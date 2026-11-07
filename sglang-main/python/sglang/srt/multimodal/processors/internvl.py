# Adapted from https://huggingface.co/OpenGVLab/InternVL2-4B/blob/main/modeling_intern_vit.py

from functools import lru_cache

import numpy as np
import torch
# HACK(Shizhuo)
# ============================================
import torch.nn.functional as F
from decord import VideoReader, cpu, gpu
from PIL import Image

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.interns1 import InternS1ForConditionalGeneration
from sglang.srt.models.internvl import InternVLChatModel
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
# HACK(Shizhuo)
import hashlib
import threading
import time
from sglang.utils import logger 
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from sglang.srt.server_args import get_global_server_args 


class InternVLImageProcessor(BaseMultimodalProcessor):
    models = [InternVLChatModel, InternS1ForConditionalGeneration]

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    @staticmethod
    @lru_cache(maxsize=1)
    def _get_normalize_tensors(device="cuda", dtype=torch.float32):
        mean = torch.tensor(
            InternVLImageProcessor.IMAGENET_MEAN, device=device, dtype=dtype
        ).view(-1, 1, 1)
        std = torch.tensor(
            InternVLImageProcessor.IMAGENET_STD, device=device, dtype=dtype
        ).view(-1, 1, 1)
        return mean, std

    def __init__(self, hf_config, server_args, _image_processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _image_processor, *args, **kwargs)

        # HACK(Shizhuo): [Config Fix] Load max_num from server_args
        # Allow external control of max_num for video (1) vs image (12)
        # Default fallback: Image=12 (High Res), Video=1 (Low Res/Efficiency)
        self.mm_process_config = server_args.mm_process_config or {}
        
        # 用户在 Engine 初始化时传入 mm_process_config={"image": {"max_num": 1}} 将在此生效
        self.default_image_max_num = self.mm_process_config.get("image", {}).get("max_num", 12)
        self.default_video_max_num = self.mm_process_config.get("video", {}).get("max_num", 1)
        
        # logger.info(
        #     f"[InternVLProcessor] Init config: "
        #     f"Image max_num={self.default_image_max_num}, "
        #     f"Video max_num={self.default_video_max_num}"
        # )
        # =========================================================

        image_size = (
            getattr(hf_config, "force_image_size", None)
            or hf_config.vision_config.image_size
        )
        patch_size = hf_config.vision_config.patch_size
        if isinstance(image_size, list):
            image_size = image_size[0]
        if isinstance(patch_size, list):
            patch_size = patch_size[0]

        self.IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
        self.IMG_START_TOKEN = "<img>"
        self.IMG_END_TOKEN = "</img>"
        self.num_image_token = int(
            (image_size // patch_size) ** 2 * (hf_config.downsample_ratio**2)
        )
        if hasattr(self._processor, "tokenizer"):
            tokenizer = self._processor.tokenizer
        else:
            tokenizer = self._processor
        self.tokenizer = tokenizer

        # HACK(Shizhuo)
        try:
            # InternVL3.5 uses llm_config.max_position_embeddings (e.g. 40960)
            max_pos = int(getattr(hf_config.llm_config, "max_position_embeddings", 0) or 0)
            if max_pos > 0:
                self.tokenizer.model_max_length = max_pos
                if hasattr(self.tokenizer, "init_kwargs"):
                    self.tokenizer.init_kwargs["model_max_length"] = max_pos

                # best-effort for fast tokenizer backend (may or may not exist)
                if hasattr(self.tokenizer, "_tokenizer"):
                    try:
                        self.tokenizer._tokenizer.model.max_length = max_pos
                    except Exception:
                        pass
        except Exception:
            pass
        # =========================================================================


        self.img_start_token_id = tokenizer.convert_tokens_to_ids(self.IMG_START_TOKEN)
        self.img_end_token_id = tokenizer.convert_tokens_to_ids(self.IMG_END_TOKEN)
        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<IMG_CONTEXT>",
            image_token_id=tokenizer.convert_tokens_to_ids(self.IMG_CONTEXT_TOKEN),
        ).build(_image_processor)

        # HACK(Shizhuo)
        # [FIX] 初始化多线程执行器
        self.cpu_executor = None
        if server_args and server_args.mm_process_config:
            num_workers = server_args.mm_process_config.get("num_workers", 0)
            if num_workers > 0:
                from concurrent.futures import ThreadPoolExecutor
                # 限制最大线程数，避免 OOM
                safe_workers = min(num_workers, 64)
                self.cpu_executor = ThreadPoolExecutor(max_workers=safe_workers)
        # Simple thread-safe LRU Cache
        self.tensor_cache = OrderedDict()
        self.cache_capacity = 1000 # Store 2000 frames (InternVL tiles are large, be careful with RAM)
        self.cache_lock = threading.Lock()
        # ========================================================

    @staticmethod
    def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / num_segments
        frame_indices = np.array(
            [
                int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
                for idx in range(num_segments)
            ]
        )
        return frame_indices

    @staticmethod
    def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
        try:
            vr = VideoReader(video_path, ctx=gpu(0), num_threads=1)
            use_gpu = True
        except (RuntimeError, OSError) as e:
            print(
                f"[WARNING] Load video on gpu decoding failed: {e}. Falling back to CPU."
            )
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            use_gpu = False

        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())

        pixel_values_list = []
        num_patches_list = []
        frame_indices = InternVLImageProcessor.get_index(
            bound, fps, max_frame, first_idx=0, num_segments=num_segments
        )

        mean, std = InternVLImageProcessor._get_normalize_tensors(device="cuda")

        for frame_index in frame_indices:
            # Load frame
            frame = vr[frame_index]
            if use_gpu:
                img = frame.cuda().permute(2, 0, 1).float() / 255.0
            else:
                img_np = frame.asnumpy()
                img = torch.from_numpy(img_np).permute(2, 0, 1).cuda().float() / 255.0

            img = (img - mean) / std

            tiles = InternVLImageProcessor.dynamic_preprocess(
                img, image_size=input_size, max_num=max_num, use_thumbnail=True
            )

            pixel_values_list.append(tiles)
            num_patches_list.append(tiles.shape[0])

        pixel_values = torch.cat(pixel_values_list, dim=0)
        return pixel_values, num_patches_list

    @staticmethod
    def dynamic_preprocess(tensor, image_size=448, max_num=12, use_thumbnail=False):
        C, H, W = tensor.shape
        aspect_ratio = W / H

        # Generate all possible aspect ratios
        target_ratios = set(
            (i, j)
            for n in range(1, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # Find closest ratio
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)

        for x, y in target_ratios:
            target_ar = x / y
            diff = abs(aspect_ratio - target_ar)
            blocks = x * y
            best_blocks = best_ratio[0] * best_ratio[1]

            if diff < best_ratio_diff:
                best_ratio_diff = diff
                best_ratio = (x, y)
            elif diff == best_ratio_diff and blocks > best_blocks:
                best_ratio = (x, y)

        target_w, target_h = image_size * best_ratio[0], image_size * best_ratio[1]
        blocks = best_ratio[0] * best_ratio[1]

        # Resize on GPU
        resized = torch.nn.functional.interpolate(
            tensor.unsqueeze(0),
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)

        # Split into tiles
        tiles = []
        for i in range(blocks):
            x = (i % best_ratio[0]) * image_size
            y = (i // best_ratio[0]) * image_size
            tile = resized[:, y : y + image_size, x : x + image_size]
            tiles.append(tile)

        # Add thumbnail if needed
        if use_thumbnail and len(tiles) > 1:
            thumb = torch.nn.functional.interpolate(
                tensor.unsqueeze(0),
                size=(image_size, image_size),
                mode="bicubic",
                align_corners=False,
            ).squeeze(0)
            tiles.append(thumb)

        return torch.stack(tiles).to(torch.bfloat16)

    # HACK(Shizhuo)
    # ============================================
    def _extract_cacheflex_cfg(self, request_obj):
        """
        只要请求里真的带了 custom_params.cacheflex，就认为 cacheflex_active=True。
        fr 模式按你的 client 逻辑不会传，所以会返回 None -> 走 Branch B。
        """
        try:
            sampling_params = getattr(request_obj, "sampling_params", None)
            if sampling_params is None and isinstance(request_obj, dict):
                sampling_params = request_obj.get("sampling_params", None)

            custom_params = getattr(sampling_params, "custom_params", None) if sampling_params else None
            if custom_params is None and isinstance(sampling_params, dict):
                custom_params = sampling_params.get("custom_params", None)

            if isinstance(custom_params, dict):
                cf = custom_params.get("cacheflex", None)
                if isinstance(cf, dict):
                    return cf
        except Exception:
            pass
        return None

    @staticmethod
    def _md5_rgb(image: Image.Image) -> str:
        b = image.convert("RGB").tobytes()
        return hashlib.md5(b).hexdigest()

    @staticmethod
    def _cpu_preprocess_pil(image: Image.Image, image_size=448, max_num=1, use_thumbnail=True):
        """
        CPU-only preprocess that mirrors GPU path exactly:
        - PIL -> torch float32 [0,1] -> normalize -> F.interpolate(bicubic) -> cast(float16)

        This ensures pixel-level consistency with GPU path by using the same torch ops.

        注意：max_num==1 走极简快_path（video 模式你基本就是这个）
        """
        image = image.convert("RGB")

        # 使用 torch tensor 的 mean/std，与 GPU 路径一致
        mean = torch.tensor(InternVLImageProcessor.IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(InternVLImageProcessor.IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)

        if max_num == 1:
            # 1) PIL -> uint8 -> torch float32 [0,1] (CHW)
            img_np = np.array(image, copy=False)  # HWC uint8
            x = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()  # CHW uint8
            x = x.float().div_(255.0)  # CHW float32 [0,1]

            # 2) normalize（复刻 GPU 路径：先 normalize）
            x = (x - mean) / std

            # 3) resize（复刻 GPU 路径参数）
            x = F.interpolate(
                x.unsqueeze(0),
                size=(image_size, image_size),
                mode="bicubic",
                align_corners=False,
                antialias=False,  # 显式锁死，与 GPU 默认一致
            ).squeeze(0)

            # 4) cast to bfloat16（与 GPU 路径一致）
            return x.to(torch.bfloat16).unsqueeze(0)  # (1,3,448,448)

        # ---- max_num > 1：复刻 dynamic_preprocess 的逻辑，但用 torch on CPU ----
        W, H = image.size
        aspect_ratio = W / max(1, H)

        target_ratios = set(
            (i, j)
            for n in range(1, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        for x, y in target_ratios:
            target_ar = x / y
            diff = abs(aspect_ratio - target_ar)
            blocks = x * y
            best_blocks = best_ratio[0] * best_ratio[1]
            if diff < best_ratio_diff or (diff == best_ratio_diff and blocks > best_blocks):
                best_ratio_diff = diff
                best_ratio = (x, y)

        target_w, target_h = image_size * best_ratio[0], image_size * best_ratio[1]
        blocks = best_ratio[0] * best_ratio[1]

        # 1) PIL -> uint8 -> torch float32 [0,1]
        img_np = np.array(image, copy=False)  # HWC uint8
        x = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()  # CHW uint8
        x = x.float().div_(255.0)  # CHW float32 [0,1]

        # 2) normalize（复刻 GPU 路径：先 normalize）
        x = (x - mean) / std

        # 3) resize（复刻 GPU 路径参数）
        resized = F.interpolate(
            x.unsqueeze(0),
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False,
            antialias=False,
        ).squeeze(0)

        # 4) split into tiles
        tiles = []
        for i in range(blocks):
            tile_x = (i % best_ratio[0]) * image_size
            tile_y = (i // best_ratio[0]) * image_size
            tile = resized[:, tile_y : tile_y + image_size, tile_x : tile_x + image_size]
            tiles.append(tile)

        # 5) add thumbnail if needed
        if use_thumbnail and len(tiles) > 1:
            thumb = F.interpolate(
                x.unsqueeze(0),
                size=(image_size, image_size),
                mode="bicubic",
                align_corners=False,
                antialias=False,
            ).squeeze(0)
            tiles.append(thumb)

        # 6) cast to bfloat16（与 GPU 路径一致）
        return torch.stack(tiles, dim=0).to(torch.bfloat16)

    def _process_single_image_task(self, image, index, current_max_num, client_hash=None):
        """
        ThreadPool 里只做 CPU 工作：PIL resize/crop + numpy normalize
        返回 CPU tensor（float16），避免并发触发 GPU kernel。
        """
        try:
            if not isinstance(image, Image.Image):
                # 防御：如果上游给的不是 PIL，就尽量转成 PIL
                if isinstance(image, torch.Tensor):
                    img_np = image.detach().cpu().numpy()
                    if img_np.ndim == 3 and img_np.shape[0] == 3:
                        img_np = np.transpose(img_np, (1, 2, 0))
                    image = Image.fromarray(img_np.astype(np.uint8))
                else:
                    image = Image.fromarray(np.asarray(image).astype(np.uint8))

            f_hash = client_hash or self._md5_rgb(image)
            tiles_cpu = self._cpu_preprocess_pil(
                image=image, image_size=448, max_num=current_max_num, use_thumbnail=True
            )
            return index, tiles_cpu, f_hash

        except Exception as e:
            print(f"[Error] CPU preprocess failed for image {index}: {e}")
            return index, None, None

    async def process_mm_data_async(
        self, image_data, input_text, request_obj, **kwargs
    ):
        # HACK(Shizhuo)
        # [Profile] Start CPU Timer
        t_cpu_start = time.perf_counter()

        # 只在“请求里真的带了 custom_params.cacheflex”时，才认为需要走 CacheFlex 加速分支
        cacheflex_cfg = self._extract_cacheflex_cfg(request_obj)
        cacheflex_active = isinstance(cacheflex_cfg, dict)
        server_args = get_global_server_args()
        mpc = getattr(server_args, "mm_process_config", None) or {}
        cacheflex_mode = mpc.get("cacheflex_mode", "")

        # fr 模式：允许线程池，但禁止跨轮复用（不启用 LRU）
        threading_only = (
            (cacheflex_mode == "fr")
            and (server_args is not None)
            and bool(getattr(server_args, "enable_cacheflex", False))
            and (self.cpu_executor is not None)
        )
        client_hashes = cacheflex_cfg.get("video_frame_hashes", []) if cacheflex_active else []

        if cacheflex_active:
            # =========================================================
            # BRANCH A: CacheFlex (Optimized + Video Support + Caching)
            # 触发条件：cacheflex_active=True（请求里确实传了 custom_params.cacheflex）
            # =========================================================

            # HACK(Shizhuo): Check whether to use GPU for image preprocessing
            server_args = get_global_server_args()
            use_gpu_preprocess = getattr(server_args, "cacheflex_use_gpu_preprocess", True)

            # 1) Load Data（保持你原逻辑：显式传 video_data）
            base_output = self.load_mm_data(
                prompt=input_text,
                image_data=image_data,
                video_data=getattr(request_obj, "video_data", None),
                multimodal_tokens=self.mm_tokens,
                discard_alpha_channel=True,
            )

            # 2) Detect Video & Config
            has_video_input = (
                (getattr(request_obj, "video_data", None) is not None)
                or (len(getattr(base_output, "videos", [])) > 0)
                or (len(getattr(base_output, "images", [])) > 1)
            )
            current_max_num = self.default_video_max_num if has_video_input else self.default_image_max_num

            # HACK(Shizhuo): GPU Path - Faster and consistent with baseline_prefix_org
            if use_gpu_preprocess:
                # ========================================================
                # BRANCH A.1: GPU Serial Path (推荐)
                # ========================================================
                use_feature_cache = (len(client_hashes) == len(base_output.images) and len(client_hashes) > 0)
                mean = torch.tensor(self.IMAGENET_MEAN, device='cuda', dtype=torch.float32).view(-1, 1, 1)
                std = torch.tensor(self.IMAGENET_STD, device='cuda', dtype=torch.float32).view(-1, 1, 1)

                pixel_values = []
                num_patches_list = []
                frame_hashes = []

                # GPU 串行处理每一帧（但会跳过 cache hit 帧）
                for idx, image in enumerate(base_output.images):
                    # 3.1 Check if we can skip this frame (optional optimization)
                    # For now, use client_hashes as skip indicator
                    should_skip = False
                    if use_feature_cache and idx < len(client_hashes):
                        c_hash = client_hashes[idx]
                        if c_hash in self.tensor_cache:
                            # Cache hit at processor level - skip processing
                            cached = self.tensor_cache[c_hash]
                            pixel_values.append(cached)
                            num_patches_list.append(int(cached.shape[0]))
                            frame_hashes.append(c_hash)
                            should_skip = True

                    if not should_skip:
                        # 3.2 GPU Processing
                        if isinstance(image, Image.Image):
                            img_np = np.array(image.convert("RGB"))
                            x = torch.from_numpy(img_np).permute(2, 0, 1).cuda().float() / 255.0
                        else:
                            x = image.cuda()

                        x = (x - mean) / std
                        tiles = self.dynamic_preprocess(
                            x,
                            image_size=448,
                            max_num=current_max_num,
                            use_thumbnail=True
                        )
                        pixel_values.append(tiles)
                        num_patches_list.append(tiles.shape[0])
                        # Use hash if available, else generate one
                        f_hash = client_hashes[idx] if idx < len(client_hashes) else self._md5_rgb(image)
                        frame_hashes.append(f_hash)

                # Concatenate results (keep float32 like native GPU path)
                pixel_values = torch.cat(pixel_values, dim=0)

            else:
                # ========================================================
                # BRANCH A.2: CPU Multithread Path (原始实现)
                # ========================================================
                # 3) Parallel CPU Processing + Feature Cache Lookup（只在 active 且 hashes 对齐时查表）
                use_feature_cache = (len(client_hashes) == len(base_output.images) and len(client_hashes) > 0)

                import asyncio
                loop = asyncio.get_running_loop()
                tasks = []
                results_map = {}

                for idx, image in enumerate(base_output.images):
                    c_hash = client_hashes[idx] if use_feature_cache else None

                    # 3.1 Lookup（只在 hashes 可用时才查）
                    hit = False
                    if use_feature_cache and c_hash:
                        with self.cache_lock:
                            if c_hash in self.tensor_cache:
                                results_map[idx] = (self.tensor_cache[c_hash], c_hash)
                                self.tensor_cache.move_to_end(c_hash)
                                hit = True

                    # 3.2 Miss -> CPU task
                    if not hit:
                        if self.cpu_executor is not None:
                            tasks.append(loop.run_in_executor(
                                self.cpu_executor,
                                self._process_single_image_task,
                                image, idx, current_max_num, c_hash
                            ))
                        else:
                            i, tiles_cpu, f_hash = self._process_single_image_task(image, idx, current_max_num, c_hash)
                            if tiles_cpu is None:
                                return None
                            results_map[i] = (tiles_cpu, f_hash)
                            if use_feature_cache and f_hash:
                                with self.cache_lock:
                                    self.tensor_cache[f_hash] = tiles_cpu
                                    if len(self.tensor_cache) > self.cache_capacity:
                                        self.tensor_cache.popitem(last=False)

                if tasks:
                    for idx, tiles_cpu, f_hash in await asyncio.gather(*tasks):
                        if tiles_cpu is None:
                            return None
                        results_map[idx] = (tiles_cpu, f_hash)
                        if use_feature_cache and f_hash:
                            with self.cache_lock:
                                self.tensor_cache[f_hash] = tiles_cpu
                                if len(self.tensor_cache) > self.cache_capacity:
                                    self.tensor_cache.popitem(last=False)

                # 4) Assemble（全部保持 CPU tensor）
                pixel_values_list = []
                num_patches_list = []
                frame_hashes = []

                for idx in range(len(base_output.images)):
                    tiles_cpu, h = results_map[idx]
                    pixel_values_list.append(tiles_cpu)
                    num_patches_list.append(int(tiles_cpu.shape[0]))
                    frame_hashes.append(h)

                if not pixel_values_list:
                    return None

                pixel_values = torch.cat(pixel_values_list, dim=0)  # CPU float16

            # 5) Text Replacement（保持原逻辑）
            original_placeholder = "<<<__IMG_CONTEXT_PLACEHOLDER__>>>"
            input_text = base_output.input_text.replace(self.IMG_CONTEXT_TOKEN, original_placeholder)

            input_text_updated = input_text
            for num_patches in num_patches_list:
                image_tokens = (
                    self.IMG_START_TOKEN
                    + self.IMG_CONTEXT_TOKEN * self.num_image_token * num_patches
                    + self.IMG_END_TOKEN
                )
                input_text_updated = input_text_updated.replace(original_placeholder, image_tokens, 1)

            input_text_updated = input_text_updated.replace(original_placeholder, self.IMG_CONTEXT_TOKEN)

            # 6) Tokenize & Offsets
            input_ids_tensor = self.tokenizer(input_text_updated, return_tensors="pt")["input_ids"].flatten()
            input_ids = input_ids_tensor.tolist()

            image_offsets = self.get_mm_items_offset(
                input_ids=input_ids_tensor.to("cuda"),
                mm_token_id=self.mm_tokens.image_token_id,
            )

            # 7) data_offsets / image_pad_len（保持你原 CacheFlex 分支的细粒度切片信息）
            image_pad_len = [int(self.num_image_token * n) for n in num_patches_list]
            mm_id = self.mm_tokens.image_token_id
            ctx_pos = (input_ids_tensor == mm_id).nonzero(as_tuple=False).flatten().tolist()

            data_offsets = []
            cursor = 0
            for L in image_pad_len:
                if cursor < len(ctx_pos):
                    data_offsets.append(ctx_pos[cursor])
                    cursor += L

            # 8) CPU Cost
            t_cpu_end = time.perf_counter()
            cpu_cost = t_cpu_end - t_cpu_start

            # ====================================================================
            # [DEBUG] Level 1: 确认每个请求的"帧序列与 hash 序列"是否一一对应
            # ====================================================================
            from sglang.srt.utils import logger
            logger.info(
                f"[CF_FHASH_SRC] cacheflex_active={cacheflex_active} "
                f"num_frames={len(frame_hashes)} "
                f"hash_head={[str(h)[:8] for h in frame_hashes[:4]] if frame_hashes else None} "
                f"hash_type={type(frame_hashes[0]).__name__ if frame_hashes else None} "
                f"source={'custom' if cacheflex_active and client_hashes else 'default'}"
            )

            # 9) Construct Item
            item = MultimodalDataItem(
                feature=pixel_values,  # ✅ CPU tensor (float16)
                modality=Modality.VIDEO if has_video_input else Modality.IMAGE,
                offsets=image_offsets,
                model_specific_data={
                    "frame_hashes": frame_hashes,
                    "cpu_preprocessing_latency": cpu_cost,
                },
            )
            setattr(item, "frame_hashes", frame_hashes)

            ret = {
                "input_ids": input_ids,
                "mm_items": [item],
                "im_start_id": self.img_start_token_id,
                "im_end_id": self.img_end_token_id,
                "im_token_id": self.mm_tokens.image_token_id,
            }
            ret["data_offsets"] = data_offsets
            ret["image_pad_len"] = image_pad_len
            return ret
        # ==================================================================
        
        else:
            # =========================================================
            # BRANCH B: Native SGLang Logic (Zero Regression)
            # + [NEW] threading-only path for fr (no LRU, now supports GPU path)
            # =========================================================

            # HACK(Shizhuo): Check if GPU path should be used (applies to fr mode as well)
            use_gpu_preprocess = getattr(server_args, "cacheflex_use_gpu_preprocess", True)

            if threading_only:
                # =========================================================
                # fr Mode - Choose CPU or GPU based on parameter
                # =========================================================
                if use_gpu_preprocess:
                    # =========================================================
                    # BRANCH B.1: fr mode with GPU serial processing
                    # =========================================================
                    base_output = self.load_mm_data(
                        prompt=input_text,
                        image_data=image_data,
                        video_data=getattr(request_obj, "video_data", None),
                        multimodal_tokens=self.mm_tokens,
                        discard_alpha_channel=True,
                    )

                    has_video_input = (
                        (getattr(request_obj, "video_data", None) is not None)
                        or (len(getattr(base_output, "videos", [])) > 0)
                        or (len(getattr(base_output, "images", [])) > 1)
                    )
                    current_max_num = self.default_video_max_num if has_video_input else self.default_image_max_num

                    # GPU serial processing (similar to baseline_prefix_org)
                    mean = torch.tensor(self.IMAGENET_MEAN, device='cuda', dtype=torch.float32).view(-1, 1, 1)
                    std = torch.tensor(self.IMAGENET_STD, device='cuda', dtype=torch.float32).view(-1, 1, 1)

                    pixel_values = []
                    num_patches_list = []
                    frame_hashes = []

                    for idx, image in enumerate(base_output.images):
                        # GPU processing
                        if isinstance(image, Image.Image):
                            img_np = np.array(image.convert("RGB"))
                            x = torch.from_numpy(img_np).permute(2, 0, 1).cuda().float() / 255.0
                        else:
                            x = image.cuda()

                        x = (x - mean) / std
                        tiles = self.dynamic_preprocess(
                            x,
                            image_size=448,
                            max_num=current_max_num,
                            use_thumbnail=True
                        )
                        pixel_values.append(tiles)
                        num_patches_list.append(tiles.shape[0])
                        f_hash = self._md5_rgb(image) if isinstance(image, Image.Image) else None
                        frame_hashes.append(f_hash)

                    pixel_values = torch.cat(pixel_values, dim=0)  # Keep float32 like native GPU path

                else:
                    # =========================================================
                    # fr mode with CPU multithread processing (original)
                    # =========================================================
                    base_output = self.load_mm_data(
                        prompt=input_text,
                        image_data=image_data,
                        video_data=getattr(request_obj, "video_data", None),
                        multimodal_tokens=self.mm_tokens,
                        discard_alpha_channel=True,
                    )

                    has_video_input = (
                        (getattr(request_obj, "video_data", None) is not None)
                        or (len(getattr(base_output, "videos", [])) > 0)
                        or (len(getattr(base_output, "images", [])) > 1)
                    )
                    current_max_num = self.default_video_max_num if has_video_input else self.default_image_max_num

                    import asyncio
                    loop = asyncio.get_running_loop()
                    tasks = []
                    for idx, image in enumerate(base_output.images):
                        tasks.append(loop.run_in_executor(
                            self.cpu_executor,
                            self._process_single_image_task,
                            image, idx, current_max_num, None
                        ))
                    gathered = await asyncio.gather(*tasks)

                    pixel_values_list, num_patches_list, frame_hashes = [], [], []
                    for idx, tiles_cpu, h in sorted(gathered, key=lambda x: x[0]):
                        if tiles_cpu is None:
                            return None
                        pixel_values_list.append(tiles_cpu)
                        num_patches_list.append(int(tiles_cpu.shape[0]))
                        frame_hashes.append(h)

                    if not pixel_values_list:
                        return None

                    pixel_values = torch.cat(pixel_values_list, dim=0)  # CPU float16

                # ---- token replacement (same as Branch A) ----
                original_placeholder = "<<<__IMG_CONTEXT_PLACEHOLDER__>>>"
                input_text_tmp = base_output.input_text.replace(self.IMG_CONTEXT_TOKEN, original_placeholder)

                input_text_updated = input_text_tmp
                for num_patches in num_patches_list:
                    image_tokens = (
                        self.IMG_START_TOKEN
                        + self.IMG_CONTEXT_TOKEN * self.num_image_token * num_patches
                        + self.IMG_END_TOKEN
                    )
                    input_text_updated = input_text_updated.replace(original_placeholder, image_tokens, 1)
                input_text_updated = input_text_updated.replace(original_placeholder, self.IMG_CONTEXT_TOKEN)

                input_ids_tensor = self.tokenizer(input_text_updated, return_tensors="pt")["input_ids"].flatten()
                input_ids = input_ids_tensor.tolist()

                image_offsets = self.get_mm_items_offset(
                    input_ids=input_ids_tensor.to("cuda"),
                    mm_token_id=self.mm_tokens.image_token_id,
                )

                image_pad_len = [int(self.num_image_token * n) for n in num_patches_list]
                mm_id = self.mm_tokens.image_token_id
                ctx_pos = (input_ids_tensor == mm_id).nonzero(as_tuple=False).flatten().tolist()

                data_offsets = []
                cursor = 0
                for L in image_pad_len:
                    if cursor < len(ctx_pos):
                        data_offsets.append(ctx_pos[cursor])
                        cursor += L

                t_cpu_end = time.perf_counter()
                cpu_cost = t_cpu_end - t_cpu_start

                item = MultimodalDataItem(
                    feature=pixel_values,  # CPU float16
                    modality=Modality.IMAGE,  # ✅ keep SAME as native InternVL behavior
                    offsets=image_offsets,
                    model_specific_data={
                        "frame_hashes": frame_hashes,
                        "cpu_preprocessing_latency": cpu_cost,
                    },
                )
                setattr(item, "frame_hashes", frame_hashes)

                ret = {
                    "input_ids": input_ids,
                    "mm_items": [item],
                    "im_start_id": self.img_start_token_id,
                    "im_end_id": self.img_end_token_id,
                    "im_token_id": self.mm_tokens.image_token_id,
                }

                ret["data_offsets"] = data_offsets
                ret["image_pad_len"] = image_pad_len

                return ret
            # ----------- 原来的 GPU native Branch B 逻辑从这里开始（保持不变，除了 cpu_cost 注入） -----------

            # Explicitly copied from your provided snippet
            base_output = self.load_mm_data(
                prompt=input_text,
                image_data=image_data,
                video_data=getattr(request_obj, "video_data", None), 
                multimodal_tokens=self.mm_tokens,
                discard_alpha_channel=True,
            )

            num_patches_list = []
            pixel_values = []
            # HACK(Shizhuo): Init frame_hashes
            frame_hashes = []
            # ================================

            mean, std = InternVLImageProcessor._get_normalize_tensors(device="cuda")

            # HACK(Shizhuo): [Logic Fix] Determine max_num based on input modality
            # Detect if the input contains video data
            has_video_input = (
                (getattr(request_obj, "video_data", None) is not None) 
                or (len(getattr(base_output, "videos", [])) > 0)
                or (len(base_output.images) > 1) 
            )

            # Select appropriate max_num
            # Note: If passing frames as a list of images (e.g. in test script), 
            # users MUST set mm_process_config={"image": {"max_num": 1}} in Engine to override default_image_max_num
            current_max_num = self.default_video_max_num if has_video_input else self.default_image_max_num
            # =========================================================

            # Process each input with allocated frames
            for image_index, image in enumerate(base_output.images):
                try:
                    # TODO: video input
                    # Convert PIL to GPU tensor
                    if isinstance(image, Image.Image):
                        img_np = np.array(image.convert("RGB"))
                        tensor = (
                            torch.from_numpy(img_np).permute(2, 0, 1).cuda().float() / 255.0
                        )
                    else:
                        tensor = image.cuda()  # assume already tensor

                    tensor = (tensor - mean) / std

                    # HACK(Shizhuo)
                    tiles = self.dynamic_preprocess(
                        tensor, 
                        image_size=448, 
                        max_num=current_max_num,  # <--- Changed from 12
                        use_thumbnail=True
                    )
                    # tiles = self.dynamic_preprocess(
                    #     tensor, image_size=448, max_num=12, use_thumbnail=True
                    # )
                    # =========================================================

                    pixel_values.append(tiles)
                    num_patches_list.append(tiles.shape[0])

                except Exception as e:
                    print(f"[Error] Failed to process image {image_index}: {e}")
                    return None

            # Concatenate all
            pixel_values = torch.cat(pixel_values, dim=0)

            original_placeholder = "<<<__IMG_CONTEXT_PLACEHOLDER__>>>"

            input_text = base_output.input_text.replace(
                self.IMG_CONTEXT_TOKEN, original_placeholder
            )

            input_text_updated = input_text
            for num_patches in num_patches_list:
                image_tokens = (
                    self.IMG_START_TOKEN
                    + self.IMG_CONTEXT_TOKEN * self.num_image_token * num_patches
                    + self.IMG_END_TOKEN
                )
                input_text_updated = input_text_updated.replace(
                    original_placeholder, image_tokens, 1
                )

            input_text_updated = input_text_updated.replace(
                original_placeholder, self.IMG_CONTEXT_TOKEN
            )

            # Tokenize
            input_ids_tensor = self.tokenizer(input_text_updated, return_tensors="pt")[
                "input_ids"
            ].flatten()
            input_ids = input_ids_tensor.tolist()

            # Get image token offsets
            image_offsets = self.get_mm_items_offset(
                input_ids=input_ids_tensor.to("cuda"),
                mm_token_id=self.mm_tokens.image_token_id,
            )

            t_cpu_end = time.perf_counter()
            cpu_cost = t_cpu_end - t_cpu_start
            items = [
                MultimodalDataItem(
                    feature=pixel_values,
                    modality=Modality.IMAGE, # Original behavior
                    offsets=image_offsets,
                    # HACK(Shizhuo): Inject CPU Latency even for native path
                    model_specific_data={
                        "cpu_preprocessing_latency": cpu_cost
                    }
                )
            ]
            return {
                "input_ids": input_ids,
                "mm_items": items,
                "im_start_id": self.img_start_token_id,
                "im_end_id": self.img_end_token_id,
                "im_token_id": self.mm_tokens.image_token_id,
            }
