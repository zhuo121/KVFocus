import logging  # HACK(Shizhuo): CacheFlex: frame-hash logging
import asyncio
from typing import List, Optional, Union

import numpy as np
from transformers.models.auto.processing_auto import (
    PROCESSOR_MAPPING_NAMES as HF_MAPPING_NAMES,
)

import sglang.srt.managers.multimodal_processor as sgl_mm_processor_utils
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.llava import (
    LlavaForConditionalGeneration,
    LlavaLlamaForCausalLM,
    LlavaMistralForCausalLM,
    LlavaQwenForCausalLM,
)
from sglang.srt.models.llavavid import LlavaVidForCausalLM
from sglang.srt.models.mistral import Mistral3ForConditionalGeneration
from sglang.srt.multimodal.mm_utils import expand2square, process_anyres_image
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.utils import ImageData, load_image, logger
from sglang.utils import get_exception_traceback

# HACK(Shizhuo)
from collections import OrderedDict
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from sglang.srt.server_args import get_global_server_args

class LlavaImageProcessor(BaseMultimodalProcessor):
    models = [
        LlavaLlamaForCausalLM,
        LlavaVidForCausalLM,
        LlavaQwenForCausalLM,
        LlavaMistralForCausalLM,
    ]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

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

        # 初始化 Tensor Cache (简单线程安全的 LRU)
        # 结构: {hash_str: (pixel_values_np, image_size)}
        self.tensor_cache = OrderedDict()
        self.cache_capacity = 1000 # 缓存 1000 帧 (约 400-800MB RAM)
        self.cache_lock = threading.Lock()

    @staticmethod
    def _process_single_image_task(
        image_data: Union[str, bytes, ImageData],
        image_aspect_ratio: Optional[str] = None,
        image_grid_pinpoints: Optional[str] = None,
        processor=None,
    ):
        # HACK(Shizhuo)
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is None:
            image_processor = processor  # assume processor itself is an image processor
        # image_processor = processor.image_processor
        # ===========================================================

        try:
            url = image_data.url if isinstance(image_data, ImageData) else image_data

            # HACK(Shizhuo)
            # image, image_size = load_image(url)
            from PIL import Image
            import hashlib
            import io
            image_hash = None

            stride = getattr(processor, "mm_spatial_pool_stride", None)
            if stride is None and hasattr(processor, "config"):
                stride = getattr(processor.config, "mm_spatial_pool_stride", None)
            if stride is None:
                stride = 2
            cfg_sig = f"stride_{stride}"
            # cfg_sig = f"stride_{getattr(processor, 'mm_spatial_pool_stride', 2)}"

            if isinstance(url, Image.Image):
                image, image_size = url, None
                image_hash = hashlib.md5(image.tobytes() + cfg_sig.encode()).hexdigest()
            elif isinstance(url, (bytes, bytearray)):
                image, image_size = Image.open(io.BytesIO(url)).convert("RGB"), None
                image_hash = hashlib.md5(url + cfg_sig.encode()).hexdigest()
            else:
                image, image_size = load_image(url)
                # 保持路径作为 Key，如果不是字符串则用 MD5
                image_hash = url if isinstance(url, str) else hashlib.md5(image.tobytes() + cfg_sig.encode()).hexdigest()         
            # ==========================================================

            if image_size is not None:
                # It is a video with multiple images

                # HACK(Shizhuo) 
                # 屏蔽原生 hash(url)
                # image_hash = hash(url)
                # ========================
                pixel_values = image_processor(image)["pixel_values"]
                for _ in range(len(pixel_values)):
                    pixel_values[_] = pixel_values[_].astype(np.float16)
                pixel_values = np.stack(pixel_values, axis=0)
                return pixel_values, image_hash, image_size
            else:
                # It is an image

                # HACK(Shizhuo) 
                # 屏蔽原生 hash(url)
                # image_hash = hash(url)
                # ========================
                
                if image_aspect_ratio == "pad":
                    image = expand2square(
                        image,
                        tuple(int(x * 255) for x in image_processor.image_mean),
                    )
                    pixel_values = image_processor(image.convert("RGB"))[
                        "pixel_values"
                    ][0]
                elif image_aspect_ratio == "anyres" or (
                    image_aspect_ratio is not None
                    and "anyres_max" in image_aspect_ratio
                ):
                    pixel_values = process_anyres_image(
                        image, image_processor, image_grid_pinpoints
                    )
                else:
                    pixel_values = image_processor(image)["pixel_values"][0]

                if isinstance(pixel_values, np.ndarray):
                    pixel_values = pixel_values.astype(np.float16)

                return pixel_values, image_hash, image.size
        except Exception:
            logger.error("Exception in TokenizerManager:\n" + get_exception_traceback())

    async def _process_single_image(
        self,
        image_data: Union[bytes, str, ImageData],
        aspect_ratio: str,
        grid_pinpoints: str,
    ):
        if self.cpu_executor is not None:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self.cpu_executor,
                LlavaImageProcessor._process_single_image_task,
                image_data,
                aspect_ratio,
                grid_pinpoints,
                self._processor,
            )
        else:
            return self._process_single_image_task(
                image_data,
                aspect_ratio,
                grid_pinpoints,
                # HACK(Shizhuo)
                self._processor,
            )

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes, ImageData]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        # HACK(Shizhuo)
        t_cpu_start = time.perf_counter()

        # 1. 解析配置与开关
        client_hashes = []
        cacheflex_cfg = None
        try:
            sampling_params = getattr(request_obj, "sampling_params", None)
            if not sampling_params and isinstance(request_obj, dict):
                sampling_params = request_obj.get("sampling_params")

            custom_params = getattr(sampling_params, "custom_params", None) if sampling_params else None
            if custom_params is None and isinstance(sampling_params, dict):
                custom_params = sampling_params.get("custom_params")

            if isinstance(custom_params, dict):
                cacheflex_cfg = custom_params.get("cacheflex", None)
                if isinstance(cacheflex_cfg, dict):
                    client_hashes = cacheflex_cfg.get("video_frame_hashes", []) or []
        except Exception:
            cacheflex_cfg = None

        cacheflex_active = isinstance(cacheflex_cfg, dict) 

        # ========================================================================
        # BRANCH 1: CacheFlex 优化路径 (Feature Cache + Multithreading + Hash Injection)
        # ========================================================================
        if cacheflex_active:
            # 1. 动态初始化 LRU Cache (如果不存在)
            if not hasattr(self, "tensor_cache"):
                from collections import OrderedDict
                import threading
                self.tensor_cache = OrderedDict()
                self.cache_lock = threading.Lock()
                self.cache_capacity = 5000 # 约 3GB 显存/内存保护
            
            # 2. 找回真实数据 (解决 Engine 槽位偏移问题)
            real_data = getattr(request_obj, "video_data", None) or image_data
            is_video = (getattr(request_obj, "video_data", None) is not None) or \
                       (isinstance(real_data, list) and len(real_data) > 1)
            
            req_modalities = getattr(request_obj, "modalities", None)
            if not req_modalities and isinstance(request_obj, dict):
                req_modalities = request_obj.get("modalities")
            modalities = ["video"] if is_video else (req_modalities or ["image"])

            aspect_ratio = getattr(self.hf_config, "image_aspect_ratio", None)
            grid_pinpoints = (
                self.hf_config.image_grid_pinpoints
                if hasattr(self.hf_config, "image_grid_pinpoints")
                and "anyres" in aspect_ratio
                else None
            )

            if real_data is not None and len(real_data) > 0:
                final_pixel_values_list = [None] * len(real_data)
                final_image_sizes_list = [None] * len(real_data)
                # 优先使用 Client Hash，如果没有则后续填补
                final_hashes_list = list(client_hashes) if (client_hashes and len(client_hashes) == len(real_data)) else [None] * len(real_data)
                miss_indices = []
                miss_data_list = []

                # 3. 查表 (Feature Cache Lookup)
                # 只有当 Client Hash 存在且对齐时才能查表
                use_cache = (len(final_hashes_list) == len(real_data) and final_hashes_list[0] is not None)
                
                if use_cache:
                    with self.cache_lock:
                        for i, h in enumerate(final_hashes_list):
                            if h in self.tensor_cache:
                                # HIT
                                cached_pv, cached_size = self.tensor_cache[h]
                                final_pixel_values_list[i] = cached_pv
                                final_image_sizes_list[i] = cached_size
                                self.tensor_cache.move_to_end(h)
                            else:
                                # MISS
                                miss_indices.append(i)
                                miss_data_list.append(real_data[i])
                else:
                    # 无法查表，全部计算
                    miss_indices = list(range(len(real_data)))
                    miss_data_list = real_data

                # 4. 处理 Miss 数据 (Multithreaded CPU Processing)
                if miss_data_list:
                    # 动态初始化线程池 (如果未初始化)
                    if getattr(self, "cpu_executor", None) is None:
                        # 默认开 16 线程，大幅加速 Resize
                        self.cpu_executor = ThreadPoolExecutor(max_workers=16)

                    # 复用原有单图处理逻辑
                    _aspect_ratio = "pad" if ("multi-images" in modalities or "video" in modalities) else aspect_ratio
                    
                    res = []
                    for img_data in miss_data_list:
                        res.append(
                            self._process_single_image(img_data, _aspect_ratio, grid_pinpoints)
                        )
                    
                    # 并行等待
                    processed_results = await asyncio.gather(*res)
                    
                    # 回填与更新缓存
                    for k, (pv, h, s) in enumerate(processed_results):
                        original_idx = miss_indices[k]
                        
                        # 决定最终使用哪个 Hash 作为 Key
                        # 如果有 Client Hash，优先用它 (逻辑一致性)；否则用 Pixel Hash
                        target_hash = final_hashes_list[original_idx]
                        if not target_hash:
                            target_hash = h
                            final_hashes_list[original_idx] = h

                        final_pixel_values_list[original_idx] = pv
                        final_image_sizes_list[original_idx] = s
                        
                        # 写入缓存
                        if target_hash:
                            with self.cache_lock:
                                self.tensor_cache[target_hash] = (pv, s)
                                if len(self.tensor_cache) > self.cache_capacity:
                                    self.tensor_cache.popitem(last=False)

                # 5. Stack Results
                if isinstance(final_pixel_values_list[0], np.ndarray):
                    pixel_values = np.stack(final_pixel_values_list, axis=0)
                else:
                    pixel_values = np.array(final_pixel_values_list) # 防御空数据
                    
                image_sizes = final_image_sizes_list
                data_hashes = final_hashes_list # 这里的 data_hashes 已经是正确的 MD5 了
                
            else:
                raise ValueError(f"Invalid image data: {real_data}")

            modality = Modality.IMAGE
            if "video" in modalities: modality = Modality.VIDEO
            elif "multi-images" in modalities: modality = Modality.MULTI_IMAGES
            
            # 统计 CPU 耗时
            t_cpu_end = time.perf_counter()
            cpu_cost = t_cpu_end - t_cpu_start
            
            return {
                "mm_items": [
                    MultimodalDataItem(
                        feature=pixel_values,
                        model_specific_data={
                            "image_sizes": image_sizes,
                            "frame_hashes": data_hashes, # 源头注入生效
                            "cpu_preprocessing_latency": cpu_cost, # 性能数据注入
                        },
                        modality=modality,
                    )
                ],
            }

        # ========================================================================
        # BRANCH 2: Original SGLang Logic (完全保留源代码逻辑，只适配多图视频处理，增加时间统计，不做其他任何修改)
        # ========================================================================
        else:
            # HACK(Shizhuo)
            # logger.info(f"[MMDBG] modalities={request_obj.modalities} type(image_data[0])={type(image_data[0])}")
            # modalities = request_obj.modalities or ["image"]
            
            t_cpu_start = time.perf_counter()
            # 1. 解决 Engine.generate 槽位位移，找回数据
            real_data = getattr(request_obj, "video_data", None) or image_data
            is_video = (getattr(request_obj, "video_data", None) is not None) or \
                    (isinstance(real_data, list) and len(real_data) > 1)
            modalities = ["video"] if is_video else (getattr(request_obj, "modalities", None) or ["image"])
            # =====================================================================

            aspect_ratio = getattr(self.hf_config, "image_aspect_ratio", None)
            grid_pinpoints = (
                self.hf_config.image_grid_pinpoints
                if hasattr(self.hf_config, "image_grid_pinpoints")
                and "anyres" in aspect_ratio
                else None
            )

            # HACK(Shizhuo)
            # if isinstance(image_data, list) and len(image_data) > 0:
            if real_data is not None and len(real_data) > 0:
            # =========================================================

                if "multi-images" in modalities or "video" in modalities:
                    # Multiple images/ Video mode
                    aspect_ratio = "pad"  # LLaVA OneVision Handling: more than one image --> interleaved image mode or video mode. We do not use anyres
                    pixel_values, data_hashes, image_sizes = [], [], []
                    res = []

                    # HACK(Shizhuo)
                    # for img_data in image_data:
                    for img_data in real_data:
                    # ===========================

                        res.append(
                            self._process_single_image(
                                img_data, aspect_ratio, grid_pinpoints
                            )
                        )

                    res = await asyncio.gather(*res)
                    for pixel_v, image_h, image_s in res:
                        pixel_values.append(pixel_v)
                        data_hashes.append(image_h)
                        image_sizes.append(image_s)

                    if isinstance(pixel_values[0], np.ndarray):
                        pixel_values = np.stack(pixel_values, axis=0)
                else:
                    # A single image

                    # HACK(Shizhuo)
                    # pixel_values, image_hash, image_size = await self._process_single_image(
                    #     image_data[0], aspect_ratio, grid_pinpoints
                    # )
                    pixel_values, image_hash, image_size = await self._process_single_image(
                        real_data[0], aspect_ratio, grid_pinpoints
                    )
                    # ========================================================================

                    image_sizes = [image_size]
                    # HACK(Shizhuo)
                    data_hashes = [image_hash]
            else:
                # raise ValueError(f"Invalid image data: {image_data}")
                raise ValueError(f"Invalid image data: real_data is {real_data}")

            modality = Modality.IMAGE
            # HACK(Shizhuo)
            # 修正模态判定逻辑
            if "video" in modalities:
                modality = Modality.VIDEO
            elif "multi-images" in modalities:
                modality = Modality.MULTI_IMAGES
            # if isinstance(request_obj.modalities, list):
            #     if request_obj.modalities[0] == "multi-images":
            #         modality = Modality.MULTI_IMAGES
            #     elif request_obj.modalities[0] == "video":
            #         modality = Modality.VIDEO
            # ===================================================

            # HACK(Shizhuo) 源头注入 Client Hash
            if logger.isEnabledFor(logging.INFO):
                logger.info("[CacheFlex Processor] Modality: %s, Frames: %d, FirstHash: %s", 
                            modality, len(data_hashes), data_hashes[0][:8] if data_hashes else "None")

            final_hashes = data_hashes

            # [PROFILE] 结束计时 (CPU Preprocessing End)
            t_cpu_end = time.perf_counter()
            cpu_cost = t_cpu_end - t_cpu_start
            # =======================

            return {
                "mm_items": [
                    MultimodalDataItem(
                        feature=pixel_values,
                        model_specific_data={
                            "image_sizes": image_sizes,
                            # HACK(Shizhuo)
                            # 核心：将帧哈希注入，供 Scheduler 匹配 KV Cache
                            "frame_hashes": final_hashes, 
                            "cpu_preprocessing_latency": cpu_cost,
                        },
                        modality=modality,
                    )
                ],
            }


class LlavaMultimodalProcessor(BaseMultimodalProcessor):
    """
    This is a wrapper class used to identify the multimodal processor for Llava architectures' vision model.
    """

    models = [LlavaForConditionalGeneration, Mistral3ForConditionalGeneration]

    def _get_sgl_processor_cls(self, model_type: str):
        if hf_name := HF_MAPPING_NAMES.get(model_type):
            sgl_mm_processor_set = sgl_mm_processor_utils.PROCESSOR_MAPPING.values()
            sgl_processor_cls = list(
                filter(lambda p: p.__name__ == hf_name, sgl_mm_processor_set)
            )
            if sgl_processor_cls:
                return sgl_processor_cls[0]
        raise ValueError(
            f"Cannot find corresponding multimodal processor registered in sglang for model type `{model_type}`"
        )

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        assert hasattr(hf_config, "vision_config")
        assert hasattr(hf_config, "text_config")
        self.vision_config = hf_config.vision_config
        self.text_config = hf_config.text_config
        self.hf_config = hf_config

        if vision_type := getattr(self.vision_config, "model_type"):
            self.inner = self._get_sgl_processor_cls(vision_type)(
                hf_config, server_args, _processor, *args, **kwargs
            )
        else:
            raise ValueError(
                f"Required `vision_config.model_type` is not found in hf_config: `{hf_config}`"
            )

    async def process_mm_data_async(self, *args, **kwargs):
        return await self.inner.process_mm_data_async(*args, **kwargs)
