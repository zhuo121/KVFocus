# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Inference-only LLaVa model compatible with HuggingFace weights."""

import math
# HACK(Shizhuo)
import time
import os
from sglang.srt.server_args import get_global_server_args
# 导入模块而非直接导入 embedding_cache 变量，避免 Python 模块导入顺序问题
from sglang.srt.managers import mm_utils
from sglang.srt.mem_cache.multimodal_cache import MultimodalCache

import re
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from torch import nn
from transformers import (
    CLIPVisionConfig,
    CLIPVisionModel,
    LlavaConfig,
    MistralConfig,
    Qwen2Config,
    SiglipVisionModel,
)
from transformers.models.auto.modeling_auto import AutoModel, AutoModelForCausalLM
from transformers.models.llava.modeling_llava import LlavaMultiModalProjector

# leave till last and symbol only in case circular import
import sglang.srt.models as sgl_models
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import general_mm_embed_routine
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.llama import LlamaForCausalLM
from sglang.srt.models.mistral import MistralForCausalLM
from sglang.srt.models.qwen2 import Qwen2ForCausalLM
from sglang.srt.multimodal.mm_utils import (
    get_anyres_image_grid_shape,
    unpad_image,
    unpad_image_shape,
)
from sglang.srt.utils import add_prefix, flatten_nested_list, logger


class LlavaBaseForCausalLM(nn.Module):
    def pad_input_ids(self, input_ids: List[int], image_inputs: MultimodalInputs):
        image_sizes = flatten_nested_list(
            [item.image_sizes for item in image_inputs.mm_items]
        )

        pad_values = [item.pad_value for item in image_inputs.mm_items]

        # hardcode for spatial_unpad + anyres
        if any(
            item.modality == Modality.MULTI_IMAGES or item.modality == Modality.VIDEO
            for item in image_inputs.mm_items
        ):
            image_aspect_ratio = "pad"
        else:
            image_aspect_ratio = "anyres"

        # HACK(Shizhuo)
        # 判定是否为视频模式：检查 mm_items 里的模态标记
        is_video = any(item.modality == Modality.VIDEO for item in image_inputs.mm_items)
        # 硬断言：如果找不到 image token，说明对已 padding 的 ids 又 padding 了一次
        img_tok = int(self.config.image_token_index)
        cnt_img_tok = input_ids.count(img_tok) if isinstance(input_ids, list) else 0
        if cnt_img_tok == 0:
            raise RuntimeError(f"[PAD_IN] image_token_index={img_tok} not found; pad_input_ids called on already-padded ids?")
        # ===========================================

        offset_list = []
        image_inputs.image_pad_len = []
        for image_idx, image_s in enumerate(image_sizes):
            # HACK(Shizhuo)
            # if len(image_sizes) > 16:
            if len(image_sizes) > 16 or is_video:
            # ===========================================
                # 2x2 pooling with stride 2
                new_image_feature_len = (
                    math.ceil(self.image_size / self.patch_size / 2) ** 2
                )
            else:
                new_image_feature_len = self.image_feature_len  # multi-image

            height = width = self.num_patches_per_side
            if "anyres" in image_aspect_ratio:
                num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                    image_s,
                    self.image_grid_pinpoints,
                    self.vision_tower.config.image_size,
                )
                h = num_patch_height * height
                w = num_patch_width * width
                new_h, new_w = unpad_image_shape(h, w, image_s)

                if "anyres_max" in self.config.image_aspect_ratio:
                    matched_anyres_max_num_patches = re.match(
                        r"anyres_max_(\d+)", self.config.image_aspect_ratio
                    )
                    if matched_anyres_max_num_patches:
                        max_num_patches = int(matched_anyres_max_num_patches.group(1))
                    # times = math.sqrt(h * w / (max_num_patches * unit**2))
                    times = math.sqrt(
                        new_h * new_w / (max_num_patches * self.image_feature_len)
                    )
                    if times > 1.1:
                        new_h = int(new_h // times)
                        new_w = int(new_w // times)
                new_image_feature_len += new_h * (new_w + 1)

            try:
                offset = input_ids.index(self.config.image_token_index)
            except ValueError:
                # HACK(Shizhuo): 原来是 offset=0，但 offset=0 会把 pad 插到开头，silent bug
                # 改成 raise 立即报错，排除"已 padding 再次 padding"的隐蔽错误
                raise RuntimeError(f"[PAD_IN] image_token_index not found in input_ids at frame={image_idx}")

            # HACK(Shizhuo): 修正 image_pad_len 为实际的 feature 长度（包含 image_newline）
            newline_len = getattr(self, "image_newline_token_len", 0)
            actual_len = new_image_feature_len + newline_len
            pad_value = pad_values[image_idx % len(pad_values)]
            # old_len + pad_len - 1, because we need to remove image_token_id
            input_ids = (
                input_ids[:offset]
                + [pad_value] * actual_len
                + input_ids[offset + 1 :]
            )
            # input_ids = (
            #     input_ids[:offset]
            #     + [pad_values[image_idx % len(pad_values)]]* new_image_feature_len
            #     + input_ids[offset + 1 :]
            # )
            # =================================================================
            offset_list.append(offset)
            image_inputs.image_pad_len.append(actual_len)

        image_inputs.image_offsets = offset_list
        return input_ids

    def encode_images(
        self, pixel_values: Union[torch.Tensor, List[torch.Tensor]]
    ) -> torch.Tensor:
        """
        encode images by vision tower and multimodal projector
        Args:
            pixel_values: torch.Tensor or List[torch.Tensor]: each tensor for an input image
        Returns:
            torch.Tensor: encoded image features from the input image; if multiple, flattened by seq_len axis
        """
        image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
        # NOTE: This is not memory efficient. (output_hidden_states=True) will save all the hidden stated.
        selected_image_feature = image_outputs.hidden_states[self.vision_feature_layer]
        if self.vision_feature_select_strategy in ["default", "patch"]:
            selected_image_feature = selected_image_feature[:, 1:]
        elif self.vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        else:
            raise ValueError(
                f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}"
            )
        image_features = self.multi_modal_projector(selected_image_feature)
        return image_features

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        # HACK(Shizhuo)
        # 检查 CacheFlex 开关 (确保只在 Extend 阶段且开启了功能时触发)
        server_args = get_global_server_args()
        mpc = getattr(server_args, "mm_process_config", None) or {}
        cacheflex_mode = mpc.get("cacheflex_mode", "")
        is_cacheflex_enabled = (
            forward_batch.forward_mode.is_extend()
            and server_args is not None
            and getattr(server_args, "enable_cacheflex", False)
            and cacheflex_mode != "fr"
        )
        cacheflex_cfgs = getattr(forward_batch, "cacheflex_configs", None)
        has_any_hit = False
        # 支持通过环境变量强制禁用 vision pruning
        disable_pruning_env = os.environ.get("SGLANG_CACHEFLEX_DISABLE_PRUNING", "0") == "1"

        if is_cacheflex_enabled and cacheflex_cfgs and not disable_pruning_env:
            for cfg in cacheflex_cfgs:
                if cfg and "sources" in cfg:
                    for src in cfg["sources"]:
                        if src.get("type") == "video_hit":
                            has_any_hit = True
                            break
                if has_any_hit:
                    break

        # [PROFILE] 1. 检查开关与初始化计时
        enable_profiling = os.environ.get("SGLANG_ENABLE_PROFILING", "false").lower() == "true"
        do_profile = enable_profiling and forward_batch.forward_mode.is_extend()
        
        t_start = 0.0
        t_pre_vision = t_post_vision = t_post_fusion = 0.0
        latency_cpu_preprocess = 0.0
        if do_profile:
            torch.cuda.synchronize()
            t_start = time.perf_counter()
            t_pre_vision = t_post_vision = t_post_fusion = t_start # Default
        # ================================================

        image_inputs = forward_batch.mm_inputs

        # HACK(Shizhuo)
        # [PROFILE] 提取 CPU 预处理时间 (Cross-Process Extraction)
        # SGLang 会把 MultimodalDataItem 序列化传过来
        if forward_batch.forward_mode.is_extend() and image_inputs:
            for im in image_inputs:
                if im:
                    for item in im.mm_items:
                        latency_cpu_preprocess += item.model_specific_data.get("cpu_preprocessing_latency", 0.0)
        # ==================================================

        if forward_batch.forward_mode.is_extend():
            # Clamp input ids. This is because the input_ids for the image tokens are
            # filled with the hash values of the image for the prefix matching in the radix attention.
            # There values are useless because their embeddings will be replaced by vision embeddings anyway.

            # HACK(Shizhuo) - 修复 Hash Token 匹配：将 clamp 改为取模映射，保留 hash 信息用于 prefix matching
            # input_inds.clamp_(min=0, max=self.config.vocab_size - 1)
            vocab = self.config.vocab_size
            # 只对异常范围做映射，避免改正常文本 token
            mask = (input_ids < 0) | (input_ids >= vocab)
            if mask.any():
                input_ids = input_ids.clone()
                input_ids[mask] = torch.remainder(input_ids[mask], vocab)
                input_ids[input_ids < 0] += vocab
            # ===========================================================

            # Embed text inputs
            input_embeds = self.language_model.model.embed_tokens(input_ids)

            # Got List[List[str]] extend it to List[str]
            # The length of the List should be equal to batch size
            modalities_list = []
            max_image_offset = []
            for im in image_inputs:
                if im:
                    modalities_list.extend([item.modality for item in im.mm_items])
                if im and im.image_offsets:
                    max_image_offset.append(
                        np.max(np.array(im.image_offsets) + np.array(im.image_pad_len))
                    )
                else:
                    max_image_offset.append(-1)

            start_positions = positions[forward_batch.extend_start_loc].cpu().numpy()
            need_vision = start_positions <= np.array(max_image_offset)


            # HACK(Shizhuo)
            # [PROFILE] 2. 视觉编码前的时间点 (用于计算 CPU 逻辑/Tensor 准备耗时)
            if do_profile:
                t_pre_vision = time.perf_counter()
            # ======================================

            if need_vision.any():
                bs = forward_batch.batch_size
                
                # HACK(Shizhuo)
                # =========================================================
                # BRANCH 1: CacheFlex 优化路径 (Vision Pruning)
                # =========================================================
                if is_cacheflex_enabled and has_any_hit:
                    extend_start_loc_cpu = forward_batch.extend_start_loc.cpu().numpy()
                    extend_seq_lens = forward_batch.extend_seq_lens.cpu().numpy()
                    prefix_lens_cpu = list(forward_batch.extend_prefix_lens_cpu)

                    # Two-way map: reuse_map (cached frames), valid_map (frames to compute)
                    valid_pixel_values = []
                    valid_image_sizes = []
                    valid_map = []  # [(b_i, img_i, image_offset, combined_hash or None)]
                    reuse_map = []  # [(b_i, img_idx, image_offset, combined_hash)]

                    total_frames = 0
                    in_window_frames = 0
                    marked_hit_frames = 0
                    cache_hit_frames = 0
                    to_compute_frames = 0

                    # Step 1: 遍历帧，分离 cached 和 to-compute
                    for i in range(bs):
                        if not need_vision[i]:
                            continue

                        seq_len = extend_seq_lens[i]
                        prefix_len = prefix_lens_cpu[i]

                        # 构建 hit hashes 集合 (从 scheduler 的 video_hit 标记)
                        # HACK(Shizhuo) - hash 规范化函数
                        def norm_hash(h):
                            if h is None:
                                return None
                            if isinstance(h, int):
                                return format(h, "x")
                            h = str(h).lower()
                            return h[2:] if h.startswith("0x") else h

                        skip_hashes = set()
                        if hasattr(forward_batch, "cacheflex_configs") and forward_batch.cacheflex_configs:
                            cfg = forward_batch.cacheflex_configs[i]
                            if cfg and "sources" in cfg:
                                for src in cfg["sources"]:
                                    if src.get("type") == "video_hit" and "hash" in src:
                                        skip_hashes.add(norm_hash(src["hash"]))

                        mm_item = image_inputs[i].mm_items[0]
                        current_frame_hashes = mm_item.model_specific_data.get("frame_hashes", [])

                        for img_idx, (offset, pad_len) in enumerate(zip(image_inputs[i].image_offsets, image_inputs[i].image_pad_len)):
                            total_frames += 1

                            # HACK(Shizhuo) - 修复 Vision Pruning: 添加 prefix/window 过滤（复刻原生分支逻辑）
                            if offset + pad_len <= prefix_len:
                                continue
                            if offset >= prefix_len + seq_len:
                                break
                            in_window_frames += 1

                            curr_hash = current_frame_hashes[img_idx] if img_idx < len(current_frame_hashes) else None
                            curr_hash = norm_hash(curr_hash)

                            combined_hash = None
                            if curr_hash:
                                hash_int = int(curr_hash, 16)
                                combined_hash = MultimodalCache.combine_hashes([hash_int])

                            if curr_hash and curr_hash in skip_hashes:
                                marked_hit_frames += 1
                                # Cache hit: 直接 get 避免 has+get 的 TOCTOU
                                if mm_utils.embedding_cache is not None:
                                    cached_feature = mm_utils.embedding_cache.get([], combined_hash=combined_hash)
                                    if cached_feature is not None:
                                        cache_hit_frames += 1
                                        reuse_map.append((i, img_idx, offset, combined_hash))
                                    else:
                                        to_compute_frames += 1
                                        valid_pixel_values.append(mm_item.feature[img_idx])
                                        valid_image_sizes.append(mm_item.image_sizes[img_idx])
                                        valid_map.append((i, img_idx, offset, combined_hash))
                                else:
                                    to_compute_frames += 1
                                    valid_pixel_values.append(mm_item.feature[img_idx])
                                    valid_image_sizes.append(mm_item.image_sizes[img_idx])
                                    valid_map.append((i, img_idx, offset, combined_hash))
                            else:
                                to_compute_frames += 1
                                valid_pixel_values.append(mm_item.feature[img_idx])
                                valid_image_sizes.append(mm_item.image_sizes[img_idx])
                                # HACK(Shizhuo) 非 hit 帧也都生成 combined_hash，算完后能写入 cache
                                valid_map.append((i, img_idx, offset, combined_hash))

                    # Step 2: 初始化 scatter 变量（已在 Step1 前初始化）

                    # Step 3: 对 valid_map 运行 Vision Encoder + Pooling + Scatter + Cache
                    if len(valid_pixel_values) > 0:
                        # [PROFILE] Vision Start
                        if do_profile:
                            torch.cuda.synchronize()
                            t_pre_vision = time.perf_counter()

                        if valid_pixel_values[0].ndim == 4:
                            concat_images = torch.tensor(np.concatenate(valid_pixel_values, axis=0), device=self.vision_tower.device)
                            image_features = self.encode_images(concat_images)
                            split_sizes = [image.shape[0] for image in valid_pixel_values]
                            image_features = torch.split(image_features, split_sizes, dim=0)
                        else:
                            pixel_values_tensor = torch.tensor(np.array(valid_pixel_values), device=self.vision_tower.device)
                            image_features = self.encode_images(pixel_values_tensor)

                        # [PROFILE] Vision End
                        if do_profile:
                            torch.cuda.synchronize()
                            t_post_vision = time.perf_counter()

                        # Pooling 处理 valid_map
                        processed_features = []
                        if self.mm_patch_merge_type.startswith("spatial"):
                            height = width = self.num_patches_per_side
                            for idx, image_feature in enumerate(image_features):
                                b_idx, _, _, combined_hash = valid_map[idx]
                                current_modality = image_inputs[b_idx].mm_items[0].modality
                                current_image_size = valid_image_sizes[idx]
                                if current_modality == Modality.IMAGE:
                                    image_aspect_ratio = self.config.image_aspect_ratio
                                elif current_modality == Modality.MULTI_IMAGES or current_modality == Modality.VIDEO:
                                    image_aspect_ratio = "pad"
                                
                                if (image_feature.shape[0] > 1 and "anyres" in image_aspect_ratio and current_modality == Modality.IMAGE):
                                    base_image_feature = image_feature[0]
                                    image_feature = image_feature[1:]
                                    assert height * width == base_image_feature.shape[0]
                                    if "anyres_max" in image_aspect_ratio:
                                        matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
                                        if matched_anyres_max_num_patches:
                                            max_num_patches = int(matched_anyres_max_num_patches.group(1))
                                    if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                                        vision_tower_image_size = self.image_size
                                        try:
                                            num_patch_width, num_patch_height = (get_anyres_image_grid_shape(current_image_size[0], self.config.image_grid_pinpoints, vision_tower_image_size))
                                        except Exception as e:
                                            # print(f"Error: {e}")
                                            num_patch_width, num_patch_height = 2, 2
                                        image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                                    else:
                                        image_feature = image_feature.view(2, 2, height, width, -1)

                                    if "unpad" in self.mm_patch_merge_type:
                                        unit = image_feature.shape[2]
                                        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                                        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                                        image_feature = unpad_image(image_feature, current_image_size[0])
                                        if "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
                                            c, h, w = image_feature.shape
                                            times = math.sqrt(h * w / (max_num_patches * unit**2))
                                            if times > 1.1:
                                                image_feature = image_feature[None]
                                                image_feature = nn.functional.interpolate(image_feature, [int(h // times), int(w // times)], mode="bilinear",)[0]
                                        image_feature = torch.cat((image_feature, self.language_model.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1)), dim=-1)
                                        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                                    else:
                                        image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                                        image_feature = image_feature.flatten(0, 3)
                                    image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                                    image_feature = image_feature.unsqueeze(0)
                                else:
                                    if current_modality == Modality.VIDEO:
                                        # [FIXED] 维度修复：确保 view 能正确处理单帧
                                        image_feature = image_feature.view(-1, height, width, image_feature.shape[-1])
                                        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
                                        h_s, w_s = image_feature.shape[2:]
                                        scaled_shape = [math.ceil(h_s / 2), math.ceil(w_s / 2)]
                                        image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode="bilinear")
                                        image_feature = image_feature.flatten(2).transpose(1, 2).contiguous()
                                    if "unpad" in self.mm_patch_merge_type:
                                        image_feature = torch.cat((image_feature, self.language_model.model.image_newline[None, None].expand(image_feature.shape[0], 1, image_feature.shape[-1]),), dim=1)
                                
                                processed_features.append(image_feature)

                        # Scatter: valid_map 写入 input_embeds + 缓存
                        for feat_idx, (b_i, _, image_offset, combined_hash) in enumerate(valid_map):
                            start_idx = extend_start_loc_cpu[b_i]
                            seq_len = extend_seq_lens[b_i]
                            prefix_len = prefix_lens_cpu[b_i]

                            # 准备特征 (设备标准化、边界裁剪) - 先获取 feature
                            feature = processed_features[feat_idx]
                            if feature.ndim == 3:
                                feature = feature.squeeze(0)
                            feature = feature.to(device=input_embeds.device, dtype=input_embeds.dtype)
                            feat_len = feature.shape[0]
                            pad_len = image_inputs[b_i].image_pad_len[img_idx] if hasattr(image_inputs[b_i], 'image_pad_len') else feat_len

                            # 验证 packed_off 在范围内
                            global_off = image_offset
                            packed_off = global_off - prefix_len

                            # 关键断言：packed_off 必须落在 [0, seq_len)
                            assert 0 <= packed_off < seq_len, f"packed_off out of range: {packed_off} not in [0, {seq_len})"

                            # 保存完整特征用于缓存
                            full_feature_for_cache = feature

                            # 边界裁剪 (裁剪后的才写入 input_embeds)
                            input_offset = image_offset - prefix_len
                            left_idx = start_idx + input_offset
                            right_idx = left_idx + feature.shape[0]

                            feature_to_scatter = feature
                            if input_offset < 0:
                                feature_to_scatter = feature_to_scatter[-input_offset:]
                                left_idx = start_idx
                            if right_idx > start_idx + seq_len:
                                feature_to_scatter = feature_to_scatter[: start_idx + seq_len - right_idx]
                                right_idx = start_idx + seq_len

                            # 边界检查
                            seg_len = right_idx - left_idx
                            assert seg_len > 0, f"empty range"
                            assert feature_to_scatter is not None, f"feature is None"
                            assert feature_to_scatter.shape[0] == seg_len, f"len mismatch seg={seg_len} feat_len={feature_to_scatter.shape[0]}"
                            assert input_embeds.device == feature_to_scatter.device, f"device mismatch"

                            input_embeds[left_idx:right_idx] = feature_to_scatter

                            # 写入缓存 (保存完整特征)
                            if combined_hash is not None and mm_utils.embedding_cache is not None:
                                mm_utils.embedding_cache.set(combined_hash, full_feature_for_cache.clone())
                    else:
                        if do_profile:
                            t_post_vision = t_pre_vision

                    # Step 4: reuse_map 回填 - 从缓存读取 + scatter
                    for (b_i, img_idx, image_offset, combined_hash) in reuse_map:
                        start_idx = extend_start_loc_cpu[b_i]
                        seq_len = extend_seq_lens[b_i]
                        prefix_len = prefix_lens_cpu[b_i]

                        if mm_utils.embedding_cache is not None:
                            # get 时传入 combined_hash，且传入空列表让内部直接使用 combined_hash
                            cached_feature = mm_utils.embedding_cache.get([], combined_hash=combined_hash)
                            if cached_feature is not None:
                                # Shape 规整: 确保是 [T,H]
                                if cached_feature.ndim == 3:
                                    assert cached_feature.shape[0] == 1, f"unexpected cached_feature shape: {cached_feature.shape}"
                                    cached_feature = cached_feature[0]
                                assert cached_feature.ndim == 2, f"cached_feature must be [T,H], got {cached_feature.shape}"
                                cached_feature = cached_feature.to(device=input_embeds.device, dtype=input_embeds.dtype)

                                # 边界裁剪 (同步裁剪 cached_feature)
                                input_offset = image_offset - prefix_len
                                left_idx = start_idx + input_offset
                                right_idx = left_idx + cached_feature.shape[0]

                                if input_offset < 0:
                                    cached_feature = cached_feature[-input_offset:]
                                    left_idx = start_idx
                                if right_idx > start_idx + seq_len:
                                    cached_feature = cached_feature[: start_idx + seq_len - right_idx]
                                    right_idx = start_idx + seq_len

                                if right_idx > left_idx:
                                    # 边界检查
                                    seg_len = right_idx - left_idx
                                    assert seg_len > 0, f"empty range"
                                    assert cached_feature is not None, f"cached_feature is None"
                                    assert cached_feature.shape[0] == seg_len, f"len mismatch seg={seg_len} feat_len={cached_feature.shape[0]}"
                                    assert input_embeds.device == cached_feature.device, f"device mismatch"

                                    input_embeds[left_idx:right_idx] = cached_feature

                # =========================================================
                # BRANCH 2: SGLang 原生逻辑
                # =========================================================
                else:
                    pixel_values = flatten_nested_list(
                        [
                            [item.feature for item in image_inputs[i].mm_items]
                            for i in range(bs)
                            if need_vision[i]
                        ]
                    )
                    image_sizes = [
                        flatten_nested_list(
                            [item.image_sizes for item in image_inputs[i].mm_items]
                        )
                        for i in range(bs)
                        if need_vision[i]
                    ]

                    ########## Encode Image ########
                    # HACK(Shizhuo)
                    # [PROFILE] 记录 Vision 开始时刻
                    if do_profile:
                        torch.cuda.synchronize()
                        t_pre_vision = time.perf_counter()
                    # =====================================

                    if pixel_values[0].ndim == 4:
                        # llava-hd: BS, num_patch, C=3, H=336, W=336, num_patch obtained from process_images
                        np.concatenate(pixel_values, axis=0)
                        # ndim=4
                        concat_images = torch.tensor(
                            np.concatenate(pixel_values, axis=0),
                            device=self.vision_tower.device,
                        )
                        image_features = self.encode_images(concat_images)
                        split_sizes = [image.shape[0] for image in pixel_values]
                        image_features = torch.split(image_features, split_sizes, dim=0)
                        # hd image_features: BS, num_patch, 576, 4096
                    else:
                        # normal pixel: BS, C=3, H=336, W=336
                        pixel_values = torch.tensor(
                            np.array(pixel_values), device=self.vision_tower.device
                        )
                        image_features = self.encode_images(pixel_values)
                        # image_features: BS, 576, 4096
                    
                    # HACK(Shizhuo)
                    # [PROFILE] Vision 结束
                    if do_profile:
                        torch.cuda.synchronize()
                        t_post_vision = time.perf_counter()
                    # =====================================

                    if self.mm_patch_merge_type.startswith("spatial"):
                        new_image_features = []
                        height = width = self.num_patches_per_side
                        for image_idx, image_feature in enumerate(image_features):
                            if modalities_list[image_idx] == Modality.IMAGE:
                                image_aspect_ratio = (
                                    self.config.image_aspect_ratio
                                )  # single image
                            elif (
                                modalities_list[image_idx] == Modality.MULTI_IMAGES
                                or modalities_list[image_idx] == Modality.VIDEO
                            ):
                                image_aspect_ratio = "pad"  # multi image
                            # image_aspect_ratio = (
                            #     "anyres" if len(image_sizes[image_idx]) == 1 else "pad"
                            # )
                            if (
                                image_feature.shape[0] > 1
                                and "anyres" in image_aspect_ratio
                                and modalities_list[image_idx] == Modality.IMAGE
                            ):
                                base_image_feature = image_feature[0]
                                image_feature = image_feature[1:]
                                assert height * width == base_image_feature.shape[0]

                                if "anyres_max" in image_aspect_ratio:
                                    matched_anyres_max_num_patches = re.match(
                                        r"anyres_max_(\d+)", image_aspect_ratio
                                    )
                                    if matched_anyres_max_num_patches:
                                        max_num_patches = int(
                                            matched_anyres_max_num_patches.group(1)
                                        )

                                if (
                                    image_aspect_ratio == "anyres"
                                    or "anyres_max" in image_aspect_ratio
                                ):
                                    vision_tower_image_size = self.image_size
                                    try:
                                        num_patch_width, num_patch_height = (
                                            get_anyres_image_grid_shape(
                                                image_sizes[image_idx][0],
                                                self.config.image_grid_pinpoints,
                                                vision_tower_image_size,
                                            )
                                        )
                                    except Exception as e:
                                        num_patch_width, num_patch_height = 2, 2
                                    image_feature = image_feature.view(
                                        num_patch_height, num_patch_width, height, width, -1
                                    )
                                else:
                                    image_feature = image_feature.view(
                                        2, 2, height, width, -1
                                    )

                                # (
                                #     num_patch_width,
                                #     num_patch_height,
                                # ) = get_anyres_image_grid_shape(
                                #     image_sizes[image_idx][0],
                                #     self.image_grid_pinpoints,
                                #     self.vision_tower.config.image_size,
                                # )

                                # image_feature = image_feature.view(
                                #     num_patch_height, num_patch_width, height, width, -1
                                # )

                                if "unpad" in self.mm_patch_merge_type:
                                    unit = image_feature.shape[2]
                                    image_feature = image_feature.permute(
                                        4, 0, 2, 1, 3
                                    ).contiguous()
                                    image_feature = image_feature.flatten(1, 2).flatten(
                                        2, 3
                                    )
                                    image_feature = unpad_image(
                                        image_feature, image_sizes[image_idx][0]
                                    )
                                    if (
                                        "anyres_max" in image_aspect_ratio
                                        and matched_anyres_max_num_patches
                                    ):
                                        c, h, w = image_feature.shape
                                        times = math.sqrt(
                                            h * w / (max_num_patches * unit**2)
                                        )
                                        if times > 1.1:
                                            image_feature = image_feature[None]
                                            image_feature = nn.functional.interpolate(
                                                image_feature,
                                                [int(h // times), int(w // times)],
                                                mode="bilinear",
                                            )[0]
                                    image_feature = torch.cat(
                                        (
                                            image_feature,
                                            self.language_model.model.image_newline[
                                                :, None, None
                                            ].expand(*image_feature.shape[:-1], 1),
                                        ),
                                        dim=-1,
                                    )
                                    image_feature = image_feature.flatten(1, 2).transpose(
                                        0, 1
                                    )
                                else:
                                    image_feature = image_feature.permute(
                                        0, 2, 1, 3, 4
                                    ).contiguous()
                                    image_feature = image_feature.flatten(0, 3)
                                image_feature = torch.cat(
                                    (base_image_feature, image_feature), dim=0
                                )
                                image_feature = image_feature.unsqueeze(0)
                            else:
                                if modalities_list[image_idx] == Modality.VIDEO:  # video
                                    # 2x2 pooling
                                    num_of_frames = image_feature.shape[0]
                                    image_feature = image_feature.view(
                                        num_of_frames, height, width, -1
                                    )
                                    image_feature = image_feature.permute(
                                        0, 3, 1, 2
                                    ).contiguous()  # N, C, H, W
                                    height, weight = image_feature.shape[2:]
                                    scaled_shape = [
                                        math.ceil(height / 2),
                                        math.ceil(weight / 2),
                                    ]
                                    image_feature = nn.functional.interpolate(
                                        image_feature, size=scaled_shape, mode="bilinear"
                                    )
                                    image_feature = (
                                        image_feature.flatten(2)
                                        .transpose(1, 2)
                                        .contiguous()
                                    )  # N, C, H*W
                                if "unpad" in self.mm_patch_merge_type:
                                    image_feature = torch.cat(
                                        (
                                            image_feature,
                                            # Expand to (bs, 1, hidden_dim) and concat at the end of the image tokens
                                            self.language_model.model.image_newline[
                                                None, None
                                            ].expand(
                                                image_feature.shape[0],
                                                1,
                                                image_feature.shape[-1],
                                            ),
                                        ),
                                        dim=1,
                                    )

                            new_image_features.append(image_feature)
                        image_features = new_image_features

                    # Fill in the placeholder for the image
                    extend_start_loc_cpu = forward_batch.extend_start_loc.cpu().numpy()
                    extend_seq_lens = forward_batch.extend_seq_lens.cpu().numpy()
                    prefix_lens_cpu = forward_batch.extend_prefix_lens_cpu
                    pt = 0
                    for i in range(bs):
                        if not need_vision[i]:
                            continue

                        start_idx = extend_start_loc_cpu[i]
                        seq_len = extend_seq_lens[i]
                        prefix_len = prefix_lens_cpu[i]

                        # Multiple images
                        for image_idx, image_offset in enumerate(
                            image_inputs[i].image_offsets
                        ):
                            if (
                                image_offset + image_inputs[i].image_pad_len[image_idx]
                                <= prefix_len
                            ):
                                continue
                            if image_offset >= prefix_len + seq_len:
                                break

                            tmp_image_feature = image_features[pt][image_idx]
                            pad_len = tmp_image_feature.shape[0]

                            input_offset = image_offset - prefix_len
                            left_idx = start_idx + input_offset
                            right_idx = left_idx + pad_len
                            assert right_idx > start_idx
                            if input_offset < 0:
                                left_idx = start_idx
                                tmp_image_feature = tmp_image_feature[-input_offset:]
                            if right_idx > start_idx + seq_len:
                                tmp_image_feature = tmp_image_feature[
                                    : start_idx + seq_len - right_idx
                                ]
                                right_idx = start_idx + seq_len
                            try:
                                input_embeds[left_idx:right_idx] = tmp_image_feature
                            except RuntimeError as e:
                                pass
                                # print(f"RuntimeError in image encoding: {e}")
                                # print(f"{image_embeds.shape=},{tmp_image_feature.shape=}")
                        pt += 1

            # HACK(Shizhuo)
            else:
                # No vision needed
                if do_profile:
                    t_post_vision = t_pre_vision # 相当于 vision 耗时为 0

            # [PROFILE] Fusion 结束
            if do_profile:
                torch.cuda.synchronize()
                t_post_fusion = time.perf_counter()

            # LLM Layers
            # HACK(Shizhuo) 使用 CUDA Event 测量 LLM Prefill 时间
            if do_profile:
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start_ev.record()

            ret = self.language_model(
                input_ids, positions, forward_batch, input_embeds=input_embeds
            )

            if do_profile:
                end_ev.record()
                torch.cuda.synchronize()
                llm_time_ms = start_ev.elapsed_time(end_ev)

            # [PROFILE] 3. 统计并"返回" (通过 Print JSON 供外部解析)
            forward_batch.profiling_metadata = {
                "latency_cpu_preprocessing_transfer": latency_cpu_preprocess
            }
            if do_profile:
                torch.cuda.synchronize()
                t_end = time.perf_counter()
                
                # 计算细分时延
                latency_vision = t_post_vision - t_pre_vision if need_vision.any() else 0.0
                latency_fusion = t_post_fusion - t_post_vision if need_vision.any() else 0.0
                latency_llm = t_end - t_post_fusion
                
                forward_batch.profiling_metadata.update({
                    "latency_vision_encoder": latency_vision,
                    "latency_fusion_embedding": latency_fusion,
                    "latency_llm_prefill": latency_llm
                })
            
            return ret             

            # return self.language_model(
            #     input_ids, positions, forward_batch, input_embeds=input_embeds
            # )
            # =================================================================   

        elif forward_batch.forward_mode.is_decode():
            return self.language_model(input_ids, positions, forward_batch)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # Load clip vision model by cfg['mm_vision_tower']:
        # huggingface_name or path_of_clip_relative_to_llava_model_dir
        # We put the initialization here instead of __init__ to allow it being reused by other subclasses.
        vision_path = self.config.mm_vision_tower
        if "clip" in vision_path:
            self.vision_tower = CLIPVisionModel.from_pretrained(
                vision_path, torch_dtype=torch.float16
            ).cuda()
        elif "siglip" in vision_path:
            self.vision_tower = SiglipVisionModel.from_pretrained(
                vision_path, torch_dtype=torch.float16
            ).cuda()
            # Siglip needs all feature tokens
            self.config.mm_vision_select_feature = "full"
        self.vision_tower.eval()

        self.vision_feature_layer = self.config.mm_vision_select_layer
        self.vision_feature_select_strategy = self.config.mm_vision_select_feature
        self.image_size = self.vision_tower.config.image_size
        self.patch_size = self.vision_tower.config.patch_size

        self.mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
        self.image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
        self.image_grid_pinpoints = getattr(self.config, "image_grid_pinpoints", None)

        self.image_feature_len = int((self.image_size // self.patch_size) ** 2)
        if (
            self.vision_feature_select_strategy == "patch"
            or self.vision_feature_select_strategy == "full"
        ):
            pass
        elif self.vision_feature_select_strategy == "cls_patch":
            self.image_feature_len += 1
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")

        # load mm_projector
        projector_weights = {
            "model.mm_projector.0": "multi_modal_projector.linear_1",
            "model.mm_projector.2": "multi_modal_projector.linear_2",
            "model.vision_tower.vision_tower": "vision_tower",
            # Update the vision tower weights if we find them in the checkpoint (it may be finetuned).
            "model.image_newline": "language_model.model.image_newline",
        }
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "projector" in name or "vision_tower" in name or "image_newline" in name:
                for weight_name, param_name in projector_weights.items():
                    if weight_name in name:
                        name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            else:
                self.language_model.load_weights([(name, loaded_weight)])

    @property
    def num_patches_per_side(self):
        return self.image_size // self.patch_size


class LlavaLlamaForCausalLM(LlavaBaseForCausalLM):
    def __init__(
        self,
        config: LlavaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        self.vision_tower = None
        self.config.vision_config.hidden_size = config.mm_hidden_size
        self.config.text_config.hidden_size = config.hidden_size

        self.multi_modal_projector = LlavaMultiModalProjector(config)
        self.language_model = LlamaForCausalLM(
            config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )
        # HACK(Shizhuo): 记录 image_newline 的 token 数量，用于修正 image_pad_len
        # 问题：unpad 模式下 vision encoder 输出 patches + image_newline，
        #      但 pad_input_ids 只记录了 patches 数量，导致 image_pad_len 少了 1
        # 修复：记录 image_newline_token_len，实际长度 = patches + image_newline_token_len
        self.image_newline_token_len = 0
        if "unpad" in getattr(config, "mm_patch_merge_type", ""):
            self.language_model.model.image_newline = nn.Parameter(
                torch.empty(config.text_config.hidden_size, dtype=torch.float16)
            )
            # HACK(Shizhuo): image_newline 目前固定为 1 个 token
            # expand 时使用 [image_feature.shape[0], 1, ...] 确认只有 1 个
            self.image_newline_token_len = 1


class LlavaQwenForCausalLM(LlavaBaseForCausalLM):
    def __init__(
        self,
        config: LlavaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        self.vision_tower = None

        if getattr(self.config, "vision_config", None) is None:
            self.config.vision_config = CLIPVisionConfig(self.config.mm_vision_tower)
        if getattr(self.config, "text_config", None) is None:
            self.config.text_config = Qwen2Config(self.config._name_or_path)

        self.config.vision_config.hidden_size = config.mm_hidden_size
        self.config.text_config.hidden_size = config.hidden_size

        if getattr(self.config, "projector_hidden_act", None) is None:
            self.config.projector_hidden_act = "gelu"
        if getattr(self.config, "image_token_index", None) is None:
            self.config.image_token_index = 151646

        self.multi_modal_projector = LlavaMultiModalProjector(config)
        self.language_model = Qwen2ForCausalLM(
            config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )
        # HACK(Shizhuo): 记录 image_newline 的 token 数量，用于修正 image_pad_len
        # 问题：unpad 模式下 vision encoder 输出 patches + image_newline，
        #      但 pad_input_ids 只记录了 patches 数量，导致 image_pad_len 少了 1
        # 修复：记录 image_newline_token_len，实际长度 = patches + image_newline_token_len
        self.image_newline_token_len = 0
        if "unpad" in getattr(config, "mm_patch_merge_type", ""):
            self.language_model.model.image_newline = nn.Parameter(
                torch.empty(config.text_config.hidden_size, dtype=torch.float16)
            )
            # HACK(Shizhuo): image_newline 目前固定为 1 个 token
            # expand 时使用 [image_feature.shape[0], 1, ...] 确认只有 1 个
            self.image_newline_token_len = 1


class LlavaMistralForCausalLM(LlavaBaseForCausalLM):
    def __init__(
        self,
        config: LlavaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        self.vision_tower = None

        if getattr(self.config, "vision_config", None) is None:
            self.config.vision_config = CLIPVisionConfig(self.config.mm_vision_tower)
        if getattr(self.config, "text_config", None) is None:
            self.config.text_config = MistralConfig(self.config._name_or_path)

        self.config.vision_config.hidden_size = config.mm_hidden_size
        self.config.text_config.hidden_size = config.hidden_size

        if getattr(self.config, "projector_hidden_act", None) is None:
            self.config.projector_hidden_act = "gelu"
        if getattr(self.config, "image_token_index", None) is None:
            self.config.image_token_index = 32000

        self.multi_modal_projector = LlavaMultiModalProjector(config)
        self.language_model = MistralForCausalLM(
            config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )
        # HACK(Shizhuo): 记录 image_newline 的 token 数量，用于修正 image_pad_len
        # 问题：unpad 模式下 vision encoder 输出 patches + image_newline，
        #      但 pad_input_ids 只记录了 patches 数量，导致 image_pad_len 少了 1
        # 修复：记录 image_newline_token_len，实际长度 = patches + image_newline_token_len
        self.image_newline_token_len = 0
        if "unpad" in getattr(config, "mm_patch_merge_type", ""):
            self.language_model.model.image_newline = nn.Parameter(
                torch.empty(config.text_config.hidden_size, dtype=torch.float16)
            )
            # HACK(Shizhuo): image_newline 目前固定为 1 个 token
            # expand 时使用 [image_feature.shape[0], 1, ...] 确认只有 1 个
            self.image_newline_token_len = 1


class LlavaForConditionalGeneration(LlavaBaseForCausalLM):
    """
    An adaptor class to enable support for multiple mmlm such as mistral-community/pixtral-12b
    It follows the structure of (vision_tower, multi_modal_projector, language_model)

    Once a model config is loaded, text_config and vision_config will be extracted, and
    LlavaForConditionalGeneration will load the language_model and vision_tower models
    according to config.
    """

    MULTIMODAL_PROJECTOR_TYPE = LlavaMultiModalProjector

    @property
    def dtype(self):
        return self.torch_dtype

    def pad_input_ids(self, input_ids: List[int], image_inputs: MultimodalInputs):
        if hasattr(self.vision_tower, "pad_input_ids"):
            return self.vision_tower.pad_input_ids(input_ids, image_inputs)
        else:
            return super().pad_input_ids(input_ids, image_inputs)

    def _get_sgl_model_cls(self, config, auto_model_type: Type[AutoModel] = AutoModel):
        """
        Get the SGLang model implementation class according to config.

        Args:
            config: The config object of the model.
            auto_model_type: The type of the auto model.

        Returns:
            The SGLang model implementation class.
        """
        config_cls_name = config.__class__.__name__
        arch_name_mapping = self._config_cls_name_to_arch_name_mapping(auto_model_type)
        if arch := arch_name_mapping.get(config_cls_name):
            if isinstance(arch, tuple):
                arch = arch[0]
                logger.warning(
                    f"Multiple {auto_model_type.__name__} models found for submodule config `{config_cls_name}`, defaulting to [0]: {arch.__name__}"
                )
            try:
                return sgl_models.registry.ModelRegistry.resolve_model_cls(arch)[0]
            except Exception as e:
                raise ValueError(
                    f"{auto_model_type.__name__} found a corresponding model `{arch}` for config class `{config_cls_name}`, but failed to load it from SGLang ModelRegistry. \n{e}"
                )
        else:
            raise ValueError(
                f"{auto_model_type.__name__} cannot find a corresponding model for config class `{config_cls_name}`"
            )

    @lru_cache
    def _config_cls_name_to_arch_name_mapping(
        self, auto_model_type: Type[AutoModel]
    ) -> Dict[str, str]:
        mapping = {}
        for config_cls in auto_model_type._model_mapping.keys():
            archs = auto_model_type._model_mapping.get(config_cls, None)
            if archs is not None:
                if isinstance(archs, tuple):
                    mapping[config_cls.__name__] = tuple(
                        arch.__name__ for arch in archs
                    )
                else:
                    mapping[config_cls.__name__] = archs.__name__
        return mapping

    def __init__(
        self,
        config: LlavaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        assert hasattr(config, "text_config")
        assert hasattr(config, "vision_config")
        self.config = config
        self.text_config = self.config.text_config
        self.vision_config = self.config.vision_config
        self.torch_dtype = getattr(self.config, "torch_dtype")

        if not getattr(self.text_config, "torch_dtype"):
            self.text_config.torch_dtype = self.torch_dtype
        if not getattr(self.vision_config, "torch_dtype"):
            self.vision_config.torch_dtype = self.torch_dtype

        if not hasattr(self.config, "vocab_size"):
            self.config.vocab_size = self.text_config.vocab_size
        if not hasattr(self.config, "image_aspect_ratio"):
            self.config.image_aspect_ratio = "anyres"
        if not hasattr(self.config, "image_grid_pinpoints"):
            # from transformers.models.llava_onevision.configuration_llava_onevision import LlavaOnevisionConfig
            # self.config.image_grid_pinpoints = LlavaOnevisionConfig().image_grid_pinpoints
            self.config.image_grid_pinpoints = [
                [96, 96],
                [224, 224],
                [384, 384],
                [512, 512],
                [768, 768],
                [1024, 1024],
            ]
        if not hasattr(self.config, "mm_patch_merge_type"):
            self.config.mm_patch_merge_type = "flat"
        if not hasattr(self.config, "image_token_index"):
            self.config.image_token_index = 10
        if not hasattr(self.config, "projector_hidden_act"):
            self.config.projector_hidden_act = "gelu"

        self.vision_feature_layer = getattr(self.config, "vision_feature_layer", -1)
        self.vision_feature_select_strategy = getattr(
            self.config, "vision_feature_select_strategy", "full"
        )
        self.image_size = self.vision_config.image_size
        self.patch_size = self.vision_config.patch_size

        self.mm_patch_merge_type = self.config.mm_patch_merge_type
        self.image_aspect_ratio = self.config.image_aspect_ratio
        self.image_grid_pinpoints = self.config.image_grid_pinpoints

        self.image_feature_len = int((self.image_size // self.patch_size) ** 2)

        self.multi_modal_projector = self.MULTIMODAL_PROJECTOR_TYPE(config)

        language_model_cls = self._get_sgl_model_cls(
            self.text_config, AutoModelForCausalLM
        )
        vision_model_cls = self._get_sgl_model_cls(self.vision_config, AutoModel)
        self.language_model = language_model_cls(
            self.text_config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )
        self.vision_tower = vision_model_cls(
            self.vision_config,
            quant_config=quant_config,
            prefix=add_prefix("vision_tower", prefix),
        )

        # HACK(Shizhuo): 记录 image_newline 的 token 数量，用于修正 image_pad_len
        # 问题：unpad 模式下 vision encoder 输出 patches + image_newline，
        #      但 pad_input_ids 只记录了 patches 数量，导致 image_pad_len 少了 1
        # 修复：记录 image_newline_token_len，实际长度 = patches + image_newline_token_len
        self.image_newline_token_len = 0
        if "unpad" in getattr(self.config, "mm_patch_merge_type", ""):
            self.language_model.model.image_newline = nn.Parameter(
                torch.empty(self.text_config.hidden_size, dtype=self.torch_dtype)
            )
            # HACK(Shizhuo): image_newline 目前固定为 1 个 token
            self.image_newline_token_len = 1

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """Extract features from image inputs.

        Args:
            items: List of MultimodalDataItem objects containing image data
                Note that an item can be either "image" or "multi-images"

        Returns:
            torch.Tensor: features from image inputs, concatenated
        """
        features = []
        for item in items:
            # in each item, we assume pixel_values is always batched
            pixel_values, image_sizes = item.feature, item.image_sizes
            image_outputs = self.vision_tower(
                pixel_values, image_sizes, output_hidden_states=True
            )
            selected_image_feature = image_outputs.hidden_states[
                self.vision_feature_layer
            ]

            if self.vision_feature_select_strategy in ["default", "patch"]:
                selected_image_feature = selected_image_feature[:, 1:]
            elif self.vision_feature_select_strategy == "full":
                selected_image_feature = selected_image_feature
            else:
                raise ValueError(
                    f"Unexpected select feature: {self.vision_feature_select_strategy}"
                )
            features.append(
                self.multi_modal_projector(selected_image_feature.squeeze(0))
            )
        ret = torch.cat(features, dim=0)
        return ret

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
    ):
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            get_embedding=get_embedding,
            language_model=self.language_model,
            data_embedding_funcs={
                Modality.IMAGE: self.get_image_feature,
            },
            placeholder_tokens=None,  # using mm_item.pad_value
            positions=positions,
        )

        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights for LlavaForConditionalGeneration.

        Unlike the base class implementation, this one doesn't need to handle
        weight name remapping as the weights are already properly structured with
        'language_model' and 'vision_tower' prefixes in the safetensors files.
        """
        if (
            self.vision_feature_select_strategy == "patch"
            or self.vision_feature_select_strategy == "full"
        ):
            pass
        elif self.vision_feature_select_strategy == "cls_patch":
            self.image_feature_len += 1
        else:
            raise ValueError(
                f"Unexpected select feature: {self.vision_feature_select_strategy}"
            )

        # Create dictionaries for direct parameter loading
        params_dict = dict(self.named_parameters())

        # Load weights directly without remapping
        for name, loaded_weight in weights:
            for part in ("language_model", "vision_tower"):
                if name.startswith(part):
                    name = name[len(part + ".") :]
                    getattr(self, part).load_weights([(name, loaded_weight)])
                    break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = [
    LlavaLlamaForCausalLM,
    LlavaQwenForCausalLM,
    LlavaMistralForCausalLM,
    LlavaForConditionalGeneration,
]
