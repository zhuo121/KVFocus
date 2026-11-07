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

# Adapted from llama2.py
# Modify details for the adaptation of Qwen2 model.
"""Inference-only Qwen2 model compatible with HuggingFace weights."""
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import nn

from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.activation import SiluAndMul
# HACK(Shizhuo): mem profile helper
from sglang.srt.layers.attention.cacheflex_utils import cacheflex_mem_snapshot
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    kv_cache_scales_loader,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, make_layers

Qwen2Config = None

# HACK(Shizhuo)
# ============================================
import logging
logger = logging.getLogger(__name__)

# =========================================================================
# HACK(Shizhuo) [DEBUG TOOL] Tensor Dump Helper
# =========================================================================
# HACK(Shizhuo): unified dump (gated by Category.LAYER_TENSORS)
import os
from sglang.srt.utils import cacheflex_dump as _cf_dump
def dump_tensor_to_disk(name, tensor, layer_id, method):
    """Save tensor/dict/list for layer-level debug analysis.

    Gated by Category.LAYER_TENSORS; output: $DUMP_DIR/<method>/layer_tensors/layer_{lid}/{name}.pt
    """
    if method is None:
        return
    save_dir = _cf_dump.get_dump_path(
        _cf_dump.Category.LAYER_TENSORS, session=method, subpath=f"layer_{layer_id}"
    )
    if save_dir is None:
        return
    try:
        file_path = save_dir / f"{name}.pt"
        if isinstance(tensor, (dict, list)):
            torch.save(tensor, file_path)
        else:
            torch.save(tensor.detach().cpu(), file_path)
    except Exception as e:
        print(f"[CacheFlex LAYER_TENSORS dump ERROR] Failed to save {name}: {e}")
# =========================================================================



class Qwen2MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        if get_global_server_args().rl_on_policy_target is not None:
            x = x.bfloat16()

        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Qwen2Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: Optional[int] = None,
        layer_id: int = 0,
        rope_theta: float = 1000000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 32768,
        quant_config: Optional[QuantizationConfig] = None,
        dual_chunk_attention_config: Optional[dict[str, Any]] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        if head_dim is not None:
            self.head_dim = head_dim
        else:
            self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)

        # attn_output = self.attn(q, k, v, forward_batch)
        # HACK(Shizhuo)
        attn_output = self.attn(
            q, k, v,
            forward_batch,
            rotary_emb=self.rotary_emb # <--- Add this
        )

        # ================= [DEBUG DUMP: OUTPUT] =================
        # HACK(Shizhuo)
        dump_name_shizhuo = getattr(forward_batch, "debug_dump_name", None)
        if dump_name_shizhuo and forward_batch.forward_mode.is_extend():
            # [ProphetKV] 检查 dump_qk_only 参数
            dump_qk_only = "qk_only" in str(dump_name_shizhuo)

            # 获取当前 Layer ID (RadixAttention 对象通常有 layer_id 属性)
            lid_c = getattr(self.attn, "layer_id", -1)
            # HACK(Shizhuo) [Oracle-Dump]: layer filter (no-op when DUMP_LAYERS unset)
            _layer_ok = _cf_dump.layer_allowed(lid_c)

            if not dump_qk_only and _layer_ok:
                # 完整dump模式：保存 Attention 输出
                dump_tensor_to_disk("attn_output", attn_output, lid_c, dump_name_shizhuo)

            # dump: q_full + 边界点 + cacheflex_slice_info + qo_indptr + input_ids
            # 只在 Extend (Prefill) 阶段保存，避免 Decode 阶段刷屏

            # === 1. 保存完整 Q (正确格式: [seq_len, num_heads, head_dim]) ===
            # ProphetKV Stage I 必需
            # q 当前 shape: [total_tokens, hidden_size]
            # 需要 reshape 为 [total_tokens, num_heads, head_dim]
            if _layer_ok:
                # HACK(Shizhuo) [Oracle-Dump]: q_full 仅在 LO_QFULL_LAYERS 指定层 dump
                # (默认全 dump); K/V 由 triton_backend 按 DUMP_LAYERS 全 dump, positions 全留。
                if _cf_dump.qfull_layer_allowed(lid_c):
                    q_full = q.view(q.size(0), self.num_heads, self.head_dim)
                    dump_tensor_to_disk("q_full", q_full, lid_c, dump_name_shizhuo)
                if dump_qk_only and _cf_dump.layer_filter_active():
                    # HACK(Shizhuo) [Oracle-Dump]: qk_only 补 positions 供跨 run 逻辑序对齐
                    dump_tensor_to_disk("positions", positions, lid_c, dump_name_shizhuo)

            if _layer_ok and not dump_qk_only:
                # 完整dump模式：保存额外的调试信息
                # === 0. 保存 input_ids (用于解码 token 内容) ===
                if hasattr(forward_batch, "input_ids") and forward_batch.input_ids is not None:
                    dump_tensor_to_disk("input_ids", forward_batch.input_ids, lid_c, dump_name_shizhuo)

                # === 2. 保存边界点信息 ===
                # 边界点定义:
                # - system_prompt_start = 0 (固定)
                # - system_prompt_end = image_offsets[0] (第一帧开始前)
                # - image_start = image_offsets[0]
                # - image_end = image_offsets[-1] + image_pad_len[-1] (最后一帧结束)
                # - question_start = image_end
                # - question_end = extend_prefix_lens[0] + extend_seq_lens[0] (prefill 总长度)
                system_prompt_end = None
                image_start = None
                image_end = None
                question_start = None

                # 获取 image_offsets 和 image_pad_len
                image_offsets_list = None
                image_pad_len_list = None

                # 方法1: 检查 forward_batch 是否直接有 image_offsets 属性
                if hasattr(forward_batch, "image_offsets") and forward_batch.image_offsets is not None:
                    image_offsets_list = forward_batch.image_offsets
                # 方法2: 检查 mm_inputs 中的 image_offsets
                elif forward_batch.mm_inputs and len(forward_batch.mm_inputs) > 0:
                    for mm_input in forward_batch.mm_inputs:
                        if mm_input and hasattr(mm_input, "image_offsets") and mm_input.image_offsets is not None:
                            image_offsets_list = mm_input.image_offsets
                            # 同时获取 image_pad_len
                            if hasattr(mm_input, "image_pad_len") and mm_input.image_pad_len is not None:
                                image_pad_len_list = mm_input.image_pad_len
                            break

                if image_offsets_list is not None and len(image_offsets_list) > 0:
                    # 扁平化处理（如果是 batch 嵌套结构）
                    if isinstance(image_offsets_list[0], list):
                        flat_offsets = []
                        for batch_offsets in image_offsets_list:
                            flat_offsets.extend(batch_offsets)
                        image_offsets_list = flat_offsets

                    # 扁平化 image_pad_len
                    if image_pad_len_list is not None:
                        if isinstance(image_pad_len_list[0], list):
                            flat_pad_len = []
                            for batch_pad_len in image_pad_len_list:
                                flat_pad_len.extend(batch_pad_len)
                            image_pad_len_list = flat_pad_len

                    # 计算边界点
                    if len(image_offsets_list) > 0:
                        system_prompt_end = image_offsets_list[0]
                        image_start = image_offsets_list[0]

                        if image_pad_len_list is not None and len(image_pad_len_list) > 0:
                            if len(image_pad_len_list) >= len(image_offsets_list):
                                image_end = image_offsets_list[-1] + image_pad_len_list[-1]
                            else:
                                # 如果 pad_len 比 offsets 少，使用最后一个 pad_len
                                image_end = image_offsets_list[-1] + image_pad_len_list[-1] if image_pad_len_list else image_offsets_list[-1] + 100
                        else:
                            # 没有 pad_len 信息，从 offsets 间隔估算帧长
                            if len(image_offsets_list) > 1:
                                estimated_frame_len = image_offsets_list[1] - image_offsets_list[0]
                                image_end = image_offsets_list[-1] + estimated_frame_len
                            else:
                                image_end = image_offsets_list[-1] + 100

                        question_start = image_end

                    # 获取 extend_prefix_lens 和 extend_seq_lens
                    question_end = None
                    if forward_batch.extend_prefix_lens is not None and forward_batch.extend_seq_lens is not None:
                        prefix_len = forward_batch.extend_prefix_lens[0].item()
                        seq_len = forward_batch.extend_seq_lens[0].item()
                        question_end = prefix_len + seq_len
                    elif hasattr(positions, '__len__'):
                        question_end = len(positions) - 1

                    # 构建边界点字典
                    boundary_info = {
                        "system_prompt_start": 0,
                        "system_prompt_end": system_prompt_end,
                        "image_start": image_start,
                        "image_end": image_end,
                        "question_start": question_start,
                        "question_end": question_end,
                    }

                    # 保存边界点信息
                    dump_tensor_to_disk("boundary_info", boundary_info, lid_c, dump_name_shizhuo)

                    # 保存 image_offsets 和 image_pad_len (用于帧级分析)
                    image_offsets_tensor = torch.tensor(image_offsets_list, dtype=torch.long)
                    dump_tensor_to_disk("image_offsets", image_offsets_tensor, lid_c, dump_name_shizhuo)
                    if image_pad_len_list is not None:
                        image_pad_len_tensor = torch.tensor(image_pad_len_list, dtype=torch.long)
                        dump_tensor_to_disk("image_pad_len", image_pad_len_tensor, lid_c, dump_name_shizhuo)

                # === 3. 保存 CacheFlex slice_info (复用策略信息) ===
                if hasattr(forward_batch, "cacheflex_slice_info") and forward_batch.cacheflex_slice_info is not None:
                    dump_tensor_to_disk("cacheflex_slice_info", forward_batch.cacheflex_slice_info, lid_c, dump_name_shizhuo)
                    if hasattr(forward_batch, "cacheflex_total_kv_lens") and forward_batch.cacheflex_total_kv_lens is not None:
                        dump_tensor_to_disk("cacheflex_total_kv_lens", forward_batch.cacheflex_total_kv_lens, lid_c, dump_name_shizhuo)
                    if forward_batch.extend_prefix_lens is not None:
                        dump_tensor_to_disk("extend_prefix_lens", forward_batch.extend_prefix_lens, lid_c, dump_name_shizhuo)
                    if forward_batch.extend_seq_lens is not None:
                        dump_tensor_to_disk("extend_seq_lens", forward_batch.extend_seq_lens, lid_c, dump_name_shizhuo)
        # ========================================================

        output, _ = self.o_proj(attn_output)
        return output


class Qwen2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_id = layer_id  # HACK(Shizhuo): Store layer_id for dump
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 32768)
        head_dim = getattr(config, "head_dim", None)
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        self.self_attn = Qwen2Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=head_dim,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            dual_chunk_attention_config=dual_chunk_attention_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = Qwen2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # HACK(Shizhuo): Dump for debugging
        dump_name = getattr(forward_batch, "debug_dump_name", None)
        layer_id = self.layer_id
        # [ProphetKV] qk_only 模式检查
        dump_qk_only = dump_name and "qk_only" in str(dump_name)

        # Dump INPUT hidden_states (Layer 0-2 only to reduce data)
        # qk_only 模式下跳过所有层内 dump
        if dump_name and not dump_qk_only and forward_batch.forward_mode.is_extend() and layer_id <= 2:
            dump_tensor_to_disk("layer_input_hidden", hidden_states, layer_id, dump_name)
            if residual is not None:
                dump_tensor_to_disk("layer_input_residual", residual, layer_id, dump_name)
        # ================================================================================

        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # HACK(Shizhuo)
        # Dump AFTER INPUT_LAYERNORM
        if dump_name and not dump_qk_only and forward_batch.forward_mode.is_extend() and layer_id <= 2:
            dump_tensor_to_disk("layer_after_input_ln_hidden", hidden_states, layer_id, dump_name)
            dump_tensor_to_disk("layer_after_input_ln_residual", residual, layer_id, dump_name)
        # ======================================================================================

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        # HACK(Shizhuo)
        # ============================================
        # [Sparse Check Layer] Slice residual after attention output.
        # hidden_states is already sparse (attention only processed sparse Q).
        # residual is still full-size from input_layernorm bypass, must slice
        # before post_attention_layernorm (fused_add_rmsnorm requires matching shapes).
        runtime = getattr(forward_batch, "cacheflex_runtime", None)
        if (runtime is not None
            and runtime.get("_check_layer_global_mask") is not None
            and not runtime.get("_check_layer_sparse_slicing_done", False)
            and self.layer_id == runtime.get("_check_layer_id", -1)):
            mask = runtime["_check_layer_global_mask"]
            residual = residual[mask]
            runtime["_check_layer_sparse_slicing_done"] = True

        # HACK(Shizhuo)
        # Dump AFTER SELF_ATTN
        if dump_name and not dump_qk_only and forward_batch.forward_mode.is_extend() and layer_id <= 2:
            dump_tensor_to_disk("layer_after_attn_hidden", hidden_states, layer_id, dump_name)
        # ======================================================================================

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # HACK(Shizhuo)
        # Dump AFTER POST_ATTN_LAYERNORM
        if dump_name and not dump_qk_only and forward_batch.forward_mode.is_extend() and layer_id <= 2:
            dump_tensor_to_disk("layer_after_post_ln_hidden", hidden_states, layer_id, dump_name)
            dump_tensor_to_disk("layer_after_post_ln_residual", residual, layer_id, dump_name)
        # ======================================================================================

        hidden_states = self.mlp(hidden_states)

        # HACK(Shizhuo)
        # Dump AFTER MLP (OUTPUT)
        if dump_name and not dump_qk_only and forward_batch.forward_mode.is_extend() and layer_id <= 2:
            dump_tensor_to_disk("layer_output_hidden", hidden_states, layer_id, dump_name)
        # ======================================================================================

        return hidden_states, residual

    def _l0_prefix_qkv_only(self, positions, hidden_states, forward_batch, out_cache_loc_prefix):
        """L0 prefix pass: QKV projection only, skip attention and MLP.
        Save K/V to pool for L1+ to read, return V for compute_diff at L1."""
        hidden_states = self.input_layernorm(hidden_states)
        qkv, _ = self.self_attn.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.self_attn.q_size, self.self_attn.kv_size, self.self_attn.kv_size], dim=-1)
        q, k = self.self_attn.rotary_emb(positions, q, k)
        k = k.view(-1, self.self_attn.attn.tp_k_head_num, self.self_attn.attn.qk_head_dim)
        v = v.view(-1, self.self_attn.attn.tp_v_head_num, self.self_attn.attn.v_head_dim)
        forward_batch.token_to_kv_pool.set_kv_buffer(
            self.self_attn.attn, out_cache_loc_prefix, k, v
        )
        # Note: contiguous buffer NOT updated here — gather_contiguous_kv at check_layer+1
        # will re-gather from paged pool (which has correct K/V from set_kv_buffer).
        return v


class Qwen2Model(nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        decoder_layer_type: type[nn.Module] = Qwen2DecoderLayer,
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                enable_tp=not is_dp_attention_enabled(),
                prefix=add_prefix("embed_tokens", prefix),
                params_dtype=(
                    torch.float32
                    if get_global_server_args().rl_on_policy_target is not None
                    else None
                ),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # Use the provided decoder layer type or default to Qwen2DecoderLayer
        decoder_layer_type = decoder_layer_type or Qwen2DecoderLayer
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: decoder_layer_type(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        if self.pp_group.is_last_rank:
            norm_kwargs = (
                dict(
                    weight_dtype=torch.float32,
                    cast_x_before_out_mul=True,
                    override_orig_dtype=torch.float32,
                    fp32_residual=True,
                )
                if get_global_server_args().rl_on_policy_target is not None
                else {}
            )
            self.norm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps, **norm_kwargs
            )
        else:
            self.norm = PPMissingLayer(return_tuple=True)

        # For EAGLE3 support
        self.layers_to_capture = []

    def get_input_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        if hasattr(self.config, "scale_emb"):
            return self.get_input_embeddings()(input_ids) * self.config.scale_emb
        else:
            return self.get_input_embeddings()(input_ids)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:

        # HACK(Shizhuo): [Mem Profile] Entry snapshot (gated by CACHEFLEX_MEM_PROFILE=1)
        cacheflex_mem_snapshot(forward_batch, "entry")

        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        # HACK(Shizhuo): [CacheFlex] Slicing & Copy Point
        # [Stage 1] Initial Slicing (Before Layer 0)
        if forward_batch.forward_mode.is_extend() and forward_batch.cacheflex_configs:
             hidden_states, positions = self._apply_cacheflex_slicing_and_copy(
                 hidden_states, positions, forward_batch
             )
        # =======================================================
        # HACK(Shizhuo): [Mem Profile] After Stage 1
        cacheflex_mem_snapshot(forward_batch, "after_stage1")

        aux_hidden_states = []
        for i in range(self.start_layer, self.end_layer):
            # HACK(Shizhuo)
            # [Stage 2] Post-Check Slicing (After check layer)
            check_layer = None
            sample_task = self._find_any_task(getattr(forward_batch, "cacheflex_dynamic_tasks_by_req", []))
            if sample_task:
                check_layer = sample_task["check_layer"]
                
            if check_layer is not None and i == (check_layer + 1) and forward_batch.forward_mode.is_extend():
                # HACK(Shizhuo)
                # ============================================================================
                # [P2 cleanup] Free L0 prefix V map before check_layer+1 Stage 2 slicing.
                # After check_layer, V map has no more readers (all token_selector /
                # kvfocus_harm_observe call sites are gated by check_layer == self.layer_id
                # in radix_attention). Freeing here reduces peak memory pressure for the
                # upcoming K/V clone in _apply_post_check_slicing.
                if hasattr(forward_batch, '_cacheflex_l0_prefix_v_map'):
                    del forward_batch._cacheflex_l0_prefix_v_map
                # ============================================================================
                hidden_states, positions, residual, _ = self._apply_post_check_slicing(
                    hidden_states, positions, residual, forward_batch
                )
            # =======================================================
            if i in self.layers_to_capture:
                aux_hidden_states.append(
                    hidden_states + residual if residual is not None else hidden_states
                )
            layer = self.layers[i]

            # HACK(Shizhuo): [P2 reverted] L0 split (prefix QKV-only + suffix full)
            # REMOVED — it left prefix tokens' hidden_states stale (L0-input instead
            # of L0-output), causing wrong Q/K/V at L1+ for any cacheblend/kvfocus
            # method where some prefix tokens get selected as "fresh". This matched
            # vLLM/epic's original behavior (no L0 split — every layer runs full
            # forward for all tokens before check_layer).
            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
            )

            # HACK(Shizhuo)
            # ============================================
            # [Sparse Check Layer] Sync positions local variable after check layer.
            if (check_layer is not None and i == check_layer
                and forward_batch.forward_mode.is_extend()
                and getattr(forward_batch, "cacheflex_runtime", None) is not None
                and forward_batch.cacheflex_runtime.get("_check_layer_sparse_slicing_done")):
                positions = forward_batch.positions

        # HACK(Shizhuo) [ProphetKV-Probe]: after ALL layers (still in the timed pass1 forward),
        # uniform mean-fuse the per-layer query->ctx alpha (Eq7) -> top-r -> inline external-replay
        # mask = ProphetKV Stage I incl fusion+topk. Flag/marker gated => OFF = byte-unchanged.
        # (covers qwen3 too: Qwen3Model(Qwen2Model) inherits this forward.)
        if (forward_batch.forward_mode.is_extend()
                and getattr(forward_batch, "cacheflex_configs", None)):
            from sglang.srt.layers.radix_attention import (
                _prophetkv_probe_enabled, _prophetkv_finalize,
            )
            if _prophetkv_probe_enabled(forward_batch):
                _prophetkv_finalize(forward_batch)

        # HACK(Shizhuo): [Mem Profile] After full layer loop (before norm/lm_head)
        cacheflex_mem_snapshot(forward_batch, "after_layers")

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
        else:
            if hidden_states.shape[0] != 0:
                if residual is None:
                    hidden_states = self.norm(hidden_states)
                else:
                    hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) == 0:
            return hidden_states

        return hidden_states, aux_hidden_states

    # If this function is called, it should always initialize KV cache scale
    # factors (or else raise an exception). Thus, handled exceptions should
    # make sure to leave KV cache scale factors in a known good (dummy) state
    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        for layer_idx, scaling_factor in kv_cache_scales_loader(
            quantization_param_path,
            tp_rank,
            tp_size,
            self.config.num_hidden_layers,
            self.config.__class__.model_type,
        ):
            if not isinstance(self.layers[layer_idx], nn.Identity):
                layer_self_attn = self.layers[layer_idx].self_attn
            if hasattr(layer_self_attn.attn, "k_scale"):
                layer_self_attn.attn.k_scale = scaling_factor
                layer_self_attn.attn.v_scale = scaling_factor
            else:
                raise RuntimeError(
                    "Self attention has no KV cache scaling " "factor attribute!"
                )

    # HACK(Shizhuo)
    # Helper to find valid task safely
    def _find_any_task(self, tasks_by_req):
        if not tasks_by_req: return None
        for tasks in tasks_by_req:
            if tasks:
                return tasks[0]
        return None

    # ========================================================================
    # HACK(Shizhuo) [CacheFlex] Helper: 处理静态切分 (EPIC / Naive)
    # ========================================================================
    def _process_static_segment(self,
                                hidden_states, positions, req_out_loc,
                                start_abs_idx, current_input_ptr,
                                src_info, parts_collector, req_idx, token_ids=None): 
        """
        EPIC/Naive: 根据 Plan 直接切分 Compute (Head) 和 Copy (Tail) 区域。
        """
        plan = src_info["plan"]
        src_len = src_info["src_len"]
        rope_delta = src_info["rope_delta"]
        sink_offset = src_info["sink_token_len"]
        
        # 1. Compute Part (Head) -> 加入 parts_collector (保留)
        c_range = plan.compute_range if plan.compute_range is not None else (0, 0)
        c_start, c_end = c_range
        
        # 边界防御
        c_start = max(0, min(int(c_start), int(src_len)))
        c_end   = max(0, min(int(c_end),   int(src_len)))
        
        if c_end > c_start:
            # 计算绝对位置: BatchStart + RelInputPtr + RelComputeStart
            head_abs_start = start_abs_idx + current_input_ptr + c_start
            head_abs_end   = start_abs_idx + current_input_ptr + c_end

            parts_collector['h'].append(hidden_states[head_abs_start : head_abs_end])
            parts_collector['p'].append(positions[head_abs_start : head_abs_end])
            parts_collector['l'].append(req_out_loc[current_input_ptr + c_start : current_input_ptr + c_end])
            # logprobs: 同步收集 token_ids（用于 logprob 对齐）
            if token_ids is not None:
                parts_collector['t'].append(token_ids[head_abs_start : head_abs_end])

        # 2. Copy Part (Tail) -> 生成 Copy 任务
        copy_range = plan.copy_range
        
        if copy_range is not None:
            cp_start, cp_end = copy_range
            cp_start = max(0, min(int(cp_start), int(src_len)))
            cp_end   = max(0, min(int(cp_end),   int(src_len)))
            
            if cp_end > cp_start:
                dst_tail_loc = req_out_loc[current_input_ptr + cp_start : current_input_ptr + cp_end]
                cache_start = sink_offset + cp_start
                cache_end   = sink_offset + cp_end
                
                parts_collector['info'].append({
                    "is_sliced": True,
                    "src_block_indices": src_info["src_block_indices"],
                    "dst_tail_loc": dst_tail_loc,
                    "copy_range": (cache_start, cache_end),
                    "prefix_len": rope_delta,
                    "req_idx": req_idx,
                })

    # ========================================================================
    # HACK(Shizhuo) [CacheFlex] Helper: 处理 selective-recompute 切分 (cacheblend / kvfocus / external)
    # 见 llama.py 同名函数注释: 为 check_layer 的选择性重算做准备 (compute-all + 铺 KV
    # 底座 + 登记 check_layer 任务), mask 来源由 radix_attention 决定。
    # ========================================================================
    def _process_selective_recompute_segment(self,
                                    hidden_states, positions, req_out_loc,
                                    start_abs_idx, current_input_ptr,
                                    src_info, parts_collector,
                                    dynamic_tasks_list, req_idx, token_ids=None,
                                    contiguous_kv_offset=0):
        """
        CacheBlend (Layer 0):
        1. 全量计算 (Compute All): 为了计算准确的 Diff。
        2. 全量拷贝 (Copy All): 为了填充 KV Cache 底座。
        3. 登记任务 (Register Task): 告诉 Attention 层需要计算 Diff。
        """
        src_len = src_info["src_len"]
        plan = src_info["plan"]
        rope_delta = src_info["rope_delta"]
        sink_offset = src_info["sink_token_len"]

        # 1. Compute All (Layer 0 需要全量 V)
        abs_start = start_abs_idx + current_input_ptr
        abs_end   = abs_start + src_len

        parts_collector['h'].append(hidden_states[abs_start : abs_end])
        parts_collector['p'].append(positions[abs_start : abs_end])
        parts_collector['l'].append(req_out_loc[current_input_ptr : current_input_ptr + src_len])
        # logprobs: 同步收集 token_ids
        if token_ids is not None:
            parts_collector['t'].append(token_ids[abs_start : abs_end])

        # 2. Copy All (填满 Cache，供后续层稀疏计算时 Attend)
        parts_collector['info'].append({
            "is_sliced": True,
            "src_block_indices": src_info["src_block_indices"],
            "dst_tail_loc": req_out_loc[current_input_ptr : current_input_ptr + src_len],
            "copy_range": (0 + sink_offset, src_len + sink_offset),
            "prefix_len": rope_delta,
            "req_idx": req_idx,
        })

        # 3. Register Dynamic Task
        # 计算该 Segment 在当前 Request 内的相对起始位置 (Relative Start)
        # 依赖于在此之前已经加入 parts_collector 的长度
        rel_start = sum(t.shape[0] for t in parts_collector['h']) - src_len # 减去刚加进去的自己

        dynamic_tasks_list.append({
            "rel_start": rel_start, # 相对当前 Req 的偏移
            "src_len": src_len,
            "src_block_indices": src_info["src_block_indices"],
            "src_offset": sink_offset,
            "contiguous_offset": contiguous_kv_offset,
            "recomp_ratio": plan.recomp_ratio,
            "check_layer": plan.check_layer,
            "kvfocus": plan.kvfocus,  # [KVFocus] probe attention mass flag
            "solver": plan.solver,
            "scaling": plan.scaling,
            "beta": plan.beta,
            "eta": plan.eta,
            "gamma": plan.gamma,
            "rho": plan.rho,
            "lam": plan.lam,
            "max_rate": plan.max_rate,
            "n_ref": plan.n_ref,
            "b0": plan.b0,
            "b0_mode": plan.b0_mode,
            "num_probes": plan.num_probes,
            "top_k_peak": plan.top_k_peak,
            "target_rate": plan.target_rate,
            "pass_mode": plan.pass_mode,
            "kvfocus_variant": plan.kvfocus_variant,  # method marker (e.g. prophetkv-probe)
            "rope_delta": rope_delta,  # task self-contained; no config re-lookup
            "source_role": src_info.get("source_role", "doc"),  # defensive: must always be "doc"
        })

    # HACK(Shizhuo): external shares _process_selective_recompute_segment
    # with cacheblend/kvfocus.

    # ========================================================================
    # [CacheFlex P2] L0 Prefix/Suffix Split
    # ========================================================================
    def _l0_split_and_run(self, hidden_states, positions, forward_batch, layer):
        """L0 optimization: prefix QKV-only + suffix full forward.
        Splits tokens into prefix (cached, QKV-proj only) and suffix (new, full L0).
        Stores L0 prefix V for L1 scorer to use (prefix hidden_states are stale at L1).

        After L0, reassembles full hidden_states (stale prefix + updated suffix) so
        that L1 can process all tokens normally — scorer needs full K for probe scores.
        """
        l0_suffix_starts = forward_batch._cacheflex_l0_suffix_starts
        batch_size = forward_batch.batch_size

        # Build prefix/suffix split masks (per-request, in sliced sequence)
        prefix_mask_parts = []
        suffix_mask_parts = []
        for req_idx in range(batch_size):
            seq_len = forward_batch.extend_seq_lens[req_idx].item()
            suffix_start = l0_suffix_starts[req_idx]
            mask = torch.zeros(seq_len, dtype=torch.bool, device=hidden_states.device)
            if suffix_start > 0:
                mask[:suffix_start] = True
            prefix_mask_parts.append(mask)
            suffix_mask_parts.append(~mask)

        prefix_mask = torch.cat(prefix_mask_parts)
        suffix_mask = torch.cat(suffix_mask_parts)

        # Save stale prefix hidden_states for L1 reassembly
        stale_prefix_h = hidden_states[prefix_mask].clone() if prefix_mask.any() else None
        stale_prefix_p = positions[prefix_mask].clone() if prefix_mask.any() else None

        # Phase 1: Prefix QKV-only
        if prefix_mask.any():
            prefix_h = hidden_states[prefix_mask]
            prefix_p = positions[prefix_mask]
            prefix_loc = forward_batch.out_cache_loc[prefix_mask]
            v_prefix = layer._l0_prefix_qkv_only(prefix_p, prefix_h, forward_batch, prefix_loc)

            # Build prefix_v_map from v_prefix using tasks_by_req
            prefix_v_map = {}
            req_prefix_offset = 0
            for req_idx, tasks in enumerate(forward_batch.cacheflex_dynamic_tasks_by_req):
                for task in tasks:
                    if task.get("kind") == "drop_all":
                        continue
                    src_len = task["src_len"]
                    rel_start = task["rel_start"]
                    task_v_start = req_prefix_offset + rel_start
                    if task_v_start + src_len <= v_prefix.shape[0]:
                        prefix_v_map[(req_idx, rel_start)] = v_prefix[task_v_start : task_v_start + src_len].clone()
                req_prefix_offset += l0_suffix_starts[req_idx]
            forward_batch._cacheflex_l0_prefix_v_map = prefix_v_map

        # Phase 2: Suffix full L0
        if suffix_mask.any():
            suffix_h = hidden_states[suffix_mask]
            suffix_p = positions[suffix_mask]
            suffix_loc = forward_batch.out_cache_loc[suffix_mask]

            # Save original forward_batch fields
            orig_seq_lens = forward_batch.seq_lens
            orig_start_loc = forward_batch.extend_start_loc
            orig_out_cache_loc = forward_batch.out_cache_loc
            orig_positions = forward_batch.positions

            # Rebuild seq_lens for suffix-only
            req_idx_per_token = torch.repeat_interleave(
                torch.arange(batch_size, device=hidden_states.device, dtype=torch.long),
                orig_seq_lens.to(torch.long)
            )
            suffix_counts = torch.bincount(
                req_idx_per_token[suffix_mask],
                minlength=batch_size
            ).to(torch.int64)

            forward_batch.seq_lens = suffix_counts
            forward_batch.extend_seq_lens = suffix_counts
            forward_batch.out_cache_loc = suffix_loc
            new_start_loc = torch.zeros_like(orig_start_loc)
            new_start_loc[1:] = torch.cumsum(suffix_counts[:-1], dim=0)
            forward_batch.extend_start_loc = new_start_loc
            forward_batch.positions = suffix_p

            # Run full L0 for suffix tokens only
            hidden_suffix, residual_suffix = layer(suffix_p, suffix_h, forward_batch, residual=None)

            # Restore forward_batch metadata
            forward_batch.seq_lens = orig_seq_lens
            forward_batch.extend_seq_lens = orig_seq_lens
            forward_batch.extend_start_loc = orig_start_loc
            forward_batch.out_cache_loc = orig_out_cache_loc
            forward_batch.positions = orig_positions

            # Reassemble full hidden_states: stale prefix + updated suffix
            hidden_states = torch.empty_like(hidden_states if stale_prefix_h is None else
                                             torch.cat([stale_prefix_h, hidden_suffix]))
            hidden_states[prefix_mask] = stale_prefix_h
            hidden_states[suffix_mask] = hidden_suffix

            # Reassemble residual: prefix gets 0, suffix gets L0 residual
            residual = torch.empty_like(hidden_states)
            residual[prefix_mask] = 0
            residual[suffix_mask] = residual_suffix

            # positions stays full (restored from orig_positions)
            positions = orig_positions
        else:
            # All prefix, no suffix — residual should be 0 for L1 input_layernorm
            residual = torch.zeros_like(hidden_states)

        return hidden_states, positions, residual

    # ========================================================================
    # HACK(Shizhuo) [CacheFlex] Stage 1: Initial Slicing & Task Registration
    # ========================================================================
    def _apply_cacheflex_slicing_and_copy(self, hidden_states, positions, forward_batch):
        """
        [CacheFlex] Input Slicing Logic.
        统一策略切片器：支持全量重算(full)、全量复用(naive)、部分重算(epic)和动态复用(cacheblend)。
        在 Model 内部执行切片：
        支持拼接：Global Prefix + Doc1 + Doc2 + ... + Query
        1. 根据 Metadata 计算 Active Indices (Head + Query)。
        2. 触发 Triton Copy Kernel 搬运 Inactive Indices (Tail)。
        3. 切割 Hidden States, Positions。
        4. 原地修改 forward_batch 的元数据，使其与切割后的 Tensor 匹配。
        Layer 0 之前调用。
        - EPIC/Naive: 执行静态切分。
        - CacheBlend: 全量保留，但生成 Copy 任务并登记动态任务。
        - External (ProphetKV): 根据 external_mask 切片。
        """
        # HACK(Shizhuo): [ProphetKV] 检查 external 方法是否与其他方法混用
        # HACK(Shizhuo): external runs on the selective-recompute path and can be
        # freely mixed with naive/epic sources (e.g. sys=naive + doc=external).
        # external requires an injected mask; a missing mask fails fast.
        has_external = any(
            (src.get("plan").method if src.get("plan") and hasattr(src.get("plan"), "method") else "") == "external"
            for cfg in forward_batch.cacheflex_configs if cfg and "sources" in cfg
            for src in cfg["sources"]
        )
        if has_external and "cacheflex_global_mask" not in forward_batch.cacheflex_runtime:
            raise ValueError(
                "[CacheFlex] cacheflex_global_mask not found. "
                "external_mask must be provided when using external method."
            )

        # 获取原始全量信息
        original_out_cache_loc = forward_batch.out_cache_loc

        new_hidden_list = []
        new_pos_list = []
        new_loc_list = []
        new_seq_lens = []
        total_kv_lens = [] # 记录全量长度供 Attention 使用
        new_extend_logprob_start_lens = []
        l0_suffix_starts_list = []  # [P2] suffix start offset in sliced sequence per request
        # HACK(Shizhuo): KVFocus probe range — start of LAST source in sliced coords.
        # Used by SGLANG_KVFOCUS_PROBE_FROM_SUFFIX to span [last_source_start, req_len)
        # so probes cover query content + chat marker (fixes Qwen chat-template issue).
        probe_starts_list = []

        slice_info = []
        current_loc_offset = 0
        has_slicing = False        

        # 初始化动态任务列表 (按 Req 索引)
        forward_batch.cacheflex_dynamic_tasks_by_req = [[] for _ in range(len(forward_batch.extend_seq_lens))]

        # 遍历 Batch 中的每个请求
        for req_idx, (seq_len, cfg) in enumerate(zip(forward_batch.extend_seq_lens, forward_batch.cacheflex_configs)):
            start = forward_batch.extend_start_loc[req_idx]
            req_len = seq_len.item()
            end = start + req_len

            # 当前请求对应的全量物理 block 地址
            req_out_loc = original_out_cache_loc[current_loc_offset : current_loc_offset + req_len]
            current_loc_offset += req_len

            cached_len = forward_batch.extend_prefix_lens[req_idx].item()

            # logprobs: 初始化当前 req 的 kept_before 计数器
            # 统计在 sliced 序列中，原 logprob_start_len 之前保留了多少 tokens
            old_start = forward_batch.extend_logprob_start_lens_cpu[req_idx] if (
                hasattr(forward_batch, "extend_logprob_start_lens_cpu") and
                forward_batch.extend_logprob_start_lens_cpu and
                req_idx < len(forward_batch.extend_logprob_start_lens_cpu)
            ) else 0
            kept_before = 0

            # 用于统计 kept_before 的辅助函数：计算某个 keep range 在 [0, old_start) 内的交集长度
            def add_keep_range(a, b):
                nonlocal kept_before
                # [a, b) 是相对当前 req 的全量坐标
                # 我们需要计算 [a, b) 与 [0, old_start) 的交集长度
                if b <= 0 or a >= old_start:
                    return
                kept_before += max(0, min(b, old_start) - max(a, 0))

            # # 逻辑范围
            # input_logical_start = cached_len
            # input_logical_end = cached_len + seq_len

            # --- Case 1: 普通请求 ---
            if not cfg or "sources" not in cfg:
                new_hidden_list.append(hidden_states[start:end])
                new_pos_list.append(positions[start:end])
                new_loc_list.append(req_out_loc)
                # logprobs: 普通请求没有切片，整个 [0, req_len) 都保留
                add_keep_range(0, req_len)
                new_seq_lens.append(req_len)
                total_kv_lens.append(cached_len + req_len)
                slice_info.append({"is_sliced": False})
                # 保存 new_start_len
                new_extend_logprob_start_lens.append(kept_before)
                l0_suffix_starts_list.append(0)  # [P2] no prefix/suffix split for non-CacheFlex
                probe_starts_list.append(0)  # no override for non-CacheFlex requests
                continue

            # --- Case 2: CacheFlex 请求 ---
            has_slicing = True

            # 初始化分段列表
            parts_collector = {'h': [], 'p': [], 'l': [], 'info': []}

            # input_ptr: 相对 start 的偏移量，强制从 0 开始
            current_input_ptr = 0

            # [P0] Track offset into contiguous buffer for compute_diff fast path.
            contiguous_kv_offset = 0

            # HACK(Shizhuo): record where the LAST processed source starts in sliced coords.
            # Used by KVFocus probe-range logic — probes should span query content + chat
            # marker, which is last source + suffix block. _cacheflex_l0_suffix_starts only
            # records AFTER all sources (= chat marker start only).
            probe_start_sliced = 0  # default: req_len → no probe range override

            # --- 2. Source Documents Loop (Doc1, Doc2, ...) ---
            # [MODIFIED] 遍历 sources 列表进行多段切片
            for i, src_info in enumerate(cfg["sources"]):
                src_len = src_info["src_len"]
                plan = src_info["plan"]
                # 边界检查
                if current_input_ptr + src_len > req_len: break # 防止溢出

                # HACK(Shizhuo): record start of this source BEFORE processing — overwrites
                # each iter, so after the loop it points to the LAST source's start.
                probe_start_sliced = sum(t.shape[0] for t in parts_collector['h'])

                method = src_info["plan"].method
                # HACK(Shizhuo) [Oracle-Dump]: cacheblend/kvfocus/external/layer_refresh 共享
                # selective-recompute Stage-1; mask 来源在 radix_attention 分叉 (见 llama.py)。
                if method in ("cacheblend", "layer_refresh", "external"):
                    self._process_selective_recompute_segment(
                        hidden_states, positions, req_out_loc,
                        start, current_input_ptr,
                        src_info, parts_collector,
                        forward_batch.cacheflex_dynamic_tasks_by_req[req_idx],
                        req_idx,
                        None,  # 不再传递 token_ids，因为它是 windowed 的
                        contiguous_kv_offset,  # [P0] offset in contiguous buffer
                    )
                    # 明确 Kind
                    last_task = forward_batch.cacheflex_dynamic_tasks_by_req[req_idx][-1]
                    if method == "layer_refresh":
                        last_task["kind"] = "drop_all"
                    else:
                        last_task["kind"] = "select_topk"
                    # logprobs: cacheblend 全量保留，全部加入统计
                    add_keep_range(current_input_ptr, current_input_ptr + src_len)

                elif method in ["naive", "epic"]:
                    self._process_static_segment(
                        hidden_states, positions, req_out_loc,
                        start, current_input_ptr,
                        src_info, parts_collector,
                        req_idx,
                        None  # 不再传递 token_ids
                    )
                    # logprobs: naive/epic 只有 compute_range 保留，只统计这部分
                    c_range = plan.compute_range if plan.compute_range is not None else (0, 0)
                    c_start, c_end = c_range
                    c_start = max(0, min(int(c_start), int(src_len)))
                    c_end = max(0, min(int(c_end), int(src_len)))
                    if c_end > c_start:
                        add_keep_range(current_input_ptr + c_start, current_input_ptr + c_end)

                else: # method == "full" (不命中或占位段)
                    # 保留完整 Embedding 进行重计算
                    abs_start = start + current_input_ptr
                    abs_end = abs_start + src_len
                    parts_collector['h'].append(hidden_states[abs_start : abs_end])
                    parts_collector['p'].append(positions[abs_start : abs_end])
                    parts_collector['l'].append(req_out_loc[current_input_ptr : current_input_ptr + src_len])
                    # logprobs: full 段全量保留，全部加入统计
                    add_keep_range(current_input_ptr, current_input_ptr + src_len)

                current_input_ptr += src_len

                # [P0] Accumulate contiguous buffer offset for all text_hit sources
                if src_info.get("type") == "text_hit" and src_info.get("src_block_indices") is not None:
                    contiguous_kv_offset += src_len

            # --- 3. Suffix (Query / User Input) ---
            # [P2] Record suffix start in sliced sequence (after all source segments)
            l0_suffix_start_sliced = sum(t.shape[0] for t in parts_collector['h'])

            if current_input_ptr < req_len:
                # HACK(Shizhuo): [ProphetKV] external 方法时，suffix 也根据 mask 处理
                if has_external:
                    global_mask = forward_batch.cacheflex_runtime["cacheflex_global_mask"]
                    local_mask = global_mask[current_input_ptr : req_len]

                    abs_start = start + current_input_ptr
                    abs_end = end
                    local_hidden = hidden_states[abs_start : abs_end]
                    local_pos = positions[abs_start : abs_end]
                    local_loc = req_out_loc[current_input_ptr : req_len]

                    # 只保留 mask=True 的部分
                    parts_collector['h'].append(local_hidden[local_mask])
                    parts_collector['p'].append(local_pos[local_mask])
                    parts_collector['l'].append(local_loc[local_mask])

                    # logprobs
                    kept_indices = torch.where(local_mask)[0]
                    for idx in kept_indices:
                        add_keep_range(current_input_ptr + idx.item(), current_input_ptr + idx.item() + 1)
                else:
                    suf_start = start + current_input_ptr
                    parts_collector['h'].append(hidden_states[suf_start : end])
                    parts_collector['p'].append(positions[suf_start : end])
                    parts_collector['l'].append(req_out_loc[current_input_ptr : req_len])
                    # logprobs: suffix 全量保留，全部加入统计
                    add_keep_range(current_input_ptr, req_len)

            # --- 组装 ---
            if parts_collector['h']:
                merged_h = torch.cat(parts_collector['h'])
                new_hidden_list.append(merged_h)
                new_pos_list.append(torch.cat(parts_collector['p']))
                new_loc_list.append(torch.cat(parts_collector['l']))

                new_seq_lens.append(len(merged_h))
                slice_info.extend(parts_collector['info'])
            else:
                new_seq_lens.append(0) # 极端情况

            # [CRITICAL] 保持全长 KV
            total_kv_lens.append(cached_len + req_len)
            # 保存当前 req 的新 start_len
            new_extend_logprob_start_lens.append(kept_before)
            # [P2] Record suffix start in sliced sequence for L0 prefix/suffix split
            l0_suffix_starts_list.append(l0_suffix_start_sliced)
            # HACK(Shizhuo): start of last source (= start of probe range for KVFocus)
            probe_starts_list.append(probe_start_sliced)

        # === Batch Reconstruction ===
        if has_slicing:
            # Reconstruct Tensors
            if len(new_hidden_list) > 0:
                sliced_hidden = torch.cat(new_hidden_list)
                sliced_positions = torch.cat(new_pos_list)
                sliced_loc = torch.cat(new_loc_list)
            else:
                sliced_hidden = torch.empty(0, dtype=hidden_states.dtype, device=hidden_states.device)
                sliced_positions = torch.empty(0, dtype=positions.dtype, device=positions.device)
                sliced_loc = torch.empty(0, dtype=original_out_cache_loc.dtype, device=original_out_cache_loc.device)

            # 原地更新 Forward Batch
            # 这样后续 Layer 的 Attention Kernel 就能读到正确的 seq_len
            new_lens_tensor = torch.tensor(new_seq_lens, dtype=torch.int64, device=forward_batch.extend_seq_lens.device)

            forward_batch.extend_seq_lens = new_lens_tensor
            forward_batch.seq_lens = new_lens_tensor # 关键：FlashInfer/Triton 用这个
            forward_batch.out_cache_loc = sliced_loc # 关键：KV Cache 写入位置

            # Recompute Start Loc (必须更新!)
            new_start_loc = torch.zeros_like(forward_batch.extend_start_loc)
            new_start_loc[1:] = torch.cumsum(new_lens_tensor[:-1], dim=0)
            forward_batch.extend_start_loc = new_start_loc

            # logprobs: 更新 extend_seq_lens_cpu 和 extend_logprob_start_lens_cpu
            # 使用正确计算的 new_extend_logprob_start_lens（已映射到 sliced 坐标）
            if hasattr(forward_batch, "extend_seq_lens_cpu") and forward_batch.extend_seq_lens_cpu is not None:
                forward_batch.extend_seq_lens_cpu = new_lens_tensor.tolist()

                if hasattr(forward_batch, "extend_logprob_start_lens_cpu") and forward_batch.extend_logprob_start_lens_cpu is not None:
                    forward_batch.extend_logprob_start_lens_cpu = new_extend_logprob_start_lens

            # logprobs: 添加防呆断言
            # 验证：token_ids 总长度 == sum_i (new_len_i - new_start_i)
            if (hasattr(forward_batch, "extend_input_logprob_token_ids_gpu") and
                forward_batch.extend_input_logprob_token_ids_gpu is not None):
                expected = sum(int(l) - int(s) for l, s in zip(forward_batch.extend_seq_lens_cpu,
                                                               forward_batch.extend_logprob_start_lens_cpu))
                actual = forward_batch.extend_input_logprob_token_ids_gpu.shape[0]
                assert actual == expected, (f"token_ids length {actual} != expected {expected}, "
                                          f"new_lens={new_extend_logprob_start_lens}, "
                                          f"new_seq_lens={new_seq_lens}")

            # 传递 Full KV Lens 给 Triton Backend (用于 Attention Mask)
            forward_batch.cacheflex_total_kv_lens = torch.tensor(total_kv_lens, dtype=torch.int32, device=forward_batch.extend_seq_lens.device)
            forward_batch.cacheflex_slice_info = slice_info # 标记给 Backend
            # [P2] Store suffix start offsets for L0 prefix/suffix split
            forward_batch._cacheflex_l0_suffix_starts = l0_suffix_starts_list
            # HACK(Shizhuo): expose probe start per req for KVFocus
            forward_batch._cacheflex_probe_starts = probe_starts_list
            # logprobs: 更新 positions，否则 Kernel 会读到错误的 mask 位置
            forward_batch.positions = sliced_positions

            # 强制后端同步
            # 这会重新计算 qo_indptr，让 Triton 知道现在只有 3300 个 Token 了
            if hasattr(forward_batch, "attn_backend") and forward_batch.attn_backend:
                forward_batch.attn_backend.init_forward_metadata(forward_batch)

            return sliced_hidden, sliced_positions
        
        return hidden_states, positions

    # ========================================================================
    # HACK(Shizhuo) [CacheFlex] Stage 2: Post-Check Slicing (Metadata Reconstruction)
    # ========================================================================
    def _apply_post_check_slicing(self, hidden_states, positions, residual, forward_batch):
        """
        After check layer: apply global_mask to sparsify tokens.
        When check layer already did sparse QKV (radix_attention.py), most work
        is done. This function handles remaining: logprobs, req metadata,
        contiguous KV gather, and backend metadata reinit.
        """
        runtime = forward_batch.cacheflex_runtime
        if "cacheflex_global_mask" not in runtime:
            return hidden_states, positions, residual, forward_batch.out_cache_loc

        # HACK(Shizhuo)
        # ============================================
        # [Sparse Check Layer] Fast path: L1 already sparsified Q/K/V and
        # updated forward_batch metadata. Skip redundant mask gather.
        if runtime.get("_check_layer_sparse_slicing_done"):
            # Reinit backend metadata (needed for L2+ layers)
            if hasattr(forward_batch, "attn_backend") and forward_batch.attn_backend:
                forward_batch.attn_backend.init_forward_metadata(forward_batch)

            # HACK(Shizhuo)
            # ============================================
            # [Contiguous KV] Gather full KV from paged cache
            if (getattr(forward_batch, "cacheflex_kv_mode", "paged") == "contiguous"
                and not getattr(forward_batch, "cacheflex_use_old_kv", False)):
                from sglang.srt.layers.attention.cacheflex_utils import gather_contiguous_kv
                check_layer_id = runtime.get("_check_layer_id", 0)
                gather_contiguous_kv(forward_batch, start_layer=check_layer_id + 1, end_layer=self.end_layer)

            return hidden_states, forward_batch.positions, residual, forward_batch.out_cache_loc

        mask = runtime["cacheflex_global_mask"]

        # 1. Gather Tensors (稀疏化)
        new_hidden = hidden_states[mask]
        new_pos = positions[mask]
        new_loc = forward_batch.out_cache_loc[mask]
        new_residual = None
        if residual is not None:
            new_residual = residual[mask]

        # HACK(Shizhuo): logprobs - 同步 mask gather token_ids
        # extend_input_logprob_token_ids_gpu 只对齐 hidden_states 中
        # logprob_start_len 之后的部分，不能直接用全局 mask gather。
        # 需要计算 logprob 偏移量，提取子 mask 来 gather。
        if (hasattr(forward_batch, "extend_input_logprob_token_ids_gpu") and
            forward_batch.extend_input_logprob_token_ids_gpu is not None):
            # 计算 logprob 区域的起始偏移 = sum(extend_logprob_start_lens_cpu[i])
            logprob_offset = 0
            if (hasattr(forward_batch, "extend_logprob_start_lens_cpu") and
                forward_batch.extend_logprob_start_lens_cpu is not None):
                for s in forward_batch.extend_logprob_start_lens_cpu:
                    logprob_offset += int(s)
            else:
                logprob_offset = hidden_states.shape[0] - forward_batch.extend_input_logprob_token_ids_gpu.shape[0]
            # 提取子 mask：只取 logprob 覆盖范围内的 mask
            sub_mask = mask[logprob_offset:]
            forward_batch.extend_input_logprob_token_ids_gpu = forward_batch.extend_input_logprob_token_ids_gpu[sub_mask]

        # 2. Rebuild Metadata (CRITICAL)
        # 我们需要知道切分后，每个 Request 剩下了多少 Token
        start_loc = forward_batch.extend_start_loc
        seq_lens = forward_batch.extend_seq_lens
        new_seq_lens_list = []

        # 统计每个 Request 内 mask=True 的数量
        # (Tensor 操作可能比 Loop 快，但 Loop 逻辑最清晰)
        for i in range(len(seq_lens)):
            start = start_loc[i].item()
            length = seq_lens[i].item()
            # 切片 mask 并求和
            keep_count = mask[start : start + length].sum().item()
            new_seq_lens_list.append(keep_count)

        # Update Batch Meta
        new_seq_lens_tensor = torch.tensor(new_seq_lens_list, dtype=torch.int64, device=hidden_states.device)
        forward_batch.extend_seq_lens = new_seq_lens_tensor
        forward_batch.seq_lens = new_seq_lens_tensor
        forward_batch.positions = new_pos

        # Update Start Loc
        new_start_loc = torch.zeros_like(forward_batch.extend_start_loc)
        new_start_loc[1:] = torch.cumsum(new_seq_lens_tensor[:-1], dim=0)
        forward_batch.extend_start_loc = new_start_loc

        # Update Out Loc
        forward_batch.out_cache_loc = new_loc

        # HACK(Shizhuo): logprobs - 同步更新 extend_seq_lens_cpu 和 extend_logprob_start_lens_cpu
        # Stage 2 的 mask 会同时去掉 logprob 区域前和区域内的 tokens，
        # 不能用 req.origin_input_ids 推算，必须直接用 mask 统计。
        # 对每个 req：new_logprob_start_len = new_len - logprob_keep_count
        # 其中 logprob_keep_count = mask 在 logprob 区域内 True 的数量
        if hasattr(forward_batch, "extend_seq_lens_cpu") and forward_batch.extend_seq_lens_cpu is not None:
            forward_batch.extend_seq_lens_cpu = new_seq_lens_tensor.tolist()

            if (hasattr(forward_batch, "extend_logprob_start_lens_cpu") and
                forward_batch.extend_logprob_start_lens_cpu is not None):
                new_logprob_start_lens = []
                for i in range(len(seq_lens)):
                    req_start = start_loc[i].item()
                    req_len = seq_lens[i].item()
                    old_logprob_start = forward_batch.extend_logprob_start_lens_cpu[i]
                    # logprob 区域：[old_logprob_start, req_len)
                    # 在 mask 中的绝对范围：[req_start + old_logprob_start, req_start + req_len)
                    abs_logprob_start = req_start + old_logprob_start
                    abs_logprob_end = req_start + req_len
                    if abs_logprob_start < abs_logprob_end:
                        logprob_keep = mask[abs_logprob_start:abs_logprob_end].sum().item()
                    else:
                        logprob_keep = 0
                    new_len = int(new_seq_lens_tensor[i])
                    new_logprob_start_lens.append(max(new_len - int(logprob_keep), 0))
                forward_batch.extend_logprob_start_lens_cpu = new_logprob_start_lens

        # HACK(Shizhuo): logprobs - 同步更新 req 级别的 extend_input_len 和 extend_logprob_start_len
        # Stage 2 的 mask 会移除 logprob 区域内的 tokens，scheduler 在处理 logprob
        # 结果时依赖 req.extend_input_len 和 req.extend_logprob_start_len，必须同步更新。
        # 注意：必须在 extend_logprob_start_lens_cpu 更新之后执行。
        if hasattr(forward_batch, "reqs") and forward_batch.return_logprob:
            for i, req in enumerate(forward_batch.reqs):
                new_len = int(new_seq_lens_tensor[i])
                new_lp_start = (
                    forward_batch.extend_logprob_start_lens_cpu[i]
                    if (hasattr(forward_batch, "extend_logprob_start_lens_cpu") and
                        forward_batch.extend_logprob_start_lens_cpu and
                        i < len(forward_batch.extend_logprob_start_lens_cpu))
                    else 0
                )
                if hasattr(req, "extend_input_len"):
                    req.extend_input_len = new_len
                if hasattr(req, "extend_logprob_start_len"):
                    req.extend_logprob_start_len = new_lp_start
                # 同步更新 req.logprob_start_len 使 _calculate_relevant_tokens_len 一致
                # relevant_tokens_len = len(origin_input_ids) - logprob_start_len
                # 应等于 extend_input_len - extend_logprob_start_len（实际 logprob tokens 数）
                if hasattr(req, "origin_input_ids") and hasattr(req, "logprob_start_len"):
                    actual_logprob_tokens = new_len - new_lp_start
                    req.logprob_start_len = len(req.origin_input_ids) - actual_logprob_tokens
                
        # 注意: cacheflex_total_kv_lens 保持全长不变，因为 Context 没变

        # 重新初始化 backend metadata
        if hasattr(forward_batch, "attn_backend") and forward_batch.attn_backend:
            forward_batch.attn_backend.init_forward_metadata(forward_batch)

        # HACK(Shizhuo)
        # ============================================
        # [Contiguous KV] Gather full KV from paged cache into contiguous tensors
        # [Contiguous P0] Skip gather when using old_kv (already provided by Stage 1)
        # — was missing here (only present in llama.py and qwen2 fast path L1418-1419).
        # Without this guard, r=1.0 (mask=all-True) falls into this slow path and
        # OVERWRITES cacheflex_contiguous_kv with garbage gathered from layers that
        # haven't computed K yet → garbled output (Qwen `'10000000...'`).
        if (getattr(forward_batch, "cacheflex_kv_mode", "paged") == "contiguous"
            and not getattr(forward_batch, "cacheflex_use_old_kv", False)):
            from sglang.srt.layers.attention.cacheflex_utils import gather_contiguous_kv
            sample_task = self._find_any_task(getattr(forward_batch, "cacheflex_dynamic_tasks_by_req", []))
            check_layer = sample_task["check_layer"] if sample_task else 0
            gather_contiguous_kv(forward_batch, start_layer=check_layer + 1, end_layer=self.end_layer)

        return new_hidden, new_pos, new_residual, new_loc


class Qwen2ForCausalLM(nn.Module):
    # BitandBytes specific attributes
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        self.model = Qwen2Model(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )

        # handle the lm head on different pp ranks
        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix),
                )
        else:
            # ranks other than the last rank will have a placeholder layer
            self.lm_head = PPMissingLayer()

        # perform weight tying for PP
        if self.pp_group.world_size > 1 and config.tie_word_embeddings:
            if self.pp_group.is_first_rank:
                self.pp_group.send(
                    self.model.embed_tokens.weight, dst=self.pp_group.last_rank
                )
            elif self.pp_group.is_last_rank:
                emb_token_weight = self.pp_group.recv(
                    size=(config.vocab_size, config.hidden_size),
                    dtype=next(self.model.parameters()).dtype,
                    src=self.pp_group.first_rank,
                )
                self.lm_head.weight.copy_(emb_token_weight)

        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)
        # For EAGLE3 support
        self.capture_aux_hidden_states = False

    def get_input_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embedding(input_ids)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        if self.pp_group.is_last_rank:
            if not get_embedding:
                return self.logits_processor(
                    input_ids,
                    hidden_states,
                    self.lm_head,
                    forward_batch,
                    aux_hidden_states,
                )
            else:
                return self.pooler(hidden_states, forward_batch)
        else:
            return hidden_states

    @torch.no_grad()
    def forward_split_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        split_interval: Tuple[int, int],  # [start, end) 0-based
        input_embeds: torch.Tensor = None,
    ):
        start, end = split_interval
        # embed
        if start == 0:
            if input_embeds is None:
                forward_batch.hidden_states = self.model.embed_tokens(input_ids)
            else:
                forward_batch.hidden_states = input_embeds
        # decoder layer
        for i in range(start, end):
            layer = self.model.layers[i]
            forward_batch.hidden_states, forward_batch.residual = layer(
                positions,
                forward_batch.hidden_states,
                forward_batch,
                forward_batch.residual,
            )

        if end == self.model.config.num_hidden_layers:
            # norm
            hidden_states, _ = self.model.norm(
                forward_batch.hidden_states, forward_batch.residual
            )
            forward_batch.hidden_states = hidden_states
            # logits process
            result = self.logits_processor(
                input_ids, forward_batch.hidden_states, self.lm_head, forward_batch
            )
        else:
            result = None

        return result

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue

            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                if self.pp_group.world_size > 1 and self.pp_group.is_last_rank:
                    # Handle pp weight tying here
                    # find the embed_tokens.weight in the weights
                    embed_token_weights = next(
                        filter(lambda x: x[0] == "model.embed_tokens.weight", weights)
                    )[1]
                    loaded_weight = embed_token_weights
                else:
                    continue
            if name.startswith("model.vision_tower") and name not in params_dict:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name in params_dict.keys():
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                else:
                    logger.warning(f"Parameter {name} not found in params_dict")

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.model.load_kv_cache_scales(quantization_param_path)

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        if not self.pp_group.is_last_rank:
            return

        self.capture_aux_hidden_states = True
        if layer_ids is None:
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [
                2,
                num_layers // 2,
                num_layers - 3,
            ]  # Specific layers for EAGLE3 support
        else:
            self.model.layers_to_capture = [val + 1 for val in layer_ids]


EntryClass = Qwen2ForCausalLM
