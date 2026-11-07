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

# Adapted from
# https://github.com/vllm-project/vllm/blob/c7f2cf2b7f67bce5842fedfdba508440fe257375/vllm/model_executor/models/llama.py#L1
"""Inference-only LLaMA model compatible with HuggingFace weights."""

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import LlamaConfig

from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.activation import SiluAndMul
# HACK(Shizhuo): mem profile helper
from sglang.srt.layers.attention.cacheflex_utils import cacheflex_mem_snapshot
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
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
    maybe_remap_kv_scale_name,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, make_layers
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)

# =========================================================================
# HACK(Shizhuo) [DEBUG TOOL] Tensor Dump Helper
# =========================================================================
# HACK(Shizhuo): unified dump (gated by Category.LAYER_TENSORS)
import os
import time
from sglang.srt.utils import cacheflex_dump as _cf_dump
def dump_tensor_to_disk(name, tensor, layer_id, method):
    """Save tensor/dict/list to disk for layer-level debug analysis.

    Gated by Category.LAYER_TENSORS; output: $DUMP_DIR/<method>/layer_tensors/layer_{lid}/{name}.pt
    Pre-refactor bug: any debug_dump_name (i.e. --dump flag) would trigger 32 layer dumps,
    blowing up to ~3.4 GB per sample. Now requires explicit DUMP_LAYER_TENSORS=1 to enable.
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


class LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        reduce_results: bool = True,
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
            reduce_results=reduce_results,
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(
        self,
        x,
        forward_batch=None,
        use_reduce_scatter: bool = False,
    ):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(
            x,
            skip_all_reduce=use_reduce_scatter,
        )
        return x


class LlamaAttention(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        rope_is_neox_style: bool = True,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        bias: bool = False,
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
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
        self.head_dim = getattr(
            config, "head_dim", self.hidden_size // self.total_num_heads
        )
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1)
        self.rotary_dim = int(partial_rotary_factor * self.head_dim)
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
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.rotary_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=rope_is_neox_style,
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

        # ================= [DEBUG DUMP: INPUTS] =================
        # HACK(Shizhuo)
        # 只在 Extend (Prefill) 阶段保存，避免 Decode 阶段刷屏
        dump_name = getattr(forward_batch, "debug_dump_name", None)
        if dump_name and forward_batch.forward_mode.is_extend():
            # 获取当前 Layer ID (RadixAttention 对象通常有 layer_id 属性)
            lid = getattr(self.attn, "layer_id", -1)
            # [ProphetKV] 检查 dump_qk_only 参数
            dump_qk_only = "qk_only" in str(dump_name)

            # HACK(Shizhuo) [Oracle-Dump]: layer filter (no-op when DUMP_LAYERS unset)
            if _cf_dump.layer_allowed(lid):
                # 保存完整 Q (split-head 格式: [seq_len, num_heads, head_dim]) 用于 ProphetKV
                # HACK(Shizhuo) [Oracle-Dump]: q_full 仅在 LO_QFULL_LAYERS 指定层 dump
                # (默认全 dump); K/V 由 triton_backend 按 DUMP_LAYERS 全 dump, positions 全留。
                if _cf_dump.qfull_layer_allowed(lid):
                    q_full = q.view(q.size(0), self.num_heads, self.head_dim)
                    dump_tensor_to_disk("q_full", q_full, lid, dump_name)

                if not dump_qk_only:
                    # 完整dump模式：保存额外的调试信息
                    # 保存 RoPE 后的 Q (flatten 格式，向后兼容)
                    dump_tensor_to_disk("q_rope", q, lid, dump_name)
                    # 保存 RoPE 后的 K (虽然 triton_backend 会存全量 K，这里存切片 K 也可以作为双重验证)
                    dump_tensor_to_disk("k_rope_sliced", k, lid, dump_name)
                    # 保存 Positions
                    dump_tensor_to_disk("positions", positions, lid, dump_name)
                elif _cf_dump.layer_filter_active():
                    # HACK(Shizhuo) [Oracle-Dump]: qk_only dumps only q_full by default;
                    # positions are also dumped for cross-run (fr/naive) logical alignment.
                    dump_tensor_to_disk("positions", positions, lid, dump_name)
        # ========================================================

        # HACK(Shizhuo)
        attn_output = self.attn(
            q, k, v,
            forward_batch,
            rotary_emb=self.rotary_emb
        )

        # ================= [DEBUG DUMP: OUTPUT] =================
        # HACK(Shizhuo)
        if dump_name and forward_batch.forward_mode.is_extend():
            dump_qk_only = "qk_only" in str(dump_name)
            if not dump_qk_only:
                lid = getattr(self.attn, "layer_id", -1)
                # 保存 Attention 的输出结果
                # 如果这里错了，说明 Kernel 逻辑（Mask/Softmax）有问题
                # HACK(Shizhuo) [Oracle-Dump]: layer filter (no-op when DUMP_LAYERS unset)
                if _cf_dump.layer_allowed(lid):
                    dump_tensor_to_disk("attn_output", attn_output, lid, dump_name)
        # ========================================================

        output, _ = self.o_proj(attn_output)
        return output


class LlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_id = layer_id  # HACK(Shizhuo): Store layer_id for dump
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
            config, "original_max_position_embeddings", None
        ):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings
            )
        rope_is_neox_style = getattr(config, "rope_is_neox_style", True)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        # Support llamafy/Qwen-Qwen2.5-7B-Instruct-llamafied with attention_bias
        # Support internlm/internlm-7b with bias
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False
        )
        self.self_attn = LlamaAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            rope_is_neox_style=rope_is_neox_style,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            bias=attention_bias,
        )
        self.mlp = LlamaMLP(
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
            if residual is not None:
                dump_tensor_to_disk("layer_after_input_ln_residual", residual, layer_id, dump_name)
        # ======================================================================================

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

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
            if residual is not None:
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


class LlamaModel(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
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
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: LlamaDecoderLayer(
                config=config, quant_config=quant_config, layer_id=idx, prefix=prefix
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix="model.layers",
        )

        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)
        self.layers_to_capture = []

        # HACK(Shizhuo)
        # ============================================
        # [Plan A] LRU caching of gathered old_kv REMOVED.
        # Each request now gathers fresh and stores result only on forward_batch
        # (per-request lifecycle). Reasons:
        # 1. The LRU never hit in eval (each sample has a different prefix_hash),
        #    so it only accumulated dead memory (~1.7–2.4 GB/sample baseline growth).
        # 2. Cached old_kv required cloning before forward_batch assignment (in-place
        #    update_contiguous_kv would corrupt cache), doubling peak by ~old_kv size.
        # Both costs eliminated by per-request gather + direct assign (no clone).

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]], PPProxyTensors]:

        # HACK(Shizhuo): [Mem Profile] Entry snapshot (gated by CACHEFLEX_MEM_PROFILE=1)
        # LRU diagnostic removed — Plan A eliminated _cacheflex_old_kvs.
        cacheflex_mem_snapshot(forward_batch, "entry")

        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            assert pp_proxy_tensors is not None
            # FIXME(@ying): reduce the number of proxy tensors by not fusing layer norms
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]
            deferred_norm = None

        # HACK(Shizhuo): [CacheFlex] Slicing & Copy Point
         # [Stage 1] Initial Slicing (Before Layer 0)
        if forward_batch.forward_mode.is_extend() and forward_batch.cacheflex_configs:
             hidden_states, positions = self._apply_cacheflex_slicing_and_copy(
                 hidden_states, positions, forward_batch
             )
        # =======================================================
        # HACK(Shizhuo): [Mem Profile] After Stage 1 (slicing + gather + clone if any)
        cacheflex_mem_snapshot(forward_batch, "after_stage1")

        aux_hidden_states = []
        # HACK(Shizhuo)
        # ============================================
        _do_layer_profile = (_cf_dump.profile_timing_enabled()
                             and forward_batch.forward_mode.is_extend()
                             and getattr(forward_batch, "cacheflex_configs", None))
        if _do_layer_profile:
            _layer_start_events = {}
            _layer_end_events = {}

        for i in range(self.start_layer, self.end_layer):
            # HACK(Shizhuo)
            # [Stage 2] Post-Check Slicing (After check layer)
            check_layer = None
            sample_task = self._find_any_task(getattr(forward_batch, "cacheflex_dynamic_tasks_by_req", []))
            if sample_task:
                check_layer = sample_task["check_layer"]
            decision_layers = [check_layer] if check_layer is not None else []

            # Stage 2 post-check 切片: check_layer 之后一层 (check_layer+1) 触发一次
            if decision_layers and (i - 1) in decision_layers and forward_batch.forward_mode.is_extend():
                if _do_layer_profile:
                    _ev_s2_start = torch.cuda.Event(enable_timing=True)
                    _ev_s2_end = torch.cuda.Event(enable_timing=True)
                    _ev_s2_start.record()
                # HACK(Shizhuo)
                # ============================================================================
                # [P2 cleanup] Free L0 prefix V map before check_layer+1 Stage 2 slicing.
                # After check_layer, V map has no more readers (all token_selector /
                # kvfocus_harm_observe call sites are gated by check_layer == self.layer_id
                # in radix_attention). Freeing here reduces peak memory pressure for the
                # upcoming K/V clone in _apply_post_check_slicing (which previously
                # OOMed on hotpotqa/musique under --use-contiguous-old-kv).
                if hasattr(forward_batch, '_cacheflex_l0_prefix_v_map'):
                    del forward_batch._cacheflex_l0_prefix_v_map
                # ============================================================================
                hidden_states, positions, residual, _ = self._apply_post_check_slicing(
                    hidden_states, positions, residual, forward_batch
                )
                if _do_layer_profile:
                    _ev_s2_end.record()
                    forward_batch._cacheflex_stage2_events = {
                        "start": _ev_s2_start, "end": _ev_s2_end
                    }
            # =======================================================
            if i in self.layers_to_capture:
                aux_hidden_states.append(hidden_states + residual)
            layer = self.layers[i]

            # HACK(Shizhuo)
            # ============================================
            if _do_layer_profile:
                _ev_layer_start = torch.cuda.Event(enable_timing=True)
                _ev_layer_start.record()

            # HACK(Shizhuo): [P2 reverted] L0 split REMOVED — see qwen2.py for full
            # rationale. Aligns with vLLM/epic: every layer runs full forward for
            # all tokens before check_layer.
            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
            )

            # HACK(Shizhuo)
            # ============================================
            # [Sparse Check Layer] Sync positions local variable after check layer.
            # radix_attention.py updated forward_batch.positions to sparse, but
            # this loop uses the local `positions` which is still full-size.
            # vLLM does: positions = positions[imp_indices] after check layer.
            if (decision_layers and i in decision_layers
                and forward_batch.forward_mode.is_extend()
                and getattr(forward_batch, "cacheflex_runtime", None) is not None
                and forward_batch.cacheflex_runtime.get("_check_layer_sparse_slicing_done")):
                positions = forward_batch.positions

            if _do_layer_profile:
                _ev_layer_end = torch.cuda.Event(enable_timing=True)
                _ev_layer_end.record()
                _layer_start_events[i] = _ev_layer_start
                _layer_end_events[i] = _ev_layer_end

        if _do_layer_profile:
            forward_batch._cacheflex_layer_start_events = _layer_start_events
            forward_batch._cacheflex_layer_end_events = _layer_end_events

        # HACK(Shizhuo) [ProphetKV-Probe]: after ALL layers (still in the timed pass1 forward),
        # uniform mean-fuse the per-layer query->ctx alpha (Eq7) -> top-r -> inline external-replay
        # mask = ProphetKV Stage I incl fusion+topk. Flag/marker gated => OFF = byte-unchanged.
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
            # HACK(Shizhuo)
            # ============================================
            if _do_layer_profile:
                _ev_final_s = torch.cuda.Event(enable_timing=True)
                _ev_final_s.record()
            hidden_states, _ = self.norm(hidden_states, residual)
            # HACK(Shizhuo)
            # ============================================
            if _do_layer_profile:
                _ev_final_e = torch.cuda.Event(enable_timing=True)
                _ev_final_e.record()
                forward_batch._cacheflex_final_events = {
                    "norm_start": _ev_final_s, "norm_end": _ev_final_e
                }

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

    def get_input_embeddings(self) -> nn.Embedding:
        """Get input embeddings from the model."""
        return self.embed_tokens
    
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
    # 名字曾叫 _process_cacheblend_segment。真实职责: 不是"全算", 而是为 check_layer 的
    # 选择性重算做准备 —— compute-all 全段进 forward (好在 check_layer 算 ΔV / 应用外部
    # mask) + 铺老 KV 底座 (供未选中 token 兜底) + 登记 check_layer 任务 (延迟到那层再选)。
    # mask 来源 (scorer 在线算 / 外部注入) 与本函数无关, 由 radix_attention 决定。
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
    # HACK(Shizhuo)
    # ============================================
    # [Contiguous P0] Gather warmup KV from src_block_indices
    # ========================================================================
    def _gather_kv_from_src_blocks(self, forward_batch, prefix_hash,
                                    dst_locs_per_source=None):
        """Gather KV from token_to_kv_pool by src_block_indices into contiguous tensors.

        Key points:
        1. Cannot use req_to_token[:total_kv_len] because CB request's req_to_token
           maps newly allocated empty physical pages. Warmup KV in radix tree is
           indexed by src_block_indices, must read from token_to_kv_pool directly.
        2. Each source's src_block_indices includes sink tokens that duplicate the
           preceding source's KV. Must skip sink_token_len per source before
           concatenating to avoid duplication and gathered_len > total_kv_len.
        3. Each source may have rope_delta != 0 (sys-prefix warmup strategy).
           K tensors are rotated by rope_delta positions in-place (Delta-RoPE).
           V tensors need no rotation (RoPE only affects K).
        4. old_kv tensor size = cacheflex_total_kv_lens[0], prefix part filled
           with warmup KV (deduplicated + Delta-RoPE adjusted), suffix part
           zero-initialized (filled by update_contiguous_kv).
        5. If dst_locs_per_source is provided, also write gathered KV (with
           Delta-RoPE applied) to paged cache at those physical locations.
           This eliminates the need for complementary_copy at every layer.
        """
        bs = forward_batch.batch_size
        assert bs == 1, "old_kv gather only supports batch_size=1"

        # Use cacheflex_total_kv_lens (set during Batch Reconstruction)
        total_kv_len = forward_batch.cacheflex_total_kv_lens[0].item()

        # Build per-source info: indices after sink skip + rope_delta
        cfg = forward_batch.cacheflex_configs[0]
        src_info_list = []  # (indices, rope_delta) per source

        for src in cfg["sources"]:
            if src.get("type") == "text_hit" and src.get("src_block_indices") is not None:
                indices = src["src_block_indices"]
                sink_len = src.get("sink_token_len", 0)
                rope_delta = src.get("rope_delta", 0)

                if sink_len > 0:
                    # Skip sink portion — already covered by a preceding source
                    indices = indices[sink_len:]

                logger.info(f"[Contiguous P0] Source: sink_len={sink_len}, "
                           f"rope_delta={rope_delta}, indices_len={len(indices)}, "
                           f"full_src_len={len(src['src_block_indices'])}")
                src_info_list.append((indices, rope_delta))

        if not src_info_list:
            logger.warning("[Contiguous P0] No src_block_indices found, cannot gather")
            # HACK(Shizhuo): [Plan A] return None tuple so caller can detect failure
            return None, None, 0

        # Compute gathered_len from all segments
        gathered_len = sum(len(indices) for indices, _ in src_info_list)

        # Safety guard (should not trigger after sink skip, but defensive)
        if gathered_len > total_kv_len:
            logger.warning(f"[Contiguous P0] gathered_len ({gathered_len}) > "
                          f"total_kv_len ({total_kv_len}), truncating")
            gathered_len = total_kv_len

        # Get rotary_emb from first layer (shared RoPE config across all layers)
        rotary_emb = getattr(self.layers[self.start_layer].self_attn, "rotary_emb", None)

        # Debug: log rotary_emb properties
        if rotary_emb is not None:
            logger.info(f"[Contiguous P0] rotary_emb: is_neox={getattr(rotary_emb, 'is_neox_style', True)}, "
                       f"rotary_dim={getattr(rotary_emb, 'rotary_dim', 'N/A')}, "
                       f"head_size={getattr(rotary_emb, 'head_size', 'N/A')}, "
                       f"cos_sin_cache shape={rotary_emb.cos_sin_cache.shape if hasattr(rotary_emb, 'cos_sin_cache') else 'N/A'}")

        # Allocate total_kv_len sized contiguous tensors
        # Prefix part filled with warmup KV, suffix part zero-initialized
        # Suffix part will be filled by update_contiguous_kv with online token KV
        old_kv = {}
        for layer_idx in range(self.start_layer, self.end_layer):
            k_buf = forward_batch.token_to_kv_pool.get_key_buffer(layer_idx)
            v_buf = forward_batch.token_to_kv_pool.get_value_buffer(layer_idx)

            k_cont = torch.zeros((total_kv_len, k_buf.shape[1], k_buf.shape[2]),
                                 dtype=k_buf.dtype, device=k_buf.device)
            v_cont = torch.zeros((total_kv_len, v_buf.shape[1], v_buf.shape[2]),
                                 dtype=v_buf.dtype, device=v_buf.device)

            # Per-source copy with Delta-RoPE for K
            offset = 0
            for src_idx, (indices, rope_delta) in enumerate(src_info_list):
                doc_len = len(indices)
                if offset + doc_len > total_kv_len:
                    doc_len = total_kv_len - offset
                    indices = indices[:doc_len]
                if doc_len <= 0:
                    break

                # Copy V (no RoPE adjustment needed — RoPE only affects K)
                v_cont[offset:offset+doc_len] = v_buf[indices]

                if rope_delta != 0 and rotary_emb is not None:
                    # Copy K then apply Delta-RoPE rotation in-place
                    k_cont[offset:offset+doc_len] = k_buf[indices]
                    self._apply_rope_delta_to_k(
                        k_cont[offset:offset+doc_len], rope_delta, rotary_emb
                    )
                else:
                    # No rotation needed (rope_delta=0 or no rotary_emb)
                    k_cont[offset:offset+doc_len] = k_buf[indices]

                # Write gathered KV (with Delta-RoPE) to paged cache
                # This eliminates the need for complementary_copy at every layer
                if dst_locs_per_source is not None and src_idx < len(dst_locs_per_source):
                    dst_loc = dst_locs_per_source[src_idx]
                    if dst_loc is not None and len(dst_loc) == doc_len:
                        v_buf[dst_loc.to(torch.long)] = v_cont[offset:offset+doc_len]
                        k_buf[dst_loc.to(torch.long)] = k_cont[offset:offset+doc_len]

                offset += doc_len

            old_kv[layer_idx] = (k_cont, v_cont)

        # Debug: log K statistics for layer 0 to verify Delta-RoPE
        k0 = old_kv[self.start_layer][0]
        v0 = old_kv[self.start_layer][1]
        logger.info(f"[Contiguous P0] Layer 0 K stats: "
                    f"mean={k0[:gathered_len].abs().mean().item():.6f}, "
                    f"max={k0[:gathered_len].abs().max().item():.6f}, "
                    f"zero_frac={((k0[:gathered_len]==0).sum().item() / k0[:gathered_len].numel()):.4f}, "
                    f"V mean={v0[:gathered_len].abs().mean().item():.6f}, "
                    f"suffix_zero_frac={((k0[gathered_len:]==0).sum().item() / max(k0[gathered_len:].numel(),1)):.4f}")

        cu_seq_lens = torch.tensor([0, total_kv_len], dtype=torch.int32,
                                   device=forward_batch.seq_lens.device)

        # HACK(Shizhuo): [Plan A] return result instead of self-storage.
        # No LRU caching (caller owns lifecycle via forward_batch).
        logger.info(f"[Contiguous P0] Gathered old_kv: prefix_hash={prefix_hash[:8] if prefix_hash else 'none'}, "
                    f"total_kv_len={total_kv_len}, gathered_prefix_len={gathered_len}, "
                    f"layers={self.end_layer - self.start_layer}")
        return old_kv, cu_seq_lens, gathered_len

    # HACK(Shizhuo)
    # ============================================
    def _apply_rope_delta_to_k(self, k_segment, rope_delta, rotary_emb):
        """Apply Delta-RoPE rotation to K tensor segment in-place.

        Numerically equivalent to complementary_copy's Delta-RoPE logic:
        1. Handle negative delta by flipping second-half signs (Neox style)
        2. Create delta_positions tensor with abs(rope_delta) for all tokens
        3. Call apply_rope_with_cos_sin_cache_inplace (in-place on key, no fused arg)
        4. For negative delta: flip second-half signs again
        """
        from sgl_kernel import apply_rope_with_cos_sin_cache_inplace

        num_tokens = k_segment.shape[0]
        head_dim = k_segment.shape[2]

        is_neox_style = getattr(rotary_emb, "is_neox_style", True)
        rotary_dim = getattr(rotary_emb, "rotary_dim", head_dim)
        cos_sin_cache = rotary_emb.cos_sin_cache

        is_negative_delta = rope_delta < 0
        abs_delta = abs(int(rope_delta))

        if is_negative_delta:
            half = rotary_dim // 2
            k_segment[..., half:rotary_dim] *= -1

        delta_positions = torch.full(
            (num_tokens,), abs_delta, dtype=torch.int64, device=k_segment.device
        )

        # Dummy Q (kernel requires both Q and K; we only care about K rotation)
        dummy_q = torch.empty_like(k_segment)

        apply_rope_with_cos_sin_cache_inplace(
            positions=delta_positions,
            query=dummy_q,
            key=k_segment,
            head_size=int(head_dim),
            cos_sin_cache=cos_sin_cache,
            is_neox=is_neox_style,
        )

        if is_negative_delta:
            half = rotary_dim // 2
            k_segment[..., half:rotary_dim] *= -1

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
            # v_prefix rows are prefix tokens in sliced order (all cacheblend segments, no naive/epic)
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
            # L1 needs all tokens for qkv_proj → scorer K for probe scores
            hidden_states = torch.empty_like(hidden_states if stale_prefix_h is None else
                                             torch.cat([stale_prefix_h, hidden_suffix]))
            hidden_states[prefix_mask] = stale_prefix_h
            hidden_states[suffix_mask] = hidden_suffix

            # Reassemble residual: prefix gets 0 (L1 input_layernorm treats as residual=None),
            # suffix gets L0 residual (attn_output + input from L0 suffix forward)
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
        new_extend_logprob_start_lens = []  # HACK(Shizhuo): logprobs 支持
        l0_suffix_starts_list = []  # [P2] suffix start offset in sliced sequence per request
        # HACK(Shizhuo): KVFocus probe range — start of LAST source in sliced coords
        # (see qwen2.py for full rationale)
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

            # HACK(Shizhuo): logprobs - 初始化当前 req 的 kept_before 计数器
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
                # HACK(Shizhuo): logprobs - 普通请求没有切片，整个 [0, req_len) 都保留
                add_keep_range(0, req_len)
                new_seq_lens.append(req_len)
                total_kv_lens.append(cached_len + req_len)
                slice_info.append({"is_sliced": False})
                # 保存 new_start_len
                new_extend_logprob_start_lens.append(kept_before)
                l0_suffix_starts_list.append(0)  # [P2] no prefix/suffix split for non-CacheFlex
                probe_starts_list.append(0)
                continue

            # --- Case 2: CacheFlex 请求 ---
            has_slicing = True

            # 初始化分段列表
            parts_collector = {'h': [], 'p': [], 'l': [], 'info': []}

            # input_ptr: 相对 start 的偏移量，强制从 0 开始
            current_input_ptr = 0

            # [P0] Track offset into contiguous buffer for compute_diff fast path.
            # Must cover ALL text_hit sources (including naive/epic), matching _gather_kv_from_src_blocks.
            contiguous_kv_offset = 0

            # HACK(Shizhuo): start of LAST processed source in sliced coords
            probe_start_sliced = 0

            # --- 2. Source Documents Loop (Doc1, Doc2, ...) ---
            # [MODIFIED] 遍历 sources 列表进行多段切片
            for i, src_info in enumerate(cfg["sources"]):
                src_len = src_info["src_len"]
                plan = src_info["plan"]
                # 边界检查
                if current_input_ptr + src_len > req_len: break # 防止溢出

                # HACK(Shizhuo): overwrites each iter → after loop = last source's start
                probe_start_sliced = sum(t.shape[0] for t in parts_collector['h'])

                method = src_info["plan"].method
                # HACK(Shizhuo) [Oracle-Dump]: cacheblend / kvfocus / external / layer_refresh
                # 都是 dynamic method, 共享 selective-recompute Stage-1 (compute-all + 铺 KV
                # 底座 + 登记 check_layer 任务)。三者区别只在 check_layer 处 mask 来源:
                # cacheblend=M scorer / kvfocus=M×U scorer / external=外部注入 (见 radix_attention)。
                if method in ("cacheblend", "layer_refresh", "external"):
                    self._process_selective_recompute_segment(
                        hidden_states, positions, req_out_loc,
                        start, current_input_ptr,
                        src_info, parts_collector,
                        forward_batch.cacheflex_dynamic_tasks_by_req[req_idx],
                        req_idx,
                        None,  # HACK(Shizhuo): 不再传递 token_ids，因为它是 windowed 的
                        contiguous_kv_offset,  # [P0] offset in contiguous buffer
                    )
                    # 明确 Kind
                    last_task = forward_batch.cacheflex_dynamic_tasks_by_req[req_idx][-1]
                    if method == "layer_refresh":
                        last_task["kind"] = "drop_all"
                    else:
                        last_task["kind"] = "select_topk"
                    # HACK(Shizhuo): logprobs - cacheblend 全量保留，全部加入统计
                    add_keep_range(current_input_ptr, current_input_ptr + src_len)

                elif method in ["naive", "epic"]:
                    self._process_static_segment(
                        hidden_states, positions, req_out_loc,
                        start, current_input_ptr,
                        src_info, parts_collector,
                        req_idx,
                        None  # HACK(Shizhuo): 不再传递 token_ids
                    )
                    # HACK(Shizhuo): logprobs - naive/epic 只有 compute_range 保留，只统计这部分
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
                    # HACK(Shizhuo): logprobs - full 段全量保留，全部加入统计
                    add_keep_range(current_input_ptr, current_input_ptr + src_len)

                # 移动指针
                current_input_ptr += src_len

                # [P0] Accumulate contiguous buffer offset for all text_hit sources
                if src_info.get("type") == "text_hit" and src_info.get("src_block_indices") is not None:
                    contiguous_kv_offset += src_len

            # --- 3. Suffix (Query / User Input) ---
            # [P2] Record suffix start in sliced sequence (after all source segments)
            # This is the offset within the sliced hidden_states, not the original sequence.
            # Used by _l0_split_and_run to separate prefix (cached) from suffix (new) tokens.
            l0_suffix_start_sliced = sum(t.shape[0] for t in parts_collector['h'])

            # 将剩余部分全部保留
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
                    parts_collector['h'].append(hidden_states[start + current_input_ptr : end])
                    parts_collector['p'].append(positions[start + current_input_ptr : end])
                    parts_collector['l'].append(req_out_loc[current_input_ptr : req_len])
                    # HACK(Shizhuo): logprobs - suffix 全量保留，全部加入统计
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
            # HACK(Shizhuo): 保存当前 req 的新 start_len
            new_extend_logprob_start_lens.append(kept_before)
            # [P2] Record suffix start in sliced sequence for L0 prefix/suffix split
            l0_suffix_starts_list.append(l0_suffix_start_sliced)
            probe_starts_list.append(probe_start_sliced)  # HACK(Shizhuo): for KVFocus

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
            forward_batch._cacheflex_probe_starts = probe_starts_list  # HACK(Shizhuo)

            # HACK(Shizhuo)
            # ============================================================================
            # [Plan A] If using old_kv, gather fresh per-request (no LRU, no clone).
            # See LlamaModel.__init__ for the rationale on removing the LRU cache.
            # The cache-hit branch is gone — gather always runs. Direct-assign to
            # forward_batch (no clone needed since there is no shared cache to protect).
            if getattr(forward_batch, "cacheflex_use_old_kv", False):
                _do_profile_s1 = (_cf_dump.profile_timing_enabled())
                if _do_profile_s1:
                    _ev_s1_assign = torch.cuda.Event(enable_timing=True)

                prefix_hash = getattr(forward_batch, "cacheflex_prefix_hash", None)
                if prefix_hash is not None:
                    # Build per-source dst_tail_loc for paged cache write-back.
                    # src_len in metadata is effective_src_len (= full_src_len - sink_len),
                    # which matches the number of tokens gather produces.
                    dst_locs_per_source = []
                    src_ptr = 0
                    for src in cfg["sources"]:
                        if src.get("type") == "text_hit" and src.get("src_block_indices") is not None:
                            src_len_i = src["src_len"]
                            dst_locs_per_source.append(
                                req_out_loc[src_ptr : src_ptr + src_len_i]
                            )
                            src_ptr += src_len_i
                        else:
                            dst_locs_per_source.append(None)

                    # CUDA event timing for gather
                    gather_start = torch.cuda.Event(enable_timing=True)
                    gather_end = torch.cuda.Event(enable_timing=True)
                    gather_start.record()
                    old_kv, cu_seq_lens, _gathered_len = self._gather_kv_from_src_blocks(
                        forward_batch, prefix_hash, dst_locs_per_source
                    )
                    gather_end.record()
                    forward_batch._cacheflex_gather_start_event = gather_start
                    forward_batch._cacheflex_gather_end_event = gather_end
                    # HACK(Shizhuo): [Mem Profile] After gather
                    cacheflex_mem_snapshot(forward_batch, "after_gather")

                    if old_kv is not None:
                        # [Plan A] Direct-assign (no clone). update_contiguous_kv will
                        # mutate the gathered tensors in-place, which is now safe because
                        # there is no shared owner (LRU removed).
                        forward_batch.cacheflex_contiguous_kv = old_kv
                        forward_batch.cacheflex_contiguous_cu_seq_lens = cu_seq_lens
                        forward_batch.cacheflex_kv_mode = "contiguous"
                        # Skip complementary_copy: gather already wrote warmup KV to paged cache
                        forward_batch.cacheflex_skip_complementary_copy = True

                        # HACK(Shizhuo): [Mem Profile] After direct assign.
                        # Pre-Plan-A this was "after_clone" — keep the tag for JSON schema
                        # compat. Now this should show ~0 delta vs after_gather (no clone).
                        if _do_profile_s1:
                            _ev_s1_assign.record()
                        cacheflex_mem_snapshot(forward_batch, "after_clone")

                        if _do_profile_s1:
                            if not hasattr(forward_batch, "_cacheflex_stage1_events"):
                                forward_batch._cacheflex_stage1_events = {}
                            # "clone_end" key retained for profiler compat. Now marks the
                            # direct-assign point; clone_ms should be ~0 post-Plan-A.
                            forward_batch._cacheflex_stage1_events["clone_end"] = _ev_s1_assign
                    else:
                        logger.warning(
                            f"[Plan A] gather returned None for prefix_hash={prefix_hash}, "
                            f"falling back to paged+complementary_copy path"
                        )
            # ============================================================================
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

        # [Sparse Check Layer] Fast path: L1 already sparsified Q/K/V and
        # updated forward_batch metadata. Skip redundant mask gather.
        if runtime.get("_check_layer_sparse_slicing_done"):
            # logprobs sync: mask is based on the ORIGINAL full-size metadata,
            # but extend_start_loc/seq_lens are already sparse. We need the
            # original values for logprob offset calculation, which were saved
            # by _apply_cacheflex_slicing_and_copy at L0.
            # Since L1 sparse already handled the token filtering, the logprob
            # token_ids should already be consistent with sparse hidden_states.
            # Skip the mask-based logprobs gather (it would double-filter).

            # req-level metadata: seq_lens already updated by L1 sparsify,
            # but extend_logprob_start_lens_cpu and req-level fields need update.
            if hasattr(forward_batch, "extend_seq_lens_cpu") and forward_batch.extend_seq_lens_cpu is not None:
                if (hasattr(forward_batch, "extend_logprob_start_lens_cpu") and
                    forward_batch.extend_logprob_start_lens_cpu is not None):
                    # Recompute logprob start lens based on sparse seq_lens
                    # This requires the original full-size metadata which we
                    # don't have here. For now, keep existing values as they
                    # were set by _apply_cacheflex_slicing_and_copy at L0
                    # and should still be approximately correct.
                    pass

            # Reinit backend metadata (needed for L2+ layers)
            if hasattr(forward_batch, "attn_backend") and forward_batch.attn_backend:
                forward_batch.attn_backend.init_forward_metadata(forward_batch)

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

        # [Contiguous KV] Gather full KV from paged cache into contiguous tensors
        # [Contiguous P0] Skip gather when using old_kv (already provided by Step 3)
        if (getattr(forward_batch, "cacheflex_kv_mode", "paged") == "contiguous"
            and not getattr(forward_batch, "cacheflex_use_old_kv", False)):
            from sglang.srt.layers.attention.cacheflex_utils import gather_contiguous_kv
            # Find check_layer to determine start_layer for gather
            sample_task = self._find_any_task(getattr(forward_batch, "cacheflex_dynamic_tasks_by_req", []))
            check_layer = sample_task["check_layer"] if sample_task else 0
            gather_contiguous_kv(forward_batch, start_layer=check_layer + 1, end_layer=self.end_layer)

        return new_hidden, new_pos, new_residual, new_loc


class LlamaForCausalLM(nn.Module):
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
    # in TP, these weights are partitioned along the column dimension (dim=-1)
    column_parallel_weights_modules = [".down_proj.", ".o_proj."]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        ".q_proj": (".qkv_proj", 0),
        ".k_proj": (".qkv_proj", 1),
        ".v_proj": (".qkv_proj", 2),
        ".gate_proj": (".gate_up_proj", 0),
        ".up_proj": (".gate_up_proj", 1),
    }

    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        self.model = self._init_model(config, quant_config, add_prefix("model", prefix))
        # Llama 3.2 1B Instruct set tie_word_embeddings to True
        # Llama 3.1 8B Instruct set tie_word_embeddings to False
        if self.config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
                use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
            )
        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)
        self.stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        self.capture_aux_hidden_states = False

    def _init_model(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        return LlamaModel(config, quant_config=quant_config, prefix=prefix)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> LogitsProcessorOutput:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        # HACK(Shizhuo)
        # ============================================
        # # [DEBUG START] 在 return 之前检查 hidden_states
        # if torch.isnan(hidden_states).any():
        #     logger.error(f"[FATAL] Hidden States contain NaN! (Rank {self.pp_group.rank_in_group})")
        # elif hidden_states.abs().max() > 1e4: 
        #     logger.warning(f"[WARN] Hidden States exploding? Max={hidden_states.abs().max()}") 
        # else: 
        #     logger.info(f"[DEBUG] Final Hidden States OK. Max={hidden_states.abs().max():.2f}") 
        # # [DEBUG END]

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        if self.pp_group.is_last_rank:
            if not get_embedding:
                # HACK(Shizhuo)
                # ============================================
                _do_final_profile = (
                    _cf_dump.profile_timing_enabled()
                    and forward_batch.forward_mode.is_extend()
                    and getattr(forward_batch, "cacheflex_configs", None)
                )
                if _do_final_profile:
                    _ev_lm_s = torch.cuda.Event(enable_timing=True)
                    _ev_lm_s.record()
                logits_out = self.logits_processor(
                    input_ids,
                    hidden_states,
                    self.lm_head,
                    forward_batch,
                    aux_hidden_states,
                )
                # HACK(Shizhuo)
                # ============================================
                if _do_final_profile:
                    _ev_lm_e = torch.cuda.Event(enable_timing=True)
                    _ev_lm_e.record()
                    _final_events = getattr(forward_batch, "_cacheflex_final_events", None)
                    if _final_events is not None:
                        _final_events["lm_start"] = _ev_lm_s
                        _final_events["lm_end"] = _ev_lm_e
                    else:
                        forward_batch._cacheflex_final_events = {
                            "lm_start": _ev_lm_s, "lm_end": _ev_lm_e
                        }
                return logits_out
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
    ) -> Optional[LogitsProcessorOutput]:
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

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def get_module_name_from_weight_name(self, name):
        for param_name, weight_name, shard_id, num_shard in self.stacked_params_mapping:
            if weight_name in name:
                return (
                    name.replace(weight_name, param_name)[: -len(".weight")],
                    num_shard,
                )
        return name[: -len(".weight")], 1

    def get_num_params(self):
        params_dict = dict(self.named_parameters())
        return len(params_dict)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
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
            if name.startswith("model.vision_tower") and name not in params_dict:
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue
            # Handle FP8 kv-scale remapping
            if "scale" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
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
                # Skip loading kv_scale from ckpts towards new design.
                if name.endswith(".kv_scale") and name not in params_dict:
                    continue
                if name in params_dict.keys():
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                else:
                    logger.warning(f"Parameter {name} not found in params_dict")

    def get_weights_by_name(
        self, name: str, truncate_size: int = 100, tp_size: int = 1
    ) -> Optional[torch.Tensor]:
        """Get the weights of the parameter by its name. Similar to `get_parameter` in Hugging Face.

        Only used for unit test with an unoptimized performance.
        For optimized performance, please use torch.save and torch.load.
        """
        try:
            if name == "lm_head.weight" and self.config.tie_word_embeddings:
                logger.info(
                    "word embedding is tied for this model, return embed_tokens.weight as lm_head.weight."
                )
                return (
                    self.model.embed_tokens.weight.cpu()
                    .to(torch.float32)
                    .numpy()
                    .tolist()[:truncate_size]
                )

            mapped_name = name
            mapped_shard_id = None
            for param_name, weight_name, shard_id in self.stacked_params_mapping:
                if weight_name in name:
                    mapped_name = name.replace(weight_name, param_name)
                    mapped_shard_id = shard_id
                    break
            params_dict = dict(self.named_parameters())
            param = params_dict[mapped_name]
            if mapped_shard_id is not None:
                if mapped_shard_id in ["q", "k", "v"]:
                    num_heads = self.config.num_attention_heads // tp_size
                    num_kv_heads = self.config.num_key_value_heads // tp_size
                    head_dim = (
                        self.config.hidden_size // self.config.num_attention_heads
                    )
                    if mapped_shard_id == "q":
                        offset = 0
                        size = num_heads * head_dim
                    elif mapped_shard_id == "k":
                        offset = num_heads * head_dim
                        size = num_kv_heads * head_dim
                    elif mapped_shard_id == "v":
                        offset = (num_heads + num_kv_heads) * head_dim
                        size = num_kv_heads * head_dim
                    weight = param.data.narrow(0, offset, size)
                elif mapped_shard_id in [0, 1]:
                    intermediate_size = self.config.intermediate_size
                    slice_size = intermediate_size // tp_size
                    if mapped_shard_id == 0:  # gate_proj
                        offset = 0
                        size = slice_size
                    elif mapped_shard_id == 1:  # up_proj
                        offset = slice_size
                        size = slice_size

                    weight = param.data.narrow(0, offset, size)
                else:
                    weight = param.data
            else:
                weight = param.data
            if tp_size > 1 and ("o_proj" in name or "down_proj" in name):
                gathered_weights = [torch.zeros_like(weight) for _ in range(tp_size)]
                torch.distributed.all_gather(gathered_weights, weight)
                weight = torch.cat(gathered_weights, dim=1)
            return weight.cpu().to(torch.float32).numpy().tolist()[:truncate_size]

        except Exception:
            logger.error(
                f"Error getting weights by name {name} in LlamaForCausalLM: {get_exception_traceback()}"
            )
            return None

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def get_embed(self):
        return self.model.embed_tokens.weight

    def set_embed(self, embed):
        # NOTE: If draft hidden size != target hidden size, the embed weight cannot be shared for EAGLE3
        if (
            hasattr(self.config, "target_hidden_size")
            and self.config.target_hidden_size != self.config.hidden_size
        ):
            return
        del self.model.embed_tokens.weight
        self.model.embed_tokens.weight = embed
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.model.load_kv_cache_scales(quantization_param_path)

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        if not self.pp_group.is_last_rank:
            return

        if layer_ids is None:
            self.capture_aux_hidden_states = True
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [2, num_layers // 2, num_layers - 3]
        else:
            self.capture_aux_hidden_states = True
            # we plus 1 here because in sglang, for the ith layer, it takes the output
            # of the (i-1)th layer as aux hidden state
            self.model.layers_to_capture = [val + 1 for val in layer_ids]


class Phi3ForCausalLM(LlamaForCausalLM):
    pass


class InternLM3ForCausalLM(LlamaForCausalLM):
    pass


EntryClass = [LlamaForCausalLM, Phi3ForCausalLM, InternLM3ForCausalLM]
