# Adapted from qwen2.py
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn

from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes
from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import QKVParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.models.qwen2 import Qwen2MLP as Qwen3MLP
from sglang.srt.models.qwen2 import Qwen2Model
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (
    add_prefix,
    get_cmo_stream,
    is_cuda,
    is_npu,
    wait_cmo_stream,
)

Qwen3Config = None

logger = logging.getLogger(__name__)
_is_cuda = is_cuda()
_is_npu = is_npu()

if _is_npu:
    from sgl_kernel_npu.norm.split_qkv_rmsnorm_rope import split_qkv_rmsnorm_rope


# =========================================================================
# HACK(Shizhuo) [DEBUG TOOL] Tensor Dump Helper
# =========================================================================
# HACK(Shizhuo): unified dump (gated by Category.LAYER_TENSORS)
import os
from sglang.srt.utils import cacheflex_dump as _cf_dump
def dump_tensor_to_disk(name, tensor, layer_id, method):
    """Save tensor for layer-level debug analysis.

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
        torch.save(tensor.detach().cpu(), save_dir / f"{name}.pt")
    except Exception as e:
        print(f"[CacheFlex LAYER_TENSORS dump ERROR] Failed to save {name}: {e}")
# =========================================================================


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 1000000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        head_dim: Optional[int] = None,
        max_position_embeddings: int = 32768,
        quant_config: Optional[QuantizationConfig] = None,
        rms_norm_eps: float = None,
        attention_bias: bool = False,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= attn_tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.tp_rank = get_tensor_model_parallel_rank()

        norm_kwargs = (
            dict(
                weight_dtype=torch.float32,
                cast_x_before_out_mul=True,
            )
            if get_global_server_args().rl_on_policy_target is not None
            else {}
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps, **norm_kwargs)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps, **norm_kwargs)

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
        )
        self.alt_stream = alt_stream

    def _apply_qk_norm(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # overlap qk norm
        if self.alt_stream is not None and get_is_capture_mode():
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            q_by_head = q.reshape(-1, self.head_dim)
            q_by_head = self.q_norm(q_by_head)
            with torch.cuda.stream(self.alt_stream):
                k_by_head = k.reshape(-1, self.head_dim)
                k_by_head = self.k_norm(k_by_head)
            current_stream.wait_stream(self.alt_stream)
        else:
            q_by_head = q.reshape(-1, self.head_dim)
            q_by_head = self.q_norm(q_by_head)
            k_by_head = k.reshape(-1, self.head_dim)
            k_by_head = self.k_norm(k_by_head)
        q = q_by_head.view(q.shape)
        k = k_by_head.view(k.shape)
        return q, k

    def forward_prepare_native(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self._apply_qk_norm(q, k)
        q, k = self.rotary_emb(positions, q, k)
        return q, k, v

    def forward_prepare_npu(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)

        if self.attn.layer_id == 0:
            self.rotary_emb.get_cos_sin_with_position(positions)
        q, k, v = split_qkv_rmsnorm_rope(
            qkv,
            self.rotary_emb.position_sin,
            self.rotary_emb.position_cos,
            self.q_norm.weight,
            self.k_norm.weight,
            self.q_size,
            self.kv_size,
            self.head_dim,
            self.q_norm.variance_epsilon,
            q_bias=getattr(self.q_norm, "bias", None),
            k_bias=getattr(self.k_norm, "bias", None),
        )
        return q, k, v

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if get_global_server_args().rl_on_policy_target is not None:
            hidden_states = hidden_states.bfloat16()

        if not _is_npu:
            q, k, v = self.forward_prepare_native(
                positions=positions,
                hidden_states=hidden_states,
            )
        else:
            q, k, v = self.forward_prepare_npu(
                positions=positions,
                hidden_states=hidden_states,
            )

        if get_global_server_args().rl_on_policy_target is not None:
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)

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
                # ProphetKV Stage I 必需
                # HACK(Shizhuo) [Oracle-Dump]: q_full 仅在 LO_QFULL_LAYERS 指定层 dump
                # (默认全 dump); 与 llama.py/qwen2.py 行为一致。
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
                    # HACK(Shizhuo) [Oracle-Dump]: qk_only 补 positions 供跨 run 逻辑序对齐
                    dump_tensor_to_disk("positions", positions, lid, dump_name)
        # ========================================================

        # attn_output = self.attn(q, k, v, forward_batch)
        # HACK(Shizhuo)
        attn_output = self.attn(
            q, k, v, 
            forward_batch,
            rotary_emb=self.rotary_emb # <--- Add this
        )

        # ================= [DEBUG DUMP: OUTPUT] =================
        # HACK(Shizhuo)
        if dump_name and forward_batch.forward_mode.is_extend() and not dump_qk_only:
            lid = getattr(self.attn, "layer_id", -1)

            # 保存 Attention 的输出结果
            # 如果这里错了，说明 Kernel 逻辑（Mask/Softmax）有问题
            # HACK(Shizhuo) [Oracle-Dump]: layer filter (no-op when DUMP_LAYERS unset)
            if _cf_dump.layer_allowed(lid):
                dump_tensor_to_disk("attn_output", attn_output, lid, dump_name)
        # ========================================================

        output, _ = self.o_proj(attn_output)
        return output


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # HACK(Shizhuo): Store layer_id for dump
        self.layer_id = layer_id
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 32768)
        head_dim = getattr(config, "head_dim", None)
        self.self_attn = Qwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=config.attention_bias,
            prefix=add_prefix("self_attn", prefix),
            alt_stream=alt_stream,
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

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
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, **norm_kwargs
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, **norm_kwargs
        )

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=False,
            is_previous_layer_sparse=False,
        )
        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # HACK(Shizhuo): [DEBUG DUMP: INPUT]
        dump_name = getattr(forward_batch, "debug_dump_name", None)
        # [ProphetKV] qk_only 模式检查 - 跳过所有层内 dump
        dump_qk_only = dump_name and "qk_only" in str(dump_name)
        layer_id = self.layer_id

        if dump_name and not dump_qk_only and forward_batch.forward_mode.is_extend() and layer_id <= 2:
            dump_tensor_to_disk("layer_input_hidden", hidden_states, layer_id, dump_name)
            if residual is not None:
                dump_tensor_to_disk("layer_input_residual", residual, layer_id, dump_name)

        # Self Attention
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch
        )

        # HACK(Shizhuo): [DEBUG DUMP: AFTER_INPUT_LN]
        if dump_name and not dump_qk_only and forward_batch.forward_mode.is_extend() and layer_id <= 2:
            dump_tensor_to_disk("after_input_ln", hidden_states, self.layer_id, dump_name)

        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        # HACK(Shizhuo) [2026-05-27 Qwen3 sparse-path residual fix]
        # ============================================================
        # [Sparse Check Layer] Slice residual after attention output.
        # hidden_states is already sparse (attention only processed sparse Q at
        # check_layer). residual is still full-size from prepare_attn (input
        # layernorm bypass), must slice before prepare_mlp's fused_add_rmsnorm
        # which requires matching shapes.
        #
        # Mirrors the [Sparse Check Layer] block in llama.py and qwen2.py. Without this, Qwen3 +
        # any sparse-path method (cacheblend / kvfocus / kvfocus-adaptive) crashes
        # at the second call site of prepare_mlp's fused_add_rmsnorm:
        #   RuntimeError: CHECK_EQ(input.size(0), residual.size(0)) failed.
        #   376 vs 7364
        # where 376 = sparse active tokens, 7364 = full prompt length.
        runtime = getattr(forward_batch, "cacheflex_runtime", None)
        if (runtime is not None
            and runtime.get("_check_layer_global_mask") is not None
            and not runtime.get("_check_layer_sparse_slicing_done", False)
            and self.layer_id == runtime.get("_check_layer_id", -1)):
            mask = runtime["_check_layer_global_mask"]
            residual = residual[mask]
            runtime["_check_layer_sparse_slicing_done"] = True
        # ============================================================

        # HACK(Shizhuo): [DEBUG DUMP: AFTER_ATTN]
        if dump_name and not dump_qk_only and forward_batch.forward_mode.is_extend() and layer_id <= 2:
            dump_tensor_to_disk("after_attn", hidden_states, self.layer_id, dump_name)

        # Fully Connected
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states,
            residual,
            forward_batch,
            cache=(
                [self.mlp.gate_up_proj.weight, self.mlp.down_proj.weight]
                if _is_npu
                else None
            ),
        )

        # HACK(Shizhuo): [DEBUG DUMP: AFTER_POST_LN]
        if dump_name and not dump_qk_only and forward_batch.forward_mode.is_extend() and layer_id <= 2:
            dump_tensor_to_disk("after_post_ln", hidden_states, self.layer_id, dump_name)

        hidden_states = self.mlp(hidden_states)
        if _is_npu and get_cmo_stream():
            wait_cmo_stream()
        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )

        # HACK(Shizhuo): [DEBUG DUMP: OUTPUT]
        if dump_name and not dump_qk_only and forward_batch.forward_mode.is_extend() and layer_id <= 2:
            dump_tensor_to_disk("layer_output_hidden", hidden_states, self.layer_id, dump_name)
            if residual is not None:
                dump_tensor_to_disk("layer_output_residual", residual, self.layer_id, dump_name)

        return hidden_states, residual


class Qwen3Model(Qwen2Model):
    def __init__(
        self,
        config: Qwen3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        alt_stream = torch.cuda.Stream() if _is_cuda else None
        super().__init__(
            config=config,
            quant_config=quant_config,
            prefix=prefix,
            decoder_layer_type=Qwen3DecoderLayer,
            alt_stream=alt_stream,
        )


class Qwen3ForCausalLM(nn.Module):
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
        config: Qwen3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        self.model = Qwen3Model(
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

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

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
            if "Embedding" in self.config.name_or_path:
                name = add_prefix(name, "model")
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


EntryClass = Qwen3ForCausalLM
