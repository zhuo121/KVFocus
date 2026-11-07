from typing import Iterable, List, Optional, Tuple, Union

import torch
import time  # HACK(Shizhuo): CacheFlex: frame-reuse profiling

# Adapted from https://raw.githubusercontent.com/vllm-project/vllm/7f62077af5159c625fe3ad1c812e6c1a2b93ba3b/vllm/model_executor/models/internlm2.py
# Adapted from https://raw.githubusercontent.com/hehesangsj/sglang/refs/heads/internvl/python/sglang/srt/models/internvl.py
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling

# HACK(Shizhuo) - 修复 Vision Pruning: 导入模块而非直接导入 embedding_cache 变量，避免 Python 模块导入顺序问题
from sglang.srt.managers import mm_utils
from sglang.srt.mem_cache.multimodal_cache import MultimodalCache

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.activation import get_act_fn
from sglang.srt.layers.attention import vision_utils
from sglang.srt.layers.attention.vision import SingletonCache, VisionAttention
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternTokenPairs,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.deepseek_janus_pro import DropPath
from sglang.srt.models.gpt_oss import GptOssForCausalLM
from sglang.srt.models.internlm2 import InternLM2ForCausalLM
from sglang.srt.models.qwen2 import Qwen2ForCausalLM
from sglang.srt.models.qwen3 import Qwen3ForCausalLM
from sglang.srt.models.qwen3_moe import Qwen3MoeForCausalLM
from sglang.srt.multimodal.mm_utils import run_dp_sharded_vision_model
from sglang.srt.server_args import get_global_server_args
from sglang.utils import logger


class InternAttention(nn.Module):
    def __init__(
        self,
        config,
        quant_config: QuantizationConfig = None,
        use_data_parallel: bool = False,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5

        self.attn = VisionAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            projection_size=self.embed_dim,
            use_qkv_parallel=True,
            quant_config=quant_config,
            dropout=getattr(config, "dropout", 0.0),
            qkv_bias=getattr(config, "qkv_bias", False)
            or getattr(config, "attention_bias", False),
            num_dummy_heads=getattr(config, "num_dummy_heads", 0),
            qk_normalization=getattr(config, "qk_normalization", False)
            or getattr(config, "use_qk_norm", False),
            flatten_batch=False,
            use_data_parallel=use_data_parallel,
        )

        self.proj_drop = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        out = self.attn(hidden_states, cu_seqlens=cu_seqlens)
        outs = self.proj_drop(out)
        return outs


class InternVisionEmbeddings(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = (
            config.image_size
            if isinstance(config.image_size, int)
            else config.image_size[0]
        )
        self.patch_size = (
            config.patch_size
            if isinstance(config.patch_size, int)
            else config.patch_size[0]
        )

        self.class_embedding = nn.Parameter(
            torch.randn(1, 1, self.embed_dim),
        )

        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_positions, self.embed_dim)
        )

    def _get_pos_embed(self, pos_embed, H, W):
        target_dtype = pos_embed.dtype
        pos_embed = (
            pos_embed.float()
            .reshape(
                1,
                self.image_size // self.patch_size,
                self.image_size // self.patch_size,
                -1,
            )
            .permute(0, 3, 1, 2)
        )
        pos_embed = (
            F.interpolate(pos_embed, size=(H, W), mode="bicubic", align_corners=False)
            .reshape(1, -1, H * W)
            .permute(0, 2, 1)
            .to(target_dtype)
        )
        return pos_embed

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(
            pixel_values
        )  # shape = [*, channel, width, height]
        batch_size, _, height, width = patch_embeds.shape
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        position_embedding = torch.cat(
            [
                self.position_embedding[:, :1, :],
                self._get_pos_embed(self.position_embedding[:, 1:, :], height, width),
            ],
            dim=1,
        )
        embeddings = embeddings + position_embedding.to(target_dtype)
        return embeddings


class InternRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class InternMLP(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        use_data_parallel: bool = False,
    ):
        super().__init__()
        self.tp_size = (
            1 if use_data_parallel else get_tensor_model_parallel_world_size()
        )
        self.tp_rank = 0 if use_data_parallel else get_tensor_model_parallel_rank()
        self.config = config
        self.act = get_act_fn(config.hidden_act)
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            quant_config=None,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
        )
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            quant_config=None,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


NORM2FN = {
    "rms_norm": InternRMSNorm,
    "layer_norm": nn.LayerNorm,
}


class InternVisionEncoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        drop_path_rate: float,
        quant_config: QuantizationConfig = None,
        use_data_parallel: bool = False,
    ):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.norm_type = config.norm_type
        self.attn = InternAttention(
            config=config,
            quant_config=quant_config,
            use_data_parallel=use_data_parallel,
        )
        self.mlp = InternMLP(config, use_data_parallel)
        self.norm1 = NORM2FN[self.norm_type](self.embed_dim, eps=config.layer_norm_eps)
        self.norm2 = NORM2FN[self.norm_type](self.embed_dim, eps=config.layer_norm_eps)

        self.ls1 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))
        self.ls2 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))
        self.drop_path1 = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )
        self.drop_path2 = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> Tuple[
        torch.FloatTensor,
        Optional[torch.FloatTensor],
        Optional[Tuple[torch.FloatTensor]],
    ]:
        """
        Args:
            hidden_states (`Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]`): input to the layer of shape `(batch, seq_len, embed_dim)`
        """

        hidden_states = hidden_states + self.drop_path1(
            self.attn(
                self.norm1(hidden_states).to(hidden_states.dtype), cu_seqlens=cu_seqlens
            )
            * self.ls1
        )

        hidden_states = hidden_states + self.drop_path2(
            self.mlp(self.norm2(hidden_states).to(hidden_states.dtype)) * self.ls2
        )

        return hidden_states


class InternVisionEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`InternEncoderLayer`].

    Args:
        config (`InternConfig`):
            The corresponding vision configuration for the `InternEncoder`.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        use_data_parallel: bool = False,
    ):
        super().__init__()
        self.config = config
        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)
        ]
        self.layers = nn.ModuleList(
            [
                InternVisionEncoderLayer(
                    config, dpr[idx], quant_config, use_data_parallel
                )
                for idx in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        inputs_embeds,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Embedded representation of the inputs. Should be float, not int tokens.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        encoder_states = () if output_hidden_states else None
        hidden_states = inputs_embeds

        cu_seqlens = SingletonCache()

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            layer_outputs = encoder_layer(hidden_states, cu_seqlens=cu_seqlens)
            hidden_states = layer_outputs

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states
        )


class InternVisionModel(PreTrainedModel):
    main_input_name = "pixel_values"
    _supports_flash_attn_2 = True
    config_class = PretrainedConfig
    _no_split_modules = ["InternVisionEncoderLayer"]

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        use_data_parallel: bool = False,
    ):
        super().__init__(config)

        self.config = config
        self.use_data_parallel = use_data_parallel
        self.embeddings = InternVisionEmbeddings(
            config,
        )
        self.encoder = InternVisionEncoder(config, quant_config, use_data_parallel)

    def resize_pos_embeddings(self, old_size, new_size, patch_size):
        pos_emb = self.embeddings.position_embedding
        _, num_positions, embed_dim = pos_emb.shape
        cls_emb = pos_emb[:, :1, :]
        pos_emb = (
            pos_emb[:, 1:, :]
            .reshape(1, old_size // patch_size, old_size // patch_size, -1)
            .permute(0, 3, 1, 2)
        )
        pos_emb = F.interpolate(
            pos_emb.float(),
            size=new_size // patch_size,
            mode="bicubic",
            align_corners=False,
        )
        pos_emb = pos_emb.to(cls_emb.dtype).reshape(1, embed_dim, -1).permute(0, 2, 1)
        pos_emb = torch.cat([cls_emb, pos_emb], dim=1)
        self.embeddings.position_embedding = nn.Parameter(pos_emb)
        self.embeddings.image_size = new_size
        logger.info(
            "Resized position embeddings from {} to {}".format(old_size, new_size)
        )

    def get_input_embeddings(self):
        return self.embeddings

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_embeds: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if pixel_values is None and pixel_embeds is None:
            raise ValueError("You have to specify pixel_values or pixel_embeds")

        if pixel_embeds is not None:
            hidden_states = pixel_embeds
        else:
            if len(pixel_values.shape) == 4:
                hidden_states = self.embeddings(pixel_values)
            else:
                raise ValueError(f"wrong pixel_values size: {pixel_values.shape}")

        if self.use_data_parallel:
            encoder_outputs = run_dp_sharded_vision_model(hidden_states, self.encoder)
            last_hidden_state = encoder_outputs
        else:
            encoder_outputs = self.encoder(
                inputs_embeds=hidden_states,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            last_hidden_state = encoder_outputs.last_hidden_state
        pooled_output = last_hidden_state[:, 0, :]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        if self.use_data_parallel:
            return BaseModelOutputWithPooling(
                last_hidden_state=last_hidden_state,
                pooler_output=pooled_output,
                hidden_states=None,
                attentions=None,
            )
        else:
            return BaseModelOutputWithPooling(
                last_hidden_state=last_hidden_state,
                pooler_output=pooled_output,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
            )


class InternVLChatModel(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        use_flash_attn=True,
    ) -> None:
        super().__init__()
        self.config = config
        self.use_data_parallel = get_global_server_args().mm_enable_dp_encoder
        self.quant_config = quant_config
        vision_utils.update_vit_attn_dummy_heads_config(self.config)
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int(
            (image_size // patch_size) ** 2 * (config.downsample_ratio**2)
        )
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version

        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.llm_config._attn_implementation = (
            "flash_attention_2" if use_flash_attn else "eager"
        )

        logger.info(f"num_image_token: {self.num_image_token}")
        logger.info(f"ps_version: {self.ps_version}")

        self.vision_model = InternVisionModel(
            config.vision_config,
            use_data_parallel=self.use_data_parallel,
        )
        if config.llm_config.architectures[0] == "Qwen2ForCausalLM":
            self.language_model = Qwen2ForCausalLM(
                config=config.llm_config, quant_config=quant_config
            )
        elif config.llm_config.architectures[0] == "InternLM2ForCausalLM":
            self.language_model = InternLM2ForCausalLM(
                config=config.llm_config, quant_config=quant_config
            )
        elif config.llm_config.architectures[0] == "Qwen3MoeForCausalLM":
            self.language_model = Qwen3MoeForCausalLM(
                config=config.llm_config, quant_config=quant_config
            )
        elif config.llm_config.architectures[0] == "GptOssForCausalLM":
            self.language_model = GptOssForCausalLM(
                config=config.llm_config, quant_config=quant_config
            )
        elif config.llm_config.architectures[0] == "Qwen3ForCausalLM":
            self.language_model = Qwen3ForCausalLM(
                config=config.llm_config, quant_config=quant_config
            )
        else:
            raise NotImplementedError(
                f"{config.llm_config.architectures[0]} is not implemented."
            )
        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(
                vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size
            ),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

        self.external_mm_data_embedding_funcs = {
            Modality.IMAGE: self.get_image_feature,
            # HACK(Shizhuo)
            Modality.VIDEO: self.get_image_feature,
            Modality.MULTI_IMAGES: self.get_image_feature
        }

        self.model = self.language_model.model

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(
            n,
            int(h * scale_factor),
            int(w * scale_factor),
            int(c / (scale_factor * scale_factor)),
        )
        if self.ps_version == "v1":
            logger.warn(
                "In ps_version 'v1', the height and width have not been swapped back, "
                "which results in a transposed image."
            )
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=False, return_dict=True
            ).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=True, return_dict=True
            ).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    # def get_image_feature(self, items: List[MultimodalDataItem]):
    #     """
    #     Projects the last hidden state from the vision model into language model space.

    #     Returns:
    #         image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
    #     """
    #     pixel_values = torch.cat([item.feature for item in items])
    #     image_features = self.extract_feature(pixel_values)
    #     return image_features

    # HACK(Shizhuo): Modified get_image_feature for CacheFlex Pruning
    def get_image_feature(self, items: List[MultimodalDataItem]):
        import os
        forward_batch = getattr(self, "_current_cacheflex_batch", None)
        server_args = get_global_server_args()
        mpc = getattr(server_args, "mm_process_config", None) or {}
        cacheflex_mode = mpc.get("cacheflex_mode", "")
        enable_cacheflex = bool(server_args and getattr(server_args, "enable_cacheflex", False) and cacheflex_mode != "fr")

        # HACK(Shizhuo): hash 规范化函数，统一处理 int/str/0x 前缀
        def norm_hash(h):
            if h is None:
                return None
            if isinstance(h, int):
                return format(h, "x")
            h = str(h).lower()
            return h[2:] if h.startswith("0x") else h

        enable_profiling = os.environ.get("SGLANG_ENABLE_PROFILING", "false").lower() == "true"
        do_profile = enable_profiling and (forward_batch is not None) and forward_batch.forward_mode.is_extend()

        if do_profile:
            torch.cuda.synchronize()
            t_vis_start = time.perf_counter()

        try:
            # HACK(Shizhuo) 判断是否需要 pruning（extend 且 cacheFlex 开启存在 video_hit）
            cacheflex_cfgs = getattr(forward_batch, "cacheflex_configs", None) if forward_batch else None
            enable_pruning = False

            if enable_cacheflex and forward_batch is not None and forward_batch.forward_mode.is_extend() and cacheflex_cfgs:
                for cfg in cacheflex_cfgs:
                    if cfg and "sources" in cfg:
                        for src in cfg["sources"]:
                            if src.get("type") == "video_hit" and "hash" in src:
                                enable_pruning = True
                                break
                    if enable_pruning:
                        break

            # 如果不需要 pruning，走原生 batch 分支
            if not enable_pruning:
                # HACK(Shizhuo): 修复 InternVL: baseline 分支 warm per-frame cache
                device = next(self.parameters()).device
                pixel_values = torch.cat([item.feature for item in items], dim=0)
                pixel_values = pixel_values.to(device=device, dtype=torch.bfloat16, non_blocking=True)
                feats = self.extract_feature(pixel_values)  # [sumN, dummy_len, hidden]

                # HACK(Shizhuo): 修复 InternVL: baseline 分支写入 per-frame cache
                if mm_utils.embedding_cache is not None:
                    pt = 0
                    for item_idx, item in enumerate(items):
                        pv_cpu = item.feature
                        n = pv_cpu.shape[0] if (pv_cpu is not None and pv_cpu.ndim == 4) else 0
                        hashes = item.model_specific_data.get("frame_hashes", []) or []
                        m = min(n, len(hashes))

                        for j in range(m):
                            h = norm_hash(hashes[j])
                            if not h:
                                continue
                            ch = MultimodalCache.combine_hashes([int(h, 16)])
                            mm_utils.embedding_cache.set(ch, feats[pt + j : pt + j + 1].clone())
                        pt += n

                return feats

            # HACK(Shizhuo) pruning + embedding_cache 回填
            device = next(self.parameters()).device
            hidden_size = self.config.llm_config.hidden_size
            dummy_len = self.num_image_token

            try:
                # 尝试跟随模型参数类型
                default_dtype = next(self.parameters()).dtype
            except Exception:
                # 兜底使用 bfloat16
                default_dtype = torch.bfloat16

            # HACK(Shizhuo) 遍历每个 item，per-item 构建 skip_hashes 并处理 cache hit
            final_features = []

            for item_idx, item in enumerate(items):
                target_dtype = default_dtype
                # item.feature: CPU float16 (N,3,448,448)；hashes 是 per-frame
                item_hashes = item.model_specific_data.get("frame_hashes", []) or []
                pv_cpu = item.feature

                # 只支持 1 tile per frame（video max_num=1）时的 pruning；否则退回原生 batch
                if len(item_hashes) != pv_cpu.shape[0] or pv_cpu.ndim != 4:
                    pv = pv_cpu.to(device=device, dtype=torch.bfloat16, non_blocking=True)
                    final_features.append(self.extract_feature(pv))
                    continue

                # HACK(Shizhuo): per-item 构建 skip_hashes，避免跨样本 hash 污染
                skip_hashes = set()
                if cacheflex_cfgs and item_idx < len(cacheflex_cfgs):
                    cfg = cacheflex_cfgs[item_idx]
                    if cfg and "sources" in cfg:
                        for src in cfg["sources"]:
                            if src.get("type") == "video_hit" and "hash" in src:
                                skip_hashes.add(norm_hash(src["hash"]))

                # HACK(Shizhuo): 调试计数器
                total_frames = len(item_hashes)
                marked_hit_frames = sum(1 for h in item_hashes if h and norm_hash(h) in skip_hashes)
                cache_hit_frames = 0
                to_compute_frames = 0

                # HACK(Shizhuo): 修复 InternVL: batch 处理 miss 帧
                # 收集所有需要计算的帧（cache miss 或非 hit 帧），统一 batch 处理
                num_frames = len(item_hashes)
                frame_features = [None] * num_frames  # 预分配结果列表
                miss_pvs = []
                miss_indices = []
                miss_hashes = []

                for frame_idx, (frame_pv_cpu, frame_hash) in enumerate(zip(pv_cpu, item_hashes)):
                    _norm_hash = norm_hash(frame_hash)

                    if frame_hash and _norm_hash in skip_hashes:
                        # Hit 帧：尝试从 embedding_cache 读取
                        hash_int = int(_norm_hash, 16)
                        combined_hash = MultimodalCache.combine_hashes([hash_int])

                        cached_feature = None
                        if mm_utils.embedding_cache is not None:
                            cached_feature = mm_utils.embedding_cache.get([], combined_hash=combined_hash)

                        if cached_feature is not None:
                            # Shape 规整: 确保是 [1, dummy_len, hidden_size]
                            if cached_feature.ndim == 3:
                                assert cached_feature.shape[0] == 1, f"unexpected cached_feature shape: {cached_feature.shape}"
                            elif cached_feature.ndim == 2:
                                cached_feature = cached_feature.unsqueeze(0)
                            assert cached_feature.ndim == 3, f"cached_feature must be [1,T,H], got {cached_feature.shape}"

                            # 设备标准化
                            cached_feature = cached_feature.to(device=device, dtype=default_dtype)
                            # HACK(Shizhuo) - 长度不匹配时当作 miss 重新算并覆盖 cache，而不是用 zeros
                            if cached_feature.shape[1] != dummy_len:
                                # 长度不匹配，当作 miss 处理
                                miss_pvs.append(frame_pv_cpu)
                                miss_indices.append(frame_idx)
                                miss_hashes.append(combined_hash)
                            else:
                                cache_hit_frames += 1
                                frame_features[frame_idx] = cached_feature
                        else:
                            # Cache miss：加入 batch
                            to_compute_frames += 1
                            miss_pvs.append(frame_pv_cpu)
                            miss_indices.append(frame_idx)
                            miss_hashes.append(combined_hash)
                    else:
                        # 非 hit 帧：加入 batch
                        to_compute_frames += 1
                        miss_pvs.append(frame_pv_cpu)
                        miss_indices.append(frame_idx)
                        # 非 hit 帧也生成 hash 以便缓存
                        if frame_hash:
                            hash_int = int(_norm_hash, 16)
                            miss_hashes.append(MultimodalCache.combine_hashes([hash_int]))
                        else:
                            miss_hashes.append(None)
                            
                # 统一 batch 处理所有 miss 帧
                if miss_pvs:
                    pv_batch = torch.stack(miss_pvs, dim=0).to(device=device, dtype=torch.bfloat16, non_blocking=True)  # [M,3,H,W]
                    feats_batch = self.extract_feature(pv_batch)  # [M, dummy_len, hidden]

                    for j, frame_idx in enumerate(miss_indices):
                        feat = feats_batch[j:j+1].to(dtype=default_dtype)  # [1, dummy_len, hidden]
                        frame_features[frame_idx] = feat
                        # 写入缓存
                        if mm_utils.embedding_cache is not None and miss_hashes[j] is not None:
                            mm_utils.embedding_cache.set(miss_hashes[j], feat.clone())

                # 填充 None（理论上不应该发生，但作为保险）
                for frame_idx in range(num_frames):
                    if frame_features[frame_idx] is None:
                        frame_features[frame_idx] = torch.zeros(
                            (1, dummy_len, hidden_size), dtype=default_dtype, device=device
                        )

                final_features.append(torch.cat(frame_features, dim=0))

            return torch.cat(final_features, dim=0)

        finally:
            if do_profile:
                torch.cuda.synchronize()
                t_vis_end = time.perf_counter()
                if forward_batch is not None:
                    prev = getattr(forward_batch, "temp_vision_latency", 0.0)
                    forward_batch.temp_vision_latency = prev + (t_vis_end - t_vis_start)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:

        # HACK(Shizhuo): Pass context & Init Temp Var
        self._current_cacheflex_batch = forward_batch
        forward_batch.temp_vision_latency = 0.0
        # ==========================================

        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            multimodal_model=self,
            data_embedding_funcs=self.external_mm_data_embedding_funcs,
            positions=positions,
        )

        # HACK(Shizhuo): Cleanup
        self._current_cacheflex_batch = None
        # ==========================================

        return hidden_states

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        # Get all special token IDs
        im_start_id: int = mm_inputs.im_start_id
        im_end_id: int = mm_inputs.im_end_id

        media_token_pairs = [(im_start_id, im_end_id)]
        helper = MultiModalityDataPaddingPatternTokenPairs(media_token_pairs)

        return helper.pad_input_tokens(input_ids, mm_inputs)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        expert_params_mapping = []
        if "InternLM2ForCausalLM" in self.config.llm_config.architectures:
            stacked_params_mapping = [
                # (param_name, shard_name, shard_id)
                ("gate_up_proj", "w1", 0),
                ("gate_up_proj", "w3", 1),
            ]
        elif "Qwen2ForCausalLM" in self.config.llm_config.architectures:
            stacked_params_mapping = [
                # (param_name, shard_name, shard_id)
                ("qkv_proj", "q_proj", "q"),
                ("qkv_proj", "k_proj", "k"),
                ("qkv_proj", "v_proj", "v"),
                ("gate_up_proj", "gate_proj", 0),
                ("gate_up_proj", "up_proj", 1),
            ]
        elif "Qwen3MoeForCausalLM" in self.config.llm_config.architectures:
            stacked_params_mapping = [
                # (param_name, shard_name, shard_id)
                ("qkv_proj", "q_proj", "q"),
                ("qkv_proj", "k_proj", "k"),
                ("qkv_proj", "v_proj", "v"),
                ("gate_up_proj", "gate_proj", 0),
                ("gate_up_proj", "up_proj", 1),
            ]

            expert_params_mapping = FusedMoE.make_expert_params_mapping(
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=self.config.num_experts,
            )
        elif "Qwen3ForCausalLM" in self.config.llm_config.architectures:
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
            if "rotary_emb.inv_freq" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:
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
                if "vision_model" in name:
                    # adapt to VisionAttention
                    name = name.replace(r"attn.", r"attn.attn.")
                    name = name.replace(r"qkv.", r"qkv_proj.")

                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    param = params_dict[name]
                    if "wqkv" in name:
                        config = self.config
                        kv_groups = (
                            config.num_attention_heads // config.num_key_value_heads
                        )
                        head_dim = config.hidden_size // config.num_attention_heads
                        loaded_weight = loaded_weight.view(
                            -1, 2 + kv_groups, head_dim, loaded_weight.shape[-1]
                        )
                        wq, wk, wv = torch.split(
                            loaded_weight, [kv_groups, 1, 1], dim=1
                        )
                        wq = wq.reshape(-1, wq.shape[-1])
                        wk = wk.reshape(-1, wk.shape[-1])
                        wv = wv.reshape(-1, wv.shape[-1])
                        weight_loader = param.weight_loader
                        weight_loader(param, wq, "q")
                        weight_loader(param, wk, "k")
                        weight_loader(param, wv, "v")
                    else:
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        if "vision_model" in name:
                            loaded_weight = vision_utils.pad_vit_attn_dummy_heads(
                                self.config, name, loaded_weight
                            )
                        weight_loader(param, loaded_weight)


EntryClass = InternVLChatModel
