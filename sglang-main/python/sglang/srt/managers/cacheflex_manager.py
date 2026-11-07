# HACK(Shizhuo) New
from enum import Enum
from typing import Dict
from sglang.srt.managers.cacheflex_types import CacheFlexPlan

class CacheFlexMethod(Enum):
    NAIVE = "naive"
    EPIC = "epic"
    CACHEBLEND = "cacheblend"
    FULL = "full"
    LAYER_REFRESH = "layer_refresh"
    EXTERNAL = "external"  # 外部传入选择位置 (如ProphetKV)
    KVFOCUS = "kvfocus"    # CacheBlend + Probe Attention Mass (M × U)
    # HACK(Shizhuo) [ProphetKV-Probe]: 全层 reused-K query→ctx U 探针 (= ProphetKV Stage-I
    # query-only 轻量 pass)。KV 行为 = naive (复用全部 doc KV、不 compute-all → 不重算上下文),
    # 但 plan.kvfocus=True 让 radix_attention 的逐层 forward-free U hook 把 doc 当作打分对象。
    PROPHETKV_PROBE = "prophetkv_probe"

class CacheFlexManager:
    def __init__(self, method: str = "naive", config: Dict = None):
        try:
            self.method = CacheFlexMethod(method)
        except ValueError:
            self.method = CacheFlexMethod.NAIVE
        self.config = config or {}

    def get_plan(self, src_len: int, prefix_len: int) -> CacheFlexPlan:
        # 1. NAIVE
        if self.method == CacheFlexMethod.NAIVE:
            return CacheFlexPlan(
                method="naive", mode="static", is_sliced=True,
                compute_range=(0, 0), copy_range=(0, src_len)
            )
        # 2. EPIC
        elif self.method == CacheFlexMethod.EPIC:
            k = self.config.get("epic_k", 32)
            k = min(k, src_len)
            return CacheFlexPlan(
                method="epic", mode="static", is_sliced=True,
                compute_range=(0, k), copy_range=(k, src_len)
            )
        # 3. CACHEBLEND
        elif self.method == CacheFlexMethod.CACHEBLEND:
            # 初始阶段全量计算，全量拷贝填底
            return CacheFlexPlan(
                method="cacheblend", mode="dynamic", is_sliced=True,
                compute_range=(0, src_len), copy_range=(0, src_len),
                check_layer=self.config.get("check_layer", 1),
                recomp_ratio=self.config.get("recomp_ratio", 0.15)
            )
        elif self.method == CacheFlexMethod.LAYER_REFRESH:
            return CacheFlexPlan(
                method="layer_refresh", mode="dynamic", is_sliced=True,
                compute_range=(0, src_len), copy_range=(0, src_len),
                check_layer=self.config.get("check_layer", 0),
                recomp_ratio=0.0
            )
        elif self.method == CacheFlexMethod.FULL:
            return CacheFlexPlan(
                method="full", mode="static", is_sliced=False,
                compute_range=(0, src_len), copy_range=None
            )
        # 6. EXTERNAL: 外部传入 mask (如 ProphetKV)。统一为 selective-recompute 路径:
        # 与 cacheblend/kvfocus 共用 _process_selective_recompute_segment (compute-all 到
        # check_layer + 铺 KV 底座), 在 check_layer 用预注入 cacheflex_global_mask 稀疏化
        # (radix_attention 的 _skip_scorer 分支, 不碰在线 scorer)。check_layer 默认 0、负数
        # 当 0 (layer-0 KV 与上下文无关, "compute-all 到 0 再稀疏" == 从 layer-0 直接选)。
        # 支持「同一 mask 不同 check_layer」扫描。可与 naive/epic 自由混用。
        elif self.method == CacheFlexMethod.EXTERNAL:
            # [Oracle-Dump]: 统一为 compute-all 到 check_layer 再按外部 mask
            # 稀疏化。check_layer 默认 0、负数按 0 处理（layer-0 KV 与上下文无关，
            # check_layer=-1 等价于 0）。
            return CacheFlexPlan(
                method="external", mode="dynamic", is_sliced=True,
                compute_range=(0, src_len), copy_range=(0, src_len),
                check_layer=max(self.config.get("check_layer", 0), 0),
                recomp_ratio=0.0
            )
        # 7. KVFOCUS: CacheBlend + Probe Attention Mass (M × U)
        # 与 CacheBlend 结构完全一致，额外启用 kvfocus 标记触发 U 计算
        elif self.method == CacheFlexMethod.KVFOCUS:
            # Defensive: ensure solver is explicitly provided
            _solver = self.config.get("solver")
            assert _solver is not None, (
                f"method_cfg missing 'solver' field. method={self.config.get('method')}, "
                f"keys={list(self.config.keys())}. "
                f"This usually means the eval script omitted solver when building method_cfg."
            )
            return CacheFlexPlan(
                method="cacheblend", mode="dynamic", is_sliced=True,
                compute_range=(0, src_len), copy_range=(0, src_len),
                check_layer=self.config.get("check_layer", 1),
                recomp_ratio=self.config.get("recomp_ratio", 0.15),
                kvfocus=True,
                solver=_solver,
                scaling=self.config.get("scaling", "linear"),
                beta=self.config.get("beta", 1.0),
                eta=self.config.get("eta", 0.5),
                gamma=self.config.get("gamma", 2.0),
                rho=self.config.get("rho", 0.05),
                lam=self.config.get("lam", None),
                max_rate=self.config.get("max_rate", 0.50),
                n_ref=self.config.get("n_ref", 11),
                b0=self.config.get("b0", 0),
                b0_mode=self.config.get("b0_mode", "topk"),
                num_probes=self.config.get("num_probes", 5),
                top_k_peak=self.config.get("top_k_peak", 3),
                target_rate=self.config.get("target_rate", 0.0),
                pass_mode=self.config.get("pass_mode", "reuse_full"),
                kvfocus_variant=self.config.get("kvfocus_variant", "s"),
            )

        # 8. PROPHETKV_PROBE: prophetkv-probe: mark sources so the radix ProphetKV Stage-I probe pass triggers.
        # plan.method="naive" → reuse the full doc KV (compute_range=(0,0), zero recompute);
        # kvfocus=True lets the reused-K gather read the doc region for the per-layer probe.
        elif self.method == CacheFlexMethod.PROPHETKV_PROBE:
            return CacheFlexPlan(
                method="naive", mode="static", is_sliced=True,
                compute_range=(0, 0), copy_range=(0, src_len),
                kvfocus=True,
                kvfocus_variant="prophetkv-probe",
                check_layer=self.config.get("check_layer", 1),
            )

        return CacheFlexPlan(method="unknown", mode="static", is_sliced=False, compute_range=(0, src_len))