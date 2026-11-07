# HACK(Shizhuo) New
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class CacheFlexPlan:
    method: str  # "naive", "epic", "cacheblend"
    mode: str    # "static", "dynamic"

    # Static Params
    compute_range: Tuple[int, int]
    copy_range: Optional[Tuple[int, int]] = None
    is_sliced: bool = False

    # Dynamic Params
    # check_layer 为 None 表示该方法不会触发 Post-Check 切片
    check_layer: Optional[int] = None
    recomp_ratio: float = 0.15

    # [KVFocus] Probe attention mass flag
    # When True, compute U (probe segment attention mass) at check_layer alongside M (V-diff)
    kvfocus: bool = False

    # [KVFocus] Adaptive selection parameters
    solver: str = "residual_budget"       # "residual_budget"|"threshold"|"coverage"|"topk"
    scaling: str = "linear"               # "linear"|"log"|"none" — g_N scaling mode
    beta: float = 1.0                     # D_probe weight (原 auto_alpha)
    eta: float = 0.5                      # D_peak weight
    gamma: float = 2.0                    # coverage saturation exponent
    rho: float = 0.05                     # residual harm budget
    lam: Optional[float] = None           # cost threshold λ (None = fixed ratio mode; coverage/threshold only)
    max_rate: float = 0.50                # max recompute rate hard cap (r_max)
    n_ref: int = 11                       # g_N normalization reference
    b0: int = 0                           # floor tokens per segment
    b0_mode: str = "topk"                 # "topk" = largest scores, "prefix" = first tokens

    # [KVFocus] Configuration parameters
    num_probes: int = 5                   # K_p: probe tokens per segment
    top_k_peak: int = 3                   # K for D_peak top-K selection
    target_rate: float = 0.0              # matched-budget precise target rate (0=use max_rate)
    pass_mode: str = "reuse_full"         # "reuse_full"|"warmup_full"|"3pass" — experimental modes

    # [KVFocus]
    # ============================================================
    #   kvfocus_variant: method marker; "s" (default, online probe-q@check_layer × K@check_layer)
    #                  | "prophetkv-probe" (ProphetKV Stage-I probe pass, eval approach)
    # ============================================================
    kvfocus_variant: str = "s"
    # ============================================================
