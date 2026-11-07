# HACK(Shizhuo) New
"""
CacheFlex unified dump + profile control system.

All .pt dumps + profiling are controlled via SGLANG_CACHEFLEX_* env vars.
This module replaces the historically scattered control points (~25 env vars
across different prefixes) with one consistent schema.

The Category list and env vars below are the full reference.

# Quick reference

Master switch (required for ANY .pt dump):
  SGLANG_CACHEFLEX_DUMP_DIR=<path>          # set this to activate dump framework

Per-category dump switches (default off, opt-in):
  SGLANG_CACHEFLEX_DUMP_LOGITS=1            # sampler first-step logits + logprobs
  SGLANG_CACHEFLEX_DUMP_LAYER_KV=1          # triton_backend per-layer K/V/meta
  SGLANG_CACHEFLEX_DUMP_LAYER_QO=1          # triton_backend per-layer qo_indptr
  SGLANG_CACHEFLEX_DUMP_LAYER_TENSORS=1     # models/* dump_tensor_to_disk intermediate tensors
  SGLANG_CACHEFLEX_DUMP_KVFOCUS_SCORES=1    # radix_attention per-layer KVFocus scores + mask
  SGLANG_CACHEFLEX_DUMP_KVFOCUS_EXTENDED=1  # per-token KVFocus + outdiff (requires SCORES)
  SGLANG_CACHEFLEX_DUMP_HARM=1              # kvfocus_harm_observe 2-pass attention harm
  SGLANG_CACHEFLEX_DUMP_HARM_MODE=both      # both | single_ablation | full_reuse
  SGLANG_CACHEFLEX_DUMP_HARM_PROBE_LEN=5
  SGLANG_CACHEFLEX_DUMP_ALL=1               # convenience: enable all categories

Session name (per-sample dedup):
  SGLANG_CACHEFLEX_DUMP_SESSION=<name>      # fallback if no debug_dump_name
  custom_params["debug_dump_name"]=<name>   # per-sample (preferred; inference sets it via --dump)

Profiling (lightweight, written to JSON profiling_metadata):
  SGLANG_CACHEFLEX_PROFILE_TIMING=1         # CUDA event timing -> profiling_metadata
  SGLANG_CACHEFLEX_PROFILE_MEM=1            # GPU memory snapshots -> profiling_metadata
  SGLANG_CACHEFLEX_PROFILE_KERNEL=1         # triton kernel launcher stdout prints (debug only)

# Output layout

  $SGLANG_CACHEFLEX_DUMP_DIR/<session>/<category_lowercase>/[...]
  e.g. ./cacheflex_dumps/naive_sys_prefix_gpu0_2wikimqa_sample42/logits/logits.pt

# Deprecated env vars (auto-mapped to new names at module load + DeprecationWarning)

  SGLANG_KVFOCUS_DUMP            -> SGLANG_CACHEFLEX_DUMP_KVFOCUS_SCORES (and DUMP_SESSION)
  SGLANG_KVFOCUS_DUMP_EXTENDED   -> SGLANG_CACHEFLEX_DUMP_KVFOCUS_EXTENDED
  SGLANG_KVFOCUS_HARM_OBSERVE    -> SGLANG_CACHEFLEX_DUMP_HARM
  SGLANG_KVFOCUS_HARM_MODE       -> SGLANG_CACHEFLEX_DUMP_HARM_MODE
  SGLANG_KVFOCUS_HARM_PROBE_LEN  -> SGLANG_CACHEFLEX_DUMP_HARM_PROBE_LEN
  KVFOCUS_PROFILE                -> SGLANG_CACHEFLEX_PROFILE_TIMING
  CACHEFLEX_MEM_PROFILE          -> SGLANG_CACHEFLEX_PROFILE_MEM
  SGLANG_KVFOCUS_PROF            -> SGLANG_CACHEFLEX_PROFILE_KERNEL
"""

import os
import warnings
from pathlib import Path
from typing import Optional


# ============================================================================
# Category constants
# ============================================================================

class Category:
    """Dump category constants. Use these instead of raw strings."""
    LOGITS = "LOGITS"
    LAYER_KV = "LAYER_KV"
    LAYER_QO = "LAYER_QO"
    LAYER_TENSORS = "LAYER_TENSORS"
    KVFOCUS_SCORES = "KVFOCUS_SCORES"
    KVFOCUS_EXTENDED = "KVFOCUS_EXTENDED"
    HARM = "HARM"

_ALL_CATEGORIES = [
    Category.LOGITS,
    Category.LAYER_KV,
    Category.LAYER_QO,
    Category.LAYER_TENSORS,
    Category.KVFOCUS_SCORES,
    Category.KVFOCUS_EXTENDED,
    Category.HARM,
]


# ============================================================================
# Deprecation auto-map (runs once at module import)
# ============================================================================

# old_env -> (new_env, extra_action_hint)
_DEPRECATION_MAP = {
    "SGLANG_KVFOCUS_DUMP":           "SGLANG_CACHEFLEX_DUMP_KVFOCUS_SCORES",  # value also -> SESSION
    "SGLANG_KVFOCUS_DUMP_EXTENDED":  "SGLANG_CACHEFLEX_DUMP_KVFOCUS_EXTENDED",
    "SGLANG_KVFOCUS_HARM_OBSERVE":   "SGLANG_CACHEFLEX_DUMP_HARM",
    "SGLANG_KVFOCUS_HARM_MODE":      "SGLANG_CACHEFLEX_DUMP_HARM_MODE",
    "SGLANG_KVFOCUS_HARM_PROBE_LEN": "SGLANG_CACHEFLEX_DUMP_HARM_PROBE_LEN",
    "KVFOCUS_PROFILE":               "SGLANG_CACHEFLEX_PROFILE_TIMING",
    "CACHEFLEX_MEM_PROFILE":         "SGLANG_CACHEFLEX_PROFILE_MEM",
    "SGLANG_KVFOCUS_PROF":           "SGLANG_CACHEFLEX_PROFILE_KERNEL",
}

_AUTOMAP_DONE = False


def _apply_deprecation_automap() -> None:
    """Scan deprecated env vars; auto-map to new names + emit DeprecationWarning once.

    Called automatically at module import. Safe to re-call (idempotent).
    """
    global _AUTOMAP_DONE
    if _AUTOMAP_DONE:
        return
    _AUTOMAP_DONE = True

    warned = []
    for old_name, new_name in _DEPRECATION_MAP.items():
        old_val = os.environ.get(old_name)
        if old_val is None or old_val == "":
            continue

        # Don't clobber if user already set new name
        if os.environ.get(new_name) is None:
            os.environ[new_name] = old_val

        warned.append(f"  {old_name}={old_val!r} -> {new_name}")

        # Special: SGLANG_KVFOCUS_DUMP sets BOTH the category flag AND session name
        # (historically it was a string session name; "set" meant "enable + use as session")
        if old_name == "SGLANG_KVFOCUS_DUMP":
            # Ensure the category enable flag is "1" not the session string
            os.environ["SGLANG_CACHEFLEX_DUMP_KVFOCUS_SCORES"] = "1"
            if os.environ.get("SGLANG_CACHEFLEX_DUMP_SESSION") is None:
                os.environ["SGLANG_CACHEFLEX_DUMP_SESSION"] = old_val
            # Auto-enable master DUMP_DIR if not set (historical behavior assumed cwd/kvfocus_dump/)
            if os.environ.get("SGLANG_CACHEFLEX_DUMP_DIR") is None:
                default_dir = os.path.abspath("./cacheflex_dumps")
                os.environ["SGLANG_CACHEFLEX_DUMP_DIR"] = default_dir
                warned.append(f"  (auto-enabled SGLANG_CACHEFLEX_DUMP_DIR={default_dir})")

        # SGLANG_KVFOCUS_HARM_OBSERVE was binary "1"/"0", same semantics as new -> OK
        # SGLANG_KVFOCUS_DUMP_EXTENDED was "1"/"0" -> OK
        # KVFOCUS_PROFILE / CACHEFLEX_MEM_PROFILE / SGLANG_KVFOCUS_PROF: all "1"/"0", OK
        # SGLANG_KVFOCUS_HARM_MODE / _HARM_PROBE_LEN: string values, OK

    if warned:
        msg = (
            "[CacheFlex] DEPRECATED env vars detected; auto-mapped to new SGLANG_CACHEFLEX_* "
            "schema. Please migrate your scripts. See the cacheflex_dump module docstring.\n"
            + "\n".join(warned)
        )
        # DeprecationWarning often filtered; use UserWarning for visibility
        warnings.warn(msg, UserWarning, stacklevel=2)


# Trigger at import
_apply_deprecation_automap()


# ============================================================================
# Dump enablement checks (hot-path; called per layer / per sample)
# ============================================================================

def is_enabled(category: str) -> bool:
    """Return True iff the master switch + this category are both enabled.

    Conditions (ALL must hold):
      1. SGLANG_CACHEFLEX_DUMP_DIR is set (master switch)
      2. SGLANG_CACHEFLEX_DUMP_ALL=1  OR  SGLANG_CACHEFLEX_DUMP_<CATEGORY>=1
    """
    if os.environ.get("SGLANG_CACHEFLEX_DUMP_DIR") is None:
        return False
    if category not in _ALL_CATEGORIES:
        raise ValueError(
            f"Unknown CacheFlex dump category {category!r}; valid: {_ALL_CATEGORIES}"
        )
    if os.environ.get("SGLANG_CACHEFLEX_DUMP_ALL", "0") == "1":
        return True
    return os.environ.get(f"SGLANG_CACHEFLEX_DUMP_{category}", "0") == "1"


# [Oracle-Dump] ===============================================
# 层级过滤：layer-attention oracle 实验只需 {shallow, mid, deep} 三层的
# LAYER_KV / LAYER_TENSORS dump，全层 dump 是 GB 级。设 SGLANG_CACHEFLEX_DUMP_LAYERS
# = "1,14,21" 后，per-layer dump 站点先过 layer_allowed(layer_id) 再落盘。
# 不设（默认）= 放行所有层，对既有 ProphetKV / KF-adaptive dump 行为完全透明。
_LAYER_FILTER_CACHE = None  # (raw_str, frozenset|None)


def layer_allowed(layer_id: int) -> bool:
    """True iff this layer_id passes SGLANG_CACHEFLEX_DUMP_LAYERS filter.

    SGLANG_CACHEFLEX_DUMP_LAYERS unset/empty -> all layers allowed (no filter).
    Comma-separated 0-indexed layer ids otherwise; only listed layers dump.
    Cheap hot-path: caches parsed set keyed by the raw env string.
    """
    global _LAYER_FILTER_CACHE
    raw = os.environ.get("SGLANG_CACHEFLEX_DUMP_LAYERS", "")
    if raw == "":
        return True
    if _LAYER_FILTER_CACHE is None or _LAYER_FILTER_CACHE[0] != raw:
        try:
            allowed = frozenset(int(x) for x in raw.split(",") if x.strip() != "")
        except ValueError:
            allowed = None  # malformed -> treat as no filter (fail open, with warn)
            warnings.warn(
                f"[CacheFlex] malformed SGLANG_CACHEFLEX_DUMP_LAYERS={raw!r}; "
                "ignoring filter (dumping all layers).",
                UserWarning,
                stacklevel=2,
            )
        _LAYER_FILTER_CACHE = (raw, allowed)
    allowed = _LAYER_FILTER_CACHE[1]
    if allowed is None:
        return True
    return layer_id in allowed


def layer_filter_active() -> bool:
    """True iff SGLANG_CACHEFLEX_DUMP_LAYERS is set (used to gate extra qk_only
    dumps like `positions` that are only needed by the oracle-dump pipeline)."""
    return os.environ.get("SGLANG_CACHEFLEX_DUMP_LAYERS", "") != ""


# [Oracle-Dump]: 控制哪些层 dump 全量 q_full（85M/层，最大头）。
# 新分析(old-KV 层间演化)主用 K/V 全层; q_full 只在 oracle 层(deep)算 U oracle 时需要。
# SGLANG_LO_QFULL_LAYERS unset -> True (保持现行为, 所有 layer_allowed 层都 dump q_full);
# 设了逗号分隔层号 -> 只这些层 dump 全量 q_full, 其余 layer_allowed 层只 dump K/V (省空间)。
_QFULL_FILTER_CACHE = None  # (raw_str, frozenset|None)


def qfull_layer_allowed(layer_id: int) -> bool:
    """True iff this layer should dump full q_full. See SGLANG_LO_QFULL_LAYERS above."""
    global _QFULL_FILTER_CACHE
    raw = os.environ.get("SGLANG_LO_QFULL_LAYERS", "")
    if raw == "":
        return True
    if _QFULL_FILTER_CACHE is None or _QFULL_FILTER_CACHE[0] != raw:
        try:
            allowed = frozenset(int(x) for x in raw.split(",") if x.strip() != "")
        except ValueError:
            allowed = None  # malformed -> fail open
        _QFULL_FILTER_CACHE = (raw, allowed)
    allowed = _QFULL_FILTER_CACHE[1]
    if allowed is None:
        return True
    return layer_id in allowed
# ============================================================================


# ============================================================================
# Session resolution + path building
# ============================================================================

def get_session(override: Optional[str] = None, default: Optional[str] = None) -> str:
    """Resolve current session name.

    Priority: override > SGLANG_CACHEFLEX_DUMP_SESSION env > default > 'default'

    Callers typically pass `override=forward_batch.debug_dump_name` so per-sample
    session names from custom_params take priority over global env default.
    """
    if override is not None and override != "":
        return override
    env_sess = os.environ.get("SGLANG_CACHEFLEX_DUMP_SESSION")
    if env_sess is not None and env_sess != "":
        return env_sess
    return default if default is not None else "default"


def get_dump_path(
    category: str,
    session: Optional[str] = None,
    subpath: str = "",
    mkdir: bool = True,
) -> Optional[Path]:
    """Return the directory Path for (session, category), or None if disabled.

    Layout: $SGLANG_CACHEFLEX_DUMP_DIR/<session>/<category_lower>/<subpath>

    Returns None if `is_enabled(category)` is False; callers should short-circuit
    on None to avoid any dump work.

    With mkdir=True (default), creates the directory tree on demand.
    """
    if not is_enabled(category):
        return None
    root = os.environ.get("SGLANG_CACHEFLEX_DUMP_DIR")
    if root is None:
        return None  # defensive
    sess = get_session(override=session)
    cat_dir = category.lower()
    p = Path(root) / sess / cat_dir
    if subpath:
        p = p / subpath
    if mkdir:
        p.mkdir(parents=True, exist_ok=True)
    return p


# ============================================================================
# Harm sub-config (read string env values for HARM category)
# ============================================================================

def get_harm_mode() -> str:
    """Return SGLANG_CACHEFLEX_DUMP_HARM_MODE, default 'both'."""
    return os.environ.get("SGLANG_CACHEFLEX_DUMP_HARM_MODE", "both")


def get_harm_probe_len() -> int:
    """Return SGLANG_CACHEFLEX_DUMP_HARM_PROBE_LEN, default 5."""
    try:
        return int(os.environ.get("SGLANG_CACHEFLEX_DUMP_HARM_PROBE_LEN", "5"))
    except ValueError:
        return 5


# ============================================================================
# Profile enablement (lightweight checks for non-.pt profiling)
# ============================================================================

def profile_timing_enabled() -> bool:
    """CUDA event-based timing (writes to ForwardBatch.profiling_metadata).

    Replaces legacy KVFOCUS_PROFILE.
    """
    return os.environ.get("SGLANG_CACHEFLEX_PROFILE_TIMING", "0") == "1"


def profile_mem_enabled() -> bool:
    """GPU memory snapshots at forward checkpoints (writes MEM_* JSON fields).

    Replaces legacy CACHEFLEX_MEM_PROFILE.
    """
    return os.environ.get("SGLANG_CACHEFLEX_PROFILE_MEM", "0") == "1"


def profile_kernel_enabled() -> bool:
    """Triton kernel launcher stdout debug prints (no JSON output).

    Replaces legacy SGLANG_KVFOCUS_PROF.
    """
    return os.environ.get("SGLANG_CACHEFLEX_PROFILE_KERNEL", "0") == "1"
