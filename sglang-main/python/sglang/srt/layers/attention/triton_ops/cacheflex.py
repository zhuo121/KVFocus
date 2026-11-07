# HACK(Shizhuo) New
import os
import time
import torch
import triton
import triton.language as tl
from typing import Optional

# unified profile control (replaces SGLANG_KVFOCUS_PROF; auto-mapped for compat)
from sglang.srt.utils import cacheflex_dump as _cf_dump
_PROF = _cf_dump.profile_kernel_enabled()

# 尝试导入 sgl_kernel 高级接口
try:
    from sgl_kernel import apply_rope_with_cos_sin_cache_inplace, FusedSetKVBufferArg
except ImportError:
    apply_rope_with_cos_sin_cache_inplace = None
    FusedSetKVBufferArg = None

# -----------------------------------------------------------------------------
# P1: Fused L2/L1+L2 diff Triton kernels (single kernel launch vs 4-5 PyTorch ops)
# -----------------------------------------------------------------------------

@triton.jit
def _fused_l2_diff_kernel(
    v_new_ptr, v_old_ptr, out_ptr,
    n_rows, n_cols,
    v_new_row_stride, v_old_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    sum_sq = 0.0
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        v_new_val = tl.load(v_new_ptr + row_idx * v_new_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
        v_old_val = tl.load(v_old_ptr + row_idx * v_old_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
        d = v_new_val - v_old_val
        sum_sq += tl.sum(d * d)
    tl.store(out_ptr + row_idx, tl.sqrt(sum_sq))


@triton.jit
def _fused_l1_l2_diff_kernel(
    v_new_ptr, v_old_ptr, out_l2_ptr, out_l1_ptr,
    n_rows, n_cols,
    v_new_row_stride, v_old_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    sum_sq = 0.0
    sum_abs = 0.0
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        v_new_val = tl.load(v_new_ptr + row_idx * v_new_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
        v_old_val = tl.load(v_old_ptr + row_idx * v_old_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
        d = v_new_val - v_old_val
        sum_sq += tl.sum(d * d)
        sum_abs += tl.sum(tl.abs(d))
    tl.store(out_l2_ptr + row_idx, tl.sqrt(sum_sq))
    tl.store(out_l1_ptr + row_idx, sum_abs)


def _fused_l2_diff(v_new_part: torch.Tensor, v_old: torch.Tensor) -> torch.Tensor:
    """Fused per-row L2 norm of (v_new - v_old) using Triton kernel."""
    n_rows = v_new_part.shape[0]
    n_cols = v_new_part.shape[1] * v_new_part.shape[2]  # H * D
    diff = torch.empty(n_rows, dtype=torch.float32, device=v_new_part.device)
    BLOCK_SIZE = 128
    _fused_l2_diff_kernel[(n_rows,)](
        v_new_part, v_old, diff,
        n_rows, n_cols,
        v_new_part.stride(0), v_old.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return diff


def _fused_l1_l2_diff(v_new_part: torch.Tensor, v_old: torch.Tensor):
    """Fused per-row L1+L2 norm of (v_new - v_old) using Triton kernel."""
    n_rows = v_new_part.shape[0]
    n_cols = v_new_part.shape[1] * v_new_part.shape[2]  # H * D
    diff_l2 = torch.empty(n_rows, dtype=torch.float32, device=v_new_part.device)
    diff_l1 = torch.empty(n_rows, dtype=torch.float32, device=v_new_part.device)
    BLOCK_SIZE = 128
    _fused_l1_l2_diff_kernel[(n_rows,)](
        v_new_part, v_old, diff_l2, diff_l1,
        n_rows, n_cols,
        v_new_part.stride(0), v_old.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return diff_l2, diff_l1

# -----------------------------------------------------------------------------
# Cacheflex partial Recompute
# -----------------------------------------------------------------------------

def compute_diff(
    v_new: torch.Tensor,       # [SegLen, H, D]
    src_block_indices: torch.Tensor, # Block Table (Token Indices if PageSize=1)
    src_offset: int,           # Token Offset
    kv_pool,                   # Memory Pool
    layer_id: int,
    contiguous_kv=None,        # [P0] Contiguous buffer for fast path
    contiguous_offset: int = 0,# [P0] Offset into contiguous buffer
    ):
    """
    Python-based Gather & Diff (no top-k selection).
    Returns: L2 diff per token [ValidLen] and valid_len.
    """
    _dt0 = time.perf_counter() if _PROF else 0

    # [P0] Fast path: read from contiguous buffer (avoid fancy indexing gather)
    if contiguous_kv is not None and layer_id in contiguous_kv:
        v_cont = contiguous_kv[layer_id][1]  # V buffer
        if len(src_block_indices) > 0:
            valid_len = min(len(src_block_indices) - src_offset, v_new.shape[0])
            if valid_len <= 0:
                return None, 0
            v_old = v_cont[contiguous_offset : contiguous_offset + valid_len]
            v_new_part = v_new[:valid_len]
            diff = _fused_l2_diff(v_new_part, v_old)
            return diff, valid_len
        return None, 0

    # Fallback: paged pool gather
    v_buffer = kv_pool.v_buffer[layer_id]

    if len(src_block_indices) > 0:
        valid_len = min(len(src_block_indices) - src_offset, v_new.shape[0])
        if valid_len <= 0:
            return None, 0
        gather_indices = src_block_indices[src_offset : src_offset + valid_len].long()

        v_old = v_buffer[gather_indices] # [ValidLen, H, D]
        v_new_part = v_new[:valid_len]

        # Diff (L2 Norm per token = ||ΔV||)
        diff = torch.sqrt(torch.sum(torch.pow(v_new_part - v_old, 2), dim=(1, 2)))

        if _PROF:
            torch.cuda.synchronize()
            _dt1 = time.perf_counter()
            print(f"[PROF] compute_diff valid_len={valid_len}: {(_dt1-_dt0)*1e3:.2f}ms", flush=True)

        return diff, valid_len

    return None, 0


def compute_diff_l1_l2(
    v_new: torch.Tensor,
    src_block_indices: torch.Tensor,
    src_offset: int,
    kv_pool,
    layer_id: int,
    contiguous_kv=None,        # [P0] Contiguous buffer for fast path
    contiguous_offset: int = 0,# [P0] Offset into contiguous buffer
    ):
    """
    Compute both L1 and L2 per-token V-drift (‖ΔV‖₁ and ‖ΔV‖₂).
    Returns: (diff_l2 [ValidLen], diff_l1 [ValidLen], valid_len)
    """
    # [P0] Fast path: read from contiguous buffer
    if contiguous_kv is not None and layer_id in contiguous_kv:
        v_cont = contiguous_kv[layer_id][1]  # V buffer
        if len(src_block_indices) > 0:
            valid_len = min(len(src_block_indices) - src_offset, v_new.shape[0])
            if valid_len <= 0:
                return None, None, 0
            v_old = v_cont[contiguous_offset : contiguous_offset + valid_len]
            v_new_part = v_new[:valid_len]
            diff_l2, diff_l1 = _fused_l1_l2_diff(v_new_part, v_old)
            return diff_l2, diff_l1, valid_len
        return None, None, 0

    # Fallback: paged pool gather
    v_buffer = kv_pool.v_buffer[layer_id]

    if len(src_block_indices) > 0:
        valid_len = min(len(src_block_indices) - src_offset, v_new.shape[0])
        if valid_len <= 0:
            return None, None, 0
        gather_indices = src_block_indices[src_offset : src_offset + valid_len].long()

        v_old = v_buffer[gather_indices]
        v_new_part = v_new[:valid_len]
        delta = v_new_part - v_old

        diff_l2 = torch.sqrt(torch.sum(torch.pow(delta, 2), dim=(1, 2)))
        diff_l1 = torch.sum(torch.abs(delta), dim=(1, 2))

        return diff_l2, diff_l1, valid_len

    return None, None, 0

# -----------------------------------------------------------------------------
# [KVFocus] Probe Segment Attention Mass
# -----------------------------------------------------------------------------

def _compute_probe_attention_scores(
    q: torch.Tensor,              # [TotalTokens, QHeads*HeadDim] - RoPE 后的 Q (flatten)
    k: torch.Tensor,              # [TotalTokens, KVHeads, HeadDim] - RoPE 后的 K (reshaped)
    tasks_by_req,                 # List[List[Dict]] - dynamic tasks per request
    start_loc: torch.Tensor,      # [BatchSize] - extend start location
    seq_lens: torch.Tensor,       # [BatchSize] - extend sequence lengths (sliced)
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    num_probes: int = 2,          # number of tail probe tokens (newline = 2 tokens for Mistral)
    suffix_starts=None,           # Optional[List[int]] sliced-coord suffix-start per req.
                                  # When provided + SGLANG_KVFOCUS_PROBE_FROM_SUFFIX=1, probes span the
                                  # full [suffix_start, req_len) instead of the last num_probes tokens.
                                  # Needed on Qwen where last 5 tokens are chat-template structural markers
                                  # (<|im_end|>\n<|im_start|>assistant\n) — bad probe Q.
    ):
    """
    [KVFocus] Shared helper: compute probe attention scores and global LSE.

    Returns:
        result dict: {req_idx: {"scores": [P, Hq, L], "lse_all": [P, Hq], "tasks": tasks}}
    """
    sm_scale = head_dim ** -0.5
    kv_group_num = num_q_heads // num_kv_heads
    result = {}

    for req_idx, tasks in enumerate(tasks_by_req):
        if not tasks:
            continue

        req_start = start_loc[req_idx].item()
        req_len = seq_lens[req_idx].item()

        # Probe range — full suffix vs tail num_probes.
        # SGLANG_KVFOCUS_PROBE_FROM_SUFFIX=1: probes = [suffix_start, req_len) (whole query
        # + chat-template marker). Fixes Qwen2 where last 5 tokens are structural markers.
        if (os.environ.get("SGLANG_KVFOCUS_PROBE_FROM_SUFFIX", "0") == "1"
            and suffix_starts is not None and req_idx < len(suffix_starts)):
            probe_start_off = max(0, min(int(suffix_starts[req_idx]), req_len))
            actual_probes = req_len - probe_start_off
        else:
            actual_probes = min(num_probes, req_len)
            probe_start_off = req_len - actual_probes
        if actual_probes <= 0:
            continue

        _t0 = time.perf_counter() if _PROF else 0

        # 1. Extract probe Q: [actual_probes, QHeads*HeadDim]
        probe_abs_start = req_start + probe_start_off
        q_probe = q[probe_abs_start : req_start + req_len]
        q_probe = q_probe.view(actual_probes, num_q_heads, head_dim)  # [P, Hq, D]

        _t1 = time.perf_counter() if _PROF else 0

        # 2. Extract all K for this request: [req_len, KVHeads, HeadDim]
        k_all = k[req_start : req_start + req_len]  # [L, Hkv, D]

        # 3. Handle GQA: expand K heads to match Q heads
        if kv_group_num > 1:
            k_all = k_all.repeat_interleave(kv_group_num, dim=1)  # [L, Hq, D]

        if _PROF:
            torch.cuda.synchronize()
        _t2 = time.perf_counter() if _PROF else 0

        # 4. Compute attention scores: [P, Hq, L]
        scores = torch.einsum('phd,lhd->phl', q_probe.float(), k_all.float()) * sm_scale

        if _PROF:
            torch.cuda.synchronize()
        _t3 = time.perf_counter() if _PROF else 0

        # 5. Causal mask
        # probe at sliced position (probe_start_off + p_idx) can attend
        # to all positions [0, probe_start_off + p_idx + 1). Was hardcoded as
        # (req_len - actual_probes + p_idx + 1) which only worked when probes = tail.
        for p_idx in range(actual_probes):
            causal_boundary = probe_start_off + p_idx + 1
            if causal_boundary < req_len:
                scores[p_idx, :, causal_boundary:] = float('-inf')

        # 6. Global LSE: [P, Hq]
        lse_all = torch.logsumexp(scores, dim=-1)  # [P, Hq]

        if _PROF:
            torch.cuda.synchronize()
            _t4 = time.perf_counter()
            print(f"[PROF] _shared req{req_idx} L={req_len}: "
                  f"slice_q={(_t1-_t0)*1e3:.2f}ms  k_slice+gqa={(_t2-_t1)*1e3:.2f}ms  "
                  f"einsum={(_t3-_t2)*1e3:.2f}ms  causal+lse={(_t4-_t3)*1e3:.2f}ms  "
                  f"TOTAL={(_t4-_t0)*1e3:.2f}ms", flush=True)

        result[req_idx] = {"scores": scores, "lse_all": lse_all, "tasks": tasks}

    return result


def compute_probe_segment_score_L1(
    q: torch.Tensor,              # [TotalTokens, QHeads*HeadDim] - RoPE 后的 Q (flatten)
    k: torch.Tensor,              # [TotalTokens, KVHeads, HeadDim] - RoPE 后的 K (reshaped)
    tasks_by_req,                 # List[List[Dict]] - dynamic tasks per request
    start_loc: torch.Tensor,      # [BatchSize] - extend start location
    seq_lens: torch.Tensor,       # [BatchSize] - extend sequence lengths (sliced)
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    num_probes: int = 2,          # number of tail probe tokens (newline = 2 tokens for Mistral)
    ) -> dict:
    """
    [KVFocus] Compute tail probe tokens' L1 attention mass to each segment.

    U_L1 = exp(LSE_C - LSE_all) = sum_{j in C} a_{u,j,h}

    Returns:
        {req_idx: segment_scores_tensor} - U_L1 for each segment
    """
    shared = _compute_probe_attention_scores(
        q, k, tasks_by_req, start_loc, seq_lens,
        num_q_heads, num_kv_heads, head_dim, num_probes,
    )
    result = {}

    for req_idx, data in shared.items():
        scores = data["scores"]
        lse_all = data["lse_all"]
        tasks = data["tasks"]

        segment_scores = []
        for task in tasks:
            seg_start = task["rel_start"]
            seg_len = task["src_len"]
            seg_scores = scores[:, :, seg_start : seg_start + seg_len]
            lse_seg = torch.logsumexp(seg_scores, dim=-1)  # [P, Hq]
            mass_l1 = torch.exp(lse_seg - lse_all)  # [P, Hq]
            segment_scores.append(mass_l1.mean())

        result[req_idx] = torch.stack(segment_scores)

    return result


def compute_probe_segment_score_L2(
    q: torch.Tensor,              # [TotalTokens, QHeads*HeadDim] - RoPE 后的 Q (flatten)
    k: torch.Tensor,              # [TotalTokens, KVHeads, HeadDim] - RoPE 后的 K (reshaped)
    tasks_by_req,                 # List[List[Dict]] - dynamic tasks per request
    start_loc: torch.Tensor,      # [BatchSize] - extend start location
    seq_lens: torch.Tensor,       # [BatchSize] - extend sequence lengths (sliced)
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    num_probes: int = 2,          # number of tail probe tokens (newline = 2 tokens for Mistral)
    ) -> dict:
    """
    [KVFocus] Compute tail probe tokens' L2 attention mass to each segment.

    U_L2 = exp(0.5 * logsumexp(2*s)_{j in C} - LSE_all) = sqrt(sum_{j in C} a_{u,j,h}^2)

    Head aggregation controlled by two env vars:
      SGLANG_KVFOCUS_U_INTRA_GROUP: "mean" (default) | "max" — aggregation within each KV group
      SGLANG_KVFOCUS_U_INTER_GROUP: "mean" (default) | "frobenius" — aggregation across KV groups

    Combinations:
      mean/mean:     original (equivalent to flat mean over all Hq)
      max/mean:      Fix 1-max — within KV group take max, across groups take mean
      mean/frobenius: Fix 1c — within KV group take mean, across groups take sqrt(sum of squares)
      max/frobenius: Fix 1c+max — within KV group take max, across groups take sqrt(sum of squares)
    """
    u_intra = os.environ.get("SGLANG_KVFOCUS_U_INTRA_GROUP", "mean")
    u_inter = os.environ.get("SGLANG_KVFOCUS_U_INTER_GROUP", "mean")
    kv_group_num = num_q_heads // num_kv_heads

    shared = _compute_probe_attention_scores(
        q, k, tasks_by_req, start_loc, seq_lens,
        num_q_heads, num_kv_heads, head_dim, num_probes,
    )
    result = {}

    for req_idx, data in shared.items():
        scores = data["scores"]
        lse_all = data["lse_all"]
        tasks = data["tasks"]

        segment_scores = []
        for task in tasks:
            seg_start = task["rel_start"]
            seg_len = task["src_len"]
            seg_scores = scores[:, :, seg_start : seg_start + seg_len]
            # L2 mass: exp(0.5 * logsumexp(2*s) - lse_all) = sqrt(sum a^2)
            lse_seg_2x = torch.logsumexp(2.0 * seg_scores, dim=-1)  # [P, Hq]
            mass_l2 = torch.exp(0.5 * lse_seg_2x - lse_all)  # [P, Hq]

            if kv_group_num > 1:
                # Group-aware aggregation: [P, Hq] → [P, num_kv_heads, kv_group_num]
                mass_grouped = mass_l2.view(-1, num_kv_heads, kv_group_num)
                # Intra-group: reduce within each KV group
                if u_intra == "max":
                    mass_grouped = mass_grouped.max(dim=-1).values  # [P, num_kv_heads]
                else:
                    mass_grouped = mass_grouped.mean(dim=-1)  # [P, num_kv_heads]
                # Inter-group: reduce across KV groups
                if u_inter == "frobenius":
                    u_c = mass_grouped.pow(2).sum(dim=-1).mean().sqrt()
                else:
                    u_c = mass_grouped.mean()
            else:
                # No GQA grouping — flat mean over all heads
                u_c = mass_l2.mean()

            segment_scores.append(u_c)

        result[req_idx] = torch.stack(segment_scores)

    return result

# -----------------------------------------------------------------------------
# [P3a] Fast U_L1: single einsum + softmax per request (no Python loops)
# Replaces compute_per_token_u_l1_only for KVFocusScorer (main path).
# -----------------------------------------------------------------------------


def _compute_u_scores_l1_fast(
    q: torch.Tensor,              # [TotalTokens, QHeads*HeadDim]
    k: torch.Tensor,              # [TotalTokens, KVHeads, HeadDim]
    tasks_by_req,
    start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    num_probes: int = 2,
    compute_seg_u_l2: bool = False,
    suffix_starts=None,           # see _compute_probe_attention_scores
) -> dict:
    """Fast U_L1: single einsum with avg Q (no per-segment Python loop, no repeat_interleave).

    Key optimizations over compute_per_token_u_l1_only:
    1. GQA: avg Q within group instead of repeat_interleave K (4x less compute for Mistral-7B)
    2. No causal mask loop (probes are suffix tokens, naturally attend to all prior)
    3. Single einsum + logsumexp + exp + mean per request (4 kernels, not 40+)
    4. No per-segment logsumexp/exp (segment U derived from per-token tensor)

    Returns same format as compute_per_token_u_l1_only for drop-in compatibility:
        {req_idx: {
            "u_l1_per_token": List[Tensor[valid_len]],
            "u_l1_seg": Tensor[num_segments],
            "u_l2_seg": Tensor[num_segments] or None,
        }}
    """
    sm_scale = head_dim ** -0.5
    kv_group_num = num_q_heads // num_kv_heads
    result = {}

    _pt_total_t0 = time.perf_counter() if _PROF else 0

    for req_idx, tasks in enumerate(tasks_by_req):
        if not tasks:
            continue

        req_start = start_loc[req_idx].item()
        req_len = seq_lens[req_idx].item()

        # Probe range — full suffix vs tail num_probes
        # (see _compute_probe_attention_scores for rationale)
        _env_on = os.environ.get("SGLANG_KVFOCUS_PROBE_FROM_SUFFIX", "0") == "1"
        if (_env_on and suffix_starts is not None and req_idx < len(suffix_starts)):
            probe_start_off = max(0, min(int(suffix_starts[req_idx]), req_len))
            actual_probes = req_len - probe_start_off
            _branch = "SUFFIX"
        else:
            actual_probes = min(num_probes, req_len)
            probe_start_off = req_len - actual_probes
            _branch = "TAIL"
        # One-time diag print (env-gated)
        if _env_on:
            print(f"[KVFocus probe-range] req_idx={req_idx} branch={_branch} "
                  f"suffix_starts={suffix_starts} req_len={req_len} "
                  f"probe_start_off={probe_start_off} actual_probes={actual_probes}",
                  flush=True)
        if actual_probes <= 0:
            continue

        # 1. Extract probe Q and reshape
        probe_abs_start = req_start + probe_start_off
        q_probe = q[probe_abs_start : req_start + req_len].view(
            actual_probes, num_q_heads, head_dim
        )  # [P, Hq, D]

        # 2. Extract all K for this request
        k_all = k[req_start : req_start + req_len]  # [L, Hkv, D]

        # 3. GQA: avg Q within group (NOT expand K) — reduces einsum compute by group_size
        if kv_group_num > 1:
            q_probe_avg = q_probe.view(
                actual_probes, num_kv_heads, kv_group_num, head_dim
            ).mean(dim=2)  # [P, Hkv, D]
            scores = torch.einsum('phd,lhd->phl', q_probe_avg.float(), k_all.float()) * sm_scale
        else:
            scores = torch.einsum('phd,lhd->phl', q_probe.float(), k_all.float()) * sm_scale
        # scores: [P, Hkv, L]

        # Causal mask. Original fast path skipped this because all probes
        # were tail (P=5) and could attend to nearly everything before. With probes spanning
        # the whole suffix (probe at query start), early probes MUST NOT attend to later
        # query tokens — otherwise softmax denominator is wrong → U values skewed.
        probe_positions = probe_start_off + torch.arange(actual_probes, device=scores.device)  # [P]
        l_idx = torch.arange(req_len, device=scores.device)
        causal_mask = l_idx[None, :] > probe_positions[:, None]  # [P, L] True = mask out
        scores = scores.masked_fill(causal_mask[:, None, :], float('-inf'))

        # 5. Softmax: single logsumexp + exp
        lse = torch.logsumexp(scores, dim=-1, keepdim=True)  # [P, Hkv, 1]
        probs = torch.exp(scores - lse)                       # [P, Hkv, L]

        # 6. Per-token U_L1: mean over probes and heads
        u_per_token = probs.mean(dim=(0, 1))  # [L]

        # 7. Segment-level aggregation
        # SGLANG_KVFOCUS_U_SOFTMAX_SCOPE controls the softmax normalization range
        # of U (NOT the L1/L2 norm — this is the softmax denominator's scope).
        #   "global" (default): u_per_token softmax-normalized over whole request → carries
        #       cross-segment recency bias (probe at query tail under-weights far docs, the
        #       "lost-in-the-middle" effect).
        #   "segment": re-softmax within each segment boundary → strips cross-segment positional
        #       decay so U measures within-segment relative importance (M handles cross-segment
        #       budget). Mirrors EMS global-local idea (arXiv 2412.08521) at segment granularity.
        # Orthogonal to _U_INTRA_GROUP / _U_INTER_GROUP (which control the GQA head dimension).
        _u_scope = os.environ.get("SGLANG_KVFOCUS_U_SOFTMAX_SCOPE", "global")
        pt_l1_list = []
        seg_l1_list = []
        seg_l2_list = [] if compute_seg_u_l2 else None

        for task in tasks:
            seg_start = task["rel_start"]
            seg_len = task["src_len"]
            src_offset = task.get("src_offset", 0)
            src_block_indices = task.get("src_block_indices")

            if src_block_indices is not None and len(src_block_indices) > 0:
                valid_len = min(len(src_block_indices) - src_offset, seg_len)
            else:
                valid_len = seg_len - src_offset

            # Per-token U_L1 for this segment
            if _u_scope == "segment":
                # segment-local softmax — renormalize within [seg_start, seg_start+seg_len).
                # scores is raw (pre-softmax, causal-masked); doc segments precede the probe so no mask.
                _seg_sc = scores[:, :, seg_start : seg_start + seg_len]      # [P, Hkv, seg_len]
                _seg_lse = torch.logsumexp(_seg_sc, dim=-1, keepdim=True)    # [P, Hkv, 1]
                _seg_probs = torch.exp(_seg_sc - _seg_lse)
                pt_l1_list.append(_seg_probs.mean(dim=(0, 1))[:valid_len])  # [valid_len]
            else:
                pt_l1_list.append(u_per_token[seg_start : seg_start + valid_len])

            # Segment-level U_L1: sum of per-token attention mass
            seg_l1_list.append(u_per_token[seg_start : seg_start + seg_len].sum())

            # Segment-level U_L2 (only if needed for dump/observe)
            if compute_seg_u_l2:
                seg_probs = probs[:, :, seg_start : seg_start + seg_len]  # [P, Hkv, seg_len]
                # L2 norm of segment probability vector: sqrt(sum p_j^2), averaged over probes and heads
                u_l2_seg = seg_probs.pow(2).sum(dim=-1).sqrt().mean()
                seg_l2_list.append(u_l2_seg)

        result[req_idx] = {
            "u_l1_per_token": pt_l1_list,
            "u_l1_seg": torch.stack(seg_l1_list),
            "u_l2_seg": torch.stack(seg_l2_list) if compute_seg_u_l2 else None,
        }

    if _PROF:
        torch.cuda.synchronize()
        _pt_total_t1 = time.perf_counter()
        print(f"[PROF] _compute_u_scores_l1_fast: "
              f"TOTAL={(_pt_total_t1-_pt_total_t0)*1e3:.2f}ms", flush=True)

    return result


# -----------------------------------------------------------------------------
# Controlled by SGLANG_KVFOCUS_DUMP_EXTENDED=1
# -----------------------------------------------------------------------------


def compute_per_token_u_l1_only(
    q: torch.Tensor,
    k: torch.Tensor,
    tasks_by_req,
    start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    num_probes: int = 2,
    compute_seg_u_l2: bool = False,
    suffix_starts=None,           # signature parity with fast path (unused here).
                                  # Pre-existing: caller always passes suffix_starts=, so the slow
                                  # path (SGLANG_KVFOCUS_USE_SLOW_U=1) would TypeError without this.
) -> dict:
    """Compute per-token U_L1 and segment-level U from a single attention score pass.

    This is the primary scoring function for KVFocus.
    Returns per-token U_L1 for s_A = m_i × U_L1_pt[i] scoring.
    Segment-level U_L1 is always computed (cheap: one logsumexp + exp).
    Segment-level U_L2 is only computed when compute_seg_u_l2=True
    (needed by harm observation / dump).

    Returns:
        {req_idx: {
            "u_l1_per_token": List[Tensor[valid_len]],
            "u_l1_seg": Tensor[num_segments],
            "u_l2_seg": Tensor[num_segments] or None,
        }}
    """
    kv_group_num = num_q_heads // num_kv_heads

    _pt_total_t0 = time.perf_counter() if _PROF else 0

    shared = _compute_probe_attention_scores(
        q, k, tasks_by_req, start_loc, seq_lens,
        num_q_heads, num_kv_heads, head_dim, num_probes,
    )

    if _PROF:
        torch.cuda.synchronize()
    _pt_post_shared = time.perf_counter() if _PROF else 0

    result = {}

    for req_idx, data in shared.items():
        scores = data["scores"]       # [P, Hq, L]
        lse_all = data["lse_all"]     # [P, Hq]
        tasks = data["tasks"]

        pt_l1_list = []
        seg_l1_list = []
        seg_l2_list = [] if compute_seg_u_l2 else None

        _seg_exp_t = 0.0
        for task in tasks:
            seg_start = task["rel_start"]
            seg_len = task["src_len"]
            src_offset = task.get("src_offset", 0)
            src_block_indices = task.get("src_block_indices")

            if src_block_indices is not None and len(src_block_indices) > 0:
                valid_len = min(len(src_block_indices) - src_offset, seg_len)
            else:
                valid_len = seg_len - src_offset

            _et0 = time.perf_counter() if _PROF else 0

            seg_scores = scores[:, :, seg_start:seg_start + seg_len]  # [P, Hq, seg_len]
            seg_probs = torch.exp(seg_scores - lse_all.unsqueeze(-1))  # [P, Hq, seg_len]

            # Per-token U_L1
            pt_l1_list.append(seg_probs[:, :, :valid_len].mean(dim=(0, 1)))  # [valid_len]

            # Segment-level U_L1 (always: cheap)
            lse_seg = torch.logsumexp(seg_scores, dim=-1)  # [P, Hq]
            mass_l1 = torch.exp(lse_seg - lse_all)  # [P, Hq]
            seg_l1_list.append(mass_l1.mean())

            # Segment-level U_L2 (only if needed for dump/observe)
            if compute_seg_u_l2:
                lse_seg_2x = torch.logsumexp(2.0 * seg_scores, dim=-1)  # [P, Hq]
                mass_l2 = torch.exp(0.5 * lse_seg_2x - lse_all)  # [P, Hq]
                u_intra = os.environ.get("SGLANG_KVFOCUS_U_INTRA_GROUP", "mean")
                u_inter = os.environ.get("SGLANG_KVFOCUS_U_INTER_GROUP", "mean")
                if kv_group_num > 1:
                    mass_grouped = mass_l2.view(-1, num_kv_heads, kv_group_num)
                    if u_intra == "max":
                        mass_grouped = mass_grouped.max(dim=-1).values
                    else:
                        mass_grouped = mass_grouped.mean(dim=-1)
                    if u_inter == "frobenius":
                        seg_l2_list.append(mass_grouped.pow(2).sum(dim=-1).mean().sqrt())
                    else:
                        seg_l2_list.append(mass_grouped.mean())
                else:
                    seg_l2_list.append(mass_l2.mean())

            if _PROF:
                _seg_exp_t += time.perf_counter() - _et0

        result[req_idx] = {
            "u_l1_per_token": pt_l1_list,
            "u_l1_seg": torch.stack(seg_l1_list),
            "u_l2_seg": torch.stack(seg_l2_list) if compute_seg_u_l2 else None,
        }

    if _PROF:
        torch.cuda.synchronize()
        _pt_total_t1 = time.perf_counter()
        print(f"[PROF] compute_per_token_u_l1_only: shared={(_pt_post_shared-_pt_total_t0)*1e3:.2f}ms  "
              f"seg_exp+mean={_seg_exp_t*1e3:.2f}ms  "
              f"TOTAL={(_pt_total_t1-_pt_total_t0)*1e3:.2f}ms", flush=True)

    return result


def compute_probe_scores_merged(
    q: torch.Tensor,
    k: torch.Tensor,
    tasks_by_req,
    start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    num_probes: int = 2,
) -> dict:
    """Compute all probe-derived signals from a single attention score pass.

    Merges compute_probe_segment_score_L1, compute_probe_segment_score_L2,
    and compute_per_token_u_scores into one call to _compute_probe_attention_scores,
    eliminating redundant einsum computation.

    Returns:
        {req_idx: {
            "u_l1_seg": Tensor[num_segments],       # segment-level U_L1
            "u_l2_seg": Tensor[num_segments],       # segment-level U_L2
            "u_l1_per_token": List[Tensor[valid_len]],  # per-token U_L1
            "u_l2_per_token": List[Tensor[valid_len]],  # per-token U_L2
        }}
    """
    u_intra = os.environ.get("SGLANG_KVFOCUS_U_INTRA_GROUP", "mean")
    u_inter = os.environ.get("SGLANG_KVFOCUS_U_INTER_GROUP", "mean")
    kv_group_num = num_q_heads // num_kv_heads

    shared = _compute_probe_attention_scores(
        q, k, tasks_by_req, start_loc, seq_lens,
        num_q_heads, num_kv_heads, head_dim, num_probes,
    )

    result = {}

    for req_idx, data in shared.items():
        scores = data["scores"]       # [P, Hq, L]
        lse_all = data["lse_all"]     # [P, Hq]
        tasks = data["tasks"]

        # Full softmax (needed for per-token U)
        probs = torch.exp(scores - lse_all.unsqueeze(-1))  # [P, Hq, L]

        seg_l1_list = []
        seg_l2_list = []
        pt_l1_list = []
        pt_l2_list = []

        for task in tasks:
            seg_start = task["rel_start"]
            seg_len = task["src_len"]
            src_offset = task.get("src_offset", 0)
            src_block_indices = task.get("src_block_indices")

            # valid_len calculation
            if src_block_indices is not None and len(src_block_indices) > 0:
                valid_len = min(len(src_block_indices) - src_offset, seg_len)
            else:
                valid_len = seg_len - src_offset

            # --- Segment-level U_L1 ---
            seg_scores = scores[:, :, seg_start:seg_start + seg_len]
            lse_seg = torch.logsumexp(seg_scores, dim=-1)  # [P, Hq]
            mass_l1 = torch.exp(lse_seg - lse_all)  # [P, Hq]
            seg_l1_list.append(mass_l1.mean())

            # --- Segment-level U_L2 ---
            lse_seg_2x = torch.logsumexp(2.0 * seg_scores, dim=-1)  # [P, Hq]
            mass_l2 = torch.exp(0.5 * lse_seg_2x - lse_all)  # [P, Hq]

            if kv_group_num > 1:
                mass_grouped = mass_l2.view(-1, num_kv_heads, kv_group_num)
                if u_intra == "max":
                    mass_grouped = mass_grouped.max(dim=-1).values
                else:
                    mass_grouped = mass_grouped.mean(dim=-1)
                if u_inter == "frobenius":
                    u_c = mass_grouped.pow(2).sum(dim=-1).mean().sqrt()
                else:
                    u_c = mass_grouped.mean()
            else:
                u_c = mass_l2.mean()
            seg_l2_list.append(u_c)

            # --- Per-token U_L1 and U_L2 ---
            seg_probs = probs[:, :, seg_start:seg_start + seg_len]  # [P, Hq, seg_len]
            seg_probs_valid = seg_probs[:, :, :valid_len]

            pt_l1_list.append(seg_probs_valid.mean(dim=(0, 1)))                           # [valid_len]
            pt_l2_list.append(seg_probs_valid.pow(2).mean(dim=(0, 1)).sqrt())             # [valid_len]

        result[req_idx] = {
            "u_l1_seg": torch.stack(seg_l1_list),
            "u_l2_seg": torch.stack(seg_l2_list),
            "u_l1_per_token": pt_l1_list,
            "u_l2_per_token": pt_l2_list,
        }

    return result


def compute_per_token_u_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    tasks_by_req,
    start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    num_probes: int = 2,
) -> dict:
    """Compute per-token attention weight from probe tokens to each token in each segment.

    Returns:
        {req_idx: {"u_l1_per_token": [Tensor[valid_len]], "u_l2_per_token": [Tensor[valid_len]]}}

    Note: Output tensors have length valid_len (same as m_per_token),
    so that per-token multiplication m_j * u_j is well-defined.
    seg_probs is already segment-relative, so slicing [:valid_len] (no src_offset)

    u_l1_per_token[j] = mean_{p,h} softmax(scores)_{p,h,j}  (L1 mass per token)
    u_l2_per_token[j] = sqrt(mean_{p,h} softmax(scores)_{p,h,j}^2)  (L2 mass per token)
    """
    shared = _compute_probe_attention_scores(
        q, k, tasks_by_req, start_loc, seq_lens,
        num_q_heads, num_kv_heads, head_dim, num_probes,
    )
    result = {}

    for req_idx, data in shared.items():
        scores = data["scores"]       # [P, Hq, L]
        lse_all = data["lse_all"]     # [P, Hq]
        tasks = data["tasks"]

        probs = torch.exp(scores - lse_all.unsqueeze(-1))  # [P, Hq, L]

        u_l1_list = []
        u_l2_list = []

        for task in tasks:
            seg_start = task["rel_start"]
            seg_len = task["src_len"]
            src_offset = task.get("src_offset", 0)
            src_block_indices = task.get("src_block_indices")
            # valid_len from compute_diff: min(len(src_block_indices) - src_offset, v_new.shape[0])
            # But seg_probs is already sliced to [seg_start:seg_start+seg_len], so
            # m_per_token[j] aligns with seg_probs[:, :, j] (not seg_probs[:, :, src_offset+j]).
            if src_block_indices is not None and len(src_block_indices) > 0:
                valid_len = min(len(src_block_indices) - src_offset, seg_len)
            else:
                valid_len = seg_len - src_offset
            seg_probs = probs[:, :, seg_start:seg_start + seg_len]  # [P, Hq, seg_len]

            # Slice to valid_len from start (NOT from src_offset) to match m_per_token
            seg_probs_valid = seg_probs[:, :, :valid_len]

            u_l1_token = seg_probs_valid.mean(dim=(0, 1))                     # [valid_len]
            u_l2_token = seg_probs_valid.pow(2).mean(dim=(0, 1)).sqrt()       # [valid_len]

            u_l1_list.append(u_l1_token)
            u_l2_list.append(u_l2_token)

        result[req_idx] = {"u_l1_per_token": u_l1_list, "u_l2_per_token": u_l2_list}

    return result


def compute_suffix_per_token_u_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    tasks_by_req,
    start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    num_probes: int = 2,
) -> dict:
    """Compute per-token attention weight from suffix tokens to each token in each segment.

    Suffix tokens = tokens after all segments end.
    Returns same format as compute_per_token_u_scores.
    """
    sm_scale = head_dim ** -0.5
    kv_group_num = num_q_heads // num_kv_heads
    result = {}

    for req_idx, tasks in enumerate(tasks_by_req):
        if not tasks:
            continue

        req_start = start_loc[req_idx].item()
        req_len = seq_lens[req_idx].item()

        # Determine suffix range
        last_seg_end = max(t["rel_start"] + t["src_len"] for t in tasks)
        suffix_len = req_len - last_seg_end
        if suffix_len <= 0:
            continue

        # Extract suffix Q
        q_suffix = q[req_start + last_seg_end: req_start + req_len]
        q_suffix = q_suffix.view(suffix_len, num_q_heads, head_dim)  # [S, Hq, D]

        # Extract all K
        k_all = k[req_start: req_start + req_len]  # [L, Hkv, D]
        if kv_group_num > 1:
            k_all = k_all.repeat_interleave(kv_group_num, dim=1)  # [L, Hq, D]

        # Compute attention scores: [S, Hq, L]
        scores = torch.einsum('shd,lhd->shl', q_suffix.float(), k_all.float()) * sm_scale

        # Causal mask
        for s_idx in range(suffix_len):
            causal_boundary = last_seg_end + s_idx + 1
            if causal_boundary < req_len:
                scores[s_idx, :, causal_boundary:] = float('-inf')

        # Global LSE: [S, Hq]
        lse_all = torch.logsumexp(scores, dim=-1)

        # Softmax probabilities: [S, Hq, L]
        probs = torch.exp(scores - lse_all.unsqueeze(-1))

        u_l1_list = []
        u_l2_list = []

        for task in tasks:
            seg_start = task["rel_start"]
            seg_len = task["src_len"]
            src_offset = task.get("src_offset", 0)
            src_block_indices = task.get("src_block_indices")
            # valid_len from compute_diff: min(len(src_block_indices) - src_offset, v_new.shape[0])
            # But seg_probs is already sliced to [seg_start:seg_start+seg_len], so
            # m_per_token[j] aligns with seg_probs[:, :, j] (not seg_probs[:, :, src_offset+j]).
            if src_block_indices is not None and len(src_block_indices) > 0:
                valid_len = min(len(src_block_indices) - src_offset, seg_len)
            else:
                valid_len = seg_len - src_offset
            seg_probs = probs[:, :, seg_start:seg_start + seg_len]  # [S, Hq, seg_len]

            # Slice to valid_len from start (NOT from src_offset) to match m_per_token
            seg_probs_valid = seg_probs[:, :, :valid_len]

            u_l1_token = seg_probs_valid.mean(dim=(0, 1))                     # [valid_len]
            u_l2_token = seg_probs_valid.pow(2).mean(dim=(0, 1)).sqrt()       # [valid_len]

            u_l1_list.append(u_l1_token)
            u_l2_list.append(u_l2_token)

        result[req_idx] = {"u_l1_per_token": u_l1_list, "u_l2_per_token": u_l2_list}

    return result

# -----------------------------------------------------------------------------
# Triton Kernel: Simple Block Reader and Writer
# -----------------------------------------------------------------------------

@triton.jit
def read_blocks_kernel(
    Src_Slot_Ptr,       # [Total_Tokens] 逻辑到物理的映射 (Page Table Slice)
    Out_Buffer_Ptr,     # [N, H, D]      输出临时 Buffer
    Src_Buffer_Ptr,     # [Pool, H, D]   源物理 Memory Pool
    stride_out_tok, stride_out_head, stride_out_dim,
    stride_buf_tok, stride_buf_head, stride_buf_dim,
    SRC_OFFSET_START,   # 源逻辑起始偏移
    HEAD_DIM: tl.constexpr, 
    DTYPE_CODE: tl.constexpr
    ):
    """
    从 Memory Pool 的不连续物理块中读取数据到连续的临时 Tensor。
    """
    # 1. Dtype 设置
    if DTYPE_CODE == 0: dtype = tl.float16
    elif DTYPE_CODE == 1: dtype = tl.bfloat16
    else: dtype = tl.float32
    
    token_idx = tl.program_id(0).to(tl.int64)
    head_idx = tl.program_id(1).to(tl.int64)
    
    # 2. 查表：获取源物理 Block Index
    logic_src = (SRC_OFFSET_START + token_idx).to(tl.int64)
    src_phys_idx = tl.load(Src_Slot_Ptr + logic_src).to(tl.int64)
    
    # 3. 计算指针偏移 (使用 int64 防止溢出)
    src_ptr = (
        src_phys_idx.to(tl.int64) * stride_buf_tok.to(tl.int64) + 
        head_idx.to(tl.int64) * stride_buf_head.to(tl.int64)
    )
    out_ptr = (
        token_idx.to(tl.int64) * stride_out_tok.to(tl.int64) + 
        head_idx.to(tl.int64) * stride_out_head.to(tl.int64)
    )
    
    # 4. 搬运数据
    offs = tl.arange(0, HEAD_DIM)
    val = tl.load(Src_Buffer_Ptr.to(tl.pointer_type(dtype)) + src_ptr + offs)
    tl.store(Out_Buffer_Ptr.to(tl.pointer_type(dtype)) + out_ptr + offs, val)

@triton.jit
def write_blocks_kernel(
    Dst_Slot_Ptr,       # [Num_Blocks] 页表 (Page Table)
    Src_Buffer_Ptr,     # [N, H, D]    我们处理好的 tmp_k / tmp_v
    Dst_Buffer_Ptr,     # [Pool_Size, H, D] 物理显存池
    stride_src_tok, stride_src_head, stride_src_dim,
    stride_dst_tok, stride_dst_head, stride_dst_dim,
    DST_OFFSET_START,   # 逻辑起始位置 (通常为0，因为我们切片出来的 loc_tail 已经是目标区域了)
    HEAD_DIM: tl.constexpr, 
    DTYPE_CODE: tl.constexpr
    ):
    if DTYPE_CODE == 0: dtype = tl.float16
    elif DTYPE_CODE == 1: dtype = tl.bfloat16
    else: dtype = tl.float32
    
    copy_idx = tl.program_id(0).to(tl.int64)
    head_idx = tl.program_id(1).to(tl.int64)
    
    # 1. 计算目标物理地址  
    phys_token_idx = tl.load(Dst_Slot_Ptr + copy_idx).to(tl.int64)
    
    src_ptr = (
        copy_idx * stride_src_tok.to(tl.int64) + 
        head_idx * stride_src_head.to(tl.int64)
    )
    
    dst_ptr = (
        phys_token_idx * stride_dst_tok.to(tl.int64) + 
        head_idx * stride_dst_head.to(tl.int64)
    )
    
    offs = tl.arange(0, HEAD_DIM)
    val = tl.load(Src_Buffer_Ptr.to(tl.pointer_type(dtype)) + src_ptr + offs)
    tl.store(Dst_Buffer_Ptr.to(tl.pointer_type(dtype)) + dst_ptr + offs, val)

def _get_dtype_code(dtype):
    if dtype == torch.float16: return 0
    elif dtype == torch.bfloat16: return 1
    elif dtype == torch.float32: return 2
    raise ValueError(f"Unsupported dtype: {dtype}")

# -----------------------------------------------------------------------------
# Main Launcher
# -----------------------------------------------------------------------------

# python 实现
def cacheflex_complementary_copy_launcher_python(slice_info, kv_pool, rotary_emb, layer_id, page_size=1):
    # 1. 获取 Buffer
    if isinstance(kv_pool.k_buffer, list):
        k_buf = kv_pool.k_buffer[layer_id]
        v_buf = kv_pool.v_buffer[layer_id]
    else:
        k_buf = kv_pool.k_buffer
        v_buf = kv_pool.v_buffer
        
    cos_sin_cache = rotary_emb.cos_sin_cache
    
    # 2. 遍历 Slice Info
    for info in slice_info:
        if not info or not info.get("is_sliced"): continue
        
        dst_loc = info["dst_tail_loc"]
        src_slots = info["src_block_indices"]
        src_offset = info["copy_range"][0]  # Old Pos (e.g., 2001)
        prefix_len = info["prefix_len"]     # Delta (e.g., 101)
        
        # [FIX] 先计算 num_tokens
        num_tokens = len(dst_loc)
        
        if num_tokens == 0: continue

        # [FIX] 然后再做安全检查
        if src_slots.numel() == 0:
            print(f"[CacheFlex ERROR] Src Block Indices is EMPTY for Layer {layer_id}! Skipping Copy.")
            continue
            
        max_slot_idx = src_slots.numel()
        # 假设 PageSize=1 (Token Level)
        # 如果 PageSize > 1，这里应该用 max_slot_idx * page_size
        total_available_tokens = max_slot_idx * page_size
        needed_len = src_offset + num_tokens
        
        if needed_len > total_available_tokens:
             print(f"[CacheFlex ERROR] Read Out of Bounds! Needed {needed_len}, Have {total_available_tokens}")
             # 可以选择 continue 跳过，或者让它 crash 以便发现问题
             # continue

        # 计算 Target Pos (New Pos)
        # 逻辑：Head 长度 = src_offset - 1 (因为 src_offset 是 2001, Head 是 0~2000 也就是 2000 个 + BOS 1 个? 
        # 不，简单点：Target = Old + Delta)
        # 验证：Old=2001, Delta=101 => Target=2102. 
        # 这与我们之前的推导一致。
        target_pos_start = src_offset + prefix_len
        
        num_tokens = len(dst_loc)
        if num_tokens == 0: continue

        # --- 3. Read K/V ---
        head_num = k_buf.shape[1]
        head_dim = k_buf.shape[2]
        dtype_code = _get_dtype_code(k_buf.dtype)
        
        tmp_k = torch.empty((num_tokens, head_num, head_dim), dtype=k_buf.dtype, device=k_buf.device)
        tmp_v = torch.empty((num_tokens, head_num, head_dim), dtype=v_buf.dtype, device=v_buf.device)
        
        grid = (num_tokens, head_num)
        
        read_blocks_kernel[grid](
            src_slots, tmp_k, k_buf,
            *tmp_k.stride(), *k_buf.stride(),
            SRC_OFFSET_START=int(src_offset),
            HEAD_DIM=int(head_dim), DTYPE_CODE=dtype_code
        )
        # V 不需要处理，直接读
        read_blocks_kernel[grid](
            src_slots, tmp_v, v_buf,
            *tmp_v.stride(), *v_buf.stride(),
            SRC_OFFSET_START=int(src_offset),
            HEAD_DIM=int(head_dim), DTYPE_CODE=dtype_code
        )

        # --- 4. Un-rotate & Re-rotate (FP32 Precision) ---
        
        # 转换为 FP32 避免精度崩坏
        k_f32 = tmp_k.to(torch.float32)
        rot_dim = rotary_emb.rotary_dim
        
        # 准备 Pos Indices
        # Old Pos: [src_offset, src_offset+1, ...]
        # New Pos: [target_pos_start, target_pos_start+1, ...]
        old_positions = torch.arange(src_offset, src_offset + num_tokens, device=k_buf.device)
        new_positions = torch.arange(target_pos_start, target_pos_start + num_tokens, device=k_buf.device)
        
        # 获取 Cos/Sin
        # Shape: [num_tokens, rot_dim]
        cs_old = cos_sin_cache[old_positions].to(torch.float32)
        cs_new = cos_sin_cache[new_positions].to(torch.float32)
        
        # Split Cos/Sin (SGLang is usually [cos, sin])
        cos_old = cs_old[:, :rot_dim//2].view(num_tokens, 1, -1) # [N, 1, D/2]
        sin_old = cs_old[:, rot_dim//2:].view(num_tokens, 1, -1)
        
        cos_new = cs_new[:, :rot_dim//2].view(num_tokens, 1, -1)
        sin_new = cs_new[:, rot_dim//2:].view(num_tokens, 1, -1)
        
        # Split K (Half-Half Layout for is_neox=True)
        mid = rot_dim // 2
        k_rot = k_f32[..., :rot_dim]
        k_pass = k_f32[..., rot_dim:]
        
        k1 = k_rot[..., :mid]
        k2 = k_rot[..., mid:]
        
        # A. Un-rotate (恢复 K_raw)
        # Inverse Rotation: Rotate by -theta
        # cos(-t) = cos(t), sin(-t) = -sin(t)
        # Formula: K_raw = K_old * cos + K_old_swap * (-sin)
        #        = K_old * cos - K_old_swap * sin
        # Half-Half Swap: [-k2, k1] 对应正向旋转
        # 逆向旋转逻辑:
        # raw_1 = k1 * c - (-k2 * s) = k1 * c + k2 * s (Wait, inverse matrix is Transpose)
        # Let's derive:
        # [ k1 ]   [ c  -s ] [ raw1 ]
        # [ k2 ] = [ s   c ] [ raw2 ]
        # Inverse:
        # [ raw1 ]   [ c   s ] [ k1 ]
        # [ raw2 ] = [ -s  c ] [ k2 ]
        # So: raw1 = k1*c + k2*s
        #     raw2 = -k1*s + k2*c
        
        raw_1 = k1 * cos_old + k2 * sin_old
        raw_2 = k2 * cos_old - k1 * sin_old
        
        # B. Re-rotate (计算 K_new)
        # Formula:
        # new_1 = raw1 * c_new - raw2 * s_new
        # new_2 = raw1 * s_new + raw2 * c_new
        
        new_1 = raw_1 * cos_new - raw_2 * sin_new
        new_2 = raw_1 * sin_new + raw_2 * cos_new
        
        # 更新 K
        tmp_k[..., :mid] = new_1.to(tmp_k.dtype)
        tmp_k[..., mid:rot_dim] = new_2.to(tmp_k.dtype)

        # --- 5. Write Back ---
        
        # 写回 K
        write_blocks_kernel[grid](
            dst_loc, tmp_k, k_buf,
            *tmp_k.stride(), *k_buf.stride(),
            DST_OFFSET_START=0,
            HEAD_DIM=int(head_dim), DTYPE_CODE=dtype_code
        )
        
        # 写回 V
        write_blocks_kernel[grid](
            dst_loc, tmp_v, v_buf,
            *tmp_v.stride(), *v_buf.stride(),
            DST_OFFSET_START=0,
            HEAD_DIM=int(head_dim), DTYPE_CODE=dtype_code
        )

# kernel 实现
def cacheflex_complementary_copy_launcher_kernel(slice_info, kv_pool, rotary_emb, layer_id):
    """
    CacheFlex 核心搬运函数 (Optimized with Delta RoPE & SGL Kernel):
    1. Read: 从源物理地址读取 K/V 到临时 Tensor。
    2. Delta RoPE & Write: 利用 sgl_kernel 接口，对 K 应用 PrefixLen 旋转，并直接写入目标物理地址。
    """
    if not slice_info: return
    if apply_rope_with_cos_sin_cache_inplace is None:
        raise ImportError("sgl_kernel not found or version incompatible!")

    # 1. 获取物理 Buffer
    # 兼容 SGLang 不同版本的 Pool 实现
    if isinstance(kv_pool.k_buffer, list):
        k_buf = kv_pool.k_buffer[layer_id]
        v_buf = kv_pool.v_buffer[layer_id]
    else:
        k_buf = kv_pool.k_buffer
        v_buf = kv_pool.v_buffer

    # 获取 Cos/Sin Cache
    cos_sin_cache = rotary_emb.cos_sin_cache
    # 获取 RoPE 风格和维度
    is_neox_style = getattr(rotary_emb, "is_neox_style", True)
    rotary_dim = getattr(rotary_emb, "rotary_dim", k_buf.shape[-1])

    for info in slice_info:
        # 仅处理需要 Copy 的部分
        if not info or not info.get("is_sliced"): continue
        
        dst_loc = info["dst_tail_loc"]          # 目标物理地址 [N]
        src_slots = info["src_block_indices"]   # 源物理地址表 (Request Page Table)
        src_offset = info["copy_range"][0]      # 源逻辑起始位置
        prefix_len = info["prefix_len"]         # 前缀长度 (作为 Delta Position)
        num_tokens = len(dst_loc)

        if num_tokens == 0: continue

        # --- Step 1: Read Source K/V to Temp Buffer ---
        head_num = k_buf.shape[1]
        head_dim = k_buf.shape[2]
        dtype_code = _get_dtype_code(k_buf.dtype)
        
        # 临时 Buffer 必须是连续的
        tmp_k = torch.empty((num_tokens, head_num, head_dim), dtype=k_buf.dtype, device=k_buf.device)
        tmp_v = torch.empty((num_tokens, head_num, head_dim), dtype=v_buf.dtype, device=v_buf.device)
        
        grid = (num_tokens, head_num)
        
        # 启动 Triton Kernel 读取不连续的 Block
        read_blocks_kernel[grid](
            src_slots, tmp_k, k_buf,
            *tmp_k.stride(), *k_buf.stride(),
            SRC_OFFSET_START=int(src_offset),
            HEAD_DIM=int(head_dim), DTYPE_CODE=dtype_code
        )
        read_blocks_kernel[grid](
            src_slots, tmp_v, v_buf,
            *tmp_v.stride(), *v_buf.stride(),
            SRC_OFFSET_START=int(src_offset),
            HEAD_DIM=int(head_dim), DTYPE_CODE=dtype_code
        )

        # --- Step 2: Apply Delta RoPE & Write (Via SGL Kernel) ---
        
        # 构造 Delta Positions
        # K_new = K_old * RoPE(Delta)
        # Delta = PrefixLen

        # 数学技巧：利用 RoPE 的奇偶性处理负数偏移
        # R(-theta) = Flip_Sign( R(theta) * Flip_Sign(K) )
        # 其中 Flip_Sign 是对向量的第二部分 (sin 相关项) 取负        
        is_negative_delta = prefix_len < 0
        abs_delta = abs(int(prefix_len))
        if is_negative_delta:
            # 1. 预处理：反转 K 的第二部分符号
            # 假设 Neox Style (Half-Half layout)
            half = rotary_dim // 2
            tmp_k[..., half:rotary_dim] *= -1

        delta_positions = torch.full(
            (num_tokens,), 
            abs_delta, 
            dtype=torch.int64, 
            device=k_buf.device
        )
        
        # 构造 Dummy Q (必须与 K 长度一致，Kernel 会对它做 inplace 操作，但我们不关心结果)
        # 注意：Head Num 必须匹配 K 的 Head Num (num_kv_heads)
        dummy_q = torch.empty((num_tokens, head_num, head_dim), dtype=k_buf.dtype, device=k_buf.device)
        
        # 构造 Fused 参数对象
        # 这个对象告诉 Kernel 把结果写到哪里
        # value: 这里传入 Source V (tmp_v)，Kernel 会把它搬运到 v_buffer[dst_loc]
        fused_arg = FusedSetKVBufferArg(
            k_buffer=k_buf,
            v_buffer=v_buf,
            value=tmp_v, 
            cache_loc=dst_loc,
            k_scale=None, # For BF16/FP16
            v_scale=None  # For BF16/FP16
        )
        
        # 调用官方接口
        # 逻辑：
        # 1. 读取 key (tmp_k)
        # 2. 读取 cos/sin cache (根据 positions 索引)
        # 3. 执行 RoPE: key_rotated = key * cos + key_swap * sin
        # 4. 写入 k_buffer[cache_loc] = key_rotated
        # 5. 写入 v_buffer[cache_loc] = fused_arg.value (即 tmp_v)
        
        apply_rope_with_cos_sin_cache_inplace(
            positions=delta_positions,
            query=dummy_q,
            key=tmp_k,  # In-place modified, but also written to k_buffer
            head_size=int(head_dim),
            cos_sin_cache=cos_sin_cache,
            is_neox=is_neox_style,
            fused_set_kv_buffer_arg=fused_arg
        )

        if is_negative_delta:
            # 2. 后处理：官方算子已将结果写入 k_buf，我们需要反转写回后的第二部分符号
            # 这样在数学上完全等价于应用了负旋转矩阵
            half = rotary_dim // 2
            k_buf[dst_loc, :, half:rotary_dim] *= -1

