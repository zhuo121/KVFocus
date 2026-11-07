# HACK(Shizhuo) New
"""
Fused Triton kernel for computing per-token U_L1 attention weights from probe positions.

Replaces the Python-level `_compute_probe_attention_scores` + `compute_per_token_u_l1_only`
pipeline, which materializes the full [P, Hq, L] score tensor (~114ms).

This kernel computes U_L1_per_token[i] = mean_{p,h} softmax(q_probe_p · k_i / sqrt(d))
for tokens within each segment, using a two-pass online softmax approach per probe:
  Pass 1: scan all KV tokens -> compute global row max (m) and normalization (l)
  Pass 2: given m/l, compute normalized per-token probs and atomic_add to output

Each (request, q_head, probe) is a separate program. After kernel, Python divides
by (actual_probes * num_q_heads) to get the mean over probes and heads.
"""

import os
import time

import torch
import triton
import triton.language as tl

# unified profile control (replaces SGLANG_KVFOCUS_PROF; auto-mapped)
from sglang.srt.utils import cacheflex_dump as _cf_dump
_PROF = _cf_dump.profile_kernel_enabled()


def _next_power_of_2(n: int) -> int:
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


@triton.jit
def _probe_ul1_kernel(
    # --- Input pointers ---
    Q_Ptr,              # [TotalTokens, QHeads, HeadDim]
    K_Ptr,              # [TotalTokens, KVHeads, HeadDim]
    Req_Meta_Ptr,       # [num_active * REQ_META_STRIDE]
    Seg_Meta_Ptr,       # [total_segments * SEG_META_STRIDE]

    # --- Output ---
    Out_Ptr,            # [total_output_size] accumulator

    # --- Strides ---
    stride_q_tok, stride_q_head,
    stride_k_tok, stride_k_head,

    # --- Params ---
    sm_scale,
    num_probes,
    num_q_heads,
    num_kv_heads,
    kv_group_num,
    REQ_META_STRIDE: tl.constexpr,
    SEG_META_STRIDE: tl.constexpr,

    # --- Block sizes ---
    head_dim: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Grid: (num_active_requests * num_probes, num_q_heads)
    Each program handles one (request, probe_idx, q_head).

    Two-pass kernel per probe:
      Pass 1: iterate all KV tokens, compute m (row max) and l (normalization)
      Pass 2: iterate segment KV tokens, compute normalized probs, atomic_add to output
    """
    pid = tl.program_id(0)
    pid_head = tl.program_id(1)

    # Decode request index and probe index from pid
    req_idx = pid // num_probes
    probe_idx = pid % num_probes

    # --- Load request metadata ---
    rm_base = req_idx * REQ_META_STRIDE
    req_q_start = tl.load(Req_Meta_Ptr + rm_base + 0)
    req_q_len = tl.load(Req_Meta_Ptr + rm_base + 1)
    seg_start_idx = tl.load(Req_Meta_Ptr + rm_base + 2)
    num_segments = tl.load(Req_Meta_Ptr + rm_base + 3)
    actual_probes = tl.load(Req_Meta_Ptr + rm_base + 4)

    # Early exit: this probe index is out of range or no segments
    if probe_idx >= actual_probes or num_segments == 0:
        return

    # KV head for GQA
    kv_head = pid_head // kv_group_num

    # --- Load single probe Q vector: [head_dim] ---
    probe_abs_idx = req_q_start + req_q_len - actual_probes + probe_idx
    offs_d = tl.arange(0, head_dim)
    mask_d = offs_d < head_dim
    off_q = probe_abs_idx.to(tl.int64) * stride_q_tok + pid_head.to(tl.int64) * stride_q_head + offs_d
    q_vec = tl.load(Q_Ptr + off_q, mask=mask_d, other=0.0)

    # Probe's logical position within request (for causal mask)
    probe_logic = req_q_len - actual_probes + probe_idx

    # --- Load segment metadata ---
    # We need to iterate in two passes, so store segment info in local arrays.
    # Max 32 segments; use scalar loop since Triton doesn't support dynamic indexing well.
    # Instead, we re-read segment metadata in both passes.

    # Compute total KV length
    total_kv_len = 0
    for s in range(32):
        if s < num_segments:
            sm_base = (seg_start_idx + s) * SEG_META_STRIDE
            kv_len_s = tl.load(Seg_Meta_Ptr + sm_base + 1)
            kv_start_s = tl.load(Seg_Meta_Ptr + sm_base + 0)
            end_s = kv_start_s + kv_len_s
            total_kv_len = tl.maximum(total_kv_len, end_s)

    if total_kv_len == 0:
        return

    NEGINF = -1e30

    # ====================================================================
    # PASS 1: Compute global m and l via online softmax over ALL KV tokens
    # Must scan the full request range [0, req_q_len) for correct normalization.
    # Python path normalizes over k[req_start : req_start + req_len], which is
    # the entire extend range, not just segment tokens.
    # ====================================================================
    m_val = NEGINF
    l_val = 0.0

    offs_n = tl.arange(0, BLOCK_N)

    for start_n in range(0, req_q_len, BLOCK_N):
        kv_logic = start_n + offs_n
        mask_n = kv_logic < req_q_len

        abs_k = req_q_start + kv_logic

        off_k = (
            abs_k[:, None].to(tl.int64) * stride_k_tok +
            kv_head.to(tl.int64) * stride_k_head +
            offs_d[None, :]
        )
        k_block = tl.load(K_Ptr + off_k, mask=(mask_n[:, None] & mask_d[None, :]), other=0.0)

        # QK dot product: [BLOCK_N]
        qk = tl.sum(q_vec[None, :] * k_block, axis=1) * sm_scale

        # Causal mask
        causal = kv_logic <= probe_logic
        qk = tl.where(mask_n & causal, qk, NEGINF)

        # Online softmax update
        row_max = tl.max(qk)
        m_new = tl.maximum(m_val, row_max)
        alpha = tl.exp(m_val - m_new) if m_val > NEGINF + 1 else 0.0
        l_val = l_val * alpha + tl.sum(tl.exp(qk - m_new))
        m_val = m_new

    # ====================================================================
    # PASS 2: Compute normalized per-token probabilities, write per segment
    # Use safe_l to avoid division by zero (if l_val==0, prob will be 0)
    # ====================================================================
    safe_l = tl.where(l_val > 0, l_val, 1.0)

    for s in range(32):
        if s < num_segments:
            sm_base = (seg_start_idx + s) * SEG_META_STRIDE
            kv_start = tl.load(Seg_Meta_Ptr + sm_base + 0)
            kv_len = tl.load(Seg_Meta_Ptr + sm_base + 1)
            out_offset = tl.load(Seg_Meta_Ptr + sm_base + 2)

            if kv_len > 0:
                for start_n in range(0, kv_len, BLOCK_N):
                    tok_logic = start_n + offs_n
                    mask_tok = tok_logic < kv_len

                    abs_k = req_q_start + kv_start + tok_logic

                    off_k = (
                        abs_k[:, None].to(tl.int64) * stride_k_tok +
                        kv_head.to(tl.int64) * stride_k_head +
                        offs_d[None, :]
                    )
                    k_block = tl.load(K_Ptr + off_k, mask=(mask_tok[:, None] & mask_d[None, :]), other=0.0)

                    # QK
                    qk = tl.sum(q_vec[None, :] * k_block, axis=1) * sm_scale

                    # Causal mask
                    kv_pos_in_req = kv_start + tok_logic
                    causal = kv_pos_in_req <= probe_logic
                    valid_mask = mask_tok & causal
                    qk = tl.where(valid_mask, qk, NEGINF)

                    # Normalized probability
                    prob = tl.exp(qk - m_val) / safe_l
                    prob = tl.where(valid_mask, prob, 0.0)

                    # Atomic add
                    out_idx = out_offset + tok_logic
                    tl.atomic_add(Out_Ptr + out_idx, prob, mask=mask_tok)


def compute_ul1_triton(
    q: torch.Tensor,              # [TotalTokens, QHeads*HeadDim]
    k: torch.Tensor,              # [TotalTokens, KVHeads, HeadDim]
    tasks_by_req,
    start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    num_probes: int = 2,
) -> dict:
    """
    Fused Triton kernel replacement for compute_per_token_u_l1_only.

    Returns:
        {req_idx: {"u_l1_per_token": List[Tensor[valid_len]]}}
    """
    kv_group_num = num_q_heads // num_kv_heads
    sm_scale = head_dim ** -0.5
    device = q.device

    _t0 = time.perf_counter() if _PROF else 0

    q_3d = q.view(-1, num_q_heads, head_dim)

    REQ_META_STRIDE = 5
    SEG_META_STRIDE = 3

    active_requests = []
    all_seg_info = []
    total_output_size = 0
    req_seg_results = {}

    for req_idx, tasks in enumerate(tasks_by_req):
        if not tasks:
            continue
        req_start = start_loc[req_idx].item()
        req_len = seq_lens[req_idx].item()
        actual_probes = min(num_probes, req_len)
        if actual_probes <= 0:
            continue

        seg_start_idx = len(all_seg_info)
        seg_results = []

        for task in tasks:
            seg_start = task["rel_start"]
            seg_len = task["src_len"]
            src_offset = task.get("src_offset", 0)
            src_block_indices = task.get("src_block_indices")

            if src_block_indices is not None and len(src_block_indices) > 0:
                valid_len = min(len(src_block_indices) - src_offset, seg_len)
            else:
                valid_len = seg_len - src_offset

            out_offset = total_output_size
            all_seg_info.append((seg_start, valid_len, out_offset))
            total_output_size += valid_len
            seg_results.append((valid_len, out_offset))

        active_requests.append((req_idx, req_start, req_len, seg_start_idx, len(tasks), actual_probes))
        req_seg_results[req_idx] = seg_results

    if not active_requests:
        return {}

    num_active = len(active_requests)

    # Build metadata tensors
    req_meta = torch.zeros(num_active * REQ_META_STRIDE, dtype=torch.int32, device=device)
    for i, (req_idx, req_start, req_len, seg_start_idx, num_segs, actual_probes) in enumerate(active_requests):
        base = i * REQ_META_STRIDE
        req_meta[base + 0] = req_start
        req_meta[base + 1] = req_len
        req_meta[base + 2] = seg_start_idx
        req_meta[base + 3] = num_segs
        req_meta[base + 4] = actual_probes

    seg_meta = torch.zeros(len(all_seg_info) * SEG_META_STRIDE, dtype=torch.int32, device=device)
    for i, (kv_start, kv_len, out_offset) in enumerate(all_seg_info):
        base = i * SEG_META_STRIDE
        seg_meta[base + 0] = kv_start
        seg_meta[base + 1] = kv_len
        seg_meta[base + 2] = out_offset

    output = torch.zeros(total_output_size, dtype=torch.float32, device=device)

    _t1 = time.perf_counter() if _PROF else 0

    # Grid: (num_active * num_probes, num_q_heads)
    BLOCK_N = 64
    grid = (num_active * num_probes, num_q_heads)

    _probe_ul1_kernel[grid](
        q_3d, k, req_meta, seg_meta, output,
        q_3d.stride(0), q_3d.stride(1),
        k.stride(0), k.stride(1),
        sm_scale=sm_scale,
        num_probes=num_probes,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        kv_group_num=kv_group_num,
        REQ_META_STRIDE=REQ_META_STRIDE,
        SEG_META_STRIDE=SEG_META_STRIDE,
        head_dim=head_dim,
        BLOCK_N=BLOCK_N,
    )

    if _PROF:
        torch.cuda.synchronize()
    _t2 = time.perf_counter() if _PROF else 0

    # Divide by (actual_probes * num_q_heads) to get mean
    result = {}
    for req_idx, req_start, req_len, seg_start_idx, num_segs, actual_probes in active_requests:
        seg_results = req_seg_results[req_idx]
        divisor = float(actual_probes * num_q_heads)
        pt_l1_list = []

        for valid_len, out_offset in seg_results:
            seg_output = output[out_offset : out_offset + valid_len] / divisor
            pt_l1_list.append(seg_output)

        result[req_idx] = {"u_l1_per_token": pt_l1_list}

    if _PROF:
        torch.cuda.synchronize()
        _t3 = time.perf_counter()
        print(f"[PROF] compute_ul1_triton: prep={(_t1-_t0)*1e3:.2f}ms  "
              f"kernel={(_t2-_t1)*1e3:.2f}ms  extract={(_t3-_t2)*1e3:.2f}ms  "
              f"TOTAL={(_t3-_t0)*1e3:.2f}ms", flush=True)

    return result
