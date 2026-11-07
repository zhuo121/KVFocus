# HACK(Shizhuo) New
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
"""
Cacheflex Memory-efficient attention for prefill.
It supports page size = 1 and prefill with KV cache (i.e. extend).
"""

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.triton_ops.prefill_attention import (
    context_attention_fwd,
)
from sglang.srt.utils import is_cuda, is_hip

_is_cuda = is_cuda()
if _is_cuda:
    CUDA_CAPABILITY = torch.cuda.get_device_capability()

_is_hip = is_hip()


def _get_block_sizes_for_extend_attention(Lq: int, Lv: int):
    """
    Get block sizes and configuration for extend attention kernels.

    Args:
        Lq: Query head dimension
        Lv: Value head dimension

    Returns:
        tuple: (BLOCK_DMODEL, BLOCK_DPE, BLOCK_DV, BLOCK_M, BLOCK_N, num_warps)
    """
    # Determine BLOCK_DMODEL and BLOCK_DPE based on head dimension
    if Lq == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif Lq == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    elif Lq == 192:
        BLOCK_DMODEL = 128
        BLOCK_DPE = 64
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lq)
        BLOCK_DPE = 0

    BLOCK_DV = triton.next_power_of_2(Lv)

    # Determine BLOCK_M, BLOCK_N, and num_warps based on hardware
    if _is_hip:
        BLOCK_M, BLOCK_N = (64, 64)
        num_warps = 4
    else:
        if _is_cuda and CUDA_CAPABILITY[0] >= 9:
            # Hopper architecture (H100, etc.)
            if Lq <= 256:
                BLOCK_M, BLOCK_N = (128, 64)
            else:
                BLOCK_M, BLOCK_N = (32, 64)
        elif _is_cuda and CUDA_CAPABILITY[0] >= 8:
            # Ampere architecture (A100, etc.)
            # sm86/sm89 has a much smaller shared memory size (100K) than sm80 (160K)
            if CUDA_CAPABILITY[1] == 9 or CUDA_CAPABILITY[1] == 6:
                if Lq <= 128:
                    BLOCK_M, BLOCK_N = (64, 128)
                elif Lq <= 256:
                    BLOCK_M, BLOCK_N = (64, 64)
                else:
                    BLOCK_M, BLOCK_N = (32, 32)
            else:
                if Lq <= 128:
                    BLOCK_M, BLOCK_N = (128, 128)
                elif Lq <= 256:
                    BLOCK_M, BLOCK_N = (64, 64)
                else:
                    BLOCK_M, BLOCK_N = (32, 64)
        else:
            # Older architectures
            BLOCK_M, BLOCK_N = (64, 64) if Lq <= 128 else (32, 32)

        num_warps = 4 if Lq <= 64 else 8

    return BLOCK_DMODEL, BLOCK_DPE, BLOCK_DV, BLOCK_M, BLOCK_N, num_warps


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _copy_unified_indices_kernel(
    # Input buffers
    prefix_kv_indptr,
    prefix_kv_indices,
    extend_start_loc,
    extend_seq_lens,
    extend_kv_indices,
    unified_kv_indptr,
    # Output buffer
    unified_kv_indices,
    # Size
    bs,
):
    """
    Triton kernel to copy indices to unified buffer (parallel per sequence).
    Each thread block processes one sequence with vectorized loads/stores.
    """
    pid = tl.program_id(0)

    if pid >= bs:
        return

    # Load sequence info
    prefix_start = tl.load(prefix_kv_indptr + pid)
    prefix_end = tl.load(prefix_kv_indptr + pid + 1)
    extend_start = tl.load(extend_start_loc + pid)
    extend_len = tl.load(extend_seq_lens + pid)

    prefix_len = prefix_end - prefix_start
    unified_start = tl.load(unified_kv_indptr + pid)

    # Copy indices in vectorized chunks
    BLOCK_SIZE: tl.constexpr = 128

    # Process prefix indices
    for block_start in range(0, prefix_len, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < prefix_len

        src_idx = prefix_start + offs
        dst_idx = unified_start + offs

        vals = tl.load(prefix_kv_indices + src_idx, mask=mask, other=0)
        tl.store(unified_kv_indices + dst_idx, vals, mask=mask)

    # Process extend indices
    for block_start in range(0, extend_len, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < extend_len

        src_idx = extend_start + offs
        dst_idx = unified_start + prefix_len + offs

        vals = tl.load(extend_kv_indices + src_idx, mask=mask, other=0)
        tl.store(unified_kv_indices + dst_idx, vals, mask=mask)


def build_unified_kv_indices(
    prefix_kv_indptr: torch.Tensor,
    prefix_kv_indices: torch.Tensor,
    extend_start_loc: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    extend_kv_indices: torch.Tensor,
    bs: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build unified KV indices efficiently:
    - Use PyTorch's optimized cumsum (NVIDIA CUB) for indptr
    - Use Triton kernel for parallel index copying

    Returns:
        (unified_kv_indptr, unified_kv_indices, prefix_lens)
    """
    device = prefix_kv_indptr.device

    prefix_lens = prefix_kv_indptr[1 : bs + 1] - prefix_kv_indptr[:bs]

    # Create unified_kv_indptr avoiding direct assignment (for CUDA graph compatibility)
    unified_lens = prefix_lens + extend_seq_lens[:bs]
    unified_kv_indptr = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.cumsum(unified_lens, dim=0),
        ]
    )

    max_unified_len = len(prefix_kv_indices) + len(extend_kv_indices)

    unified_kv_indices = torch.empty(max_unified_len, dtype=torch.int64, device=device)

    # Launch Triton kernel for parallel index copying
    _copy_unified_indices_kernel[(bs,)](
        prefix_kv_indptr,
        prefix_kv_indices,
        extend_start_loc,
        extend_seq_lens,
        extend_kv_indices,
        unified_kv_indptr,
        unified_kv_indices,
        bs,
    )

    return unified_kv_indptr, unified_kv_indices, prefix_lens

# -----------------------------------------------------------------------------
# Triton Kernel (Updated with Int64 Fixes & Dot Product Fix)
# -----------------------------------------------------------------------------

@triton.jit
def _cacheflex_fwd_kernel(
    # --- Tensor Pointers ---
    Q_Sparse,           # [Total_Sparse_Tokens, H, D]
    O_Sparse,           # [Total_Sparse_Tokens, H, D]
    K_Buffer,           # [Pool_Size, H, D] - Full Physical Memory
    V_Buffer,           # [Pool_Size, H, D]
    
    # --- Metadata Pointers ---
    qo_indptr,          # [Batch+1] Sparse Q Ranges
    kv_indptr,          # [Batch+1] Full KV Ranges (Logic Length)
    Req_To_Token_Ptr,   # [Max_Reqs, Max_Context] Physical Page Table
    Req_Pool_Indices,   # [B] Request IDs
    Positions_Ptr,      # [Total_Sparse_Tokens] Real Positions for Q
    
    # --- Strides ---
    stride_q_tok, stride_q_head,      # Q Sparse
    stride_o_tok, stride_o_head,      # O Sparse
    stride_buf_k_tok, stride_buf_k_head, # Pool (Physical)
    stride_buf_v_tok, stride_buf_v_head,
    stride_req_b, stride_req_s,       # Page Table
    
    # --- Params ---
    sm_scale,
    kv_group_num,
    logit_cap,
    
    # --- Constants (Kwargs) ---
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr, BLOCK_DV: tl.constexpr,
    Lq: tl.constexpr, Lv: tl.constexpr,
    IS_CAUSAL: tl.constexpr, 
    USE_CUSTOM_MASK: tl.constexpr, 
    HAS_SINK: tl.constexpr
):
    # -----------------------------------------------------------
    # 1. Setup & Ranges
    # -----------------------------------------------------------
    cur_seq = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_block_m = tl.program_id(2)
    
    cur_kv_head = cur_head // kv_group_num
    cur_req_idx = tl.load(Req_Pool_Indices + cur_seq) # Req ID for Page Table
    
    # Q Ranges (Sparse)
    cur_seq_q_start_idx = tl.load(qo_indptr + cur_seq)
    cur_seq_q_len = tl.load(qo_indptr + cur_seq + 1) - cur_seq_q_start_idx
    
    # KV Ranges (Full: Prefix + Head + Tail + Query)
    cur_seq_kv_start_idx = tl.load(kv_indptr + cur_seq)
    cur_seq_kv_len = tl.load(kv_indptr + cur_seq + 1) - cur_seq_kv_start_idx
    
    # -----------------------------------------------------------
    # 2. Load Q
    # -----------------------------------------------------------
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = tl.arange(0, BLOCK_M)
    mask_m = (cur_block_m * BLOCK_M + offs_m) < cur_seq_q_len
    mask_d = offs_d < Lq
    
    # Global Q Offset
    q_global_offs = cur_seq_q_start_idx + cur_block_m * BLOCK_M + offs_m
    
    # Load Q Real Position (For Causal Mask)
    q_real_pos = tl.load(Positions_Ptr + q_global_offs, mask=mask_m, other=0)
    
    # Load Q Tensor
    # [FIX] Use int64 for large offsets
    off_q = (
        q_global_offs[:, None].to(tl.int64) * stride_q_tok + 
        cur_head.to(tl.int64) * stride_q_head + 
        offs_d[None, :]
    )
    q = tl.load(Q_Sparse + off_q, mask=(mask_m[:, None] & mask_d[None, :]), other=0.0)
    
    # -----------------------------------------------------------
    # 3. Attention Loop (Iterate over Full KV via Page Table)
    # -----------------------------------------------------------
    offs_n = tl.arange(0, BLOCK_N)
    
    # 初始化累加器
    # acc: 累加的加权和 (Numerator)
    # l_i: 累加的归一化因子 (Denominator)
    # m_i: 当前遇到的全局最大 Logit
    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    
    # [Diff] Official iterates prefix then extend. We iterate EVERYTHING via Page Table.
    for start_n in range(0, cur_seq_kv_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        # --- 1. Mask 计算 ---
        kv_logic_idx = start_n + offs_n
        mask_n = kv_logic_idx < cur_seq_kv_len
        
        # --- Causal Masking ---
        if IS_CAUSAL:
            # 必须使用 positions 里的真实位置进行比较
            # q_real_pos shape: [BLOCK_M], kv_logic_idx shape: [BLOCK_N]
            mask_causal = kv_logic_idx[None, :] <= q_real_pos[:, None]
        else:
            mask_causal = True
            
        final_mask = mask_n[None, :] & mask_causal
        
        # [Optimization] Skip block if fully masked
        # if tl.max(final_mask) == 0: continue (Optional, compiler might optimize)
        
        # --- 2. 加载 K ---
        # 查页表获取物理地址
        req_offset = (
            cur_req_idx.to(tl.int64) * stride_req_b + 
            kv_logic_idx.to(tl.int64) * stride_req_s
        )
        kv_phys_idx = tl.load(Req_To_Token_Ptr + req_offset, mask=mask_n, other=0)
        
        # 计算 K Buffer 偏移 (转置加载 [D, N] 以优化 dot)
        off_k = (
            kv_phys_idx[None, :].to(tl.int64) * stride_buf_k_tok + 
            cur_kv_head.to(tl.int64) * stride_buf_k_head + 
            offs_d[:, None]
        )
        # mask_d[:, None] -> [D, 1]
        # mask_n[None, :] -> [1, N]
        k = tl.load(K_Buffer + off_k, mask=(mask_n[None, :] & mask_d[:, None]), other=0.0)
        
        # --- 3. 计算 QK Score ---
        # q: [M, D], k: [D, N] -> qk: [M, N]
        qk = tl.dot(q, k, out_dtype=tl.float32)
        qk *= sm_scale
        
        if logit_cap > 0:
            qk = logit_cap * tanh(qk / logit_cap)
        
        # 应用 Mask (将无效位置设为 -inf)
        qk = tl.where(final_mask, qk, float("-inf"))
            
        # --- 4. Online Softmax 更新 (标准 SGLang 逻辑) ---
        
        # A. 计算当前块的 Row Max
        row_max = tl.max(qk, 1)
        
        # [Safety] 修复 -inf 导致的 NaN 问题
        # 如果一行全被 Mask 掉，row_max 为 -inf。直接减法可能导致 NaN。
        # SGLang 原版使用了这个 trick：如果是 -inf，替换为一个极小值但不为 inf
        row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
        
        # B. 更新全局 Max (m_i_new)
        m_i_new = tl.maximum(m_i, row_max_fixed)
        
        # C. 计算旧累加值的缩放因子 (alpha)
        # alpha = exp(m_old - m_new)
        alpha = tl.exp(m_i - m_i_new)
        
        # D. 计算当前块的 P (概率值)
        # 关键点：直接使用 m_i_new 进行减法
        # p = exp(qk - m_new)
        p = tl.exp(qk - m_i_new[:, None])
        
        # --- 5. 加载 V 并累加 ---
        # Official: offs_kv_loc[:, None] * stride_buf_vbs + offs_dv[None, :]
        # Shape: [N, D] (Normal load)
        mask_dv = tl.arange(0, BLOCK_DV) < Lv
        off_v = (
            kv_phys_idx[:, None].to(tl.int64) * stride_buf_v_tok + 
            cur_kv_head.to(tl.int64) * stride_buf_v_head + 
            tl.arange(0, BLOCK_DV)[None, :]
        )
        v = tl.load(V_Buffer + off_v, mask=(mask_n[:, None] & mask_dv[None, :]), other=0.0)
        
        # 累加 Acc
        # [修正] acc * alpha[:, None] 根据最新的m更新历史累加器
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
        
        # 累加 Denominator (l_i)
        l_i = l_i * alpha + tl.sum(p, 1)
        
        # 更新 m_i 为下一轮做准备
        m_i = m_i_new
        
    # -----------------------------------------------------------
    # 4. Store Output
    # -----------------------------------------------------------
    # [Safety] 防止除以 0 (如果一行全是 Mask)
    acc = tl.where(l_i[:, None] > 0.0, acc / l_i[:, None], 0.0)
    
    mask_dv = tl.arange(0, BLOCK_DV) < Lv
    off_o = (
        q_global_offs[:, None].to(tl.int64) * stride_o_tok + 
        cur_head.to(tl.int64) * stride_o_head + 
        tl.arange(0, BLOCK_DV)[None, :]
    )
    
    tl.store(O_Sparse + off_o, acc.to(O_Sparse.dtype.element_ty), mask=(mask_m[:, None] & mask_dv[None, :]))

# -----------------------------------------------------------------------------
# Python Wrapper (Matched with extend_attention_fwd signature)
# -----------------------------------------------------------------------------

def cacheflex_attention_fwd(
    q, k_buffer, v_buffer, o,
    qo_indptr, kv_indptr,
    req_to_token, req_pool_indices, positions,
    stride_req_b, stride_req_s,
    custom_mask, is_causal, mask_indptr, max_len_extend,
    sm_scale=None, logit_cap=0.0, sliding_window_size=-1, 
    sinks=None, window_kv_offsets=None, xai_temperature_len=-1
):
    Lq, Lv = q.shape[-1], v_buffer.shape[-1]
    BLOCK_DMODEL, _, BLOCK_DV, BLOCK_M, BLOCK_N, num_warps = \
        _get_block_sizes_for_extend_attention(Lq, Lv)
    
    sm_scale = sm_scale or 1.0 / (Lq**0.5)
    batch_size = qo_indptr.shape[0] - 1
    head_num = q.shape[1]
    kv_group_num = int(q.shape[1] // k_buffer.shape[1])
    
    q_lens = qo_indptr[1:] - qo_indptr[:-1]
    real_max_q = torch.max(q_lens).item()
    grid = (batch_size, head_num, triton.cdiv(real_max_q, BLOCK_M))
    
    # Dummy handlers for optional args
    if custom_mask is None: custom_mask = q
    if mask_indptr is None: mask_indptr = q
    if sinks is None: sinks = q
    USE_CUSTOM_MASK = False

    extra_kargs = {}
    if _is_hip:
        extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}

    _cacheflex_fwd_kernel[grid](
        q, o, k_buffer, v_buffer,
        qo_indptr, kv_indptr,
        req_to_token, req_pool_indices, positions,
        
        q.stride(0), q.stride(1),
        o.stride(0), o.stride(1),
        k_buffer.stride(0), k_buffer.stride(1),
        v_buffer.stride(0), v_buffer.stride(1),
        stride_req_b, stride_req_s,
        
        float(sm_scale), 
        int(kv_group_num), 
        float(logit_cap), 
        
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL, BLOCK_DV=BLOCK_DV,
        Lq=Lq, Lv=Lv,
        IS_CAUSAL=is_causal,
        USE_CUSTOM_MASK=USE_CUSTOM_MASK,
        HAS_SINK=False,
        num_warps=num_warps,
        num_stages=1,
        **extra_kargs
    )


# =============================================================================
# Contiguous KV Attention: direct memory access without page table
# =============================================================================

@triton.jit
def _cacheflex_contiguous_fwd_kernel(
    # --- Tensor Pointers ---
    Q_Sparse,           # [Total_Sparse_Tokens, H, D]
    O_Sparse,           # [Total_Sparse_Tokens, H, D]
    K_Contiguous,       # [Total_KV_Tokens, H_kv, D] - Contiguous KV buffer
    V_Contiguous,       # [Total_KV_Tokens, H_kv, D]

    # --- Metadata Pointers ---
    qo_indptr,          # [Batch+1] Sparse Q Ranges
    kv_indptr,          # [Batch+1] Full KV Ranges (Logic Length)
    Positions_Ptr,      # [Total_Sparse_Tokens] Real Positions for Q

    # --- Strides ---
    stride_q_tok, stride_q_head,
    stride_o_tok, stride_o_head,
    stride_k_tok, stride_k_head,
    stride_v_tok, stride_v_head,

    # --- Params ---
    sm_scale,
    kv_group_num,
    logit_cap,

    # --- Constants (Kwargs) ---
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr, BLOCK_DV: tl.constexpr,
    Lq: tl.constexpr, Lv: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    # -----------------------------------------------------------
    # 1. Setup & Ranges
    # -----------------------------------------------------------
    cur_seq = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_block_m = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    # Q Ranges (Sparse)
    cur_seq_q_start_idx = tl.load(qo_indptr + cur_seq)
    cur_seq_q_len = tl.load(qo_indptr + cur_seq + 1) - cur_seq_q_start_idx

    # KV Ranges (Full)
    cur_seq_kv_start_idx = tl.load(kv_indptr + cur_seq)
    cur_seq_kv_len = tl.load(kv_indptr + cur_seq + 1) - cur_seq_kv_start_idx

    # -----------------------------------------------------------
    # 2. Load Q
    # -----------------------------------------------------------
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = tl.arange(0, BLOCK_M)
    mask_m = (cur_block_m * BLOCK_M + offs_m) < cur_seq_q_len
    mask_d = offs_d < Lq

    q_global_offs = cur_seq_q_start_idx + cur_block_m * BLOCK_M + offs_m
    q_real_pos = tl.load(Positions_Ptr + q_global_offs, mask=mask_m, other=0)

    off_q = (
        q_global_offs[:, None].to(tl.int64) * stride_q_tok +
        cur_head.to(tl.int64) * stride_q_head +
        offs_d[None, :]
    )
    q = tl.load(Q_Sparse + off_q, mask=(mask_m[:, None] & mask_d[None, :]), other=0.0)

    # -----------------------------------------------------------
    # 3. Attention Loop (Direct Contiguous K/V Access - NO Page Table)
    # -----------------------------------------------------------
    offs_n = tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    for start_n in range(0, cur_seq_kv_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # --- 1. Mask ---
        kv_logic_idx = start_n + offs_n
        mask_n = kv_logic_idx < cur_seq_kv_len

        # Causal Masking (same as paged kernel: position-based)
        if IS_CAUSAL:
            mask_causal = kv_logic_idx[None, :] <= q_real_pos[:, None]
        else:
            mask_causal = True
        final_mask = mask_n[None, :] & mask_causal

        # --- 2. Load K (DIRECT access, no page table) ---
        kv_cont_offset = cur_seq_kv_start_idx + kv_logic_idx
        off_k = (
            kv_cont_offset[None, :].to(tl.int64) * stride_k_tok +
            cur_kv_head.to(tl.int64) * stride_k_head +
            offs_d[:, None]
        )
        k = tl.load(K_Contiguous + off_k, mask=(mask_n[None, :] & mask_d[:, None]), other=0.0)

        # --- 3. QK Score ---
        qk = tl.dot(q, k, out_dtype=tl.float32)
        qk *= sm_scale

        if logit_cap > 0:
            qk = logit_cap * tanh(qk / logit_cap)

        qk = tl.where(final_mask, qk, float("-inf"))

        # --- 4. Online Softmax ---
        row_max = tl.max(qk, 1)
        row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
        m_i_new = tl.maximum(m_i, row_max_fixed)
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])

        # --- 5. Load V (DIRECT access) ---
        mask_dv = tl.arange(0, BLOCK_DV) < Lv
        off_v = (
            kv_cont_offset[:, None].to(tl.int64) * stride_v_tok +
            cur_kv_head.to(tl.int64) * stride_v_head +
            tl.arange(0, BLOCK_DV)[None, :]
        )
        v = tl.load(V_Contiguous + off_v, mask=(mask_n[:, None] & mask_dv[None, :]), other=0.0)

        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    # -----------------------------------------------------------
    # 4. Store Output
    # -----------------------------------------------------------
    acc = tl.where(l_i[:, None] > 0.0, acc / l_i[:, None], 0.0)

    mask_dv = tl.arange(0, BLOCK_DV) < Lv
    off_o = (
        q_global_offs[:, None].to(tl.int64) * stride_o_tok +
        cur_head.to(tl.int64) * stride_o_head +
        tl.arange(0, BLOCK_DV)[None, :]
    )
    tl.store(O_Sparse + off_o, acc.to(O_Sparse.dtype.element_ty), mask=(mask_m[:, None] & mask_dv[None, :]))


def cacheflex_contiguous_attention_fwd(
    q, k_contiguous, v_contiguous, o,
    qo_indptr, kv_indptr,
    positions,
    is_causal=True,
    sm_scale=None, logit_cap=0.0,
):
    """
    Contiguous KV attention for CacheFlex post-check layers.
    Same logic as cacheflex_attention_fwd but uses contiguous KV buffers
    instead of page table indirect access, eliminating ~34% per-token overhead.
    """
    Lq, Lv = q.shape[-1], v_contiguous.shape[-1]
    BLOCK_DMODEL, _, BLOCK_DV, BLOCK_M, BLOCK_N, num_warps = \
        _get_block_sizes_for_extend_attention(Lq, Lv)

    sm_scale = sm_scale or 1.0 / (Lq**0.5)
    batch_size = qo_indptr.shape[0] - 1
    head_num = q.shape[1]
    kv_group_num = int(q.shape[1] // k_contiguous.shape[1])

    q_lens = qo_indptr[1:] - qo_indptr[:-1]
    real_max_q = torch.max(q_lens).item()
    grid = (batch_size, head_num, triton.cdiv(real_max_q, BLOCK_M))

    extra_kargs = {}
    if _is_hip:
        extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}

    _cacheflex_contiguous_fwd_kernel[grid](
        q, o, k_contiguous, v_contiguous,
        qo_indptr, kv_indptr,
        positions,

        q.stride(0), q.stride(1),
        o.stride(0), o.stride(1),
        k_contiguous.stride(0), k_contiguous.stride(1),
        v_contiguous.stride(0), v_contiguous.stride(1),

        float(sm_scale),
        int(kv_group_num),
        float(logit_cap),

        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL, BLOCK_DV=BLOCK_DV,
        Lq=Lq, Lv=Lv,
        IS_CAUSAL=is_causal,
        num_warps=num_warps,
        num_stages=1,
        **extra_kargs
    )


# =============================================================================
# KV-Sparse: 稀疏 KV Attention Kernel
# =============================================================================

@triton.jit
def _cacheflex_fwd_kernel_kvsparse(
    # --- Tensor Pointers ---
    Q_Sparse,           # [Total_Sparse_Tokens, H, D]
    O_Sparse,           # [Total_Sparse_Tokens, H, D]
    K_Buffer,           # [Pool_Size, H, D] - Full Physical Memory
    V_Buffer,           # [Pool_Size, H, D]

    # --- Metadata Pointers ---
    qo_indptr,          # [Batch+1] Sparse Q Ranges
    kv_indptr,          # [Batch+1] Full KV Ranges (Logic Length) - NOT USED for KV-Sparse!
    KV_Sparse_Indptr,   # [Batch+1] Sparse KV Ranges (NEW!)
    KV_Sparse_Indices,  # [Total_Sparse_KV] Physical KV indices (NEW!)
    KV_Sparse_Pos,      # [Total_Sparse_KV] Real positions for sparse KV (NEW!)
    Req_To_Token_Ptr,   # [Max_Reqs, Max_Context] Physical Page Table
    Req_Pool_Indices,   # [B] Request IDs
    Positions_Ptr,      # [Total_Sparse_Tokens] Real Positions for Q

    # --- Strides ---
    stride_q_tok, stride_q_head,      # Q Sparse
    stride_o_tok, stride_o_head,      # O Sparse
    stride_buf_k_tok, stride_buf_k_head, # Pool (Physical)
    stride_buf_v_tok, stride_buf_v_head,
    stride_req_b, stride_req_s,       # Page Table

    # --- Params ---
    sm_scale,
    kv_group_num,
    logit_cap,

    # --- Constants (Kwargs) ---
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr, BLOCK_DV: tl.constexpr,
    Lq: tl.constexpr, Lv: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    USE_CUSTOM_MASK: tl.constexpr,
    HAS_SINK: tl.constexpr
):
    """
    KV-Sparse CacheFlex Attention Kernel

    关键区别：不再通过 req_to_token 查全量 KV，直接使用稀疏索引。
    这允许跳过不重要的视觉 token，只计算 Q 对稀疏 KV 的 attention。
    """
    # -----------------------------------------------------------
    # 1. Setup & Ranges
    # -----------------------------------------------------------
    cur_seq = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_block_m = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num
    cur_req_idx = tl.load(Req_Pool_Indices + cur_seq)

    # Q Ranges (Sparse)
    cur_seq_q_start_idx = tl.load(qo_indptr + cur_seq)
    cur_seq_q_len = tl.load(qo_indptr + cur_seq + 1) - cur_seq_q_start_idx

    # KV Ranges (Sparse - NOT Full!)
    cur_seq_kv_start_idx = tl.load(KV_Sparse_Indptr + cur_seq)
    cur_seq_kv_len = tl.load(KV_Sparse_Indptr + cur_seq + 1) - cur_seq_kv_start_idx

    # -----------------------------------------------------------
    # 2. Load Q
    # -----------------------------------------------------------
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = tl.arange(0, BLOCK_M)
    mask_m = (cur_block_m * BLOCK_M + offs_m) < cur_seq_q_len
    mask_d = offs_d < Lq

    # Global Q Offset
    q_global_offs = cur_seq_q_start_idx + cur_block_m * BLOCK_M + offs_m

    # Load Q Real Position (For Causal Mask)
    q_real_pos = tl.load(Positions_Ptr + q_global_offs, mask=mask_m, other=0)

    # Load Q Tensor
    off_q = (
        q_global_offs[:, None].to(tl.int64) * stride_q_tok +
        cur_head.to(tl.int64) * stride_q_head +
        offs_d[None, :]
    )
    q = tl.load(Q_Sparse + off_q, mask=(mask_m[:, None] & mask_d[None, :]), other=0.0)

    # -----------------------------------------------------------
    # 3. Attention Loop (Iterate over Sparse KV via Direct Indices)
    # -----------------------------------------------------------
    offs_n = tl.arange(0, BLOCK_N)

    # Initialize accumulators
    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    for start_n in range(0, cur_seq_kv_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # --- 1. 计算稀疏 KV 的局部索引 ---
        kv_local_idx = start_n + offs_n
        mask_n = kv_local_idx < cur_seq_kv_len

        # --- 2. Causal Masking ---
        if IS_CAUSAL:
            # 加载稀疏 KV 的真实位置
            kv_sparse_pos_idx = cur_seq_kv_start_idx + kv_local_idx
            kv_real_pos = tl.load(KV_Sparse_Pos + kv_sparse_pos_idx, mask=mask_n, other=0)
            mask_causal = kv_real_pos[None, :] <= q_real_pos[:, None]
        else:
            mask_causal = True

        final_mask = mask_n[None, :] & mask_causal

        # --- 3. 加载 K (直接使用稀疏索引，不查 Page Table) ---
        # 全局索引 in sparse_indices
        kv_global_idx = cur_seq_kv_start_idx + kv_local_idx

        # 直接获取物理 KV 索引
        kv_phys_idx = tl.load(KV_Sparse_Indices + kv_global_idx, mask=mask_n, other=0)

        # 计算 K Buffer 偏移 (转置加载 [D, N] 以优化 dot)
        off_k = (
            kv_phys_idx[None, :].to(tl.int64) * stride_buf_k_tok +
            cur_kv_head.to(tl.int64) * stride_buf_k_head +
            offs_d[:, None]
        )
        k = tl.load(K_Buffer + off_k, mask=(mask_n[None, :] & mask_d[:, None]), other=0.0)

        # --- 4. 计算 QK Score ---
        qk = tl.dot(q, k, out_dtype=tl.float32)
        qk *= sm_scale

        if logit_cap > 0:
            qk = logit_cap * tanh(qk / logit_cap)

        # 应用 Mask
        qk = tl.where(final_mask, qk, float("-inf"))

        # --- 5. Online Softmax 更新 (标准 SGLang 逻辑) ---
        row_max = tl.max(qk, 1)
        row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
        m_i_new = tl.maximum(m_i, row_max_fixed)

        # 计算旧累加值的缩放因子 (alpha)
        alpha = tl.exp(m_i - m_i_new)

        # 计算当前块的 P (概率值)
        p = tl.exp(qk - m_i_new[:, None])

        # --- 6. 加载 V (使用相同物理 KV 索引) ---
        off_v = (
            kv_phys_idx[:, None].to(tl.int64) * stride_buf_v_tok +
            cur_kv_head.to(tl.int64) * stride_buf_v_head +
            tl.arange(0, BLOCK_DV)[None, :]
        )
        mask_dv = tl.arange(0, BLOCK_DV) < Lv
        v = tl.load(V_Buffer + off_v, mask=(mask_n[:, None] & mask_dv[None, :]), other=0.0)

        # --- 7. 累加结果 ---
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    # -----------------------------------------------------------
    # 4. 归一化与输出
    # -----------------------------------------------------------
    acc = tl.where(l_i[:, None] > 0.0, acc / l_i[:, None], 0.0)

    mask_dv = tl.arange(0, BLOCK_DV) < Lv
    off_o = (
        q_global_offs[:, None].to(tl.int64) * stride_o_tok +
        cur_head.to(tl.int64) * stride_o_head +
        tl.arange(0, BLOCK_DV)[None, :]
    )

    tl.store(O_Sparse + off_o, acc.to(O_Sparse.dtype.element_ty),
            mask=(mask_m[:, None] & mask_dv[None, :]))



def cacheflex_attention_fwd_kvsparse(
    q, k_buffer, v_buffer, o,
    qo_indptr, kv_indptr,
    req_to_token, req_pool_indices, positions,
    stride_req_b, stride_req_s,
    custom_mask, is_causal, mask_indptr, max_len_extend,
    kv_sparse_indptr=None, kv_sparse_indices=None, kv_sparse_pos=None,
    sm_scale=None, logit_cap=0.0, sliding_window_size=-1,
    sinks=None, window_kv_offsets=None, xai_temperature_len=-1
):
    """
    KV-Sparse CacheFlex Attention Wrapper

    使用稀疏 KV 索引 (kv_sparse_indices, kv_sparse_pos, kv_sparse_indptr)
    而不是通过 req_to_token 查询全量 KV。
    """
    # --- diagnostics: CacheFlex KV-Sparse wrapper ---
    # print once to avoid debug overhead
    import os
    KV_SPARSE_DEBUG = os.environ.get("KV_SPARSE_DEBUG", "0") == "1"
    if KV_SPARSE_DEBUG:
        # 使用静态变量确保只打印一次
        if not hasattr(cacheflex_attention_fwd_kvsparse, "_printed_once"):
            q_len = q.shape[0]
            has_sparse = (
                kv_sparse_indptr is not None
                and kv_sparse_indices is not None
                and kv_sparse_pos is not None
            )
            sparse_info = {
                "q_len": q_len,
                "kv_sparse_indices_numel": kv_sparse_indices.numel() if kv_sparse_indices is not None else 0,
                "kv_sparse_indptr_numel": kv_sparse_indptr.numel() if kv_sparse_indptr is not None else 0,
                "has_sparse": has_sparse,
            }
            print(f"[KV-Sparse DIAG] cacheflex_attention_fwd_kvsparse called: {sparse_info}")
            cacheflex_attention_fwd_kvsparse._printed_once = True
    # --- END DIAG ---

    # 完整实现 KV-sparse kernel 调用

    # 如果没有稀疏索引，回退到标准 kernel
    if kv_sparse_indptr is None or kv_sparse_indices is None or kv_sparse_pos is None:
        from sglang.srt.layers.attention.triton_ops.cacheflex_extend_attention import cacheflex_attention_fwd
        return cacheflex_attention_fwd(
            q, k_buffer, v_buffer, o,
            qo_indptr, kv_indptr,
            req_to_token, req_pool_indices, positions,
            stride_req_b, stride_req_s,
            custom_mask, is_causal, mask_indptr, max_len_extend,
            sm_scale, logit_cap, sliding_window_size,
            sinks, window_kv_offsets, xai_temperature_len
        )

    Lq, Lv = q.shape[-1], v_buffer.shape[-1]
    BLOCK_DMODEL, _, BLOCK_DV, BLOCK_M, BLOCK_N, num_warps = \
        _get_block_sizes_for_extend_attention(Lq, Lv)

    sm_scale = sm_scale or 1.0 / (Lq**0.5)
    batch_size = qo_indptr.shape[0] - 1
    head_num = q.shape[1]
    kv_group_num = int(q.shape[1] // k_buffer.shape[1])

    q_lens = qo_indptr[1:] - qo_indptr[:-1]
    real_max_q = torch.max(q_lens).item()
    grid = (batch_size, head_num, triton.cdiv(real_max_q, BLOCK_M))

    # Dummy handlers for optional args
    if custom_mask is None:
        custom_mask = q
    if mask_indptr is None:
        mask_indptr = q
    if sinks is None:
        sinks = q

    extra_kargs = {}
    if _is_hip:
        extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}

    # debug print
    import os
    should_print = os.environ.get("KV_SPARSE_DEBUG_PRINT", "0") == "1"
    if should_print and kv_sparse_indices is not None:
        print(f"\n===== KV-Sparse Debug Info =====")
        print(f"[Shape] q.shape={q.shape}, k_buffer.shape={k_buffer.shape}")
        print(f"[Batch] batch_size={batch_size}, qo_indptr={qo_indptr.tolist()}")
        print(f"[Q] seq_lens={(qo_indptr[1:] - qo_indptr[:-1]).tolist()}")
        print(f"[KV sparse] kv_sparse_indptr={kv_sparse_indptr.tolist()}")
        print(f"[KV sparse] len(kv_sparse_indices)={len(kv_sparse_indices)}")
        print(f"[KV sparse] kv_sparse_indices.min()/max()={kv_sparse_indices.min().item()}/{kv_sparse_indices.max().item()}")
        print(f"[KV sparse] has_negative={(kv_sparse_indices < 0).any()}")
        print(f"[Positions Q] forward_batch.positions[:20]={positions[:20] if len(positions) >= 20 else positions.tolist()}")
        print(f"[Positions Q] min/max={positions.min().item()}/{positions.max().item()}")
        print(f"[Positions KV] kv_sparse_pos[:20]={kv_sparse_pos[:20] if len(kv_sparse_pos) >= 20 else kv_sparse_pos.tolist()}")
        print(f"[Positions KV] min/max={kv_sparse_pos.min().item()}/{kv_sparse_pos.max().item()}")
        print(f"=====================================================\n")
    # --- END DEBUG PRINT ---

    # 调用 KV-sparse kernel
    _cacheflex_fwd_kernel_kvsparse[grid](
        q, o, k_buffer, v_buffer,
        qo_indptr, kv_indptr,
        kv_sparse_indptr, kv_sparse_indices, kv_sparse_pos,  # 稀疏 KV 参数
        req_to_token, req_pool_indices, positions,

        q.stride(0), q.stride(1),
        o.stride(0), o.stride(1),
        k_buffer.stride(0), k_buffer.stride(1),
        v_buffer.stride(0), v_buffer.stride(1),
        stride_req_b, stride_req_s,

        float(sm_scale),
        int(kv_group_num),
        float(logit_cap),

        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL, BLOCK_DV=BLOCK_DV,
        Lq=Lq, Lv=Lv,
        IS_CAUSAL=is_causal,
        USE_CUSTOM_MASK=False,
        HAS_SINK=False,
        num_warps=num_warps,
        num_stages=1,
        **extra_kargs
    )
# --- END KV-Sparse ---
# =============================================================================


def redundant_attention(
    q_extend,
    o_extend,
    k_buffer,
    v_buffer,
    b_req_idx,
    b_start_loc,
    b_seq_len,
    b_seq_len_prefix,
    max_len_in_batch,
):
    total_token_num = k_buffer.shape[0]
    B, H_Q, D = b_req_idx.shape[0], q_extend.shape[-2], q_extend.shape[-1]
    q_buffer = torch.empty(
        (total_token_num, H_Q, D), dtype=q_extend.dtype, device=q_extend.device
    )

    pt = 0
    for i in range(B):
        cur_seq_len_extend = b_seq_len[i] - b_seq_len_prefix[i]
        pl, pr = b_start_loc[i] + b_seq_len_prefix[i], b_start_loc[i] + b_seq_len[i]
        q_buffer[pl:pr] = q_extend[pt : pt + cur_seq_len_extend]
        pt += cur_seq_len_extend

    o_buffer = torch.empty_like(q_buffer)
    context_attention_fwd(
        q_buffer, k_buffer, v_buffer, o_buffer, b_start_loc, b_seq_len, max_len_in_batch
    )

    pt = 0
    for i in range(B):
        cur_seq_len_extend = b_seq_len[i] - b_seq_len_prefix[i]
        pl, pr = b_start_loc[i] + b_seq_len_prefix[i], b_start_loc[i] + b_seq_len[i]
        o_extend[pt : pt + cur_seq_len_extend] = o_buffer[pl:pr]
        pt += cur_seq_len_extend


@triton.jit
def _fwd_kernel_unified(
    Q,
    O,
    K_Buffer,
    V_Buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    prefix_lens,
    mask_ptr,
    mask_indptr,
    sink_ptr,
    window_start_pos,
    sm_scale,
    kv_group_num,
    stride_qbs,
    stride_qh,
    stride_obs,
    stride_oh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    SLIDING_WINDOW_SIZE: tl.constexpr,
    logit_cap: tl.constexpr,
    xai_temperature_len: tl.constexpr,
    Lq: tl.constexpr,
    Lv: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    USE_CUSTOM_MASK: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    """
    Unified 1-stage kernel for deterministic extend attention.
    Both prefix and extend KV are accessed through the unified kv_indices.
    """
    cur_seq = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_block_m = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num

    # Load sequence information
    cur_seq_q_start_idx = tl.load(qo_indptr + cur_seq)
    cur_seq_q_len = tl.load(qo_indptr + cur_seq + 1) - cur_seq_q_start_idx
    cur_seq_kv_start_idx = tl.load(kv_indptr + cur_seq)
    cur_seq_kv_len = tl.load(kv_indptr + cur_seq + 1) - cur_seq_kv_start_idx
    cur_seq_prefix_len = tl.load(prefix_lens + cur_seq)

    # Load window start position for sliding window attention
    # This is the absolute position of the first key in the window (0 if no sliding window)
    cur_window_start = 0
    if SLIDING_WINDOW_SIZE > 0:
        cur_window_start = tl.load(window_start_pos + cur_seq)

    # Load custom mask start index if using custom mask (for speculative decoding)
    if USE_CUSTOM_MASK:
        cur_seq_mask_start_idx = tl.load(mask_indptr + cur_seq)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_m = tl.arange(0, BLOCK_M)
    mask_m = (cur_block_m * BLOCK_M + offs_m) < cur_seq_q_len
    mask_d = offs_d < Lq
    mask_dv = offs_dv < Lv

    # XAI temperature handling
    if xai_temperature_len > 0:
        offs_qidx = cur_seq_prefix_len + cur_block_m * BLOCK_M + offs_m
        xai_temperature_reg = tl.where(
            offs_qidx < xai_temperature_len,
            1.0,
            xai_temperature_len / (offs_qidx + 1.0),
        )

    # Load Q
    offs_q = (
        (cur_seq_q_start_idx + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(Q + offs_q, mask=(mask_m[:, None]) & (mask_d[None, :]), other=0.0)

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        offs_qpe = (
            (cur_seq_q_start_idx + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_qbs
            + cur_head * stride_qh
            + offs_dpe[None, :]
        )
        qpe = tl.load(Q + offs_qpe, mask=mask_m[:, None], other=0.0)

    # Initialize accumulators
    offs_n = tl.arange(0, BLOCK_N)
    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)
    deno = tl.zeros([BLOCK_M], dtype=tl.float32)
    e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    # Unified loop: process all KV tokens (prefix + extend)
    for start_n in range(0, cur_seq_kv_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < cur_seq_kv_len

        # Compute mask
        final_mask = mask_m[:, None] & mask_n[None, :]

        # Apply custom mask if provided
        if USE_CUSTOM_MASK:
            custom_mask = tl.load(
                mask_ptr
                + cur_seq_mask_start_idx
                + (cur_block_m * BLOCK_M + offs_m[:, None]) * cur_seq_kv_len
                + start_n
                + offs_n[None, :],
                mask=(mask_m[:, None] & mask_n[None, :]),
                other=0,
            )
            final_mask &= custom_mask

        # Apply causal mask for extend part
        if IS_CAUSAL and not USE_CUSTOM_MASK:
            # Determine if current KV block is in extend region
            # Only apply causal mask when both Q and K are in extend region
            q_idx = cur_block_m * BLOCK_M + offs_m[:, None]
            k_idx_in_total = start_n + offs_n[None, :]

            # Causal mask: q_idx >= (k_idx - prefix_len) when k_idx >= prefix_len
            # For prefix region (k_idx < prefix_len), no causal mask
            k_is_extend = k_idx_in_total >= cur_seq_prefix_len
            k_idx_in_extend = k_idx_in_total - cur_seq_prefix_len
            causal_mask = tl.where(
                k_is_extend,
                q_idx >= k_idx_in_extend,
                True,  # No causal mask for prefix
            )
            final_mask &= causal_mask

        if SLIDING_WINDOW_SIZE > 0:
            # Sliding window mask with correct absolute positions
            # Q absolute position: window_start + prefix_len + q_position_in_extend
            q_abs_pos = (
                cur_window_start
                + cur_seq_prefix_len
                + cur_block_m * BLOCK_M
                + offs_m[:, None]
            )

            # K absolute position: window_start + k_index_in_unified_array
            k_abs_pos = cur_window_start + start_n + offs_n[None, :]

            # Sliding window: query can attend to keys within window_size
            window_mask = q_abs_pos <= (k_abs_pos + SLIDING_WINDOW_SIZE)
            final_mask &= window_mask

        # Check if we can skip this tile
        SKIP_TILE = False
        if USE_CUSTOM_MASK or SLIDING_WINDOW_SIZE > 0:
            SKIP_TILE = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0

        if not SKIP_TILE:
            # Load KV indices
            offs_kv_loc = tl.load(
                kv_indices + cur_seq_kv_start_idx + start_n + offs_n,
                mask=mask_n,
                other=0,
            )

            # Load K
            offs_buf_k = (
                offs_kv_loc[None, :] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_d[:, None]
            )
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(mask_n[None, :]) & (mask_d[:, None]),
                other=0.0,
            )

            # Compute QK
            qk = tl.dot(q.to(k.dtype), k)
            if BLOCK_DPE > 0:
                offs_kpe = (
                    offs_kv_loc[None, :] * stride_buf_kbs
                    + cur_kv_head * stride_buf_kh
                    + offs_dpe[:, None]
                )
                kpe = tl.load(
                    K_Buffer + offs_kpe,
                    mask=mask_n[None, :],
                    other=0.0,
                )
                qk += tl.dot(qpe.to(kpe.dtype), kpe)

            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            if xai_temperature_len > 0:
                qk *= xai_temperature_reg[:, None]

            qk = tl.where(final_mask, qk, float("-inf"))

            # Online softmax
            row_max = tl.max(qk, 1)
            row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
            n_e_max = tl.maximum(row_max_fixed, e_max)

            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            deno = deno * re_scale + tl.sum(p, 1)

            # Load V
            offs_buf_v = (
                offs_kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_dv[None, :]
            )
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=mask_n[:, None] & mask_dv[None, :],
                other=0.0,
            )
            p = p.to(v.dtype)
            acc = acc * re_scale[:, None] + tl.dot(p, v)

            e_max = n_e_max

    # Handle sink tokens
    if HAS_SINK:
        cur_sink = tl.load(sink_ptr + cur_head)
        deno += tl.exp(cur_sink - e_max)

    # Store output
    offs_o = (
        (cur_seq_q_start_idx + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_dv[None, :]
    )
    tl.store(
        O + offs_o,
        acc / deno[:, None],
        mask=mask_m[:, None] & mask_dv[None, :],
    )


def extend_attention_fwd_unified(
    q,
    o,
    k_buffer,
    v_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    prefix_lens,
    max_len_extend,
    custom_mask=None,
    mask_indptr=None,
    sm_scale=None,
    logit_cap=0.0,
    is_causal=True,
    sliding_window_size=-1,
    sinks=None,
    window_start_pos=None,
    xai_temperature_len=-1,
):
    """
    Unified 1-stage extend attention for deterministic inference.

    Args:
        q: Query tensor [num_tokens, num_heads, head_dim]
        o: Output tensor [num_tokens, num_heads, head_dim]
        k_buffer: Key cache buffer
        v_buffer: Value cache buffer
        qo_indptr: Query offsets [batch_size + 1]
        kv_indptr: KV offsets [batch_size + 1] (includes both prefix and extend)
        kv_indices: Unified KV indices (both prefix and extend)
        prefix_lens: Prefix length for each sequence [batch_size]
        max_len_extend: Maximum extend length
        custom_mask: Custom attention mask (for speculative decoding tree attention)
        mask_indptr: Mask offsets [batch_size + 1]
        sm_scale: Softmax scale
        logit_cap: Logit capping value
        is_causal: Whether to apply causal mask
        sliding_window_size: Sliding window size (-1 for no sliding window)
        sinks: Sink tokens
        window_start_pos: Absolute position of first key in sliding window [batch_size]
                         (None if sliding window not used)
        xai_temperature_len: XAI temperature length
    """
    Lq, Lv = q.shape[-1], v_buffer.shape[-1]

    # Get block sizes and configuration
    BLOCK_DMODEL, BLOCK_DPE, BLOCK_DV, BLOCK_M, BLOCK_N, num_warps = (
        _get_block_sizes_for_extend_attention(Lq, Lv)
    )

    sm_scale = sm_scale or 1.0 / (Lq**0.5)
    batch_size, head_num = qo_indptr.shape[0] - 1, q.shape[1]
    kv_group_num = q.shape[1] // k_buffer.shape[1]

    USE_CUSTOM_MASK = custom_mask is not None
    HAS_SINK = sinks is not None

    # For sliding window attention, window_start_pos tracks the absolute position
    # of the first key in each sequence's window
    if sliding_window_size > 0 and window_start_pos is None:
        # If not provided, assume window starts at position 0
        window_start_pos = torch.zeros(batch_size, dtype=torch.int32, device=q.device)

    grid = (batch_size, head_num, triton.cdiv(max_len_extend, BLOCK_M))
    num_stages = 1

    extra_kargs = {}
    if _is_hip:
        extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}

    _fwd_kernel_unified[grid](
        q,
        o,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        prefix_lens,
        custom_mask,
        mask_indptr,
        sinks,
        window_start_pos,
        sm_scale,
        kv_group_num,
        q.stride(0),
        q.stride(1),
        o.stride(0),
        o.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        SLIDING_WINDOW_SIZE=sliding_window_size,
        logit_cap=logit_cap,
        xai_temperature_len=xai_temperature_len,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        Lq=Lq,
        Lv=Lv,
        IS_CAUSAL=is_causal,
        USE_CUSTOM_MASK=USE_CUSTOM_MASK,
        HAS_SINK=HAS_SINK,
        num_warps=num_warps,
        num_stages=num_stages,
        **extra_kargs,
    )
