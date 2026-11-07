# HACK(Shizhuo) New
"""
Shared utilities for CacheFlex contiguous KV attention path.

These functions gather full KV from the paged cache into contiguous tensors
(at check_layer+1) and update them per-layer (scatter recomputed K/V).
Used by both llama.py and qwen2.py to avoid code duplication.
"""

import os
import time
import torch
import logging

# unified profile control
from sglang.srt.utils import cacheflex_dump as _cf_dump

logger = logging.getLogger(__name__)


# ============================================================================
# [Mem Profile] Read-only GPU memory snapshot at a checkpoint.
# Gated by CACHEFLEX_MEM_PROFILE=1.
#
# IMPORTANT: This helper does NOT call torch.cuda.reset_peak_memory_stats().
# An earlier version did, which (empirically) caused between-sample deadlocks
# in sglang's scheduler ("Cache not flushed because pending requests" → hang).
# Likely root cause: reset_peak_memory_stats() races with the scheduler's
# concurrent KV-pool allocator access. Now we only READ stats; no writes.
#
# Semantics change vs reset-based version:
#   peak_gb is CUMULATIVE since process start (or sglang's last internal reset).
#   To get per-segment peak GROWTH: compute delta(peak[t+1], peak[t]).
#     - delta > 0 → new high water during segment t→t+1
#     - delta == 0 → no growth; segment did not push peak past prior high
#   cur_gb is precise (current allocated at this checkpoint).
#
# For LRU-accumulation diagnostic: compare entry_cur_gb across samples.
# For clone-spike diagnostic: compare after_clone_cur_gb − after_gather_cur_gb.
def cacheflex_mem_snapshot(forward_batch, tag: str):
    if not _cf_dump.profile_mem_enabled():
        return
    if not hasattr(forward_batch, "_cacheflex_mem_profile") or forward_batch._cacheflex_mem_profile is None:
        forward_batch._cacheflex_mem_profile = {}
    forward_batch._cacheflex_mem_profile[f"{tag}_peak_gb"] = (
        torch.cuda.max_memory_allocated() / 1e9
    )
    forward_batch._cacheflex_mem_profile[f"{tag}_cur_gb"] = (
        torch.cuda.memory_allocated() / 1e9
    )
# ============================================================================


def gather_contiguous_kv(forward_batch, start_layer: int, end_layer: int):
    """
    Gather full KV from paged cache into contiguous tensors, once.

    Called at check_layer+1, after _apply_post_check_slicing.
    Stores per-layer contiguous KV in forward_batch.cacheflex_contiguous_kv.

    Args:
        forward_batch: ForwardBatch with cacheflex_total_kv_lens, req_pool_indices, etc.
        start_layer: First layer to gather (typically check_layer+1)
        end_layer: Last layer (exclusive), typically self.end_layer
    """
    _do_profile = _cf_dump.profile_timing_enabled()
    if _do_profile:
        torch.cuda.synchronize()
        _gather_t0 = time.perf_counter()

    total_kv_lens = forward_batch.cacheflex_total_kv_lens  # [bs]
    bs = forward_batch.batch_size

    # Build cu_seq_lens for contiguous KV
    cu_seq_lens = torch.zeros(bs + 1, dtype=torch.int32, device=total_kv_lens.device)
    cu_seq_lens[1:] = torch.cumsum(total_kv_lens, dim=0)
    forward_batch.cacheflex_contiguous_cu_seq_lens = cu_seq_lens

    total_kv_tokens = cu_seq_lens[-1].item()
    req_to_token = forward_batch.req_to_token_pool.req_to_token

    logger.info(
        f"[CacheFlex Contiguous] Gather: bs={bs}, total_kv_tokens={total_kv_tokens}, "
        f"total_kv_lens={total_kv_lens.tolist()}, layers=[{start_layer},{end_layer})"
    )

    # Build flat per-token mapping for vectorized gather (zero .item() in loop)
    # req_idx_per_token[i] = which request token i belongs to
    req_idx_per_token = torch.repeat_interleave(
        torch.arange(bs, device=total_kv_lens.device, dtype=torch.long),
        total_kv_lens.to(torch.long)
    )
    # kv_bases = starting offset of each request in cont buffer
    kv_bases = cu_seq_lens[:-1]  # [bs]
    # Intra-request offset: 0, 1, ..., kv_len-1 for each request
    intra_offsets = torch.arange(total_kv_tokens, device=total_kv_lens.device, dtype=torch.long)
    cum_bases = torch.repeat_interleave(kv_bases, total_kv_lens.to(torch.long))
    intra_req_offset = intra_offsets - cum_bases
    # Build flat physical indices via 2D advanced indexing
    req_pool_indices = forward_batch.req_pool_indices  # [bs]
    flat_pool_idx = req_pool_indices[req_idx_per_token]  # [total_kv_tokens]
    phys_indices = req_to_token[flat_pool_idx, intra_req_offset].to(torch.long)

    # Gather per-layer
    contiguous_kv = {}
    for layer_idx in range(start_layer, end_layer):
        k_buf = forward_batch.token_to_kv_pool.get_key_buffer(layer_idx)
        v_buf = forward_batch.token_to_kv_pool.get_value_buffer(layer_idx)
        num_kv_heads = k_buf.shape[1]
        head_dim = k_buf.shape[2]

        k_cont = torch.empty(
            (total_kv_tokens, num_kv_heads, head_dim),
            dtype=k_buf.dtype, device=k_buf.device,
        )
        v_cont = torch.empty(
            (total_kv_tokens, num_kv_heads, head_dim),
            dtype=v_buf.dtype, device=v_buf.device,
        )

        k_cont[:] = k_buf[phys_indices]
        v_cont[:] = v_buf[phys_indices]
        contiguous_kv[layer_idx] = (k_cont, v_cont)

    forward_batch.cacheflex_contiguous_kv = contiguous_kv

    if _do_profile:
        torch.cuda.synchronize()
        _gather_ms = (time.perf_counter() - _gather_t0) * 1000
        if not hasattr(forward_batch, "_cacheflex_copy_profile"):
            forward_batch._cacheflex_copy_profile = []
        forward_batch._cacheflex_copy_profile.append({
            "layer": -1,
            "copy_ms": 0.0,
            "sync_ms": 0.0,
            "gather_ms": _gather_ms,
        })

    logger.info(
        f"[CacheFlex Contiguous] Gathered {len(contiguous_kv)} layers × "
        f"{total_kv_tokens} KV tokens = "
        f"{total_kv_tokens * len(contiguous_kv) * 2 * k_buf.shape[1] * k_buf.shape[2] * 2 / 1e9:.2f} GB"
    )


def update_contiguous_kv(k, v, layer_id, forward_batch, tp_k_head_num, tp_v_head_num,
                          qk_head_dim, v_head_dim):
    """
    Scatter recomputed K/V into the contiguous KV buffer.

    Called after set_kv_buffer writes to the paged cache, to also
    update the contiguous buffer at the correct logical positions.

    Args:
        k: newly computed K tensor [num_tokens, tp_k_head_num * qk_head_dim]
        v: newly computed V tensor [num_tokens, tp_v_head_num * v_head_dim]
        layer_id: current layer index
        forward_batch: ForwardBatch
        tp_k_head_num, tp_v_head_num: number of KV heads after TP
        qk_head_dim, v_head_dim: head dimensions
    """
    contiguous_kv = forward_batch.cacheflex_contiguous_kv
    if contiguous_kv is None or layer_id not in contiguous_kv:
        return

    k_cont, v_cont = contiguous_kv[layer_id]
    cu_seq_lens = forward_batch.cacheflex_contiguous_cu_seq_lens
    positions = forward_batch.positions
    seq_lens = forward_batch.seq_lens  # sliced (recompute token count per req)

    k_view = k.view(-1, tp_k_head_num, qk_head_dim)
    v_view = v.view(-1, tp_v_head_num, v_head_dim)

    # Vectorized scatter: build per-token request index, then compute
    # cont_offsets = kv_bases[req_idx] + positions all on GPU.
    # This replaces the Python loop + .item() synchronization.
    kv_bases = cu_seq_lens[:-1]  # [bs], starting offset of each request in cont buffer

    req_idx_per_token = torch.repeat_interleave(
        torch.arange(seq_lens.shape[0], device=seq_lens.device, dtype=torch.long),
        seq_lens.to(torch.long)
    )

    cont_offsets = kv_bases[req_idx_per_token] + positions

    # DEBUG: verify vectorized offsets match loop-based offsets
    if os.environ.get("SGLANG_CACHEFLEX_VERIFY_SCATTER") == "1":
        start_loc = forward_batch.extend_start_loc
        # Verify req_idx_per_token: for each token, check that
        # start_loc[req_idx] <= token_idx < start_loc[req_idx] + seq_lens[req_idx]
        token_idx = torch.arange(positions.shape[0], device=positions.device)
        expected_start = start_loc[req_idx_per_token]
        expected_end = expected_start + seq_lens[req_idx_per_token]
        assert ((token_idx >= expected_start) & (token_idx < expected_end)).all(), \
            f"req_idx_per_token mismatch: some tokens assigned to wrong request"

        # Verify cont_offsets match loop-based offsets
        loop_offsets = []
        for req_idx in range(forward_batch.batch_size):
            req_start = start_loc[req_idx].item()
            req_len = seq_lens[req_idx].item()
            if req_len == 0:
                continue
            kv_base = kv_bases[req_idx].item()
            req_positions = positions[req_start : req_start + req_len]
            loop_offsets.append(kv_base + req_positions)
        loop_offsets = torch.cat(loop_offsets)
        vectorized_offsets = kv_bases[req_idx_per_token] + positions
        assert torch.equal(loop_offsets.sort()[0], vectorized_offsets.sort()[0]), \
            f"cont_offsets mismatch: loop vs vectorized"

    k_cont[cont_offsets] = k_view
    v_cont[cont_offsets] = v_view


def sync_complementary_copy_to_contiguous(forward_batch, slice_info, layer_id):
    """
    Sync complementary_copy writes from paged cache to contiguous buffer.

    complementary_copy writes Delta-RoPE adjusted K/V to paged cache at
    dst_tail_loc (physical indices). We must copy these updated values
    into the contiguous buffer at the corresponding logical positions,
    otherwise the contiguous buffer contains stale KV without Delta-RoPE.

    Called in radix_attention.py after cacheflex_complementary_copy_launcher_kernel.

    Args:
        forward_batch: ForwardBatch
        slice_info: the slice_info list used by complementary_copy
        layer_id: current layer index
    """
    contiguous_kv = forward_batch.cacheflex_contiguous_kv
    if contiguous_kv is None or layer_id not in contiguous_kv:
        return

    k_cont, v_cont = contiguous_kv[layer_id]
    cu_seq_lens = forward_batch.cacheflex_contiguous_cu_seq_lens
    kv_bases = cu_seq_lens[:-1]  # [bs], GPU tensor — avoid .item() per req
    k_buf = forward_batch.token_to_kv_pool.get_key_buffer(layer_id)
    v_buf = forward_batch.token_to_kv_pool.get_value_buffer(layer_id)
    req_to_token = forward_batch.req_to_token_pool.req_to_token

    for info in slice_info:
        if not info or not info.get("is_sliced"):
            continue

        req_idx = info["req_idx"]
        dst_loc = info["dst_tail_loc"]
        copy_range = info["copy_range"]
        prefix_len = info["prefix_len"]
        num_tokens = len(dst_loc)
        if num_tokens == 0:
            continue

        kv_base = kv_bases[req_idx]  # GPU scalar, no .item()
        # copy_range[0] is the logical position in the warmup request (Old Pos).
        # complementary_copy applies Delta-RoPE (offset prefix_len), so the
        # updated K/V in paged cache corresponds to the current request's
        # logical position = copy_range[0] + prefix_len (New Pos).
        # This matches cacheflex.py: target_pos_start = src_offset + prefix_len
        target_pos_start = copy_range[0] + prefix_len
        logical_positions = torch.arange(
            target_pos_start, target_pos_start + num_tokens,
            dtype=torch.long, device=dst_loc.device,
        )
        cont_offsets = kv_base + logical_positions

        # Read updated K/V from paged cache at dst_loc and write to contiguous buffer
        k_cont[cont_offsets] = k_buf[dst_loc.to(torch.long)]
        v_cont[cont_offsets] = v_buf[dst_loc.to(torch.long)]


# [ProphetKV] ==============================================================
# Cached reused-K gather + Delta-RoPE.
#
# The cached old-K of each reused segment lives in the paged pool
# (src_block_indices point at warmup pages, kept alive by src_nodes locks
# through Stage 1), so K at any layer can be read straight from the pool
# without a forward pass. The ProphetKV Stage-I probe uses this to let its
# query rows attend to the reused context forward-free.
#
# apply_rope_delta_to_k is the free-function form of llama.py::_apply_rope_delta_to_k
# (identical math) so qwen2/qwen3 (no llama-only method) can reuse it. gather is
# driven by the dynamic tasks (which already carry src_block_indices/src_offset/
# rope_delta) → same token order as the L0 copy, doc-only.
# ============================================================================
def apply_rope_delta_to_k(k_segment, rope_delta, rotary_emb):
    """Apply Delta-RoPE rotation to a K segment in-place.

    Free-function copy of llama.py::_apply_rope_delta_to_k — numerically identical:
      1. negative delta → flip second-half signs (Neox style) before & after
      2. delta_positions = abs(rope_delta) for all tokens
      3. apply_rope_with_cos_sin_cache_inplace (in-place on key)

    Args:
        k_segment: [num_tokens, Hkv, head_dim] K to rotate (modified in-place)
        rope_delta: int position shift (current_offset - cached_offset)
        rotary_emb: the layer's rotary embedding (cos_sin_cache / is_neox_style / rotary_dim)
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


def gather_old_k_at_deep_layer(forward_batch, layer_id, rotary_emb):
    """Gather the whole reuse-context (sys + docs) cached K at a given layer, Delta-RoPE adjusted.

    Public role: gather the cached reused-K at a given layer, Delta-RoPE aligned;
    used by the ProphetKV Stage-I probe.

    Mirrors the column set that offline ``attn_doc_scores`` normalizes its softmax over —
    i.e. the full reused context — minus the recomputed-query columns, which are unavailable
    forward-free (query is being recomputed; the layer is never reached for it). This is
    the only deliberate deviation from offline; it shifts only the per-probe-row softmax
    denominator (a cross-row reweighting in the mean).

    Concatenates all ``text_hit`` reuse sources of each request in logical order (sink-skipped,
    Delta-RoPE'd) into one [N_reuse, Hkv, Dh] tensor, and returns ``doc_start`` = number of
    leading sys columns. The softmax denominator spans the whole [0, N_reuse); only the doc
    region [doc_start:] is read out as per-doc U and fed to the selector (sys is denominator-
    only, never scored — matching offline doc_start=sys_len). A source is classified as a doc
    iff its plan has ``kvfocus=True`` (sys = naive → kvfocus=False). If sys is a recompute gap
    (not text_hit) its K is not cached and is simply absent → doc_start=0 (denominator = docs
    only), a graceful degradation.

    K is returned at native Hkv heads / native dtype; the consumer does the GQA expansion to
    Hq and upcasts to float32 for the q·K softmax.

    Returns:
        {req_idx: (k_reuse [N_reuse, Hkv, Dh], doc_start, doc_end)} ; (None, 0, 0) if no reuse K.
        Scored doc region = k_reuse[doc_start:doc_end] (kvfocus sources only); sys (leading) and a
        forced-recompute query (trailing, method="full") stay in k_reuse as denominator-only.
    """
    configs = getattr(forward_batch, "cacheflex_configs", None)
    if configs is None:
        return {}
    k_buf = forward_batch.token_to_kv_pool.get_key_buffer(layer_id)  # [pool, Hkv, Dh]
    out = {}
    for req_idx, cfg in enumerate(configs):
        if not cfg or "sources" not in cfg:
            continue
        parts = []
        cum = 0
        doc_start = None  # start col of first scored (kvfocus) doc source
        doc_end = None    # end col of last scored doc source
        for src in cfg["sources"]:
            if src.get("type") != "text_hit":
                continue
            idx = src.get("src_block_indices")
            if idx is None:
                continue
            sink = src.get("sink_token_len", 0)
            if sink:
                idx = idx[sink:]  # skip sink (covered by a preceding source)
            if not torch.is_tensor(idx):
                idx = torch.as_tensor(idx, device=k_buf.device)
            idx = idx.to(device=k_buf.device, dtype=torch.long)
            if idx.numel() == 0:
                continue
            # advanced indexing returns a fresh (non-view) tensor → safe for in-place RoPE
            k_seg = k_buf[idx]  # [seg_len, Hkv, Dh]
            rope_delta = src.get("rope_delta", 0)
            if rope_delta and rotary_emb is not None:
                apply_rope_delta_to_k(k_seg, int(rope_delta), rotary_emb)
            seg_len = k_seg.shape[0]
            plan = src.get("plan")
            is_doc = bool(getattr(plan, "kvfocus", False)) if plan is not None else False
            # Only kvfocus (doc) sources are scored. Leading sys (naive) is denominator-only;
            # a trailing query source (if present, e.g. method="full") is also denominator-only —
            # so the scored region is exactly the [first_doc, last_doc] span, never sys/query.
            if is_doc:
                if doc_start is None:
                    doc_start = cum
                doc_end = cum + seg_len
            cum += seg_len
            parts.append(k_seg)
        if not parts:
            out[req_idx] = (None, 0, 0)
            continue
        if doc_start is None:  # no doc source registered → degenerate, score nothing
            doc_start, doc_end = 0, 0
        out[req_idx] = (torch.cat(parts, dim=0), doc_start, doc_end)
    return out
