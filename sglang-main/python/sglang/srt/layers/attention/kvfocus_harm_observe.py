# HACK(Shizhuo) New
"""KVFocus Realized Harm Observation Module.

Implements dual-pass attention at the check layer (Reuse → Full order):
  Pass 1: Reuse-style attention (warmup K/V for cached segments) → output_reuse
  Pass 2: Full attention (current K/V for all tokens) → output_full

Then computes per-online-token differences:
  D_out_u = ||o_u^full - o_u^reuse||_2

Online tokens = probe (last 5 of each segment) + suffix (after all segments).

Ordering rationale:
  - Reuse pass only observes, does not modify model state
  - Full pass is production: writes correct K/V to pool, returns output to model
  - If reuse pass crashes, full pass still runs correctly

Gated by Category.HARM via sglang.srt.utils.cacheflex_dump:
  SGLANG_CACHEFLEX_DUMP_DIR=<path>             # master switch (must be set)
  SGLANG_CACHEFLEX_DUMP_HARM=1                 # enable harm observation
  SGLANG_CACHEFLEX_DUMP_HARM_MODE=both         # single_ablation | full_reuse | both
  SGLANG_CACHEFLEX_DUMP_HARM_PROBE_LEN=5       # probe tokens per segment tail
"""

import logging
import os
from typing import Dict, List, Optional

import torch

# unified dump control
from sglang.srt.utils import cacheflex_dump as _cf_dump

logger = logging.getLogger("KVFocus.HarmObserve")


def _get_env_config():
    enabled = _cf_dump.is_enabled(_cf_dump.Category.HARM)
    mode = _cf_dump.get_harm_mode()
    probe_len = _cf_dump.get_harm_probe_len()
    return enabled, mode, probe_len


def _identify_online_tokens(
    tasks: List[Dict],
    req_start: int,
    seq_len: int,
    probe_len: int,
) -> Dict:
    """Identify online token positions for a single request.

    Online tokens = last probe_len tokens of each segment + suffix.
    Suffix = tokens after all segment tasks end.

    Returns dict with:
      online_indices: list of absolute indices in the flat token array
      token_map: category → [start, end) ranges
      suffix_range: [suffix_start, suffix_end)
    """
    segments = []
    for seg_idx, task in enumerate(tasks):
        abs_start = req_start + task["rel_start"]
        length = task["src_len"]
        segments.append((seg_idx, abs_start, length))

    online_indices = []
    token_map = {
        "sys_probes": None,
        "doc_probes": [],
        "query_probes": None,
        "suffix": None,
    }

    for seg_idx, abs_start, length in segments:
        if length <= probe_len:
            probe_start = abs_start
            probe_end = abs_start + length
        else:
            probe_start = abs_start + length - probe_len
            probe_end = abs_start + length

        probe_range = [probe_start, probe_end]
        online_indices.extend(range(probe_start, probe_end))

        if seg_idx == 0:
            token_map["sys_probes"] = probe_range
        elif seg_idx == len(segments) - 1:
            token_map["query_probes"] = probe_range
        else:
            token_map["doc_probes"].append(probe_range)

    # Suffix: tokens after the last segment ends
    last_seg_end = segments[-1][1] + segments[-1][2] if segments else req_start
    req_end = req_start + seq_len
    if last_seg_end < req_end:
        suffix_range = [last_seg_end, req_end]
        online_indices.extend(range(last_seg_end, req_end))
        token_map["suffix"] = suffix_range

    return {
        "online_indices": online_indices,
        "token_map": token_map,
        "segments": segments,
    }


def _write_current_kv_to_pool(
    indices: List[int],
    k: torch.Tensor,
    v: torch.Tensor,
    kv_pool,
    layer_id: int,
    out_cache_loc: torch.Tensor,
):
    """Write current k[i], v[i] to pool at out_cache_loc[i] for given indices."""
    if not indices:
        return
    idx = torch.tensor(indices, device=k.device, dtype=torch.long)
    loc = out_cache_loc[idx]
    # k, v are [TotalTokens, H, D] at this point
    kv_pool.k_buffer[layer_id][loc] = k[idx]
    kv_pool.v_buffer[layer_id][loc] = v[idx]


def _run_cacheflex_attention(
    q: torch.Tensor,
    layer,
    forward_batch,
) -> torch.Tensor:
    """Run CacheFlex attention with current pool state (no save_kv_cache)."""
    from sglang.srt.layers.attention.triton_ops.cacheflex_extend_attention import (
        cacheflex_attention_fwd,
    )

    q_view = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)

    if layer.qk_head_dim != layer.v_head_dim:
        o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        o_view = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)
    else:
        o = torch.empty_like(q)
        o_view = o.view(-1, layer.tp_q_head_num, layer.qk_head_dim)

    if not q_view.is_contiguous():
        q_view = q_view.contiguous()

    bs = forward_batch.batch_size
    qo_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=q.device)
    qo_indptr[1:] = torch.cumsum(forward_batch.seq_lens, dim=0)

    kv_lens = forward_batch.cacheflex_total_kv_lens
    kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=q.device)
    kv_indptr[1:] = torch.cumsum(kv_lens, dim=0)

    logit_cap = 0.0
    if layer.logit_cap > 0 and layer.logit_capping_method == "tanh":
        logit_cap = layer.logit_cap

    cacheflex_attention_fwd(
        q=q_view,
        k_buffer=forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
        v_buffer=forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
        o=o_view,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        req_to_token=forward_batch.req_to_token_pool.req_to_token,
        req_pool_indices=forward_batch.req_pool_indices,
        positions=forward_batch.positions,
        stride_req_b=forward_batch.req_to_token_pool.req_to_token.stride(0),
        stride_req_s=forward_batch.req_to_token_pool.req_to_token.stride(1),
        custom_mask=forward_batch.attn_backend.forward_metadata.custom_mask,
        is_causal=True,
        mask_indptr=forward_batch.attn_backend.forward_metadata.mask_indptr,
        max_len_extend=torch.max(forward_batch.seq_lens).item(),
        sm_scale=layer.scaling,
        logit_cap=logit_cap,
        sliding_window_size=layer.sliding_window_size or -1,
        sinks=None,
        window_kv_offsets=None,
        xai_temperature_len=layer.xai_temperature_len,
    )

    return o


def _compute_harm_metrics(
    output_full: torch.Tensor,
    output_reuse: torch.Tensor,
    online_indices: List[int],
) -> Dict:
    """Compute per-online-token harm metrics.

    H_chk_out = (∑ ||o_full - o_reuse||²)^{1/2}
    """
    if not online_indices:
        return {
            "D_out_per_online_token": torch.tensor([]),
            "H_chk_out": 0.0,
        }

    idx = torch.tensor(online_indices, device=output_full.device, dtype=torch.long)
    out_full_online = output_full[idx]
    out_reuse_online = output_reuse[idx]

    diff = out_full_online - out_reuse_online
    D_out = torch.norm(diff, dim=-1) if diff.dim() == 2 else torch.norm(diff, dim=(1, 2))

    return {
        "D_out_per_online_token": D_out.cpu(),
        "H_chk_out": torch.norm(D_out).item(),
    }


def _compute_category_harm(
    output_full: torch.Tensor,
    output_reuse: torch.Tensor,
    index_range: Optional[List[int]],
) -> float:
    """Compute H_chk for a specific category of online tokens."""
    if index_range is None or len(index_range) != 2 or index_range[1] <= index_range[0]:
        return 0.0
    indices = list(range(index_range[0], index_range[1]))
    if not indices:
        return 0.0
    metrics = _compute_harm_metrics(output_full, output_reuse, indices)
    return metrics["H_chk_out"]


def _compute_m_stats(task: Dict, abs_start: int, v: torch.Tensor, kv_pool, layer_id: int, forward_batch=None) -> Dict:
    """Compute per-segment m (||ΔV||) statistics."""
    from sglang.srt.layers.attention.triton_ops.cacheflex import compute_diff

    length = task["src_len"]
    # [P2] Use stored L0 prefix V for source segments (prefix hidden_states are stale at L1)
    l0_prefix_v_map = getattr(forward_batch, '_cacheflex_l0_prefix_v_map', None) if forward_batch is not None else None
    req_idx = task.get("req_idx", None)
    rel_start = task.get("rel_start", None)
    if l0_prefix_v_map is not None and req_idx is not None and rel_start is not None and (req_idx, rel_start) in l0_prefix_v_map:
        v_seg = l0_prefix_v_map[(req_idx, rel_start)]
    else:
        v_seg = v[abs_start : abs_start + length]

    m_diff, valid_len = compute_diff(
        v_seg, task["src_block_indices"], task["src_offset"], kv_pool, layer_id
    )

    if m_diff is None or valid_len == 0:
        return {"m_mean": 0.0, "m_max": 0.0, "m_sum": 0.0,
                "m_top1pct_mean": 0.0, "m_top5pct_mean": 0.0}

    m = m_diff[:valid_len]
    n = len(m)
    top1pct_mean = torch.topk(m, max(1, n // 100)).values.mean().item()
    top5pct_mean = torch.topk(m, max(1, n // 20)).values.mean().item()

    return {
        "m_mean": m.mean().item(),
        "m_max": m.max().item(),
        "m_sum": m.sum().item(),
        "m_top1pct_mean": top1pct_mean,
        "m_top5pct_mean": top5pct_mean,
    }


def run_harm_observation_reuse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    forward_batch,
    layer,
    tasks_by_req: List[List[Dict]],
) -> Optional[Dict]:
    """Steps 3-4: Prepare pool and run reuse-style attention.

    Called AFTER cacheflex_complementary_copy and KVFocus Hook,
    BEFORE the production full attention.

    Returns a dict with output_reuse and online token info,
    or None if disabled.
    """
    enabled, mode, probe_len = _get_env_config()
    if not enabled:
        return None

    # HARM dumps share the master DUMP_DIR + session resolution
    session = _cf_dump.get_session(default=None)
    if session is None or session == "default":
        logger.warning(
            "SGLANG_CACHEFLEX_DUMP_HARM=1 but no session resolvable "
            "(set SGLANG_CACHEFLEX_DUMP_SESSION or pass debug_dump_name via custom_params)"
        )
        return None

    if not getattr(forward_batch, "cacheflex_slice_info", None):
        logger.warning("Harm observation requires CacheFlex slice info, skipping")
        return None

    layer_id = layer.layer_id
    start_loc = forward_batch.extend_start_loc
    out_cache_loc = forward_batch.out_cache_loc
    kv_pool = forward_batch.token_to_kv_pool

    per_req_results = []

    for req_idx, tasks in enumerate(tasks_by_req):
        if not tasks:
            continue

        req_start = start_loc[req_idx].item()
        seq_len = forward_batch.extend_seq_lens[req_idx].item()

        # Identify online token positions
        online_info = _identify_online_tokens(tasks, req_start, seq_len, probe_len)

        # Collect all online token indices (probe + suffix)
        online_indices = online_info["online_indices"]

        req_result = {
            "segments": [],
            "online_token_map": online_info["token_map"],
            "num_segments": len(tasks),
            "req_start": req_start,
            "online_indices": online_indices,
        }

        # =============================================
        # Full Reuse Mode
        # =============================================
        # Pool state after cacheflex_complementary_copy:
        #   segment positions = warmup K/V
        #   suffix positions = Layer 0 K/V (unchanged)
        #
        # Step 3: Write probe + suffix current K/V to pool
        #   (non-probe segment positions keep warmup K/V from complementary copy)
        if mode in ("full_reuse", "both"):
            # Write probe tokens (last probe_len of each segment)
            probe_indices = []
            for seg_idx, task in enumerate(tasks):
                abs_start = req_start + task["rel_start"]
                length = task["src_len"]
                p_start = abs_start + max(0, length - probe_len)
                p_end = abs_start + length
                probe_indices.extend(range(p_start, p_end))

            # Write suffix tokens
            suffix_range = online_info["token_map"].get("suffix")
            suffix_indices = list(range(suffix_range[0], suffix_range[1])) if suffix_range else []

            all_online = probe_indices + suffix_indices
            _write_current_kv_to_pool(all_online, k, v, kv_pool, layer_id, out_cache_loc)

            # Pool now: segment non-probe = warmup K/V, probe + suffix = current K/V
            # This is the correct reuse state!

            # Step 4: Run reuse attention
            try:
                import sys
                print(f"[HarmObserve] Running full-reuse attention...", flush=True)
                output_reuse_full = _run_cacheflex_attention(q, layer, forward_batch)
                req_result["full_reuse_output"] = output_reuse_full
                print(f"[HarmObserve] Full-reuse attention done, output shape={output_reuse_full.shape}", flush=True)
            except Exception as e:
                import traceback
                print(f"[HarmObserve] Full-reuse FAILED: {e}\n{traceback.format_exc()}")
                logger.error(f"Harm obs full-reuse failed: {e}")
                import traceback
                traceback.print_exc()
                req_result["full_reuse_error"] = str(e)

        # =============================================
        # Single-segment Ablation Mode
        # =============================================
        # Strategy: cacheflex_complementary_copy has already written
        # Delta-RoPE-adjusted warmup K/V to segment pool positions.
        # 1. Save the warmup K/V that complementary copy wrote (these have
        #    correct RoPE for current positions)
        # 2. Overwrite ALL positions with current K/V (ablation base)
        # 3. For each segment, restore its warmup K/V, run attention, then
        #    re-overwrite with current K/V
        if mode in ("single_ablation", "both"):
            # Step 1: Save Delta-RoPE-adjusted warmup K/V per segment
            # (pool currently has warmup K/V from complementary copy)
            warmup_kv_per_seg = []
            for seg_idx, task in enumerate(tasks):
                abs_start = req_start + task["rel_start"]
                length = task["src_len"]
                non_probe_len = max(0, length - probe_len)
                if non_probe_len == 0 or task.get("kind") == "drop_all":
                    warmup_kv_per_seg.append(None)
                    continue
                seg_loc = out_cache_loc[abs_start : abs_start + non_probe_len]
                warmup_k = kv_pool.k_buffer[layer_id][seg_loc].clone()
                warmup_v = kv_pool.v_buffer[layer_id][seg_loc].clone()
                warmup_kv_per_seg.append((warmup_k, warmup_v, seg_loc))

            # Step 2: Write ALL current K/V to pool (ablation base state)
            req_out_loc = out_cache_loc[req_start : req_start + seq_len]
            kv_pool.k_buffer[layer_id][req_out_loc] = k[req_start : req_start + seq_len]
            kv_pool.v_buffer[layer_id][req_out_loc] = v[req_start : req_start + seq_len]

            # Step 3: For each segment, restore its warmup K/V, run
            # attention, then re-overwrite with current K/V
            for seg_idx, task in enumerate(tasks):
                if task.get("kind") == "drop_all":
                    req_result["segments"].append({
                        "seg_idx": seg_idx, "H_d_chk_out": 0.0,
                        "valid_len": 0, "kind": "drop_all",
                    })
                    continue

                abs_start = req_start + task["rel_start"]
                length = task["src_len"]
                non_probe_len = max(0, length - probe_len)

                saved = warmup_kv_per_seg[seg_idx]
                if saved is None or non_probe_len == 0:
                    req_result["segments"].append({
                        "seg_idx": seg_idx, "H_d_chk_out": 0.0, "valid_len": length,
                    })
                    continue

                warmup_k, warmup_v, seg_loc = saved

                # Overwrite this segment's non-probe pool positions with
                # Delta-RoPE-adjusted warmup K/V
                kv_pool.k_buffer[layer_id][seg_loc] = warmup_k
                kv_pool.v_buffer[layer_id][seg_loc] = warmup_v

                try:
                    output_reuse_d = _run_cacheflex_attention(q, layer, forward_batch)

                    seg_result = {
                        "seg_idx": seg_idx,
                        "valid_len": length,
                        "probe_token_indices": list(range(max(0, length - probe_len), length)),
                        "output_reuse": output_reuse_d,
                        "abs_start": abs_start,
                    }

                    # m stats
                    m_stats = _compute_m_stats(task, abs_start, v, kv_pool, layer_id, forward_batch)
                    seg_result.update(m_stats)

                    # U scores
                    u_l2 = u_l1 = 0.0
                    u_scores_l2 = forward_batch.cacheflex_runtime.get("kvfocus_u_scores_l2")
                    u_scores_l1 = forward_batch.cacheflex_runtime.get("kvfocus_u_scores_l1")
                    if u_scores_l2 and req_idx in u_scores_l2 and seg_idx < len(u_scores_l2[req_idx]):
                        u_l2 = u_scores_l2[req_idx][seg_idx].item()
                    if u_scores_l1 and req_idx in u_scores_l1 and seg_idx < len(u_scores_l1[req_idx]):
                        u_l1 = u_scores_l1[req_idx][seg_idx].item()
                    seg_result["u_l2"] = u_l2
                    seg_result["u_l1"] = u_l1

                    req_result["segments"].append(seg_result)

                except Exception as e:
                    logger.error(f"Harm obs single-ablation seg {seg_idx} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    req_result["segments"].append({
                        "seg_idx": seg_idx, "error": str(e),
                    })
                finally:
                    # Always restore current K/V for this segment's non-probe
                    req_seg_loc = out_cache_loc[abs_start : abs_start + non_probe_len]
                    kv_pool.k_buffer[layer_id][req_seg_loc] = k[abs_start : abs_start + non_probe_len]
                    kv_pool.v_buffer[layer_id][req_seg_loc] = v[abs_start : abs_start + non_probe_len]

        per_req_results.append(req_result)

    return {
        "per_req": per_req_results,
        "session": session,
        "layer_id": layer_id,
    }


def finalize_harm_metrics(
    output_full: torch.Tensor,
    reuse_data: Dict,
    forward_batch,
    tasks_by_req: List[List[Dict]],
    probe_len: int,
):
    """Step 6: Compute harm metrics after full attention completes.

    Compares output_full with stored output_reuse from the reuse pass.
    Dumps results to disk.
    """
    layer_id = reuse_data["layer_id"]
    session = reuse_data["session"]
    start_loc = forward_batch.extend_start_loc

    for req_idx, req_result in enumerate(reuse_data["per_req"]):
        if not req_result:
            continue

        req_start = start_loc[req_idx].item()
        seq_len = forward_batch.extend_seq_lens[req_idx].item()
        tasks = tasks_by_req[req_idx]

        online_info = _identify_online_tokens(tasks, req_start, seq_len, probe_len)
        online_indices = online_info["online_indices"]
        token_map = online_info["token_map"]

        # Full reuse metrics
        if "full_reuse_output" in req_result:
            output_reuse_full = req_result.pop("full_reuse_output")
            metrics = _compute_harm_metrics(output_full, output_reuse_full, online_indices)
            req_result["full_reuse"] = {
                "D_out_per_online_token": metrics["D_out_per_online_token"],
                "H_total_chk_out": metrics["H_chk_out"],
            }
        if "full_reuse_error" in req_result:
            req_result["full_reuse"] = {"error": req_result.pop("full_reuse_error")}

        # Single-segment ablation metrics
        for seg in req_result.get("segments", []):
            if "output_reuse" not in seg:
                continue

            output_reuse_d = seg.pop("output_reuse")
            abs_start = seg.pop("abs_start")

            # Overall harm for this segment
            metrics = _compute_harm_metrics(output_full, output_reuse_d, online_indices)

            # Per-category harm
            H_query = _compute_category_harm(output_full, output_reuse_d, token_map.get("query_probes"))
            H_probe_self = _compute_category_harm(
                output_full, output_reuse_d,
                [abs_start + max(0, seg["valid_len"] - probe_len),
                 abs_start + seg["valid_len"]]
            )
            H_suffix = _compute_category_harm(output_full, output_reuse_d, token_map.get("suffix"))
            H_all_probes = 0.0
            for probe_range in [token_map["sys_probes"]] + token_map["doc_probes"] + [token_map["query_probes"]]:
                H_all_probes += _compute_category_harm(output_full, output_reuse_d, probe_range) ** 2
            H_all_probes = H_all_probes ** 0.5 if H_all_probes > 0 else 0.0

            seg["H_d_chk_out"] = metrics["H_chk_out"]
            seg["D_out_per_online_token"] = metrics["D_out_per_online_token"]
            seg["H_d_chk_query_out"] = H_query
            seg["H_d_chk_probe_self_out"] = H_probe_self
            seg["H_d_chk_suffix_out"] = H_suffix
            seg["H_d_chk_all_probes_out"] = H_all_probes

    # Dump results
    _dump_harm_results(layer_id, reuse_data["per_req"], session)


_harm_dump_counter = 0


def _dump_harm_results(layer_id: int, results: List[Dict], session: str):
    """Save harm observation results to disk under Category.HARM."""
    global _harm_dump_counter
    sid = _harm_dump_counter
    _harm_dump_counter += 1

    save_dir = _cf_dump.get_dump_path(
        _cf_dump.Category.HARM, session=session, subpath=f"layer_{layer_id}"
    )
    if save_dir is None:
        return  # defensive; _get_env_config should have already short-circuited

    try:
        torch.save(results, os.path.join(save_dir, f"sample_{sid}.pt"))
        logger.info(f"[HarmObserve] Saved sample_{sid} with {len(results)} requests")
    except Exception as e:
        logger.error(f"Harm dump failed for sample_{sid}: {e}")
