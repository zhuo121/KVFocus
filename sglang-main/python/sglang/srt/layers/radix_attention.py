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
"""Radix attention."""
from __future__ import annotations

# HACK(Shizhuo)
# ============================================
import logging as _logging_module
import math
import os
import time
from enum import Enum
from typing import TYPE_CHECKING, Optional

import torch
from torch import nn

from sglang.srt.compilation.piecewise_context_manager import get_forward_context
from sglang.srt.utils import direct_register_custom_op
# HACK(Shizhuo): unified dump/profile control
from sglang.srt.utils import cacheflex_dump as _cf_dump

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


# HACK(Shizhuo)
# ============================================
_kvfocus_dump_counter = 0


def _kvfocus_next_sample_id():
    """Return a monotonically increasing sample ID for dump file naming."""
    global _kvfocus_dump_counter
    sid = _kvfocus_dump_counter
    _kvfocus_dump_counter += 1
    return sid


def _dump_kvfocus_scores(layer_id, seg_scores_list, session=None):
    """Dump per-token M×U_L2 scores for KVFocus auto analysis.

    Gated by Category.KVFOCUS_SCORES; output: $DUMP_DIR/<session>/kvfocus_scores/layer_{lid}/sample_{sid}.pt
    Returns: sample ID (sid) for matching mask file, or -1 if dump disabled.
    """
    save_dir = _cf_dump.get_dump_path(
        _cf_dump.Category.KVFOCUS_SCORES, session=session, subpath=f"layer_{layer_id}"
    )
    if save_dir is None:
        return -1
    sid = _kvfocus_next_sample_id()
    try:
        torch.save(seg_scores_list, save_dir / f"sample_{sid}.pt")
    except Exception as e:
        _logging_module.getLogger("KVFocus").warning(f"Dump failed for sample_{sid}: {e}")
    return sid


def _dump_kvfocus_mask(layer_id, seg_mask_data, sid, session=None):
    """Dump per-segment recompute mask after top-k.

    Gated by Category.KVFOCUS_SCORES (same gate as scores).
    """
    save_dir = _cf_dump.get_dump_path(
        _cf_dump.Category.KVFOCUS_SCORES, session=session, subpath=f"layer_{layer_id}"
    )
    if save_dir is None:
        return
    try:
        torch.save(seg_mask_data, save_dir / f"sample_{sid}_mask.pt")
    except Exception as e:
        _logging_module.getLogger("KVFocus").warning(f"Mask dump failed for sample_{sid}: {e}")


def _dump_kvfocus_output_diff(layer_id, sid, per_token_output_diff, segment_boundaries=None, session=None):
    """Dump per-token output diff ‖o_full[t] - o_warm[t]‖₂ for all tokens.

    Gated by Category.KVFOCUS_EXTENDED (independent of KVFOCUS_SCORES; can enable separately).

    Args:
        segment_boundaries: list of (rel_start, src_len) tuples for each segment.
    """
    if sid < 0:
        return
    save_dir = _cf_dump.get_dump_path(
        _cf_dump.Category.KVFOCUS_EXTENDED, session=session, subpath=f"layer_{layer_id}"
    )
    if save_dir is None:
        return
    data = {"per_token_output_diff": per_token_output_diff}
    if segment_boundaries is not None:
        data["segment_boundaries"] = segment_boundaries
    try:
        torch.save(data, save_dir / f"sample_{sid}_outdiff.pt")
    except Exception as e:
        _logging_module.getLogger("KVFocus").warning(f"Output diff dump failed for sample_{sid}: {e}")


# HACK(Shizhuo): [KVFocus-Adaptive] Two-pass (or three-pass) attention for adaptive recompute rate
def run_two_pass_attention(
    q, k, v, forward_batch, layer, tasks, req_idx, sr,
    num_probes, global_mask, **kwargs
):
    """Run 2-pass (or 3-pass) attention at check layer.

    Operational modes:
      - pass_mode="reuse_full" (DEFAULT): 2-pass reuse+full, D^R observations
      - pass_mode="warmup_full" (experimental): 2-pass warmup+full, D^W observations
      - pass_mode="3pass" (reference): 3-pass warmup→reuse→full, D^R + D^W dual group

    Returns dict with keys:
        output_full, recompute_rate, global_mask_updates, per_segment_counts,
        H, threshold, k_star, max_prop, mean_prop,
        probe_residual dict (always contains R0, d_probe_per_seg, d_probe_mean, g_N, d_peak),
        [optional] warmup diagnostics (only when warmup pass is executed)
        [optional] timing (only when KVFOCUS_PROFILE=1)
    or None if no auto-rate path is triggered.
    """
    from sglang.srt.layers.attention.kvfocus_harm_observe import (
        _identify_online_tokens,
        _write_current_kv_to_pool,
        _run_cacheflex_attention,
    )
    from sglang.srt.layers.attention.kvfocus_adaptive_rate import _compute_probe_metrics

    import logging as _logging
    _kvfocus_logger = _logging.getLogger("KVFocus")

    # Profile timing hooks
    profile_enabled = _cf_dump.profile_timing_enabled()
    timing = {}

    # Determine pass_mode
    pass_mode = tasks[0].get("pass_mode", "reuse_full") if tasks else "reuse_full"

    # Backward compat: SGLANG_KVFOCUS_WARMUP_PASS=1 → treat as pass_mode="3pass"
    if os.environ.get("SGLANG_KVFOCUS_WARMUP_PASS", "0") == "1" and pass_mode != "3pass":
        import warnings
        warnings.warn(
            "SGLANG_KVFOCUS_WARMUP_PASS is deprecated. Use pass_mode='3pass' in CacheFlexPlan.",
            DeprecationWarning, stacklevel=2,
        )
        pass_mode = "3pass"

    # Detect adaptive mode parameters
    solver_val = tasks[0].get("solver", "residual_budget") if tasks else "residual_budget"
    lam_val = None
    for task in tasks:
        if task.get("lam") is not None:
            lam_val = task["lam"]
            break

    # Adaptive solvers (residual_budget/coverage/threshold) require 2-pass attention
    is_adaptive = solver_val in ("residual_budget", "coverage", "threshold")
    if not is_adaptive or not sr.token_scores_list:
        return None

    # Step 1: Identify probe + suffix indices (per-segment)
    req_start = forward_batch.extend_start_loc[req_idx]
    seq_len = forward_batch.extend_seq_lens[req_idx].item()

    online_info = _identify_online_tokens(tasks, req_start, seq_len, num_probes)
    probe_indices = []
    probe_indices_per_seg = []
    for seg_idx_a, task_a in enumerate(tasks):
        abs_start_a = req_start + task_a["rel_start"]
        length_a = task_a["src_len"]
        p_start = abs_start_a + max(0, length_a - num_probes)
        p_end = abs_start_a + length_a
        seg_probe = list(range(p_start, p_end))
        probe_indices.extend(seg_probe)
        probe_indices_per_seg.append(seg_probe)
    suffix_range = online_info["token_map"].get("suffix")
    suffix_indices = list(range(suffix_range[0], suffix_range[1])) if suffix_range else []

    kv_pool = forward_batch.token_to_kv_pool
    out_cache_loc = forward_batch.out_cache_loc

    # ========================================================================
    # Mode 1: pass_mode="reuse_full" (DEFAULT)
    #   1. Write probe+suffix current KV to pool (one step)
    #   2. Reuse pass → o_reuse
    #   3. Write all current KV to pool
    #   4. Full pass → o_full
    #   5. R0 = ||o_full[S] - o_reuse[S]|| / ||o_full[S]||
    #   6. D_probe_d = ||o_full[probe_d] - o_reuse[probe_d]|| / norm
    #   7. No delta_online/delta_full_warmup diagnostics
    # ========================================================================
    if pass_mode == "reuse_full":
        # Write probe+suffix current KV to pool in one step
        combined_indices = suffix_indices + probe_indices
        _write_current_kv_to_pool(combined_indices, k, v, kv_pool, layer.layer_id, out_cache_loc)

        # Pass 1: Reuse attention (non-probe=warmup KV, probe+suffix=current KV)
        if profile_enabled:
            torch.cuda.synchronize()
            t_start = time.perf_counter()
        output_reuse = _run_cacheflex_attention(q, layer, forward_batch)
        if profile_enabled:
            torch.cuda.synchronize()
            timing["reuse_pass_ms"] = (time.perf_counter() - t_start) * 1000

        # Save per-segment probe output from reuse pass
        reuse_probe_per_seg = []
        for seg_probe_idx in probe_indices_per_seg:
            if seg_probe_idx:
                reuse_probe_per_seg.append(output_reuse[seg_probe_idx].clone())
            else:
                reuse_probe_per_seg.append(None)
        if suffix_indices:
            R0_baseline_suffix = output_reuse[suffix_indices].clone()

        # Write all current KV to pool
        req_out_loc = out_cache_loc[req_start : req_start + seq_len]
        kv_pool.k_buffer[layer.layer_id][req_out_loc] = k[req_start : req_start + seq_len]
        kv_pool.v_buffer[layer.layer_id][req_out_loc] = v[req_start : req_start + seq_len]

        # Pass 2: Full attention (save_kv_cache=True)
        if profile_enabled:
            torch.cuda.synchronize()
            t_start = time.perf_counter()
        output_full = forward_batch.attn_backend.forward(
            q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs
        )
        if profile_enabled:
            torch.cuda.synchronize()
            timing["full_pass_ms"] = (time.perf_counter() - t_start) * 1000

        # Compute R0 from suffix positions (all norms on GPU, one .item() at end)
        if suffix_indices:
            diff = output_full[suffix_indices] - R0_baseline_suffix
            R0_abs_t = torch.norm(diff)
            O_full_norm_t = torch.norm(output_full[suffix_indices])
            R0 = (R0_abs_t / (O_full_norm_t + 1e-12)).item()
        else:
            R0 = 0.0

        # Compute per-segment probe damage D_probe (GPU norms, batch .item())
        d_probe_per_seg = [0.0] * len(probe_indices_per_seg)
        d_probe_rel_per_seg = [0.0] * len(probe_indices_per_seg)
        _norms_gpu = []
        _norm_indices = []  # seg_idx for mapping back
        for seg_idx_d, seg_probe_idx in enumerate(probe_indices_per_seg):
            if seg_probe_idx and reuse_probe_per_seg[seg_idx_d] is not None:
                probe_diff = output_full[seg_probe_idx] - reuse_probe_per_seg[seg_idx_d]
                _norms_gpu.append(torch.norm(probe_diff))
                _norms_gpu.append(torch.norm(output_full[seg_probe_idx]))
                _norm_indices.append(seg_idx_d)

        if _norms_gpu:
            _norms_cpu = torch.stack(_norms_gpu).cpu().tolist()
            for i, seg_idx_d in enumerate(_norm_indices):
                d_abs = _norms_cpu[2 * i]
                d_norm = _norms_cpu[2 * i + 1]
                d_probe_per_seg[seg_idx_d] = d_abs
                d_probe_rel_per_seg[seg_idx_d] = d_abs / (d_norm + 1e-12)

        d_probe_mean = sum(d_probe_rel_per_seg) / max(len(d_probe_rel_per_seg), 1)

        # No warmup diagnostics in reuse_full mode
        probe_residual = {
            "d_probe_per_seg": d_probe_rel_per_seg,
            "d_probe_rel_per_seg": d_probe_rel_per_seg,
            "d_probe_mean": d_probe_mean,
            "R0": R0,
            "n_segments": len(probe_indices_per_seg),
            "n_total_tokens": sr.total_cached_tokens,
        }

    # ========================================================================
    # Mode 2: pass_mode="warmup_full" (experimental)
    #   1. Write suffix current KV only (probe stays warmup)
    #   2. Warmup pass → o_warm
    #   3. Write all current KV to pool → full pass → o_full
    #   4. R0'/D_probe' from o_full - o_warm
    # ========================================================================
    elif pass_mode == "warmup_full":
        # Write suffix current KV only (probe stays warmup)
        _write_current_kv_to_pool(suffix_indices, k, v, kv_pool, layer.layer_id, out_cache_loc)

        # Pass 1: Warmup attention (all=warmup KV except suffix=current KV)
        if profile_enabled:
            torch.cuda.synchronize()
            t_start = time.perf_counter()
        output_warmup = _run_cacheflex_attention(q, layer, forward_batch)
        if profile_enabled:
            torch.cuda.synchronize()
            timing["warmup_pass_ms"] = (time.perf_counter() - t_start) * 1000

        # Save per-segment probe output from warmup pass
        warmup_probe_per_seg = []
        for seg_probe_idx in probe_indices_per_seg:
            if seg_probe_idx:
                warmup_probe_per_seg.append(output_warmup[seg_probe_idx].clone())
            else:
                warmup_probe_per_seg.append(None)
        if suffix_indices:
            R0_baseline_suffix = output_warmup[suffix_indices].clone()

        # Write all current KV to pool
        req_out_loc = out_cache_loc[req_start : req_start + seq_len]
        kv_pool.k_buffer[layer.layer_id][req_out_loc] = k[req_start : req_start + seq_len]
        kv_pool.v_buffer[layer.layer_id][req_out_loc] = v[req_start : req_start + seq_len]

        # Pass 2: Full attention (save_kv_cache=True)
        if profile_enabled:
            torch.cuda.synchronize()
            t_start = time.perf_counter()
        output_full = forward_batch.attn_backend.forward(
            q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs
        )
        if profile_enabled:
            torch.cuda.synchronize()
            timing["full_pass_ms"] = (time.perf_counter() - t_start) * 1000

        # Compute R0 from suffix positions (warmup baseline, GPU norms, one .item())
        if suffix_indices:
            diff = output_full[suffix_indices] - R0_baseline_suffix
            R0_abs_t = torch.norm(diff)
            O_full_norm_t = torch.norm(output_full[suffix_indices])
            R0 = (R0_abs_t / (O_full_norm_t + 1e-12)).item()
        else:
            R0 = 0.0

        # Compute per-segment probe damage D_probe (warmup baseline, GPU norms, batch .item())
        d_probe_per_seg = [0.0] * len(probe_indices_per_seg)
        d_probe_rel_per_seg = [0.0] * len(probe_indices_per_seg)
        _norms_gpu = []
        _norm_indices = []
        for seg_idx_d, seg_probe_idx in enumerate(probe_indices_per_seg):
            if seg_probe_idx and warmup_probe_per_seg[seg_idx_d] is not None:
                probe_diff = output_full[seg_probe_idx] - warmup_probe_per_seg[seg_idx_d]
                _norms_gpu.append(torch.norm(probe_diff))
                _norms_gpu.append(torch.norm(output_full[seg_probe_idx]))
                _norm_indices.append(seg_idx_d)

        if _norms_gpu:
            _norms_cpu = torch.stack(_norms_gpu).cpu().tolist()
            for i, seg_idx_d in enumerate(_norm_indices):
                d_abs = _norms_cpu[2 * i]
                d_norm = _norms_cpu[2 * i + 1]
                d_probe_per_seg[seg_idx_d] = d_abs
                d_probe_rel_per_seg[seg_idx_d] = d_abs / (d_norm + 1e-12)

        d_probe_mean = sum(d_probe_rel_per_seg) / max(len(d_probe_rel_per_seg), 1)

        # Compute per-token output diff when extended dump is enabled
        per_token_output_diff = None
        if _cf_dump.is_enabled(_cf_dump.Category.KVFOCUS_EXTENDED):
            per_token_output_diff = torch.norm(
                output_full[req_start : req_start + seq_len] - output_warmup[req_start : req_start + seq_len],
                dim=-1,
            ).cpu()

        # No delta_online diagnostics in warmup_full mode
        probe_residual = {
            "d_probe_per_seg": d_probe_rel_per_seg,
            "d_probe_rel_per_seg": d_probe_rel_per_seg,
            "d_probe_mean": d_probe_mean,
            "R0": R0,
            "n_segments": len(probe_indices_per_seg),
            "n_total_tokens": sr.total_cached_tokens,
        }
        if per_token_output_diff is not None:
            probe_residual["per_token_output_diff"] = per_token_output_diff

    # ========================================================================
    # Mode 3: pass_mode="3pass" (V1 reference — superset of reuse_full + warmup_full)
    #   1. Write suffix → warmup pass → o_warm
    #   2. Write probe → reuse pass → o_reuse
    #   3. Write all → full pass → o_full
    #   4. H formula uses D^R group (reuse baseline, same as reuse_full)
    #   5. D^W group computed and stored in probe_residual for sanity check
    #   6. delta_online/delta_full_warmup diagnostics computed
    # ========================================================================
    elif pass_mode == "3pass":
        # Write suffix current KV to pool
        _write_current_kv_to_pool(suffix_indices, k, v, kv_pool, layer.layer_id, out_cache_loc)

        # Pass 1: Warmup attention (segment=warmup KV, suffix=current KV)
        if profile_enabled:
            torch.cuda.synchronize()
            t_start = time.perf_counter()
        output_warmup = _run_cacheflex_attention(q, layer, forward_batch)
        if profile_enabled:
            torch.cuda.synchronize()
            timing["warmup_pass_ms"] = (time.perf_counter() - t_start) * 1000

        # Save per-segment probe output from warmup pass
        warmup_probe_per_seg = []
        for seg_probe_idx in probe_indices_per_seg:
            if seg_probe_idx:
                warmup_probe_per_seg.append(output_warmup[seg_probe_idx].clone())
            else:
                warmup_probe_per_seg.append(None)
        if suffix_indices:
            R0_warmup_suffix = output_warmup[suffix_indices].clone()

        # Write probe current KV to pool
        _write_current_kv_to_pool(probe_indices, k, v, kv_pool, layer.layer_id, out_cache_loc)

        # Pass 2: Reuse attention (non-probe=warmup KV, probe+suffix=current KV)
        if profile_enabled:
            torch.cuda.synchronize()
            t_start = time.perf_counter()
        output_reuse = _run_cacheflex_attention(q, layer, forward_batch)
        if profile_enabled:
            torch.cuda.synchronize()
            timing["reuse_pass_ms"] = (time.perf_counter() - t_start) * 1000

        # Save per-segment probe output from reuse pass
        reuse_probe_per_seg = []
        for seg_probe_idx in probe_indices_per_seg:
            if seg_probe_idx:
                reuse_probe_per_seg.append(output_reuse[seg_probe_idx].clone())
            else:
                reuse_probe_per_seg.append(None)
        if suffix_indices:
            R0_reuse_suffix = output_reuse[suffix_indices].clone()

        # Write all current KV to pool
        req_out_loc = out_cache_loc[req_start : req_start + seq_len]
        kv_pool.k_buffer[layer.layer_id][req_out_loc] = k[req_start : req_start + seq_len]
        kv_pool.v_buffer[layer.layer_id][req_out_loc] = v[req_start : req_start + seq_len]

        # Pass 3: Full attention (save_kv_cache=True)
        if profile_enabled:
            torch.cuda.synchronize()
            t_start = time.perf_counter()
        output_full = forward_batch.attn_backend.forward(
            q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs
        )
        if profile_enabled:
            torch.cuda.synchronize()
            timing["full_pass_ms"] = (time.perf_counter() - t_start) * 1000

        # Compute R0 from suffix positions (reuse baseline — GPU norms, batch .item())
        # Also compute D^W group (warmup baseline) for sanity check
        R0_warmup = 0.0
        if suffix_indices:
            diff = output_full[suffix_indices] - R0_reuse_suffix
            R0_abs_t = torch.norm(diff)
            O_full_norm_t = torch.norm(output_full[suffix_indices])
            # Batch all suffix-level norms together
            warmup_suffix_diff = output_full[suffix_indices] - R0_warmup_suffix
            R0_warmup_abs_t = torch.norm(warmup_suffix_diff)
            full_warmup_suffix_diff = output_full[suffix_indices] - R0_warmup_suffix
            delta_fw_suffix_abs_t = torch.norm(full_warmup_suffix_diff)
            reuse_warmup_suffix_diff = R0_reuse_suffix - R0_warmup_suffix
            delta_rw_suffix_abs_t = torch.norm(reuse_warmup_suffix_diff)
            # One GPU→CPU transfer for all suffix norms
            _suffix_norms = torch.stack([R0_abs_t, O_full_norm_t, R0_warmup_abs_t,
                                         delta_fw_suffix_abs_t, delta_rw_suffix_abs_t]).cpu().tolist()
            R0 = _suffix_norms[0] / (_suffix_norms[1] + 1e-12)
            R0_warmup = _suffix_norms[2] / (_suffix_norms[1] + 1e-12)
            delta_full_warmup_suffix_rel = _suffix_norms[3] / (_suffix_norms[1] + 1e-12)
            delta_reuse_warmup_suffix_rel = _suffix_norms[4] / (_suffix_norms[1] + 1e-12)
        else:
            R0 = 0.0
            delta_full_warmup_suffix_rel = 0.0
            delta_reuse_warmup_suffix_rel = 0.0

        # Per-segment probe damage metrics (GPU norms, batch .item())
        d_probe_per_seg = [0.0] * len(probe_indices_per_seg)
        d_probe_rel_per_seg = [0.0] * len(probe_indices_per_seg)
        d_probe_warmup_rel_per_seg = [0.0] * len(probe_indices_per_seg)
        delta_online_per_seg = [0.0] * len(probe_indices_per_seg)
        delta_online_rel_per_seg = [0.0] * len(probe_indices_per_seg)
        delta_full_warmup_probe_per_seg = [0.0] * len(probe_indices_per_seg)
        delta_full_warmup_probe_rel_per_seg = [0.0] * len(probe_indices_per_seg)
        _seg_norms_gpu = []
        _seg_norm_indices = []
        for seg_idx_d, seg_probe_idx in enumerate(probe_indices_per_seg):
            if seg_probe_idx and warmup_probe_per_seg[seg_idx_d] is not None:
                # Collect all 4 norms per segment on GPU
                probe_diff = output_full[seg_probe_idx] - reuse_probe_per_seg[seg_idx_d]
                warmup_probe_diff = output_full[seg_probe_idx] - warmup_probe_per_seg[seg_idx_d]
                online_diff = reuse_probe_per_seg[seg_idx_d] - warmup_probe_per_seg[seg_idx_d]
                full_warmup_diff = output_full[seg_probe_idx] - warmup_probe_per_seg[seg_idx_d]
                _seg_norms_gpu.extend([
                    torch.norm(probe_diff),      # d_abs
                    torch.norm(output_full[seg_probe_idx]),  # d_norm
                    torch.norm(warmup_probe_diff),  # d_warmup_abs
                    torch.norm(online_diff),      # d_online_abs
                    torch.norm(full_warmup_diff), # d_fw_abs
                ])
                _seg_norm_indices.append(seg_idx_d)

        if _seg_norms_gpu:
            _seg_norms_cpu = torch.stack(_seg_norms_gpu).cpu().tolist()
            for i, seg_idx_d in enumerate(_seg_norm_indices):
                b = i * 5
                d_abs = _seg_norms_cpu[b]
                d_norm = _seg_norms_cpu[b + 1]
                d_warmup_abs = _seg_norms_cpu[b + 2]
                d_online_abs = _seg_norms_cpu[b + 3]
                d_fw_abs = _seg_norms_cpu[b + 4]
                d_probe_per_seg[seg_idx_d] = d_abs
                d_probe_rel_per_seg[seg_idx_d] = d_abs / (d_norm + 1e-12)
                d_probe_warmup_rel_per_seg[seg_idx_d] = d_warmup_abs / (d_norm + 1e-12)
                delta_online_per_seg[seg_idx_d] = d_online_abs
                delta_online_rel_per_seg[seg_idx_d] = d_online_abs / (d_norm + 1e-12)
                delta_full_warmup_probe_per_seg[seg_idx_d] = d_fw_abs
                delta_full_warmup_probe_rel_per_seg[seg_idx_d] = d_fw_abs / (d_norm + 1e-12)

        d_probe_mean = sum(d_probe_rel_per_seg) / max(len(d_probe_rel_per_seg), 1)
        delta_online_mean = sum(delta_online_per_seg) / max(len(delta_online_per_seg), 1)

        # D^W group: d_probe_warmup mean and d_peak_warmup
        d_probe_warmup_mean = sum(d_probe_warmup_rel_per_seg) / max(len(d_probe_warmup_rel_per_seg), 1)
        _, d_peak_warmup = _compute_probe_metrics(
            d_probe_warmup_rel_per_seg,
            eta=tasks[0].get("eta", 0.5) if tasks else 0.5,
            top_k_peak=tasks[0].get("top_k_peak", 3) if tasks else 3,
        )

        # Full probe_residual with D^R + D^W dual group
        probe_residual = {
            "d_probe_per_seg": d_probe_rel_per_seg,
            "delta_online_per_seg": delta_online_per_seg,
            "d_probe_rel_per_seg": d_probe_rel_per_seg,
            "delta_online_rel_per_seg": delta_online_rel_per_seg,
            "delta_full_warmup_probe_per_seg": delta_full_warmup_probe_per_seg,
            "delta_full_warmup_probe_rel_per_seg": delta_full_warmup_probe_rel_per_seg,
            "delta_full_warmup_suffix_rel": delta_full_warmup_suffix_rel,
            "delta_reuse_warmup_suffix_rel": delta_reuse_warmup_suffix_rel,
            "d_probe_mean": d_probe_mean,
            "delta_online_mean": delta_online_mean,
            "R0": R0,
            # D^W group (warmup baseline — V1 sanity check)
            "R0_warmup": R0_warmup,
            "d_probe_warmup_per_seg": d_probe_warmup_rel_per_seg,
            "d_probe_warmup_mean": d_probe_warmup_mean,
            "d_peak_warmup": d_peak_warmup,
            "n_segments": len(probe_indices_per_seg),
            "n_total_tokens": sr.total_cached_tokens,
        }

    # ========================================================================
    # Token selection via Selector (common to all modes)
    # ========================================================================
    from sglang.srt.layers.attention.token_selector import create_selector_from_tasks
    selector = create_selector_from_tasks(tasks)

    if profile_enabled:
        torch.cuda.synchronize()
        t_solver_start = time.perf_counter()

    result = selector.select(
        scorer_result=sr,
        R0=R0,
        d_probe_per_seg=d_probe_rel_per_seg,
        lam=lam_val,
        n_ref=tasks[0].get("n_ref", 11) if tasks else 11,
        max_rate=tasks[0].get("max_rate", 0.50) if tasks else 0.50,
        b0=tasks[0].get("b0", 0) if tasks else 0,
        b0_mode=tasks[0].get("b0_mode", "topk") if tasks else "topk",
        scaling=tasks[0].get("scaling", "linear") if tasks else "linear",
        beta=tasks[0].get("beta", 1.0) if tasks else 1.0,
        eta=tasks[0].get("eta", 0.5) if tasks else 0.5,
        gamma=tasks[0].get("gamma", 2.0) if tasks else 2.0,
        rho=tasks[0].get("rho", 0.05) if tasks else 0.05,
        top_k_peak=tasks[0].get("top_k_peak", 3) if tasks else 3,
        target_rate=tasks[0].get("target_rate", 0.0) if tasks else 0.0,
    )

    if profile_enabled:
        torch.cuda.synchronize()
        timing["solver_ms"] = (time.perf_counter() - t_solver_start) * 1000

    # Generate global_mask updates
    global_mask_updates = []
    if sr.mapping:
        for abs_start, valid_len, _ in sr.mapping:
            global_mask[abs_start : abs_start + valid_len] = False
        for abs_start, local_indices in result["global_mask_updates"]:
            global_mask[abs_start + local_indices] = True

    # Store timing data if profiling
    if profile_enabled and timing:
        probe_residual["timing"] = timing

    # Propagate g_N and d_peak from selector result to probe_residual
    if "g_N" in result:
        probe_residual["g_N"] = result["g_N"]
    if "d_peak" in result:
        probe_residual["d_peak"] = result["d_peak"]

    # Logging
    total_selected = sum(result["per_segment_counts"])
    if _kvfocus_logger.isEnabledFor(_logging.INFO):
        _kvfocus_logger.info(
            f"[KVFocus] Req{req_idx}: "
            f"pass_mode={pass_mode}, solver={solver_val}, scaling={tasks[0].get('scaling', 'linear') if tasks else 'linear'}, "
            f"R0_suffix={R0:.4f}, "
            f"D_probe_mean={d_probe_mean:.4f}, "
            f"H={result.get('H', 0):.4f}, "
            f"|I|={sr.total_cached_tokens}, "
            f"τ_π={result['threshold']:.2e}, "
            f"π_max={result.get('max_prop', 0):.2e}, "
            f"π_mean={result.get('mean_prop', 0):.2e}, "
            f"recompute_rate={result['recompute_rate']:.4f}, "
            f"selected={total_selected}/{sr.total_cached_tokens}, "
            f"k*={result.get('k_star', 'N/A')}, "
            f"per_seg={result['per_segment_counts']}"
        )

    return {
        "output_full": output_full,
        "recompute_rate": result["recompute_rate"],
        "global_mask_updates": result["global_mask_updates"],
        "per_segment_counts": result["per_segment_counts"],
        "H": result.get("H", 0),
        "threshold": result["threshold"],
        "k_star": result.get("k_star"),
        "max_prop": result.get("max_prop", 0),
        "mean_prop": result.get("mean_prop", 0),
        "probe_residual": probe_residual,
    }


# ========================================================================
# End of HACK(Shizhuo): run_two_pass_attention
# ========================================================================


# HACK(Shizhuo) [ProphetKV-Probe] ========================================================
# All-layer reused-K query->ctx U probe (= ProphetKV Stage-I probe pass).
#
# Runs on a *naive-KV* forward (docs reused, context recompute 0; query=full =
# ProphetKV's query-only Stage-I pass). At EVERY layer it gathers the cached reused-K @ THIS
# layer (per-layer call passing self.layer_id) and scores the real query rows against that
# reused context forward-free, then dumps u_doc per layer. Offline: mean-fuse U across
# layers -> top-r -> external mask.
#
# Forward-free: reads cached src-block K; no doc tokens in the forward, no M,
# no scorer block (naive => no dynamic tasks => Block B skipped), no selection, no sparsify.
# Triggered by env SGLANG_PROPHETKV_PROBE=1 OR any source plan.kvfocus_variant=="prophetkv-probe"
# (the eval's prophetkv-probe approach). The per-layer .pt dump is separately gated by
# SGLANG_PROPHETKV_PROBE_DUMP_DIR so timing runs can incur the probe cost without dump I/O.
# All gated -> OFF by default -> KVFocus / every other path byte-unchanged.
def _prophetkv_probe_enabled(forward_batch):
    if os.environ.get("SGLANG_PROPHETKV_PROBE", "0") == "1":
        return True
    cfgs = getattr(forward_batch, "cacheflex_configs", None)
    if not cfgs:
        return False
    for cfg in cfgs:
        if not cfg or "sources" not in cfg:
            continue
        for src in cfg["sources"]:
            plan = src.get("plan")
            if plan is not None and getattr(plan, "kvfocus_variant", "s") == "prophetkv-probe":
                return True
    return False


def _prophetkv_probe_step(layer, forward_batch, rotary_emb, q):
    """ProphetKV Stage-I, FAITHFUL (real user-query attention — NO prior). At THIS layer:
    real query Q (the forward's `q`, query rows) x reused context K@layer (gather, Delta-RoPE'd)
    -> softmax over context -> column-sum over query rows + mean over heads -> alpha_l(t) per
    context token. Accumulate into a running sum across layers (uniform fuse, Eq7);
    `_prophetkv_finalize` (after the model layer loop, still in the timed pass1 forward) does
    /L + top-r -> inline external-replay mask JSON.

    Faithful to ProphetKV (arXiv 2602.02579 §4.2-4.3 + Eq7): Q = real user query tokens (NOT a
    prior); per-layer query->context attention; column-sum; all-layer uniform mean. An earlier
    prior-based approximation of this probe was unfaithful to Stage-I and has been removed.

    Sanity check (query must be full/active, NOT reused): in the naive probe the query source is
    method="full" => recomputed => active in the forward, so its real Q is computed each layer.
    SGLANG_PROPHETKV_DEBUG=1 logs shapes for req0 @ first probe layer to confirm.
    """
    cfgs = getattr(forward_batch, "cacheflex_configs", None)
    if not cfgs:
        return
    has_probe = any(
        src.get("plan") is not None and getattr(src["plan"], "kvfocus_variant", "s") == "prophetkv-probe"
        for cfg in cfgs if cfg and "sources" in cfg for src in cfg["sources"]
    )
    if not has_probe:
        return
    rt = getattr(forward_batch, "cacheflex_runtime", None)
    if rt is None:
        rt = forward_batch.cacheflex_runtime = {}

    # reused context K @ THIS layer (Delta-RoPE'd): {req: (k_reuse[N,Hkv,D], doc_start, doc_end)}
    # k_reuse = sys + docs (+ maybe a trailing full-query source); doc region = [doc_start,doc_end].
    from sglang.srt.layers.attention.cacheflex_utils import gather_old_k_at_deep_layer
    r_emb = rotary_emb or getattr(layer, "rotary_emb", None)
    gathered = gather_old_k_at_deep_layer(forward_batch, layer.layer_id, r_emb)
    if not gathered:
        return

    start_loc = forward_batch.extend_start_loc
    seq_lens = forward_batch.extend_seq_lens
    probe_starts = getattr(forward_batch, "_cacheflex_probe_starts", None)
    sm_scale = layer.qk_head_dim ** -0.5
    group = layer.tp_q_head_num // max(layer.tp_k_head_num, 1)
    _dbg = os.environ.get("SGLANG_PROPHETKV_DEBUG", "0") == "1"
    # forward `q` arrives flat [N_active, Hq*D]; reshape to [N_active, Hq, D] for the einsum.
    q3 = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)

    alpha_sum = rt.setdefault("pkv_alpha_sum", {})   # req -> tensor[doc_len] (running sum over layers)
    doc_meta = rt.setdefault("pkv_doc_meta", {})     # req -> (doc_start, doc_end)
    rt["pkv_layer_count"] = rt.get("pkv_layer_count", 0) + 1

    for req_idx, entry in gathered.items():
        if entry is None:
            continue
        k_reuse, doc_start, doc_end = entry
        if k_reuse is None or doc_end <= doc_start:
            continue
        # HACK(Shizhuo) [PKV-QIP] query-in-pool scoring variant (env SGLANG_PROPHETKV_QUERY_IN_POOL) =====
        # 默认 (off): context = sys + docs = k_reuse[:doc_end]; 排除尾部 full-query 列
        #   —— 这是默认行为, 逐字节保留。
        # Variant on: 尾部 query-reuse 列 (k_reuse[doc_end:], 即 gather_old_k_at_deep_layer
        #   docstring 里 "a forced-recompute query (trailing, method=full) stays in
        #   k_reuse as denominator-only" 的那段 cached K) 进入**打分域**, 与 doc 争同一个
        #   r% 预算 (query 进池、参与选择、计入 r%)。
        # 用 cached (stale) K 而非 forward 里的 fresh K 打分: doc 列也是 stale K, 用 fresh
        #   会给 query 一个 doc 拿不到的精确度优势 —— 且真实部署里 query 同样来自 cache。
        # 三处随之改动:
        #   ① 分母含 query-reuse 列;
        #   ② query 行与 query-reuse 列在位置上重叠 → causal 不再是 no-op, 必须遮;
        #   ③ 各列可 attend 的行数不同 → 按有效行数归一化, 否则靠后的 query 列纯因
        #      "能看到它的行少" 而被系统性低估。
        _qip = os.environ.get("SGLANG_PROPHETKV_QUERY_IN_POOL", "0") == "1"
        _n_reuse = int(k_reuse.shape[0])
        q_rs = int(doc_end)
        q_re = _n_reuse if _qip else int(doc_end)
        if _qip and q_re <= q_rs:
            raise ValueError(
                f"[PKV-QIP] no query-reuse columns in k_reuse (n_reuse={_n_reuse}, "
                f"doc_end={doc_end}). The query source must be a cache-hit 'text_hit' "
                f"source; check that warmup ran with --warmup-with-query and that the "
                f"query source kept its token_ids/sink_token_len."
            )
        k_ctx = k_reuse[:q_re]                            # [N_ctx, Hkv, D]
        # =========================================================================
        # real query Q rows for this req in the sliced (active) forward = the query region
        # (last source start .. end of req), located via Stage-1's probe_starts.
        s = int(start_loc[req_idx]); e = s + int(seq_lens[req_idx])
        ps = int(probe_starts[req_idx]) if (probe_starts is not None and req_idx < len(probe_starts)) else 0
        q_query = q3[s + ps:e]                            # [n_q, Hq, D]
        if _dbg and rt["pkv_layer_count"] == 1 and req_idx == 0:
            _logging_module.getLogger("ProphetKVProbe").warning(
                f"[pkv-probe] L{layer.layer_id} req0: q_query={tuple(q_query.shape)} "
                f"k_ctx={tuple(k_ctx.shape)} doc=[{doc_start}:{doc_end}] "
                f"active_seq_len={int(seq_lens[req_idx])} probe_start={ps}"
            )
        if q_query.shape[0] == 0:
            continue  # query not active (reused?) -> no real query Q (will surface as no mask)
        k_exp = k_ctx.repeat_interleave(group, dim=1) if group > 1 else k_ctx   # [N_ctx, Hq, D]
        # query->context attention (ctx all precede query -> causal is a no-op)
        scores = torch.einsum("qhd,nhd->qhn", q_query.float(), k_exp.float()) * sm_scale  # [n_q,Hq,N_ctx]
        # HACK(Shizhuo) [PKV-QIP] ② causal over the overlapping query-reuse block =====
        # 位置对应: query 行 r 的全局位置 = doc_end + r; query-reuse 列 c 的全局位置
        # = doc_end + c (两者是同一段 token 的行/列视角) → 允许当且仅当 c <= r。
        _nq = int(q_query.shape[0])
        _qn = q_re - q_rs
        if _qip and _qn > 0:
            _r = torch.arange(_nq, device=scores.device).unsqueeze(1)      # [n_q,1]
            _c = torch.arange(_qn, device=scores.device).unsqueeze(0)      # [1,q_reuse]
            scores[:, :, q_rs:q_re] = scores[:, :, q_rs:q_re].masked_fill(
                (_c > _r).unsqueeze(1), float("-inf"))
        # =========================================================================
        attn = torch.softmax(scores, dim=-1)
        # ProphetKV: column-sum over query rows + mean over heads -> [N_ctx]
        alpha = attn.sum(dim=0).mean(dim=0)
        # HACK(Shizhuo) [PKV-QIP] ③ 按有效 attend 行数归一化 =========================
        # sys/doc 列被全部 n_q 行 attend; query-reuse 列 c 只被 r>=c 的 (n_q - c) 行
        # attend。不归一化则靠后的 query 列纯因行数少而系统性低估。除以有效行数 =
        # 把 α 从 "行求和" 变成 "可见行上的均值", 对 doc 列只是同一个全局常数 (不改
        # doc 内部排序), 但把 query-reuse 列放到了同一量纲上。
        if _qip and _qn > 0:
            _eff = torch.full((alpha.numel(),), float(_nq), device=alpha.device, dtype=alpha.dtype)
            _eff[q_rs:q_re] = (_nq - torch.arange(_qn, device=alpha.device)).clamp(min=1).to(alpha.dtype)
            alpha = alpha / _eff
        # =========================================================================
        alpha_doc = alpha[doc_start:doc_end].detach()    # [doc_len]
        # HACK(Shizhuo) [PKV-QIP] 打分域 = doc ∪ query-reuse (拼接, finalize 再切回) ===
        if _qip and _qn > 0:
            alpha_doc = torch.cat([alpha_doc, alpha[q_rs:q_re].detach()])
            rt.setdefault("pkv_qip_split", {})[req_idx] = (
                int(doc_end - doc_start), int(_qn), int(_nq))
        # =========================================================================
        # gate (attention faithful): dump inputs+output for req0@first probe layer so an offline
        # torch reference can independently recompute softmax(q.K).sum(0).mean(0) and byte-compare.
        if _dbg and rt["pkv_layer_count"] == 1 and req_idx == 0:
            _gdir = os.environ.get("SGLANG_PROPHETKV_GATE_DIR")
            if _gdir:
                try:
                    os.makedirs(_gdir, exist_ok=True)
                    torch.save({"q_query": q_query.detach().float().cpu(),
                                "k_ctx": k_ctx.detach().float().cpu(),
                                "group": int(group), "sm_scale": float(sm_scale),
                                "alpha": alpha.detach().float().cpu(),
                                "doc_start": int(doc_start), "doc_end": int(doc_end),
                                "layer": int(layer.layer_id)},
                               os.path.join(_gdir, "gate_attn_L0_req0.pt"))
                except Exception as _ge:
                    _logging_module.getLogger("ProphetKVProbe").warning(f"gate dump failed: {_ge}")
        prev = alpha_sum.get(req_idx)
        alpha_sum[req_idx] = alpha_doc if prev is None else prev + alpha_doc
        doc_meta[req_idx] = (int(doc_start), int(doc_end))
        # HACK(Shizhuo) [Fig2-perlayer] ===============================================
        # env-gated per-layer alpha stash (Fig.2 obs: answer-query->reused-token attention
        # per layer). Gated by SGLANG_PROPHETKV_PROBE_PERLAYER_DIR -> OFF by default ->
        # every path (incl. faithful ProphetKV mask) byte-unchanged. Only stores CPU copies.
        if os.environ.get("SGLANG_PROPHETKV_PROBE_PERLAYER_DIR"):
            # store FULL-context alpha (sys+doc), not just doc slice, so offline can compute
            # both doc-only (gold/right-panel) AND global position-0 sink share (left-panel).
            _pl = rt.setdefault("pkv_perlayer_alpha", {})
            _pl.setdefault(req_idx, []).append((int(layer.layer_id), alpha.float().cpu()))
        # ============================================================================


def _prophetkv_finalize(forward_batch):
    """After the model layer loop (still INSIDE the timed pass1 forward): uniform mean-fuse the
    per-layer alpha (Eq7) -> top-r -> inline external-replay mask JSON for pass2. Optionally dump
    the fused alpha for the offline-builder oracle cross-check (gate: inline-mask==offline-mask).
    The ONLY thing excluded from the ProphetKV pass1 selector cost = the mask-file write I/O
    (two-launch artifact). No-op unless pass1 accumulated alpha."""
    rt = getattr(forward_batch, "cacheflex_runtime", None)
    if not rt or not rt.get("pkv_alpha_sum"):
        return
    import json as _json, math as _math, re as _re
    L = max(1, int(rt.get("pkv_layer_count", 1)))
    rate = float(os.environ.get("SGLANG_PROPHETKV_RATE", "0.15"))
    mask_dir = os.environ.get("SGLANG_PROPHETKV_MASK_DIR")
    alpha_dir = os.environ.get("SGLANG_PROPHETKV_ALPHA_DIR")   # optional: dump fused alpha for x-check
    sess = getattr(forward_batch, "debug_dump_name", None) or "probe"
    _m = _re.search(r"sample(\d+)", sess)
    sidx = int(_m.group(1)) if _m else 0
    alpha_sum = rt["pkv_alpha_sum"]; doc_meta = rt.get("pkv_doc_meta", {})
    for req_idx, asum in alpha_sum.items():
        abar = (asum.float() / L).cpu()                  # [doc_len] fused importance (uniform mean)
        doc_start, doc_end = doc_meta.get(req_idx, (0, int(abar.numel())))
        n = int(abar.numel())
        # HACK(Shizhuo) [PKV-QIP] 统一预算公式 + doc/query 切分 ======================
        # off (default): ksel = ceil(rate * doc_len), selection over the doc region only.
        # Variant on: n 已是 pool = doc_len + q_reuse_len; 预算改用与 KVFocus / CacheBlend
        #   完全相同的 k = max(int(r * pool), 1) (KVFocusSelector / CacheBlendSelector), 即分母、
        #   取整、下限三者全部对齐。选完再按 doc_len 切回两段。
        _split = rt.get("pkv_qip_split", {}).get(req_idx)
        if _split is not None:
            _dlen, _qrlen, _nq = _split
            if _dlen + _qrlen != n:
                raise ValueError(
                    f"[PKV-QIP] split mismatch: doc {_dlen} + query-reuse {_qrlen} != "
                    f"scored {n} (req {req_idx})")
            ksel = max(1, int(rate * n))
        else:
            ksel = max(1, _math.ceil(rate * n))
        mask_doc = torch.zeros(n, dtype=torch.bool)
        if ksel >= n:
            mask_doc[:] = True
        else:
            mask_doc[torch.topk(abar, ksel).indices] = True
        mask_qre = None
        if _split is not None:
            mask_qre = mask_doc[_dlen:]
            mask_doc = mask_doc[:_dlen]
            n = _dlen
        # =========================================================================
        if alpha_dir:
            try:
                os.makedirs(alpha_dir, exist_ok=True)
                torch.save({"alpha_bar": abar, "doc_start": int(doc_start), "doc_end": int(doc_end),
                            "layers": L, "rate": rate}, os.path.join(alpha_dir, f"{sess}__req{req_idx}__abar.pt"))
            except Exception as _e:
                _logging_module.getLogger("ProphetKVProbe").warning(f"alpha dump failed: {_e}")
        # HACK(Shizhuo) [Fig2-perlayer] ===============================================
        # dump per-layer alpha stack [L, N_ctx] (full sys+doc; doc_start/end mark reused) for Fig.2 obs.
        _pldir = os.environ.get("SGLANG_PROPHETKV_PROBE_PERLAYER_DIR")
        _plall = rt.get("pkv_perlayer_alpha", {})
        if _pldir and _plall.get(req_idx):
            try:
                os.makedirs(_pldir, exist_ok=True)
                _rows = sorted(_plall[req_idx], key=lambda x: x[0])
                _stack = torch.stack([a for _, a in _rows], dim=0)   # [L, doc_len]
                torch.save({"alpha_per_layer": _stack, "layer_ids": [li for li, _ in _rows],
                            "doc_start": int(doc_start), "doc_end": int(doc_end)},
                           os.path.join(_pldir, f"{sess}__req{req_idx}__perlayer.pt"))
            except Exception as _e:
                _logging_module.getLogger("ProphetKVProbe").warning(f"perlayer dump failed: {_e}")
        # ============================================================================
        if mask_dir and req_idx == 0:   # one req per sample in this eval; mask file keyed by sample idx
            try:
                os.makedirs(mask_dir, exist_ok=True)
                # docs_mask = [False]*sys_len + doc_topk_bools ; consumer appends [True]*query_len.
                docs_mask = [False] * int(doc_start) + [bool(x) for x in mask_doc.tolist()]
                _rec = {"docs_mask": docs_mask, "docs_len": len(docs_mask),
                        "doc_region_len": int(n), "selected": int(mask_doc.sum().item()),
                        "sys_len": int(doc_start), "variant": "prophetkv"}
                # HACK(Shizhuo) [PKV-QIP] 额外写 query_mask, 供 mask 消费端使用。
                # 长度 = 完整 query 长度 n_q; 末 (n_q - q_reuse_len) 位 = suffix, 全臂共有
                # 的必重算段, 恒 True 且不占预算。不开该变体时不写此字段（query 视为全 True）。
                if mask_qre is not None:
                    _sfx = int(_nq) - int(_qrlen)
                    if _sfx < 0:
                        raise ValueError(
                            f"[PKV-QIP] suffix len negative: n_q={_nq} q_reuse={_qrlen}")
                    _rec["query_mask"] = [bool(x) for x in mask_qre.tolist()] + [True] * _sfx
                    _rec["query_selected"] = int(mask_qre.sum().item())
                    _rec["pool"] = int(_dlen) + int(_qrlen)
                    _rec["budget_k"] = int(ksel)
                    _rec["variant"] = "prophetkv-e1"
                with open(os.path.join(mask_dir, f"sample_{sidx}.json"), "w") as f:
                    _json.dump(_rec, f)
            except Exception as _e:
                _logging_module.getLogger("ProphetKVProbe").warning(f"mask write failed: {_e}")
    # free running state so it doesn't leak across samples (CLEAR_CACHE_EACH_SAMPLE re-inits anyway)
    rt.pop("pkv_alpha_sum", None); rt.pop("pkv_doc_meta", None); rt.pop("pkv_layer_count", None)
    rt.pop("pkv_qip_split", None)   # HACK(Shizhuo) [PKV-QIP] 同上, 防跨 sample 泄漏


class AttentionType(Enum):
    """
    Attention type.
    Use string to be compatible with `torch.compile`.
    """

    # Decoder attention between previous layer Q/K/V
    DECODER = "decoder"
    # Encoder attention between previous layer Q/K/V
    ENCODER_ONLY = "encoder_only"


class RadixAttention(nn.Module):
    """
    The attention layer implementation.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scaling: float,
        num_kv_heads: int,
        layer_id: int,
        logit_cap: float = 0.0,
        v_head_dim: int = -1,
        sliding_window_size: int = -1,
        is_cross_attention: bool = False,
        pos_encoding_mode: str = "NONE",
        logit_capping_method: str = "tanh",
        quant_config: Optional[QuantizationConfig] = None,
        attn_type: AttentionType = AttentionType.DECODER,
        use_irope: bool = False,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_q_head_num = num_heads
        self.tp_k_head_num = num_kv_heads
        self.tp_v_head_num = num_kv_heads
        self.head_dim = head_dim
        self.qk_head_dim = head_dim
        self.v_head_dim = v_head_dim if v_head_dim != -1 else head_dim
        self.scaling = scaling
        self.layer_id = layer_id
        self.logit_cap = logit_cap
        self.sliding_window_size = sliding_window_size or -1
        self.is_cross_attention = is_cross_attention
        self.use_irope = use_irope
        self.k_scale = None
        self.v_scale = None
        self.k_scale_float = None
        self.v_scale_float = None
        self.quant_method = None

        if quant_config is not None:
            self.quant_method = quant_config.get_quant_method(self, prefix=prefix)
        if self.quant_method is not None:
            self.quant_method.create_weights(self)
        self.attn_type = attn_type

        self.pos_encoding_mode = pos_encoding_mode
        self.logit_capping_method = logit_capping_method
        self.xai_temperature_len = -1

    def forward(
        self,
        q,
        k,
        v,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        # HACK(Shizhuo)
        rotary_emb: Optional[nn.Module] = None,
        **kwargs,
    ):
        if k is not None:
            # For cross-layer sharing, kv can be None
            assert v is not None
            if "k_rope" not in kwargs:
                k = k.view(-1, self.tp_k_head_num, self.qk_head_dim)
                v = v.view(-1, self.tp_v_head_num, self.v_head_dim)
            else:
                k = k.view(-1, self.tp_k_head_num, self.v_head_dim)
        
        # HACK(Shizhuo): [CacheFlex] Trigger Copy
        _harm_reuse_data = None  # HACK(Shizhuo): [KVFocus Harm Observation]
        _copy_time = 0.0
        _sync_time = 0.0

        # HACK(Shizhuo)
        # ============================================
        # [KVFOCUS_PROFILE] CUDA event timing for attention sub-steps
        _do_attn_profile = (_cf_dump.profile_timing_enabled()
                            and forward_batch.forward_mode.is_extend()
                            and getattr(forward_batch, "cacheflex_configs", None))
        if _do_attn_profile:
            _attn_ev = {}

        # Block A: complementary_copy (can be skipped for Contiguous P0)
        # HACK(Shizhuo): SGLANG_FORCE_SKIP_COMPLEMENTARY_COPY=1 forcibly skips Block A
        # for diagnostic — tests if complementary_copy overwrites fresh K at r=1.0.
        _force_skip_cc = os.environ.get("SGLANG_FORCE_SKIP_COMPLEMENTARY_COPY", "0") == "1"
        if (forward_batch.forward_mode.is_extend()
            and getattr(forward_batch, "cacheflex_slice_info", None) is not None
            and not getattr(forward_batch, "cacheflex_skip_complementary_copy", False)
            and not _force_skip_cc):
            if _do_attn_profile:
                _attn_ev["copy_start"] = torch.cuda.Event(enable_timing=True)
                _attn_ev["copy_start"].record()
            from sglang.srt.server_args import get_global_server_args
            server_args = get_global_server_args()
            # TODO(Shizhuo) page_size>1逻辑
            page_size = server_args.page_size

            # 确保获取到 rotary_emb
            r_emb = rotary_emb or kwargs.get("rotary_emb") or getattr(self, "rotary_emb", None)

            if r_emb:
                _do_profile = _cf_dump.profile_timing_enabled()
                if _do_profile:
                    torch.cuda.synchronize()
                    _t0 = time.perf_counter()

                from sglang.srt.layers.attention.triton_ops.cacheflex import cacheflex_complementary_copy_launcher_kernel
                cacheflex_complementary_copy_launcher_kernel(
                    slice_info=forward_batch.cacheflex_slice_info,
                    kv_pool=forward_batch.token_to_kv_pool,
                    rotary_emb=r_emb,
                    layer_id=self.layer_id
                )

                if _do_profile:
                    torch.cuda.synchronize()
                    _copy_time = (time.perf_counter() - _t0) * 1000

                # HACK(Shizhuo)
                # ============================================
                # [Contiguous KV] Sync complementary_copy writes to contiguous buffer
                if (getattr(forward_batch, "cacheflex_kv_mode", "paged") == "contiguous"
                    and getattr(forward_batch, "cacheflex_contiguous_kv", None) is not None
                    and self.layer_id in forward_batch.cacheflex_contiguous_kv):
                    if _do_profile:
                        torch.cuda.synchronize()
                        _t1 = time.perf_counter()

                    from sglang.srt.layers.attention.cacheflex_utils import sync_complementary_copy_to_contiguous
                    sync_complementary_copy_to_contiguous(
                        forward_batch, forward_batch.cacheflex_slice_info, self.layer_id
                    )

                    if _do_profile:
                        torch.cuda.synchronize()
                        _sync_time = (time.perf_counter() - _t1) * 1000

                if _do_profile:
                    if not hasattr(forward_batch, "_cacheflex_copy_profile"):
                        forward_batch._cacheflex_copy_profile = []
                    forward_batch._cacheflex_copy_profile.append({
                        "layer": self.layer_id,
                        "copy_ms": _copy_time,
                        "sync_ms": _sync_time,
                    })
            else:
                # 如果是 CacheFlex 请求但没有 Rotary，这是一个 Bug，应该打印错误
                import logging
                logger = logging.getLogger(__name__)
                logger.error("CacheFlex triggered but rotary_emb is missing!")

            if _do_attn_profile:
                _attn_ev["copy_end"] = torch.cuda.Event(enable_timing=True)
                _attn_ev["copy_end"].record()

        # HACK(Shizhuo) [ProphetKV-Probe]: all-layer reused-K U probe (forward-free).
        # Per-layer gather(self.layer_id) + query->ctx scoring + dump. Naive-KV forward =>
        # no dynamic tasks => Block B below is skipped (no scorer/selection/sparsify). Flag/marker
        # gated => OFF by default => KVFocus & all other paths byte-unchanged.
        if (forward_batch.forward_mode.is_extend()
                and getattr(forward_batch, "cacheflex_configs", None)
                and _prophetkv_probe_enabled(forward_batch)):
            try:
                _prophetkv_probe_step(
                    self, forward_batch,
                    rotary_emb or kwargs.get("rotary_emb") or getattr(self, "rotary_emb", None),
                    q,
                )
            except Exception as _probe_e:
                _logging_module.getLogger("ProphetKVProbe").error(
                    f"probe step failed at layer {self.layer_id}: {_probe_e}", exc_info=True
                )

        # Block B: Check layer scoring + mask selection
        # Must run even when complementary_copy is skipped (Contiguous P0)
        # SGLANG_FORCE_SKIP_SCORING=1 — diagnostic: skips scoring entirely (no global_mask
        # written, no sparse slicing). Tests if scoring block has side effects.
        _force_skip_scoring = os.environ.get("SGLANG_FORCE_SKIP_SCORING", "0") == "1"
        # HACK(Shizhuo) [Oracle-Dump]: 去掉原来的 `"cacheflex_global_mask" not in runtime`
        # gate。external+check_layer 模式下 global_mask 由 init_new 预注入，需要在 check_layer
        # 仍进入本块完成稀疏化。内层 `check_layer == layer_id` gate 保证只在 check_layer 执行一次；
        # 非 check_layer 层 sample_task 不匹配，no-op。external(check_layer=-1) 不注册 dynamic
        # task → cacheflex_dynamic_tasks_by_req 空 → 不进块, 向后兼容。
        if (forward_batch.forward_mode.is_extend()
            and getattr(forward_batch, "cacheflex_slice_info", None) is not None
            and forward_batch.cacheflex_dynamic_tasks_by_req
            and not _force_skip_scoring):
            # Robust task finding
            tasks_by_req = forward_batch.cacheflex_dynamic_tasks_by_req
            sample_task = None
            for tasks in tasks_by_req:
                if tasks:
                    sample_task = tasks[0]
                    break

            if sample_task and self.layer_id == sample_task["check_layer"]:
                if _do_attn_profile:
                    _attn_ev["scorer_start"] = torch.cuda.Event(enable_timing=True)
                    _attn_ev["scorer_start"].record()
                # HACK(Shizhuo) [Oracle-Dump]: external+check_layer 模式 —— global_mask 已由
                # init_new 从 custom_params["external_mask"] 预注入。跳过 scorer/selector，
                # 直接用预注入 mask 做 check_layer 稀疏化。
                _preset_mask = forward_batch.cacheflex_runtime.get("cacheflex_global_mask")
                if _preset_mask is not None and _preset_mask.shape[0] == v.shape[0]:
                    global_mask = _preset_mask
                    _skip_scorer = True
                elif _preset_mask is not None and _preset_mask.shape[0] > v.shape[0]:
                    # HACK(Shizhuo) [Oracle-Dump]: 注入的 external_mask 含 naive-reused 前缀
                    # (如主 eval 里 sys 源 method=naive, 不进 compute-all extend)。compute-all
                    # 的 token 集是 [..., docs, query] (排掉前面被 naive 复用的)，取 mask 末尾
                    # v.shape[0] 个对齐。两个 eval (sys=external / sys=naive) 都兼容。
                    global_mask = _preset_mask[-v.shape[0]:].contiguous()
                    _skip_scorer = True
                else:
                    # Init Global Mask (True=Keep)
                    # 注意：v 的 shape 是 [TotalTokens, H, D]
                    global_mask = torch.ones(v.shape[0], dtype=torch.bool, device=v.device)
                    _skip_scorer = False

                # 遍历 Batch 里的每个 Req
                # 注意：这里的 v 是全局 flatten 的，我们需要根据 extend_start_loc 定位
                start_loc = forward_batch.extend_start_loc

                # Check if any task has kvfocus enabled
                has_kvfocus = any(task.get("kvfocus", False) for tasks in tasks_by_req for task in tasks)

                if _skip_scorer:
                    # HACK(Shizhuo) [Oracle-Dump]: external 预注入 mask, 不算 scorer/selector,
                    # global_mask 已是预注入 mask, 直接进下面的稀疏化。
                    pass
                elif has_kvfocus:
                    # ============================================================
                    # [KVFocus Hook] sqrt(M) × U_L2 score → Global Top-K across all segments
                    # ============================================================
                    import logging as _logging
                    _kvfocus_logger = _logging.getLogger("KVFocus")

                    # Determine num_probes from cacheflex_configs (default=5)
                    # Note: cacheflex_configs carries num_probes from the
                    # eval script via custom_params["cacheflex"]["num_probes"].
                    # Reading from cacheflex_configs preserves backward compatibility.
                    num_probes = 5
                    if getattr(forward_batch, "cacheflex_configs", None):
                        for cfg in forward_batch.cacheflex_configs:
                            if cfg and "num_probes" in cfg:
                                num_probes = cfg["num_probes"]
                                break

                    # [Scorer] Compute per-token M×U scores.
                    # HACK(Shizhuo) [KVFocus]
                    from sglang.srt.layers.attention.token_selector import create_scorer_from_tasks
                    _flat_tasks = [t for _tasks in tasks_by_req for t in _tasks]
                    scorer = create_scorer_from_tasks(_flat_tasks, has_kvfocus=True)
                    scorer_results = scorer.compute(
                        v=v, q=q, k=k,
                        tasks_by_req=tasks_by_req,
                        start_loc=start_loc,
                        forward_batch=forward_batch,
                        layer=self,
                        num_probes=num_probes,
                    )

                    # Apply drop_all masks from ScorerResult
                    for req_idx, sr in scorer_results.items():
                        for seg_idx, stats in enumerate(sr.seg_log_stats):
                            if stats is None:  # drop_all segment
                                task = tasks_by_req[req_idx][seg_idx]
                                abs_start = start_loc[req_idx] + task["rel_start"]
                                length = task["src_len"]
                                global_mask[abs_start : abs_start + length] = False
                            elif sr.m_diff_list[seg_idx] is None:  # no diff → reuse all
                                abs_start = stats["abs_start"]
                                length = stats["length"]
                                global_mask[abs_start : abs_start + length] = False

                    # Per-request: dump + selection + mask application
                    u_scores_l1 = forward_batch.cacheflex_runtime["kvfocus_u_scores_l1"]
                    u_scores_l2 = forward_batch.cacheflex_runtime["kvfocus_u_scores_l2"]

                    for req_idx, tasks in enumerate(forward_batch.cacheflex_dynamic_tasks_by_req):
                        if req_idx not in scorer_results:
                            continue
                        sr = scorer_results[req_idx]
                        req_start = start_loc[req_idx]
                        u_tensor_l1 = u_scores_l1.get(req_idx)
                        u_tensor_l2 = u_scores_l2.get(req_idx)

                        # [Dump] Per-token M×U_L2 scores for auto analysis
                        _kvfocus_dump_sid = -1
                        _dump_extended = _cf_dump.is_enabled(_cf_dump.Category.KVFOCUS_EXTENDED)
                        if _cf_dump.is_enabled(_cf_dump.Category.KVFOCUS_SCORES):
                            seg_scores_list = []
                            for seg_i, (abs_start, valid_len, seg_idx) in enumerate(sr.mapping):
                                m = sr.token_scores_list[seg_i]  # [valid_len] M×U_L2
                                u_l2_val = u_tensor_l2[seg_idx].item() if u_tensor_l2 is not None and seg_idx < len(u_tensor_l2) else 0.0
                                u_l1_val = u_tensor_l1[seg_idx].item() if u_tensor_l1 is not None and seg_idx < len(u_tensor_l1) else 0.0
                                # Recover raw M from M×U_L2
                                m_raw = sr.m_diff_list[seg_idx] if seg_idx < len(sr.m_diff_list) and sr.m_diff_list[seg_idx] is not None else m
                                seg_dict = {
                                    "m_per_token": m_raw,              # ||ΔV||₂ per token [valid_len]
                                    "u_l2": u_l2_val,                  # segment-level probe attention L2 mass
                                    "u_l1": u_l1_val,                  # segment-level probe attention L1 mass
                                    "mxu_l2_per_token": m,             # M×U_L2 per token [valid_len]
                                    "valid_len": valid_len,
                                    "src_len": sr.seg_log_stats[seg_idx]["length"] if seg_idx < len(sr.seg_log_stats) and sr.seg_log_stats[seg_idx] else valid_len,
                                }
                                # Extended per-token fields (gated by env var)
                                if _dump_extended:
                                    if seg_idx < len(sr.m_diff_l1_list) and sr.m_diff_l1_list[seg_idx] is not None:
                                        seg_dict["m_l1_per_token"] = sr.m_diff_l1_list[seg_idx].cpu()
                                    if seg_idx < len(sr.u_l1_per_token_list) and sr.u_l1_per_token_list[seg_idx] is not None:
                                        seg_dict["u_l1_per_token"] = sr.u_l1_per_token_list[seg_idx].cpu()
                                    if seg_idx < len(sr.u_l2_per_token_list) and sr.u_l2_per_token_list[seg_idx] is not None:
                                        seg_dict["u_l2_per_token"] = sr.u_l2_per_token_list[seg_idx].cpu()
                                    if seg_idx < len(sr.suffix_u_l1_per_token_list) and sr.suffix_u_l1_per_token_list[seg_idx] is not None:
                                        seg_dict["suffix_u_l1_per_token"] = sr.suffix_u_l1_per_token_list[seg_idx].cpu()
                                    if seg_idx < len(sr.suffix_u_l2_per_token_list) and sr.suffix_u_l2_per_token_list[seg_idx] is not None:
                                        seg_dict["suffix_u_l2_per_token"] = sr.suffix_u_l2_per_token_list[seg_idx].cpu()
                                seg_scores_list.append(seg_dict)
                            req_dump_data = {
                                "segments": seg_scores_list,
                                "total_cached_tokens": sr.total_cached_tokens,
                                "recomp_ratio": sr.recomp_ratio,
                                "num_segments": len(seg_scores_list),
                            }
                            _kvfocus_dump_sid = _dump_kvfocus_scores(
                                self.layer_id, req_dump_data,
                                session=getattr(forward_batch, "debug_dump_name", None),
                            )

                        # Pass 2: Token selection (auto-rate or fixed ratio)
                        # Adaptive solvers require 2-pass attention; topk uses fixed ratio
                        solver_val = tasks[0].get("solver", "residual_budget") if tasks else "residual_budget"
                        is_adaptive = solver_val in ("residual_budget", "coverage", "threshold")
                        lam_val = None
                        for task in tasks:
                            if task.get("lam") is not None:
                                lam_val = task["lam"]
                                break

                        if is_adaptive and sr.token_scores_list:
                            # ============================================================
                            # [KVFocus-Adaptive] Adaptive Threshold over Normalized M×U
                            # 2-pass (reuse_full/warmup_full) or 3-pass (pass_mode="3pass")
                            # ============================================================
                            tp_result = run_two_pass_attention(
                                q, k, v, forward_batch, self, tasks, req_idx, sr,
                                num_probes, global_mask, **kwargs
                            )

                            if tp_result is not None:
                                # Save output + skip flag + recompute_rate + probe_residual data
                                forward_batch.cacheflex_runtime["_kvfocus_output"] = tp_result["output_full"]
                                forward_batch.cacheflex_runtime["_skip_production_attention"] = True
                                forward_batch.cacheflex_runtime["recompute_rate"] = tp_result["recompute_rate"]
                                forward_batch.cacheflex_runtime["probe_residual"] = tp_result["probe_residual"]
                                forward_batch.cacheflex_runtime["probe_residual"]["H"] = tp_result["H"]

                                # [Dump Extended] Per-token output diff
                                if _dump_extended and _kvfocus_dump_sid >= 0:
                                    pr = tp_result["probe_residual"]
                                    if "per_token_output_diff" in pr:
                                        seg_boundaries = [
                                            (t["rel_start"], t["src_len"]) for t in tasks
                                        ]
                                        _dump_kvfocus_output_diff(
                                            self.layer_id, _kvfocus_dump_sid,
                                            pr["per_token_output_diff"],
                                            segment_boundaries=seg_boundaries,
                                            session=getattr(forward_batch, "debug_dump_name", None),
                                        )

                        elif sr.token_scores_list and sr.total_cached_tokens > 0:
                            # ============================================================
                            # [KVFocus] Fixed Ratio Mode — Global Top-K (via KVFocusSelector)
                            # ============================================================
                            from sglang.srt.layers.attention.token_selector import KVFocusSelector
                            selector = KVFocusSelector(recomp_ratio=sr.recomp_ratio)
                            result = selector.select(scorer_result=sr)

                            # Apply mask: mark all cached tokens as False first
                            for abs_start, valid_len, _ in sr.mapping:
                                global_mask[abs_start : abs_start + valid_len] = False
                            # Mark selected tokens as True (vectorized, zero CPU sync)
                            for abs_start, local_indices in result["global_mask_updates"]:
                                global_mask[abs_start + local_indices] = True

                            # Record recompute_rate for profiling
                            if not hasattr(forward_batch, "cacheflex_runtime") or forward_batch.cacheflex_runtime is None:
                                forward_batch.cacheflex_runtime = {}
                            forward_batch.cacheflex_runtime["recompute_rate"] = result["recompute_rate"]

                        # [Dump] Global mask after selection (shared by auto and fixed-ratio)
                        if _kvfocus_dump_sid >= 0 and sr.mapping:
                            seg_mask_data = []
                            for abs_start, valid_len, seg_idx in sr.mapping:
                                seg_mask = global_mask[abs_start : abs_start + valid_len]
                                seg_mask_data.append({
                                    "mask": seg_mask,         # bool tensor [valid_len], True=recompute
                                    "num_recompute": int(seg_mask.sum().item()),
                                    "valid_len": valid_len,
                                })
                            _dump_kvfocus_mask(
                                self.layer_id, seg_mask_data, _kvfocus_dump_sid,
                                session=getattr(forward_batch, "debug_dump_name", None),
                            )

                        # Logging (shared by auto and fixed-ratio)
                        if _kvfocus_logger.isEnabledFor(_logging.INFO):
                            total_tokens = 0
                            total_recompute = 0
                            for seg_idx, stats in enumerate(sr.seg_log_stats):
                                if stats is None:
                                    continue
                                abs_start = stats["abs_start"]
                                length = stats["length"]
                                recompute_count = global_mask[abs_start : abs_start + stats["valid_len"]].sum().item()
                                total_tokens += length
                                total_recompute += recompute_count

                                _kvfocus_logger.info(
                                    f"[KVFocus] Req{req_idx} Seg{seg_idx}: "
                                    f"rel_start={abs_start - req_start}, src_len={length}, "
                                    f"recomp_ratio={sr.recomp_ratio:.2f}, "
                                    f"recompute={recompute_count}/{stats['valid_len']} ({recompute_count/max(stats['valid_len'],1)*100:.1f}%), "
                                    f"U_L1={stats['u_l1']:.4f}, U_L2={stats['u_l2']:.4f}, "
                                    f"M_mean={stats['m_mean']:.4f}, M_max={stats['m_max']:.4f}"
                                )

                            if total_tokens > 0:
                                total_reuse = total_tokens - total_recompute
                                _kvfocus_logger.info(
                                    f"[KVFocus] Req{req_idx} Summary: "
                                    f"total={total_tokens}, recompute={total_recompute} ({total_recompute/total_tokens*100:.1f}%), "
                                    f"reuse={total_reuse} ({total_reuse/total_tokens*100:.1f}%)"
                                )

                else:
                    # ============================================================
                    # [CacheBlend Hook] M score only - Per-segment Top-K + keep-last
                    # ============================================================
                    from sglang.srt.layers.attention.token_selector import CacheBlendScorer, CacheBlendSelector

                    # [Scorer] Compute per-token M scores (no U multiplier)
                    cb_scorer = CacheBlendScorer()
                    cb_scorer_results = cb_scorer.compute(
                        v=v,
                        tasks_by_req=tasks_by_req,
                        start_loc=start_loc,
                        forward_batch=forward_batch,
                        layer=self,
                    )

                    for req_idx, tasks in enumerate(forward_batch.cacheflex_dynamic_tasks_by_req):
                        if req_idx not in cb_scorer_results:
                            continue
                        sr = cb_scorer_results[req_idx]
                        req_start = start_loc[req_idx]

                        # Apply drop_all masks
                        for seg_idx, stats in enumerate(sr.seg_log_stats):
                            if stats is None:  # drop_all segment
                                task = tasks[seg_idx]
                                abs_start = req_start + task["rel_start"]
                                length = task["src_len"]
                                global_mask[abs_start : abs_start + length] = False

                        # [Selector] Per-segment Top-K + keep-last
                        if sr.token_scores_list and sr.total_cached_tokens > 0:
                            cb_selector = CacheBlendSelector(recomp_ratio=sr.recomp_ratio)
                            result = cb_selector.select(scorer_result=sr)

                            # Mark all cached tokens as False first
                            for abs_start, valid_len, _ in sr.mapping:
                                global_mask[abs_start : abs_start + valid_len] = False
                            # Mark selected tokens as True (vectorized, zero CPU sync)
                            for abs_start, local_indices in result["global_mask_updates"]:
                                global_mask[abs_start + local_indices] = True

                            # Record recompute_rate for profiling
                            if not hasattr(forward_batch, "cacheflex_runtime") or forward_batch.cacheflex_runtime is None:
                                forward_batch.cacheflex_runtime = {}
                            forward_batch.cacheflex_runtime["recompute_rate"] = result["recompute_rate"]

                # 保存 Mask
                forward_batch.cacheflex_runtime["cacheflex_global_mask"] = global_mask

                if _do_attn_profile:
                    _attn_ev["scorer_end"] = torch.cuda.Event(enable_timing=True)
                    _attn_ev["scorer_end"].record()

                # [Sparse Check Layer] Filter Q/K/V to only recompute tokens,
                # like vLLM's query = query[top_indices]. Attention becomes
                # sparse Q × full KV (K/V written to paged cache, kernel reads
                # full KV via page table). MLP also processes only sparse tokens.
                # Apply for CacheBlend and KVFocus-fixed-rate (not KVFocus-Adaptive
                # which already computed attention in run_two_pass_attention).
                if not forward_batch.cacheflex_runtime.get("_skip_production_attention"):
                    if _do_attn_profile:
                        _attn_ev["sparsify_start"] = torch.cuda.Event(enable_timing=True)
                        _attn_ev["sparsify_start"].record()
                    keep_count = global_mask.sum()  # GPU scalar
                    total_count = global_mask.shape[0]
                    if keep_count < total_count:  # 1 implicit sync (__bool__)
                        keep_count = int(keep_count)  # cache as CPU int, avoid 2nd sync
                        # Save mask for llama.py to slice residual
                        forward_batch.cacheflex_runtime["_check_layer_global_mask"] = global_mask
                        forward_batch.cacheflex_runtime["_check_layer_sparse_slicing_done"] = False
                        forward_batch.cacheflex_runtime["_check_layer_id"] = self.layer_id

                        # Filter Q/K/V (already [N, H, D] shaped)
                        q = q[global_mask]
                        k = k[global_mask]
                        v = v[global_mask]

                        # Filter out_cache_loc and positions
                        forward_batch.out_cache_loc = forward_batch.out_cache_loc[global_mask]
                        forward_batch.positions = forward_batch.positions[global_mask]

                        # Rebuild per-request seq_lens and start_loc
                        old_start_loc = forward_batch.extend_start_loc
                        old_seq_lens = forward_batch.seq_lens

                        # Save original metadata for logprobs calculation in
                        # _apply_post_check_slicing fast path
                        forward_batch.cacheflex_runtime["_pre_sparse_start_loc"] = old_start_loc.clone()
                        forward_batch.cacheflex_runtime["_pre_sparse_seq_lens"] = old_seq_lens.clone()

                        # Vectorized: count recompute tokens per request on GPU
                        # replaces Python loop + .item() synchronization
                        req_idx_per_token = torch.repeat_interleave(
                            torch.arange(forward_batch.batch_size, device=global_mask.device, dtype=torch.long),
                            old_seq_lens.to(torch.long)
                        )
                        new_seq_lens = torch.bincount(
                            req_idx_per_token[global_mask],
                            minlength=forward_batch.batch_size
                        ).to(torch.int64)

                        # DEBUG: verify bincount matches loop-based computation
                        if os.environ.get("SGLANG_CACHEFLEX_VERIFY_SPARSE") == "1":
                            old_result = []
                            for req_idx in range(forward_batch.batch_size):
                                start = old_start_loc[req_idx].item()
                                length = old_seq_lens[req_idx].item()
                                keep = global_mask[start : start + length].sum().item()
                                old_result.append(keep)
                            assert new_seq_lens.tolist() == old_result, \
                                f"Mismatch: {new_seq_lens.tolist()} vs {old_result}"

                        forward_batch.seq_lens = new_seq_lens
                        forward_batch.extend_seq_lens = new_seq_lens

                        new_start_loc = torch.zeros_like(old_start_loc)
                        new_start_loc[1:] = torch.cumsum(new_seq_lens[:-1], dim=0)
                        forward_batch.extend_start_loc = new_start_loc

                        # Sync CPU-side metadata (used by logprobs logic)
                        # .tolist() is unavoidable: extend_seq_lens_cpu must be Python list
                        forward_batch.extend_seq_lens_cpu = new_seq_lens.tolist()

                        # Reinit backend metadata (qo_indptr, etc.)
                        if hasattr(forward_batch, "attn_backend") and forward_batch.attn_backend:
                            forward_batch.attn_backend.init_forward_metadata(forward_batch)

                        if _do_attn_profile:
                            _attn_ev["sparsify_end"] = torch.cuda.Event(enable_timing=True)
                            _attn_ev["sparsify_end"].record()

                        _logging_module.getLogger("KVFocus").info(
                            f"[CacheFlex Sparse Check Layer] L{self.layer_id}: "
                            f"{total_count} -> {keep_count} tokens "
                            f"({keep_count/total_count*100:.1f}%)"
                        )

                # HACK(Shizhuo): [KVFocus Harm Observation] Steps 3-4
                # Prepare pool and run reuse attention BEFORE production full attention.
                # Must happen AFTER cacheflex_complementary_copy (Step 1) and
                # KVFocus Hook (Step 2), but BEFORE save_kv_cache.
                # Skip if auto-rate already computed attention (pool state would be corrupted)
                if _cf_dump.is_enabled(_cf_dump.Category.HARM) and not forward_batch.cacheflex_runtime.get("_skip_production_attention"):
                    try:
                        from sglang.srt.layers.attention.kvfocus_harm_observe import (
                            run_harm_observation_reuse,
                        )
                        _harm_reuse_data = run_harm_observation_reuse(
                            q=q, k=k, v=v,
                            forward_batch=forward_batch,
                            layer=self,
                            tasks_by_req=tasks_by_req,
                        )
                    except Exception as e:
                        import traceback
                        _logging_module.getLogger("KVFocus.HarmObserve").error(
                            f"Harm observation reuse pass failed at layer {self.layer_id}: {e}\n{traceback.format_exc()}"
                        )
        # ============================================================================

        # HACK(Shizhuo)
        # ============================================
        # [KVFOCUS_PROFILE] Attention sub-step timing: attn phase
        if _do_attn_profile:
            _attn_ev["attn_start"] = torch.cuda.Event(enable_timing=True)
            _attn_ev["attn_start"].record()

        # [KVFocus-Adaptive] If attention was already computed in the Hook, skip production attention
        if getattr(forward_batch, "cacheflex_runtime", None) and forward_batch.cacheflex_runtime.get("_skip_production_attention"):
            output = forward_batch.cacheflex_runtime.pop("_kvfocus_output")
            forward_batch.cacheflex_runtime.pop("_skip_production_attention", None)
            if _do_attn_profile:
                _attn_ev["attn_end"] = torch.cuda.Event(enable_timing=True)
                _attn_ev["attn_end"].record()
                if not hasattr(forward_batch, "_cacheflex_attn_events"):
                    forward_batch._cacheflex_attn_events = {}
                forward_batch._cacheflex_attn_events[self.layer_id] = _attn_ev
            return output

        if forward_batch.forward_mode.is_extend() and get_forward_context() is not None:
            if self.qk_head_dim != self.v_head_dim:
                output = q.new_empty((q.shape[0], self.tp_q_head_num * self.v_head_dim))
            else:
                output = torch.empty_like(q)
            torch.ops.sglang.unified_attention_with_output(
                q, k, v, output, save_kv_cache, self.layer_id, **kwargs
            )
            if _do_attn_profile:
                _attn_ev["attn_end"] = torch.cuda.Event(enable_timing=True)
                _attn_ev["attn_end"].record()
                if not hasattr(forward_batch, "_cacheflex_attn_events"):
                    forward_batch._cacheflex_attn_events = {}
                forward_batch._cacheflex_attn_events[self.layer_id] = _attn_ev
            return output
        else:
            output = forward_batch.attn_backend.forward(
                q,
                k,
                v,
                self,
                forward_batch,
                save_kv_cache,
                **kwargs,
            )

            # HACK(Shizhuo): [KVFocus Harm Observation] Step 6
            # After full attention (CacheFlex path), compute harm metrics
            if _harm_reuse_data is not None:
                try:
                    from sglang.srt.layers.attention.kvfocus_harm_observe import (
                        finalize_harm_metrics,
                        _get_env_config,
                    )
                    _, _, _probe_len = _get_env_config()
                    _tasks_by_req = getattr(forward_batch, "cacheflex_dynamic_tasks_by_req", None)
                    if _tasks_by_req:
                        finalize_harm_metrics(
                            output_full=output,
                            reuse_data=_harm_reuse_data,
                            forward_batch=forward_batch,
                            tasks_by_req=_tasks_by_req,
                            probe_len=_probe_len,
                        )
                except Exception as e:
                    import traceback
                    _logging_module.getLogger("KVFocus.HarmObserve").error(
                        f"Harm observation finalize failed at layer {self.layer_id}: {e}\n{traceback.format_exc()}"
                    )
            # ============================================================================
            if _do_attn_profile:
                _attn_ev["attn_end"] = torch.cuda.Event(enable_timing=True)
                _attn_ev["attn_end"].record()
                if not hasattr(forward_batch, "_cacheflex_attn_events"):
                    forward_batch._cacheflex_attn_events = {}
                forward_batch._cacheflex_attn_events[self.layer_id] = _attn_ev

            return output


def unified_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    save_kv_cache: bool,
    layer_id: int,
    *,
    q_rope: Optional[torch.Tensor] = None,
    k_rope: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
) -> None:
    # =========================================================================
    # HACK(Shizhuo) - 最小化 dump 用于实验分析
    # 只 dump 必要数据：k_cache_full, image_offsets, query_tokens, rope
    # attn_weights 从这些数据计算，不直接 dump
    # =========================================================================
    def dump_minimal_for_experiment(name, tensor, layer_id):
        if os.environ.get("SGLANG_ENABLE_MINIMAL_DUMP") != "true":
            return
        method = os.environ.get("DEBUG_METHOD", "unknown")
        # 使用绝对路径
        save_dir = os.path.abspath(f"debug_dump_full/{method}/layer_{layer_id}")
        os.makedirs(save_dir, exist_ok=True)
        try:
            file_path = os.path.join(save_dir, f"{name}.pt")
            torch.save(tensor.detach().cpu(), file_path)
            print(f"[DUMP] Saved {name} to {file_path} (shape: {tensor.shape})")
        except Exception as e:
            print(f"[DUMP ERROR] Failed to save {name}: {e}")
            import traceback
            traceback.print_exc()

    # 仅在 extend 模式下 dump（prefill 阶段有完整的视觉 tokens）
    is_extend_mode = forward_batch.forward_mode.is_extend()

    if is_extend_mode and os.environ.get("SGLANG_ENABLE_MINIMAL_DUMP") == "true":
        try:
            # 1. Dump query tokens（计算 attention 必需）
            # query: [extend_tokens, q_heads, head_dim]
            dump_minimal_for_experiment("query_tokens", query, layer_id)

            # 2. 获取完整 KV Cache
            k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer_id)
            # req_to_token: [max_num_reqs, max_context_len]，映射逻辑 KV 索引到物理索引
            req_to_token = forward_batch.req_to_token_pool.req_to_token
            req_pool_indices = forward_batch.req_pool_indices
            seq_lens = forward_batch.seq_lens

            # 3. 获取 batch 中所有请求的完整 KV（包括之前的缓存的）
            all_k_full = []
            for req_idx in range(seq_lens.shape[0]):
                req_len = seq_lens[req_idx]
                req_tokens = req_to_token[req_pool_indices[req_idx], :req_len]
                req_k = k_cache[req_tokens]  # [seq_len, kv_heads, head_dim]
                all_k_full.append(req_k)
            k_cache_full = torch.cat(all_k_full, dim=0)  # [total_kv_tokens, kv_heads, head_dim]
            dump_minimal_for_experiment("k_cache_full", k_cache_full, layer_id)

            # 4. Dump rotary embeddings（计算 attention 必需）
            if q_rope is not None:
                dump_minimal_for_experiment("q_rope", q_rope, layer_id)
            if k_rope is not None:
                dump_minimal_for_experiment("k_rope", k_rope, layer_id)

            # 5. 获取 image_offsets（每帧的 token 范围，必须知道才能划分帧）
            if hasattr(forward_batch, 'image_offsets'):
                dump_minimal_for_experiment("image_offsets", forward_batch.image_offsets, layer_id)
            elif hasattr(forward_batch, 'image_offset_range'):
                dump_minimal_for_experiment("image_offsets", forward_batch.image_offset_range, layer_id)

            # 6. 保存序列元数据
            metadata = {
                "total_kv_tokens": int(k_cache_full.shape[0]),
                "extend_tokens": int(query.shape[0]),
                "q_heads": int(attention_layer.tp_q_head_num),
                "kv_heads": int(attention_layer.tp_k_head_num),
                "head_dim": int(attention_layer.qk_head_dim),
                "scaling": float(attention_layer.scaling) if hasattr(attention_layer, 'scaling') else float(1.0 / (attention_layer.qk_head_dim ** 0.5)),
            }
            dump_minimal_for_experiment("metadata", torch.tensor([v for v in metadata.values()]), layer_id)

        except Exception as e:
            print(f"[DUMP WARNING] Attention dump failed at layer {layer_id}: {e}")
            import traceback
            traceback.print_exc()
    # =========================================================================

    context = get_forward_context()
    forward_batch = context.forward_batch
    attention_layers = context.attention_layers
    attention_layer = attention_layers[layer_id]

    kwargs = {}
    if q_rope is not None:
        kwargs["q_rope"] = q_rope
    if k_rope is not None:
        kwargs["k_rope"] = k_rope
    if sinks is not None:
        kwargs["sinks"] = sinks

    ret = forward_batch.attn_backend.forward(
        query, key, value, attention_layer, forward_batch, save_kv_cache, **kwargs
    )
    assert (
        output.numel() == ret.numel()
    ), f"Output tensor element mismatch: {output.numel()} != {ret.numel()}"

    output.view(ret.shape).copy_(ret)
    return


def unified_attention_with_output_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    save_kv_cache: bool,
    layer_id: int,
    *,
    q_rope: Optional[torch.Tensor] = None,
    k_rope: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
) -> None:
    return


direct_register_custom_op(
    op_name="unified_attention_with_output",
    op_func=unified_attention_with_output,
    mutates_args=["output"],
    fake_impl=unified_attention_with_output_fake,
)
