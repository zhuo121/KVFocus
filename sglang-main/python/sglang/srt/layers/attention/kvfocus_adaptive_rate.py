# HACK(Shizhuo) New
"""
KVFocus: Adaptive Token Selection for KV Cache Recomputation.

Implements decoupled scoring and selection:
  Scoring (determines H harm metric):
    scaling="linear":  H = g_N × (R0 + β·D_probe + η·D_peak)   [default]
    scaling="log":     H = s_N × (R0 + β·D_probe + η·D_peak)   [preserved, not in ablation]
    scaling="none":    H = R0 + β·D_probe + η·D_peak            [ablation]

  Selection (determines mask from token scores):
    solver="residual_budget":  top-p by residual harm budget ρ   [default]
    solver="coverage":         saturated coverage prefix scan     [ablation]
    solver="threshold":        CRH-threshold on π_i              [ablation]
    solver="topk":             fixed-rate top-k                   [ablation]

Where:
  π_i = s_i / Σ s_j           — token's share of total risk
  s_i = m_i × U_L2,d          — raw M×U score per token
  H = harm metric              — computed from R0, D_probe, D_peak
  |I| = total cached tokens   — normalizes per-token cost
"""

import math
import torch
from typing import List, Dict, Tuple, Optional, Set


def _compute_probe_metrics(
    d_probe_per_seg: List[float],
    eta: float,
    top_k_peak: int,
) -> Tuple[float, float]:
    """Extract D_probe (mean) and D_peak (top-K mean) from per-segment probe residuals."""
    D_probe = sum(d_probe_per_seg) / max(len(d_probe_per_seg), 1)
    if d_probe_per_seg:
        vals = sorted(d_probe_per_seg, reverse=True)
        K = min(top_k_peak, len(vals))
        D_peak = sum(vals[:K]) / max(K, 1)
    else:
        D_peak = 0.0
    return D_probe, D_peak


def compute_crh_harm(
    R0: float,
    d_probe_per_seg: List[float],
    num_segments: int,
    n_ref: int = 11,
    g_max: float = 2.5,
    beta: float = 1.0,
    eta: float = 0.0,
    top_k_peak: int = 3,
    eps: float = 1e-12,
) -> float:
    """Compute CRH harm: H = g_N * (R0 + beta * D_probe + eta * D_peak)."""
    g_N = min(num_segments / n_ref, g_max)
    D_probe, D_peak = _compute_probe_metrics(d_probe_per_seg, eta, top_k_peak)
    return g_N * (R0 + beta * D_probe + eta * D_peak)


def _map_global_to_per_segment(
    selected_global_indices: torch.Tensor,
    token_scores_list: List[torch.Tensor],
) -> Dict[int, Set[int]]:
    """Map global sorted indices back to {seg_idx: set(local_idx)}."""
    seg_boundaries = []
    cumulative = 0
    for seg_idx, scores in enumerate(token_scores_list):
        seg_len = scores.numel()
        seg_boundaries.append((seg_idx, cumulative, cumulative + seg_len))
        cumulative += seg_len

    selected_segs: Dict[int, Set[int]] = {}
    for gidx in selected_global_indices.tolist():
        for seg_idx, start, end in seg_boundaries:
            if start <= gidx < end:
                local_idx = gidx - start
                if seg_idx not in selected_segs:
                    selected_segs[seg_idx] = set()
                selected_segs[seg_idx].add(local_idx)
                break
    return selected_segs


def _add_floor_tokens(
    selected_segs: Dict[int, Set[int]],
    token_scores_list: List[torch.Tensor],
    b0: int,
    b0_mode: str,
    B: int,
) -> None:
    """Add floor tokens to selected_segs in-place after prefix scan selection.
    Ensures total selected count does not exceed B.
    """
    total = sum(len(v) for v in selected_segs.values())
    for seg_idx, scores in enumerate(token_scores_list):
        if scores.numel() == 0:
            continue
        if total >= B:
            break
        k_floor = min(b0, scores.numel())
        if b0_mode == "prefix":
            floor_local = set(range(k_floor))
        else:  # "topk"
            _, top_idx = torch.topk(scores.float(), k_floor)
            floor_local = set(top_idx.tolist())
        if seg_idx not in selected_segs:
            selected_segs[seg_idx] = set()
        new_tokens = floor_local - selected_segs[seg_idx]
        actual_add = min(len(new_tokens), B - total)
        if actual_add > 0:
            added = set(list(new_tokens)[:actual_add])
            selected_segs[seg_idx].update(added)
            total += actual_add


def select_adaptive_kvfocus_tokens(
    token_scores_list: List[torch.Tensor],
    R0: float,
    d_probe_per_seg: Optional[List[float]] = None,
    num_segments: int = 0,
    lam: float = 0.5,
    all_kvfocus_mapping: Optional[List[Tuple[int, int, int]]] = None,
    n_ref: int = 11,
    max_rate: float = 0.50,
    b0: int = 0,
    b0_mode: str = "topk",
    eps: float = 1e-12,
    # Decoupled scaling + solver
    solver: str = "residual_budget",
    scaling: str = "linear",
    beta: float = 1.0,
    eta: float = 0.5,
    gamma: float = 2.0,
    rho: float = 0.05,
    top_k_peak: int = 3,
    target_rate: float = 0.0,
) -> Dict:
    """Adaptive token selection for KVFocus.

    Decoupled design: scaling determines H formula, solver determines selection strategy.
    """
    if num_segments == 0 or not token_scores_list:
        return {
            "global_mask_updates": [],
            "recompute_rate": 0.0,
            "per_segment_counts": [],
            "R0": R0,
            "threshold": 0.0,
            "H": 0.0,
            "k_star": None,
            "g_N": 0.0,
            "d_peak": 0.0,
        }

    # ========== Step A: Compute harm metric H ==========
    g_N = 0.0
    D_peak = 0.0
    if scaling == "none":
        # No g_N scaling: H = R0 + beta*D_probe + eta*D_peak
        # When beta=0, eta=0, naturally degrades to H = R0 (ablation)
        # g_N still computed for diagnostic output
        g_N = min(num_segments / n_ref, 2.5)
        if d_probe_per_seg is not None and len(d_probe_per_seg) > 0:
            D_probe, D_peak = _compute_probe_metrics(d_probe_per_seg, eta, top_k_peak)
            H = R0 + beta * D_probe + eta * D_peak
        else:
            H = R0
    elif scaling == "log":
        # [Preserved but ablation table does not use]
        s_N = math.log1p(num_segments) / math.log1p(n_ref)
        g_N = s_N  # log-scaled g_N equivalent
        if d_probe_per_seg is not None and len(d_probe_per_seg) > 0:
            D_probe, D_peak = _compute_probe_metrics(d_probe_per_seg, eta, top_k_peak)
            H = s_N * (R0 + beta * D_probe + eta * D_peak)
        else:
            H = s_N * R0
    else:
        # Default (scaling="linear") + ablation variants:
        # g_N × (R0 + beta*D_probe + eta*D_peak)
        if d_probe_per_seg is not None and len(d_probe_per_seg) > 0:
            # Compute g_N and D_peak manually so we can return them
            g_N = min(num_segments / n_ref, 2.5)
            D_probe_val, D_peak = _compute_probe_metrics(d_probe_per_seg, eta, top_k_peak)
            H = g_N * (R0 + beta * D_probe_val + eta * D_peak)
        else:
            # Fallback: no D_probe data → g_N × R0
            g_N = min(num_segments / n_ref, 2.5)
            H = g_N * R0

    H = max(H, eps)

    # Concat all token scores
    all_scores = torch.cat(token_scores_list)
    total_cached_tokens = all_scores.numel()

    if total_cached_tokens == 0:
        return {
            "global_mask_updates": [],
            "recompute_rate": 0.0,
            "per_segment_counts": [0] * len(token_scores_list),
            "R0": R0,
            "threshold": 0.0,
            "H": H,
            "k_star": None,
            "g_N": g_N,
            "d_peak": D_peak,
        }

    # Compute proportional scores: π_i = s_i / Σs
    sum_scores = all_scores.float().sum().item() + eps
    prop_scores_list = [s.float() / sum_scores for s in token_scores_list]
    all_prop = torch.cat(prop_scores_list)

    # Diagnostic stats — batch GPU→CPU transfer
    _diag_vals = torch.stack([
        torch.quantile(all_prop, 0.95),
        all_prop.mean(),
        all_prop.max(),
    ]).cpu().tolist()
    p95_prop = _diag_vals[0]
    mean_prop = _diag_vals[1]
    max_prop = _diag_vals[2]

    # Budget cap
    k_max = math.floor(max_rate * total_cached_tokens) if max_rate > 0 else total_cached_tokens

    # ========== Step B: Token selection strategy ==========
    k_star = None
    threshold = 0.0

    if solver == "topk":
        # Fixed-rate top-k (ablation reference)
        rate = target_rate if target_rate > 0 else max_rate
        k = min(int(rate * total_cached_tokens), total_cached_tokens)
        if k <= 0:
            seg_selected, per_segment_counts, total_selected = {}, [0] * len(token_scores_list), 0
        else:
            sorted_pi, sorted_idx = torch.sort(all_prop, descending=True)
            seg_selected = _map_global_to_per_segment(sorted_idx[:k], token_scores_list)
            per_segment_counts = [len(seg_selected.get(i, set())) for i in range(len(token_scores_list))]
            total_selected = sum(per_segment_counts)

    elif solver == "threshold":
        # CRH-threshold: τ = λ / (H × |I|)
        threshold = lam / (H * total_cached_tokens + eps)
        seg_selected, per_segment_counts, total_selected = _select_threshold(
            prop_scores_list, threshold, k_max, b0, b0_mode,
        )

    elif solver == "coverage":
        # Saturated coverage prefix scan
        B = k_max
        if B <= 0 or H <= eps:
            seg_selected, per_segment_counts, total_selected = {}, [0] * len(token_scores_list), 0
        else:
            sorted_pi, sorted_idx = torch.sort(all_prop, descending=True)
            sorted_pi = sorted_pi[:B]
            sorted_idx = sorted_idx[:B]

            C = torch.cumsum(sorted_pi, dim=0)
            benefit = H * (1.0 - torch.clamp(1.0 - C, min=0.0) ** gamma)
            cost = lam * torch.arange(1, B + 1, device=C.device, dtype=C.dtype) / total_cached_tokens
            J = benefit - cost

            best_val, best_pos = torch.max(J, dim=0)
            if best_val.item() <= 0:
                k_star = 0
                seg_selected = {}
            else:
                k_star = best_pos.item() + 1
                seg_selected = _map_global_to_per_segment(sorted_idx[:k_star], token_scores_list)

            # Add floor tokens after prefix scan
            if b0 > 0:
                _add_floor_tokens(seg_selected, token_scores_list, b0, b0_mode, k_max)

            per_segment_counts = [len(seg_selected.get(i, set())) for i in range(len(token_scores_list))]
            total_selected = sum(per_segment_counts)

    elif solver == "residual_budget":
        # Residual-harm constrained top-p (default)
        B = k_max
        if H <= rho:
            # Harm already below budget → no recompute
            k_star = 0
            seg_selected, per_segment_counts, total_selected = {}, [0] * len(token_scores_list), 0
        elif B <= 0:
            seg_selected, per_segment_counts, total_selected = {}, [0] * len(token_scores_list), 0
        else:
            q = 1.0 - (rho / (H + eps)) ** (1.0 / gamma)
            q = max(0.0, min(q, 1.0))

            sorted_pi, sorted_idx = torch.sort(all_prop, descending=True)
            sorted_pi = sorted_pi[:B]
            sorted_idx = sorted_idx[:B]

            C = torch.cumsum(sorted_pi, dim=0)
            hit = torch.nonzero(C >= q).flatten()

            if hit.numel() == 0:
                k_star = B
            else:
                k_star = hit[0].item() + 1

            seg_selected = _map_global_to_per_segment(sorted_idx[:k_star], token_scores_list)

            # Add floor tokens after top-p selection
            if b0 > 0:
                _add_floor_tokens(seg_selected, token_scores_list, b0, b0_mode, k_max)

            per_segment_counts = [len(seg_selected.get(i, set())) for i in range(len(token_scores_list))]
            total_selected = sum(per_segment_counts)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Build global_mask_updates
    global_mask_updates = []
    if all_kvfocus_mapping is not None:
        for seg_idx in range(len(token_scores_list)):
            if token_scores_list[seg_idx].numel() == 0:
                continue
            abs_start, valid_len, _ = all_kvfocus_mapping[seg_idx]
            local_indices = seg_selected.get(seg_idx, set())
            if local_indices:
                global_mask_updates.append(
                    (abs_start, torch.tensor(sorted(local_indices), dtype=torch.long, device=abs_start.device))
                )

    recompute_rate = total_selected / max(total_cached_tokens, 1)

    return {
        "global_mask_updates": global_mask_updates,
        "recompute_rate": recompute_rate,
        "per_segment_counts": per_segment_counts,
        "R0": R0,
        "threshold": threshold,
        "H": H,
        "k_star": k_star,
        "p95_prop": p95_prop,
        "mean_prop": mean_prop,
        "max_prop": max_prop,
        "sum_scores": sum_scores,
        "g_N": g_N,
        "d_peak": D_peak,
    }


def _select_threshold(
    prop_scores_list: List[torch.Tensor],
    threshold: float,
    k_max: int,
    b0: int,
    b0_mode: str,
) -> Tuple[Dict[int, Set[int]], List[int], int]:
    """Threshold selection: original candidate + floor token logic."""
    candidates = []
    floor_indices_per_seg = {}

    for seg_idx, scores in enumerate(prop_scores_list):
        if scores.numel() == 0:
            continue

        # Floor: b0 tokens per segment
        if b0 > 0:
            k_floor = min(b0, scores.numel())
            if b0_mode == "prefix":
                floor_indices_per_seg[seg_idx] = set(range(k_floor))
            else:  # "topk"
                _, top_idx = torch.topk(scores, k_floor)
                floor_indices_per_seg[seg_idx] = set(top_idx.tolist())

        # All tokens above threshold are candidates
        above_mask = scores > threshold
        above_indices = above_mask.nonzero(as_tuple=True)[0].tolist()

        for local_idx in above_indices:
            candidates.append((scores[local_idx].item(), seg_idx, local_idx))

    # Sort candidates by proportional score descending
    candidates.sort(key=lambda x: -x[0])

    # Select tokens: floor first, then fill from candidates respecting budget
    seg_selected: Dict[int, Set[int]] = {}
    per_segment_counts = [0] * len(prop_scores_list)
    total_selected = 0

    # Step 1: Add floor tokens
    for seg_idx, floor_set in floor_indices_per_seg.items():
        if seg_idx not in seg_selected:
            seg_selected[seg_idx] = set()
        seg_selected[seg_idx].update(floor_set)
        per_segment_counts[seg_idx] = len(floor_set)
        total_selected += len(floor_set)

    # Step 2: Fill from candidates (skip already-selected floor tokens)
    for score, seg_idx, local_idx in candidates:
        if total_selected >= k_max:
            break
        if seg_idx in seg_selected and local_idx in seg_selected[seg_idx]:
            continue
        if seg_idx not in seg_selected:
            seg_selected[seg_idx] = set()
        seg_selected[seg_idx].add(local_idx)
        per_segment_counts[seg_idx] += 1
        total_selected += 1

    return seg_selected, per_segment_counts, total_selected
