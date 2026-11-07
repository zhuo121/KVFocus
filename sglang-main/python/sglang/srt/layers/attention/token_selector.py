# HACK(Shizhuo) New
"""
TokenScorer / TokenSelector: Decoupled scoring and selection for CacheFlex.

Scorer computes per-token risk scores (signal).
Selector chooses which tokens to recompute from those scores (mask).

Design:
  - Scorer subclasses differ in WHAT they compute (M-only vs M×U)
  - Selector subclasses differ in HOW they select (top-k, threshold, coverage, residual_budget)
  - All selectors (except CacheBlendSelector) delegate to select_adaptive_kvfocus_tokens()
"""

import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import torch

# unified dump control
from sglang.srt.utils import cacheflex_dump as _cf_dump


# ---------------------------------------------------------------------------
# ScorerResult: output of TokenScorer.compute()
# ---------------------------------------------------------------------------

@dataclass
class ScorerResult:
    """Per-request output of a TokenScorer."""
    token_scores_list: List[torch.Tensor]           # per-segment scores [valid_len]
    mapping: List[Tuple[int, int, int]]              # (abs_start, valid_len, seg_idx)
    total_cached_tokens: int = 0
    seg_log_stats: List[Optional[Dict]] = field(default_factory=list)
    # KVFocus-specific (absent for CacheBlend)
    m_diff_list: List[Optional[torch.Tensor]] = field(default_factory=list)
    seg_scores_auto: list = field(default_factory=list)
    recomp_ratio: float = 0.0
    # Extended per-token data (populated only when SGLANG_KVFOCUS_DUMP_EXTENDED=1)
    m_diff_l1_list: List[Optional[torch.Tensor]] = field(default_factory=list)
    u_l1_per_token_list: List[Optional[torch.Tensor]] = field(default_factory=list)
    u_l2_per_token_list: List[Optional[torch.Tensor]] = field(default_factory=list)
    suffix_u_l1_per_token_list: List[Optional[torch.Tensor]] = field(default_factory=list)
    suffix_u_l2_per_token_list: List[Optional[torch.Tensor]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# TokenScorer base + subclasses
# ---------------------------------------------------------------------------

class TokenScorer(ABC):
    """Base class: compute per-token risk scores from check layer signals."""

    @abstractmethod
    def compute(
        self,
        v: torch.Tensor,
        tasks_by_req: List[List[Dict]],
        start_loc: torch.Tensor,
        forward_batch,
        layer,
        **kwargs,
    ) -> Dict[int, ScorerResult]:
        """Return {req_idx: ScorerResult} per request in the batch."""


class KVFocusScorerV1(TokenScorer):
    """KVFocus V1 (legacy): m_i × U_{L2}(seg) per-token scoring.

    Uses segment-level attention mass (2 independent einsum calls).
    Kept for backward compatibility. Use KVFocusScorer (V2) for the default path.
    """

    def compute(
        self,
        v: torch.Tensor,
        tasks_by_req: List[List[Dict]],
        start_loc: torch.Tensor,
        forward_batch,
        layer,
        q: torch.Tensor = None,
        k: torch.Tensor = None,
        num_probes: int = 2,
        **kwargs,
    ) -> Dict[int, ScorerResult]:
        import os
        from sglang.srt.layers.attention.triton_ops.cacheflex import (
            compute_probe_segment_score_L1,
            compute_probe_segment_score_L2,
            compute_diff,
        )

        # Extended per-token analysis (gated by env var)
        dump_extended = _cf_dump.is_enabled(_cf_dump.Category.KVFOCUS_EXTENDED)
        if dump_extended:
            from sglang.srt.layers.attention.triton_ops.cacheflex import (
                compute_diff_l1_l2,
                compute_per_token_u_scores,
                compute_suffix_per_token_u_scores,
            )

        seq_lens = forward_batch.extend_seq_lens

        # [P0] Check if contiguous buffer is available for fast path
        _use_old_kv = getattr(forward_batch, "cacheflex_use_old_kv", False)
        contiguous_kv = getattr(forward_batch, "cacheflex_contiguous_kv", None) if _use_old_kv else None

        # Compute segment-level U scores (batch-wide)
        u_scores_l1 = compute_probe_segment_score_L1(
            q=q, k=k, tasks_by_req=tasks_by_req,
            start_loc=start_loc, seq_lens=seq_lens,
            num_q_heads=layer.tp_q_head_num,
            num_kv_heads=layer.tp_k_head_num,
            head_dim=layer.qk_head_dim,
            num_probes=num_probes,
        )
        u_scores_l2 = compute_probe_segment_score_L2(
            q=q, k=k, tasks_by_req=tasks_by_req,
            start_loc=start_loc, seq_lens=seq_lens,
            num_q_heads=layer.tp_q_head_num,
            num_kv_heads=layer.tp_k_head_num,
            head_dim=layer.qk_head_dim,
            num_probes=num_probes,
        )

        # Compute per-token U scores (batch-wide, only if extended dump enabled)
        per_token_u = None
        suffix_per_token_u = None
        if dump_extended:
            per_token_u = compute_per_token_u_scores(
                q=q, k=k, tasks_by_req=tasks_by_req,
                start_loc=start_loc, seq_lens=seq_lens,
                num_q_heads=layer.tp_q_head_num,
                num_kv_heads=layer.tp_k_head_num,
                head_dim=layer.qk_head_dim,
                num_probes=num_probes,
            )
            suffix_per_token_u = compute_suffix_per_token_u_scores(
                q=q, k=k, tasks_by_req=tasks_by_req,
                start_loc=start_loc, seq_lens=seq_lens,
                num_q_heads=layer.tp_q_head_num,
                num_kv_heads=layer.tp_k_head_num,
                head_dim=layer.qk_head_dim,
                num_probes=num_probes,
            )

        # Store U scores in runtime for harm observation
        forward_batch.cacheflex_runtime["kvfocus_u_scores_l1"] = u_scores_l1
        forward_batch.cacheflex_runtime["kvfocus_u_scores_l2"] = u_scores_l2

        results = {}

        for req_idx, tasks in enumerate(tasks_by_req):
            if not tasks:
                continue
            req_start = start_loc[req_idx]
            u_tensor_l1 = u_scores_l1.get(req_idx)
            u_tensor_l2 = u_scores_l2.get(req_idx)

            token_scores_list = []
            mapping = []
            m_diff_list = []
            seg_scores_auto = []
            seg_log_stats = []
            total_cached_tokens = 0
            recomp_ratio = 0.0

            # Extended lists
            m_diff_l1_list = []
            u_l1_per_token_list = []
            u_l2_per_token_list = []
            suffix_u_l1_per_token_list = []
            suffix_u_l2_per_token_list = []

            for seg_idx, task in enumerate(tasks):
                abs_start = req_start + task["rel_start"]
                length = task["src_len"]

                if task.get("kind") == "drop_all":
                    seg_log_stats.append(None)
                    m_diff_list.append(None)
                    seg_scores_auto.append(0.0)
                    if dump_extended:
                        m_diff_l1_list.append(None)
                        u_l1_per_token_list.append(None)
                        u_l2_per_token_list.append(None)
                        suffix_u_l1_per_token_list.append(None)
                        suffix_u_l2_per_token_list.append(None)
                    continue

                # [P2] Use stored L0 prefix V for source segments (prefix hidden_states are stale at L1)
                l0_prefix_v_map = getattr(forward_batch, '_cacheflex_l0_prefix_v_map', None)
                if l0_prefix_v_map is not None and (req_idx, task["rel_start"]) in l0_prefix_v_map:
                    v_seg = l0_prefix_v_map[(req_idx, task["rel_start"])]
                else:
                    v_seg = v[abs_start : abs_start + length]

                if dump_extended:
                    m_diff, m_diff_l1, valid_len = compute_diff_l1_l2(
                        v_seg,
                        task["src_block_indices"],
                        task["src_offset"],
                        forward_batch.token_to_kv_pool,
                        layer.layer_id,
                        contiguous_kv=contiguous_kv,
                        contiguous_offset=task.get("contiguous_offset", 0),
                    )
                else:
                    m_diff, valid_len = compute_diff(
                        v_seg,
                        task["src_block_indices"],
                        task["src_offset"],
                        forward_batch.token_to_kv_pool,
                        layer.layer_id,
                        contiguous_kv=contiguous_kv,
                        contiguous_offset=task.get("contiguous_offset", 0),
                    )
                    m_diff_l1 = None

                u_val_l2 = u_tensor_l2[seg_idx] if u_tensor_l2 is not None and seg_idx < len(u_tensor_l2) else torch.tensor(0.0, device=v.device)
                u_val_l1 = u_tensor_l1[seg_idx] if u_tensor_l1 is not None and seg_idx < len(u_tensor_l1) else torch.tensor(0.0, device=v.device)

                # Extract per-token U for this segment (extended dump only)
                if dump_extended and per_token_u is not None:
                    ptu = per_token_u.get(req_idx)
                    if ptu is not None and seg_idx < len(ptu["u_l1_per_token"]):
                        u_l1_pt = ptu["u_l1_per_token"][seg_idx]
                        u_l2_pt = ptu["u_l2_per_token"][seg_idx]
                    else:
                        u_l1_pt = None
                        u_l2_pt = None
                else:
                    u_l1_pt = None
                    u_l2_pt = None

                if dump_extended and suffix_per_token_u is not None:
                    sfx_ptu = suffix_per_token_u.get(req_idx)
                    if sfx_ptu is not None and seg_idx < len(sfx_ptu["u_l1_per_token"]):
                        suffix_u_l1_pt = sfx_ptu["u_l1_per_token"][seg_idx]
                        suffix_u_l2_pt = sfx_ptu["u_l2_per_token"][seg_idx]
                    else:
                        suffix_u_l1_pt = None
                        suffix_u_l2_pt = None
                else:
                    suffix_u_l1_pt = None
                    suffix_u_l2_pt = None

                if m_diff is not None:
                    token_scores = m_diff * u_val_l2
                    token_scores_list.append(token_scores)
                    mapping.append((abs_start, valid_len, seg_idx))
                    total_cached_tokens += valid_len
                    recomp_ratio = task["recomp_ratio"]

                    m_max_d = m_diff.max()
                    seg_scores_auto.append(m_max_d * u_val_l2)
                    m_diff_list.append(m_diff)
                    if dump_extended:
                        m_diff_l1_list.append(m_diff_l1)
                        u_l1_per_token_list.append(u_l1_pt)
                        u_l2_per_token_list.append(u_l2_pt)
                        suffix_u_l1_per_token_list.append(suffix_u_l1_pt)
                        suffix_u_l2_per_token_list.append(suffix_u_l2_pt)
                else:
                    m_diff_list.append(None)
                    seg_scores_auto.append(torch.tensor(0.0, device=v.device))
                    if dump_extended:
                        m_diff_l1_list.append(None)
                        u_l1_per_token_list.append(None)
                        u_l2_per_token_list.append(None)
                        suffix_u_l1_per_token_list.append(None)
                        suffix_u_l2_per_token_list.append(None)

                seg_log_stats.append({
                    "abs_start": abs_start,
                    "length": length,
                    "valid_len": valid_len if m_diff is not None else 0,
                    "u_l1": u_val_l1,
                    "u_l2": u_val_l2,
                    "m_mean": m_diff.mean() if m_diff is not None and m_diff.numel() > 0 else torch.tensor(0.0, device=v.device),
                    "m_max": m_diff.max() if m_diff is not None and m_diff.numel() > 0 else torch.tensor(0.0, device=v.device),
                })

            results[req_idx] = ScorerResult(
                token_scores_list=token_scores_list,
                mapping=mapping,
                total_cached_tokens=total_cached_tokens,
                seg_log_stats=seg_log_stats,
                m_diff_list=m_diff_list,
                seg_scores_auto=seg_scores_auto,
                recomp_ratio=recomp_ratio,
                m_diff_l1_list=m_diff_l1_list if dump_extended else [],
                u_l1_per_token_list=u_l1_per_token_list if dump_extended else [],
                u_l2_per_token_list=u_l2_per_token_list if dump_extended else [],
                suffix_u_l1_per_token_list=suffix_u_l1_per_token_list if dump_extended else [],
                suffix_u_l2_per_token_list=suffix_u_l2_per_token_list if dump_extended else [],
            )

        return results


# [CacheTune-baseline]: frequency-domain selector (related-work ablation).
def compute_freq_energy(k_seg, v_seg, alpha: float = 0.5):
    """CacheTune low-frequency selection score (arXiv 2605.24022 §4.1, Eqs 2-6).

    Faithful reproduction of CacheTune's *selector only* (not its transfer/async/RoPE system):
    along the sequence dim, rFFT K and V, keep the lowest `alpha` fraction of frequency bins,
    irFFT back, take per-token L2 energy, return the joint 0.5*(||K_lp_i|| + ||V_lp_i||).
    alpha=0.5 is the paper default (verified robust over [0.3, 0.7]). The per-token score feeds
    the same global top-k selector as M×U for an apples-to-apples selector ablation at matched
    rate. k_seg / v_seg: [seq, ...] (trailing dims flattened for the energy). Returns [seq].
    """
    def _lowfreq_energy(x):
        x = x.reshape(x.shape[0], -1).float()         # [seq, C]
        seq = x.shape[0]
        xf = torch.fft.rfft(x, dim=0)                 # [seq//2+1, C] complex; rFFT over sequence
        cutoff = max(1, int(alpha * xf.shape[0]))     # keep lowest alpha fraction of freq bins
        if cutoff < xf.shape[0]:
            xf[cutoff:] = 0                            # low-pass: zero the high band
        xr = torch.fft.irfft(xf, n=seq, dim=0)        # [seq, C] low-frequency reconstruction
        return xr.norm(dim=1)                         # [seq] per-token L2 energy over channels
    return 0.5 * (_lowfreq_energy(k_seg) + _lowfreq_energy(v_seg))


class KVFocusScorer(TokenScorer):
    """KVFocus per-token scorer, mode-dispatched via SGLANG_KVFOCUS_SCORER_MODE.

    Modes:
      mxu (default) = KVFocus: m_i × U_i at the check layer
      m_only = value deviation alone | u_only = U alone | m_add_u = M+U additive
      freq = CacheTune baseline | random = diagnostic
    Per-token U via compute_per_token_u_l1_only; falls back to segment-level U.
    """

    def compute(
        self,
        v: torch.Tensor,
        tasks_by_req: List[List[Dict]],
        start_loc: torch.Tensor,
        forward_batch,
        layer,
        q: torch.Tensor = None,
        k: torch.Tensor = None,
        num_probes: int = 2,
        **kwargs,
    ) -> Dict[int, ScorerResult]:
        # SGLANG_KVFOCUS_USE_SLOW_U=1 switches to per-Q-head slow
        # path (compute_per_token_u_l1_only). Investigation of Qwen garbled output:
        # the fast path's avg-Q approximation may degrade on GQA ratios > 4
        # (Qwen2.5-7B = 7-way), since softmax(mean(qk)) != mean(softmax(qk)) and
        # the divergence grows with group size.
        from sglang.srt.layers.attention.triton_ops.cacheflex import (
            compute_diff,
            _compute_u_scores_l1_fast,
            compute_per_token_u_l1_only,
        )
        _use_slow_u = os.environ.get("SGLANG_KVFOCUS_USE_SLOW_U", "0") == "1"
        _u_fn = compute_per_token_u_l1_only if _use_slow_u else _compute_u_scores_l1_fast

        dump_extended = _cf_dump.is_enabled(_cf_dump.Category.KVFOCUS_EXTENDED)
        # Segment-level U_L2 needed only for dump or harm observation
        need_seg_u_l2 = (dump_extended
                        or _cf_dump.is_enabled(_cf_dump.Category.HARM)
                        or _cf_dump.is_enabled(_cf_dump.Category.KVFOCUS_SCORES))
        if dump_extended:
            from sglang.srt.layers.attention.triton_ops.cacheflex import (
                compute_diff_l1_l2,
                compute_suffix_per_token_u_scores,
            )

        seq_lens = forward_batch.extend_seq_lens

        # [P0] Check if contiguous buffer is available for fast path
        _use_old_kv = getattr(forward_batch, "cacheflex_use_old_kv", False)
        contiguous_kv = getattr(forward_batch, "cacheflex_contiguous_kv", None) if _use_old_kv else None

        # pass per-req probe range starts so probes can span the LAST
        # source (= query content) + suffix block (= chat-template marker), when
        # SGLANG_KVFOCUS_PROBE_FROM_SUFFIX=1. _cacheflex_probe_starts records the
        # start of the last source in sliced coords (Stage 1 in llama.py / qwen2.py).
        # _cacheflex_l0_suffix_starts only marks AFTER all sources (= chat marker
        # start) which equals tail-5 → no-op.
        _suffix_starts = getattr(forward_batch, "_cacheflex_probe_starts", None)

        # Single-pass: per-token U_L1 (always) + segment-level U_L1 (always) + U_L2 (only if needed)
        # [KVFocus]: U = online probe-q @ check-layer K via _u_fn.
        per_token_u = self._get_per_token_u(
            _u_fn, q, k, tasks_by_req, start_loc, seq_lens, layer,
            num_probes, need_seg_u_l2, _suffix_starts, forward_batch,
        )

        # Extended dump: suffix U (only when dump is enabled)
        suffix_per_token_u = None
        if dump_extended:
            suffix_per_token_u = compute_suffix_per_token_u_scores(
                q=q, k=k, tasks_by_req=tasks_by_req,
                start_loc=start_loc, seq_lens=seq_lens,
                num_q_heads=layer.tp_q_head_num,
                num_kv_heads=layer.tp_k_head_num,
                head_dim=layer.qk_head_dim,
                num_probes=num_probes,
            )

        # Extract segment-level U scores (u_l2_seg is None if compute_seg_u_l2=False)
        u_scores_l1 = {rid: d["u_l1_seg"] for rid, d in per_token_u.items()}
        u_scores_l2 = {rid: d["u_l2_seg"] for rid, d in per_token_u.items() if d["u_l2_seg"] is not None}
        forward_batch.cacheflex_runtime["kvfocus_u_scores_l1"] = u_scores_l1
        forward_batch.cacheflex_runtime["kvfocus_u_scores_l2"] = u_scores_l2

        # Read scorer mode once per compute() call (not per segment)
        _scorer_mode = os.environ.get("SGLANG_KVFOCUS_SCORER_MODE", "mxu")

        results = {}

        for req_idx, tasks in enumerate(tasks_by_req):
            if not tasks:
                continue
            req_start = start_loc[req_idx]
            ptu_data = per_token_u.get(req_idx)
            u_tensor_l1 = u_scores_l1.get(req_idx)
            u_tensor_l2 = u_scores_l2.get(req_idx)
            sfx_ptu = suffix_per_token_u.get(req_idx) if suffix_per_token_u is not None else None

            token_scores_list = []
            mapping = []
            m_diff_list = []
            seg_scores_auto = []
            seg_log_stats = []
            total_cached_tokens = 0
            recomp_ratio = 0.0

            m_diff_l1_list = []
            u_l1_per_token_list = []
            u_l2_per_token_list = []
            suffix_u_l1_per_token_list = []
            suffix_u_l2_per_token_list = []

            for seg_idx, task in enumerate(tasks):
                abs_start = req_start + task["rel_start"]
                length = task["src_len"]

                if task.get("kind") == "drop_all":
                    seg_log_stats.append(None)
                    m_diff_list.append(None)
                    seg_scores_auto.append(0.0)
                    if dump_extended:
                        m_diff_l1_list.append(None)
                        u_l1_per_token_list.append(None)
                        u_l2_per_token_list.append(None)
                        suffix_u_l1_per_token_list.append(None)
                        suffix_u_l2_per_token_list.append(None)
                    continue

                # [P2] Use stored L0 prefix V for source segments (prefix hidden_states are stale at L1)
                l0_prefix_v_map = getattr(forward_batch, '_cacheflex_l0_prefix_v_map', None)
                if l0_prefix_v_map is not None and (req_idx, task["rel_start"]) in l0_prefix_v_map:
                    v_seg = l0_prefix_v_map[(req_idx, task["rel_start"])]
                else:
                    v_seg = v[abs_start : abs_start + length]

                if dump_extended:
                    m_diff, m_diff_l1, valid_len = compute_diff_l1_l2(
                        v_seg,
                        task["src_block_indices"],
                        task["src_offset"],
                        forward_batch.token_to_kv_pool,
                        layer.layer_id,
                        contiguous_kv=contiguous_kv,
                        contiguous_offset=task.get("contiguous_offset", 0),
                    )
                else:
                    m_diff, valid_len = compute_diff(
                        v_seg,
                        task["src_block_indices"],
                        task["src_offset"],
                        forward_batch.token_to_kv_pool,
                        layer.layer_id,
                        contiguous_kv=contiguous_kv,
                        contiguous_offset=task.get("contiguous_offset", 0),
                    )
                    m_diff_l1 = None

                u_val_l2 = u_tensor_l2[seg_idx] if u_tensor_l2 is not None and seg_idx < len(u_tensor_l2) else torch.tensor(0.0, device=v.device)
                u_val_l1 = u_tensor_l1[seg_idx] if u_tensor_l1 is not None and seg_idx < len(u_tensor_l1) else torch.tensor(0.0, device=v.device)

                # Per-token U_L1: from online path (always available)
                u_l1_pt = None
                if ptu_data is not None and seg_idx < len(ptu_data["u_l1_per_token"]):
                    u_l1_pt = ptu_data["u_l1_per_token"][seg_idx]

                u_l2_pt = None

                suffix_u_l1_pt = None
                suffix_u_l2_pt = None
                if dump_extended and sfx_ptu is not None and seg_idx < len(sfx_ptu["u_l1_per_token"]):
                    suffix_u_l1_pt = sfx_ptu["u_l1_per_token"][seg_idx]
                    suffix_u_l2_pt = sfx_ptu["u_l2_per_token"][seg_idx]

                if m_diff is not None:
                    # Scorer mode (SGLANG_KVFOCUS_SCORER_MODE; default "mxu" = KVFocus headline).
                    _u_pt = u_l1_pt[:valid_len].to(v.device) if (u_l1_pt is not None and len(u_l1_pt) >= valid_len) else None

                    if _scorer_mode == "random":            # diagnostic
                        token_scores = torch.rand_like(m_diff)
                    elif _scorer_mode == "u_only":          # U-only (w/o M): deep U alone
                        token_scores = _u_pt.clone() if _u_pt is not None else u_val_l1.clone()
                    elif _scorer_mode == "m_only":          # M-only (w/o U): value deviation alone
                        token_scores = m_diff
                    elif _scorer_mode == "m_add_u":         # M+U (additive): rank-sum vs product in mxu
                        _N = float(len(m_diff))
                        _m_rank = torch.argsort(torch.argsort(m_diff)).float() / _N
                        _u_src = _u_pt if _u_pt is not None else u_val_l1.expand(len(m_diff))
                        _u_rank = torch.argsort(torch.argsort(_u_src)).float() / _N
                        token_scores = _m_rank + _u_rank
                    elif _scorer_mode == "freq":
                        # [CacheTune-baseline]: low-frequency energy selector
                        # (arXiv 2605.24022). Uses the segment's check-layer K/V (same layer/tokens
                        # as M×U) -> per-token score -> existing global selector at matched rate
                        # (apples-to-apples). alpha via SGLANG_KVFOCUS_FREQ_ALPHA (default 0.5).
                        _alpha = float(os.environ.get("SGLANG_KVFOCUS_FREQ_ALPHA", "0.5"))
                        _k_seg = k[abs_start : abs_start + length][:valid_len]
                        token_scores = compute_freq_energy(
                            _k_seg, v_seg[:valid_len], _alpha
                        ).to(dtype=m_diff.dtype, device=m_diff.device)
                    else:                                   # "mxu" (default) = KVFocus: m_i × U_i at the check layer
                        token_scores = m_diff * (_u_pt if _u_pt is not None else u_val_l1)

                    token_scores_list.append(token_scores)
                    mapping.append((abs_start, valid_len, seg_idx))
                    total_cached_tokens += valid_len
                    recomp_ratio = task["recomp_ratio"]

                    m_max_d = m_diff.max()
                    # Segment score: use per-token U if available, else segment-level
                    if _u_pt is not None and _scorer_mode in ("mxu", "m_add_u", "u_only"):
                        seg_scores_auto.append((m_max_d * _u_pt).max())
                    else:
                        seg_scores_auto.append(m_max_d * u_val_l1)
                    m_diff_list.append(m_diff)
                    if dump_extended:
                        m_diff_l1_list.append(m_diff_l1)
                        u_l1_per_token_list.append(u_l1_pt)
                        u_l2_per_token_list.append(u_l2_pt)
                        suffix_u_l1_per_token_list.append(suffix_u_l1_pt)
                        suffix_u_l2_per_token_list.append(suffix_u_l2_pt)
                else:
                    m_diff_list.append(None)
                    seg_scores_auto.append(torch.tensor(0.0, device=v.device))
                    if dump_extended:
                        m_diff_l1_list.append(None)
                        u_l1_per_token_list.append(None)
                        u_l2_per_token_list.append(None)
                        suffix_u_l1_per_token_list.append(None)
                        suffix_u_l2_per_token_list.append(None)

                seg_log_stats.append({
                    "abs_start": abs_start,
                    "length": length,
                    "valid_len": valid_len if m_diff is not None else 0,
                    "u_l1": u_val_l1,
                    "u_l2": u_val_l2 if u_tensor_l2 is not None else 0.0,
                    "m_mean": m_diff.mean() if m_diff is not None and m_diff.numel() > 0 else torch.tensor(0.0, device=v.device),
                    "m_max": m_diff.max() if m_diff is not None and m_diff.numel() > 0 else torch.tensor(0.0, device=v.device),
                })

            results[req_idx] = ScorerResult(
                token_scores_list=token_scores_list,
                mapping=mapping,
                total_cached_tokens=total_cached_tokens,
                seg_log_stats=seg_log_stats,
                m_diff_list=m_diff_list,
                seg_scores_auto=seg_scores_auto,
                recomp_ratio=recomp_ratio,
                m_diff_l1_list=m_diff_l1_list if dump_extended else [],
                u_l1_per_token_list=u_l1_per_token_list if dump_extended else [],
                u_l2_per_token_list=u_l2_per_token_list if dump_extended else [],
                suffix_u_l1_per_token_list=suffix_u_l1_per_token_list if dump_extended else [],
                suffix_u_l2_per_token_list=suffix_u_l2_per_token_list if dump_extended else [],
            )

        return results

    def _get_per_token_u(self, _u_fn, q, k, tasks_by_req, start_loc, seq_lens, layer,
                         num_probes, need_seg_u_l2, suffix_starts, forward_batch):
        """Online U: probe-q at the check layer against check-layer K
        (per-token U_L1, full-range softmax-normalized)."""
        return _u_fn(
            q=q, k=k, tasks_by_req=tasks_by_req,
            start_loc=start_loc, seq_lens=seq_lens,
            num_q_heads=layer.tp_q_head_num,
            num_kv_heads=layer.tp_k_head_num,
            head_dim=layer.qk_head_dim,
            num_probes=num_probes,
            compute_seg_u_l2=need_seg_u_l2,
            suffix_starts=suffix_starts,
        )


class CacheBlendScorer(TokenScorer):
    """CacheBlend: M-only (||ΔV||) per-token scoring.

    Calls compute_diff only — no segment-level U computation.
    """

    def compute(
        self,
        v: torch.Tensor,
        tasks_by_req: List[List[Dict]],
        start_loc: torch.Tensor,
        forward_batch,
        layer,
        **kwargs,
    ) -> Dict[int, ScorerResult]:
        from sglang.srt.layers.attention.triton_ops.cacheflex import compute_diff

        # [P0] Check if contiguous buffer is available for fast path
        _use_old_kv = getattr(forward_batch, "cacheflex_use_old_kv", False)
        contiguous_kv = getattr(forward_batch, "cacheflex_contiguous_kv", None) if _use_old_kv else None

        results = {}

        for req_idx, tasks in enumerate(tasks_by_req):
            if not tasks:
                continue
            req_start = start_loc[req_idx]

            token_scores_list = []
            mapping = []
            seg_log_stats = []
            total_cached_tokens = 0
            recomp_ratio = 0.0

            for task in tasks:
                abs_start = req_start + task["rel_start"]
                length = task["src_len"]

                if task.get("kind") == "drop_all":
                    seg_log_stats.append(None)
                    continue

                # [P2] Use stored L0 prefix V for source segments (prefix hidden_states are stale at L1)
                l0_prefix_v_map = getattr(forward_batch, '_cacheflex_l0_prefix_v_map', None)
                if l0_prefix_v_map is not None and (req_idx, task["rel_start"]) in l0_prefix_v_map:
                    v_seg = l0_prefix_v_map[(req_idx, task["rel_start"])]
                else:
                    v_seg = v[abs_start : abs_start + length]
                m_diff, valid_len = compute_diff(
                    v_seg,
                    task["src_block_indices"],
                    task["src_offset"],
                    forward_batch.token_to_kv_pool,
                    layer.layer_id,
                    contiguous_kv=contiguous_kv,
                    contiguous_offset=task.get("contiguous_offset", 0),
                )

                if m_diff is not None:
                    # CacheBlend uses raw M scores (no U multiplier)
                    token_scores_list.append(m_diff)
                    mapping.append((abs_start, valid_len, -1))
                    total_cached_tokens += valid_len
                    recomp_ratio = task["recomp_ratio"]

                seg_log_stats.append({
                    "abs_start": abs_start,
                    "length": length,
                    "valid_len": valid_len if m_diff is not None else 0,
                    "m_mean": m_diff.mean() if m_diff is not None and m_diff.numel() > 0 else torch.tensor(0.0, device=v.device),
                    "m_max": m_diff.max() if m_diff is not None and m_diff.numel() > 0 else torch.tensor(0.0, device=v.device),
                })

            results[req_idx] = ScorerResult(
                token_scores_list=token_scores_list,
                mapping=mapping,
                total_cached_tokens=total_cached_tokens,
                seg_log_stats=seg_log_stats,
                recomp_ratio=recomp_ratio,
            )

        return results


# ---------------------------------------------------------------------------
# TokenSelector base + subclasses
# ---------------------------------------------------------------------------

class TokenSelector(ABC):
    """Base class: select recompute mask from per-token scores."""

    @abstractmethod
    def select(
        self,
        scorer_result: ScorerResult,
        R0: float = 0.0,
        d_probe_per_seg: Optional[List[float]] = None,
        **kwargs,
    ) -> Dict:
        """Return selection result dict with keys:
        global_mask_updates, recompute_rate, per_segment_counts, R0, threshold, H, k_star
        """


class TopKSelector(TokenSelector):
    """solver='topk': fixed-rate top-k selection."""

    def __init__(self, target_rate: float = 0.0, max_rate: float = 0.50):
        self.target_rate = target_rate
        self.max_rate = max_rate

    def select(self, scorer_result, R0=0.0, d_probe_per_seg=None, **kwargs):
        from sglang.srt.layers.attention.kvfocus_adaptive_rate import select_adaptive_kvfocus_tokens
        return select_adaptive_kvfocus_tokens(
            token_scores_list=scorer_result.token_scores_list,
            R0=R0,
            d_probe_per_seg=d_probe_per_seg,
            num_segments=len(scorer_result.token_scores_list),
            lam=kwargs.get("lam", 0.01),
            all_kvfocus_mapping=scorer_result.mapping if scorer_result.mapping else None,
            n_ref=kwargs.get("n_ref", 11),
            max_rate=self.max_rate,
            b0=kwargs.get("b0", 0),
            b0_mode=kwargs.get("b0_mode", "topk"),
            solver="topk",
            scaling=kwargs.get("scaling", "linear"),
            beta=kwargs.get("beta", 1.0),
            eta=kwargs.get("eta", 0.5),
            gamma=kwargs.get("gamma", 2.0),
            rho=kwargs.get("rho", 0.05),
            top_k_peak=kwargs.get("top_k_peak", 3),
            target_rate=self.target_rate,
        )


class ResidualBudgetSelector(TokenSelector):
    """solver='residual_budget': q* = 1-(ρ/H)^{1/γ} top-p prefix scan."""

    def __init__(
        self,
        rho: float = 0.05,
        gamma: float = 2.0,
        max_rate: float = 0.50,
        b0: int = 0,
        b0_mode: str = "topk",
        scaling: str = "linear",
        beta: float = 1.0,
        eta: float = 0.5,
        n_ref: int = 11,
        top_k_peak: int = 3,
    ):
        self.rho = rho
        self.gamma = gamma
        self.max_rate = max_rate
        self.b0 = b0
        self.b0_mode = b0_mode
        self.scaling = scaling
        self.beta = beta
        self.eta = eta
        self.n_ref = n_ref
        self.top_k_peak = top_k_peak

    def select(self, scorer_result, R0=0.0, d_probe_per_seg=None, **kwargs):
        from sglang.srt.layers.attention.kvfocus_adaptive_rate import select_adaptive_kvfocus_tokens
        return select_adaptive_kvfocus_tokens(
            token_scores_list=scorer_result.token_scores_list,
            R0=R0,
            d_probe_per_seg=d_probe_per_seg,
            num_segments=len(scorer_result.token_scores_list),
            lam=kwargs.get("lam", 0.01),
            all_kvfocus_mapping=scorer_result.mapping if scorer_result.mapping else None,
            n_ref=self.n_ref,
            max_rate=self.max_rate,
            b0=self.b0,
            b0_mode=self.b0_mode,
            solver="residual_budget",
            scaling=self.scaling,
            beta=self.beta,
            eta=self.eta,
            gamma=self.gamma,
            rho=self.rho,
            top_k_peak=self.top_k_peak,
            target_rate=0.0,
        )


class CoverageSelector(TokenSelector):
    """solver='coverage': saturated coverage prefix scan."""

    def __init__(
        self,
        lam: float = 0.5,
        gamma: float = 2.0,
        max_rate: float = 0.50,
        b0: int = 0,
        b0_mode: str = "topk",
        scaling: str = "linear",
        beta: float = 1.0,
        eta: float = 0.5,
        n_ref: int = 11,
        top_k_peak: int = 3,
    ):
        self.lam = lam
        self.gamma = gamma
        self.max_rate = max_rate
        self.b0 = b0
        self.b0_mode = b0_mode
        self.scaling = scaling
        self.beta = beta
        self.eta = eta
        self.n_ref = n_ref
        self.top_k_peak = top_k_peak

    def select(self, scorer_result, R0=0.0, d_probe_per_seg=None, **kwargs):
        from sglang.srt.layers.attention.kvfocus_adaptive_rate import select_adaptive_kvfocus_tokens
        return select_adaptive_kvfocus_tokens(
            token_scores_list=scorer_result.token_scores_list,
            R0=R0,
            d_probe_per_seg=d_probe_per_seg,
            num_segments=len(scorer_result.token_scores_list),
            lam=self.lam,
            all_kvfocus_mapping=scorer_result.mapping if scorer_result.mapping else None,
            n_ref=self.n_ref,
            max_rate=self.max_rate,
            b0=self.b0,
            b0_mode=self.b0_mode,
            solver="coverage",
            scaling=self.scaling,
            beta=self.beta,
            eta=self.eta,
            gamma=self.gamma,
            rho=kwargs.get("rho", 0.05),
            top_k_peak=self.top_k_peak,
            target_rate=0.0,
        )


class ThresholdSelector(TokenSelector):
    """solver='threshold': π_i > τ = λ/(H·|I|)."""

    def __init__(
        self,
        lam: float = 0.5,
        max_rate: float = 0.50,
        b0: int = 0,
        b0_mode: str = "topk",
        scaling: str = "linear",
        beta: float = 1.0,
        eta: float = 0.5,
        n_ref: int = 11,
        top_k_peak: int = 3,
    ):
        self.lam = lam
        self.max_rate = max_rate
        self.b0 = b0
        self.b0_mode = b0_mode
        self.scaling = scaling
        self.beta = beta
        self.eta = eta
        self.n_ref = n_ref
        self.top_k_peak = top_k_peak

    def select(self, scorer_result, R0=0.0, d_probe_per_seg=None, **kwargs):
        from sglang.srt.layers.attention.kvfocus_adaptive_rate import select_adaptive_kvfocus_tokens
        return select_adaptive_kvfocus_tokens(
            token_scores_list=scorer_result.token_scores_list,
            R0=R0,
            d_probe_per_seg=d_probe_per_seg,
            num_segments=len(scorer_result.token_scores_list),
            lam=self.lam,
            all_kvfocus_mapping=scorer_result.mapping if scorer_result.mapping else None,
            n_ref=self.n_ref,
            max_rate=self.max_rate,
            b0=self.b0,
            b0_mode=self.b0_mode,
            solver="threshold",
            scaling=self.scaling,
            beta=self.beta,
            eta=self.eta,
            gamma=kwargs.get("gamma", 2.0),
            rho=kwargs.get("rho", 0.05),
            top_k_peak=self.top_k_peak,
            target_rate=0.0,
        )


class KVFocusSelector(TokenSelector):
    """KVFocus fixed-rate global top-k selector.

    Selects tokens by global top-k across all segments using M×U score.
    No per-segment guarantee — allows some segments to have zero recompute.
    """

    def __init__(self, recomp_ratio: float = 0.15):
        self.recomp_ratio = recomp_ratio

    def select(self, scorer_result, R0=0.0, d_probe_per_seg=None, **kwargs):
        token_scores_list = scorer_result.token_scores_list
        mapping = scorer_result.mapping
        total_cached_tokens = scorer_result.total_cached_tokens

        if not token_scores_list or total_cached_tokens == 0:
            return {
                "global_mask_updates": [],
                "recompute_rate": 0.0,
                "per_segment_counts": [0] * len(token_scores_list),
                "R0": R0,
                "threshold": 0.0,
                "H": 0.0,
                "k_star": None,
            }

        # Global top-k across all segments (2 kernels: cat + topk)
        total_scores = torch.cat(token_scores_list)
        num_recompute = max(int(total_cached_tokens * self.recomp_ratio), 1)
        _, global_topk_indices = torch.topk(total_scores, min(num_recompute, len(total_scores)))

        # Build cumulative lengths on GPU (replaces Python list + linear search)
        valid_lens = torch.tensor(
            [v for _, v, _ in mapping],
            device=global_topk_indices.device, dtype=torch.long,
        )
        cum_lens = torch.cat([
            torch.zeros(1, device=valid_lens.device, dtype=torch.long),
            valid_lens.cumsum(0),
        ])  # [S+1]

        # Vectorized segment lookup: searchsorted replaces O(N×S) linear scan
        seg_indices = torch.searchsorted(cum_lens, global_topk_indices, right=True) - 1  # [K]
        local_indices = global_topk_indices - cum_lens[seg_indices]  # [K]

        # Per-segment counts via bincount
        num_segments = len(mapping)
        per_segment_counts = torch.bincount(seg_indices, minlength=num_segments)

        # Sort by segment to group them, then build per-segment local index lists
        sort_order = torch.argsort(seg_indices)
        sorted_local = local_indices[sort_order]

        # Build global_mask_updates: (abs_start, sorted local_indices tensor)
        global_mask_updates = []
        offset = 0
        counts_list = per_segment_counts.tolist()
        for seg_i in range(num_segments):
            count = counts_list[seg_i]
            if count > 0:
                abs_start = mapping[seg_i][0]
                seg_local = torch.sort(sorted_local[offset : offset + count])[0]
                global_mask_updates.append((abs_start, seg_local))
                offset += count

        total_selected = int(per_segment_counts.sum().item())

        return {
            "global_mask_updates": global_mask_updates,
            "recompute_rate": total_selected / max(total_cached_tokens, 1),
            "per_segment_counts": counts_list,
            "R0": R0,
            "threshold": 0.0,
            "H": 0.0,
            "k_star": num_recompute,
        }


class CacheBlendSelector(TokenSelector):
    """CacheBlend global top-k with keep-last (faithful to official CacheBlend).

    HACK(Shizhuo) [cacheblend-global]: all reuse tokens compete in ONE global ‖ΔV‖
    pool for top-R% (NOT per-segment), then the sequence's last token is force-kept —
    matching official CacheBlend (YaoJiayi/CacheBlend vllm_blend/.../xformers.py:
    global topk over docs + last_len keep). Replaces the earlier per-segment variant
    (which diverged from the official global-pool mechanism).
    """

    def __init__(self, recomp_ratio: float = 0.15):
        self.recomp_ratio = recomp_ratio

    def select(self, scorer_result, R0=0.0, d_probe_per_seg=None, **kwargs):
        token_scores_list = scorer_result.token_scores_list
        mapping = scorer_result.mapping
        total_cached_tokens = scorer_result.total_cached_tokens

        if not token_scores_list or total_cached_tokens == 0:
            return {
                "global_mask_updates": [],
                "recompute_rate": 0.0,
                "per_segment_counts": [0] * len(token_scores_list),
                "R0": R0,
                "threshold": 0.0,
                "H": 0.0,
                "k_star": None,
            }

        # HACK(Shizhuo) [cacheblend-global]: ONE global ‖ΔV‖ pool → top-R% + last-token keep
        # (faithful to official CacheBlend xformers; replaces the per-segment variant).
        from sglang.srt.layers.attention.kvfocus_adaptive_rate import _map_global_to_per_segment

        all_scores = torch.cat([s for s in token_scores_list])  # [total] across all segments
        total = all_scores.numel()
        k = max(int(total * self.recomp_ratio), 1)
        topk_global = torch.topk(all_scores, min(k, total)).indices

        # Keep-last: force the sequence's final token (= official last_len keep; QA last_len=1)
        last_global = total - 1
        if not (topk_global == last_global).any():
            topk_global = torch.cat(
                [topk_global, torch.tensor([last_global], device=all_scores.device)]
            )

        seg_selected = _map_global_to_per_segment(topk_global, token_scores_list)

        global_mask_updates = []
        per_segment_counts = []
        for seg_i, (abs_start, valid_len, _) in enumerate(mapping):
            local = seg_selected.get(seg_i, set())
            per_segment_counts.append(len(local))
            if local:
                local_idx = torch.tensor(sorted(local), device=all_scores.device)
                global_mask_updates.append((abs_start, local_idx))

        total_selected = sum(per_segment_counts)

        return {
            "global_mask_updates": global_mask_updates,
            "recompute_rate": total_selected / max(total_cached_tokens, 1),
            "per_segment_counts": per_segment_counts,
            "R0": R0,
            "threshold": 0.0,
            "H": 0.0,
            "k_star": k,
        }


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def create_scorer(has_kvfocus: bool) -> TokenScorer:
    """Create scorer based on method type.

    KVFocus scorer version is controlled by SGLANG_KVFOCUS_SCORER:
      - "v2" (default): per-token U_L1 scoring (s_i = m_i × U_L1_pt[i])
      - "v1": segment-level U_L2 scoring (s_i = m_i × U_L2(seg))
    """
    if has_kvfocus:
        scorer_version = os.environ.get("SGLANG_KVFOCUS_SCORER", "v2")
        if scorer_version == "v1":
            return KVFocusScorerV1()
        return KVFocusScorer()
    return CacheBlendScorer()


def create_scorer_from_tasks(tasks: List[Dict], has_kvfocus: bool) -> TokenScorer:
    """Create the scorer from the dynamic tasks (KVFocus or CacheBlend)."""
    return create_scorer(has_kvfocus)


def create_selector_from_tasks(tasks: List[Dict]) -> TokenSelector:
    """Create selector from dynamic task dicts (backward compat).

    Used when tasks carry solver/hyperparams from CacheFlexPlan.
    For the fixed-ratio (non-adaptive) KVFocus path, returns CacheBlendSelector
    since it uses the same top-k + keep-last logic.
    """
    if not tasks:
        return ResidualBudgetSelector()

    sample = tasks[0]
    solver = sample.get("solver", None)
    recomp_ratio = sample.get("recomp_ratio", 0.15)
    has_lam = any(t.get("lam") is not None for t in tasks)

    if solver is None:
        # Fixed-ratio mode (no adaptive selection)
        return CacheBlendSelector(recomp_ratio=recomp_ratio)

    if solver == "topk":
        return TopKSelector(
            target_rate=sample.get("target_rate", 0.0),
            max_rate=sample.get("max_rate", 0.50),
        )
    elif solver == "residual_budget":
        return ResidualBudgetSelector(
            rho=sample.get("rho", 0.05),
            gamma=sample.get("gamma", 2.0),
            max_rate=sample.get("max_rate", 0.50),
            b0=sample.get("b0", 0),
            b0_mode=sample.get("b0_mode", "topk"),
            scaling=sample.get("scaling", "linear"),
            beta=sample.get("beta", 1.0),
            eta=sample.get("eta", 0.5),
            n_ref=sample.get("n_ref", 11),
            top_k_peak=sample.get("top_k_peak", 3),
        )
    elif solver == "coverage":
        return CoverageSelector(
            lam=sample.get("lam", 0.5),
            gamma=sample.get("gamma", 2.0),
            max_rate=sample.get("max_rate", 0.50),
            b0=sample.get("b0", 0),
            b0_mode=sample.get("b0_mode", "topk"),
            scaling=sample.get("scaling", "linear"),
            beta=sample.get("beta", 1.0),
            eta=sample.get("eta", 0.5),
            n_ref=sample.get("n_ref", 11),
            top_k_peak=sample.get("top_k_peak", 3),
        )
    elif solver == "threshold":
        return ThresholdSelector(
            lam=sample.get("lam", 0.5),
            max_rate=sample.get("max_rate", 0.50),
            b0=sample.get("b0", 0),
            b0_mode=sample.get("b0_mode", "topk"),
            scaling=sample.get("scaling", "linear"),
            beta=sample.get("beta", 1.0),
            eta=sample.get("eta", 0.5),
            n_ref=sample.get("n_ref", 11),
            top_k_peak=sample.get("top_k_peak", 3),
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")
