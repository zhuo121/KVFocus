from __future__ import annotations

from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.utils import convert_to_bigram_key

"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
The radix tree data structure for managing the KV cache.
"""

import heapq
import logging
import sys
import time
from collections import defaultdict
from functools import lru_cache, partial
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)

# HACK(Shizhuo) [2026-05-26 lock_ref leak hunt instrumentation]
# ============================================================
# Per-process trace of inc_lock_ref / dec_lock_ref / cache_finished_req
# Enable via CACHEFLEX_LOCK_TRACE=1. Writes to /tmp/locktrace_<pid>.log.
# Use to find unmatched inc_lock_ref under multi-proc multihoprag hang.
import os as _os
import sys as _sys

_LOCKTRACE_ENABLED = _os.environ.get("CACHEFLEX_LOCK_TRACE") == "1"
_LOCKTRACE_FILE = None

def _locktrace_log(msg: str):
    global _LOCKTRACE_FILE
    if not _LOCKTRACE_ENABLED:
        return
    if _LOCKTRACE_FILE is None:
        _LOCKTRACE_FILE = open(f"/tmp/locktrace_{_os.getpid()}.log", "a", buffering=1)
    _LOCKTRACE_FILE.write(f"{time.time():.4f} {msg}\n")
# ============================================================

from sglang.srt.disaggregation.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
)
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, MatchResult
from sglang.srt.mem_cache.evict_policy import (
    EvictionStrategy,
    FIFOStrategy,
    FILOStrategy,
    LFUStrategy,
    LRUStrategy,
    MRUStrategy,
    PriorityStrategy,
)
from sglang.srt.mem_cache.hicache_storage import get_hash_str, hash_str_to_int64

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class RadixKey:
    def __init__(
        self,
        token_ids: List[int],
        extra_key: Optional[str] = None,
        is_bigram: bool = False,
    ):
        # token ids sequence
        self.token_ids = token_ids
        # extra key (e.g. lora_id, cache_salt)
        self.extra_key = extra_key
        # is bigram key
        self.is_bigram = is_bigram

    def __len__(self) -> int:
        return len(self.token_ids)

    def __iter__(self) -> Iterator[int]:
        return iter(self.token_ids)

    def __getitem__(self, idx: Union[int, slice]) -> "RadixKey":
        if isinstance(idx, slice):
            return RadixKey(self.token_ids[idx], self.extra_key)
        return RadixKey([self.token_ids[idx]], self.extra_key)

    def __repr__(self) -> str:
        preview = self.token_ids[:10]
        return f"RadixKey(extra_key={self.extra_key!r}, token_ids={preview}{'...' if len(self.token_ids) > 10 else ''})"


class TreeNode:

    counter = 0

    def __init__(self, id: Optional[int] = None, priority: int = 0):
        self.children = defaultdict(TreeNode)
        self.parent: TreeNode = None
        self.key: RadixKey = None
        self.value: Optional[torch.Tensor] = None
        self.lock_ref = 0
        self.last_access_time = time.monotonic()
        self.creation_time = time.monotonic()

        self.hit_count = 0
        # indicating the node is locked to protect from eviction
        # incremented when the node is referenced by a storage operation
        self.host_ref_counter = 0
        # store the host indices of KV cache
        self.host_value: Optional[torch.Tensor] = None
        # store hash values of each pages
        self.hash_value: Optional[List[str]] = None
        # priority for priority-aware eviction
        self.priority = priority

        # HACK(Shizhuo)
        self.is_temporary = False  # 标记是否为 CacheFlex 游离节点

        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1

    @property
    def evicted(self):
        return self.value is None

    @property
    def backuped(self):
        return self.host_value is not None

    def protect_host(self):
        """Protect the host value from eviction."""
        self.host_ref_counter += 1

    def release_host(self):
        """Release the host value, allowing it to be evicted."""
        if self.host_ref_counter > 0:
            self.host_ref_counter -= 1
        else:
            raise RuntimeError("Host reference counter is already zero.")

    def get_last_hash_value(self) -> Optional[str]:
        """Returns the hash value of the last page in this node."""
        if self.hash_value is None or len(self.hash_value) == 0:
            return None
        return self.hash_value[-1]

    @lru_cache(maxsize=1)
    def get_prefix_hash_values(self, node: TreeNode) -> List[str]:
        if node is None or node.hash_value is None:
            return []

        return node.get_prefix_hash_values(node.parent) + node.hash_value

    def __lt__(self, other: "TreeNode"):
        return self.last_access_time < other.last_access_time


def _check_extra_key(key0: RadixKey, key1: RadixKey):
    if key0.extra_key != key1.extra_key:
        raise ValueError(
            f"_key_match should be run on the same extra key, but got key0.extra_key={key0.extra_key} != key1.extra_key={key1.extra_key}"
        )


def _key_match_page_size1(key0: RadixKey, key1: RadixKey):
    _check_extra_key(key0, key1)
    i = 0
    for k0, k1 in zip(key0.token_ids, key1.token_ids):
        if k0 != k1:
            break
        i += 1
    return i


def _key_match_paged(key0: RadixKey, key1: RadixKey, page_size: int):
    _check_extra_key(key0, key1)
    min_len = min(len(key0), len(key1))

    i = 0
    while i < min_len:
        if key0.token_ids[i : i + page_size] != key1.token_ids[i : i + page_size]:
            break
        i += page_size

    return i


def get_child_key(key: RadixKey, page_size: int = 1):
    if page_size == 1:
        plain_key = key.token_ids[0]
    else:
        plain_key = tuple(key.token_ids[:page_size])
    if key.extra_key is None:
        return plain_key
    else:
        return (key.extra_key, plain_key)


def compute_node_hash_values(node: "TreeNode", page_size: int) -> List[str]:
    """Compute SHA256-based hash values for position-aware identification.

    Args:
        node: The TreeNode to compute hash values for
        page_size: The page size for chunking tokens

    Returns:
        List of SHA256 hex strings, one per page
    """
    hash_values = []

    # Get parent's last hash value if parent exists
    parent_hash = None
    if node.parent is not None and node.parent.hash_value is not None:
        # Check if parent is root by checking if it has empty key
        if len(node.parent.key) > 0 and len(node.parent.hash_value) > 0:
            parent_hash = node.parent.hash_value[-1]

    # Iterate through node's pages
    for start in range(0, len(node.key), page_size):
        page_tokens = node.key.token_ids[start : start + page_size]
        if not page_tokens:
            continue

        # Use SHA256-based chaining via get_hash_str
        hash_val = get_hash_str(page_tokens, prior_hash=parent_hash)
        hash_values.append(hash_val)
        parent_hash = hash_val

    return hash_values


def split_node_hash_value(
    child_hash_value: Optional[List[str]], split_len: int, page_size: int
) -> tuple[Optional[List[str]], Optional[List[str]]]:
    """Split hash_value between parent and child nodes during node splitting.

    Args:
        child_hash_value: The hash_value list from the child node being split
        split_len: The length at which to split (in tokens)
        page_size: The page size for calculating number of pages

    Returns:
        Tuple of (new_node_hash_value, updated_child_hash_value)
    """
    if child_hash_value is None:
        return None, None

    if page_size == 1:
        split_pages = split_len
    else:
        split_pages = split_len // page_size

    new_node_hash = child_hash_value[:split_pages]
    child_hash = child_hash_value[split_pages:]

    return new_node_hash, child_hash


class RadixCache(BasePrefixCache):
    def __init__(self, params: CacheInitParams):
        self.disable = params.disable
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.page_size = params.page_size
        self.enable_kv_cache_events = params.enable_kv_cache_events
        self.is_eagle = params.is_eagle
        self.disable_finished_insert = params.disable_finished_insert
        self.eviction_policy = params.eviction_policy.lower()

        self.kv_event_queue = []

        # HACK(Shizhuo)
        # 帧指纹索引库：frame_hash -> {physical_indices, old_offset, token_len}
        self.frame_to_kv_info = {}
        # ===========================

        if params.enable_metrics:
            self.init_metrics_collector()

        if self.token_to_kv_pool_allocator:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")

        if self.page_size == 1:
            self.key_match_fn = _key_match_page_size1
            self.get_child_key_fn = get_child_key
        else:
            self.key_match_fn = partial(_key_match_paged, page_size=self.page_size)
            self.get_child_key_fn = partial(get_child_key, page_size=self.page_size)

        if self.eviction_policy == "lru":
            self.eviction_strategy: EvictionStrategy = LRUStrategy()
        elif self.eviction_policy == "lfu":
            self.eviction_strategy: EvictionStrategy = LFUStrategy()
        elif self.eviction_policy == "fifo":
            self.eviction_strategy: EvictionStrategy = FIFOStrategy()
        elif self.eviction_policy == "mru":
            self.eviction_strategy: EvictionStrategy = MRUStrategy()
        elif self.eviction_policy == "filo":
            self.eviction_strategy: EvictionStrategy = FILOStrategy()
        elif self.eviction_policy == "priority":
            self.eviction_strategy: EvictionStrategy = PriorityStrategy()
        else:
            raise ValueError(
                f"Unknown eviction policy: {self.eviction_policy}. Supported policies: 'lru', 'lfu', 'fifo', 'mru', 'filo', 'priority'."
            )
        self.reset()

    @classmethod
    def create_simulated(
        self,
        disable: bool = False,
        mock_allocator: Optional[Any] = None,
        page_size: int = 1,
        enable_kv_cache_events: bool = False,
    ) -> RadixCache:
        """Init a radix cache without memory pools for simulation purpose."""
        params = CacheInitParams(
            disable=disable,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=mock_allocator,
            page_size=page_size,
            enable_kv_cache_events=enable_kv_cache_events,
        )
        return RadixCache(params)

    ##### Public API #####

    def reset(self):
        # HACK(Shizhuo): Clear video frame fingerprint DB to prevent hash collisions
        # across different videos in multi-shot experiments
        if hasattr(self, "frame_to_kv_info"):
            self.frame_to_kv_info.clear()
        # Initialize root with minimum priority so any real priority overrides it
        self.root_node = TreeNode(priority=-sys.maxsize)
        self.root_node.key = RadixKey(token_ids=[], extra_key=None)
        self.root_node.value = []
        self.root_node.host_value = []
        self.root_node.lock_ref = 1
        self.root_node.hash_value = []
        self.evictable_size_ = 0
        self.protected_size_ = 0
        self._record_all_cleared_event()

    def maybe_bigram_convert(
        self, key: RadixKey, value: Optional[torch.Tensor] = None
    ) -> Tuple[RadixKey, Optional[torch.Tensor]]:
        if self.is_eagle and not key.is_bigram:
            key.token_ids = convert_to_bigram_key(key.token_ids)
            if value is not None:
                value = value[: len(key)]

        return key, value

    def match_prefix(self, key: RadixKey, **kwargs) -> MatchResult:
        """Find the longest cached prefix of ``key`` in the radix tree.

        The logical namespace for prefix matching is determined by both the
        token id sequence and the optional ``extra_key`` carried by ``RadixKey``.
        Entries that share identical leading token ids but have *different*
        ``extra_key`` values are intentionally kept disjoint and never share
        prefix nodes. This is useful to:

        * Isolate KV cache lines for different LoRA / adapter IDs.
        * Separate requests that intentionally should not share state (e.g.,
          different sampling salt, cache version, or retrieval augmentation
          context) by supplying a distinct ``extra_key``.

        Args:
            key (RadixKey): The lookup key containing a list of token ids and an
                optional ``extra_key`` namespace tag. If ``page_size > 1`` the
                length is internally truncated to a multiple of ``page_size``
                before matching. Passing an empty key returns an empty result
                with the root as the last node.
            **kwargs: Reserved for future extensions (ignored currently).

        Returns:
            MatchResult: ``device_indices`` is a 1-D ``torch.int64`` tensor of
            the concatenated KV cache indices corresponding to the longest
            cached prefix (may be length 0). ``last_device_node`` and
            ``last_host_node`` (currently the same) are the tree node objects
            representing the terminal node of the matched prefix. This method
            may mutate internal structure by splitting an existing node if the
            match ends inside a stored segment.

        Internal updates:
            * Refreshes access metadata (timestamps) used by the
                configured eviction strategy.
            * If the lookup ends inside a stored segment the node is split once
                to expose a precise boundary; this structural refinement improves
                subsequent match efficiency and does not duplicate data.
        """
        key, _ = self.maybe_bigram_convert(key)

        def empty_match_result():
            return MatchResult(
                device_indices=torch.empty(
                    (0,),
                    dtype=torch.int64,
                    device=self.device,
                ),
                last_device_node=self.root_node,
                last_host_node=self.root_node,
            )

        if self.disable or len(key) == 0:
            return empty_match_result()

        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]

        if len(key) == 0:
            return empty_match_result()

        value, last_node = self._match_prefix_helper(self.root_node, key)
        if value:
            value = torch.cat(value)
        else:
            value = torch.empty((0,), dtype=torch.int64, device=self.device)
        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_node,
        )

    def insert(self, key: RadixKey, value=None, chunked=False, priority: int = 0):
        if self.disable:
            return 0

        if value is None:
            value = torch.tensor(key.token_ids, dtype=torch.int64)

        key, value = self.maybe_bigram_convert(key, value)

        return self._insert_helper(self.root_node, key, value, priority)

    def _page_align_keys(self, key: list) -> list:
        if self.page_size == 1:
            return key
        page_aligned_len = len(key) // self.page_size * self.page_size
        return key[:page_aligned_len]

    # HACK(Shizhuo)
    # HACK(Shizhuo) [CacheFlex] Phantom Size 管理接口
    @property
    def _cf_lock(self):
        if not hasattr(self, "_cacheflex_lock"):
            import threading
            self._cacheflex_lock = threading.Lock()
        return self._cacheflex_lock

    def inc_phantom_size(self, num: int): 
        with self._cf_lock:
            # 确保 protected_size_ 属性存在
            if not hasattr(self, "protected_size_"):
                self.protected_size_ = 0
            self.protected_size_ += num
        
    def dec_phantom_size(self, num: int): 
        with self._cf_lock:
            if not hasattr(self, "protected_size_"):
                self.protected_size_ = 0
            self.protected_size_ -= num
            if self.protected_size_ < 0: 
                self.protected_size_ = 0

    # HACK(Shizhuo) [CacheFlex] JIT 查找接口
    def lookup_indices_with_node(self, key):
        res = self.match_prefix(key)
        # 仅当完全匹配时返回 Block Indices 和 Node
        if len(res.device_indices) * self.page_size >= len(key.token_ids):
             return res.device_indices, res.last_device_node
        return None, None

    def cache_finished_req(self, req: Req, is_insert: bool = True):
        """Cache request when it finishes."""
        # HACK(Shizhuo)
        if hasattr(req, "multimodal_inputs") and req.multimodal_inputs:
            new_frames_this_req = 0 # 记录本轮新注册的帧数

            for item in req.multimodal_inputs.mm_items:
                if item.is_video() and hasattr(item, "frame_hashes"):
                    # 1. 拿到该请求进入 Prefill 时的前缀长度 (基准偏移)
                    base_offset = getattr(req, "extend_prefix_len", 0)

                    # 2. 获取该请求对应的物理页表切片
                    phys_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx]
                    
                    for i, f_hash in enumerate(item.frame_hashes):
                        if f_hash not in self.frame_to_kv_info:
                            start = req.multimodal_inputs.image_offsets[i]
                            length = req.multimodal_inputs.image_pad_len[i]

                            # 计算绝对 Position ID (1D 语义下的"相位")
                            # 这在 Qwen2 (LLaVA-Video 基座) 下是绝对准确的
                            actual_pos_id = base_offset + start
                        
                            # 记录这一帧指纹、物理块、以及【首 Token 的 Position ID】
                            self.frame_to_kv_info[f_hash] = {
                                "physical_indices": phys_indices[start : start + length].clone(),
                                "old_pos_id": actual_pos_id,
                                "old_offset": start, # 冗余记录用于校验
                                "token_len": length
                            }
                            new_frames_this_req += 1

            # 仅在有新帧存入且开启 INFO 级别日志时打印
            if new_frames_this_req > 0 and logger.isEnabledFor(logging.INFO):
                logger.info(
                    "[CacheFlex Cache] Incremental storage success for Req: %s. "
                    "New frames registered: %d, Fingerprint DB size: %d",
                    req.rid[:4], 
                    new_frames_this_req, 
                    len(self.frame_to_kv_info)
                )
        # ==================================================================

        # HACK(Shizhuo) [CacheFlex src_node leak fix, 2026-05-26]
        # ===================================================
        # The scheduler's CacheFlex timeline-scan block (scheduler.py) calls
        # inc_lock_ref(src_node) for every text-cache hit, then saves them on
        # req.cacheflex_src_nodes. The matching dec_lock_ref was originally placed
        # inside the `is_active_cacheflex` branch below — but that branch is gated on
        # `cacheflex_allocated_indices`, a field nothing in the codebase ever writes.
        # So is_active_cacheflex is always False, the entire branch is dead code, and
        # every CacheFlex hit permanently leaked a +1 lock_ref on its src_node prefix
        # path. Symptom: protected_size in the radix tree grows unbounded across
        # samples until allocation fails -> waiting_queue stalls -> scheduler spins
        # in event_loop_overlap forever -> bench hang. Fix is unconditional: any req
        # that has cacheflex_src_nodes (regardless of allocated_indices) gets its
        # src locks released exactly once here, before falling through to either the
        # dead CacheFlex branch (no-op) or the standard sglang cleanup.
        src_nodes_to_release = getattr(req, "cacheflex_src_nodes", None)
        if _LOCKTRACE_ENABLED:
            _has_cf_meta = getattr(req, "cacheflex_metadata", None) is not None
            _is_cf_proc = getattr(req, "is_cacheflex_processed", False)
            _already_released = getattr(req, "cacheflex_src_released", False)
            _n_srcn = len(src_nodes_to_release) if src_nodes_to_release else 0
            _has_allocated = getattr(req, "cacheflex_allocated_indices", None) is not None
            _rid_short = req.rid[:8] if hasattr(req, "rid") and req.rid else "?"
            _locktrace_log(f"FIN rid={_rid_short} cf_proc={_is_cf_proc} cf_meta={_has_cf_meta} srcn={_n_srcn} released={_already_released} alloc_indices={_has_allocated} proto_at_entry={self.protected_size_}")
        if src_nodes_to_release and not getattr(req, "cacheflex_src_released", False):
            for node in src_nodes_to_release:
                if node is not None and node != self.root_node:
                    self.dec_lock_ref(node)
            req.cacheflex_src_released = True
        # ===================================================

        # HACK(Shizhuo)
        # HACK(Shizhuo) [CacheFlex] 请求结束清理：物理释放 + 逻辑销账
         # 同样仅针对被 Scheduler 激活了 CacheFlex 的请求进行特殊处理
        is_active_cacheflex = getattr(req, "cacheflex_allocated_indices", None) is not None

        if is_active_cacheflex:
            # 防止重复调用导致 Double Free
            if getattr(req, "cacheflex_cleaned", False): return
            req.cacheflex_cleaned = True

            try:
                # 1. 释放 Prefix (CacheFlex 拼接部分)
                # 这部分显存由 Scheduler 直接通过 allocator 分配，必须手动释放
                kv_indices = req.cacheflex_allocated_indices
                allocator = self.token_to_kv_pool_allocator
                
                if kv_indices.device != allocator.device:
                    kv_indices = kv_indices.to(allocator.device)
                
                # 强制释放物理内存 (绕过可能的 group 限制)
                was_group = not allocator.is_not_in_free_group
                if was_group: allocator.is_not_in_free_group = True
                allocator.free(kv_indices)
                if was_group: allocator.is_not_in_free_group = False
                
                # 2. 销账 (归还 Phantom Size)
                # 将 Scheduler 借用的逻辑账目归还，确保 available + phantom + evictable = total
                if hasattr(req, "cacheflex_token_len"):
                    self.dec_phantom_size(req.cacheflex_token_len)
                else:
                    self.dec_phantom_size(len(kv_indices) * self.page_size)

                # 3. 释放 Suffix (新生成部分)
                # 因为我们拦截了入树操作，SGLang 原生逻辑不会接管这部分显存，
                # 必须根据 token 长度计算范围并手动释放，否则会发生物理显存泄漏。
                total_len = req.pop_committed_kv_cache()
                
                # 获取后缀起始位置 (即 CacheFlex 前缀长度)
                suffix_start_idx = getattr(req, "cacheflex_token_len", 0)
                
                if total_len > suffix_start_idx:
                    suffix_indices = self.req_to_token_pool.req_to_token[
                        req.req_pool_idx, suffix_start_idx:total_len
                    ].clone()
                    
                    if suffix_indices.device != self.token_to_kv_pool_allocator.device:
                        suffix_indices = suffix_indices.to(self.token_to_kv_pool_allocator.device)
                    suffix_indices = suffix_indices.to(torch.int64)
                    
                    self.token_to_kv_pool_allocator.free(suffix_indices)

            except Exception:
                logger.warning(f"[CacheFlex][CLEANUP-ERR] rid={req.rid[:4]}")

            # 4. 释放 Request Slot (页表项)
            self.req_to_token_pool.free(req.req_pool_idx)
            
            # 5. 清理引用计数
            # CacheFlex 请求的 last_node 被强制指向了 root，这里解锁 root
            if req.last_node is not None and req.last_node != self.root_node:
                self.dec_lock_ref(req.last_node)
            
            # 6. [FIX] 释放源节点引用 (Source Node)
            # 关键修复：释放我们在 Scheduler 中手动锁定的源节点
            # HACK(Shizhuo) [2026-05-26] guarded by cacheflex_src_released so we don't
            # double-dec if the top-of-function unconditional cleanup already ran.
            src_nodes = getattr(req, "cacheflex_src_nodes", None)
            if (
                getattr(req, "cacheflex_src_locked", False)
                and src_nodes
                and not getattr(req, "cacheflex_src_released", False)
            ):
                n_nodes = len(src_nodes)

                for node in src_nodes:
                    if node is not None and node != self.root_node:
                        self.dec_lock_ref(node)
                req.cacheflex_src_released = True
            src_node = getattr(req, "cacheflex_src_node", None)
            if (not src_nodes) and (src_node is not None) and (src_node != self.root_node):
                self.dec_lock_ref(src_node)
                req.cacheflex_src_node = None
            
            # [关键] 提前返回，跳过下方所有原生逻辑
            return
        # ==============================================

        # In deterministic mode, disable finished request insertion to radix cache
        if self.disable_finished_insert:
            is_insert = False

        kv_committed_len = req.pop_committed_kv_cache()
        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_committed_len
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            self.req_to_token_pool.free(req.req_pool_idx)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        # Maybe convert to bigram keys for EAGLE
        keys = convert_to_bigram_key(req.fill_ids) if self.is_eagle else req.fill_ids
        keys = self._page_align_keys(keys)
        values = kv_indices[: len(keys)].to(dtype=torch.int64, copy=True)
        radix_key = RadixKey(keys, req.extra_key, is_bigram=self.is_eagle)

        # Radix Cache takes one ref in memory pool
        if is_insert:
            priority = getattr(req, "priority", 0) or 0
            new_prefix_len = self.insert(radix_key, values, priority=priority)
            # Free the duplicates that were already in the tree
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len : new_prefix_len]
            )
        else:
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len : len(keys)]
            )

        # free the unaligned tail
        self.token_to_kv_pool_allocator.free(kv_indices[len(keys) :])

        # Remove req slot release the cache lock
        self.req_to_token_pool.free(req.req_pool_idx)
        self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: Req, chunked=False):
        """Cache request when it is unfinished."""
        # HACK(Shizhuo)
        # HACK(Shizhuo) [CacheFlex] 拦截入树：防止游离 Block 被 Radix 树接管导致账目不平
        # 仅当 Scheduler 确实为该请求分配了 CacheFlex 专用物理块时才拦截。
        # 如果只检查 sampling_params，会导致 disable 模式下误拦截正常请求。
        is_active_cacheflex = getattr(req, "cacheflex_allocated_indices", None) is not None
        
        if is_active_cacheflex:
            # CacheFlex 的显存块是“游离”的（记录在 Phantom Size 中），
            # 禁止它们进入 Radix Tree，防止 RadixCache 将其重复统计为 Evictable，
            # 从而导致账目不平 (Surplus Error)。
            return

        # ===============================================

        if self.disable:
            return

        token_ids = req.fill_ids
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        # Maybe convert to bigram keys for EAGLE
        keys = convert_to_bigram_key(req.fill_ids) if self.is_eagle else req.fill_ids
        keys = self._page_align_keys(keys)
        values = kv_indices[: len(keys)].to(dtype=torch.int64, copy=True)
        radix_key = RadixKey(keys, req.extra_key, is_bigram=self.is_eagle)

        # Radix Cache takes one ref in memory pool
        new_prefix_len = self.insert(
            radix_key,
            values,
            chunked=chunked,
            priority=getattr(req, "priority", 0) or 0,
        )

        self.token_to_kv_pool_allocator.free(
            kv_indices[req.cache_protected_len : new_prefix_len]
        )

        # The prefix indices could be updated, reuse it
        match_result = self.match_prefix(radix_key)
        (new_indices, new_last_node) = (
            match_result.device_indices,
            match_result.last_device_node,
        )
        assert len(new_indices) == len(keys), f"{len(new_indices)=}, {len(keys)=}"

        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(req.cache_protected_len, len(new_indices))),
            new_indices[req.cache_protected_len :],
        )

        # The cache_protected_len is not always equal to len(req.prefix_indices)
        # since for page_size > 1, the partial part is added to req.prefix_indices, but that part of kv indices is not added to the tree.
        # It should be freed in the next cache_unfinished_req and final cache_finished_req to avoid memory leak.
        # So we introduce this `cache_protected_len` field to make sure the partial part can be freed correctly.
        req.cache_protected_len = len(new_indices)

        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)

        # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
        # - page_size != 1: there is a partial page at the end, keep the full kv_indices
        # - eagle case: bigram keys will only cache len - 1 kv indices
        if len(new_indices) < len(kv_indices):
            req.prefix_indices = torch.cat(
                [new_indices, kv_indices[len(new_indices) :]]
            )
        else:
            req.prefix_indices = new_indices

        req.last_node = new_last_node

    def pretty_print(self):
        self._print_helper(self.root_node, 0)
        print(f"#tokens: {self.total_size()}")

    def total_size(self):
        return self._total_size_helper()

    def evict(self, num_tokens: int, 
        # HACK(Shizhuo)
        evict_all: bool = False
        ):
        if self.disable:
            return

        # HACK(Shizhuo)
        # HACK(Shizhuo) [CacheFlex] 处理全量清空请求
        if evict_all:
            # 1. 清空视频帧指纹库
            # 必须做！否则下个视频推理时，Scheduler 会查到旧视频的物理地址（已释放），导致严重的显存访问错误
            if hasattr(self, "frame_to_kv_info"):
                self.frame_to_kv_info.clear()
            
            # 2. 归还所有幽灵显存计数
            if hasattr(self, "protected_size_"):
                self.protected_size_ = 0
            
            # 3. 将目标驱逐数量设为无穷大，强制下面的循环清空整个 Radix Tree
            num_tokens = float("inf")
        # =====================================

        start_time = time.perf_counter()
        leaves = self._collect_leaves()
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        while num_evicted < num_tokens and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)

            self.token_to_kv_pool_allocator.free(x.value)
            num_evicted += len(x.value)
            self._delete_leaf(x)

            if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

            self._record_remove_event(x)

        self.update_eviction_metrics(num_evicted, start_time)

    def inc_lock_ref(self, node: TreeNode):
        if self.disable:
            return 0

        # HACK(Shizhuo) [locktrace] capture caller for leak hunt
        if _LOCKTRACE_ENABLED:
            _f = _sys._getframe(1)
            _caller = f"{_os.path.basename(_f.f_code.co_filename)}:{_f.f_lineno}"
            _proto_before = self.protected_size_
            _node_id = id(node)
            _key_len = len(node.key)

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.key)
                self.protected_size_ += len(node.key)
                delta -= len(node.key)
            node.lock_ref += 1
            node = node.parent

        if _LOCKTRACE_ENABLED:
            _locktrace_log(f"INC node={_node_id} klen={_key_len} delta={delta} proto={_proto_before}->{self.protected_size_} caller={_caller}")
        return delta

    def dec_lock_ref(self, node: TreeNode):
        if self.disable:
            return 0

        # HACK(Shizhuo) [locktrace]
        if _LOCKTRACE_ENABLED:
            _f = _sys._getframe(1)
            _caller = f"{_os.path.basename(_f.f_code.co_filename)}:{_f.f_lineno}"
            _proto_before = self.protected_size_
            _node_id = id(node)
            _key_len = len(node.key)

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.key)
                self.protected_size_ -= len(node.key)
                delta += len(node.key)
            node.lock_ref -= 1
            if node.parent is None:
                assert (
                    node is self.root_node
                ), f"This request holds the node from another tree"
            node = node.parent

        if _LOCKTRACE_ENABLED:
            _locktrace_log(f"DEC node={_node_id} klen={_key_len} delta={delta} proto={_proto_before}->{self.protected_size_} caller={_caller}")
        return delta

    def evictable_size(self):
        return self.evictable_size_

    def protected_size(self):
        # protected size refers to the size of the cache that is locked
        return self.protected_size_

    def all_values_flatten(self):
        values = []

        def _dfs_helper(node: TreeNode):
            for _, child in node.children.items():
                values.append(child.value)
                _dfs_helper(child)

        _dfs_helper(self.root_node)
        return torch.cat(values)

    ##### Internal Helper Functions #####

    def _match_prefix_helper(self, node: TreeNode, key: RadixKey):
        access_time = time.monotonic()
        node.last_access_time = access_time

        child_key = self.get_child_key_fn(key)

        value = []
        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = access_time
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                node = new_node
                break
            else:
                value.append(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = self.get_child_key_fn(key)

        return value, node

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int):
        # new_node -> child
        # New node inherits child's priority (represents shared prefix)
        self._record_remove_event(child)
        new_node = TreeNode(priority=child.priority)
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len]
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:]
        new_node.parent.children[self.get_child_key_fn(key)] = new_node

        # Split hash_value if it was already computed, otherwise leave as None
        new_node.hash_value, child.hash_value = split_node_hash_value(
            child.hash_value, split_len, self.page_size
        )

        return new_node

    def _insert_helper(self, node: TreeNode, key: RadixKey, value, priority: int = 0):
        # Convert None priority to 0
        if priority is None:
            priority = 0
        access_time = time.monotonic()
        node.last_access_time = access_time
        # Update priority along the path (take max to propagate higher priority)
        node.priority = max(node.priority, priority)
        if len(key) == 0:
            return 0

        child_key = self.get_child_key_fn(key)

        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = access_time
            prefix_len = self.key_match_fn(node.key, key)
            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                new_node.priority = max(new_node.priority, priority)
                node = new_node
            else:
                node.priority = max(node.priority, priority)

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode(priority=priority)
            new_node.parent = node
            new_node.key = key
            new_node.value = value
            node.children[child_key] = new_node
            self.evictable_size_ += len(key)
            # Hash will be computed lazily during event emission
            self._record_store_event(new_node)
        return total_prefix_length

    def _print_helper(self, node: TreeNode, indent: int):
        """Prints the radix tree in a human-readable format."""
        stack = [(node, indent)]
        while stack:
            current_node, current_indent = stack.pop()
            print(
                " " * current_indent,
                len(current_node.key),
                current_node.key.token_ids[:10],
                f"r={current_node.lock_ref}",
            )
            for key, child in current_node.children.items():
                stack.append((child, current_indent + 2))

                assert key == self.get_child_key_fn(
                    child.key
                ), f"{key=}, {self.get_child_key_fn(child.key)=}"

    def _delete_leaf(self, node):
        for k, v in node.parent.children.items():
            if v == node:
                break
        del node.parent.children[k]
        self.evictable_size_ -= len(node.key)

    def _total_size_helper(self):
        total_size = 0
        stack = [self.root_node]
        while stack:
            current_node = stack.pop()
            total_size += len(current_node.value)
            for child in current_node.children.values():
                if child.evicted:
                    continue
                stack.append(child)
        return total_size

    def _collect_leaves(self):
        ret_list = []
        stack = list(self.root_node.children.values())

        while stack:
            cur_node = stack.pop()
            if len(cur_node.children) == 0:
                if cur_node.lock_ref == 0:
                    ret_list.append(cur_node)
            else:
                stack.extend(cur_node.children.values())

        return ret_list

    def _record_store_event(self, node: TreeNode):
        # One BlockStored per ``page_size`` chunk.
        if self.enable_kv_cache_events:
            # Compute hash_value lazily if not already set
            if node.hash_value is None:
                node.hash_value = compute_node_hash_values(node, self.page_size)

            # Get parent's last hash value for first page
            parent_block_hash = None
            if node.parent is not None and node.parent != self.root_node:
                if (
                    node.parent.hash_value is not None
                    and len(node.parent.hash_value) > 0
                ):
                    parent_block_hash = hash_str_to_int64(node.parent.hash_value[-1])

            page_index = 0
            for start in range(0, len(node.key), self.page_size):
                page_tokens = node.key.token_ids[start : start + self.page_size]
                if not page_tokens:
                    continue

                block_hash = hash_str_to_int64(node.hash_value[page_index])

                self.kv_event_queue.append(
                    BlockStored(
                        block_hashes=[block_hash],
                        parent_block_hash=parent_block_hash,
                        token_ids=page_tokens,
                        block_size=len(page_tokens),
                        lora_id=None,
                    )
                )

                parent_block_hash = block_hash
                page_index += 1

    def _record_remove_event(self, node: TreeNode):
        # One BlockRemoved per chunk.
        if self.enable_kv_cache_events:
            # Compute hash_value lazily if not already set (must match what was stored)
            if node.hash_value is None:
                node.hash_value = compute_node_hash_values(node, self.page_size)

            page_index = 0
            for start in range(0, len(node.key), self.page_size):
                page_tokens = node.key.token_ids[start : start + self.page_size]
                if not page_tokens:
                    continue

                block_hash = hash_str_to_int64(node.hash_value[page_index])

                self.kv_event_queue.append(BlockRemoved(block_hashes=[block_hash]))

                page_index += 1

    def _record_all_cleared_event(self):
        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

    def take_events(self):
        """Atomically takes all events and clears the queue.

        Returns:
            A list of KV cache events.
        """
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events


if __name__ == "__main__":
    tree = RadixCache.create_simulated()

    # Example token id sequences (as lists of ints)
    tree.insert(RadixKey(token_ids=[1, 2, 3], extra_key=None))
    tree.insert(RadixKey(token_ids=[1, 2, 3], extra_key=None))
    tree.insert(RadixKey(token_ids=[1, 2, 4, 5], extra_key=None))
    tree.insert(RadixKey(token_ids=[1, 2, 4, 5, 6, 7], extra_key=None))
    tree.insert(RadixKey(token_ids=[8, 9, 10, 11, 12], extra_key=None))
    tree.pretty_print()

    print(tree.match_prefix(RadixKey(token_ids=[1, 2, 3, 13, 14], extra_key=None)))
