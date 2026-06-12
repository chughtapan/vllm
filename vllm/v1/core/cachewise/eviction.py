# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Predictive free-block queue for CacheWise KV cache eviction."""

from bisect import insort
from collections.abc import Callable, Iterator

from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock

# Reserved segment ids. Session segments use the (non-negative) session ids
# assigned by the SessionTracker, plus its PREEMPTED_SESSION_ID (-1).
RECLAIM_SEGMENT = -2
UNHASHED_SEGMENT = -3
DEFAULT_SEGMENT = -4
_RESERVED_SEGMENTS = (RECLAIM_SEGMENT, UNHASHED_SEGMENT, DEFAULT_SEGMENT)


class PredictiveFreeBlockQueue(FreeKVCacheBlockQueue):
    """Drop-in replacement for FreeKVCacheBlockQueue with predictive order.

    Free blocks are partitioned into segments, each an intrusive
    FreeKVCacheBlockQueue (so per-block remove() stays O(1)):

    - reclaim: blocks prepended for immediate reuse (uncached scratch).
      Always evicted first.
    - unhashed: blocks with no cached content. Evicted next, since they can
      never produce a prefix cache hit.
    - per-session segments and a default segment (cached blocks without
      session metadata), evicted in order of decreasing predicted
      time-to-next-reuse. Within a session, blocks are evicted suffix
      before prefix, preserving the usable head of the chain.

    The cross-segment order is refreshed via rebuild(); between rebuilds,
    newly created session segments are inserted at their at-creation
    priority.
    """

    def __init__(
        self,
        blocks: list[KVCacheBlock],
        classify_fn: Callable[[int], int | None],
        priority_fn: Callable[[int], float],
        on_evicted_fn: Callable[[int, int], None],
        default_priority: float,
    ) -> None:
        # Deliberately does NOT call super().__init__: blocks live in
        # per-segment FreeKVCacheBlockQueues instead of one linked list.
        self._classify_fn = classify_fn
        self._priority_fn = priority_fn
        self._on_evicted_fn = on_evicted_fn
        self._default_priority = default_priority

        self._segments: dict[int, FreeKVCacheBlockQueue] = {
            RECLAIM_SEGMENT: FreeKVCacheBlockQueue([]),
            UNHASHED_SEGMENT: FreeKVCacheBlockQueue(blocks),
            DEFAULT_SEGMENT: FreeKVCacheBlockQueue([]),
        }
        self._block_segment: dict[int, int] = {
            block.block_id: UNHASHED_SEGMENT for block in blocks
        }
        # Prioritized segments (default + sessions), most-distant reuse
        # first. The reclaim and unhashed segments always precede these.
        self._order: list[int] = [DEFAULT_SEGMENT]
        self._priorities: dict[int, float] = {DEFAULT_SEGMENT: default_priority}
        self.num_free_blocks = len(blocks)
        self.num_predictive_evictions = 0
        self.eviction_epoch = 0

    def popleft(self) -> KVCacheBlock:
        """Pop the best eviction candidate.

        Raises:
            ValueError: If the queue is empty.
        """
        blocks = self.popleft_n(1) if self.num_free_blocks else []
        if not blocks:
            raise ValueError("No free blocks available")
        return blocks[0]

    def popleft_n(self, n: int) -> list[KVCacheBlock]:
        if n == 0:
            return []
        assert self.num_free_blocks >= n
        ret: list[KVCacheBlock] = []
        remaining = n
        for segment_id in self._eviction_order():
            if remaining == 0:
                break
            segment = self._segments[segment_id]
            take = min(remaining, segment.num_free_blocks)
            if take == 0:
                continue
            popped = segment.popleft_n(take)
            for block in popped:
                del self._block_segment[block.block_id]
            if segment_id not in _RESERVED_SEGMENTS:
                self.num_predictive_evictions += take
                for block in popped:
                    self._on_evicted_fn(block.block_id, segment_id)
            ret.extend(popped)
            remaining -= take
        # Decrement by what was actually popped, not the request size, so a
        # segment/aggregate accounting mismatch surfaces here instead of
        # silently desyncing num_free_blocks.
        assert remaining == 0, "free block accounting desync"
        self.num_free_blocks -= len(ret)
        return ret

    def remove(self, block: KVCacheBlock) -> None:
        segment_id = self._block_segment.pop(block.block_id)
        self._segments[segment_id].remove(block)
        self.num_free_blocks -= 1

    def append(self, block: KVCacheBlock) -> None:
        segment_id = self._classify(block)
        self._get_or_create_segment(segment_id).append(block)
        self._block_segment[block.block_id] = segment_id
        self.num_free_blocks += 1

    def append_n(self, blocks: list[KVCacheBlock]) -> None:
        for block in blocks:
            self.append(block)

    def prepend_n(self, blocks: list[KVCacheBlock]) -> None:
        if not blocks:
            return
        self._segments[RECLAIM_SEGMENT].prepend_n(blocks)
        for block in blocks:
            self._block_segment[block.block_id] = RECLAIM_SEGMENT
        self.num_free_blocks += len(blocks)

    def get_all_free_blocks(self) -> list[KVCacheBlock]:
        """Get all free blocks in eviction order. Mainly used for testing."""
        ret: list[KVCacheBlock] = []
        for segment_id in self._eviction_order():
            ret.extend(self._segments[segment_id].get_all_free_blocks())
        return ret

    def iter_eviction_candidates(
        self, start_after: KVCacheBlock | None = None
    ) -> Iterator[KVCacheBlock]:
        """Iterate free blocks in eviction order (soonest evicted first).

        Walks the reclaim and unhashed segments, then the prioritized
        segments in decreasing predicted time-to-next-reuse, so consumers
        such as proactive KV offload copy out the furthest-reuse blocks
        first. Resuming via ``start_after`` is only valid while
        ``eviction_epoch`` is unchanged and the block is still free.
        """
        order = self._eviction_order()
        start_index = 0
        resume_block: KVCacheBlock | None = None
        if start_after is not None:
            segment_id = self._block_segment.get(start_after.block_id)
            if segment_id is not None and segment_id in order:
                start_index = order.index(segment_id)
                resume_block = start_after
        for segment_id in order[start_index:]:
            yield from self._segments[segment_id].iter_eviction_candidates(resume_block)
            resume_block = None

    def rebuild(self, priorities: dict[int, float]) -> None:
        """Re-order session segments by fresh priority estimates.

        Sessions absent from ``priorities`` (pruned by the tracker) have
        their blocks merged into the default segment.
        """
        self.eviction_epoch += 1
        for segment_id in list(self._segments):
            if segment_id in _RESERVED_SEGMENTS:
                continue
            if segment_id not in priorities:
                self._merge_into_default(segment_id)
            elif self._segments[segment_id].num_free_blocks == 0:
                # Garbage-collect empty session segments.
                del self._segments[segment_id]
        self._priorities = {DEFAULT_SEGMENT: self._default_priority}
        for segment_id, priority in priorities.items():
            if segment_id in self._segments:
                self._priorities[segment_id] = priority
        self._order = sorted(
            self._priorities,
            key=lambda sid: (-self._priorities[sid], sid),
        )

    def reset(self) -> None:
        """Demote all blocks to the unhashed segment (prefix cache reset)."""
        self.eviction_epoch += 1
        for segment_id in list(self._segments):
            if segment_id == UNHASHED_SEGMENT:
                continue
            if segment_id in _RESERVED_SEGMENTS:
                segment = self._segments[segment_id]
            else:
                segment = self._segments.pop(segment_id)
            self._drain_into(segment, UNHASHED_SEGMENT)
        self._order = [DEFAULT_SEGMENT]
        self._priorities = {DEFAULT_SEGMENT: self._default_priority}

    def _eviction_order(self) -> list[int]:
        return [RECLAIM_SEGMENT, UNHASHED_SEGMENT, *self._order]

    def _classify(self, block: KVCacheBlock) -> int:
        if block.block_hash is None:
            return UNHASHED_SEGMENT
        session_id = self._classify_fn(block.block_id)
        if session_id is None:
            return DEFAULT_SEGMENT
        return session_id

    def _get_or_create_segment(self, segment_id: int) -> FreeKVCacheBlockQueue:
        segment = self._segments.get(segment_id)
        if segment is None:
            segment = self._segments[segment_id] = FreeKVCacheBlockQueue([])
            self._priorities[segment_id] = self._priority_fn(segment_id)
            # Keep most-distant reuse first.
            insort(self._order, segment_id, key=lambda sid: -self._priorities[sid])
        return segment

    def _merge_into_default(self, segment_id: int) -> None:
        segment = self._segments.pop(segment_id)
        self._priorities.pop(segment_id, None)
        if segment_id in self._order:
            self._order.remove(segment_id)
        self._drain_into(segment, DEFAULT_SEGMENT)

    def _drain_into(self, segment: FreeKVCacheBlockQueue, target_id: int) -> None:
        target = self._segments[target_id]
        for block in segment.get_all_free_blocks():
            segment.remove(block)
            target.append(block)
            self._block_segment[block.block_id] = target_id
