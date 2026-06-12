# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the CacheWise predictive free-block queue."""

import random

import pytest

from vllm.v1.core.cachewise.eviction import (
    PredictiveFreeBlockQueue,
)
from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock


def make_blocks(n: int) -> list[KVCacheBlock]:
    return [KVCacheBlock(i) for i in range(n)]


def set_hash(block: KVCacheBlock, tag: bytes | None = None) -> None:
    block._block_hash = tag or bytes([block.block_id % 256]) * 8


def make_queue(
    blocks: list[KVCacheBlock],
    owners: dict[int, int] | None = None,
    priorities: dict[int, float] | None = None,
    default_priority: float = 120.0,
):
    """Queue with dict-driven classification for testing."""
    owners = owners if owners is not None else {}
    priorities = priorities if priorities is not None else {}
    evicted: list[tuple[int, int]] = []
    queue = PredictiveFreeBlockQueue(
        blocks=blocks,
        classify_fn=owners.get,
        priority_fn=lambda sid: priorities.get(sid, default_priority),
        on_evicted_fn=lambda block_id, sid: evicted.append((block_id, sid)),
        default_priority=default_priority,
    )
    return queue, evicted


def test_cold_start_matches_lru_order():
    blocks = make_blocks(8)
    queue, _ = make_queue(blocks)
    assert queue.num_free_blocks == 8
    # All blocks start unhashed: popped in block-id order, like LRU.
    assert [queue.popleft().block_id for _ in range(8)] == list(range(8))
    with pytest.raises(ValueError, match="No free blocks"):
        queue.popleft()


def test_lru_equivalence_property():
    """With no session tags, random free/touch/pop traces must match LRU
    within each hash class (unhashed evicts before cached by design)."""
    rng = random.Random(42)
    for all_hashed in (True, False):
        blocks_a = make_blocks(64)
        blocks_b = make_blocks(64)
        reference = FreeKVCacheBlockQueue(blocks_a)
        queue, _ = make_queue(blocks_b)

        in_queue_a = {b.block_id: b for b in blocks_a}
        in_queue_b = {b.block_id: b for b in blocks_b}
        out_a: dict[int, KVCacheBlock] = {}
        out_b: dict[int, KVCacheBlock] = {}

        for _ in range(500):
            op = rng.choice(["pop", "append", "remove"])
            if op == "pop" and reference.num_free_blocks > 0:
                got_a = reference.popleft()
                got_b = queue.popleft()
                assert got_a.block_id == got_b.block_id
                del in_queue_a[got_a.block_id]
                del in_queue_b[got_b.block_id]
                out_a[got_a.block_id] = got_a
                out_b[got_b.block_id] = got_b
                # Mimic eviction-on-allocation.
                got_a.reset_hash()
                got_b.reset_hash()
            elif op == "append" and out_a:
                block_id = rng.choice(sorted(out_a))
                block_a = out_a.pop(block_id)
                block_b = out_b.pop(block_id)
                if all_hashed:
                    set_hash(block_a)
                    set_hash(block_b)
                reference.append(block_a)
                queue.append(block_b)
                in_queue_a[block_id] = block_a
                in_queue_b[block_id] = block_b
            elif op == "remove" and in_queue_a:
                block_id = rng.choice(sorted(in_queue_a))
                block_a = in_queue_a.pop(block_id)
                block_b = in_queue_b.pop(block_id)
                reference.remove(block_a)
                queue.remove(block_b)
                out_a[block_id] = block_a
                out_b[block_id] = block_b
            assert reference.num_free_blocks == queue.num_free_blocks


def test_unhashed_evicted_before_cached():
    blocks = make_blocks(4)
    queue, _ = make_queue(blocks)
    popped = queue.popleft_n(4)
    # Re-free: 0 and 1 cached, 2 and 3 unhashed.
    set_hash(popped[0])
    set_hash(popped[1])
    queue.append_n(popped)
    order = [queue.popleft().block_id for _ in range(4)]
    assert order == [2, 3, 0, 1]


def test_session_eviction_order_furthest_reuse_first():
    blocks = make_blocks(9)
    owners: dict[int, int] = {}
    priorities = {1: 5.0, 2: 500.0}
    queue, evicted = make_queue(blocks, owners, priorities)
    popped = queue.popleft_n(9)
    for block in popped:
        set_hash(block)
    # Session 1 (reuse in 5s) owns 0-2, session 2 (reuse in 500s) owns 3-5,
    # blocks 6-8 are untagged (default priority 120s).
    owners.update({0: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2})
    # Free tail-first within each session, like the kv cache manager does.
    queue.append_n([popped[2], popped[1], popped[0]])
    queue.append_n([popped[5], popped[4], popped[3]])
    queue.append_n([popped[6], popped[7], popped[8]])
    queue.rebuild(priorities)

    order = [queue.popleft().block_id for _ in range(9)]
    # Session 2 evicts first (furthest reuse), suffix before prefix; then
    # the default bucket; session 1 last.
    assert order == [5, 4, 3, 6, 7, 8, 2, 1, 0]
    assert [(bid, sid) for bid, sid in evicted if sid == 2] == [
        (5, 2),
        (4, 2),
        (3, 2),
    ]


def test_rebuild_reorders_after_priority_change():
    blocks = make_blocks(4)
    owners = {0: 1, 1: 1, 2: 2, 3: 2}
    priorities = {1: 10.0, 2: 100.0}
    queue, _ = make_queue(blocks, owners, priorities)
    popped = queue.popleft_n(4)
    for block in popped:
        set_hash(block)
    queue.append_n(popped)
    queue.rebuild(dict(priorities))
    # Session 2 is furthest: evicts first.
    assert queue.popleft().block_id in (2, 3)
    # Session 1 becomes furthest after the update.
    queue.rebuild({1: 1000.0, 2: 100.0})
    assert queue.popleft().block_id in (0, 1)


def test_rebuild_merges_pruned_sessions_into_default():
    blocks = make_blocks(2)
    owners = {0: 1, 1: 1}
    queue, _ = make_queue(blocks, owners, {1: 50.0})
    popped = queue.popleft_n(2)
    for block in popped:
        set_hash(block)
    queue.append_n(popped)
    # Session 1 disappears (TTL pruned).
    queue.rebuild({})
    assert queue.num_free_blocks == 2
    assert [queue.popleft().block_id for _ in range(2)] == [0, 1]


def test_remove_from_any_segment():
    blocks = make_blocks(6)
    owners = {0: 1}
    queue, _ = make_queue(blocks, owners, {1: 50.0})
    popped = queue.popleft_n(6)
    for block in popped[:2]:
        set_hash(block)
    queue.append_n(popped)
    # Touch blocks from session, default, and unhashed segments.
    for block in (popped[0], popped[1], popped[4]):
        queue.remove(block)
    assert queue.num_free_blocks == 3
    remaining = {queue.popleft().block_id for _ in range(3)}
    assert remaining == {2, 3, 5}


def test_prepend_n_reclaim_priority():
    blocks = make_blocks(5)
    queue, _ = make_queue(blocks)
    popped = queue.popleft_n(3)
    for block in popped:
        set_hash(block)
    queue.append_n(popped[:1])
    queue.prepend_n(popped[1:])
    # Reclaimed blocks (1, 2) pop before everything else.
    assert queue.popleft().block_id == 1
    assert queue.popleft().block_id == 2
    # Then remaining unhashed initial blocks, then the cached one.
    assert [queue.popleft().block_id for _ in range(3)] == [3, 4, 0]


def test_popleft_n_spans_segments():
    blocks = make_blocks(4)
    owners = {0: 1, 1: 1}
    queue, _ = make_queue(blocks, owners, {1: 600.0})
    popped = queue.popleft_n(4)
    for block in popped:
        set_hash(block)
    queue.append_n(popped)
    queue.rebuild({1: 600.0})
    got = [block.block_id for block in queue.popleft_n(4)]
    # Session 1 first (further than default), then default blocks.
    assert got == [0, 1, 2, 3]


def test_reset_demotes_everything_to_unhashed():
    blocks = make_blocks(4)
    owners = {0: 1, 1: 1}
    queue, _ = make_queue(blocks, owners, {1: 50.0})
    popped = queue.popleft_n(4)
    for block in popped:
        set_hash(block)
    queue.append_n(popped)
    queue.reset()
    assert queue.num_free_blocks == 4
    assert len(queue.get_all_free_blocks()) == 4
    # No session callbacks fire after a reset.
    queue.popleft_n(4)


def test_get_all_free_blocks_eviction_order():
    blocks = make_blocks(3)
    queue, _ = make_queue(blocks)
    listed = [block.block_id for block in queue.get_all_free_blocks()]
    assert listed == [0, 1, 2]


def test_iter_eviction_candidates_order_and_resume():
    blocks = make_blocks(6)
    owners = {0: 1, 1: 1, 2: 2, 3: 2}
    priorities = {1: 5.0, 2: 500.0}
    queue, _ = make_queue(blocks, owners, priorities)
    popped = queue.popleft_n(6)
    for block in popped[:4]:
        set_hash(block)
    # Session segments freed tail-first; 4 and 5 stay unhashed.
    queue.append_n([popped[1], popped[0]])
    queue.append_n([popped[3], popped[2]])
    queue.append_n([popped[4], popped[5]])
    queue.rebuild(priorities)

    listed = [b.block_id for b in queue.iter_eviction_candidates()]
    # Unhashed first, then session 2 (furthest reuse), default, session 1.
    assert listed == [4, 5, 3, 2, 1, 0]

    # Resume mid-iteration: continue after block 3 (session 2 segment).
    resumed = [b.block_id for b in queue.iter_eviction_candidates(blocks[3])]
    assert resumed == [2, 1, 0]

    # A stale cursor (block no longer free) restarts cleanly from None.
    gone = queue.popleft()
    assert [b.block_id for b in queue.iter_eviction_candidates(gone)] == [
        b.block_id for b in queue.iter_eviction_candidates()
    ]


def test_eviction_epoch_bumps_on_rebuild_and_reset():
    blocks = make_blocks(2)
    queue, _ = make_queue(blocks)
    epoch = queue.eviction_epoch
    queue.rebuild({})
    assert queue.eviction_epoch == epoch + 1
    queue.reset()
    assert queue.eviction_epoch == epoch + 2
