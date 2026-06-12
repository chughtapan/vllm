# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration tests for lazy CPU offload over the CacheWise predictive
free-block queue: offload candidates must follow predicted-reuse order, and
the offload cursor must reset when the eviction order is rebuilt."""

from tests.v1.simple_kv_offload.test_scheduler import (
    _BYTES_PER_BLOCK,
    BLOCK_SIZE,
    _make_kv_cache_config,
    _make_vllm_config,
)
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.cachewise.eviction import PredictiveFreeBlockQueue
from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
from vllm.v1.simple_kv_offload.manager import SimpleCPUOffloadScheduler


def make_predictive_fixture(
    owners: dict[int, int],
    priorities: dict[int, float],
    num_cpu_blocks: int = 12,
    num_gpu_blocks: int = 16,
):
    kv_cache_config = _make_kv_cache_config(num_gpu_blocks, 1)
    vllm_config = _make_vllm_config()
    sched = SimpleCPUOffloadScheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        cpu_capacity_bytes=_BYTES_PER_BLOCK * num_cpu_blocks,
        scheduler_block_size=BLOCK_SIZE,
        hash_block_size=BLOCK_SIZE,
        lazy_offload=True,
    )
    queues: list[PredictiveFreeBlockQueue] = []

    def factory(blocks):
        queue = PredictiveFreeBlockQueue(
            blocks=blocks,
            classify_fn=owners.get,
            priority_fn=lambda sid: priorities.get(sid, 120.0),
            on_evicted_fn=lambda block_id, sid: None,
            default_priority=120.0,
        )
        queues.append(queue)
        return queue

    gpu_block_pool = BlockPool(
        num_gpu_blocks=num_gpu_blocks,
        enable_caching=True,
        hash_block_size=BLOCK_SIZE,
        free_block_queue_factory=factory,
    )
    sched.bind_gpu_block_pool(gpu_block_pool)
    return sched, gpu_block_pool, queues[0]


def populate_sessions(pool, queue, owners, priorities):
    """Allocate, hash, and free two sessions' blocks plus untagged ones.

    Session 1 owns blocks [b0, b1], session 2 owns [b2, b3], and [b4, b5]
    are untagged cached blocks. Blocks are freed tail-first per session.
    """
    blocks = pool.get_new_blocks(6)
    for block in blocks:
        block.block_hash = make_block_hash_with_group_id(
            bytes([block.block_id]) * 32, 0
        )
    owners.update(
        {
            blocks[0].block_id: 1,
            blocks[1].block_id: 1,
            blocks[2].block_id: 2,
            blocks[3].block_id: 2,
        }
    )
    pool.free_blocks([blocks[1], blocks[0]])
    pool.free_blocks([blocks[3], blocks[2]])
    pool.free_blocks([blocks[4], blocks[5]])
    queue.rebuild(priorities)
    return blocks


def test_lazy_offload_follows_predicted_reuse_order():
    owners: dict[int, int] = {}
    priorities = {1: 5.0, 2: 500.0}
    sched, pool, queue = make_predictive_fixture(owners, priorities)
    blocks = populate_sessions(pool, queue, owners, priorities)

    # Only enough budget to cover the two best candidates. With all
    # initial unallocated blocks unhashed, the walk starts at the
    # unhashed segment but only session blocks carry hashes... so use a
    # budget covering unhashed + 2.
    num_unhashed = pool.get_num_free_blocks() - 6
    sched._target_free = num_unhashed + 2

    gpu_ids, cpu_ids, _ = sched._prepare_lazy_store_specs()
    # Session 2 has the furthest predicted reuse: its blocks offload
    # first, suffix before prefix.
    assert gpu_ids == [blocks[3].block_id, blocks[2].block_id]
    assert len(cpu_ids) == 2


def test_lazy_offload_cursor_resets_on_rebuild():
    owners: dict[int, int] = {}
    priorities = {1: 5.0, 2: 500.0}
    sched, pool, queue = make_predictive_fixture(owners, priorities)
    blocks = populate_sessions(pool, queue, owners, priorities)

    num_unhashed = pool.get_num_free_blocks() - 6
    sched._target_free = num_unhashed + 2

    first_ids, _, _ = sched._prepare_lazy_store_specs()
    assert first_ids == [blocks[3].block_id, blocks[2].block_id]
    epoch_before = sched._cursor_epoch

    # Predictions change: session 1 is now the furthest. The rebuild bumps
    # the epoch, so the next pass restarts instead of resuming after the
    # stale cursor.
    priorities = {1: 800.0, 2: 1.0}
    queue.rebuild(priorities)

    second_ids, _, _ = sched._prepare_lazy_store_specs()
    assert sched._cursor_epoch == epoch_before + 1
    # Session 2's blocks were touch()-pinned by the first pass, so the
    # restarted walk offloads session 1's blocks, suffix first.
    assert second_ids == [blocks[1].block_id, blocks[0].block_id]


def test_lazy_offload_unchanged_epoch_resumes_cursor():
    """Without a rebuild, the second pass resumes after the cursor and
    finds nothing new (LRU-compatible behavior preserved)."""
    owners: dict[int, int] = {}
    priorities = {1: 5.0, 2: 500.0}
    sched, pool, queue = make_predictive_fixture(owners, priorities)
    blocks = populate_sessions(pool, queue, owners, priorities)

    sched._target_free = pool.get_num_free_blocks()
    first_ids, _, _ = sched._prepare_lazy_store_specs()
    assert set(first_ids) == {b.block_id for b in blocks[:6]}

    second_ids, _, _ = sched._prepare_lazy_store_specs()
    assert second_ids == []
