# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-level integration tests for CacheWise predictive eviction and
prefix-aware scheduling."""

import json

from tests.v1.core.utils import EOS_TOKEN_ID, create_scheduler
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.cachewise.session_tracker import NEXT_TOOLS_HINT_KEY
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

BLOCK_SIZE = 16

# NONE_HASH seeds randomly on each init; initialize once so block hash
# chains match across requests within the test session.
init_none_hash(sha256)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def make_request(
    request_id: str,
    token_ids: list[int],
    hint: list[dict] | None = None,
    max_tokens: int = 4,
) -> Request:
    extra_args = {NEXT_TOOLS_HINT_KEY: json.dumps(hint)} if hint is not None else None
    sampling_params = SamplingParams(
        max_tokens=max_tokens, ignore_eos=True, extra_args=extra_args
    )
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)
    return Request(
        request_id=request_id,
        prompt_token_ids=token_ids,
        sampling_params=sampling_params,
        pooling_params=None,
        mm_features=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )


def run_to_completion(scheduler: Scheduler, request: Request) -> None:
    """Add a request, run prefill + a few decode steps, and finish it."""
    scheduler.add_request(request)
    for _ in range(3):
        output = scheduler.schedule()
        model_output = ModelRunnerOutput(
            req_ids=[req.request_id for req in scheduler.running],
            req_id_to_index={
                req.request_id: i for i, req in enumerate(scheduler.running)
            },
            sampled_token_ids=[
                [1000] if output.num_scheduled_tokens.get(req.request_id) else []
                for req in scheduler.running
            ],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )
        scheduler.update_from_output(output, model_output)
    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_STOPPED)
    # Flush finished bookkeeping.
    scheduler.schedule()


def num_cached_tokens(scheduler: Scheduler, request: Request) -> int:
    _, hits = scheduler.kv_cache_manager.coordinator.find_longest_cache_hit(
        request.block_hashes, request.num_tokens - 1
    )
    return hits


def test_predictive_eviction_end_to_end():
    """Two sessions finish with hinted tool calls; under memory pressure the
    session with the longer predicted reuse loses its blocks first."""
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        kv_cache_eviction_policy="predictive",
        num_blocks=20,
        block_size=BLOCK_SIZE,
    )
    assert scheduler.cachewise is not None
    clock = FakeClock()
    scheduler.cachewise.time_fn = clock
    scheduler.cachewise.tracker.time_fn = clock
    # Seed duration distributions so the hinted tools differ in
    # predicted reuse.
    scheduler.cachewise.predictor.record([("fast_tool", "")], 1.0)
    scheduler.cachewise.predictor.record([("slow_tool", "")], 1000.0)

    req_a = make_request(
        "session-a", [1] * 48, hint=[{"name": "fast_tool", "args": ""}]
    )
    req_b = make_request(
        "session-b", [2] * 48, hint=[{"name": "slow_tool", "args": ""}]
    )
    run_to_completion(scheduler, req_a)
    run_to_completion(scheduler, req_b)
    assert len(scheduler.cachewise.tracker.sessions) == 2

    # Both sessions' prompt blocks are cached.
    ext_a = make_request("probe-a", [1] * 48 + [1000] * 4 + [3] * 30)
    ext_b = make_request("probe-b", [2] * 48 + [1000] * 4 + [3] * 30)
    assert num_cached_tokens(scheduler, ext_a) == 48
    assert num_cached_tokens(scheduler, ext_b) == 48

    # Force an eviction-order rebuild with fresh priorities.
    for _ in range(scheduler.cachewise.rebuild_interval + 1):
        scheduler.schedule()

    # A large request forces eviction of cached blocks.
    big = make_request("big", [4] * (16 * BLOCK_SIZE))
    run_to_completion(scheduler, big)

    # Session B (slow tool, furthest predicted reuse) was evicted before
    # session A (fast tool).
    hits_a = num_cached_tokens(scheduler, ext_a)
    hits_b = num_cached_tokens(scheduler, ext_b)
    assert hits_b < hits_a

    # When session A returns, an online duration sample is recorded.
    clock.now += 5.0
    ext_a.arrival_time = clock.now
    samples_before = scheduler.cachewise.tracker.num_samples_recorded
    scheduler.add_request(ext_a)
    scheduler.schedule()
    assert scheduler.cachewise.tracker.num_samples_recorded == (samples_before + 1)


def test_predictive_eviction_default_lru_when_no_metadata():
    """Without hints or reports, eviction degrades to LRU-like behavior and
    nothing crashes."""
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        kv_cache_eviction_policy="predictive",
        num_blocks=20,
        block_size=BLOCK_SIZE,
    )
    for i in range(3):
        run_to_completion(scheduler, make_request(f"req-{i}", [i + 1] * 48))
    big = make_request("big", [9] * (16 * BLOCK_SIZE))
    run_to_completion(scheduler, big)
    assert scheduler.cachewise is not None
    assert len(scheduler.cachewise.tracker.sessions) > 0


def test_prefix_aware_scheduling_prefers_cached_request():
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        scheduling_policy="prefix_aware",
        num_blocks=200,
        block_size=BLOCK_SIZE,
        max_num_seqs=1,
    )
    # Warm the cache with a session.
    warm = make_request("warm", [7] * 64)
    run_to_completion(scheduler, warm)
    assert not scheduler.running

    # Two waiting requests: "cold" arrives first with no overlap; "hot"
    # extends the warm session's prefix.
    cold = make_request("cold", [8] * 64)
    hot = make_request("hot", [7] * 64 + [1000] * 4 + [9] * 10)
    scheduler.add_request(cold)
    scheduler.add_request(hot)

    output = scheduler.schedule()
    scheduled = [req.req_id for req in output.scheduled_new_reqs]
    assert scheduled == ["hot"]


def test_prefix_aware_starvation_override():
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        scheduling_policy="prefix_aware",
        num_blocks=200,
        block_size=BLOCK_SIZE,
        max_num_seqs=1,
    )
    warm = make_request("warm", [7] * 64)
    run_to_completion(scheduler, warm)

    cold = make_request("cold", [8] * 64)
    hot = make_request("hot", [7] * 64 + [1000] * 4 + [9] * 10)
    # The cold request has been waiting far past the starvation limit.
    cold.arrival_time -= 10 * scheduler.scheduler_config.prefix_aware_max_wait_s
    scheduler.add_request(cold)
    scheduler.add_request(hot)

    output = scheduler.schedule()
    scheduled = [req.req_id for req in output.scheduled_new_reqs]
    assert scheduled == ["cold"]


def test_fcfs_default_unaffected():
    """With default flags, no CacheWise manager is created and FCFS order
    is preserved."""
    scheduler = create_scheduler(enable_prefix_caching=True, max_num_seqs=1)
    assert scheduler.cachewise is None
    first = make_request("first", [1] * 32)
    second = make_request("second", [2] * 32)
    scheduler.add_request(first)
    scheduler.add_request(second)
    output = scheduler.schedule()
    assert [req.req_id for req in output.scheduled_new_reqs] == ["first"]
