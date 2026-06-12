# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the CacheWise session tracker."""

import json
from types import SimpleNamespace

import pytest

from tests.v1.core.cachewise.utils import FakeClock
from vllm.v1.core.cachewise.predictor import ToolReusePredictor
from vllm.v1.core.cachewise.session_tracker import (
    NEXT_TOOLS_HINT_KEY,
    PREEMPTED_SESSION_ID,
    SessionTracker,
)


def make_tracker(default_reuse_s: float = 120.0, ttl: float = 1800.0):
    clock = FakeClock()
    predictor = ToolReusePredictor(default_reuse_s=default_reuse_s)
    tracker = SessionTracker(
        predictor=predictor,
        default_reuse_s=default_reuse_s,
        session_ttl_s=ttl,
        time_fn=clock,
    )
    return tracker, predictor, clock


def make_request(
    request_id: str,
    block_hashes: list[bytes],
    arrival_time: float = 0.0,
    hint: list[dict] | None = None,
):
    extra_args = {NEXT_TOOLS_HINT_KEY: json.dumps(hint)} if hint is not None else {}
    return SimpleNamespace(
        request_id=request_id,
        block_hashes=block_hashes,
        arrival_time=arrival_time,
        sampling_params=SimpleNamespace(extra_args=extra_args),
    )


def hashes(*tags: str) -> list[bytes]:
    return [tag.encode() for tag in tags]


def test_finish_then_return_emits_sample():
    tracker, predictor, clock = make_tracker()
    req1 = make_request("r1", hashes("a", "b"))
    tracker.on_request_finished(req1, block_ids=[10, 11])
    tracker.on_tool_report("r1", [("Bash", "pytest")])

    # The session returns 30s later with an extended prefix.
    clock.now += 30.0
    req2 = make_request("r2", hashes("a", "b", "c"), arrival_time=clock.now)
    tracker.on_request_scheduled(req2)
    assert tracker.num_samples_recorded == 1
    assert predictor.predict_remaining([("Bash", "x")], 0.0) == pytest.approx(
        30.0, rel=0.3
    )

    # Finishing the return request updates the same session's tail.
    tracker.on_request_finished(req2, block_ids=[10, 11, 12])
    assert len(tracker.sessions) == 1
    session = next(iter(tracker.sessions.values()))
    assert session.tail_block_hash == b"c"
    assert session.block_ids == {10, 11, 12}


def test_deepest_tail_match_disambiguates_shared_prefix():
    """Two sessions sharing a system prompt block: the deeper chain wins."""
    tracker, _, clock = make_tracker()
    tracker.on_request_finished(
        make_request("r1", hashes("sys", "a1")), block_ids=[1, 2]
    )
    tracker.on_request_finished(
        make_request("r2", hashes("sys", "a1", "a2")), block_ids=[1, 2, 3]
    )
    sid_short = tracker.by_tail_hash[b"a1"]
    sid_long = tracker.by_tail_hash[b"a2"]
    assert sid_short != sid_long

    ret = make_request("r3", hashes("sys", "a1", "a2", "a3"), arrival_time=clock.now)
    tracker.on_request_scheduled(ret)
    assert tracker.by_request_id["r3"] == sid_long


def test_prefix_break_creates_new_session():
    """Context compaction breaks the hash chain: no match, new session."""
    tracker, _, clock = make_tracker()
    tracker.on_request_finished(make_request("r1", hashes("a", "b")), block_ids=[1])
    compacted = make_request("r2", hashes("x", "y"), arrival_time=clock.now)
    tracker.on_request_scheduled(compacted)
    assert "r2" not in tracker.by_request_id
    assert tracker.num_samples_recorded == 0
    tracker.on_request_finished(compacted, block_ids=[2])
    assert len(tracker.sessions) == 2


def test_hint_takes_precedence_over_report():
    tracker, predictor, clock = make_tracker()
    req = make_request(
        "r1",
        hashes("a"),
        hint=[{"name": "Bash", "args": "sleep 60"}],
    )
    tracker.on_request_finished(req, block_ids=[1])
    session = next(iter(tracker.sessions.values()))
    assert session.pending_tools == [("Bash", "sleep 60")]
    # Hinted finishes never await a parsed report.
    assert "r1" not in tracker.recently_finished
    # A later parsed report must not override the hint.
    tracker.on_tool_report("r1", [("Read", "x")])
    assert session.pending_tools == [("Bash", "sleep 60")]


def test_tool_report_for_unknown_request_is_noop():
    tracker, _, clock = make_tracker()
    tracker.on_tool_report("unknown", [("Bash", "x")])
    assert not tracker.sessions


def test_priorities_reflect_predictions_and_flight_state():
    tracker, predictor, clock = make_tracker(default_reuse_s=77.0)
    predictor.record([("Bash", "x")], 50.0)

    tracker.on_request_finished(make_request("r1", hashes("a")), block_ids=[1])
    tracker.on_tool_report("r1", [("Bash", "y")])
    tracker.on_request_finished(
        make_request("r2", hashes("b")), block_ids=[2]
    )  # awaiting report

    sid1 = tracker.by_tail_hash[b"a"]
    sid2 = tracker.by_tail_hash[b"b"]
    priorities = tracker.priorities(clock.now)
    assert priorities[sid1] == pytest.approx(50.0, rel=0.3)
    assert priorities[sid2] == 77.0

    # In-flight sessions have the most imminent reuse.
    ret = make_request("r3", hashes("a", "c"), arrival_time=clock.now)
    tracker.on_request_scheduled(ret)
    priorities = tracker.priorities(clock.now)
    assert priorities[sid1] == 0.0


def test_classify_block_picks_min_priority_owner():
    tracker, predictor, clock = make_tracker()
    predictor.record([("Fast", "x")], 1.0)
    predictor.record([("Slow", "x")], 1000.0)
    # Both sessions own the shared block 1.
    tracker.on_request_finished(make_request("r1", hashes("s", "a")), block_ids=[1, 2])
    tracker.on_tool_report("r1", [("Fast", "x")])
    tracker.on_request_finished(make_request("r2", hashes("s", "b")), block_ids=[1, 3])
    tracker.on_tool_report("r2", [("Slow", "x")])

    sid_fast = tracker.by_tail_hash[b"a"]
    tracker.priorities(clock.now)
    # The shared block belongs to the soonest-reuse session.
    assert tracker.classify_block(1) == sid_fast
    assert tracker.classify_block(99) is None


def test_preempted_blocks_pseudo_session():
    tracker, _, clock = make_tracker()
    req = make_request("r1", hashes("a"))
    tracker.on_request_preempted(req, block_ids=[5, 6])
    assert tracker.classify_block(5) == PREEMPTED_SESSION_ID
    assert tracker.session_priority(PREEMPTED_SESSION_ID) == 0.0
    assert tracker.priorities(clock.now)[PREEMPTED_SESSION_ID] == 0.0
    # Rescheduling clears the tag.
    tracker.on_request_scheduled(req)
    assert tracker.classify_block(5) is None


def test_preempted_then_aborted_clears_tag():
    # A preempted request that finishes (aborts) instead of rescheduling must
    # not leave its blocks pinned at PREEMPTED priority forever.
    tracker, _, clock = make_tracker()
    req = make_request("r1", hashes("a"))
    tracker.on_request_preempted(req, block_ids=[5, 6])
    assert tracker.classify_block(5) == PREEMPTED_SESSION_ID
    tracker.on_request_finished(req, block_ids=[7])
    assert tracker.classify_block(5) is None
    assert PREEMPTED_SESSION_ID not in tracker.priorities(clock.now)


def test_session_count_is_capped(monkeypatch):
    monkeypatch.setattr("vllm.v1.core.cachewise.session_tracker._MAX_SESSIONS", 4)
    tracker, _, clock = make_tracker()
    for i in range(20):
        tracker.on_request_finished(
            make_request(f"r{i}", hashes(f"h{i}")), block_ids=[i]
        )
        clock.now += 1.0
    assert len(tracker.sessions) <= 4
    # The oldest sessions were dropped; their tail hashes no longer match.
    assert tracker.by_tail_hash.get(b"h0") is None
    # Maps stay consistent with the surviving sessions.
    assert set(tracker.by_tail_hash.values()) == set(tracker.sessions)


def test_in_flight_sessions_survive_cap(monkeypatch):
    monkeypatch.setattr("vllm.v1.core.cachewise.session_tracker._MAX_SESSIONS", 2)
    tracker, _, clock = make_tracker()
    # An active session (returned, in flight) must not be evicted by the cap.
    tracker.on_request_finished(make_request("r0", hashes("keep")), block_ids=[0])
    ret = make_request("r0b", hashes("keep", "more"), arrival_time=clock.now)
    tracker.on_request_scheduled(ret)
    kept_sid = tracker.by_request_id["r0b"]
    for i in range(10):
        tracker.on_request_finished(
            make_request(f"r{i}", hashes(f"h{i}")), block_ids=[100 + i]
        )
    assert kept_sid in tracker.sessions


def test_hint_size_is_capped():
    tracker, _, _ = make_tracker()
    big_hint = [{"name": "x" * 9999, "args": "y" * 99999} for _ in range(1000)]
    req = make_request("r1", hashes("a"), hint=big_hint)
    tracker.on_request_finished(req, block_ids=[1])
    session = next(iter(tracker.sessions.values()))
    assert session.pending_tools is not None
    assert len(session.pending_tools) <= 32
    name, args = session.pending_tools[0]
    assert len(name) <= 128
    assert len(args) <= 4096


def test_ttl_prunes_idle_sessions():
    tracker, _, clock = make_tracker(ttl=100.0)
    tracker.on_request_finished(make_request("r1", hashes("a")), block_ids=[1])
    clock.now += 50.0
    tracker.prune(clock.now)
    assert len(tracker.sessions) == 1
    clock.now += 100.0
    tracker.prune(clock.now)
    assert not tracker.sessions
    assert not tracker.block_owners
    assert tracker.classify_block(1) is None


def test_eviction_callback_removes_ownership():
    tracker, _, clock = make_tracker()
    tracker.on_request_finished(make_request("r1", hashes("a")), block_ids=[1, 2])
    sid = tracker.by_tail_hash[b"a"]
    tracker.on_block_evicted(1, sid)
    assert tracker.classify_block(1) is None
    assert tracker.sessions[sid].block_ids == {2}


def test_clear_resets_everything():
    tracker, _, clock = make_tracker()
    tracker.on_request_finished(make_request("r1", hashes("a")), block_ids=[1])
    tracker.clear()
    assert not tracker.sessions
    assert not tracker.by_tail_hash
    assert not tracker.block_owners
    assert not tracker.recently_finished


def test_malformed_hint_ignored():
    tracker, _, clock = make_tracker()
    req = make_request("r1", hashes("a"))
    req.sampling_params.extra_args = {NEXT_TOOLS_HINT_KEY: "not json"}
    tracker.on_request_finished(req, block_ids=[1])
    session = next(iter(tracker.sessions.values()))
    assert session.pending_tools is None
    # A malformed hint degrades to awaiting a parsed report.
    assert "r1" in tracker.recently_finished
