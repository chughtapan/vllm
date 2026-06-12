# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the prefix-aware request queue."""

import time

import pytest

from vllm.v1.core.sched.request_queue import (
    FCFSRequestQueue,
    PrefixAwareRequestQueue,
    SchedulingPolicy,
    create_request_queue,
)


class FakeRequest:
    def __init__(self, request_id: str, arrival_time: float | None = None):
        self.request_id = request_id
        self.arrival_time = arrival_time if arrival_time is not None else time.time()


def make_request(request_id: str, arrival_time: float | None = None):
    return FakeRequest(request_id, arrival_time)


def make_queue(scores: dict[str, int], **kwargs) -> PrefixAwareRequestQueue:
    return PrefixAwareRequestQueue(
        scorer=lambda request: scores[request.request_id], **kwargs
    )


def test_factory_creates_prefix_aware_queue():
    queue = create_request_queue(SchedulingPolicy.PREFIX_AWARE)
    assert isinstance(queue, PrefixAwareRequestQueue)
    # Other policies are unaffected.
    assert isinstance(create_request_queue(SchedulingPolicy.FCFS), FCFSRequestQueue)
    assert not isinstance(
        create_request_queue(SchedulingPolicy.FCFS), PrefixAwareRequestQueue
    )


def test_selects_minimum_score():
    queue = make_queue({"a": 5, "b": 1, "c": 3})
    for request_id in ("a", "b", "c"):
        queue.add_request(make_request(request_id))
    assert queue.peek_request().request_id == "b"
    assert queue.pop_request().request_id == "b"
    assert queue.pop_request().request_id == "c"
    assert queue.pop_request().request_id == "a"
    with pytest.raises(IndexError):
        queue.peek_request()


def test_ties_break_by_arrival_order():
    queue = make_queue({"x": 7, "y": 7})
    queue.add_request(make_request("x"))
    queue.add_request(make_request("y"))
    assert queue.pop_request().request_id == "x"


def test_peek_and_pop_return_same_request():
    queue = make_queue({"a": 9, "b": 2})
    queue.add_request(make_request("a"))
    queue.add_request(make_request("b"))
    peeked = queue.peek_request()
    assert queue.pop_request() is peeked


def test_starvation_override():
    now = time.time()
    queue = make_queue({"old": 100, "new": 0}, max_wait_s=30.0)
    queue.add_request(make_request("old", arrival_time=now - 60))
    queue.add_request(make_request("new", arrival_time=now))
    assert queue.peek_request().request_id == "old"


def test_max_candidates_bounds_scan():
    scored: list[str] = []

    def scorer(request):
        scored.append(request.request_id)
        return {"a": 9, "b": 8, "c": 0}[request.request_id]

    queue = PrefixAwareRequestQueue(scorer=scorer, max_candidates=2)
    for request_id in ("a", "b", "c"):
        queue.add_request(make_request(request_id))
    # Only the first two candidates are scored; "c" is out of range.
    assert queue.peek_request().request_id == "b"
    assert set(scored) == {"a", "b"}


def test_mutation_invalidates_cached_selection():
    scores = {"a": 5, "b": 1}
    queue = make_queue(scores)
    queue.add_request(make_request("a"))
    assert queue.peek_request().request_id == "a"
    queue.add_request(make_request("b"))
    assert queue.peek_request().request_id == "b"


def test_new_epoch_invalidates_cached_selection():
    scores = {"a": 5, "b": 1}
    queue = make_queue(scores)
    queue.add_request(make_request("a"))
    queue.add_request(make_request("b"))
    assert queue.peek_request().request_id == "b"
    # Cache state changed: "a" now has the best score.
    scores["a"] = 0
    assert queue.peek_request().request_id == "b"  # cached
    queue.new_epoch()
    assert queue.peek_request().request_id == "a"


def test_prepended_requests_are_candidates():
    queue = make_queue({"preempted": 3, "waiting": 5})
    queue.add_request(make_request("waiting"))
    queue.prepend_request(make_request("preempted"))
    assert queue.pop_request().request_id == "preempted"


def test_no_scorer_falls_back_to_fcfs():
    queue = PrefixAwareRequestQueue(scorer=None)
    queue.add_request(make_request("first"))
    queue.add_request(make_request("second"))
    assert queue.pop_request().request_id == "first"


def test_remove_requests_keeps_consistency():
    queue = make_queue({"a": 1, "b": 2, "c": 3})
    requests = [make_request(r) for r in ("a", "b", "c")]
    for request in requests:
        queue.add_request(request)
    queue.remove_requests(requests[:1])
    assert len(queue) == 2
    assert queue.pop_request().request_id == "b"
