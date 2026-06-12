# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import heapq
import time
from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable, Iterator
from enum import Enum

from vllm.v1.request import Request


class SchedulingPolicy(Enum):
    """Enum for scheduling policies."""

    FCFS = "fcfs"
    PRIORITY = "priority"
    PREFIX_AWARE = "prefix_aware"


class RequestQueue(ABC):
    """Abstract base class for request queues."""

    @abstractmethod
    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to the policy."""
        pass

    @abstractmethod
    def pop_request(self) -> Request:
        """Pop a request from the queue according to the policy."""
        pass

    @abstractmethod
    def peek_request(self) -> Request:
        """Peek at the request at the front of the queue without removing it."""
        pass

    @abstractmethod
    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        pass

    @abstractmethod
    def prepend_requests(self, requests: "RequestQueue") -> None:
        """Prepend all requests from another queue to the front of this
        queue."""
        pass

    @abstractmethod
    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        pass

    @abstractmethod
    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get number of requests in queue."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to the policy."""
        pass

    def new_epoch(self) -> None:
        """Notify the queue that external state affecting its ordering may
        have changed (e.g. a new scheduling step started). A no-op for
        policies whose ordering depends only on queue contents."""
        return None


class FCFSRequestQueue(deque[Request], RequestQueue):
    """A first-come-first-served queue that supports deque operations."""

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to FCFS policy."""
        self.append(request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to FCFS policy."""
        return self.popleft()

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        self.appendleft(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Prepend all requests from another queue to the front of this
        queue.

        Note: The requests will be prepended in reverse order of their
        appearance in the `requests` queue.
        """
        self.extendleft(requests)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        filtered_requests = [req for req in self if req not in requests_to_remove]
        # deque does not support in-place filtering, so we need to clear
        # and extend
        self.clear()
        self.extend(filtered_requests)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return len(self) > 0

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return super().__len__()

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to FCFS policy."""
        return super().__iter__()


class PriorityRequestQueue(RequestQueue):
    """
    A priority queue that supports heap operations.

    Respects the ordering defined in the Request class, where
    requests with a smaller value of `priority` are processed first.
    If multiple requests have the same priority, the one with the earlier
    `arrival_time` is processed first.
    """

    def __init__(self) -> None:
        self._heap: list[Request] = []

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy."""
        heapq.heappush(self._heap, request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to priority policy."""
        if not self._heap:
            raise IndexError("pop from empty heap")
        return heapq.heappop(self._heap)

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self._heap:
            raise IndexError("peek from empty heap")
        return self._heap[0]

    def prepend_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time)."""
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Add all requests from another queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time)."""
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self._heap.remove(request)
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = requests if isinstance(requests, set) else set(requests)
        self._heap = [r for r in self._heap if r not in requests_to_remove]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return bool(self._heap)

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return len(self._heap)

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to priority policy."""
        heap_copy = self._heap[:]
        while heap_copy:
            yield heapq.heappop(heap_copy)


class PrefixAwareRequestQueue(RequestQueue):
    """A queue that serves the request needing the fewest new KV blocks.

    Requests are stored in arrival order, but peek/pop select the request
    with the highest prefix cache overlap (fewest additional KV blocks to
    allocate), breaking ties by arrival order. To bound both starvation and
    scheduling cost, a request older than ``max_wait_s`` is served FCFS
    regardless of its score, and at most ``max_candidates`` of the oldest
    requests are scored per selection.

    Backed by an ``OrderedDict`` keyed by request id so that the
    select-then-remove-from-middle access pattern is O(1) per operation
    (a deque would make pop_request O(n)). The score of a waiting request
    changes as blocks are cached and evicted, so selection is re-evaluated
    lazily: the cached choice is invalidated by any queue mutation, and
    per-request scores are memoized until ``new_epoch()`` (i.e. for one
    scheduling step, bounding staleness).
    """

    def __init__(
        self,
        scorer: Callable[[Request], int] | None = None,
        max_wait_s: float = 0.0,
        max_candidates: int = 64,
    ) -> None:
        self._requests: OrderedDict[str, Request] = OrderedDict()
        self._scorer = scorer
        self._max_wait_s = max_wait_s
        self._max_candidates = max_candidates
        self._selected: Request | None = None
        self._scores: dict[str, int] = {}

    def _score(self, request: Request) -> int:
        assert self._scorer is not None
        score = self._scores.get(request.request_id)
        if score is None:
            score = self._scores[request.request_id] = self._scorer(request)
        return score

    def _select_request(self) -> Request:
        if not self._requests:
            raise IndexError("peek from an empty queue")
        if self._selected is not None:
            return self._selected
        head = next(iter(self._requests.values()))
        if self._scorer is None or (
            self._max_wait_s > 0 and time.time() - head.arrival_time > self._max_wait_s
        ):
            self._selected = head
            return head
        best = head
        best_score = self._score(head)
        for index, request in enumerate(self._requests.values()):
            if index == 0:
                continue
            if index >= self._max_candidates:
                break
            score = self._score(request)
            if score < best_score:
                best, best_score = request, score
        self._selected = best
        return best

    def peek_request(self) -> Request:
        """Peek at the best-overlap request without removing it."""
        return self._select_request()

    def pop_request(self) -> Request:
        """Pop the best-overlap request (the one last peeked)."""
        request = self._select_request()
        del self._requests[request.request_id]
        self._scores.pop(request.request_id, None)
        self._selected = None
        return request

    def add_request(self, request: Request) -> None:
        self._selected = None
        self._requests[request.request_id] = request

    def prepend_request(self, request: Request) -> None:
        self._selected = None
        self._requests[request.request_id] = request
        self._requests.move_to_end(request.request_id, last=False)

    def prepend_requests(self, requests: RequestQueue) -> None:
        # Iterate forward, moving each to the front, so the donor's order is
        # reversed at the head -- matching FCFSRequestQueue.extendleft, which
        # the scheduler relies on when requeuing skipped requests.
        self._selected = None
        for request in requests:
            self._requests[request.request_id] = request
            self._requests.move_to_end(request.request_id, last=False)

    def remove_request(self, request: Request) -> None:
        self._selected = None
        del self._requests[request.request_id]
        self._scores.pop(request.request_id, None)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        self._selected = None
        for request in requests:
            self._requests.pop(request.request_id, None)
            self._scores.pop(request.request_id, None)

    def __bool__(self) -> bool:
        return bool(self._requests)

    def __len__(self) -> int:
        return len(self._requests)

    def __iter__(self) -> Iterator[Request]:
        return iter(self._requests.values())

    def new_epoch(self) -> None:
        self._selected = None
        self._scores.clear()


def create_request_queue(
    policy: SchedulingPolicy,
    prefix_aware_scorer: Callable[[Request], int] | None = None,
    prefix_aware_max_wait_s: float = 0.0,
    prefix_aware_max_candidates: int = 64,
) -> RequestQueue:
    """Create request queue based on scheduling policy."""
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    elif policy == SchedulingPolicy.PREFIX_AWARE:
        return PrefixAwareRequestQueue(
            scorer=prefix_aware_scorer,
            max_wait_s=prefix_aware_max_wait_s,
            max_candidates=prefix_aware_max_candidates,
        )
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")
