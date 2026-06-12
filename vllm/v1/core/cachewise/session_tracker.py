# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Session tracking for CacheWise predictive KV cache eviction.

A session is an agent conversation whose requests repeatedly extend a
growing token prefix. Sessions are identified purely from request block
hashes: the tail block hash recorded when a request finishes is matched
against the block hashes of later requests, requiring no client
cooperation.
"""

import json
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.core.cachewise.predictor import ToolReusePredictor

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)

# extra_args key through which clients can pass ground-truth metadata about
# the tool calls they are about to execute (JSON list of {"name", "args"}).
NEXT_TOOLS_HINT_KEY = "cachewise_next_tools"

# Pseudo-session for blocks freed by preemption: the preempted request is
# requeued immediately, so its blocks have the most imminent reuse.
PREEMPTED_SESSION_ID = -1

# How long a finished request id stays addressable for late tool reports
# from the API layer.
_TOOL_REPORT_WINDOW_S = 60.0

ToolCalls = list[tuple[str, str]]


@dataclass
class Session:
    session_id: int
    tail_block_hash: bytes
    block_ids: set[int] = field(default_factory=set)
    last_finish_ts: float = 0.0
    # None: awaiting a tool report from the API layer.
    # []: the turn ended with no tool call (human pause).
    pending_tools: ToolCalls | None = None
    # True when pending_tools came from a client hint and must not be
    # overwritten by a parsed tool report.
    tools_from_hint: bool = False
    last_request_id: str = ""
    in_flight: bool = False


class SessionTracker:
    """Tracks agent sessions and the reuse priority of their KV blocks."""

    def __init__(
        self,
        predictor: ToolReusePredictor,
        default_reuse_s: float,
        session_ttl_s: float,
        time_fn: Callable[[], float] = time.time,
    ) -> None:
        self.predictor = predictor
        self.default_reuse_s = default_reuse_s
        self.session_ttl_s = session_ttl_s
        self.time_fn = time_fn

        self.sessions: dict[int, Session] = {}
        self.by_tail_hash: dict[bytes, int] = {}
        # In-flight request id -> session id.
        self.by_request_id: dict[str, int] = {}
        # Recently finished request id -> (session id, expiry timestamp),
        # kept so tool reports arriving after blocks were freed still land.
        self.recently_finished: OrderedDict[str, tuple[int, float]] = OrderedDict()
        # Block id -> session ids whose recorded prefix includes the block.
        self.block_owners: dict[int, set[int]] = {}
        # Blocks freed by preemption, keyed by the preempted request id.
        self.preempted_blocks: dict[str, set[int]] = {}
        self._preempted_block_to_req: dict[int, str] = {}

        self._next_session_id = 0
        # Session priorities as of the last rebuild, used to classify blocks
        # freed between rebuilds.
        self._last_priorities: dict[int, float] = {}
        self.num_samples_recorded = 0
        self.num_sessions_created = 0

    def on_request_scheduled(self, request: "Request") -> None:
        """Called when a request is admitted for its first prefill.

        Detects returning sessions by matching the request's block hashes
        against recorded session tail hashes (deepest match wins), and emits
        an online duration sample for the tool calls the session executed.
        """
        request_id = request.request_id
        self._clear_preempted(request_id)
        if request_id in self.by_request_id:
            return
        session = self._find_session(request.block_hashes)
        if session is None:
            return
        if not session.in_flight and session.pending_tools is not None:
            duration_s = request.arrival_time - session.last_finish_ts
            if duration_s > 0:
                self.predictor.record(session.pending_tools, duration_s)
                self.num_samples_recorded += 1
        session.in_flight = True
        self.by_request_id[request_id] = session.session_id

    def on_request_finished(self, request: "Request", block_ids: list[int]) -> None:
        """Called when a request finishes, before its blocks are freed.

        Updates (or creates) the session, records which blocks belong to it,
        and attaches tool metadata from a client hint when present; otherwise
        the session awaits a tool report from the API layer.
        """
        if not request.block_hashes:
            return
        request_id = request.request_id
        tail_hash = bytes(request.block_hashes[-1])
        session_id = self.by_request_id.pop(request_id, None)
        if session_id is not None and session_id in self.sessions:
            session = self.sessions[session_id]
            self.by_tail_hash.pop(session.tail_block_hash, None)
        else:
            session = Session(
                session_id=self._next_session_id, tail_block_hash=tail_hash
            )
            self._next_session_id += 1
            self.num_sessions_created += 1
            self.sessions[session.session_id] = session

        # Re-point the tail hash. On a collision (two requests sharing the
        # exact same full-block prefix), the most recent finisher wins.
        session.tail_block_hash = tail_hash
        self.by_tail_hash[tail_hash] = session.session_id
        session.last_finish_ts = self.time_fn()
        session.last_request_id = request_id
        session.in_flight = False

        new_block_ids = set(block_ids)
        for block_id in session.block_ids - new_block_ids:
            self._remove_owner(block_id, session.session_id)
        for block_id in new_block_ids - session.block_ids:
            self.block_owners.setdefault(block_id, set()).add(session.session_id)
        session.block_ids = new_block_ids

        hint_tools = self._parse_hint(request)
        if hint_tools is not None:
            session.pending_tools = hint_tools
            session.tools_from_hint = True
        else:
            session.pending_tools = None
            session.tools_from_hint = False
            self.recently_finished[request_id] = (
                session.session_id,
                session.last_finish_ts + _TOOL_REPORT_WINDOW_S,
            )

    def on_tool_report(
        self, request_id: str, tools: ToolCalls, finish_ts: float
    ) -> None:
        """Attach tool calls parsed by the API layer to a finished request."""
        entry = self.recently_finished.pop(request_id, None)
        if entry is None:
            return
        session = self.sessions.get(entry[0])
        if session is None or session.tools_from_hint:
            return
        if session.last_request_id == request_id:
            session.pending_tools = tools

    def on_request_preempted(self, request: "Request", block_ids: list[int]) -> None:
        """Tag a preempted request's freed blocks as imminently reused."""
        request_id = request.request_id
        block_id_set = set(block_ids)
        self.preempted_blocks[request_id] = block_id_set
        for block_id in block_id_set:
            self._preempted_block_to_req[block_id] = request_id

    def classify_block(self, block_id: int) -> int | None:
        """Session id whose segment a freed block belongs to, or None.

        Blocks owned by several sessions go to the owner with the soonest
        predicted reuse, so shared prefixes are protected by their most
        imminent session.
        """
        if block_id in self._preempted_block_to_req:
            return PREEMPTED_SESSION_ID
        owners = self.block_owners.get(block_id)
        if not owners:
            return None
        return min(
            owners,
            key=lambda sid: self._last_priorities.get(sid, self.default_reuse_s),
        )

    def session_priority(self, session_id: int) -> float:
        if session_id == PREEMPTED_SESSION_ID:
            return 0.0
        return self._last_priorities.get(session_id, self.default_reuse_s)

    def on_block_evicted(self, block_id: int, session_id: int) -> None:
        """Called by the eviction queue when a tagged block is evicted."""
        if session_id == PREEMPTED_SESSION_ID:
            request_id = self._preempted_block_to_req.pop(block_id, None)
            if request_id is not None:
                blocks = self.preempted_blocks.get(request_id)
                if blocks is not None:
                    blocks.discard(block_id)
            return
        self._remove_owner(block_id, session_id)
        session = self.sessions.get(session_id)
        if session is not None:
            session.block_ids.discard(block_id)

    def priorities(self, now: float) -> dict[int, float]:
        """Predicted seconds-to-next-reuse per session, for queue rebuilds."""
        priorities: dict[int, float] = {}
        for session_id, session in self.sessions.items():
            if session.in_flight:
                priorities[session_id] = 0.0
            elif session.pending_tools is None:
                priorities[session_id] = self.default_reuse_s
            else:
                elapsed_s = max(0.0, now - session.last_finish_ts)
                priorities[session_id] = self.predictor.predict_remaining(
                    session.pending_tools, elapsed_s
                )
        if self._preempted_block_to_req:
            priorities[PREEMPTED_SESSION_ID] = 0.0
        self._last_priorities = priorities
        return priorities

    def prune(self, now: float) -> None:
        """Drop expired sessions and stale tool-report entries."""
        expired = [
            session
            for session in self.sessions.values()
            if not session.in_flight
            and now - session.last_finish_ts > self.session_ttl_s
        ]
        for session in expired:
            self._drop_session(session)
        while self.recently_finished:
            request_id, (_, deadline) = next(iter(self.recently_finished.items()))
            if deadline > now:
                break
            del self.recently_finished[request_id]

    def clear(self) -> None:
        """Reset all tracking state (e.g. when the prefix cache is reset)."""
        self.sessions.clear()
        self.by_tail_hash.clear()
        self.by_request_id.clear()
        self.recently_finished.clear()
        self.block_owners.clear()
        self.preempted_blocks.clear()
        self._preempted_block_to_req.clear()
        self._last_priorities.clear()

    def stats(self) -> dict[str, int]:
        return {
            "num_sessions": len(self.sessions),
            "num_sessions_created": self.num_sessions_created,
            "num_samples_recorded": self.num_samples_recorded,
            "num_tracked_blocks": len(self.block_owners),
        }

    def _find_session(self, block_hashes: Sequence[bytes]) -> Session | None:
        for block_hash in reversed(block_hashes):
            session_id = self.by_tail_hash.get(bytes(block_hash))
            if session_id is not None:
                return self.sessions[session_id]
        return None

    def _parse_hint(self, request: "Request") -> ToolCalls | None:
        params = request.sampling_params
        if params is None or not params.extra_args:
            return None
        raw = params.extra_args.get(NEXT_TOOLS_HINT_KEY)
        if raw is None:
            return None
        try:
            entries = json.loads(raw) if isinstance(raw, str) else raw
            return [
                (str(entry["name"]), str(entry.get("args", ""))) for entry in entries
            ]
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning_once(
                'Malformed %s hint; expected a JSON list of {"name", "args"} objects.',
                NEXT_TOOLS_HINT_KEY,
            )
            return None

    def _remove_owner(self, block_id: int, session_id: int) -> None:
        owners = self.block_owners.get(block_id)
        if owners is None:
            return
        owners.discard(session_id)
        if not owners:
            del self.block_owners[block_id]

    def _clear_preempted(self, request_id: str) -> None:
        blocks = self.preempted_blocks.pop(request_id, None)
        if blocks is None:
            return
        for block_id in blocks:
            self._preempted_block_to_req.pop(block_id, None)

    def _drop_session(self, session: Session) -> None:
        for block_id in session.block_ids:
            self._remove_owner(block_id, session.session_id)
        self.by_tail_hash.pop(session.tail_block_hash, None)
        self.by_request_id.pop(session.last_request_id, None)
        self.sessions.pop(session.session_id, None)
        self._last_priorities.pop(session.session_id, None)
