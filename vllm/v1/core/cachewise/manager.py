# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CacheWise manager: facade tying together session tracking, the tool
duration predictor, and the predictive eviction queue."""

import time
from typing import TYPE_CHECKING

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.core.cachewise.eviction import PredictiveFreeBlockQueue
from vllm.v1.core.cachewise.predictor import ToolReusePredictor
from vllm.v1.core.cachewise.session_tracker import SessionTracker, ToolCalls
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.request import Request

logger = init_logger(__name__)

_STATS_LOG_INTERVAL_S = 300.0


def supports_predictive_eviction(kv_cache_config: KVCacheConfig) -> bool:
    """Predictive eviction currently supports full-attention-only models.

    Hybrid and sliding-window layouts reclaim scratch blocks with different
    per-group semantics that the predictive queue does not yet model.
    """
    return len(kv_cache_config.kv_cache_groups) == 1 and isinstance(
        kv_cache_config.kv_cache_groups[0].kv_cache_spec, FullAttentionSpec
    )


class CacheWiseManager:
    """Owns CacheWise state for one scheduler.

    Created by the scheduler when ``kv_cache_eviction_policy`` is
    "predictive"; its ``create_free_queue`` is injected into the BlockPool
    so freed blocks are ordered by predicted time-to-next-reuse.
    """

    def __init__(self, vllm_config: VllmConfig) -> None:
        cache_config = vllm_config.cache_config
        self.rebuild_interval = cache_config.cachewise_rebuild_interval
        self.default_reuse_s = cache_config.cachewise_default_reuse_s
        # Attribute (not a constructor parameter) so tests can patch it.
        self.time_fn = time.time

        self.predictor = ToolReusePredictor(
            default_reuse_s=self.default_reuse_s,
            use_clustering=cache_config.cachewise_predictor == "tfidf_kmeans",
            bootstrap_path=cache_config.cachewise_bootstrap_path,
        )
        self.tracker = SessionTracker(
            predictor=self.predictor,
            default_reuse_s=self.default_reuse_s,
            session_ttl_s=cache_config.cachewise_session_ttl,
        )
        self.queue: PredictiveFreeBlockQueue | None = None
        self._last_rebuild_step = 0
        self._last_stats_log_ts = self.time_fn()

    def create_free_queue(
        self, blocks: list["KVCacheBlock"]
    ) -> PredictiveFreeBlockQueue:
        """Factory for the BlockPool's free block queue."""
        self.queue = PredictiveFreeBlockQueue(
            blocks=blocks,
            classify_fn=self.tracker.classify_block,
            priority_fn=self.tracker.session_priority,
            on_evicted_fn=self.tracker.on_block_evicted,
            default_priority=self.default_reuse_s,
        )
        return self.queue

    def step(self, current_step: int) -> None:
        """Periodic maintenance, called once per scheduler iteration."""
        if current_step - self._last_rebuild_step < self.rebuild_interval:
            return
        self._last_rebuild_step = current_step
        now = self.time_fn()
        self.tracker.prune(now)
        self.predictor.maybe_refit()
        if self.queue is not None:
            self.queue.rebuild(self.tracker.priorities(now))
        if now - self._last_stats_log_ts >= _STATS_LOG_INTERVAL_S:
            self._last_stats_log_ts = now
            logger.info("CacheWise stats: %s", self.stats())

    def on_request_scheduled(self, request: "Request") -> None:
        self.tracker.on_request_scheduled(request)

    def on_request_finished(self, request: "Request", block_ids: list[int]) -> None:
        self.tracker.on_request_finished(request, block_ids)

    def on_request_preempted(self, request: "Request", block_ids: list[int]) -> None:
        self.tracker.on_request_preempted(request, block_ids)

    def report_tool_calls(self, request_id: str, tool_calls: ToolCalls) -> None:
        self.tracker.on_tool_report(request_id, tool_calls)

    def on_reset_prefix_cache(self) -> None:
        self.tracker.clear()
        if self.queue is not None:
            self.queue.reset()

    def stats(self) -> dict[str, int]:
        stats = {**self.tracker.stats(), **self.predictor.stats()}
        if self.queue is not None:
            stats["num_predictive_evictions"] = self.queue.num_predictive_evictions
        return stats
