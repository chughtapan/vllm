# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tool-call duration predictor for CacheWise predictive eviction."""

import json
import math
import os
from collections import OrderedDict, deque
from collections.abc import Sequence
from concurrent.futures import Executor, Future
from functools import partial

from vllm.logger import init_logger
from vllm.utils.import_utils import has_sklearn

logger = init_logger(__name__)

# Key for samples of turns that ended without any tool call: the session
# returns when the human responds, on a much slower timescale than tools.
HUMAN_PAUSE_KEY = "__human__"

# Histogram resolution shared by all duration distributions.
_NUM_BINS = 64
_MIN_DURATION_S = 0.01
_MAX_DURATION_S = 3600.0
_LOG_MIN = math.log(_MIN_DURATION_S)
_LOG_MAX = math.log(_MAX_DURATION_S)
_LOG_STEP = (_LOG_MAX - _LOG_MIN) / _NUM_BINS

# Minimum number of samples per tool before argument clustering kicks in.
_MIN_SAMPLES_FOR_CLUSTERING = 100
# Number of new samples per tool between clustering refits.
_REFIT_SAMPLE_INTERVAL = 256
# Refits run inline on the engine loop; bounding the per-tool sample buffer
# keeps each refit's cost and memory constant.
_MAX_SAMPLES_PER_KEY = 2048
# Cap on distinct tool keys retained; tool names are client-influenced, so an
# unbounded key space would be a memory-exhaustion vector. Evicted keys fall
# back to the global distribution.
_MAX_TOOL_KEYS = 4096
_NUM_CLUSTERS = 8
_TFIDF_MAX_FEATURES = 512


def _bin_index(duration_s: float) -> int:
    if duration_s <= _MIN_DURATION_S:
        return 0
    idx = int((math.log(duration_s) - _LOG_MIN) / _LOG_STEP)
    return min(idx, _NUM_BINS - 1)


# Geometric mean of each bin's bounds, used as its representative value.
_BIN_MEANS = [math.exp(_LOG_MIN + (idx + 0.5) * _LOG_STEP) for idx in range(_NUM_BINS)]


class DurationDistribution:
    """Log-spaced histogram of observed durations.

    Supports conditional expectation queries of the form
    E[duration - elapsed | duration > elapsed].
    """

    def __init__(self) -> None:
        self.counts = [0] * _NUM_BINS
        self.num_samples = 0
        self.max_observed_s = 0.0

    def record(self, duration_s: float) -> None:
        self.counts[_bin_index(duration_s)] += 1
        self.num_samples += 1
        self.max_observed_s = max(self.max_observed_s, duration_s)

    def expected_remaining(self, elapsed_s: float) -> float | None:
        """Expected remaining seconds until completion given elapsed time.

        Returns None when the distribution has no samples. When the elapsed
        time exceeds (almost) all observed durations, returns a bounded
        heuristic tail instead of extrapolating from an empty histogram.
        """
        if self.num_samples == 0:
            return None
        start_bin = _bin_index(elapsed_s) if elapsed_s > _MIN_DURATION_S else 0
        surviving = 0
        total_s = 0.0
        for idx in range(start_bin, _NUM_BINS):
            count = self.counts[idx]
            if count == 0:
                continue
            bin_mean = _BIN_MEANS[idx]
            if bin_mean <= elapsed_s:
                continue
            surviving += count
            total_s += count * (bin_mean - elapsed_s)
        if surviving == 0:
            # Prediction is overdue: every observed duration has passed.
            return max(self.max_observed_s * 1.5, elapsed_s * 0.5)
        return total_s / surviving


class ToolReusePredictor:
    """Predicts how long until an agent session issues its next request.

    Maintains per-tool empirical duration distributions, optionally refined
    by clustering tool-call arguments (TF-IDF + KMeans, requires
    scikit-learn). Falls back from cluster to tool-name to global
    distributions, and finally to ``default_reuse_s``.
    """

    def __init__(
        self,
        default_reuse_s: float,
        use_clustering: bool = False,
        bootstrap_path: str | None = None,
        refit_executor: "Executor | None" = None,
    ) -> None:
        self.default_reuse_s = default_reuse_s
        self.global_dist = DurationDistribution()
        # LRU-bounded so a client minting unique tool names cannot grow
        # per-key state without limit; evicted keys fall back to global_dist.
        self.tool_dists: OrderedDict[str, DurationDistribution] = OrderedDict()

        self.use_clustering = use_clustering and has_sklearn()
        if use_clustering and not self.use_clustering:
            logger.warning_once(
                "cachewise_predictor='tfidf_kmeans' requires scikit-learn; "
                "falling back to per-tool-name distributions."
            )
        # Per-tool raw samples retained for clustering refits (bounded so
        # refit cost and memory stay constant), and the fitted per-tool
        # models: (vectorizer, kmeans, cluster distributions).
        self._samples: dict[str, deque[tuple[str, float]]] = {}
        self._samples_since_refit: dict[str, int] = {}
        self._cluster_models: dict[str, tuple] = {}
        # When set, cluster fits run here instead of on the calling (engine)
        # thread; keys with a fit in flight are tracked to avoid re-submitting.
        self._refit_executor = refit_executor
        self._inflight_refits: set[str] = set()

        if bootstrap_path is not None:
            self._load_bootstrap(bootstrap_path)

    @staticmethod
    def _key(tools: list[tuple[str, str]]) -> str:
        """Distribution key for a turn's tool calls.

        Parallel tool batches map to a composite key of sorted names so that
        single-tool distributions are not polluted by batched turns.
        """
        if not tools:
            return HUMAN_PAUSE_KEY
        names = sorted(name for name, _ in tools)
        return "+".join(names)

    def record(self, tools: list[tuple[str, str]], duration_s: float) -> None:
        """Record an observed (tool calls -> session return) duration."""
        key = self._key(tools)
        dist = self.tool_dists.get(key)
        if dist is None:
            dist = self.tool_dists[key] = DurationDistribution()
        else:
            self.tool_dists.move_to_end(key)
        dist.record(duration_s)
        self.global_dist.record(duration_s)
        if self.use_clustering and key != HUMAN_PAUSE_KEY:
            args = " ".join(arg for _, arg in tools)
            samples = self._samples.get(key)
            if samples is None:
                samples = self._samples[key] = deque(maxlen=_MAX_SAMPLES_PER_KEY)
            samples.append((args, duration_s))
            self._samples_since_refit[key] = self._samples_since_refit.get(key, 0) + 1
        self._evict_keys_over_cap()

    def _evict_keys_over_cap(self) -> None:
        """Drop least-recently-recorded tool keys past the cardinality cap."""
        while len(self.tool_dists) > _MAX_TOOL_KEYS:
            evicted, _ = self.tool_dists.popitem(last=False)
            self._samples.pop(evicted, None)
            self._samples_since_refit.pop(evicted, None)
            self._cluster_models.pop(evicted, None)

    def predict_remaining(
        self, tools: list[tuple[str, str]], elapsed_s: float
    ) -> float:
        """Expected seconds until the session's next request arrives."""
        key = self._key(tools)
        if self.use_clustering:
            estimate = self._predict_from_cluster(key, tools, elapsed_s)
            if estimate is not None:
                return estimate
        dist = self.tool_dists.get(key)
        if dist is not None:
            estimate = dist.expected_remaining(elapsed_s)
            if estimate is not None:
                return estimate
        estimate = self.global_dist.expected_remaining(elapsed_s)
        if estimate is not None:
            return estimate
        return self.default_reuse_s

    def _predict_from_cluster(
        self, key: str, tools: list[tuple[str, str]], elapsed_s: float
    ) -> float | None:
        model = self._cluster_models.get(key)
        if model is None:
            return None
        vectorizer, kmeans, cluster_dists = model
        args = " ".join(arg for _, arg in tools)
        try:
            cluster = int(kmeans.predict(vectorizer.transform([args]))[0])
        except Exception:
            return None
        return cluster_dists[cluster].expected_remaining(elapsed_s)

    def maybe_refit(self) -> None:
        """Refit argument clusters for tools that accumulated enough samples.

        Amortized: each tool refits only after ``_REFIT_SAMPLE_INTERVAL`` new
        samples. With an executor the fit runs off-thread and the model is
        swapped in atomically on completion, so the scheduler loop never
        blocks on sklearn; without one it fits synchronously (tests, simple
        deployments).
        """
        if not self.use_clustering:
            return
        for key, pending in list(self._samples_since_refit.items()):
            samples = self._samples.get(key, ())
            if len(samples) < _MIN_SAMPLES_FOR_CLUSTERING:
                continue
            if key in self._cluster_models and pending < _REFIT_SAMPLE_INTERVAL:
                continue
            # Reset the counter only when a refit is actually started; if one
            # is already in flight for this key, leave the counter so the new
            # samples aren't silently discarded and we retry next tick.
            if self._submit_refit(key):
                self._samples_since_refit[key] = 0

    def _submit_refit(self, key: str) -> bool:
        """Start a refit for ``key``. Returns whether one was started."""
        samples = self._samples.get(key)
        if not samples:
            return False
        if self._refit_executor is None:
            model = self._fit_tool(key, list(samples))
            if model is not None:
                self._cluster_models[key] = model
            return True
        if key in self._inflight_refits:
            return False
        # Zero-copy handoff: give the worker the accumulated deque and install
        # a fresh one, so the 2048-sample snapshot never happens on the engine
        # thread and the worker reads a buffer nothing else mutates.
        self._inflight_refits.add(key)
        batch = self._samples[key]
        self._samples[key] = deque(maxlen=_MAX_SAMPLES_PER_KEY)
        future = self._refit_executor.submit(self._fit_tool, key, batch)
        future.add_done_callback(partial(self._on_refit_done, key))
        return True

    def _on_refit_done(self, key: str, future: Future) -> None:
        self._inflight_refits.discard(key)
        try:
            model = future.result()
        except Exception:
            logger.warning_once("CacheWise cluster refit failed for %r", key)
            return
        # dict assignment is atomic under CPython; predict only reads the tuple.
        # Skip keys evicted from the cardinality cap while the fit ran, so a
        # stale callback can't reinsert state past _MAX_TOOL_KEYS.
        if model is not None and key in self.tool_dists:
            self._cluster_models[key] = model

    @staticmethod
    def _fit_tool(key: str, samples: "Sequence[tuple[str, float]]") -> "tuple | None":
        """Fit TF-IDF + KMeans for one tool key. Pure: returns the model
        tuple (vectorizer, kmeans, cluster distributions) or None."""
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.feature_extraction.text import TfidfVectorizer

        args_list = [args for args, _ in samples]
        n_clusters = min(_NUM_CLUSTERS, len(samples))
        try:
            vectorizer = TfidfVectorizer(max_features=_TFIDF_MAX_FEATURES)
            matrix = vectorizer.fit_transform(args_list)
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters, n_init="auto", random_state=0
            )
            labels = kmeans.fit_predict(matrix)
        except ValueError:
            # E.g. empty vocabulary when all arguments are stop words.
            return None
        cluster_dists = [DurationDistribution() for _ in range(n_clusters)]
        for (_, duration_s), label in zip(samples, labels):
            cluster_dists[label].record(duration_s)
        return (vectorizer, kmeans, cluster_dists)

    def _load_bootstrap(self, path: str) -> None:
        if not os.path.exists(path):
            logger.warning(
                "CacheWise bootstrap file %s does not exist; starting cold.",
                path,
            )
            return
        num_loaded = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    tool = sample["tool"]
                    duration_s = float(sample["duration_s"])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    logger.warning_once(
                        "Skipping malformed lines in CacheWise bootstrap file %s",
                        path,
                    )
                    continue
                args = str(sample.get("args", ""))
                tools = [(tool, args)] if tool != HUMAN_PAUSE_KEY else []
                self.record(tools, duration_s)
                num_loaded += 1
        self.maybe_refit()
        logger.info(
            "CacheWise predictor bootstrapped with %d samples from %s",
            num_loaded,
            path,
        )

    def stats(self) -> dict[str, int]:
        return {
            "num_samples": self.global_dist.num_samples,
            "num_tool_keys": len(self.tool_dists),
            "num_clustered_tools": len(self._cluster_models),
        }
