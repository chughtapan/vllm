# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the CacheWise tool duration predictor."""

import json

import pytest

from vllm.v1.core.cachewise import predictor as predictor_mod
from vllm.v1.core.cachewise.predictor import (
    HUMAN_PAUSE_KEY,
    DurationDistribution,
    ToolReusePredictor,
)


def test_tool_key_cardinality_is_capped(monkeypatch):
    monkeypatch.setattr(predictor_mod, "_MAX_TOOL_KEYS", 8)
    predictor = ToolReusePredictor(default_reuse_s=10.0)
    for i in range(100):
        predictor.record([(f"tool_{i}", "")], 5.0)
    assert len(predictor.tool_dists) <= 8
    # Global distribution still absorbed every sample.
    assert predictor.global_dist.num_samples == 100


def test_empty_distribution_returns_none():
    dist = DurationDistribution()
    assert dist.expected_remaining(0.0) is None


def test_expected_remaining_single_mode():
    dist = DurationDistribution()
    for _ in range(100):
        dist.record(10.0)
    estimate = dist.expected_remaining(0.0)
    # Histogram bins are log-spaced; expect the estimate near 10s.
    assert estimate == pytest.approx(10.0, rel=0.2)


def test_expected_remaining_conditions_on_elapsed():
    """A bimodal distribution must shift to the slow mode once the fast
    mode's duration has passed."""
    dist = DurationDistribution()
    for _ in range(50):
        dist.record(1.0)
    for _ in range(50):
        dist.record(100.0)
    fresh = dist.expected_remaining(0.0)
    # Mean of ~1 and ~100 is ~50.
    assert fresh == pytest.approx(50.0, rel=0.3)
    # After 10 seconds, only the 100s mode survives: ~90s remaining.
    conditioned = dist.expected_remaining(10.0)
    assert conditioned == pytest.approx(90.0, rel=0.3)


def test_expected_remaining_overdue_tail():
    dist = DurationDistribution()
    for _ in range(10):
        dist.record(5.0)
    # All observed durations have passed; bounded heuristic, not None/0.
    overdue = dist.expected_remaining(50.0)
    assert overdue is not None
    assert overdue > 0


def test_fallback_chain():
    predictor = ToolReusePredictor(default_reuse_s=42.0)
    # Cold start falls back to the default.
    assert predictor.predict_remaining([("Bash", "ls")], 0.0) == 42.0
    # With only other-tool samples, falls back to the global distribution.
    predictor.record([("Read", "file.py")], 2.0)
    estimate = predictor.predict_remaining([("Bash", "ls")], 0.0)
    assert estimate == pytest.approx(2.0, rel=0.3)
    # With same-tool samples, uses the per-tool distribution.
    predictor.record([("Bash", "pytest")], 60.0)
    estimate = predictor.predict_remaining([("Bash", "ls")], 0.0)
    assert estimate == pytest.approx(60.0, rel=0.3)


def test_parallel_tool_batches_use_composite_key():
    predictor = ToolReusePredictor(default_reuse_s=10.0)
    predictor.record([("Bash", "x")], 1.0)
    predictor.record([("Read", "y"), ("Bash", "x")], 500.0)
    # The batched sample must not pollute the single-tool distribution.
    single = predictor.predict_remaining([("Bash", "x")], 0.0)
    assert single == pytest.approx(1.0, rel=0.3)
    batch = predictor.predict_remaining([("Bash", "a"), ("Read", "b")], 0.0)
    assert batch == pytest.approx(500.0, rel=0.3)


def test_human_pause_key():
    predictor = ToolReusePredictor(default_reuse_s=10.0)
    predictor.record([], 900.0)
    assert predictor._key([]) == HUMAN_PAUSE_KEY
    estimate = predictor.predict_remaining([], 0.0)
    assert estimate == pytest.approx(900.0, rel=0.3)


def test_bootstrap_loading(tmp_path):
    path = tmp_path / "samples.jsonl"
    samples = [
        {"tool": "Bash", "args": "pytest -x", "duration_s": 30.0},
        {"tool": "Bash", "args": "ls", "duration_s": 0.1},
        {"tool": "__human__", "duration_s": 1000.0},
        {"bad": "sample"},
    ]
    path.write_text("\n".join(json.dumps(s) for s in samples))
    predictor = ToolReusePredictor(default_reuse_s=10.0, bootstrap_path=str(path))
    assert predictor.global_dist.num_samples == 3
    assert "Bash" in predictor.tool_dists
    assert HUMAN_PAUSE_KEY in predictor.tool_dists


def test_bootstrap_missing_file():
    predictor = ToolReusePredictor(
        default_reuse_s=10.0, bootstrap_path="/nonexistent/file.jsonl"
    )
    assert predictor.global_dist.num_samples == 0


def test_clustering_unavailable_falls_back(monkeypatch):
    monkeypatch.setattr("vllm.v1.core.cachewise.predictor.has_sklearn", lambda: False)
    predictor = ToolReusePredictor(default_reuse_s=10.0, use_clustering=True)
    assert not predictor.use_clustering
    predictor.record([("Bash", "x")], 5.0)
    predictor.maybe_refit()
    assert predictor.predict_remaining([("Bash", "x")], 0.0) == pytest.approx(
        5.0, rel=0.3
    )


def test_clustering_refit_and_predict():
    pytest.importorskip("sklearn")
    predictor = ToolReusePredictor(default_reuse_s=10.0, use_clustering=True)
    # Two argument populations with very different durations.
    for _ in range(60):
        predictor.record([("Bash", "pytest tests/unit slow suite")], 100.0)
        predictor.record([("Bash", "ls -la quick listing")], 0.1)
    predictor.maybe_refit()
    assert "Bash" in predictor._cluster_models
    slow = predictor.predict_remaining(
        [("Bash", "pytest tests/integration slow suite")], 0.0
    )
    fast = predictor.predict_remaining([("Bash", "ls quick listing")], 0.0)
    assert slow > fast
