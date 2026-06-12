# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CacheWise: predictive KV cache management for agentic workloads.

Implements session tracking, tool-call duration prediction, and predictive
KV cache block eviction for long-running, closed-loop agent sessions.
"""

from vllm.v1.core.cachewise.manager import CacheWiseManager

__all__ = ["CacheWiseManager"]
