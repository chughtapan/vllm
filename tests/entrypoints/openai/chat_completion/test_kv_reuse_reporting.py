# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for CacheWise tool-call reporting from the chat serving layer."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat


def make_serving(enabled: bool = True) -> OpenAIServingChat:
    """A bare serving object with only the reporting state initialized."""
    serving = object.__new__(OpenAIServingChat)
    serving.enable_kv_reuse_reporting = enabled
    serving._kv_reuse_report_tasks = set()
    serving.engine_client = AsyncMock()
    return serving


@pytest.mark.asyncio
async def test_reports_tool_calls():
    serving = make_serving()
    serving._report_kv_reuse_tool_calls(
        "req-1", num_choices=1, tool_calls=[("Bash", "pytest -x")]
    )
    await asyncio.gather(*serving._kv_reuse_report_tasks)
    serving.engine_client.cachewise_report_tool_calls.assert_awaited_once()
    args = serving.engine_client.cachewise_report_tool_calls.await_args.args
    assert args[0] == "req-1"
    assert args[1] == [("Bash", "pytest -x")]
    assert isinstance(args[2], float)


@pytest.mark.asyncio
async def test_reports_empty_tool_calls_as_human_turn():
    serving = make_serving()
    serving._report_kv_reuse_tool_calls("req-1", num_choices=1, tool_calls=[])
    await asyncio.gather(*serving._kv_reuse_report_tasks)
    args = serving.engine_client.cachewise_report_tool_calls.await_args.args
    assert args[1] == []


@pytest.mark.asyncio
async def test_skips_when_disabled():
    serving = make_serving(enabled=False)
    serving._report_kv_reuse_tool_calls(
        "req-1", num_choices=1, tool_calls=[("Bash", "x")]
    )
    assert not serving._kv_reuse_report_tasks
    serving.engine_client.cachewise_report_tool_calls.assert_not_called()


@pytest.mark.asyncio
async def test_skips_multi_choice_requests():
    serving = make_serving()
    serving._report_kv_reuse_tool_calls(
        "req-1", num_choices=2, tool_calls=[("Bash", "x")]
    )
    assert not serving._kv_reuse_report_tasks
    serving.engine_client.cachewise_report_tool_calls.assert_not_called()


@pytest.mark.asyncio
async def test_reporting_errors_do_not_propagate():
    serving = make_serving()
    serving.engine_client.cachewise_report_tool_calls.side_effect = RuntimeError(
        "engine down"
    )
    serving._report_kv_reuse_tool_calls(
        "req-1", num_choices=1, tool_calls=[("Bash", "x")]
    )
    await asyncio.gather(*serving._kv_reuse_report_tasks, return_exceptions=True)
