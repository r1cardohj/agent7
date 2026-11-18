"""Test-only runner for exercising the CLI assembly without exporting an API."""

from __future__ import annotations

from typing import Any

from agent0_runtime import RunStatus

from agent7.agent import run_agent_result


async def run_agent(task: str, **kwargs: Any) -> str:
    result = await run_agent_result(task, **kwargs)
    if result.status is RunStatus.BLOCKED and result.stop_reason == "max_turns":
        return f"(stopped after {result.turns} turns)"
    return result.text
