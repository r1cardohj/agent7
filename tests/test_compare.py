"""Offline tests for the external-agent comparison harness."""

from __future__ import annotations

import asyncio
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from evals.compare import (
    AgentResult,
    KimiRunner,
    PiRunner,
    _run_external,
    build_agent_plan,
    load_tasks,
    main,
    make_kimi_config,
    make_pi_models,
    parse_kimi_output,
    parse_pi_json,
)
from evals.report import render_comparison_report
from evals.run import Task


def record(
    task: str,
    agent: str,
    status: str,
    success: bool,
    *,
    seconds: float,
    prompt: int = 0,
    completion: int = 0,
    total: int = 0,
    usage: bool = True,
) -> dict:
    return {
        "task_id": task,
        "category": "coding",
        "strategy": agent,
        "repeat": 1,
        "status": status,
        "success": success,
        "valid": status != "infrastructure_error",
        "agent_seconds": seconds,
        "wall_seconds": seconds + 1,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "usage_observed": usage,
    }


class PlanTests(unittest.TestCase):
    def test_default_plan_is_one_full_round(self) -> None:
        tasks = load_tasks()
        plan = build_agent_plan(tasks, ["agent7", "pi", "kimi"], 1)
        self.assertEqual(len(plan), 36)
        self.assertEqual(
            [(agent, repeat) for _, agent, repeat in plan[:3]],
            [("agent7", 1), ("pi", 1), ("kimi", 1)],
        )
        # Rotate which agent goes first so every agent is not always last.
        self.assertEqual(plan[3][1], "pi")

    def test_dry_run_needs_no_credentials_or_commands(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["--dry-run", "--repeat", "1", "--task", "quick-greeting"])
        self.assertEqual(code, 0)
        self.assertIn("3 runs (1 tasks × 3 agents × 1)", output.getvalue())
        self.assertIn("quick-greeting", output.getvalue())


class PiParserTests(unittest.TestCase):
    def test_parser_totals_assistant_nested_and_compaction_usage(self) -> None:
        events = [
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [],
                    "stopReason": "toolUse",
                    "usage": {
                        "input": 10,
                        "output": 2,
                        "cacheRead": 3,
                        "cacheWrite": 1,
                        "totalTokens": 16,
                    },
                },
            },
            {"type": "tool_execution_end", "toolName": "edit", "isError": False},
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "done"}],
                    "stopReason": "stop",
                    "usage": {
                        "input": 20,
                        "output": 5,
                        "cacheRead": 4,
                        "cacheWrite": 0,
                        "totalTokens": 29,
                    },
                },
            },
            {
                "type": "message_end",
                "message": {
                    "role": "toolResult",
                    "isError": False,
                    "usage": {
                        "input": 1,
                        "output": 1,
                        "cacheRead": 0,
                        "cacheWrite": 0,
                        "totalTokens": 2,
                    },
                },
            },
            {
                "type": "compaction_end",
                "result": {
                    "usage": {
                        "input": 7,
                        "output": 1,
                        "cacheRead": 0,
                        "cacheWrite": 0,
                        "totalTokens": 8,
                    },
                },
            },
        ]
        result = AgentResult()
        parse_pi_json(result, "\n".join(json.dumps(event) for event in events))
        self.assertEqual(result.answer, "done")
        self.assertEqual(result.prompt_tokens, 46)
        self.assertEqual(result.completion_tokens, 9)
        self.assertEqual(result.total_tokens, 55)
        self.assertEqual(result.cache_read_tokens, 7)
        self.assertEqual(result.cache_write_tokens, 1)
        self.assertEqual(result.model_rounds, 2)
        self.assertEqual(result.tool_calls, 1)
        self.assertEqual(result.compactions, 1)
        self.assertTrue(result.usage_observed)
        self.assertIsNone(result.error)

    def test_terminal_error_is_reported_without_hiding_tokens(self) -> None:
        result = AgentResult()
        parse_pi_json(
            result,
            json.dumps(
                {
                    "type": "message_end",
                    "message": {
                        "role": "assistant",
                        "content": "partial",
                        "stopReason": "error",
                        "errorMessage": "provider exploded",
                        "usage": {
                            "input": 5,
                            "output": 1,
                            "cacheRead": 0,
                            "cacheWrite": 0,
                            "totalTokens": 6,
                        },
                    },
                }
            ),
        )
        self.assertEqual(result.total_tokens, 6)
        self.assertIsInstance(result.error, RuntimeError)
        self.assertIn("provider exploded", str(result.error))

    def test_overload_is_an_infrastructure_error_not_a_task_failure(self) -> None:
        result = AgentResult()
        parse_pi_json(
            result,
            json.dumps(
                {
                    "type": "message_end",
                    "message": {
                        "role": "assistant",
                        "content": "partial",
                        "stopReason": "error",
                        "errorMessage": (
                            "The engine is currently overloaded, please try again later"
                        ),
                        "usage": {
                            "input": 5,
                            "output": 1,
                            "cacheRead": 0,
                            "cacheWrite": 0,
                            "totalTokens": 6,
                        },
                    },
                }
            ),
        )
        self.assertEqual(result.error.status_code, 503)

    def test_connection_error_is_an_infrastructure_error(self) -> None:
        result = AgentResult()
        parse_pi_json(
            result,
            json.dumps(
                {
                    "type": "message_end",
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "stopReason": "error",
                        "errorMessage": "Connection error.",
                        "usage": {
                            "input": 0,
                            "output": 0,
                            "cacheRead": 0,
                            "cacheWrite": 0,
                            "totalTokens": 0,
                        },
                    },
                }
            ),
        )
        self.assertEqual(result.error.status_code, 503)
        self.assertFalse(result.usage_observed)
        self.assertEqual(result.total_tokens, 0)

    def test_core_pi_command_disables_local_extensions(self) -> None:
        runner = PiRunner(
            "pi",
            "kimi-bench/k3-256k",
            thinking="high",
            provider="kimi-bench",
            model_id="k3-256k",
            base_url="https://api.kimi.com/coding/v1",
            api_key="secret",
            api="openai-completions",
        )
        task = Task(id="x", category="coding", prompt="fix it")
        command = runner.command(task)
        for flag in ("--no-session", "--no-extensions", "--no-mcp", "--no-skills"):
            self.assertIn(flag, command)
        self.assertLess(command.index("--"), len(command) - 1)


class KimiParserTests(unittest.TestCase):
    def test_parser_totals_main_and_subagent_wire_usage(self) -> None:
        result = AgentResult()
        stdout = "\n".join(
            [
                json.dumps({"role": "meta", "type": "system.version", "version": "2.4.1"}),
                json.dumps(
                    {
                        "role": "assistant",
                        "content": "working",
                        "tool_calls": [{"id": "call-1", "function": {"name": "edit"}}],
                    }
                ),
                json.dumps({"role": "tool", "tool_call_id": "call-1", "content": "ok"}),
                json.dumps({"role": "assistant", "content": "final answer"}),
            ]
        )
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            main_wire = home / "sessions/wd/session-1/agents/main/wire.jsonl"
            sub_wire = home / "sessions/wd/session-1/agents/sub-1/wire.jsonl"
            main_wire.parent.mkdir(parents=True)
            sub_wire.parent.mkdir(parents=True)
            main_wire.write_text(
                "\n".join(
                    [
                        json.dumps({"type": "llm.request", "model": "k3-256k"}),
                        json.dumps(
                            {
                                "type": "usage.record",
                                "usage": {
                                    "inputOther": 100,
                                    "output": 10,
                                    "inputCacheRead": 50,
                                    "inputCacheCreation": 0,
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "type": "context.append_loop_event",
                                "event": {"type": "tool.call", "name": "edit"},
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            sub_wire.write_text(
                "\n".join(
                    [
                        json.dumps({"type": "llm.request", "model": "k3-256k"}),
                        json.dumps(
                            {
                                "type": "usage.record",
                                "usage": {
                                    "inputOther": 20,
                                    "output": 2,
                                    "inputCacheRead": 0,
                                    "inputCacheCreation": 0,
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "type": "context.append_loop_event",
                                "event": {"type": "tool.call", "name": "read"},
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            parse_kimi_output(result, stdout, home)

        self.assertEqual(result.answer, "final answer")
        self.assertEqual(result.prompt_tokens, 170)
        self.assertEqual(result.completion_tokens, 12)
        self.assertEqual(result.total_tokens, 182)
        self.assertEqual(result.cache_read_tokens, 50)
        self.assertEqual(result.model_rounds, 2)
        self.assertEqual(result.tool_calls, 2)
        self.assertTrue(result.usage_observed)
        self.assertEqual(result.token_source, "kimi-wire-all-agents")
        self.assertIsNone(result.error)

    def test_kimi_command_uses_stream_json(self) -> None:
        runner = KimiRunner(
            "kimi",
            "kimi-bench/k3-256k",
            provider="kimi-bench",
            model_id="k3-256k",
            base_url="https://api.kimi.com/coding/v1",
            api_key="secret",
        )
        command = runner.command(Task(id="x", category="coding", prompt="fix it"))
        self.assertIn("--output-format", command)
        self.assertIn("stream-json", command)

    def test_bench_configs_reference_the_shared_environment_key(self) -> None:
        pi_models = json.loads(make_pi_models("kimi-bench", "k3-256k", "https://api.example/v1"))
        provider = pi_models["providers"]["kimi-bench"]
        self.assertEqual(provider["api"], "openai-completions")
        self.assertEqual(provider["apiKey"], "$AGENT7_KIMI_API_KEY")
        self.assertFalse(provider["compat"]["supportsDeveloperRole"])
        config = make_kimi_config("kimi-bench", "k3-256k", "https://api.example/v1")
        self.assertIn('api_key_env = "AGENT7_KIMI_API_KEY"', config)
        self.assertIn('default_model = "kimi-bench/k3-256k"', config)


class SubprocessTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancellation_kills_process_and_keeps_output_handles(self) -> None:
        result = AgentResult()
        command = [
            sys.executable,
            "-c",
            "import sys, time; print('partial', flush=True); time.sleep(30)",
        ]
        with (
            tempfile.TemporaryDirectory() as temporary,
            self.assertRaises(asyncio.TimeoutError),
        ):
            await asyncio.wait_for(
                _run_external(command, Path(temporary), dict(), result),
                timeout=0.3,
            )
        self.assertEqual(result.raw_output.strip(), "partial")
        self.assertEqual(result.raw_error, "")


class ComparisonReportTests(unittest.TestCase):
    def test_report_separates_completion_time_tokens_and_missing_usage(self) -> None:
        records = [
            record("a", "agent7", "passed", True, seconds=10, prompt=90, completion=10, total=100),
            record("a", "pi", "passed", True, seconds=20, prompt=50, completion=5, total=55),
            record("a", "kimi", "passed", True, seconds=15, prompt=70, completion=7, total=77),
            record(
                "b", "agent7", "failed", False, seconds=30, prompt=190, completion=20, total=210
            ),
            record("b", "pi", "failed", False, seconds=40, usage=False),
            record("b", "kimi", "infrastructure_error", False, seconds=1),
        ]
        rendered = render_comparison_report(
            records,
            config={
                "models": {
                    "agent7": "kimi/k3-256k",
                    "pi": "kimi-coding/k3-256k",
                    "kimi": "kimi-code/k3-256k",
                },
                "agent_versions": {"agent7": "0.3.2", "pi": "1.4.0", "kimi": "2.4.1"},
                "pi_thinking": "high",
            },
        )
        self.assertIn("Coding-agent head-to-head evaluation", rendered)
        self.assertIn("| agent7 | 2 | 2 | 1 | 50.0 | 2/2 |", rendered)
        self.assertIn("| pi | 2 | 2 | 1 | 50.0 | 1/2 |", rendered)
        self.assertIn("| kimi | 2 | 1 | 1 | 100.0 | 1/1 |", rendered)
        self.assertIn("n/a tok", rendered)
        self.assertIn("`kimi-coding/k3-256k`", rendered)
        self.assertIn("agent7 minus baseline", rendered)


if __name__ == "__main__":
    unittest.main()
