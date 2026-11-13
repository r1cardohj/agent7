"""Offline evaluation harness tests; none of these tests calls a provider."""

import asyncio
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from evals.report import paired_success_delta, render_report
from evals.run import (
    EVAL_ROOT,
    AgentResult,
    FailoverRoute,
    ObservedRoute,
    RouteTarget,
    build_plan,
    execute_run,
    load_tasks,
    main,
)
from evals.verify import VERIFIERS
from agent7.providers import Credential
from agent7.routing import MODEL_CAPABILITIES
from responses_fakes import FakeClient, text_turn


class FakeRunner:
    def __init__(self, mode="pass") -> None:
        self.mode = mode

    async def run(self, task, cwd, strategy):
        result = AgentResult(
            answer="fake answer", selected_model="fake/model", initial_model="fake/model",
            prompt_tokens=10, completion_tokens=5, total_tokens=15,
        )
        if self.mode == "pass":
            path = cwd / "app" / "text.py"
            text = path.read_text(encoding="utf-8")
            text = text.replace(
                '    return f"Hello {name}"',
                '    name = name.strip() or "world"\n    return f"Hello, {name}!"',
            )
            path.write_text(text, encoding="utf-8")
        elif self.mode == "infra":
            error = RuntimeError("provider unavailable")
            error.status_code = 429
            result.error = error
        return result


def capability_target(ref):
    capability = next(cap for cap in MODEL_CAPABILITIES if cap.ref == ref)
    credential = Credential("secret", "https://example.test", "test")
    return RouteTarget(
        capability.provider_id, capability.model, credential,
        FakeClient([text_turn("unused")]), capability.context_limit, capability,
    )


class TaskManifestTests(unittest.TestCase):
    def test_shipped_tasks_form_the_documented_plan(self) -> None:
        tasks = load_tasks()
        self.assertEqual(len(tasks), 12)
        self.assertEqual(Counter(task.category for task in tasks), {
            "quick": 3, "coding": 3, "debugging": 3, "reasoning": 3,
        })
        self.assertEqual({task.id for task in tasks}, set(VERIFIERS))
        self.assertEqual(len(build_plan(
            tasks, ["fixed-fast", "fixed-strong", "auto-tier", "auto-full"], 2,
        )), 96)

    def test_dry_run_needs_no_credentials(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["--dry-run", "--repeat", "1", "--task", "quick-greeting"])
        self.assertEqual(code, 0)
        self.assertIn("4 runs (1 tasks × 4 strategies × 1)", output.getvalue())

    def test_every_hidden_verifier_rejects_the_pristine_fixture(self) -> None:
        for task in load_tasks():
            with self.subTest(task=task.id), tempfile.TemporaryDirectory() as temporary:
                cwd = Path(temporary) / "repo"
                shutil.copytree(EVAL_ROOT / "repos" / task.repo, cwd)
                shutil.copy2(EVAL_ROOT / "verify.py", cwd / "_agent7_eval_verify.py")
                completed = subprocess.run(
                    [sys.executable, "_agent7_eval_verify.py"], cwd=cwd,
                    env=dict(os.environ, AGENT7_EVAL_TASK=task.id, PYTHONDONTWRITEBYTECODE="1"),
                    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=5,
                )
                self.assertNotEqual(completed.returncode, 0, completed.stdout)


class RouteAblationTests(unittest.TestCase):
    def start(self, tier_only: bool) -> ObservedRoute:
        refs = ("deepseek/deepseek-flash", "xai/grok-4.7", "kimi/k3.1-pro")
        base = FailoverRoute(tuple(capability_target(ref) for ref in refs), intent_routing=True)
        route = ObservedRoute(base, tier_only=tier_only)

        async def fake_score(*args):
            return "debugging", {
                "deepseek/deepseek-flash": 10,
                "xai/grok-4.7": 90,
                "kimi/k3.1-pro": 100,
            }

        with patch("agent7.routing._score_targets", new=fake_score):
            asyncio.run(route.start("debug", [{"role": "user", "content": "debug"}], 10))
        return route

    def test_tier_ablation_uses_the_first_strong_model(self) -> None:
        route = self.start(tier_only=True)
        self.assertEqual(route.current.ref, "xai/grok-4.7")
        asyncio.run(route.close())

    def test_full_policy_still_uses_the_score_argmax(self) -> None:
        route = self.start(tier_only=False)
        self.assertEqual(route.current.ref, "kimi/k3.1-pro")
        asyncio.run(route.close())


class ExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.output = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.task = next(task for task in load_tasks() if task.id == "quick-greeting")

    def execute(self, mode: str):
        return asyncio.run(execute_run(
            self.task, "fixed-fast", 1, self.output, FakeRunner(mode),
        ))

    def test_fake_agent_passes_hidden_verification_and_saves_artifacts(self) -> None:
        record = self.execute("pass")
        self.assertEqual(record["status"], "passed")
        self.assertTrue(record["success"])
        self.assertEqual(record["verify_exit"], 0)
        artifact = self.output / record["artifact"]
        self.assertEqual((artifact / "answer.txt").read_text(encoding="utf-8"), "fake answer")
        patch_text = (artifact / "changes.patch").read_text(encoding="utf-8")
        self.assertIn("Hello,", patch_text)
        self.assertNotIn("_agent7_eval_verify.py", patch_text)
        self.assertIn("verified: quick-greeting", (artifact / "verify.txt").read_text())
        saved = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["run_id"], record["run_id"])

    def test_failed_verification_is_a_valid_unsuccessful_run(self) -> None:
        record = self.execute("fail")
        self.assertEqual(record["status"], "failed")
        self.assertFalse(record["success"])
        self.assertTrue(record["valid"])

    def test_provider_failure_is_reported_but_not_a_task_failure(self) -> None:
        record = self.execute("infra")
        self.assertEqual(record["status"], "infrastructure_error")
        self.assertFalse(record["valid"])
        self.assertIsNone(record["verify_exit"])

    def test_cancelled_agent_is_a_valid_failure_and_keeps_partial_artifacts(self) -> None:
        class SlowRunner:
            def begin(self, strategy):
                self.result = AgentResult(
                    selected_model="late/model", intent="coding", prompt_tokens=99,
                )
                return self.result

            async def run(self, task, cwd, strategy):
                await asyncio.sleep(30)

        record = asyncio.run(execute_run(
            self.task, "fixed-fast", 1, self.output, SlowRunner(), timeout=0.05,
        ))
        self.assertEqual(record["status"], "timeout")
        self.assertTrue(record["valid"])
        self.assertEqual(record["selected_model"], "late/model")
        self.assertEqual(record["prompt_tokens"], 99)
        artifact = self.output / record["artifact"]
        self.assertTrue((artifact / "changes.patch").is_file())
        self.assertTrue((artifact / "events.jsonl").is_file())

    def test_outer_cancellation_stops_the_agent_and_retains_progress(self) -> None:
        class CancellableRunner:
            def __init__(self):
                self.cancelled = False
                self.current = None

            def begin(self, strategy):
                self.current = AgentResult(selected_model="cancelled/model", prompt_tokens=7)
                return self.current

            async def run(self, task, cwd, strategy):
                try:
                    await asyncio.sleep(30)
                except asyncio.CancelledError:
                    self.cancelled = True
                    raise

        async def exercise():
            runner = CancellableRunner()
            running = asyncio.create_task(execute_run(
                self.task, "auto-full", 1, self.output, runner,
            ))
            await asyncio.sleep(0.05)
            running.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await running
            return runner

        runner = asyncio.run(exercise())
        self.assertTrue(runner.cancelled)
        self.assertEqual(runner.current.selected_model, "cancelled/model")
        artifacts = list((self.output / "artifacts").iterdir())
        self.assertEqual(len(artifacts), 1)
        self.assertTrue((artifacts[0] / "changes.patch").is_file())


class ReportTests(unittest.TestCase):
    @staticmethod
    def record(task: str, strategy: str, status: str, success: bool) -> dict:
        return {
            "task_id": task, "category": "coding", "strategy": strategy, "repeat": 1,
            "status": status, "success": success, "wall_seconds": 10,
            "score_seconds": 1, "prompt_tokens": 100, "completion_tokens": 20,
            "score_prompt_tokens": 5, "score_completion_tokens": 2, "failovers": 0,
        }

    def test_report_counts_invalid_runs_and_bootstraps_paired_tasks(self) -> None:
        records = [
            self.record("a", "auto-full", "passed", True),
            self.record("a", "fixed-fast", "failed", False),
            self.record("b", "auto-full", "passed", True),
            self.record("b", "fixed-fast", "passed", True),
            self.record("c", "auto-full", "infrastructure_error", False),
            self.record("c", "fixed-fast", "passed", True),
        ]
        delta = paired_success_delta(records, "auto-full", "fixed-fast", samples=100)
        self.assertEqual(delta["tasks"], 2)
        self.assertEqual(delta["delta_pp"], 50)
        rendered = render_report(records)
        self.assertIn("| auto-full | 3 | 2 | 2 | 100.0 | 1", rendered)
        self.assertIn("auto-full minus baseline", rendered)
        self.assertIn("Tokens include execution and auto-scoring", rendered)


if __name__ == "__main__":
    unittest.main()
