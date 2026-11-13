"""Zero-config intent scoring chooses and orders the confirmed auto candidates."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import agent7.agent as A
from agent7.agent import run_agent
from agent7.providers import Credential
from agent7.routing import MODEL_CAPABILITIES, FailoverRoute, RouteTarget
from responses_fakes import FakeClient, call_turn, text_turn


def score_response(text: str):
    return SimpleNamespace(output=[{
        "type": "message",
        "content": [{"type": "output_text", "text": text}],
    }])


class ScoringClient:
    """A Responses fake with separate non-stream scoring and streaming execution."""

    def __init__(self, score_text: str = "", script=None, error: Exception | None = None) -> None:
        self.worker = FakeClient([text_turn("unused")] if script is None else script)
        self.score_text = score_text
        self.error = error
        self.score_requests = []
        self.responses = self

    @property
    def calls(self) -> int:
        return self.worker.calls

    @property
    def requests(self):
        return self.worker.requests

    async def create(self, **kwargs):
        if kwargs.get("stream"):
            return await self.worker.responses.create(**kwargs)
        self.score_requests.append(kwargs)
        if self.error:
            raise self.error
        return score_response(self.score_text)

    async def close(self) -> None:
        pass


def capability_target(ref: str, client: ScoringClient) -> RouteTarget:
    capability = next(cap for cap in MODEL_CAPABILITIES if cap.ref == ref)
    credential = Credential("secret", f"https://{capability.provider_id}.test", "test")
    return RouteTarget(
        capability.provider_id, capability.model, credential, client,
        capability.context_limit, capability,
    )


def scores(intent: str, values: dict[str, float]) -> str:
    return json.dumps({"intent": intent, "scores": values})


class IntentSelectionTests(unittest.TestCase):
    def start(self, route: FailoverRoute, task: str = "fix the bug", tokens: int = 10) -> None:
        messages = [{"role": "user", "content": task}]
        asyncio.run(route.start(task, messages, tokens))

    def test_fast_model_scores_all_candidates_once_and_orders_by_score(self) -> None:
        deepseek = ScoringClient(scores("debugging", {
            "deepseek/deepseek-flash": 40,
            "kimi/kimi-for-coding": 95,
            "kimi/k3.1-pro": 80,
        }))
        kimi = ScoringClient()
        strong_kimi = ScoringClient()
        route = FailoverRoute((
            capability_target("deepseek/deepseek-flash", deepseek),
            capability_target("kimi/kimi-for-coding", kimi),
            capability_target("kimi/k3.1-pro", strong_kimi),
        ), intent_routing=True)

        self.start(route, "find the race condition")

        self.assertEqual(route.current.ref, "kimi/kimi-for-coding")
        self.assertEqual([target.ref for target in route.active_targets], [
            "kimi/kimi-for-coding", "kimi/k3.1-pro", "deepseek/deepseek-flash",
        ])
        self.assertEqual(route.selection.intent, "debugging")
        self.assertEqual(route.selection.scorer_ref, "deepseek/deepseek-flash")
        self.assertEqual(len(deepseek.score_requests), 1)
        self.assertEqual(kimi.score_requests, [])
        self.assertEqual(strong_kimi.score_requests, [])
        request = deepseek.score_requests[0]
        self.assertFalse(request["stream"])
        self.assertEqual(request["temperature"], 0)
        self.assertEqual(request["reasoning"], {"effort": "none"})
        payload = json.loads(request["input"])
        self.assertEqual(payload["task"], "find the race condition")
        self.assertEqual(len(payload["candidates"]), 3)
        self.assertTrue(all(c["routing_hint"] for c in payload["candidates"]))

    def test_first_task_uses_the_highest_score_without_an_incumbent_margin(self) -> None:
        deepseek = ScoringClient(scores("coding", {
            "deepseek/deepseek-flash": 88,
            "kimi/kimi-for-coding": 90,
        }))
        kimi = ScoringClient()
        route = FailoverRoute((
            capability_target("deepseek/deepseek-flash", deepseek),
            capability_target("kimi/kimi-for-coding", kimi),
        ), intent_routing=True)

        self.start(route)

        self.assertEqual(route.current.ref, "kimi/kimi-for-coding")

    def test_unknown_scorer_output_falls_back_instead_of_injecting_a_target(self) -> None:
        deepseek = ScoringClient(scores("coding", {"not-configured/model": 100}))
        kimi = ScoringClient()
        route = FailoverRoute((
            capability_target("deepseek/deepseek-flash", deepseek),
            capability_target("kimi/kimi-for-coding", kimi),
        ), intent_routing=True)

        self.start(route)

        self.assertEqual(route.current.ref, "deepseek/deepseek-flash")
        self.assertEqual(route.selection.intent, "default")
        self.assertEqual(route.selection.error, "ValueError")
        self.assertEqual(len(deepseek.score_requests), 1)

    def test_scorer_error_keeps_the_current_target(self) -> None:
        deepseek = ScoringClient(scores("coding", {
            "deepseek/deepseek-flash": 10,
            "kimi/kimi-for-coding": 90,
        }))
        kimi = ScoringClient()
        route = FailoverRoute((
            capability_target("deepseek/deepseek-flash", deepseek),
            capability_target("kimi/kimi-for-coding", kimi),
        ), intent_routing=True)
        self.start(route, "first task")
        self.assertEqual(route.current.ref, "kimi/kimi-for-coding")
        deepseek.error = TimeoutError()

        self.start(route, "second task")

        self.assertEqual(route.current.ref, "kimi/kimi-for-coding")
        self.assertEqual(route.selection.error, "TimeoutError")
        self.assertEqual([target.ref for target in route.active_targets], [
            "kimi/kimi-for-coding", "deepseek/deepseek-flash",
        ])

    def test_one_candidate_skips_scoring(self) -> None:
        deepseek = ScoringClient("not json")
        route = FailoverRoute((
            capability_target("deepseek/deepseek-flash", deepseek),
        ), intent_routing=True)

        self.start(route)

        self.assertEqual(deepseek.score_requests, [])
        self.assertIsNone(route.selection)

    def test_context_filter_runs_before_scoring(self) -> None:
        # 300k fits deepseek-flash (1M) but not a 256k fallback target.
        deepseek = ScoringClient("not json")
        small = ScoringClient()
        route = FailoverRoute((
            capability_target("deepseek/deepseek-flash", deepseek),
            RouteTarget("test", "small", Credential("s", "https://test", "test"), small, 256_000),
        ), intent_routing=True)

        self.start(route, tokens=300_000)

        self.assertEqual(small.score_requests, [])

        self.assertEqual(route.current.ref, "deepseek/deepseek-flash")
        self.assertEqual(route.active_targets, (route.current,))
        self.assertEqual(deepseek.score_requests, [])

    def previous_kimi_route(self, deepseek: ScoringClient, kimi: ScoringClient) -> FailoverRoute:
        route = FailoverRoute((
            capability_target("deepseek/deepseek-flash", deepseek),
            capability_target("kimi/kimi-for-coding", kimi),
        ), intent_routing=True)
        deepseek.score_text = scores("coding", {
            "deepseek/deepseek-flash": 10,
            "kimi/kimi-for-coding": 90,
        })
        self.start(route, "first task")
        self.assertEqual(route.current.ref, "kimi/kimi-for-coding")
        return route

    def test_close_scores_keep_the_current_target(self) -> None:
        deepseek, kimi = ScoringClient(), ScoringClient()
        route = self.previous_kimi_route(deepseek, kimi)
        deepseek.score_text = scores("coding", {
            "deepseek/deepseek-flash": 87,
            "kimi/kimi-for-coding": 85,
        })

        self.start(route, "second task")

        self.assertEqual(route.current.ref, "kimi/kimi-for-coding")

    def test_three_point_score_gain_switches_the_current_target(self) -> None:
        deepseek, kimi = ScoringClient(), ScoringClient()
        route = self.previous_kimi_route(deepseek, kimi)
        deepseek.score_text = scores("coding", {
            "deepseek/deepseek-flash": 88,
            "kimi/kimi-for-coding": 85,
        })

        self.start(route, "second task")

        self.assertEqual(route.current.ref, "deepseek/deepseek-flash")


class IntentExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        retry = patch.object(A, "RETRY_SLEEPS", (0, 0, 0))
        retry.start()
        self.addCleanup(retry.stop)

    def test_scores_once_and_pins_the_winner_through_tool_rounds(self) -> None:
        (self.cwd / "note.txt").write_text("hello", encoding="utf-8")
        deepseek = ScoringClient(scores("coding", {
            "deepseek/deepseek-flash": 20,
            "kimi/kimi-for-coding": 95,
        }))
        kimi = ScoringClient(script=[
            call_turn(("read", {"path": "note.txt"})), text_turn("done"),
        ])
        route = FailoverRoute((
            capability_target("deepseek/deepseek-flash", deepseek),
            capability_target("kimi/kimi-for-coding", kimi),
        ), intent_routing=True)

        answer = asyncio.run(run_agent(
            "fix the file", cwd=self.cwd, client=deepseek, model="auto",
            messages=[], route=route, on_tool=lambda *args: None,
        ))

        self.assertEqual(answer, "done")
        self.assertEqual(len(deepseek.score_requests), 1)
        self.assertEqual(deepseek.calls, 0)
        self.assertEqual(kimi.calls, 2)
        self.assertEqual(route.current.ref, "kimi/kimi-for-coding")

    def test_failover_follows_the_existing_score_order_without_rescoring(self) -> None:
        class _Status(Exception):
            status_code = 503

        deepseek = ScoringClient(scores("debugging", {
            "deepseek/deepseek-flash": 10,
            "kimi/kimi-for-coding": 90,
            "kimi/k3.1-pro": 80,
        }))
        kimi = ScoringClient(script=[_Status("unavailable")])
        strong_kimi = ScoringClient(script=[text_turn("ok")])
        route = FailoverRoute((
            capability_target("deepseek/deepseek-flash", deepseek),
            capability_target("kimi/kimi-for-coding", kimi),
            capability_target("kimi/k3.1-pro", strong_kimi),
        ), intent_routing=True)

        answer = asyncio.run(run_agent(
            "debug it", cwd=self.cwd, client=deepseek, model="auto",
            messages=[], route=route,
        ))

        self.assertEqual(answer, "ok")
        self.assertEqual(kimi.calls, 4)
        self.assertEqual(strong_kimi.calls, 1)
        self.assertEqual(deepseek.calls, 0)
        self.assertEqual(len(deepseek.score_requests), 1)


if __name__ == "__main__":
    unittest.main()
