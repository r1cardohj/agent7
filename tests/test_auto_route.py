"""Ordered auto routes fail over only when a request is safe to replay."""

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import agent7.agent as A
from agent7 import cli
from agent7.agent import run_agent
from agent7.commands import switch_model
from agent7.providers import Credential
from agent7.responses import RAW_ITEMS_KEY, raw_state
from agent7.routing import (
    MODEL_CAPABILITIES,
    FailoverRoute,
    RouteTarget,
    build_auto_route,
)
from agent7.session import Session
from agent7.store import load, save
from responses_fakes import FakeClient, call_turn, text_turn


class _Status(Exception):
    def __init__(self, status: int, message: str = "") -> None:
        self.status_code = status
        super().__init__(message or f"status {status}")


def target(provider: str, model: str, script, *, limit: int | None = None) -> RouteTarget:
    credential = Credential("secret", f"https://{provider}.test", "test")
    return RouteTarget(provider, model, credential, FakeClient(script), limit)


class DefaultAutoRouteTests(unittest.TestCase):
    def test_capability_table_is_the_confirmed_candidate_set(self) -> None:
        self.assertEqual({cap.ref for cap in MODEL_CAPABILITIES}, {
            "deepseek/deepseek-flash",
            "kimi/kimi-for-coding",
            "xai/grok-build-0.1",
            "xai/grok-4.7",
            "kimi/k3.1-pro",
        })
        for ref in ("deepseek/deepseek-flash", "kimi/kimi-for-coding", "xai/grok-build-0.1"):
            cap = next(cap for cap in MODEL_CAPABILITIES if cap.ref == ref)
            self.assertEqual(cap.strengths, frozenset({"fast", "coding"}))
        for ref in ("xai/grok-4.7", "kimi/k3.1-pro"):
            cap = next(cap for cap in MODEL_CAPABILITIES if cap.ref == ref)
            self.assertEqual(cap.strengths, frozenset({"strong"}))
        grok = next(cap for cap in MODEL_CAPABILITIES if cap.ref == "xai/grok-4.7")
        kimi = next(cap for cap in MODEL_CAPABILITIES if cap.ref == "kimi/k3.1-pro")
        coding_kimi = next(
            cap for cap in MODEL_CAPABILITIES if cap.ref == "kimi/kimi-for-coding"
        )
        self.assertEqual(grok.context_limit, 500_000)
        self.assertEqual(coding_kimi.context_limit, 1_048_576)
        self.assertIn("debugging", grok.routing_hint)
        self.assertIn("architecture", kimi.routing_hint)

    def test_auto_uses_logged_in_reviewed_models_without_configuration(self) -> None:
        credentials = {
            pid: Credential("secret", f"https://{pid}.test", "test")
            for pid in ("deepseek", "kimi")
        }
        providers = [
            SimpleNamespace(
                id=pid, resolve=lambda pid=pid: credentials.get(pid),
                make_client=lambda cred: FakeClient([text_turn("ok")]),
            )
            for pid in ("deepseek", "xai", "kimi")
        ]
        with patch("agent7.routing.list_providers", return_value=providers):
            route = build_auto_route(credentials)

        self.assertTrue(route.intent_routing)
        self.assertEqual([target.ref for target in route.targets], [
            "deepseek/deepseek-flash", "kimi/kimi-for-coding", "kimi/k3.1-pro",
        ])

    def test_current_model_list_filters_candidates(self) -> None:
        credential = Credential("secret", "https://deepseek.test", "test")
        provider = SimpleNamespace(
            id="deepseek", resolve=lambda: credential,
            make_client=lambda cred: FakeClient([text_turn("ok")]),
        )
        with patch("agent7.routing.list_providers", return_value=[provider]):
            route = build_auto_route(
                {"deepseek": credential},
                models=["deepseek/deepseek-chat", "deepseek/deepseek-flash"],
            )
        self.assertEqual([target.ref for target in route.targets], ["deepseek/deepseek-flash"])

    def test_no_supported_logged_in_model_is_an_error(self) -> None:
        with patch("agent7.routing.list_providers", return_value=[]):
            with self.assertRaisesRegex(ValueError, "no supported models"):
                build_auto_route({})


class AutoFailoverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        retry = patch.object(A, "RETRY_SLEEPS", (0, 0, 0))
        retry.start()
        self.addCleanup(retry.stop)

    def run_route(self, route: FailoverRoute, *, seed=None, on_delta=None):
        messages = list(seed or [])
        changes = []
        answer = asyncio.run(run_agent(
            "go", cwd=self.cwd, client=route.current.client, model="auto",
            messages=messages, route=route, on_delta=on_delta,
            on_route=lambda *args: changes.append(args), on_tool=lambda *args: None,
        ))
        return answer, messages, changes

    def test_retries_one_target_then_fails_over(self) -> None:
        first = target("deepseek", "chat", [_Status(429)])
        second = target("kimi", "k2", [text_turn("ok")])
        route = FailoverRoute((first, second))

        answer, messages, changes = self.run_route(route)

        self.assertEqual(answer, "ok")
        self.assertEqual(first.client.calls, 4)
        self.assertEqual(second.client.calls, 1)
        self.assertEqual(route.current, second)
        self.assertEqual(messages[-1][RAW_ITEMS_KEY]["provider"], "kimi")
        self.assertEqual(len([m for m in messages if m.get("role") == "user"]), 1)
        self.assertTrue(any(change[1] is first and change[0] is second for change in changes))

    def test_auth_and_missing_model_errors_fail_over_without_retrying(self) -> None:
        for status in (401, 403, 404):
            with self.subTest(status=status):
                first = target("deepseek", "chat", [_Status(status)])
                second = target("kimi", "k2", [text_turn("ok")])
                route = FailoverRoute((first, second))

                answer, _, _ = self.run_route(route)

                self.assertEqual(answer, "ok")
                self.assertEqual(first.client.calls, 1)
                self.assertEqual(second.client.calls, 1)
                self.assertEqual(route.current, second)

    def test_non_retryable_error_does_not_fail_over(self) -> None:
        first = target("deepseek", "chat", [_Status(400, "bad tools")])
        second = target("kimi", "k2", [text_turn("wrong")])
        route = FailoverRoute((first, second))

        with self.assertRaises(_Status):
            self.run_route(route)

        self.assertEqual(first.client.calls, 1)
        self.assertEqual(second.client.calls, 0)
        self.assertEqual(route.current, first)

    def test_partial_output_is_never_replayed_on_another_target(self) -> None:
        first = target("deepseek", "chat", [(text_turn("partial"), _Status(503))])
        second = target("kimi", "k2", [text_turn("wrong")])
        route = FailoverRoute((first, second))
        seen = []

        with self.assertRaises(_Status):
            self.run_route(route, on_delta=seen.append)

        self.assertEqual(seen, ["partial"])
        self.assertEqual(first.client.calls, 1)
        self.assertEqual(second.client.calls, 0)

    def test_tool_continuation_stays_on_the_failover_target(self) -> None:
        (self.cwd / "note.txt").write_text("hello", encoding="utf-8")
        first = target("deepseek", "chat", [_Status(503)])
        second = target("kimi", "k2", [
            call_turn(("read", {"path": "note.txt"})), text_turn("done"),
        ])
        route = FailoverRoute((first, second))

        answer, messages, _ = self.run_route(route)

        self.assertEqual(answer, "done")
        self.assertEqual(second.client.calls, 2)
        self.assertEqual(route.current, second)
        self.assertEqual(messages[-2]["role"], "tool")
        self.assertIn("hello", messages[-2]["content"])

    def test_cross_provider_history_uses_portable_messages(self) -> None:
        old_raw = [{"id": "msg_old", "type": "message", "role": "assistant", "content": []}]
        seed = [
            {"role": "user", "content": "earlier"},
            {"role": "assistant", "content": "old answer", RAW_ITEMS_KEY: raw_state("deepseek", old_raw)},
        ]
        first = target("deepseek", "chat", [_Status(503)])
        second = target("kimi", "k2", [text_turn("ok")])
        route = FailoverRoute((first, second))

        self.run_route(route, seed=seed)

        replayed = second.client.requests[0]["input"]
        self.assertIn({"role": "assistant", "content": "old answer"}, replayed)
        self.assertNotIn(old_raw[0], replayed)


class AutoSessionTests(unittest.TestCase):
    def test_cli_auto_builds_a_route_and_saves_the_logical_ref(self) -> None:
        async def exercise(root: Path, route: FailoverRoute):
            credential = Credential("secret", "https://deepseek.test", "test")
            provider = SimpleNamespace(
                id="deepseek", make_client=lambda cred: FakeClient([text_turn("unused")]),
            )
            seen = []

            async def fake_run(session, task, **kwargs):
                seen.append((session.display_model, session.route, task))
                kwargs["on_saved"]()
                return "done"

            with patch.object(cli, "resolve_all", return_value={"deepseek": credential}), \
                 patch.object(cli, "resolve_default", return_value=(provider, credential)), \
                 patch.object(cli, "fetch_models", return_value=asyncio.Future()) as fetched, \
                 patch("agent7.commands.build_auto_route", return_value=route), \
                 patch.object(cli, "run_turn", new=fake_run):
                fetched.return_value.set_result([])
                await cli.async_main(cli.Options.parse(["--cwd", str(root), "-m", "auto", "go"]))
            self.assertEqual(seen, [("auto → deepseek/chat", route, "go")])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            route = FailoverRoute((target("deepseek", "chat", [text_turn("ok")]),))
            with patch.dict(os.environ, {"XDG_STATE_HOME": str(root / "state")}):
                asyncio.run(exercise(root, route))
                self.assertEqual(load(root)["model_ref"], "auto")

    def test_switch_keeps_logical_auto_and_saves_it(self) -> None:
        async def exercise(root: Path):
            route = FailoverRoute((target("deepseek", "chat", [text_turn("ok")]),))
            session = Session(model="pending", cwd=root, client=None)
            with patch("agent7.commands.build_auto_route", return_value=route):
                result = await switch_model(session, "auto")
            self.assertEqual(session.model_ref, "auto")
            self.assertEqual(session.display_model, "auto → deepseek/chat")
            self.assertIn("auto → deepseek/chat", result.message)
            save(session, created_at="2025-01-01T00:00:00+00:00")
            await session.close()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, {"XDG_STATE_HOME": str(root / "state")}):
                asyncio.run(exercise(root))
                self.assertEqual(load(root)["model_ref"], "auto")


if __name__ == "__main__":
    unittest.main()
