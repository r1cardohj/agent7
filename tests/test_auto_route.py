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
    FailoverRoute,
    RouteTarget,
    build_auto_route,
    parse_auto_refs,
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


class AutoRouteConfigTests(unittest.TestCase):
    def test_refs_are_explicit_and_deduplicated(self) -> None:
        self.assertEqual(
            parse_auto_refs("deepseek/chat, kimi/k2,deepseek/chat"),
            [("deepseek", "chat"), ("kimi", "k2")],
        )
        for raw in ("", "auto", "deepseek/", "auto/foo"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_auto_refs(raw)

    def test_logged_out_targets_are_skipped(self) -> None:
        credential = Credential("secret", "https://deepseek.test", "test")
        client = FakeClient([text_turn("ok")])
        providers = {
            "deepseek": SimpleNamespace(
                id="deepseek", resolve=lambda: credential, make_client=lambda cred: client,
            ),
            "kimi": SimpleNamespace(id="kimi", resolve=lambda: None),
        }
        with patch("agent7.routing.get_provider", side_effect=providers.get):
            route = build_auto_route({}, raw="deepseek/chat,kimi/k2")

        self.assertEqual([t.ref for t in route.targets], ["deepseek/chat"])
        self.assertEqual(route.skipped, ("kimi/k2",))

    def test_no_logged_in_target_is_an_error(self) -> None:
        provider = SimpleNamespace(id="deepseek", resolve=lambda: None)
        with patch("agent7.routing.get_provider", return_value=provider):
            with self.assertRaisesRegex(ValueError, "no usable targets"):
                build_auto_route({}, raw="deepseek/chat")


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
