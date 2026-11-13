"""Hidden verifiers copied into a workdir only after the agent has finished."""

from __future__ import annotations

import asyncio
import copy
import os
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path


def quick_greeting() -> None:
    from app.text import greet

    assert greet(" Alice ") == "Hello, Alice!"
    assert greet("") == "Hello, world!"
    assert greet("   ") == "Hello, world!"


def quick_readme_command() -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    assert "python -m app report --format json" in text
    assert "python -m app report --format text" not in text


def quick_boolean_flag() -> None:
    from app.cli import parse_boolean

    for value in ("1", "true", "TRUE", " yes ", "On"):
        assert parse_boolean(value) is True, value
    for value in ("", "0", "false", "off", "no", "truthy"):
        assert parse_boolean(value) is False, value


def coding_slugify() -> None:
    from app.text import slugify

    assert slugify(" Hello, agent7! ") == "hello-agent7"
    assert slugify("One__two--three") == "one-two-three"
    assert slugify("--already--") == "already"
    assert slugify("") == ""


def coding_json_lines() -> None:
    from app.jsonlines import loads_lines

    assert loads_lines('{"a": 1}\n\n[2, 3]\n') == [{"a": 1}, [2, 3]]
    try:
        loads_lines('{"ok": 1}\n\nnot-json\n')
    except ValueError as error:
        assert "line 3" in str(error), str(error)
    else:
        raise AssertionError("invalid JSON did not raise ValueError")


def coding_retry_after() -> None:
    from app.http import parse_retry_after

    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert parse_retry_after(None, now) is None
    assert parse_retry_after("2.5", now) == 2.5
    assert parse_retry_after("-3", now) == 0
    assert parse_retry_after("not-a-date", now) is None
    http_date = "Wed, 01 Jan 2026 00:00:05 GMT"
    delay = parse_retry_after(http_date, now)
    assert delay is not None and abs(delay - 5) < 0.001, delay
    past = (now - timedelta(seconds=5)).strftime("%a, %d %b %Y %H:%M:%S GMT")
    assert parse_retry_after(past, now) == 0


def debug_pagination() -> None:
    from app.pagination import paginate

    items = list(range(10))
    assert paginate(items, 1, 3) == [0, 1, 2]
    assert paginate(items, 2, 3) == [3, 4, 5]
    assert paginate(items, 4, 3) == [9]
    assert paginate([], 1, 3) == []


def debug_memoize() -> None:
    from app.cache import memoize

    calls = []

    @memoize
    def combine(left, right=0):
        calls.append((left, right))
        return left + right

    assert combine(1, right=2) == 3
    assert combine(1, right=2) == 3
    assert calls == [(1, 2)]
    assert combine(2, right=1) == 3
    assert combine(1, right=3) == 4
    assert calls == [(1, 2), (2, 1), (1, 3)]

    @memoize
    def other(left, right=0):
        return left - right

    assert other(1, right=2) == -1


def debug_async_close() -> None:
    from app.async_worker import Worker

    async def scenario() -> None:
        worker = Worker()
        worker.start()
        await asyncio.sleep(0)
        await asyncio.wait_for(worker.close(), timeout=0.25)
        assert worker._task.done()

    asyncio.run(scenario())


def reasoning_transaction_rollback() -> None:
    from app.rollback import run_transaction

    log = []

    def operation(name, fail=False):
        def do():
            log.append(f"do:{name}")
            if fail:
                raise RuntimeError(name)

        def undo():
            log.append(f"undo:{name}")

        return do, undo

    try:
        run_transaction([operation("a"), operation("b", fail=True), operation("c")])
    except RuntimeError as error:
        assert str(error) == "b"
    else:
        raise AssertionError("transaction failure was not re-raised")
    assert log == ["do:a", "do:b", "undo:a"], log

    log.clear()
    run_transaction([operation("a"), operation("b")])
    assert log == ["do:a", "do:b"], log


def reasoning_snapshot_migration() -> None:
    from app.snapshot import migrate_snapshot

    original = {
        "model": "deepseek/old",
        "history": [{"role": "user", "content": "hello"}],
        "cwd": "/workspace",
    }
    before = copy.deepcopy(original)
    migrated = migrate_snapshot(original)
    assert original == before
    assert migrated["version"] == 2
    assert migrated["model_ref"] == "deepseek/old"
    assert migrated["messages"] == original["history"]
    assert migrated["cwd"] == "/workspace"
    assert "model" not in migrated and "history" not in migrated

    legacy = {"model": "legacy/model", "history": [{"role": "user"}]}
    migrated_legacy = migrate_snapshot(legacy)
    assert migrated_legacy["version"] == 2
    assert migrated_legacy["model_ref"] == "legacy/model"
    assert migrated_legacy["messages"] == [{"role": "user"}]

    version_two = {"version": 2, "model_ref": "x", "messages": [{"role": "user"}]}
    copied = migrate_snapshot(version_two)
    assert copied == version_two and copied is not version_two
    assert copied["messages"] is not version_two["messages"]
    try:
        migrate_snapshot({"version": 3, "messages": []})
    except ValueError:
        pass
    else:
        raise AssertionError("future snapshot version was accepted")


def reasoning_config_precedence() -> None:
    from app.config import resolve_config

    defaults = {"host": "default", "port": 8000, "debug": False}
    files = {"host": "file", "port": 9000}
    environment = {"APP_HOST": "environment", "APP_PORT": "9100", "APP_UNUSED": "ignored"}
    overrides = {"host": "explicit"}
    before = copy.deepcopy((defaults, files, environment, overrides))
    resolved = resolve_config(defaults, files, environment, overrides)
    assert resolved == {"host": "explicit", "port": "9100", "debug": False}, resolved
    assert (defaults, files, environment, overrides) == before


VERIFIERS = {
    "quick-greeting": quick_greeting,
    "quick-readme-command": quick_readme_command,
    "quick-boolean-flag": quick_boolean_flag,
    "coding-slugify": coding_slugify,
    "coding-json-lines": coding_json_lines,
    "coding-retry-after": coding_retry_after,
    "debug-pagination": debug_pagination,
    "debug-memoize": debug_memoize,
    "debug-async-close": debug_async_close,
    "reasoning-transaction-rollback": reasoning_transaction_rollback,
    "reasoning-snapshot-migration": reasoning_snapshot_migration,
    "reasoning-config-precedence": reasoning_config_precedence,
}


def main() -> int:
    task = os.environ.get("AGENT7_EVAL_TASK", "")
    if task not in VERIFIERS:
        print(f"unknown evaluation task: {task!r}", file=sys.stderr)
        return 2
    try:
        VERIFIERS[task]()
    except BaseException:
        traceback.print_exc()
        return 1
    print(f"verified: {task}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
