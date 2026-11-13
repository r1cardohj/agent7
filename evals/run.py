"""Run isolated coding tasks against fixed and auto model-routing policies.

This is a developer tool, not part of the agent7 wheel. It deliberately uses
real agent/API calls unless a test supplies a fake runner.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import suppress
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from agent7.agent import SYSTEM, run_agent
from agent7.providers import Credential, get_provider, resolve_all
from agent7.responses import response_usage
from agent7.routing import FailoverRoute, RouteTarget, build_auto_route
from agent7.session import resolve_context_limit

from .report import render_report

EVAL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_ROOT.parent
DEFAULT_MANIFEST = EVAL_ROOT / "tasks.json"
STRATEGIES = ("fixed-fast", "fixed-strong", "auto-tier", "auto-full")
AUTO_STRATEGIES = frozenset({"auto-tier", "auto-full"})
INVALID_STATUSES = frozenset({
    "infrastructure_error", "verify_error", "runner_error", "interrupted",
})
_INFRA_STATUSES = frozenset({401, 403, 404, 408, 409, 425, 429, 500, 502, 503, 504})
_INFRA_ERRORS = frozenset({
    "APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError",
    "ServiceUnavailableError", "BadGatewayError", "AuthenticationError",
    "PermissionDeniedError", "NotFoundError",
})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Task:
    id: str
    category: str
    prompt: str
    repo: str = "pyapp"
    language: str = "en"
    setup: str = ""
    verify: str = "{python} _agent7_eval_verify.py"
    timeout: int = 240
    verify_timeout: int = 60
    max_turns: int = 40


@dataclass
class AgentResult:
    answer: str = ""
    selected_model: str = ""
    initial_model: str = ""
    intent: str = ""
    scores: dict[str, float] = field(default_factory=dict)
    scorer: str = ""
    score_error: str = ""
    score_seconds: float = 0.0
    score_prompt_tokens: int = 0
    score_completion_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model_rounds: int = 0
    tool_rounds: int = 0
    tool_calls: int = 0
    compactions: int = 0
    failovers: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)
    error: BaseException | None = None


def load_tasks(path: Path = DEFAULT_MANIFEST) -> list[Task]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("tasks")
    if not isinstance(data, list):
        raise ValueError("task manifest must be a JSON array or contain a tasks array")
    allowed = {item.name for item in fields(Task)}
    tasks: list[Task] = []
    seen: set[str] = set()
    for raw in data:
        if not isinstance(raw, dict) or not set(raw) >= {"id", "category", "prompt"}:
            raise ValueError("each task needs id, category, and prompt")
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown task fields: {sorted(unknown)}")
        task = Task(**raw)
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", task.id):
            raise ValueError(f"invalid task id: {task.id!r}")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", task.repo):
            raise ValueError(f"invalid fixture repository: {task.repo!r}")
        if task.id in seen:
            raise ValueError(f"duplicate task id: {task.id}")
        if task.timeout <= 0 or task.verify_timeout <= 0 or task.max_turns <= 0:
            raise ValueError(f"task limits must be positive: {task.id}")
        seen.add(task.id)
        tasks.append(task)
    if not tasks:
        raise ValueError("task manifest is empty")
    return tasks


def build_plan(tasks: list[Task], strategies: list[str], repeat: int) -> list[tuple[Task, str, int]]:
    """Interleave policies within a task instead of running one provider in a block."""
    if repeat < 1:
        raise ValueError("repeat must be at least one")
    unknown = set(strategies) - set(STRATEGIES)
    if unknown or not strategies:
        raise ValueError(f"unknown strategies: {sorted(unknown)}")
    return [(task, strategy, number) for task in tasks for number in range(1, repeat + 1)
            for strategy in strategies]


def _responses_spy(responses: Any, usage: list[dict[str, int]]):
    class ResponsesSpy:
        async def create(self, **kwargs):
            response = await responses.create(**kwargs)
            if not kwargs.get("stream"):
                if counts := response_usage(response):
                    usage.append(counts)
            return response

    return ResponsesSpy()


def _client_spy(client: Any, score_usage: list[dict[str, int]]):
    class ClientSpy:
        def __init__(self) -> None:
            self.responses = _responses_spy(client.responses, score_usage)

        async def close(self) -> None:
            close = getattr(client, "close", None)
            if close is not None:
                result = close()
                if asyncio.iscoroutine(result):
                    await result

    return ClientSpy()


class ObservedRoute(FailoverRoute):
    """Add measurements and the evaluation-only fast/strong tier ablation."""

    def __init__(self, base: FailoverRoute, *, tier_only: bool) -> None:
        self.tier_only = tier_only
        self.score_seconds = 0.0
        self.score_usage: list[dict[str, int]] = []
        targets = tuple(RouteTarget(
            provider_id=target.provider_id,
            model=target.model,
            credential=target.credential,
            client=_client_spy(target.client, self.score_usage),
            context_limit=target.context_limit,
            capability=target.capability,
        ) for target in base.targets)
        super().__init__(targets, intent_routing=True)

    async def start(self, task: str, messages: list[dict[str, Any]], estimated_tokens: int) -> None:
        started = time.perf_counter()
        try:
            await super().start(task, messages, estimated_tokens)
        finally:
            self.score_seconds += time.perf_counter() - started
        if not self.tier_only or self.selection is None or self.selection.error:
            return
        tier = "strong" if self.selection.intent in {"debugging", "reasoning"} else "fast"
        eligible = self._active
        chosen = next((target for target in self.targets if target in eligible
                       and target.capability and tier in target.capability.strengths), None)
        if chosen is not None:
            self._active = (chosen, *(target for target in eligible if target is not chosen))
            self.index = 0


async def _close(client: Any) -> None:
    if client is None:
        return
    with suppress(Exception):
        result = client.close()
        if asyncio.iscoroutine(result):
            await result


def _split_ref(ref: str) -> tuple[str, str]:
    provider, separator, model = ref.partition("/")
    if not separator or not provider or not model:
        raise ValueError(f"model must be provider/model, got {ref!r}")
    return provider, model


class AgentRunner:
    """Execute one real agent turn and retain machine-readable routing metrics."""

    def __init__(
        self, credentials: dict[str, Credential], listed_models: list[str], *,
        fast_model: str, strong_model: str, max_turns: int | None = None,
    ) -> None:
        self.current: AgentResult | None = None
        self.credentials = credentials
        self.listed_models = listed_models
        self.fast_model = fast_model
        self.strong_model = strong_model
        self.max_turns = max_turns

    def begin(self, strategy: str) -> AgentResult:
        """Track progress from the start, so a cancelled run still reports it."""
        result = AgentResult()
        self.current = result
        return result

    async def run(self, task: Task, cwd: Path, strategy: str) -> AgentResult:
        result = self.current or self.begin(strategy)
        route: ObservedRoute | None = None
        client: Any = None
        try:
            if strategy in AUTO_STRATEGIES:
                base = build_auto_route(
                    self.credentials, models=self.listed_models,
                    context_limit=resolve_context_limit,
                )
                route = ObservedRoute(base, tier_only=strategy == "auto-tier")
                client = route.current.client
                model, provider_id = "auto", route.current.provider_id
                context_limit = None
            else:
                ref = self.fast_model if strategy == "fixed-fast" else self.strong_model
                provider_id, model = _split_ref(ref)
                provider = get_provider(provider_id)
                credential = self.credentials.get(provider_id)
                if provider is None or credential is None:
                    raise ValueError(f"no credentials for {ref}")
                client = provider.make_client(credential)
                context_limit = resolve_context_limit(model)
                result.selected_model = result.initial_model = ref

            def on_usage(usage: dict[str, int]) -> None:
                result.prompt_tokens += usage.get("prompt_tokens", 0)
                result.completion_tokens += usage.get("completion_tokens", 0)
                result.total_tokens += usage.get("total_tokens", 0)
                result.events.append({"event": "usage", **usage})

            def on_route(target: RouteTarget, previous: RouteTarget | None, error) -> None:
                result.selected_model = target.ref
                if previous is None:
                    result.initial_model = target.ref
                else:
                    result.failovers += 1
                result.events.append({
                    "event": "route", "target": target.ref,
                    "previous": previous.ref if previous else None,
                    "error": str(error) if error else None,
                })

            def on_tool(turn: int, name: str, args: dict, output: str) -> None:
                result.tool_rounds = max(result.tool_rounds, turn)
                result.tool_calls += 1
                result.events.append({"event": "tool", "turn": turn, "name": name})

            def on_model_start(turn: int) -> None:
                result.model_rounds = max(result.model_rounds, turn)

            def on_compact(count: int) -> None:
                result.compactions += 1
                result.events.append({"event": "compact", "messages": count})

            result.answer = await run_agent(
                task.prompt, cwd=cwd, client=client, model=model,
                max_turns=self.max_turns or task.max_turns,
                messages=[{"role": "system", "content": SYSTEM}],
                provider_id=provider_id, context_limit=context_limit,
                route=route, on_route=on_route if route else None,
                on_usage=on_usage, on_tool=on_tool, on_model_start=on_model_start,
                on_compact=on_compact,
            )
        except Exception as error:  # the outer runner classifies and records it
            result.error = error
        finally:
            if route is not None:
                result.score_seconds = route.score_seconds
                result.score_prompt_tokens = sum(
                    usage.get("prompt_tokens", 0) for usage in route.score_usage
                )
                result.score_completion_tokens = sum(
                    usage.get("completion_tokens", 0) for usage in route.score_usage
                )
                if route.selection is not None:
                    selection = route.selection
                    result.intent = selection.intent
                    result.scores = dict(selection.scores)
                    result.scorer = selection.scorer_ref
                    result.score_error = selection.error
                if not result.selected_model:
                    result.selected_model = route.current.ref
                await route.close()
            else:
                await _close(client)
        return result


def _record_workdir(artifact: Path, cwd: Path, baseline: str, outcome: AgentResult) -> None:
    """Persist what the agent did even when the run is cancelled or crashes."""
    (artifact / "answer.txt").write_text(outcome.answer, encoding="utf-8")
    (artifact / "events.jsonl").write_text(
        "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in outcome.events),
        encoding="utf-8",
    )
    if not baseline or not cwd.is_dir():
        return
    _run(["git", "add", "-N", "."], cwd)
    diff = _run(["git", "diff", "--binary", baseline], cwd, check=False).stdout
    (artifact / "changes.patch").write_text(diff, encoding="utf-8")


def _infrastructure_error(error: BaseException) -> bool:
    status = getattr(error, "status_code", None)
    return status in _INFRA_STATUSES or type(error).__name__ in _INFRA_ERRORS


def _run(command: list[str], cwd: Path, *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        check=check,
    )


def _git_baseline(cwd: Path) -> str:
    _run(["git", "init", "-q"], cwd)
    _run(["git", "add", "."], cwd)
    _run([
        "git", "-c", "user.name=agent7-eval", "-c", "user.email=eval@localhost",
        "commit", "-q", "-m", "evaluation baseline",
    ], cwd)
    return _run(["git", "rev-parse", "HEAD"], cwd).stdout.strip()


def _project_revision() -> str:
    try:
        head = _run(["git", "rev-parse", "HEAD"], PROJECT_ROOT).stdout.strip()
        dirty = bool(_run(["git", "status", "--porcelain"], PROJECT_ROOT).stdout.strip())
        return head + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _shell(command: str, cwd: Path, timeout: int, env: dict[str, str] | None = None):
    return subprocess.run(
        command, cwd=cwd, shell=True, text=True, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout,
    )


async def _with_timeout(runner: Any, task: Task, cwd: Path, strategy: str,
                        timeout: int) -> AgentResult:
    started = runner.begin(strategy) if hasattr(runner, "begin") else None
    pending_task = asyncio.create_task(runner.run(task, cwd, strategy))
    try:
        done, _ = await asyncio.wait({pending_task}, timeout=timeout)
    except asyncio.CancelledError:
        pending_task.cancel()
        with suppress(BaseException):
            await pending_task
        raise
    if done:
        return pending_task.result()
    pending_task.cancel()
    with suppress(BaseException):
        await pending_task
    result = started or AgentResult()
    result.error = TimeoutError(f"agent exceeded {timeout}s")
    return result


async def execute_run(
    task: Task, strategy: str, repeat: int, output: Path, runner: Any, *,
    timeout: int | None = None,
) -> dict[str, Any]:
    """Run one isolated task; hidden verification arrives only after the agent stops."""
    run_id = f"{task.id}-{strategy}-r{repeat}-{uuid.uuid4().hex[:8]}"
    artifact = output / "artifacts" / run_id
    artifact.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    started = time.perf_counter()
    outcome = AgentResult()
    status = "runner_error"
    verify_exit: int | None = None
    baseline = ""
    workdir_recorded = False
    with tempfile.TemporaryDirectory(prefix="agent7-eval-") as temporary:
        cwd = Path(temporary) / "repo"
        try:
            shutil.copytree(EVAL_ROOT / "repos" / task.repo, cwd)
            if task.setup:
                setup = _shell(task.setup.format(python=shlex.quote(sys.executable)),
                               cwd, task.timeout)
                (artifact / "setup.txt").write_text(setup.stdout, encoding="utf-8")
                if setup.returncode:
                    raise RuntimeError(f"setup exited {setup.returncode}")
            baseline = _git_baseline(cwd)
            outcome = await _with_timeout(runner, task, cwd, strategy, timeout or task.timeout)
            _record_workdir(artifact, cwd, baseline, outcome)
            workdir_recorded = True
            if outcome.error is not None:
                if isinstance(outcome.error, TimeoutError):
                    status = "timeout"
                elif _infrastructure_error(outcome.error):
                    status = "infrastructure_error"
                else:
                    status = "agent_error"
            else:
                shutil.copy2(EVAL_ROOT / "verify.py", cwd / "_agent7_eval_verify.py")
                env = dict(os.environ, AGENT7_EVAL_TASK=task.id, PYTHONDONTWRITEBYTECODE="1")
                try:
                    verified = _shell(
                        task.verify.format(python=shlex.quote(sys.executable)), cwd,
                        task.verify_timeout, env,
                    )
                    verify_exit = verified.returncode
                    (artifact / "verify.txt").write_text(verified.stdout, encoding="utf-8")
                    status = "passed" if verified.returncode == 0 else "failed"
                except subprocess.TimeoutExpired as error:
                    (artifact / "verify.txt").write_text(str(error), encoding="utf-8")
                    status = "verify_error"
        except asyncio.CancelledError as error:
            outcome = getattr(runner, "current", None) or outcome
            outcome.error = error
            status = "interrupted"
            raise
        except Exception as error:
            outcome.error = error
            status = "runner_error"
        finally:
            if not workdir_recorded:
                with suppress(Exception):
                    _record_workdir(artifact, cwd, baseline, outcome)

    error = outcome.error
    record = {
        "run_id": run_id,
        "task_id": task.id,
        "category": task.category,
        "language": task.language,
        "strategy": strategy,
        "repeat": repeat,
        "started_at": started_at,
        "wall_seconds": round(time.perf_counter() - started, 3),
        "status": status,
        "success": status == "passed",
        "valid": status not in INVALID_STATUSES,
        "verify_exit": verify_exit,
        "selected_model": outcome.selected_model,
        "initial_model": outcome.initial_model,
        "intent": outcome.intent,
        "scores": outcome.scores,
        "scorer": outcome.scorer,
        "score_error": outcome.score_error,
        "score_seconds": round(outcome.score_seconds, 3),
        "score_prompt_tokens": outcome.score_prompt_tokens,
        "score_completion_tokens": outcome.score_completion_tokens,
        "prompt_tokens": outcome.prompt_tokens,
        "completion_tokens": outcome.completion_tokens,
        "total_tokens": outcome.total_tokens,
        "model_rounds": outcome.model_rounds,
        "tool_rounds": outcome.tool_rounds,
        "tool_calls": outcome.tool_calls,
        "compactions": outcome.compactions,
        "failovers": outcome.failovers,
        "error_type": type(error).__name__ if error else "",
        "error": str(error)[:2000] if error else "",
        "artifact": str(artifact.relative_to(output)),
        "agent7_revision": _project_revision(),
        "python": sys.version.split()[0],
    }
    (artifact / "result.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return record


async def _listed_models(credentials: dict[str, Credential]) -> list[str]:
    refs: list[str] = []
    for provider_id, credential in credentials.items():
        provider = get_provider(provider_id)
        if provider is None:
            continue
        try:
            models = await provider.list_models(credential)
        except Exception as error:
            print(f"warning: {provider_id}/models failed: {error}", file=sys.stderr)
        else:
            refs.extend(f"{provider_id}/{model}" for model in models)
    return sorted(set(refs))


def _parse_strategies(raw: str) -> list[str]:
    strategies = list(dict.fromkeys(item.strip() for item in raw.split(",") if item.strip()))
    unknown = set(strategies) - set(STRATEGIES)
    if unknown or not strategies:
        raise ValueError(f"choose strategies from {', '.join(STRATEGIES)}")
    return strategies


def _preflight(runner: AgentRunner, strategies: Iterable[str]) -> None:
    for strategy in strategies:
        if strategy in AUTO_STRATEGIES:
            continue
        ref = runner.fast_model if strategy == "fixed-fast" else runner.strong_model
        provider_id, _ = _split_ref(ref)
        if get_provider(provider_id) is None or provider_id not in runner.credentials:
            raise ValueError(f"{strategy} needs credentials for {ref}")


async def _amain(args: argparse.Namespace) -> int:
    tasks = load_tasks(args.manifest)
    if args.task:
        wanted = {item for raw in args.task for item in raw.split(",") if item}
        unknown = wanted - {task.id for task in tasks}
        if unknown:
            raise ValueError(f"unknown tasks: {sorted(unknown)}")
        tasks = [task for task in tasks if task.id in wanted]
    strategies = _parse_strategies(args.strategies)
    plan = build_plan(tasks, strategies, args.repeat)
    if args.dry_run:
        print(f"{len(plan)} runs ({len(tasks)} tasks × {len(strategies)} strategies × {args.repeat})")
        for task, strategy, repeat in plan:
            print(f"{task.id:30} {task.category:10} {strategy:13} r{repeat}")
        return 0

    credentials = resolve_all()
    if not credentials:
        raise ValueError("no provider credentials; set an API key or use agent7 /login")
    listed = await _listed_models(credentials) if set(strategies) & AUTO_STRATEGIES else []
    runner = AgentRunner(
        credentials, listed, fast_model=args.fast_model, strong_model=args.strong_model,
        max_turns=args.max_turns,
    )
    _preflight(runner, strategies)
    if set(strategies) & AUTO_STRATEGIES:
        probe = build_auto_route(credentials, models=listed, context_limit=resolve_context_limit)
        await probe.close()

    output = args.output or EVAL_ROOT / "results" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    config = {
        "started_at": utc_now(), "strategies": strategies, "repeat": args.repeat,
        "tasks": [asdict(task) for task in tasks], "fast_model": args.fast_model,
        "strong_model": args.strong_model, "listed_models": listed,
        "providers": sorted(credentials), "max_turns": args.max_turns,
        "timeout": args.timeout, "agent7_revision": _project_revision(),
    }
    (output / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    records: list[dict[str, Any]] = []
    results = output / "results.jsonl"
    interrupted = False
    try:
        with results.open("a", encoding="utf-8") as stream:
            for index, (task, strategy, repeat) in enumerate(plan, 1):
                print(f"[{index}/{len(plan)}] {task.id} {strategy} r{repeat}", flush=True)
                record = await execute_run(
                    task, strategy, repeat, output, runner, timeout=args.timeout,
                )
                records.append(record)
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                print(f"  {record['status']} {record['selected_model']} "
                      f"{record['wall_seconds']:.1f}s", flush=True)
    except (KeyboardInterrupt, asyncio.CancelledError):
        interrupted = True
        print("\ninterrupted; keeping the completed runs", file=sys.stderr)
    finally:
        if records:
            (output / "report.md").write_text(render_report(records), encoding="utf-8")
    print(f"results: {results}\nreport:  {output / 'report.md'}")
    return 130 if interrupted else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--task", action="append", help="task id or comma-separated ids")
    parser.add_argument("--strategies", default=",".join(STRATEGIES))
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--fast-model", default="deepseek/deepseek-flash")
    parser.add_argument("--strong-model", default="xai/grok-4.7")
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        return asyncio.run(_amain(args))
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
