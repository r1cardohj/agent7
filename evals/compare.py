"""Head-to-head benchmark: agent7 vs external coding-agent CLIs.

Every run gets a fresh copy of the same fixture and the same hidden verifier.
Metrics are normalized to cumulative per-model-call tokens, agent wall time,
and independent pass/fail completion.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from contextlib import suppress
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent7 import __version__
from agent7.providers import Credential, get_provider, resolve_all

from .report import render_comparison_report
from .run import (
    DEFAULT_MANIFEST,
    EVAL_ROOT,
    AgentResult,
    AgentRunner,
    Task,
    _project_revision,
    execute_run,
    load_tasks,
)

AGENTS = ("agent7", "pi", "kimi")
BENCH_PROVIDER = "kimi-bench"
BENCH_MODEL = "k3-256k"
BENCH_BASE_URL = "https://api.kimi.com/coding/v1"
DEFAULT_MODELS = {
    "agent7": f"kimi/{BENCH_MODEL}",
    "pi": f"{BENCH_PROVIDER}/{BENCH_MODEL}",
    "kimi": f"{BENCH_PROVIDER}/{BENCH_MODEL}",
}
BENCH_API_KEY_ENV = "AGENT7_KIMI_API_KEY"
DEFAULT_OUTPUT = EVAL_ROOT / "results" / "compare"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_agents(raw: str) -> list[str]:
    agents = list(dict.fromkeys(item.strip() for item in raw.split(",") if item.strip()))
    unknown = set(agents) - set(AGENTS)
    if unknown or not agents:
        raise ValueError(f"choose agents from {', '.join(AGENTS)}")
    return agents


def build_agent_plan(
    tasks: list[Task], agents: list[str], repeat: int
) -> list[tuple[Task, str, int]]:
    """Interleave agents within each task and rotate first-agent order."""
    if repeat < 1:
        raise ValueError("repeat must be at least one")
    plan: list[tuple[Task, str, int]] = []
    for task_index, task in enumerate(tasks):
        rotated = agents[task_index % len(agents) :] + agents[: task_index % len(agents)]
        for number in range(1, repeat + 1):
            plan.extend((task, agent, number) for agent in rotated)
    return plan


def _json_objects(text: str, *, strict: bool) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            if strict:
                raise ValueError(f"invalid JSON line {number}: {error}") from error
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


class ExternalProviderError(RuntimeError):
    def __init__(self, message: str, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(message)


def _provider_error(message: str) -> RuntimeError:
    """Keep provider/network failures separate from task-completion failures."""
    lowered = message.lower()
    if any(
        marker in lowered
        for marker in (
            "overloaded",
            "temporarily unavailable",
            "connection error",
            "connection_error",
            "connection failed",
            "network error",
            "timed out",
        )
    ):
        return ExternalProviderError(message, 503)
    if "rate limit" in lowered or "too many requests" in lowered:
        return ExternalProviderError(message, 429)
    if "oauth" in lowered or "unauthorized" in lowered or "authentication" in lowered:
        return ExternalProviderError(message, 401)
    match = re.search(r"(?:api error|status) \((\d{3})\)|\bstatus (\d{3})\b", message)
    if match:
        return ExternalProviderError(message, int(match.group(1) or match.group(2)))
    return RuntimeError(message)


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(
        part.get("text", "")
        for part in content
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    )


def _add_usage(
    result: AgentResult,
    usage: dict[str, Any],
    *,
    source: str,
) -> None:
    input_other = int(usage.get("inputOther") or usage.get("input") or 0)
    output = int(usage.get("output") or usage.get("output_tokens") or 0)
    cache_read = int(usage.get("inputCacheRead") or usage.get("cacheRead") or 0)
    cache_write = int(usage.get("inputCacheCreation") or usage.get("cacheWrite") or 0)
    prompt = input_other + cache_read + cache_write
    total_value = usage.get("totalTokens") or usage.get("total_tokens")
    total = int(total_value) if total_value is not None else prompt + output
    if not any((input_other, output, cache_read, cache_write, total)):
        # Error responses often carry a zero-valued usage placeholder.
        return
    result.prompt_tokens += prompt
    result.completion_tokens += output
    result.total_tokens += total
    result.cache_read_tokens += cache_read
    result.cache_write_tokens += cache_write
    result.usage_observed = True
    result.token_source = source
    result.events.append(
        {
            "event": "usage",
            "source": source,
            "prompt_tokens": prompt,
            "completion_tokens": output,
            "total_tokens": total,
            "cache_read_tokens": cache_read,
            "cache_write_tokens": cache_write,
        }
    )


def parse_pi_json(result: AgentResult, stdout: str) -> None:
    """Normalize one Pi JSON-mode invocation, including compaction usage."""
    result.token_source = "missing"
    answers: list[str] = []
    terminal_errors: list[RuntimeError] = []
    records = _json_objects(stdout, strict=True)
    for event in records:
        kind = event.get("type")
        if kind == "message_end":
            message = event.get("message") or {}
            if message.get("role") == "assistant":
                result.model_rounds += 1
                if (usage := message.get("usage")) and isinstance(usage, dict):
                    _add_usage(result, usage, source="pi-message")
                stop = message.get("stopReason")
                if stop in {"error", "aborted", "length"}:
                    detail = message.get("errorMessage") or stop
                    terminal_errors.append(_provider_error(f"Pi response ended with {detail}"))
                text = _message_text(message)
                if text:
                    answers.append(text)
            elif message.get("role") == "toolResult":
                # Nested model work contributes to a full-session total.
                if (usage := message.get("usage")) and isinstance(usage, dict):
                    _add_usage(result, usage, source="pi-nested")
        elif kind == "tool_execution_end":
            result.tool_calls += 1
        elif kind == "turn_end":
            result.tool_rounds += 1
        elif kind == "compaction_end" and event.get("result"):
            result.compactions += 1
            usage = event["result"].get("usage")
            if isinstance(usage, dict):
                _add_usage(result, usage, source="pi-compaction")
        elif kind == "agent_settled" and event.get("aborted"):
            terminal_errors.append(RuntimeError("Pi run was aborted"))
    if answers:
        result.answer = answers[-1]
    if terminal_errors:
        result.error = next(
            (error for error in terminal_errors if getattr(error, "status_code", None)),
            terminal_errors[-1],
        )


async def _run_external(
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    result: AgentResult,
) -> int:
    """Run one CLI, always retaining partial output for cancellation."""
    with tempfile.TemporaryDirectory(prefix=f"agent7-{result.token_source or 'cli'}-") as temporary:
        stdout_path = Path(temporary) / "stdout"
        stderr_path = Path(temporary) / "stderr"
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            try:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=cwd,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=(os.name == "posix"),
                )
            except OSError as error:
                result.raw_error = str(error)
                raise RuntimeError(f"cannot start {command[0]}: {error}") from error
            try:
                return_code = await process.wait()
            except asyncio.CancelledError:
                with suppress(ProcessLookupError, PermissionError, OSError):
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                with suppress(asyncio.CancelledError):
                    await process.wait()
                raise
            finally:
                result.raw_output = stdout_path.read_text(errors="replace")
                result.raw_error = stderr_path.read_text(errors="replace")
            return return_code


class Agent7Runner:
    """Run agent7 through the production in-process loop on one fixed model."""

    def __init__(self, runner: AgentRunner, model_ref: str) -> None:
        self.runner = runner
        self.model_ref = model_ref
        self.current: AgentResult | None = None

    def begin(self, strategy: str) -> AgentResult:
        self.current = self.runner.begin(strategy)
        return self.current

    async def run(self, task: Task, cwd: Path, _strategy: str) -> AgentResult:
        # AgentRunner maps fixed-fast/fixed-strong to configured model refs. The
        # comparison runner passes both through the same fast ref, so strategy
        # is only the display name supplied by execute_run.
        return await self.runner.run(task, cwd, "fixed-fast")


def make_pi_models(
    provider: str,
    model: str,
    base_url: str,
    api: str = "openai-completions",
) -> str:
    """Pi models.json for the same Kimi Coding API used by agent7."""
    return (
        json.dumps(
            {
                "$schema": "https://pi.dev/schemas/models.schema.json",
                "providers": {
                    provider: {
                        "baseUrl": base_url,
                        "api": api,
                        "apiKey": f"${BENCH_API_KEY_ENV}",
                        "compat": {"supportsDeveloperRole": False},
                        "models": [
                            {
                                "id": model,
                                "name": f"Kimi {model}",
                                "reasoning": True,
                                "input": ["text"],
                                "contextWindow": 262144,
                                "maxTokens": 65536,
                            }
                        ],
                    },
                },
            },
            indent=2,
        )
        + "\n"
    )


def make_kimi_config(provider: str, model: str, base_url: str) -> str:
    """Minimal isolated Kimi config backed by the benchmark API key."""
    quoted_base = json.dumps(base_url)
    return f'''default_model = "{provider}/{model}"

[providers.{provider}]
type = "kimi"
base_url = {quoted_base}
api_key_env = "{BENCH_API_KEY_ENV}"

[models."{provider}/{model}"]
provider = "{provider}"
model = "{model}"
max_context_size = 262144
support_efforts = ["low", "high", "max"]
default_effort = "high"
'''


class PiRunner:
    def __init__(
        self,
        executable: str,
        model: str,
        *,
        thinking: str,
        provider: str,
        model_id: str,
        base_url: str,
        api_key: str,
        api: str,
    ) -> None:
        self.executable = executable
        self.model = model
        self.thinking = thinking
        self.provider = provider
        self.model_id = model_id
        self.base_url = base_url
        self.api_key = api_key
        self.api = api
        self.current: AgentResult | None = None

    def begin(self, _strategy: str) -> AgentResult:
        self.current = AgentResult()
        return self.current

    def command(self, task: Task) -> list[str]:
        return [
            self.executable,
            "--mode",
            "json",
            "--no-session",
            "--no-extensions",
            "--no-mcp",
            "--no-skills",
            "--no-prompt-templates",
            "--no-context-files",
            "--model",
            self.model,
            "--thinking",
            self.thinking,
            "--",
            task.prompt,
        ]

    async def run(self, task: Task, cwd: Path, strategy: str) -> AgentResult:
        result = self.current or self.begin(strategy)
        result.selected_model = result.initial_model = self.model
        try:
            with tempfile.TemporaryDirectory(prefix="agent7-pi-home-") as temporary:
                agent_dir = Path(temporary) / "agent"
                agent_dir.mkdir()
                (agent_dir / "models.json").write_text(
                    make_pi_models(
                        self.provider,
                        self.model_id,
                        self.base_url,
                        self.api,
                    ),
                    encoding="utf-8",
                )
                env = dict(
                    os.environ,
                    PI_CODING_AGENT_DIR=str(agent_dir),
                    **{BENCH_API_KEY_ENV: self.api_key},
                )
                return_code = await _run_external(
                    self.command(task),
                    cwd,
                    env,
                    result,
                )
            parse_pi_json(result, result.raw_output)
            if result.error is not None:
                raise result.error
            if return_code:
                raise _provider_error(
                    f"pi exited {return_code}: {result.raw_error.strip()[-1000:]}"
                )
        except Exception as error:
            result.error = error
        return result


class KimiRunner:
    def __init__(
        self,
        executable: str,
        model: str,
        *,
        provider: str,
        model_id: str,
        base_url: str,
        api_key: str,
    ) -> None:
        self.executable = executable
        self.model = model
        self.provider = provider
        self.model_id = model_id
        self.base_url = base_url
        self.api_key = api_key
        self.current: AgentResult | None = None

    def begin(self, _strategy: str) -> AgentResult:
        self.current = AgentResult()
        return self.current

    def command(self, task: Task) -> list[str]:
        return [
            self.executable,
            "--model",
            self.model,
            "--prompt",
            task.prompt,
            "--output-format",
            "stream-json",
        ]

    async def run(self, task: Task, cwd: Path, strategy: str) -> AgentResult:
        result = self.current or self.begin(strategy)
        result.selected_model = result.initial_model = self.model
        try:
            with tempfile.TemporaryDirectory(prefix="agent7-kimi-home-") as temporary:
                home = Path(temporary) / "home"
                home.mkdir()
                (home / "config.toml").write_text(
                    make_kimi_config(self.provider, self.model_id, self.base_url),
                    encoding="utf-8",
                )
                skills = home / "empty-skills"
                skills.mkdir()
                env = dict(
                    os.environ,
                    KIMI_CODE_HOME=str(home),
                    **{BENCH_API_KEY_ENV: self.api_key},
                )
                return_code = await _run_external(
                    [*self.command(task), "--skills-dir", str(skills)],
                    cwd,
                    env,
                    result,
                )
                parse_kimi_output(result, result.raw_output, home)
                if result.error is not None:
                    raise result.error
                if return_code:
                    raise _provider_error(
                        f"kimi exited {return_code}: {result.raw_error.strip()[-1000:]}"
                    )
        except Exception as error:
            result.error = error
        return result


def _wire_usages(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    usages = [
        obj["usage"]
        for obj in records
        if obj.get("type") == "usage.record" and isinstance(obj.get("usage"), dict)
    ]
    if usages:
        return usages
    return [
        event["usage"]
        for obj in records
        if obj.get("type") == "context.append_loop_event"
        and isinstance((event := obj.get("event")), dict)
        and event.get("type") == "step.end"
        and isinstance(event.get("usage"), dict)
    ]


def parse_kimi_output(result: AgentResult, stdout: str, home: Path) -> None:
    """Parse Kimi stream-json and the fresh isolated session journal."""
    result.token_source = "missing"
    answers: list[str] = []
    stream_tool_ids: set[str] = set()
    stream_rounds = 0
    for event in _json_objects(stdout, strict=True):
        role = event.get("role")
        if role == "assistant":
            stream_rounds += 1
            text = _message_text(event)
            if text:
                answers.append(text)
            for call in event.get("tool_calls") or []:
                if isinstance(call, dict):
                    stream_tool_ids.add(str(call.get("id", "")))
        elif role == "tool":
            if call_id := event.get("tool_call_id"):
                stream_tool_ids.add(str(call_id))
        elif role == "error":
            result.error = _provider_error(str(event.get("content") or event))

    wire_rounds = 0
    wire_tools = 0
    wire_usages: list[dict[str, Any]] = []
    for wire_path in sorted(home.glob("sessions/*/*/agents/*/wire.jsonl")):
        try:
            records = _json_objects(wire_path.read_text(errors="replace"), strict=True)
        except OSError:
            continue
        agent_name = wire_path.parent.name
        wire_rounds += sum(1 for obj in records if obj.get("type") == "llm.request")
        for usage in _wire_usages(records):
            wire_usages.append(usage)
            _add_usage(result, usage, source=f"kimi-wire:{agent_name}")
        for obj in records:
            event = obj.get("event")
            if (
                obj.get("type") == "context.append_loop_event"
                and isinstance(event, dict)
                and event.get("type") == "tool.call"
            ):
                wire_tools += 1

    result.model_rounds = max(result.model_rounds, wire_rounds or stream_rounds)
    result.tool_calls = max(result.tool_calls, wire_tools or len(stream_tool_ids))
    if answers:
        result.answer = answers[-1]
    if result.usage_observed:
        result.token_source = "kimi-wire-all-agents"


def _split_model_ref(ref: str) -> tuple[str, str]:
    provider, separator, model = ref.partition("/")
    if not separator or not provider or not model:
        raise ValueError(f"model must be provider/model, got {ref!r}")
    return provider, model


def _preflight(
    agents: list[str],
    *,
    agent7_model: str,
    pi_executable: str,
    kimi_executable: str,
) -> dict[str, Credential]:
    credentials = resolve_all()
    if "agent7" in agents:
        provider_id, _ = _split_model_ref(agent7_model)
        provider = get_provider(provider_id)
        if provider is None or provider_id not in credentials:
            raise ValueError(f"agent7 needs credentials for {agent7_model}")
    # Pi and Kimi Code are deliberately provisioned from the same Kimi Coding
    # API key as agent7. Using their local OAuth stores would make benchmark
    # runs depend on interactive login state and could rotate refresh tokens.
    if ({"pi", "kimi"} & set(agents)) and "kimi" not in credentials:
        raise ValueError("pi/kimi comparison runs need the agent7 kimi credential")
    if "pi" in agents and shutil.which(pi_executable) is None:
        raise ValueError(f"pi executable not found: {pi_executable}")
    if "kimi" in agents and shutil.which(kimi_executable) is None:
        raise ValueError(f"kimi executable not found: {kimi_executable}")
    return credentials


def _cli_version(executable: str) -> str:
    try:
        completed = subprocess.run(
            [executable, "--version"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    return completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else "unknown"


async def _amain(args: argparse.Namespace) -> int:
    tasks = load_tasks(args.manifest)
    if args.task:
        wanted = {item for raw in args.task for item in raw.split(",") if item}
        unknown = wanted - {task.id for task in tasks}
        if unknown:
            raise ValueError(f"unknown tasks: {sorted(unknown)}")
        tasks = [task for task in tasks if task.id in wanted]
    agents = parse_agents(args.agents)
    plan = build_agent_plan(tasks, agents, args.repeat)
    if args.dry_run:
        print(f"{len(plan)} runs ({len(tasks)} tasks × {len(agents)} agents × {args.repeat})")
        for task, agent, repeat in plan:
            print(f"{task.id:30} {task.category:10} {agent:7} r{repeat}")
        return 0

    credentials = _preflight(
        agents,
        agent7_model=args.agent7_model,
        pi_executable=args.pi_executable,
        kimi_executable=args.kimi_executable,
    )
    bench_api_key = credentials["kimi"].api_key

    agent7_delegate: AgentRunner | None = None
    if "agent7" in agents:
        agent7_delegate = AgentRunner(
            credentials,
            [],
            fast_model=args.agent7_model,
            strong_model=args.agent7_model,
            max_turns=args.agent7_max_turns,
        )
    runners: dict[str, Any] = {}
    if agent7_delegate is not None:
        runners["agent7"] = Agent7Runner(agent7_delegate, args.agent7_model)
    if "pi" in agents:
        pi_provider, pi_model_id = _split_model_ref(args.pi_model)
        runners["pi"] = PiRunner(
            args.pi_executable,
            args.pi_model,
            thinking=args.pi_thinking,
            provider=pi_provider,
            model_id=pi_model_id,
            base_url=args.bench_base_url,
            api_key=bench_api_key,
            api=args.pi_api,
        )
    if "kimi" in agents:
        kimi_provider, kimi_model_id = _split_model_ref(args.kimi_model)
        runners["kimi"] = KimiRunner(
            args.kimi_executable,
            args.kimi_model,
            provider=kimi_provider,
            model_id=kimi_model_id,
            base_url=args.bench_base_url,
            api_key=bench_api_key,
        )

    output = args.output or DEFAULT_OUTPUT.with_name(
        DEFAULT_OUTPUT.name + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    output.mkdir(parents=True, exist_ok=False)
    agent_versions = {
        "agent7": __version__,
        "pi": _cli_version(args.pi_executable) if "pi" in agents else None,
        "kimi": _cli_version(args.kimi_executable) if "kimi" in agents else None,
    }
    config = {
        "started_at": utc_now(),
        "agents": agents,
        "repeat": args.repeat,
        "models": {
            "agent7": args.agent7_model if "agent7" in agents else None,
            "pi": args.pi_model if "pi" in agents else None,
            "kimi": args.kimi_model if "kimi" in agents else None,
        },
        "pi_thinking": args.pi_thinking if "pi" in agents else None,
        "pi_api": args.pi_api if "pi" in agents else None,
        "bench_base_url": args.bench_base_url,
        "bench_api_key_env": BENCH_API_KEY_ENV,
        "agent_versions": agent_versions,
        "tasks": [asdict(task) for task in tasks],
        "agent7_max_turns": args.agent7_max_turns,
        "timeout": args.timeout,
        "agent7_revision": _project_revision(),
        "token_semantics": (
            "cumulative model-call tokens; Kimi includes all session agents; "
            "Pi includes message, nested, and compaction usage"
        ),
    }
    (output / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    records: list[dict[str, Any]] = []
    results = output / "results.jsonl"
    interrupted = False
    try:
        with results.open("a", encoding="utf-8") as stream:
            for index, (task, agent, repeat) in enumerate(plan, 1):
                print(f"[{index}/{len(plan)}] {task.id} {agent} r{repeat}", flush=True)
                record = await execute_run(
                    task,
                    agent,
                    repeat,
                    output,
                    runners[agent],
                    timeout=args.timeout,
                )
                records.append(record)
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                tokens = f"{record['total_tokens']:,}" if record["usage_observed"] else "n/a"
                print(
                    f"  {record['status']} {record['agent_seconds']:.1f}s {tokens} tokens",
                    flush=True,
                )
    except (KeyboardInterrupt, asyncio.CancelledError):
        interrupted = True
        print("\ninterrupted; keeping the completed runs", file=sys.stderr)
    finally:
        if records:
            report = render_comparison_report(records, config=config)
            (output / "report.md").write_text(report, encoding="utf-8")
    print(f"results: {results}")
    if records:
        print(f"report:  {output / 'report.md'}")
    return 130 if interrupted else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--task", action="append", help="task id or comma-separated ids")
    parser.add_argument("--agents", default=",".join(AGENTS))
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--agent7-model", default=DEFAULT_MODELS["agent7"])
    parser.add_argument("--pi-model", default=DEFAULT_MODELS["pi"])
    parser.add_argument("--kimi-model", default=DEFAULT_MODELS["kimi"])
    parser.add_argument("--pi-thinking", default="high")
    parser.add_argument(
        "--pi-api",
        choices=("openai-completions", "openai-responses"),
        default="openai-completions",
    )
    parser.add_argument("--pi-executable", default="pi")
    parser.add_argument("--kimi-executable", default="kimi")
    parser.add_argument("--bench-base-url", default=BENCH_BASE_URL)
    parser.add_argument("--agent7-max-turns", type=int)
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
