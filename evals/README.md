# Auto-routing evaluation

This directory measures **downstream task outcomes**, not whether a routing score looks
plausible. The same task is executed by fixed and auto policies in fresh, isolated workdirs;
verification code is copied in only after the agent finishes.

The harness is a development tool. It is not included in the agent7 wheel, and CI never makes
these live API calls.

## Quick start

Inspect the default plan without credentials or API usage:

```bash
uv run python -m evals.run --dry-run
```

The default is 12 tasks × 4 strategies × 2 repeats = **96 real agent runs**. Start with a
small pilot instead:

```bash
uv run python -m evals.run \
  --task quick-greeting,debug-async-close \
  --strategies fixed-fast,fixed-strong,auto-tier,auto-full \
  --repeat 1 \
  --output /tmp/agent7-eval-pilot
```

Run the complete benchmark only when the API spend is acceptable:

```bash
uv run python -m evals.run
```

Credentials come from the normal agent7 environment/login store. `git` and Python are required.
The default fixed refs can be changed with `--fast-model provider/model` and
`--strong-model provider/model`.

## Head-to-head comparison

`evals.compare` runs agent7, Pi, and Kimi Code CLI against the same fixture and hidden
verifier. The default is 12 tasks × 3 agents × 1 repeat = **36 real agent runs**. Inspect
it without credentials or API usage:

```bash
uv run python -m evals.compare --dry-run
```

Start with a four-category, 12-run pilot:

```bash
uv run python -m evals.compare \
  --task quick-greeting,coding-json-lines,debug-async-close,reasoning-config-precedence \
  --output /tmp/agent7-compare-pilot
```

Run the full suite only when the API spend is acceptable:

```bash
uv run python -m evals.compare
```

All three agents request the same K3-256k model from the Kimi Coding endpoint. Pi and
Kimi Code use an isolated `kimi-bench/k3-256k` provider backed by the agent7 Kimi API key,
rather than their local OAuth login state:

| Agent | Default model | Token source |
| --- | --- | --- |
| agent7 | `kimi/k3-256k` | OpenAI Responses usage events |
| Pi | `kimi-bench/k3-256k` | JSON-mode assistant usage, nested usage, and compaction usage |
| Kimi Code | `kimi-bench/k3-256k` | The fresh session journal, including every session agent |

Token totals are cumulative per model call, including repeated context, cache reads, and
reasoning output. They are an efficiency signal, not a provider price. Kimi Code's
`stream-json` stdout has no usage fields; parsing its `wire.jsonl` is required. Missing token
data is reported as `n/a`, not zero.

Agent time starts immediately before the agent loop/process and ends when it exits or is
interrupted. Fixture copying and hidden verification are excluded. Completion is binary
independent verification, not a model-written score or partial human rubric.

For reproducibility, Pi runs in an isolated agent directory with sessions, local extensions,
MCP, skills, prompt templates, and context-file discovery disabled. Kimi Code runs with a
fresh `KIMI_CODE_HOME`, a minimal generated config, and an empty skills directory; built-in
Kimi capabilities such as subagents remain available, and their tokens are included. The API
key is passed through `AGENT7_KIMI_API_KEY` and is never written to results or artifacts.
`--timeout` applies to every agent. `--agent7-max-turns` applies only to agent7; Pi and Kimi
retain their configured loop limits. Agent7 uses Responses; Pi and Kimi Code use their compatible
Chat Completions configurations (`--pi-api openai-responses` remains available for protocol
debugging). This is a fixed-endpoint harness comparison, not a proof that every generation
parameter is identical.

One repeat is a smoke measurement of this small suite. Repeat at least three times before
treating completion or token differences as stable.

## Strategies

| Name | Behavior |
| --- | --- |
| `fixed-fast` | Always execute with `--fast-model` (default `deepseek/deepseek-flash`). |
| `fixed-strong` | Always execute with `--strong-model` (default `kimi/k3.1-pro`). |
| `auto-tier` | Use the normal scorer and intent, then choose the table's first eligible fast model for quick/coding or strong model for debugging/reasoning. |
| `auto-full` | Use the production score ordering, first-task argmax, and subsequent-task hysteresis. |

`auto-tier` is the important ablation: if `auto-full` does not outperform it, per-model ranking
may not justify its complexity. The current suite uses a fresh route for every task; sequential
session stickiness needs a separate suite and is not inferred from these results.

Policies are interleaved within each task and repeat. Execution is intentionally serial to limit
rate-limit noise. Auto scoring latency and tokens are included in the run measurements.

## Results

Each output directory contains:

```text
config.json          # task definitions, available models, revision; never API keys
results.jsonl        # one flushed record per completed run
report.md            # aggregate and paired-bootstrap report
artifacts/<run-id>/
  result.json
  answer.txt
  events.jsonl       # route/tool/usage events, not prompts or credentials
  agent-output.jsonl # raw external CLI stdout when present; may contain prompts/code
  agent-stderr.txt   # raw external CLI stderr when present
  changes.patch      # diff against the pristine fixture, including new files
  verify.txt         # independent verifier output
```

Regenerate a report without rerunning agents:

```bash
uv run python -m evals.report /tmp/agent7-eval-pilot
# or write it somewhere else
uv run python -m evals.report /tmp/agent7-eval-pilot/results.jsonl -o /tmp/report.md
```

Statuses have distinct meanings:

- `passed` / `failed`: the agent completed and independent verification passed/failed.
- `agent_error` / `timeout`: a valid but unsuccessful task attempt.
- `infrastructure_error`: provider/auth/network failure; reported but excluded from success-rate
  denominators.
- `verify_error` / `runner_error`: harness failure; inspect artifacts before comparing policies.

The report shows raw counts as well as rates, so excluded failures cannot disappear silently.
Paired success deltas bootstrap **tasks**, not individual repeats; an interval spanning zero is
inconclusive. Raw input/output tokens are retained so cost reports can use dated provider prices.

## Task pack

`tasks.json` contains 12 standard-library tasks: three each in quick, coding, debugging, and
reasoning. They share `repos/pyapp`, but every run receives a fresh copy. `verify.py` contains the
hidden checks and is selected with `AGENT7_EVAL_TASK` only after execution.

A task manifest entry supports:

```json
{
  "id": "unique-task-id",
  "category": "coding",
  "language": "en",
  "repo": "pyapp",
  "prompt": "What the agent is asked to change",
  "setup": "optional trusted shell command",
  "verify": "{python} _agent7_eval_verify.py",
  "timeout": 240,
  "verify_timeout": 60,
  "max_turns": 40
}
```

Add a verifier to `VERIFIERS` for each new task. A useful task must fail on the pristine fixture
and pass for a reference fix. Do not put hidden tests under the fixture repo itself.

## Limits and safety

This first pack consists of small, objectively verified tasks. It does not measure open-ended
architecture quality, human preference, or production workload frequency; add a blind rubric and
separate reports before drawing those conclusions. Small samples can detect large regressions,
not prove small improvements.

Agent tools are **not sandboxed**. Use only trusted task manifests and fixtures, preferably inside
a disposable container. `evals/results/` is ignored by git because raw artifacts can be large.
