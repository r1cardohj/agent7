# agent7

<p align="center">
  <img src="https://raw.githubusercontent.com/r1cardohj/agent7/master/assets/demo.gif"
       alt="A real agent7 session: the REPL opens with Seven, one sentence is typed, and agent7 changes and tests greet.py" width="820">
</p>

<p align="center"><sub>A real session — one sentence in, six tools out.</sub></p>

**The simplest coding agent. Six tools, one loop.**

A deliberately minimal harness — all-in on modern, carefully chosen model providers.
Fast and precise. No seatbelt; hold on.

No orchestration: no plan mode, no subagents, no parallel tasks, no background jobs.
You give it one sentence; it reads files, edits files, runs commands, searches and fetches web pages, then stops.
Built for people who work on one thing at a time — you hold one task in your head, and so does it.

## Meet Seven

<img align="right" src="https://raw.githubusercontent.com/r1cardohj/agent7/master/assets/mascot.png" alt="Seven, the little pixel robot" width="100">

**Seven** is agent7's tiny coding companion: a lavender pixel robot with a seven-shaped antenna,
cursor-green eyes, and a lucky `7` on its chest. It stays out of the way until you open the REPL.

The [SVG](assets/mascot.svg) and [PNG](assets/mascot.png) are original pixel artwork with matching
terminal art. Set `AGENT7_NO_MASCOT=1` to hide the REPL welcome; one-shot commands never show it.
To tweak the design, edit `agent7/mascot.py`, then run `python scripts/generate_mascot.py`.

## Features

- **Deliberately minimal harness** — six tools, one loop, no orchestration
- **Web fetch** — fetch an `http(s)` page and extract its readable Markdown text with metadata; proxy-aware urllib3 handles downloads, trafilatura extracts HTML, and raw Markdown/JSON pass through without HTML conversion
- **Web search** — ranked results from the DeepSeek Search API with `site:` / `lang:` / `after:` syntax; the tool ships an on-demand skill (`skills/websearch/SKILL.md`) holding the syntax table and result-triage rules, so the always-on schema stays short
- **Complete CLI agent UX** — streaming thinking + live Markdown answers, tool lines with ✓/✗, edit/write preview, vi-mode REPL with a bottom status bar, fuzzy `@path` completion, isolated session resume and safe deletion
- **Slash commands** — `/help` `/session` `/model` `/status` `/login` `/logout` `/compact`
- **Streaming Markdown rendering** — answer deltas update a Rich Live display as they arrive, coalesced into 30 repaints per second. Rendering is incremental: completed blocks are parsed and pre-rendered once, and only the in-progress block is re-parsed per repaint. Long replies stay in the terminal scrollback; there is no pager.
- **Provider login** — paste an API key for DeepSeek or Kimi (`/login` or env); switch models in-session
- **Auto intent routing** — one-shot tasks use it by default; `/model auto` scores the current task against reviewed available models, then fails over in score order
- **Optional Herdr background runs** — inside a Herdr terminal with `hc` on `PATH`, a bundled on-demand skill manages long tests/builds and services while agent7 keeps working; outside Herdr agent7 uses ordinary synchronous bash


## Install

```bash
pip install agent7
# or: uv tool install agent7
# or from source: uv sync && uv run agent7
```

## Auth

Paste an API key; there is no OAuth. Keys are stored in `~/.config/agent7/auth.json`.

```bash
export DEEPSEEK_API_KEY=sk-...   # or /login deepseek
export KIMI_API_KEY=sk-...       # Kimi Code, from the Kimi Code Console (or /login kimi)
# export AGENT7_MODEL=kimi-for-coding
```

You can also log in from inside the REPL:

```text
/login deepseek
/login kimi              # Kimi Code — kimi-for-coding and other available models
/model                   # list models for every logged-in provider
/model deepseek-flash
/model kimi-for-coding
```

## Usage

```bash
uv run agent7 "create hello.py that prints hi"   # one-shot, auto-routed
uv run agent7 -m deepseek-flash "say hi"         # one-shot on a chosen model
uv run agent7 -m                                 # pick the model from a menu
uv run agent7                                    # new REPL session
uv run agent7 -r                                 # pick a session to resume
uv run agent7 -r SESSION_ID                      # resume a specific session
uv run agent7 -r "continue"                      # latest session + one-shot
uv run agent7 --delete                           # pick a session to delete
uv run agent7 --delete SESSION_ID                # delete a specific session
uv run agent7 --help                             # --cwd, --max-turns, -m, -r, --delete, --version
```

The command line is a [click](https://click.palletsprojects.com) command in `agent7/cli.py`
(options, session bootstrap, one-shot runs); the prompt, completion menu and reply rendering
live in `agent7/repl.py`.

Every conversation is a separate session with a unique `sess-…` ID and its own snapshot, including
multiple sessions in the same workspace. Its readable name is generated automatically from the first
user message, so no naming step is needed. Starting without `-r` begins a fresh conversation but
reuses the most recently selected model in the REPL. A one-shot task with no explicit model starts on
`auto` instead; the saved model remains its fallback. Set `$AGENT7_MODEL` to override the model choice.

A bare `-r` opens an fzp picker over the current workspace's sessions, showing each name and ID;
`agent7 -r SESSION_ID` resumes one directly. Inside the REPL, `/session` opens the same picker and
switches the conversation in place (`/session SESSION_ID` also works). With `-r`, the saved
user/assistant conversation is printed before the REPL prompt or one-shot task; system and tool
messages are omitted. For compatibility, `-r "continue"` still resumes the latest session and runs
that one-shot without opening the picker. Opening or leaving an untouched new REPL creates no file;
active sessions are saved automatically after tasks, commands, interruptions, and exit.

`--delete` uses the same picker when no ID is given and always asks for confirmation before moving
the snapshot to `${XDG_STATE_HOME:-~/.local/state}/agent7/trash/`. It never overwrites an older
trashed session: if the same ID was recreated, later copies become `SESSION_ID.2.json`,
`SESSION_ID.3.json`, and so on instead of blocking deletion. Inside the REPL, use
`/session delete` or `/session delete SESSION_ID`; the delete picker omits the active session,
while naming the active ID directly is refused so a later save cannot silently recreate it.
There is no automatic trash cleanup yet.
Model calls use the OpenAI Responses API with local function tools. Sessions replay raw output
items to their originating provider where required, without using server-side conversation state;
older snapshots and provider switches use portable function calls and results instead.
New files are snapshot format 2. Format 1 still loads; older releases refuse format 2
instead of replaying fields they do not know.

Snapshots contain resumable model context, not a separate full transcript; compaction is reflected
in both the saved context and the displayed history. Files live in
`${XDG_STATE_HOME:-~/.local/state}/agent7/sessions/` and do not store API keys. Hard process kills
can lose the in-progress turn.

`-m/--model` chooses the model for this run. `-m deepseek-flash` resolves a name exactly like
`/model` does; a bare `-m` opens an [fzp](https://github.com/r1cardohj/fzp) picker over every model
the logged-in providers offer — type to filter, Enter to choose, Esc or Ctrl-C to cancel. The menu
starts from the model the workspace would otherwise use, and falls back to each provider's default
when the model list cannot be fetched. Precedence is `-m` > `$AGENT7_MODEL` > the default, where the
default is the workspace's last model in the REPL and `auto` for a one-shot task. If `auto` cannot
build a route, a one-shot falls back to the saved model, then the fetched model list, then the
active provider's default. A `-m` that names nothing usable stops the run instead of quietly
picking another model, and a run that never starts (cancelled menu, bad `-m`) leaves the workspace
snapshot untouched. Because `-m` takes the following word as its value, a bare `-m` before a task
needs a separator:
`agent7 -m -- "fix it"` (or put the task first: `agent7 "fix it" -m`).
Reusing a saved model does not block startup on provider discovery. Interactive sessions
refresh the model list on a daemon worker; `/model` joins that in-flight refresh instead of
issuing duplicate requests, or reads the populated cache immediately. Process exit never
waits for the worker, so a blocked DNS or HTTP call cannot delay Ctrl-D.

Choose `auto` with `agent7 -m auto` or `/model auto`; one-shot tasks use it by default, and there
is no route configuration. agent7
builds the candidate pool from its reviewed capability table, your logged-in providers, and
the models currently listed by those providers:

| Models | Tags |
| --- | --- |
| `deepseek-flash`, `kimi-for-coding` | fast, coding |
| `k3.1-pro` | strong |

The first available fast model scores all candidates for each new user task in one short,
non-streaming, zero-temperature request. The highest-scoring model handles the whole task,
including its tool rounds. The first task has no incumbent and always uses the highest score;
on later tasks, a close score keeps the current model to avoid needless switching. Only reviewed
models join auto — other listed models remain available for explicit `/model` selection. If
the model list cannot be fetched, reviewed models from logged-in providers are still tried.

Each execution target keeps the normal transient-error retry budget. Authentication, permission,
and missing-model errors (401/403/404) skip same-target retries but can still fail over. Only a
failure before any text or reasoning was emitted can move to the next target, in score order.
Scoring timeout, failure, or invalid JSON falls back to the current/default target instead of
blocking the task. The snapshot keeps the logical choice `auto`, while the status bar shows
the concrete target, intent, and score as `auto → provider/model · coding · 91`. `/status`
includes the scorer and candidate scores. No quota-aware selection is performed.

REPL commands: `/help` `/session` `/model` `/status` `/login` `/logout` `/compact`.
Status lives in a bar docked to the bottom row of the terminal while you type:
`model · context · cwd`, with the context gauge turning yellow then red as the window fills,
and the vi mode (`INSERT` / `NORMAL` / `REPLACE`) on the right. The prompt itself stays a bare
`>`. Set `AGENT7_NO_STATUSBAR=1` to go back to a plain prompt.
Press Ctrl-E to edit multiline input in `$VISUAL` / `$EDITOR` (falling back to the system editor).
After `:wq`, the text returns to the prompt for review; Enter submits it. Set `EDITOR=vim` to use Vim.
Exit with Ctrl-D (or Ctrl-Z on Windows). Leaving a saved session prints
`agent7 -r SESSION_ID` so you can copy the command to resume it quickly. Ctrl-C discards the current
input or interrupts the running turn, then returns to the prompt instead of exiting the REPL.

Reasoning models (DeepSeek and Kimi) stream their Responses reasoning events live: they
show up dimmed under a `thinking` header, before the answer and above each tool line. In a
terminal, the latest six visual lines scroll inside a fixed-height viewport that is cleared
afterward; redirected output retains the full text without control codes. Reasoning text is
display only; opaque protocol items needed to continue a tool call are kept with the session.

Answer deltas also appear immediately in a terminal: Rich Live repaints up to 30 times
per second, then renders the complete reply when the stream ends. Rendering is
incremental — a block split at blank lines outside code fences is parsed and pre-rendered
once when it closes, and only the block being typed is re-parsed per repaint, so long
answers stay smooth. The live view is capped to one screen and follows the latest lines;
when the stream ends it is replaced by one complete static reply, so intermediate frames
never duplicate text in the terminal scrollback. A draft emitted before a tool call is
cleared rather than mistaken for the final answer. Redirected answers remain buffered
and are rendered once without terminal control codes.

Type `@` in the REPL to complete files and directories from the cwd:
`fix @agent7/agent.py`.
A bare `@` lists the cwd, `@dir/` lists that directory, and a typed fragment searches the whole
tree (`@setup` → `docs/guides/setup.md`). Scanning runs in a worker thread and stops after
10k entries, skipping generated directories, so a large working directory or a slow disk never
blocks the prompt. Tab accepts, directories keep going (`@agent7/` → `@agent7/session.py`),
and spaces are escaped (`@my\ file.txt`), so the path the model sees is the path it can read.

The menu is fuzzy, powered by [fzp](https://github.com/r1cardohj/fzp): `@mn` finds `main.py`,
`/mdl` finds `/model`, `/model sonnet` finds `anthropic/claude-sonnet-4`, and `-r`, `/session`, or
`--delete` filters sessions by name or ID. Candidates are
subsequence-matched and ranked by score (consecutive hits and word starts win). Matched
characters are marked in bright green and bold, foreground only; the background is reserved
for the selected row, so a match never punches a colored hole in the highlight. Commands,
`@paths`, `/model` refs and `/login` providers all go through the same scorer.

`--cwd` sets the working directory. `--max-turns` caps the loop at **200 model rounds
per task** — high enough that ordinary work never hits it, low enough to stop a runaway.
The current model round and its tool calls finish before the run pauses. When it runs out,
a yellow notice explains that the task may be unfinished and how to recover. The session is
preserved: type `continue` in the REPL, run `agent7 -r "continue"` for a one-shot, or use
`agent7 --max-turns 500 -r "continue"` to resume with a higher cap. Reaching the cap never
silently starts a new session or extends the same run indefinitely.

The cap is **per run**, not per session: `-r` never pins you to the `--max-turns` an
already-saved session was created with. Precedence is `--max-turns` > `$AGENT7_MAX_TURNS`
> 200.

A run that ends with no visible text never ends silently: you get either
`# (stopped after N turns)`, `# no text in the reply …` or `# … hit the output token limit …`.

## Tools

| Tool | What it does | Output budget |
|---|---|---|
| `read` | Read a file with line numbers | 200 lines per call by default; page with `offset=` |
| `write` | Create or overwrite a whole file | `+N -M` diff stats |
| `edit` | Precise replacements; one call may include multiple `edits[]` against the original file (no overlaps) | `+N -M` diff stats |
| `bash` | Host shell in the cwd (bash / PowerShell / cmd) | ~12k chars |
| `fetch` | Fetch an `http(s)` page with proxy-aware downloads; extract readable HTML Markdown (trafilatura), pass through Markdown/JSON, or fall back to clean plain text | 12k chars by default, capped at 50k via `max_chars` |
| `search` | Web search via the DeepSeek Search API; returns ranked titles, URLs, summaries, dates and authority | 12k chars; 8 results by default |

That's it. No `grep`, no `list_dir`, no `todo`, no `task`.
Need more capability? Let the agent use `bash`.

On startup agent7 picks a shell from `sys.platform`: **Git Bash → PowerShell → cmd** on
Windows, **bash** elsewhere. The choice is written into the system prompt so the model
matches the host. Override with `AGENT7_SHELL=bash|powershell|cmd|/path/to/shell`.

Tools run **serially**, one after another — so the log always reads in order and
you never get three outputs interleaved. Edits are applied without a confirmation
prompt; you just get a one-line diff stat.

## Herdr

Inside a [herdr-core](https://github.com/herdrdev/herdr-core) terminal, agent7 reports its
lifecycle through OSC 7501: `idle` at startup, `working` during a task, `done` on success,
`blocked` at the turn limit, `idle` after an interruption, and `error` after a failed turn.
This is an observer only: it adds no model tools, is inactive outside herdr, and never writes
protocol bytes to a non-terminal stream. Set `AGENT7_NO_HERDR=1` to disable it.

When both `HC_TERMINAL_ID` and the `hc` command are available, agent7 adds only a
short pointer to its packaged [Herdr skill](agent7/skills/herdr/SKILL.md). Before
starting any test or build expected to exceed about 60 seconds, or any service, the
model must read that skill with the ordinary `read` tool and use `hc` as directed.
This keeps the full protocol out of the resident system prompt. Tests and builds
start once with `hc run --bg --no-notify`, allow independent work, then collect the
result with `hc wait` — `hc read` only when failure logs matter. Services use a
stable name, never `--rm`, confirm readiness with `hc wait --output` or `hc read`,
stay running while needed, and are stopped only on request with `hc kill`. Fast
commands still use plain bash. This is an optional enhancement, not a runtime
dependency: without herdr, without `hc`, with an unreadable skill, or with
`AGENT7_NO_HERDR=1`, agent7 keeps using ordinary synchronous `bash`.

Detaching leaves the original process running. When herdr restores a terminal as a fresh shell
rather than rerunning agent7, run `agent7 -r` in the restored directory to resume the workspace
conversation.

### Interrupts

During a turn, both Esc and Ctrl-C interrupt agent7, so direct launches work with `hc interrupt`.
Input typed while a task runs is preserved for the next prompt. To force Ctrl-C instead, install
[the supplied template](assets/herdr-agent7.md) as `agent7.md` in
`~/.config/herdr-core/templates` (`%APPDATA%\herdr-core\templates` on Windows), then start the
agent with `hc run agent7`; it contains `stop: ctrl+c`.

### Windows and WSL

When PowerShell runs inside herdr but you enter WSL manually, pass the terminal identity across
the WSL boundary. In that PowerShell, before running `wsl`:

```powershell
if ($env:WSLENV) {
    $env:WSLENV += ":HC_TERMINAL_ID/u"
} else {
    $env:WSLENV = "HC_TERMINAL_ID/u"
}
wsl
```

In WSL, `printf '%s\n' "$HC_TERMINAL_ID"` should print the terminal ID. OSC 7501 output then
travels back through `wsl.exe`; WSL does not need a separate status-reporting bridge. For a
core-managed WSL machine instead, herdr also supports
`hc machine add --via 'wsl.exe -d Ubuntu -e sh -c' --name wsl` from PowerShell.

## Context

At 75% context usage, old tool/assistant text is compacted automatically: the last
3 user turns are kept, or for one-shot tasks the last 4 tool results. `/compact`
triggers the same compaction manually.

Providers that never report `usage` are measured by an estimate, so compaction still
kicks in; if the server rejects a request as too long anyway, it compacts harder and
retries that turn once.

`AGENT7_CONTEXT_LIMIT` overrides the context limit.

## When things go wrong

- **429 / 5xx / connection errors** are retried up to 3 times (0.5s, 1.5s, 3s) — but
  only while nothing has been printed yet, so you never see a reply twice. After that
  you get `# error: …` and, in the REPL, the prompt comes back instead of the process dying.
- **Ctrl-C in the middle of a tool batch** answers every remaining tool call with
  `error: interrupted`, so the next request is still valid.
- **Writes are atomic** (temp file + rename): an interrupted `write`/`edit` leaves the
  original file untouched, and symlinks keep pointing where they did.
- **Large outputs are bounded**: `read` streams the file and keeps only the window it
  returns; `bash` captures to temp files and reads back ~12k chars. A 100 MB log costs
  a few hundred KB of memory, not 300 MB.

## Head-to-head benchmark

On **2026-11-16**, agent7, Pi, and Kimi Code ran the same 12 coding tasks once each:
**12 tasks × 3 agents × 1 repeat = 36 live model runs**. Every run started from a fresh copy of
`evals/repos/pyapp`; the hidden checks in `evals/verify.py` were copied in only after the agent
stopped. Completion is therefore an independent pass/fail result, not a model-written score.

### Model and harness configuration

All three agents requested **K3-256k from the same Kimi Coding endpoint**. The provider aliases
are different only because each harness has its own provider configuration:

| Harness | Version measured | Requested model | API and thinking configuration |
| --- | --- | --- | --- |
| agent7 | 0.3.2-based working tree with the fixes in this commit | `kimi/k3-256k` | Responses; provider default |
| Pi | 1.4.0 | `kimi-bench/k3-256k` | Chat Completions; thinking `high` |
| Kimi Code | 2.4.1 | `kimi-bench/k3-256k` | Chat Completions; default effort `high` |

Pi was rerun as a separate batch after correcting its compatible API configuration to Chat
Completions and `system` rather than the unsupported `developer` role. Its earlier Responses
runs, which failed during tool-history replay, are not included. The agent7 measurement includes
this commit's Responses replay and transport-retry fixes.

### What the 12 tasks test

The prompts, including one Chinese prompt per category, are in
[`evals/tasks.json`](evals/tasks.json). These are small, objectively verifiable tasks, not open-ended
architecture or human-preference evaluations.

| Category | Task | Required change |
| --- | --- | --- |
| Quick | `quick-greeting` | Trim a name, default blank input to `world`, and return an exact greeting. |
| Quick | `quick-readme-command` | Replace a README example with one exact report command and no unrelated edits. |
| Quick | `quick-boolean-flag` | Parse only `1`, `true`, `yes`, and `on` as true, ignoring case and whitespace. |
| Coding | `coding-slugify` | Implement ASCII lowercasing, non-alphanumeric-run collapsing, and hyphen trimming. |
| Coding | `coding-json-lines` | Parse JSON Lines, skip blanks, and report the actual source line for invalid JSON. |
| Coding | `coding-retry-after` | Parse delay seconds and HTTP dates, clamp negatives, and reject missing/invalid values. |
| Debugging | `debug-pagination` | Fix an off-by-one slice so non-final pages contain exactly `per_page` items. |
| Debugging | `debug-memoize` | Make cache keys distinguish positional args, keyword args, and decorated functions. |
| Debugging | `debug-async-close` | Wake a runner blocked on its tick so `close()` finishes promptly without API changes. |
| Reasoning | `reasoning-transaction-rollback` | Undo only completed operations in reverse order and re-raise the original failure. |
| Reasoning | `reasoning-snapshot-migration` | Migrate v1 snapshots to v2 without input mutation or shared mutable nested data. |
| Reasoning | `reasoning-config-precedence` | Resolve defaults < file < `APP_*` environment < explicit overrides, preserving inputs. |

### Results

| Agent | Completion | Median agent time | Median total tokens | Total tokens | Tokens per passed task |
| --- | ---: | ---: | ---: | ---: | ---: |
| **agent7** | **12/12 (100%)** | **27.4s** | **7,430** | **87,733** | **7,311** |
| Pi | 12/12 (100%) | 29.5s | 9,514 | 124,892 | 10,408 |
| Kimi Code | 12/12 (100%) | 34.1s | 51,820 | 963,933 | 80,328 |

Agent time starts immediately before the agent loop/process and ends when it exits, excluding
fixture setup and hidden verification. Tokens are cumulative across model calls, including repeated
context, cache reads, and reasoning output; they are an efficiency signal, not a dollar cost.
Kimi Code totals include every session agent. No outlier was removed: Kimi Code's `debug-memoize`
run alone used **368,024 tokens over 206.7s**, which raises its total but not its median.

On this suite all three tied on completion, while agent7 had the lowest median time and token use.
This is **one repeat of a small task pack**, not a general coding-agent ranking. Repeat at least
three times before treating completion or efficiency differences as stable. The runner, metric
definitions, raw-artifact layout, and reproduction commands are documented in
[`evals/README.md`](evals/README.md); inspect the plan with
`uv run python -m evals.compare --dry-run`.

## Evaluating auto routing

`evals/` contains an opt-in, live-API benchmark that compares fixed models, a simple
two-tier ablation, and full auto routing on isolated coding tasks. It records independent
verification, routing decisions, scoring/execution tokens, latency, and failovers, then renders
a paired-bootstrap Markdown report. See [evals/README.md](evals/README.md); start with
`uv run python -m evals.run --dry-run`. CI never runs these paid evaluations.

## What it is not

- Not an orchestrator: no branching, no subagents
- Not a parallel framework: single-threaded, one thing at a time
- Not a background service: work runs only in the foreground
- No sandbox: tools run as you, and `bash` can do anything your account can do

If that kind of "small" is exactly what you want, this is enough. If it's missing
the abstraction you had in mind, that omission is probably the point — not an oversight.

