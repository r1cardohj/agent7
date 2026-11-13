# agent7

<p align="center">
  <img src="https://raw.githubusercontent.com/r1cardohj/agent7/master/assets/demo.gif"
       alt="A real agent7 session: the REPL opens with Seven, one sentence is typed, and agent7 changes and tests greet.py" width="820">
</p>

<p align="center"><sub>A real session — one sentence in, four tools out.</sub></p>

**The simplest coding agent. Four tools, one loop.**

A deliberately minimal harness — all-in on modern, carefully chosen model providers.
Fast and precise. No seatbelt; hold on.

No orchestration: no plan mode, no subagents, no parallel tasks, no background jobs.
You give it one sentence; it reads files, edits files, runs commands, then stops.
Built for people who work on one thing at a time — you hold one task in your head, and so does it.

## Meet Seven

<img align="right" src="https://raw.githubusercontent.com/r1cardohj/agent7/master/assets/mascot.png" alt="Seven, the little pixel robot" width="100">

**Seven** is agent7's tiny coding companion: a lavender pixel robot with a seven-shaped antenna,
cursor-green eyes, and a lucky `7` on its chest. It stays out of the way until you open the REPL.

The [SVG](assets/mascot.svg) and [PNG](assets/mascot.png) are original pixel artwork with matching
terminal art. Set `AGENT7_NO_MASCOT=1` to hide the REPL welcome; one-shot commands never show it.
To tweak the design, edit `agent7/mascot.py`, then run `python scripts/generate_mascot.py`.

## Features

- **Deliberately minimal harness** — four tools, one loop, no orchestration
- **Complete CLI agent UX** — streaming thinking + answer, tool lines with ✓/✗, edit/write preview, vi-mode REPL with a bottom status bar, fuzzy `@path` completion, session resume
- **Slash commands** — `/help` `/model` `/status` `/login` `/logout` `/compact`
- **Markdown rendering** — final replies rendered as Rich Markdown in the terminal, thanks rich.
- **Paged final replies** — interactive answers that exceed one screen open in your system pager (usually `less`), or a built-in pager on Windows; shorter answers print directly. The built-in pager shows which lines are on screen and accepts the arrow keys, PageUp/PageDown, Space, Home/End, the mouse wheel, and `q` (or Esc) to continue. The rendered reply stays in the terminal after you quit. Redirected output is never paged.
- **Provider login** — paste an API key for DeepSeek or Kimi (`/login` or env); switch models in-session
- **Auto intent routing** — `/model auto` scores the current task against reviewed available models, then fails over in score order


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
uv run agent7 "create hello.py that prints hi"   # one-shot
uv run agent7 -m deepseek-flash "say hi"         # one-shot on a chosen model
uv run agent7 -m                                 # pick the model from a menu
uv run agent7                                    # new REPL session
uv run agent7 -r                                 # resume this cwd's session
uv run agent7 -r "continue"                      # resume + one-shot
uv run agent7 --help                             # --cwd, --max-turns, -m, -r, --version
```

The command line is a [click](https://click.palletsprojects.com) command in `agent7/cli.py`
(options, session bootstrap, one-shot runs); the prompt, completion menu and reply rendering
live in `agent7/repl.py`.

Each workspace keeps a single snapshot — starting without `-r` begins a fresh conversation but
reuses the last selected model. Opening or leaving an empty REPL does not touch the saved snapshot;
a new conversation replaces it only after the first agent task. Set `$AGENT7_MODEL` to override the
model choice. Active sessions are saved automatically after tasks, commands, interruptions, and exit.
With `-r`, the saved user/assistant conversation is printed before the REPL prompt or one-shot task;
system and tool messages are omitted.
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
when the model list cannot be fetched. Precedence is `-m` > `$AGENT7_MODEL` > the workspace's last
model. A `-m` that names nothing usable stops the run instead of quietly picking another model, and
a run that never starts (cancelled menu, bad `-m`) leaves the workspace snapshot untouched. Because
`-m` takes the following word as its value, a bare `-m` before a task needs a separator:
`agent7 -m -- "fix it"` (or put the task first: `agent7 "fix it" -m`).

Choose `auto` with `agent7 -m auto` or `/model auto`; there is no route configuration. agent7
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

REPL commands: `/help` `/model` `/status` `/login` `/logout` `/compact`.
Status lives in a bar docked to the bottom row of the terminal while you type:
`model · context · cwd`, with the context gauge turning yellow then red as the window fills,
and the vi mode (`INSERT` / `NORMAL` / `REPLACE`) on the right. The prompt itself stays a bare
`>`. Set `AGENT7_NO_STATUSBAR=1` to go back to a plain prompt.
Press Ctrl-E to edit multiline input in `$VISUAL` / `$EDITOR` (falling back to the system editor).
After `:wq`, the text returns to the prompt for review; Enter submits it. Set `EDITOR=vim` to use Vim.
Exit with Ctrl-D (or Ctrl-Z on Windows). Ctrl-C discards the current input or interrupts
the running turn, then returns to the prompt instead of exiting the REPL.

Reasoning models (DeepSeek and Kimi) stream their Responses reasoning events live: they
show up dimmed under a `thinking` header, before the answer and above each tool line. In a
terminal, the latest six visual lines scroll inside a fixed-height viewport that is cleared
afterward; redirected output retains the full text without control codes. Reasoning text is
display only; opaque protocol items needed to continue a tool call are kept with the session.

Type `@` in the REPL to complete files and directories from the cwd:
`fix @agent7/agent.py`.
A bare `@` lists the cwd, `@dir/` lists that directory, and a typed fragment searches the whole
tree (`@setup` → `docs/guides/setup.md`). Scanning runs in a worker thread and stops after
10k entries, skipping generated directories, so a large working directory or a slow disk never
blocks the prompt. Tab accepts, directories keep going (`@agent7/` → `@agent7/session.py`),
and spaces are escaped (`@my\ file.txt`), so the path the model sees is the path it can read.

The menu is fuzzy, powered by [fzp](https://github.com/r1cardohj/fzp): `@mn` finds `main.py`,
`/mdl` finds `/model`, `/model sonnet` finds `anthropic/claude-sonnet-4`. Candidates are
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

