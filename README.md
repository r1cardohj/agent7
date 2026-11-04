# agent7

**The simplest coding agent. ~1800 lines of Python, four tools, one loop.**

A deliberately minimal harness — all-in on modern, carefully chosen model providers.
Fast and precise. No seatbelt; hold on.

No orchestration: no plan mode, no subagents, no parallel tasks, no background jobs.
You give it one sentence; it reads files, edits files, runs commands, then stops.
Built for people who work on one thing at a time — you hold one task in your head, and so does it.

## Features

- **Deliberately minimal harness** — four tools, one loop, no orchestration
- **Complete CLI agent UX** — streaming thinking + answer, tool lines with ✓/✗, edit/write preview, vi-mode REPL, `@path` completion, session resume
- **Slash commands** — `/help` `/model` `/status` `/login` `/logout` `/compact`
- **Markdown rendering** — final replies rendered as Rich Markdown in the terminal, thanks rich.
- **Paged final replies** — interactive answers open in your system pager (usually `less`); use the arrow keys/Space to scroll and `q` to continue. The rendered reply stays in the terminal after you quit. Redirected output is never paged.
- **Provider login** — paste an API key for DeepSeek, Grok, or Kimi (`/login` or env); switch models in-session


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
export XAI_API_KEY=xai-...       # Grok, from console.x.ai  (or /login grok)
export KIMI_API_KEY=sk-...       # Kimi Code, from the Kimi Code Console (or /login kimi)
# export AGENT7_MODEL=grok-4
```

You can also log in from inside the REPL:

```text
/login deepseek
/login grok              # xAI — Grok models
/login kimi              # Kimi Code — kimi-for-coding and other available models
/model                   # list models for every logged-in provider
/model deepseek-flash
/model grok-4
/model kimi-for-coding
```

## Usage

```bash
uv run agent7 "create hello.py that prints hi"   # one-shot
uv run agent7                                    # new REPL session
uv run agent7 -r                                 # resume this cwd's session
uv run agent7 -r "continue"                      # resume + one-shot
```

Each workspace keeps a single snapshot — starting without `-r` replaces it.
Sessions are saved automatically after tasks, commands, interruptions, and exit.
Snapshots contain resumable model context, not a separate full transcript; compaction is reflected
in the saved context. Files live in `${XDG_STATE_HOME:-~/.local/state}/agent7/sessions/` and do not
store API keys. Hard process kills can lose the in-progress turn.

REPL commands: `/help` `/model` `/status` `/login` `/logout` `/compact`.
Exit with Ctrl-D (or Ctrl-Z on Windows). Ctrl-C discards the current input or interrupts
the running turn, then returns to the prompt instead of exiting the REPL.

Reasoning models (DeepSeek, Grok, Kimi) stream their `reasoning_content` live: it shows up
dimmed under a `thinking` header, before the answer and above each tool line. It is
display only — reasoning is never replayed into the prompt or the saved session.

Type `@` in the REPL to recursively complete files and directories from the cwd:
`fix @agent7/agent.py`.
Tab accepts, directories keep going (`@agent7/` → `@agent7/session.py`), and spaces
are escaped (`@my\ file.txt`), so the path the model sees is the path it can read.

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

## What it is not

- Not an orchestrator: no branching, no subagents
- Not a parallel framework: single-threaded, one thing at a time
- Not a background service: work runs only in the foreground
- No sandbox: tools run as you, and `bash` can do anything your account can do

If that kind of "small" is exactly what you want, this is enough. If it's missing
the abstraction you had in mind, that omission is probably the point — not an oversight.

