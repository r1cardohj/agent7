# agent7

**The simplest coding agent. ~1600 lines of Python, four tools, one loop.**

No orchestration: no plan mode, no subagents, no parallel tasks, no background jobs.
You give it one sentence; it reads files, edits files, runs commands, then stops.
Built for people who work on one thing at a time — you hold one task in your head, and so does it.

```
agent7/        1634 lines
  agent.py      513   tools + agent loop + compaction
  __main__.py   472   CLI + REPL + @path completion + session wiring
  commands.py   237   slash commands + model switching
  providers.py  161   API key auth (DeepSeek, Grok)
  store.py      149   atomic JSON session snapshots
  session.py     99   session state
  __init__.py     3   version
```

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
# export AGENT7_MODEL=grok-4
```

You can also log in from inside the REPL:

```text
/login deepseek
/login grok              # xAI — Grok models
/model                   # list models for every logged-in provider
/model deepseek-flash
/model grok-4
```

## Usage

```bash
uv run agent7 "create hello.py that prints hi"   # one-shot
uv run agent7                                    # new REPL session
uv run agent7 -r                                 # resume latest session in this cwd
uv run agent7 -r <session-id>                    # resume a specific session
uv run agent7 session -l                         # list this cwd's sessions
uv run agent7 session "fix authentication"       # new named REPL session
```

Sessions are saved automatically after tasks, commands, interruptions, and exit.
`agent7 session -l` prints tab-separated `ID`, `NAME`, and `PATH` columns and needs no login.
Snapshots contain resumable model context, not a separate full transcript; compaction is reflected
in the saved context. Files live in `${XDG_STATE_HOME:-~/.local/state}/agent7/sessions/` and do not
store API keys. Hard process kills can lose the in-progress turn.

REPL commands: `/help` `/model` `/status` `/login` `/logout` `/compact`.
Exit with an empty line or Ctrl-D.

Reasoning models (DeepSeek, Grok) stream their `reasoning_content` live: it shows up
dimmed under a `thinking` header, before the answer and above each tool line. It is
display only — reasoning is never replayed into the prompt or the saved session.

Type `@` in the REPL to recursively complete files and directories from the cwd:
`fix @agent7/agent.py`.
Tab accepts, directories keep going (`@agent7/` → `@agent7/session.py`), and spaces
are escaped (`@my\ file.txt`), so the path the model sees is the path it can read.

`--cwd` sets the working directory, `--max-turns` caps the loop (default 30 turns;
when it runs out you get `# (stopped after N turns)`). A run that ends with no visible
text never ends silently: you get either `# (stopped after N turns)`,
`# no text in the reply …` or `# … hit the output token limit …`.

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

`AGENT7_CONTEXT_LIMIT` overrides the context limit.

## What it is not

- Not an orchestrator: no branching, no subagents
- Not a parallel framework: single-threaded, one thing at a time
- Not a background service: a foreground process, stopped with Ctrl-C
- No sandbox: tools run as you, and `bash` can do anything your account can do

If that kind of "small" is exactly what you want, this is enough. If it's missing
the abstraction you had in mind, that omission is probably the point — not an oversight.

