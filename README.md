# agent7

**The simplest coding agent. ~1200 lines of Python, four tools, one loop.**

No orchestration: no plan mode, no subagents, no parallel tasks, no background jobs.
You give it one sentence; it reads files, edits files, runs commands, then stops.
Built for people who work on one thing at a time — you hold one task in your head, and so does it.

```
agent7/        1252 lines
  __main__.py   355   CLI + REPL + @path completion
  agent.py      398   tools + agent loop + compaction
  commands.py   236   slash commands + model switching
  providers.py  161   API key auth
  session.py    101   session state
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
export DEEPSEEK_API_KEY=sk-...
# export KIMI_API_KEY=...
# export MOONSHOT_API_KEY=sk-...
# export AGENT7_MODEL=deepseek-flash
```

You can also log in from inside the REPL:

```text
/login deepseek
/login kimi              # optional: /login kimi global
/login moonshot
/model                   # list models for every logged-in provider
/model deepseek-flash
```

## Usage

```bash
uv run agent7 "create hello.py that prints hi"   # one-shot
uv run agent7                                    # REPL
```

REPL commands: `/help` `/model` `/status` `/login` `/logout` `/compact`.
Exit with an empty line or Ctrl-D.

Type `@` in the REPL to recursively complete files and directories from the cwd:
`fix @agent7/agent.py`.
Tab accepts, directories keep going (`@agent7/` → `@agent7/session.py`), and spaces
are escaped (`@my\ file.txt`), so the path the model sees is the path it can read.

`--cwd` sets the working directory, `--max-turns` caps the loop (default 30 turns;
when it runs out you get `# (stopped after N turns)`). A run that ends with no visible
text never ends silently: you get either `# (stopped after N turns)`,
`# no text in the reply …` or `# … hit the output token limit …`.

## Only four tools

| Tool | What it does | Output budget |
|---|---|---|
| `read` | Read a file with line numbers | 200 lines per call by default; page with `offset=` |
| `write` | Write a file; with `old_string=` replace one unique span | `path  +N -M` diff stats |
| `bash` | Run a shell command in the cwd | ~12k chars |
| `grep` | rg/grep search | 100 lines / ~12k chars |

That's it. No `edit`, no `list_dir`, no `todo`, no `task`.
Need more capability? Let the agent use `bash`.

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
