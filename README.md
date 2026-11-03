# agent7

**The simplest coding agent. ~1500 lines of Python, four tools, one loop.**

No orchestration: no plan mode, no subagents, no parallel tasks, no background jobs.
You give it one sentence; it reads files, edits files, runs commands, then stops.
Built for people who work on one thing at a time — you hold one task in your head, and so does it.

```
agent7/        1482 lines
  __main__.py   434   CLI + REPL + @path completion + session wiring
  agent.py      398   tools + agent loop + compaction
  commands.py   236   slash commands + model switching
  providers.py  161   API key auth
  store.py      149   atomic JSON session snapshots
  session.py    101   session state
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
| `bash` | Shell in the cwd (incl. rg/grep/find/ls) | ~12k chars |

That's it. No `grep`, no `list_dir`, no `todo`, no `task`.
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

## Releasing

CI (`.github/workflows/ci.yml`) runs the unittest suite on Python 3.10–3.13 for every
push to `main`/`master` and every pull request. It also builds the sdist + wheel with
`python -m build`, runs `twine check --strict`, and installs the wheel in a clean venv
to run `agent7 --help`.

Publishing to PyPI is tag-driven (`.github/workflows/release.yml`):

```bash
# 1. bump the version in pyproject.toml and agent7/__init__.py (they must match)
# 2. commit it on main/master and push
# 3. tag and push the tag — this is what publishes
git tag v0.2.0
git push origin master --follow-tags
```

On a `v*` tag the workflow re-runs the tests, verifies the tag commit is on the main
branch, verifies the tag equals the package version, builds the sdist + wheel, runs
`twine check --strict`, publishes with `pypa/gh-action-pypi-publish`, then creates a
GitHub release.

Publishing authenticates through OIDC trusted publishing: PyPI hands the workflow a
short-lived token, so no API token is stored in GitHub, and PEP 740 attestations are
attached by default.

One-time setup on PyPI (see *Your projects → agent7 → Publishing → Add a new pending
publisher / GitHub*):

| Field | Value |
|---|---|
| Owner | `r1cardohj` |
| Repository name | `agent7` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

The environment name is optional, but if you set it here it must match the `environment:`
name in the `publish` job.

Notes:

- Publishing an already-released version fails on purpose (PyPI rejects duplicate
  filenames). Bump the version rather than moving a tag.
- Prefer an API token over OIDC? Set `secrets.PYPI_API_TOKEN` and add
  `with: password: ${{ secrets.PYPI_API_TOKEN }}` to the publish step.
- CI installs the test matrix with pip, not `uv sync`, so whatever index was baked into
  `uv.lock` on a dev machine cannot leak into CI.
