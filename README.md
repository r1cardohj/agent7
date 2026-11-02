# agent7

Minimal coding agent (`read` / `write` / `bash` / `grep`) for OpenAI-compatible providers.

## Install

```bash
uv sync
```

## Auth

Paste an API key — no OAuth. Stored in `~/.config/agent7/auth.json`.

```bash
export DEEPSEEK_API_KEY=sk-...
# export KIMI_API_KEY=...
# export MOONSHOT_API_KEY=sk-...
# export AGENT7_MODEL=deepseek-flash
```

```text
/login deepseek
/login kimi              # optional: /login kimi global
/login moonshot
/model                   # all logged-in providers
/model deepseek-flash
/model kimi-coding/k3
```

## Usage

```bash
uv run agent7 "create hello.py that prints hi"
uv run agent7   # REPL — /help /login /model /status /compact
```

Writes show a `path  +N -M` line (green/red counts). `write(old_string=...)` replaces one unique span.

Tool output is budgeted: `read` ~200 lines/call with `offset=` hints; `bash`/`grep` ~12k chars (grep: 100 lines).

At ~75% context usage, old tool/assistant text is auto-shrunk (last 3 user turns; one-shot loops keep last 4 tool results). `/compact` runs that pass manually.

No sandbox — tools run as your user.
