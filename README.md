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
uv run agent7   # REPL — /help /login /model /status
```

No sandbox — tools run as your user.
