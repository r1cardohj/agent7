# agent7

Minimal coding agent with four tools: `read`, `write`, `bash`, `grep`.

OpenAI-compatible APIs (DeepSeek, Kimi, …).

## Install

```bash
uv sync
```

## Config

```bash
cp .env.example .env
# or:
export AGENT7_API_KEY=sk-...
export AGENT7_BASE_URL=https://api.deepseek.com   # optional
export AGENT7_MODEL=deepseek-flash                # optional; else first /models id
```

Also accepts `DEEPSEEK_API_KEY` / `MOONSHOT_API_KEY`.

## Usage

```bash
uv run agent7 "create hello.py that prints hi"
uv run agent7 --cwd /path/to/repo "fix the failing test"
uv run agent7   # REPL
```

REPL slash commands: `/help`, `/model`, `/status` (`/model` loads ids from `GET /models`).

No sandbox — `bash` / `write` run as your user.
