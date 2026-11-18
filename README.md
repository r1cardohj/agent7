# agent7

<p align="center">
  <img src="https://raw.githubusercontent.com/r1cardohj/agent7/master/assets/demo.gif"
       alt="A real agent7 session: one sentence is entered, and agent7 edits and tests greet.py" width="820">
</p>

<p align="center"><sub>A one-sentence task: agent7 reads, edits, and verifies <code>greet.py</code>.</sub></p>

<p align="center">
  <a href="https://pypi.org/project/agent7/"><img src="https://img.shields.io/pypi/v/agent7" alt="PyPI version"></a>
  <a href="https://pypi.org/project/agent7/"><img src="https://img.shields.io/pypi/pyversions/agent7" alt="Python versions"></a>
  <a href="https://github.com/r1cardohj/agent7/actions/workflows/ci.yml"><img src="https://github.com/r1cardohj/agent7/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License: MIT"></a>
</p>

agent7 is a minimal coding agent for the terminal. It pairs a single model loop with six
tools: read, write, edit, bash, fetch, and search. Given one task, it explores the workspace,
makes changes, verifies them, and stops.

The design is deliberately narrow. There is no plan mode, no subagents, no parallel tool
execution, and no built-in background scheduler. agent7 works on one task at a time in a
single foreground loop, and streams its reasoning, tool calls, and Markdown answer as they
happen. Long tests, builds, and services can still run detached through Herdr's managed
terminals when agent7 starts inside herdr-core; that is the only background path, and it is
off elsewhere.

## Features

- **Small harness** — six tools and one loop; `bash` covers search, tests, builds, and Git.
- **Terminal-native UI** — streaming Markdown, live reasoning, tool status, diff previews,
  a vi-mode prompt, a docked status bar, and fuzzy completion for commands and `@paths`.
- **Session management** — one resumable snapshot per conversation, grouped by workspace,
  readable names, an fzf-style picker, and safe multi-select deletion.
- **Model routing** — DeepSeek and Kimi through API keys; one-shot tasks use automatic
  intent-aware routing by default, and any model can be selected explicitly.
- **Web tools** — proxy-aware page fetching and ranked web search with query operators.
- **Optional Herdr integration** — terminal lifecycle reporting and managed background runs
  for long tests, builds, and services.
- **Remote access setup** — `agent7 deploy` prepares Tailscale and SSH for Herdr Mobile.

## Requirements

- Python 3.10 or later
- Linux, macOS, or Windows
- An API key for DeepSeek, Kimi, or both

## Installation

```bash
pip install agent7
```

Or install it as an isolated command:

```bash
uv tool install agent7
```

From a source checkout:

```bash
git clone https://github.com/r1cardohj/agent7.git
cd agent7
uv sync
uv run agent7
```

## Authentication

agent7 uses API keys; there is no OAuth flow. A key can come from the environment or from
`/login` inside the REPL:

```bash
export DEEPSEEK_API_KEY=sk-...
export KIMI_API_KEY=sk-...
```

```text
> /login deepseek
> /login kimi
> /model
```

Keys entered through `/login` are stored in `~/.config/agent7/auth.json` (or
`$XDG_CONFIG_HOME/agent7/auth.json`) with owner-only permissions. The file never contains
session data. Use `/logout <provider>` to remove a key.

`search` uses the DeepSeek Search API and therefore requires a DeepSeek key even when the
active model is provided by Kimi.

## Usage

Run a one-shot task:

```bash
agent7 "create hello.py that prints hi"
agent7 -m deepseek-flash "add type hints to utils.py"
agent7 --cwd ../project --max-turns 500 "fix the failing tests"
```

Omit `TASK` to open the interactive REPL:

```bash
agent7          # new session
agent7 -r       # pick a saved session
agent7 -r ID    # resume a specific session
agent7 -m       # pick a model for this run
agent7 --delete # pick sessions and move them to trash
```

`-m/--model` accepts `provider/model` or a bare model name. A bare `-m` opens a model picker.
Because `-m` consumes the following word, use a separator when a one-shot task follows a bare
`-m`: `agent7 -m -- "fix it"`.

`--max-turns` applies per run. Resuming a session does not pin it to the cap it was created
with. Other options are listed by `agent7 --help`.

## REPL

The prompt is a multiline vi-mode editor. A status bar shows the active model, context usage,
working directory, and editing mode. Argument completion is fuzzy: type `@` for workspace
paths, `/` for commands, and `/model <fragment>` for models. `Ctrl-E` opens the current input
in `$VISUAL` or `$EDITOR`; `Ctrl-C` cancels the input or interrupts a running turn;
`Ctrl-D` exits (`Ctrl-Z` on Windows).

`Ctrl-V` (or `Alt-V`, when the terminal owns Ctrl-V) pastes an image from the clipboard
(in vi insert mode): the image is written to a temporary file and its path is inserted into
the prompt, so you can keep typing around it. Nothing is attached automatically — the model
sees the picture by calling `read` on that path. Text is pasted normally when the clipboard
has no image. Clipboard backends: `wl-paste` (Wayland) or `xclip` (X11) on Linux, `pngpaste`
on macOS, PowerShell on Windows and WSL.

| Command | Purpose |
| --- | --- |
| `/help` | List commands |
| `/session [ID]` | Pick or switch sessions |
| `/session delete [ID...]` | Delete one or more other sessions |
| `/model [REF]` | List or switch models; `/model auto` enables routing |
| `/model refresh` | Refresh the provider model list |
| `/status` | Show model, provider, usage, and workspace details |
| `/login <provider>` | Store an API key |
| `/logout <provider>` | Remove a stored API key |
| `/compact` | Compact the conversation manually |

The REPL opens with Seven, the project's pixel mascot. Set `AGENT7_NO_MASCOT=1` to hide it,
or `AGENT7_NO_STATUSBAR=1` to use a plain prompt.

## Tools

| Tool | Description | Output limit |
| --- | --- | --- |
| `read` | Read a text file with line numbers, or an image (png/jpg/jpeg/gif/webp/bmp) as an inline attachment | 200 lines by default; 50k character cap; images resized to 2000px / 4.5MB |
| `write` | Create or overwrite a whole file | One-line result; diff preview in the REPL |
| `edit` | Exact replacements, with one or more non-overlapping `edits[]` per call | One-line result; diff preview in the REPL |
| `bash` | Run a command in the workspace using the host shell | About 12k characters |
| `fetch` | Download an `http(s)` page and extract readable text or Markdown | 12k by default, 50k maximum |
| `search` | Search the web through the DeepSeek Search API | 12k; 8 results by default |

Edits are applied without a confirmation prompt. All tools run serially, so output is never
interleaved. Writes are atomic: an interrupted `write` or `edit` leaves the original file
untouched.

Images travel inside the `read` tool result, so a session snapshot keeps them and later rounds
can still see them. Vision is checked against the reviewed capability table first (DeepSeek
Flash and the Kimi coding models both accept images, including in tool results); text-only
models get the text note instead of the image, and compaction drops old images while keeping
their notes.

On Windows, agent7 selects Git Bash, then PowerShell, then `cmd`; on other platforms it uses
bash. The selected shell is written into the system prompt. Override it with
`AGENT7_SHELL=bash|powershell|cmd|/path/to/shell`.

## Models and routing

| Provider | Environment variable | Default endpoint | Default model |
| --- | --- | --- | --- |
| DeepSeek | `DEEPSEEK_API_KEY` | `https://api.deepseek.com` | `deepseek-chat` |
| Kimi Code | `KIMI_API_KEY` | `https://api.kimi.com/coding/v1` | `kimi-for-coding` |

Model references have the form `provider/model`; aliases such as `ds` are accepted for
DeepSeek. When a bare name is unambiguous, `/model` and `-m` resolve it automatically.

A one-shot task without an explicit model starts on `auto`. Auto builds its candidate pool
from a reviewed capability table and the providers that are logged in. When several targets
are available, the first available fast model scores the task for intent, and the
highest-scoring model handles the whole task, including its tool rounds. Later tasks keep a
close-scoring incumbent to avoid unnecessary switching. If scoring fails, the route falls
back to the current or first available target. A conversation that already contains images
only routes among models marked as vision-capable, and the scorer is told the task has
attachments so it can price that in; it still reads text only.

| Reviewed model | Strength | Images |
| --- | --- | --- |
| `deepseek-flash` | fast, coding | yes |
| `kimi-for-coding` | fast, coding | yes |
| `k3.1-pro` | strong | yes |

Only reviewed models participate in auto. Other models listed by a provider remain available
through `/model` and `-m`. The status bar shows the concrete target when routing is active,
for example `auto → kimi/kimi-for-coding · coding · 91`.

Precedence for model selection is `-m` > `AGENT7_MODEL` > the workspace's last model. For a
one-shot task, the implicit default is `auto`; the saved model remains its fallback.

## Sessions

Each workspace has its own sessions. Every conversation is stored as a single snapshot with
an ID such as `sess-0123456789abcdef...` and a name derived from the first user message.
Startup without `-r` creates a fresh conversation but reuses the last selected model.

Snapshots live in `${XDG_STATE_HOME:-~/.local/state}/agent7/sessions/` and contain the
resumable model context, not API keys. New snapshots use format 2; format 1 snapshots still
load. Sessions are saved after tasks, commands, interruptions, and exit. A hard process kill
can lose the turn in progress.

`--delete` moves sessions to `${XDG_STATE_HOME:-~/.local/state}/agent7/trash/`. Existing
trash entries are never overwritten; repeated IDs receive numbered filenames. The active
session cannot be deleted from inside the REPL.

## Context management

When prompt usage reaches 75% of the model's context window, agent7 compacts older content.
The most recent three user turns are preserved; for long single-turn tasks, the last four
tool results are kept intact. Older large tool outputs and assistant text are truncated.

`/compact` runs the same compaction manually. If a provider does not report usage, agent7
uses an estimate. A request rejected as too long is compacted more aggressively and retried
a bounded number of times. `AGENT7_CONTEXT_LIMIT` overrides the detected context window.

## Configuration

| Variable | Effect |
| --- | --- |
| `AGENT7_MODEL` | Default model reference |
| `AGENT7_MAX_TURNS` | Maximum model rounds per task; default `200` |
| `AGENT7_CONTEXT_LIMIT` | Override the model context window in tokens |
| `AGENT7_SHELL` | Shell used by the `bash` tool |
| `AGENT7_HOME` | Directory containing `auth.json` |
| `AGENT7_BASE_URL` | Override the provider API base URL |
| `AGENT7_NO_MASCOT` | Disable the REPL mascot |
| `AGENT7_NO_STATUSBAR` | Disable the docked status bar |
| `AGENT7_NO_HERDR` | Disable all Herdr integration |
| `XDG_CONFIG_HOME` | Base directory for the default `auth.json` path |
| `XDG_STATE_HOME` | Base directory for sessions and trash |

## Herdr integration

Inside a [herdr-core](https://github.com/herdrdev/herdr-core) terminal, agent7 reports its
lifecycle over OSC 7501: idle, working, done, blocked, error, and clear on exit. This is an
observer only; it adds no model tools and writes nothing to redirected output. Set
`AGENT7_NO_HERDR=1` to disable it.

When both `HC_TERMINAL_ID` and the `hc` command are available, agent7 adds a short pointer to
its bundled Herdr skill. For tests or builds expected to exceed about 60 seconds, and for
services, the model reads that skill and uses `hc` as directed. Fast commands still use plain
synchronous `bash`. Outside Herdr, with `hc` unavailable, or with Herdr disabled, agent7 always
uses ordinary synchronous `bash`.

`agent7 deploy` prepares a machine for Herdr Mobile over Tailscale and SSH:

```bash
agent7 deploy              # install and start Tailscale, join the tailnet, enable SSH
agent7 deploy --check      # verify readiness without changing the system
agent7 deploy --stop       # stop tailscaled; keep binaries and login state
agent7 deploy --sshd       # use the system OpenSSH server instead of Tailscale SSH
agent7 deploy --auth-key K # unattended tailnet login ($TAILSCALE_AUTHKEY also works)
```

The command is idempotent and currently targets Linux hosts. Automatic Tailscale
installation uses the official static archive and systemd; `--sshd` uses the host package
manager (currently apt).

## Evaluation

The `evals/` directory contains opt-in benchmarks that run real API calls; CI never executes
them. `evals.run` compares fixed models, a two-tier ablation, and full auto routing on
isolated coding tasks. `evals.compare` runs agent7, Pi, and Kimi Code CLI against the same
fixture and hidden verifier. See [evals/README.md](evals/README.md).

```bash
uv run python -m evals.run --dry-run
uv run python -m evals.compare --dry-run
```

A historical head-to-head run from **2026-11-16** used 12 coding tasks, one repeat, and the
same K3-256k Kimi endpoint for all three agents:

| Agent | Completion | Median time | Median tokens |
| --- | ---: | ---: | ---: |
| agent7 | 12/12 | 27.4s | 7,430 |
| Pi | 12/12 | 29.5s | 9,514 |
| Kimi Code | 12/12 | 34.1s | 51,820 |

agent7 was a 0.3.2-based working tree that included the later tool-replay fixes. Completion
was verified independently after each agent stopped. This is a small historical suite, not a
general coding-agent ranking; repeat it before drawing conclusions about the current release.

## Security and limitations

- **No sandbox.** Tools run with your account's privileges, and `bash` can do anything your
  shell can do. Run agent7 only on code and prompts you trust.
- **No orchestration.** There are no subagents, parallel tasks, plan mode, or background
  job scheduler. Background work exists only as Herdr-managed runs (see the Herdr integration
  section); outside herdr-core every command runs synchronously in the foreground.
- **No confirmation prompts for edits.** Review changes with your normal version control
  workflow.
- **Network content is untrusted.** Fetched pages and search results are data, not
  instructions; the system prompt tells the model to treat them accordingly.
- **Hard kills lose the current turn.** Sessions are durable only after they are saved.

## Runtime

agent7 builds on the standalone [`agent0-runtime`](https://github.com/r1cardohj/agent0-runtime)
package. The runtime owns the headless loop, model/tool interfaces, structured events,
lifecycle hooks, cancellation, and plugin SDK; agent7 owns the terminal interface and
application policy. See the runtime repository for installation, embedding examples,
and its [`docs/runtime.md`](https://github.com/r1cardohj/agent0-runtime/blob/main/docs/runtime.md).

## Development

```bash
uv sync
uv run python -m unittest discover -s tests -v
```

CI tests agent7 on Python 3.10 through 3.13 and smoke-tests its installed wheel.
The standalone runtime has its own repository, CI, and release pipeline.
`scripts/generate_mascot.py` regenerates the mascot assets from `agent7/mascot.py`.

## License

[MIT](LICENSE)
