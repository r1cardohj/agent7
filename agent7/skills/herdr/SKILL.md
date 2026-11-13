---
name: herdr-background-runs
description: Run long tests, builds, and services in managed Herdr terminals from agent7.
---

# Herdr background runs

`hc` is available through agent7's bash tool. Use it for long non-interactive
tests/builds (expected beyond about 60 seconds) or services that must stay
alive. Use plain bash directly for fast commands such as `ls`, `grep`,
`git status`, or one small test file; never wrap a command that could finish
in one ordinary bash call.

## Test or build

1. Start it once with a unique name, e.g.
   `hc run --bg --no-notify --name a7-<purpose> --cwd <cwd> <command>`, and
   record the returned terminal address. For pass/fail-only work where logs
   are never needed, add `--rm`; after `hc wait` succeeds, skip both log
   reading and cleanup. Otherwise omit `--rm` so failure logs remain
   available. Never start a duplicate because a wait timed out or status is
   still running.
2. Continue independent work. Never sleep or poll.
3. Before relying on the result, run `hc wait <terminal>`; only a `--rm` run
   needs no further log call. Otherwise inspect logs with
   `hc read --lines 100 <terminal>`. `hc wait` returns status, not program
   output; never claim success before observing exit code 0. If it reports
   ended with a missing or ambiguous exit code, wait on the same terminal
   again rather than rerunning it.
4. Remove a finished non-`--rm` terminal after collecting output with
   `hc rm <terminal>`; stop one you started with `hc kill <terminal>`.

## Long-running service

1. Use a stable, meaningful name and start it once; never pass `--rm`. If
   `hc get <name>` shows a live terminal, reuse that service instead of
   starting a duplicate.
2. Never call `hc wait` without `--output`; a service does not exit. Confirm
   readiness with `hc wait --timeout 30 --output <ready-marker> <terminal>`
   or inspect `hc read --lines 100 <terminal>`. Give the outer bash tool a
   longer timeout than the wait. If that output wait times out while the
   service may already be running, inspect logs and the listening endpoint
   before deciding; never restart merely because the wait timed out.
3. Leave the terminal running while it is needed; do not read it repeatedly
   or clean it up. Stop it only on the user's explicit request with
   `hc kill <terminal>`.
4. If startup fails, read the screen and diagnose errors such as address or
   port reuse before restarting; never blindly relaunch the service.

## Common rules

Give any `hc wait` a shorter timeout than the outer bash tool. Herdr may not
inherit this shell's PATH; if spawn fails with command-not-found before
returning an address, resolve the executable to an absolute path and start it
once. If hc still cannot start an absolute executable, fall back to ordinary
bash with a suitable timeout; never retry a launch with an ambiguous result.
Do not use hc for agents, interactive programs, remote machines, or
destructive operations. Keep the task tied to this terminal unless the user
asks for `--outlive-parent`.
