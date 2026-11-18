"""Host shell selection and bounded command execution."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_BASH_CHARS = 12_000
_DEFAULT_TIMEOUT = 60


@dataclass(frozen=True)
class Shell:
    """Host shell used by the bash tool and named in an application prompt."""

    name: str  # bash | powershell | cmd
    argv: tuple[str, ...]  # prefix; the command string is appended as one arg


_SHELL_HINTS = {
    "bash": "rg, grep, find, ls (POSIX)",
    "powershell": "rg if available, Get-ChildItem, Select-String (PowerShell)",
    "cmd": "dir, findstr; rg if on PATH (cmd.exe)",
}


def shell_hints(shell: Shell) -> str:
    return _SHELL_HINTS.get(shell.name, shell.name)


def _is_wsl_bash_stub(path: str) -> bool:
    # Windows ships System32\bash.exe as a WSL launcher — skip it for local Git Bash.
    norm = path.replace("/", "\\").lower()
    return norm.endswith("\\system32\\bash.exe") or norm.endswith("\\sysnative\\bash.exe")


def _git_bash_path() -> str | None:
    found = shutil.which("bash")
    if found and not _is_wsl_bash_stub(found):
        return found
    roots = [
        os.environ.get("PROGRAMFILES", r"C:\Program Files"),
        os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"),
        os.environ.get("LOCALAPPDATA", ""),
    ]
    for root in roots:
        if not root:
            continue
        for rel in (r"Git\bin\bash.exe", r"Git\usr\bin\bash.exe", r"Programs\Git\bin\bash.exe"):
            cand = Path(root) / rel
            if cand.is_file():
                return str(cand)
    return None


def _shell_for_name(name: str, platform: str) -> Shell:
    if name == "bash":
        if platform == "win32":
            bash = _git_bash_path() or shutil.which("bash") or "bash"
        else:
            bash = shutil.which("bash") or "/bin/bash"
        return Shell("bash", (bash, "-lc"))
    if name == "powershell":
        exe = shutil.which("pwsh") or shutil.which("powershell") or "powershell"
        return Shell("powershell", (exe, "-NoProfile", "-NonInteractive", "-Command"))
    comspec = os.environ.get("COMSPEC", "cmd.exe")
    return Shell("cmd", (comspec, "/c"))


def resolve_shell(*, platform: str | None = None, override: str | None = None) -> Shell:
    """Pick a shell from sys.platform, honoring an explicit executable/name."""

    plat = sys.platform if platform is None else platform
    if override is None:
        override = os.environ.get("AGENT0_SHELL", "")
    override = override.strip()
    if override:
        key = override.lower()
        if key in ("bash", "powershell", "cmd", "sh"):
            return _shell_for_name("bash" if key == "sh" else key, plat)
        base = Path(override).name.lower()
        if "bash" in base:
            return Shell("bash", (override, "-lc"))
        if base in ("pwsh", "pwsh.exe") or "powershell" in base:
            return Shell("powershell", (override, "-NoProfile", "-NonInteractive", "-Command"))
        return Shell("cmd", (override, "/c"))
    if plat == "win32":
        if bash := _git_bash_path():
            return Shell("bash", (bash, "-lc"))
        if pwsh := shutil.which("pwsh") or shutil.which("powershell"):
            return Shell("powershell", (pwsh, "-NoProfile", "-NonInteractive", "-Command"))
        comspec = os.environ.get("COMSPEC", "cmd.exe")
        return Shell("cmd", (comspec, "/c"))
    bash = shutil.which("bash") or "/bin/bash"
    return Shell("bash", (bash, "-lc"))


_BASH_HEAD_BYTES = MAX_BASH_CHARS * 4  # one char is at most 4 bytes in UTF-8


def _head(f: Any) -> str:
    """First ~MAX_BASH_CHARS chars of a temp file; the rest never enters memory."""

    f.flush()
    f.seek(0)
    return f.read(_BASH_HEAD_BYTES).decode("utf-8", "replace")


def _process_group_options() -> dict[str, Any]:
    if os.name == "posix":
        return {"start_new_session": True}
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {}


def _kill_process_tree(proc: subprocess.Popen) -> None:
    if os.name == "posix":
        with suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        return
    if sys.platform == "win32":
        with suppress(FileNotFoundError, subprocess.TimeoutExpired):
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
    if proc.poll() is None:
        proc.kill()


def _truncate(s: str, limit: int, *, tag: str = "truncated") -> str:
    return s if len(s) <= limit else s[:limit] + f"\n...[{tag} {len(s) - limit} chars]"


def run_shell(
    cwd: Path | None,
    command: str,
    timeout: int | None = None,
    *,
    shell: Shell | None = None,
) -> str:
    if cwd is None:
        return "error: no workspace is configured"
    selected = shell or resolve_shell()
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("LANG", "C.UTF-8")
    env.setdefault("LC_ALL", "C.UTF-8")
    argv = [*selected.argv, command]
    limit = timeout or _DEFAULT_TIMEOUT
    # Output goes to temp files instead of buffering everything in memory. The
    # shell gets its own process group so a timeout kills its children as well.
    with tempfile.TemporaryFile() as out_f, tempfile.TemporaryFile() as err_f:
        try:
            proc = subprocess.Popen(
                argv,
                cwd=cwd,
                stdout=out_f,
                stderr=err_f,
                env=env,
                **_process_group_options(),
            )
        except FileNotFoundError:
            return f"error: shell not found: {selected.argv[0]}"
        try:
            proc.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc)
            proc.wait()
            return _truncate(
                f"timeout after {limit}s\n{_head(out_f)}{_head(err_f)}", MAX_BASH_CHARS
            )
        parts = [f"exit={proc.returncode}"]
        if stdout := _head(out_f):
            parts.append("stdout:\n" + stdout)
        if stderr := _head(err_f):
            parts.append("stderr:\n" + stderr)
    return _truncate("\n".join(parts), MAX_BASH_CHARS)
