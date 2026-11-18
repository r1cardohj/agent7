"""Runtime shell selection and process cleanup."""

import contextlib
import os
import select
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from agent0_runtime.shell import resolve_shell, run_shell


class ResolveShellTests(unittest.TestCase):
    def test_win32_falls_back_to_powershell(self) -> None:
        with (
            patch.dict(os.environ, {"COMSPEC": "C:\\Windows\\System32\\cmd.exe"}, clear=False),
            patch("agent0_runtime.shell._git_bash_path", return_value=None),
            patch(
                "agent0_runtime.shell.shutil.which",
                side_effect=lambda name: "C:\\pwsh.exe" if name == "pwsh" else None,
            ),
        ):
            shell = resolve_shell(platform="win32", override="")

        self.assertEqual(shell.name, "powershell")
        self.assertIn("-Command", shell.argv)

    def test_win32_uses_cmd_when_nothing_else_is_available(self) -> None:
        with (
            patch.dict(os.environ, {"COMSPEC": "C:\\Windows\\System32\\cmd.exe"}, clear=False),
            patch("agent0_runtime.shell._git_bash_path", return_value=None),
            patch("agent0_runtime.shell.shutil.which", return_value=None),
        ):
            shell = resolve_shell(platform="win32", override="")

        self.assertEqual(shell.name, "cmd")
        self.assertEqual(shell.argv[-1], "/c")


class BashTimeoutTests(unittest.TestCase):
    @staticmethod
    def _process_running(pid: int) -> bool:
        if sys.platform.startswith("linux"):
            try:
                stat = Path(f"/proc/{pid}/stat").read_text()
            except (FileNotFoundError, ProcessLookupError):
                return False
            state = stat.rpartition(") ")[2].split(maxsplit=1)[0]
            return state not in {"Z", "X", "x"}
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        return True

    def _wait_for_exit(self, pid: int, timeout: float) -> bool:
        if sys.platform.startswith("linux") and hasattr(os, "pidfd_open"):
            try:
                fd = os.pidfd_open(pid)
            except ProcessLookupError:
                return True
            try:
                ready, _, _ = select.select([fd], [], [], timeout)
                return bool(ready)
            finally:
                os.close(fd)
        deadline = time.monotonic() + timeout
        while self._process_running(pid):
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.02)
        return True

    @unittest.skipUnless(os.name == "posix", "POSIX process-group integration test")
    def test_timeout_kills_shell_children(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pidfile = Path(tmp) / "child.pid"
            command = (
                "sleep 30 & child=$!; "
                f"printf '%s\\n' \"$child\" > {shlex.quote(str(pidfile))}; "
                'wait "$child"'
            )
            pid = None
            real_wait = subprocess.Popen.wait

            def wait_after_ready(proc, timeout=None):
                if timeout is not None:
                    deadline = time.monotonic() + 5
                    while not pidfile.exists() and time.monotonic() < deadline:
                        time.sleep(0.005)
                return real_wait(proc, timeout=timeout)

            try:
                started = time.monotonic()
                with patch("agent0_runtime.shell.subprocess.Popen.wait", new=wait_after_ready):
                    result = run_shell(Path(tmp), command, 1)
                self.assertLess(time.monotonic() - started, 8)
                self.assertIn("timeout after 1s", result)
                pid = int(pidfile.read_text().strip())
                self.assertTrue(self._wait_for_exit(pid, 2), "timed-out child is still running")
            finally:
                if pid is not None and self._process_running(pid):
                    with contextlib.suppress(ProcessLookupError):
                        os.kill(pid, signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
