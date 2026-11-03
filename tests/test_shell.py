"""Host shell detection for Windows / Unix."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent7.agent import (
    DEEPSEEK_RL_SYSTEM,
    SHELL,
    SYSTEM,
    BashSession,
    is_deepseek_model,
    make_system,
    resolve_shell,
    tool_bash,
)


class ResolveShellTests(unittest.TestCase):
    def test_linux_defaults_to_bash(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AGENT7_SHELL", None)
            sh = resolve_shell(platform="linux")
        self.assertEqual(sh.name, "bash")
        self.assertEqual(sh.argv[-1], "-lc")

    def test_win32_falls_back_to_powershell_or_cmd(self) -> None:
        with patch.dict(os.environ, {"COMSPEC": "C:\\Windows\\System32\\cmd.exe"}, clear=False):
            os.environ.pop("AGENT7_SHELL", None)
            with patch("agent7.agent._git_bash_path", return_value=None), \
                 patch("agent7.agent.shutil.which", side_effect=lambda n: "C:\\pwsh.exe" if n == "pwsh" else None):
                sh = resolve_shell(platform="win32")
        self.assertEqual(sh.name, "powershell")
        self.assertIn("-Command", sh.argv)

    def test_win32_cmd_when_nothing_else(self) -> None:
        with patch.dict(os.environ, {"COMSPEC": "C:\\Windows\\System32\\cmd.exe"}, clear=False):
            os.environ.pop("AGENT7_SHELL", None)
            with patch("agent7.agent._git_bash_path", return_value=None), \
                 patch("agent7.agent.shutil.which", return_value=None):
                sh = resolve_shell(platform="win32")
        self.assertEqual(sh.name, "cmd")
        self.assertEqual(sh.argv[-1], "/c")

    def test_agent7_shell_override_name(self) -> None:
        with patch.dict(os.environ, {"AGENT7_SHELL": "cmd", "COMSPEC": "cmd.exe"}):
            sh = resolve_shell(platform="win32")
        self.assertEqual(sh.name, "cmd")

    def test_agent7_shell_override_path(self) -> None:
        with patch.dict(os.environ, {"AGENT7_SHELL": r"C:\Git\bin\bash.exe"}):
            sh = resolve_shell(platform="win32")
        self.assertEqual(sh.name, "bash")
        self.assertEqual(sh.argv[0], r"C:\Git\bin\bash.exe")

    def test_system_prompt_names_shell_and_os(self) -> None:
        sh = resolve_shell(platform="linux")
        text = make_system(sh)
        self.assertIn("via bash", text)
        self.assertIn("persist", text)
        self.assertIn(SHELL.name, SYSTEM)  # import-time SYSTEM matches live shell

    def test_deepseek_uses_frozen_rl_persona(self) -> None:
        self.assertTrue(is_deepseek_model("deepseek-chat"))
        self.assertTrue(is_deepseek_model("deepseek-flash"))
        self.assertTrue(is_deepseek_model("other", provider_id="deepseek"))
        self.assertFalse(is_deepseek_model("grok-4"))
        self.assertEqual(make_system(model="deepseek-chat"), DEEPSEEK_RL_SYSTEM)
        self.assertEqual(make_system(model="grok-4", provider_id="deepseek"), DEEPSEEK_RL_SYSTEM)
        self.assertNotEqual(make_system(model="grok-4"), DEEPSEEK_RL_SYSTEM)


class BashToolShellTests(unittest.TestCase):
    def test_runs_through_resolved_shell(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = tool_bash(Path(tmp), "echo hello-shell")
        self.assertIn("exit=0", result)
        self.assertIn("hello-shell", result)

    def test_persistent_bash_keeps_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "nested"
            nested.mkdir()
            bash = BashSession(SHELL, root)
            try:
                first = bash.run(f"cd nested && pwd")
                self.assertIn("exit=0", first)
                second = bash.run("pwd")
                self.assertIn("exit=0", second)
                self.assertIn(str(nested.resolve()), second)
                third = bash.run("export AGENT7_MARK=alive")
                self.assertIn("exit=0", third)
                fourth = bash.run("printf '%s\\n' \"$AGENT7_MARK\"")
                self.assertIn("exit=0", fourth)
                self.assertIn("alive", fourth)
            finally:
                bash.close()

    def test_persistent_bash_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bash = BashSession(SHELL, Path(tmp))
            try:
                result = bash.run("sleep 5", timeout=1)
                self.assertIn("timeout after 1s", result)
                # session recovers for a later command
                again = bash.run("echo recovered")
                self.assertIn("exit=0", again)
                self.assertIn("recovered", again)
            finally:
                bash.close()


if __name__ == "__main__":
    unittest.main()
