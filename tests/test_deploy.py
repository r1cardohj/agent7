"""The native `agent7 deploy` remote-access command."""

import hashlib
import io
import json
import subprocess
import tarfile
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from agent7.cli import Options, main
from agent7.remote import (
    RemoteError,
    TailscaleNode,
    _download,
    _extract_archive_member,
    _install_tailscale,
    _run,
    _tailscale_archive_url,
    _tailscale_node,
    deploy_mobile_ssh,
    stop_tailscale,
)


def have_tailscale_only(command: str) -> bool:
    return command in {"tailscale", "tailscaled"}


class DeployCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_deploy_help_is_separate_from_agent_help(self) -> None:
        result = self.runner.invoke(main, ["deploy", "--help"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Usage: agent7 deploy [OPTIONS]", result.output)
        self.assertIn("--check", result.output)
        self.assertIn("--sshd", result.output)
        self.assertIn("--auth-key", result.output)
        self.assertNotIn("model rounds per task", result.output)

    def test_deploy_options_are_forwarded(self) -> None:
        with patch("agent7.remote.deploy_mobile_ssh", return_value=0) as deploy:
            result = self.runner.invoke(
                main, ["deploy", "--check", "--sshd", "--auth-key", "tskey-test"]
            )

        self.assertEqual(result.exit_code, 0)
        deploy.assert_called_once_with(check=True, sshd=True, auth_key="tskey-test")

    def test_check_exit_code_is_forwarded(self) -> None:
        with patch("agent7.remote.deploy_mobile_ssh", return_value=1):
            result = self.runner.invoke(main, ["deploy", "--check"])

        self.assertEqual(result.exit_code, 1)

    def test_stop_forwards_to_the_remote_lifecycle(self) -> None:
        with patch("agent7.remote.stop_tailscale") as stop:
            result = self.runner.invoke(main, ["deploy", "--stop"])

        self.assertEqual(result.exit_code, 0)
        stop.assert_called_once_with()

    def test_stop_cannot_be_combined_with_deploy_options(self) -> None:
        result = self.runner.invoke(main, ["deploy", "--stop", "--check"])

        self.assertEqual(result.exit_code, 2)
        self.assertIn("cannot be combined", result.output)

    def test_deploy_errors_become_exit_1(self) -> None:
        deploy = patch(
            "agent7.remote.deploy_mobile_ssh",
            side_effect=RemoteError("no sudo"),
        )
        with deploy:
            result = self.runner.invoke(main, ["deploy"])

        self.assertEqual(result.exit_code, 1)
        self.assertIn("no sudo", result.output)

    def test_similar_natural_language_still_reaches_the_agent(self) -> None:
        async def fake_async_main(options: Options) -> None:
            self.seen_options = options

        with patch("agent7.cli.async_main", side_effect=fake_async_main):
            result = self.runner.invoke(main, ["deployment", "docs"])

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(self.seen_options.task, "deployment docs")


class StopTailscaleTests(unittest.TestCase):
    def test_stop_uses_systemd_and_preserves_installation(self) -> None:
        out = io.StringIO()
        with (
            patch("agent7.remote._systemd_unit_exists", return_value=True),
            patch("agent7.remote._service_active", side_effect=[True, False]),
            patch("agent7.remote._require") as require,
            redirect_stdout(out),
        ):
            stop_tailscale()

        require.assert_called_once_with(
            ["systemctl", "stop", "tailscaled.service"],
            root=True,
            what="stop tailscaled",
        )
        self.assertIn("binaries and login state are preserved", out.getvalue())

    def test_stop_is_idempotent(self) -> None:
        out = io.StringIO()
        with (
            patch("agent7.remote._systemd_unit_exists", return_value=True),
            patch("agent7.remote._service_active", return_value=False),
            patch("agent7.remote._require") as require,
            redirect_stdout(out),
        ):
            stop_tailscale()

        require.assert_not_called()
        self.assertIn("already stopped", out.getvalue())


class RemoteProcessTests(unittest.TestCase):
    def test_root_commands_refresh_sudo_on_the_terminal_first(self) -> None:
        ok = subprocess.CompletedProcess(["sudo", "-v"], 0)
        done = subprocess.CompletedProcess(["systemctl"], 0, stdout="ok\n")
        with (
            patch("agent7.remote.os.geteuid", return_value=1000),
            patch("agent7.remote.subprocess.run", side_effect=[ok, done]) as run,
        ):
            result = _run(["systemctl", "status"], root=True)

        self.assertEqual(result.stdout, "ok\n")
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args_list[0].args[0][:2], ["sudo", "-p"])

    def test_failed_sudo_is_a_clean_deploy_error(self) -> None:
        denied = subprocess.CompletedProcess(["sudo", "-v"], 1)
        with (
            patch("agent7.remote.os.geteuid", return_value=1000),
            patch("agent7.remote.subprocess.run", return_value=denied),
            self.assertRaisesRegex(RemoteError, "sudo authentication failed"),
        ):
            _run(["systemctl", "status"], root=True)


class FakeIndexResponse:
    def __init__(self, body: str) -> None:
        self.body = body.encode()

    def read(self) -> bytes:
        return self.body

    def __enter__(self) -> "FakeIndexResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class FakeDownloadResponse:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.headers = {"Content-Length": str(sum(map(len, chunks)))}

    def read(self, _size: int) -> bytes:
        return self.chunks.pop(0) if self.chunks else b""

    def __enter__(self) -> "FakeDownloadResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class StaticTailscaleInstallTests(unittest.TestCase):
    def test_extracts_nested_systemd_files_without_extracting_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "tailscale.tgz"
            with tarfile.open(archive, "w:gz") as tf:
                for name, body in {
                    "tailscale_1.0_amd64/tailscale": b"cli",
                    "tailscale_1.0_amd64/systemd/tailscaled.service": b"service",
                }.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(body)
                    tf.addfile(info, io.BytesIO(body))
            target = Path(tmp) / "tailscaled.service"

            _extract_archive_member(str(archive), "tailscaled.service", str(target))

            self.assertEqual(target.read_bytes(), b"service")

    def test_archive_url_uses_newest_versioned_official_build(self) -> None:
        index = """
            <a href="tailscale_1.20.3_amd64.tgz">old</a>
            <a href="tailscale_1.104.7_amd64.tgz">new</a>
            <a href="tailscale_1.104.7_amd64.tgz.sha256">checksum</a>
            <a href="tailscale_2.0_arm64.tgz">other arch</a>
        """
        with (
            patch("agent7.remote.platform.machine", return_value="x86_64"),
            patch(
                "agent7.remote.urllib.request.urlopen",
                return_value=FakeIndexResponse(index),
            ),
        ):
            url, version = _tailscale_archive_url()

        self.assertEqual(version, "1.104.7")
        self.assertEqual(
            url,
            "https://pkgs.tailscale.com/stable/tailscale_1.104.7_amd64.tgz",
        )

    def test_static_install_verifies_checksum_and_avoids_apt(self) -> None:
        archive_bytes = b"official tailscale archive"
        archive_digest = hashlib.sha256(archive_bytes).hexdigest().upper()

        def fake_download(
            url: str, target: str, show_progress: bool | None = None,
        ) -> None:
            del url, show_progress
            Path(target).write_text(
                archive_digest if target.endswith(".sha256") else "",
                encoding="ascii",
            )
            if not target.endswith(".sha256"):
                Path(target).write_bytes(archive_bytes)

        def fake_extract(_archive: str, name: str, target: str) -> None:
            Path(target).write_bytes(name.encode())

        with (
            patch(
                "agent7.remote._tailscale_archive_url",
                return_value=("https://example.test/tailscale.tgz", "1.0.0"),
            ),
            patch("agent7.remote._download", side_effect=fake_download),
            patch("agent7.remote._extract_archive_member", side_effect=fake_extract),
            patch("agent7.remote._require") as require,
        ):
            _install_tailscale()

        commands = [call.args[0] for call in require.call_args_list]
        self.assertTrue(any(command[:2] == ["install", "-m"] for command in commands))
        self.assertTrue(any("/usr/sbin/tailscaled" in command for command in commands))
        self.assertFalse(any("apt-get" in command for command in commands))

    def test_checksum_mismatch_refuses_installation(self) -> None:
        def fake_download(
            _url: str, target: str, show_progress: bool | None = None,
        ) -> None:
            del show_progress
            if target.endswith(".sha256"):
                Path(target).write_text("0" * 64, encoding="ascii")
            else:
                Path(target).write_bytes(b"tampered")

        with (
            patch(
                "agent7.remote._tailscale_archive_url",
                return_value=("https://example.test/tailscale.tgz", "1.0.0"),
            ),
            patch("agent7.remote._download", side_effect=fake_download),
            patch("agent7.remote._extract_archive_member") as extract,
            self.assertRaisesRegex(RemoteError, "checksum mismatch"),
        ):
            _install_tailscale()

        extract.assert_not_called()


class DownloadTests(unittest.TestCase):
    def test_download_writes_every_chunk_and_reports_progress(self) -> None:
        response = FakeDownloadResponse([b"abc", b"def", b"ghi", b"jkl"])
        err = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "installer.sh"
            with (
                patch("agent7.remote.urllib.request.urlopen", return_value=response),
                redirect_stderr(err),
            ):
                _download(
                    "https://example.test/install.sh", str(target), show_progress=True,
                )
            self.assertEqual(target.read_bytes(), b"abcdefghijkl")

        self.assertIn("100%", err.getvalue())
        self.assertIn("\x1b[KDownloaded", err.getvalue())

    def test_download_can_stay_quiet_for_captured_output(self) -> None:
        response = FakeDownloadResponse([b"abc"])
        err = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "installer.sh"
            with (
                patch("agent7.remote.urllib.request.urlopen", return_value=response),
                redirect_stderr(err),
            ):
                _download(
                    "https://example.test/install.sh", str(target), show_progress=False,
                )
            self.assertEqual(target.read_bytes(), b"abc")

        self.assertEqual(err.getvalue(), "")


class DeployMobileSshTests(unittest.TestCase):
    def test_logged_out_tailscale_is_not_treated_as_joined(self) -> None:
        status = json.dumps({
            "BackendState": "NeedsLogin",
            "Self": {"DNSName": "", "TailscaleIPs": []},
        })
        with (
            patch("agent7.remote._have", return_value=True),
            patch(
                "agent7.remote._run",
                return_value=subprocess.CompletedProcess(
                    ["tailscale"], 0, stdout=status,
                ),
            ),
        ):
            self.assertIsNone(_tailscale_node())

    def test_deploy_runs_tailscale_up_when_login_is_missing(self) -> None:
        node = TailscaleNode(ip="100.64.0.2", dns_name="devbox.tailnet.test")
        out = io.StringIO()
        nodes = patch(
            "agent7.remote._tailscale_node",
            side_effect=[None, node],
        )
        with (
            patch("agent7.remote._have", side_effect=have_tailscale_only),
            patch("agent7.remote._service_active", return_value=True),
            nodes,
            patch("agent7.remote._join_tailnet") as join,
            patch("agent7.remote._tailscale_ssh_enabled", return_value=True),
            redirect_stdout(out),
        ):
            code = deploy_mobile_ssh(auth_key="tskey-test")

        self.assertEqual(code, 0)
        join.assert_called_once_with("tskey-test")
        self.assertIn("+ tailnet — devbox.tailnet.test", out.getvalue())

    def test_check_reports_every_missing_piece(self) -> None:
        out = io.StringIO()
        with (
            patch("agent7.remote._have", return_value=False),
            patch("agent7.remote._tailscaled_present", return_value=False),
            patch("agent7.remote._service_active", return_value=False),
            redirect_stdout(out),
        ):
            code = deploy_mobile_ssh(check=True)

        self.assertEqual(code, 1)
        text = out.getvalue()
        for name in ("tailscale", "tailscaled", "tailnet", "tailscale ssh"):
            self.assertIn(name, text)
        self.assertIn("not ready", text)

    def test_check_is_ready_with_tailscale_ssh(self) -> None:
        node = TailscaleNode(ip="100.64.0.2", dns_name="devbox.tailnet.test")
        out = io.StringIO()
        with (
            patch("agent7.remote._have", side_effect=have_tailscale_only),
            patch("agent7.remote._service_active", return_value=True),
            patch("agent7.remote._tailscale_node", return_value=node),
            patch("agent7.remote._tailscale_ssh_enabled", return_value=True),
            redirect_stdout(out),
        ):
            code = deploy_mobile_ssh(check=True)

        self.assertEqual(code, 0)
        self.assertIn("ready", out.getvalue())

    def test_check_with_sshd_reports_system_ssh_state(self) -> None:
        out = io.StringIO()
        service_active = patch(
            "agent7.remote._service_active",
            side_effect=lambda unit: unit == "tailscaled.service",
        )
        with (
            patch("agent7.remote._have", side_effect=have_tailscale_only),
            service_active,
            patch("agent7.remote._tailscale_node", return_value=None),
            patch("agent7.remote._openssh_present", return_value=False),
            redirect_stdout(out),
        ):
            code = deploy_mobile_ssh(check=True, sshd=True)

        self.assertEqual(code, 1)
        text = out.getvalue()
        self.assertIn("openssh-server", text)
        self.assertIn("sshd", text)

    def test_setup_enables_tailscale_ssh_and_prints_phone_steps(self) -> None:
        node = TailscaleNode(ip="100.64.0.2", dns_name="devbox.tailnet.test")
        out = io.StringIO()
        with (
            patch("agent7.remote._have", side_effect=have_tailscale_only),
            patch("agent7.remote._service_active", return_value=True),
            patch("agent7.remote._tailscale_node", return_value=node),
            patch("agent7.remote._tailscale_ssh_enabled", return_value=False),
            patch("agent7.remote._ensure_tailscale_ssh") as enable_ssh,
            redirect_stdout(out),
        ):
            code = deploy_mobile_ssh()

        self.assertEqual(code, 0)
        enable_ssh.assert_called_once_with()
        text = out.getvalue()
        self.assertIn("ready: host=devbox.tailnet.test", text)
        self.assertIn("Herdr Mobile", text)


if __name__ == "__main__":
    unittest.main()
