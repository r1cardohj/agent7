"""One-command remote access setup for Herdr Mobile.

`agent7 deploy` is intentionally a native operations path, not an agent task:
it installs Tailscale from the verified official static archive, starts it, joins
the tailnet, and enables an SSH backend. The default backend is Tailscale SSH;
`--sshd` uses the system OpenSSH server and therefore the host package manager.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

try:
    import pwd
except ImportError:  # Windows: USERNAME carries the interactive account name
    pwd = None  # type: ignore[assignment]

TAILSCALE_INDEX_URL = "https://pkgs.tailscale.com/stable/"
BINARY_UNIT = 1024
DOWNLOAD_CHUNK_SIZE = 64 * BINARY_UNIT

StepState = Literal["ok", "changed", "needed", "unknown", "failed"]


class RemoteError(RuntimeError):
    """A deploy step failed in a way the user must fix."""


@dataclass(frozen=True)
class Step:
    name: str
    state: StepState
    detail: str = ""


@dataclass(frozen=True)
class TailscaleNode:
    ip: str
    dns_name: str
    backend_state: str = "Running"


def _have(command: str) -> bool:
    return shutil.which(command) is not None


def _root_command(args: list[str]) -> list[str]:
    if os.geteuid() == 0:
        return args
    if not _have("sudo"):
        raise RemoteError("root privileges are required and sudo was not found")
    # An explicit prompt makes a password request attributable to agent7.
    return ["sudo", "-p", "[agent7 deploy] password: ", *args]


def _run(
    args: list[str],
    *,
    root: bool = False,
    capture: bool = True,
    text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    command = _root_command(args) if root else args
    try:
        if root and os.geteuid() != 0:
            # Validate sudo on the inherited terminal before capturing output:
            # otherwise its password prompt would disappear into the pipe.
            auth = subprocess.run(
                ["sudo", "-p", "[agent7 deploy] password: ", "-v"],
                check=False,
            )
            if auth.returncode != 0:
                raise RemoteError("sudo authentication failed")
        return subprocess.run(
            command,
            input=text,
            text=True,
            capture_output=capture,
            check=False,
        )
    except OSError as e:
        raise RemoteError(f"cannot run {' '.join(args)}: {e}") from None


def _require(
    args: list[str], *, root: bool = False, what: str = "", capture: bool = True,
) -> None:
    proc = _run(args, root=root, capture=capture)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip() if capture else ""
        label = what or " ".join(args)
        suffix = f": {detail}" if detail else ""
        raise RemoteError(f"{label} failed{suffix}")


def _systemd_unit_exists(unit: str) -> bool:
    proc = _run(["systemctl", "cat", unit])
    return proc.returncode == 0


def _service_active(unit: str) -> bool:
    return _run(["systemctl", "is-active", "--quiet", unit]).returncode == 0


def _tailscale_node() -> TailscaleNode | None:
    if not _have("tailscale"):
        return None
    proc = _run(["tailscale", "status", "--json"])
    if proc.returncode != 0 or not proc.stdout:
        return None
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    # A logged-out daemon still reports an empty Self; only a usable identity
    # counts as joined. Checking the backend state prevents deploy from
    # printing a blank host and skipping the interactive login step.
    backend_state = str(data.get("BackendState") or "")
    if backend_state == "NeedsLogin":
        return None
    self_info = data.get("Self") or {}
    ips = self_info.get("TailscaleIPs") or []
    ip = next((str(item) for item in ips if ":" not in str(item)), "")
    dns_name = str(self_info.get("DNSName") or "").rstrip(".")
    if backend_state != "Running" or (not ip and not dns_name):
        return None
    return TailscaleNode(ip=ip, dns_name=dns_name, backend_state=backend_state)


def _tailscale_ssh_enabled() -> bool | None:
    """Best-effort read of the RunSSH preference.

    Returns None when Tailscale is absent or the preference cannot be read.
    `debug prefs` is not a stability boundary, so treat it as an optimization:
    when unavailable, published SSH host keys prove it is enabled, while their
    absence is only unknown (the tailnet ACL may not have admitted SSH yet).
    """
    if not _have("tailscale"):
        return None
    prefs = _run(["tailscale", "debug", "prefs"])
    if prefs.returncode == 0 and prefs.stdout:
        try:
            value = json.loads(prefs.stdout).get("RunSSH")
        except json.JSONDecodeError:
            value = None
        if isinstance(value, bool):
            return value
    node = _tailscale_node()
    if node is not None:
        status = _run(["tailscale", "status", "--json"])
        if status.returncode == 0 and status.stdout:
            try:
                keys = (json.loads(status.stdout).get("Self") or {}).get("SSH_HostKeys")
            except json.JSONDecodeError:
                keys = None
            if keys:
                return True
    return None


def _tailscale_arch() -> str:
    machine = platform.machine().lower()
    aliases = {
        "x86_64": "amd64",
        "amd64": "amd64",
        "aarch64": "arm64",
        "arm64": "arm64",
        "i386": "386",
        "i686": "386",
        "mips": "mips",
        "mipsle": "mipsle",
        "mips64": "mips64",
        "mips64le": "mips64le",
        "riscv64": "riscv64",
    }
    if machine in aliases:
        return aliases[machine]
    if machine.startswith("armv"):
        return "arm"
    raise RemoteError(f"unsupported Tailscale architecture: {platform.machine()}")


def _tailscale_archive_url() -> tuple[str, str]:
    """Resolve the newest versioned static archive from Tailscale's index."""
    arch = _tailscale_arch()
    try:
        with urllib.request.urlopen(TAILSCALE_INDEX_URL, timeout=30) as response:
            index = response.read().decode("utf-8", errors="replace")
    except OSError as e:
        raise RemoteError(f"cannot fetch the Tailscale package index: {e}") from None
    pattern = re.compile(rf"tailscale_([0-9]+(?:\.[0-9]+)+)_{re.escape(arch)}\.tgz")
    versions: dict[tuple[int, ...], str] = {}
    for match in pattern.finditer(index):
        text = match.group(1)
        versions[tuple(int(part) for part in text.split("."))] = text
    if not versions:
        message = f"no official Tailscale archive found for architecture {arch}"
        raise RemoteError(message)
    version = versions[max(versions)]
    return f"{TAILSCALE_INDEX_URL}tailscale_{version}_{arch}.tgz", version


def _format_bytes(count: int) -> str:
    if count < BINARY_UNIT:
        return f"{count} B"
    value = float(count)
    for unit in ("KiB", "MiB", "GiB"):
        value /= BINARY_UNIT
        if value < BINARY_UNIT:
            return f"{value:.1f} {unit}"
    return f"{value:.1f} TiB"


def _download(url: str, target: str, *, show_progress: bool | None = None) -> None:
    """Download a file, rendering progress only on an interactive terminal."""
    if show_progress is None:
        show_progress = sys.stderr.isatty()
    request = urllib.request.Request(url)
    with urllib.request.urlopen(request, timeout=30) as response:
        total_header = response.headers.get("Content-Length", "")
        try:
            total = int(total_header)
        except ValueError:
            total = 0
        downloaded = 0
        last_percent = -1
        if show_progress:
            size = f" ({_format_bytes(total)})" if total else " (unknown size)"
            print(f"Downloading {url}{size}...", file=sys.stderr)
        with open(target, "wb") as f:
            while chunk := response.read(DOWNLOAD_CHUNK_SIZE):
                f.write(chunk)
                downloaded += len(chunk)
                if not show_progress:
                    continue
                if total:
                    percent = downloaded * 100 // total
                    if percent != last_percent:
                        print(
                            f"\rDownloading {url}: {percent:3d}% "
                            f"({_format_bytes(downloaded)} / {_format_bytes(total)})",
                            end="",
                            file=sys.stderr,
                            flush=True,
                        )
                        last_percent = percent
                elif downloaded and downloaded % (BINARY_UNIT ** 2) < len(chunk):
                    print(
                        f"\rDownloading {url}: {_format_bytes(downloaded)}",
                        end="",
                        file=sys.stderr,
                        flush=True,
                    )
        if show_progress:
            size = f" / {_format_bytes(total)}" if total else ""
            # Clear the longer percentage line before writing the summary.
            summary = f"\r\x1b[KDownloaded {_format_bytes(downloaded)}{size}"
            print(summary, file=sys.stderr)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_archive_member(archive: str, name: str, target: str) -> None:
    """Copy one regular file out of the official tarball without extracting paths."""
    with tarfile.open(archive, mode="r:gz") as tf:
        for member in tf.getmembers():
            parts = [part for part in member.name.split("/") if part]
            if parts[-1:] != [name] or not member.isfile() or ".." in parts:
                continue
            source = tf.extractfile(member)
            if source is None:
                break
            with open(target, "wb") as output:
                shutil.copyfileobj(source, output)
            return
    raise RemoteError(f"Tailscale archive does not contain {name}")


def _install_tailscale() -> None:
    if sys.platform != "linux":
        raise RemoteError("automatic Tailscale installation currently supports Linux")
    archive_url, version = _tailscale_archive_url()
    message = f"Installing Tailscale {version} from the official static archive..."
    print(message, file=sys.stderr)
    with tempfile.TemporaryDirectory(prefix="agent7-tailscale-") as tmp:
        archive = os.path.join(tmp, "tailscale.tgz")
        checksum = os.path.join(tmp, "tailscale.tgz.sha256")
        _download(archive_url, archive)
        _download(f"{archive_url}.sha256", checksum, show_progress=False)
        checksum_text = Path(checksum).read_text(encoding="ascii")
        expected = re.search(r"\b([0-9a-fA-F]{64})\b", checksum_text)
        if expected is None:
            raise RemoteError("Tailscale returned an invalid sha256 checksum file")
        actual = _sha256_file(archive)
        if not hmac.compare_digest(actual, expected.group(1).lower()):
            message = "Tailscale archive checksum mismatch; refusing installation"
            raise RemoteError(message)

        files = {
            "tailscale": (os.path.join(tmp, "tailscale"), 0o755),
            "tailscaled": (os.path.join(tmp, "tailscaled"), 0o755),
            "tailscaled.service": (os.path.join(tmp, "tailscaled.service"), 0o644),
            "tailscaled.defaults": (os.path.join(tmp, "tailscaled.defaults"), 0o644),
        }
        for name, (target, mode) in files.items():
            _extract_archive_member(archive, name, target)
            os.chmod(target, mode)

        _require(
            ["install", "-m", "0755", files["tailscaled"][0], "/usr/sbin/tailscaled"],
            root=True, what="install tailscaled",
        )
        _require(
            ["install", "-m", "0755", files["tailscale"][0], "/usr/bin/tailscale"],
            root=True, what="install tailscale",
        )
        _require(
            ["install", "-m", "0644", files["tailscaled.service"][0],
             "/etc/systemd/system/tailscaled.service"],
            root=True, what="install the Tailscale service",
        )
        _require(
            ["install", "-D", "-m", "0644", files["tailscaled.defaults"][0],
             "/etc/default/tailscaled"],
            root=True, what="install Tailscale defaults",
        )
        _require(["systemctl", "daemon-reload"], root=True, what="reload systemd")


def _join_tailnet(auth_key: str) -> None:
    args = ["tailscale", "up"]
    if auth_key:
        message = "# Authenticating Tailscale with the provided auth key..."
        print(message, file=sys.stderr)
        args.append(f"--auth-key={auth_key}")
        _require(args, root=True, what="tailscale up")
    else:
        if not (sys.stdin.isatty() and sys.stderr.isatty()):
            raise RemoteError(
                "Tailscale is not logged in. Run `agent7 deploy` in a terminal, "
                "or pass --auth-key for unattended setup"
            )
        message = (
            "# Tailscale login required: open the URL shown below, "
            "then return here"
        )
        print(message, file=sys.stderr)
        # Keep stdio attached so the login URL and sudo prompt are visible.
        proc = _run(args, root=True, capture=False)
        if proc.returncode != 0:
            raise RemoteError("tailscale up failed")
    if _tailscale_node() is None:
        raise RemoteError("tailscale up finished, but this host is not in a tailnet")


def _ensure_tailscale_ssh() -> None:
    _require(["tailscale", "set", "--ssh"], root=True, what="enable Tailscale SSH")


def stop_tailscale() -> None:
    """Stop only the Tailscale daemon; preserve its binaries and login state."""
    unit = "tailscaled.service"
    if not _systemd_unit_exists(unit):
        raise RemoteError("Tailscale is not installed; run `agent7 deploy` first")
    if not _service_active(unit):
        print("tailscaled is already stopped")
        return
    _require(["systemctl", "stop", unit], root=True, what="stop tailscaled")
    if _service_active(unit):
        raise RemoteError("tailscaled did not stop; check: systemctl status tailscaled")
    print("tailscaled stopped; binaries and login state are preserved")
    print("start it again with: agent7 deploy")


def _install_openssh_server() -> None:
    if not _have("apt-get"):
        message = "automatic openssh-server installation currently requires apt"
        raise RemoteError(message)
    _require(["apt-get", "update", "-qq"], root=True, what="apt update", capture=False)
    install = [
        "env", "DEBIAN_FRONTEND=noninteractive", "apt-get",
        "install", "-y", "openssh-server",
    ]
    _require(install, root=True, what="install openssh-server", capture=False)


def _enable_openssh_server() -> None:
    unit = "ssh.service" if _systemd_unit_exists("ssh.service") else "sshd.service"
    _require(["systemctl", "enable", "--now", unit], root=True, what=f"start {unit}")
    if not _service_active(unit):
        raise RemoteError(f"{unit} did not start; check: journalctl -u {unit}")


def _print_step(step: Step) -> None:
    marks = {
        "ok": "✓",
        "changed": "+",
        "needed": "!",
        "unknown": "?",
        "failed": "✗",
    }
    detail = f" — {step.detail}" if step.detail else ""
    print(f"  {marks[step.state]} {step.name}{detail}")


def _openssh_present() -> bool:
    return _have("sshd") or os.path.exists("/usr/sbin/sshd")


def _openssh_active() -> bool:
    return _service_active("ssh.service") or _service_active("sshd.service")


def _tailscaled_present() -> bool:
    # A static install puts the daemon in /usr/sbin, which is not always on a
    # non-root user's PATH, so do not infer absence from `shutil.which` alone.
    return (
        _have("tailscaled")
        or os.path.exists("/usr/sbin/tailscaled")
        or _systemd_unit_exists("tailscaled.service")
    )


def _tailscale_steps(
    *, check: bool, auth_key: str,
) -> tuple[list[Step], TailscaleNode | None]:
    steps: list[Step] = []
    if _have("tailscale"):
        steps.append(Step("tailscale", "ok", "installed"))
    elif check:
        steps.append(Step("tailscale", "needed", "not installed"))
    else:
        _install_tailscale()
        steps.append(Step("tailscale", "changed", "installed"))

    if _tailscaled_present():
        if _service_active("tailscaled.service"):
            steps.append(Step("tailscaled", "ok", "running"))
        elif check:
            steps.append(Step("tailscaled", "needed", "service not running"))
        else:
            _require(["systemctl", "enable", "--now", "tailscaled.service"], root=True,
                     what="start tailscaled")
            steps.append(Step("tailscaled", "changed", "service running"))
    elif check:
        steps.append(Step("tailscaled", "needed", "not installed"))
    else:
        raise RemoteError("tailscaled is missing after installation")

    node = _tailscale_node()
    if node is not None:
        steps.append(Step("tailnet", "ok", node.dns_name or node.ip))
    elif check:
        steps.append(Step("tailnet", "needed", "not logged in"))
    else:
        _join_tailnet(auth_key)
        node = _tailscale_node()
        if node is None:
            raise RemoteError("could not read this host's Tailscale identity")
        steps.append(Step("tailnet", "changed", node.dns_name or node.ip))
    return steps, node


def _ssh_steps(*, check: bool, sshd: bool) -> list[Step]:
    if not sshd:
        enabled = _tailscale_ssh_enabled()
        if enabled is True:
            return [Step("tailscale ssh", "ok", "enabled")]
        if check:
            state: StepState = "unknown" if enabled is None else "needed"
            return [Step("tailscale ssh", state, "not confirmed")]
        _ensure_tailscale_ssh()
        return [Step("tailscale ssh", "changed", "enabled")]

    steps: list[Step] = []
    if _openssh_present():
        steps.append(Step("openssh-server", "ok", "installed"))
    elif check:
        steps.append(Step("openssh-server", "needed", "not installed"))
    else:
        _install_openssh_server()
        steps.append(Step("openssh-server", "changed", "installed"))
    if _openssh_active():
        steps.append(Step("sshd", "ok", "running"))
    elif check:
        steps.append(Step("sshd", "needed", "service not running"))
    else:
        _enable_openssh_server()
        steps.append(Step("sshd", "changed", "running"))
    return steps


def deploy_mobile_ssh(
    *, check: bool = False, sshd: bool = False, auth_key: str = "",
) -> int:
    """Prepare this machine for Herdr Mobile over Tailscale.

    Returns 0 when ready. In check mode returns 0 only when every required
    component is already installed, running, joined, and exposing SSH.
    """
    tailscale_steps, node = _tailscale_steps(check=check, auth_key=auth_key)
    steps = [*tailscale_steps, *_ssh_steps(check=check, sshd=sshd)]

    print("\n# agent7 deploy — mobile SSH")
    for step in steps:
        _print_step(step)

    ready = all(step.state == "ok" for step in steps)
    if check:
        if ready:
            print("\nready")
        else:
            print("\nnot ready — run: agent7 deploy")
        return 0 if ready else 1

    if node is None:
        raise RemoteError("Tailscale identity is unavailable")
    host = node.dns_name or node.ip
    if not host:
        raise RemoteError("Tailscale is running, but did not assign a host identity")
    backend = "OpenSSH server" if sshd else "Tailscale SSH"
    user = os.environ.get("USER") or os.environ.get("USERNAME")
    if not user and pwd is not None:
        user = pwd.getpwuid(os.getuid()).pw_name
    print(f"\nready: host={host} user={user or 'unknown'} ssh={backend}")
    print("\nOn your phone:")
    print("  1. Install Tailscale and sign in to the same account.")
    print(f"  2. In Herdr Mobile, add this machine as `{host}`.")
    print("  3. Start agent7 in a herdr-core terminal; it appears in the Inbox.")
    if not sshd:
        print("\nIf SSH is denied later, check the tailnet's Access controls for SSH.")
    return 0
