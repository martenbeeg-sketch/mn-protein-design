from __future__ import annotations

import os
import subprocess
from pathlib import Path

from typer.testing import CliRunner

from mn_protein_design.cli import app
from mn_protein_design.launchers import install_user_launchers


def _fake_cli(tmp_path: Path) -> Path:
    executable = tmp_path / "active env" / "bin" / "mn-protein-design"
    executable.parent.mkdir(parents=True)
    executable.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$@\" > \"$MN_PROTEIN_DESIGN_TEST_ARGS\"\n"
    )
    executable.chmod(0o755)
    return executable


def test_install_user_launchers_pins_runtime_paths_and_creates_desktop_shortcut(
    tmp_path: Path, monkeypatch
) -> None:
    executable = _fake_cli(tmp_path)
    monkeypatch.setenv("MN_PROTEIN_DESIGN_CLI", str(executable))
    monkeypatch.setenv("MN_PROTEIN_DESIGN_TEST_ARGS", str(tmp_path / "args.txt"))
    monkeypatch.setattr("mn_protein_design.launchers.shutil.which", lambda _: None)
    paths = {
        "app_home": tmp_path / "runtime home",
        "runs_dir": tmp_path / "runs folder",
        "reference_dir": tmp_path / "reference files",
        "tmpdir": tmp_path / "temp files",
    }

    installed = install_user_launchers(
        bin_dir=tmp_path / "local bin",
        desktop_dir=tmp_path / "Desktop",
        **paths,
    )

    cli_wrapper = Path(installed["cli_wrapper"])
    app_wrapper = Path(installed["app_wrapper"])
    assert cli_wrapper.stat().st_mode & 0o111
    assert app_wrapper.stat().st_mode & 0o111
    assert subprocess.run(["sh", "-n", str(app_wrapper)], check=False).returncode == 0
    assert installed["runtime_paths"] == {key: str(value) for key, value in paths.items()}
    assert subprocess.run(
        [str(app_wrapper), "--server.port", "8600"],
        env=os.environ.copy(),
        check=True,
    ).returncode == 0
    actual_args = (tmp_path / "args.txt").read_text().splitlines()
    assert actual_args == [
        "app",
        "--app-home",
        str(paths["app_home"].resolve()),
        "--runs-dir",
        str(paths["runs_dir"].resolve()),
        "--reference-dir",
        str(paths["reference_dir"].resolve()),
        "--tmpdir",
        str(paths["tmpdir"].resolve()),
        "--server.port",
        "8600",
    ]
    desktop = Path(installed["desktop_launcher"])
    assert desktop.stat().st_mode & 0o111
    desktop_text = desktop.read_text()
    assert f"Exec={app_wrapper}" in desktop_text
    assert "Name=MN Protein Design" in desktop_text
    assert "Terminal=true" in desktop_text


def test_install_user_launchers_can_skip_desktop_and_use_current_runtime(
    tmp_path: Path, monkeypatch
) -> None:
    executable = _fake_cli(tmp_path)
    monkeypatch.setenv("MN_PROTEIN_DESIGN_CLI", str(executable))
    monkeypatch.setenv("MN_PROTEIN_DESIGN_APP_HOME", str(tmp_path / "active home"))
    monkeypatch.setenv("MN_PROTEIN_DESIGN_RUN_DIR", str(tmp_path / "active runs"))
    monkeypatch.setenv("MN_PROTEIN_DESIGN_REFERENCE_DIR", str(tmp_path / "active references"))
    monkeypatch.setenv("TMPDIR", str(tmp_path / "active temp"))

    installed = install_user_launchers(
        bin_dir=tmp_path / "bin",
        create_desktop=False,
    )

    assert installed["desktop_launcher"] == ""
    assert installed["runtime_paths"] == {
        "app_home": str((tmp_path / "active home").resolve()),
        "runs_dir": str((tmp_path / "active runs").resolve()),
        "reference_dir": str((tmp_path / "active references").resolve()),
        "tmpdir": str((tmp_path / "active temp").resolve()),
    }


def test_install_launchers_cli_can_skip_desktop(tmp_path: Path, monkeypatch) -> None:
    executable = _fake_cli(tmp_path)
    monkeypatch.setenv("MN_PROTEIN_DESIGN_CLI", str(executable))

    result = CliRunner().invoke(
        app,
        ["install-launchers", "--no-desktop", "--bin-dir", str(tmp_path / "bin")],
    )

    assert result.exit_code == 0, result.output
    assert f"Start the app with: {tmp_path / 'bin' / 'mn-protein-design-app'}" in result.output
    assert "Desktop shortcut:" not in result.output
