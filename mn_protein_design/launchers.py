from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _atomic_text(path: Path, text: str, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    temporary.chmod(mode)
    temporary.replace(path)


def _environment_cli() -> Path:
    override = os.getenv("MN_PROTEIN_DESIGN_CLI", "").strip()
    candidate = (
        Path(override).expanduser()
        if override
        else Path(sys.executable).with_name("mn-protein-design")
    ).resolve()
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise FileNotFoundError(
            f"mn-protein-design executable was not found beside the active Python: {candidate}. "
            "Install the project in this environment with `python -m pip install -e .`."
        )
    return candidate


def install_user_launchers(
    *,
    bin_dir: Path | None = None,
    desktop_dir: Path | None = None,
    create_desktop: bool = True,
    app_home: Path | None = None,
    runs_dir: Path | None = None,
    reference_dir: Path | None = None,
    tmpdir: Path | None = None,
) -> dict[str, Any]:
    """Install activation-free CLI and app launchers for the current user."""
    from mn_protein_design.runtime import (
        DEFAULT_TMPDIR,
        app_home as active_app_home,
        reference_root,
        runs_root,
    )

    cli = _environment_cli()
    if app_home is None:
        selected_app_home = active_app_home()
    else:
        selected_app_home = Path(app_home).expanduser().resolve()
    if runs_dir is None:
        selected_runs = (
            selected_app_home / "workdir" / "runs"
            if app_home is not None
            else runs_root()
        )
    else:
        selected_runs = Path(runs_dir).expanduser().resolve()
    selected_reference = (
        Path(reference_dir).expanduser().resolve()
        if reference_dir is not None
        else reference_root()
    )
    selected_tmpdir = (
        Path(tmpdir).expanduser().resolve()
        if tmpdir is not None
        else Path(os.getenv("TMPDIR", str(DEFAULT_TMPDIR))).expanduser().resolve()
    )

    selected_bin = Path(bin_dir or Path.home() / ".local" / "bin").expanduser().resolve()
    cli_wrapper = selected_bin / "mn-protein-design"
    app_wrapper = selected_bin / "mn-protein-design-app"
    quoted_cli = shlex.quote(str(cli))
    _atomic_text(
        cli_wrapper,
        f"#!/bin/sh\nset -eu\nexec {quoted_cli} \"$@\"\n",
        mode=0o755,
    )

    runtime_args = [
        "--app-home",
        str(selected_app_home),
        "--runs-dir",
        str(selected_runs),
        "--reference-dir",
        str(selected_reference),
        "--tmpdir",
        str(selected_tmpdir),
    ]
    quoted_runtime_args = " ".join(shlex.quote(value) for value in runtime_args)
    _atomic_text(
        app_wrapper,
        "#!/bin/sh\nset -eu\n"
        f"exec {quoted_cli} app {quoted_runtime_args} \"$@\"\n",
        mode=0o755,
    )

    desktop_path: Path | None = None
    if create_desktop:
        selected_desktop = Path(
            desktop_dir or Path.home() / "Desktop"
        ).expanduser().resolve()
        desktop_path = selected_desktop / "mn-protein-design.desktop"
        _atomic_text(
            desktop_path,
            "\n".join(
                (
                    "[Desktop Entry]",
                    "Type=Application",
                    "Version=1.0",
                    "Name=MN Protein Design",
                    "Comment=Start the MN Protein Design workbench",
                    f"Exec={app_wrapper}",
                    "Icon=applications-science",
                    "Terminal=true",
                    "Categories=Science;Education;",
                    "StartupNotify=true",
                    "",
                )
            ),
            mode=0o755,
        )
        gio = shutil.which("gio")
        if gio:
            subprocess.run(
                [gio, "set", str(desktop_path), "metadata::trusted", "true"],
                capture_output=True,
                check=False,
                text=True,
            )

    return {
        "environment_cli": str(cli),
        "cli_wrapper": str(cli_wrapper),
        "app_wrapper": str(app_wrapper),
        "desktop_launcher": str(desktop_path) if desktop_path else "",
        "runtime_paths": {
            "app_home": str(selected_app_home),
            "runs_dir": str(selected_runs),
            "reference_dir": str(selected_reference),
            "tmpdir": str(selected_tmpdir),
        },
    }
