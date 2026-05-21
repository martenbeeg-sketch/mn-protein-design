from __future__ import annotations

import os
import runpy
import socket
import sys
from pathlib import Path

import typer

from mn_protein_design.runtime import DEFAULT_APP_HOME, DEFAULT_TMPDIR, ensure_runtime_home


app = typer.Typer(
    pretty_exceptions_enable=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=True,
)


@app.callback()
def cli() -> None:
    """Command line helpers for the standalone mn-protein-design app."""


@app.command(name="init")
def init_home(
    app_home: str = typer.Option(str(DEFAULT_APP_HOME), "--app-home", help="Runtime directory for jobs and app state."),
    tmpdir: str = typer.Option(str(DEFAULT_TMPDIR), "--tmpdir", help="Writable temporary directory for Streamlit startup."),
) -> None:
    """Initialize the file-backed runtime directory."""
    home_path = ensure_runtime_home(app_home, tmpdir)
    typer.echo(f"Initialized mn-protein-design runtime directory: {home_path}")


def _has_streamlit_port(args: list[str]) -> bool:
    return any(arg == "--server.port" or arg.startswith("--server.port=") for arg in args)


def _first_free_port(start: int = 8501, stop: int = 8599) -> int:
    for port in range(start, stop + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"No free port found from {start} to {stop}.")


@app.command(name="app", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run_app(
    ctx: typer.Context,
    app_home: str = typer.Option(str(DEFAULT_APP_HOME), "--app-home", help="Runtime directory for jobs and app state."),
    tmpdir: str = typer.Option(str(DEFAULT_TMPDIR), "--tmpdir", help="Writable temporary directory for Streamlit startup."),
    port: int = typer.Option(8501, "--port", help="First port to try when choosing a free Streamlit port."),
    port_max: int = typer.Option(8599, "--port-max", help="Highest port to try when choosing a free Streamlit port."),
) -> None:
    """Run the Streamlit app."""
    home_path = ensure_runtime_home(app_home, tmpdir)
    os.environ.setdefault("TMPDIR", str(Path(tmpdir).expanduser().resolve()))
    os.environ.setdefault("MN_PROTEIN_DESIGN_APP_HOME", str(home_path))
    os.environ.setdefault("MN_PROTEIN_DESIGN_RUN_DIR", str(home_path / "workdir" / "runs"))
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)

    streamlit_script_path = Path(__file__).resolve().parent / "run_app.py"
    sys.argv = ["streamlit", "run", str(streamlit_script_path)]
    sys.argv.extend(["--browser.gatherUsageStats", "0"])
    sys.argv.extend(["--server.showEmailPrompt", "0"])
    sys.argv.extend(["--logger.enableRich", "0"])
    extra_args = list(ctx.args)
    if not _has_streamlit_port(extra_args):
        selected_port = _first_free_port(port, port_max)
        typer.echo(f"Starting mn-protein-design on http://localhost:{selected_port}")
        sys.argv.extend(["--server.port", str(selected_port)])
    sys.argv.extend(extra_args)
    runpy.run_module("streamlit", run_name="__main__")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
