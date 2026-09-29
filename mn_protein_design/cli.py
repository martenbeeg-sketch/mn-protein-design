from __future__ import annotations

import os
import runpy
import signal
import socket
import sys
from pathlib import Path

import typer

from mn_protein_design.runtime import DEFAULT_APP_HOME, DEFAULT_TMPDIR, ensure_runtime_home, reference_root


app = typer.Typer(
    pretty_exceptions_enable=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=True,
)
portability_app = typer.Typer(
    help="Audit, export, verify, and import workdir data without modifying the source runtime.",
    no_args_is_help=True,
)
campaign_app = typer.Typer(
    help="Submit local design campaigns through the app's normal scheduler.",
    no_args_is_help=True,
)
jobs_app = typer.Typer(
    help="Inspect, wait for, and cancel local app jobs.",
    no_args_is_help=True,
)
workflow_app = typer.Typer(
    help="Run allowlisted app workflows from versioned JSON requests.",
    no_args_is_help=True,
)
app.add_typer(portability_app, name="portability")
app.add_typer(campaign_app, name="campaign")
app.add_typer(jobs_app, name="jobs")
app.add_typer(workflow_app, name="workflow")


@app.callback()
def cli() -> None:
    """Command line helpers for the standalone mn-protein-design app."""


@app.command(name="init")
def init_home(
    app_home: str = typer.Option(str(DEFAULT_APP_HOME), "--app-home", help="Runtime directory for jobs and app state."),
    tmpdir: str = typer.Option(str(DEFAULT_TMPDIR), "--tmpdir", help="Writable temporary directory for Streamlit startup."),
    reference_dir: str | None = typer.Option(None, "--reference-dir", help="Shared model/reference directory."),
    runs_dir: str | None = typer.Option(None, "--runs-dir", help="Job result directory; defaults to APP_HOME/workdir/runs."),
) -> None:
    """Initialize the file-backed runtime directory."""
    home_path = ensure_runtime_home(app_home, tmpdir)
    if reference_dir:
        os.environ["MN_PROTEIN_DESIGN_REFERENCE_DIR"] = str(Path(reference_dir).expanduser().resolve())
    if runs_dir:
        runs_path = Path(runs_dir).expanduser().resolve()
        runs_path.mkdir(parents=True, exist_ok=True)
        os.environ["MN_PROTEIN_DESIGN_RUN_DIR"] = str(runs_path)
    reference_path = reference_root()
    reference_path.mkdir(parents=True, exist_ok=True)
    typer.echo(f"Initialized mn-protein-design runtime directory: {home_path}")
    typer.echo(f"Reference directory: {reference_path}")


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
    reference_dir: str | None = typer.Option(None, "--reference-dir", help="Shared model/reference directory."),
    runs_dir: str | None = typer.Option(None, "--runs-dir", help="Job result directory; defaults to APP_HOME/workdir/runs."),
    port: int = typer.Option(8501, "--port", help="First port to try when choosing a free Streamlit port."),
    port_max: int = typer.Option(8599, "--port-max", help="Highest port to try when choosing a free Streamlit port."),
) -> None:
    """Run the Streamlit app."""
    home_path = ensure_runtime_home(app_home, tmpdir)
    os.environ.setdefault("TMPDIR", str(Path(tmpdir).expanduser().resolve()))
    os.environ.setdefault("MN_PROTEIN_DESIGN_APP_HOME", str(home_path))
    if runs_dir:
        os.environ["MN_PROTEIN_DESIGN_RUN_DIR"] = str(Path(runs_dir).expanduser().resolve())
        Path(os.environ["MN_PROTEIN_DESIGN_RUN_DIR"]).mkdir(parents=True, exist_ok=True)
    else:
        os.environ.setdefault("MN_PROTEIN_DESIGN_RUN_DIR", str(home_path / "workdir" / "runs"))
    if reference_dir:
        os.environ["MN_PROTEIN_DESIGN_REFERENCE_DIR"] = str(Path(reference_dir).expanduser().resolve())
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    reference_root().mkdir(parents=True, exist_ok=True)

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


@app.command(name="install-launchers")
def install_launchers(
    desktop: bool = typer.Option(
        True,
        "--desktop/--no-desktop",
        help="Also create a desktop shortcut.",
    ),
    bin_dir: Path | None = typer.Option(
        None,
        "--bin-dir",
        help="Wrapper destination; defaults to ~/.local/bin.",
    ),
    desktop_dir: Path | None = typer.Option(
        None,
        "--desktop-dir",
        help="Desktop shortcut destination; defaults to ~/Desktop.",
    ),
    app_home: Path | None = typer.Option(
        None,
        "--app-home",
        help="Runtime data directory to pin into the app launcher.",
    ),
    runs_dir: Path | None = typer.Option(
        None,
        "--runs-dir",
        help="Job/result directory to pin into the app launcher.",
    ),
    reference_dir: Path | None = typer.Option(
        None,
        "--reference-dir",
        help="Reference/model directory to pin into the app launcher.",
    ),
    tmpdir: Path | None = typer.Option(
        None,
        "--tmpdir",
        help="Writable temporary directory to pin into the app launcher.",
    ),
) -> None:
    """Install activation-free command and desktop app launchers for this user."""
    from mn_protein_design.launchers import install_user_launchers

    try:
        installed = install_user_launchers(
            bin_dir=bin_dir,
            desktop_dir=desktop_dir,
            create_desktop=desktop,
            app_home=app_home,
            runs_dir=runs_dir,
            reference_dir=reference_dir,
            tmpdir=tmpdir,
        )
    except (OSError, RuntimeError) as exc:
        typer.echo(f"Launcher installation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Command launcher: {installed['cli_wrapper']}")
    typer.echo(f"One-command app launcher: {installed['app_wrapper']}")
    if installed["desktop_launcher"]:
        typer.echo(f"Desktop shortcut: {installed['desktop_launcher']}")
    typer.echo("Runtime paths pinned into the app launcher:")
    for name, path in installed["runtime_paths"].items():
        typer.echo(f"  {name}: {path}")
    typer.echo(f"Start the app with: {installed['app_wrapper']}")


def _portability_roots_options(
    app_home: str | None,
    runs_dir: str | None,
    reference_dir: str | None,
) -> dict[str, Path | None]:
    return {
        "app_home_path": Path(app_home).expanduser() if app_home else None,
        "runs_path": Path(runs_dir).expanduser() if runs_dir else None,
        "references_path": Path(reference_dir).expanduser() if reference_dir else None,
    }


@portability_app.command(name="audit")
def portability_audit(
    app_home: str | None = typer.Option(None, "--app-home", help="Runtime app-home to audit."),
    runs_dir: str | None = typer.Option(None, "--runs-dir", help="Effective runs directory."),
    reference_dir: str | None = typer.Option(None, "--reference-dir", help="Reference root used to map stored references."),
    json_output: bool = typer.Option(False, "--json", help="Print the full read-only audit as JSON."),
) -> None:
    """Audit workdir path portability and report data size without changing it."""
    import json

    from mn_protein_design.core.portability import PortabilityError, audit_runtime_portability

    try:
        report = audit_runtime_portability(**_portability_roots_options(app_home, runs_dir, reference_dir))
    except (OSError, PortabilityError) as exc:
        typer.echo(f"Portability audit failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(json.dumps(report, indent=2, sort_keys=True))
        return
    typer.echo(f"Workdir: {report['workdir']}")
    typer.echo(f"Runs: {report['runs_dir']}")
    typer.echo(f"Estimated workdir + runs size: {report['estimated_bytes']['workdir_and_runs']:,} bytes")
    typer.echo(f"Active jobs: {len(report['active_jobs'])}")
    typer.echo(f"Absolute operational paths: {report['counts'].get('absolute_operational_paths', 0)}")
    typer.echo(f"Paths into separately managed references: {report['counts'].get('reference_root_paths', 0)}")
    typer.echo(f"Other external paths: {report['counts'].get('external_paths', 0)}")
    typer.echo("Reference files and Docker images are excluded from the workdir export.")


@portability_app.command(name="export")
def portability_export(
    destination: Path = typer.Argument(..., help="New directory where the portable workdir bundle will be created."),
    app_home: str | None = typer.Option(None, "--app-home", help="Source app-home."),
    runs_dir: str | None = typer.Option(None, "--runs-dir", help="Source runs directory."),
    reference_dir: str | None = typer.Option(None, "--reference-dir", help="Reference root for path mapping; it is not copied."),
    json_output: bool = typer.Option(False, "--json", help="Print the export summary as JSON."),
) -> None:
    """Copy all workdir data and configured runs into a verified bundle."""
    import json

    from mn_protein_design.core.portability import PortabilityError, export_portable_workdir

    try:
        report = export_portable_workdir(destination, **_portability_roots_options(app_home, runs_dir, reference_dir))
    except (OSError, PortabilityError) as exc:
        typer.echo(f"Portable export failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        typer.echo(f"Portable workdir bundle created: {report['destination']}")
        typer.echo(f"Rewritten paths: {report['counts']['rewritten_paths']}")
        typer.echo("Source workdir remains unchanged. Run portability verify before transfer.")


@portability_app.command(name="verify")
def portability_verify(
    bundle: Path = typer.Argument(..., help="Portable workdir bundle to verify."),
    json_output: bool = typer.Option(False, "--json", help="Print verification details as JSON."),
) -> None:
    """Verify the bundle's file checksums and portable paths."""
    import json

    from mn_protein_design.core.portability import verify_portable_export

    report = verify_portable_export(bundle)
    if json_output:
        typer.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        typer.echo(f"Bundle verification: {'valid' if report['valid'] else 'FAILED'}")
        typer.echo(f"Files checked: {report['counts'].get('files', 0)}")
        typer.echo(f"External reference paths: {report['counts'].get('external_reference_paths', 0)}")
        for error in report["errors"][:20]:
            typer.echo(f"- {error['file']}: {error['error']}", err=True)
    if not report["valid"]:
        raise typer.Exit(code=1)


@portability_app.command(name="import")
def portability_import(
    bundle: Path = typer.Argument(..., help="Verified portable workdir bundle."),
    app_home: Path = typer.Option(..., "--app-home", help="New destination app-home. It must not already exist."),
    json_output: bool = typer.Option(False, "--json", help="Print import paths as JSON."),
) -> None:
    """Restore a bundle into a new app-home without overwriting an existing installation."""
    import json

    from mn_protein_design.core.portability import PortabilityError, import_portable_workdir

    try:
        report = import_portable_workdir(bundle, app_home)
    except (OSError, PortabilityError) as exc:
        typer.echo(f"Portable import failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(json.dumps(report, indent=2, sort_keys=True))
        return
    typer.echo("Portable workdir imported:")
    typer.echo(f"  app home: {report['app_home']}")
    typer.echo(f"  workdir:  {report['workdir']}")
    typer.echo(f"  runs:     {report['runs_dir']}")
    typer.echo(f"  references: {report['reference_dir']}")
    typer.echo("Launch the app with --app-home and --runs-dir. Set --reference-dir to the separately copied reference files.")


@app.command(name="worker")
def run_worker(
    cpu_slots: int | None = typer.Option(
        None,
        "--cpu-slots",
        min=1,
        help="CPU slots to allocate across local jobs (default: up to 32, or the detected CPU count when lower).",
    ),
    poll_seconds: float = typer.Option(
        2.0,
        "--poll-seconds",
        min=0.25,
        help="How often the worker checks the file-backed job queue.",
    ),
) -> None:
    """Run the persistent local job scheduler in the foreground."""
    from mn_protein_design.core.scheduler import run_worker_service

    run_worker_service(cpu_slots=cpu_slots, poll_seconds=poll_seconds)


@app.command(name="worker-status")
def worker_status() -> None:
    """Show the local worker service health snapshot."""
    from mn_protein_design.core.scheduler import scheduler_snapshot, service_owner_pid

    snapshot = scheduler_snapshot()
    pid = service_owner_pid()
    alive = False
    if pid:
        try:
            os.kill(pid, 0)
            alive = True
        except PermissionError:
            alive = True
        except ProcessLookupError:
            alive = False
    typer.echo(f"Worker service: {'running' if alive else 'stopped'}" + (f" (PID {pid})" if alive else ""))
    if snapshot:
        typer.echo(f"CPU slots: {snapshot.get('cpu_slots', 'unknown')}")
        typer.echo(f"Active jobs: {len(snapshot.get('active_workers') or [])}")
        typer.echo(f"Heartbeat: {snapshot.get('heartbeat_at', 'unknown')}")


@app.command(name="worker-stop")
def stop_worker() -> None:
    """Stop the queue service; already running jobs continue to completion."""
    from mn_protein_design.core.scheduler import service_owner_pid

    pid = service_owner_pid()
    if not pid:
        typer.echo("Worker service is not running.")
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        typer.echo("Worker service is not running.")
        return
    typer.echo(f"Sent a stop request to worker service PID {pid}. Active jobs will continue.")


def _automation_error(exc: Exception) -> None:
    typer.echo(f"Automation command failed: {exc}", err=True)
    raise typer.Exit(code=1) from exc


@campaign_app.command(name="validate")
def validate_campaign_request(
    request: Path = typer.Option(..., "--request", "-r", exists=True, dir_okay=False, readable=True),
) -> None:
    """Validate a campaign request JSON file without creating a job."""
    import json

    from mn_protein_design.services.local_automation import AutomationError, load_campaign_request

    try:
        normalized = load_campaign_request(request)
    except (AutomationError, OSError, TypeError, ValueError) as exc:
        _automation_error(exc)
    normalized["target_pdb"] = str(normalized["target_pdb"])
    typer.echo(json.dumps({"valid": True, "request": normalized}, indent=2, sort_keys=True))


@campaign_app.command(name="submit")
def submit_campaign_request(
    request: Path = typer.Option(..., "--request", "-r", exists=True, dir_okay=False, readable=True),
) -> None:
    """Queue a design campaign from a versioned JSON request."""
    import json

    from mn_protein_design.services.local_automation import AutomationError, submit_design_campaign_request

    try:
        job = submit_design_campaign_request(request)
    except (AutomationError, OSError, RuntimeError, TypeError, ValueError) as exc:
        _automation_error(exc)
    typer.echo(json.dumps(job, indent=2, sort_keys=True))


@jobs_app.command(name="list")
def list_local_jobs(
    task_group: str | None = typer.Option(None, "--task-group", help="Limit results to one task group."),
    status: str | None = typer.Option(None, "--status", help="Filter by exact status, such as running or completed."),
    limit: int = typer.Option(100, "--limit", min=1, max=10000, help="Maximum number of jobs to return."),
    include_hidden: bool = typer.Option(False, "--include-hidden", help="Include internal child jobs hidden in the app list."),
) -> None:
    """Print local jobs as JSON for scripts and command-line inspection."""
    import json

    from mn_protein_design.services.local_automation import AutomationError, list_jobs

    try:
        rows = list_jobs(task_group=task_group, status=status, limit=limit, include_hidden=include_hidden)
    except (AutomationError, OSError, TypeError, ValueError) as exc:
        _automation_error(exc)
    typer.echo(json.dumps(rows, indent=2, sort_keys=True))


@jobs_app.command(name="show")
def show_local_job(
    reference: str = typer.Argument(..., help="Job code, run ID, or task-group/run-ID."),
    include_hidden: bool = typer.Option(False, "--include-hidden", help="Allow lookup of an internal child job."),
) -> None:
    """Print a job's status, inputs, result, and candidate artifact location."""
    import json

    from mn_protein_design.services.local_automation import AutomationError, inspect_job

    try:
        details = inspect_job(reference, include_hidden=include_hidden)
    except (AutomationError, OSError, TypeError, ValueError) as exc:
        _automation_error(exc)
    typer.echo(json.dumps(details, indent=2, sort_keys=True))


@jobs_app.command(name="results")
def show_local_job_results(
    reference: str = typer.Argument(..., help="Job code, run ID, or task-group/run-ID."),
    include_candidates: bool = typer.Option(False, "--include-candidates", help="Include normalized candidate records in the JSON output."),
    include_hidden: bool = typer.Option(False, "--include-hidden", help="Allow lookup of an internal child job."),
) -> None:
    """Print result metadata and, optionally, normalized candidate records."""
    import json

    from mn_protein_design.services.local_automation import AutomationError, job_results

    try:
        result = job_results(
            reference,
            include_candidates=include_candidates,
            include_hidden=include_hidden,
        )
    except (AutomationError, OSError, TypeError, ValueError) as exc:
        _automation_error(exc)
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@jobs_app.command(name="wait")
def wait_for_local_job(
    reference: str = typer.Argument(..., help="Job code, run ID, or task-group/run-ID."),
    timeout: float = typer.Option(3600, "--timeout", min=0, help="Maximum wait in seconds; 0 checks once."),
    poll_seconds: float = typer.Option(2, "--poll-seconds", min=0.05, help="Status polling interval."),
    include_hidden: bool = typer.Option(False, "--include-hidden", help="Allow lookup of an internal child job."),
) -> None:
    """Wait for a job to finish; exit 0 only when it completes successfully."""
    import json

    from mn_protein_design.services.local_automation import AutomationError, wait_for_job

    try:
        row = wait_for_job(
            reference,
            timeout_seconds=timeout,
            poll_seconds=poll_seconds,
            include_hidden=include_hidden,
        )
    except (AutomationError, OSError, TypeError, ValueError) as exc:
        _automation_error(exc)
    if row is None:
        typer.echo(json.dumps({"job": reference, "status": "timeout"}, sort_keys=True))
        raise typer.Exit(code=2)
    typer.echo(json.dumps(row, indent=2, sort_keys=True))
    if row["status"] != "completed" or row.get("success") is False:
        raise typer.Exit(code=1)


@jobs_app.command(name="cancel")
def cancel_local_job(
    reference: str = typer.Argument(..., help="Job code, run ID, or task-group/run-ID."),
    reason: str = typer.Option("Cancelled through local automation CLI", "--reason", help="Reason recorded in the job metadata."),
) -> None:
    """Cancel a queued, running, or paused job while preserving its artifacts."""
    import json

    from mn_protein_design.services.local_automation import AutomationError, cancel_job

    try:
        row = cancel_job(reference, reason=reason)
    except (AutomationError, OSError, RuntimeError, TypeError, ValueError) as exc:
        _automation_error(exc)
    typer.echo(json.dumps(row, indent=2, sort_keys=True))


@workflow_app.command(name="list")
def list_automation_workflows(
    category: str | None = typer.Option(
        None,
        "--category",
        help="Filter by target, detection, design, sequence, refolding, analysis, import, campaign, or benchmark.",
    ),
) -> None:
    """List supported workflow IDs and their accepted parameters as JSON."""
    import json

    from mn_protein_design.services.workflow_automation import workflow_catalog

    typer.echo(json.dumps(workflow_catalog(category=category), indent=2, sort_keys=True))


@workflow_app.command(name="schema")
def show_workflow_schema(workflow_id: str = typer.Argument(..., help="Workflow ID from `workflow list`.")) -> None:
    """Show the request fields for one supported workflow."""
    import json

    from mn_protein_design.services.local_automation import AutomationError
    from mn_protein_design.services.workflow_automation import workflow_schema

    try:
        schema = workflow_schema(workflow_id)
    except AutomationError as exc:
        _automation_error(exc)
    typer.echo(json.dumps(schema, indent=2, sort_keys=True))


@workflow_app.command(name="validate")
def validate_workflow_request_file(
    request: Path = typer.Option(..., "--request", "-r", exists=True, dir_okay=False, readable=True),
) -> None:
    """Validate workflow inputs and resources without creating a job."""
    import json

    from mn_protein_design.services.local_automation import AutomationError
    from mn_protein_design.services.workflow_automation import validate_workflow_request

    try:
        report = validate_workflow_request(request)
    except (AutomationError, OSError, TypeError, ValueError) as exc:
        _automation_error(exc)
    typer.echo(json.dumps(report, indent=2, sort_keys=True))


@workflow_app.command(name="submit")
def submit_workflow_request_file(
    request: Path = typer.Option(..., "--request", "-r", exists=True, dir_okay=False, readable=True),
) -> None:
    """Submit an allowlisted workflow through the local worker/job system."""
    import json

    from mn_protein_design.services.local_automation import AutomationError
    from mn_protein_design.services.workflow_automation import submit_workflow_request

    try:
        result = submit_workflow_request(request)
    except (AutomationError, OSError, RuntimeError, TypeError, ValueError) as exc:
        _automation_error(exc)
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
