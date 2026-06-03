from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

import mn_protein_design
from mn_protein_design.runtime import runs_root


def _page(package_root: Path, relative: str, title: str, slug: str, default: bool = False, hidden: bool = False) -> st.Page:
    kwargs = {"page": str(package_root / relative), "title": title, "url_path": slug}
    if default:
        kwargs["default"] = True
    if hidden:
        kwargs["visibility"] = "hidden"
    return st.Page(**kwargs)


def main() -> None:
    package_root = Path(mn_protein_design.__file__).resolve().parent
    icon_path = package_root / "app" / "assets" / "icon.png"

    st.set_page_config(page_title="mn-protein-design", page_icon=str(icon_path), layout="wide")
    st.sidebar.title("mn-protein-design")
    st.sidebar.caption("Docker-backed protein design workbench")

    pg = st.navigation(
        {
            "Jobs": [
                _page(package_root, "app/pages/jobs.py", "Jobs", "jobs", default=True),
            ],
            "Tasks": [
                _page(package_root, "app/pages/target_preparation.py", "Target Preparation", "target-preparation"),
                _page(package_root, "app/pages/ppi_detection.py", "PPI / Hotspot Detection", "ppi-hotspot-detection"),
                _page(package_root, "app/pages/target_cropping.py", "Target Cropping", "target-cropping"),
                _page(package_root, "app/pages/design.py", "Design", "design"),
                _page(package_root, "app/pages/sequence_design.py", "Sequence Design", "sequence-design"),
                _page(package_root, "app/pages/refolding.py", "Refolding / Validation", "refolding-validation"),
                _page(package_root, "app/pages/analysis.py", "Analysis", "analysis"),
                _page(package_root, "app/pages/benchmark.py", "Binder Benchmark", "binder-benchmark"),
            ],
            "": [
                _page(package_root, "app/pages/results.py", "Result Details", "results", hidden=True),
            ],
        },
        position="sidebar",
    )
    st.sidebar.divider()
    st.sidebar.caption(f"Runs: {os.getenv('MN_PROTEIN_DESIGN_RUN_DIR', str(runs_root()))}")
    pg.run()


if __name__ == "__main__":
    main()
