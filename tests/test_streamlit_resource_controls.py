from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_cpu_resource_control_uses_configured_worker_capacity(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MN_PROTEIN_DESIGN_WORKER_CPU_SLOTS", "8")
    page = tmp_path / "resource_control_app.py"
    page.write_text(
        "from mn_protein_design.app.pages.common import cpu_run_panel\n"
        "reserved = cpu_run_panel(key='test_resource', default=4)\n"
    )

    app = AppTest.from_file(str(page)).run()

    assert not app.exception
    assert app.selectbox[0].label == "Reserved CPU slots"
    assert list(app.selectbox[0].options) == ["1", "2", "4", "8"]
    assert app.selectbox[0].value == 4
    app.selectbox[0].set_value(8).run()
    assert app.selectbox[0].value == 8
