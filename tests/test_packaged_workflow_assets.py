from __future__ import annotations

from mn_protein_design.workflows import design


def test_bindcraft_defaults_are_packaged_with_the_app() -> None:
    advanced = design.BINDCRAFT_RESOURCE_DIR / "settings_advanced"
    filters = design.BINDCRAFT_RESOURCE_DIR / "settings_filters"

    assert (advanced / "default_4stage_multimer.json").is_file()
    assert (filters / "default_filters.json").is_file()


def test_rfdiffusion_scaffold_archive_is_packaged_with_the_app() -> None:
    assert design.RFDIFFUSION_BUNDLED_SCAFFOLD_TAR.is_file()
