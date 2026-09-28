from __future__ import annotations

from dataclasses import asdict, dataclass, field

from mn_protein_design.core.candidates import (
    STAGE_ANALYSIS,
    STAGE_COMPLEX_REFOLDING,
    STAGE_BENCHMARK,
    STAGE_GENERATION_BACKBONE,
    STAGE_GENERATION_BACKBONE_SEQUENCE,
    STAGE_MONOMER_REFOLDING,
    STAGE_SEQUENCE_DESIGN,
)


@dataclass(frozen=True)
class ModuleSpec:
    module_id: str
    label: str
    family: str
    tools: tuple[str, ...]
    consumes: tuple[str, ...]
    produces: tuple[str, ...]
    description: str = ""
    default_backend: str = "docker"
    supported_backends: tuple[str, ...] = ("docker",)
    params: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)


MODULE_SPECS: dict[str, ModuleSpec] = {
    "generation": ModuleSpec(
        module_id="generation",
        label="Generation",
        family="design",
        tools=(
            "rfdiffusion_classic",
            "bindcraft",
            "rfdiffusion3_foundry",
            "boltzgen",
            "pxdesign",
            "genie3",
            "protpardelle_1c",
            "proteina_complexa",
        ),
        consumes=(),
        produces=(STAGE_GENERATION_BACKBONE, STAGE_GENERATION_BACKBONE_SEQUENCE),
        description="Generate binder backbones or backbone+sequence candidates against a prepared target.",
        supported_backends=("docker", "nextflow"),
    ),
    "sequence_design": ModuleSpec(
        module_id="sequence_design",
        label="Sequence Design / Optimization",
        family="design",
        tools=("ligandmpnn", "proteinmpnn", "solublempnn", "foundry_mpnn", "proteinmpnn_fastrelax"),
        consumes=(STAGE_GENERATION_BACKBONE,),
        produces=(STAGE_SEQUENCE_DESIGN, STAGE_GENERATION_BACKBONE_SEQUENCE),
        description="Assign or optimize binder sequences and side chains for generated backbones.",
        supported_backends=("docker", "nextflow"),
    ),
    "monomer_refolding": ModuleSpec(
        module_id="monomer_refolding",
        label="Monomer Refolding",
        family="refolding-validation",
        tools=("af2_monomer", "boltz2_monomer", "esmfold2_monomer", "esmfold"),
        consumes=(STAGE_SEQUENCE_DESIGN, STAGE_GENERATION_BACKBONE_SEQUENCE, STAGE_COMPLEX_REFOLDING),
        produces=(STAGE_MONOMER_REFOLDING,),
        description="Refold designed binder chains without the target complex.",
        supported_backends=("docker", "nextflow"),
    ),
    "complex_refolding": ModuleSpec(
        module_id="complex_refolding",
        label="Complex Refolding",
        family="refolding-validation",
        tools=(
            "af2_initial_guess",
            "boltz2_initial_guess",
            "esmfold2_complex_validation",
            "esmfold2_initial_guess_validation",
        ),
        consumes=(STAGE_SEQUENCE_DESIGN, STAGE_GENERATION_BACKBONE_SEQUENCE, STAGE_MONOMER_REFOLDING, STAGE_COMPLEX_REFOLDING),
        produces=(STAGE_COMPLEX_REFOLDING,),
        description="Predict and score the binder-target complex.",
        supported_backends=("docker", "nextflow"),
    ),
    "analysis": ModuleSpec(
        module_id="analysis",
        label="Analysis",
        family="analysis",
        tools=("filters", "ranking", "clustering", "reports"),
        consumes=(
            STAGE_GENERATION_BACKBONE,
            STAGE_GENERATION_BACKBONE_SEQUENCE,
            STAGE_SEQUENCE_DESIGN,
            STAGE_MONOMER_REFOLDING,
            STAGE_COMPLEX_REFOLDING,
        ),
        produces=(STAGE_ANALYSIS,),
        description="Filter, rank, cluster, and report candidate sets.",
        supported_backends=("docker",),
    ),
    "binder_benchmark": ModuleSpec(
        module_id="binder_benchmark",
        label="Binder Benchmark",
        family="benchmark",
        tools=(
            "esmfold2_benchmark",
            "de_novo_binder_scoring_metrics",
            "de_novo_binder_scoring_scripts",
        ),
        consumes=(),
        produces=(STAGE_BENCHMARK,),
        description="Score known, non-binding, and unknown candidates with ESM/ESMFold2 features.",
        supported_backends=("docker",),
    ),
}


def list_module_specs() -> list[dict]:
    return [spec.to_json() for spec in MODULE_SPECS.values()]


def module_for_stage(stage: str) -> str | None:
    for module_id, spec in MODULE_SPECS.items():
        if stage in spec.produces:
            return module_id
    return None


def downstream_modules(stage: str) -> list[dict]:
    return [spec.to_json() for spec in MODULE_SPECS.values() if stage in spec.consumes]
