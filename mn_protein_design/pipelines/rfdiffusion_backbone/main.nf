nextflow.enable.dsl = 2

params.input_pdb = null
params.publish_dir = "output"
params.rfdiffusion_image = "ovo-rfdiffusion:latest"
params.rfdiffusion_num_designs = 1
params.rfdiffusion_contig = null
params.hotspot = ""
params.rfdiffusion_run_parameters = "diffuser.T=50"
params.write_trajectory = false

process RFDIFFUSION_BACKBONE {
    container params.rfdiffusion_image
    publishDir params.publish_dir, mode: "copy", overwrite: true

    input:
    path input_pdb

    output:
    path "output/*"

    script:
    def hotspotArg = params.hotspot ? "\"ppi.hotspot_res=[${params.hotspot}]\"" : ""
    """
    set -euxo pipefail
    export PYTHONPATH=/opt/RFdiffusion:/opt/RFdiffusion/env/SE3Transformer
    mkdir -p output
    python3 /opt/RFdiffusion/scripts/run_inference.py \\
      inference.output_prefix=output/design \\
      inference.model_directory_path=/models \\
      inference.schedule_directory_path=/opt/RFdiffusion/schedules \\
      inference.input_pdb=${input_pdb} \\
      inference.num_designs=${params.rfdiffusion_num_designs} \\
      "contigmap.contigs=[${params.rfdiffusion_contig}]" \\
      ${hotspotArg} \\
      inference.write_trajectory=${params.write_trajectory} \\
      ${params.rfdiffusion_run_parameters}
    test -n "\$(find output -maxdepth 1 -type f -name 'design_*.pdb' -print -quit)"
    """
}

workflow {
    if (!params.input_pdb) {
        error "Missing --input_pdb"
    }
    if (!params.rfdiffusion_contig) {
        error "Missing --rfdiffusion_contig"
    }
    RFDIFFUSION_BACKBONE(file(params.input_pdb))
}
