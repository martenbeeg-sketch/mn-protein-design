# Current Problem: AF2-IG Internal Screen With Staged Chain Roles

Status as of 2026-07-17: archived historical debugging note. The original
staged-chain-role failure described here was used to harden the current staged
workflow, role sidecars, evidence-mode presets, and AF2-IG handling. Keep this
file as context for why AF2-IG must never assume binder chain `A`, but verify the
current implementation in code before treating this as an active open bug.

This note captures the current failing design-campaign issue so future work does
not have to reconstruct it from chat history.

## Symptom

Design campaign run:

```text
20260715-191128-621903b1
```

failed during:

```text
Collect generator outputs
```

The campaign used only RFdiffusion classic. RFdiffusion generation and
ProteinMPNN-style sequence refinement completed, but the internal AF2 initial
guess screen produced no usable candidates, so the staged campaign ended with:

```text
harmonized_candidate_count: 1
refined_candidate_count: 0
sequence_refinement_status: completed
status: skipped
success: false
```

## What Worked

The generator and sequence-design side of the staged workflow appears to be
doing the right thing for this run:

* RFdiffusion classic produced a generator candidate.
* MPNN sequence refinement produced five redesigned candidates.
* The staged AF2-IG input PDBs used the current app role convention:
  * binder chain: `Z`
  * target chain: `A`
* Sidecars and chain maps were present next to the staged AF2-IG input:
  * `.binder_source_chains.txt` contained `Z`
  * `.target_chains.txt` contained `A`
  * `.binder_sequence.txt` contained the correct redesigned binder sequence
  * `.chain_map.json` recorded explicit role metadata

This means the previous RFdiffusion -> MPNN binder-sequence mismatch was not the
active failure in this run.

## What Failed

The AF2-IG child benchmark/refolding run was:

```text
/mnt/data/RESULTS/mn-protein-design-workdir/workdir/runs/benchmark/20260715-191207-1c8f59e8
```

Its AF2-IG engine subrun returned:

```text
return_code: 1
candidate_count: 0
```

The Docker command passed:

```text
--designed_chains A
```

but the staged binder chain in the input PDB was `Z`, while `A` was the target.
The helper then attempted to prepare the wrong chain as binder and failed with a
shape mismatch:

```text
ValueError: Incompatible shapes for broadcasting: shapes=[(1, 0, 20), (230, 20)]
```

## Likely Root Cause

`run_af2_initial_guess_complex_refolding(...)` in
`mn_protein_design/workflows/refolding.py` still defaults to:

```python
designed_chains_arg = "A"
```

for normal complex AF2-IG runs. It only derives chain IDs from the staged
structure for capacity target-only runs.

The AF2-IG script also historically assumed whole-complex binder chains were
`A` or `B`:

```python
if not target_only_template_mode and options.designed_chains not in {"A", "B"}:
    raise NotImplementedError("Expected binder chain to be A or B")
```

That assumption conflicts with the app's current staged convention where binders
are `Z`, `Y`, `X`, ... and targets are `A`, `B`, `C`, ... .

## Fix Direction

The fix should make AF2-IG consume the same role metadata as the other staged
refolding/evaluation paths.

Recommended implementation:

1. In `run_af2_initial_guess_complex_refolding(...)`, derive
   `--designed_chains` from staged input sidecars/chain maps:
   * prefer `<safe_id>.binder_source_chains.txt`
   * then `<safe_id>.chain_map.json`
   * then normalized candidate `binder_chains`
   * only fall back to `A` for legacy data
2. Teach `af2_initial_guess_binder_eval.py` to accept non-legacy binder chain
   IDs such as `Z` when that chain exists in the input PDB.
3. Keep target selection based on `.target_chains.txt` when present.
4. If ColabDesign/AF2-IG cannot handle arbitrary chain IDs directly, add a
   narrow AF2-IG staging adapter that rewrites only the AF2-IG input copy into
   its legacy local layout while preserving app-normalized role metadata in the
   output candidate.
5. Re-test with a tiny RFdiffusion-only staged campaign and confirm:
   * MPNN produces expected sequence count
   * AF2-IG child receives `--designed_chains Z` or a documented AF2-local
     legacy copy
   * AF2-IG emits predicted structures
   * `first_rmsd_filter.missing_rmsd_count` is no longer equal to all inputs

## Important Constraint

Do not change the global app convention back to binder `A` / target `B` to fix
AF2-IG. The correct boundary is either:

* make AF2-IG understand the staged role metadata, or
* create an AF2-IG-local compatibility input while normalizing outputs back into
  the app chain-role contract.

## Runs Mentioned While Debugging

Earlier related failed runs:

```text
20260715-184711-df280c35
20260715-183205-26191362
20260715-175412-8544d6aa
```

The most diagnostic run for the current AF2-IG chain-role problem is:

```text
20260715-191128-621903b1
```
