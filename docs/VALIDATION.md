# Repository publication checks — November 2, 2026

The source-only snapshot was copied to an isolated temporary directory on GPU1.
Validation used the existing MolmoAct Python environment and pinned source
checkouts; it did not replace files in the active experiment directories.

Passed checks:

- Source/credential-pattern/artifact audit over all 88 candidate files.
- Python AST parsing of every included Python file.
- `bash -n` for every included shell script.
- Applying the GPU1 patch to a clean checkout of the pinned TRAP commit.
- A second bootstrap invocation recognized the already applied patch.
- Importing and showing CLI help for collection, table-mask replay,
  target-completion evaluation, and the TRAP optimization entry point.
- Expanding both matched configs against clone-relative paths, finding the
  camera calibration, and confirming that only conditioning mode/output differ.
- Both existing CPU surrogate tests.
- The portable launcher's help path and checkpoint metadata command.

The existing Python environment and simulator assets were reused during these
checks. A fresh dependency installation, model download, recollection, and full
GPU optimization were not rerun for publication. The recorded GPU1 experiments
are the evidence for the underlying model code; the complete fresh-machine
workflow still needs an end-to-end reproduction run.

On November 3 the completed 150 main-condition episode files were checked by
the matched comparison script. Each episode's recorded target completion
agreed with its per-step grasp/contact/lift flags. Object positions matched
between conditions, and identical repeats were confirmed in every layout.
Control configs were checked against the executed teacher-forced config:
each disables only its intended loss, uses the same data/model/seed, and points
to byte-identical camera calibration. Both control drivers passed `bash -n`
and started their GPU smoke checks. On November 4 both smoke checks, 12,000-step
optimizations, and 50-trial evaluations completed. The comparison script parsed
all 3,705 action-only and 4,000 CoT-only reasoning traces, confirmed matched
layouts and identical repeats, and found zero orange predicate disagreements.
The random-patch evaluation was still in progress at 35/50 trials.
