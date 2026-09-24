# EnsembleFold wrapper stage status — 2026-09-24

Public task input remains JSON with Boltz's own schema. Defaults:
`write_input_json=true`, `compress_fold_input=false`, and
`compress_full_confidence=false`. Detailed JSON/NPZ compression does not enable
optional PAE/PDE output. Prepared and processed resources retain the documented
public/private separation. See `ENSEMBLEFOLD_CHANGES.md` for usage.

Earlier-stage CPU/offline wrapper regression: 144 tests; not the full native/GPU suite.
The proposed AF3-style mandatory template declaration was cancelled: omission
still means no templates, and `templates: []` remains valid.

Follow-up: structure and affinity failures now produce a nonzero command exit,
including failures on another distributed rank. Successful files remain; later
work stops at the failed stage/seed. CPU regression: 152 passed, including real
two-rank gloo failure tests; not a GPU OOM reproduction.

Final verification (2026-09-24): the same 152 CPU tests passed again (job
93940). A real-model single-GPU smoke was attempted in allocation 91413;
the installed PyTorch 2.7.1+cu126 lacks kernels for its Blackwell sm_120 GPU
and failed with `no kernel image`. This is not a GPU acceptance pass.
No production dependency or native algorithm was changed to work around it.
Read-only redundancy review found an unused affinity-filter helper and a
currently unreachable bytes-publication branch; no broad refactor was made.

Deferred limitations: nontransactional sample replacement,
some cross-job MSA filename collisions, case-insensitive
filesystem collisions, and the previously identified native insertion-feature
issue. None is claimed fixed by this stage. Skip intentionally checks only the
required non-empty files and does not compare changed input conditions.
