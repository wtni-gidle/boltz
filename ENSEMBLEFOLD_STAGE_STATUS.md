# EnsembleFold wrapper stage status — 2026-09-23

Public task input remains JSON with Boltz's own schema. Defaults:
`write_input_json=true`, `compress_fold_input=false`, and
`compress_full_confidence=false`. Detailed JSON/NPZ compression does not enable
optional PAE/PDE output. Prepared and processed resources retain the documented
public/private separation. See `ENSEMBLEFOLD_CHANGES.md` for usage.

CPU/offline wrapper regression: 144 tests; not the full native/GPU suite.
The proposed AF3-style mandatory template declaration was cancelled: omission
still means no templates, and `templates: []` remains valid.

Deferred limitations: nontransactional sample replacement, failure exit status
after native OOM, some cross-job MSA filename collisions, case-insensitive
filesystem collisions, and the previously identified native insertion-feature
issue. None is claimed fixed by this stage. Skip intentionally checks only the
required non-empty files and does not compare changed input conditions.
