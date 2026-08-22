# EnsembleFold wrapper changes

This branch is based on upstream commit
`b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc` and changes prediction I/O only.
Model architecture, checkpoints, featurization, and diffusion sampling are unchanged.

## Output layout

Each record is written below `predictions/<record.id>/`:

```text
models/seed-<seed>_sample-<sample>_model.cif
summary_confidences/seed-<seed>_sample-<sample>_summary_confidences.json
full_data/plddt_seed-<seed>_sample-<sample>.npz
full_data/pae_seed-<seed>_sample-<sample>.npz
full_data/pde_seed-<seed>_sample-<sample>.npz
embeddings/seed-<seed>_embeddings.npz
affinity/seed-<seed>_affinity.json
```

PAE, PDE, embeddings, and affinity files are only written when returned or requested
by the corresponding prediction path. pLDDT, PAE, and PDE remain separate NPZ files
with their original keys.

`sample` is the original diffusion sample index. Confidence ranking is not written
into filenames, and no ranking CSV or best-model copy is produced. Confidence scores
remain available in each summary JSON.

## Skip and affinity behavior

Structure skip checks the expected model, summary, and pLDDT file for every requested
sample. It additionally checks PAE, PDE, or embeddings when those outputs are explicitly
requested. It does not hash inputs or parse output contents.

Affinity still uses the highest-confidence structure internally. Its private hand-off
is named `pre_affinity_seed-<seed>.npz`, preventing different seeds from overwriting one
another. The affinity stage is reseeded so a resumed affinity-only run matches a run in
which structure and affinity inference execute together.

If `--seed` is omitted, the CLI generates, reports, and uses a concrete random seed so
outputs never collapse under a shared `seed-None` name.
