# EnsembleFold wrapper changes

This branch is based on upstream commit
`b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc` and changes pipeline orchestration and
prediction I/O. Model architecture, checkpoints, featurization, and diffusion sampling
are unchanged.

## Two-stage MSA pipeline

`boltz predict` accepts two boolean stage controls:

```bash
boltz predict target.yaml --out_dir results -D true -P false --use_msa_server
boltz predict results/target/target_data.yaml --out_dir results -D false -P true --seed 42
```

Data-only search writes one paired and one unpaired A3M for every auto-MSA protein
entity, then writes an executable `target_data.yaml`. It stops before keyed CSV
creation, preprocessing, checkpoint download, or model inference. Files are named:

```text
results/<target>/<target>_data.yaml
results/<target>/msa/<target>_<entity>_paired.a3m
results/<target>/msa/<target>_<entity>_unpaired.a3m
```

The generated YAML preserves the original Boltz input and replaces each automatically
searched protein MSA with explicit `msa.paired` and `msa.unpaired` paths. The unpaired
A3M can be replaced by DeepMSA2 output or its path can be edited in the YAML.
Inference-only reads `target_data.yaml` directly, derives target id `target` by removing
the `_data` suffix, verifies the unpaired query, creates the native keyed CSV, refreshes
processed MSA arrays, and then follows the original inference path. The keyed CSV and
all `processed/` files are runtime-only artifacts in a process-private temporary
directory. They are removed after structure and affinity inference, so the persistent
job directory contains no `processed/` tree. Concurrent inference processes therefore
never share preprocessing files. Slurm jobs automatically prefer `SLURM_TMPDIR` when it
exists; ordinary servers use the platform temporary directory. There is no work-directory
CLI option.

`--seed` remains the single-seed interface. `--seeds 40,41,42,43,44` runs several
seeds in one process and is mutually exclusive with `--seed`. The prepared YAML is
preprocessed once, the structure checkpoint is loaded once for all requested seeds,
and the affinity checkpoint is likewise loaded at most once. RNG state is reset before
every structure and affinity prediction so each seed follows the same path as an
equivalent single-seed invocation. Skip checks and output names remain seed-specific.

Prepared MSA paths may point to plain text, gzip, xz, or zstd data. Compression is
detected from file magic bytes, not the filename: a real zstd stream is decompressed even
without a `.zst` suffix, while a plain-text file named `.a3m.zst` is read as plain text.

## Output layout

The requested output directory is a collection root. For `target.yaml` or
`target_data.yaml`, job artifacts are written below `results/target/`, and prediction
files for the normal single-record job are written directly below
`results/target/predictions/`:

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
requested. It does not hash inputs or parse output contents. Prediction runs again by
default; this check is activated only by the explicit `--skip` flag.

Affinity still uses the highest-confidence structure internally. Its private hand-off
is named `pre_affinity_seed-<seed>.npz`, preventing different seeds from overwriting one
another. The affinity stage is reseeded so a resumed affinity-only run matches a run in
which structure and affinity inference execute together.

If `--seed` is omitted, the CLI generates, reports, and uses a concrete random seed so
outputs never collapse under a shared `seed-None` name.
