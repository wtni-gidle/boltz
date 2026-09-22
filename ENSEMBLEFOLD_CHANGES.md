# EnsembleFold wrapper changes

## Current wrapper contract — 2026-09-20 implementation

This section supersedes the historical notes below. The five targeted wrapper
test files passed on stat CPU (99 tests, job 89685); full six-method native-equivalence acceptance is deferred. No new
commit or deployment is implied by this document.

- Molecule inputs are **JSON only**, with optional top-level `name`. YAML/FASTA
  molecule inputs are rejected. Runtime configuration formats are unaffected.
- `-D/--run_data_pipeline` and `-P/--run_inference` select stages.
  `--write_input_json true|false` (shell `-J`) independently controls prepared
  JSON persistence. When omitted it follows `-D` for compatibility. True updates
  an existing `<name>_data.json`; false neither creates nor modifies it.
- All inference invocations rebuild derived features from the specified JSON
  and current referenced resources. No existing processed record is authoritative.
  Disabling data forbids new MSA searches, not interpretation of supplied templates.
- `processed/`, keyed CSV, search working files, template conversion files,
  Lightning internal files, and affinity handoffs belong in private temporary
  directories. Cleanup follows the full invocation, including failures handled
  by Python. SIGKILL/power loss still needs scheduler/OS scratch reclamation.
  Prefer valid `SLURM_TMPDIR`, then standard `TMPDIR`/system scratch. Launch one
  entry process; external multi-rank launch is rejected before work. The existing
  `--devices` local fork route remains, but GPU execution is not tested this round.
- Public results are directly under `<output>/<name>/{models,summary_confidences,full_data}`;
  optional embeddings and affinity results remain public. Structure output is CIF.
  Shared weights/CCD resources and reusable data artifacts are not scratch files.
- Auto MSA resources use `msas/<name>__<first-chain>_pairedmsa.a3m.zst` and
  `_unpairedmsa.a3m.zst`; native paired row keys, limits and CSV conversion remain.
  Replace unpaired without replacing paired or templates.
- `--skip` checks required files for existence and nonzero size only. If affinity
  is requested, its final JSON is also required. A missing result reruns the whole
  seed, including structure samples and affinity; an old handoff NPZ does not count.

### Template input

No `templates` (or `[]`) means no templates, with no automatic template search.
Legacy Boltz JSON template entries with `cif`/`pdb`, `chain_id`, `template_id`
remain valid. `template_id` in that legacy syntax means template chain ID.

The new grouped format uses this template-field fragment:

```json
{"templates": [{"groupId": "complex_0", "chains": [
  {"queryChain": "A", "mmcifPath": "complex.cif", "templateChain": "X",
   "queryIndices": [0, 2], "templateIndices": [1, 3]},
  {"queryChain": "B", "mmcifPath": "complex.cif", "templateChain": "Y"}
], "force": false}]}
```

`templateChain` uses the native Boltz **label_asym_id**, not author chain numbering.
It may be omitted only when the CIF has one protein chain. Paths are relative to
the input JSON. Omit BOTH index lists for native automatic mapping, including in
inference-only runs; provide both to use exactly those pairs, without realignment.
Explicit empty lists mean no pairs. Invalid lists fail, rather than trigger fallback.
Indices are 0-based positions in the full query/template polymer sequences, including
unresolved positions; absent coordinates retain their native masks.

Each group declares a common coordinate frame. Members can reference different
single-chain CIFs, but must not be independently centered or rotated. Separate,
unrelated templates belong to different groups. Native `force`/`threshold` apply
to the group. Do not mix legacy and grouped entries in a single templates list.

When prepared JSON is requested, selected template chains are exported as
`msas/<name>__<query-chain>_template_<i>.cif.zst`, with group identity and explicit
residue pairs. Default pairs follow the native **actual feature consumer's offset**,
not an assumed truncation at the local alignment end. Export/readback token
coordinates and masks are checked; a persistence failure is reported, never used
to silently drop a template. With writing disabled, mappings remain runtime-only.
All resources are staged before publication; caught publication errors restore
previous resources. If rollback itself fails, recovery backups are retained and
reported. This is not a concurrent-reader snapshot transaction: use one writer per
target bundle, and disable input writing for parallel seed prediction. Data-only
with writing disabled still persists searched MSA resources, but not a data JSON.

Network architecture, checkpoint and diffusion code are unchanged. Template feature
selection is deliberately extended to honor arbitrary explicit residue pairs and
groups spanning multiple files; it is no longer accurate to claim all featurization
code is untouched. Full native-equivalence acceptance remains pending.

## Historical implementation notes (superseded where different above)

This branch is based on upstream commit
`b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc` and changes pipeline orchestration and
prediction I/O. Model architecture, checkpoints, featurization, and diffusion sampling
are unchanged.

## Two-stage MSA pipeline

The repository root includes `run_boltz.sh`, a Bash entry point for data-only,
inference-only, multi-seed, and resume runs. Its layout follows
`run_alphafold_pro.sh`: select `server` near the top, parse the documented short
options, set defaults, activate the selected environment, print the full native
command, and run it. `server="current"` uses the already active environment;
`server="delta"` contains the paths used by the Delta test installation. Add another
explicit server branch when deploying elsewhere.

`boltz predict` accepts two boolean stage controls:

```bash
boltz predict target.yaml --out_dir results -D true -P false --use_msa_server
boltz predict results/target/target_data.yaml --out_dir results -D false -P true --seeds 42
```

YAML may contain an optional top-level `name`. It takes precedence over the input
filename and defines the record ID, generated `_data.yaml` filename, and generated MSA
prefix. For the normal single-YAML invocation it also names the job directory. If
omitted, the filename stem remains the fallback for backward compatibility. Data-only
output persists the resolved name in the generated YAML, so renaming that prepared file
does not change its target identity. A directory input remains one multi-record job
whose outer directory is named after the input directory.

Data-only search writes one paired and one unpaired A3M for every auto-MSA protein
entity, then writes an executable `target_data.yaml`. It stops before keyed CSV
creation, preprocessing, checkpoint download, or model inference. Files are named:

```text
results/<target>/<target>_data.yaml
results/<target>/msa/<target>_<first-chain>_paired.a3m.zst
results/<target>/msa/<target>_<first-chain>_unpaired.a3m.zst
```

The generated YAML preserves the original Boltz input and replaces each automatically
searched protein MSA with explicit `msa.paired` and `msa.unpaired` paths. The unpaired
A3M can be replaced by DeepMSA2 output or its path can be edited in the YAML.
For a deduplicated protein entry such as `id: [A, B]`, both chains retain their IDs and
share zstd-compressed files named after the first ID, for example
`target_A_unpaired.a3m.zst`.
Inference-only reads `target_data.yaml` directly, derives target id `target` by removing
the `_data` suffix, verifies the unpaired query, creates the native keyed CSV, refreshes
processed MSA arrays, and then follows the original inference path. The keyed CSV and
all `processed/` files are runtime-only artifacts in a process-private temporary
directory. They are removed after structure and affinity inference, so the persistent
job directory contains no `processed/` tree. Concurrent inference processes therefore
never share preprocessing files. Slurm jobs automatically prefer `SLURM_TMPDIR` when it
exists; ordinary servers use the platform temporary directory. There is no work-directory
CLI option.

`--seeds 42` runs one seed, while `--seeds 40,41,42,43,44` runs several seeds in one
process. The prepared YAML is
preprocessed once, the structure checkpoint is loaded once for all requested seeds,
and the affinity checkpoint is likewise loaded at most once. RNG state is reset before
every structure and affinity prediction. Skip checks and output names remain
seed-specific. Because all seeds share one GPU process and model instance, outputs are
not guaranteed to be bitwise or coordinate-identical to separate single-seed processes;
use independent `--seeds 42` invocations when strict cross-process reproducibility matters.

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

If `--seeds` is omitted, the CLI generates, reports, and uses one concrete random seed so
outputs never collapse under a shared `seed-None` name.
