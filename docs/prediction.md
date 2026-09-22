# Prediction

Once `boltz` is installed, you can run predictions with:

`boltz predict <INPUT_PATH> [OPTIONS]`

* `<INPUT_PATH>` is a single `.json` file or a directory of JSON input files. The staged wrapper accepts JSON only: YAML and FASTA inputs are rejected, including YAML content merely renamed to `.json`.
* If you include `--use_msa_server`, the MSA will be generated automatically via the mmseqs2 server. Without this flag, you must provide a pre-computed MSA.
* If you include `--use_potentials`, Boltz will apply inference-time potentials to improve the physical plausibility of the predicted poses.
* By default, Boltz runs structure and affinity prediction even if matching output files already exist. With `--skip`, a seed is skipped only when every requested sample's model, summary confidence, and pLDDT outputs, plus requested PAE/PDE and seed embeddings, exist as nonempty regular files. For affinity records, the final affinity JSON is part of the same completion gate. Missing or empty required files rerun all requested samples of the affected seed; if affinity is incomplete, both structure and affinity stages rerun. Complete seeds remain untouched. The check does not parse JSON/NPZ/CIF contents or compare MSA, templates, constraints, checkpoint, or inference parameters, so changed conditions can still skip; use a separate output directory or disable skip when new predictions are wanted.
* The optional top-level JSON `name` defines the target/record ID, generated `_data.json` filename, job directory, and MSA prefix. If omitted, the input filename stem is used (with `_data` removed for a prepared filename). A name must be a non-empty, single filename component. Prepared JSON always saves the resolved name, so renaming the file does not change the task identity. Duplicate task names in a batch are rejected before preparation.
* `-D true -P false` runs only the MSA data pipeline and writes `<out_dir>/<name>/<name>_data.json` plus separate paired/unpaired A3M files. `-D false -P true` accepts that generated JSON directly, creates the native keyed CSV at runtime, and runs preprocessing and inference. Directory inputs also write one prepared bundle per task, without an extra input-directory-name layer. At least one stage must be enabled.
* `--seeds 42` runs one seed, while `--seeds 40,41,42` runs several seeds sequentially in one process while sharing preprocessing and checkpoint loading. When omitted, one concrete seed is generated and printed.


## Input format

Boltz takes a JSON object describing the complex, using Boltz's own schema, not AF3's schema.
Relative MSA and template paths are resolved from the declaring JSON's directory, not the shell's working directory.
Prepared output rebases these references; it does not promise to copy every external resource.
Old YAML inputs must be converted to valid JSON; there is no automatic compatibility reader or migration command.

The following JSON-syntax illustration shows the supported sections. Replace the illustrative sequences, paths, CCD/atom identifiers and residue indices with those for your own system; optional sections may be omitted.

```json
{
  "name": "TARGET_NAME",
  "version": 1,
  "sequences": [
    {"protein": {
      "id": ["A", "B"], "sequence": "MCDE", "msa": "msa/protein.csv",
      "modifications": [{"position": 1, "ccd": "MSE"}], "cyclic": false
    }},
    {"ligand": {"id": "C", "ccd": "SAH"}},
    {"ligand": {"id": "D", "smiles": "CCO"}}
  ],
  "constraints": [
    {"bond": {"atom1": ["A", 2, "SG"], "atom2": ["B", 2, "SG"]}},
    {"pocket": {"binder": "C", "contacts": [["A", 2]], "max_distance": 6, "force": false}},
    {"contact": {"token1": ["A", 3], "token2": ["B", 3], "max_distance": 6, "force": false}}
  ],
  "templates": [
    {"cif": "template.cif", "force": true, "threshold": 2},
    {"cif": "template.cif", "chain_id": "A"},
    {"cif": "template.cif", "chain_id": ["A", "B"], "template_id": ["X", "Y"]},
    {"pdb": "template.pdb", "chain_id": ["A", "B"], "template_id": ["A1", "B1"]}
  ],
  "properties": [{"affinity": {"binder": "D"}}]
}
```

### Sequences and molecules

The sequences section has one entry per unique chain or molecule.
* Polymers: use `ENTITY_TYPE` equals to `protein`, `dna`, or `rna`, and provide a `sequence`.
* Ligands (non-polymers): use `ENTITY_TYPE` equals `ligand`, and provide either a `smiles` string or a `ccd` code (but not both).
* `CHAIN_ID`: unique identifier for each chain/molecule. If multiple identical entities exist, set id as a list (e.g. `["A", "B"]`).

For proteins:
* By default, an `msa` must be provided.
* If `--use_msa_server` is set, the MSA is auto-generated (so `msa` can be omitted).
* To use a precomputed custom MSA, set `"msa": "MSA_PATH"` pointing to a `.a3m` file. If you have more than one protein chain, use a CSV format instead of a3m with two columns: `sequence` (protein sequence) and `key` (a unique identifier for matching rows across chains). Sequences with the same key are mutually aligned.
* To force single-sequence mode (not recommended, as it reduces accuracy), set `"msa": "empty"`.
* A generated `*_data.json` represents an automatically searched MSA as `"msa": {"paired": "PATH", "unpaired": "PATH"}`. This mapping is a wrapper extension intended as the data-pipeline/inference hand-off. The normal scalar `"msa": "PATH"` form remains supported.
* Prepared MSA filenames use the first chain ID of each deduplicated protein entry and are written as zstd streams by default. For example, `"id": ["A", "B"]` produces `<name>_A_paired.a3m.zst` and `<name>_A_unpaired.a3m.zst`, which are shared by both chains.
* Prepared A3M paths may contain plain text, gzip, xz, or zstd data. Compression is detected from magic bytes rather than the filename, so both a real compressed `.a3m.zst` and a plain-text file carrying that suffix are handled correctly.

The `modifications` field is optional and allows specification of modified residues in polymers (`protein`, `dna`, or `rna`).  
- `position`: index of the residue (starting from 1)  
- `ccd`: CCD code of the modified residue (currently supported only for CCD ligands)  

The `cyclic` flag indicates whether a polymer chain (not ligands) is cyclic.

### Constraints

`constraints` is an optional field that allows you to specify additional information about the input structure. 


* The `bond` constraint specifies covalent bonds between two atoms (`atom1` and `atom2`). It is currently only supported for CCD ligands and canonical residues, `CHAIN_ID` refers to the id of the residue set above, `RES_IDX` is the index (starting from 1) of the residue (1 for ligands), and `ATOM_NAME` is the standardized atom name (can be verified in CIF file of that component on the RCSB website).

* The `pocket` constraint specifies the residues associated with binding interaction, where `binder` refers to the chain binding to the pocket (which can be a molecule, protein, DNA or RNA) and `contacts` is the list of chain and residue indices (starting from 1, or atom names if the chain is a molecule) that form the binding site for the `binder`. `max_distance` specifies the maximum distance (in Angstrom, supported between 4A and 20A with 6A as default) between any atom in the `binder` and any atom in each of the `contacts` elements. If `force` is set to true, a potential will be used to enforce the pocket constraint.

* The `contact` constraint specifies a contact between two residues or atoms, where `token1` and `token2` are the identifiers of the residues or atoms (in the format `[CHAIN_ID, RES_IDX/ATOM_NAME]`). `max_distance` specifies the maximum distance (in Angstrom, supported between 4A and 20A with 6A as default) between any pair of atoms in the two elements. If `force` is set to true, a potential will be used to enforce the contact constraint. 

### Templates
`templates` is optional and allows specification of structural templates for protein chains. At minimum, provide the path to a CIF or PDB file.

If you wish to explicitly define which of the chains in your JSON should be templated using this file, you can use the `chain_id` entry to specify them. If providing a PDB file, chain ids will be incrementally assigned to each subchain in a parent PDB chain resulting in template chain ids of A1, A2, B1, etc for PDB chains A and B. Make sure to look at the structure of the template PDB file to determine the corresponding value of `template_id` to provide. Whether a set of ids is provided or not, Boltz will find the best matching chains from the provided template. If you wish to explicitly define the mapping yourself, you may provide the corresponding `template_id`.

For any template you provide, you can also specify a `force` flag which will use a potential to enforce that the backbone does not deviate excessively from the template during the prediction. When using `force` one must specify also the `threshold` field which controls the distance (in Angstroms) that the prediction can deviate from the template. 

### Properties (affinity)
`properties` is an optional field that allows you to specify whether you want to compute the affinity. If enabled, you must also provide the chain_id corresponding to the small molecule against which the affinity will be computed. Only one single small molecule can be specified for affinity computation. It must be a ligand chain (not a protein, DNA or RNA) and has to be at most 128 atoms counting heavy atoms and hydrogens kept by `RDKit RemoveHs`, however, we do not recommend running the affinity module with ligands significantly larger than 56 atoms (counted as above, limit set during training). At this point, Boltz only supports the computation of affinity of small molecules to protein targets, if ran with an RNA/DNA/co-factor target, the code will not crash but the output will be unreliable.


### Example

Save this example as `examples/example_complex.json` so its MSA path resolves
to the provided `examples/msa/seq1.a3m`. Other ready-to-edit JSON inputs are in
`examples/`; select individual files when that directory also contains resources
and historical FASTA examples.

```json
{
  "name": "example_complex",
  "version": 1,
  "sequences": [
    {
      "protein": {
        "id": [
          "A",
          "B"
        ],
        "sequence": "MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQFYERLIGLWAPAVMEAAHELGVFAALAEAPADSGELARRLDCDARAMRVLLDALYAYDVIDRIHDTNGFRYLLSAEARECLLPGTLFSLVGKFMHDINVAWPAWRNLAEVVRHGARDTSGAESPNGIAQEDYESLVGGINFWAPPIVTTLSRKLRASGRSGDATASVLDVGCGTGLYSQLLLREFPRWTATGLDVERIATLANAQALRLGVEERFATRAGDFWRGGWGTGYDLVLFANIFHLQTPASAVRLMRHAAACLAPDGLVAVVDQIVDADREPKTPQDRFALLFAASMTNTGGGDAYTFQEYEEWFTAAGLQRIETLDTPMHRILLARRATEPSAVPEGQASENLYFQ",
        "msa": "msa/seq1.a3m"
      }
    },
    {
      "ligand": {
        "id": [
          "C",
          "D"
        ],
        "ccd": "SAH"
      }
    },
    {
      "ligand": {
        "id": [
          "E",
          "F"
        ],
        "smiles": "N[C@@H](Cc1ccc(O)cc1)C(=O)O"
      }
    }
  ]
}
```




## Options

The following options are available for the `predict` command:

    boltz predict input_path [OPTIONS]

Examples of common options include:

* Adding `--use_msa_server` flag, Boltz auto-generates the MSA using the mmseqs2 server. 

* Adding the `--use_potentials` flag, Boltz uses an inference time potential that significantly improve the physical quality of the poses. 

* To predict a structure using 10 recycling steps and 25 samples (the default parameters for AlphaFold3) use (note however that the prediction will take significantly longer): `--recycling_steps 10 --diffusion_samples 25`


| **Option**               | **Type**        | **Default**                 | **Description**                                                                                                                                                                     |
|--------------------------|-----------------|-----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--out_dir`              | `PATH`          | `./`                        | Collection root below which a `<target>/` job directory is created.                                                                                                                  |
| `--cache`                | `PATH`          | `~/.boltz`                  | The directory where to download the data and model. Will use environment variable `BOLTZ_CACHE` as an absolute path if set                                                          |
| `--checkpoint`           | `PATH`          | None                        | An optional checkpoint. Uses the provided Boltz-2 model by default.                                                                                                                 |
| `--devices`              | `INTEGER`       | `1`                         | The number of devices to use for prediction.                                                                                                                                        |
| `--accelerator`          | `[gpu,cpu,tpu]` | `gpu`                       | The accelerator to use for prediction.                                                                                                                                              |
| `--recycling_steps`      | `INTEGER`       | `3`                         | The number of recycling steps to use for prediction.                                                                                                                                |
| `--sampling_steps`       | `INTEGER`       | `200`                       | The number of sampling steps to use for prediction.                                                                                                                                 |
| `--diffusion_samples`    | `INTEGER`       | `1`                         | The number of diffusion samples to use for prediction.                                                                                                                              |
| `--max_parallel_samples` | `INTEGER` | `5`                       | maximum number of samples to predict in parallel. |
| `--step_scale`           | `FLOAT`         | `1.638` for Boltz-1 and `1.5` for Boltz-2                     | The step size is related to the temperature at which the diffusion process samples the distribution. The lower the higher the diversity among samples (recommended between 1 and 2). |
| `--output_format`        | `[mmcif]`       | `mmcif`                     | Wrapper predictions use mmCIF. The lower-level writer still supports PDB for library callers.                                                                                       |
| `--num_workers`          | `INTEGER`       | `2`                         | The number of dataloader workers to use for prediction.                                                                                                                             |
| `-D`, `--run_data_pipeline` | `BOOLEAN`    | `True`                      | Whether to search and save separate paired/unpaired MSA files.                                                                                                                       |
| `-P`, `--run_inference`  | `BOOLEAN`       | `True`                      | Whether to run preprocessing and model inference.                                                                                                                                   |
| `--method`          | str       | None                         | The method to use for prediction.                                                                                                                             |
| `--preprocessing-threads`          | `INTEGER`       | `multiprocessing.cpu_count()` | The number of threads to use for preprocessing.                                                                                                                             |
| `--affinity_mw_correction`          | `FLAG`       | `False` | Whether to add the Molecular Weight correction to the affinity value head.                                                                                                                             |
| `--sampling_steps_affinity`          | `INTEGER`       | `200` | The number of sampling steps to use for affinity prediction.                                                                                                                             |
| `--diffusion_samples_affinity`          | `INTEGER`       | `5` | The number of diffusion samples to use for affinity prediction.                                                                                                                             |
| `--affinity_checkpoint`          | `PATH`          | None | An optional checkpoint for affinity. Uses the provided Boltz-2 model by default.                                                                                                                             |
| `--max_msa_seqs`          | `INTEGER`       | `8192` |The maximum number of MSA sequences to use for prediction.                                                                                                                             |
| `--subsample_msa`          | `FLAG`       | `False` | Whether to subsample the MSA.                                                                                                                             |
| `--num_subsampled_msa`          | `INTEGER`       | `1024` | The number of MSA sequences to subsample.                                                                                                                             |
| `--no_kernels`          | `FLAG`       | `False` | Whether to not use trifast kernels for triangular updates..                                                                                                                             |
| `--skip`                 | `FLAG`          | `False`                     | Whether to skip seeds with complete existing prediction outputs.                                                                                                                    |
| `--seeds`                | `TEXT`          | generated                   | One 32-bit seed or a comma-separated list to run sequentially in one process. When omitted, one seed is generated and printed. Preprocessing and checkpoint loading are shared, while output and skip checks remain seed-specific. |
| `--use_msa_server`       | `FLAG`          | `False`                     | Whether to use the msa server to generate msa's.                                                                                                                                    |
| `--msa_server_url`       | str             | `https://api.colabfold.com` | MSA server url. Used only if --use_msa_server is set.                                                                                                                               |
| `--msa_pairing_strategy` | str             | `greedy`                    | Pairing strategy to use. Used only if --use_msa_server is set. Options are 'greedy' and 'complete'                                                                                  |
| `--use_potentials`        | `FLAG`          | `False`                     | Whether to run the original Boltz-2 model using inference time potentials.                                                                                                        |
| `--write_full_pae`       | `FLAG`          | `False`                     | Whether to save the full PAE matrix as a file.                                                                                                                                      |
| `--write_full_pde`       | `FLAG`          | `False`                     | Whether to save the full PDE matrix as a file.                                                                                                                                      |

## Output

For `target.json` and `--out_dir results`, job artifacts are organized below `results/target/` unless an explicit `name` chooses another task identity. Generated `target_data.json` saves that name and can be renamed without changing the task.

After a data-only run followed by inference-only, the persistent layout is:

```
results/
└── target/
    ├── target_data.json
    ├── msa/
    │   ├── target_A_paired.a3m.zst
    │   └── target_A_unpaired.a3m.zst
    ├── models/seed-[seed]_sample-0_model.cif
    ├── summary_confidences/seed-[seed]_sample-0_summary_confidences.json
    ├── full_data/plddt_seed-[seed]_sample-0.npz
    ├── full_data/pae_seed-[seed]_sample-0.npz
    ├── full_data/pde_seed-[seed]_sample-0.npz
    ├── embeddings/seed-[seed]_embeddings.npz
    └── affinity/seed-[seed]_affinity.json
```

Inference-only creates the keyed CSV, `processed/` data, manifest, and Lightning working files in a process-private temporary directory and removes them after prediction. It therefore does **not** persist `target_A.csv`, `processed/`, or `lightning_logs/` below `results/target/`. A combined `-D true -P true` run retains the native persistent preprocessing behavior and may include those paths.

For a file input, prediction categories are written directly below the job
directory. For a directory input, each record is written below
`<out_dir>/<record.id>/`, even if the directory contains only one record or
skip filtering leaves one record, so names cannot collide. Existing files below
the former `predictions/` directory are not migrated or treated as current
outputs. Samples retain their original diffusion sample index; they are not
renamed by confidence rank. Confidence scores remain available in the summary
JSON. With `--seeds`, every seed uses the same layout and filename pattern;
requested seeds are run sequentially, and `--skip` evaluates completeness
independently for each seed.

Each output folder includes a confidence `.json` file with aggregated confidence scores for that sample. Its structure is:
```yaml
{
    "confidence_score": 0.8367,       # Aggregated score used to sort the predictions, corresponds to 0.8 * complex_plddt + 0.2 * iptm (ptm for single chains)
    "ptm": 0.8425,                    # Predicted TM score for the complex
    "iptm": 0.8225,                   # Predicted TM score when aggregating at the interfaces
    "ligand_iptm": 0.0,               # ipTM but only aggregating at protein-ligand interfaces
    "protein_iptm": 0.8225,           # ipTM but only aggregating at protein-protein interfaces
    "complex_plddt": 0.8402,          # Average pLDDT score for the complex
    "complex_iplddt": 0.8241,         # Average pLDDT score when upweighting interface tokens
    "complex_pde": 0.8912,            # Average PDE score for the complex
    "complex_ipde": 5.1650,           # Average PDE score when aggregating at interfaces  
    "chains_ptm": {                   # Predicted TM score within each chain
        "0": 0.8533,
        "1": 0.8330
    },
    "pair_chains_iptm": {             # Predicted (interface) TM score between each pair of chains
        "0": {
            "0": 0.8533,
            "1": 0.8090
        },
        "1": {
            "0": 0.8225,
            "1": 0.8330
        }
    }
}
```
`confidence_score`, `ptm` and `plddt` scores (and their interface and individual chain analogues) have a range of [0, 1], where higher values indicate higher confidence. `pde` scores have a unit of angstroms, where lower values indicate higher confidence.

The output affinity `.json` file is organized as follows:
```yaml
{
    "affinity_pred_value": 0.8367,             # Predicted binding affinity from the ensemble model
    "affinity_probability_binary": 0.8425,     # Predicted binding likelihood from the ensemble model
    "affinity_pred_value1": 0.8225,            # Predicted binding affinity from the first model of the ensemble
    "affinity_probability_binary1": 0.0,       # Predicted binding likelihood from the first model in the ensemble
    "affinity_pred_value2": 0.8225,            # Predicted binding affinity from the second model of the ensemble
    "affinity_probability_binary2": 0.8402,    # Predicted binding likelihood from the second model in the ensemble
}
```

There are two main predictions in the affinity output: `affinity_pred_value` and `affinity_probability_binary`. They are trained on largely different datasets, with different supervisions, and should be used in different contexts. 

The `affinity_probability_binary` field should be used to detect binders from decoys, for example in a hit-discovery stage. It's value ranges from 0 to 1 and represents the predicted probability that the ligand is a binder.

The `affinity_pred_value` aims to measure the specific affinity of different binders and how this changes with small modifications of the molecule (*note that this implies that it should only be used when comparing different active molecules, not inactives*). This should be used in ligand optimization stages such as hit-to-lead and lead-optimization. It reports a binding affinity value as `log10(IC50)`, derived from an `IC50` measured in `μM`. Lower values indicate stronger predicted binding, for instance:
- IC50 of $10^{-9}$ M $\longrightarrow$ our model outputs $-3$ (strong binder)
- IC50 of $10^{-6}$ M $\longrightarrow$ our model outputs $0$ (moderate binder)
- IC50 of $10^{-4}$ M $\longrightarrow$ our model outputs $2$ (weak binder / decoy)

You can convert the model's output to pIC50 in `kcal/mol` by using `y --> (6 - y) * 1.364` where `y` is the model's prediction.


## Authentication to MSA Server

When using the `--use_msa_server` option with a server that requires authentication, you can provide credentials in one of two ways:

### 1. Basic Authentication

- Use the CLI options `--msa_server_username` and `--msa_server_password`.
- Or, set the environment variables:
  - `BOLTZ_MSA_USERNAME` (for the username)
  - `BOLTZ_MSA_PASSWORD` (for the password, recommended for security)

**Example:**
```bash
export BOLTZ_MSA_USERNAME=myuser
export BOLTZ_MSA_PASSWORD=mypassword
boltz predict ... --use_msa_server
```
Or:
```bash
boltz predict ... --use_msa_server --msa_server_username myuser --msa_server_password mypassword
```

### 2. API Key Authentication

- Use the CLI options `--api_key_header` (default: `X-API-Key`) and `--api_key_value` to specify the header and value for API key authentication.
- Or, set the API key value via the environment variable `MSA_API_KEY_VALUE` (recommended for security).

**Example using CLI:**
```bash
boltz predict ... --use_msa_server --api_key_header X-API-Key --api_key_value <your-api-key>
```

**Example using environment variable:**
```bash
export MSA_API_KEY_VALUE=<your-api-key>
boltz predict ... --use_msa_server --api_key_header X-API-Key
```
If both the CLI option and environment variable are set, the CLI option takes precedence.

> If your server expects a different header, set `--api_key_header` accordingly (e.g., `--api_key_header X-Gravitee-Api-Key`).

---

**Note:**  
Only one authentication method (basic or API key) can be used at a time. If both are provided, the program will raise an error.


## Legacy inputs

The staged wrapper no longer accepts YAML or FASTA. Native standalone FASTA
parser code and historical `.fasta` examples remain for reference, not as
supported `boltz predict` inputs. Use the JSON examples instead. Runtime/training
configuration YAML is unrelated and remains unchanged.

## Troubleshooting

 - When running on old NVIDIA GPUs, you may encounter an error related to the `cuequivariance` library. In this case, you should run the model with the `--no_kernels` flag, which will disable the use of the `cuequivariance` library and allow the model to run without it. This may result in slightly lower performance, but it will allow you to run the model on older hardware.
