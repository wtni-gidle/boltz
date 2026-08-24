import multiprocessing
import os
import pickle
import platform
import secrets
import tarfile
import tempfile
import urllib.request
import warnings
from dataclasses import asdict, dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Literal, Optional

import click
import torch
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only
from rdkit import Chem
from tqdm import tqdm

from boltz.data import const
from boltz.data.module.inference import BoltzInferenceDataModule
from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.data.mol import load_canonicals
from boltz.data.msa.pipeline import materialize_msa_csvs, search_msa_components
from boltz.data.parse.a3m import parse_a3m
from boltz.data.parse.csv import parse_csv
from boltz.data.parse.fasta import parse_fasta
from boltz.data.parse.yaml import parse_yaml, target_name_from_path
from boltz.data.types import MSA, Manifest, Record, Target
from boltz.data.write.writer import BoltzAffinityWriter, BoltzWriter
from boltz.model.models.boltz1 import Boltz1
from boltz.model.models.boltz2 import Boltz2

CCD_URL = "https://huggingface.co/boltz-community/boltz-1/resolve/main/ccd.pkl"
MOL_URL = "https://huggingface.co/boltz-community/boltz-2/resolve/main/mols.tar"

BOLTZ1_URL_WITH_FALLBACK = [
    "https://model-gateway.boltz.bio/boltz1_conf.ckpt",
    "https://huggingface.co/boltz-community/boltz-1/resolve/main/boltz1_conf.ckpt",
]

BOLTZ2_URL_WITH_FALLBACK = [
    "https://model-gateway.boltz.bio/boltz2_conf.ckpt",
    "https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_conf.ckpt",
]

BOLTZ2_AFFINITY_URL_WITH_FALLBACK = [
    "https://model-gateway.boltz.bio/boltz2_aff.ckpt",
    "https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_aff.ckpt",
]


@dataclass
class BoltzProcessedInput:
    """Processed input data."""

    manifest: Manifest
    targets_dir: Path
    msa_dir: Path
    constraints_dir: Optional[Path] = None
    template_dir: Optional[Path] = None
    extra_mols_dir: Optional[Path] = None


def resolve_prediction_seeds(
    seed: Optional[int],
    seed_values: Optional[str],
) -> list[int]:
    """Resolve the single- or multi-seed CLI options."""
    if seed is not None and seed_values is not None:
        msg = "--seed and --seeds are mutually exclusive."
        raise click.UsageError(msg)

    if seed_values is None:
        if seed is None:
            seed = secrets.randbits(32)
            click.echo(f"No seed provided; using generated seed {seed}.")
        resolved = [seed]
    else:
        raw_values = [value.strip() for value in seed_values.split(",")]
        if not raw_values or any(not value for value in raw_values):
            msg = "--seeds must be a comma-separated list of integers."
            raise click.BadParameter(msg, param_hint="--seeds")
        try:
            resolved = [int(value) for value in raw_values]
        except ValueError as exc:
            msg = "--seeds must be a comma-separated list of integers."
            raise click.BadParameter(msg, param_hint="--seeds") from exc

    if any(value < 0 or value >= 2**32 for value in resolved):
        msg = "Seeds must be between 0 and 4294967295 inclusive."
        raise click.BadParameter(msg, param_hint="--seed/--seeds")
    if len(resolved) != len(set(resolved)):
        msg = "--seeds must not contain duplicate values."
        raise click.BadParameter(msg, param_hint="--seeds")
    return resolved


@dataclass
class PairformerArgs:
    """Pairformer arguments."""

    num_blocks: int = 48
    num_heads: int = 16
    dropout: float = 0.0
    activation_checkpointing: bool = False
    offload_to_cpu: bool = False
    v2: bool = False


@dataclass
class PairformerArgsV2:
    """Pairformer arguments."""

    num_blocks: int = 64
    num_heads: int = 16
    dropout: float = 0.0
    activation_checkpointing: bool = False
    offload_to_cpu: bool = False
    v2: bool = True


@dataclass
class MSAModuleArgs:
    """MSA module arguments."""

    msa_s: int = 64
    msa_blocks: int = 4
    msa_dropout: float = 0.0
    z_dropout: float = 0.0
    use_paired_feature: bool = True
    pairwise_head_width: int = 32
    pairwise_num_heads: int = 4
    activation_checkpointing: bool = False
    offload_to_cpu: bool = False
    subsample_msa: bool = False
    num_subsampled_msa: int = 1024


@dataclass
class BoltzDiffusionParams:
    """Diffusion process parameters."""

    gamma_0: float = 0.605
    gamma_min: float = 1.107
    noise_scale: float = 0.901
    rho: float = 8
    step_scale: float = 1.638
    sigma_min: float = 0.0004
    sigma_max: float = 160.0
    sigma_data: float = 16.0
    P_mean: float = -1.2
    P_std: float = 1.5
    coordinate_augmentation: bool = True
    alignment_reverse_diff: bool = True
    synchronize_sigmas: bool = True
    use_inference_model_cache: bool = True


@dataclass
class Boltz2DiffusionParams:
    """Diffusion process parameters."""

    gamma_0: float = 0.8
    gamma_min: float = 1.0
    noise_scale: float = 1.003
    rho: float = 7
    step_scale: float = 1.5
    sigma_min: float = 0.0001
    sigma_max: float = 160.0
    sigma_data: float = 16.0
    P_mean: float = -1.2
    P_std: float = 1.5
    coordinate_augmentation: bool = True
    alignment_reverse_diff: bool = True
    synchronize_sigmas: bool = True


@dataclass
class BoltzSteeringParams:
    """Steering parameters."""

    fk_steering: bool = False
    num_particles: int = 3
    fk_lambda: float = 4.0
    fk_resampling_interval: int = 3
    physical_guidance_update: bool = False
    contact_guidance_update: bool = True
    num_gd_steps: int = 20


@rank_zero_only
def download_boltz1(cache: Path, *, download_weights: bool = True) -> None:
    """Download all the required data.

    Parameters
    ----------
    cache : Path
        The cache directory.

    """
    # Download CCD
    ccd = cache / "ccd.pkl"
    if not ccd.exists():
        click.echo(
            f"Downloading the CCD dictionary to {ccd}. You may "
            "change the cache directory with the --cache flag."
        )
        urllib.request.urlretrieve(CCD_URL, str(ccd))  # noqa: S310

    if not download_weights:
        return

    # Download model
    model = cache / "boltz1_conf.ckpt"
    if not model.exists():
        click.echo(
            f"Downloading the model weights to {model}. You may "
            "change the cache directory with the --cache flag."
        )
        for i, url in enumerate(BOLTZ1_URL_WITH_FALLBACK):
            try:
                urllib.request.urlretrieve(url, str(model))  # noqa: S310
                break
            except Exception as e:  # noqa: BLE001
                if i == len(BOLTZ1_URL_WITH_FALLBACK) - 1:
                    msg = f"Failed to download model from all URLs. Last error: {e}"
                    raise RuntimeError(msg) from e
                continue


@rank_zero_only
def download_boltz2(cache: Path, *, download_weights: bool = True) -> None:
    """Download all the required data.

    Parameters
    ----------
    cache : Path
        The cache directory.

    """
    # Download CCD
    mols = cache / "mols"
    tar_mols = cache / "mols.tar"
    if not tar_mols.exists():
        click.echo(
            f"Downloading the CCD data to {tar_mols}. "
            "This may take a bit of time. You may change the cache directory "
            "with the --cache flag."
        )
        urllib.request.urlretrieve(MOL_URL, str(tar_mols))  # noqa: S310
    if not mols.exists():
        click.echo(
            f"Extracting the CCD data to {mols}. "
            "This may take a bit of time. You may change the cache directory "
            "with the --cache flag."
        )
        with tarfile.open(str(tar_mols), "r") as tar:
            tar.extractall(cache)  # noqa: S202

    if not download_weights:
        return

    # Download model
    model = cache / "boltz2_conf.ckpt"
    if not model.exists():
        click.echo(
            f"Downloading the Boltz-2 weights to {model}. You may "
            "change the cache directory with the --cache flag."
        )
        for i, url in enumerate(BOLTZ2_URL_WITH_FALLBACK):
            try:
                urllib.request.urlretrieve(url, str(model))  # noqa: S310
                break
            except Exception as e:  # noqa: BLE001
                if i == len(BOLTZ2_URL_WITH_FALLBACK) - 1:
                    msg = f"Failed to download model from all URLs. Last error: {e}"
                    raise RuntimeError(msg) from e
                continue

    # Download affinity model
    affinity_model = cache / "boltz2_aff.ckpt"
    if not affinity_model.exists():
        click.echo(
            f"Downloading the Boltz-2 affinity weights to {affinity_model}. You may "
            "change the cache directory with the --cache flag."
        )
        for i, url in enumerate(BOLTZ2_AFFINITY_URL_WITH_FALLBACK):
            try:
                urllib.request.urlretrieve(url, str(affinity_model))  # noqa: S310
                break
            except Exception as e:  # noqa: BLE001
                if i == len(BOLTZ2_AFFINITY_URL_WITH_FALLBACK) - 1:
                    msg = f"Failed to download model from all URLs. Last error: {e}"
                    raise RuntimeError(msg) from e
                continue


def get_cache_path() -> str:
    """Determine the cache path, prioritising the BOLTZ_CACHE environment variable.

    Returns
    -------
    str: Path
        Path to use for boltz cache location.

    """
    env_cache = os.environ.get("BOLTZ_CACHE")
    if env_cache:
        resolved_cache = Path(env_cache).expanduser().resolve()
        if not resolved_cache.is_absolute():
            msg = f"BOLTZ_CACHE must be an absolute path, got: {env_cache}"
            raise ValueError(msg)
        return str(resolved_cache)

    return str(Path("~/.boltz").expanduser())


def check_inputs(data: Path) -> list[Path]:
    """Check the input data and output directory.

    Parameters
    ----------
    data : Path
        The input data.

    Returns
    -------
    list[Path]
        The list of input data.

    """
    click.echo("Checking input data.")

    # Check if data is a directory
    if data.is_dir():
        data: list[Path] = list(data.glob("*"))

        # Filter out non .fasta or .yaml files, raise
        # an error on directory and other file types
        for d in data:
            if d.is_dir():
                msg = f"Found directory {d} instead of .fasta or .yaml."
                raise RuntimeError(msg)
            if d.suffix.lower() not in (".fa", ".fas", ".fasta", ".yml", ".yaml"):
                msg = (
                    f"Unable to parse filetype {d.suffix}, "
                    "please provide a .fasta or .yaml file."
                )
                raise RuntimeError(msg)
    else:
        data = [data]

    return data


def filter_inputs_structure(  # noqa: C901
    manifest: Manifest,
    outdir: Path,
    skip: bool = False,
    *,
    seed: Optional[int] = None,
    diffusion_samples: int = 1,
    output_format: Literal["pdb", "mmcif"] = "mmcif",
    write_full_pae: bool = False,
    write_full_pde: bool = False,
    write_embeddings: bool = False,
) -> Manifest:
    """Filter the manifest to only include missing predictions.

    Parameters
    ----------
    manifest : Manifest
        The manifest of the input data.
    outdir : Path
        The output directory.
    skip: bool
        Whether to skip seeds with complete existing predictions.
    seed : Optional[int]
        The random seed used for the predictions.
    diffusion_samples : int
        The number of diffusion samples expected for each record.
    output_format : Literal["pdb", "mmcif"]
        The structure output format.
    write_full_pae : bool
        Whether a PAE file is expected for every sample.
    write_full_pde : bool
        Whether a PDE file is expected for every sample.
    write_embeddings : bool
        Whether a seed-level embeddings file is expected.

    Returns
    -------
    Manifest
        The manifest of the filtered input data.

    """
    structure_suffix = "pdb" if output_format == "pdb" else "cif"
    use_record_subdir = len(manifest.records) > 1

    def outputs_complete(record: Record) -> bool:
        """Check the lightweight AF3-style output contract for one seed."""
        target_dir = outdir / "predictions"
        if use_record_subdir:
            target_dir = target_dir / record.id
        models_dir = target_dir / "models"
        summary_dir = target_dir / "summary_confidences"
        full_data_dir = target_dir / "full_data"

        for sample_idx in range(diffusion_samples):
            basename = f"seed-{seed}_sample-{sample_idx}"
            required_paths = [
                models_dir / f"{basename}_model.{structure_suffix}",
                summary_dir / f"{basename}_summary_confidences.json",
                full_data_dir / f"plddt_{basename}.npz",
            ]
            if write_full_pae:
                required_paths.append(full_data_dir / f"pae_{basename}.npz")
            if write_full_pde:
                required_paths.append(full_data_dir / f"pde_{basename}.npz")
            if not all(path.is_file() for path in required_paths):
                return False

        if write_embeddings:
            embeddings_path = (
                target_dir / "embeddings" / f"seed-{seed}_embeddings.npz"
            )
            if not embeddings_path.is_file():
                return False

        # Affinity consumes a private structure selected by confidence. If the
        # public affinity result is still missing, keep the structure record in
        # the manifest when that hand-off file also needs to be regenerated.
        if record.affinity:
            affinity_path = target_dir / "affinity" / f"seed-{seed}_affinity.json"
            handoff_path = target_dir / f"pre_affinity_seed-{seed}.npz"
            if not affinity_path.is_file() and not handoff_path.is_file():
                return False

        return True

    existing = {r.id for r in manifest.records if outputs_complete(r)}

    # Remove complete records only when skip is explicitly requested.
    if existing and skip:
        manifest = Manifest([r for r in manifest.records if r.id not in existing])
        msg = (
            f"Found some existing predictions ({len(existing)}), "
            "skipping them and running only the missing ones, if any."
        )
        click.echo(msg)
    elif existing:
        msg = f"Found {len(existing)} existing predictions, will run them again."
        click.echo(msg)

    return manifest


def filter_inputs_affinity(
    manifest: Manifest,
    outdir: Path,
    skip: bool = False,
    *,
    seed: Optional[int] = None,
) -> Manifest:
    """Check the input data and output directory for affinity.

    Parameters
    ----------
    manifest : Manifest
        The manifest.
    outdir : Path
        The output directory.
    skip: bool
        Whether to skip seeds with existing affinity predictions.
    seed : Optional[int]
        The random seed used for the affinity prediction.

    Returns
    -------
    Manifest
        The manifest of the filtered input data.

    """
    click.echo("Checking input data for affinity.")
    use_record_subdir = len(manifest.records) > 1

    # Get all affinity targets
    existing = {
        r.id
        for r in manifest.records
        if r.affinity
        and (
            outdir
            / "predictions"
            / (r.id if use_record_subdir else "")
            / "affinity"
            / f"seed-{seed}_affinity.json"
        ).is_file()
    }

    # Remove complete records only when skip is explicitly requested.
    if existing and skip:
        manifest = Manifest([r for r in manifest.records if r.id not in existing])
        num_skipped = len(existing)
        msg = (
            f"Found some existing affinity predictions ({num_skipped}), "
            "skipping them and running only the missing ones, if any."
        )
        click.echo(msg)
    elif existing:
        msg = "Found existing affinity predictions, will run them again."
        click.echo(msg)

    return manifest


def compute_msa(
    data: dict[str, str],
    target_id: str,
    msa_dir: Path,
    msa_server_url: str,
    msa_pairing_strategy: str,
    msa_server_username: Optional[str] = None,
    msa_server_password: Optional[str] = None,
    api_key_header: Optional[str] = None,
    api_key_value: Optional[str] = None,
) -> None:
    """Compute the MSA for the input data.

    Parameters
    ----------
    data : dict[str, str]
        The input protein sequences.
    target_id : str
        The target id.
    msa_dir : Path
        The msa directory.
    msa_server_url : str
        The MSA server URL.
    msa_pairing_strategy : str
        The MSA pairing strategy.
    msa_server_username : str, optional
        Username for basic authentication with MSA server.
    msa_server_password : str, optional
        Password for basic authentication with MSA server.
    api_key_header : str, optional
        Custom header key for API key authentication (default: X-API-Key).
    api_key_value : str, optional
        Custom header value for API key authentication (overrides --api_key if set).

    """
    click.echo(f"Calling MSA server for target {target_id} with {len(data)} sequences")
    click.echo(f"MSA server URL: {msa_server_url}")
    click.echo(f"MSA pairing strategy: {msa_pairing_strategy}")
    search_msa_components(
        data=data,
        target_id=target_id,
        msa_dir=msa_dir,
        msa_server_url=msa_server_url,
        msa_pairing_strategy=msa_pairing_strategy,
        msa_server_username=msa_server_username,
        msa_server_password=msa_server_password,
        api_key_header=api_key_header,
        api_key_value=api_key_value,
    )
    materialize_msa_csvs(data=data, msa_dir=msa_dir)


def parse_input_target(
    path: Path,
    ccd: dict,
    mol_dir: Path,
    boltz2: bool,
    msa_materialization_dir: Optional[Path] = None,
) -> Target:
    """Parse one supported Boltz input file."""
    if path.suffix.lower() in (".fa", ".fas", ".fasta"):
        return parse_fasta(path, ccd, mol_dir, boltz2)
    if path.suffix.lower() in (".yml", ".yaml"):
        return parse_yaml(
            path,
            ccd,
            mol_dir,
            boltz2,
            msa_materialization_dir=msa_materialization_dir,
        )
    if path.is_dir():
        msg = f"Found directory {path} instead of .fasta or .yaml."
        raise RuntimeError(msg)
    msg = (
        f"Unable to parse filetype {path.suffix}, "
        "please provide a .fasta or .yaml file."
    )
    raise RuntimeError(msg)


def collect_auto_msas(target: Target, msa_dir: Path) -> dict[str, str]:
    """Assign runtime CSV paths and collect auto-MSA protein entities."""
    to_generate: dict[str, str] = {}
    prot_id = const.chain_type_ids["PROTEIN"]
    for chain in target.record.chains:
        if (chain.mol_type == prot_id) and (chain.msa_id == 0):
            entity_id = chain.entity_id
            msa_id = f"{target.record.id}_{entity_id}"
            to_generate[msa_id] = target.sequences[entity_id]
            chain.msa_id = msa_dir / f"{msa_id}.csv"
        elif chain.msa_id == 0:
            chain.msa_id = -1
    return to_generate


def write_data_yaml(
    source_path: Path,
    out_dir: Path,
    target: Target,
    auto_msas: dict[str, str],
) -> Path:
    """Write the post-data-pipeline Boltz YAML used for later inference."""
    if source_path.suffix.lower() not in (".yml", ".yaml"):
        msg = "Staged data-pipeline output currently requires a YAML input."
        raise ValueError(msg)

    with source_path.open() as handle:
        prepared_schema = yaml.safe_load(handle)

    msa_id_by_sequence = {
        sequence: msa_id for msa_id, sequence in auto_msas.items()
    }
    for item in prepared_schema.get("sequences", []):
        protein = item.get("protein")
        if protein is None or protein.get("msa") not in (None, "", 0):
            continue
        sequence = str(protein["sequence"])
        msa_id = msa_id_by_sequence.get(sequence)
        if msa_id is None:
            continue
        paired_path = out_dir / "msa" / f"{msa_id}_paired.a3m"
        unpaired_path = out_dir / "msa" / f"{msa_id}_unpaired.a3m"
        protein["msa"] = {
            "paired": str(paired_path.relative_to(out_dir)),
            "unpaired": str(unpaired_path.relative_to(out_dir)),
        }

    data_path = out_dir / f"{target.record.id}_data.yaml"
    data_path.write_text(yaml.safe_dump(prepared_schema, sort_keys=False))
    click.echo(f"Prepared Boltz input written to {data_path}")
    return data_path


def process_input(  # noqa: C901, PLR0912, PLR0915, D103
    path: Path,
    ccd: dict,
    msa_dir: Path,
    mol_dir: Path,
    boltz2: bool,
    run_data_pipeline: bool,
    use_msa_server: bool,
    msa_server_url: str,
    msa_pairing_strategy: str,
    msa_server_username: Optional[str],
    msa_server_password: Optional[str],
    api_key_header: Optional[str],
    api_key_value: Optional[str],
    max_msa_seqs: int,
    processed_msa_dir: Path,
    processed_constraints_dir: Path,
    processed_templates_dir: Path,
    processed_mols_dir: Path,
    structure_dir: Path,
    records_dir: Path,
) -> None:
    try:
        target = parse_input_target(
            path,
            ccd,
            mol_dir,
            boltz2,
            msa_materialization_dir=msa_dir,
        )

        # Get target id
        target_id = target.record.id

        # Resolve auto-MSA entities to runtime CSV paths.
        to_generate = collect_auto_msas(target, msa_dir)

        if to_generate:
            if run_data_pipeline:
                if not use_msa_server:
                    msg = "Missing MSA's in input and --use_msa_server flag not set."
                    raise RuntimeError(msg)  # noqa: TRY301
                msg = (
                    f"Generating MSA for {path} with {len(to_generate)} "
                    "protein entities."
                )
                click.echo(msg)
                compute_msa(
                    data=to_generate,
                    target_id=target_id,
                    msa_dir=msa_dir,
                    msa_server_url=msa_server_url,
                    msa_pairing_strategy=msa_pairing_strategy,
                    msa_server_username=msa_server_username,
                    msa_server_password=msa_server_password,
                    api_key_header=api_key_header,
                    api_key_value=api_key_value,
                )
            else:
                click.echo(f"Materializing prepared MSA files for {path}.")
                materialize_msa_csvs(data=to_generate, msa_dir=msa_dir)

        if run_data_pipeline:
            write_data_yaml(
                source_path=path,
                out_dir=msa_dir.parent,
                target=target,
                auto_msas=to_generate,
            )

        # Parse MSA data
        msas = sorted({c.msa_id for c in target.record.chains if c.msa_id != -1})
        msa_id_map = {}
        for msa_idx, msa_id in enumerate(msas):
            # Check that raw MSA exists
            msa_path = Path(msa_id)
            if not msa_path.exists():
                msg = f"MSA file {msa_path} not found."
                raise FileNotFoundError(msg)  # noqa: TRY301

            # Dump processed MSA
            processed = processed_msa_dir / f"{target_id}_{msa_idx}.npz"
            msa_id_map[msa_id] = f"{target_id}_{msa_idx}"
            if not processed.exists() or not run_data_pipeline:
                # Parse A3M
                if msa_path.suffix.lower() in {".a3m", ".gz", ".xz", ".zst"}:
                    msa: MSA = parse_a3m(
                        msa_path,
                        taxonomy=None,
                        max_seqs=max_msa_seqs,
                    )
                elif msa_path.suffix == ".csv":
                    msa: MSA = parse_csv(msa_path, max_seqs=max_msa_seqs)
                else:
                    msg = (
                        f"MSA file {msa_path} not supported, only a3m, "
                        "a3m.gz, a3m.xz, a3m.zst, or csv."
                    )
                    raise RuntimeError(msg)  # noqa: TRY301

                msa.dump(processed)

        # Modify records to point to processed MSA
        for c in target.record.chains:
            if (c.msa_id != -1) and (c.msa_id in msa_id_map):
                c.msa_id = msa_id_map[c.msa_id]

        # Dump templates
        for template_id, template in target.templates.items():
            name = f"{target.record.id}_{template_id}.npz"
            template_path = processed_templates_dir / name
            template.dump(template_path)

        # Dump constraints
        constraints_path = processed_constraints_dir / f"{target.record.id}.npz"
        target.residue_constraints.dump(constraints_path)

        # Dump extra molecules
        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
        with (processed_mols_dir / f"{target.record.id}.pkl").open("wb") as f:
            pickle.dump(target.extra_mols, f)

        # Dump structure
        struct_path = structure_dir / f"{target.record.id}.npz"
        target.structure.dump(struct_path)

        # Dump record
        record_path = records_dir / f"{target.record.id}.json"
        target.record.dump(record_path)

    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        if not run_data_pipeline:
            msg = f"Failed to materialize prepared MSA input {path}."
            raise RuntimeError(msg) from e
        print(f"Failed to process {path}. Skipping. Error: {e}.")  # noqa: T201


@rank_zero_only
def process_inputs(
    data: list[Path],
    out_dir: Path,
    ccd_path: Path,
    mol_dir: Path,
    msa_server_url: str,
    msa_pairing_strategy: str,
    max_msa_seqs: int = 8192,
    run_data_pipeline: bool = True,
    use_msa_server: bool = False,
    msa_server_username: Optional[str] = None,
    msa_server_password: Optional[str] = None,
    api_key_header: Optional[str] = None,
    api_key_value: Optional[str] = None,
    boltz2: bool = False,
    preprocessing_threads: int = 1,
) -> Manifest:
    """Process the input data and output directory.

    Parameters
    ----------
    data : list[Path]
        The input data.
    out_dir : Path
        The output directory.
    ccd_path : Path
        The path to the CCD dictionary.
    max_msa_seqs : int, optional
        Max number of MSA sequences, by default 8192.
    use_msa_server : bool, optional
        Whether to use the MMSeqs2 server for MSA generation, by default False.
    run_data_pipeline : bool, optional
        Whether to search missing MSAs instead of reading prepared A3M files.
    msa_server_username : str, optional
        Username for basic authentication with MSA server, by default None.
    msa_server_password : str, optional
        Password for basic authentication with MSA server, by default None.
    api_key_header : str, optional
        Custom header key for API key authentication (default: X-API-Key).
    api_key_value : str, optional
        Custom header value for API key authentication (overrides --api_key if set).
    boltz2: bool, optional
        Whether to use Boltz2, by default False.
    preprocessing_threads: int, optional
        The number of threads to use for preprocessing, by default 1.

    Returns
    -------
    Manifest
        The manifest of the processed input data.

    """
    # Validate mutually exclusive authentication methods
    has_basic_auth = msa_server_username and msa_server_password
    has_api_key = api_key_value is not None
    
    if has_basic_auth and has_api_key:
        raise ValueError(
            "Cannot use both basic authentication (--msa_server_username/--msa_server_password) "
            "and API key authentication (--api_key_header/--api_key_value). Please use only one authentication method."
        )

    # Check if records exist at output path
    records_dir = out_dir / "processed" / "records"
    if records_dir.exists() and run_data_pipeline:
        # Load existing records
        existing = [Record.load(p) for p in records_dir.glob("*.json")]
        processed_ids = {record.id for record in existing}

        # Filter to missing only
        data = [d for d in data if target_name_from_path(d) not in processed_ids]

        # Nothing to do, update the manifest and return
        if data:
            click.echo(
                f"Found {len(existing)} existing processed inputs, skipping them."
            )
        else:
            click.echo("All inputs are already processed.")
            updated_manifest = Manifest(existing)
            updated_manifest.dump(out_dir / "processed" / "manifest.json")

    # Create output directories
    msa_dir = out_dir / "msa"
    records_dir = out_dir / "processed" / "records"
    structure_dir = out_dir / "processed" / "structures"
    processed_msa_dir = out_dir / "processed" / "msa"
    processed_constraints_dir = out_dir / "processed" / "constraints"
    processed_templates_dir = out_dir / "processed" / "templates"
    processed_mols_dir = out_dir / "processed" / "mols"
    predictions_dir = out_dir / "predictions"

    out_dir.mkdir(parents=True, exist_ok=True)
    msa_dir.mkdir(parents=True, exist_ok=True)
    records_dir.mkdir(parents=True, exist_ok=True)
    structure_dir.mkdir(parents=True, exist_ok=True)
    processed_msa_dir.mkdir(parents=True, exist_ok=True)
    processed_constraints_dir.mkdir(parents=True, exist_ok=True)
    processed_templates_dir.mkdir(parents=True, exist_ok=True)
    processed_mols_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Load CCD
    if boltz2:
        ccd = load_canonicals(mol_dir)
    else:
        with ccd_path.open("rb") as file:
            ccd = pickle.load(file)  # noqa: S301

    # Create partial function
    process_input_partial = partial(
        process_input,
        ccd=ccd,
        msa_dir=msa_dir,
        mol_dir=mol_dir,
        boltz2=boltz2,
        run_data_pipeline=run_data_pipeline,
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_pairing_strategy=msa_pairing_strategy,
        msa_server_username=msa_server_username,
        msa_server_password=msa_server_password,
        api_key_header=api_key_header,
        api_key_value=api_key_value,
        max_msa_seqs=max_msa_seqs,
        processed_msa_dir=processed_msa_dir,
        processed_constraints_dir=processed_constraints_dir,
        processed_templates_dir=processed_templates_dir,
        processed_mols_dir=processed_mols_dir,
        structure_dir=structure_dir,
        records_dir=records_dir,
    )

    # Parse input data
    preprocessing_threads = min(preprocessing_threads, len(data))
    click.echo(f"Processing {len(data)} inputs with {preprocessing_threads} threads.")

    if preprocessing_threads > 1 and len(data) > 1:
        with Pool(preprocessing_threads) as pool:
            list(tqdm(pool.imap(process_input_partial, data), total=len(data)))
    else:
        for path in tqdm(data):
            process_input_partial(path)

    # Load all records and write manifest
    record_paths = list(records_dir.glob("*.json"))
    if not run_data_pipeline:
        requested_ids = {target_name_from_path(path) for path in data}
        record_paths = [path for path in record_paths if path.stem in requested_ids]
    records = [Record.load(path) for path in record_paths]
    manifest = Manifest(records)
    manifest.dump(out_dir / "processed" / "manifest.json")


def prepare_msa_inputs(
    data: list[Path],
    out_dir: Path,
    ccd_path: Path,
    mol_dir: Path,
    boltz2: bool,
    use_msa_server: bool,
    msa_server_url: str,
    msa_pairing_strategy: str,
    msa_server_username: Optional[str] = None,
    msa_server_password: Optional[str] = None,
    api_key_header: Optional[str] = None,
    api_key_value: Optional[str] = None,
) -> None:
    """Search MSAs and write an executable ``<target>_data.yaml`` input."""
    msa_dir = out_dir / "msa"
    msa_dir.mkdir(parents=True, exist_ok=True)
    if boltz2:
        ccd = load_canonicals(mol_dir)
    else:
        with ccd_path.open("rb") as file:
            ccd = pickle.load(file)  # noqa: S301

    for path in data:
        target = parse_input_target(path, ccd, mol_dir, boltz2)
        to_generate = collect_auto_msas(target, msa_dir)
        if to_generate and not use_msa_server:
            msg = (
                f"Input {path} has auto MSA entities, but --use_msa_server "
                "was not set."
            )
            raise RuntimeError(msg)
        if to_generate:
            click.echo(
                f"Preparing paired/unpaired MSA files for {path} "
                f"({len(to_generate)} protein entities)."
            )
            search_msa_components(
                data=to_generate,
                target_id=target.record.id,
                msa_dir=msa_dir,
                msa_server_url=msa_server_url,
                msa_pairing_strategy=msa_pairing_strategy,
                msa_server_username=msa_server_username,
                msa_server_password=msa_server_password,
                api_key_header=api_key_header,
                api_key_value=api_key_value,
            )

        write_data_yaml(
            source_path=path,
            out_dir=out_dir,
            target=target,
            auto_msas=to_generate,
        )


@click.group()
def cli() -> None:
    """Boltz."""
    return


@cli.command()
@click.argument("data", type=click.Path(exists=True))
@click.option(
    "--out_dir",
    type=click.Path(exists=False),
    help="The collection root where the target job directory is created.",
    default="./",
)
@click.option(
    "-D",
    "--run_data_pipeline",
    type=bool,
    default=True,
    show_default=True,
    help="Search and save paired/unpaired MSA files.",
)
@click.option(
    "-P",
    "--run_inference",
    type=bool,
    default=True,
    show_default=True,
    help="Run preprocessing and model inference.",
)
@click.option(
    "--cache",
    type=click.Path(exists=False),
    help=(
        "The directory where to download the data and model. "
        "Default is ~/.boltz, or $BOLTZ_CACHE if set."
    ),
    default=get_cache_path,
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True),
    help="An optional checkpoint, will use the provided Boltz-1 model by default.",
    default=None,
)
@click.option(
    "--devices",
    type=int,
    help="The number of devices to use for prediction. Default is 1.",
    default=1,
)
@click.option(
    "--accelerator",
    type=click.Choice(["gpu", "cpu", "tpu"]),
    help="The accelerator to use for prediction. Default is gpu.",
    default="gpu",
)
@click.option(
    "--recycling_steps",
    type=int,
    help="The number of recycling steps to use for prediction. Default is 3.",
    default=3,
)
@click.option(
    "--sampling_steps",
    type=int,
    help="The number of sampling steps to use for prediction. Default is 200.",
    default=200,
)
@click.option(
    "--diffusion_samples",
    type=int,
    help="The number of diffusion samples to use for prediction. Default is 1.",
    default=1,
)
@click.option(
    "--max_parallel_samples",
    type=int,
    help="The maximum number of samples to predict in parallel. Default is None.",
    default=5,
)
@click.option(
    "--step_scale",
    type=float,
    help=(
        "The step size is related to the temperature at "
        "which the diffusion process samples the distribution. "
        "The lower the higher the diversity among samples "
        "(recommended between 1 and 2). "
        "Default is 1.638 for Boltz-1 and 1.5 for Boltz-2. "
        "If not provided, the default step size will be used."
    ),
    default=None,
)
@click.option(
    "--write_full_pae",
    type=bool,
    is_flag=True,
    help="Whether to dump the pae into a npz file. Default is True.",
)
@click.option(
    "--write_full_pde",
    type=bool,
    is_flag=True,
    help="Whether to dump the pde into a npz file. Default is False.",
)
@click.option(
    "--output_format",
    type=click.Choice(["pdb", "mmcif"]),
    help="The output format to use for the predictions. Default is mmcif.",
    default="mmcif",
)
@click.option(
    "--num_workers",
    type=int,
    help="The number of dataloader workers to use for prediction. Default is 2.",
    default=2,
)
@click.option(
    "--skip",
    is_flag=True,
    help="Whether to skip seeds with complete existing outputs. Default is False.",
)
@click.option(
    "--seed",
    type=int,
    help=(
        "Seed to use for random number generation. If omitted, a random seed is "
        "generated and reported."
    ),
    default=None,
)
@click.option(
    "--seeds",
    "seed_values",
    type=str,
    help=(
        "Comma-separated seeds to run in one process. Preprocessing and model "
        "loading are shared across seeds. Mutually exclusive with --seed."
    ),
    default=None,
)
@click.option(
    "--use_msa_server",
    is_flag=True,
    help="Whether to use the MMSeqs2 server for MSA generation. Default is False.",
)
@click.option(
    "--msa_server_url",
    type=str,
    help="MSA server url. Used only if --use_msa_server is set. ",
    default="https://api.colabfold.com",
)
@click.option(
    "--msa_pairing_strategy",
    type=str,
    help=(
        "Pairing strategy to use. Used only if --use_msa_server is set. "
        "Options are 'greedy' and 'complete'"
    ),
    default="greedy",
)
@click.option(
    "--msa_server_username",
    type=str,
    help="MSA server username for basic auth. Used only if --use_msa_server is set. Can also be set via BOLTZ_MSA_USERNAME environment variable.",
    default=None,
)
@click.option(
    "--msa_server_password",
    type=str,
    help="MSA server password for basic auth. Used only if --use_msa_server is set. Can also be set via BOLTZ_MSA_PASSWORD environment variable.",
    default=None,
)
@click.option(
    "--api_key_header",
    type=str,
    help="Custom header key for API key authentication (default: X-API-Key).",
    default=None,
)
@click.option(
    "--api_key_value",
    type=str,
    help="Custom header value for API key authentication.",
    default=None,
)
@click.option(
    "--use_potentials",
    is_flag=True,
    help="Whether to use potentials for steering. Default is False.",
)
@click.option(
    "--model",
    default="boltz2",
    type=click.Choice(["boltz1", "boltz2"]),
    help="The model to use for prediction. Default is boltz2.",
)
@click.option(
    "--method",
    type=str,
    help="The method to use for prediction. Default is None.",
    default=None,
)
@click.option(
    "--preprocessing-threads",
    type=int,
    help="The number of threads to use for preprocessing. Default is 1.",
    default=multiprocessing.cpu_count(),
)
@click.option(
    "--affinity_mw_correction",
    is_flag=True,
    type=bool,
    help="Whether to add the Molecular Weight correction to the affinity value head.",
)
@click.option(
    "--sampling_steps_affinity",
    type=int,
    help="The number of sampling steps to use for affinity prediction. Default is 200.",
    default=200,
)
@click.option(
    "--diffusion_samples_affinity",
    type=int,
    help="The number of diffusion samples to use for affinity prediction. Default is 5.",
    default=5,
)
@click.option(
    "--affinity_checkpoint",
    type=click.Path(exists=True),
    help="An optional checkpoint, will use the provided Boltz-1 model by default.",
    default=None,
)
@click.option(
    "--max_msa_seqs",
    type=int,
    help="The maximum number of MSA sequences to use for prediction. Default is 8192.",
    default=8192,
)
@click.option(
    "--subsample_msa",
    is_flag=True,
    help="Whether to subsample the MSA. Default is True.",
)
@click.option(
    "--num_subsampled_msa",
    type=int,
    help="The number of MSA sequences to subsample. Default is 1024.",
    default=1024,
)
@click.option(
    "--no_kernels",
    is_flag=True,
    help="Whether to disable the kernels. Default False",
)
@click.option(
    "--write_embeddings",
    is_flag=True,
    help=" to dump the s and z embeddings into a npz file. Default is False.",
)
def predict(  # noqa: C901, PLR0915, PLR0912
    data: str,
    out_dir: str,
    run_data_pipeline: bool = True,
    run_inference: bool = True,
    cache: str = "~/.boltz",
    checkpoint: Optional[str] = None,
    affinity_checkpoint: Optional[str] = None,
    devices: int = 1,
    accelerator: str = "gpu",
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    diffusion_samples: int = 1,
    sampling_steps_affinity: int = 200,
    diffusion_samples_affinity: int = 3,
    max_parallel_samples: Optional[int] = None,
    step_scale: Optional[float] = None,
    write_full_pae: bool = False,
    write_full_pde: bool = False,
    output_format: Literal["pdb", "mmcif"] = "mmcif",
    num_workers: int = 2,
    skip: bool = False,
    seed: Optional[int] = None,
    seed_values: Optional[str] = None,
    use_msa_server: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    msa_pairing_strategy: str = "greedy",
    msa_server_username: Optional[str] = None,
    msa_server_password: Optional[str] = None,
    api_key_header: Optional[str] = None,
    api_key_value: Optional[str] = None,
    use_potentials: bool = False,
    model: Literal["boltz1", "boltz2"] = "boltz2",
    method: Optional[str] = None,
    affinity_mw_correction: Optional[bool] = False,
    preprocessing_threads: int = 1,
    max_msa_seqs: int = 8192,
    subsample_msa: bool = True,
    num_subsampled_msa: int = 1024,
    no_kernels: bool = False,
    write_embeddings: bool = False,
) -> None:
    """Run predictions with Boltz."""
    if not run_data_pipeline and not run_inference:
        msg = "At least one of --run_data_pipeline or --run_inference must be true."
        raise click.UsageError(msg)

    # Set rdkit pickle logic
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

    prediction_seeds: list[int] = []
    if run_inference:
        # If cpu, write a friendly warning
        if accelerator == "cpu":
            msg = "Running on CPU, this will be slow. Consider using a GPU."
            click.echo(msg)

        # Supress some lightning warnings
        warnings.filterwarnings(
            "ignore", ".*that has Tensor Cores. To properly utilize them.*"
        )

        # Set no grad
        torch.set_grad_enabled(False)

        # Ignore matmul precision warning
        torch.set_float32_matmul_precision("highest")

        prediction_seeds = resolve_prediction_seeds(seed, seed_values)
        seed_everything(prediction_seeds[0])

        for key in ["CUEQ_DEFAULT_CONFIG", "CUEQ_DISABLE_AOT_TUNING"]:
            # Disable kernel tuning by default,
            # but do not modify envvar if already set by caller
            os.environ[key] = os.environ.get(key, "1")

    # Set cache path
    cache = Path(cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)

    # Get MSA server credentials from environment variables if not provided
    if run_data_pipeline and use_msa_server:
        if msa_server_username is None:
            msa_server_username = os.environ.get("BOLTZ_MSA_USERNAME")
        if msa_server_password is None:
            msa_server_password = os.environ.get("BOLTZ_MSA_PASSWORD")
        if api_key_value is None:
            api_key_value = os.environ.get("MSA_API_KEY_VALUE")
        
        click.echo(f"MSA server enabled: {msa_server_url}")
        if api_key_value:
            click.echo("MSA server authentication: using API key header")
        elif msa_server_username and msa_server_password:
            click.echo("MSA server authentication: using basic auth")
        else:
            click.echo("MSA server authentication: no credentials provided")

    # Create one AF3-style job directory below the requested output root. A
    # generated ``*_data.yaml`` deliberately maps back to the original target
    # name so data-only and inference-only runs share the same directory.
    data = Path(data).expanduser()
    out_dir = Path(out_dir).expanduser() / target_name_from_path(data)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download necessary data and model
    if model == "boltz1":
        download_boltz1(cache, download_weights=run_inference)
    elif model == "boltz2":
        download_boltz2(cache, download_weights=run_inference)
    else:
        msg = f"Model {model} not supported. Supported: boltz1, boltz2."
        raise ValueError(f"Model {model} not supported.")

    # Validate inputs
    data = check_inputs(data)

    # Check method
    if method is not None:
        if model == "boltz1":
            msg = "Method conditioning is not supported for Boltz-1."
            raise ValueError(msg)
        if method.lower() not in const.method_types_ids:
            method_names = list(const.method_types_ids.keys())
            msg = f"Method {method} not supported. Supported: {method_names}"
            raise ValueError(msg)

    # Process inputs
    ccd_path = cache / "ccd.pkl"
    mol_dir = cache / "mols"
    if run_data_pipeline and not run_inference:
        prepare_msa_inputs(
            data=data,
            out_dir=out_dir,
            ccd_path=ccd_path,
            mol_dir=mol_dir,
            boltz2=model == "boltz2",
            use_msa_server=use_msa_server,
            msa_server_url=msa_server_url,
            msa_pairing_strategy=msa_pairing_strategy,
            msa_server_username=msa_server_username,
            msa_server_password=msa_server_password,
            api_key_header=api_key_header,
            api_key_value=api_key_value,
        )
        return

    # Inference-only runs rebuild all derived features from the current
    # ``*_data.yaml`` and its referenced MSA/template inputs. Keep those
    # artifacts private to this process so concurrent seeds never share CSV,
    # NPZ, record, manifest, or Lightning log files. TemporaryDirectory uses
    # the system TMPDIR on ordinary servers; Slurm jobs prefer SLURM_TMPDIR.
    temporary_workspace = None
    processing_out_dir = out_dir
    if run_inference and not run_data_pipeline:
        slurm_tmp = os.environ.get("SLURM_TMPDIR")
        temp_base = Path(slurm_tmp) if slurm_tmp else None
        if temp_base is not None and not temp_base.is_dir():
            temp_base = None
        temporary_workspace = tempfile.TemporaryDirectory(
            prefix=f"boltz-{out_dir.name}-",
            dir=temp_base,
        )
        processing_out_dir = Path(temporary_workspace.name)
        click.echo("Using process-private temporary preprocessing directory.")

    process_inputs(
        data=data,
        out_dir=processing_out_dir,
        ccd_path=ccd_path,
        mol_dir=mol_dir,
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_pairing_strategy=msa_pairing_strategy,
        run_data_pipeline=run_data_pipeline,
        msa_server_username=msa_server_username,
        msa_server_password=msa_server_password,
        api_key_header=api_key_header,
        api_key_value=api_key_value,
        boltz2=model == "boltz2",
        preprocessing_threads=preprocessing_threads,
        max_msa_seqs=max_msa_seqs,
    )

    # Load manifest
    manifest = Manifest.load(processing_out_dir / "processed" / "manifest.json")
    use_record_subdir = len(manifest.records) > 1

    # Load processed data
    processed_dir = processing_out_dir / "processed"
    processed = BoltzProcessedInput(
        manifest=manifest,
        targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
        constraints_dir=(
            (processed_dir / "constraints")
            if (processed_dir / "constraints").exists()
            else None
        ),
        template_dir=(
            (processed_dir / "templates")
            if (processed_dir / "templates").exists()
            else None
        ),
        extra_mols_dir=(
            (processed_dir / "mols") if (processed_dir / "mols").exists() else None
        ),
    )

    # Determine which structure predictions are still needed for every seed.
    structure_manifests = {
        current_seed: filter_inputs_structure(
            manifest=manifest,
            outdir=out_dir,
            skip=skip,
            seed=current_seed,
            diffusion_samples=diffusion_samples,
            output_format=output_format,
            write_full_pae=write_full_pae,
            write_full_pde=write_full_pde,
            write_embeddings=write_embeddings,
        )
        for current_seed in prediction_seeds
    }

    # Set up trainer
    strategy = "auto"
    if (isinstance(devices, int) and devices > 1) or (
        isinstance(devices, list) and len(devices) > 1
    ):
        start_method = "fork" if platform.system() != "win32" and platform.system() != "Windows" else "spawn"
        strategy = DDPStrategy(start_method=start_method)
        if len(manifest.records) < devices:
            msg = (
                "Number of requested devices is greater "
                "than the number of predictions, taking the minimum."
            )
            click.echo(msg)
            if isinstance(devices, list):
                devices = devices[: max(1, len(manifest.records))]
            else:
                devices = max(1, min(len(manifest.records), devices))

    # Set up model parameters
    if model == "boltz2":
        diffusion_params = Boltz2DiffusionParams()
        step_scale = 1.5 if step_scale is None else step_scale
        diffusion_params.step_scale = step_scale
        pairformer_args = PairformerArgsV2()
    else:
        diffusion_params = BoltzDiffusionParams()
        step_scale = 1.638 if step_scale is None else step_scale
        diffusion_params.step_scale = step_scale
        pairformer_args = PairformerArgs()

    msa_args = MSAModuleArgs(
        subsample_msa=subsample_msa,
        num_subsampled_msa=num_subsampled_msa,
        use_paired_feature=model == "boltz2",
    )

    trainer: Optional[Trainer] = None
    pending_structure_seeds = [
        current_seed
        for current_seed, filtered in structure_manifests.items()
        if filtered.records
    ]
    if pending_structure_seeds:
        # Load the structure checkpoint once, then reset the RNG before every
        # prediction so each seed matches an equivalent single-seed process.
        seed_everything(pending_structure_seeds[0])
        if checkpoint is None:
            if model == "boltz2":
                checkpoint = cache / "boltz2_conf.ckpt"
            else:
                checkpoint = cache / "boltz1_conf.ckpt"

        predict_args = {
            "recycling_steps": recycling_steps,
            "sampling_steps": sampling_steps,
            "diffusion_samples": diffusion_samples,
            "max_parallel_samples": max_parallel_samples,
            "write_confidence_summary": True,
            "write_full_pae": write_full_pae,
            "write_full_pde": write_full_pde,
        }

        steering_args = BoltzSteeringParams()
        steering_args.fk_steering = use_potentials
        steering_args.physical_guidance_update = use_potentials

        model_cls = Boltz2 if model == "boltz2" else Boltz1
        model_module = model_cls.load_from_checkpoint(
            checkpoint,
            strict=True,
            predict_args=predict_args,
            map_location="cpu",
            diffusion_process_args=asdict(diffusion_params),
            ema=False,
            use_kernels=not no_kernels,
            pairformer_args=asdict(pairformer_args),
            msa_args=asdict(msa_args),
            steering_args=asdict(steering_args),
        )
        model_module.eval()

        for current_seed in pending_structure_seeds:
            filtered_manifest = structure_manifests[current_seed]
            seed_everything(current_seed)
            msg = (
                f"Running structure prediction for {len(filtered_manifest.records)} "
                f"input{'s' if len(filtered_manifest.records) > 1 else ''} "
                f"with seed {current_seed}."
            )
            click.echo(msg)

            pred_writer = BoltzWriter(
                data_dir=processed.targets_dir,
                output_dir=out_dir / "predictions",
                output_format=output_format,
                boltz2=model == "boltz2",
                write_embeddings=write_embeddings,
                seed=current_seed,
                use_record_subdir=use_record_subdir,
            )
            if trainer is None:
                trainer = Trainer(
                    default_root_dir=processing_out_dir,
                    strategy=strategy,
                    callbacks=[pred_writer],
                    accelerator=accelerator,
                    devices=devices,
                    precision=32 if model == "boltz1" else "bf16-mixed",
                )
            else:
                trainer.callbacks[0] = pred_writer

            if model == "boltz2":
                data_module = Boltz2InferenceDataModule(
                    manifest=filtered_manifest,
                    target_dir=processed.targets_dir,
                    msa_dir=processed.msa_dir,
                    mol_dir=mol_dir,
                    num_workers=num_workers,
                    constraints_dir=processed.constraints_dir,
                    template_dir=processed.template_dir,
                    extra_mols_dir=processed.extra_mols_dir,
                    override_method=method,
                )
            else:
                data_module = BoltzInferenceDataModule(
                    manifest=filtered_manifest,
                    target_dir=processed.targets_dir,
                    msa_dir=processed.msa_dir,
                    num_workers=num_workers,
                    constraints_dir=processed.constraints_dir,
                )

            trainer.predict(
                model_module,
                datamodule=data_module,
                return_predictions=False,
            )

    # Check if affinity predictions are needed
    if any(r.affinity for r in manifest.records):
        click.echo("\nPredicting property: affinity\n")
        affinity_manifests = {
            current_seed: filter_inputs_affinity(
                manifest=manifest,
                outdir=out_dir,
                skip=skip,
                seed=current_seed,
            )
            for current_seed in prediction_seeds
        }
        pending_affinity_seeds = [
            current_seed
            for current_seed, filtered in affinity_manifests.items()
            if filtered.records
        ]

        if pending_affinity_seeds:
            predict_affinity_args = {
                "recycling_steps": 5,
                "sampling_steps": sampling_steps_affinity,
                "diffusion_samples": diffusion_samples_affinity,
                "max_parallel_samples": 1,
                "write_confidence_summary": False,
                "write_full_pae": False,
                "write_full_pde": False,
            }
            if affinity_checkpoint is None:
                affinity_checkpoint = cache / "boltz2_aff.ckpt"

            steering_args = BoltzSteeringParams()
            steering_args.fk_steering = False
            steering_args.physical_guidance_update = False
            steering_args.contact_guidance_update = False

            seed_everything(pending_affinity_seeds[0])
            model_module = Boltz2.load_from_checkpoint(
                affinity_checkpoint,
                strict=True,
                predict_args=predict_affinity_args,
                map_location="cpu",
                diffusion_process_args=asdict(diffusion_params),
                ema=False,
                pairformer_args=asdict(pairformer_args),
                msa_args=asdict(msa_args),
                steering_args=asdict(steering_args),
                affinity_mw_correction=affinity_mw_correction,
            )
            model_module.eval()

            for current_seed in pending_affinity_seeds:
                manifest_filtered = affinity_manifests[current_seed]
                seed_everything(current_seed)
                msg = (
                    f"Running affinity prediction for "
                    f"{len(manifest_filtered.records)} "
                    f"input{'s' if len(manifest_filtered.records) > 1 else ''} "
                    f"with seed {current_seed}."
                )
                click.echo(msg)

                pred_writer = BoltzAffinityWriter(
                    data_dir=processed.targets_dir,
                    output_dir=out_dir / "predictions",
                    seed=current_seed,
                    use_record_subdir=use_record_subdir,
                )
                if trainer is None:
                    trainer = Trainer(
                        default_root_dir=processing_out_dir,
                        strategy=strategy,
                        callbacks=[pred_writer],
                        accelerator=accelerator,
                        devices=devices,
                        precision="bf16-mixed",
                    )
                else:
                    trainer.callbacks[0] = pred_writer

                data_module = Boltz2InferenceDataModule(
                    manifest=manifest_filtered,
                    target_dir=out_dir / "predictions",
                    msa_dir=processed.msa_dir,
                    mol_dir=mol_dir,
                    num_workers=num_workers,
                    constraints_dir=processed.constraints_dir,
                    template_dir=processed.template_dir,
                    extra_mols_dir=processed.extra_mols_dir,
                    override_method="other",
                    affinity=True,
                    seed=current_seed,
                    use_record_subdir=use_record_subdir,
                )
                trainer.predict(
                    model_module,
                    datamodule=data_module,
                    return_predictions=False,
                )
        else:
            click.echo("Found existing affinity predictions for all seeds, skipping.")

    if temporary_workspace is not None:
        temporary_workspace.cleanup()


if __name__ == "__main__":
    cli()
