import json
import os
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Literal

import click
import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import BasePredictionWriter
from torch import Tensor

from boltz.data.types import Coords, Interface, Record, Structure, StructureV2
from boltz.data.write.mmcif import to_mmcif
from boltz.data.write.pdb import to_pdb


def _check_prediction_failures(
    failed: int, *, stage: str, seed: int, pl_module: LightningModule
) -> None:
    """Report failed predictions before distributed workers are torn down."""
    if torch.distributed.is_initialized():
        # Every rank participates, including ranks with no local failures.
        # Checking only after Trainer.predict returns loses forked-worker state.
        count = torch.tensor(failed, dtype=torch.int64, device=pl_module.device)
        torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM)
        failed = int(count.item())
    print(f"Number of failed examples: {failed}")  # noqa: T201
    if failed:
        raise click.ClickException(
            f"{stage.capitalize()} prediction for seed {seed}: {failed} failed "
            "example(s). Successful output files have been preserved."
        )


class BoltzWriter(BasePredictionWriter):
    """Custom writer for predictions."""

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        *,
        seed: int,
        output_format: Literal["pdb", "mmcif"] = "mmcif",
        boltz2: bool = False,
        write_embeddings: bool = False,
        use_record_subdir: bool = True,
        affinity_output_dir: Path | None = None,
        compress_full_confidence: bool = False,
    ) -> None:
        """Initialize the writer.

        Parameters
        ----------
        output_dir : str
            The directory to save the predictions.

        """
        super().__init__(write_interval="batch")
        if output_format not in ["pdb", "mmcif"]:
            msg = f"Invalid output format: {output_format}"
            raise ValueError(msg)

        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.affinity_output_dir = (
            Path(affinity_output_dir) if affinity_output_dir is not None
            else self.data_dir.parent / "affinity_handoff"
        )
        self.seed = seed
        self.compress_full_confidence = compress_full_confidence
        self.output_format = output_format
        self.failed = 0
        self.boltz2 = boltz2
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.write_embeddings = write_embeddings
        self.use_record_subdir = use_record_subdir

    def _write_confidence(self, directory: Path, name: str, key: str, value: np.ndarray) -> None:
        suffix = ".npz" if self.compress_full_confidence else ".json"
        path = directory / (name + suffix)
        descriptor, filename = tempfile.mkstemp(prefix=f".{path.name}.", dir=directory)
        temporary = Path(filename)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                if self.compress_full_confidence:
                    np.savez_compressed(handle, **{key: value})
                else:
                    handle.write(json.dumps({key: value.tolist()}).encode("utf-8"))
            os.replace(temporary, path)
            path.with_suffix(".json" if self.compress_full_confidence else ".npz").unlink(missing_ok=True)
        finally:
            temporary.unlink(missing_ok=True)

    def write_on_batch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
        prediction: dict[str, Tensor],
        batch_indices: list[int],  # noqa: ARG002
        batch: dict[str, Tensor],
        batch_idx: int,  # noqa: ARG002
        dataloader_idx: int,  # noqa: ARG002
    ) -> None:
        """Write the predictions to disk."""
        if prediction["exception"]:
            self.failed += 1
            return

        # Get the records
        records: list[Record] = batch["record"]

        # Get the predictions
        coords = prediction["coords"]
        coords = coords.unsqueeze(0)

        pad_masks = prediction["masks"]

        # Select the best sample only for the internal affinity hand-off. The
        # published sample names preserve diffusion order and do not expose a rank.
        if "confidence_score" in prediction:
            best_model_idx = int(torch.argmax(prediction["confidence_score"]).item())
        # Handles cases where confidence summary is False
        else:
            best_model_idx = 0

        # Iterate over the records
        for record, coord, pad_mask in zip(records, coords, pad_masks):
            # Load the structure
            path = self.data_dir / f"{record.id}.npz"
            if self.boltz2:
                structure: StructureV2 = StructureV2.load(path)
            else:
                structure: Structure = Structure.load(path)

            # Compute chain map with masked removed, to be used later
            chain_map = {}
            for i, mask in enumerate(structure.mask):
                if mask:
                    chain_map[len(chain_map)] = i

            # Remove masked chains completely
            structure = structure.remove_invalid_chains()

            # A file input writes directly below its job directory. Directory
            # inputs retain record subdirectories to prevent collisions.
            record_dir = (
                self.output_dir / record.id
                if self.use_record_subdir
                else self.output_dir
            )
            record_dir.mkdir(exist_ok=True)
            models_dir = record_dir / "models"
            summary_dir = record_dir / "summary_confidences"
            full_data_dir = record_dir / "full_data"
            models_dir.mkdir(exist_ok=True)
            summary_dir.mkdir(exist_ok=True)
            full_data_dir.mkdir(exist_ok=True)

            for model_idx in range(coord.shape[0]):
                # Get model coord
                model_coord = coord[model_idx]
                # Unpad
                coord_unpad = model_coord[pad_mask.bool()]
                coord_unpad = coord_unpad.cpu().numpy()

                # New atom table
                atoms = structure.atoms
                atoms["coords"] = coord_unpad
                atoms["is_present"] = True
                if self.boltz2:
                    structure: StructureV2
                    coord_unpad = [(x,) for x in coord_unpad]
                    coord_unpad = np.array(coord_unpad, dtype=Coords)

                # Mew residue table
                residues = structure.residues
                residues["is_present"] = True

                # Update the structure
                interfaces = np.array([], dtype=Interface)
                if self.boltz2:
                    new_structure: StructureV2 = replace(
                        structure,
                        atoms=atoms,
                        residues=residues,
                        interfaces=interfaces,
                        coords=coord_unpad,
                    )
                else:
                    new_structure: Structure = replace(
                        structure,
                        atoms=atoms,
                        residues=residues,
                        interfaces=interfaces,
                    )

                # Update chain info
                chain_info = []
                for chain in new_structure.chains:
                    old_chain_idx = chain_map[chain["asym_id"]]
                    old_chain_info = record.chains[old_chain_idx]
                    new_chain_info = replace(
                        old_chain_info,
                        chain_id=int(chain["asym_id"]),
                        valid=True,
                    )
                    chain_info.append(new_chain_info)

                # Get plddt's
                plddts = None
                if "plddt" in prediction:
                    plddts = prediction["plddt"][model_idx]

                # Create path name
                outname = f"seed-{self.seed}_sample-{model_idx}"

                # Save the structure
                if self.output_format == "pdb":
                    path = models_dir / f"{outname}_model.pdb"
                    with path.open("w") as f:
                        f.write(
                            to_pdb(new_structure, plddts=plddts, boltz2=self.boltz2)
                        )
                elif self.output_format == "mmcif":
                    path = models_dir / f"{outname}_model.cif"
                    with path.open("w") as f:
                        f.write(
                            to_mmcif(new_structure, plddts=plddts, boltz2=self.boltz2)
                        )
                else:
                    path = models_dir / f"{outname}_model.npz"
                    np.savez_compressed(path, **asdict(new_structure))

                if self.boltz2 and record.affinity and model_idx == best_model_idx:
                    handoff_dir = self.affinity_output_dir
                    if self.use_record_subdir:
                        handoff_dir = handoff_dir / record.id
                    handoff_dir.mkdir(parents=True, exist_ok=True)
                    path = handoff_dir / f"pre_affinity_seed-{self.seed}.npz"
                    np.savez_compressed(path, **asdict(new_structure))
                    np.array(atoms["coords"][:, None], dtype=Coords)

                # Save confidence summary
                if "plddt" in prediction:
                    path = summary_dir / f"{outname}_summary_confidences.json"
                    confidence_summary_dict = {}
                    for key in [
                        "confidence_score",
                        "ptm",
                        "iptm",
                        "ligand_iptm",
                        "protein_iptm",
                        "complex_plddt",
                        "complex_iplddt",
                        "complex_pde",
                        "complex_ipde",
                    ]:
                        confidence_summary_dict[key] = prediction[key][model_idx].item()
                    confidence_summary_dict["chains_ptm"] = {
                        idx: prediction["pair_chains_iptm"][idx][idx][model_idx].item()
                        for idx in prediction["pair_chains_iptm"]
                    }
                    confidence_summary_dict["pair_chains_iptm"] = {
                        idx1: {
                            idx2: prediction["pair_chains_iptm"][idx1][idx2][
                                model_idx
                            ].item()
                            for idx2 in prediction["pair_chains_iptm"][idx1]
                        }
                        for idx1 in prediction["pair_chains_iptm"]
                    }
                    with path.open("w") as f:
                        f.write(
                            json.dumps(
                                confidence_summary_dict,
                                indent=4,
                            )
                        )

                    # Save plddt
                    plddt = prediction["plddt"][model_idx]
                    self._write_confidence(full_data_dir, f"plddt_{outname}", "plddt", plddt.cpu().numpy())

                # Save pae
                if "pae" in prediction:
                    pae = prediction["pae"][model_idx]
                    self._write_confidence(full_data_dir, f"pae_{outname}", "pae", pae.cpu().numpy())

                # Save pde
                if "pde" in prediction:
                    pde = prediction["pde"][model_idx]
                    self._write_confidence(full_data_dir, f"pde_{outname}", "pde", pde.cpu().numpy())

            # Save embeddings
            if self.write_embeddings and "s" in prediction and "z" in prediction:
                s = prediction["s"].cpu().numpy()
                z = prediction["z"].cpu().numpy()

                embeddings_dir = record_dir / "embeddings"
                embeddings_dir.mkdir(exist_ok=True)
                path = embeddings_dir / f"seed-{self.seed}_embeddings.npz"
                np.savez_compressed(path, s=s, z=z)

    def on_predict_epoch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,
    ) -> None:
        """Fail the invocation if any rank failed to predict an example."""
        _check_prediction_failures(
            self.failed, stage="structure", seed=self.seed, pl_module=pl_module
        )


class BoltzAffinityWriter(BasePredictionWriter):
    """Custom writer for predictions."""

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        *,
        seed: int,
        use_record_subdir: bool = True,
    ) -> None:
        """Initialize the writer.

        Parameters
        ----------
        output_dir : str
            The directory to save the predictions.

        """
        super().__init__(write_interval="batch")
        self.failed = 0
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.use_record_subdir = use_record_subdir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_on_batch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
        prediction: dict[str, Tensor],
        batch_indices: list[int],  # noqa: ARG002
        batch: dict[str, Tensor],
        batch_idx: int,  # noqa: ARG002
        dataloader_idx: int,  # noqa: ARG002
    ) -> None:
        """Write the predictions to disk."""
        if prediction["exception"]:
            self.failed += 1
            return
        # Dump affinity summary
        affinity_summary = {}
        pred_affinity_value = prediction["affinity_pred_value"]
        pred_affinity_probability = prediction["affinity_probability_binary"]
        affinity_summary = {
            "affinity_pred_value": pred_affinity_value.item(),
            "affinity_probability_binary": pred_affinity_probability.item(),
        }
        if "affinity_pred_value1" in prediction:
            pred_affinity_value1 = prediction["affinity_pred_value1"]
            pred_affinity_probability1 = prediction["affinity_probability_binary1"]
            pred_affinity_value2 = prediction["affinity_pred_value2"]
            pred_affinity_probability2 = prediction["affinity_probability_binary2"]
            affinity_summary["affinity_pred_value1"] = pred_affinity_value1.item()
            affinity_summary["affinity_probability_binary1"] = (
                pred_affinity_probability1.item()
            )
            affinity_summary["affinity_pred_value2"] = pred_affinity_value2.item()
            affinity_summary["affinity_probability_binary2"] = (
                pred_affinity_probability2.item()
            )

        # Save the affinity summary
        record_dir = (
            self.output_dir / batch["record"][0].id
            if self.use_record_subdir
            else self.output_dir
        )
        record_dir.mkdir(exist_ok=True)
        affinity_dir = record_dir / "affinity"
        affinity_dir.mkdir(exist_ok=True)
        path = affinity_dir / f"seed-{self.seed}_affinity.json"

        with path.open("w") as f:
            f.write(json.dumps(affinity_summary, indent=4))

    def on_predict_epoch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,
    ) -> None:
        """Fail the invocation if any rank failed to predict an example."""
        _check_prediction_failures(
            self.failed, stage="affinity", seed=self.seed, pl_module=pl_module
        )
