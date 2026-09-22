import json
from pathlib import Path

import numpy as np
import pytest
import torch

from boltz.data.module.inferencev2 import load_input
from boltz.data.types import (
    AffinityInfo,
    AtomV2,
    BondV2,
    Chain,
    ChainInfo,
    Coords,
    Ensemble,
    Interface,
    Manifest,
    Record,
    Residue,
    StructureInfo,
    StructureV2,
)
from boltz.data.write import writer as writer_module
from boltz.data.write.writer import BoltzAffinityWriter, BoltzWriter
from boltz.main import filter_inputs_affinity, filter_inputs_structure


def _record(record_id: str, *, affinity: bool = False) -> Record:
    affinity_info = AffinityInfo(chain_id=0, mw=100.0) if affinity else None

    return Record(
        id=record_id,
        structure=StructureInfo(num_chains=1),
        chains=[
            ChainInfo(
                chain_id=0,
                chain_name="A",
                mol_type=0,
                cluster_id=-1,
                msa_id=-1,
                num_residues=1,
            )
        ],
        interfaces=[],
        affinity=affinity_info,
    )


def _write_structure(data_dir: Path, record_id: str) -> None:
    atoms = np.array(
        [("C", [0.0, 0.0, 0.0], True, 0.0, 1.0)],
        dtype=AtomV2,
    )
    residues = np.array(
        [("ALA", 0, 0, 0, 1, 0, 0, True, True)],
        dtype=Residue,
    )
    chains = np.array(
        [("A", 0, 0, 0, 0, 0, 1, 0, 1, 0)],
        dtype=Chain,
    )
    structure = StructureV2(
        atoms=atoms,
        bonds=np.array([], dtype=BondV2),
        residues=residues,
        chains=chains,
        interfaces=np.array([], dtype=Interface),
        mask=np.ones(1, dtype=bool),
        coords=np.array([([0.0, 0.0, 0.0],)], dtype=Coords),
        ensemble=np.array([(0, 1)], dtype=Ensemble),
    )
    structure.dump(data_dir / f"{record_id}.npz")


def _prediction(
    *,
    include_pae: bool = True,
    include_pde: bool = True,
    include_embeddings: bool = True,
) -> dict:
    prediction = {
        "exception": False,
        "coords": torch.tensor(
            [[[1.0, 0.0, 0.0]], [[9.0, 0.0, 0.0]]],
            dtype=torch.float32,
        ),
        "masks": torch.tensor([[True]]),
        # Sample 1 deliberately ranks above sample 0. Public names must still
        # use diffusion sample indices, while affinity consumes sample 1.
        "confidence_score": torch.tensor([0.1, 0.9]),
        "ptm": torch.tensor([0.8, 0.7]),
        "iptm": torch.tensor([0.7, 0.6]),
        "ligand_iptm": torch.tensor([0.6, 0.5]),
        "protein_iptm": torch.tensor([0.5, 0.4]),
        "complex_plddt": torch.tensor([0.9, 0.8]),
        "complex_iplddt": torch.tensor([0.8, 0.7]),
        "complex_pde": torch.tensor([0.1, 0.2]),
        "complex_ipde": torch.tensor([0.2, 0.3]),
        "plddt": torch.tensor([[0.91], [0.81]]),
        "pair_chains_iptm": {0: {0: torch.tensor([0.7, 0.6])}},
    }
    if include_pae:
        prediction["pae"] = torch.tensor([[[1.0]], [[2.0]]])
    if include_pde:
        prediction["pde"] = torch.tensor([[[3.0]], [[4.0]]])
    if include_embeddings:
        prediction["s"] = torch.tensor([[1.0, 2.0]])
        prediction["z"] = torch.tensor([[[3.0, 4.0]]])
    return prediction


@pytest.mark.parametrize(
    ("output_format", "suffix", "converter_name"),
    [("mmcif", "cif", "to_mmcif"), ("pdb", "pdb", "to_pdb")],
)
def test_writer_uses_seed_sample_layout_and_preserves_full_data_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output_format: str,
    suffix: str,
    converter_name: str,
) -> None:
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "predictions"
    data_dir.mkdir()
    record = _record("target")
    _write_structure(data_dir, record.id)
    monkeypatch.setattr(
        writer_module,
        converter_name,
        lambda *_args, **_kwargs: "MODEL",
    )

    writer = BoltzWriter(
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        seed=42,
        output_format=output_format,
        boltz2=True,
        write_embeddings=True,
        use_record_subdir=False,
    )
    writer.write_on_batch_end(
        None,
        None,
        _prediction(),
        [],
        {"record": [record]},
        0,
        0,
    )

    target_dir = output_dir
    for sample_idx in range(2):
        basename = f"seed-42_sample-{sample_idx}"
        assert (target_dir / "models" / f"{basename}_model.{suffix}").read_text() == (
            "MODEL"
        )
        assert (
            target_dir / "summary_confidences" / f"{basename}_summary_confidences.json"
        ).is_file()
        for metric in ("plddt", "pae", "pde"):
            metric_path = target_dir / "full_data" / f"{metric}_{basename}.npz"
            with np.load(metric_path) as data:
                assert metric in data.files

    assert not list(target_dir.rglob("*rank*"))
    assert not list(target_dir.rglob("ranking.csv"))
    sample_zero_summary = json.loads(
        (
            target_dir
            / "summary_confidences"
            / "seed-42_sample-0_summary_confidences.json"
        ).read_text()
    )
    assert sample_zero_summary["confidence_score"] == pytest.approx(0.1)
    embeddings_path = target_dir / "embeddings" / "seed-42_embeddings.npz"
    with np.load(embeddings_path) as data:
        assert set(data.files) == {"s", "z"}


def test_writer_omits_optional_arrays_and_embeddings_when_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "predictions"
    data_dir.mkdir()
    record = _record("target")
    _write_structure(data_dir, record.id)
    monkeypatch.setattr(
        writer_module,
        "to_mmcif",
        lambda *_args, **_kwargs: "MODEL",
    )

    writer = BoltzWriter(
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        seed=7,
        boltz2=True,
        write_embeddings=True,
        use_record_subdir=False,
    )
    writer.write_on_batch_end(
        None,
        None,
        _prediction(
            include_pae=False,
            include_pde=False,
            include_embeddings=False,
        ),
        [],
        {"record": [record]},
        0,
        0,
    )

    target_dir = output_dir
    assert len(list((target_dir / "full_data").glob("plddt_*.npz"))) == 2
    assert not list((target_dir / "full_data").glob("pae_*.npz"))
    assert not list((target_dir / "full_data").glob("pde_*.npz"))
    assert not (target_dir / "embeddings").exists()


def test_writer_keeps_records_separate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "predictions"
    data_dir.mkdir()
    monkeypatch.setattr(
        writer_module,
        "to_mmcif",
        lambda *_args, **_kwargs: "MODEL",
    )
    writer = BoltzWriter(
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        seed=3,
        boltz2=True,
    )

    for record_id in ("target_a", "target_b"):
        record = _record(record_id)
        _write_structure(data_dir, record.id)
        writer.write_on_batch_end(
            None,
            None,
            _prediction(include_embeddings=False),
            [],
            {"record": [record]},
            0,
            0,
        )

    for record_id in ("target_a", "target_b"):
        assert (
            output_dir / record_id / "models" / "seed-3_sample-0_model.cif"
        ).is_file()


def test_affinity_handoff_uses_best_confidence_sample(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "predictions"
    data_dir.mkdir()
    record = _record("target", affinity=True)
    _write_structure(data_dir, record.id)
    monkeypatch.setattr(
        writer_module,
        "to_mmcif",
        lambda *_args, **_kwargs: "MODEL",
    )
    writer = BoltzWriter(
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        seed=11,
        boltz2=True,
        use_record_subdir=False,
        affinity_output_dir=tmp_path / "private_affinity",
    )
    writer.write_on_batch_end(
        None,
        None,
        _prediction(include_embeddings=False),
        [],
        {"record": [record]},
        0,
        0,
    )

    handoff_path = tmp_path / "private_affinity" / "pre_affinity_seed-11.npz"
    assert not (output_dir / "pre_affinity_seed-11.npz").exists()
    handoff = StructureV2.load(handoff_path)
    np.testing.assert_allclose(handoff.atoms["coords"], [[9.0, 0.0, 0.0]])

    msa_dir = tmp_path / "msa"
    msa_dir.mkdir()
    affinity_input = load_input(
        record,
        target_dir=tmp_path / "private_affinity",
        msa_dir=msa_dir,
        affinity=True,
        seed=11,
        use_record_subdir=False,
    )
    np.testing.assert_allclose(
        affinity_input.structure.atoms["coords"],
        [[9.0, 0.0, 0.0]],
    )


def test_affinity_writer_uses_seed_level_output(tmp_path: Path) -> None:
    output_dir = tmp_path / "predictions"
    record = _record("target", affinity=True)
    writer = BoltzAffinityWriter(
        data_dir=str(tmp_path / "data"),
        output_dir=str(output_dir),
        seed=13,
        use_record_subdir=False,
    )
    writer.write_on_batch_end(
        None,
        None,
        {
            "exception": False,
            "affinity_pred_value": torch.tensor(1.5),
            "affinity_probability_binary": torch.tensor(0.75),
        },
        [],
        {"record": [record]},
        0,
        0,
    )

    assert (output_dir / "affinity" / "seed-13_affinity.json").is_file()


def _touch_complete_structure_outputs(
    out_dir: Path,
    record_id: str,
    *,
    seed: int,
    diffusion_samples: int,
    suffix: str = "cif",
    include_pae: bool = False,
    include_pde: bool = False,
    include_embeddings: bool = False,
    use_record_subdir: bool = True,
) -> None:
    target_dir = out_dir
    if use_record_subdir:
        target_dir = target_dir / record_id
    models_dir = target_dir / "models"
    summary_dir = target_dir / "summary_confidences"
    full_data_dir = target_dir / "full_data"
    for directory in (models_dir, summary_dir, full_data_dir):
        directory.mkdir(parents=True, exist_ok=True)

    for sample_idx in range(diffusion_samples):
        basename = f"seed-{seed}_sample-{sample_idx}"
        (models_dir / f"{basename}_model.{suffix}").write_bytes(b"nonempty")
        (summary_dir / f"{basename}_summary_confidences.json").write_bytes(b"nonempty")
        (full_data_dir / f"plddt_{basename}.npz").write_bytes(b"nonempty")
        if include_pae:
            (full_data_dir / f"pae_{basename}.npz").write_bytes(b"nonempty")
        if include_pde:
            (full_data_dir / f"pde_{basename}.npz").write_bytes(b"nonempty")

    if include_embeddings:
        embeddings_dir = target_dir / "embeddings"
        embeddings_dir.mkdir(exist_ok=True)
        (embeddings_dir / f"seed-{seed}_embeddings.npz").write_bytes(b"nonempty")


def test_skip_uses_direct_job_categories(tmp_path: Path) -> None:
    prefix = "seed-7_sample-0"
    for relative in (
        f"models/{prefix}_model.cif",
        f"summary_confidences/{prefix}_summary_confidences.json",
        f"full_data/plddt_{prefix}.npz",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"nonempty")

    filtered = filter_inputs_structure(
        Manifest([_record("target")]),
        tmp_path,
        skip=True,
        seed=7,
    )

    assert filtered.records == []


def test_wrapper_rejects_pdb_output(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from boltz.main import cli

    source = tmp_path / "target.json"
    source.write_text('{"sequences": []}\n', encoding="utf-8")
    result = CliRunner().invoke(
        cli,
        ["predict", str(source), "--output_format", "pdb"],
    )

    assert result.exit_code == 2
    assert "Invalid value for '--output_format'" in result.output


def test_legacy_prediction_directory_does_not_complete_new_root(
    tmp_path: Path,
) -> None:
    record = _record("target")
    _touch_complete_structure_outputs(
        tmp_path / "predictions",
        record.id,
        seed=7,
        diffusion_samples=1,
        use_record_subdir=False,
    )

    filtered = filter_inputs_structure(
        Manifest([record]),
        tmp_path,
        skip=True,
        seed=7,
    )

    assert [item.id for item in filtered.records] == [record.id]


def test_structure_skip_is_seed_sample_and_record_specific(tmp_path: Path) -> None:
    complete = _record("complete")
    incomplete = _record("incomplete")
    manifest = Manifest([complete, incomplete])
    _touch_complete_structure_outputs(
        tmp_path,
        complete.id,
        seed=17,
        diffusion_samples=2,
        include_pae=True,
        include_pde=True,
        include_embeddings=True,
    )
    _touch_complete_structure_outputs(
        tmp_path,
        incomplete.id,
        seed=17,
        diffusion_samples=1,
        include_pae=True,
        include_pde=True,
        include_embeddings=True,
    )

    filtered = filter_inputs_structure(
        manifest,
        tmp_path,
        skip=True,
        seed=17,
        diffusion_samples=2,
        write_full_pae=True,
        write_full_pde=True,
        write_embeddings=True,
    )

    assert [record.id for record in filtered.records] == [incomplete.id]


def test_structure_skip_honors_pdb_format(tmp_path: Path) -> None:
    record = _record("target")
    manifest = Manifest([record])
    _touch_complete_structure_outputs(
        tmp_path,
        record.id,
        seed=19,
        diffusion_samples=1,
        suffix="pdb",
        use_record_subdir=False,
    )

    filtered = filter_inputs_structure(
        manifest,
        tmp_path,
        skip=True,
        seed=19,
        output_format="pdb",
    )

    assert not filtered.records


def test_affinity_skip_is_seed_and_record_specific(tmp_path: Path) -> None:
    complete = _record("complete", affinity=True)
    incomplete = _record("incomplete", affinity=True)
    manifest = Manifest([complete, incomplete])
    affinity_dir = tmp_path / complete.id / "affinity"
    affinity_dir.mkdir(parents=True)
    (affinity_dir / "seed-23_affinity.json").write_bytes(b"nonempty")

    filtered = filter_inputs_affinity(manifest, tmp_path, skip=True, seed=23)

    assert [record.id for record in filtered.records] == [incomplete.id]


def test_single_record_affinity_skip_uses_flat_prediction_dir(tmp_path: Path) -> None:
    record = _record("target", affinity=True)
    manifest = Manifest([record])
    affinity_dir = tmp_path / "affinity"
    affinity_dir.mkdir(parents=True)
    (affinity_dir / "seed-23_affinity.json").write_bytes(b"nonempty")

    filtered = filter_inputs_affinity(manifest, tmp_path, skip=True, seed=23)

    assert not filtered.records


def test_structure_skip_regenerates_missing_affinity_handoff(tmp_path: Path) -> None:
    record = _record("target", affinity=True)
    manifest = Manifest([record])
    _touch_complete_structure_outputs(
        tmp_path,
        record.id,
        seed=29,
        diffusion_samples=1,
        use_record_subdir=False,
    )
    target_dir = tmp_path

    filtered = filter_inputs_structure(manifest, tmp_path, skip=True, seed=29)
    assert [item.id for item in filtered.records] == [record.id]

    (target_dir / "pre_affinity_seed-29.npz").write_bytes(b"nonempty")
    filtered = filter_inputs_structure(manifest, tmp_path, skip=True, seed=29)
    assert filtered.records

    (target_dir / "pre_affinity_seed-29.npz").unlink()
    affinity_dir = target_dir / "affinity"
    affinity_dir.mkdir()
    (affinity_dir / "seed-29_affinity.json").write_bytes(b"nonempty")
    filtered = filter_inputs_structure(manifest, tmp_path, skip=True, seed=29)
    assert not filtered.records


def test_structure_predictions_run_again_without_skip(tmp_path: Path) -> None:
    record = _record("target")
    manifest = Manifest([record])
    _touch_complete_structure_outputs(
        tmp_path,
        record.id,
        seed=31,
        diffusion_samples=1,
        use_record_subdir=False,
    )

    filtered = filter_inputs_structure(manifest, tmp_path, seed=31)

    assert [item.id for item in filtered.records] == [record.id]


@pytest.mark.parametrize("artifact", ["model", "summary", "plddt", "pae", "pde", "embeddings"])
@pytest.mark.parametrize("state", ["empty", "missing", "directory"])
def test_structure_skip_requires_nonempty_requested_files(tmp_path, artifact, state):
    manifest = Manifest([_record("target")])
    for seed in (7, 9):
        _touch_complete_structure_outputs(
            tmp_path, "target", seed=seed, diffusion_samples=2,
            include_pae=True, include_pde=True, include_embeddings=True,
            use_record_subdir=False,
        )
    paths = {
        "model": "models/seed-9_sample-1_model.cif",
        "summary": "summary_confidences/seed-9_sample-1_summary_confidences.json",
        "plddt": "full_data/plddt_seed-9_sample-1.npz",
        "pae": "full_data/pae_seed-9_sample-1.npz",
        "pde": "full_data/pde_seed-9_sample-1.npz",
        "embeddings": "embeddings/seed-9_embeddings.npz",
    }
    options = dict(skip=True, diffusion_samples=2, write_full_pae=True,
                   write_full_pde=True, write_embeddings=True)
    assert not filter_inputs_structure(manifest, tmp_path, seed=9, **options).records
    damaged = tmp_path / paths[artifact]
    damaged.unlink()
    if state == "empty":
        damaged.touch()
    elif state == "directory":
        damaged.mkdir()
    assert not filter_inputs_structure(manifest, tmp_path, seed=7, **options).records
    assert filter_inputs_structure(manifest, tmp_path, seed=9, **options).records == manifest.records
    if artifact in ("pae", "pde", "embeddings"):
        assert not filter_inputs_structure(manifest, tmp_path, seed=9, skip=True, diffusion_samples=2).records


@pytest.mark.parametrize("damage", ["missing", "empty"])
def test_cli_reruns_all_samples_only_for_incomplete_seed(tmp_path, monkeypatch, damage):
    from types import SimpleNamespace
    from click.testing import CliRunner
    import boltz.main as main

    source = tmp_path / "target_data.json"
    source.write_text('{"sequences": []}')
    output_root = tmp_path / "out"
    job = output_root / "target"
    for seed in (7, 9):
        _touch_complete_structure_outputs(job, "target", seed=seed,
                                          diffusion_samples=2, use_record_subdir=False)
    preserved = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in job.rglob("*") if p.is_file()}
    calls = []

    def process_inputs(*, out_dir, **kwargs):
        processed = out_dir / "processed"
        processed.mkdir(parents=True)
        Manifest([_record("target")]).dump(processed / "manifest.json")

    def load_model(*args, **kwargs):
        return SimpleNamespace(eval=lambda: None, samples=kwargs["predict_args"]["diffusion_samples"])

    class Trainer:
        def __init__(self, **kwargs):
            self.callbacks = kwargs["callbacks"]

        def predict(self, model, *, datamodule, return_predictions):
            seed = self.callbacks[0].seed
            calls.append((seed, model.samples, [r.id for r in datamodule.manifest.records]))
            _touch_complete_structure_outputs(job, "target", seed=seed,
                                              diffusion_samples=model.samples, use_record_subdir=False)

    monkeypatch.setattr(main, "download_boltz2", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "process_inputs", process_inputs)
    monkeypatch.setattr(main.Boltz2, "load_from_checkpoint", load_model)
    monkeypatch.setattr(main, "Boltz2InferenceDataModule", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(main, "Trainer", Trainer)
    monkeypatch.setattr(main, "seed_everything", lambda *args: None)
    arguments = ["predict", str(source), "--out_dir", str(output_root), "--cache", str(tmp_path / "cache"),
                 "--seeds", "7,9", "--diffusion_samples", "2", "--sampling_steps", "17",
                 "--recycling_steps", "3", "--accelerator", "cpu", "--skip", "-D", "false", "-P", "true"]
    complete = CliRunner().invoke(main.cli, arguments)
    assert complete.exit_code == 0, complete.exception
    assert calls == []
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in preserved} == preserved
    damaged = job / "models/seed-9_sample-1_model.cif"
    if damage == "missing":
        damaged.unlink()
    else:
        damaged.write_bytes(b"")
    rerun = CliRunner().invoke(main.cli, arguments)
    assert rerun.exit_code == 0, rerun.exception
    assert calls == [(9, 2, ["target"])]
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in preserved if "seed-7_" in p.name} == {
        p: value for p, value in preserved.items() if "seed-7_" in p.name
    }
    assert damaged.stat().st_size > 0
