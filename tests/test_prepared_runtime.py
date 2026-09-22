"""Runtime files must not become persistent inputs or completion markers."""
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from boltz import main
from boltz.data.types import Manifest
from test_af3_output_layout import (
    _prediction, _record, _touch_complete_structure_outputs, _write_structure,
)


@pytest.mark.parametrize("affinity_bytes", [None, b"", b"not parsed"])
def test_affinity_seed_requires_nonempty_final_result(tmp_path, affinity_bytes):
    record = _record("job", affinity=True)
    _touch_complete_structure_outputs(
        tmp_path, "job", seed=4, diffusion_samples=1, use_record_subdir=False
    )
    for path in tmp_path.rglob("*"):
        if path.is_file():
            path.write_bytes(b"exists, not parsed")
    (tmp_path / "pre_affinity_seed-4.npz").write_bytes(b"old handoff")
    affinity = tmp_path / "affinity" / "seed-4_affinity.json"
    if affinity_bytes is not None:
        affinity.parent.mkdir()
        affinity.write_bytes(affinity_bytes)
    pending = main.filter_inputs_structure(
        Manifest([record]), tmp_path, skip=True, seed=4
    )
    assert bool(pending.records) == (not affinity_bytes)


def test_empty_structure_files_do_not_complete_seed(tmp_path):
    record = _record("job")
    _touch_complete_structure_outputs(
        tmp_path, "job", seed=4, diffusion_samples=1, use_record_subdir=False
    )
    (tmp_path / "models" / "seed-4_sample-0_model.cif").write_bytes(b"")
    assert main.filter_inputs_structure(
        Manifest([record]), tmp_path, skip=True, seed=4
    ).records


@pytest.mark.parametrize("run_data", [True, False])
@pytest.mark.parametrize("write_json", [True, False])
@pytest.mark.parametrize("fail", [True, False])
def test_cli_separates_write_flag_and_cleans_runtime(
    tmp_path, monkeypatch, run_data, write_json, fail
):
    source = tmp_path / "job.json"
    source.write_text('{"name": "job", "sequences": []}')
    original = source.read_bytes()
    output = tmp_path / "out"
    private = tmp_path / "scratch"
    private.mkdir()
    monkeypatch.setenv("SLURM_TMPDIR", str(private))
    monkeypatch.setattr(main, "download_boltz2", lambda *a, **k: None)
    captured = {}

    # Heavy molecular parsing/model execution are outside this lifecycle test.
    # The real CLI owns flag dispatch, workspace allocation and cleanup.
    def process(*, out_dir, run_data_pipeline, write_input_json, **kwargs):
        captured.update(path=out_dir, data=run_data_pipeline, write=write_input_json)
        processed = out_dir / "processed"
        processed.mkdir(parents=True)
        Manifest([]).dump(processed / "manifest.json")
        if fail:
            raise ValueError("fixture preprocessing failure")

    monkeypatch.setattr(main, "process_inputs", process)
    result = CliRunner().invoke(main.cli, [
        "predict", str(source), "--out_dir", str(output),
        "--cache", str(tmp_path / "cache"), "--seeds", "4",
        "-D", str(run_data).lower(), "-P", "true",
        "--write_input_json", str(write_json).lower(),
    ])
    assert "No such option" not in result.output
    assert result.exit_code == (1 if fail else 0), result.output
    assert captured["data"] is run_data
    assert captured["write"] is write_json
    assert captured["path"].parent == private
    assert not captured["path"].exists()
    assert not list(output.rglob("processed"))
    assert source.read_bytes() == original


@pytest.mark.parametrize("write_json", [True, False])
def test_data_only_write_switch_controls_overwrite(tmp_path, monkeypatch, write_json):
    source = tmp_path / "job.json"
    original = {"name": "job", "sequences": [{"ligand": {"id": "A", "smiles": "C"}}]}
    source.write_text(json.dumps(original))
    out = tmp_path / "out"
    destination = out / "job" / "job_data.json"
    destination.parent.mkdir(parents=True)
    destination.write_text('{"old": true}')
    monkeypatch.setattr(main, "load_canonicals", lambda path: {})
    # A real SMILES-only molecule exercises parsing/writing without a CCD download.
    main.prepare_msa_inputs(
        data=[source], out_dir=out, prepared_output_root=out,
        ccd_path=tmp_path / "ccd", mol_dir=tmp_path / "mols",
        boltz2=True, use_msa_server=False, msa_server_url="unused",
        msa_pairing_strategy="greedy", write_input_json=write_json,
    )
    saved = json.loads(destination.read_text())
    assert saved == (original if write_json else {"old": True})


@pytest.mark.parametrize("missing", ["structure", "affinity", "neither"])
def test_affinity_restarts_both_stages_as_one_seed(tmp_path, monkeypatch, missing):
    from types import SimpleNamespace
    import numpy as np
    import torch
    from boltz.data.module.inferencev2 import load_input
    from boltz.data.write import writer as writers

    source = tmp_path / "job.json"
    source.write_text('{"name": "job", "sequences": []}')
    output = tmp_path / "out"
    job = output / "job"
    record = _record("job", affinity=True)
    _touch_complete_structure_outputs(job, "job", seed=4, diffusion_samples=2, use_record_subdir=False)
    affinity_path = job / "affinity" / "seed-4_affinity.json"
    affinity_path.parent.mkdir()
    affinity_path.write_text("old affinity, not parsed by skip")
    if missing == "structure":
        (job / "models" / "seed-4_sample-0_model.cif").unlink()
    elif missing == "affinity":
        affinity_path.unlink()
    monkeypatch.setattr(main, "download_boltz2", lambda *a, **k: None)
    monkeypatch.setattr(writers, "to_mmcif", lambda *a, **k: "new structure")
    calls, workspaces = [], []

    def process(*, out_dir, **kwargs):
        workspaces.append(out_dir)
        for directory in [out_dir / "processed" / "structures", out_dir / "processed" / "msa"]:
            directory.mkdir(parents=True)
        _write_structure(out_dir / "processed" / "structures", "job")
        Manifest([record]).dump(out_dir / "processed" / "manifest.json")

    # Only model evaluation is replaced. Real CLI selects stages; real writers
    # and the real affinity loader exchange the selected structure via scratch.
    class Trainer:
        def __init__(self, *, callbacks, **kwargs):
            self.callbacks = callbacks

        def predict(self, model, *, datamodule, **kwargs):
            writer = self.callbacks[0]
            if isinstance(writer, writers.BoltzAffinityWriter):
                calls.append("affinity")
                loaded = load_input(
                    record, target_dir=datamodule.target_dir, msa_dir=datamodule.msa_dir,
                    affinity=True, seed=4, use_record_subdir=False,
                )
                np.testing.assert_allclose(loaded.structure.atoms["coords"], [[9, 0, 0]])
                prediction = {"exception": False, "affinity_pred_value": torch.tensor(1.0),
                              "affinity_probability_binary": torch.tensor(0.5)}
            else:
                calls.append("structure")
                prediction = _prediction(include_embeddings=False)
            writer.write_on_batch_end(None, None, prediction, [], {"record": [record]}, 0, 0)

    monkeypatch.setattr(main, "process_inputs", process)
    monkeypatch.setattr(main, "Trainer", Trainer)
    monkeypatch.setattr(main, "Boltz2InferenceDataModule", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(main.Boltz2, "load_from_checkpoint", lambda *a, **k: SimpleNamespace(eval=lambda: None))
    result = CliRunner().invoke(main.cli, [
        "predict", str(source), "--out_dir", str(output), "--cache", str(tmp_path / "cache"),
        "-D", "false", "--write_input_json", "false", "--seeds", "4", "--diffusion_samples", "2", "--skip",
    ])
    assert result.exit_code == 0, f"{result.output}\n{result.exception}"
    assert calls == ([] if missing == "neither" else ["structure", "affinity"])
    assert affinity_path.stat().st_size > 0
    assert all(not path.exists() for path in workspaces)
    assert not list(output.rglob("pre_affinity*"))
    assert not list(output.rglob("processed"))


def test_external_multirank_launch_is_rejected_before_work(tmp_path, monkeypatch):
    source = tmp_path / "job.json"
    source.write_text('{"name": "job", "sequences": []}')
    monkeypatch.setenv("WORLD_SIZE", "2")
    def forbid_download(*args, **kwargs):
        raise AssertionError("must not download in a rejected launch")
    monkeypatch.setattr(main, "download_boltz2", forbid_download)
    result = CliRunner().invoke(main.cli, ["predict", str(source), "--out_dir", str(tmp_path / "out")])
    assert result.exit_code == 2
    assert "one entry process" in result.output
    assert not (tmp_path / "out").exists()


def test_failed_json_publication_restores_previous_resources(tmp_path, monkeypatch):
    from types import SimpleNamespace
    source = tmp_path / "job.json"
    (tmp_path / "new.csv").write_text("key,sequence\n-1,AAAA\n")
    source.write_text(json.dumps({"name": "job", "sequences": [{"protein": {
        "id": "A", "sequence": "AAAA", "msa": "new.csv",
    }}]}))
    output = tmp_path / "out"
    old_msa = output / "msas" / "job__A_msa.csv"
    old_msa.parent.mkdir(parents=True)
    old_msa.write_bytes(b"old MSA")
    old_json = output / "job_data.json"
    old_json.write_bytes(b"old JSON")
    original_replace = main.os.replace
    def fail_publish(source, destination):
        if Path(destination) == old_json:
            raise OSError("fixture JSON publish failure")
        return original_replace(source, destination)
    monkeypatch.setattr(main.os, "replace", fail_publish)
    with pytest.raises(OSError, match="fixture JSON"):
        main.write_data_json(source, output, SimpleNamespace(record=SimpleNamespace(id="job")), {})
    assert old_msa.read_bytes() == b"old MSA"
    assert old_json.read_bytes() == b"old JSON"
