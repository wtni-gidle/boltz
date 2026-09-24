"""Swallowed prediction failures must become a failing CLI/job outcome."""

import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from click.testing import CliRunner

from boltz import main
from boltz.data.types import Manifest
from boltz.data.write import writer as writers
from test_af3_output_layout import _prediction, _record, _write_structure


@pytest.mark.parametrize("stage", ["structure", "affinity"])
@pytest.mark.parametrize("failed_records", [0, 1, 2])
def test_cli_fails_if_any_prediction_failed_and_preserves_successes(
    tmp_path, monkeypatch, stage, failed_records
):
    # Removing either writer's failure propagation must break the exit assertion.
    source = tmp_path / "inputs"
    source.mkdir()
    records = [_record(name, affinity=stage == "affinity") for name in ["first", "second"]]
    for record in records:
        (source / f"{record.id}.json").write_text('{"sequences": []}')
    output = tmp_path / "out"
    workspaces = []

    def process(*, out_dir, **kwargs):
        workspaces.append(out_dir)
        structures = out_dir / "processed" / "structures"
        structures.mkdir(parents=True)
        for record in records:
            _write_structure(structures, record.id)
        Manifest(records).dump(out_dir / "processed" / "manifest.json")

    # Checkpoint loading and GPU evaluation are expensive. Keep CLI dispatch,
    # real callback batch/epoch hooks, and public file writing in this test.
    class Trainer:
        def __init__(self, *, callbacks, **kwargs):
            self.callbacks = callbacks

        def predict(self, model, *, datamodule, **kwargs):
            writer = self.callbacks[0]
            current_stage = "affinity" if isinstance(writer, writers.BoltzAffinityWriter) else "structure"
            for index, record in enumerate(datamodule.manifest.records):
                fail = current_stage == stage and writer.seed == 5 and index < failed_records
                if fail:
                    prediction = {"exception": True}
                elif current_stage == "affinity":
                    prediction = {"exception": False, "affinity_pred_value": torch.tensor(1.0),
                                  "affinity_probability_binary": torch.tensor(0.5)}
                else:
                    prediction = _prediction(include_embeddings=False)
                writer.write_on_batch_end(self, model, prediction, [], {"record": [record]}, index, 0)
            writer.on_predict_epoch_end(self, model)

    monkeypatch.setattr(main, "download_boltz2", lambda *a, **k: None)
    monkeypatch.setattr(main, "process_inputs", process)
    monkeypatch.setattr(main, "Trainer", Trainer)
    monkeypatch.setattr(main, "Boltz2InferenceDataModule", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(main.Boltz2, "load_from_checkpoint", lambda *a, **k: SimpleNamespace(eval=lambda: None, device=torch.device("cpu")))
    monkeypatch.setattr(writers, "to_mmcif", lambda *a, **k: "successful structure")

    result = CliRunner().invoke(main.cli, [
        "predict", str(source), "--out_dir", str(output), "--cache", str(tmp_path / "cache"),
        "-D", "false", "--write_input_json", "false", "--seeds", "4,5", "--diffusion_samples", "2",
    ])
    assert result.exit_code == (1 if failed_records else 0), f"{result.output}\n{result.exception}"
    if failed_records:
        summary = f"{result.output}\n{result.exception}".lower()
        assert stage in summary and "seed 5" in summary and "failed" in summary
        assert f"{failed_records} failed" in summary

    def public_result(name, seed):
        if stage == "affinity":
            return output / name / "affinity" / f"seed-{seed}_affinity.json"
        return output / name / "models" / f"seed-{seed}_sample-0_model.cif"

    for index, record in enumerate(records):
        assert public_result(record.id, 4).stat().st_size > 0
        assert public_result(record.id, 5).exists() == (index >= failed_records)
    assert all(not workspace.exists() for workspace in workspaces)


def _distributed_writer_end(rank, rendezvous, output, stage):
    """Exercise a real collective with only rank one observing a failure."""
    torch.distributed.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        writer_type = writers.BoltzWriter if stage == "structure" else writers.BoltzAffinityWriter
        writer = writer_type(data_dir=output, output_dir=output, seed=19)
        if rank == 1:
            writer.write_on_batch_end(None, None, {"exception": True}, [], {}, 0, 0)
        error = None
        try:
            writer.on_predict_epoch_end(None, SimpleNamespace(device=torch.device("cpu")))
        except Exception as exc:
            error = str(exc)
        (Path(output) / f"rank-{rank}.json").write_text(json.dumps(error))
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("stage", ["structure", "affinity"])
@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="requires CPU gloo")
def test_failure_on_other_rank_fails_every_rank(tmp_path, stage):
    # A local-only check (or a collective after a local conditional) fails here.
    output = tmp_path / "out"
    output.mkdir()
    torch.multiprocessing.spawn(
        _distributed_writer_end, args=(str(tmp_path / "rendezvous"), str(output), stage), nprocs=2,
    )
    for rank in range(2):
        error = json.loads((output / f"rank-{rank}.json").read_text())
        assert error is not None, f"rank {rank} incorrectly reported success"
        assert stage in error.lower() and "seed 19" in error.lower()
        assert "1" in error and "failed" in error.lower()
