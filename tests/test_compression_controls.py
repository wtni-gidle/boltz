"""Compression choices preserve native MSA routes and confidence array groups."""
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest

from boltz import main
from boltz.data.parse.compression import open_maybe_compressed_text
from boltz.data.parse.json import parse_json
from boltz.data.types import Manifest
from boltz.data.write import writer as writer_module
from test_af3_output_layout import _prediction, _record, _write_structure
from test_prepared_templates import alanine_ccd, template_cif
from test_unpaired_replacement_acceptance import _source, _process


@pytest.mark.parametrize("compressed", [None, False, True])
def test_bundle_zstd_to_selected_text_preserves_templates(tmp_path, alanine_ccd, template_cif, compressed):
    source = _source(tmp_path, template_cif)
    target = parse_json(source, alanine_ccd, tmp_path, True, tmp_path / "runtime")
    old = main.write_data_json(source, tmp_path / "old", target, {}, alanine_ccd, tmp_path, compress_fold_input=True)
    target = parse_json(old, alanine_ccd, tmp_path, True, tmp_path / "runtime2")
    options = {} if compressed is None else {"compress_fold_input": compressed}
    new = main.write_data_json(old, tmp_path / "new", target, {}, alanine_ccd, tmp_path, **options)
    schema = json.loads(new.read_text())
    msa = schema["sequences"][0]["protein"]["msa"]
    assert msa["unpaired"] == "msas/job__A_unpairedmsa.a3m" + (".zst" if compressed else "")
    with open_maybe_compressed_text(new.parent / msa["unpaired"]) as handle:
        assert handle.read() == ">query\nAAAA\n>unpaired\nCCAA\n"
    assert schema["templates"][0]["chains"][0]["mmcifPath"].endswith(".cif" + (".zst" if compressed else ""))
    loaded, _, _ = _process(new, alanine_ccd, tmp_path / "consumed", tmp_path / "unused", consume_features=False)
    assert loaded.msa


@pytest.mark.parametrize("compressed", [False, True])
def test_scalar_csv_preserves_keys_order_through_compressed_publication(tmp_path, alanine_ccd, compressed):
    csv = tmp_path / "input.csv"
    contents = "key,sequence\n-1,AAAA\n9,CAAA\n3,ACAA\n"
    csv.write_text(contents)
    source = tmp_path / "job.json"
    source.write_text(json.dumps({"name": "job", "sequences": [{"protein": {"id": "A", "sequence": "AAAA", "msa": str(csv)}}]}))
    target = parse_json(source, alanine_ccd, tmp_path, True, tmp_path / "native")
    prepared = main.write_data_json(source, tmp_path / "public/job", target, {}, alanine_ccd, tmp_path, compress_fold_input=compressed)
    ref = json.loads(prepared.read_text())["sequences"][0]["protein"]["msa"]
    assert ref == "msas/job__A_msa.csv" + (".zst" if compressed else "")
    with open_maybe_compressed_text(prepared.parent / ref) as handle:
        assert handle.read() == contents
    loaded, _, _ = _process(prepared, alanine_ccd, tmp_path / "consume", tmp_path / "unused", consume_features=False)
    np.testing.assert_array_equal(loaded.msa[0].sequences["taxonomy"], [-1, 9, 3])


@pytest.mark.parametrize("compressed", [None, False, True])
def test_confidence_formats_shapes_cleanup_and_current_skip(tmp_path, monkeypatch, compressed):
    data = tmp_path / "data"
    data.mkdir()
    _write_structure(data, "job")
    monkeypatch.setattr(writer_module, "to_mmcif", lambda *a, **k: "data_model\n#\n")
    output = tmp_path / "out"
    record = _record("job")
    for selection in (not bool(compressed), compressed):
        options = {} if selection is None else {"compress_full_confidence": selection}
        writer = writer_module.BoltzWriter(str(data), str(output), seed=7, boltz2=True, write_embeddings=True, use_record_subdir=False, **options)
        prediction = _prediction()
        writer.write_on_batch_end(None, None, prediction, [], {"record": [record]}, 0, 0)
        for kind in ("plddt", "pae", "pde"):
            path = output / "full_data" / (f"{kind}_seed-7_sample-0." + ("npz" if selection else "json"))
            assert path.is_file()
            assert not path.with_suffix(".json" if selection else ".npz").exists()
            if selection:
                with ZipFile(path) as archive:
                    assert all(item.compress_type == ZIP_DEFLATED for item in archive.infolist())
                with np.load(path) as archive:
                    actual = dict(archive)
            else:
                actual = json.loads(path.read_text())
            assert set(actual) == {kind}
            np.testing.assert_array_equal(actual[kind], prediction[kind][0].numpy())
        assert not main.filter_inputs_structure(Manifest([record]), output, skip=True, seed=7, diffusion_samples=2, write_full_pae=True, write_full_pde=True, **options).records
        assert (output / "embeddings/seed-7_embeddings.npz").is_file()
    path.unlink()
    assert main.filter_inputs_structure(Manifest([record]), output, skip=True, seed=7, diffusion_samples=2, write_full_pae=True, write_full_pde=True, **options).records


@pytest.mark.parametrize("write", [None, False, True])
def test_inference_cli_skip_publishes_current_plain_input_without_search(tmp_path, monkeypatch, alanine_ccd, write):
    from click.testing import CliRunner
    from boltz.data.msa import pipeline
    msa = tmp_path / "input.a3m"
    msa.write_text(">q\nAAAA\n>hit\nCAAA\n")
    source = tmp_path / "job.json"
    source.write_text(json.dumps({"name": "job", "sequences": [{"protein": {"id": "A", "sequence": "AAAA", "msa": str(msa)}}]}))
    output = tmp_path / "out"
    for relative in ("models/seed-7_sample-0_model.cif", "summary_confidences/seed-7_sample-0_summary_confidences.json", "full_data/plddt_seed-7_sample-0.json"):
        path = output / "job" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("complete without parsing")
    monkeypatch.setattr(main, "download_boltz2", lambda *a, **k: None)
    monkeypatch.setattr(main, "load_canonicals", lambda *a, **k: alanine_ccd)
    def forbidden(*args, **kwargs):
        pytest.fail("inference-only fully skipped job must not search or load a model")
    monkeypatch.setattr(pipeline, "run_mmseqs2", forbidden)
    monkeypatch.setattr(main.Boltz2, "load_from_checkpoint", forbidden)
    args = ["predict", str(source), "--out_dir", str(output), "--cache", str(tmp_path / "cache"), "-D", "false", "--seeds", "7", "--diffusion_samples", "1", "--skip", "--accelerator", "cpu"]
    if write is not None:
        args += ["--write_input_json", str(write).lower()]
    result = CliRunner().invoke(main.cli, args)
    assert result.exit_code == 0, (result.output, result.exception)
    prepared = output / "job/job_data.json"
    assert prepared.exists() is (write is not False)
    if prepared.exists():
        resource = output / "job/msas/job__A_msa.a3m"
        assert resource.read_text().endswith(">hit\nCAAA\n")
        msa.write_text(">q\nAAAA\n>hit\nACAA\n")
        result = CliRunner().invoke(main.cli, args)
        assert result.exit_code == 0, (result.output, result.exception)
        assert resource.read_text().endswith(">hit\nACAA\n")
