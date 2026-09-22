"""Replacement acceptance through the current fork's real feature consumers.

These are characterization/regression tests, not evidence of equivalence to an
official upstream revision. No network, model, GPU, or full featurizer is needed.
"""
import json
import pickle

import numpy as np
import pytest

from boltz import main
from boltz.data import const
from boltz.data.feature.featurizerv2 import process_msa_features, process_template_features
from boltz.data.module.inferencev2 import load_input
from boltz.data.msa import pipeline
from boltz.data.parse.compression import write_zstd_text
from boltz.data.parse.json import parse_json
from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
from boltz.data.types import Record
from test_prepared_templates import alanine_ccd, template_cif  # noqa: F401


def _source(tmp_path, template_cif):
    source = tmp_path / "job.json"
    proteins = []
    for chain, query, paired, unpaired in [
        ("A", "AAAA", "A-AA", "CCAA"),
        ("B", "AAA", "A-A", "CCA"),
    ]:
        (tmp_path / f"{chain}_paired.a3m").write_text(f">query\n{query}\n>paired\n{paired}\n")
        (tmp_path / f"{chain}_unpaired.a3m").write_text(f">query\n{query}\n>unpaired\n{unpaired}\n")
        proteins.append({"protein": {"id": chain, "sequence": query, "msa": {
            "paired": f"{chain}_paired.a3m", "unpaired": f"{chain}_unpaired.a3m",
        }}})
    source.write_text(json.dumps({"name": "job", "sequences": proteins, "templates": [{
        "groupId": "complex", "chains": [
            {"queryChain": "A", "mmcifPath": template_cif.name, "templateChain": "X",
             "queryIndices": [0, 2], "templateIndices": [1, 3]},
            {"queryChain": "B", "mmcifPath": template_cif.name, "templateChain": "Y",
             "queryIndices": [0, 1, 2], "templateIndices": [0, 1, 2]},
        ],
    }]}))
    return source


def _process(source, ccd, runtime, public, *, run_data=False, write_json=False, consume_features=True):
    directories = {
        "msa_dir": runtime / "msa",
        "processed_msa_dir": runtime / "processed" / "msa",
        "processed_constraints_dir": runtime / "processed" / "constraints",
        "processed_templates_dir": runtime / "processed" / "templates",
        "processed_mols_dir": runtime / "processed" / "mols",
        "structure_dir": runtime / "processed" / "structures",
        "records_dir": runtime / "processed" / "records",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    main.process_input(
        path=source, ccd=ccd, mol_dir=source.parent, boltz2=True,
        run_data_pipeline=run_data, write_input_json=write_json,
        use_msa_server=True, msa_server_url="https://unused.invalid",
        msa_pairing_strategy="greedy", msa_server_username=None,
        msa_server_password=None, api_key_header=None, api_key_value=None,
        max_msa_seqs=128, prepared_output_root=public, **directories,
    )
    record = Record.load(directories["records_dir"] / "job.json")
    loaded = load_input(
        record, target_dir=directories["structure_dir"],
        msa_dir=directories["processed_msa_dir"],
        constraints_dir=directories["processed_constraints_dir"],
        template_dir=directories["processed_templates_dir"],
        extra_mols_dir=directories["processed_mols_dir"],
    )
    if not consume_features:
        return loaded, None, None
    tokenized = Boltz2Tokenizer().tokenize(loaded)
    msa = process_msa_features(
        tokenized, np.random.default_rng(7), max_seqs_batch=128, max_seqs=128,
    )
    templates = process_template_features(tokenized, len(tokenized.tokens))
    return loaded, msa, templates


def _msa_rows(features):
    letters = {const.token_ids[const.prot_letter_to_token[letter]]: letter for letter in "ACG-"}
    return {"".join(letters[int(token)] for token in row) for row in features["msa"]}


def _chain_a_rows(features):
    return {row[:4] for row in _msa_rows(features)}


def _snapshot(folder):
    return {str(path.relative_to(folder)): path.read_bytes()
            for path in folder.rglob("*") if path.is_file()}


def _assert_templates_equal(before, after):
    # Independent mapping expectation guards against two equally empty results.
    np.testing.assert_array_equal(after["template_mask"], [[1, 0, 1, 0, 1, 1, 1]])
    assert after["template_cb"].shape[0] == 1
    for field in before:
        np.testing.assert_allclose(before[field], after[field], atol=1e-5, rtol=1e-5, err_msg=field)


@pytest.mark.parametrize("run_data,write_json", [(False, False), (False, True), (True, False), (True, True)])
def test_same_path_unpaired_replacement_rebuilds_real_msa_and_preserves_templates(
    tmp_path, monkeypatch, alanine_ccd, template_cif, run_data, write_json,
):
    def forbidden_search(*args, **kwargs):
        raise AssertionError("Explicit prepared MSA resources must not trigger search")

    monkeypatch.setattr(pipeline, "run_mmseqs2", forbidden_search)
    source = _source(tmp_path, template_cif)
    original = parse_json(source, alanine_ccd, tmp_path, True, tmp_path / "export-runtime")
    public = tmp_path / "public"
    prepared = main.write_data_json(source, public / "job", original, {}, alanine_ccd, tmp_path)
    schema = json.loads(prepared.read_text())
    resources = [item["protein"]["msa"] for item in schema["sequences"]]
    assert resources[0] == {
        "paired": "msas/job__A_pairedmsa.a3m.zst",
        "unpaired": "msas/job__A_unpairedmsa.a3m.zst",
    }
    runtime = tmp_path / "runtime"
    _, before_msa, before_templates = _process(prepared, alanine_ccd, runtime, public)
    assert "CCAA" in _chain_a_rows(before_msa)
    # Reuse the exact source and processed paths: an existence/cache shortcut
    # would retain CCAA, whereas the current file now supplies GGAA.
    write_zstd_text(prepared.parent / resources[0]["unpaired"], ">query\nAAAA\n>replacement\nGGAA\n")
    prepared.write_text(json.dumps(schema))  # Makes J=true publication observable.
    public_before = _snapshot(public)
    after_loaded, after_msa, after_templates = _process(
        prepared, alanine_ccd, runtime, public, run_data=run_data, write_json=write_json,
    )
    rows = _chain_a_rows(after_msa)
    assert "GGAA" in rows
    assert "CCAA" not in rows
    assert "A-AA" in rows  # Nonempty paired input reaches native processing.
    for features in (before_msa, after_msa):
        assert "A-AAA-A" in _msa_rows(features)
    np.testing.assert_array_equal(after_loaded.msa[0].sequences["taxonomy"], [0, 1, -1])
    _assert_templates_equal(before_templates, after_templates)
    public_after = _snapshot(public)
    if write_json:
        assert json.loads(prepared.read_text()) == schema
        assert public_after["job/job_data.json"] != public_before["job/job_data.json"]
        # Export may rewrite the JSON, but every already-prepared resource is stable.
        assert {key: value for key, value in public_after.items() if key != "job/job_data.json"} == {
            key: value for key, value in public_before.items() if key != "job/job_data.json"
        }
    else:
        assert public_after == public_before


def test_direct_combined_path_reads_replacement_without_search_or_publication(
    tmp_path, monkeypatch, alanine_ccd, template_cif,
):
    monkeypatch.setattr(pipeline, "run_mmseqs2", lambda *a, **k: pytest.fail("unexpected search"))
    source = _source(tmp_path, template_cif)
    public = tmp_path / "public"
    _, before_msa, before_templates = _process(source, alanine_ccd, tmp_path / "runtime", public, run_data=True)
    assert "CCAA" in _chain_a_rows(before_msa)
    (tmp_path / "A_unpaired.a3m").write_text(">query\nAAAA\n>replacement\nGGAA\n")
    untouched = {path: path.read_bytes() for path in [source, template_cif, tmp_path / "A_paired.a3m", tmp_path / "B_paired.a3m"]}
    _, after_msa, after_templates = _process(source, alanine_ccd, tmp_path / "runtime", public, run_data=True)
    assert "GGAA" in _chain_a_rows(after_msa)
    assert "CCAA" not in _chain_a_rows(after_msa)
    for features in (before_msa, after_msa):
        assert "A-AAA-A" in _msa_rows(features)
    _assert_templates_equal(before_templates, after_templates)
    assert all(path.read_bytes() == contents for path, contents in untouched.items())
    assert not public.exists()


@pytest.mark.parametrize("data_only", [False, True])
@pytest.mark.parametrize("write_json", [False, True])
@pytest.mark.parametrize("existing_public", [False, True])
def test_auto_search_publication_obeys_write_control(
    tmp_path, monkeypatch, alanine_ccd, data_only, write_json, existing_public,
):
    """Search must not create or overwrite public resources when J=false."""
    source = tmp_path / "job.json"
    source.write_text(json.dumps({"name": "job", "sequences": [{"protein": {
        "id": "A", "sequence": "AAAA",
    }}]}))
    public = tmp_path / "public"
    if existing_public:
        resources = public / "job" / "msas"
        resources.mkdir(parents=True)
        (public / "job" / "job_data.json").write_text("previous prepared JSON")
        for kind in ("paired", "unpaired"):
            (resources / f"job__A_{kind}msa.a3m.zst").write_bytes(b"previous MSA")
    before = _snapshot(public)
    monkeypatch.setattr(pipeline, "run_mmseqs2", lambda sequences, prefix, **kwargs: [">query\nAAAA\n>hit\nCCAA\n"])
    if data_only:
        from rdkit import Chem

        ccd_path = tmp_path / "ccd.pkl"
        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
        with ccd_path.open("wb") as handle:
            pickle.dump(alanine_ccd, handle)
        main.prepare_msa_inputs(
            data=[source], out_dir=tmp_path / "runtime", ccd_path=ccd_path,
            mol_dir=tmp_path, boltz2=False, use_msa_server=True,
            msa_server_url="https://unused.invalid", msa_pairing_strategy="greedy",
            prepared_output_root=public, write_input_json=write_json,
        )
    else:
        loaded, _, _ = _process(
            source, alanine_ccd, tmp_path / "runtime", public,
            run_data=True, write_json=write_json, consume_features=False,
        )
        assert len(loaded.msa[0].sequences) >= 2
    after = _snapshot(public)
    if not write_json:
        assert after == before, f"J=false changed public files: {sorted(after)}"
    else:
        prepared = json.loads((public / "job" / "job_data.json").read_text())
        assert prepared["sequences"][0]["protein"]["msa"] == {
            "paired": "msas/job__A_pairedmsa.a3m.zst",
            "unpaired": "msas/job__A_unpairedmsa.a3m.zst",
        }
        for kind in ("paired", "unpaired"):
            data = after[f"job/msas/job__A_{kind}msa.a3m.zst"]
            assert data and data != b"previous MSA"


def test_inference_only_missing_msa_fails_without_search_or_publication(tmp_path, monkeypatch, alanine_ccd):
    source = tmp_path / "job.json"
    source.write_text(json.dumps({"name": "job", "sequences": [{"protein": {
        "id": "A", "sequence": "AAAA",
    }}]}))
    monkeypatch.setattr(pipeline, "run_mmseqs2", lambda *a, **k: pytest.fail("D=false searched"))
    public = tmp_path / "public"
    with pytest.raises(RuntimeError, match="Prepared paired MSA not found"):
        _process(source, alanine_ccd, tmp_path / "runtime", public, run_data=False)
    assert not public.exists()
