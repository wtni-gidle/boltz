"""Check the actual template feature consumer, not just JSON round-tripping."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from boltz.data.feature.featurizerv2 import process_template_features
from boltz.data.types import Chain, TemplateInfo, TokenV2


@pytest.fixture
def alanine_ccd():
    from rdkit import Chem
    molecule = Chem.RWMol()
    ids = {}
    for name, number in [("N", 7), ("CA", 6), ("C", 6), ("O", 8), ("CB", 6)]:
        atom = Chem.Atom(number)
        atom.SetProp("name", name)
        ids[name] = molecule.AddAtom(atom)
    for left, right in [("N", "CA"), ("CA", "C"), ("CA", "CB")]:
        molecule.AddBond(ids[left], ids[right], Chem.BondType.SINGLE)
    molecule.AddBond(ids["C"], ids["O"], Chem.BondType.DOUBLE)
    molecule = molecule.GetMol()
    conformer = Chem.Conformer(5)
    for index in range(5):
        conformer.SetAtomPosition(index, (float(index), float(index % 2), float(index % 3)))
    molecule.AddConformer(conformer)
    return {"ALA": molecule}


@pytest.fixture
def template_cif(tmp_path, alanine_ccd):
    from boltz.data.parse.schema import parse_boltz_schema
    from boltz.data.write.mmcif import to_mmcif
    target = parse_boltz_schema("template", {
        "sequences": [{"protein": {"id": ["X", "Y"], "sequence": "AAAA", "msa": "empty"}}]
    }, alanine_ccd, tmp_path, True)
    atoms = target.structure.atoms
    for index in range(len(atoms)):
        atoms["coords"][index] = (index * 0.3, index % 3, index % 2)
    path = tmp_path / "complex.cif"
    path.write_text(to_mmcif(target.structure, boltz2=True))
    return path


@pytest.mark.parametrize("explicit", [False, True])
def test_group_json_is_consumed_without_rewriting_input(tmp_path, alanine_ccd, template_cif, explicit):
    import json
    from boltz.data.parse.json import parse_json
    mapping = {"queryChain": "A", "mmcifPath": template_cif.name, "templateChain": "X"}
    if explicit:
        mapping.update(queryIndices=[0, 2], templateIndices=[1, 3])
    schema = {
        "sequences": [{"protein": {"id": "A", "sequence": "AAA", "msa": "empty"}}],
        "templates": [{"groupId": "complex", "chains": [mapping]}],
    }
    source = tmp_path / "job.json"
    source.write_text(json.dumps(schema))
    before = {path: path.read_bytes() for path in [source, template_cif]}
    target = parse_json(source, alanine_ccd, tmp_path, True, tmp_path / "private")
    assert target.record.templates
    record = target.record.templates[0]
    assert record.name == "complex"
    assert record.query_chain == "A"
    assert record.template_chain == "X"
    if explicit:
        assert record.query_indices == [0, 2]
        assert record.template_indices == [1, 3]
    assert all(path.read_bytes() == content for path, content in before.items())
    assert not list(tmp_path.rglob("*_data.json"))


@pytest.mark.parametrize("q,t", [([0], [9]), ([0, 1], [0]), ([True], [0]), ([0, 0], [0, 1]), ([-1], [0])])
def test_invalid_explicit_mapping_is_not_silently_realigned(tmp_path, alanine_ccd, template_cif, q, t):
    import json
    from boltz.data.parse.json import parse_json
    source = tmp_path / "job.json"
    source.write_text(json.dumps({
        "sequences": [{"protein": {"id": "A", "sequence": "AAA", "msa": "empty"}}],
        "templates": [{"groupId": "g", "chains": [{
            "queryChain": "A", "mmcifPath": template_cif.name, "templateChain": "X",
            "queryIndices": q, "templateIndices": t,
        }]}],
    }))
    with pytest.raises(ValueError, match="indices|Indices|mapping"):
        parse_json(source, alanine_ccd, tmp_path, True, tmp_path / "private")


def test_exported_template_is_single_chain_and_preserves_explicit_mapping(tmp_path, alanine_ccd, template_cif):
    import json
    import gemmi
    from boltz import main
    from boltz.data.parse.json import parse_json
    from boltz.data.parse.compression import open_maybe_compressed_text
    source = tmp_path / "job.json"
    source.write_text(json.dumps({
        "sequences": [{"protein": {"id": "A", "sequence": "AAA", "msa": "empty"}}],
        "templates": [{"groupId": "g", "chains": [{
            "queryChain": "A", "mmcifPath": template_cif.name, "templateChain": "X",
            "queryIndices": [0, 2], "templateIndices": [1, 3],
        }]}],
    }))
    target = parse_json(source, alanine_ccd, tmp_path, True, tmp_path / "private")
    destination = main.write_data_json(source, tmp_path / "out", target, {}, alanine_ccd, tmp_path)
    prepared = json.loads(destination.read_text())
    mapping = prepared["templates"][0]["chains"][0]
    assert mapping["queryIndices"] == [0, 2]
    assert mapping["templateIndices"] == [1, 3]
    exported = destination.parent / mapping["mmcifPath"]
    assert exported.name == "job__A_template_0.cif.zst"
    with open_maybe_compressed_text(exported) as handle:
        structure = gemmi.make_structure_from_block(gemmi.cif.read_string(handle.read()).sole_block())
    assert len(structure[0]) == 1
    assert list(structure.entities[0].full_sequence) == ["ALA"] * 4
    rebuilt = parse_json(destination, alanine_ccd, tmp_path, True, tmp_path / "again")
    assert rebuilt.record.templates[0].template_indices == [1, 3]


def test_existing_msas_are_copied_into_prepared_bundle(tmp_path, alanine_ccd):
    import json
    from boltz import main
    from boltz.data.parse.json import parse_json
    from boltz.data.msa.pipeline import read_a3m_sequences
    (tmp_path / "paired.a3m").write_text("")
    (tmp_path / "unpaired.a3m").write_text(">query\nAAA\n>hit\nAA-\n")
    schema = {"sequences": [{"protein": {"id": "A", "sequence": "AAA", "msa": {
        "paired": "paired.a3m", "unpaired": "unpaired.a3m",
    }}}]}
    source = tmp_path / "job.json"
    source.write_text(json.dumps(schema))
    target = parse_json(source, alanine_ccd, tmp_path, True, tmp_path / "private")
    destination = main.write_data_json(source, tmp_path / "out", target, {})
    msa = json.loads(destination.read_text())["sequences"][0]["protein"]["msa"]
    assert msa == {
        "paired": "msas/job__A_pairedmsa.a3m.zst",
        "unpaired": "msas/job__A_unpairedmsa.a3m.zst",
    }
    assert read_a3m_sequences(destination.parent / msa["unpaired"]) == ["AAA", "AA-"]


def test_native_multichain_template_roundtrip_with_missing_residue(tmp_path, alanine_ccd, template_cif):
    import json
    import gemmi
    from boltz import main
    from boltz.data.parse.json import parse_json
    from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
    from boltz.data.types import Input

    structure = gemmi.read_structure(str(template_cif))
    del structure[0]["X"][1]
    template_cif.write_text(structure.make_mmcif_document().as_string())
    source = tmp_path / "job.json"
    source.write_text(json.dumps({
        "sequences": [{"protein": {"id": ["A", "B"], "sequence": "AAA", "msa": "empty"}}],
        "templates": [{"cif": template_cif.name, "chain_id": ["A", "B"], "template_id": ["X", "Y"]}],
    }))
    original = parse_json(source, alanine_ccd, tmp_path, True, tmp_path / "private")
    destination = main.write_data_json(source, tmp_path / "out", original, {}, alanine_ccd, tmp_path)
    # Renaming a prepared JSON must not change its task or resource resolution.
    renamed = destination.with_name("renamed.json")
    destination.rename(renamed)
    rebuilt = parse_json(renamed, alanine_ccd, tmp_path, True, tmp_path / "again")
    assert rebuilt.record.id == "job"
    features = []
    for target in [original, rebuilt]:
        tokens = Boltz2Tokenizer().tokenize(Input(
            structure=target.structure, msa={}, record=target.record,
            residue_constraints=target.residue_constraints,
            templates=target.templates, extra_mols=target.extra_mols,
        ))
        features.append(process_template_features(tokens, len(tokens.tokens)))
    assert features[0]["template_cb"].shape[0] == 1
    for field in features[0]:
        np.testing.assert_allclose(features[0][field], features[1][field], atol=1e-5, rtol=1e-5, err_msg=field)


def test_failed_export_preserves_existing_bundle(tmp_path, monkeypatch, alanine_ccd, template_cif):
    import json
    from boltz import main
    from boltz.data.parse import prepared_templates
    from boltz.data.parse.json import parse_json
    source = tmp_path / "job.json"
    source.write_text(json.dumps({
        "sequences": [{"protein": {"id": "A", "sequence": "AAA", "msa": "empty"}}],
        "templates": [{"cif": template_cif.name, "chain_id": "A", "template_id": "X"}],
    }))
    target = parse_json(source, alanine_ccd, tmp_path, True, tmp_path / "private")
    output = tmp_path / "out"
    old_template = output / "msas" / "job__A_template_0.cif.zst"
    old_template.parent.mkdir(parents=True)
    old_template.write_bytes(b"previous template must survive")
    old_json = output / "job_data.json"
    old_json.write_text('{"previous": true}')
    def fail_read(*args, **kwargs):
        raise ValueError("roundtrip fixture failure")
    monkeypatch.setattr(prepared_templates, "_read_cif", fail_read)
    with pytest.raises(ValueError, match="roundtrip fixture"):
        main.write_data_json(source, output, target, {}, alanine_ccd, tmp_path)
    assert old_template.read_bytes() == b"previous template must survive"
    assert old_json.read_text() == '{"previous": true}'


def _chains(names):
    result = np.zeros(len(names), dtype=Chain)
    result["name"] = names
    result["asym_id"] = np.arange(len(names))
    return result


def _tokens(size, *, x_offset=0):
    result = np.zeros(size, dtype=TokenV2)
    result["token_idx"] = np.arange(size)
    result["res_idx"] = np.arange(size)
    result["center_coords"][:, 0] = np.arange(size) + x_offset
    result["disto_coords"][:, 0] = np.arange(size) + x_offset
    result["disto_mask"] = True
    result["frame_mask"] = True
    result["frame_rot"] = np.eye(3).reshape(9)
    return result


def test_explicit_mapping_reaches_feature_consumer_without_filling_gaps():
    info = TemplateInfo(
        name="group", query_chain="A", query_st=0, query_en=3,
        template_chain="X", template_st=1, template_en=4,
        query_indices=[0, 2], template_indices=[1, 4],
        structure_name="single_X",
    )
    data = SimpleNamespace(
        tokens=_tokens(3), structure=SimpleNamespace(chains=_chains(["A"])),
        record=SimpleNamespace(templates=[info]),
        templates={"single_X": SimpleNamespace(chains=_chains(["X"]))},
        template_tokens={"single_X": _tokens(5, x_offset=10)},
    )
    features = process_template_features(data, 3)
    np.testing.assert_array_equal(features["template_mask"], [[1, 0, 1]])
    np.testing.assert_array_equal(features["template_cb"][0, :, 0], [11, 0, 14])


def test_same_group_across_files_preserves_cross_chain_visibility_and_coordinates():
    infos = [TemplateInfo(
        name="group", query_chain=query, query_st=0, query_en=1,
        template_chain=template, template_st=0, template_en=1,
        query_indices=[0], template_indices=[0], structure_name=template,
    ) for query, template in [("A", "X"), ("B", "Y")]]
    query = _tokens(2)
    query["asym_id"] = [0, 1]
    query["res_idx"] = [0, 0]
    data = SimpleNamespace(
        tokens=query, structure=SimpleNamespace(chains=_chains(["A", "B"])),
        record=SimpleNamespace(templates=infos),
        templates={name: SimpleNamespace(chains=_chains([name])) for name in ["X", "Y"]},
        template_tokens={"X": _tokens(1), "Y": _tokens(1, x_offset=9)},
    )
    features = process_template_features(data, 2)
    assert features["template_cb"].shape[0] == 1
    np.testing.assert_array_equal(features["visibility_ids"], [[0, 0]])
    np.testing.assert_array_equal(features["template_cb"][0, :, 0], [0, 9])


def test_native_offset_still_uses_overlap_outside_local_alignment_end():
    info = TemplateInfo(
        name="X", query_chain="A", query_st=0, query_en=1,
        template_chain="X", template_st=1, template_en=2,
    )
    data = SimpleNamespace(
        tokens=_tokens(3), structure=SimpleNamespace(chains=_chains(["A"])),
        record=SimpleNamespace(templates=[info]),
        templates={"X": SimpleNamespace(chains=_chains(["X"]))},
        template_tokens={"X": _tokens(5, x_offset=10)},
    )
    features = process_template_features(data, 3)
    np.testing.assert_array_equal(features["template_mask"], [[1, 1, 1]])
    np.testing.assert_array_equal(features["template_cb"][0, :, 0], [11, 12, 13])
