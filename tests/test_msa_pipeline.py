import json
from pathlib import Path
from types import SimpleNamespace

import click
import pytest
import zstandard as zstd
from click.testing import CliRunner
from rdkit import Chem

from boltz import main as main_module
from boltz.data.msa import pipeline
from boltz.data.msa.pipeline import (
    component_paths,
    materialize_msa_csv,
    read_a3m_sequences,
    search_msa_components,
)
from boltz.data.parse import json as json_parser
from boltz.data.parse.a3m import parse_a3m
from boltz.data.parse.json import materialize_prepared_msas, target_name_from_path
from boltz.data.types import Manifest


@pytest.mark.parametrize(
    ("seed_values", "expected"),
    [
        ("7", [7]),
        ("7, 11,19", [7, 11, 19]),
        ("0,4294967295", [0, 4294967295]),
    ],
)
def test_resolve_prediction_seeds(
    seed_values: str,
    expected: list[int],
) -> None:
    assert main_module.resolve_prediction_seeds(seed_values) == expected


def test_resolve_prediction_seeds_generates_one_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main_module.secrets, "randbits", lambda _bits: 12345)

    assert main_module.resolve_prediction_seeds(None) == [12345]


@pytest.mark.parametrize(
    "seed_values",
    [
        "",
        "1,",
        "one,2",
        "1,1",
        "-1",
        "4294967296",
    ],
)
def test_resolve_prediction_seeds_rejects_invalid_values(
    seed_values: str,
) -> None:
    with pytest.raises(click.ClickException):
        main_module.resolve_prediction_seeds(seed_values)


def test_cli_exposes_only_plural_seeds_option() -> None:
    result = CliRunner().invoke(main_module.cli, ["predict", "--help"])

    assert result.exit_code == 0
    assert "--seeds TEXT" in result.output
    assert "--seed INTEGER" not in result.output


def test_search_saves_separate_paired_and_unpaired_a3m(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_run_mmseqs2(sequences, prefix, *, use_pairing, **kwargs):
        calls.append((sequences, Path(prefix), use_pairing, kwargs))
        if use_pairing:
            return [">q\nAAAA\n>p\nAA-A\n", ">q\nBBBB\n>p\nBB-B\n"]
        return [">q\nAAAA\n>u\nAACC\n", ">q\nBBBB\n>u\nBBCC\n"]

    monkeypatch.setattr(pipeline, "run_mmseqs2", fake_run_mmseqs2)
    data = {"target_0": "AAAA", "target_1": "BBBB"}

    search_msa_components(
        data=data,
        target_id="target",
        msa_dir=tmp_path,
        msa_server_url="https://example.test",
        msa_pairing_strategy="greedy",
    )

    assert [call[2] for call in calls] == [True, False]
    for msa_id in data:
        paired_path, unpaired_path = component_paths(tmp_path, msa_id)
        assert paired_path.is_file()
        assert unpaired_path.is_file()
        assert paired_path.name.endswith("_pairedmsa.a3m.zst")
        assert unpaired_path.name.endswith("_unpairedmsa.a3m.zst")
        assert paired_path.read_bytes().startswith(b"\x28\xb5\x2f\xfd")
        assert unpaired_path.read_bytes().startswith(b"\x28\xb5\x2f\xfd")
        assert not (tmp_path / f"{msa_id}.csv").exists()


def test_collect_auto_msas_uses_first_chain_name_for_each_entity(
    tmp_path: Path,
) -> None:
    prot_id = main_module.const.chain_type_ids["PROTEIN"]
    chains = [
        SimpleNamespace(
            chain_name="A", entity_id=0, mol_type=prot_id, msa_id=0
        ),
        SimpleNamespace(
            chain_name="B", entity_id=0, mol_type=prot_id, msa_id=0
        ),
        SimpleNamespace(
            chain_name="D", entity_id=1, mol_type=prot_id, msa_id=0
        ),
        SimpleNamespace(
            chain_name="E", entity_id=1, mol_type=prot_id, msa_id=0
        ),
    ]
    target = SimpleNamespace(
        record=SimpleNamespace(id="target", chains=chains),
        sequences={0: "AAAA", 1: "BBBB"},
    )

    auto_msas = main_module.collect_auto_msas(target, tmp_path)

    assert auto_msas == {"target__A": "AAAA", "target__D": "BBBB"}
    assert [chain.msa_id for chain in chains] == [
        tmp_path / "target__A.csv",
        tmp_path / "target__A.csv",
        tmp_path / "target__D.csv",
        tmp_path / "target__D.csv",
    ]


def test_monomer_search_writes_empty_paired_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_run_mmseqs2(sequences, prefix, *, use_pairing, **kwargs):
        calls.append((sequences, Path(prefix), use_pairing, kwargs))
        return [">q\nAAAA\n>u\nAACC\n"]

    monkeypatch.setattr(pipeline, "run_mmseqs2", fake_run_mmseqs2)
    search_msa_components(
        data={"target_0": "AAAA"},
        target_id="target",
        msa_dir=tmp_path,
        msa_server_url="https://example.test",
        msa_pairing_strategy="greedy",
    )

    paired_path, unpaired_path = component_paths(tmp_path, "target_0")
    assert read_a3m_sequences(paired_path) == []
    assert read_a3m_sequences(unpaired_path) == ["AAAA", "AACC"]
    assert [call[2] for call in calls] == [False]


@pytest.mark.parametrize("filename", ["compressed.a3m.zst", "compressed.a3m"])
def test_a3m_reader_detects_zstd_from_magic_bytes(
    tmp_path: Path,
    filename: str,
) -> None:
    path = tmp_path / filename
    content = ">query\nAAAA\n>hit\nAA-A\n"
    path.write_bytes(zstd.ZstdCompressor().compress(content.encode()))

    assert read_a3m_sequences(path) == ["AAAA", "AA-A"]
    assert len(parse_a3m(path, taxonomy=None).sequences) == 2


def test_plain_text_with_zst_suffix_is_not_decompressed(tmp_path: Path) -> None:
    path = tmp_path / "plain.a3m.zst"
    path.write_text(">query\nAAAA\n")

    assert read_a3m_sequences(path) == ["AAAA"]


def test_materialize_csv_preserves_paired_row_keys_and_replaces_unpaired(
    tmp_path: Path,
) -> None:
    paired_path = tmp_path / "paired.a3m"
    unpaired_path = tmp_path / "unpaired.a3m"
    csv_path = tmp_path / "msa.csv"
    paired_path.write_text(">q\nAAAA\n>gap\n----\n>p\nAA-A\n")
    unpaired_path.write_text(">q\nAAAA\n>u1\nAACC\n>u2\nAADD\n")

    materialize_msa_csv(
        paired_path=paired_path,
        unpaired_path=unpaired_path,
        csv_path=csv_path,
        query_sequence="AAAA",
    )

    assert csv_path.read_text().splitlines() == [
        "key,sequence",
        "0,AAAA",
        "2,AA-A",
        "-1,AACC",
        "-1,AADD",
    ]


def test_materialize_monomer_keeps_unpaired_query(tmp_path: Path) -> None:
    paired_path = tmp_path / "paired.a3m"
    unpaired_path = tmp_path / "unpaired.a3m"
    csv_path = tmp_path / "msa.csv"
    paired_path.write_text("")
    unpaired_path.write_text(">q\nAAAA\n>u\nAACC\n")

    materialize_msa_csv(
        paired_path=paired_path,
        unpaired_path=unpaired_path,
        csv_path=csv_path,
        query_sequence="AAAA",
    )

    assert csv_path.read_text().splitlines() == [
        "key,sequence",
        "-1,AAAA",
        "-1,AACC",
    ]


def test_materialize_rejects_replacement_with_wrong_query(tmp_path: Path) -> None:
    paired_path = tmp_path / "paired.a3m"
    unpaired_path = tmp_path / "unpaired.a3m"
    paired_path.write_text("")
    unpaired_path.write_text(">q\nBBBB\n")

    with pytest.raises(ValueError, match="does not match the query"):
        materialize_msa_csv(
            paired_path=paired_path,
            unpaired_path=unpaired_path,
            csv_path=tmp_path / "msa.csv",
            query_sequence="AAAA",
        )


def test_data_only_cli_stops_after_preparing_msas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "target.json"
    input_path.write_text(json.dumps({"sequences": []}))
    calls = []
    downloads = []
    monkeypatch.setattr(
        main_module,
        "download_boltz2",
        lambda _cache, *, download_weights: downloads.append(download_weights),
    )
    monkeypatch.setattr(
        main_module,
        "prepare_msa_inputs",
        lambda **kwargs: calls.append(kwargs),
    )

    result = CliRunner().invoke(
        main_module.cli,
        [
            "predict",
            str(input_path),
            "--out_dir",
            str(tmp_path / "out"),
            "--cache",
            str(tmp_path / "cache"),
            "-D",
            "true",
            "-P",
            "false",
            "--use_msa_server",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0]["use_msa_server"] is True
    assert calls[0]["out_dir"] != tmp_path / "out" / "target"
    assert not calls[0]["out_dir"].exists()
    assert downloads == [False]
    assert "No seed provided" not in result.output


def test_data_only_cli_uses_top_level_name_for_job_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input_filename.json"
    input_path.write_text(
        json.dumps({"name": "json_job_name", "sequences": []})
    )
    calls = []
    monkeypatch.setattr(
        main_module,
        "download_boltz2",
        lambda _cache, *, download_weights: None,
    )
    monkeypatch.setattr(
        main_module,
        "prepare_msa_inputs",
        lambda **kwargs: calls.append(kwargs),
    )

    result = CliRunner().invoke(
        main_module.cli,
        [
            "predict",
            str(input_path),
            "--out_dir",
            str(tmp_path / "out"),
            "--cache",
            str(tmp_path / "cache"),
            "-D",
            "true",
            "-P",
            "false",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls[0]["out_dir"] != tmp_path / "out" / "json_job_name"
    assert not calls[0]["out_dir"].exists()
    assert calls[0]["prepared_output_root"] == tmp_path / "out"


@pytest.mark.parametrize("use_slurm_tmp", [False, True])
def test_inference_only_uses_and_cleans_private_processed_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    use_slurm_tmp: bool,
) -> None:
    input_path = tmp_path / "target_data.json"
    input_path.write_text(json.dumps({"sequences": []}))
    output_root = tmp_path / "out"
    captured = {}
    if use_slurm_tmp:
        slurm_tmp = tmp_path / "slurm_tmp"
        slurm_tmp.mkdir()
        monkeypatch.setenv("SLURM_TMPDIR", str(slurm_tmp))
    else:
        monkeypatch.delenv("SLURM_TMPDIR", raising=False)

    monkeypatch.setattr(main_module, "download_boltz2", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main_module, "seed_everything", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main_module, "Trainer", lambda **_kwargs: SimpleNamespace())

    def fake_process_inputs(*, out_dir: Path, **_kwargs) -> Manifest:
        captured["work_dir"] = out_dir
        processed_dir = out_dir / "processed"
        processed_dir.mkdir(parents=True)
        manifest = Manifest([])
        manifest.dump(processed_dir / "manifest.json")
        return manifest

    monkeypatch.setattr(main_module, "process_inputs", fake_process_inputs)
    seed_args = ["--seeds", "7,8"] if use_slurm_tmp else ["--seeds", "7"]

    result = CliRunner().invoke(
        main_module.cli,
        [
            "predict",
            str(input_path),
            "--out_dir",
            str(output_root),
            "--cache",
            str(tmp_path / "cache"),
            *seed_args,
            "-D",
            "false",
            "-P",
            "true",
        ],
    )

    assert result.exit_code == 0, result.output
    work_dir = captured["work_dir"]
    assert work_dir != output_root / "target"
    if use_slurm_tmp:
        assert work_dir.parent == tmp_path / "slurm_tmp"
    assert not work_dir.exists()
    assert not (output_root / "target" / "processed").exists()


def test_prepare_msa_inputs_writes_executable_json_without_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "target.json"
    input_path.write_text(
        json.dumps(
            {
                "version": 1,
                "sequences": [
                    {
                        "protein": {
                            "id": "A",
                            "sequence": "AAAA",
                        }
                    }
                ],
                "templates": [],
                "constraints": [
                    {
                        "contact": {
                            "token1": ["A", 1],
                            "token2": ["A", 2],
                            "max_distance": 8,
                        }
                    }
                ],
                "properties": [{"affinity": {"binder": "A"}}],
            }
        )
    )
    source_bytes = input_path.read_bytes()
    target = SimpleNamespace(record=SimpleNamespace(id="target"))
    monkeypatch.setattr(main_module, "load_canonicals", lambda _path: {})
    monkeypatch.setattr(
        main_module,
        "parse_input_target",
        lambda *_args, **_kwargs: target,
    )
    monkeypatch.setattr(
        main_module,
        "collect_auto_msas",
        lambda *_args, **_kwargs: {"target__A": "AAAA"},
    )

    def fake_search(*, data, msa_dir, **_kwargs):
        for msa_id in data:
            paired_path, unpaired_path = component_paths(msa_dir, msa_id)
            compressor = zstd.ZstdCompressor()
            paired_path.write_bytes(compressor.compress(b""))
            unpaired_path.write_bytes(compressor.compress(b">q\nAAAA\n"))

    monkeypatch.setattr(main_module, "search_msa_components", fake_search)
    out_dir = tmp_path / "out"

    main_module.prepare_msa_inputs(
        data=[input_path],
        out_dir=out_dir,
        prepared_output_root=out_dir,
        ccd_path=tmp_path / "ccd.pkl",
        mol_dir=tmp_path / "mols",
        boltz2=True,
        use_msa_server=True,
        msa_server_url="https://example.test",
        msa_pairing_strategy="greedy",
    )

    data_path = out_dir / "target" / "target_data.json"
    prepared = json.loads(data_path.read_text())
    protein = prepared["sequences"][0]["protein"]
    assert protein["msa"] == {
        "paired": "msas/target__A_pairedmsa.a3m.zst",
        "unpaired": "msas/target__A_unpairedmsa.a3m.zst",
    }
    assert prepared["name"] == "target"
    assert prepared["templates"] == []
    assert prepared["constraints"][0]["contact"]["max_distance"] == 8
    assert prepared["properties"] == [{"affinity": {"binder": "A"}}]
    assert input_path.read_bytes() == source_bytes
    assert not (out_dir / "target" / "prepared_msa_manifest.json").exists()
    assert not (out_dir / "target" / "msa" / "target_A.csv").exists()


def test_prepared_json_materializes_relative_msa_paths(tmp_path: Path) -> None:
    msa_dir = tmp_path / "msa"
    msa_dir.mkdir()
    (msa_dir / "target_0_paired.a3m").write_text("")
    (msa_dir / "target_0_unpaired.a3m").write_text(
        ">query\nAAAA\n>hit\nAA-A\n"
    )
    data_path = tmp_path / "target_data.json"
    schema = {
        "sequences": [
            {
                "protein": {
                    "id": "A",
                    "sequence": "AAAA",
                    "msa": {
                        "paired": "msa/target_0_paired.a3m",
                        "unpaired": "msa/target_0_unpaired.a3m",
                    },
                }
            }
        ]
    }

    materialize_prepared_msas(data_path, schema)

    csv_path = msa_dir / "target__entity_0.csv"
    assert csv_path.read_text().splitlines() == [
        "key,sequence",
        "-1,AAAA",
        "-1,AA-A",
    ]
    assert schema["sequences"][0]["protein"]["msa"] == str(csv_path)
    assert target_name_from_path(data_path) == "target"


def test_prepared_json_materializes_msa_in_private_directory(tmp_path: Path) -> None:
    msa_dir = tmp_path / "msa"
    msa_dir.mkdir()
    paired_path = msa_dir / "target_0_paired.a3m.zst"
    unpaired_path = msa_dir / "target_0_unpaired.a3m.zst"
    compressor = zstd.ZstdCompressor()
    paired_path.write_bytes(compressor.compress(b""))
    unpaired_path.write_bytes(compressor.compress(b">query\nAAAA\n>hit\nAA-A\n"))
    private_dir = tmp_path / "private" / "msa"
    data_path = tmp_path / "target_data.json"
    schema = {
        "sequences": [
            {
                "protein": {
                    "id": "A",
                    "sequence": "AAAA",
                    "msa": {
                        "paired": "msa/target_0_paired.a3m.zst",
                        "unpaired": "msa/target_0_unpaired.a3m.zst",
                    },
                }
            }
        ]
    }

    materialize_prepared_msas(data_path, schema, output_dir=private_dir)

    csv_path = private_dir / "target__entity_0.csv"
    assert csv_path.read_text().splitlines() == [
        "key,sequence",
        "-1,AAAA",
        "-1,AA-A",
    ]
    assert schema["sequences"][0]["protein"]["msa"] == str(csv_path)
    assert not list(msa_dir.glob("*.csv"))


def test_parse_data_json_uses_original_target_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    msa_dir = tmp_path / "msa"
    msa_dir.mkdir()
    (msa_dir / "target_0_paired.a3m").write_text("")
    (msa_dir / "target_0_unpaired.a3m").write_text(">query\nAAAA\n")
    data_path = tmp_path / "target_data.json"
    data_path.write_text(
        json.dumps(
            {
                "version": 1,
                "sequences": [
                    {
                        "protein": {
                            "id": "A",
                            "sequence": "AAAA",
                            "msa": {
                                "paired": "msa/target_0_paired.a3m",
                                "unpaired": "msa/target_0_unpaired.a3m",
                            },
                        }
                    }
                ],
            }
        )
    )
    parsed = {}

    def fake_parse(name, schema, *_args, **_kwargs):
        parsed["name"] = name
        parsed["schema"] = schema
        return SimpleNamespace()

    monkeypatch.setattr(json_parser, "parse_boltz_schema", fake_parse)

    json_parser.parse_json(data_path, {}, tmp_path, boltz2=True)

    assert parsed["name"] == "target"
    assert parsed["schema"]["sequences"][0]["protein"]["msa"] == str(
        msa_dir / "target__entity_0.csv"
    )


def test_native_parse_uses_absolute_private_csv_with_relative_output_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "nested" / "inputs"
    msa_dir = input_dir / "msa"
    msa_dir.mkdir(parents=True)
    paired = msa_dir / "job_A_paired.a3m"
    unpaired = msa_dir / "job_A_unpaired.a3m"
    paired.write_text("")
    unpaired.write_text(">query\nAAAA\n")
    source = input_dir / "renamed.json"
    source.write_text(
        json.dumps(
            {
                "name": "native_job",
                "sequences": [
                    {
                        "protein": {
                            "id": "A",
                            "sequence": "AAAA",
                            "msa": {
                                "paired": "msa/job_A_paired.a3m",
                                "unpaired": "msa/job_A_unpaired.a3m",
                            },
                        }
                    }
                ],
            }
        )
    )
    monkeypatch.chdir(tmp_path)
    relative_runtime_dir = Path("relative-private") / "msa"
    alanine = Chem.RWMol()
    atom_ids = {}
    for atom_name, atomic_number in (
        ("N", 7),
        ("CA", 6),
        ("C", 6),
        ("O", 8),
        ("CB", 6),
    ):
        atom = Chem.Atom(atomic_number)
        atom.SetProp("name", atom_name)
        atom_ids[atom_name] = alanine.AddAtom(atom)
    alanine.AddBond(atom_ids["N"], atom_ids["CA"], Chem.BondType.SINGLE)
    alanine.AddBond(atom_ids["CA"], atom_ids["C"], Chem.BondType.SINGLE)
    alanine.AddBond(atom_ids["C"], atom_ids["O"], Chem.BondType.DOUBLE)
    alanine.AddBond(atom_ids["CA"], atom_ids["CB"], Chem.BondType.SINGLE)
    alanine = alanine.GetMol()
    conformer = Chem.Conformer(alanine.GetNumAtoms())
    for atom_idx in range(alanine.GetNumAtoms()):
        conformer.SetAtomPosition(atom_idx, (float(atom_idx), 0.0, 0.0))
    alanine.AddConformer(conformer)

    target = json_parser.parse_json(
        source,
        {"ALA": alanine},
        tmp_path / "mols",
        boltz2=True,
        msa_materialization_dir=relative_runtime_dir,
    )

    csv_path = (tmp_path / relative_runtime_dir / "native_job__entity_0.csv").resolve()
    assert target.record.id == "native_job"
    assert Path(target.record.chains[0].msa_id) == csv_path
    assert csv_path.is_file()
    assert not list(msa_dir.glob("*.csv"))


def test_parse_json_uses_top_level_name_instead_of_filename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_path = tmp_path / "renamed.json"
    data_path.write_text(json.dumps({"name": "stable_name", "sequences": []}))
    parsed = {}

    def fake_parse(name, schema, *_args, **_kwargs):
        parsed["name"] = name
        parsed["schema"] = schema
        return SimpleNamespace()

    monkeypatch.setattr(json_parser, "parse_boltz_schema", fake_parse)

    json_parser.parse_json(data_path, {}, tmp_path, boltz2=True)

    assert parsed["name"] == "stable_name"
    assert target_name_from_path(data_path) == "stable_name"


@pytest.mark.parametrize("name", ["", "../escape", "nested/job", "nested\\job"])
def test_json_rejects_unsafe_top_level_name(tmp_path: Path, name: str) -> None:
    data_path = tmp_path / "input.json"
    data_path.write_text(json.dumps({"name": name, "sequences": []}))

    with pytest.raises(ValueError, match="name"):
        target_name_from_path(data_path)


def test_single_yaml_input_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "job.yaml"
    source.write_text("version: 1\nsequences: []\n")

    with pytest.raises((ValueError, RuntimeError), match="JSON|json"):
        main_module.check_inputs(source)


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("job.fasta", ">A\nAAAA\n"),
        ("job.json", "version: 1\nsequences: []\n"),
    ],
)
def test_single_input_rejects_fasta_and_yaml_disguised_as_json(
    tmp_path: Path,
    filename: str,
    content: str,
) -> None:
    source = tmp_path / filename
    source.write_text(content)

    with pytest.raises((ValueError, RuntimeError), match="JSON|json"):
        main_module.check_inputs(source)


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("job.yaml", "version: 1\nsequences: []\n"),
        ("job.fasta", ">A\nAAAA\n"),
        ("job.json", "version: 1\nsequences: []\n"),
    ],
)
def test_directory_input_rejects_non_json_and_invalid_json(
    tmp_path: Path,
    filename: str,
    content: str,
) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    (input_dir / filename).write_text(content)

    with pytest.raises((ValueError, RuntimeError), match="JSON|json"):
        main_module.check_inputs(input_dir)


def test_json_input_requires_top_level_object(tmp_path: Path) -> None:
    source = tmp_path / "job.json"
    source.write_text("[]")

    with pytest.raises(ValueError, match="object"):
        json_parser.load_input_schema(source)


def test_directory_rejects_duplicate_target_names(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    for filename in ("first.json", "second.json"):
        (input_dir / filename).write_text(
            json.dumps({"name": "same_job", "sequences": []})
        )

    with pytest.raises(ValueError, match="Duplicate|duplicate"):
        main_module.check_inputs(input_dir)


def test_invalid_input_is_rejected_before_download_or_output_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "job.yaml"
    source.write_text("sequences: []\n")
    called = []
    monkeypatch.setattr(
        main_module,
        "download_boltz2",
        lambda *_args, **_kwargs: called.append(True),
    )
    output_root = tmp_path / "out"
    cache = tmp_path / "cache"

    result = CliRunner().invoke(
        main_module.cli,
        [
            "predict",
            str(source),
            "--out_dir",
            str(output_root),
            "--cache",
            str(cache),
            "-D",
            "true",
            "-P",
            "false",
        ],
    )

    assert result.exit_code != 0
    assert isinstance(result.exception, RuntimeError)
    assert "JSON" in str(result.exception)
    assert called == []
    assert not output_root.exists()
    assert not cache.exists()


def test_data_only_directory_writes_one_bundle_per_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = []
    for name in ("alpha", "beta"):
        source = tmp_path / f"{name}.json"
        source.write_text(json.dumps({"name": name, "sequences": []}))
        sources.append(source)

    monkeypatch.setattr(main_module, "load_canonicals", lambda _path: {})
    monkeypatch.setattr(
        main_module,
        "parse_input_target",
        lambda path, *_args, **_kwargs: SimpleNamespace(
            record=SimpleNamespace(id=json.loads(path.read_text())["name"])
        ),
    )
    monkeypatch.setattr(main_module, "collect_auto_msas", lambda *_args: {})
    output_root = tmp_path / "out"

    main_module.prepare_msa_inputs(
        data=sources,
        out_dir=tmp_path / "private",
        prepared_output_root=output_root,
        ccd_path=tmp_path / "ccd.pkl",
        mol_dir=tmp_path / "mols",
        boltz2=True,
        use_msa_server=False,
        msa_server_url="https://example.test",
        msa_pairing_strategy="greedy",
    )

    assert json.loads(
        (output_root / "alpha" / "alpha_data.json").read_text()
    )["name"] == "alpha"
    assert json.loads(
        (output_root / "beta" / "beta_data.json").read_text()
    )["name"] == "beta"
    assert not (output_root / tmp_path.name).exists()


def test_data_only_existing_prepared_msa_uses_clean_private_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "source"
    source_msa_dir = source_dir / "msa"
    source_msa_dir.mkdir(parents=True)
    paired = source_msa_dir / "existing_paired.a3m"
    unpaired = source_msa_dir / "existing_unpaired.a3m"
    paired.write_text("")
    unpaired.write_text(">query\nAAAA\n")
    source = source_dir / "prepared.json"
    source.write_text(
        json.dumps(
            {
                "name": "stable_job",
                "sequences": [
                    {
                        "protein": {
                            "id": "A",
                            "sequence": "AAAA",
                            "msa": {
                                "paired": "msa/existing_paired.a3m",
                                "unpaired": "msa/existing_unpaired.a3m",
                            },
                        }
                    }
                ],
            }
        )
    )
    target = SimpleNamespace(record=SimpleNamespace(id="stable_job"))
    monkeypatch.setattr(main_module, "load_canonicals", lambda _path: {})
    monkeypatch.setattr(
        json_parser,
        "parse_boltz_schema",
        lambda *_args, **_kwargs: target,
    )
    monkeypatch.setattr(main_module, "collect_auto_msas", lambda *_args: {})
    output_root = tmp_path / "out"

    main_module.prepare_msa_inputs(
        data=[source],
        out_dir=tmp_path / "private",
        prepared_output_root=output_root,
        ccd_path=tmp_path / "ccd.pkl",
        mol_dir=tmp_path / "mols",
        boltz2=True,
        use_msa_server=False,
        msa_server_url="https://example.test",
        msa_pairing_strategy="greedy",
    )

    prepared_path = output_root / "stable_job" / "stable_job_data.json"
    prepared = json.loads(prepared_path.read_text())
    msa = prepared["sequences"][0]["protein"]["msa"]
    assert msa["paired"] == "msas/stable_job__A_pairedmsa.a3m.zst"
    assert read_a3m_sequences(prepared_path.parent / msa["paired"]) == []
    assert read_a3m_sequences(prepared_path.parent / msa["unpaired"]) == ["AAAA"]
    assert not list(source_msa_dir.glob("*.csv"))
    assert not list(output_root.rglob("*.csv"))
    assert not (tmp_path / "private").exists()


def test_combined_processing_writes_one_bundle_per_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Dumpable:
        def dump(self, path: Path) -> None:
            path.write_bytes(b"test")

    sources = []
    targets = {}
    for name in ("alpha", "beta"):
        source = tmp_path / f"{name}.json"
        source.write_text(json.dumps({"name": name, "sequences": []}))
        sources.append(source)
        record = SimpleNamespace(id=name, chains=[], dump=Dumpable().dump)
        targets[source] = SimpleNamespace(
            record=record,
            templates={},
            residue_constraints=Dumpable(),
            extra_mols={},
            structure=Dumpable(),
        )

    monkeypatch.setattr(
        main_module,
        "parse_input_target",
        lambda path, *_args, **_kwargs: targets[path],
    )
    monkeypatch.setattr(main_module, "collect_auto_msas", lambda *_args: {})
    private_root = tmp_path / "private"
    directories = {
        name: private_root / name
        for name in (
            "msa",
            "processed_msa",
            "processed_constraints",
            "processed_templates",
            "processed_mols",
            "structures",
            "records",
        )
    }
    for directory in directories.values():
        directory.mkdir(parents=True)
    output_root = tmp_path / "out"

    for source in sources:
        main_module.process_input(
            path=source,
            ccd={},
            msa_dir=directories["msa"],
            mol_dir=tmp_path / "mols",
            boltz2=True,
            run_data_pipeline=True,
            use_msa_server=False,
            msa_server_url="https://example.test",
            msa_pairing_strategy="greedy",
            msa_server_username=None,
            msa_server_password=None,
            api_key_header=None,
            api_key_value=None,
            max_msa_seqs=8192,
            processed_msa_dir=directories["processed_msa"],
            processed_constraints_dir=directories["processed_constraints"],
            processed_templates_dir=directories["processed_templates"],
            processed_mols_dir=directories["processed_mols"],
            structure_dir=directories["structures"],
            records_dir=directories["records"],
            prepared_output_root=output_root,
        )

    for name in ("alpha", "beta"):
        prepared = output_root / name / f"{name}_data.json"
        assert json.loads(prepared.read_text())["name"] == name
    assert not list(private_root.rglob("*_data.json"))


def test_rebased_legacy_json_resolves_resources_from_its_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resources = tmp_path / "resources"
    resources.mkdir()
    msa_path = resources / "custom.a3m"
    cif_path = resources / "template.cif"
    pdb_path = resources / "template.pdb"
    for path in (msa_path, cif_path, pdb_path):
        path.write_text("resource")
    source = tmp_path / "source" / "input.json"
    source.parent.mkdir()
    empty_named_template = source.parent / "empty"
    empty_named_template.write_text("resource")
    source.write_text(
        json.dumps(
            {
                "name": "stable_job",
                "sequences": [
                    {
                        "protein": {
                            "id": "A",
                            "sequence": "AAAA",
                            "msa": "../resources/custom.a3m",
                        }
                    },
                    {
                        "protein": {
                            "id": "B",
                            "sequence": "BBBB",
                            "msa": "empty",
                        }
                    },
                    {"ligand": {"id": "C", "smiles": "C/C=C\\C"}},
                    {"ligand": {"id": "D", "ccd": "ATP"}},
                ],
                "templates": [
                    {"cif": "../resources/template.cif", "chain_id": "A"},
                    {"pdb": "../resources/template.pdb", "chain_id": "B"},
                    {"cif": "empty", "chain_id": "A"},
                ],
                "constraints": [
                    {"bond": {"atom1": ["A", 1, "CA"], "atom2": ["C", 1, "C1"]}}
                ],
            }
        )
    )
    prepared_dir = tmp_path / "out" / "stable_job"
    # Exercise path rebasing independently of template export. Real grouped
    # CIF export/readback is covered in test_prepared_templates.py.
    prepared_dir.mkdir(parents=True)
    prepared = prepared_dir / "stable_job_data.json"
    schema = json.loads(source.read_text())
    json_parser.rebase_schema_resource_paths(schema, source, prepared_dir)
    prepared.write_text(json.dumps(schema))
    renamed = prepared.with_name("renamed.json")
    prepared.rename(renamed)
    parsed = {}

    def fake_parse(name, schema, *_args, **_kwargs):
        parsed["name"] = name
        parsed["schema"] = schema
        return SimpleNamespace()

    monkeypatch.setattr(json_parser, "parse_boltz_schema", fake_parse)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    json_parser.parse_json(renamed, {}, tmp_path / "mols", boltz2=True)

    schema = parsed["schema"]
    assert parsed["name"] == "stable_job"
    assert Path(schema["sequences"][0]["protein"]["msa"]) == msa_path
    assert schema["sequences"][1]["protein"]["msa"] == "empty"
    assert schema["sequences"][2]["ligand"]["smiles"] == "C/C=C\\C"
    assert schema["sequences"][3]["ligand"]["ccd"] == "ATP"
    assert Path(schema["templates"][0]["cif"]) == cif_path
    assert Path(schema["templates"][1]["pdb"]) == pdb_path
    assert Path(schema["templates"][2]["cif"]) == empty_named_template
    assert schema["constraints"][0]["bond"]["atom1"] == ["A", 1, "CA"]


def test_cli_rejects_disabling_both_stages(tmp_path: Path) -> None:
    input_path = tmp_path / "target.json"
    input_path.write_text(json.dumps({"sequences": []}))

    result = CliRunner().invoke(
        main_module.cli,
        ["predict", str(input_path), "-D", "false", "-P", "false"],
    )

    assert result.exit_code == 2
    assert "At least one" in result.output
