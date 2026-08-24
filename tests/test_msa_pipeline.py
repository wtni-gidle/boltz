from pathlib import Path
from types import SimpleNamespace

import click
import pytest
import yaml
import zstandard as zstd
from click.testing import CliRunner

from boltz import main as main_module
from boltz.data.msa import pipeline
from boltz.data.msa.pipeline import (
    component_paths,
    materialize_msa_csv,
    read_a3m_sequences,
    search_msa_components,
)
from boltz.data.parse import yaml as yaml_parser
from boltz.data.parse.a3m import parse_a3m
from boltz.data.parse.yaml import materialize_prepared_msas, target_name_from_path
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
        assert paired_path.name.endswith("_paired.a3m.zst")
        assert unpaired_path.name.endswith("_unpaired.a3m.zst")
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

    assert auto_msas == {"target_A": "AAAA", "target_D": "BBBB"}
    assert [chain.msa_id for chain in chains] == [
        tmp_path / "target_A.csv",
        tmp_path / "target_A.csv",
        tmp_path / "target_D.csv",
        tmp_path / "target_D.csv",
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
    input_path = tmp_path / "target.yaml"
    input_path.write_text("sequences: []\n")
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
    assert calls[0]["out_dir"] == tmp_path / "out" / "target"
    assert downloads == [False]
    assert "No seed provided" not in result.output


def test_data_only_cli_uses_top_level_name_for_job_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input_filename.yaml"
    input_path.write_text("name: yaml_job_name\nsequences: []\n")
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
    assert calls[0]["out_dir"] == tmp_path / "out" / "yaml_job_name"


@pytest.mark.parametrize("use_slurm_tmp", [False, True])
def test_inference_only_uses_and_cleans_private_processed_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    use_slurm_tmp: bool,
) -> None:
    input_path = tmp_path / "target_data.yaml"
    input_path.write_text("sequences: []\n")
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


def test_prepare_msa_inputs_writes_executable_yaml_without_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "target.yaml"
    input_path.write_text(
        """version: 1
sequences:
  - protein:
      id: A
      sequence: AAAA
templates:
  - cif: template.cif
"""
    )
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
        lambda *_args, **_kwargs: {"target_A": "AAAA"},
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
        ccd_path=tmp_path / "ccd.pkl",
        mol_dir=tmp_path / "mols",
        boltz2=True,
        use_msa_server=True,
        msa_server_url="https://example.test",
        msa_pairing_strategy="greedy",
    )

    prepared = yaml.safe_load((out_dir / "target_data.yaml").read_text())
    protein = prepared["sequences"][0]["protein"]
    assert protein["msa"] == {
        "paired": "msa/target_A_paired.a3m.zst",
        "unpaired": "msa/target_A_unpaired.a3m.zst",
    }
    assert prepared["name"] == "target"
    assert prepared["templates"] == [{"cif": "template.cif"}]
    assert not (out_dir / "prepared_msa_manifest.json").exists()
    assert not (out_dir / "msa" / "target_A.csv").exists()


def test_prepared_yaml_materializes_relative_msa_paths(tmp_path: Path) -> None:
    msa_dir = tmp_path / "msa"
    msa_dir.mkdir()
    (msa_dir / "target_0_paired.a3m").write_text("")
    (msa_dir / "target_0_unpaired.a3m").write_text(
        ">query\nAAAA\n>hit\nAA-A\n"
    )
    data_path = tmp_path / "target_data.yaml"
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

    csv_path = msa_dir / "target_0.csv"
    assert csv_path.read_text().splitlines() == [
        "key,sequence",
        "-1,AAAA",
        "-1,AA-A",
    ]
    assert schema["sequences"][0]["protein"]["msa"] == str(csv_path)
    assert target_name_from_path(data_path) == "target"


def test_prepared_yaml_materializes_msa_in_private_directory(tmp_path: Path) -> None:
    msa_dir = tmp_path / "msa"
    msa_dir.mkdir()
    paired_path = msa_dir / "target_0_paired.a3m.zst"
    unpaired_path = msa_dir / "target_0_unpaired.a3m.zst"
    compressor = zstd.ZstdCompressor()
    paired_path.write_bytes(compressor.compress(b""))
    unpaired_path.write_bytes(compressor.compress(b">query\nAAAA\n>hit\nAA-A\n"))
    private_dir = tmp_path / "private" / "msa"
    data_path = tmp_path / "target_data.yaml"
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

    csv_path = private_dir / "target_0_paired.csv"
    assert csv_path.read_text().splitlines() == [
        "key,sequence",
        "-1,AAAA",
        "-1,AA-A",
    ]
    assert schema["sequences"][0]["protein"]["msa"] == str(csv_path)
    assert not list(msa_dir.glob("*.csv"))


def test_parse_data_yaml_uses_original_target_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    msa_dir = tmp_path / "msa"
    msa_dir.mkdir()
    (msa_dir / "target_0_paired.a3m").write_text("")
    (msa_dir / "target_0_unpaired.a3m").write_text(">query\nAAAA\n")
    data_path = tmp_path / "target_data.yaml"
    data_path.write_text(
        """version: 1
sequences:
  - protein:
      id: A
      sequence: AAAA
      msa:
        paired: msa/target_0_paired.a3m
        unpaired: msa/target_0_unpaired.a3m
"""
    )
    parsed = {}

    def fake_parse(name, schema, *_args, **_kwargs):
        parsed["name"] = name
        parsed["schema"] = schema
        return SimpleNamespace()

    monkeypatch.setattr(yaml_parser, "parse_boltz_schema", fake_parse)

    yaml_parser.parse_yaml(data_path, {}, tmp_path, boltz2=True)

    assert parsed["name"] == "target"
    assert parsed["schema"]["sequences"][0]["protein"]["msa"] == str(
        msa_dir / "target_0.csv"
    )


def test_parse_yaml_uses_top_level_name_instead_of_filename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_path = tmp_path / "renamed.yaml"
    data_path.write_text("name: stable_name\nsequences: []\n")
    parsed = {}

    def fake_parse(name, schema, *_args, **_kwargs):
        parsed["name"] = name
        parsed["schema"] = schema
        return SimpleNamespace()

    monkeypatch.setattr(yaml_parser, "parse_boltz_schema", fake_parse)

    yaml_parser.parse_yaml(data_path, {}, tmp_path, boltz2=True)

    assert parsed["name"] == "stable_name"
    assert target_name_from_path(data_path) == "stable_name"


@pytest.mark.parametrize("name", ["", "../escape", "nested/job", "nested\\job"])
def test_yaml_rejects_unsafe_top_level_name(tmp_path: Path, name: str) -> None:
    data_path = tmp_path / "input.yaml"
    data_path.write_text(yaml.safe_dump({"name": name, "sequences": []}))

    with pytest.raises(ValueError, match="name"):
        target_name_from_path(data_path)


def test_cli_rejects_disabling_both_stages(tmp_path: Path) -> None:
    input_path = tmp_path / "target.yaml"
    input_path.write_text("sequences: []\n")

    result = CliRunner().invoke(
        main_module.cli,
        ["predict", str(input_path), "-D", "false", "-P", "false"],
    )

    assert result.exit_code == 2
    assert "At least one" in result.output
