from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from click.testing import CliRunner

from boltz import main as main_module
from boltz.data.msa import pipeline
from boltz.data.msa.pipeline import (
    component_paths,
    materialize_msa_csv,
    search_msa_components,
)
from boltz.data.parse import yaml as yaml_parser
from boltz.data.parse.yaml import materialize_prepared_msas, target_name_from_path


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
        assert not (tmp_path / f"{msa_id}.csv").exists()


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
    assert paired_path.read_text() == ""
    assert unpaired_path.read_text().startswith(">q\nAAAA")
    assert [call[2] for call in calls] == [False]


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
        lambda *_args, **_kwargs: {"target_0": "AAAA"},
    )

    def fake_search(*, data, msa_dir, **_kwargs):
        for msa_id in data:
            paired_path, unpaired_path = component_paths(msa_dir, msa_id)
            paired_path.write_text("")
            unpaired_path.write_text(">q\nAAAA\n")

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
        "paired": "msa/target_0_paired.a3m",
        "unpaired": "msa/target_0_unpaired.a3m",
    }
    assert prepared["templates"] == [{"cif": "template.cif"}]
    assert not (out_dir / "prepared_msa_manifest.json").exists()
    assert not (out_dir / "msa" / "target_0.csv").exists()


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


def test_cli_rejects_disabling_both_stages(tmp_path: Path) -> None:
    input_path = tmp_path / "target.yaml"
    input_path.write_text("sequences: []\n")

    result = CliRunner().invoke(
        main_module.cli,
        ["predict", str(input_path), "-D", "false", "-P", "false"],
    )

    assert result.exit_code == 2
    assert "At least one" in result.output
