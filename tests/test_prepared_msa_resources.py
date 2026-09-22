import json
from pathlib import Path

from boltz.data.msa import pipeline
from boltz.data.parse.json import materialize_prepared_msas


def test_persistence_snapshots_swapped_source_paths_before_writing(tmp_path):
    from boltz.data.parse.json import persist_msa_resources
    folder = tmp_path / "msas"
    folder.mkdir()
    (folder / "job__A_msa.csv").write_text("key,sequence\n-1,CCCC\n")
    (folder / "job__B_msa.csv").write_text("key,sequence\n-1,AAAA\n")
    schema = {"name": "job", "sequences": [
        {"protein": {"id": "A", "sequence": "AAAA", "msa": "msas/job__B_msa.csv"}},
        {"protein": {"id": "B", "sequence": "CCCC", "msa": "msas/job__A_msa.csv"}},
    ]}
    persist_msa_resources(schema, tmp_path)
    assert (folder / "job__A_msa.csv").read_text() == "key,sequence\n-1,AAAA\n"
    assert (folder / "job__B_msa.csv").read_text() == "key,sequence\n-1,CCCC\n"


def test_search_workdirs_are_private_fresh_and_cleaned(tmp_path, monkeypatch):
    calls = []

    def service(sequences, prefix, **kwargs):
        prefix = Path(prefix)
        assert not (prefix / "stale").exists()
        prefix.mkdir(parents=True)
        (prefix / "stale").write_text("cached query")
        calls.append(prefix)
        return [f">q\n{seq}\n" for seq in sequences]

    monkeypatch.setattr(pipeline, "run_mmseqs2", service)
    for sequence in ["AAAA", "CCCC"]:
        pipeline.search_msa_components(
            {"job__A": sequence}, "job", tmp_path / "msas", "unused", "greedy"
        )
    assert len(set(calls)) == 2
    assert all(not path.exists() for path in calls)
    assert all(tmp_path / "msas" not in path.parents for path in calls)
    assert pipeline.read_a3m_sequences(
        tmp_path / "msas" / "job__A_unpairedmsa.a3m.zst"
    ) == ["CCCC"]


def test_same_basename_msas_do_not_overwrite_other_entities(tmp_path):
    schema = {"sequences": []}
    for index, sequence in enumerate(["AAAA", "CCCC"]):
        folder = tmp_path / str(index)
        folder.mkdir()
        (folder / "paired.a3m").write_text("")
        (folder / "unpaired.a3m").write_text(f">q\n{sequence}\n")
        schema["sequences"].append({"protein": {
            "id": chr(65 + index), "sequence": sequence,
            "msa": {"paired": f"{index}/paired.a3m", "unpaired": f"{index}/unpaired.a3m"},
        }})
    materialize_prepared_msas(tmp_path / "job.json", schema, tmp_path / "private")
    paths = [Path(item["protein"]["msa"]) for item in schema["sequences"]]
    assert len(set(paths)) == 2
    assert paths[0].read_text().splitlines() == ["key,sequence", "-1,AAAA"]
    assert paths[1].read_text().splitlines() == ["key,sequence", "-1,CCCC"]
