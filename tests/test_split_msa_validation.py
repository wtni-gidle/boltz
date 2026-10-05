"""Split A3M validation must precede conversion and public input publication."""

import gzip
import io
import json
import lzma
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import zstandard as zstd

from boltz import main
from boltz.data import const
from boltz.data.msa import pipeline
from boltz.data.parse import json as json_parser
from boltz.data.parse.json import materialize_prepared_msas, persist_msa_resources


class SplitMSAValidationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.paired = self.root / "paired.a3m"
        self.unpaired = self.root / "unpaired.a3m"
        self.csv = self.root / "demo__A.csv"
        self.paired.write_text(">q\nAAAA\n")
        self.unpaired.write_text(">q\nAAAA\n")

    def convert(self, query="AAAA"):
        pipeline.materialize_msa_csv(self.paired, self.unpaired, self.csv, query)
        return self.csv.read_text().splitlines()

    def assert_invalid(self, channel, content, *, record=1):
        path = self.paired if channel == "paired" else self.unpaired
        path.write_text(content)
        self.csv.write_text("OLD CSV")
        with self.assertRaises(ValueError) as caught:
            self.convert()
        message = str(caught.exception)
        self.assertIn(channel, message)
        self.assertIn(str(path), message)
        self.assertIn(f"record {record}", message)
        self.assertEqual(self.csv.read_text(), "OLD CSV")

    def test_valid_insertions_wrapping_and_gaps_keep_original_pair_keys(self):
        self.paired.write_text("\n>q\nAA\nAA\n\n>gap\n----\n>hit\nAAggAA\n")
        self.unpaired.write_text(">q\nAAAA\n>other\nAA-A\n")
        self.assertEqual(self.convert(), ["key,sequence", "0,AAAA", "2,AAggAA", "-1,AA-A"])

    def test_empty_paired_and_query_only_unpaired_are_valid(self):
        for empty in ("", "\n  \n"):
            with self.subTest(empty=empty):
                self.paired.write_text(empty)
                self.assertEqual(self.convert(), ["key,sequence", "-1,AAAA"])

    def test_supported_ambiguous_letters_are_preserved(self):
        self.paired.write_text("")
        self.unpaired.write_text(">query\nXBZJUO\n>hit\nXBzZJU-\n")
        self.assertEqual(self.convert("XBZJUO"), ["key,sequence", "-1,XBZJUO", "-1,XBzZJU-"])

    def test_both_queries_must_match_exactly_without_gaps_or_insertions(self):
        for channel in ("paired", "unpaired"):
            for query in ("CCCC", "AAA", "AAAAA", "aaAA", "AAaAA", "AA-A", "----"):
                with self.subTest(channel=channel, query=query):
                    self.paired.write_text(">q\nAAAA\n")
                    self.unpaired.write_text(">q\nAAAA\n")
                    self.assert_invalid(channel, f">q\n{query}\n>valid_later\nAAAA\n")

    def test_every_hit_has_query_alignment_width(self):
        for channel in ("paired", "unpaired"):
            for hit in ("AAA", "AAAAA", "AAaA", "---"):
                with self.subTest(channel=channel, hit=hit):
                    self.paired.write_text(">q\nAAAA\n")
                    self.unpaired.write_text(">q\nAAAA\n")
                    self.assert_invalid(channel, f">q\nAAAA\n>bad\n{hit}\n", record=2)

    def test_invalid_symbols_are_rejected(self):
        for hit in ("AA.A", "AA*A", "AA1A", "AAßA", "AA A"):
            with self.subTest(hit=hit):
                self.assert_invalid("unpaired", f">q\nAAAA\n>bad\n{hit}\n", record=2)

    def test_missing_headers_and_empty_records_are_rejected(self):
        for content, record in (("AAAA\n", 1), (">q\n", 1), (">\nAAAA\n", 1),
                                (">empty\n>q\nAAAA\n", 1),
                                (">q\nAAAA\n>empty\n", 2)):
            for channel in ("paired", "unpaired"):
                with self.subTest(channel=channel, content=content):
                    self.paired.write_text(">q\nAAAA\n")
                    self.unpaired.write_text(">q\nAAAA\n")
                    self.assert_invalid(channel, content, record=record)

    def test_unpaired_cannot_be_empty_even_with_paired_query(self):
        self.assert_invalid("unpaired", "")

    def test_validation_precedes_sequence_count_limits(self):
        for channel, limit in (("paired", "max_paired_seqs"), ("unpaired", "max_msa_seqs")):
            with self.subTest(channel=channel), patch.object(const, limit, 1):
                self.paired.write_text(">q\nAAAA\n")
                self.unpaired.write_text(">q\nAAAA\n")
                self.assert_invalid(channel, ">q\nAAAA\n>outside_quota\nAAA\n", record=2)

    def test_supported_compression_is_detected_by_content(self):
        text = b">q\nAAAA\n>hit\nAAggAA\n"
        for compressor in (lambda data: data, gzip.compress, lzma.compress,
                           zstd.ZstdCompressor().compress):
            with self.subTest(compressor=compressor):
                self.paired.write_text("")
                self.unpaired.write_bytes(compressor(text))
                self.assertEqual(self.convert(), ["key,sequence", "-1,AAAA", "-1,AAggAA"])

    def schema(self):
        return {"name": "demo", "sequences": [{"protein": {
            "id": "A", "sequence": "AAAA", "msa": {
                "paired": str(self.paired), "unpaired": str(self.unpaired),
            },
        }}]}

    def test_prepared_conversion_reports_target_chain_and_channel(self):
        self.paired.write_text(">q\nCCCC\n")
        with self.assertRaises(ValueError) as caught:
            materialize_prepared_msas(self.root / "input.json", self.schema(), self.root / "runtime")
        for field in ("demo", "chain", "A", "paired", "record 1"):
            self.assertIn(field, str(caught.exception))
        self.assertFalse(list((self.root / "runtime").glob("*.csv")))

    def test_publication_validates_all_sources_before_replacing_any(self):
        schema = self.schema()
        bad = self.root / "bad.a3m"
        bad.write_text(">q\nCCCC\n>wrong_width\nCCC\n")
        schema["sequences"].append({"protein": {
            "id": "B", "sequence": "CCCC", "msa": {
                "paired": str(bad), "unpaired": str(bad),
            },
        }})
        output = self.root / "public"
        old = output / "msas/demo__A_unpairedmsa.a3m"
        old.parent.mkdir(parents=True)
        old.write_text("OLD RESOURCE")
        with self.assertRaises(ValueError):
            persist_msa_resources(schema, output)
        self.assertEqual(old.read_text(), "OLD RESOURCE")
        self.assertEqual(list(old.parent.iterdir()), [old])

    def test_auto_data_only_checks_search_results_with_or_without_publication(self):
        source = self.root / "input.json"
        source.write_text(json.dumps({"name": "demo", "sequences": [
            {"protein": {"id": "A", "sequence": "AAAA"}},
        ]}))
        target = SimpleNamespace(
            record=SimpleNamespace(id="demo", chains=[SimpleNamespace(
                chain_name="A", entity_id=0, mol_type=const.chain_type_ids["PROTEIN"], msa_id=0)]),
            sequences={0: "AAAA"},
        )

        def search(*, msa_dir, **kwargs):
            paired, unpaired = pipeline.component_paths(msa_dir, "demo__A")
            paired.write_text(f">q\n{query}\n")
            unpaired.write_text(">q\nAAAA\n")

        for query in ("CCCC", "AAAA"):
            for write in (False, True):
                with self.subTest(query=query, write=write), \
                     patch.object(main, "load_canonicals", return_value={}), \
                     patch.object(main, "parse_input_target", return_value=target), \
                     patch.object(main, "search_msa_components", side_effect=search):
                    target.record.chains[0].msa_id = 0
                    output = self.root / f"public-{query}-{write}"
                    args = ([source], self.root / "runtime", self.root / "ccd",
                            self.root / "mols", True, True, "unused", "greedy")
                    kwargs = {"prepared_output_root": output, "write_input_json": write}
                    if query != "AAAA":
                        with self.assertRaisesRegex(ValueError, "paired"):
                            main.prepare_msa_inputs(*args, **kwargs)
                        self.assertFalse(list(output.rglob("*.json")))
                        self.assertFalse(list(output.rglob("*.a3m")))
                    else:
                        main.prepare_msa_inputs(*args, **kwargs)
                        saved = output / "demo/demo_data.json"
                        self.assertEqual(saved.exists(), write)
                        if write:
                            msa = json.loads(saved.read_text())["sequences"][0]["protein"]["msa"]
                            self.assertEqual((saved.parent / msa["paired"]).read_text(), ">q\nAAAA\n")

    def test_custom_data_only_rejects_invalid_split_with_or_without_publication(self):
        self.paired.write_text(">q\nCCCC\n")
        source = self.root / "input.json"
        source.write_text(json.dumps(self.schema()))
        for write in (False, True):
            with self.subTest(write=write), \
                 patch.object(main, "load_canonicals", return_value={}), \
                 patch.object(json_parser, "parse_boltz_schema",
                              side_effect=AssertionError("native parsing should not be reached")):
                output = self.root / f"custom-{write}"
                with self.assertRaisesRegex(ValueError, "paired"):
                    main.prepare_msa_inputs(
                        [source], self.root / "runtime", self.root / "ccd", self.root / "mols",
                        True, False, "unused", "greedy", prepared_output_root=output,
                        write_input_json=write,
                    )
                self.assertFalse(list(output.rglob("*.json")))

    def test_scalar_csv_and_explicit_empty_routes_keep_their_publication_behavior(self):
        scalar = self.root / "scalar.csv"
        contents = "key,sequence\n-1,AAAA\n9,AA-A\n"
        scalar.write_text(contents)
        schema = self.schema()
        schema["sequences"][0]["protein"]["msa"] = str(scalar)
        schema["sequences"].append({"protein": {"id": "B", "sequence": "CCCC", "msa": "empty"}})
        output = self.root / "scalar-public"
        persist_msa_resources(schema, output)
        self.assertEqual((output / "msas/demo__A_msa.csv").read_text(), contents)
        self.assertEqual(schema["sequences"][1]["protein"]["msa"], "empty")

    def test_inference_entry_rejects_custom_split_before_publishing_for_all_stage_flags(self):
        self.paired.write_text(">q\nCCCC\n")
        source = self.root / "input.json"
        source.write_text(json.dumps(self.schema()))
        for data in (False, True):
            for write in (False, True):
                with self.subTest(data=data, write=write), patch.object(
                    json_parser, "parse_boltz_schema",
                    side_effect=AssertionError("native parsing reached before invalid split was rejected"),
                ), redirect_stderr(io.StringIO()):
                    output = self.root / f"out-{data}-{write}"
                    with self.assertRaisesRegex(RuntimeError, "paired") as caught:
                        main.process_input(
                            path=source, ccd={}, msa_dir=self.root / "runtime",
                            mol_dir=self.root / "mols", boltz2=True,
                            run_data_pipeline=data, use_msa_server=False,
                            msa_server_url="unused", msa_pairing_strategy="greedy",
                            msa_server_username=None, msa_server_password=None,
                            api_key_header=None, api_key_value=None, max_msa_seqs=8192,
                            processed_msa_dir=self.root / "processed/msa",
                            processed_constraints_dir=self.root / "processed/constraints",
                            processed_templates_dir=self.root / "processed/templates",
                            processed_mols_dir=self.root / "processed/mols",
                            structure_dir=self.root / "processed/structures",
                            records_dir=self.root / "processed/records",
                            prepared_output_root=output, write_input_json=write,
                        )
                    self.assertIsInstance(caught.exception.__cause__, ValueError)
                    self.assertFalse(list(output.rglob("*.json")))


if __name__ == "__main__":
    unittest.main()
