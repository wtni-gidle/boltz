"""Exercise the real shell entry point without invoking a model."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class RunBoltzScriptTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "input.json"
        self.source.write_text("{}")
        executable = self.root / "boltz"
        executable.write_text(
            '#!/bin/bash\nprintf "GPU=%s\\n" "${CUDA_VISIBLE_DEVICES-unset}"\n'
            'printf "ARG=%s\\n" "$@"\n'
        )
        executable.chmod(0o755)
        self.environment = dict(os.environ, PATH=f"{self.root}:{os.environ['PATH']}",
                                CUDA_VISIBLE_DEVICES="9")
        self.script = Path(__file__).resolve().parents[1] / "run_boltz.sh"

    def run_script(self, *options):
        return subprocess.run(
            ["bash", str(self.script), "-i", str(self.source), "-o", str(self.root / "out"),
             "-d", "7,8", *options],
            env=self.environment, text=True, capture_output=True, check=False,
        )

    def test_boolean_spellings_control_devices_and_flag_forwarding(self):
        values = (("True", True), ("YES", True), ("1", True), (" On ", True),
                  ("False", False), ("NO", False), ("0", False), (" off ", False))
        value_flags = {"-D": "--run_data_pipeline", "-P": "--run_inference",
                       "-J": "--write_input_json", "-z": "--compress_fold_input",
                       "-f": "--compress_full_confidence"}
        for value, enabled in values:
            with self.subTest(value=value):
                options = [part for flag in (*value_flags, "-M", "-S")
                           for part in (flag, value)]
                if not enabled:
                    options += ["-D", "true"]
                result = self.run_script(*options)
                self.assertEqual(result.returncode, 0, result.stderr)
                lines = result.stdout.splitlines()
                self.assertIn("GPU=7,8" if enabled else "GPU=9", lines)
                args = [line[4:] for line in lines if line.startswith("ARG=")]
                self.assertEqual(args[args.index("--devices") + 1], "2" if enabled else "1")
                for flag, forwarded in value_flags.items():
                    expected = "true" if enabled or flag == "-D" else "false"
                    self.assertEqual(args[args.index(forwarded) + 1], expected)
                self.assertEqual("--use_msa_server" in args, enabled)
                self.assertEqual("--skip" in args, enabled)

    def test_invalid_boolean_values_fail_before_executable_runs(self):
        for flag in ("-D", "-P", "-J", "-z", "-f", "-M", "-S"):
            for value in ("typo", "", " "):
                with self.subTest(flag=flag, value=value):
                    result = self.run_script(flag, value)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(flag, result.stderr)
                    self.assertNotIn("ARG=", result.stdout)

    def test_both_stages_disabled_is_rejected_after_normalization(self):
        result = self.run_script("-D", "FALSE", "-P", "OFF")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cannot both be false", result.stdout)
        self.assertNotIn("ARG=", result.stdout)


if __name__ == "__main__":
    unittest.main()
