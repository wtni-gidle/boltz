"""Check production prediction dispatch against real output files, without folding."""

import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from boltz import main


class StopAfterSkip(Exception):
    """Stop dispatch after its real completeness decision, before model loading."""


class WrapperSkipContractTest(unittest.TestCase):
    def pending(self, root, *, model, compressed, pae=False, pde=False, skip=True,
                checkpoint_parameters=None):
        source = root / "target.json"
        source.write_text("{}")
        record = SimpleNamespace(id="target", affinity=None)
        manifest = main.Manifest([record])
        observed = {}
        original_filter = main.filter_inputs_structure

        def filter_actual_files(*args, **kwargs):
            result = original_filter(*args, **kwargs)
            observed[kwargs["seed"]] = [item.id for item in result.records]
            return result

        with ExitStack() as stack:
            stack.enter_context(patch.dict(main.os.environ, {"WORLD_SIZE": "1", "SLURM_NTASKS": "1"}))
            replacements = {
                "check_inputs": lambda path: [path],
                "target_name_from_path": lambda path: path.stem,
                "download_boltz1": lambda *a, **k: None,
                "download_boltz2": lambda *a, **k: None,
                "process_inputs": lambda **k: None,
                "seed_everything": lambda *a: None,
                "filter_inputs_structure": filter_actual_files,
            }
            for name, replacement in replacements.items():
                stack.enter_context(patch.object(main, name, replacement))
            stack.enter_context(patch.object(main.Manifest, "load", return_value=manifest))
            stack.enter_context(patch.object(main.click, "get_current_context",
                return_value=SimpleNamespace(call_on_close=stack.callback)))
            stack.enter_context(patch.object(main, "MSAModuleArgs", side_effect=StopAfterSkip))
            checkpoint = None
            if checkpoint_parameters is not None:
                checkpoint = str(root / "custom.ckpt")
                def read_checkpoint(path, *, map_location, weights_only):
                    self.assertEqual(path, checkpoint)
                    self.assertEqual(map_location, "cpu")
                    self.assertFalse(weights_only)
                    return {"hyper_parameters": checkpoint_parameters}
                stack.enter_context(patch.object(main.torch, "load", read_checkpoint, create=True))
            with self.assertRaises(StopAfterSkip):
                main.predict.callback(
                    str(source), str(root / "out"), run_data_pipeline=False,
                    write_input_json=False, cache=str(root / "cache"), model=model,
                    seeds="7,9", diffusion_samples=2, skip=skip,
                    compress_full_confidence=compressed,
                    write_full_pae=pae, write_full_pde=pde,
                    checkpoint=checkpoint,
                )
        return observed

    def write_outputs(self, root, *, compressed, matrices):
        suffix = "npz" if compressed else "json"
        for seed in (7, 9):
            for sample in range(2):
                stem = f"seed-{seed}_sample-{sample}"
                paths = [f"models/{stem}_model.cif",
                         f"summary_confidences/{stem}_summary_confidences.json",
                         f"full_data/plddt_{stem}.{suffix}"]
                paths += [f"full_data/{kind}_{stem}.{suffix}" for kind in matrices]
                for name in paths:
                    path = root / "out/target" / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"nonempty; the skip contract does not parse content")

    def test_default_boltz2_requires_each_pae_and_pde_in_current_format(self):
        for compressed in (False, True):
            for matrix in ("pae", "pde"):
                for damage in ("missing", "empty", "wrong_format"):
                    with self.subTest(compressed=compressed, matrix=matrix, damage=damage), \
                         tempfile.TemporaryDirectory() as temporary:
                        root = Path(temporary)
                        self.write_outputs(root, compressed=compressed, matrices=("pae", "pde"))
                        suffix = "npz" if compressed else "json"
                        path = root / f"out/target/full_data/{matrix}_seed-9_sample-1.{suffix}"
                        if damage == "empty":
                            path.write_bytes(b"")
                        elif damage == "missing":
                            path.unlink()
                        else:
                            path.rename(path.with_suffix(".json" if compressed else ".npz"))
                        self.assertEqual(self.pending(root, model="boltz2", compressed=compressed),
                                         {7: [], 9: ["target"]})

    def test_boltz2_complete_outputs_skip_but_skip_false_reruns(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_outputs(root, compressed=False, matrices=("pae", "pde"))
            self.assertEqual(self.pending(root, model="boltz2", compressed=False), {7: [], 9: []})
            self.assertEqual(self.pending(root, model="boltz2", compressed=False, skip=False),
                             {7: ["target"], 9: ["target"]})

    def test_boltz1_keeps_flag_controlled_matrix_requirements(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_outputs(root, compressed=False, matrices=())
            self.assertEqual(self.pending(root, model="boltz1", compressed=False), {7: [], 9: []})
            for option in ("pae", "pde"):
                with self.subTest(option=option):
                    self.assertEqual(self.pending(root, model="boltz1", compressed=False,
                                                  **{option: True}),
                                     {7: ["target"], 9: ["target"]})

    def test_custom_boltz2_without_pae_does_not_rerun_forever(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_outputs(root, compressed=False, matrices=("pde",))
            # Boltz-2 predict_step ignores the full-output flags; capabilities
            # come from the checkpoint, including when PAE was explicitly set.
            for pae_flag in (False, True):
                with self.subTest(pae_flag=pae_flag):
                    self.assertEqual(self.pending(root, model="boltz2", compressed=False,
                        pae=pae_flag,
                        checkpoint_parameters={"confidence_prediction": True, "alpha_pae": 0}),
                        {7: [], 9: []})
            self.assertEqual(self.pending(root, model="boltz2", compressed=False,
                checkpoint_parameters={"confidence_prediction": True, "alpha_pae": 1}),
                {7: ["target"], 9: ["target"]})
            (root / "out/target/full_data/pde_seed-9_sample-1.json").unlink()
            # Missing hyperparameters use the actual Boltz2 constructor defaults:
            # confidence_prediction=True and alpha_pae=0.0.
            self.assertEqual(self.pending(root, model="boltz2", compressed=False,
                                          checkpoint_parameters={}),
                             {7: [], 9: ["target"]})


if __name__ == "__main__":
    unittest.main()
