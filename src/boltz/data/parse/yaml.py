from pathlib import Path
from typing import Optional

import yaml
from rdkit.Chem.rdchem import Mol

from boltz.data.msa.pipeline import materialize_msa_csv
from boltz.data.parse.schema import parse_boltz_schema
from boltz.data.types import Target


def target_name_from_path(path: Path) -> str:
    """Return the target name encoded by a Boltz YAML filename."""
    name = path.stem
    if path.suffix.lower() in (".yml", ".yaml") and name.endswith("_data"):
        return name.removesuffix("_data")
    return name


def _resolve_path(path: str, input_dir: Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = input_dir / resolved
    return resolved


def materialize_prepared_msas(
    path: Path,
    schema: dict,
    output_dir: Optional[Path] = None,
) -> None:
    """Resolve paired/unpaired MSA mappings to native Boltz CSV files.

    When ``output_dir`` is provided, generated CSV files are kept away from the
    prepared A3M inputs. Inference uses this to put all materialized data in its
    process-private temporary directory.
    """
    csv_by_sequence: dict[str, Path] = {}
    spec_by_sequence: dict[str, tuple[Path, Path]] = {}

    for item in schema.get("sequences", []):
        protein = item.get("protein")
        if protein is None or not isinstance(protein.get("msa"), dict):
            continue

        msa = protein["msa"]
        unknown = set(msa) - {"paired", "unpaired"}
        if unknown or set(msa) != {"paired", "unpaired"}:
            msg = (
                "Prepared protein MSA must contain exactly 'paired' and "
                f"'unpaired' paths, got {sorted(msa)}."
            )
            raise ValueError(msg)

        paired_path = _resolve_path(str(msa["paired"]), path.parent)
        unpaired_path = _resolve_path(str(msa["unpaired"]), path.parent)
        sequence = str(protein["sequence"])
        spec = (paired_path, unpaired_path)

        if sequence in spec_by_sequence and spec_by_sequence[sequence] != spec:
            msg = "Proteins with the same sequence must share one prepared MSA."
            raise ValueError(msg)

        if sequence not in csv_by_sequence:
            paired_suffix = "_paired.a3m"
            if paired_path.name.endswith(paired_suffix):
                csv_name = paired_path.name.removesuffix(paired_suffix) + ".csv"
            else:
                csv_name = paired_path.with_suffix("").with_suffix(".csv").name
            csv_path = (
                output_dir / csv_name
                if output_dir is not None
                else paired_path.with_name(csv_name)
            )
            materialize_msa_csv(
                paired_path=paired_path,
                unpaired_path=unpaired_path,
                csv_path=csv_path,
                query_sequence=sequence,
            )
            csv_by_sequence[sequence] = csv_path
            spec_by_sequence[sequence] = spec

        protein["msa"] = str(csv_by_sequence[sequence])


def parse_yaml(
    path: Path,
    ccd: dict[str, Mol],
    mol_dir: Path,
    boltz2: bool = False,
    msa_materialization_dir: Optional[Path] = None,
) -> Target:
    """Parse a Boltz input yaml / json.

    The input file should be a yaml file with the following format:

    sequences:
        - protein:
            id: A
            sequence: "MADQLTEEQIAEFKEAFSLF"
        - protein:
            id: [B, C]
            sequence: "AKLSILPWGHC"
        - rna:
            id: D
            sequence: "GCAUAGC"
        - ligand:
            id: E
            smiles: "CC1=CC=CC=C1"
        - ligand:
            id: [F, G]
            ccd: []
    constraints:
        - bond:
            atom1: [A, 1, CA]
            atom2: [A, 2, N]
        - pocket:
            binder: E
            contacts: [[B, 1], [B, 2]]
    templates:
        - path: /path/to/template.pdb
          ids: [A] # optional, specify which chains to template

    version: 1

    Parameters
    ----------
    path : Path
        Path to the YAML input format.
    components : Dict
        Dictionary of CCD components.
    boltz2 : bool
        Whether to parse the input for Boltz2.

    Returns
    -------
    Target
        The parsed target.

    """
    with path.open("r") as file:
        data = yaml.safe_load(file)

    materialize_prepared_msas(path, data, output_dir=msa_materialization_dir)
    name = target_name_from_path(path)
    return parse_boltz_schema(name, data, ccd, mol_dir, boltz2)
