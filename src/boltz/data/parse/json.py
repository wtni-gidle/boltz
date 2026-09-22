import json
import os
from pathlib import Path
from typing import Mapping, Optional

from rdkit.Chem.rdchem import Mol

from boltz.data.msa.pipeline import component_paths, materialize_msa_csv
from boltz.data.parse.compression import open_maybe_compressed_text, write_zstd_text
from boltz.data.parse.schema import parse_boltz_schema
from boltz.data.types import Target


def load_input_schema(path: Path) -> dict:
    """Load one strict JSON object from ``path``."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            schema = json.load(handle)
    except json.JSONDecodeError as exc:
        msg = f"Boltz input {path} must contain valid JSON: {exc.msg}."
        raise ValueError(msg) from exc

    if not isinstance(schema, dict):
        schema_type = type(schema).__name__
        msg = f"Boltz JSON input must contain an object, got {schema_type}."
        raise ValueError(msg)
    return schema


def _fallback_target_name(path: Path) -> str:
    """Return the target name encoded by a Boltz input filename."""
    name = path.stem
    if path.suffix.lower() == ".json" and name.endswith("_data"):
        return name.removesuffix("_data")
    return name


def target_name_from_schema(path: Path, schema: Mapping) -> str:
    """Return and validate the target name from a parsed Boltz JSON schema."""
    name = schema.get("name", _fallback_target_name(path))
    if not isinstance(name, str) or not name.strip():
        msg = "Top-level JSON 'name' must be a non-empty string."
        raise ValueError(msg)

    name = name.strip()
    invalid_chars = ("/", "\\", "\0", "\n", "\r")
    if name in {".", ".."} or any(char in name for char in invalid_chars):
        msg = (
            "Top-level JSON 'name' must be a single safe filename component "
            f"without path separators, got {name!r}."
        )
        raise ValueError(msg)
    return name


def target_name_from_path(path: Path) -> str:
    """Return the JSON ``name`` or fall back to the input filename."""
    if path.is_file() and path.suffix.lower() == ".json":
        return target_name_from_schema(path, load_input_schema(path))
    return _fallback_target_name(path)


def _resolve_path(value: str, input_dir: Path) -> Path:
    resource = Path(value).expanduser()
    if not resource.is_absolute():
        resource = input_dir / resource
    return resource.resolve()


def _is_resource_reference(value: object) -> bool:
    return isinstance(value, str) and value != ""


def _is_scalar_msa_reference(value: object) -> bool:
    return _is_resource_reference(value) and value != "empty"


def _relative_resource_reference(
    value: str,
    source_path: Path,
    destination_dir: Path,
) -> str:
    resource = _resolve_path(value, source_path.parent)
    return os.path.relpath(resource, start=destination_dir.resolve())


def rebase_schema_resource_paths(
    schema: dict,
    source_path: Path,
    destination_dir: Path,
) -> None:
    """Rebase supported JSON resource references for a prepared copy."""
    for item in schema.get("sequences", []):
        protein = item.get("protein") if isinstance(item, dict) else None
        if not isinstance(protein, dict):
            continue
        msa = protein.get("msa")
        if isinstance(msa, dict):
            for key in ("paired", "unpaired"):
                value = msa.get(key)
                if _is_resource_reference(value):
                    msa[key] = _relative_resource_reference(
                        value,
                        source_path,
                        destination_dir,
                    )
        elif _is_scalar_msa_reference(msa):
            protein["msa"] = _relative_resource_reference(
                msa,
                source_path,
                destination_dir,
            )

    for template in schema.get("templates", []):
        if not isinstance(template, dict):
            continue
        for member in template.get("chains", []):
            value = member.get("mmcifPath")
            if _is_resource_reference(value):
                member["mmcifPath"] = _relative_resource_reference(value, source_path, destination_dir)
        for key in ("cif", "pdb"):
            value = template.get(key)
            if _is_resource_reference(value):
                template[key] = _relative_resource_reference(
                    value,
                    source_path,
                    destination_dir,
                )


def _resolve_schema_resource_paths(schema: dict, source_path: Path) -> None:
    """Resolve supported scalar input references against their JSON file."""
    for item in schema.get("sequences", []):
        protein = item.get("protein") if isinstance(item, dict) else None
        if not isinstance(protein, dict):
            continue
        msa = protein.get("msa")
        if _is_scalar_msa_reference(msa):
            protein["msa"] = str(_resolve_path(msa, source_path.parent))

    for template in schema.get("templates", []):
        if not isinstance(template, dict):
            continue
        for member in template.get("chains", []):
            value = member.get("mmcifPath")
            if _is_resource_reference(value):
                member["mmcifPath"] = str(_resolve_path(value, source_path.parent))
        for key in ("cif", "pdb"):
            value = template.get(key)
            if _is_resource_reference(value):
                template[key] = str(_resolve_path(value, source_path.parent))


def materialize_prepared_msas(
    path: Path,
    schema: dict,
    output_dir: Optional[Path] = None,
) -> None:
    """Resolve paired/unpaired MSA mappings to native Boltz CSV files."""
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
            # Neither basenames from distinct directories nor entity indices
            # from different targets may collide in a shared runtime root.
            target_name = target_name_from_schema(path, schema)
            csv_name = f"{target_name}__entity_{len(csv_by_sequence)}.csv"
            csv_path = (
                output_dir / csv_name
                if output_dir is not None
                else paired_path.with_name(csv_name)
            ).resolve()
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            materialize_msa_csv(
                paired_path=paired_path,
                unpaired_path=unpaired_path,
                csv_path=csv_path,
                query_sequence=sequence,
            )
            csv_by_sequence[sequence] = csv_path
            spec_by_sequence[sequence] = spec

        protein["msa"] = str(csv_by_sequence[sequence])


def persist_msa_resources(schema: dict, output_dir: Path, source_dir: Optional[Path] = None) -> None:
    """Copy reusable MSA resources, preserving native scalar A3M/CSV routes.

    Paths in schema have already been rebased against output_dir. This is only
    called when exporting input, never by inference's read/convert path.
    """
    source_dir = output_dir if source_dir is None else source_dir
    by_sequence = {}
    writes = []
    for item in schema.get("sequences", []):
        protein = item.get("protein")
        if protein is None:
            continue
        msa = protein.get("msa")
        if not isinstance(msa, dict) and not _is_scalar_msa_reference(msa):
            continue
        sequence = protein["sequence"]
        if sequence in by_sequence:
            protein["msa"] = by_sequence[sequence]
            continue
        chain = protein["id"]
        if isinstance(chain, list):
            chain = chain[0]
        if not isinstance(chain, str) or any(char in chain for char in "/\\\0") or chain in (".", ".."):
            raise ValueError(f"Unsafe MSA chain filename component: {chain!r}")
        identifier = f"{schema['name']}__{chain}"
        if isinstance(msa, dict):
            paths = component_paths(output_dir / "msas", identifier)
            sources = [_resolve_path(msa[key], source_dir) for key in ("paired", "unpaired")]
            # Read both first, so updating the same bundle cannot clobber a
            # resource before another component has read it.
            contents = []
            for source in sources:
                with open_maybe_compressed_text(source) as handle:
                    contents.append(handle.read())
            for destination, content in zip(paths, contents, strict=True):
                writes.append((destination, content))
            prepared = {key: str(path.relative_to(output_dir)) for key, path in zip(("paired", "unpaired"), paths, strict=True)}
        else:
            source = _resolve_path(msa, source_dir)
            if source.suffix.lower() == ".csv":
                destination = output_dir / "msas" / f"{identifier}_msa.csv"
                content = source.read_bytes()
                writes.append((destination, content))
            else:
                destination = output_dir / "msas" / f"{identifier}_msa.a3m.zst"
                with open_maybe_compressed_text(source) as handle:
                    content = handle.read()
                writes.append((destination, content))
            prepared = str(destination.relative_to(output_dir))
        protein["msa"] = prepared
        by_sequence[sequence] = prepared

    # Snapshot every source before publishing any destination. In-place bundle
    # updates can swap paths between entities, not just paired/unpaired fields.
    for destination, content in writes:
        if isinstance(content, bytes):
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        else:
            write_zstd_text(destination, content)


def parse_json(
    path: Path,
    ccd: dict[str, Mol],
    mol_dir: Path,
    boltz2: bool = False,
    msa_materialization_dir: Optional[Path] = None,
) -> Target:
    """Parse a strict Boltz JSON input."""
    data = load_input_schema(path)
    _resolve_schema_resource_paths(data, path)
    materialize_prepared_msas(path, data, output_dir=msa_materialization_dir)
    name = target_name_from_schema(path, data)
    if any("chains" in item for item in data.get("templates", [])):
        from boltz.data.parse.prepared_templates import parse_grouped_templates
        return parse_grouped_templates(name, data, ccd, mol_dir, boltz2)
    return parse_boltz_schema(name, data, ccd, mol_dir, boltz2)
