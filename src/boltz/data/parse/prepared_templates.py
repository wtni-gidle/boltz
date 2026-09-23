"""Boltz template groups: native automatic matching or explicit residue pairs.

Groups determine joint geometry; structure keys determine file loading. Neither
writing prepared inputs nor splitting files may change the template conditions.
"""
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from boltz.data import const
from boltz.data.parse.compression import open_maybe_compressed_text, write_zstd_text
from boltz.data.parse.mmcif import parse_mmcif
from boltz.data.parse.prepared_io import temporary_directory
from boltz.data.parse.schema import get_template_records_from_matching, parse_boltz_schema
from boltz.data.tokenize.boltz2 import tokenize_structure
from boltz.data.types import Target, TemplateInfo
from boltz.data.write.mmcif import to_mmcif


def _read_cif(path: Path, ccd, mol_dir: Path):
    # parse_mmcif requires a filename. Decompression is private, even when
    # input writing is disabled; it never writes beside the user's CIF.
    with open_maybe_compressed_text(path) as handle:
        text = handle.read()
    with temporary_directory("boltz-template-") as scratch:
        unpacked = Path(scratch) / "template.cif"
        unpacked.write_text(text)
        return parse_mmcif(
            unpacked, mols=ccd, moldir=mol_dir,
            use_assembly=False, compute_interfaces=False,
        )


def _indices(mapping: dict, query_length: int, template_length: int):
    keys = {"queryIndices", "templateIndices"} & mapping.keys()
    if not keys:
        return None
    if len(keys) != 2:
        raise ValueError("Template mapping requires both queryIndices and templateIndices")
    query, template = mapping["queryIndices"], mapping["templateIndices"]
    if not isinstance(query, list) or not isinstance(template, list) or len(query) != len(template):
        raise ValueError("Template indices must be two equally sized lists")
    for values, size in [(query, query_length), (template, template_length)]:
        if any(type(value) is not int or value < 0 or value >= size for value in values):
            raise ValueError(f"Template mapping indices must be integers in [0, {size})")
        if len(set(values)) != len(values):
            raise ValueError("Template mapping indices must not repeat within a mapping")
    return query, template


def parse_grouped_templates(name: str, schema: dict, ccd, mol_dir: Path, boltz2: bool) -> Target:
    if not boltz2:
        raise ValueError("Templates are not supported in Boltz 1.0!")
    groups = schema["templates"]
    if any("chains" not in group for group in groups):
        raise ValueError("Use either grouped templates or legacy template entries, not both")
    target = parse_boltz_schema(name, {**schema, "templates": []}, ccd, mol_dir, boltz2)
    sequences = {
        chain.chain_name: target.sequences[chain.entity_id]
        for chain in target.record.chains
        if chain.mol_type == const.chain_type_ids["PROTEIN"]
    }
    records, structures, seen_groups = [], {}, set()
    for group_index, group in enumerate(groups):
        unknown = group.keys() - {"groupId", "chains", "force", "threshold"}
        if unknown:
            raise ValueError(f"Unknown template group fields: {sorted(unknown)}")
        group_id = group.get("groupId", f"template_{group_index}")
        if not isinstance(group_id, str) or not group_id or group_id in seen_groups:
            raise ValueError("Template groupId must be a unique nonempty string")
        seen_groups.add(group_id)
        members = group["chains"]
        if not isinstance(members, list) or not members:
            raise ValueError(f"Template group {group_id} needs a nonempty chains list")
        force = group.get("force", False)
        if type(force) is not bool:
            raise ValueError("Template force must be a boolean")
        threshold = group.get("threshold") if force else float("inf")
        if force and (type(threshold) not in (int, float) or not math.isfinite(threshold)):
            raise ValueError("Forced template group needs a finite threshold")
        parsed_by_path = {}
        for member in members:
            unknown = member.keys() - {"queryChain", "mmcifPath", "templateChain", "queryIndices", "templateIndices"}
            if unknown:
                raise ValueError(f"Unknown template mapping fields: {sorted(unknown)}")
            query_chain = member.get("queryChain")
            if query_chain not in sequences:
                raise ValueError(f"Template mapping queryChain {query_chain!r} is not a protein chain")
            value = member.get("mmcifPath")
            if not isinstance(value, str) or not value:
                raise ValueError("Template mapping requires mmcifPath")
            path = Path(value).resolve()
            if path not in parsed_by_path:
                structure_key = f"group_{group_index}_source_{len(parsed_by_path)}"
                parsed = _read_cif(path, ccd, mol_dir)
                parsed_by_path[path] = structure_key, parsed
                structures[structure_key] = parsed.data
            structure_key, parsed = parsed_by_path[path]
            protein_names = {
                str(chain["name"]) for chain in parsed.data.chains
                if chain["mol_type"] == const.chain_type_ids["PROTEIN"]
            }
            template_chain = member.get("templateChain")
            if template_chain is None and len(protein_names) == 1:
                template_chain = next(iter(protein_names))
            if template_chain not in protein_names:
                raise ValueError(
                    f"Template {path}: specify templateChain using a protein label_asym_id; "
                    f"available chains: {sorted(protein_names)}"
                )
            pairs = _indices(member, len(sequences[query_chain]), len(parsed.sequences[template_chain]))
            if pairs is None:
                matches = get_template_records_from_matching(
                    group_id, [query_chain], sequences, [template_chain], parsed.sequences,
                    force=force, threshold=threshold,
                )
                records.extend(replace(info, structure_name=structure_key) for info in matches)
            else:
                query_indices, template_indices = pairs
                records.append(TemplateInfo(
                    name=group_id, query_chain=query_chain, query_st=0,
                    query_en=len(sequences[query_chain]), template_chain=template_chain,
                    template_st=0, template_en=len(parsed.sequences[template_chain]),
                    force=force, threshold=threshold,
                    query_indices=query_indices, template_indices=template_indices,
                    structure_name=structure_key,
                ))
    return replace(target, record=replace(target.record, templates=records), templates=structures)


def _single_chain(structure, chain_name):
    mask = np.array([chain["name"] == chain_name for chain in structure.chains])
    if np.count_nonzero(mask) != 1:
        raise ValueError(f"Cannot export unique template chain {chain_name!r}")
    return replace(structure, mask=mask).remove_invalid_chains()


def export_templates(target: Target, out_dir: Path, ccd, mol_dir: Path, *, compress_fold_input: bool = False) -> list[dict]:
    """Write single-chain resources and record the native consumer's mapping.

    A persistence error fails explicitly: write=true must never silently remove
    a template that write=false would use.
    """
    groups, files, counts = {}, {}, {}
    suffix = ".zst" if compress_fold_input else ""
    for info in target.record.templates or []:
        group = groups.setdefault(info.name, {"groupId": info.name, "chains": [], "force": info.force})
        if info.force:
            group["threshold"] = info.threshold
        structure_key = info.structure_name or info.name
        structure = target.templates[structure_key]
        query = info.query_chain
        if any(char in query for char in "/\\\0") or query in (".", ".."):
            raise ValueError(f"Unsafe query chain filename component: {query!r}")
        file_key = (info.name, query, structure_key, info.template_chain)
        if file_key not in files:
            index = counts.get(query, 0)
            counts[query] = index + 1
            destination = out_dir / "msas" / f"{target.record.id}__{query}_template_{index}.cif{suffix}"
            single = _single_chain(structure, info.template_chain)
            write_zstd_text(destination, to_mmcif(single, boltz2=True), compress=compress_fold_input)
            rebuilt = _read_cif(destination, ccd, mol_dir)
            # Compare actual token inputs after native parsing/tokenization,
            # including missing positions and coordinates in the original frame.
            before, _ = tokenize_structure(single)
            after, _ = tokenize_structure(rebuilt.data)
            if len(before) != len(after):
                raise ValueError(f"Template export changed token count: {destination}")
            for field in ["res_idx", "res_type", "disto_mask", "frame_mask", "resolved_mask"]:
                if not np.array_equal(before[field], after[field]):
                    raise ValueError(f"Template export changed {field}: {destination}")
            for field in ["center_coords", "disto_coords", "frame_rot", "frame_t"]:
                if not np.allclose(before[field], after[field], rtol=1e-5, atol=1e-5, equal_nan=True):
                    raise ValueError(f"Template export changed {field}: {destination}")
            files[file_key] = destination, str(rebuilt.data.chains[0]["name"])
        destination, exported_chain = files[file_key]
        if info.query_indices is None:
            query_chain = next(chain for chain in target.record.chains if chain.chain_name == query)
            size = len(target.sequences[query_chain.entity_id])
            tokens, _ = tokenize_structure(structure)
            chain_id = next(chain["asym_id"] for chain in structure.chains if chain["name"] == info.template_chain)
            offset = info.template_st - info.query_st
            template_indices = list(dict.fromkeys(
                int(token["res_idx"]) for token in tokens
                if token["asym_id"] == chain_id and 0 <= int(token["res_idx"]) - offset < size
            ))
            query_indices = [index - offset for index in template_indices]
        else:
            query_indices, template_indices = info.query_indices, info.template_indices
        group["chains"].append({
            "queryChain": query, "mmcifPath": str(destination.relative_to(out_dir)),
            "templateChain": exported_chain,
            "queryIndices": query_indices, "templateIndices": template_indices,
        })
    return list(groups.values())
