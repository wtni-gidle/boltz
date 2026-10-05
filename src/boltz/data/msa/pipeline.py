from pathlib import Path
import os
import tempfile
from typing import Optional

from boltz.data import const
from boltz.data.msa.mmseqs2 import run_mmseqs2
from boltz.data.parse.compression import open_maybe_compressed_text, write_zstd_text


def component_paths(msa_dir: Path, msa_id: str) -> tuple[Path, Path]:
    """Return the prepared paired and unpaired A3M paths for an entity."""
    return (
        msa_dir / f"{msa_id}_pairedmsa.a3m.zst",
        msa_dir / f"{msa_id}_unpairedmsa.a3m.zst",
    )


def _auth_headers(
    api_key_header: Optional[str],
    api_key_value: Optional[str],
) -> Optional[dict[str, str]]:
    if api_key_value is None:
        return None
    return {
        "Content-Type": "application/json",
        api_key_header or "X-API-Key": api_key_value,
    }


def search_msa_components(
    data: dict[str, str],
    target_id: str,
    msa_dir: Path,
    msa_server_url: str,
    msa_pairing_strategy: str,
    msa_server_username: Optional[str] = None,
    msa_server_password: Optional[str] = None,
    api_key_header: Optional[str] = None,
    api_key_value: Optional[str] = None,
) -> None:
    """Search and save paired/unpaired A3M files without creating CSV files."""
    msa_dir.mkdir(parents=True, exist_ok=True)
    sequences = list(data.values())
    auth_headers = _auth_headers(api_key_header, api_key_value)

    temp_base = os.environ.get("SLURM_TMPDIR")
    if temp_base and not Path(temp_base).is_dir():
        temp_base = None
    with tempfile.TemporaryDirectory(prefix="boltz-search-", dir=temp_base) as scratch:
        if len(data) > 1:
            paired_msas = run_mmseqs2(
                sequences,
                Path(scratch) / "paired",
                use_env=True,
                use_pairing=True,
                host_url=msa_server_url,
                pairing_strategy=msa_pairing_strategy,
                msa_server_username=msa_server_username,
                msa_server_password=msa_server_password,
                auth_headers=auth_headers,
            )
        else:
            paired_msas = [""] * len(data)

        unpaired_msas = run_mmseqs2(
            sequences,
            Path(scratch) / "unpaired",
            use_env=True,
            use_pairing=False,
            host_url=msa_server_url,
            pairing_strategy=msa_pairing_strategy,
            msa_server_username=msa_server_username,
            msa_server_password=msa_server_password,
            auth_headers=auth_headers,
        )

    for index, msa_id in enumerate(data):
        paired_path, unpaired_path = component_paths(msa_dir, msa_id)
        write_zstd_text(paired_path, paired_msas[index])
        write_zstd_text(unpaired_path, unpaired_msas[index])


def read_a3m_sequences(path: Path) -> list[str]:
    """Read FASTA records, accepting wrapped sequence lines and empty files."""
    sequences: list[str] = []
    current: list[str] = []
    record = 0
    with open_maybe_compressed_text(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if record:
                    if not current:
                        raise ValueError(f"{path}, record {record}: header has no sequence")
                    sequences.append("".join(current))
                    current = []
                record += 1
                if not line[1:].strip():
                    raise ValueError(f"{path}, record {record}: empty FASTA header")
            else:
                if not record:
                    raise ValueError(f"{path}, record 1: sequence has no FASTA header")
                current.append(line)
    if record:
        if not current:
            raise ValueError(f"{path}, record {record}: header has no sequence")
        sequences.append("".join(current))
    return sequences


def validate_msa_components(
    paired_path: Path,
    unpaired_path: Path,
    query_sequence: str,
    *,
    context: str = "split MSA",
) -> tuple[list[str], list[str]]:
    """Validate both complete A3Ms before truncating, converting or publishing.

    Query records must be exact, uppercase and ungapped. Lowercase insertions
    do not count toward hit alignment width; gap characters do. Record positions
    are preserved so filtering paired gap rows cannot change pairing keys.
    """
    symbols = set(const.prot_letter_to_token)
    symbols.update(char.lower() for char in const.prot_letter_to_token)
    components: list[list[str]] = []
    for channel, path in (("paired", paired_path), ("unpaired", unpaired_path)):
        label = f"{context}, {channel} MSA {path}"
        if not path.is_file():
            raise FileNotFoundError(f"{label}: file not found")
        try:
            rows = read_a3m_sequences(path)
        except ValueError as error:
            raise ValueError(f"{context}, {channel} MSA: {error}") from error
        if not rows:
            if channel == "unpaired":
                raise ValueError(f"{label}, record 1: missing required query sequence")
            components.append(rows)
            continue

        query = rows[0]
        if "-" in query or any(char.islower() for char in query):
            raise ValueError(
                f"{label}, record 1: query must be uppercase and ungapped, without insertions"
            )
        if query != query_sequence:
            position = next(
                (
                    index
                    for index, (actual, expected) in enumerate(zip(query, query_sequence), 1)
                    if actual != expected
                ),
                min(len(query), len(query_sequence)) + 1,
            )
            raise ValueError(
                f"{label}, record 1: first sequence does not match the query sequence; "
                f"expected length {len(query_sequence)}, observed {len(query)}, "
                f"first difference at position {position} (1-based)"
            )
        for record, sequence in enumerate(rows, 1):
            for position, char in enumerate(sequence, 1):
                if char not in symbols:
                    raise ValueError(
                        f"{label}, record {record}: unsupported symbol {char!r} "
                        f"at position {position} (1-based)"
                    )
            width = sum(not char.islower() for char in sequence)
            if width != len(query_sequence):
                raise ValueError(
                    f"{label}, record {record}: aligned width {width}, "
                    f"expected {len(query_sequence)}"
                )
        components.append(rows)
    return components[0], components[1]


def materialize_msa_csv(
    paired_path: Path,
    unpaired_path: Path,
    csv_path: Path,
    query_sequence: str,
    *,
    context: Optional[str] = None,
) -> None:
    """Combine prepared paired/unpaired A3M files into a Boltz keyed CSV."""
    paired_rows, unpaired_rows = validate_msa_components(
        paired_path, unpaired_path, query_sequence,
        context=context if context is not None else csv_path.stem,
    )
    paired_rows = paired_rows[: const.max_paired_seqs]
    paired_keys = [
        row_index
        for row_index, sequence in enumerate(paired_rows)
        if sequence != "-" * len(sequence)
    ]
    paired_rows = [
        sequence for sequence in paired_rows if sequence != "-" * len(sequence)
    ]

    unpaired_rows = unpaired_rows[: (const.max_msa_seqs - len(paired_rows))]
    if paired_rows:
        unpaired_rows = unpaired_rows[1:]

    sequences = paired_rows + unpaired_rows
    keys = paired_keys + [-1] * len(unpaired_rows)
    csv_lines = ["key,sequence"]
    csv_lines.extend(f"{key},{sequence}" for key, sequence in zip(keys, sequences))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("\n".join(csv_lines))


def materialize_msa_csvs(data: dict[str, str], msa_dir: Path) -> None:
    """Materialize keyed CSV files for all prepared protein entities."""
    for msa_id, query_sequence in data.items():
        paired_path, unpaired_path = component_paths(msa_dir, msa_id)
        materialize_msa_csv(
            paired_path=paired_path,
            unpaired_path=unpaired_path,
            csv_path=msa_dir / f"{msa_id}.csv",
            query_sequence=query_sequence,
        )
