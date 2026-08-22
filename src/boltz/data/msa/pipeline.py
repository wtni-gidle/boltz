from pathlib import Path
from typing import Optional

from boltz.data import const
from boltz.data.msa.mmseqs2 import run_mmseqs2
from boltz.data.parse.compression import open_maybe_compressed_text


def component_paths(msa_dir: Path, msa_id: str) -> tuple[Path, Path]:
    """Return the prepared paired and unpaired A3M paths for an entity."""
    return (
        msa_dir / f"{msa_id}_paired.a3m",
        msa_dir / f"{msa_id}_unpaired.a3m",
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

    if len(data) > 1:
        paired_msas = run_mmseqs2(
            sequences,
            msa_dir / f"{target_id}_paired_tmp",
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
        msa_dir / f"{target_id}_unpaired_tmp",
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
        paired_path.write_text(paired_msas[index])
        unpaired_path.write_text(unpaired_msas[index])


def read_a3m_sequences(path: Path) -> list[str]:
    """Read sequences from an A3M file, accepting wrapped sequence lines."""
    sequences: list[str] = []
    current: list[str] = []
    with open_maybe_compressed_text(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                    current = []
            else:
                current.append(line)
    if current:
        sequences.append("".join(current))
    return sequences


def _query_sequence(sequence: str) -> str:
    return "".join(char for char in sequence if char != "-" and not char.islower())


def materialize_msa_csv(
    paired_path: Path,
    unpaired_path: Path,
    csv_path: Path,
    query_sequence: str,
) -> None:
    """Combine prepared paired/unpaired A3M files into a Boltz keyed CSV."""
    if not paired_path.is_file():
        msg = f"Prepared paired MSA not found: {paired_path}"
        raise FileNotFoundError(msg)
    if not unpaired_path.is_file():
        msg = f"Prepared unpaired MSA not found: {unpaired_path}"
        raise FileNotFoundError(msg)

    paired_rows = read_a3m_sequences(paired_path)[: const.max_paired_seqs]
    paired_keys = [
        row_index
        for row_index, sequence in enumerate(paired_rows)
        if sequence != "-" * len(sequence)
    ]
    paired_rows = [
        sequence for sequence in paired_rows if sequence != "-" * len(sequence)
    ]

    unpaired_rows = read_a3m_sequences(unpaired_path)
    if not unpaired_rows:
        msg = f"Prepared unpaired MSA is empty: {unpaired_path}"
        raise ValueError(msg)
    if _query_sequence(unpaired_rows[0]).upper() != query_sequence.upper():
        msg = (
            f"The first sequence in {unpaired_path} does not match the query sequence."
        )
        raise ValueError(msg)

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
