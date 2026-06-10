"""Minimal FASTA parser with duplicate-identifier safeguards."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FastaRecord:
    identifier: str
    description: str
    sequence: str


def read_fasta(path: str | Path) -> list[FastaRecord]:
    """Parse FASTA records without silently accepting duplicate identifiers."""

    records: list[FastaRecord] = []
    seen: set[str] = set()
    identifier: str | None = None
    description = ""
    sequence_parts: list[str] = []

    def flush() -> None:
        nonlocal identifier, description, sequence_parts
        if identifier is None:
            return
        if identifier in seen:
            raise ValueError(f"Duplicate FASTA identifier: {identifier}")
        seen.add(identifier)
        records.append(FastaRecord(identifier, description, "".join(sequence_parts)))
        identifier = None
        description = ""
        sequence_parts = []

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header = line[1:].strip()
                if not header:
                    raise ValueError(f"Empty FASTA header at line {line_number}")
                fields = header.split(maxsplit=1)
                identifier = fields[0]
                description = fields[1] if len(fields) > 1 else ""
            else:
                if identifier is None:
                    raise ValueError(f"Sequence data before first FASTA header at line {line_number}")
                sequence_parts.append(line)
    flush()
    if not records:
        raise ValueError("No FASTA records found")
    return records
