"""Quality-control summaries stored separately from descriptor features."""

from __future__ import annotations

from dataclasses import asdict

from .fcgr import fcgr_matrix


def qc_summary(sequence: str, scales: tuple[int, ...]) -> list[dict[str, int | float]]:
    """Return per-scale FCGR masking statistics without adding them to features."""

    rows: list[dict[str, int | float]] = []
    for k in scales:
        result = fcgr_matrix(sequence, k)
        rows.append({
            "k": result.k,
            "sequence_length": result.sequence_length,
            "total_windows": result.total_windows,
            "valid_windows": result.valid_windows,
            "masked_windows": result.masked_windows,
            "valid_fraction": result.valid_fraction,
        })
    return rows
