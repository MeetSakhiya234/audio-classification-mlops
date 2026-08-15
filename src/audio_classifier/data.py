from pathlib import Path
from typing import Dict, List

import pyarrow.parquet as pq


TARGET_LABELS: Dict[int, int] = {
    0: 0,   # yes
    1: 1,   # no
    2: 2,   # up
    3: 3,   # down
    4: 4,   # left
    5: 5,   # right
    6: 6,   # on
    7: 7,   # off
    8: 8,   # stop
    9: 9,   # go
    35: 11, # _silence_
}


CLASS_NAMES: List[str] = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
    "_unknown_",
    "_silence_",
]


def map_label(original_label: int) -> int:
    """
    Convert the original Google Speech Commands label
    into our 12-class keyword-spotting label.
    """
    if original_label in TARGET_LABELS:
        return TARGET_LABELS[original_label]

    return 10  # unknown


def inspect_parquet(parquet_path: str) -> dict:
    """
    Read lightweight metadata from a Speech Commands Parquet file.
    Does not load audio into memory.
    """
    path = Path(parquet_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    table = pq.read_table(
        path,
        columns=["file", "label", "is_unknown", "speaker_id", "utterance_id"],
    )

    labels = table.column("label").to_pylist()

    mapped_labels = [map_label(label) for label in labels]

    distribution = {
        CLASS_NAMES[class_id]: mapped_labels.count(class_id)
        for class_id in range(len(CLASS_NAMES))
    }

    return {
        "path": str(path),
        "samples": len(labels),
        "class_distribution": distribution,
    }


def get_audio_path(parquet_path: str, row_index: int = 0) -> str:
    """
    Return the original audio filename stored in a Parquet row.
    """
    path = Path(parquet_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    table = pq.read_table(
        path,
        columns=["file"],
    ).slice(row_index, 1)

    files = table.column("file").to_pylist()

    if not files:
        raise IndexError(f"No row found at index {row_index}")

    return files[0]
