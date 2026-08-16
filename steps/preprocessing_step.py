from collections import defaultdict
from typing import List

import pyarrow.parquet as pq
from zenml import step

from src.audio_classifier.data import (
    CLASS_NAMES,
    map_label,
)


SAMPLES_PER_CLASS = 10


@step
def prepare_evaluation_data(
    test_path: str,
) -> List[int]:
    """
    Select the exact same balanced evaluation samples
    used by the baseline evaluation.

    12 classes × 10 samples = 120 samples.
    """

    print("Preparing balanced evaluation samples...")

    table = pq.read_table(
        test_path,
        columns=["label"],
    )

    selected = defaultdict(list)

    labels = table.column(
        "label"
    ).to_pylist()

    for index, original_label in enumerate(labels):

        class_id = map_label(
            original_label
        )

        if (
            len(selected[class_id])
            < SAMPLES_PER_CLASS
        ):
            selected[class_id].append(index)

        if all(
            len(selected[class_id])
            >= SAMPLES_PER_CLASS
            for class_id in range(
                len(CLASS_NAMES)
            )
        ):
            break

    indices = []

    for class_id in range(
        len(CLASS_NAMES)
    ):
        indices.extend(
            selected[class_id]
        )

    expected_samples = (
        len(CLASS_NAMES)
        * SAMPLES_PER_CLASS
    )

    print(
        f"Selected samples: {len(indices)}"
    )

    print(
        f"Expected samples: {expected_samples}"
    )

    if len(indices) != expected_samples:
        raise RuntimeError(
            "Balanced sample selection failed."
        )

    return indices
