import os

import pyarrow.parquet as pq
from zenml import step


TEST_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--google--speech_commands/"
    "snapshots/a751309c0fd613e8a5d30d77900f30e8b42bc2da/"
    "v0.02/test-00000-of-00001.parquet"
)


@step
def load_dataset() -> str:
    """
    Verify and register the Speech Commands
    evaluation dataset for the ZenML pipeline.
    """

    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_PATH}"
        )

    table = pq.read_table(
        TEST_PATH,
        columns=["label"],
    )

    print("Dataset loaded successfully.")
    print(f"Total dataset samples: {len(table)}")

    return TEST_PATH
