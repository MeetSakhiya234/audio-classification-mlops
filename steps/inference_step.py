import gc
from typing import List, Tuple

import pyarrow.parquet as pq
from zenml import step

from src.audio_classifier.data import map_label
from src.audio_classifier.model import load_model, predict
from src.audio_classifier.preprocessing import (
    prepare_audio_bytes,
)


@step
def run_model_inference(
    test_path: str,
    indices: List[int],
) -> Tuple[List[int], List[int]]:
    """
    Load Wav2Vec2 and run inference on the
    balanced evaluation samples.
    """

    print("Loading Wav2Vec2 model...")

    model = load_model()

    table = pq.read_table(
        test_path,
        columns=["audio", "label"],
    )

    y_true = []
    y_pred = []

    print("Running model inference...")

    for count, index in enumerate(
        indices,
        start=1,
    ):

        audio = table.column(
            "audio"
        )[index].as_py()

        original_label = table.column(
            "label"
        )[index].as_py()

        audio_bytes = audio["bytes"]

        input_tensor = prepare_audio_bytes(
            audio_bytes
        )

        predicted_class, confidence, _ = predict(
            model,
            input_tensor,
        )

        true_class = map_label(
            original_label
        )

        y_true.append(true_class)
        y_pred.append(predicted_class)

        del input_tensor
        gc.collect()

        if count % 20 == 0:
            print(
                f"Processed {count}/{len(indices)}"
            )

    del model
    gc.collect()

    print("Model inference completed.")

    return y_true, y_pred
