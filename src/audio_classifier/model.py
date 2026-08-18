import os

import numpy as np
import onnxruntime as ort


MODEL_PATH = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(__file__)
        )
    ),
    "models",
    "wav2vec2_int8_linear.onnx",
)


def load_model():
    """Load the INT8 Wav2Vec2 ONNX model."""

    print("Loading INT8 Wav2Vec2 ONNX model...")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"ONNX model not found: {MODEL_PATH}"
        )

    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"],
    )

    print("INT8 Wav2Vec2 ONNX model loaded successfully.")

    return session


def predict(model, input_values):
    """
    Run inference using the INT8 ONNX Wav2Vec2 model.

    Parameters
    ----------
    model:
        ONNX Runtime inference session.

    input_values:
        PyTorch tensor with shape [batch_size, audio_length].

    Returns
    -------
    predicted_class:
        Integer class ID.

    confidence:
        Probability of predicted class.

    probabilities:
        Probability distribution over all 12 classes.
    """

    if hasattr(input_values, "detach"):
        input_values = (
            input_values
            .detach()
            .cpu()
            .numpy()
        )

    input_values = np.asarray(
        input_values,
        dtype=np.float32,
    )

    input_name = model.get_inputs()[0].name

    outputs = model.run(
        None,
        {
            input_name: input_values,
        },
    )

    logits = outputs[0]

    logits = logits - np.max(
        logits,
        axis=-1,
        keepdims=True,
    )

    probabilities = np.exp(logits)

    probabilities /= np.sum(
        probabilities,
        axis=-1,
        keepdims=True,
    )

    predicted_class = int(
        np.argmax(
            probabilities,
            axis=-1,
        )[0]
    )

    confidence = float(
        probabilities[
            0,
            predicted_class,
        ]
    )

    return (
        predicted_class,
        confidence,
        probabilities[0].tolist(),
    )
