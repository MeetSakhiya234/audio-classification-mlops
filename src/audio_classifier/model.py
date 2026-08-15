import torch
from transformers import Wav2Vec2ForSequenceClassification


MODEL_NAME = "superb/wav2vec2-base-superb-ks"


def load_model():
    """Load the pretrained Hugging Face keyword-spotting model."""
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_NAME
    )

    model.eval()

    return model


def predict(model, input_values):
    """
    Run inference on preprocessed audio.

    Parameters
    ----------
    model:
        Loaded Wav2Vec2 classification model.

    input_values:
        Tensor with shape [batch_size, audio_length].

    Returns
    -------
    predicted_class:
        Integer class ID.

    confidence:
        Probability of predicted class.

    probabilities:
        Probability distribution over all 12 classes.
    """

    with torch.no_grad():
        outputs = model(input_values=input_values)

    probabilities = torch.softmax(outputs.logits, dim=-1)

    confidence, predicted_class = torch.max(probabilities, dim=-1)

    return (
        predicted_class.item(),
        confidence.item(),
        probabilities.squeeze(0).tolist(),
    )
