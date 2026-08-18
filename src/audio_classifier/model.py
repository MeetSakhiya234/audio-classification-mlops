import torch
from transformers import Wav2Vec2ForSequenceClassification


MODEL_NAME = "superb/wav2vec2-base-superb-ks"


def load_model():
    """Load the pretrained Hugging Face keyword-spotting model efficiently."""

    print("Loading Wav2Vec2 model in memory-efficient mode...")

    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    model.eval()

    print("Wav2Vec2 model loaded successfully.")

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

    # Keep inference tensors in the same dtype as the model.
    input_values = input_values.to(dtype=torch.float16)

    with torch.no_grad():
        outputs = model(input_values=input_values)

    probabilities = torch.softmax(outputs.logits.float(), dim=-1)

    confidence, predicted_class = torch.max(probabilities, dim=-1)

    return (
        predicted_class.item(),
        confidence.item(),
        probabilities.squeeze(0).tolist(),
    )
