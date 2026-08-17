from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response

import io
import time
from contextlib import asynccontextmanager

import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile

from src.audio_classifier.data import CLASS_NAMES
from src.audio_classifier.model import load_model, predict
from src.audio_classifier.preprocessing import prepare_audio_bytes


MODEL = None

REQUEST_COUNT = Counter(
    "audio_api_requests_total",
    "Total number of API requests",
    ["endpoint"],
)

PREDICTION_COUNT = Counter(
    "audio_predictions_total",
    "Total number of audio predictions",
    ["predicted_class"],
)

PREDICTION_ERRORS = Counter(
    "audio_prediction_errors_total",
    "Total number of prediction errors",
)

INFERENCE_LATENCY = Histogram(
    "audio_inference_latency_seconds",
    "Audio model inference latency in seconds",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL

    print("Loading Wav2Vec2 model...")

    MODEL = load_model()

    print("Wav2Vec2 model loaded successfully.")

    yield

    MODEL = None

    print("Model released.")


app = FastAPI(
    title="Audio Classification MLOps API",
    description=(
        "12-class Speech Commands keyword "
        "classification using Wav2Vec2."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "service": "Audio Classification MLOps API",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "prediction_endpoint": "/predict",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "superb/wav2vec2-base-superb-ks",
        "model_loaded": MODEL is not None,
    }

@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        media_type="text/plain",
    )

@app.post("/predict")
async def predict_audio(
    file: UploadFile = File(...),
):
    REQUEST_COUNT.labels(
        endpoint="/predict"
    ).inc()
    global MODEL

    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    allowed_types = {
        "audio/wav",
        "audio/x-wav",
        "audio/wave",
        "audio/mpeg",
        "audio/flac",
        "audio/x-flac",
        "application/octet-stream",
    }

    filename = file.filename or ""

    allowed_extensions = {
        ".wav",
        ".mp3",
        ".flac",
    }

    extension = ""

    if "." in filename:
        extension = (
            "." + filename.rsplit(".", 1)[1].lower()
        )

    if (
        file.content_type not in allowed_types
        and extension not in allowed_extensions
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported audio format. "
                "Use WAV, MP3, or FLAC."
            ),
        )

    audio_bytes = await file.read()

    if not audio_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded audio file is empty.",
        )

    try:
        sf.read(
            io.BytesIO(audio_bytes),
            dtype="float32",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "Could not decode the uploaded "
                f"audio file: {exc}"
            ),
        )

    start_time = time.perf_counter()

    try:
        input_tensor = prepare_audio_bytes(
            audio_bytes
        )

        (
            predicted_class,
            confidence,
            probabilities,
        ) = predict(
            MODEL,
            input_tensor,
        )

    except Exception as exc:
        PREDICTION_ERRORS.inc()

        raise HTTPException(
            status_code=500,
            detail=f"Audio inference failed: {exc}",
        )

    inference_time = (
        time.perf_counter()
        - start_time
    )

    INFERENCE_LATENCY.observe(
        inference_time
    )

    PREDICTION_COUNT.labels(
        predicted_class=CLASS_NAMES[predicted_class]
    ).inc()

    predicted_label = CLASS_NAMES[
        predicted_class
    ]

    return {
        "filename": file.filename,
        "predicted_class": predicted_class,
        "predicted_label": predicted_label,
        "confidence": round(
            confidence,
            6,
        ),
        "inference_time_seconds": round(
            inference_time,
            4,
        ),
        "probabilities": {
            CLASS_NAMES[index]: round(
                probability,
                6,
            )
            for index, probability in enumerate(
                probabilities
            )
        },
    }
