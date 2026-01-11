import logging
import time
from fastapi import FastAPI, Depends, Request, APIRouter, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from slowapi import Limiter
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.batcher import DynamicBatcher
from app.deps import get_batcher
from app.config import settings
from app.security import get_api_key
from app.schemas import (
    EmotionRequest,
    EmotionBatchRequest,
    EmotionResponse,
    EmotionBatchResponse,
)
from app.logging import setup_logging
from app.utils.prediction import argmax_with_confidence
from app.metrics import metrics

from src.preprocess import clean_text

LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

# ──────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────
setup_logging()
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────
# Rate limiting
# ──────────────────────────────────────────────────
def rate_limit_key(request: Request) -> str:
    return request.headers.get(settings.API_KEY_NAME) or get_remote_address(request)

limiter = Limiter(key_func=rate_limit_key)

# ──────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────
app = FastAPI(
    title="Emotion Classification API",
    version="1.0.0",
)

app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

# ──────────────────────────────────────────────────
# Startup / Shutdown
# ──────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    batcher = get_batcher()
    await batcher.start()
    logger.info("DynamicBatcher successfully started")

@app.on_event("shutdown")
async def shutdown_event():
    batcher = get_batcher()
    await batcher.stop()
    logger.info("DynamicBatcher successfully stopped")

# ──────────────────────────────────────────────────
# Exception handlers
# ──────────────────────────────────────────────────
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request, exc):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Try again later."},
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.exception("Unhandled exception occurred")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

# ──────────────────────────────────────────────────
# Router with API prefix
# ──────────────────────────────────────────────────
router = APIRouter(prefix=settings.API_PREFIX)

# ──────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────
@router.get("/internal/metrics", tags=["Internal"])
async def get_metrics():
    return metrics.snapshot()

# ──────────────────────────────────────────────────
# Health check
# ──────────────────────────────────────────────────
@router.get("/health/live", tags=["System"])
async def health_live(batcher: DynamicBatcher = Depends(get_batcher)):
    try:
        result = await batcher.predict("healthcheck")
        if not result:
            raise RuntimeError("Empty result")

        return {"status": "healthy", "inference": "ok"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

# ──────────────────────────────────────────────────
# Single prediction
# ──────────────────────────────────────────────────
@router.post(
    "/predict-emotion",
    response_model=EmotionResponse,
    tags=["Inference"],
)
@limiter.limit("100/minute")
async def predict_emotion(
    request: Request,
    payload: EmotionRequest,
    api_key: str = Depends(get_api_key),
    batcher: DynamicBatcher = Depends(get_batcher),
):
    cleaned = clean_text(payload.text)

    if not cleaned.strip():
        return EmotionResponse(emotion="neutral", confidence=1.0)

    start = time.perf_counter()
    probs = await batcher.predict(cleaned)
    latency = (time.perf_counter() - start) * 1000
    logger.info("API latency | %.2f ms", latency)

    idx, confidence = argmax_with_confidence(probs)
    return EmotionResponse(emotion=LABELS[idx], confidence=confidence)

# ──────────────────────────────────────────────────
# Batch prediction
# ──────────────────────────────────────────────────
@router.post(
    "/predict-emotion-batch",
    response_model=EmotionBatchResponse,
    tags=["Inference"],
)
@limiter.limit("10/minute")
async def predict_emotion_batch(
    request: Request,
    payload: EmotionBatchRequest,
    api_key: str = Depends(get_api_key),
    batcher: DynamicBatcher = Depends(get_batcher),
):
    if not payload.texts:
        return EmotionBatchResponse(results=[])

    cleaned_texts = [t for t in map(clean_text, payload.texts) if t.strip()]
    if not cleaned_texts:
        return EmotionBatchResponse(results=[])

    probs_batch = await batcher.predict_batch(cleaned_texts)

    results = []
    for probs in probs_batch:
        idx, confidence = argmax_with_confidence(probs)
        results.append(
            EmotionResponse(
                emotion=LABELS[idx],
                confidence=confidence,
            )
        )

    return EmotionBatchResponse(results=results)

# ──────────────────────────────────────────────────
# Register router
# ──────────────────────────────────────────────────
app.include_router(router)
