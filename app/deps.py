from functools import lru_cache
from app.config import settings
from app.model import EmotionModel
from app.batcher import DynamicBatcher
from scripts.select_batcher_params import select_best

@lru_cache(maxsize=1)
def get_model() -> EmotionModel:
    return EmotionModel(
        onnx_path=settings.ONNX_MODEL_PATH,
        model_name=settings.HF_MODEL_NAME,
    )

@lru_cache(maxsize=1)
def get_batcher() -> DynamicBatcher:
    best = select_best()

    return DynamicBatcher(
        model=get_model(),
        max_batch_size=16,
        min_batch_size=best["min_batch_size"],
        max_wait_ms=int(best["max_wait"] * 1000),
    )
