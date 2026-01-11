from typing import List
from pydantic import BaseModel, Field


class EmotionRequest(BaseModel):
    text: str = Field(min_length=3, max_length=5000, example="I feel very happy today")


class EmotionBatchRequest(BaseModel):
    texts: List[str] = Field(
        example=[
            "I am excited about this",
            "This makes me angry",
            "I feel sad and tired",
        ]
    )


class EmotionResponse(BaseModel):
    emotion: str
    confidence: float


class EmotionBatchResponse(BaseModel):
    results: List[EmotionResponse]
