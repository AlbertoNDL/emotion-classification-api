import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from typing import List
import logging
import os
from app.logging import setup_logging
import time


setup_logging()
logger = logging.getLogger(__name__)

class EmotionModel:
    def __init__(self, onnx_path: str, model_name: str, model_provider: str = "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        available_providers = ort.get_available_providers()
        selected_providers = self._select_providers(
            requested_provider=model_provider,
            available_providers=available_providers,
        )

        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=selected_providers,
        )

        logger.info("MODEL_PROVIDER requested: %s", model_provider)
        logger.info("ONNX available providers: %s", available_providers)
        logger.info("ONNX providers: %s", self.session.get_providers())

        self.input_names = {i.name for i in self.session.get_inputs()}
        self.output_name = self.session.get_outputs()[0].name

    def predict_proba(self, texts: List[str]) -> List[List[float]]:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="np",
        )

        ort_inputs = {
            k: v.astype(np.int64) for k, v in inputs.items() if k in self.input_names
        }

        start = time.perf_counter()

        logits = self.session.run(
            [self.output_name],
            ort_inputs
        )[0]

        latency_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "ONNX inference | batch=%d | latency=%.2f ms",
            len(texts),
            latency_ms,
        )

        probs = self._softmax(logits)
        print("**************** probs: ", probs)
        print("**************** logits: ", logits)

        logger.info("Running ONNX batch of size %d", len(texts))
        return probs.tolist()

    @staticmethod
    def _select_providers(
        requested_provider: str,
        available_providers: List[str],
    ) -> List[str]:
        provider = (requested_provider or os.getenv("MODEL_PROVIDER", "cpu")).strip().lower()

        if provider == "cuda":
            if "CUDAExecutionProvider" in available_providers:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]

            logger.warning(
                "MODEL_PROVIDER=cuda requested but CUDAExecutionProvider is unavailable. Falling back to CPUExecutionProvider."
            )
            return ["CPUExecutionProvider"]

        if provider != "cpu":
            logger.warning(
                "Unsupported MODEL_PROVIDER=%s. Falling back to CPUExecutionProvider.",
                provider,
            )

        return ["CPUExecutionProvider"]

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
