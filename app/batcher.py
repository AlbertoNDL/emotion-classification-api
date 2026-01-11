import asyncio
import logging
import time
from typing import Tuple, List, Optional

from app.model import EmotionModel
from app.metrics import metrics

logger = logging.getLogger(__name__)


class DynamicBatcher:

    def __init__(
        self,
        model: EmotionModel,
        min_batch_size: int = 4,
        max_batch_size: int = 16,
        max_wait_ms: int = 25,
        startup_timeout: float = 45.0,
    ):
        self.model = model
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.max_wait = max_wait_ms / 1000.0
        self.startup_timeout = startup_timeout

        # queue item = (text, future, enqueue_ts)
        self._queue: asyncio.Queue[Tuple[str, asyncio.Future, float]] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None

        self._running = asyncio.Event()
        self._stopped = asyncio.Event()
        self._ready = asyncio.Event()

    # ------------------------
    # lifecycle
    # ------------------------

    async def start(self) -> None:
        if self._worker_task is not None:
            logger.warning("DynamicBatcher already started")
            return

        logger.info("Iniciando DynamicBatcher...")
        self._running.set()

        try:
            logger.info("Warming up the model...")
            await asyncio.to_thread(self.model.predict_proba, ["warmup text"] * 4)
            logger.info("Warm-up completed")
        except Exception:
            logger.error("Error during model warm-up", exc_info=True)
            raise

        self._ready.set()
        logger.info("DynamicBatcher is ready to receive requests.")

        self._worker_task = asyncio.create_task(
            self._worker_loop(),
            name="DynamicBatcher-worker",
        )

    async def stop(self) -> None:
        if self._worker_task is None:
            logger.warning("DynamicBatcher is not running")
            return

        logger.info("Stopping DynamicBatcher...")
        self._running.clear()
        self._stopped.set()

        try:
            await asyncio.wait_for(self._worker_task, timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting worker, cancelling")
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        while not self._queue.empty():
            _, future, _ = self._queue.get_nowait()
            if not future.done():
                future.set_exception(RuntimeError("Batcher stopped"))

        self._worker_task = None
        logger.info("DynamicBatcher stopped cleanly")

    # ------------------------
    # public API
    # ------------------------

    def is_ready(self) -> bool:
        return self._ready.is_set()

    async def wait_ready(self, timeout: Optional[float] = None) -> bool:
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def predict(self, text: str) -> List[float]:
        if not self._running.is_set():
            raise RuntimeError("Batcher is not running")

        if not self._ready.is_set():
            raise RuntimeError("Model not initialised yet")

        loop = asyncio.get_running_loop()
        future = loop.create_future()

        enqueue_ts = time.perf_counter()
        await self._queue.put((text, future, enqueue_ts))

        return await future

    async def predict_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        futures = [self.predict(text) for text in texts]
        return await asyncio.gather(*futures)

    # ------------------------
    # worker
    # ------------------------

    async def _worker_loop(self):
        try:
            while self._running.is_set() or not self._queue.empty():
                batch: list[Tuple[str, asyncio.Future, float]] = []
                loop = asyncio.get_running_loop()
                batch_start = loop.time()

                while len(batch) < self.max_batch_size and (loop.time() - batch_start) < self.max_wait:
                    remaining = max(0.001, self.max_wait - (loop.time() - batch_start))
                    try:
                        item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                        batch.append(item)

                        if hasattr(self, "min_batch_size") and len(batch) >= self.min_batch_size:
                            if (loop.time() - batch_start) >= (self.max_wait / 2):
                                logger.debug("Early flush triggered | batch=%d", len(batch))
                                break

                    except asyncio.TimeoutError:
                        break

                if not batch:
                    continue

                texts, futures, enqueue_times = zip(*batch)
                batch_size = len(batch)

                now = time.perf_counter()
                queue_delay_ms = (now - min(enqueue_times)) * 1000

                t0 = time.perf_counter()
                probs = await asyncio.to_thread(self.model.predict_proba, list(texts))
                infer_ms = (time.perf_counter() - t0) * 1000

                throughput = batch_size / (infer_ms / 1000)

                metrics.inc("batch.count")
                metrics.observe("batch.size", batch_size)
                metrics.observe("inference.ms", infer_ms)
                metrics.observe("queue.delay.ms", queue_delay_ms)

                if len(probs) != len(futures):
                    err = RuntimeError("Model returned mismatched batch size")
                    for fut in futures:
                        fut.set_exception(err)
                    continue

                for prob, future in zip(probs, futures):
                    future.set_result(prob)

                logger.info(
                    "Batch processed | batch=%d | queue=%.2f ms | infer=%.2f ms | throughput=%.1f req/s",
                    batch_size,
                    queue_delay_ms,
                    infer_ms,
                    throughput,
                )

        except asyncio.CancelledError:
            logger.info("DynamicBatcher worker cancelled")
        except Exception:
            logger.critical("DynamicBatcher worker crashed", exc_info=True)
        finally:
            logger.debug("DynamicBatcher worker finished")
