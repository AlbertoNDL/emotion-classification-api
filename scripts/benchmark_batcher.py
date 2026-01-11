import asyncio
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from app.deps import get_batcher
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("benchmark")

# ---------- Configurable parameters ----------
test_texts = [
    "I am happy today!",
    "This is so frustrating...",
    "I am scared of the dark",
    "What a wonderful surprise!",
    "I feel very sad right now",
] * 20  # 100 requests

max_wait_options = [0.01, 0.02, 0.05, 0.1, 0.2]  # flush interval in seconds
min_batch_size_options = [1, 2, 4, 8]

results = []  # store benchmark results


async def benchmark_batcher(batcher, texts, max_wait, min_batch_size):
    batcher.max_wait = max_wait
    batcher.min_batch_size = min_batch_size

    if not batcher.is_ready():
        await batcher.wait_ready(timeout=30.0)

    latencies = []

    async def send_request(text):
        start = time.perf_counter()
        await batcher.predict(text)
        latencies.append((time.perf_counter() - start) * 1000)  # ms

    start_time = time.perf_counter()
    await asyncio.gather(*(send_request(t) for t in texts))
    total_time = time.perf_counter() - start_time

    latencies = np.array(latencies)
    throughput = len(texts) / total_time

    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)

    logger.info(
        "Benchmark | max_wait=%.3f s | min_batch_size=%d | throughput=%.1f req/s | "
        "p50=%.2f ms | p95=%.2f ms | p99=%.2f ms",
        max_wait,
        min_batch_size,
        throughput,
        p50,
        p95,
        p99,
    )

    results.append({
        "max_wait": max_wait,
        "min_batch_size": min_batch_size,
        "throughput": throughput,
        "p50": p50,
        "p95": p95,
        "p99": p99,
    })


async def main():
    batcher = get_batcher()
    await batcher.start()

    for max_wait in max_wait_options:
        for min_batch_size in min_batch_size_options:
            await benchmark_batcher(batcher, test_texts, max_wait, min_batch_size)

    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Heatmaps for p95, p99, and throughput
    for metric in ["p95", "p99", "throughput"]:
        pivot = df.pivot(index="min_batch_size", columns="max_wait", values=metric)
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="viridis")
        plt.title(f"{metric.upper()} heatmap")
        plt.ylabel("min_batch_size")
        plt.xlabel("max_wait (s)")
        plt.show()

    await batcher.stop()


if __name__ == "__main__":
    asyncio.run(main())
