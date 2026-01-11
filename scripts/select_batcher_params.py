import numpy as np

# Rows = min_batch_size: [1, 2, 4, 8]
# Columns = max_wait: [0.01, 0.02, 0.05, 0.1, 0.2]

P95 = np.array([
    [47499.0, 26.3, 22.8, 24.2, 23.9],
    [32.9,   30.3, 26.8, 38.2, 25.1],
    [31.4,   24.7, 27.7, 24.1, 25.3],
    [30.2,   25.3, 25.5, 24.5, 25.1],
])

THROUGHPUT = np.array([
    [2.1,    1546.0, 1223.7, 727.7, 409.7],
    [2598.1, 1568.6, 1128.6, 588.8, 429.4],
    [2678.5, 1871.2, 1181.6, 758.0, 438.3],
    [2837.3, 1547.4, 1104.5, 759.5, 419.4],
])

MIN_BATCHES = np.array([1, 2, 4, 8])
MAX_WAITS = np.array([0.01, 0.02, 0.05, 0.1, 0.2])

THROUGHPUT_RATIO = 0.7  # 70% del máximo

def select_best():
    max_tp = THROUGHPUT.max()
    tp_threshold = max_tp * THROUGHPUT_RATIO

    valid = THROUGHPUT >= tp_threshold
    masked_p95 = np.where(valid, P95, np.inf)

    idx = np.unravel_index(masked_p95.argmin(), masked_p95.shape)

    return {
        "min_batch_size": int(MIN_BATCHES[idx[0]]),
        "max_wait": float(MAX_WAITS[idx[1]]),
        "p95_ms": float(P95[idx]),
        "throughput": float(THROUGHPUT[idx]),
    }


if __name__ == "__main__":
    best = select_best()
    print("Optimal batcher config:")
    for k, v in best.items():
        print(f"  {k}: {v}")
