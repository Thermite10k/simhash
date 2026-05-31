"""
Banded SimHash for SIFT descriptors using PySpark
==================================================
- m x d random hyperplane matrix (m = number of hash bits, d = 128 for SIFT)
- Signature: sign(H @ q) for each query vector q
- Banded LSH: split m bits into b bands of r = m/b bits
- Candidate pairs: share the same band bucket in >= 1 band
- Ground truth: cosine similarity > 0.8
- Sweeps over (m, b) to compare precision / recall / F1
"""

import numpy as np
import itertools
import time
from pyspark.sql import SparkSession


# ─────────────────────────────────────────────
# 1. Data loading
# ─────────────────────────────────────────────

def load_sift(path: str) -> np.ndarray:
    """
    Load SIFT descriptors from a .fvecs file (standard ANN benchmark format).
    Returns an (n, 128) float32 array.
    """
    with open(path, "rb") as f:
        raw = np.frombuffer(f.read(), dtype=np.float32)
    dim = raw[:1].view(np.int32)[0]
    assert dim == 128, f"Expected 128-d SIFT, got {dim}"
    n = len(raw) // (1 + dim)
    # Each vector is stored as [int32 dim, float32*128]
    raw = raw.reshape(n, 1 + dim)
    return raw[:, 1:]  # drop the leading dimension field


def make_synthetic_sift(n: int = 2000, d: int = 128, seed: int = 42) -> np.ndarray:
    """
    Generate synthetic unit-normalised vectors that mimic SIFT structure.
    Creates clusters so there are genuine near-duplicates to find.
    """
    rng = np.random.default_rng(seed)
    n_clusters = max(1, n // 20)
    centres = rng.standard_normal((n_clusters, d)).astype(np.float32)
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)

    vecs = []
    for i in range(n):
        c = centres[i % n_clusters]
        noise = rng.standard_normal(d).astype(np.float32) * 0.15
        v = c + noise
        v /= np.linalg.norm(v)
        vecs.append(v)
    return np.stack(vecs)


# ─────────────────────────────────────────────
# 2. SimHash core
# ─────────────────────────────────────────────

def build_hyperplane_matrix(m: int, d: int, seed: int = 0) -> np.ndarray:
    """
    Returns an (m, d) matrix of i.i.d. standard normal random hyperplane normals.
    Each row defines one hyperplane through the origin.
    """
    rng = np.random.default_rng(seed)
    H = rng.standard_normal((m, d)).astype(np.float32)
    return H


def compute_signatures(vectors: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    vectors : (n, d)
    H       : (m, d)
    returns : (n, m) binary array in {0, 1}
              sig[i, j] = 1 if dot(H[j], vectors[i]) > 0 else 0
    """
    projections = vectors @ H.T          # (n, m)
    return (projections > 0).astype(np.uint8)


def signature_to_bands(sig: np.ndarray, b: int) -> list[tuple[int, int]]:
    """
    Split an m-bit signature into b bands of r = m/b bits each.
    Returns a list of (band_index, band_value_as_int) pairs.
    """
    m = len(sig)
    r = m // b
    bands = []
    for band_idx in range(b):
        chunk = sig[band_idx * r: (band_idx + 1) * r]
        # Pack bits into a Python int for hashing
        val = int("".join(chunk.astype(str)), 2)
        bands.append((band_idx, val))
    return bands


# ─────────────────────────────────────────────
# 3. Ground truth
# ─────────────────────────────────────────────

def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """Returns the (n, n) cosine similarity matrix for unit-normalised vectors."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normed = vectors / np.clip(norms, 1e-9, None)
    return normed @ normed.T


def ground_truth_pairs(vectors: np.ndarray, threshold: float = 0.8) -> set[tuple[int, int]]:
    """Brute-force: all (i, j) pairs with cosine similarity > threshold."""
    sim = cosine_similarity_matrix(vectors)
    n = len(vectors)
    pairs = set()
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] > threshold:
                pairs.add((i, j))
    return pairs


# ─────────────────────────────────────────────
# 4. Spark LSH job
# ─────────────────────────────────────────────

def run_banded_simhash(
    spark: SparkSession,
    vectors: np.ndarray,
    m: int,
    b: int,
    seed: int = 0,
) -> set[tuple[int, int]]:
    """
    Run banded SimHash LSH on `vectors` using Spark.

    Parameters
    ----------
    vectors : (n, d) float array
    m       : number of hash bits  (must be divisible by b)
    b       : number of bands

    Returns
    -------
    Set of candidate (i, j) pairs (i < j) that share at least one band bucket.
    """
    assert m % b == 0, f"m ({m}) must be divisible by b ({b})"

    H = build_hyperplane_matrix(m, vectors.shape[1], seed=seed)
    sigs = compute_signatures(vectors, H)   # (n, m)

    # Build RDD: (doc_id, signature_array)
    indexed = list(enumerate(sigs.tolist()))  # list of (int, list[int])

    rdd = spark.sparkContext.parallelize(indexed, numSlices=8)

    # Emit ((band_idx, band_val) -> doc_id)
    def emit_bands(item):
        doc_id, sig = item
        sig_arr = np.array(sig, dtype=np.uint8)
        return [((bi, bv), doc_id) for bi, bv in signature_to_bands(sig_arr, b)]

    band_rdd = rdd.flatMap(emit_bands)

    # Group by band bucket, keep buckets with >= 2 docs
    grouped = (
        band_rdd
        .groupByKey()
        .mapValues(list)
        .filter(lambda x: len(x[1]) >= 2)
    )

    # Emit all pairs within each bucket, deduplicate
    def emit_pairs(item):
        _, doc_ids = item
        return [
            (min(a, b_), max(a, b_))
            for a, b_ in itertools.combinations(doc_ids, 2)
        ]

    candidate_pairs = (
        grouped
        .flatMap(emit_pairs)
        .distinct()
        .collect()
    )

    return set(candidate_pairs)


# ─────────────────────────────────────────────
# 5. Evaluation
# ─────────────────────────────────────────────

def evaluate(candidates: set, ground_truth: set) -> dict:
    """Compute precision, recall, F1 given candidate and ground-truth pair sets."""
    tp = len(candidates & ground_truth)
    fp = len(candidates - ground_truth)
    fn = len(ground_truth - candidates)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "tp":        tp,
        "fp":        fp,
        "fn":        fn,
        "candidates": len(candidates),
        "ground_truth": len(ground_truth),
    }


def print_results(results: list[dict]) -> None:
    header = (
        f"{'m':>5} {'b':>5} {'r':>5} | "
        f"{'Precision':>10} {'Recall':>8} {'F1':>8} | "
        f"{'Cands':>7} {'GT':>6} {'TP':>6} {'FP':>6} {'FN':>6} | "
        f"{'Time(s)':>8}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['m']:>5} {r['b']:>5} {r['m']//r['b']:>5} | "
            f"{r['precision']:>10.3f} {r['recall']:>8.3f} {r['f1']:>8.3f} | "
            f"{r['candidates']:>7} {r['ground_truth']:>6} {r['tp']:>6} {r['fp']:>6} {r['fn']:>6} | "
            f"{r['elapsed']:>8.2f}"
        )
    print("=" * len(header) + "\n")


# ─────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────

def main():
    # ── Configuration ──────────────────────────────────────────────────────
    SIFT_PATH = None          # Set to your .fvecs file path, e.g. "sift_base.fvecs"
                              # If None, synthetic data is used instead.
    N_VECTORS  = 500          # Number of vectors to use (keep small for local testing)
    THRESHOLD  = 0.8          # Cosine similarity threshold for "similar"
    SEED       = 42

    # Sweep: all (m, b) combinations where m % b == 0
    # m = total hash bits, b = number of bands, r = m/b bits per band
    M_VALUES   = [32, 64, 128]
    B_VALUES   = [4, 8, 16]

    # ── Load data ──────────────────────────────────────────────────────────
    if SIFT_PATH:
        print(f"Loading SIFT from {SIFT_PATH} ...")
        vectors = load_sift(SIFT_PATH)[:N_VECTORS]
    else:
        print(f"Generating {N_VECTORS} synthetic 128-d unit vectors ...")
        vectors = make_synthetic_sift(n=N_VECTORS, seed=SEED)

    print(f"Data shape: {vectors.shape}  (n={vectors.shape[0]}, d={vectors.shape[1]})")

    # ── Ground truth ───────────────────────────────────────────────────────
    print(f"\nComputing ground truth (cosine similarity > {THRESHOLD}) ...")
    gt_start = time.time()
    gt = ground_truth_pairs(vectors, threshold=THRESHOLD)
    print(f"Ground truth pairs: {len(gt)}  ({time.time() - gt_start:.2f}s)")

    # ── Spark session ──────────────────────────────────────────────────────
    spark = (
        SparkSession.builder
        .appName("BandedSimHash_SIFT")
        .master("local[*]")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # ── Sweep ──────────────────────────────────────────────────────────────
    results = []
    for m, b in itertools.product(M_VALUES, B_VALUES):
        if m % b != 0:
            continue
        r = m // b
        print(f"  Running m={m:>4}, b={b:>3}, r={r:>3} bits/band ...", end="", flush=True)
        t0 = time.time()
        candidates = run_banded_simhash(spark, vectors, m=m, b=b, seed=SEED)
        elapsed = time.time() - t0
        metrics = evaluate(candidates, gt)
        metrics.update({"m": m, "b": b, "elapsed": elapsed})
        results.append(metrics)
        print(f"  precision={metrics['precision']:.3f}  recall={metrics['recall']:.3f}  F1={metrics['f1']:.3f}  ({elapsed:.2f}s)")

    spark.stop()

    # ── Summary table ──────────────────────────────────────────────────────
    print_results(results)

    # ── Key observations ───────────────────────────────────────────────────
    best_f1 = max(results, key=lambda x: x["f1"])
    best_recall = max(results, key=lambda x: x["recall"])
    print(f"Best F1:     m={best_f1['m']}, b={best_f1['b']}  →  F1={best_f1['f1']:.3f}")
    print(f"Best Recall: m={best_recall['m']}, b={best_recall['b']}  →  Recall={best_recall['recall']:.3f}\n")


if __name__ == "__main__":
    main()
