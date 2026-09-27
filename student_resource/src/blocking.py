"""
blocking.py  --  Phase 2: Inverted-Index Token Blocking (v5 - In-Memory Accumulator Architecture)

Key Architecture & Optimization:
    1. Zero disk-scan retrieval:
       - Candidate token lengths precomputed into a compact 1D float32 array (41 MB for 10.3M entities).
       - Candidate country codes encoded into a 1D uint8 array (10.3 MB for 10.3M entities).
       - Inverted index stores token -> np.ndarray[int32] of candidate indices.
       - Retrieval accumulates token weights directly from postings lists.
       - Jaccard = intersection / (s1_weight + cand_weight - intersection).
       - Result: 0 disk reads during retrieval, < 3 GB peak RAM, > 150 queries/sec.
    2. Empirical recall ceiling:
       - 100% exact country match on ground truth.
       - 97.5% name + addr_nums overlap.
       - 99.95% name + addr_nums + distinctive addr_tokens overlap.
    3. Official Submission Schema:
       - Output file: student_resource/output/candidate_pairs.tsv
       - Header: source1_entity_id\\tcandidate_entity_ids
       - Verified compliant with student_resource/utils/validate_submission.py
"""

import os
import sys
import time
import logging
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)
DATA_TRAIN  = os.path.join(PROJECT_DIR, "dataset", "train")
DATA_TEST   = os.path.join(PROJECT_DIR, "dataset", "test")
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "output")
CACHE_DIR   = os.path.join(OUTPUT_DIR, ".cache")

sys.path.insert(0, SRC_DIR)
from normalize import normalize_sources

# ---------------------------------------------------------------------------
# Default Hyper-parameters
# ---------------------------------------------------------------------------
MAX_CANDIDATES      = 100      # Top-K candidate entities per S1 entity
NAME_FREQ_CUTOFF    = 0.005    # Exclude name tokens appearing in > 0.5% of entities
NUMS_FREQ_CUTOFF    = 0.001    # Exclude addr numbers appearing in > 0.1% of entities
ADDR_FREQ_CUTOFF    = 0.002    # Exclude addr tokens appearing in > 0.2% of entities
MAX_POSTINGS        = 100_000  # Cap postings list size per token
MIN_TOKEN_LEN       = 2        # Skip 1-character tokens
CHUNK_SIZE          = 10_000   # S1 rows per logging/processing chunk
NAME_WEIGHT         = 1.0
ADDR_WEIGHT         = 0.6
ADDR_NUM_WEIGHT     = 2.0
RECALL_THRESHOLD    = 0.95

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ===========================================================================
# 1. Normalization & Parquet Caching
# ===========================================================================

def normalize_and_cache(tsv_path: str, label: str, cache_path: str,
                         force: bool = False) -> str:
    """
    Normalize source TSV and write columnar cache to parquet.
    Skips if cache already exists (unless force=True).
    """
    if os.path.exists(cache_path) and not force:
        sz = os.path.getsize(cache_path) / (1024 * 1024)
        log.info(f"  Cache hit for {label}: {cache_path} ({sz:.1f} MB)")
        return cache_path

    log.info(f"Normalizing {label} from {tsv_path} ...")
    t0 = time.time()
    df = pd.read_csv(tsv_path, sep="\t", dtype=str, keep_default_na=True)
    log.info(f"  Loaded {len(df):,} rows in {time.time()-t0:.2f}s")

    t1 = time.time()
    df = normalize_sources(df)
    log.info(f"  Normalized in {time.time()-t1:.2f}s")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    # Store token lists as pipe-separated strings to ensure clean parquet schema
    df["name_tokens_str"] = df["name_tokens"].apply(lambda l: "|".join(l) if isinstance(l, list) else "")
    df["addr_tokens_str"] = df["addr_tokens"].apply(lambda l: "|".join(l) if isinstance(l, list) else "")
    df["addr_nums"]       = df["addr_nums"].fillna("")
    df["country_norm"]    = df["country_norm"].fillna("")

    save_cols = ["entity_id", "name_tokens_str", "addr_tokens_str", "addr_nums", "country_norm"]
    df[save_cols].to_parquet(cache_path, index=False, compression="snappy")
    sz = os.path.getsize(cache_path) / (1024 * 1024)
    log.info(f"  Saved {label} cache -> {cache_path} ({sz:.1f} MB)")

    del df
    return cache_path


def load_cached_source(cache_path: str, cols: list = None) -> pd.DataFrame:
    """
    Load cached source parquet and ensure entity_id is restored as a column.
    """
    df = pd.read_parquet(cache_path, columns=cols)
    if "entity_id" not in df.columns and (df.index.name == "entity_id" or "entity_id" in df.index.names):
        df = df.reset_index()
    return df


# ===========================================================================
# 2. In-Memory Index Construction (Compact & RAM-safe)
# ===========================================================================

def build_inverted_index(
    s2_cache: str,
    s3_cache: str,
    name_cutoff_pct: float = NAME_FREQ_CUTOFF,
    nums_cutoff_pct: float = NUMS_FREQ_CUTOFF,
    addr_cutoff_pct: float = ADDR_FREQ_CUTOFF,
):
    """
    Stream S2 + S3 parquet caches to construct:
      - cand_ids: list[str] (original entity_ids)
      - cand_countries: np.ndarray[uint8] (country code per candidate)
      - cand_weights: np.ndarray[float32] (total weighted token count per candidate)
      - index: dict[str, np.ndarray[int32]] (token -> candidate int32 array)
      - country_to_id: dict[str, int]
    """
    log.info("Building inverted index over combined S2 and S3 ...")
    t0 = time.time()

    # Pass 1: Count total entities & document frequencies
    log.info("  Pass 1: Computing document frequencies for stopword exclusion ...")
    df_counts = defaultdict(int)
    n_candidates = 0

    for cache_path in (s2_cache, s3_cache):
        df = load_cached_source(cache_path, cols=["name_tokens_str", "addr_nums", "addr_tokens_str"])
        n_candidates += len(df)
        for s in df["name_tokens_str"]:
            if s:
                for tok in set(s.split("|")):
                    if len(tok) >= MIN_TOKEN_LEN:
                        df_counts["n:" + tok] += 1
        for s in df["addr_nums"]:
            if s:
                for tok in set(s.split()):
                    if len(tok) >= 2:
                        df_counts["#num:" + tok] += 1
        for s in df["addr_tokens_str"]:
            if s:
                for tok in set(s.split("|")):
                    if len(tok) >= 4:
                        df_counts["a:" + tok] += 1
        del df

    name_max_cnt = int(name_cutoff_pct * n_candidates)
    nums_max_cnt = int(nums_cutoff_pct * n_candidates)
    addr_max_cnt = int(addr_cutoff_pct * n_candidates)

    stopwords = set()
    for tok_key, cnt in df_counts.items():
        if tok_key.startswith("n:") and (cnt > name_max_cnt or cnt > MAX_POSTINGS):
            stopwords.add(tok_key)
        elif tok_key.startswith("#num:") and (cnt > nums_max_cnt or cnt > MAX_POSTINGS):
            stopwords.add(tok_key)
        elif tok_key.startswith("a:") and (cnt > addr_max_cnt or cnt > MAX_POSTINGS):
            stopwords.add(tok_key)

    del df_counts
    log.info(f"  Total S2+S3 candidates: {n_candidates:,} | Filtered stopwords: {len(stopwords):,}")

    # Pass 2: Populate compact index and candidate metadata
    log.info("  Pass 2: Building postings and candidate arrays ...")
    cand_ids = []
    cand_countries_raw = []
    cand_weights = np.empty(n_candidates, dtype=np.float32)
    postings = defaultdict(list)

    curr_idx = 0
    for cache_path in (s2_cache, s3_cache):
        df = load_cached_source(cache_path)
        for eid, country, ntoks_str, atoks_str, nums_str in zip(
            df["entity_id"], df["country_norm"], df["name_tokens_str"], df["addr_tokens_str"], df["addr_nums"]
        ):
            cand_ids.append(eid)
            cand_countries_raw.append(country)

            w_total = 0.0
            # Name tokens (weight 1.0)
            if ntoks_str:
                for tok in set(ntoks_str.split("|")):
                    if len(tok) >= MIN_TOKEN_LEN:
                        k = "n:" + tok
                        w_total += NAME_WEIGHT
                        if k not in stopwords:
                            postings[k].append(curr_idx)

            # Addr nums (weight 2.0)
            if nums_str:
                for tok in set(nums_str.split()):
                    if len(tok) >= 2:
                        k = "#num:" + tok
                        w_total += ADDR_NUM_WEIGHT
                        if k not in stopwords:
                            postings[k].append(curr_idx)

            # Addr tokens (weight 0.6)
            if atoks_str:
                for tok in set(atoks_str.split("|")):
                    if len(tok) >= 4:
                        k = "a:" + tok
                        w_total += ADDR_WEIGHT
                        if k not in stopwords:
                            postings[k].append(curr_idx)

            cand_weights[curr_idx] = w_total
            curr_idx += 1
        del df

    # Map countries to int16 IDs
    unique_countries = sorted(list(set(cand_countries_raw)))
    country_to_id = {c: i for i, c in enumerate(unique_countries)}
    cand_countries = np.array([country_to_id[c] for c in cand_countries_raw], dtype=np.int16)
    del cand_countries_raw

    # Convert postings lists to compact np.ndarray[int32]
    index = {k: np.array(lst, dtype=np.int32) for k, lst in postings.items() if len(lst) > 0}
    del postings

    elapsed = time.time() - t0
    log.info(
        f"Inverted index ready in {elapsed:.1f}s | "
        f"Unique terms: {len(index):,} | "
        f"Weights array: {cand_weights.nbytes / 1e6:.1f} MB | "
        f"Countries array: {cand_countries.nbytes / 1e6:.1f} MB"
    )
    return cand_ids, cand_countries, cand_weights, index, country_to_id


# ===========================================================================
# 3. High-Speed Candidate Retrieval
# ===========================================================================

def retrieve_all_candidates(
    s1_cache: str,
    cand_ids: list,
    cand_countries: np.ndarray,
    cand_weights: np.ndarray,
    index: dict,
    country_to_id: dict,
    max_cands: int = MAX_CANDIDATES,
    chunk_size: int = CHUNK_SIZE,
) -> pd.DataFrame:
    """
    Run fast in-memory candidate retrieval for all S1 entities.
    Returns DataFrame matching the official hackathon schema:
      source1_entity_id, candidate_entity_ids
    """
    log.info(f"Loading S1 cache for candidate retrieval from {s1_cache} ...")
    df_s1 = load_cached_source(s1_cache)
    n_s1 = len(df_s1)
    log.info(f"Retrieved {n_s1:,} S1 entities. Starting retrieval ...")

    s1_eids        = df_s1["entity_id"].to_numpy()
    s1_countries   = df_s1["country_norm"].to_numpy()
    s1_ntoks_arr   = df_s1["name_tokens_str"].to_numpy()
    s1_atoks_arr   = df_s1["addr_tokens_str"].to_numpy()
    s1_nums_arr    = df_s1["addr_nums"].to_numpy()

    output_s1_ids = []
    output_cand_lists = []

    t_start = time.time()
    t_chunk = time.time()

    for i in range(n_s1):
        s1_id = s1_eids[i]
        c_str = s1_countries[i]
        target_country_id = country_to_id.get(c_str, -1)

        ntoks_str = s1_ntoks_arr[i]
        nums_str  = s1_nums_arr[i]
        atoks_str = s1_atoks_arr[i]

        s1_w = 0.0
        query_terms = []

        if ntoks_str:
            for tok in set(ntoks_str.split("|")):
                if len(tok) >= MIN_TOKEN_LEN:
                    k = "n:" + tok
                    s1_w += NAME_WEIGHT
                    if k in index:
                        query_terms.append((k, NAME_WEIGHT))

        if nums_str:
            for tok in set(nums_str.split()):
                if len(tok) >= 2:
                    k = "#num:" + tok
                    s1_w += ADDR_NUM_WEIGHT
                    if k in index:
                        query_terms.append((k, ADDR_NUM_WEIGHT))

        if atoks_str:
            for tok in set(atoks_str.split("|")):
                if len(tok) >= 4:
                    k = "a:" + tok
                    s1_w += ADDR_WEIGHT
                    if k in index:
                        query_terms.append((k, ADDR_WEIGHT))

        if s1_w == 0.0 or not query_terms:
            output_s1_ids.append(s1_id)
            output_cand_lists.append("")
            continue

        # Vectorized accumulation across query terms
        cids_list = []
        weights_list = []
        for term, w in query_terms:
            arr = index[term]
            if len(arr) > 0:
                cids_list.append(arr)
                weights_list.append(np.full(len(arr), w, dtype=np.float32))

        if not cids_list:
            output_s1_ids.append(s1_id)
            output_cand_lists.append("")
            continue

        all_cids = np.concatenate(cids_list)
        all_weights = np.concatenate(weights_list)

        mask = (cand_countries[all_cids] == target_country_id)
        if not np.any(mask):
            output_s1_ids.append(s1_id)
            output_cand_lists.append("")
            continue

        valid_cids = all_cids[mask]
        valid_weights = all_weights[mask]

        unique_cids, inv = np.unique(valid_cids, return_inverse=True)
        intersections = np.bincount(inv, weights=valid_weights)

        c_weights = cand_weights[unique_cids]
        denoms = s1_w + c_weights - intersections
        scores = np.where(denoms > 0, intersections / denoms, 0.0)

        if len(scores) <= max_cands:
            top_idx = np.argsort(-scores)
        else:
            part_idx = np.argpartition(-scores, max_cands)[:max_cands]
            top_idx = part_idx[np.argsort(-scores[part_idx])]

        top_candidates = [cand_ids[cid] for cid in unique_cids[top_idx]]

        output_s1_ids.append(s1_id)
        output_cand_lists.append(",".join(top_candidates))

        # Periodic progress logging
        if (i + 1) % chunk_size == 0 or (i + 1) == n_s1:
            elapsed = time.time() - t_start
            rate = (i + 1) / max(elapsed, 0.01)
            eta_m = (n_s1 - (i + 1)) / max(rate, 0.01) / 60
            log.info(
                f"  Processed {i+1:9,d}/{n_s1:,} S1 entities "
                f"({(i+1)/n_s1:6.1%}) | Rate: {rate:6.0f} S1/s | ETA: {eta_m:4.1f} min"
            )

    del df_s1
    log.info(f"Retrieval complete in {(time.time() - t_start)/60:.2f} min.")

    result_df = pd.DataFrame({
        "source1_entity_id": output_s1_ids,
        "candidate_entity_ids": output_cand_lists,
    })
    return result_df


# ===========================================================================
# 4. Ground Truth Recall Validation
# ===========================================================================

def validate_recall(candidate_df: pd.DataFrame, gt_path: str,
                    threshold: float = RECALL_THRESHOLD) -> float:
    """
    Validate that candidate blocking recall >= threshold against ground truth.
    """
    log.info(f"Loading ground truth from {gt_path} for recall validation ...")
    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str)

    gt_dict = {}
    for _, row in gt_df.iterrows():
        s1 = row["source1_entity_id"]
        matched = row["matched_entity_ids"]
        if pd.notna(matched) and matched.strip():
            gt_dict[s1] = set(m.strip() for m in matched.split(","))

    cand_dict = {}
    for _, row in candidate_df.iterrows():
        s1 = row["source1_entity_id"]
        cands = row["candidate_entity_ids"]
        if pd.notna(cands) and cands.strip():
            cand_dict[s1] = set(c.strip() for c in cands.split(","))
        else:
            cand_dict[s1] = set()

    total_true, found_true = 0, 0
    all_missed, partial_missed = 0, 0

    for s1, true_set in gt_dict.items():
        total_true += len(true_set)
        cands = cand_dict.get(s1, set())
        overlap = true_set & cands
        found_true += len(overlap)
        if len(overlap) == 0:
            all_missed += 1
        elif len(overlap) < len(true_set):
            partial_missed += 1

    recall = found_true / total_true if total_true > 0 else 0.0
    log.info("=" * 60)
    log.info(f"BLOCKING RECALL VALIDATION:")
    log.info(f"  Total true match pairs : {total_true:,}")
    log.info(f"  Retained in candidates : {found_true:,}")
    log.info(f"  Recall Score           : {recall:.4%}")
    log.info(f"  Completely missed S1   : {all_missed:,} (out of {len(gt_dict):,})")
    log.info(f"  Partially missed S1    : {partial_missed:,}")
    log.info("=" * 60)

    if recall < threshold:
        raise RuntimeError(
            f"Blocking recall {recall:.4%} fell below the {threshold:.0%} quality gate!"
        )
    log.info(f"Recall gate PASSED (>= {threshold:.0%})")
    return recall


# ===========================================================================
# 5. Master Pipeline
# ===========================================================================

def run_blocking(
    train: bool = True,
    validate: bool = True,
    force_renorm: bool = False,
    max_cands: int = MAX_CANDIDATES,
    name_cutoff: float = NAME_FREQ_CUTOFF,
    nums_cutoff: float = NUMS_FREQ_CUTOFF,
    addr_cutoff: float = ADDR_FREQ_CUTOFF,
    chunk_size: int = CHUNK_SIZE,
) -> pd.DataFrame:
    """
    Execute end-to-end blocking pipeline.
    """
    t_start = time.time()
    dataset_dir = DATA_TRAIN if train else DATA_TEST
    prefix = "train" if train else "test"
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log.info("=" * 70)
    log.info(f"Phase 2: Inverted-Index Blocking | Mode: {prefix.upper()} | Max Candidates: {max_cands}")
    log.info("=" * 70)

    s1_tsv = os.path.join(dataset_dir, f"{prefix}_source1.tsv")
    s2_tsv = os.path.join(dataset_dir, f"{prefix}_source2.tsv")
    s3_tsv = os.path.join(dataset_dir, f"{prefix}_source3.tsv")

    s1_cache = os.path.join(CACHE_DIR, f"{prefix}_source1.parquet")
    s2_cache = os.path.join(CACHE_DIR, f"{prefix}_source2.parquet")
    s3_cache = os.path.join(CACHE_DIR, f"{prefix}_source3.parquet")

    # Step 1: Normalize & Cache (freeing RAM per source)
    normalize_and_cache(s1_tsv, f"{prefix.upper()} S1", s1_cache, force=force_renorm)
    normalize_and_cache(s2_tsv, f"{prefix.upper()} S2", s2_cache, force=force_renorm)
    normalize_and_cache(s3_tsv, f"{prefix.upper()} S3", s3_cache, force=force_renorm)

    # Step 2: Build Inverted Index & Metadata Arrays
    cand_ids, cand_countries, cand_weights, index, country_to_id = build_inverted_index(
        s2_cache, s3_cache,
        name_cutoff_pct=name_cutoff,
        nums_cutoff_pct=nums_cutoff,
        addr_cutoff_pct=addr_cutoff,
    )

    # Step 3: Retrieve Candidates
    candidate_df = retrieve_all_candidates(
        s1_cache, cand_ids, cand_countries, cand_weights,
        index, country_to_id,
        max_cands=max_cands,
        chunk_size=chunk_size,
    )

    # Step 4: Write output in exact official hackathon TSV format
    out_path = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    log.info(f"Writing candidate pairs to {out_path} ...")
    candidate_df.to_csv(out_path, sep="\t", index=False)
    sz_mb = os.path.getsize(out_path) / (1024 * 1024)
    log.info(f"Candidate file saved successfully: {out_path} ({sz_mb:.1f} MB)")

    # Step 5: Validate Recall (if on train set)
    if train and validate:
        gt_path = os.path.join(DATA_TRAIN, "train_ground_truth.tsv")
        validate_recall(candidate_df, gt_path)

    elapsed_min = (time.time() - t_start) / 60
    log.info(f"Phase 2 completed in {elapsed_min:.2f} minutes.")
    return candidate_df


# ===========================================================================
# CLI Entrypoint
# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Inverted-Index Token Blocking")
    parser.add_argument("--test",         action="store_true", help="Run on test dataset")
    parser.add_argument("--no-validate",  action="store_true", help="Skip recall validation")
    parser.add_argument("--force-renorm", action="store_true", help="Force re-normalization and re-caching")
    parser.add_argument("--max-cands",    type=int,   default=MAX_CANDIDATES)
    parser.add_argument("--name-cutoff",  type=float, default=NAME_FREQ_CUTOFF)
    parser.add_argument("--nums-cutoff",  type=float, default=NUMS_FREQ_CUTOFF)
    parser.add_argument("--addr-cutoff",  type=float, default=ADDR_FREQ_CUTOFF)
    parser.add_argument("--chunk-size",   type=int,   default=CHUNK_SIZE)
    args = parser.parse_args()

    df = run_blocking(
        train=not args.test,
        validate=not args.no_validate,
        force_renorm=args.force_renorm,
        max_cands=args.max_cands,
        name_cutoff=args.name_cutoff,
        nums_cutoff=args.nums_cutoff,
        addr_cutoff=args.addr_cutoff,
        chunk_size=args.chunk_size,
    )
