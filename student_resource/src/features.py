"""
features.py  --  Phase 3: Feature Engineering for Entity Resolution.

Goal: For each candidate pair in candidate_pairs.tsv, compute rich similarity
features for downstream model training (Phase 4).

Features computed (10 total):
  name_jaccard          : Jaccard over name_tokens sets
  name_jaro_winkler     : RapidFuzz Jaro-Winkler on name_norm strings
  name_token_sort_ratio : RapidFuzz token sort ratio (order-invariant name match)
  addr_jaccard          : Jaccard over addr_tokens sets
  addr_jaro_winkler     : Jaro-Winkler on addr_norm strings
  addr_num_exact        : bool — any addr_nums tokens overlap (house#, ZIP)
  country_match         : bool — country_norm equals (always 1 after Phase 2 filter)
  is_addr_missing_s1    : bool passthrough from Phase 1
  is_addr_missing_cand  : bool passthrough from Phase 1
  jaccard_score         : passthrough from Phase 2 (blocking score, recomputed
                          using the same weighted-Jaccard formula from blocking.py)

Label: 1 if candidate_id appears in ground_truth for that source1_id, else 0.

Design principles:
  - Memory-safe chunked processing: pairs are exploded and merged in S1-entity
    chunks (default 10k S1 entities per chunk ≈ up to 1M pairs), avoiding the
    need to hold all ~220M exploded pairs in memory simultaneously.
  - Feature-specific parquet cache: stores ALL columns needed for feature
    engineering (name_norm, addr_norm, is_addr_missing — not in blocking cache).
  - np.frompyfunc vectorized wrappers for per-pair computations (lower overhead
    than pd.Series.apply).
  - Decoupled from pair generation: reads candidate_pairs.tsv produced by the
    blocking/pair-construction stage; does NOT decide which pairs to create.

Entry point:
  run_features(train=True) -> DataFrame with features + labels
  CLI: py features.py [--test] [--force-renorm] [--chunk-size N]
"""

import os
import sys
import time
import logging
import argparse

import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler
from rapidfuzz import fuzz

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
# Weights (matching blocking.py for jaccard_score recomputation)
# ---------------------------------------------------------------------------
NAME_WEIGHT     = 1.0
ADDR_WEIGHT     = 0.6
ADDR_NUM_WEIGHT = 2.0
MIN_TOKEN_LEN   = 2

# Processing chunk size: number of S1 entities processed per chunk.
# At MAX_CANDIDATES=100, 10k S1 entities ≈ up to 1M pairs per chunk.
CHUNK_SIZE = 10_000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ===========================================================================
# 1. Normalization & Caching (feature-specific, ALL columns)
# ===========================================================================

def normalize_and_cache_features(tsv_path: str, label: str, cache_path: str,
                                  force: bool = False) -> str:
    """
    Normalize source TSV and write columnar cache to parquet.

    This cache is a superset of blocking.py's cache — it includes the string
    columns (name_norm, addr_norm) and the is_addr_missing flag that blocking
    does not persist but that feature engineering needs.

    Cached columns:
      entity_id, name_norm, name_tokens_str, addr_norm, addr_tokens_str,
      addr_nums, country_norm, is_addr_missing

    Skips if cache already exists (unless force=True).
    """
    if os.path.exists(cache_path) and not force:
        sz = os.path.getsize(cache_path) / (1024 * 1024)
        log.info(f"  Feature cache hit for {label}: {cache_path} ({sz:.1f} MB)")
        return cache_path

    log.info(f"Normalizing {label} from {tsv_path} ...")
    t0 = time.time()
    df = pd.read_csv(tsv_path, sep="\t", dtype=str, keep_default_na=True)
    log.info(f"  Loaded {len(df):,} rows in {time.time()-t0:.2f}s")

    t1 = time.time()
    df = normalize_sources(df)
    log.info(f"  Normalized in {time.time()-t1:.2f}s")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    # Convert list columns to pipe-separated strings for clean parquet schema
    df["name_tokens_str"] = df["name_tokens"].apply(
        lambda l: "|".join(l) if isinstance(l, list) else ""
    )
    df["addr_tokens_str"] = df["addr_tokens"].apply(
        lambda l: "|".join(l) if isinstance(l, list) else ""
    )
    df["addr_nums"]    = df["addr_nums"].fillna("")
    df["country_norm"] = df["country_norm"].fillna("")
    df["name_norm"]    = df["name_norm"].fillna("")
    df["addr_norm"]    = df["addr_norm"].fillna("")

    save_cols = [
        "entity_id", "name_norm", "name_tokens_str",
        "addr_norm", "addr_tokens_str", "addr_nums",
        "country_norm", "is_addr_missing",
    ]
    df[save_cols].to_parquet(cache_path, index=False, compression="snappy")
    sz = os.path.getsize(cache_path) / (1024 * 1024)
    log.info(f"  Saved {label} feature cache -> {cache_path} ({sz:.1f} MB)")

    del df
    return cache_path


def load_feature_source(cache_path: str) -> pd.DataFrame:
    """Load normalized source from features parquet cache."""
    df = pd.read_parquet(cache_path)
    # Ensure entity_id is a column (not index)
    if "entity_id" not in df.columns and (
        df.index.name == "entity_id" or "entity_id" in df.index.names
    ):
        df = df.reset_index()
    # Fill NaNs for safe string operations downstream
    for col in ["name_norm", "name_tokens_str", "addr_norm",
                "addr_tokens_str", "addr_nums", "country_norm"]:
        if col in df.columns:
            df[col] = df[col].fillna("")
    if "is_addr_missing" in df.columns:
        df["is_addr_missing"] = df["is_addr_missing"].fillna(False).astype(bool)
    return df


# ===========================================================================
# 2. Candidate Pair Loading
# ===========================================================================

def load_candidate_pairs(pairs_path: str) -> pd.DataFrame:
    """
    Read candidate_pairs.tsv (non-exploded, one row per S1 entity).

    Input format (from Phase 2 blocking):
      source1_entity_id\\tcandidate_entity_ids
      S1-00001\\tS2-12345,S3-67890,...

    Returns the raw DataFrame — pairs are NOT exploded here; they are
    exploded per chunk inside compute_features() to keep memory bounded.
    """
    log.info(f"Loading candidate pairs from {pairs_path} ...")
    df = pd.read_csv(pairs_path, sep="\t", dtype=str)
    n_total = len(df)
    n_with_cands = df["candidate_entity_ids"].notna().sum()
    n_with_cands -= (df["candidate_entity_ids"].fillna("").str.strip() == "").sum()
    log.info(f"  {n_total:,} S1 entities ({n_with_cands:,} with candidates)")
    return df


# ===========================================================================
# 3. Feature Computation Primitives
# ===========================================================================

def _jaccard_from_str(s1_str: str, cand_str: str, sep: str = "|") -> float:
    """Jaccard similarity from separator-delimited token strings."""
    s1_set = set(s1_str.split(sep)) if s1_str else set()
    cand_set = set(cand_str.split(sep)) if cand_str else set()
    s1_set.discard("")
    cand_set.discard("")
    union_sz = len(s1_set | cand_set)
    if union_sz == 0:
        return 0.0
    return len(s1_set & cand_set) / union_sz


def _addr_num_exact(s1_nums: str, cand_nums: str) -> int:
    """1 if any addr_nums tokens overlap (house #, ZIP, PIN), else 0."""
    if not s1_nums or not cand_nums:
        return 0
    s1_set = set(s1_nums.split())
    cand_set = set(cand_nums.split())
    s1_set.discard("")
    cand_set.discard("")
    return int(bool(s1_set & cand_set))


def _weighted_jaccard(
    nt_s1: str, nt_cand: str,
    at_s1: str, at_cand: str,
    an_s1: str, an_cand: str,
) -> float:
    """
    Recompute the blocking-phase weighted Jaccard score.

    Uses the same token-type weights and length filters as blocking.py:
      name tokens  (weight 1.0, min length 2)
      addr nums    (weight 2.0, min length 2)
      addr tokens  (weight 0.6, min length 4)

    Formula: weighted_intersection / (s1_weight + cand_weight - weighted_intersection)

    Note: blocking.py excludes high-frequency stopword tokens from the inverted
    index, so its intersection only counts non-stopword shared tokens while the
    denominator counts all tokens.  This recomputation counts ALL shared tokens
    in the intersection (no frequency-based exclusion), producing a "clean"
    weighted Jaccard.  The ML model in Phase 4 learns the appropriate weight
    regardless of this minor difference.
    """
    # Name tokens (weight 1.0, min length 2 — already guaranteed by normalize)
    ns1 = set(t for t in nt_s1.split("|") if t and len(t) >= MIN_TOKEN_LEN) if nt_s1 else set()
    nc  = set(t for t in nt_cand.split("|") if t and len(t) >= MIN_TOKEN_LEN) if nt_cand else set()

    # Addr tokens (weight 0.6, min length 4 — matching blocking.py)
    as1 = set(t for t in at_s1.split("|") if t and len(t) >= 4) if at_s1 else set()
    ac  = set(t for t in at_cand.split("|") if t and len(t) >= 4) if at_cand else set()

    # Addr nums (weight 2.0, min length 2)
    an1 = set(t for t in an_s1.split() if t and len(t) >= 2) if an_s1 else set()
    anc = set(t for t in an_cand.split() if t and len(t) >= 2) if an_cand else set()

    s1_w = len(ns1) * NAME_WEIGHT + len(an1) * ADDR_NUM_WEIGHT + len(as1) * ADDR_WEIGHT
    c_w  = len(nc)  * NAME_WEIGHT + len(anc) * ADDR_NUM_WEIGHT + len(ac)  * ADDR_WEIGHT

    isect_w = (
        len(ns1 & nc)  * NAME_WEIGHT
      + len(an1 & anc) * ADDR_NUM_WEIGHT
      + len(as1 & ac)  * ADDR_WEIGHT
    )

    denom = s1_w + c_w - isect_w
    return isect_w / denom if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Vectorized wrappers (np.frompyfunc returns ufunc objects — lower overhead
# than pd.Series.apply for element-wise Python functions).
# ---------------------------------------------------------------------------
_vec_jaccard = np.frompyfunc(_jaccard_from_str, 2, 1)
_vec_addr_num = np.frompyfunc(_addr_num_exact, 2, 1)
_vec_jaro_winkler = np.frompyfunc(
    lambda a, b: JaroWinkler.similarity(a, b) if a and b else 0.0, 2, 1
)
_vec_token_sort = np.frompyfunc(
    lambda a, b: fuzz.token_sort_ratio(a, b) / 100.0 if a and b else 0.0, 2, 1
)
_vec_weighted_jacc = np.frompyfunc(_weighted_jaccard, 6, 1)


# ===========================================================================
# 4. Feature Computation (chunked by S1 entities)
# ===========================================================================

# Columns needed from normalized sources for feature computation
_NEEDED_COLS = [
    "entity_id", "name_norm", "name_tokens_str",
    "addr_norm", "addr_tokens_str", "addr_nums",
    "country_norm", "is_addr_missing",
]

# Output feature columns (in order)
FEATURE_COLS = [
    "name_jaccard", "name_jaro_winkler", "name_token_sort_ratio",
    "addr_jaccard", "addr_jaro_winkler", "addr_num_exact",
    "country_match", "is_addr_missing_s1", "is_addr_missing_cand",
    "jaccard_score",
]


def _compute_chunk_features(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 10 features on a merged chunk.

    Expects columns: source1_entity_id, candidate_entity_id, plus
    S1-suffixed and cand-suffixed normalized columns from the merge step.

    Returns the same rows with only ID columns + 10 feature columns.
    """
    # Fill NaN after merge for safety
    str_cols = ["nn_s1", "nt_s1", "an_s1", "at_s1", "anum_s1", "cn_s1",
                "nn_c",  "nt_c",  "an_c",  "at_c",  "anum_c",  "cn_c"]
    for col in str_cols:
        chunk[col] = chunk[col].fillna("")
    chunk["is_addr_missing_s1"]   = chunk["is_addr_missing_s1"].fillna(False).astype(int)
    chunk["is_addr_missing_cand"] = chunk["is_addr_missing_cand"].fillna(False).astype(int)

    # --- 1. name_jaccard: Jaccard over name_tokens sets ---
    chunk["name_jaccard"] = _vec_jaccard(
        chunk["nt_s1"].values, chunk["nt_c"].values
    ).astype(np.float32)

    # --- 2. name_jaro_winkler: RapidFuzz Jaro-Winkler on name_norm strings ---
    chunk["name_jaro_winkler"] = _vec_jaro_winkler(
        chunk["nn_s1"].values, chunk["nn_c"].values
    ).astype(np.float32)

    # --- 3. name_token_sort_ratio: RapidFuzz token sort ratio (order-invariant) ---
    chunk["name_token_sort_ratio"] = _vec_token_sort(
        chunk["nn_s1"].values, chunk["nn_c"].values
    ).astype(np.float32)

    # --- 4. addr_jaccard: Jaccard over addr_tokens sets ---
    chunk["addr_jaccard"] = _vec_jaccard(
        chunk["at_s1"].values, chunk["at_c"].values
    ).astype(np.float32)

    # --- 5. addr_jaro_winkler: Jaro-Winkler on addr_norm strings ---
    chunk["addr_jaro_winkler"] = _vec_jaro_winkler(
        chunk["an_s1"].values, chunk["an_c"].values
    ).astype(np.float32)

    # --- 6. addr_num_exact: bool — any addr_nums tokens overlap ---
    chunk["addr_num_exact"] = _vec_addr_num(
        chunk["anum_s1"].values, chunk["anum_c"].values
    ).astype(np.int8)

    # --- 7. country_match: bool — country_norm equals ---
    chunk["country_match"] = (chunk["cn_s1"] == chunk["cn_c"]).astype(np.int8)

    # --- 8 & 9: is_addr_missing_s1, is_addr_missing_cand ---
    # Already present as renamed columns from the merge step.

    # --- 10. jaccard_score: blocking-phase weighted Jaccard (recomputed) ---
    chunk["jaccard_score"] = _vec_weighted_jacc(
        chunk["nt_s1"].values,   chunk["nt_c"].values,
        chunk["at_s1"].values,   chunk["at_c"].values,
        chunk["anum_s1"].values, chunk["anum_c"].values,
    ).astype(np.float32)

    # Keep only IDs + features
    keep = ["source1_entity_id", "candidate_entity_id"] + FEATURE_COLS
    return chunk[keep].copy()


def compute_features(
    pairs_raw: pd.DataFrame,
    s1_df: pd.DataFrame,
    cand_df: pd.DataFrame,
    chunk_size: int = CHUNK_SIZE,
) -> pd.DataFrame:
    """
    Compute all 10 features for every candidate pair.

    Processes the non-exploded pairs_raw in S1-entity chunks to bound memory:
    each chunk explodes at most chunk_size S1 entities (up to 100 candidates
    each), merges with source data, computes features, and discards the
    intermediate merged columns before moving to the next chunk.

    Parameters
    ----------
    pairs_raw : DataFrame
        Non-exploded candidate_pairs.tsv with columns
        [source1_entity_id, candidate_entity_ids].
    s1_df : DataFrame
        Normalized Source 1 (from feature cache).
    cand_df : DataFrame
        Normalized combined S2+S3 (from feature cache).
    chunk_size : int
        Number of S1 entities per processing chunk (default 10,000).

    Returns
    -------
    DataFrame with columns:
        source1_entity_id, candidate_entity_id, + 10 feature columns.
    """
    n_s1 = len(pairs_raw)
    log.info(f"Computing features for {n_s1:,} S1 entities "
             f"(chunk_size={chunk_size:,}) ...")
    t_start = time.time()

    # Prepare sub-DataFrames for merge (only needed columns)
    s1_sub   = s1_df[_NEEDED_COLS].copy()
    cand_sub = cand_df[_NEEDED_COLS].copy()

    all_chunks = []
    total_pairs = 0

    for start in range(0, n_s1, chunk_size):
        end = min(start + chunk_size, n_s1)
        raw_chunk = pairs_raw.iloc[start:end]

        # --- Explode this chunk's comma-separated candidate IDs ---
        chunk = raw_chunk[["source1_entity_id", "candidate_entity_ids"]].copy()
        mask = (
            chunk["candidate_entity_ids"].notna()
            & (chunk["candidate_entity_ids"].str.strip() != "")
        )
        chunk = chunk[mask].copy()

        if len(chunk) == 0:
            continue

        chunk["candidate_entity_id"] = chunk["candidate_entity_ids"].str.split(",")
        chunk = chunk.explode("candidate_entity_id")
        chunk["candidate_entity_id"] = chunk["candidate_entity_id"].str.strip()
        chunk = chunk[["source1_entity_id", "candidate_entity_id"]].reset_index(drop=True)
        n_chunk_pairs = len(chunk)

        if n_chunk_pairs == 0:
            continue

        # --- Merge S1 data ---
        chunk = chunk.merge(
            s1_sub, left_on="source1_entity_id", right_on="entity_id", how="left"
        ).drop(columns=["entity_id"])
        chunk.rename(columns={
            "name_norm": "nn_s1",       "name_tokens_str": "nt_s1",
            "addr_norm": "an_s1",       "addr_tokens_str": "at_s1",
            "addr_nums": "anum_s1",     "country_norm": "cn_s1",
            "is_addr_missing": "is_addr_missing_s1",
        }, inplace=True)

        # --- Merge candidate (S2+S3) data ---
        chunk = chunk.merge(
            cand_sub, left_on="candidate_entity_id", right_on="entity_id", how="left"
        ).drop(columns=["entity_id"])
        chunk.rename(columns={
            "name_norm": "nn_c",        "name_tokens_str": "nt_c",
            "addr_norm": "an_c",        "addr_tokens_str": "at_c",
            "addr_nums": "anum_c",      "country_norm": "cn_c",
            "is_addr_missing": "is_addr_missing_cand",
        }, inplace=True)

        # --- Compute features for this chunk ---
        features_chunk = _compute_chunk_features(chunk)
        all_chunks.append(features_chunk)
        total_pairs += n_chunk_pairs

        # --- Progress logging ---
        elapsed = time.time() - t_start
        rate = total_pairs / max(elapsed, 0.01)
        pct = end / n_s1
        eta_m = (n_s1 - end) / max(end / max(elapsed, 0.01), 0.01) / 60
        log.info(
            f"  S1 {end:,}/{n_s1:,} ({pct:.1%}) | "
            f"Pairs: {total_pairs:,} | "
            f"Rate: {rate:,.0f} pairs/s | ETA: {eta_m:.1f} min"
        )

    if not all_chunks:
        log.warning("No candidate pairs found — returning empty DataFrame.")
        cols = ["source1_entity_id", "candidate_entity_id"] + FEATURE_COLS
        return pd.DataFrame(columns=cols)

    features_df = pd.concat(all_chunks, ignore_index=True)
    log.info(
        f"Feature computation complete in {(time.time()-t_start)/60:.2f} min | "
        f"Total pairs: {len(features_df):,} | Shape: {features_df.shape}"
    )
    return features_df


# ===========================================================================
# 5. Label Attachment
# ===========================================================================

def attach_labels(features_df: pd.DataFrame, gt_path: str) -> pd.DataFrame:
    """
    Attach binary labels from ground truth.

    Label = 1  if candidate_entity_id appears in matched_entity_ids for that
               source1_entity_id in train_ground_truth.tsv.
    Label = 0  otherwise.
    """
    log.info(f"Attaching labels from {gt_path} ...")
    t0 = time.time()
    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str)

    # Filter to S1 entities present in features_df for efficiency
    unique_s1 = set(features_df["source1_entity_id"].unique())
    gt_sub = gt_df[gt_df["source1_entity_id"].isin(unique_s1)].dropna(subset=["matched_entity_ids"])
    del gt_df

    # Explode ground truth matched pairs for vectorized merge
    gt_sub = gt_sub[gt_sub["matched_entity_ids"].str.strip() != ""].copy()
    gt_sub["candidate_entity_id"] = gt_sub["matched_entity_ids"].str.split(",")
    gt_pairs = gt_sub.explode("candidate_entity_id")[["source1_entity_id", "candidate_entity_id"]]
    gt_pairs["candidate_entity_id"] = gt_pairs["candidate_entity_id"].str.strip()
    gt_pairs["label"] = np.int8(1)
    gt_pairs.drop_duplicates(subset=["source1_entity_id", "candidate_entity_id"], inplace=True)

    # Vectorized left merge
    features_df = features_df.merge(
        gt_pairs,
        on=["source1_entity_id", "candidate_entity_id"],
        how="left"
    )
    features_df["label"] = features_df["label"].fillna(0).astype(np.int8)

    n_pos = int(features_df["label"].sum())
    n_total = len(features_df)
    log.info(
        f"  Labels attached in {time.time()-t0:.2f}s: {n_pos:,} positive / {n_total:,} total "
        f"({n_pos/max(n_total,1):.4%} match rate)"
    )
    return features_df


# ===========================================================================
# 6. Master Pipeline
# ===========================================================================

def run_features(
    train: bool = True,
    force_renorm: bool = False,
    chunk_size: int = CHUNK_SIZE,
) -> pd.DataFrame:
    """
    Execute end-to-end Phase 3 feature engineering pipeline.

    Steps:
      1. Normalize & cache all 3 sources (feature-specific cache with ALL columns)
      2. Load candidate_pairs.tsv (non-exploded)
      3. Load normalized sources into memory
      4. Compute 10 features for every pair (chunked by S1 entity)
      5. Attach labels (train only)
      6. Save to output/features_{train|test}.parquet
    """
    t_start = time.time()
    dataset_dir = DATA_TRAIN if train else DATA_TEST
    prefix = "train" if train else "test"
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log.info("=" * 70)
    log.info(f"Phase 3: Feature Engineering | Mode: {prefix.upper()} | "
             f"Chunk: {chunk_size:,} S1 entities")
    log.info("=" * 70)

    # --- Step 1: Normalize & Cache ---
    s1_tsv = os.path.join(dataset_dir, f"{prefix}_source1.tsv")
    s2_tsv = os.path.join(dataset_dir, f"{prefix}_source2.tsv")
    s3_tsv = os.path.join(dataset_dir, f"{prefix}_source3.tsv")

    # Feature caches use a separate filename to avoid collisions with
    # blocking.py's caches (which store a subset of columns).
    s1_fcache = os.path.join(CACHE_DIR, f"{prefix}_source1_features.parquet")
    s2_fcache = os.path.join(CACHE_DIR, f"{prefix}_source2_features.parquet")
    s3_fcache = os.path.join(CACHE_DIR, f"{prefix}_source3_features.parquet")

    normalize_and_cache_features(s1_tsv, f"{prefix.upper()} S1", s1_fcache, force=force_renorm)
    normalize_and_cache_features(s2_tsv, f"{prefix.upper()} S2", s2_fcache, force=force_renorm)
    normalize_and_cache_features(s3_tsv, f"{prefix.upper()} S3", s3_fcache, force=force_renorm)

    # --- Step 2: Load candidate pairs (non-exploded) ---
    pairs_path = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    pairs_raw = load_candidate_pairs(pairs_path)

    # --- Step 3: Load normalized sources ---
    log.info("Loading normalized sources ...")
    s1_df = load_feature_source(s1_fcache)
    log.info(f"  S1: {len(s1_df):,} rows")

    s2_df = load_feature_source(s2_fcache)
    s3_df = load_feature_source(s3_fcache)
    cand_df = pd.concat([s2_df, s3_df], ignore_index=True)
    del s2_df, s3_df
    log.info(f"  S2+S3 combined: {len(cand_df):,} rows")

    # --- Step 4: Compute features ---
    features_df = compute_features(pairs_raw, s1_df, cand_df, chunk_size=chunk_size)
    del s1_df, cand_df, pairs_raw

    # --- Step 5: Attach labels (train only) ---
    if train:
        gt_path = os.path.join(DATA_TRAIN, "train_ground_truth.tsv")
        features_df = attach_labels(features_df, gt_path)

    # --- Step 6: Save output ---
    out_path = os.path.join(OUTPUT_DIR, f"features_{prefix}.parquet")
    features_df.to_parquet(out_path, index=False, compression="snappy")
    sz_mb = os.path.getsize(out_path) / (1024 * 1024)
    log.info(f"Features saved to {out_path} ({sz_mb:.1f} MB)")

    elapsed_min = (time.time() - t_start) / 60
    log.info(f"Phase 3 completed in {elapsed_min:.2f} minutes.")
    return features_df


# ===========================================================================
# CLI Entrypoint
# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 3: Feature Engineering for Entity Resolution"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run on test dataset (default: train)"
    )
    parser.add_argument(
        "--force-renorm", action="store_true",
        help="Force re-normalization and re-caching of all sources"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=CHUNK_SIZE,
        help=f"S1 entities per processing chunk (default: {CHUNK_SIZE:,})"
    )
    args = parser.parse_args()

    df = run_features(
        train=not args.test,
        force_renorm=args.force_renorm,
        chunk_size=args.chunk_size,
    )
