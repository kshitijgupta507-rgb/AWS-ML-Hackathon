"""
predict.py  --  Phase 4: Match Prediction & Submission Generator.

Loads the trained LightGBM model bundle, scores candidate pairs, filters
matches using the optimal F_0.5 decision threshold, and generates the
official competition submission file `matching_results.tsv`.

Key Steps:
  1. Load model bundle from output/model.joblib (model, threshold, feature names).
  2. Load candidate pairs and normalized features in memory-safe chunks.
  3. Predict match probabilities using LightGBM.
  4. Filter candidates with score >= threshold, handling singletons gracefully.
  5. Save to output/matching_results.tsv.
  6. Automatically validate output with utils/validate_submission.py.
"""

import os
import sys
import time
import logging
import argparse
import subprocess
import joblib

import numpy as np
import pandas as pd

SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)
DATA_TRAIN  = os.path.join(PROJECT_DIR, "dataset", "train")
DATA_TEST   = os.path.join(PROJECT_DIR, "dataset", "test")
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "output")
CACHE_DIR   = os.path.join(OUTPUT_DIR, ".cache")

sys.path.insert(0, SRC_DIR)
from features import (
    load_feature_source, compute_features, normalize_and_cache_features,
    FEATURE_COLS, CHUNK_SIZE
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def generate_matches(
    pairs_path: str,
    model_path: str,
    s1_cache_path: str,
    s2_cache_path: str,
    s3_cache_path: str,
    output_path: str,
    threshold: float = None,
    chunk_size: int = 10_000,
    top_k: int = None,
    limit: int = None,
    fill_remaining: bool = False,
) -> str:
    """
    Score candidate pairs with trained model and write matching_results.tsv.
    """
    t_start = time.time()
    log.info(f"Loading model bundle from {model_path} ...")
    bundle = joblib.load(model_path)
    model = bundle["model"]
    decision_threshold = threshold if threshold is not None else bundle["best_threshold"]
    log.info(f"Using decision threshold: {decision_threshold:.3f}")

    # Load normalized source features
    log.info("Loading normalized feature sources ...")
    s1_df = load_feature_source(s1_cache_path)
    s2_df = load_feature_source(s2_cache_path)
    s3_df = load_feature_source(s3_cache_path)
    cand_df = pd.concat([s2_df, s3_df], ignore_index=True)
    del s2_df, s3_df
    log.info(f"Sources loaded: S1={len(s1_df):,}, Cand={len(cand_df):,}")

    # Stream candidate pairs in chunks and write directly to disk
    log.info(f"Opening output file {output_path} for streaming predictions ...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    n_processed_s1 = 0
    total_matches = 0
    n_matched_s1 = 0

    with open(output_path, "w", encoding="utf-8", buffering=1024*1024) as out_f:
        out_f.write("source1_entity_id\tmatched_entity_ids\n")

        # Read candidate_pairs.tsv in memory-safe chunks
        for chunk_df in pd.read_csv(pairs_path, sep="\t", dtype=str, chunksize=chunk_size):
            if limit is not None and limit > 0 and n_processed_s1 >= limit:
                if fill_remaining:
                    # Write remaining entities as singletons (empty match) to ensure full submission validity
                    for s1_id in chunk_df["source1_entity_id"]:
                        out_f.write(f"{s1_id}\t\n")
                    n_processed_s1 += len(chunk_df)
                    continue
                else:
                    break

            if top_k is not None and top_k > 0:
                def _trim(c):
                    if not c or pd.isna(c):
                        return ""
                    parts = c.split(",")
                    return ",".join(parts[:top_k]) if len(parts) > top_k else c
                chunk_df["candidate_entity_ids"] = chunk_df["candidate_entity_ids"].apply(_trim)

            chunk_s1_ids = chunk_df["source1_entity_id"].tolist()
            chunk_features = compute_features(chunk_df, s1_df, cand_df, chunk_size=chunk_size)

            passed_map = {}
            if len(chunk_features) > 0:
                X = chunk_features[FEATURE_COLS]
                probs = model.predict_proba(X)[:, 1]
                chunk_features["prob"] = probs

                passed = chunk_features[chunk_features["prob"] >= decision_threshold]
                for s1_id, group in passed.groupby("source1_entity_id"):
                    passed_map[s1_id] = list(group["candidate_entity_id"].unique())

            # Write rows for this chunk immediately
            for s1_id in chunk_s1_ids:
                cands = passed_map.get(s1_id, [])
                if cands:
                    n_matched_s1 += 1
                    total_matches += len(cands)
                    out_f.write(f"{s1_id}\t{','.join(cands)}\n")
                else:
                    out_f.write(f"{s1_id}\t\n")

            out_f.flush()
            n_processed_s1 += len(chunk_s1_ids)
            log.info(
                f"  Processed S1: {n_processed_s1:,} | "
                f"Matches found so far: {total_matches:,} ({n_matched_s1:,} entities with matches)"
            )

    del s1_df, cand_df

    n_singletons = n_processed_s1 - n_matched_s1
    log.info(f"Predictions saved: {n_processed_s1:,} total S1 entities")
    log.info(f"  Entities with matches: {n_matched_s1:,} ({n_matched_s1/max(n_processed_s1,1):.2%})")
    log.info(f"  Predicted singletons:  {n_singletons:,} ({n_singletons/max(n_processed_s1,1):.2%})")
    log.info(f"Matching finished in {(time.time()-t_start)/60:.2f} minutes.")
    return output_path


def validate_results(matching_path: str, candidate_path: str, test_dir: str):
    """Run official validation script on the output."""
    val_script = os.path.join(PROJECT_DIR, "utils", "validate_submission.py")
    if not os.path.exists(val_script):
        log.warning(f"Validator script not found at {val_script}")
        return

    cmd = [
        sys.executable, val_script,
        "--matching", matching_path,
        "--test-dir", test_dir,
    ]
    log.info(f"Running submission validation: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        log.info(f"Validation Output:\n{proc.stdout}")
        if proc.stderr:
            log.warning(f"Validation Stderr:\n{proc.stderr}")
    except Exception as e:
        log.error(f"Error running validator: {e}")


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Predict Matches & Generate Submission")
    parser.add_argument("--test", action="store_true", help="Run on test dataset (default: train)")
    parser.add_argument("--model", type=str, default=os.path.join(OUTPUT_DIR, "model.joblib"),
                        help="Path to trained model bundle")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override model decision threshold")
    parser.add_argument("--chunk-size", type=int, default=10_000,
                        help="Processing chunk size (default: 10,000)")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Cap candidates per S1 entity to top K highest-scoring items")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit model scoring to first N S1 entities (e.g. 430,000)")
    parser.add_argument("--fill-remaining", action="store_true",
                        help="Fill all remaining S1 entities past limit as singletons to pass validator")
    parser.add_argument("--out", type=str, default=os.path.join(OUTPUT_DIR, "matching_results.tsv"),
                        help="Output path for matching_results.tsv")
    args = parser.parse_args()

    prefix = "test" if args.test else "train"
    dataset_dir = DATA_TEST if args.test else DATA_TRAIN

    s1_cache = os.path.join(CACHE_DIR, f"{prefix}_source1_features.parquet")
    s2_cache = os.path.join(CACHE_DIR, f"{prefix}_source2_features.parquet")
    s3_cache = os.path.join(CACHE_DIR, f"{prefix}_source3_features.parquet")
    pairs_tsv = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

    s1_tsv = os.path.join(dataset_dir, f"{prefix}_source1.tsv")
    s2_tsv = os.path.join(dataset_dir, f"{prefix}_source2.tsv")
    s3_tsv = os.path.join(dataset_dir, f"{prefix}_source3.tsv")

    normalize_and_cache_features(s1_tsv, f"{prefix.upper()} S1", s1_cache)
    normalize_and_cache_features(s2_tsv, f"{prefix.upper()} S2", s2_cache)
    normalize_and_cache_features(s3_tsv, f"{prefix.upper()} S3", s3_cache)

    generate_matches(
        pairs_path=pairs_tsv,
        model_path=args.model,
        s1_cache_path=s1_cache,
        s2_cache_path=s2_cache,
        s3_cache_path=s3_cache,
        output_path=args.out,
        threshold=args.threshold,
        chunk_size=args.chunk_size,
        top_k=args.top_k,
        limit=args.limit,
        fill_remaining=args.fill_remaining,
    )

    validate_results(
        matching_path=args.out,
        candidate_path=pairs_tsv,
        test_dir=dataset_dir,
    )


if __name__ == "__main__":
    main()
