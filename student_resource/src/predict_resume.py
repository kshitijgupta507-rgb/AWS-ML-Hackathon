"""
predict_resume.py  --  Resume streaming match prediction from where we left off.

Reads the existing matching_results.tsv to find which S1 entities are already
scored, then continues processing the remaining entities from candidate_pairs.tsv
and appends new predictions to the file.

Usage:
    python src/predict_resume.py --test --chunk-size 10000
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


def resume_predictions(
    pairs_path: str,
    model_path: str,
    s1_cache_path: str,
    s2_cache_path: str,
    s3_cache_path: str,
    output_path: str,
    threshold: float = None,
    chunk_size: int = 10_000,
):
    """
    Resume scoring candidate pairs. Reads existing output to find already-done
    S1 IDs, skips them, and appends remaining predictions.
    """
    t_start = time.time()

    # --- Load model ---
    log.info(f"Loading model bundle from {model_path} ...")
    bundle = joblib.load(model_path)
    model = bundle["model"]
    decision_threshold = threshold if threshold is not None else bundle["best_threshold"]
    log.info(f"Using decision threshold: {decision_threshold:.3f}")

    # --- Figure out which S1 IDs are already done ---
    done_s1_ids = set()
    if os.path.exists(output_path):
        log.info(f"Reading existing results from {output_path} to find resume point ...")
        for chunk in pd.read_csv(output_path, sep="\t", dtype=str, chunksize=50_000, usecols=["source1_entity_id"]):
            done_s1_ids.update(chunk["source1_entity_id"].tolist())
        log.info(f"Already completed: {len(done_s1_ids):,} S1 entities")
    else:
        log.info("No existing output file found — starting from scratch.")

    # --- Load normalized source features ---
    log.info("Loading normalized feature sources ...")
    s1_df = load_feature_source(s1_cache_path)
    s2_df = load_feature_source(s2_cache_path)
    s3_df = load_feature_source(s3_cache_path)
    cand_df = pd.concat([s2_df, s3_df], ignore_index=True)
    del s2_df, s3_df
    log.info(f"Sources loaded: S1={len(s1_df):,}, Cand={len(cand_df):,}")

    # --- Determine write mode ---
    write_mode = "a" if done_s1_ids else "w"
    need_header = not done_s1_ids

    n_processed_s1 = len(done_s1_ids)
    total_matches = 0
    n_matched_s1 = 0
    n_skipped_chunks = 0
    n_new = 0

    with open(output_path, write_mode, encoding="utf-8", buffering=1024*1024) as out_f:
        if need_header:
            out_f.write("source1_entity_id\tmatched_entity_ids\n")

        for chunk_df in pd.read_csv(pairs_path, sep="\t", dtype=str, chunksize=chunk_size):
            # Filter out already-done S1 IDs
            if done_s1_ids:
                mask = ~chunk_df["source1_entity_id"].isin(done_s1_ids)
                if not mask.any():
                    n_skipped_chunks += 1
                    continue
                if not mask.all():
                    chunk_df = chunk_df[mask].reset_index(drop=True)
                else:
                    # Passed the resume boundary
                    done_s1_ids.clear()

            chunk_s1_ids = chunk_df["source1_entity_id"].tolist()
            chunk_features = compute_features(chunk_df, s1_df, cand_df, chunk_size=chunk_size)

            passed_map = {}
            if len(chunk_features) > 0:
                X = chunk_features[FEATURE_COLS]
                probs = model.predict_proba(X)[:, 1]
                chunk_features["prob"] = probs

                passed = chunk_features[chunk_features["prob"] >= decision_threshold]
                for s1_id, cand_id in zip(passed["source1_entity_id"].values, passed["candidate_entity_id"].values):
                    if s1_id in passed_map:
                        if cand_id not in passed_map[s1_id]:
                            passed_map[s1_id].append(cand_id)
                    else:
                        passed_map[s1_id] = [cand_id]

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
            n_new += len(chunk_s1_ids)
            n_processed_s1 += len(chunk_s1_ids)
            pct = n_processed_s1 / 1_732_544 * 100
            elapsed = time.time() - t_start
            rate = n_new / elapsed if elapsed > 0 else 0.0
            rem_s1 = max(0, 1_732_544 - n_processed_s1)
            eta_min = (rem_s1 / rate) / 60 if rate > 0 else 0.0
            log.info(
                f"  Processed S1: {n_processed_s1:,}/1,732,544 ({pct:.1f}%) | "
                f"New: {n_new:,} | "
                f"Rate: {rate:.1f} S1/s | "
                f"ETA: {eta_min:.1f} min | "
                f"Matches: {total_matches:,} ({n_matched_s1:,} entities)"
            )
            del chunk_features

    del s1_df, cand_df

    log.info(f"Resume prediction complete.")
    log.info(f"  Previously done:   {len(done_s1_ids):,}")
    log.info(f"  Newly processed:   {n_new:,}")
    log.info(f"  Total now:         {n_processed_s1:,}")
    log.info(f"  Skipped chunks:    {n_skipped_chunks:,}")
    log.info(f"  New matches found: {total_matches:,} ({n_matched_s1:,} entities)")
    log.info(f"  Elapsed: {(time.time()-t_start)/60:.2f} minutes")
    return output_path


def validate_results(matching_path: str, test_dir: str):
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
    parser = argparse.ArgumentParser(description="Resume match prediction from existing results")
    parser.add_argument("--test", action="store_true", help="Run on test dataset (default: train)")
    parser.add_argument("--model", type=str, default=os.path.join(OUTPUT_DIR, "model.joblib"),
                        help="Path to trained model bundle")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override model decision threshold")
    parser.add_argument("--chunk-size", type=int, default=10_000,
                        help="Processing chunk size (default: 10,000)")
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

    # Ensure feature caches exist
    normalize_and_cache_features(s1_tsv, f"{prefix.upper()} S1", s1_cache)
    normalize_and_cache_features(s2_tsv, f"{prefix.upper()} S2", s2_cache)
    normalize_and_cache_features(s3_tsv, f"{prefix.upper()} S3", s3_cache)

    resume_predictions(
        pairs_path=pairs_tsv,
        model_path=args.model,
        s1_cache_path=s1_cache,
        s2_cache_path=s2_cache,
        s3_cache_path=s3_cache,
        output_path=args.out,
        threshold=args.threshold,
        chunk_size=args.chunk_size,
    )

    validate_results(
        matching_path=args.out,
        test_dir=dataset_dir,
    )


if __name__ == "__main__":
    main()
