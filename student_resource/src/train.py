"""
train.py  --  Phase 4: Model Training & Threshold Optimization for Entity Resolution.

Trains a LightGBM GBDT binary classifier on candidate pair features,
evaluates using the competition's exact macro-averaged F_0.5 metric,
and tunes the decision threshold for optimal precision-recall balance.

Key Steps:
  1. Generate / load training dataset of candidate pairs with 10 features + labels.
  2. Split by source1_entity_id (train/val split) to prevent entity leakage.
  3. Train LightGBM model with class weighting / focal loss handling.
  4. Optimize threshold T in [0.2, 0.9] to maximize macro-averaged F_0.5.
  5. Save model bundle (model + threshold + feature list) to output/model.joblib.
"""

import os
import sys
import time
import logging
import argparse
import joblib

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score

SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)
DATA_TRAIN  = os.path.join(PROJECT_DIR, "dataset", "train")
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "output")
CACHE_DIR   = os.path.join(OUTPUT_DIR, ".cache")

sys.path.insert(0, SRC_DIR)
from features import (
    load_feature_source, compute_features, attach_labels,
    FEATURE_COLS, CHUNK_SIZE
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ===========================================================================
# 1. Macro-Averaged F_0.5 Metric (Exact Competition Formula)
# ===========================================================================

def compute_macro_f05(
    val_df: pd.DataFrame,
    pred_col: str = "pred_prob",
    threshold: float = 0.5,
    gt_dict: dict = None,
) -> float:
    """
    Compute competition-exact macro-averaged F_0.5 across all Source 1 entities.

    Formula per entity:
      F_0.5 = (1.25 * P * R) / (0.25 * P + R)
    Singletons:
      If ground truth has no match: 1.0 if prediction is empty, else 0.0.
      If prediction is empty when GT has match: 0.0.
    """
    # Filter predictions above threshold
    passed = val_df[val_df[pred_col] >= threshold]
    preds_by_s1 = passed.groupby("source1_entity_id")["candidate_entity_id"].apply(set).to_dict()

    all_s1 = val_df["source1_entity_id"].unique()
    scores = []

    for s1 in all_s1:
        pred_set = preds_by_s1.get(s1, set())
        true_set = gt_dict.get(s1, set()) if gt_dict is not None else set()

        # Singleton evaluation
        if len(true_set) == 0:
            scores.append(1.0 if len(pred_set) == 0 else 0.0)
            continue

        if len(pred_set) == 0:
            scores.append(0.0)
            continue

        tp = len(pred_set & true_set)
        prec = tp / len(pred_set)
        rec = tp / len(true_set)

        denom = 0.25 * prec + rec
        if denom == 0:
            scores.append(0.0)
        else:
            f05 = (1.25 * prec * rec) / denom
            scores.append(f05)

    return float(np.mean(scores)) if scores else 0.0


# ===========================================================================
# 2. Data Preparation
# ===========================================================================

def prepare_training_data(
    n_s1_samples: int = 20_000,
    chunk_size: int = 5_000,
) -> pd.DataFrame:
    """
    Extract candidate pair features and labels for a stratified/sampled set of S1 entities.

    If n_s1_samples <= 0, processes all available candidate pairs.
    """
    s1_fcache = os.path.join(CACHE_DIR, "train_source1_features.parquet")
    s2_fcache = os.path.join(CACHE_DIR, "train_source2_features.parquet")
    s3_fcache = os.path.join(CACHE_DIR, "train_source3_features.parquet")
    pairs_path = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    gt_path = os.path.join(DATA_TRAIN, "train_ground_truth.tsv")

    log.info("Loading feature caches...")
    s1_df = load_feature_source(s1_fcache)
    s2_df = load_feature_source(s2_fcache)
    s3_df = load_feature_source(s3_fcache)
    cand_df = pd.concat([s2_df, s3_df], ignore_index=True)
    del s2_df, s3_df
    log.info(f"Loaded: S1={s1_df.shape}, Candidate S2+S3={cand_df.shape}")

    log.info(f"Reading candidate pairs from {pairs_path} ...")
    if n_s1_samples > 0:
        pairs_raw = pd.read_csv(pairs_path, sep="\t", dtype=str, nrows=n_s1_samples)
    else:
        pairs_raw = pd.read_csv(pairs_path, sep="\t", dtype=str)

    log.info(f"Processing candidate features for {len(pairs_raw):,} S1 entities...")
    features_df = compute_features(pairs_raw, s1_df, cand_df, chunk_size=chunk_size)
    del s1_df, cand_df, pairs_raw

    log.info("Attaching ground truth labels...")
    labeled_df = attach_labels(features_df, gt_path)
    return labeled_df


# ===========================================================================
# 3. Model Training & Optimization
# ===========================================================================

def train_and_optimize(
    data_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Train LightGBM on candidate pairs and find optimal decision threshold.
    """
    log.info(f"Total dataset shape: {data_df.shape}")
    s1_entities = np.array(data_df["source1_entity_id"].unique(), dtype=str)
    np.random.seed(random_state)
    np.random.shuffle(s1_entities)

    # Train/Val split by entity ID (prevents data leakage)
    n_val = int(len(s1_entities) * test_size)
    val_s1 = set(s1_entities[:n_val])
    train_s1 = set(s1_entities[n_val:])

    train_mask = data_df["source1_entity_id"].isin(train_s1)
    val_mask = data_df["source1_entity_id"].isin(val_s1)

    train_df = data_df[train_mask]
    val_df = data_df[val_mask].copy()

    log.info(f"Train split: {len(train_df):,} pairs ({train_df['label'].sum():,} positives, {train_df['label'].mean():.2%})")
    log.info(f"Val split:   {len(val_df):,} pairs ({val_df['label'].sum():,} positives, {val_df['label'].mean():.2%})")

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["label"]
    X_val = val_df[FEATURE_COLS]
    y_val = val_df["label"]

    # Calculate scale_pos_weight to handle ~3% positive class imbalance
    neg_count = (y_train == 0).sum()
    pos_count = max((y_train == 1).sum(), 1)
    scale_weight = float(neg_count / pos_count)
    log.info(f"Negative/Positive ratio: {scale_weight:.1f}x")

    log.info("Training LightGBM Classifier...")
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1.0,  # Keep unweighted probabilities for clean threshold tuning
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )

    t0 = time.time()
    model.fit(X_train, y_train)
    log.info(f"Model trained in {time.time()-t0:.2f}s")

    # Predict probabilities on validation set
    val_probs = model.predict_proba(X_val)[:, 1]
    val_df["pred_prob"] = val_probs

    auc = roc_auc_score(y_val, val_probs)
    log.info(f"Validation Pair AUC-ROC: {auc:.4f}")

    # Build GT dictionary for validation S1s
    gt_path = os.path.join(DATA_TRAIN, "train_ground_truth.tsv")
    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str)
    val_gt = gt_df[gt_df["source1_entity_id"].isin(val_s1)]
    gt_dict = {
        row["source1_entity_id"]: set(m.strip() for m in row["matched_entity_ids"].split(","))
        for _, row in val_gt.iterrows()
        if pd.notna(row["matched_entity_ids"]) and row["matched_entity_ids"].strip()
    }

    # Threshold search
    log.info("Optimizing threshold for macro-averaged F_0.5 ...")
    best_t = 0.5
    best_f05 = 0.0

    thresholds = np.linspace(0.1, 0.9, 17)
    for t in thresholds:
        score = compute_macro_f05(val_df, pred_col="pred_prob", threshold=t, gt_dict=gt_dict)
        log.info(f"  Threshold {t:.2f} -> Macro F_0.5: {score:.4f}")
        if score > best_f05:
            best_f05 = score
            best_t = t

    log.info(f"\n>>> Best Threshold: {best_t:.2f} with Macro F_0.5: {best_f05:.4f} <<<")

    # Feature Importance
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    log.info(f"Feature Importances:\n{importances.to_string()}")

    return model, best_t, best_f05, importances


# ===========================================================================
# 4. CLI Execution
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 4: LightGBM Matching Model Training")
    parser.add_argument("--samples", type=int, default=5000,
                        help="Number of S1 entities to sample for training (default: 5,000)")
    parser.add_argument("--chunk-size", type=int, default=2500,
                        help="Chunk size for feature computation (default: 2,500)")
    parser.add_argument("--out-model", type=str, default=os.path.join(OUTPUT_DIR, "model.joblib"),
                        help="Path to save model bundle")
    args = parser.parse_args()

    t_start = time.time()
    log.info("=" * 70)
    log.info(f"Phase 4: Training LightGBM Matcher | S1 Entities: {args.samples:,}")
    log.info("=" * 70)

    # 1. Prepare data
    labeled_df = prepare_training_data(n_s1_samples=args.samples, chunk_size=args.chunk_size)

    # 2. Train and tune
    model, best_t, best_f05, importances = train_and_optimize(labeled_df)

    # 3. Save bundle
    bundle = {
        "model": model,
        "best_threshold": best_t,
        "best_macro_f05": best_f05,
        "features": FEATURE_COLS,
        "importances": importances.to_dict(),
    }
    os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
    joblib.dump(bundle, args.out_model)
    sz_kb = os.path.getsize(args.out_model) / 1024
    log.info(f"Model bundle saved to {args.out_model} ({sz_kb:.1f} KB)")
    log.info(f"Phase 4 Training complete in {(time.time()-t_start)/60:.2f} minutes.")


if __name__ == "__main__":
    main()
