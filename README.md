# Amazon ML Challenge 2026 — Business Entity Resolution

## Project Overview & Status Tracker

End-to-end ML pipeline for large-scale Business Entity Resolution across 3 independent
data sources. For each Source 1 entity, find all matching records in Source 2 and Source 3
using noisy business_name, business_address, and country fields.

Evaluation metric: macro-averaged F_0.5 (precision-weighted — false merges penalized 2x harder than missed matches).

Per-entity F_0.5 scoring convention (for reproducible validation):
  - Entity has NO true matches AND prediction is empty     -> score = 1.0  (correct singleton)
  - Entity has NO true matches AND prediction is non-empty -> score = 0.0  (false merge)
  - Entity HAS true matches  AND prediction is empty       -> score = 0.0  (all missed; defined
    as 0 to avoid undefined division when both P and R are zero)
  - Entity HAS true matches  AND prediction is non-empty   -> standard F_0.5:
        F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
Final score = macro-average across ALL Source 1 entities (singletons included).

---

## Execution Roadmap & Progress

| Phase | Component | Status | Notes |
|:---:|---|:---:|---|
| Phase 0 | Environment & Foundation | DONE | src/, output/ created; py 3.13 launcher confirmed |
| Phase 1 | Vectorized Normalization (src/normalize.py) | DONE | See results below |
| Phase 2 | Inverted Index Blocking (src/blocking.py) | Pending | |
| Phase 3 | Pair Construction & Features (src/features.py) | Pending | |
| Phase 4 | Model Training & F0.5 Tuning (src/train_model.py) | Pending | |
| Phase 5 | Test Inference & Formatting (src/predict.py) | Pending | |
| Phase 6 | Validation & Submission Packaging | Pending | |

---

## Phase 0: Environment Setup — DONE

Directory structure:
  student_resource/
  ├── dataset/train/    (train_source1/2/3.tsv + train_ground_truth.tsv)
  ├── dataset/test/     (test_source1/2/3.tsv)
  ├── utils/            (validate_submission.py — do not modify)
  ├── src/              (NEW — all pipeline code here)
  ├── output/           (NEW — matching_results.tsv + candidate_pairs.tsv)
  └── notebooks/        (NEW — optional EDA)

Python: py 3.13 (launcher: py <script>)
Key packages required: pandas, numpy, scikit-learn, xgboost or lightgbm, rapidfuzz

---

## Phase 1: Vectorized Normalization (src/normalize.py) — DONE

### What it does
Cleans and standardizes business_name, business_address, country columns.
Produces 7 new columns used by all downstream stages.

### Output columns
  name_norm       : cleaned/expanded name string
  name_tokens     : list of name tokens (feeds blocking inverted index)
  is_addr_missing : bool — TRUE if original address was NaN (set BEFORE fillna)
  addr_norm       : cleaned/expanded address string
  addr_tokens     : list of address tokens (feeds blocking)
  addr_nums       : space-joined numeric substrings (house numbers, PINs, ZIPs)
  country_norm    : lowercased, stripped country

### Key design decisions
  - Vectorized .str operations throughout — no .apply() (critical at 10M+ rows)
  - Unicode-aware punctuation regex [^\w\s] — preserves Devanagari & French accented chars
  - Legal suffix expansion: pvt->private, ltd->limited, corp->corporation, inc->incorporated,
    llc->limited liability company, &->and (all word-boundary anchored)
  - Address abbrev: rd->road, st->street, ave->avenue, apt->apartment, blvd->boulevard, etc.
  - is_addr_missing flagged BEFORE fillna (it is a downstream feature, not just a cleanup step)
  - Numeric token extraction via regex \d+ for house numbers, ZIPs, PINs

### Verified results (sanity check)
  "Reliance Pvt Ltd"           -> "reliance private limited"       tokens: [reliance, private, limited]
  "Apple Corp & Associates"    -> "apple corporation associates"   tokens: [apple, corporation, associates]
  "Apt 4B, 12 rue de la Paix" -> "apartment 4b 12 rue de la paix paris"  addr_nums: "4 12"
  NaN name                     -> ""                               tokens: []
  NaN address                  -> is_addr_missing = True

### Speed test (100k rows of train_source1.tsv)
  Load time : 0.22s
  Norm time : 8.54s  (extrapolates to ~3.5 min for full 2.2M S1; ~18 min for all 3 sources)
  Output shape: (100000, 11)

### Performance note
  8.54s / 100k rows is mainly due to the abbreviation expansion loop (.str.replace per rule).
  Acceptable for a one-time preprocessing step. If needed, can be optimized with a single
  compiled combined regex replace, but not required for v1.

---

## Phase 2 (Next): Inverted-Index Token Blocking (src/blocking.py)

Goal: reduce 2.2M x 10.3M space to <=200 candidates per S1 entity.
Mandatory checkpoint: blocking recall >= 95% on validation split.
Key steps:
  1. Build inverted index: token -> set(S2/S3 entity_ids) over S2+S3 combined
  2. Exclude high-frequency tokens (stopword-like: "the", "inc", "llc", "store")
  3. For each S1 entity: candidates = union of all S2/S3 sharing a name token
  4. Union with address-token candidates
  5. Hard filter: same country_norm only
  6. Cap at top-200 by token-Jaccard score
  7. Compute blocking recall on validation; target >= 95% before proceeding

---

## Session Change Log

2026-09-26:
  - Initialized project, confirmed 6-phase architecture
  - Phase 0: Created src/, output/, notebooks/ directories
  - Phase 1: Built and tested src/normalize.py
    * All correctness assertions pass
    * Speed: 100k rows in 8.54s (vectorized, no .apply)
    * Unicode safety confirmed (Devanagari, French accented chars preserved)
    * is_addr_missing correctly flagged before NaN fill
