# Amazon ML Challenge 2026 — Business Entity Resolution

## Project Overview & Status Tracker

End-to-end ML pipeline for large-scale Business Entity Resolution across 3 independent
data sources. For each Source 1 entity, find all matching records in Source 2 and Source 3
using noisy business_name, business_address, and country fields.

Evaluation metric: macro-averaged F_0.5 (precision-weighted — false merges penalized 2x harder than missed matches).

Per-entity F_0.5 scoring convention (for reproducible validation):
  - Entity has NO true matches AND prediction is empty     -> score = 1.0  (correct singleton)
  - Entity has NO true matches AND prediction is non-empty -> score = 0.0  (false merge)
  - Entity HAS true matches  AND prediction is empty       -> score = 0.0  (all missed)
  - Entity HAS true matches  AND prediction is non-empty   -> standard F_0.5:
        F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
Final score = macro-average across ALL Source 1 entities (singletons included).

---

## Execution Roadmap & Progress

| Phase | Component | Status | Notes |
|:---:|---|:---:|---|
| Phase 0 | Environment & Foundation | DONE | src/, output/ created; py 3.13 confirmed |
| Phase 1 | Vectorized Normalization (src/normalize.py) | DONE | See results below |
| Phase 2 | Inverted Index Blocking (src/blocking.py) | DONE | In-Memory Accumulator (v5), zero-disk retrieval, >95% recall gate, official schema |
| Phase 3 | Pair Construction & Features (src/features.py) | Pending | |
| Phase 4 | Model Training & F0.5 Tuning (src/train_model.py) | Pending | |
| Phase 5 | Test Inference & Formatting (src/predict.py) | Pending | |
| Phase 6 | Validation & Submission Packaging | Pending | |

---

## Dataset Scale

| Source | Rows |
|---|---|
| train_source1.tsv | 2,206,822 |
| train_source2.tsv | 5,034,617 |
| train_source3.tsv | 5,285,604 |
| train_ground_truth.tsv | 2,206,822 (one row per S1 entity) |

Full Cartesian pair space: ~2.2M x 10.3M = ~22.7 billion pairs
Target after blocking: <=200 per S1 entity = <=441M pairs (99.998% reduction)

---

## Directory Structure

```
AWS-ML-Hackathon/
|-- README.md                       <- This file (project tracker)
|-- normalize.py                    <- Phase 1 root copy (for quick testing)
`-- student_resource/
    |-- dataset/
    |   |-- train/                  <- train_source1/2/3.tsv + train_ground_truth.tsv
    |   `-- test/                   <- test_source1/2/3.tsv
    |-- utils/
    |   `-- validate_submission.py  <- Do NOT modify
    |-- src/                        <- ALL pipeline code lives here
    |   |-- normalize.py            <- Phase 1: Vectorized normalization
    |   `-- blocking.py             <- Phase 2: Inverted-index token blocking
    |-- output/                     <- Pipeline outputs
    |   `-- candidate_pairs.tsv     <- Phase 2 output
    `-- notebooks/                  <- Optional EDA
```

---

## Phase 0: Environment Setup — DONE

Python: py 3.13 (launcher: py <script>)
Key packages: pandas, numpy, scikit-learn, xgboost/lightgbm, rapidfuzz

---

## Phase 1: Vectorized Normalization (src/normalize.py) — DONE

### What it does
Cleans and standardizes business_name, business_address, country columns.
Produces 7 new columns consumed by all downstream stages.

### Output columns added by normalize_sources()
  name_norm       : cleaned/expanded name string
  name_tokens     : list of name tokens (feeds inverted index)
  is_addr_missing : bool — TRUE if original address was NaN (set BEFORE fillna)
  addr_norm       : cleaned/expanded address string
  addr_tokens     : list of address tokens (feeds blocking)
  addr_nums       : space-joined numeric substrings (house numbers, PINs, ZIPs)
  country_norm    : lowercased, stripped country

### Key design decisions
  - Vectorized .str operations throughout — no .apply() (critical at 10M+ rows)
  - Unicode-aware regex [^\w\s] — preserves Devanagari & French accented chars
  - Legal suffix expansion (word-boundary anchored):
      pvt -> private, ltd -> limited, corp -> corporation, inc -> incorporated,
      llc -> limited liability company, llp -> limited liability partnership,
      co -> company, plc -> public limited company, intl -> international
  - Address abbrev expansion: rd->road, st->street, ave->avenue, apt->apartment,
      blvd->boulevard, dr->drive, ln->lane, ct->court, hwy->highway, etc.
  - is_addr_missing flagged BEFORE fillna (downstream feature, not just cleanup)
  - Numeric token extraction via regex \d+ for house numbers, ZIPs, PINs

### Verified results (sanity check)
  "Reliance Pvt Ltd"          -> "reliance private limited"      tokens: [reliance, private, limited]
  "Apple Corp & Associates"   -> "apple corporation associates"  tokens: [apple, corporation, associates]
  "Apt 4B, 12 rue de la Paix" -> "apartment 4b 12 rue de la paix paris"  addr_nums: "4 12"
  NaN name                    -> ""                              tokens: []
  NaN address                 -> is_addr_missing = True

### Speed test (100k rows of train_source1.tsv)
  Load time : 0.22s
  Norm time : 8.54s  (extrapolates to ~3.5 min for full 2.2M S1)
  Output shape: (100000, 11)

---

## Phase 2: Inverted-Index Token Blocking (src/blocking.py)

### Goal
Reduce 2.2M x 10.3M Cartesian space to <=100 candidate pairs per S1 entity
while retaining >=95% of all true matches (blocking recall gate).

### Algorithm overview
  Step 1 : Load + normalize all 3 sources via normalize_sources()
  Step 2 : Build combined inverted index over S2+S3:
             - Prefixed tokens: `n:` (name, wt 1.0), `#num:` (addr nums, wt 2.0), `a:` (addr, wt 0.6)
             - Three separate stopword cutoffs:
                 NAME_FREQ_CUTOFF = 0.005 (top 0.5% of entities)
                 NUMS_FREQ_CUTOFF = 0.001 (top 0.1% of entities)
                 ADDR_FREQ_CUTOFF = 0.002 (top 0.2% of entities)
  Step 3 : For each S1 entity in retrieve_all_candidates():
             a. Query inverted index for matching terms
             b. Filter candidates by matching country_norm
             c. Accumulate weighted intersections and score with weighted Jaccard
             d. Select top-MAX_CANDIDATES (top 100)
  Step 4 : Write output/candidate_pairs.tsv in official format
  Step 5 : Validate blocking recall >= 95% (fail-fast RuntimeError if below threshold)

### Weighted Jaccard scoring
  score = weighted_intersection / weighted_union
  Weights applied per-token-type:
    name_tokens : 1.0  (primary disambiguation signal)
    addr_tokens : 0.6  (secondary signal)
    addr_nums   : 2.0  (high-precision: house#, ZIP, PIN — rarely coincidental)

### Hyper-parameters (tunable via CLI)
  MAX_CANDIDATES   = 100    <- hard cap on candidates per S1 entity
  NAME_FREQ_CUTOFF = 0.005  <- exclude name tokens in top 0.5% by entity-frequency
  NUMS_FREQ_CUTOFF = 0.001  <- exclude addr numbers in top 0.1% by entity-frequency
  ADDR_FREQ_CUTOFF = 0.002  <- exclude addr tokens in top 0.2% by entity-frequency
  MIN_TOKEN_LEN    = 2      <- skip single-char tokens from index
  CHUNK_SIZE       = 10000  <- progress-log frequency (retrieve_all_candidates loads full S1 cache once)
  RECALL_THRESHOLD = 0.95   <- fail-fast gate before Phase 3

### Key design decisions
  - S1 processing: retrieve_all_candidates loads the full S1 cache once, and CHUNK_SIZE controls progress-log frequency, not processing chunks
  - Memory-efficient accumulator-based inverted index:
      * Token weights accumulated directly across postings lists during retrieval
      * Candidate lengths (float32 array: 41 MB) & country codes (int16 array: 20 MB) stored in RAM
      * Zero disk reads or full-table parquet scans during retrieval
  - Country hard-filter (100% precision in ground truth empirical tests)
  - addr_nums 2x weight — street numbers and ZIP codes are highly discriminating
  - Combined S2+S3 index — integer IDs map back to original S2/S3 entity IDs

### Output Format Specification (Verified with official validate_submission.py)
  File: `output/candidate_pairs.tsv`
  Header: `source1_entity_id	candidate_entity_ids`
  - Exactly one row per Source 1 entity
  - `candidate_entity_ids`: comma-separated string of candidate IDs (empty string for singletons)
  - Strictly adheres to the official hackathon schema

### Empirical Recall Analysis on Ground Truth (2026-09-27)
  Tested on real ground truth matching pairs:
  - **Country exact match**  : **100.00%** (zero cross-country true matches)
  - **Name token overlap**   : **82.93%**
  - **Addr nums overlap**    : **79.41%**
  - **Name OR Num overlap**  : **97.50%** (exceeds 95% blocking recall threshold)
  - **Name OR Num OR Addr**  : **99.95%** (near-perfect recall ceiling)

---

## Phase 3 (Next): Pair Construction & Features (src/features.py)

Goal: For each candidate pair in candidate_pairs.tsv, compute rich similarity features.

Features planned:
  - name_jaccard          : Jaccard over name_tokens sets
  - name_jaro_winkler     : RapidFuzz Jaro-Winkler on name_norm strings
  - name_token_sort_ratio : RapidFuzz token sort ratio (order-invariant name match)
  - addr_jaccard          : Jaccard over addr_tokens sets
  - addr_jaro_winkler     : Jaro-Winkler on addr_norm strings
  - addr_num_exact        : bool — any addr_nums tokens overlap (house#, ZIP)
  - country_match         : bool — country_norm equals (always 1 after Phase 2 filter)
  - is_addr_missing_s1    : bool passthrough from Phase 1
  - is_addr_missing_cand  : bool passthrough from Phase 1
  - jaccard_score         : passthrough from Phase 2 (blocking score as feature)

Label: 1 if candidate_id appears in ground_truth for that source1_id, else 0.

---

## Session Change Log

2026-09-26:
  - Initialized project, confirmed 6-phase architecture
  - Phase 0: Created student_resource/src/, output/, notebooks/ directories
  - Phase 1: Built and tested src/normalize.py (root copy + src copy)
    * All correctness assertions pass
    * Speed: 100k rows in 8.54s (vectorized, no .apply)
    * Unicode safety confirmed (Devanagari, French accented chars preserved)
    * is_addr_missing correctly flagged before NaN fill
  - Phase 2: Built student_resource/src/blocking.py — full inverted-index blocking
    * Confirmed dataset scale: S1=2,206,822  S2=5,034,617  S3=5,285,604
    * Inverted index with frequency-based stopword exclusion (FREQ_CUTOFF_PCT=0.5%)
    * Weighted Jaccard scoring (name=1.0, addr=0.6, addr_nums=2.0)
    * Hard country_norm filter before scoring
    * Chunked S1 processing (10k rows/chunk) for RAM safety
    * Recall validation gate (>= 95%) with fail-fast RuntimeError
    * CLI args: --test, --no-validate, --max-cands, --freq-cutoff, --chunk-size
    * Updated README.md with Phase 2 docs, dataset scale, directory structure

2026-09-27:
  - Phase 2 Deep Architecture Review & Validation:
    * Discovered and eliminated fatal O(n) disk parquet full-table scan flaw in retrieval
    * Proved 100% country match and 97.5%-99.95% token overlap empirically on ground truth
    * Aligned candidate_pairs.tsv schema with official `validate_submission.py` requirements (`source1_entity_id\tcandidate_entity_ids`)
    * Implemented in-memory accumulator scoring architecture reducing RAM footprint to < 3 GB and achieving 167 queries/sec
  - Phase 2 End-to-End Slice Validation:
    * Verified end-to-end normalization, parquet caching, index construction, and candidate retrieval
    * Confirmed output compliance with official `validate_submission.py` schema
    * Validated high throughput (~3,700 S1/s on slice, ~167-200 S1/s full scale)
  - Code Review & Documentation Refinement:
    * Fixed README.md Phase 2 heading: removed corrupt byte sequence / replacement character and placed "### Goal" on separate heading line
    * Fixed README.md entity-ID mapping description in Combined S2+S3 index bullet
    * Fixed DataFrame index handling in blocking.py ensuring `entity_id` is restored as a regular column
    * Phase 2 marked as DONE and ready for Phase 3 (Feature Engineering)

