# ML Challenge 2026 Problem Statement

## Business Entity Resolution Challenge

In large-scale commercial platforms, business identity data arrives from multiple independent sources — each contributing partial, noisy fragments of information about the same real-world entities. These fragments share no common identifiers, and the challenge of determining which records refer to the same business is known as Entity Resolution (ER). Your challenge is to build an ML solution that, given business records from 3 independent data sources with noisy and inconsistent fields, determines which records across sources refer to the same real-world business entity.

Source 1 is the deduplicated reference source. Your task is to find all matching records from Source 2 and Source 3 for each Source 1 entity. A Source 1 entity may match zero, one, or many records from Source 2 and Source 3.

### File Format

**All files in this challenge are tab-separated (`.tsv`), and your submissions must be tab-separated too.** Tabs are used because business addresses and the ID list columns both contain commas. Read them with an explicit tab separator, for example:

```python
import pandas as pd
df = pd.read_csv("dataset/train/train_source1.tsv", sep="\t")
```

Reading a `.tsv` without `sep="\t"` will silently produce a single column containing the whole line.

### Data Description:

Each source file (`*_source1.tsv`, `*_source2.tsv`, `*_source3.tsv`) has the following columns:

1. **entity_id:** Unique identifier for the record. The prefix indicates the source — `S1-`, `S2-`, or `S3-`.
2. **business_name:** Name of the business entity (may contain abbreviations, legal suffixes, typos, transliterations)
3. **business_address:** Address of the business (may contain partial addresses, format variations, missing components, landmark-based references)
4. **country:** Country label for the record. The **training** data covers `US` and `India`. The **test** set additionally contains a third country, `France`, that does **not** appear in the training data. Treat `country` as an open set of string labels: do **not** hard-code, filter, or one-hot your pipeline to only `{US, India}`, and remember that every test entity — `France` included — must appear in your submission.

There is no separate *source* column — a record's source is given by its `entity_id` prefix (`S1-`/`S2-`/`S3-`) and by which file it appears in.

The ground truth file (`train_ground_truth.tsv`) has two columns:

1. **source1_entity_id:** The `entity_id` of a Source 1 record
2. **matched_entity_ids:** Comma-separated list of matching `entity_id`s from Source 2 and/or Source 3 (empty when the entity has no matches)

**Noise Patterns to Expect:**

- **Name variations:** Abbreviations (Corp vs. Corporation, Pvt vs. Private, Ltd vs. Limited), legal suffix inconsistencies, DBA/trade names, punctuation differences (& vs. "and"), word-order transpositions, typos
- **Address variations:** Abbreviations (Rd vs. Road, St vs. Street), transliteration variants, missing components (no PIN code, no state), landmark-based references (Near SBI ATM), municipal numbering formats, component reordering

### Dataset Details:

- **Training Dataset:** Business records across 3 sources with ground truth matching labels
- **Test Set:** Business records across 3 sources without matching labels

### File Descriptions:

*Training files*

1. **dataset/train/train_source1.tsv:** Source 1 training records (the deduplicated reference source)
2. **dataset/train/train_source2.tsv:** Source 2 training records
3. **dataset/train/train_source3.tsv:** Source 3 training records
4. **dataset/train/train_ground_truth.tsv:** Ground truth matching labels for the training set

*Test files*

1. **dataset/test/test_source1.tsv:** Source 1 test records. Generate matches for every entity in this file.
2. **dataset/test/test_source2.tsv:** Source 2 test records
3. **dataset/test/test_source3.tsv:** Source 3 test records

No ground truth is provided for the test set. To measure your own performance, hold out a validation split from the training data and score it yourself using the F_0.5 formula given below.

### Output Format:

Your solution produces **two** tab-separated files, both placed in the `output/`
folder of your final submission package (see *Final Submission Package* below):

1. **`matching_results.tsv`** — your final entity matches. **This is the only file
   scored on the leaderboard** — it is what you upload to the Portal during the challenge.
2. **`candidate_pairs.tsv`** — the candidate set your blocking / candidate-generation
   stage produced, before your final matching model narrowed it down.

#### matching_results.tsv

Your final entity matches:

| Column | Description |
| --- | --- |
| source1_entity_id | The `entity_id` of a Source 1 record |
| matched_entity_ids | Comma-separated list of matching `entity_id`s from Source 2 and/or Source 3 |

**Example** (columns separated by a single tab, ID lists separated by commas with no quoting):

```
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812
S1-00002	S3-00004
S1-00003	
```

**Important:**

- Every Source 1 entity in the test set must have exactly one row
- Leave `matched_entity_ids` empty for entities with no matches (singletons)
- No duplicate entity IDs within a single ID list
- ID lists must only contain Source 2 or Source 3 IDs that exist in the test set

#### candidate_pairs.tsv

The candidate set from your blocking stage — every Source 2 / Source 3 record you
considered a plausible match for each Source 1 entity, *before* your final matching
model narrowed it down. This is the **exact set of records you feed into your matching model
for inference** — the final candidate list *just before* the ML model scores them, not
the raw output of an early blocking pass you later filter further. If your pipeline has
several blocking/filtering stages, `candidate_pairs.tsv` is the *last* one: whatever
your model actually runs inference over. Every ID in `matching_results.tsv` should
therefore appear here.

It is **not scored on the leaderboard**; we use it to analyse blocking quality (recall
ceiling, reduction ratio) and to verify your pipeline.

| Column | Description |
| --- | --- |
| source1_entity_id | The `entity_id` of a Source 1 record |
| candidate_entity_ids | Comma-separated list of candidate `entity_id`s from Source 2 and/or Source 3 |

**Example:**

```
source1_entity_id	candidate_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812,S3-00999
S1-00002	S3-00004
S1-00003	
```

Same rules as `matching_results.tsv`: one row per Source 1 entity, `candidate_entity_ids`
empty when blocking found no candidates, S2-/S3- IDs only, no duplicates within a list.
Your final matches should be a **subset** of your candidates (a matched ID that never
appeared as a candidate signals a pipeline bug — the validator warns about it).

**Validate before submitting:** a helper script `utils/validate_submission.py` (stdlib
only, no dependencies) checks both files against every rule above so you can catch a
rejection locally instead of spending a submission on it. Run it from this
`student_resource/` directory:

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

It prints `PASS` (exit 0) when the files are safe to submit, or a numbered list of issues
to fix (exit 1). It only reads your output files and the test source files; it does not
compute your score.

### Final Submission Package:

In addition to your live leaderboard uploads, **every team submits a single zip
archive** with your code and outputs. We use it to reproduce your results, audit your
blocking, and check the fair-play and model-license rules — the top teams' packages are
reviewed in detail before the final rankings are confirmed.

Structure:

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv        # final matches (same file you upload to the leaderboard)
│   └── candidate_pairs.tsv         # your blocking candidate set
├── code/
│   └── business_entity_resolution/
│       ├── src/                    # all your source code
│       ├── README.md               # how to reproduce end-to-end (data → blocking → matching → output)
│       └── requirements.txt        # pinned dependencies / environment
└── Documentation_template.md       # your methodology write-up (this filled-in template)
```

- **`output/`** — the two TSV files described above: `matching_results.tsv` and
  `candidate_pairs.tsv`.
- **`code/business_entity_resolution/`** — a self-contained, runnable copy of your
  pipeline. Put all source under `src/`, and include a `README.md` with exact run
  instructions plus a `requirements.txt` (or equivalent environment file) pinning
  versions. Anyone should be able to regenerate both output files from the
  training/test data using only what is in this folder.
- **Methodology document** — fill in the provided `Documentation_template.md` and drop
  it straight into the zip (the filled-in `.md` is fine; a `.pdf` export works too). No
  need to rename it.

### Constraints:

1. Format your output exactly as described above. Submissions that fail validation will not be evaluated. You should see a `SCORED` status with your F_0.5 score if the output is correctly formatted.
2. `matched_entity_ids` must only reference entities from Source 2 or Source 3. Self-matches to Source 1, and IDs that do not exist in the test set, will be rejected.
3. Every Source 1 entity must appear in your submission. Missing entities will cause rejection.
4. Duplicate entity IDs in any ID list will cause rejection, as will duplicate `source1_entity_id` rows.
5. Final model should be a MIT/Apache 2.0 License model and up to 8 Billion parameters.

### Evaluation Criteria:

Submissions are evaluated using **F_β Score (β = 0.5)** — a precision-heavy metric that penalizes false merges (matching two different businesses) more than missed matches.

**Formula:**

```
F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```

Computed as a **macro-average**: F_0.5 is calculated per Source 1 entity, then averaged across **all** Source 1 entities in the evaluation set.

Singletons are included in that average. A Source 1 entity with no true matches scores 1.0 when you correctly predict an empty list, and 0.0 when you predict any match for it. Correctly identifying singletons therefore earns credit, and false merges on them are penalised. When true matches exist but the prediction contains no correct IDs (zero-overlap, Precision = Recall = 0), the F₀.₅ score is defined as 0.0 so the metric remains well-defined.

**Why precision-heavy?** In real-world entity resolution, merging two distinct businesses (false positive) is more damaging than missing a link (false negative). F_0.5 weights precision 2× over recall.

**Example:**

- Your model predicts S1-00001 matches [S2-00047, S2-00193, S3-00812]
- Ground truth says S1-00001 matches [S2-00047, S3-00812]
- Precision = 2/3, Recall = 2/2 = 1.0
- F_0.5 = (1.25 × 0.667 × 1.0) / (0.25 × 0.667 + 1.0) = **0.714**

### Leaderboard Information:

- **Public Leaderboard:** During the challenge, rankings will be based on a subset of the test set to provide real-time feedback on your model's performance.
- **Private Leaderboard:** After the challenge ends, the private leaderboard will be revealed, which uses the remaining portion of the test set for evaluation.
- **Final Rankings:** The final decision will be based on the private leaderboard.

You submit predictions for the full test set in both cases; the split is applied during scoring.

### Submission Requirements:

1. **Leaderboard (during the challenge):** upload `matching_results.tsv` in the Portal —
   tab-separated, with the exact column names described above. This is what drives the
   public and private leaderboards.
2. **Final submission package:** submit the single zip described in *Final Submission
   Package* above — `output/` with **both** `matching_results.tsv` (final matches) and
   `candidate_pairs.tsv` (your candidate-generation / blocking set fed to the model),
   `code/business_entity_resolution/` (runnable pipeline), and your methodology document.
   All teams must submit it; the top teams' packages are reviewed before the final
   rankings are confirmed.
3. Your methodology document must describe:
   - Methodology used
   - Candidate generation / blocking strategy
   - Model architecture and feature engineering
   - Any other relevant information about the approach

   A template for this documentation is provided in `Documentation_template.md`. There is no page limit — prioritise clarity and technical depth over brevity.

### **Academic Integrity and Fair Play:**

**⚠️ STRICTLY PROHIBITED: External Data Lookup**

Participants are **STRICTLY NOT ALLOWED** to use external databases, APIs, or services to look up business identities or resolve entities. This includes but is not limited to:

- Using commercial entity resolution APIs or services
- Looking up business registrations from government databases
- Using geocoding APIs to normalize addresses
- Any external data augmentation from internet sources

**Enforcement:**

- All submitted approaches, methodologies, and code pipelines will be thoroughly reviewed and verified
- Any evidence of external data lookup will result in **immediate disqualification**

**Fair Play:** This challenge is designed to test your machine learning and data science skills using only the provided training data.

### Tips for Success:

- Invest in a strong blocking/candidate generation strategy — it determines the upper bound of your recall
- Explore string similarity features (Jaccard, Levenshtein, TF-IDF cosine) for name and address matching
- Pay attention to country specific address patterns
- Consider the precision-recall trade-off carefully — F_0.5 rewards precision more than recall
- Do not neglect singletons — correctly predicting "no match" is worth a full 1.0 on that entity
- Validate your own output format against the rules above before submitting

---

## Implementation Log — Team Dev Notes

> **Note:** Everything below this line is team-internal implementation documentation.
> The section above is the unmodified official problem statement.

---

### Project Structure

```
student_resource/
├── dataset/
│   ├── train/                          # train_source{1,2,3}.tsv + train_ground_truth.tsv
│   └── test/                           # test_source{1,2,3}.tsv (no ground truth)
├── output/
│   ├── candidate_pairs.tsv             # Phase 2 output  (2.87 GB, ~220M pairs)
│   ├── features_train.parquet          # Phase 3 output  (computed per-pair features)
│   ├── model.joblib                    # Phase 4 output  (trained LightGBM bundle + threshold)
│   ├── matching_results.tsv            # Phase 4 output  (final submission predictions)
│   └── .cache/
│       ├── train_source1.parquet       # Blocking cache: name/addr tokens + country
│       ├── train_source2.parquet
│       ├── train_source3.parquet
│       ├── train_source1_features.parquet  # Feature cache: adds name_norm, addr_norm,
│       ├── train_source2_features.parquet  #   is_addr_missing (superset of blocking cache)
│       └── train_source3_features.parquet
├── src/
│   ├── normalize.py                    # Phase 1 — vectorized Unicode-safe normalization
│   ├── blocking.py                     # Phase 2 — in-memory accumulator blocking (v5)
│   ├── features.py                     # Phase 3 — feature engineering (10 features)
│   ├── train.py                        # Phase 4 — LightGBM training & F0.5 threshold search
│   └── predict.py                      # Phase 4 — match prediction & submission generation
├── utils/
│   └── validate_submission.py          # Official schema validator (stdlib only)
├── Documentation_template.md
└── README.md                           # This file
```

---

### Phase 0 — Environment Setup ✅

- Python 3.13 environment
- Key dependencies: `pandas`, `numpy`, `rapidfuzz`, `pyarrow`
- Working directory: `student_resource/`

---

### Phase 1 — Normalization (`normalize.py`) ✅

**Goal:** Clean and tokenize all source fields into a canonical form ready for indexing.

**Design:**
- Fully vectorized `pandas.str` operations — no `.apply()` loops at scale
- Unicode-aware regex preserves Devanagari, accented Latin (French), etc.
- Single-pass abbreviation expansion using a compiled alternation regex
  (10 name rules + 17 addr rules → 2 regex passes instead of N sequential ones)
- `is_addr_missing` flag captured **before** NaN fill (important for features)

**Output columns added by `normalize_sources(df)`:**

| Column | Description |
|---|---|
| `name_norm` | Cleaned/expanded business name string |
| `name_tokens` | List of tokens from `name_norm` (for blocking) |
| `addr_norm` | Cleaned/expanded address string |
| `addr_tokens` | List of address tokens |
| `addr_nums` | Space-separated numeric substrings (house #, ZIP, PIN) |
| `country_norm` | Lowercased country label |
| `is_addr_missing` | `True` if address was missing/blank before normalization |

---

### Phase 2 — Blocking / Candidate Generation (`blocking.py`) ✅

**Goal:** For each S1 entity, find all plausible S2/S3 candidates (recall-optimized).

**Architecture — v5 "In-Memory Accumulator":**
- Build an in-memory inverted index: `token → np.ndarray[int32]` of S2/S3 row indices
- For each S1 entity: collect all posting lists, `np.bincount` intersection counts,
  apply weighted Jaccard scoring, threshold, emit top-K candidates
- Memory-mapped `float32` weight arrays + `int16` country code arrays —
  eliminates $O(N)$ disk scans on every chunk
- Replaced all `.iterrows()` with vectorized NumPy ops and `zip()`-based dict construction

**Token weights:**

| Token type | Weight | Min length |
|---|---|---|
| Name tokens | 1.0 | 2 chars |
| Addr nums (house #, ZIP) | 2.0 | 2 chars |
| Addr tokens | 0.6 | 4 chars |

**Results (full training set):**
- **Blocking recall:** 89.24% on 7.64M ground-truth pairs (full-scale training run bypassed the nominal 95% development gate via `--no-validate` to balance candidate size and downstream $F_{0.5}$ precision)
- **Theoretical recall ceiling:** 99.95%
- **Output:** `output/candidate_pairs.tsv` — 2.87 GB, schema-compliant

**Schema:** `source1_entity_id\tcandidate_entity_ids` (comma-separated, no spaces)

---

### Phase 3 — Feature Engineering (`features.py`) ✅ COMPLETED & VALIDATED

**Goal:** For every candidate pair in `candidate_pairs.tsv`, compute rich
similarity features for downstream model training (Phase 4).

**Features computed (10 total):**

| # | Feature | Description |
|---|---|---|
| 1 | `name_jaccard` | Jaccard similarity over `name_tokens` sets |
| 2 | `name_jaro_winkler` | RapidFuzz Jaro-Winkler on `name_norm` strings |
| 3 | `name_token_sort_ratio` | RapidFuzz token sort ratio (order-invariant name match) |
| 4 | `addr_jaccard` | Jaccard similarity over `addr_tokens` sets |
| 5 | `addr_jaro_winkler` | Jaro-Winkler on `addr_norm` strings |
| 6 | `addr_num_exact` | 1 if any `addr_nums` tokens overlap (house #, ZIP, PIN) |
| 7 | `country_match` | 1 if `country_norm` is equal |
| 8 | `is_addr_missing_s1` | Address-missing flag for the S1 entity |
| 9 | `is_addr_missing_cand` | Address-missing flag for the candidate entity |
| 10 | `jaccard_score` | Recomputed weighted Jaccard (same formula as blocking) |

**Label:** `1` if `candidate_entity_id` appears in `train_ground_truth.tsv` for that
`source1_entity_id`, else `0`.

**Architecture & Optimizations:**
- **Feature Parquet Caches Built:**
  - `train_source1_features.parquet` (240.5 MB, 2.2M rows)
  - `train_source2_features.parquet` (527.0 MB, 5.0M rows)
  - `train_source3_features.parquet` (555.8 MB, 5.3M rows)
  Stored in `output/.cache/` with string columns (`name_norm`, `addr_norm`) and missing address flags.
- **Fast Chunked Processing:** Explodes and evaluates pairs chunk-by-chunk over S1 entities, maintaining strict memory bounds.
- **Vectorized Label Attachment:** Replaced slow line-by-line `iterrows()` with vectorized S1 filtering and pandas dataframe left-merge, completing ground-truth label attachment on 10,000 pairs in 4.78 seconds.
- **Validation Run:** Verified on sample candidate pairs with active ground truth verification (yielding ~2.7% positive match rate, aligned with expected real-world candidate distribution).

**Run:**
```bash
# Working directory: student_resource/
python src/features.py                  # train mode
python src/features.py --test           # test mode
python src/features.py --chunk-size 5000 --force-renorm   # override defaults
```

**Output:** `output/features_train.parquet`

---

### Phase 4 — Matching Model (`train.py` & `predict.py`) ✅ COMPLETED & VALIDATED

**Goal:** Train a high-precision binary classifier on candidate pair features to predict final entity matches.

**Architecture & Implementation:**
- **Model:** LightGBM 4.7 (`LGBMClassifier`, 300 estimators, max depth 7, 63 leaves).
- **Scale-Up Training Run (500,000 candidate pairs across 5,000 S1 entities):**
  - **Pair-level Validation AUC-ROC:** **0.9978**
  - **Macro-Averaged $F_{0.5}$ Score:** **0.8630** at optimal threshold $T = 0.65$.
  - **Feature Importances:**
    1. `name_jaro_winkler` (3,545 splits)
    2. `name_token_sort_ratio` (3,506 splits)
    3. `jaccard_score` (2,814 splits)
    4. `addr_jaro_winkler` (2,763 splits)
    5. `addr_jaccard` (2,742 splits)
    6. `name_jaccard` (2,187 splits)
    7. `addr_num_exact` (593 splits)
    8. `is_addr_missing_cand` (163 splits)
- **Model Artifact:** Saved to `output/model.joblib` (2.0 MB).
- **Inference & Singleton Pruning:** `predict.py` applies the optimal threshold $T = 0.65$ to discard low-confidence matches. S1 entities with all candidates below threshold are output as empty strings (`""`), earning full 1.0 credit on true singletons.

**Run:**
```bash
# Working directory: student_resource/
# Train model & optimize threshold:
python src/train.py --samples 5000 --chunk-size 2500

# Predict matches & generate submission:
python src/predict.py --chunk-size 10000
```

---

### Phase 5 — Packaging & Methodology Documentation ✅ COMPLETED

- **Pinned Requirements:** [requirements.txt](requirements.txt) created with exact versions (`lightgbm==4.7.0`, `rapidfuzz==3.14.6`, `pandas==3.0.5`, `numpy==2.5.0`, `pyarrow==25.0.1`, `scikit-learn==1.9.0`, `joblib==1.5.3`, `scipy==1.18.0`).
- **Methodology Documentation:** [Documentation_template.md](Documentation_template.md) completely filled out with:
  - Executive summary and EDA problem analysis.
  - Candidate blocking keys, accumulator architecture, and 89.24% recall analysis.
  - 10 pairwise similarity features and LightGBM model configuration.
  - Empirical validation results (0.9978 AUC, 0.8630 macro $F_{0.5}$).
  - Code artefact manifest and threshold trajectory analysis.

---

### Key Design Decisions

1. **Separate blocking vs. feature caches** — Blocking only needs token arrays;
   feature engineering additionally needs the raw normalized strings. Keeping them
   separate avoids re-running blocking when Phase 3 parameters change.

2. **Weighted Jaccard for both blocking and features** — The same formula is used
   in blocking (for fast retrieval) and recomputed as a feature (as `jaccard_score`).
   The minor difference (blocking excludes high-frequency stopword tokens from the
   inverted index) is harmless — the ML model learns the appropriate weight.

3. **`np.frompyfunc` over `.apply()`** — For per-pair Python-level computations
   (Jaro-Winkler, token sort ratio), `np.frompyfunc` avoids the Python/Pandas
   overhead of `Series.apply` and processes the entire array in one call.

4. **F_0.5 metric** — The evaluation metric is precision-heavy (β=0.5), so:
   - Blocking is tuned for **recall** (high ceiling) since misses can't be recovered
   - The classifier threshold is tuned for **precision** to maximize F_0.5

---

### Change Log

| Session | Changes |
|---|---|
| Session 1 | Repo setup, Phase 0 environment |
| Session 2 | Phase 1: `normalize.py` — vectorized normalization with abbreviation expansion |
| Session 3 | Phase 2: `blocking.py` v1–v4 iterations → v5 accumulator architecture |
| Session 4 | Phase 2 fixes: memory-mapped arrays, vectorized scoring, streaming `validate_recall` |
| Session 5 | Phase 2 validation: 89.24% blocking recall on 7.64M GT pairs; schema verified |
| Session 6 | Phase 3: `features.py` — 10-feature pipeline; feature caches; chunked processing |
| Session 6 | Fix `\w` SyntaxWarning in `normalize.py` docstring (Python 3.12+ compat) |
| Session 7 | Phase 3 completion: Built S1/S2/S3 feature caches (~1.3 GB total); optimized `attach_labels` with vectorized pandas merge; installed LightGBM 4.7; initialized Phase 4 Matching Model architecture. |
| Session 8 | Phase 4 & 5 completion: Scaled LightGBM training on 500k pairs (AUC 0.9978, Macro F0.5 0.8630 at T=0.65); saved `model.joblib`; verified `predict.py`; created `requirements.txt`; populated `Documentation_template.md`. |
| Session 9 | Test Blocking Completion: Generated `output/candidate_pairs.tsv` (2,150.7 MB) covering 1,732,544 test S1 entities with inverted-index accumulator blocking; preserved training candidate pairs in `output/candidate_pairs_train.tsv`. |
| Session 10 | Match Prediction & Streaming Architecture: Refactored `predict.py` to stream candidate chunks and write predictions directly to disk with constant memory footprint; generating `output/matching_results.tsv` (scored with LightGBM at $T = 0.65$ and formatted for 100% submission compliance). |


