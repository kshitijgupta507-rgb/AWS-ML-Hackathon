# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Team EntityRes  
**Team Members:** Kshitij Gupta  
**Submission Date:** 2026-09-27  

---

## 1. Executive Summary
We designed and implemented a scalable, high-precision Business Entity Resolution pipeline capable of resolving millions of commercial records across heterogeneous sources without external lookup services. Our solution pairs an ultra-fast **In-Memory Inverted Index Accumulator** for candidate blocking (achieving 89.24% recall across 7.64M ground-truth pairs) with a **LightGBM Gradient Boosted Decision Tree** classifier trained on 10 rich string, token, and numerical similarity features. By optimizing the decision threshold directly against the macro-averaged $F_{0.5}$ metric ($\beta = 0.5$) with strict singleton credit handling, our model achieved a validation pair AUC-ROC of **0.9978** and a macro $F_{0.5}$ score of **0.8630**.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory data analysis of Source 1, 2, and 3 revealed significant challenges:
- **Lexical and Syntactical Noise:** Business names frequently contain corporate suffix abbreviations (`CORP`, `INC`, `LTD`, `PVT`), varying punctuation, and international Unicode scripts (e.g., Devanagari and accented Latin).
- **Address Heterogeneity:** Street types (`ST`, `AVE`, `RD`, `BLVD`), postal codes (ZIP, PIN), and suite numbers vary wildly in formatting. Furthermore, approximately 15% of records have completely missing address fields.
- **Extreme Class Imbalance:** Comparing 2.2M Source 1 entities against ~10.3M Source 2/3 candidates spans a Cartesian space of $>2.2 \times 10^{13}$ pairs, where true matches comprise less than 0.0001% of all combinations.
- **Evaluation Metric Dynamics:** The official macro-averaged $F_{0.5}$ metric penalizes false merges (false positives) twice as severely as missed matches ($\beta = 0.5$) while granting a full 1.0 score to correctly identified singletons (entities with zero matches).

### 2.2 Solution Strategy
**Approach Type:** Hybrid Multi-Stage Pipeline (Vectorized Unicode Normalization $\rightarrow$ In-Memory Inverted Index Blocking $\rightarrow$ Pairwise Feature Engineering $\rightarrow$ Precision-Tuned LightGBM Matching).  
**Core Innovation:**
1. **Single-Pass Vectorized Regex Normalization:** Canonicalizes abbreviations and strips noise across millions of records in seconds using compiled regex alternations rather than row-by-row iteration.
2. **In-Memory Accumulator Blocking (v5):** Employs memory-mapped weight and country arrays with inverted posting lists to accumulate shared token weights via `np.bincount`, producing high-quality candidate pairs without disk I/O bottlenecks.
3. **Threshold Calibration for Macro $F_{0.5}$:** Tunes the decision boundary ($T = 0.65$) to strictly suppress false merges and safely predict empty match sets for singletons.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:**
  - Standardized business name tokens ($\ge 2$ characters, weight = 1.0).
  - Exact address numeric tokens (house numbers, building numbers, PIN/ZIP codes, $\ge 2$ digits, weight = 2.0).
  - High-information address tokens ($\ge 4$ characters, weight = 0.6).
  - Strict exact country partition filtering (`country_norm` equality).
- **Candidate pairs generated:**
  - Up to top-$K = 100$ highest-scoring candidates retained per Source 1 entity above a weighted Jaccard threshold of 0.15.
  - Full training set produced 2.87 GB `candidate_pairs.tsv` (~220M candidate pairs across 2,206,821 S1 entities).
- **How true matches were preserved:**
  - Numeric tokens (house numbers / ZIP codes) were heavily prioritized with a 2.0x weight, preventing entities sharing generic names in different physical locations from crowding out true matches.
  - Verified empirical blocking recall reached **89.24%** on all 7.64M ground-truth matches, with a theoretical ceiling of **99.95%**.

---

## 4. Matching Model

**Features used (10 total):**
- **Name Features:**
  - `name_jaccard`: Jaccard similarity over normalized token sets.
  - `name_jaro_winkler`: RapidFuzz string-level similarity on full canonical names.
  - `name_token_sort_ratio`: Order-invariant fuzzy token match ratio.
- **Address & Numeric Features:**
  - `addr_jaccard`: Jaccard similarity over address tokens.
  - `addr_jaro_winkler`: RapidFuzz string similarity on street/city strings.
  - `addr_num_exact`: Binary indicator (1 if house numbers or ZIP codes exactly overlap, else 0).
- **Structural & Contextual Features:**
  - `country_match`: Exact country equality indicator.
  - `is_addr_missing_s1`: Flag indicating whether Source 1 entity lacked an address.
  - `is_addr_missing_cand`: Flag indicating whether candidate entity lacked an address.
  - `jaccard_score`: Full recomputed multi-field weighted Jaccard score.

**Model type:** LightGBM GBDT (`LGBMClassifier`, 300 estimators, max depth = 7, num leaves = 63, learning rate = 0.05).  
**Threshold selection method:** Grid evaluation over $T \in [0.10, 0.90]$ with step 0.05 directly evaluating the official competition Macro-Averaged $F_{0.5}$ metric on an entity-isolated validation holdout set.

---

## 5. Results & Error Analysis

- **Pair-Level Validation AUC-ROC:** **0.9978**
- **Macro-Averaged $F_{0.5}$ Score:** **0.8630** (achieved at optimal decision threshold $T = 0.65$)
- **Top Feature Importances (Gain/Splits):**
  1. `name_jaro_winkler`: 3,545 splits
  2. `name_token_sort_ratio`: 3,506 splits
  3. `jaccard_score`: 2,814 splits
  4. `addr_jaro_winkler`: 2,763 splits
  5. `addr_jaccard`: 2,742 splits
  6. `name_jaccard`: 2,187 splits
  7. `addr_num_exact`: 593 splits
- **Common false positives (wrong merges):** Entities sharing a franchise or chain name located within the same municipality where street numbers were omitted or slightly ambiguous. The elevated threshold $T = 0.65$ effectively suppresses these.
- **Common false negatives (missed matches):** Entities with extreme spelling corruptions in both name and street address combined with missing postal codes.

---

## 6. Conclusion
The developed solution delivers an end-to-end entity resolution pipeline combining linear-time inverted-index blocking with non-linear gradient-boosted pairwise matching. Achieving a 0.9978 AUC-ROC and an 0.8630 macro $F_{0.5}$ on high-dimensional candidate pairs demonstrates that domain-specific normalization and precision-calibrated thresholds produce state-of-the-art matching accuracy while maintaining strict memory and computational efficiency.

---

## Appendix

### A. Code Artefacts
All reproducible source code is organized within the self-contained directory `student_resource/src/`:
- `normalize.py`: Phase 1 vectorized normalization and abbreviation expansion.
- `blocking.py`: Phase 2 v5 in-memory accumulator candidate generation (`candidate_pairs.tsv`).
- `features.py`: Phase 3 10-feature pairwise similarity extraction with parquet caching.
- `train.py`: Phase 4 LightGBM training, entity cross-validation, and $F_{0.5}$ threshold optimization (`model.joblib`).
- `predict.py`: Phase 4 candidate scoring, singleton handling, and output generation (`matching_results.tsv`).
- `requirements.txt`: Pinned dependency specification.
- `utils/validate_submission.py`: Official standalone validation utility.

### B. Additional Results
Threshold search trajectory on validation set:
| Threshold | Macro $F_{0.5}$ | Precision / Recall Behavior |
|:---:|:---:|:---|
| 0.30 | 0.8243 | High recall, modest false merges |
| 0.50 | 0.8565 | Standard classification boundary |
| **0.65** | **0.8630** | **Optimal F0.5: Maximum precision on ambiguous candidates** |
| 0.75 | 0.8576 | Highly conservative; drops minor edge-case true matches |
| 0.90 | 0.8230 | Severe under-prediction |
