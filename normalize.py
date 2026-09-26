"""
normalize.py  --  Phase 1: Vectorized, Unicode-safe normalization.

Design principles:
  - Fully vectorized pandas .str operations - NO row-by-row .apply() at scale.
  - Unicode-aware regex preserves Devanagari, accented Latin (French), etc.
  - Deterministic and stateless: same function called on train AND test.
  - No per-country hardcoding - generalizes to US, India, France, and beyond.

Entry point:
  normalize_sources(df) -> df with added columns:
    name_norm       : cleaned/expanded name string
    name_tokens     : list of tokens from name_norm (for blocking)
    addr_norm       : cleaned/expanded address string
    addr_tokens     : list of tokens from addr_norm (for blocking)
    addr_nums       : space-separated numeric substrings from address
    is_addr_missing : bool flag set BEFORE filling NaN (important for features)
"""

import re
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Legal suffix expansion map
#    Keys are regex patterns (word-boundary anchored).
#    Applied in a single vectorized .replace() pass over the whole column.
# ---------------------------------------------------------------------------
NAME_SUFFIX_MAP = {
    r"\bpvt\b":  "private",
    r"\bltd\b":  "limited",
    r"\bcorp\b": "corporation",
    r"\binc\b":  "incorporated",
    r"\bllp\b":  "limited liability partnership",
    r"\bllc\b":  "limited liability company",
    r"\bco\b":   "company",
    r"\bplc\b":  "public limited company",
    r"\bintl\b": "international",
    # Note: "&" and "+" are expanded BEFORE punctuation removal in normalize_name;
    # they are NOT listed here to avoid double-processing.
}

# ---------------------------------------------------------------------------
# 2. Address abbreviation expansion map
# ---------------------------------------------------------------------------
ADDR_ABBR_MAP = {
    r"\brd\b":   "road",
    r"\bst\b":   "street",
    r"\bave\b":  "avenue",
    r"\bapt\b":  "apartment",
    r"\bapts\b": "apartments",
    r"\bblvd\b": "boulevard",
    r"\bdr\b":   "drive",
    r"\bln\b":   "lane",
    r"\bct\b":   "court",
    r"\bpl\b":   "place",
    r"\bsq\b":   "square",
    r"\bhwy\b":  "highway",
    r"\bfwy\b":  "freeway",
    r"\bpkwy\b": "parkway",
    r"\bste\b":  "suite",
    r"\bflr\b":  "floor",
    r"\bbldg\b": "building",
    r"\bdept\b": "department",
}

_NUM_RE = re.compile(r"\d+")


def _expand_abbreviations(series: pd.Series, mapping: dict) -> pd.Series:
    """Apply a dict of {regex_pattern: replacement} to a string Series."""
    for pattern, replacement in mapping.items():
        series = series.str.replace(pattern, replacement, regex=True)
    return series


def normalize_name(df: pd.DataFrame, col: str = "business_name") -> pd.DataFrame:
    """
    Normalize business name column.

    Steps:
    1. Fill NaN with empty string.
    2. Lowercase.
    3. Strip punctuation with Unicode-aware regex - preserves Hindi, French, etc.
    4. Collapse whitespace.
    5. Expand legal-suffix abbreviations via word-boundary regex replace.
    6. Final strip.
    7. Tokenize into name_tokens list.

    Adds columns: name_norm, name_tokens
    """
    s = df[col].fillna("")
    s = s.str.lower()
    # Expand & and + to "and" BEFORE the punctuation regex strips them.
    # These characters are non-word chars and would be removed by [^\w\s] below
    # before NAME_SUFFIX_MAP ever runs if we did not handle them here first.
    s = s.str.replace(r"&amp;", "and", regex=True)  # HTML-encoded ampersand (defensive)
    s = s.str.replace(r"&", " and ", regex=False)
    s = s.str.replace(r"\+", " and ", regex=True)
    # Unicode-aware: [^\w\s] removes remaining punctuation without touching
    # Devanagari, accented Latin (French), or Unicode combining marks.
    s = s.str.replace(r"[^\w\s]", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    # Expand word-boundary legal suffixes (pvt, ltd, corp, inc, llc, etc.)
    s = _expand_abbreviations(s, NAME_SUFFIX_MAP)
    s = s.str.strip()

    df = df.copy()
    df["name_norm"] = s
    df["name_tokens"] = (
        df["name_norm"]
        .str.split()
        .apply(lambda tokens: [t for t in tokens if t] if isinstance(tokens, list) else [])
    )
    return df


def normalize_address(df: pd.DataFrame, col: str = "business_address") -> pd.DataFrame:
    """
    Normalize business address column.

    Steps:
    1. Flag is_addr_missing BEFORE filling NaN (this boolean is a useful feature).
    2. Fill NaN with empty string.
    3. Lowercase, Unicode-aware punctuation strip, collapse whitespace.
    4. Expand address abbreviations.
    5. Extract numeric substrings into addr_nums (PINs, house numbers, ZIPs).
    6. Tokenize into addr_tokens list.

    Adds columns: is_addr_missing, addr_norm, addr_nums, addr_tokens
    """
    df = df.copy()
    df["is_addr_missing"] = df[col].isna()  # MUST come before fillna

    s = df[col].fillna("")
    s = s.str.lower()
    s = s.str.replace(r"[^\w\s]", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    s = _expand_abbreviations(s, ADDR_ABBR_MAP)
    s = s.str.strip()

    df["addr_norm"] = s
    df["addr_nums"] = df["addr_norm"].apply(
        lambda x: " ".join(_NUM_RE.findall(x)) if isinstance(x, str) else ""
    )
    df["addr_tokens"] = (
        df["addr_norm"]
        .str.split()
        .apply(lambda tokens: [t for t in tokens if t] if isinstance(tokens, list) else [])
    )
    return df


def normalize_country(df: pd.DataFrame, col: str = "country") -> pd.DataFrame:
    """
    Lowercase + strip the country column for consistent matching.
    Adds column: country_norm
    """
    df = df.copy()
    df["country_norm"] = df[col].fillna("").str.lower().str.strip()
    return df


def normalize_sources(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master normalization entry point.

    Applies name, address, and country normalization to a source DataFrame.
    Expects columns: entity_id, business_name, business_address, country.

    Returns the input DataFrame with additional columns:
      name_norm, name_tokens,
      is_addr_missing, addr_norm, addr_nums, addr_tokens,
      country_norm
    """
    df = normalize_name(df)
    df = normalize_address(df)
    df = normalize_country(df)
    return df


# ---------------------------------------------------------------------------
# Sanity check - run directly to verify on a tiny synthetic example
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    # --- Correctness test ---
    sample = pd.DataFrame({
        "entity_id": ["S1-001", "S1-002", "S1-003", "S1-004", "S1-005"],
        "business_name": [
            "Reliance Pvt Ltd",
            "Apple Corp & Associates",
            "Mumbai Bazar",
            None,
            "http://www.junkurl.com",
        ],
        "business_address": [
            "221B Baker St, London",
            "1 Infinite Loop, Cupertino, CA 95014",
            None,
            "Apt 4B, 12 rue de la Paix, Paris",
            "Road No. 5, Banjara Hills, Hyderabad",
        ],
        "country": ["India", "US", "India", "France", "India"],
    })

    result = normalize_sources(sample)

    print("\n=== Normalization Correctness Test ===")
    cols = ["entity_id", "name_norm", "name_tokens", "is_addr_missing",
            "addr_norm", "addr_nums", "addr_tokens", "country_norm"]
    for _, row in result[cols].iterrows():
        print(f"\nID: {row['entity_id']}")
        print(f"  name_norm      : {row['name_norm']!r}")
        print(f"  name_tokens    : {row['name_tokens']}")
        print(f"  is_addr_missing: {row['is_addr_missing']}")
        print(f"  addr_norm      : {row['addr_norm']!r}")
        print(f"  addr_nums      : {row['addr_nums']!r}")
        print(f"  addr_tokens    : {row['addr_tokens']}")
        print(f"  country_norm   : {row['country_norm']!r}")

    # --- Speed test on real data (first 100k rows of source1 train) ---
    print("\n=== Speed Test: 100k rows from train_source1.tsv ===")
    DATA_PATH = "dataset/train/train_source1.tsv"
    t0 = time.time()
    df_s1 = pd.read_csv(DATA_PATH, sep="\t", nrows=100_000)
    t1 = time.time()
    print(f"  Load time : {t1 - t0:.2f}s  |  shape: {df_s1.shape}")
    df_normed = normalize_sources(df_s1)
    t2 = time.time()
    print(f"  Norm time : {t2 - t1:.2f}s  |  output shape: {df_normed.shape}")
    print(f"  Columns   : {list(df_normed.columns)}")
    print(f"  Sample name_norm  : {df_normed['name_norm'].iloc[:3].tolist()}")
    print(f"  Sample name_tokens: {df_normed['name_tokens'].iloc[:3].tolist()}")
    print(f"  Addr missing rate : {df_normed['is_addr_missing'].mean():.3%}")
