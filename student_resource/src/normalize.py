"""
normalize.py  --  Phase 1: Vectorized, Unicode-safe normalization.

Design principles:
  - Fully vectorized pandas .str operations - NO row-by-row .apply() at scale.
  - Unicode-aware regex preserves Devanagari, accented Latin (French), etc.
  - Deterministic and stateless: same function called on train AND test.
  - No per-country hardcoding - generalizes to US, India, France, and beyond.

Performance optimization (v2):
  - Abbreviation expansion uses a SINGLE compiled alternation regex + re.sub
    lookup function instead of N sequential .str.replace() passes.
    e.g., 10 name rules + 17 addr rules -> 2 regex passes total (10-17x faster).
  - Token extraction uses str.findall(r'\w{2,}') — vectorized, no .apply().

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
# 1. Legal suffix expansion map  (word: replacement)
#    Keys are plain words — combined into one alternation regex at module load.
# ---------------------------------------------------------------------------
_NAME_SUFFIX_RAW = {
    "pvt":  "private",
    "ltd":  "limited",
    "corp": "corporation",
    "inc":  "incorporated",
    "llp":  "limited liability partnership",
    "llc":  "limited liability company",
    "co":   "company",
    "plc":  "public limited company",
    "intl": "international",
    # "&" and "+" are expanded inline before punctuation removal (see normalize_name)
}

# ---------------------------------------------------------------------------
# 2. Address abbreviation expansion map  (word: replacement)
# ---------------------------------------------------------------------------
_ADDR_ABBR_RAW = {
    "rd":   "road",
    "st":   "street",
    "ave":  "avenue",
    "apt":  "apartment",
    "apts": "apartments",
    "blvd": "boulevard",
    "dr":   "drive",
    "ln":   "lane",
    "ct":   "court",
    "pl":   "place",
    "sq":   "square",
    "hwy":  "highway",
    "fwy":  "freeway",
    "pkwy": "parkway",
    "ste":  "suite",
    "flr":  "floor",
    "bldg": "building",
    "dept": "department",
}


def _make_expander(raw_map: dict):
    """
    Compile a single alternation regex from a word->replacement dict and
    return a vectorized expander function (Series -> Series).

    This replaces N sequential .str.replace() calls with ONE regex pass,
    giving ~N× speedup (10-17x for our maps).

    The compiled pattern is:  \\b(pvt|ltd|corp|...)\\b
    and the replacement uses a re.sub lookup into the dict.
    """
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(k) for k in raw_map) + r")\b"
    )
    # Pre-build a plain-str lookup (keys already lowercased)
    lookup = raw_map

    def _repl(m: re.Match) -> str:
        return lookup.get(m.group(0), m.group(0))

    def expand(series: pd.Series) -> pd.Series:
        """Apply the compiled abbreviation regex to every string in the Series."""
        return series.str.replace(pattern, _repl, regex=True)

    return expand


# Module-level compiled expanders (built once at import time)
_expand_name_suffixes = _make_expander(_NAME_SUFFIX_RAW)
_expand_addr_abbrs    = _make_expander(_ADDR_ABBR_RAW)

# Keep public dicts for backward-compat references
NAME_SUFFIX_MAP = {rf"\b{k}\b": v for k, v in _NAME_SUFFIX_RAW.items()}
ADDR_ABBR_MAP   = {rf"\b{k}\b": v for k, v in _ADDR_ABBR_RAW.items()}

_NUM_RE = re.compile(r"\d+")


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
    # Expand word-boundary legal suffixes using single compiled alternation regex
    s = _expand_name_suffixes(s)
    s = s.str.strip()

    df = df.copy()
    df["name_norm"] = s
    # Vectorized tokenization: str.findall(r'\w{2,}') returns list of tokens
    # that are >= 2 chars (skips single-char noise), no .apply() needed.
    df["name_tokens"] = df["name_norm"].str.findall(r"\w{2,}")
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
    s = _expand_addr_abbrs(s)
    s = s.str.strip()

    df["addr_norm"] = s
    # Vectorized numeric extraction and tokenization (no .apply())
    df["addr_nums"]   = df["addr_norm"].str.findall(r"\d+").str.join(" ")
    df["addr_tokens"] = df["addr_norm"].str.findall(r"\w{2,}")
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
