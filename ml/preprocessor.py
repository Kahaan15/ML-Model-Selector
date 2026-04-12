"""
preprocessor.py
===============
Ultimate, plug-and-play preprocessing for any numerical CSV dataset.
Handles every edge case. Zero manual intervention needed after this.

Usage:
    from preprocessor import preprocess

    result = preprocess("your_dataset.csv", target_column="price")
    # or for auto-detected target:
    result = preprocess("your_dataset.csv")

Returns a PreprocessResult dataclass with everything the rest of the pipeline needs.
"""

import os
import warnings
import numpy as np
import pandas as pd

from dataclasses import dataclass, field
from typing import Optional, Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# CONSTANTS  (tweak here if ever needed)
# ─────────────────────────────────────────────
TRAIN_TEST_SPLIT_RATIO   = 0.20          # 80 / 20 split
RANDOM_STATE             = 42
MIN_ROWS_AFTER_CLEAN     = 10            # refuse to train on tiny leftovers
MAX_FEATURES_ALLOWED     = 50            # guard against accidental wide files
CLASSIFICATION_THRESHOLD = 15           # ≤ this many unique int values → classification
HIGH_CARDINALITY_LIMIT   = 0.95         # drop a categorical col if unique% > this
OUTLIER_IQR_MULTIPLIER   = 3.0          # very conservative: only removes extreme outliers
NEAR_ZERO_VAR_THRESHOLD  = 1e-6         # columns with variance below this are dropped
CORR_DROP_THRESHOLD      = 0.98         # drop one of a perfectly-correlated pair


# ─────────────────────────────────────────────
# RESULT CONTAINER
# ─────────────────────────────────────────────
@dataclass
class PreprocessResult:
    # Core train/test arrays (ready for sklearn)
    X_train: np.ndarray
    X_test:  np.ndarray
    y_train: np.ndarray
    y_test:  np.ndarray

    # Metadata the rest of the pipeline needs
    task_type:        str               # "regression" | "classification"
    feature_names:    list
    target_column:    str
    n_classes:        int               # 0 for regression
    class_labels:     list              # [] for regression
    class_balance:    dict              # {} for regression

    # Fitted objects (needed if we want to inverse-transform predictions later)
    scaler:           StandardScaler
    label_encoder:    Optional[LabelEncoder]   # only for classification targets

    # Audit log shown to the user in the UI
    log:              list = field(default_factory=list)

    # Raw shape info
    original_shape:   Tuple[int, int] = (0, 0)
    final_shape:      Tuple[int, int] = (0, 0)


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────
def preprocess(
    source,                              # filepath str OR a pandas DataFrame
    target_column: Optional[str] = None,
    task_type:     Optional[str] = None, # "regression" | "classification" | None=auto
    test_size:     float = TRAIN_TEST_SPLIT_RATIO,
    random_state:  int   = RANDOM_STATE,
) -> PreprocessResult:
    """
    Full preprocessing pipeline. Returns a PreprocessResult ready for training.

    Parameters
    ----------
    source        : str path to CSV, or a pandas DataFrame already in memory.
    target_column : name of the target column. If None, last column is used.
    task_type     : force "regression" or "classification". None = auto-detect.
    test_size     : fraction for test split (default 0.20).
    random_state  : for reproducibility.
    """

    log = []

    # ── 1. LOAD ──────────────────────────────────────────────────────────────
    df = _load(source, log)
    original_shape = df.shape
    log.append(f"✅ Loaded dataset: {original_shape[0]} rows × {original_shape[1]} columns.")

    # ── 2. IDENTIFY TARGET ────────────────────────────────────────────────────
    target_column = _resolve_target(df, target_column, log)

    # ── 3. BASIC STRUCTURAL FIXES ─────────────────────────────────────────────
    df = _fix_column_names(df, log)
    target_column = _safe_col_name(target_column)   # match cleaned name

    # ── 4. DROP USELESS COLUMNS ───────────────────────────────────────────────
    df = _drop_id_like_columns(df, target_column, log)
    df = _drop_non_numeric_high_cardinality(df, target_column, log)

    # ── 5. ENCODE REMAINING CATEGORICAL FEATURES ──────────────────────────────
    df = _encode_low_cardinality_categoricals(df, target_column, log)

    # ── 6. HANDLE MISSING VALUES ──────────────────────────────────────────────
    df = _handle_missing(df, target_column, log)

    # ── 6b. FINAL CATCH — force-encode any remaining object columns ──────────
    df = _force_encode_remaining(df, target_column, log)

    # ── 7. ENFORCE FEATURE LIMIT ──────────────────────────────────────────────
    df = _enforce_feature_limit(df, target_column, log)

    # ── 8. REMOVE DUPLICATE ROWS ──────────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed:
        log.append(f"🔁 Removed {removed} duplicate rows.")

    # ── 9. DROP NEAR-ZERO VARIANCE FEATURES ───────────────────────────────────
    df = _drop_near_zero_variance(df, target_column, log)

    # ── 10. DROP HIGHLY CORRELATED FEATURES ───────────────────────────────────
    df = _drop_highly_correlated(df, target_column, log)

    # ── 11. DETECT / VALIDATE TASK TYPE ───────────────────────────────────────
    task_type, n_classes, class_labels = _detect_task(df, target_column, task_type, log)

    # ── 12. OUTLIER CLIPPING (features only, regression only) ─────────────────
    if task_type == "regression":
        df = _clip_outliers(df, target_column, log)

    # ── 13. SPLIT X / y ───────────────────────────────────────────────────────
    X = df.drop(columns=[target_column])
    y = df[target_column].copy()
    feature_names = list(X.columns)
    class_balance = {}

    # ── 14. ENCODE TARGET FOR CLASSIFICATION ──────────────────────────────────
    label_encoder = None
    if task_type == "classification":
        label_counts = y.astype(str).value_counts()
        class_balance = {
            str(label): round(float(count / len(y)), 4)
            for label, count in label_counts.items()
        }

        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y.astype(str)), name=target_column)
        label_encoder = le
        log.append(f"🏷️  Target encoded: {list(le.classes_)} → {list(range(len(le.classes_)))}")

    # ── 15. SANITY CHECK ──────────────────────────────────────────────────────
    _sanity_check(X, y, log)

    # ── 16. TRAIN / TEST SPLIT ────────────────────────────────────────────────
    # Stratify only for classification AND only if each class has ≥2 samples
    if task_type == "classification":
        class_counts = y.value_counts()
        stratify = y if (class_counts >= 2).all() else None
    else:
        stratify = None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    log.append(f"✂️  Train/test split: {len(X_train)} train / {len(X_test)} test  ({int((1-test_size)*100)}/{int(test_size*100)})")

    # ── 17. FEATURE SCALING (fit on train, transform both) ────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    log.append(f"📐 Features standardised (StandardScaler fit on training set only).")

    # ── 18. FINAL SUMMARY ─────────────────────────────────────────────────────
    final_shape = (len(df), len(feature_names) + 1)
    log.append(f"🎯 Task type : {task_type.upper()}")
    log.append(f"🎯 Target    : '{target_column}'")
    log.append(f"📊 Features  : {len(feature_names)}  →  {feature_names}")
    log.append(f"📦 Final dataset: {final_shape[0]} rows × {final_shape[1]} columns (incl. target).")

    return PreprocessResult(
        X_train        = X_train_scaled,
        X_test         = X_test_scaled,
        y_train        = y_train.to_numpy(),
        y_test         = y_test.to_numpy(),
        task_type      = task_type,
        feature_names  = feature_names,
        target_column  = target_column,
        n_classes      = n_classes,
        class_labels   = class_labels,
        class_balance  = class_balance,
        scaler         = scaler,
        label_encoder  = label_encoder,
        log            = log,
        original_shape = original_shape,
        final_shape    = final_shape,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _load(source, log) -> pd.DataFrame:
    """Accept a filepath string, bytes-like object, or an existing DataFrame."""
    if isinstance(source, pd.DataFrame):
        return source.copy()

    if isinstance(source, (str, os.PathLike)):
        ext = os.path.splitext(str(source))[-1].lower()
        if ext == ".csv":
            # Try comma, then semicolon, then tab separator
            for sep in [",", ";", "\t"]:
                try:
                    df = pd.read_csv(source, sep=sep)
                    if df.shape[1] > 1:
                        return df
                except Exception:
                    continue
            raise ValueError("Could not parse CSV with common separators (, ; tab).")
        elif ext in (".xls", ".xlsx"):
            return pd.read_excel(source)
        else:
            raise ValueError(f"Unsupported file type: {ext}. Please upload a CSV.")

    # Fallback: treat as file-like object
    return pd.read_csv(source)


def _safe_col_name(col: str) -> str:
    """Match the column name transformation done in _fix_column_names."""
    return col.strip().replace(" ", "_").replace("-", "_").lower()


def _fix_column_names(df: pd.DataFrame, log) -> pd.DataFrame:
    """Lowercase, strip, replace spaces/dashes with underscores."""
    original = list(df.columns)
    df.columns = [c.strip().replace(" ", "_").replace("-", "_").lower() for c in df.columns]
    changed = [(o, n) for o, n in zip(original, df.columns) if o != n]
    if changed:
        log.append(f"🔤 Renamed {len(changed)} column(s) for consistency (e.g. '{changed[0][0]}' → '{changed[0][1]}').")
    return df


def _resolve_target(df: pd.DataFrame, target_column, log) -> str:
    if target_column is None:
        target_column = df.columns[-1]
        log.append(f"🎯 No target specified — using last column: '{target_column}'.")
    else:
        # Be forgiving: try exact match first, then case-insensitive
        cleaned = _safe_col_name(target_column)
        cols_cleaned = [_safe_col_name(c) for c in df.columns]
        if target_column not in df.columns:
            if cleaned in cols_cleaned:
                target_column = df.columns[cols_cleaned.index(cleaned)]
            else:
                raise ValueError(
                    f"Target column '{target_column}' not found. "
                    f"Available columns: {list(df.columns)}"
                )
    return target_column


def _is_string_col(series: pd.Series) -> bool:
    """Return True for both legacy object dtype and pandas 2.x StringDtype."""
    return series.dtype == object or isinstance(series.dtype, pd.StringDtype) or str(series.dtype) == "str"


def _force_encode_remaining(df: pd.DataFrame, target: str, log) -> pd.DataFrame:
    """Last-resort: label-encode any object columns still left (excluding target)."""
    remaining = [c for c in df.columns if c != target and _is_string_col(df[c])]
    for col in remaining:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    if remaining:
        log.append(f"🔢 Force-encoded remaining object columns: {remaining}")
    return df


def _drop_id_like_columns(df: pd.DataFrame, target: str, log) -> pd.DataFrame:
    """
    Drop columns that are almost certainly ID / index columns:
      - Column named 'id', 'index', 'row_id', 'unnamed:_0', etc.
      - Integer column where every value is unique (and it's not the target).
    """
    to_drop = []
    id_keywords = {"id", "index", "row", "unnamed", "rowid", "row_id", "no", "num", "number", "sr", "sno"}

    for col in df.columns:
        if col == target:
            continue
        name_lower = col.lower().replace("_", "").replace(".", "")
        if any(name_lower == kw or name_lower.endswith(kw) or name_lower.startswith(kw)
               for kw in id_keywords):
            to_drop.append(col)
            continue
        # Monotonically increasing integers that are unique → index-like
        if pd.api.types.is_integer_dtype(df[col]):
            if df[col].nunique() == len(df) and df[col].is_monotonic_increasing:
                to_drop.append(col)

    if to_drop:
        df = df.drop(columns=to_drop)
        log.append(f"🗑️  Dropped ID-like columns: {to_drop}")
    return df


def _drop_non_numeric_high_cardinality(df: pd.DataFrame, target: str, log) -> pd.DataFrame:
    """Drop text columns that have too many unique values to be useful."""
    to_drop = []
    for col in df.columns:
        if col == target:
            continue
        if _is_string_col(df[col]) or str(df[col].dtype) == "category":
            unique_ratio = df[col].nunique() / max(len(df), 1)
            if unique_ratio > HIGH_CARDINALITY_LIMIT:
                to_drop.append(col)
    if to_drop:
        df = df.drop(columns=to_drop)
        log.append(f"🗑️  Dropped high-cardinality text columns (>{int(HIGH_CARDINALITY_LIMIT*100)}% unique): {to_drop}")
    return df


def _encode_low_cardinality_categoricals(df: pd.DataFrame, target: str, log) -> pd.DataFrame:
    """
    Encode remaining categorical/object feature columns with label encoding.
    These are low-cardinality columns that carry real signal (e.g. 'gender', 'region').
    """
    encoded = []
    for col in df.columns:
        if col == target:
            continue
        if _is_string_col(df[col]) or str(df[col].dtype) == "category":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoded.append(col)
    if encoded:
        log.append(f"🔢 Label-encoded low-cardinality categorical features: {encoded}")
    return df


def _handle_missing(df: pd.DataFrame, target: str, log) -> pd.DataFrame:
    """
    Strategy:
      - Target column: drop rows with missing target (can't train without ground truth).
      - Numeric features: fill with median (robust to outliers, better than mean).
      - Categorical features: fill with mode.
      - Columns with >60% missing: drop entirely.
    """
    # Drop columns with >60% missing first
    threshold = 0.60
    missing_ratio = df.isnull().mean()
    to_drop = [c for c in df.columns if c != target and missing_ratio[c] > threshold]
    if to_drop:
        df = df.drop(columns=to_drop)
        log.append(f"🗑️  Dropped columns with >{int(threshold*100)}% missing values: {to_drop}")

    # Drop rows where target is missing
    before = len(df)
    df = df.dropna(subset=[target])
    dropped = before - len(df)
    if dropped:
        log.append(f"🗑️  Dropped {dropped} rows with missing target values.")

    # Fill remaining missing values
    total_filled = 0
    for col in df.columns:
        if col == target:
            continue
        n_missing = df[col].isnull().sum()
        if n_missing == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "unknown")
        total_filled += n_missing

    if total_filled:
        log.append(f"🩹 Imputed {total_filled} missing value(s) across feature columns (median for numeric, mode for categorical).")

    return df


def _enforce_feature_limit(df: pd.DataFrame, target: str, log) -> pd.DataFrame:
    """
    If there are too many features, keep the top N most correlated with the target.
    This prevents memory blowup and keeps the bias-variance visualisation clean.
    """
    features = [c for c in df.columns if c != target]
    if len(features) <= MAX_FEATURES_ALLOWED:
        return df

    # Compute absolute correlation with target (works for numeric target)
    if pd.api.types.is_numeric_dtype(df[target]):
        corr = df[features].corrwith(df[target]).abs().sort_values(ascending=False)
        keep = list(corr.head(MAX_FEATURES_ALLOWED).index)
    else:
        # For categorical target just keep first MAX_FEATURES_ALLOWED
        keep = features[:MAX_FEATURES_ALLOWED]

    dropped = [c for c in features if c not in keep]
    df = df[[target] + keep] if target in df.columns else df[keep]
    # Preserve target in df
    log.append(
        f"✂️  Too many features ({len(features)}). "
        f"Kept top {MAX_FEATURES_ALLOWED} most correlated with target. "
        f"Dropped: {dropped[:5]}{'...' if len(dropped) > 5 else ''}"
    )
    return df


def _drop_near_zero_variance(df: pd.DataFrame, target: str, log) -> pd.DataFrame:
    """Drop feature columns where all (or almost all) values are the same."""
    to_drop = []
    for col in df.columns:
        if col == target:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].var() < NEAR_ZERO_VAR_THRESHOLD:
                to_drop.append(col)
    if to_drop:
        df = df.drop(columns=to_drop)
        log.append(f"🗑️  Dropped near-zero variance columns (constant or near-constant): {to_drop}")
    return df


def _drop_highly_correlated(df: pd.DataFrame, target: str, log) -> pd.DataFrame:
    """
    Drop one column from any pair of features with Pearson |r| > CORR_DROP_THRESHOLD.
    Keeps the one with higher correlation to the target.
    """
    numeric_features = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]
    if len(numeric_features) < 2:
        return df

    corr_matrix = df[numeric_features].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = set()
    for col in upper.columns:
        highly_correlated_with = upper.index[upper[col] > CORR_DROP_THRESHOLD].tolist()
        for pair in highly_correlated_with:
            # Keep the one more correlated with target
            if pd.api.types.is_numeric_dtype(df[target]):
                corr_col  = abs(df[col].corr(df[target]))
                corr_pair = abs(df[pair].corr(df[target]))
                drop_candidate = col if corr_col < corr_pair else pair
            else:
                drop_candidate = col   # arbitrary tiebreak
            to_drop.add(drop_candidate)

    to_drop = list(to_drop)
    if to_drop:
        df = df.drop(columns=to_drop)
        log.append(f"🔗 Dropped {len(to_drop)} highly correlated feature(s) (|r| > {CORR_DROP_THRESHOLD}): {to_drop}")
    return df


def _detect_task(df, target, task_type_override, log):
    """
    Auto-detect regression vs classification from the target column.
    Rules:
      - If task_type_override is given, trust it.
      - Else: if target is float → regression.
      - If target is int/object with ≤ CLASSIFICATION_THRESHOLD unique values → classification.
      - Otherwise → regression.
    """
    col = df[target]
    n_unique = col.nunique()
    class_labels = []
    n_classes = 0

    if task_type_override is not None:
        task_type = task_type_override.lower()
        assert task_type in ("regression", "classification"), \
            "task_type must be 'regression' or 'classification'"
        log.append(f"⚙️  Task type manually set to: {task_type.upper()}")
    else:
        is_float_col   = pd.api.types.is_float_dtype(col)
        is_int_col     = pd.api.types.is_integer_dtype(col)
        is_object_col  = _is_string_col(col)

        if is_float_col and n_unique > CLASSIFICATION_THRESHOLD:
            task_type = "regression"
        elif _is_string_col(col):
            task_type = "classification"
        elif is_int_col and n_unique <= CLASSIFICATION_THRESHOLD:
            task_type = "classification"
        elif is_int_col and (n_unique / max(len(col), 1)) < 0.05:
            # Many rows but very few unique int values → classification (e.g. 0/1, 1-5 ratings)
            task_type = "classification"
        else:
            task_type = "regression"

        log.append(
            f"🤖 Auto-detected task type: {task_type.upper()} "
            f"(target '{target}' has {n_unique} unique values, dtype={col.dtype})."
        )

    if task_type == "classification":
        class_labels = sorted(col.dropna().unique().tolist())
        n_classes    = len(class_labels)
        log.append(f"🏷️  Classes found: {class_labels} ({n_classes} total)")
        if n_classes < 2:
            raise ValueError(f"Classification requires at least 2 classes. Only found: {class_labels}")

    return task_type, n_classes, class_labels


def _clip_outliers(df: pd.DataFrame, target: str, log) -> pd.DataFrame:
    """
    Conservative IQR-based clipping for numeric feature columns only.
    Uses multiplier=3.0 so only extreme outliers are touched.
    Does NOT touch the target column.
    """
    clipped_cols = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if col == target:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - OUTLIER_IQR_MULTIPLIER * IQR
        upper = Q3 + OUTLIER_IQR_MULTIPLIER * IQR
        n_clipped = ((df[col] < lower) | (df[col] > upper)).sum()
        if n_clipped > 0:
            df[col] = df[col].clip(lower=lower, upper=upper)
            clipped_cols.append(f"{col}({n_clipped})")

    if clipped_cols:
        log.append(f"✂️  Clipped extreme outliers (IQR×{OUTLIER_IQR_MULTIPLIER}) in: {clipped_cols}")
    return df


def _sanity_check(X, y, log):
    """Final guardrails before we train anything."""
    if len(X) < MIN_ROWS_AFTER_CLEAN:
        raise ValueError(
            f"Only {len(X)} rows remain after preprocessing. "
            f"Minimum required: {MIN_ROWS_AFTER_CLEAN}. "
            "Please provide a larger dataset."
        )
    if X.shape[1] == 0:
        raise ValueError(
            "No feature columns remain after preprocessing. "
            "Check that your CSV has at least one non-target column."
        )
    if y.isnull().any():
        raise ValueError("Target column still contains NaN values after cleaning. Cannot train.")

    log.append(f"✅ Sanity check passed: {X.shape[0]} rows, {X.shape[1]} features, 0 nulls.")


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST  (run: python preprocessor.py)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import io

    # ── Regression test ──────────────────────────────────────────────────────
    reg_csv = """id,age,salary,experience,years_education,department,target_income
1,25,50000,2,16,Engineering,55000
2,32,75000,8,18,Marketing,80000
3,28,60000,5,16,Engineering,65000
4,45,95000,20,20,Management,100000
5,38,85000,14,18,Engineering,90000
6,29,62000,6,16,Marketing,67000
7,52,110000,28,22,Management,115000
8,35,78000,11,18,Engineering,83000
9,26,53000,3,16,Marketing,58000
10,41,92000,17,20,Engineering,97000
11,33,72000,9,18,Management,77000
12,27,56000,4,16,Engineering,61000
13,48,105000,24,20,Marketing,110000
14,36,81000,12,18,Engineering,86000
15,30,67000,7,16,Management,72000"""

    print("=" * 60)
    print("TEST 1: REGRESSION")
    print("=" * 60)
    result = preprocess(io.StringIO(reg_csv), target_column="target_income")
    for line in result.log:
        print(line)
    print(f"\nX_train shape : {result.X_train.shape}")
    print(f"X_test  shape : {result.X_test.shape}")
    print(f"Features      : {result.feature_names}")
    print(f"Task          : {result.task_type}")

    # ── Classification test ──────────────────────────────────────────────────
    clf_csv = """sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,setosa
4.9,3.0,1.4,0.2,setosa
6.7,3.1,4.7,1.5,versicolor
6.3,3.3,6.0,2.5,virginica
5.8,2.7,5.1,1.9,virginica
5.7,2.8,4.5,1.3,versicolor
6.4,3.2,4.5,1.5,versicolor
5.2,3.5,1.5,0.2,setosa
7.7,3.8,6.7,2.2,virginica
5.5,2.4,3.8,1.1,versicolor
4.6,3.1,1.5,0.2,setosa
6.9,3.1,5.1,2.3,virginica
5.0,3.4,1.5,0.2,setosa
6.1,2.9,4.7,1.4,versicolor
7.2,3.2,6.0,1.8,virginica
5.4,3.7,1.5,0.2,setosa
6.5,3.0,5.5,1.8,virginica
5.6,2.9,3.6,1.3,versicolor
4.8,3.4,1.6,0.2,setosa
7.4,2.8,6.1,1.9,virginica"""

    print("\n" + "=" * 60)
    print("TEST 2: CLASSIFICATION")
    print("=" * 60)
    result2 = preprocess(io.StringIO(clf_csv), target_column="species")
    for line in result2.log:
        print(line)
    print(f"\nX_train shape : {result2.X_train.shape}")
    print(f"X_test  shape : {result2.X_test.shape}")
    print(f"Classes       : {result2.class_labels}")
    print(f"Task          : {result2.task_type}")
