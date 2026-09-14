"""Leakage-safe feature typing, splits, and sklearn preprocessors."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

MAX_CATEGORIES = 50
TFIDF_MAX_FEATURES = 300

_ISO = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}(?:[ T]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?$")
_YMD = re.compile(r"^(\d{4})[/-](\d{1,2})[/-](\d{1,2})(?:$|[ T])")
_DMY = re.compile(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})(?:$|[ T])")


class FrequencyCap(BaseEstimator, TransformerMixin):
    """Keep the top `max_categories` values per column; map the rest (and unseen) to other."""

    def __init__(self, max_categories: int = MAX_CATEGORIES, other: str = "__other__"):
        self.max_categories = max_categories
        self.other = other

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=object)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        self.n_features_in_ = arr.shape[1]
        keep: list[set[str]] = []
        for i in range(arr.shape[1]):
            col = pd.Series(arr[:, i]).map(_as_token)
            vc = col.value_counts()
            keep.append(set(vc.head(self.max_categories).index.tolist()))
        self.keep_ = keep
        return self

    def transform(self, X):
        check_is_fitted(self, "keep_")
        arr = np.asarray(X, dtype=object)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        out = np.empty(arr.shape, dtype=object)
        for i in range(arr.shape[1]):
            keep = self.keep_[i]
            col = pd.Series(arr[:, i]).map(_as_token)
            out[:, i] = np.where(col.isin(keep), col, self.other)
        return out

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        return np.asarray(input_features, dtype=object)


class TextJoin(BaseEstimator, TransformerMixin):
    """Join several text columns into one document per row."""

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=object)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        self.n_features_in_ = arr.shape[1]
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=object)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        joined = []
        for row in arr:
            parts = ["" if _is_na(v) else str(v) for v in row]
            joined.append(" ".join(parts))
        return np.asarray(joined, dtype=object)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(["joined_text"], dtype=object)


@dataclass
class FeatureSpec:
    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    text: list[str] = field(default_factory=list)

    @property
    def columns(self) -> list[str]:
        seen: list[str] = []
        for c in self.numeric + self.categorical + self.text:
            if c not in seen:
                seen.append(c)
        return seen

    def to_dict(self) -> dict:
        return {
            "numeric": list(self.numeric),
            "categorical": list(self.categorical),
            "text": list(self.text),
        }

    @classmethod
    def from_dict(cls, data: dict) -> FeatureSpec:
        return cls(
            numeric=list(data.get("numeric") or []),
            categorical=list(data.get("categorical") or []),
            text=list(data.get("text") or []),
        )


def _is_na(v) -> bool:
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except (ValueError, TypeError):
        return False


def _as_token(v) -> str:
    if _is_na(v):
        return "__missing__"
    return str(v)


def parse_date_column(series: pd.Series, column_name: str = "date") -> pd.Series:
    """Parse dates. Slash dates that could be M/D or D/M raise ValueError."""
    if pd.api.types.is_datetime64_any_dtype(series):
        out = pd.to_datetime(series, errors="raise")
        if getattr(out.dt, "tz", None) is not None:
            out = out.dt.tz_convert("UTC").dt.tz_localize(None)
        return out

    raw = series.copy()
    as_str = raw.map(lambda v: "" if _is_na(v) else str(v).strip())
    mask_na = as_str.eq("") | as_str.str.lower().isin(["nan", "nat", "none"])
    vals = as_str.loc[~mask_na]
    if vals.empty:
        raise ValueError(f"Date column '{column_name}' is empty")

    parsed = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")

    iso_ok = vals.map(lambda x: bool(_ISO.match(x) or _YMD.match(x)))
    dmy_ok = vals.map(lambda x: bool(_DMY.match(x)))

    if bool(iso_ok.all()):
        coerced = pd.to_datetime(vals, errors="coerce", format="ISO8601")
        if coerced.isna().any():
            coerced = pd.to_datetime(vals, errors="coerce")
        if coerced.isna().any():
            bad = vals[coerced.isna()].iloc[0]
            raise ValueError(f"Unparseable date in column '{column_name}': {bad!r}")
        parsed.loc[vals.index] = coerced
        return parsed

    if not bool(dmy_ok.all()):
        mixed = vals.iloc[0]
        raise ValueError(
            f"Unparseable or mixed date formats in column '{column_name}': {mixed!r}"
        )

    parts = vals.str.extract(_DMY)
    a = parts[0].astype(int)
    b = parts[1].astype(int)
    both_le_12 = bool((a <= 12).all() and (b <= 12).all())
    disagree = bool((a != b).any())
    if both_le_12 and disagree:
        raise ValueError(
            f"Ambiguous date format in column '{column_name}': "
            "values could be month-first or day-first"
        )
    if bool((a > 12).any() and (b > 12).any()):
        raise ValueError(f"Invalid calendar dates in column '{column_name}'")
    if bool((a > 12).any() and (b <= 12).all()):
        dayfirst = True
    elif bool((b > 12).any() and (a <= 12).all()):
        dayfirst = False
    else:
        dayfirst = False
    coerced = pd.to_datetime(vals, errors="coerce", dayfirst=dayfirst)
    if coerced.isna().any():
        bad = vals[coerced.isna()].iloc[0]
        raise ValueError(f"Unparseable date in column '{column_name}': {bad!r}")
    parsed.loc[vals.index] = coerced
    return parsed


def infer_roles(df: pd.DataFrame, config: dict) -> dict[str, str]:
    roles = dict(config.get("roles") or {})
    excluded = set(config.get("excluded") or [])
    target = config.get("target")
    date_column = config.get("date_column")
    group_columns = list(config.get("group_columns") or [])
    for col in df.columns:
        if col == target:
            continue
        if col in roles:
            continue
        if col in excluded:
            roles[col] = "excluded"
            continue
        if col == date_column:
            roles[col] = "date"
            continue
        if col in group_columns:
            roles[col] = "categorical"
            continue
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            roles[col] = "categorical"
            continue
        if pd.api.types.is_datetime64_any_dtype(s):
            roles[col] = "date"
            continue
        if pd.api.types.is_numeric_dtype(s):
            roles[col] = "numeric"
            continue
        n = len(s)
        nunq = int(s.nunique(dropna=True))
        sample = s.dropna().astype(str)
        avg_len = float(sample.str.len().mean()) if len(sample) else 0.0
        if nunq > min(50, max(1, int(0.45 * n))) and avg_len > 20:
            roles[col] = "text"
        elif n > 20 and nunq > 0.9 * n:
            roles[col] = "identifier"
        else:
            roles[col] = "categorical"
    return roles


def select_feature_columns(df: pd.DataFrame, config: dict, roles: dict[str, str]) -> list[str]:
    target = config.get("target")
    excluded = set(config.get("excluded") or [])
    requested = config.get("features")
    skip_roles = {"excluded", "identifier"}
    if requested:
        pool = list(requested)
    else:
        pool = [c for c in df.columns if c != target]
    cols: list[str] = []
    for c in pool:
        if c == target or c in excluded:
            continue
        if c not in df.columns:
            raise ValueError(f"Unknown feature column '{c}'")
        role = roles.get(c, "numeric")
        if role in skip_roles:
            continue
        if role == "date" and not requested:
            continue
        cols.append(c)
    if not cols:
        raise ValueError("No feature columns available after applying roles and exclusions")
    return cols


def build_feature_spec(df: pd.DataFrame, config: dict) -> tuple[FeatureSpec, dict[str, str]]:
    roles = infer_roles(df, config)
    cols = select_feature_columns(df, config, roles)
    numeric: list[str] = []
    categorical: list[str] = []
    text: list[str] = []
    for c in cols:
        role = roles.get(c, "numeric")
        if role == "text":
            text.append(c)
        elif role in {"categorical", "date"}:
            if role == "date":
                numeric.append(c)
            else:
                categorical.append(c)
        else:
            numeric.append(c)
    spec = FeatureSpec(numeric=numeric, categorical=categorical, text=text)
    return spec, roles


def coerce_features(df: pd.DataFrame, spec: FeatureSpec, roles: dict[str, str] | None = None) -> pd.DataFrame:
    """Align by name and coerce dtypes. Missing columns become NA / empty text."""
    roles = roles or {}
    out = pd.DataFrame(index=df.index)
    for c in spec.numeric:
        if c in df.columns:
            if roles.get(c) == "date" or pd.api.types.is_datetime64_any_dtype(df[c]):
                dt = parse_date_column(df[c], c)
                out[c] = (dt - pd.Timestamp("1970-01-01")).dt.total_seconds() / 86400.0
            else:
                out[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            out[c] = np.nan
    for c in spec.categorical:
        if c in df.columns:
            s = df[c]
            out[c] = s.where(s.isna(), s.map(lambda v: str(v))).astype("object")
        else:
            out[c] = np.nan
    for c in spec.text:
        if c in df.columns:
            s = df[c]
            out[c] = s.where(s.notna(), "").map(lambda v: "" if _is_na(v) else str(v))
        else:
            out[c] = ""
    return out.loc[:, spec.columns]


def build_preprocessor(spec: FeatureSpec, *, scale_numeric: bool) -> ColumnTransformer:
    transformers: list[tuple] = []
    if spec.numeric:
        steps: list[tuple] = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(steps), list(spec.numeric)))
    if spec.categorical:
        cat = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
                ("cap", FrequencyCap(max_categories=MAX_CATEGORIES)),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", cat, list(spec.categorical)))
    if spec.text:
        tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, min_df=1)
        if len(spec.text) == 1:
            transformers.append(("text", tfidf, spec.text[0]))
        else:
            transformers.append(
                ("text", Pipeline([("join", TextJoin()), ("tfidf", tfidf)]), list(spec.text))
            )
    if not transformers:
        raise ValueError("No usable feature columns after role filtering")
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)


def grouping_values(df: pd.DataFrame, config: dict) -> np.ndarray | None:
    entity = config.get("entity_column")
    if entity:
        if entity not in df.columns:
            raise ValueError(f"entity_column '{entity}' is not in the dataset")
        return df[entity].astype(str).to_numpy()
    gcols = list(config.get("group_columns") or [])
    if not gcols:
        return None
    missing = [c for c in gcols if c not in df.columns]
    if missing:
        raise ValueError(f"group_columns missing from dataset: {missing}")
    if len(gcols) == 1:
        return df[gcols[0]].astype(str).to_numpy()
    return df[gcols].astype(str).agg("|".join, axis=1).to_numpy()


def series_keys(df: pd.DataFrame, group_columns: list[str] | None) -> pd.Series:
    group_columns = list(group_columns or [])
    if not group_columns:
        return pd.Series(["__all__"] * len(df), index=df.index, dtype=object)
    missing = [c for c in group_columns if c not in df.columns]
    if missing:
        raise ValueError(f"group_columns missing from dataset: {missing}")
    if len(group_columns) == 1:
        return df[group_columns[0]].astype(str)
    return df[group_columns].astype(str).agg("|".join, axis=1)


LEAKAGE_CORR_THRESHOLD = 0.999


def flag_target_leakage(
    df: pd.DataFrame, target: str, feature_columns: list[str] | None = None
) -> list[str]:
    """Warn when a feature is identical to the target or nearly perfectly correlated."""
    if target not in df.columns:
        return []
    cols = list(feature_columns) if feature_columns is not None else [c for c in df.columns if c != target]
    y = df[target]
    y_num = pd.to_numeric(y, errors="coerce")
    y_numeric = bool(pd.api.types.is_numeric_dtype(y) or y_num.notna().mean() > 0.9)
    out: list[str] = []
    for col in cols:
        if col == target or col not in df.columns:
            continue
        x = df[col]
        if x.equals(y) or _as_token_series(x).equals(_as_token_series(y)):
            out.append(f"Possible leakage: feature '{col}' is identical to the target")
            continue
        if not y_numeric:
            continue
        x_num = pd.to_numeric(x, errors="coerce")
        mask = x_num.notna() & y_num.notna()
        if int(mask.sum()) < 4:
            continue
        if int(x_num.loc[mask].nunique()) < 2 or int(y_num.loc[mask].nunique()) < 2:
            continue
        corr = float(x_num.loc[mask].corr(y_num.loc[mask]))
        if np.isfinite(corr) and abs(corr) >= LEAKAGE_CORR_THRESHOLD:
            out.append(
                f"Possible leakage: feature '{col}' is nearly perfectly correlated with the target"
            )
    return out


def _as_token_series(s: pd.Series) -> pd.Series:
    return s.map(_as_token).astype("object")


def classification_split_feasible(
    n_rows,
    n_classes=None,
    test_size: float = 0.2,
    min_support: int = 2,
) -> bool:
    """Return whether a stratified train/val/test split can keep min_support.

    Call from experiment creation (`app.services.experiments`) so weak class
    support is rejected before enqueue. `split_supervised` still raises at train.

    Accepts either:
    - ``(n_rows, n_classes, test_size, min_support)`` totals, or
    - ``(counts: dict[str, int], test_size=...)`` per-class counts.
    """
    if isinstance(n_rows, dict):
        counts = n_rows
        if n_classes is not None:
            try:
                maybe = float(n_classes)
            except (TypeError, ValueError):
                maybe = None
            if maybe is not None and 0.0 < maybe < 1.0:
                test_size = maybe
        return _classification_split_from_counts(counts, test_size)
    try:
        n_rows_i = int(n_rows)
        n_classes_i = int(n_classes)
        min_support_i = int(min_support)
        test_size_f = float(test_size)
    except (TypeError, ValueError):
        return False
    if n_classes_i < 2 or min_support_i < 1 or n_rows_i < 4:
        return False
    if not 0.0 < test_size_f < 1.0:
        return False
    if n_rows_i < n_classes_i * min_support_i:
        return False
    n_test = max(1, math.ceil(test_size_f * n_rows_i))
    n_train = n_rows_i - n_test
    if n_test < n_classes_i or n_train < n_classes_i:
        return False
    if n_train < n_classes_i * min_support_i:
        return False
    if n_train < 3:
        n_val, n_fit = 1, n_train - 1
    else:
        n_val = max(1, math.ceil(0.2 * n_train))
        n_val = min(n_val, n_train - 1)
        n_fit = n_train - n_val
    return n_val >= n_classes_i and n_fit >= n_classes_i


def _classification_split_from_counts(counts: dict, test_size: float) -> bool:
    n = sum(int(c) for c in counts.values())
    n_classes = len(counts)
    if n_classes < 2 or n <= 0:
        return False
    try:
        test_size_f = float(test_size)
    except (TypeError, ValueError):
        return False
    if not 0.0 < test_size_f < 1.0:
        return False
    n_test = min(n - 1, max(1, math.ceil(test_size_f * n - 1e-12)))
    for count in counts.values():
        if (n_test * int(count)) / n < 0.5:
            return False
    return True


def split_supervised(
    y,
    *,
    mode: str,
    test_size: float,
    seed: int,
    groups: np.ndarray | None = None,
    dates=None,
    stratify: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_fit_idx, val_idx, test_idx). train_fit+val is the outer train."""
    y = np.asarray(y)
    n = len(y)
    if n < 4:
        raise ValueError("Need at least 4 rows to train, validate, and test")
    test_size = float(test_size)
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")
    idx = np.arange(n)
    mode = (mode or "random")
    mode = str(mode).strip().lower() or "random"
    if mode not in {"random", "group", "time"}:
        raise ValueError(
            f"Unknown split {mode!r}; expected 'random', 'group', or 'time'"
        )

    if mode == "time":
        if dates is None:
            raise ValueError("time split requires date_column")
        order = np.argsort(np.asarray(pd.to_datetime(dates)))
        n_test = max(1, round(n * test_size))
        if n - n_test < 2:
            raise ValueError(
                f"time split cannot reserve an untouched test partition of "
                f"{n_test} rows ({test_size:.0%} of {n}) while leaving train "
                f"and validation rows"
            )
        test_idx = np.sort(order[-n_test:])
        train_all = np.sort(order[:-n_test])
    elif mode == "group":
        if groups is None:
            raise ValueError("group split requires group_columns or entity_column")
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        try:
            train_all, test_idx = next(splitter.split(idx, y, groups))
        except ValueError as exc:
            raise ValueError(f"Cannot perform group split: {exc}") from exc
        train_all = np.sort(train_all)
        test_idx = np.sort(test_idx)
    else:
        strat = None
        if stratify:
            counts = pd.Series(y).value_counts()
            if int(counts.min()) >= 2 and int(counts.size) >= 2:
                strat = y
        train_all, test_idx = train_test_split(
            idx, test_size=test_size, random_state=seed, stratify=strat
        )
        train_all = np.sort(train_all)
        test_idx = np.sort(test_idx)

    train_fit, val_idx = _inner_val(
        train_all,
        y,
        mode=mode,
        seed=seed,
        groups=groups,
        dates=dates,
        stratify=stratify,
    )
    if set(train_fit) & set(test_idx) or set(val_idx) & set(test_idx):
        raise RuntimeError("Inner split leaked into the test partition")
    return train_fit, val_idx, test_idx


def _inner_val(train_all, y, *, mode, seed, groups, dates, stratify):
    train_all = np.asarray(train_all)
    val_frac = 0.2
    if mode == "group" and groups is not None:
        g_inner = groups[train_all]
        if int(pd.Series(g_inner).nunique()) < 2 or len(train_all) < 2:
            raise ValueError("insufficient groups for group split")
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
        try:
            fit_rel, val_rel = next(splitter.split(train_all, y[train_all], g_inner))
        except ValueError as exc:
            raise ValueError("insufficient groups for group split") from exc
        fit_idx = np.sort(train_all[fit_rel])
        val_idx = np.sort(train_all[val_rel])
        if set(groups[fit_idx]) & set(groups[val_idx]):
            raise ValueError("insufficient groups for group split")
        return fit_idx, val_idx
    if len(train_all) < 3:
        return train_all[:-1], train_all[-1:]
    if mode == "time":
        if dates is None:
            n_val = max(1, round(len(train_all) * val_frac))
            n_val = min(n_val, len(train_all) - 1)
            return train_all[:-n_val], train_all[-n_val:]
        order = train_all[np.argsort(np.asarray(pd.to_datetime(dates))[train_all])]
        n_val = max(1, round(len(order) * val_frac))
        n_val = min(n_val, len(order) - 1)
        return np.sort(order[:-n_val]), np.sort(order[-n_val:])
    strat = None
    y_tr = y[train_all]
    if stratify:
        counts = pd.Series(y_tr).value_counts()
        if int(counts.min()) >= 2 and int(counts.size) >= 2:
            strat = y_tr
    fit_rel, val_rel = train_test_split(
        np.arange(len(train_all)), test_size=val_frac, random_state=seed, stratify=strat
    )
    return np.sort(train_all[fit_rel]), np.sort(train_all[val_rel])


def drop_target_na(df: pd.DataFrame, target: str) -> pd.DataFrame:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' is not in the dataset")
    out = df.loc[df[target].notna()].copy()
    out.reset_index(drop=True, inplace=True)
    if out.empty:
        raise ValueError(f"Target column '{target}' has no non-missing values")
    return out
