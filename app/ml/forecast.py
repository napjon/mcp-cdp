"""Grouped forecasting: last-value, seasonal-naive, pooled XGBoost."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.ml.metrics import jsonable, library_versions, regression_metrics
from app.ml.plots import plot_feature_importance, plot_forecast
from app.ml.preprocess import parse_date_column, series_keys

try:
    from xgboost import XGBRegressor

    HAS_XGBOOST = True
except Exception:  # noqa: BLE001  # missing package or native lib (libomp)
    XGBRegressor = None
    HAS_XGBOOST = False

MAX_SERIES = 100
HORIZON_MIN = 1
HORIZON_MAX = 90
SEASON = {"D": 7, "W": 52, "M": 12}
FREQ_ALIAS = {"D": "D", "W": "W-SUN", "M": "MS"}


def lag_rolling_features(
    df: pd.DataFrame,
    *,
    target: str,
    group_columns: list[str] | None,
    lags: list[int],
    roll_windows: list[int],
) -> pd.DataFrame:
    """Lags and rolling means within group only; rolling uses shifted values (no current y)."""
    out = df.copy()
    group_columns = list(group_columns or [])
    if group_columns:
        grouped = out.groupby(group_columns, dropna=False, sort=False)[target]
    else:
        grouped = out.groupby(np.zeros(len(out), dtype=int), sort=False)[target]
    for lag in lags:
        out[f"lag_{lag}"] = grouped.shift(lag)
    for window in roll_windows:
        out[f"roll_mean_{window}"] = grouped.transform(
            lambda s, w=window: s.shift(1).rolling(window=w, min_periods=1).mean()
        )
    return out


def calendar_features(dates, frequency: str) -> pd.DataFrame:
    dt = pd.to_datetime(dates)
    frame = pd.DataFrame(index=getattr(dates, "index", None))
    frame["cal_month"] = dt.dt.month.astype(int)
    if frequency == "D":
        frame["cal_dow"] = dt.dt.dayofweek.astype(int)
        frame["cal_day"] = dt.dt.day.astype(int)
        frame["cal_week"] = dt.dt.isocalendar().week.astype(int)
    elif frequency == "W":
        frame["cal_week"] = dt.dt.isocalendar().week.astype(int)
        frame["cal_dow"] = dt.dt.dayofweek.astype(int)
    else:
        frame["cal_quarter"] = dt.dt.quarter.astype(int)
    return frame


def default_lags(frequency: str) -> list[int]:
    if frequency == "D":
        return [1, 2, 7]
    if frequency == "W":
        return [1, 2, 4]
    return [1, 2, 3]


def default_rolls(frequency: str) -> list[int]:
    if frequency == "D":
        return [3, 7]
    if frequency == "W":
        return [4]
    return [3]


def freq_alias(frequency: str) -> str:
    key = (frequency or "").upper()
    if key not in FREQ_ALIAS:
        raise ValueError(f"Unsupported frequency {frequency!r}; expected D, W, or M")
    return FREQ_ALIAS[key]


def snap_to_freq(dt: pd.Series, frequency: str) -> pd.Series:
    dt = pd.to_datetime(dt)
    alias = freq_alias(frequency)
    frequency = frequency.upper()
    if frequency == "D":
        return dt.dt.normalize()
    if frequency == "W":
        # W-* period labels use the week end; start_time is a different weekday
        # than date_range(..., freq=freq_alias("W")).
        return dt.dt.to_period(alias).dt.end_time.dt.normalize()
    return dt.dt.to_period("M").dt.to_timestamp(how="start")


def period_offset(frequency: str):
    return pd.tseries.frequencies.to_offset(freq_alias(frequency))


def _allow_exclude(config: dict) -> bool:
    return (
        config.get("allow_exclude_insufficient") is True
        or config.get("exclude_insufficient_history") is True
    )


def _duplicate_policy(config: dict) -> str:
    policy = (config.get("duplicate_timestamp") or "reject").strip().lower()
    if policy not in {"reject", "aggregate_mean"}:
        raise ValueError("duplicate_timestamp must be 'reject' or 'aggregate_mean'")
    return policy


def _apply_duplicates(df: pd.DataFrame, keys: list[str], target: str, policy: str) -> pd.DataFrame:
    dup = df.duplicated(keys, keep=False)
    if not bool(dup.any()):
        return df
    if policy == "reject":
        sample = df.loc[dup, keys].head(3).to_dict(orient="records")
        raise ValueError(
            f"Duplicate timestamps for {int(dup.sum())} rows (group+date). "
            f"Set duplicate_timestamp='aggregate_mean' or fix the data. Sample: {sample}"
        )
    agg: dict[str, str] = {}
    for col in df.columns:
        if col in keys:
            continue
        if col == target or pd.api.types.is_numeric_dtype(df[col]):
            agg[col] = "mean"
        else:
            agg[col] = "first"
    return df.groupby(keys, as_index=False).agg(agg)


def _prepare_frame(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    target = config.get("target")
    date_column = config.get("date_column")
    if not target or target not in df.columns:
        raise ValueError("Forecast requires config['target'] present in the dataset")
    if not date_column or date_column not in df.columns:
        raise ValueError("Forecast requires config['date_column'] present in the dataset")
    frequency = (config.get("frequency") or "").upper()
    alias = freq_alias(frequency)
    horizon = int(config.get("horizon"))
    if horizon < HORIZON_MIN or horizon > HORIZON_MAX:
        raise ValueError(f"horizon must be between {HORIZON_MIN} and {HORIZON_MAX}")
    group_columns = list(config.get("group_columns") or [])
    missing_g = [c for c in group_columns if c not in df.columns]
    if missing_g:
        raise ValueError(f"group_columns missing from dataset: {missing_g}")

    out = df.copy().reset_index(drop=True)
    out[date_column] = parse_date_column(out[date_column], date_column)
    out[target] = pd.to_numeric(out[target], errors="coerce")
    keys = group_columns + [date_column]
    out = _apply_duplicates(out, keys, target, _duplicate_policy(config))
    out[date_column] = snap_to_freq(out[date_column], frequency)
    out = _apply_duplicates(out, keys, target, _duplicate_policy(config))
    out = out.sort_values(group_columns + [date_column]).reset_index(drop=True)

    keys_series = series_keys(out, group_columns)
    n_series = int(keys_series.nunique())
    if n_series > MAX_SERIES:
        raise ValueError(f"Forecast has {n_series} series; maximum is {MAX_SERIES}")

    meta = {
        "target": target,
        "date_column": date_column,
        "frequency": frequency,
        "freq_alias": alias,
        "horizon": horizon,
        "group_columns": group_columns,
        "offset": period_offset(frequency),
        "season": SEASON[frequency],
    }
    return out, meta


def _regularize_group(
    gdf: pd.DataFrame, date_column: str, target: str, frequency: str
) -> tuple[pd.Series, list[str]]:
    gdf = gdf.sort_values(date_column)
    dates = pd.DatetimeIndex(gdf[date_column].tolist())
    y = pd.Series(gdf[target].to_numpy(dtype=float), index=dates)
    y = y[~y.index.duplicated(keep="first")]
    full = pd.date_range(dates.min(), dates.max(), freq=freq_alias(frequency))
    y = y.reindex(full)
    missing = [d.strftime("%Y-%m-%d") for d in full[y.isna()]]
    return y, missing


def _history_payload(series_map: dict[str, pd.Series]) -> dict:
    payload = {}
    for key, ser in series_map.items():
        payload[key] = {
            "dates": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in ser.index],
            "y": [None if pd.isna(v) else float(v) for v in ser.to_numpy()],
        }
    return payload


def _series_from_payload(payload: dict) -> dict[str, pd.Series]:
    out = {}
    for key, blob in payload.items():
        idx = pd.to_datetime(blob["dates"])
        y = [np.nan if v is None else float(v) for v in blob["y"]]
        out[key] = pd.Series(y, index=pd.DatetimeIndex(idx), dtype=float)
    return out


def last_value_forecast(history: pd.Series, horizon: int) -> np.ndarray:
    observed = history.dropna()
    last = float(observed.iloc[-1]) if len(observed) else np.nan
    return np.full(horizon, last, dtype=float)


def seasonal_naive_forecast(history: pd.Series, horizon: int, season: int) -> np.ndarray:
    arr = history.to_numpy(dtype=float)
    last_obs = np.nan
    for value in arr:
        if not np.isnan(value):
            last_obs = value
    out = np.empty(horizon, dtype=float)
    for h in range(horizon):
        src = len(arr) - season + h
        if 0 <= src < len(arr) and not np.isnan(arr[src]):
            out[h] = arr[src]
            continue
        src2 = len(arr) - season + (h % season)
        if 0 <= src2 < len(arr) and not np.isnan(arr[src2]):
            out[h] = arr[src2]
        else:
            out[h] = last_obs
    return out


def _future_index(last_date, horizon: int, frequency: str) -> pd.DatetimeIndex:
    alias = freq_alias(frequency)
    start = pd.Timestamp(last_date) + pd.tseries.frequencies.to_offset(alias)
    return pd.date_range(start, periods=horizon, freq=alias)


def _train_origins(n: int, horizon: int, n_origins: int, min_origin: int = 2) -> list[int]:
    """Origins whose horizon window stays inside the outer train prefix."""
    train_end = n - horizon
    origins: list[int] = []
    origin = train_end - horizon
    while origin >= min_origin and len(origins) < n_origins:
        origins.append(origin)
        origin -= max(1, horizon // 2)
    return origins


def _origin_scores(history: pd.Series, horizon: int, season: int, n_origins: int) -> dict[str, float]:
    n = len(history)
    train_end = n - horizon
    scores = {"last_value": [], "seasonal_naive": []}
    if train_end < 2:
        return {"last_value": np.nan, "seasonal_naive": np.nan}
    origins = _train_origins(n, horizon, n_origins)
    if not origins:
        origins = [max(1, train_end - 1)]
    enough_season = int(history.iloc[:train_end].notna().sum()) > season
    for origin in origins:
        hist = history.iloc[:origin]
        end = min(origin + horizon, train_end)
        actual = history.iloc[origin:end]
        mask = actual.notna().to_numpy()
        if not mask.any():
            continue
        y_true = actual.to_numpy(dtype=float)[mask]
        lv = last_value_forecast(hist, len(actual))[mask]
        scores["last_value"].append(float(np.mean(np.abs(y_true - lv))))
        if enough_season:
            sn = seasonal_naive_forecast(hist, len(actual), season)[mask]
            scores["seasonal_naive"].append(float(np.mean(np.abs(y_true - sn))))
    return {
        name: (float(np.mean(vals)) if vals else np.nan) for name, vals in scores.items()
    }


def _xgb_feature_frame(
    df: pd.DataFrame,
    *,
    target: str,
    date_column: str,
    group_columns: list[str],
    lags: list[int],
    rolls: list[int],
    frequency: str,
    group_id_map: dict[str, int],
    extra_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    work = lag_rolling_features(
        df,
        target=target,
        group_columns=group_columns,
        lags=lags,
        roll_windows=rolls,
    )
    cal = calendar_features(work[date_column], frequency)
    cal.index = work.index
    keys = series_keys(work, group_columns)
    work["group_id"] = keys.map(lambda k: float(group_id_map.get(str(k), -1)))
    feat_cols = [f"lag_{lag}" for lag in lags] + [f"roll_mean_{w}" for w in rolls]
    feat_cols.append("group_id")
    for col in extra_columns:
        work[col] = pd.to_numeric(work[col], errors="coerce")
        feat_cols.append(col)
    X = pd.concat([work[feat_cols], cal], axis=1)
    y = pd.to_numeric(work[target], errors="coerce")
    return X, y


def _fit_xgb(X: pd.DataFrame, y: pd.Series, seed: int, budget: str):
    mask = y.notna()
    if int(mask.sum()) < 8:
        return None
    quick = (budget or "quick") != "thorough"
    model = XGBRegressor(
        n_estimators=20 if quick else 40,
        max_depth=2 if quick else 3,
        learning_rate=0.3 if quick else 0.1,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=seed,
        n_jobs=1,
        verbosity=0,
        objective="reg:squarederror",
    )
    model.fit(X.loc[mask], y.loc[mask])
    return model


def _recursive_xgb(
    model,
    hist_df: pd.DataFrame,
    future_dates: pd.DatetimeIndex,
    *,
    target: str,
    date_column: str,
    group_columns: list[str],
    lags,
    rolls,
    frequency,
    group_id_map,
    extra_columns,
    extra_future: pd.DataFrame | None,
) -> np.ndarray:
    work = hist_df.copy()
    preds = []
    last_extras = {}
    for col in extra_columns:
        if col in work.columns:
            ser = pd.to_numeric(work[col], errors="coerce").dropna()
            last_extras[col] = float(ser.iloc[-1]) if len(ser) else np.nan
        else:
            last_extras[col] = np.nan
    gvals = {c: work[c].iloc[-1] for c in group_columns}
    for dt in future_dates:
        row = {date_column: pd.Timestamp(dt), target: np.nan}
        row.update(gvals)
        for col in extra_columns:
            value = np.nan
            if extra_future is not None and col in extra_future.columns:
                hit = extra_future.loc[extra_future[date_column] == pd.Timestamp(dt)]
                if len(hit):
                    value = pd.to_numeric(hit[col], errors="coerce").iloc[0]
            if pd.isna(value):
                value = last_extras.get(col, np.nan)
            row[col] = value
        work = pd.concat([work, pd.DataFrame([row])], ignore_index=True)
        X, _ = _xgb_feature_frame(
            work,
            target=target,
            date_column=date_column,
            group_columns=group_columns,
            lags=lags,
            rolls=rolls,
            frequency=frequency,
            group_id_map=group_id_map,
            extra_columns=extra_columns,
        )
        pred = float(model.predict(X.iloc[[-1]])[0])
        preds.append(pred)
        work.iat[-1, work.columns.get_loc(target)] = pred
    return np.asarray(preds, dtype=float)


def train(df: pd.DataFrame, config: dict, artifact_dir: Path) -> dict:
    warnings: list[str] = []
    prepared, meta = _prepare_frame(df, config)
    target = meta["target"]
    date_column = meta["date_column"]
    frequency = meta["frequency"]
    horizon = meta["horizon"]
    group_columns = meta["group_columns"]
    season = meta["season"]
    seed = int(config.get("seed", 42))
    budget = config.get("budget") or "quick"
    n_origins = 2 if budget != "thorough" else 4

    keys = series_keys(prepared, group_columns)
    groups_excluded: list[str] = []
    series_map: dict[str, pd.Series] = {}
    missing_periods: dict[str, list[str]] = {}
    templates: dict[str, pd.DataFrame] = {}
    min_obs = horizon + 1

    for key, gdf in prepared.groupby(keys, sort=False):
        key_s = str(key)
        y, missing = _regularize_group(gdf, date_column, target, frequency)
        if missing:
            missing_periods[key_s] = missing
            warnings.append(
                f"Series {key_s} has {len(missing)} missing {frequency} periods"
            )
        n_obs = int(y.notna().sum())
        if n_obs < min_obs:
            groups_excluded.append(key_s)
            continue
        series_map[key_s] = y
        templates[key_s] = gdf

    if groups_excluded:
        msg = (
            "Insufficient history for groups: "
            + ", ".join(f"{g} (need >= {min_obs} observations)" for g in groups_excluded)
        )
        if not _allow_exclude(config):
            raise ValueError(
                msg + ". Set allow_exclude_insufficient=true to skip them."
            )
        warnings.append(msg)

    if not series_map:
        raise ValueError("No forecast series remain after applying history requirements")

    extra_columns = [
        c
        for c in (config.get("features") or [])
        if c not in {target, date_column, *group_columns} and c in prepared.columns
    ]

    family_scores: dict[str, list[float]] = {
        "last_value": [],
        "seasonal_naive": [],
        "xgboost": [],
    }
    candidate_failures: list[dict] = []
    seasonal_ok = any(
        int(s.iloc[: max(len(s) - horizon, 0)].dropna().shape[0]) > season
        for s in series_map.values()
    )

    for key, ser in series_map.items():
        sc = _origin_scores(ser, horizon, season, n_origins)
        if not np.isnan(sc["last_value"]):
            family_scores["last_value"].append(sc["last_value"])
        if seasonal_ok and not np.isnan(sc["seasonal_naive"]):
            family_scores["seasonal_naive"].append(sc["seasonal_naive"])

    if not family_scores["last_value"]:
        family_scores["last_value"] = [float("inf")]

    xgb_model = None
    lags = default_lags(frequency)
    rolls = default_rolls(frequency)
    group_id_map = {k: i for i, k in enumerate(sorted(series_map))}
    xgb_feature_names: list[str] = []

    def _frame_prefix(key: str, ser: pd.Series, end: int) -> pd.DataFrame:
        hist = ser.iloc[:end]
        part = pd.DataFrame({date_column: hist.index, target: hist.to_numpy()})
        src = templates[key]
        for gc in group_columns:
            part[gc] = src[gc].iloc[0]
        for col in extra_columns:
            part[col] = np.nan
        return part

    if HAS_XGBOOST:
        try:
            xgb_mae: list[float] = []
            min_origin = 1 + max(lags + [1])
            for origin in _train_origins(
                min(len(s) for s in series_map.values()), horizon, n_origins, min_origin
            ):
                parts = [_frame_prefix(key, ser, origin) for key, ser in series_map.items()]
                train_df = pd.concat(parts, ignore_index=True)
                train_df = train_df.sort_values(group_columns + [date_column]).reset_index(drop=True)
                X_o, y_o = _xgb_feature_frame(
                    train_df,
                    target=target,
                    date_column=date_column,
                    group_columns=group_columns,
                    lags=lags,
                    rolls=rolls,
                    frequency=frequency,
                    group_id_map=group_id_map,
                    extra_columns=extra_columns,
                )
                model_o = _fit_xgb(X_o, y_o, seed, budget)
                if model_o is None:
                    continue
                for key, ser in series_map.items():
                    train_end = len(ser) - horizon
                    actual = ser.iloc[origin : min(origin + horizon, train_end)]
                    mask = actual.notna().to_numpy()
                    if not mask.any():
                        continue
                    pred = _recursive_xgb(
                        model_o,
                        _frame_prefix(key, ser, origin),
                        actual.index,
                        target=target,
                        date_column=date_column,
                        group_columns=group_columns,
                        lags=lags,
                        rolls=rolls,
                        frequency=frequency,
                        group_id_map=group_id_map,
                        extra_columns=extra_columns,
                        extra_future=None,
                    )
                    xgb_mae.append(
                        float(np.mean(np.abs(actual.to_numpy(dtype=float)[mask] - pred[mask])))
                    )
            train_parts = [
                _frame_prefix(key, ser, len(ser) - horizon) for key, ser in series_map.items()
            ]
            train_df = pd.concat(train_parts, ignore_index=True)
            train_df = train_df.sort_values(group_columns + [date_column]).reset_index(drop=True)
            X_all, y_all = _xgb_feature_frame(
                train_df,
                target=target,
                date_column=date_column,
                group_columns=group_columns,
                lags=lags,
                rolls=rolls,
                frequency=frequency,
                group_id_map=group_id_map,
                extra_columns=extra_columns,
            )
            xgb_model = _fit_xgb(X_all, y_all, seed, budget)
            if xgb_model is not None:
                xgb_feature_names = list(X_all.columns)
            if xgb_mae:
                family_scores["xgboost"] = xgb_mae
            elif xgb_model is None:
                candidate_failures.append(
                    {"candidate": "xgboost", "error": "insufficient rows after lag features"}
                )
            else:
                candidate_failures.append(
                    {"candidate": "xgboost", "error": "no backtest origins"}
                )
        except Exception as exc:  # noqa: BLE001
            candidate_failures.append({"candidate": "xgboost", "error": str(exc)})
            xgb_model = None
    else:
        candidate_failures.append({"candidate": "xgboost", "error": "xgboost unavailable"})

    if not seasonal_ok:
        candidate_failures.append(
            {"candidate": "seasonal_naive", "error": "insufficient history for one season"}
        )

    val_mae = {}
    for family, vals in family_scores.items():
        if family == "seasonal_naive" and not seasonal_ok:
            continue
        if family == "xgboost" and xgb_model is None:
            continue
        if vals:
            val_mae[family] = float(np.mean(vals))

    if not val_mae:
        raise ValueError("All forecast candidates failed")

    selected = min(val_mae, key=val_mae.get)
    y_true: list[float] = []
    y_pred: list[float] = []
    y_group: list[str] = []
    y_date: list[str] = []
    plot_dates = []
    plot_true = []
    plot_pred = []

    for key, ser in series_map.items():
        n = len(ser)
        hist = ser.iloc[: n - horizon]
        actual = ser.iloc[n - horizon :]
        if selected == "last_value":
            pred = last_value_forecast(hist, horizon)
        elif selected == "seasonal_naive":
            pred = seasonal_naive_forecast(hist, horizon, season)
        else:
            src = templates[key]
            hist_df = pd.DataFrame({date_column: hist.index, target: hist.to_numpy()})
            for gc in group_columns:
                hist_df[gc] = src[gc].iloc[0]
            for col in extra_columns:
                hist_df[col] = np.nan
            pred = _recursive_xgb(
                xgb_model,
                hist_df,
                actual.index,
                target=target,
                date_column=date_column,
                group_columns=group_columns,
                lags=lags,
                rolls=rolls,
                frequency=frequency,
                group_id_map=group_id_map,
                extra_columns=extra_columns,
                extra_future=None,
            )
        mask = actual.notna().to_numpy()
        if not mask.any():
            continue
        y_true.extend(actual.to_numpy(dtype=float)[mask].tolist())
        y_pred.extend(pred[mask].tolist())
        y_group.extend([key] * int(mask.sum()))
        y_date.extend([pd.Timestamp(d).strftime("%Y-%m-%d") for d in actual.index[mask]])
        if len(plot_true) < int(mask.sum()):
            plot_dates = list(actual.index[mask])
            plot_true = actual.to_numpy(dtype=float)[mask].tolist()
            plot_pred = pred[mask].tolist()

    test_metrics = regression_metrics(y_true, y_pred)
    plots_dir = Path(artifact_dir) / "plots"
    plot_files: list[str] = []
    if plot_dates:
        plot_files.append(
            plot_forecast(
                plot_dates,
                plot_true,
                plot_pred,
                plots_dir / "forecast.png",
                title=f"Forecast ({selected})",
            )
        )
    if selected == "xgboost" and xgb_model is not None and xgb_feature_names:
        try:
            plot_path = plot_feature_importance(
                xgb_feature_names,
                xgb_model.feature_importances_,
                plots_dir / "feature_importance.png",
            )
            if plot_path:
                plot_files.append(plot_path)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"feature importance plot skipped: {exc}")

    # History stored for inference is the full series including the holdout so
    # production predict continues from the last observed point. Test metrics
    # above used only the prefix before the untouched horizon.
    state = {
        "kind": selected,
        "frequency": frequency,
        "freq_alias": meta["freq_alias"],
        "horizon": horizon,
        "date_column": date_column,
        "target": target,
        "group_columns": group_columns,
        "season": season,
        "lags": lags,
        "roll_windows": rolls,
        "group_id_map": group_id_map,
        "extra_columns": extra_columns,
        "history": _history_payload(series_map),
        "xgb_feature_names": xgb_feature_names,
        "xgb_model": xgb_model if selected == "xgboost" else None,
        "seed": seed,
    }

    candidates_out = {}
    for family, mae in val_mae.items():
        candidates_out[family] = {"val_mae": mae, "status": "ok"}
    for fail in candidate_failures:
        candidates_out.setdefault(fail["candidate"], {"status": "failed", "error": fail["error"]})

    metrics = jsonable(
        {
            "task": "forecast",
            "seed": seed,
            "selected_candidate": selected,
            "test": test_metrics,
            "validation": {"mae": val_mae.get(selected)},
            "candidates": candidates_out,
            "candidate_failures": candidate_failures,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_group": y_group,
            "y_date": y_date,
            "groups_excluded": groups_excluded,
            "missing_periods": {k: v[:20] for k, v in missing_periods.items()},
            "missing_period_counts": {k: len(v) for k, v in missing_periods.items()},
            "library_versions": library_versions(),
            "params": {"horizon": horizon, "frequency": frequency, "budget": budget},
        }
    )
    return {
        "metrics": metrics,
        "warnings": warnings,
        "selected_candidate": selected,
        "plot_files": plot_files,
        "forecast_state": state,
        "feature_columns": extra_columns + group_columns + [date_column],
    }


def predict_forecast(df: pd.DataFrame, bundle: dict, output_path: Path) -> tuple[pd.DataFrame, dict | None, list[str]]:
    state = bundle["forecast_state"]
    warnings: list[str] = []
    date_column = state["date_column"]
    target = state["target"]
    group_columns = list(state["group_columns"] or [])
    frequency = state["frequency"]
    horizon = int(state["horizon"])
    season = int(state["season"])
    history = _series_from_payload(state["history"])
    kind = state["kind"]

    work = df.copy()
    if date_column in work.columns:
        work[date_column] = parse_date_column(work[date_column], date_column)
        work[date_column] = snap_to_freq(work[date_column], frequency)
    if group_columns:
        pred_keys = set(series_keys(work, group_columns).astype(str).tolist())
        trained_keys = set(history)
        unseen = sorted(k for k in pred_keys if k not in trained_keys)
        if unseen:
            raise ValueError(
                "Unseen forecast group keys not in trained history: " + ", ".join(unseen)
            )
    rows = []
    for key, ser in history.items():
        future = None
        extra_future = None
        if date_column in work.columns:
            if group_columns:
                mask = series_keys(work, group_columns).astype(str).eq(key)
                sub = work.loc[mask]
            else:
                sub = work
            last_train = pd.Timestamp(ser.index.max())
            future_sub = sub.loc[sub[date_column] > last_train].sort_values(date_column)
            if len(future_sub):
                future = pd.DatetimeIndex(future_sub[date_column].tolist()[:horizon])
                extra_future = future_sub
        if future is None or len(future) == 0:
            future = _future_index(ser.index.max(), horizon, frequency)
        if kind == "last_value":
            pred = last_value_forecast(ser, len(future))
        elif kind == "seasonal_naive":
            pred = seasonal_naive_forecast(ser, len(future), season)
        else:
            hist_df = pd.DataFrame({date_column: ser.index, target: ser.to_numpy()})
            parts = str(key).split("|") if group_columns else []
            for i, gc in enumerate(group_columns):
                hist_df[gc] = parts[i] if i < len(parts) else key
            for col in state.get("extra_columns") or []:
                hist_df[col] = np.nan
            pred = _recursive_xgb(
                state["xgb_model"],
                hist_df,
                future,
                target=target,
                date_column=date_column,
                group_columns=group_columns,
                lags=state["lags"],
                rolls=state["roll_windows"],
                frequency=frequency,
                group_id_map=state["group_id_map"],
                extra_columns=state.get("extra_columns") or [],
                extra_future=extra_future,
            )
        actuals = None
        if extra_future is not None and target in extra_future.columns:
            actuals = pd.to_numeric(extra_future[target], errors="coerce")
            actuals = actuals.tolist()[: len(future)]
        for i, dt in enumerate(future):
            row = {}
            if group_columns:
                parts = str(key).split("|")
                for j, gc in enumerate(group_columns):
                    row[gc] = parts[j] if j < len(parts) else key
            row[date_column] = pd.Timestamp(dt).strftime("%Y-%m-%d")
            row["prediction"] = float(pred[i]) if i < len(pred) else np.nan
            if actuals is not None and i < len(actuals):
                row[target] = actuals[i]
            rows.append(row)

    out = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    metrics = None
    if target in out.columns and out[target].notna().any():
        mask = out[target].notna() & out["prediction"].notna()
        if mask.any():
            metrics = jsonable(
                {
                    "task": "forecast",
                    "test": regression_metrics(out.loc[mask, target], out.loc[mask, "prediction"]),
                    "y_true": out.loc[mask, target].astype(float).tolist(),
                    "y_pred": out.loc[mask, "prediction"].astype(float).tolist(),
                }
            )
    if not rows:
        warnings.append("No forecast rows produced")
    return out, metrics, warnings
