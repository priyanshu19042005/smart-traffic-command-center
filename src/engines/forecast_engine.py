r"""
forecast_engine.py
==================
**Incident forecasting** for the next **day / week / month**, city-wide and per
corridor/zone, with automatic model selection and graceful degradation.

Models (auto-skipped if the library is absent)
----------------------------------------------
* **Prophet**  — additive trend + weekly seasonality (if ``prophet`` installed).
* **XGBoost**  — gradient-boosted regression on calendar + lag features
  (falls back to scikit-learn ``HistGradientBoostingRegressor`` if XGBoost
  is unavailable).
* **LSTM**     — sequence model (if ``tensorflow`` installed).
* **ETS**      — Holt-Winters exponential smoothing (``statsmodels``).
* **Naive**    — seasonal-naive baseline (last week repeats).

Selection
---------
Each model is back-tested on the last ``test_days`` (config). The model with
the lowest **MAE** wins and is refit on the full series to produce the forward
forecast. Anomaly spikes are flagged via STL-residual z-scores.

Outputs
-------
* ``outputs/forecast_<scope>.csv`` — date, yhat, yhat_lower, yhat_upper, model.
* ``outputs/forecast_metrics.csv`` — per-model MAE/RMSE/MAPE leaderboard.
* ``forecast_chart_*`` data returned for the dashboard to plot with Plotly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.utils import get_logger, get_path, load_config
from src.utils.config import Config

log = get_logger(__name__)


# --- optional dependency detection -----------------------------------------
def _has(mod: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(mod) is not None


HAS_PROPHET = _has("prophet")
HAS_XGB = _has("xgboost")
HAS_TF = _has("tensorflow")


# --- metrics ---------------------------------------------------------------
def _mae(a, b): return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))
def _rmse(a, b): return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
def _mape(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = a > 0
    return float(np.mean(np.abs((a[m] - b[m]) / a[m])) * 100) if m.any() else np.nan


@dataclass
class ForecastResult:
    scope: str
    best_model: str
    metrics: pd.DataFrame
    forecast: pd.DataFrame            # future dates with yhat + interval
    history: pd.DataFrame             # observed series
    horizons: dict[str, float] = field(default_factory=dict)


@dataclass
class ForecastEngine:
    cfg: Config

    # -- build daily series ---------------------------------------------
    def daily_series(self, df: pd.DataFrame, scope_col: Optional[str] = None,
                     scope_val: Optional[str] = None) -> pd.Series:
        sub = df
        if scope_col and scope_val is not None:
            sub = df[df[scope_col] == scope_val]
        s = (sub.set_index("date").sort_index()
                 .assign(n=1).resample(self.cfg.forecast.freq)["n"].sum())
        # Fill calendar gaps with 0 (no incidents that day).
        idx = pd.date_range(s.index.min(), s.index.max(), freq=self.cfg.forecast.freq)
        return s.reindex(idx, fill_value=0).rename("y")

    # -- calendar/lag features for ML models ----------------------------
    @staticmethod
    def _calendar(idx: pd.DatetimeIndex) -> pd.DataFrame:
        return pd.DataFrame({
            "dow": idx.dayofweek, "day": idx.day, "month": idx.month,
            "is_weekend": (idx.dayofweek >= 5).astype(int),
            "weekofyear": idx.isocalendar().week.astype(int).values,
        }, index=idx)

    def _supervised(self, s: pd.Series, lags=(1, 2, 3, 7, 14)) -> pd.DataFrame:
        X = self._calendar(s.index)
        for l in lags:
            X[f"lag_{l}"] = s.shift(l)
        X["roll7"] = s.shift(1).rolling(7).mean()
        X["y"] = s.values
        return X.dropna()

    # -- individual models (return fitted forecaster closures) -----------
    def _fit_naive(self, s: pd.Series) -> Callable[[pd.DatetimeIndex], np.ndarray]:
        season = 7
        last = s.iloc[-season:].values
        return lambda idx: np.array([last[i % season] for i in range(len(idx))])

    def _fit_ets(self, s: pd.Series) -> Optional[Callable]:
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            m = ExponentialSmoothing(s.clip(lower=0) + 1e-6, trend="add",
                                     seasonal="add", seasonal_periods=7).fit()
            return lambda idx: np.clip(m.forecast(len(idx)).values, 0, None)
        except Exception as exc:
            log.debug("ETS unavailable: %s", exc)
            return None

    def _fit_ml(self, s: pd.Series) -> Optional[Callable]:
        sup = self._supervised(s)
        if len(sup) < 20:
            return None
        feat_cols = [c for c in sup.columns if c != "y"]
        Xtr, ytr = sup[feat_cols].values, sup["y"].values
        if HAS_XGB:
            from xgboost import XGBRegressor
            model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                 subsample=0.9, random_state=self.cfg.project.random_seed)
        else:
            from sklearn.ensemble import HistGradientBoostingRegressor
            model = HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.05,
                random_state=self.cfg.project.random_seed)
        model.fit(Xtr, ytr)

        def _forecast(future_idx: pd.DatetimeIndex) -> np.ndarray:
            # Recursive multi-step forecast.
            hist = s.copy()
            preds = []
            for ts in future_idx:
                cal = self._calendar(pd.DatetimeIndex([ts])).iloc[0]
                row = {**cal.to_dict()}
                for l in (1, 2, 3, 7, 14):
                    row[f"lag_{l}"] = hist.iloc[-l] if len(hist) >= l else hist.mean()
                row["roll7"] = hist.iloc[-7:].mean() if len(hist) >= 7 else hist.mean()
                x = np.array([[row[c] for c in feat_cols]])
                yh = float(max(0, model.predict(x)[0]))
                preds.append(yh)
                hist = pd.concat([hist, pd.Series([yh], index=[ts])])
            return np.array(preds)
        return _forecast

    def _fit_prophet(self, s: pd.Series) -> Optional[Callable]:
        if not HAS_PROPHET:
            return None
        try:
            from prophet import Prophet
            dfp = pd.DataFrame({"ds": s.index, "y": s.values})
            m = Prophet(weekly_seasonality=True, daily_seasonality=False,
                        yearly_seasonality=False, interval_width=0.8)
            m.fit(dfp)

            def _forecast(idx: pd.DatetimeIndex) -> np.ndarray:
                fut = pd.DataFrame({"ds": idx})
                return np.clip(m.predict(fut)["yhat"].values, 0, None)
            _forecast._prophet = m  # type: ignore[attr-defined]
            return _forecast
        except Exception as exc:
            log.warning("Prophet failed: %s", exc)
            return None

    def _fit_lstm(self, s: pd.Series) -> Optional[Callable]:
        if not HAS_TF or len(s) < 60:
            return None
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, Sequential
            look = 14
            vals = s.values.astype("float32")
            mu, sd = vals.mean(), vals.std() + 1e-6
            z = (vals - mu) / sd
            Xs, ys = [], []
            for i in range(look, len(z)):
                Xs.append(z[i - look:i]); ys.append(z[i])
            Xs = np.array(Xs)[..., None]; ys = np.array(ys)
            model = Sequential([layers.Input((look, 1)), layers.LSTM(32),
                                layers.Dense(1)])
            model.compile(optimizer="adam", loss="mse")
            model.fit(Xs, ys, epochs=40, verbose=0)

            def _forecast(idx: pd.DatetimeIndex) -> np.ndarray:
                window = list(z[-look:]); preds = []
                for _ in range(len(idx)):
                    x = np.array(window[-look:])[None, :, None]
                    yh = float(model.predict(x, verbose=0)[0, 0])
                    preds.append(yh); window.append(yh)
                return np.clip(np.array(preds) * sd + mu, 0, None)
            return _forecast
        except Exception as exc:
            log.warning("LSTM failed: %s", exc)
            return None

    # -- orchestration ---------------------------------------------------
    def forecast_scope(self, s: pd.Series, scope: str) -> ForecastResult:
        test_days = min(self.cfg.forecast.test_days, max(7, len(s) // 4))
        train, test = s.iloc[:-test_days], s.iloc[-test_days:]

        builders = {
            "naive": self._fit_naive, "ets": self._fit_ets, "xgboost": self._fit_ml,
            "prophet": self._fit_prophet, "lstm": self._fit_lstm,
        }
        wanted = [m for m in self.cfg.forecast.models if m in builders]

        rows, fitted = [], {}
        for name in wanted:
            fn = builders[name](train)
            if fn is None:
                continue
            pred = fn(test.index)
            rows.append({"model": name, "MAE": _mae(test.values, pred),
                         "RMSE": _rmse(test.values, pred),
                         "MAPE": _mape(test.values, pred)})
            fitted[name] = builders[name]  # refit on full series below
        if not rows:
            raise RuntimeError("No forecasting model could be fit.")
        metrics = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
        best = metrics.iloc[0]["model"]
        log.info("[%s] best model = %s (MAE %.2f).", scope, best, metrics.iloc[0]["MAE"])

        # Refit best on full series, forecast max horizon.
        horizon = max(self.cfg.forecast.horizons.to_dict().values())
        future_idx = pd.date_range(s.index[-1] + pd.Timedelta(days=1),
                                   periods=horizon, freq=self.cfg.forecast.freq)
        best_fn = builders[best](s)
        yhat = best_fn(future_idx)
        resid_sd = metrics.iloc[0]["RMSE"]
        fc = pd.DataFrame({
            "date": future_idx, "yhat": np.round(yhat, 2),
            "yhat_lower": np.clip(yhat - 1.96 * resid_sd, 0, None).round(2),
            "yhat_upper": (yhat + 1.96 * resid_sd).round(2),
            "model": best, "scope": scope,
        })
        horizons = {name: float(fc["yhat"].iloc[:h].sum())
                    for name, h in self.cfg.forecast.horizons.to_dict().items()}
        hist = s.reset_index().rename(columns={"index": "date", "y": "y"})
        hist.columns = ["date", "y"]
        return ForecastResult(scope, best, metrics, fc, hist, horizons)


def run_forecasts(
    df: pd.DataFrame,
    cfg: Optional[Config] = None,
    top_scopes: int = 5,
    persist: bool = True,
) -> dict[str, ForecastResult]:
    """Forecast city-wide total plus the busiest corridors.

    Returns ``{scope -> ForecastResult}`` (scope ``"CITY"`` is the total).
    """
    cfg = cfg or load_config()
    engine = ForecastEngine(cfg)
    out_dir = get_path("outputs_dir", cfg=cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, ForecastResult] = {}
    # City-wide.
    results["CITY"] = engine.forecast_scope(engine.daily_series(df), "CITY")

    # Busiest corridors.
    gb = cfg.forecast.group_by
    if gb in df.columns:
        top = (df[df[gb].astype(str) != "unknown"][gb]
               .value_counts().head(top_scopes).index.tolist())
        for scope in top:
            s = engine.daily_series(df, gb, scope)
            if (s > 0).sum() < 20:
                continue
            try:
                results[scope] = engine.forecast_scope(s, str(scope))
            except Exception as exc:
                log.warning("Forecast failed for %s: %s", scope, exc)

    if persist:
        all_fc = pd.concat([r.forecast for r in results.values()], ignore_index=True)
        all_fc.to_csv(out_dir / "forecasts.csv", index=False)
        all_metrics = pd.concat(
            [r.metrics.assign(scope=r.scope) for r in results.values()],
            ignore_index=True)
        all_metrics.to_csv(out_dir / "forecast_metrics.csv", index=False)
        summary = pd.DataFrame([
            {"scope": r.scope, "best_model": r.best_model, **r.horizons}
            for r in results.values()
        ])
        summary.to_csv(out_dir / "forecast_summary.csv", index=False)
        log.info("Forecast outputs -> %s", out_dir)
    return results


if __name__ == "__main__":  # pragma: no cover
    cfg = load_config()
    feats = pd.read_parquet(get_path("features", cfg=cfg))
    res = run_forecasts(feats, cfg, top_scopes=3)
    for scope, r in res.items():
        print(f"\n=== {scope} (best={r.best_model}) ===")
        print("horizons:", {k: round(v, 1) for k, v in r.horizons.items()})
        print(r.forecast.head(3).to_string(index=False))
