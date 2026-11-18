from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

N_LAGS = 20
FEATURES = [f"r_lag_{k}" for k in range(1, N_LAGS + 1)]
MIN_CANDLES = N_LAGS + 1  # 21 cierres → 20 rendimientos


class CandleValidationError(ValueError):
    """Velas vacías, desordenadas, duplicadas, con valores inválidos o atrasadas."""


def validate_candles(df: pd.DataFrame, *, min_rows: int = MIN_CANDLES, now: datetime | None = None,
                     max_age_hours: float | None = None) -> dict:
    if df is None or len(df) == 0:
        raise CandleValidationError("No hay velas cerradas: conjunto vacío.")
    if len(df) < min_rows:
        raise CandleValidationError(f"Ventana insuficiente: se esperaban ≥ {min_rows} velas y se recibieron {len(df)}.")
    if df["candle_date"].duplicated().any():
        raise CandleValidationError(f"Velas duplicadas: {df.loc[df.candle_date.duplicated(), 'candle_date'].tolist()[:3]}")
    if not df["open_time"].is_monotonic_increasing:
        raise CandleValidationError("Las velas no están en orden cronológico.")
    precios = df[["open", "high", "low", "close"]].to_numpy(float)
    if not (np.isfinite(precios).all() and (precios > 0).all()):
        raise CandleValidationError("Hay precios faltantes, infinitos o no positivos.")
    gaps = pd.to_datetime(df["candle_date"]).diff().dt.days.dropna()
    info = {"rows": len(df), "first": df["candle_date"].iloc[0], "last": df["candle_date"].iloc[-1],
            "gaps_gt_1d": int((gaps > 1).sum())}
    if now is not None:
        edad = (now - df["close_time"].iloc[-1]).total_seconds() / 3600
        info["data_age_hours"] = round(edad, 3)
        if edad < 0:
            raise CandleValidationError("La última vela aún no cierra: no se usa una vela abierta como definitiva.")
        if max_age_hours is not None and edad > max_age_hours:
            raise CandleValidationError(
                f"Dato atrasado: la última vela cerró hace {edad:.1f} h (máximo {max_age_hours} h).")
    return info


def lag_vector(closes: np.ndarray) -> np.ndarray:
    """Vector (20,) con [r_t, r_{t-1}, …, r_{t-19}] a partir de los últimos 21 cierres."""
    closes = np.asarray(closes, dtype=float)
    if closes.ndim != 1 or len(closes) < MIN_CANDLES:
        raise CandleValidationError(f"Se requieren ≥ {MIN_CANDLES} cierres; forma recibida {closes.shape}.")
    r = closes[-MIN_CANDLES:][1:] / closes[-MIN_CANDLES:][:-1] - 1  # 20 rendimientos, del más antiguo al más reciente
    return r[::-1].copy()


def supervised(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla de entrenamiento: rezagos, objetivo y fechas de emisión/objetivo (sin filas incompletas)."""
    out = pd.DataFrame({"fecha_emision": df["candle_date"].values, "P": df["close"].astype(float).values})
    r = out["P"].pct_change()
    for k in range(1, N_LAGS + 1):
        out[f"r_lag_{k}"] = r.shift(k - 1)
    out["y"] = out["P"].shift(-1) / out["P"] - 1
    out["fecha_objetivo"] = out["fecha_emision"].shift(-1)
    return out.dropna(subset=FEATURES + ["y"]).reset_index(drop=True)


def indicators(closes: np.ndarray) -> dict:
    closes = np.asarray(closes, dtype=float)
    out: dict = {"sma20": None, "dist_sma20": None, "vol20": None, "rsi14": None}
    if len(closes) >= 20:
        sma = float(closes[-20:].mean())
        out["sma20"], out["dist_sma20"] = sma, float(closes[-1] / sma - 1)
    r = closes[1:] / closes[:-1] - 1
    if len(r) >= 20:
        out["vol20"] = float(np.std(r[-20:], ddof=1))
    if len(r) >= 14:
        out["rsi14"] = rsi_wilder(closes, 14)
    return out


def rsi_wilder(closes: np.ndarray, n: int = 14) -> float:
    d = np.diff(np.asarray(closes, dtype=float))
    gan, per = np.clip(d, 0, None), np.clip(-d, 0, None)
    g, p = gan[:n].mean(), per[:n].mean()
    for i in range(n, len(d)):
        g = (g * (n - 1) + gan[i]) / n
        p = (p * (n - 1) + per[i]) / n
    if g == 0 and p == 0:
        return 50.0       # sin ganancias ni pérdidas: convención neutral
    if p == 0:
        return 100.0      # sin pérdidas: RS infinito
    return float(100 - 100 / (1 + g / p))


def utcnow() -> datetime:
    return datetime.now(timezone.utc)