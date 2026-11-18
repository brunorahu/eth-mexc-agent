from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Callable

import httpx
import pandas as pd

DAY_MS = 86_400_000
MAX_LIMIT = 500  # MEXC devuelve como máximo 500 velas por solicitud


class MarketDataError(RuntimeError):
    """La API no respondió, limitó solicitudes o devolvió datos no válidos."""


class MexcClient:
    def __init__(self, base_url: str = "https://api.mexc.com", timeout: float = 10.0, retries: int = 3,
                 backoff: float = 1.0, transport: httpx.BaseTransport | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.base_url = base_url.rstrip("/")
        self.retries, self.backoff, self.sleep = retries, backoff, sleep
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout, transport=transport,
                                    headers={"User-Agent": "mexc-paper-agent/1.0"})

    # ------------------------------------------------------------------ HTTP con reintentos
    def _get(self, path: str, params: dict) -> object:
        ultimo = ""
        for intento in range(1, self.retries + 1):
            try:
                r = self._client.get(path, params=params)
                if r.status_code == 429 or r.status_code >= 500:
                    ultimo = f"HTTP {r.status_code} ({'límite de solicitudes' if r.status_code == 429 else 'error del servidor'})"
                    espera = float(r.headers.get("Retry-After", self.backoff * 2 ** (intento - 1)))
                else:
                    if r.status_code >= 400:
                        raise MarketDataError(f"MEXC rechazó la solicitud {path}: HTTP {r.status_code} {r.text[:200]}")
                    return r.json()
            except (httpx.TimeoutException, httpx.TransportError) as e:
                ultimo = f"{type(e).__name__}: sin respuesta de MEXC"
                espera = self.backoff * 2 ** (intento - 1)
            except ValueError as e:  # JSON inválido
                raise MarketDataError(f"MEXC devolvió una respuesta que no es JSON en {path}: {e}") from e
            if intento < self.retries:
                self.sleep(espera)  # espera creciente: 1 s, 2 s, 4 s…
        raise MarketDataError(f"MEXC no disponible tras {self.retries} intentos en {path}: {ultimo}")

    # ------------------------------------------------------------------ Endpoints
    def klines(self, symbol: str, interval: str = "1d", limit: int = 100,
               start_ms: int | None = None, end_ms: int | None = None) -> pd.DataFrame:
        params: dict = {"symbol": symbol, "interval": interval, "limit": min(limit, MAX_LIMIT)}
        if start_ms is not None:
            params["startTime"] = start_ms
        if end_ms is not None:
            params["endTime"] = end_ms
        data = self._get("/api/v3/klines", params)
        if not isinstance(data, list):
            raise MarketDataError(f"Respuesta inesperada de klines: {str(data)[:200]}")
        return parse_klines(data)

    def closed_klines(self, symbol: str, interval: str = "1d", limit: int = 120,
                      now: datetime | None = None) -> pd.DataFrame:
        """Solo velas CERRADAS: se descarta la vela en curso (close_time > ahora)."""
        now = now or datetime.now(timezone.utc)
        df = self.klines(symbol, interval, limit)
        return df[df["close_time"] <= now].reset_index(drop=True)

    def history(self, symbol: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Histórico diario paginado en ventanas de 500 velas (startTime + endTime)."""
        partes, s = [], int(start.timestamp() * 1000)
        e_final = int(end.timestamp() * 1000)
        while s < e_final:
            e = min(s + (MAX_LIMIT - 1) * DAY_MS, e_final)
            df = self.klines(symbol, interval, MAX_LIMIT, s, e)
            if len(df):
                partes.append(df)
            s = e + 1
        if not partes:
            raise MarketDataError(f"MEXC no devolvió velas de {symbol} entre {start.date()} y {end.date()}")
        return pd.concat(partes, ignore_index=True)

    def ticker_price(self, symbol: str) -> float:
        data = self._get("/api/v3/ticker/price", {"symbol": symbol})
        try:
            precio = float(data["price"])  # type: ignore[index]
        except (KeyError, TypeError, ValueError) as e:
            raise MarketDataError(f"Ticker inválido para {symbol}: {data}") from e
        if not precio > 0:
            raise MarketDataError(f"Precio no positivo para {symbol}: {precio}")
        return precio

    def symbol_info(self, symbol: str) -> dict:
        data = self._get("/api/v3/exchangeInfo", {"symbol": symbol})
        simbolos = data.get("symbols", []) if isinstance(data, dict) else []
        if not simbolos or simbolos[0].get("symbol") != symbol:
            raise MarketDataError(f"El símbolo {symbol} no existe en MEXC Spot")
        s = simbolos[0]
        return {"symbol": s["symbol"], "base": s.get("baseAsset"), "quote": s.get("quoteAsset"),
                "status": s.get("status"), "qty_step": float(s.get("baseSizePrecision", "0.000001")),
                "quote_precision": int(s.get("quotePrecision", 2)),
                "taker_commission": float(s.get("takerCommission", 0)),
                "maker_commission": float(s.get("makerCommission", 0))}


def parse_klines(rows: list) -> pd.DataFrame:
    """[open_time, open, high, low, close, volume, close_time, quote_volume] → DataFrame tipado (UTC)."""
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume"]
    if not rows:
        return pd.DataFrame(columns=cols)
    if any(len(r) < 8 for r in rows):
        raise MarketDataError("Velas con un número de campos distinto al esperado (8)")
    df = pd.DataFrame([r[:8] for r in rows], columns=cols)
    for c in ("open", "high", "low", "close", "volume", "quote_volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df["candle_date"] = df["open_time"].dt.strftime("%Y-%m-%d")
    return df