import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.config import RiskConfig, load_settings  # noqa: E402
from app.llm import Proposal  # noqa: E402
from app.model import Predictor  # noqa: E402
from app.store import Store  # noqa: E402

NOW = datetime(2026, 11, 17, 12, 0, tzinfo=timezone.utc)
TEST_KEY = "clave-de-prueba-suficientemente-larga"


def make_candles(n=60, last_date=datetime(2026, 11, 16, tzinfo=timezone.utc), seed=0, start_price=60000.0):
    rng = np.random.default_rng(seed)
    closes = start_price * np.cumprod(1 + rng.normal(0.001, 0.02, n))
    opens = np.concatenate([[start_price], closes[:-1]])
    days = [last_date - timedelta(days=n - 1 - i) for i in range(n)]
    return pd.DataFrame({
        "open_time": pd.to_datetime(days, utc=True), "open": opens, "high": np.maximum(opens, closes) * 1.01,
        "low": np.minimum(opens, closes) * 0.99, "close": closes, "volume": 100.0,
        "close_time": pd.to_datetime([d + timedelta(days=1) for d in days], utc=True), "quote_volume": 1.0,
        "candle_date": [d.strftime("%Y-%m-%d") for d in days]})


class FakeMexc:
    def __init__(self, candles=None, price=None, error=None):
        self.candles = make_candles() if candles is None else candles
        self.price = price or float(self.candles["close"].iloc[-1])
        self.error = error
        self.calls = 0

    def closed_klines(self, symbol, interval, limit=120, now=None):
        self.calls += 1
        if self.error:
            raise self.error
        return self.candles[self.candles["close_time"] <= (now or NOW)].tail(limit).reset_index(drop=True)

    def ticker_price(self, symbol):
        if self.error:
            raise self.error
        return self.price

    def history(self, symbol, interval, start, end):
        return self.candles


class FakeLLM:
    """Responde con la acción indicada (o la primera permitida) usando la fecha y símbolo del contexto."""
    model = "fake-llm"

    def __init__(self, action=None, error=None):
        self.action, self.error, self.calls = action, error, 0

    def propose(self, ctx):
        self.calls += 1
        if self.error:
            raise self.error
        d = ctx["datos"]
        accion = self.action if self.action else d["allowed_actions"][0]
        return (Proposal(action=accion, symbol=d["symbol"], candle_time=d["candle_time"],
                         justification="prueba automatizada"),
                {"raw": "{}", "tool_calls": [], "tools_rechazadas": [], "latency_ms": 1.0})

    def health(self):
        return {"model": self.model}


@pytest.fixture
def settings(tmp_path):
    s = load_settings()
    return replace(s, api_key=TEST_KEY, db_path=tmp_path / "agent.db", risk_path=tmp_path / "risk.json",
                   model_dir=BACKEND / "models", cycle_seconds=3600)


@pytest.fixture(scope="session")
def predictor():
    return Predictor(BACKEND / "models")


@pytest.fixture
def risk_cfg():
    return RiskConfig()


def new_agent(settings, predictor, mexc=None, llm=None, store=None, risk_cfg=None):
    from app.agent import Agent
    return Agent(settings, store or Store(settings.db_path), mexc or FakeMexc(), llm or FakeLLM(), predictor,
                 risk_cfg or RiskConfig(), clock=lambda: NOW)
