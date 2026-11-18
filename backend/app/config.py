from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_DIR / ".env")


def _path(env: str, default: str) -> Path:
    p = Path(os.environ.get(env, default))
    return p if p.is_absolute() else BACKEND_DIR / p


@dataclass(frozen=True)
class Settings:
    api_key: str
    symbol: str
    interval: str
    mexc_base_url: str
    ollama_url: str
    ollama_model: str
    ollama_timeout: float
    db_path: Path
    model_dir: Path
    risk_path: Path
    cycle_seconds: int
    cors_origins: tuple[str, ...]
    live_trading_enabled: bool = False


def load_settings() -> Settings:
    return Settings(
        api_key=os.environ.get("API_KEY", ""),
        symbol=os.environ.get("SYMBOL", "ETHUSDT").upper(),
        interval=os.environ.get("INTERVAL", "1d"),
        mexc_base_url=os.environ.get("MEXC_BASE_URL", "https://api.mexc.com"),
        ollama_url=os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"),
        ollama_model=os.environ.get("OLLAMA_MODEL", "qwen2.5:3b"),
        ollama_timeout=float(os.environ.get("OLLAMA_TIMEOUT", "120")),
        db_path=_path("DB_PATH", "data/agent.db"),
        model_dir=_path("MODEL_DIR", "models"),
        risk_path=_path("RISK_CONFIG", "config/risk.json"),
        cycle_seconds=int(os.environ.get("CYCLE_SECONDS", "300")),
        cors_origins=tuple(
            o.strip()
            for o in os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
            if o.strip()
        ),
    )


@dataclass
class RiskConfig:
    """Límites deterministas. Se definen ANTES de la prueba y el LLM no puede modificarlos."""
    initial_capital: float = 50_000.0
    fee_rate: float = 0.0
    slippage_bps: float = 5.0
    max_exposure_pct: float = 0.95
    max_daily_loss_pct: float = 0.05
    max_data_age_hours: float = 30.0
    min_notional: float = 5.0
    qty_step: float = 0.0001               # Precisión base para ETHUSDT en MEXC
    signal_threshold: float = 0.0005

    def validate(self) -> "RiskConfig":
        errores = []
        if self.initial_capital <= 0:
            errores.append("initial_capital debe ser > 0")
        if not 0 <= self.fee_rate <= 0.01:
            errores.append("fee_rate debe estar entre 0 y 0.01")
        if not 0 <= self.slippage_bps <= 100:
            errores.append("slippage_bps debe estar entre 0 y 100")
        if not 0 < self.max_exposure_pct <= 1:
            errores.append("max_exposure_pct debe estar en (0, 1]: no se permite apalancamiento")
        if not 0 < self.max_daily_loss_pct < 1:
            errores.append("max_daily_loss_pct debe estar en (0, 1)")
        if self.max_data_age_hours <= 0:
            errores.append("max_data_age_hours debe ser > 0")
        if self.min_notional <= 0 or self.qty_step <= 0:
            errores.append("min_notional y qty_step deben ser > 0")
        if self.signal_threshold < 0:
            errores.append("signal_threshold debe ser >= 0")
        if errores:
            raise ValueError("; ".join(errores))
        return self

    @classmethod
    def load(cls, path: Path) -> "RiskConfig":
        if path.exists():
            data = json.loads(path.read_text())
            conocidos = {f.name for f in fields(cls)}
            return cls(**{k: v for k, v in data.items() if k in conocidos}).validate()
        cfg = cls().validate()
        cfg.save(path)
        return cfg

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))