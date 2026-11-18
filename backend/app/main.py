"""API del backend (FastAPI). Documentación interactiva en /docs.

Todas las rutas, salvo /api/health, exigen la cabecera X-API-Key. La clave vive solo en el servidor
(backend/.env); la UI la pide al usuario en tiempo de ejecución y nunca se compila en el bundle.
"""
from __future__ import annotations

import asyncio
import hmac
import platform
import subprocess
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from . import __version__
from .agent import LIVE, STRATEGIES, Agent, run_metrics
from .config import RiskConfig, Settings, load_settings
from .llm import LLMUnavailable, OllamaClient
from .mexc import MarketDataError, MexcClient
from .model import Predictor
from .store import Store

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class Pregunta(BaseModel):
    prompt: str = Field(min_length=1, max_length=500)


class ReplayReq(BaseModel):
    start: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    end: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    strategies: list[str] = list(STRATEGIES)
    limit: int | None = Field(default=30, ge=1, le=1000)


def hardware() -> dict:
    chip = platform.processor()
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                              timeout=3).stdout.strip() or chip
        mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=3).stdout)
    except (OSError, ValueError, subprocess.SubprocessError):
        mem = None
    return {"sistema": f"{platform.system()} {platform.release()}", "arquitectura": platform.machine(),
            "cpu": chip, "ram_gb": round(mem / 2**30, 1) if mem else None, "python": platform.python_version()}


def create_app(settings: Settings | None = None, *, mexc: MexcClient | None = None, llm: OllamaClient | None = None,
               predictor: Predictor | None = None, store: Store | None = None, start_loop: bool = True) -> FastAPI:
    s = settings or load_settings()
    if not s.api_key or len(s.api_key) < 16:
        raise RuntimeError("API_KEY ausente o demasiado corta (mínimo 16 caracteres). Configúrenla en backend/.env")
    store = store or Store(s.db_path)
    predictor = predictor or Predictor(s.model_dir)
    if not s.risk_path.exists():  # primera ejecución: δ = umbral elegido en VA durante el entrenamiento
        RiskConfig(signal_threshold=float(predictor.card.get("signal_threshold", 0.0005))).validate().save(s.risk_path)
    risk_cfg = RiskConfig.load(s.risk_path)
    agent = Agent(s, store, mexc or MexcClient(s.mexc_base_url), llm or OllamaClient(s.ollama_url, s.ollama_model,
                  s.ollama_timeout), predictor, risk_cfg)
    stop_event = asyncio.Event()

    async def loop() -> None:
        """Ciclo programado: revisa cada CYCLE_SECONDS si hay una vela nueva cerrada."""
        while not stop_event.is_set():
            if agent.status == "RUNNING":
                await asyncio.to_thread(agent.live_cycle)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=s.cycle_seconds)
            except asyncio.TimeoutError:
                pass

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        agent.recover_on_startup()
        tarea = asyncio.create_task(loop()) if start_loop else None
        yield
        stop_event.set()
        if tarea:
            await tarea

    app = FastAPI(title="MEXC Paper Trading Agent", version=__version__, lifespan=lifespan,
                  description="Agente GRU + LLM local (Ollama) con controles deterministas y ejecución Paper.")
    app.add_middleware(CORSMiddleware, allow_origins=list(s.cors_origins), allow_methods=["GET", "POST", "PUT"],
                       allow_headers=["X-API-Key", "Content-Type"])
    app.state.agent, app.state.store, app.state.settings = agent, store, s

    def require_key(request: Request, key: str | None = Depends(api_key_header)) -> None:
        if not key:
            store.log_event("warning", "auth", f"Solicitud sin API key a {request.url.path}")
            raise HTTPException(401, "Falta la cabecera X-API-Key")
        if not hmac.compare_digest(key.encode(), s.api_key.encode()):
            store.log_event("warning", "auth", f"API key incorrecta en {request.url.path}")
            raise HTTPException(401, "API key incorrecta")

    protegido = [Depends(require_key)]

    # ------------------------------------------------------------------ públicas
    @app.get("/api/health")
    def health():
        return {"ok": True, "version": __version__, "symbol": s.symbol, "paper_only": True}

    # ------------------------------------------------------------------ estado y control
    @app.get("/api/status", dependencies=protegido)
    def status():
        p = store.portfolio(LIVE)
        snap = store.kv_get("last_snapshot")
        precio = snap["close"] if snap else None
        ult = store.decisions(LIVE, 1)
        return {
            "version": __version__, "symbol": s.symbol, "interval": s.interval, "mode": "PAPER",
            "live_trading_enabled": False,
            "agent": {"status": agent.status, "reason": store.kv_get("live_status_reason"),
                      "since": store.kv_get("live_status_at"), "cycle_seconds": s.cycle_seconds},
            "data": snap, "last_decision": ult[0] if ult else None,
            "portfolio": {**p, "price": precio, "equity": (p["cash"] + p["qty"] * precio) if precio else p["cash"],
                          "exposure": (p["qty"] * precio / (p["cash"] + p["qty"] * precio)) if precio else 0.0},
            "risk": asdict(agent.risk_cfg), "model": {"version": predictor.version, "type": predictor.tipo,
                                                       "hidden": predictor.h,
                                                       "device_entrenamiento": predictor.card["training"]["device"]},
            "llm": {"model": s.ollama_model, "url": s.ollama_url},
            "replay_progress": store.kv_get("replay_progress"),
        }

    @app.post("/api/agent/start", dependencies=protegido)
    def start():
        agent.start(); return {"status": agent.status}

    @app.post("/api/agent/pause", dependencies=protegido)
    def pause():
        agent.pause(); return {"status": agent.status}

    @app.post("/api/agent/stop", dependencies=protegido)
    def stop():
        agent.stop(); return {"status": agent.status, "portfolio": store.portfolio(LIVE)}

    @app.post("/api/agent/cycle", dependencies=protegido)
    async def cycle():
        return await asyncio.to_thread(agent.live_cycle, True)

    @app.post("/api/agent/liquidate", dependencies=protegido)
    async def liquidate():
        try:
            return await asyncio.to_thread(agent.liquidate)
        except MarketDataError as e:
            raise HTTPException(503, str(e))

    # ------------------------------------------------------------------ registros
    @app.get("/api/decisions", dependencies=protegido)
    def decisions(run_id: str = LIVE, limit: int = 50):
        return store.decisions(run_id, min(limit, 500))

    @app.get("/api/orders", dependencies=protegido)
    def orders(run_id: str = LIVE, limit: int = 100):
        return store.orders(run_id, min(limit, 1000))

    @app.get("/api/equity", dependencies=protegido)
    def equity(run_id: str = LIVE):
        return store.equity(run_id)

    @app.get("/api/events", dependencies=protegido)
    def events(run_id: str | None = None, level: str | None = None, limit: int = 100):
        return store.events(run_id, min(limit, 1000), level)

    # ------------------------------------------------------------------ configuración (solo humanos)
    @app.get("/api/config/risk", dependencies=protegido)
    def get_risk():
        return asdict(agent.risk_cfg)

    @app.put("/api/config/risk", dependencies=protegido)
    def put_risk(cambios: dict):
        if agent.status == "RUNNING":
            raise HTTPException(409, "Pausen o detengan el agente antes de cambiar los límites de riesgo")
        datos = asdict(agent.risk_cfg) | cambios
        try:
            nuevo = RiskConfig(**datos).validate()
        except (TypeError, ValueError) as e:
            raise HTTPException(422, f"Configuración inválida: {e}")
        nuevo.save(s.risk_path)
        agent.risk_cfg = nuevo
        store.log_event("warning", "config", "Límites de riesgo modificados por el usuario", None, cambios)
        return asdict(nuevo)

    # ------------------------------------------------------------------ LLM local (gateway protegido)
    @app.get("/api/llm/health", dependencies=protegido)
    def llm_health():
        try:
            return agent.llm.health() | {"hardware": hardware()}
        except LLMUnavailable as e:
            raise HTTPException(503, str(e))

    @app.post("/api/llm/ask", dependencies=protegido)
    async def llm_ask(q: Pregunta):
        """Consulta directa al modelo local a través del backend (demuestra inferencia sin nube)."""
        t0 = datetime.now(timezone.utc)
        try:
            r = await asyncio.to_thread(agent.llm.chat, [{"role": "user", "content": q.prompt}])
        except LLMUnavailable as e:
            raise HTTPException(503, str(e))
        return {"model": s.ollama_model, "respuesta": r.get("message", {}).get("content"),
                "latencia_ms": round((datetime.now(timezone.utc) - t0).total_seconds() * 1000, 1),
                "url_ollama": s.ollama_url}

    # ------------------------------------------------------------------ replay y evaluación
    @app.post("/api/replay", dependencies=protegido)
    async def replay(req: ReplayReq):
        if agent.status == "RUNNING":
            raise HTTPException(409, "Pausen el agente en vivo antes de ejecutar un replay")
        try:
            return await asyncio.to_thread(agent.replay, req.strategies, req.start, req.end, req.limit)
        except (ValueError, MarketDataError, LLMUnavailable) as e:
            raise HTTPException(422 if isinstance(e, ValueError) else 503, str(e))

    @app.get("/api/runs", dependencies=protegido)
    def runs():
        return [r | {"metrics": run_metrics(store, r["id"], agent.risk_cfg)} for r in store.runs()]

    @app.get("/api/model", dependencies=protegido)
    def model_card():
        return predictor.card

    return app


def app_factory() -> FastAPI:  # uvicorn app.main:app_factory --factory
    return create_app()
