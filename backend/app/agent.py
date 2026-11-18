from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import risk as riskmod
from .config import RiskConfig, Settings
from .features import CandleValidationError, MIN_CANDLES, indicators, lag_vector, utcnow, validate_candles
from .llm import LLMInvalidResponse, LLMUnavailable, OllamaClient
from .mexc import MarketDataError, MexcClient
from .model import Predictor
from .paper import InsufficientBalance, fill
from .store import DuplicateCandle, Store

LIVE = "live"
STATUSES = ("STOPPED", "RUNNING", "PAUSED", "HALTED")
STRATEGIES = ("agent_llm", "model_only", "buy_hold")


def _pct(x):
    return None if x is None else f"{100 * float(x):+.2f}%"


class Agent:
    def __init__(self, settings: Settings, store: Store, mexc: MexcClient, llm: OllamaClient,
                 predictor: Predictor, risk_cfg: RiskConfig, clock=utcnow):
        self.s, self.store, self.mexc, self.llm, self.pred = settings, store, mexc, llm, predictor
        self.risk_cfg, self.clock = risk_cfg, clock
        self._cycle_lock = threading.Lock()
        self.store.ensure_run(LIVE, "live", "agent_llm", risk_cfg.initial_capital,
                              {"symbol": settings.symbol, "interval": settings.interval})

    # ================================================================== estado y control humano
    @property
    def status(self) -> str:
        return self.store.kv_get("live_status", "STOPPED")

    def set_status(self, status: str, reason: str, level: str = "info") -> None:
        assert status in STATUSES
        anterior = self.status
        self.store.kv_set("live_status", status)
        self.store.kv_set("live_status_reason", reason)
        self.store.kv_set("live_status_at", self.clock().isoformat(timespec="seconds"))
        self.store.log_event(level, "estado", f"{anterior} → {status}: {reason}", LIVE)

    def recover_on_startup(self) -> None:
        """Al reiniciar: el portafolio y las velas procesadas se recuperan de SQLite.
        Si el agente estaba en ejecución, queda en PAUSA hasta que una persona lo reanude."""
        p = self.store.portfolio(LIVE)
        self.store.log_event("info", "reinicio", "Backend iniciado; estado recuperado de SQLite", LIVE,
                             {"cash": p["cash"], "qty": p["qty"], "status_previo": self.status})
        if self.status == "RUNNING":
            self.set_status("PAUSED", "Reinicio del backend: estado recuperado, reanudar manualmente", "warning")

    def start(self) -> None:
        self.set_status("RUNNING", "Iniciado por el usuario")

    def pause(self, reason: str = "Pausado por el usuario") -> None:
        self.set_status("PAUSED", reason)

    def stop(self) -> None:
        # Detener NO liquida: se bloquean nuevas órdenes y la posición se conserva y se muestra.
        self.set_status("STOPPED", "Detenido por el usuario: órdenes bloqueadas, la posición se conserva")

    def liquidate(self) -> dict:
        """Acción EXPLÍCITA del usuario: vender toda la posición en Paper al precio actual."""
        with self._cycle_lock:
            p = self.store.portfolio(LIVE)
            if p["qty"] <= 0:
                return {"ok": False, "message": "No hay posición que liquidar"}
            precio = self.mexc.ticker_price(self.s.symbol)
            orden, nuevo = fill("SELL", p["qty"], precio, p, self.risk_cfg)
            etiqueta = "manual-" + self.clock().strftime("%Y%m%dT%H%M%S")
            dec = {"close": precio, "forecast": None, "indicators": {}, "proposed_action": "LIQUIDATE",
                   "justification": "Liquidación manual solicitada por el usuario", "final_action": "SELL",
                   "rejected": False}
            eq = {"candle_time": etiqueta, "cash": nuevo["cash"], "qty": nuevo["qty"], "price": precio,
                  "equity": nuevo["cash"]}
            self.store.commit_cycle(LIVE, etiqueta, dec, orden, nuevo, eq)
            self.store.log_event("warning", "liquidacion", f"Posición liquidada: {orden['qty']} a {orden['price']:.2f}", LIVE)
            return {"ok": True, "order": orden}

    # ================================================================== pasos 1 y 2
    def snapshot(self, candles: pd.DataFrame, now: datetime, max_age_hours: float | None) -> dict:
        info = validate_candles(candles, min_rows=MIN_CANDLES, now=now, max_age_hours=max_age_hours)
        closes = candles["close"].to_numpy(float)
        lags = lag_vector(closes)
        forecast = self.pred.predict(lags)
        last = candles.iloc[-1]
        return {"symbol": self.s.symbol, "interval": self.s.interval, "candle_time": last["candle_date"],
                "close": float(last["close"]), "forecast": forecast, "lags": lags,
                "indicators": indicators(closes), "data_age_hours": info.get("data_age_hours", 0.0),
                "recent": [{"fecha": d, "cierre": round(float(c), 2)}
                           for d, c in zip(candles["candle_date"].iloc[-30:], candles["close"].iloc[-30:])],
                "validation": info}

    # ================================================================== pasos 3 y 4
    def propose(self, strategy: str, snap: dict, p: dict) -> dict:
        delta = self.risk_cfg.signal_threshold
        en_pos = p["qty"] > 0
        allowed = ["HOLD", "SELL"] if en_pos else ["BUY", "HOLD"]
        if strategy == "buy_hold":
            return {"proposed_action": "HOLD" if en_pos else "BUY", "justification": "Comprar y mantener (referencia pasiva)"}
        if strategy == "model_only":
            arriba = snap["forecast"] > delta
            accion = ("HOLD" if en_pos else "BUY") if arriba else ("SELL" if en_pos else "HOLD")
            return {"proposed_action": accion,
                    "justification": f"Regla sin LLM: pronóstico {snap['forecast']:+.4%} {'>' if arriba else '≤'} δ={delta:.4%}"}
        # agent_llm
        precio = snap["close"]
        datos = {
            "symbol": snap["symbol"], "interval": snap["interval"], "candle_time": snap["candle_time"],
            "close": round(precio, 2),
            "pronostico_rendimiento_siguiente": f"{100 * snap['forecast']:+.3f}%",
            "umbral_senal": f"{100 * delta:.3f}%",
            "senal_modelo": "COMPRAR/MANTENER" if snap["forecast"] > delta else "EFECTIVO/VENDER",
            "nota_modelo": "Pronóstico GRU del rendimiento de la próxima vela diaria; su error típico (RMSE) ≈ 2.5 puntos porcentuales.",
            "indicadores": {"distancia_sma20": _pct(snap["indicators"]["dist_sma20"]),
                            "volatilidad20": _pct(snap["indicators"]["vol20"]),
                            "rsi14": None if snap["indicators"]["rsi14"] is None else round(snap["indicators"]["rsi14"], 1)},
            "rendimientos_recientes": [f"{100 * float(x):+.2f}%" for x in snap["lags"][:5]],
            "portafolio": {"efectivo_usdt": round(p["cash"], 2), "cantidad": p["qty"], "en_posicion": en_pos,
                           "precio_promedio": round(p["avg_price"], 2), "patrimonio_usdt": round(p["cash"] + p["qty"] * precio, 2)},
            "allowed_actions": allowed,
        }
        ctx = {"datos": datos, "herramientas": {
            "velas_recientes": snap["recent"],
            "limites_riesgo": {"max_exposicion": self.risk_cfg.max_exposure_pct,
                               "max_perdida_diaria": self.risk_cfg.max_daily_loss_pct,
                               "max_antiguedad_horas": self.risk_cfg.max_data_age_hours,
                               "apalancamiento": False, "ventas_en_corto": False}}}
        try:
            prop, meta = self.llm.propose(ctx)
        except LLMInvalidResponse as e:
            return {"proposed_action": "INVALID", "justification": None, "llm_raw": str(e),
                    "invalid": str(e)}
        return {"proposed_action": prop.action, "justification": prop.justification, "llm_raw": meta["raw"],
                "tool_calls": meta["tool_calls"] + [f"RECHAZADA:{t}" for t in meta["tools_rechazadas"]],
                "llm_latency_ms": meta["latency_ms"]}

    # ================================================================== pasos 5 y 6
    def process(self, run_id: str, strategy: str, snap: dict, ref_price: float, valuation_price: float,
                blocked: bool) -> dict:
        t0 = time.perf_counter()
        if self.store.is_processed(run_id, snap["candle_time"]):
            raise DuplicateCandle(f"La vela {snap['candle_time']} ya fue procesada en {run_id}")
        p = self.store.portfolio(run_id)
        ultimo = self.store.last_equity(run_id)
        prev_equity = ultimo["equity"] if ultimo else p["initial_cash"]

        prop = self.propose(strategy, snap, p)
        motivos, orden, nuevo, halt = [], None, dict(p), False
        if "invalid" in prop:
            motivos.append(f"Propuesta del LLM rechazada: {prop['invalid']}")
            final = "HOLD"
        else:
            rr = riskmod.evaluate(prop["proposed_action"], price=ref_price, cash=p["cash"], qty=p["qty"],
                                  prev_equity=prev_equity, data_age_hours=snap["data_age_hours"],
                                  blocked=blocked, cfg=self.risk_cfg)
            final, motivos, halt = rr.action, rr.reasons, rr.halt
            if rr.approved and rr.action in ("BUY", "SELL"):
                try:
                    orden, nuevo = fill(rr.action, rr.qty, ref_price, p, self.risk_cfg)
                except InsufficientBalance as e:
                    final, orden, nuevo = "HOLD", None, dict(p)
                    motivos.append(str(e))
        rechazada = bool(motivos)
        eq_val = nuevo["cash"] + nuevo["qty"] * valuation_price
        dec = {"close": snap["close"], "forecast": snap["forecast"], "indicators": snap["indicators"],
               "proposed_action": prop["proposed_action"], "justification": prop.get("justification"),
               "llm_raw": prop.get("llm_raw"), "tool_calls": prop.get("tool_calls"),
               "llm_latency_ms": prop.get("llm_latency_ms"), "final_action": final, "rejected": rechazada,
               "reject_reason": " | ".join(motivos) or None, "cycle_ms": round((time.perf_counter() - t0) * 1000, 1)}
        eq = {"candle_time": snap["candle_time"], "cash": nuevo["cash"], "qty": nuevo["qty"],
              "price": valuation_price, "equity": eq_val}
        dec_id = self.store.commit_cycle(run_id, snap["candle_time"], dec, orden, nuevo, eq)
        if rechazada:
            self.store.log_event("warning", "rechazo", f"{snap['candle_time']}: {dec['reject_reason']}", run_id)
        if orden:
            self.store.log_event("info", "orden_paper",
                                 f"{orden['side']} {orden['qty']:.6f} @ {orden['price']:.2f} (ref {ref_price:.2f})", run_id)
        return {"decision_id": dec_id, **dec, "order": orden, "equity": eq_val, "halt": halt,
                "invalid": "invalid" in prop}

    # ================================================================== ciclo en vivo
    def live_cycle(self, manual: bool = False) -> dict:
        if not self._cycle_lock.acquire(blocking=False):
            return {"ok": False, "message": "Ya hay un ciclo en ejecución"}
        try:
            estado = self.status
            if estado != "RUNNING" and not manual:
                return {"ok": False, "message": f"Agente en {estado}: no se toman decisiones"}
            now = self.clock()
            try:
                velas = self.mexc.closed_klines(self.s.symbol, self.s.interval, limit=120, now=now)
                snap = self.snapshot(velas, now, self.risk_cfg.max_data_age_hours)
                if self.store.is_processed(LIVE, snap["candle_time"]):
                    msg = f"Vela {snap['candle_time']} ya procesada: se espera la siguiente (sin duplicar)"
                    if manual:
                        self.store.log_event("info", "duplicado", msg, LIVE)
                    return {"ok": True, "message": msg, "duplicate": True}
                precio = self.mexc.ticker_price(self.s.symbol)
                res = self.process(LIVE, "agent_llm", snap, ref_price=precio, valuation_price=precio,
                                   blocked=estado in ("STOPPED", "HALTED"))
            except DuplicateCandle as e:
                self.store.log_event("info", "duplicado", str(e), LIVE)
                return {"ok": True, "message": str(e), "duplicate": True}
            except (MarketDataError, CandleValidationError, LLMUnavailable) as e:
                tipo = {MarketDataError: "mexc", CandleValidationError: "datos", LLMUnavailable: "ollama"}[type(e)]
                self.store.log_event("error", tipo, str(e), LIVE)
                if estado == "RUNNING":
                    self.set_status("PAUSED", f"Pausa automática por error de {tipo}: {e}", "error")
                return {"ok": False, "message": str(e), "error": tipo}
            if res["halt"] and estado == "RUNNING":
                self.set_status("HALTED", res["reject_reason"], "error")
            elif res["invalid"] and estado == "RUNNING":
                self.set_status("PAUSED", f"Pausa automática: {res['reject_reason']}", "error")
            self.store.kv_set("last_snapshot", {k: v for k, v in snap.items() if k not in ("lags", "recent")}
                              | {"lags": [float(x) for x in snap["lags"]]})
            return {"ok": True, "result": res}
        finally:
            self._cycle_lock.release()

    # ================================================================== replay histórico
    def replay(self, strategies: list[str], start: str, end: str, limit: int | None = None,
               tag: str | None = None) -> dict:
        """Repetición con datos históricos de MEXC, identificada como REPLAY y separada del modo en vivo.
        Decisión al cierre de t; ejecución a la APERTURA de t+1; valoración al cierre de t+1."""
        for s in strategies:
            if s not in STRATEGIES:
                raise ValueError(f"Estrategia desconocida: {s}")
        ini, fin = datetime.fromisoformat(start).replace(tzinfo=timezone.utc), datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
        hist = self.mexc.history(self.s.symbol, self.s.interval, ini - timedelta(days=60), fin + timedelta(days=2))
        hist = hist[hist["close_time"] <= self.clock()].drop_duplicates("candle_date", keep="last").reset_index(drop=True)
        idx = [i for i in range(MIN_CANDLES - 1, len(hist) - 1) if start <= hist.loc[i, "candle_date"] <= end]
        if limit:
            idx = idx[:limit]
        if not idx:
            raise ValueError("No hay velas en el periodo solicitado")
        tag = tag or self.clock().strftime("%Y%m%dT%H%M%S")
        runs = {s: f"replay-{tag}-{s}" for s in strategies}
        for s, rid in runs.items():
            self.store.ensure_run(rid, "replay", s, self.risk_cfg.initial_capital,
                                  {"start": start, "end": end, "n": len(idx), "risk": self.risk_cfg.__dict__,
                                   "model": self.pred.version, "llm": self.llm.model if s == "agent_llm" else None})
            self.store.set_run_status(rid, "running")
        self.store.log_event("info", "replay", f"Replay {tag}: {len(idx)} velas {hist.loc[idx[0],'candle_date']} → "
                             f"{hist.loc[idx[-1],'candle_date']} | estrategias {strategies}")
        errores = 0
        for k, i in enumerate(idx):
            ventana = hist.iloc[max(0, i - 59): i + 1].reset_index(drop=True)
            snap = self.snapshot(ventana, now=ventana["close_time"].iloc[-1], max_age_hours=None)
            siguiente = hist.iloc[i + 1]
            for s, rid in runs.items():
                if self.store.is_processed(rid, snap["candle_time"]):
                    continue
                try:
                    self.process(rid, s, snap, ref_price=float(siguiente["open"]),
                                 valuation_price=float(siguiente["close"]), blocked=False)
                except LLMUnavailable as e:
                    errores += 1
                    self.store.log_event("error", "ollama", f"Replay {snap['candle_time']}: {e}", rid)
                    if errores >= 3:
                        for r in runs.values():
                            self.store.set_run_status(r, "failed")
                        raise
            self.store.kv_set("replay_progress", {"tag": tag, "done": k + 1, "total": len(idx),
                                                  "candle": snap["candle_time"]})
        for rid in runs.values():
            self.store.set_run_status(rid, "done")
        return {"tag": tag, "runs": runs, "candles": len(idx)}


def run_metrics(store: Store, run_id: str, cfg: RiskConfig) -> dict:
    """Rendimiento neto (liquidando al último precio con costo), drawdown, operaciones, Win Rate, costos,
    rechazos y latencias."""
    eq = store.equity(run_id, limit=100_000)
    p = store.portfolio(run_id)
    decs = store.all_decisions(run_id)
    ords = store.all_orders(run_id)
    if not eq:
        return {"run_id": run_id, "ciclos": 0}
    serie = np.array([e["equity"] for e in eq], float)
    ultimo = eq[-1]
    slip = cfg.slippage_bps / 1e4
    final_liq = ultimo["cash"] + ultimo["qty"] * ultimo["price"] * (1 - slip) * (1 - cfg.fee_rate)
    pico = np.maximum.accumulate(np.concatenate([[p["initial_cash"]], serie]))
    dd = float((np.concatenate([[p["initial_cash"]], serie]) / pico - 1).min())
    ventas = [o for o in ords if o["side"] == "SELL"]
    lat = [d["llm_latency_ms"] for d in decs if d["llm_latency_ms"] is not None]
    return {
        "run_id": run_id, "ciclos": len(decs), "desde": decs[0]["candle_time"] if decs else None,
        "hasta": decs[-1]["candle_time"] if decs else None,
        "patrimonio_final_liquidado": round(final_liq, 2),
        "rendimiento_neto": final_liq / p["initial_cash"] - 1,
        "max_drawdown": dd, "compras": sum(o["side"] == "BUY" for o in ords), "ventas": len(ventas),
        "operaciones_cerradas": len(ventas),
        "win_rate": (sum((o["realized_pnl"] or 0) > 0 for o in ventas) / len(ventas)) if ventas else None,
        "costos_totales": round(p["fees"] + p["slippage"], 2), "comisiones": round(p["fees"], 2),
        "deslizamiento": round(p["slippage"], 2),
        "acciones_rechazadas": sum(d["rejected"] for d in decs),
        "latencia_llm_ms_media": float(np.mean(lat)) if lat else None,
        "latencia_llm_ms_p95": float(np.percentile(lat, 95)) if lat else None,
        "ciclo_ms_medio": float(np.mean([d["cycle_ms"] for d in decs if d["cycle_ms"] is not None])) if decs else None,
    }
