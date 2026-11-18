"""Controles de riesgo, validación del LLM, herramientas, variables y ausencia de órdenes reales."""
import json
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.agent import LIVE, run_metrics
from app.config import RiskConfig
from app.features import FEATURES, lag_vector, supervised, validate_candles, CandleValidationError
from app.llm import LLMInvalidResponse, OllamaClient
from app.main import create_app
from app.risk import evaluate
from conftest import NOW, TEST_KEY, FakeLLM, FakeMexc, make_candles, new_agent

DATOS = {"symbol": "ETHUSDT", "candle_time": "2026-11-16", "allowed_actions": ["BUY", "HOLD"]}


# ---------------------------------------------------------------- riesgo
def test_sin_ventas_en_corto():
    r = evaluate("SELL", price=65000, cash=50000, qty=0, prev_equity=50000, data_age_hours=1, blocked=False, cfg=RiskConfig())
    assert not r.approved and "corto" in r.reasons[0]


def test_limite_de_exposicion_y_sin_apalancamiento():
    cfg = RiskConfig(max_exposure_pct=0.5)
    r = evaluate("BUY", price=50000, cash=50000, qty=0, prev_equity=50000, data_age_hours=1, blocked=False, cfg=cfg)
    assert r.approved and r.qty * 50000 <= 25000 + 1e-6
    r2 = evaluate("BUY", price=50000, cash=25000, qty=0.5, prev_equity=50000, data_age_hours=1, blocked=False, cfg=cfg)
    assert not r2.approved and "Exposición máxima" in r2.reasons[0]
    with pytest.raises(ValueError, match="apalancamiento"):
        RiskConfig(max_exposure_pct=2.0).validate()


def test_perdida_diaria_maxima_bloquea():
    r = evaluate("BUY", price=50000, cash=40000, qty=0, prev_equity=50000, data_age_hours=1, blocked=False,
                 cfg=RiskConfig(max_daily_loss_pct=0.05))
    assert not r.approved and r.halt and "pérdida diaria" in r.reasons[0]


def test_config_de_riesgo_solo_por_api_protegida(settings, predictor):
    app = create_app(settings, mexc=FakeMexc(), llm=FakeLLM(), predictor=predictor, start_loop=False)
    with TestClient(app) as c:
        h = {"X-API-Key": TEST_KEY}
        assert c.put("/api/config/risk", json={"max_exposure_pct": 0.5}).status_code == 401
        assert c.put("/api/config/risk", json={"max_exposure_pct": 3}, headers=h).status_code == 422
        assert c.put("/api/config/risk", json={"max_exposure_pct": 0.5}, headers=h).json()["max_exposure_pct"] == 0.5
        c.post("/api/agent/start", headers=h)
        assert c.put("/api/config/risk", json={"max_exposure_pct": 0.6}, headers=h).status_code == 409


# ---------------------------------------------------------------- validación de la respuesta del LLM
@pytest.mark.parametrize("raw,msg", [
    ('{"action":"BUY","symbol":"ETHUSDT","candle_time":"2026-11-16","justification":"x".join(["a","b"])}', "JSON"),
    ('{"action":"BUY","symbol":"BTCUSDT","candle_time":"2026-11-16","justification":"pronóstico alto"}', "Símbolo"),
    ('{"action":"BUY","symbol":"ETHUSDT","candle_time":"2026-11-15","justification":"pronóstico alto"}', "Fecha"),
    ('{"action":"SELL","symbol":"ETHUSDT","candle_time":"2026-11-16","justification":"pronóstico bajo"}', "no permitida"),
    ('{"action":"SHORT","symbol":"ETHUSDT","candle_time":"2026-11-16","justification":"pronóstico bajo"}', "Campos"),
    ('{"action":"BUY","symbol":"ETHUSDT","candle_time":"2026-11-16"}', "Campos"),
])
def test_validacion_estricta_de_la_propuesta(raw, msg):
    with pytest.raises(LLMInvalidResponse, match=msg):
        OllamaClient.validate(raw, DATOS)


def test_propuesta_valida_y_herramienta_no_permitida_rechazada():
    pasos = []

    def handler(req):
        body = json.loads(req.content)
        pasos.append(body)
        if len(pasos) == 1:   # el modelo pide una herramienta permitida y una inexistente
            return httpx.Response(200, json={"message": {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "obtener_portafolio", "arguments": {}}},
                {"function": {"name": "enviar_orden_real", "arguments": {"qty": 99}}}]}})
        return httpx.Response(200, json={"message": {"role": "assistant", "content": json.dumps(
            {"action": "BUY", "symbol": "ETHUSDT", "candle_time": "2026-11-16", "justification": "pronóstico > δ"})}})

    llm = OllamaClient("http://127.0.0.1:11434", "m", transport=httpx.MockTransport(handler))
    ctx = {"datos": DATOS | {"portafolio": {"efectivo_usdt": 50000}}, "herramientas": {}}
    prop, meta = llm.propose(ctx)
    assert prop.action == "BUY" and meta["tool_calls"] == ["obtener_portafolio"]
    assert meta["tools_rechazadas"] == ["enviar_orden_real"]
    assert "format" in pasos[1]                                   # segunda llamada exige el esquema JSON
    assert all("api_key" not in json.dumps(p).lower() for p in pasos)  # el LLM no recibe claves


# ---------------------------------------------------------------- variables y velas
def test_mismas_variables_que_en_entrenamiento():
    df = make_candles(80)
    sup = supervised(df)
    fila = sup.iloc[-1]
    i = df.index[df.candle_date == fila.fecha_emision][0]
    np.testing.assert_allclose(lag_vector(df["close"].to_numpy()[: i + 1]), fila[FEATURES].to_numpy(float))


def test_vela_abierta_no_se_usa(settings, predictor):
    df = make_candles(60)
    abierta = df.copy()
    abierta.loc[abierta.index[-1], "close_time"] = NOW.replace(hour=23)   # cierra en el futuro
    with pytest.raises(CandleValidationError, match="no cierra"):
        validate_candles(abierta, now=NOW)
    ag = new_agent(settings, predictor, mexc=FakeMexc(abierta))
    snap = ag.snapshot(FakeMexc(abierta).closed_klines("ETHUSDT", "1d", now=NOW), NOW, 48)
    assert snap["candle_time"] == df.candle_date.iloc[-2]                 # se usó la última CERRADA


def test_ventana_insuficiente_y_desorden():
    with pytest.raises(CandleValidationError, match="insuficiente"):
        validate_candles(make_candles(10))
    with pytest.raises(CandleValidationError, match="orden"):
        validate_candles(make_candles(40).iloc[::-1].reset_index(drop=True))


# ---------------------------------------------------------------- Paper y ausencia de órdenes reales
def test_no_existe_ruta_de_ordenes_reales():
    fuente = "".join(p.read_text() for p in (Path(__file__).parents[1] / "app").glob("*.py"))
    assert "/api/v3/order" not in fuente and "signature" not in fuente.lower()


def test_replay_y_metricas(settings, predictor):
    ag = new_agent(settings, predictor, mexc=FakeMexc(make_candles(90)), llm=FakeLLM())
    out = ag.replay(["agent_llm", "model_only", "buy_hold"], "2026-09-20", "2026-11-14")
    assert out["candles"] >= 20
    m = run_metrics(ag.store, out["runs"]["buy_hold"], ag.risk_cfg)
    assert m["ciclos"] == out["candles"] and m["compras"] == 1
    assert ag.store.decisions(LIVE) == []          # el replay está separado del modo en vivo
