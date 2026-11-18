"""Los nueve escenarios de error del protocolo (sección 15) con su respuesta esperada:
API sin respuesta o con límite de solicitudes; dato atrasado; Ollama detenido; JSON inválido; clave incorrecta;
saldo insuficiente; evento duplicado; reinicio del backend; botón de detención."""
import httpx
import pytest
from fastapi.testclient import TestClient

from app.agent import LIVE
from app.config import RiskConfig
from app.llm import LLMUnavailable, OllamaClient
from app.main import create_app
from app.mexc import MarketDataError, MexcClient
from app.paper import InsufficientBalance, fill
from app.risk import evaluate
from app.store import Store
from conftest import NOW, TEST_KEY, FakeLLM, FakeMexc, make_candles, new_agent


# 1 ─ API de MEXC sin respuesta o con límite de solicitudes ───────────────────────────────────────────
def test_1_mexc_rate_limit_reintenta_con_espera_creciente():
    respuestas = iter([httpx.Response(429), httpx.Response(429), httpx.Response(200, json={"price": "65000.5"})])
    esperas = []
    cli = MexcClient(transport=httpx.MockTransport(lambda req: next(respuestas)), sleep=esperas.append)
    assert cli.ticker_price("ETHUSDT") == 65000.5
    assert esperas == [1.0, 2.0]  # espera creciente


def test_1_mexc_sin_respuesta_pausa_al_agente(settings, predictor):
    def caida(req):
        raise httpx.ConnectTimeout("timeout")
    esperas = []
    cli = MexcClient(transport=httpx.MockTransport(caida), sleep=esperas.append)
    with pytest.raises(MarketDataError, match="tras 3 intentos"):
        cli.ticker_price("ETHUSDT")
    assert esperas == [1.0, 2.0]

    ag = new_agent(settings, predictor, mexc=FakeMexc(error=MarketDataError("MEXC no disponible")))
    ag.start()
    r = ag.live_cycle()
    assert r["error"] == "mexc" and ag.status == "PAUSED"
    assert ag.store.decisions(LIVE) == [] and ag.store.orders(LIVE) == []


# 2 ─ Dato atrasado ───────────────────────────────────────────────────────────────────────────────────
def test_2_dato_atrasado_no_opera(settings, predictor):
    from datetime import datetime, timezone
    viejas = make_candles(last_date=datetime(2026, 11, 10, tzinfo=timezone.utc))
    ag = new_agent(settings, predictor, mexc=FakeMexc(viejas))
    ag.start()
    r = ag.live_cycle()
    assert r["error"] == "datos" and "atrasado" in r["message"]
    assert ag.status == "PAUSED" and ag.store.orders(LIVE) == []


# 3 ─ Ollama detenido ─────────────────────────────────────────────────────────────────────────────────
def test_3_ollama_detenido(settings, predictor):
    def caido(req):
        raise httpx.ConnectError("connection refused")
    llm = OllamaClient("http://127.0.0.1:11434", "qwen2.5:3b", transport=httpx.MockTransport(caido), sleep=lambda s: None)
    ag = new_agent(settings, predictor, llm=llm)
    ag.start()
    r = ag.live_cycle()
    assert r["error"] == "ollama" and ag.status == "PAUSED"
    assert ag.store.orders(LIVE) == []
    assert not ag.store.is_processed(LIVE, "2026-11-16")  # la vela se reintentará al reanudar


# 4 ─ JSON inválido ───────────────────────────────────────────────────────────────────────────────────
def test_4_json_invalido_se_rechaza(settings, predictor):
    def basura(req):
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "Compra todo YA!!!"}})
    llm = OllamaClient("http://127.0.0.1:11434", "qwen2.5:3b", transport=httpx.MockTransport(basura), sleep=lambda s: None)
    ag = new_agent(settings, predictor, llm=llm)
    ag.start()
    r = ag.live_cycle()
    dec = ag.store.decisions(LIVE)[0]
    assert dec["rejected"] == 1 and "no es JSON" in dec["reject_reason"] and dec["final_action"] == "HOLD"
    assert ag.store.orders(LIVE) == [] and ag.status == "PAUSED"


# 5 ─ Clave incorrecta ────────────────────────────────────────────────────────────────────────────────
def test_5_api_key_ausente_o_incorrecta(settings, predictor):
    app = create_app(settings, mexc=FakeMexc(), llm=FakeLLM(), predictor=predictor, start_loop=False)
    with TestClient(app) as c:
        assert c.get("/api/health").status_code == 200
        assert c.get("/api/status").status_code == 401
        assert c.get("/api/status", headers={"X-API-Key": "incorrecta"}).status_code == 401
        assert c.post("/api/agent/start", headers={"X-API-Key": "incorrecta"}).status_code == 401
        assert c.get("/api/status", headers={"X-API-Key": TEST_KEY}).status_code == 200
        eventos = c.get("/api/events", headers={"X-API-Key": TEST_KEY}).json()
        assert any(e["kind"] == "auth" for e in eventos)


def test_5_sin_api_key_configurada_no_arranca(settings, predictor):
    from dataclasses import replace
    with pytest.raises(RuntimeError, match="API_KEY"):
        create_app(replace(settings, api_key=""), mexc=FakeMexc(), llm=FakeLLM(), predictor=predictor)


# 6 ─ Saldo insuficiente ──────────────────────────────────────────────────────────────────────────────
def test_6_saldo_insuficiente():
    cfg = RiskConfig()
    r = evaluate("BUY", price=65000, cash=1.0, qty=0, prev_equity=1.0, data_age_hours=1, blocked=False, cfg=cfg)
    assert not r.approved and r.action == "HOLD" and "Saldo insuficiente" in r.reasons[0]
    with pytest.raises(InsufficientBalance):
        fill("BUY", 1.0, 65000, {"cash": 100.0, "qty": 0.0, "avg_price": 0.0}, cfg)
    with pytest.raises(InsufficientBalance):
        fill("SELL", 1.0, 65000, {"cash": 100.0, "qty": 0.5, "avg_price": 60000.0}, cfg)


# 7 ─ Evento duplicado ────────────────────────────────────────────────────────────────────────────────
def test_7_vela_duplicada_no_se_procesa_dos_veces(settings, predictor):
    ag = new_agent(settings, predictor, llm=FakeLLM("BUY"))
    ag.start()
    assert ag.live_cycle()["ok"]
    r2 = ag.live_cycle(manual=True)
    assert r2.get("duplicate") is True
    assert len(ag.store.decisions(LIVE)) == 1 and len(ag.store.orders(LIVE)) == 1


# 8 ─ Reinicio del backend ────────────────────────────────────────────────────────────────────────────
def test_8_reinicio_recupera_estado_sin_duplicar(settings, predictor):
    ag = new_agent(settings, predictor, llm=FakeLLM("BUY"))
    ag.start()
    ag.live_cycle()
    antes = ag.store.portfolio(LIVE)
    assert antes["qty"] > 0
    ag.store.con.close()                                    # "se cae" el backend

    ag2 = new_agent(settings, predictor, llm=FakeLLM("BUY"), store=Store(settings.db_path))
    ag2.recover_on_startup()
    despues = ag2.store.portfolio(LIVE)
    assert (despues["cash"], despues["qty"]) == (antes["cash"], antes["qty"])
    assert ag2.status == "PAUSED"                           # requiere reanudación humana
    ag2.start()
    assert ag2.live_cycle().get("duplicate") is True        # misma vela: no se vuelve a comprar
    assert len(ag2.store.orders(LIVE)) == 1


# 9 ─ Botón de detención ──────────────────────────────────────────────────────────────────────────────
def test_9_detener_bloquea_ordenes_y_conserva_posicion(settings, predictor):
    app = create_app(settings, mexc=FakeMexc(), llm=FakeLLM("BUY"), predictor=predictor, start_loop=False)
    h = {"X-API-Key": TEST_KEY}
    with TestClient(app) as c:
        assert c.post("/api/agent/stop", headers=h).json()["status"] == "STOPPED"
        r = c.post("/api/agent/cycle", headers=h).json()       # ciclo manual con el agente detenido
        assert r["result"]["final_action"] == "HOLD" and "bloqueadas" in r["result"]["reject_reason"]
        assert c.get("/api/orders", headers=h).json() == []
        # el ciclo automático no decide si no está en RUNNING
        assert app.state.agent.live_cycle()["ok"] is False
