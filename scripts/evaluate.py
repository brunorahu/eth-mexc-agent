from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.agent import STRATEGIES, Agent, run_metrics  # noqa: E402
from app.config import RiskConfig, load_settings  # noqa: E402
from app.llm import OllamaClient  # noqa: E402
from app.mexc import MexcClient  # noqa: E402
from app.model import Predictor  # noqa: E402
from app.store import Store  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--db", default=str(ROOT / "backend/data/evaluacion.db"))
    a = ap.parse_args()

    s = load_settings()
    s = replace(s, db_path=Path(a.db))
    pred = Predictor(s.model_dir)
    te = pred.card["splits"]["TE"]["emision"]
    start, end = a.start or te[0], a.end or te[1]
    cfg = RiskConfig(signal_threshold=float(pred.card["signal_threshold"])).validate()
    llm = OllamaClient(s.ollama_url, s.ollama_model, s.ollama_timeout)
    salud = llm.health()
    agent = Agent(s, Store(s.db_path), MexcClient(s.mexc_base_url), llm, pred, cfg)

    print(f"Replay {start} → {end} (limit={a.limit}) | modelo {pred.version} | LLM {s.ollama_model}", flush=True)
    t0 = datetime.now(timezone.utc)
    out = agent.replay(list(STRATEGIES), start, end, a.limit)
    dur = (datetime.now(timezone.utc) - t0).total_seconds()

    nombres = {"model_only": "Modelo sin LLM (regla δ)", "agent_llm": "Agente con LLM local", "buy_hold": "Comprar y mantener"}
    filas = []
    for strat, rid in out["runs"].items():
        m = run_metrics(agent.store, rid, cfg)
        filas.append({"estrategia": nombres[strat], **m})

    dest = ROOT / "docs" / "evaluacion"
    dest.mkdir(parents=True, exist_ok=True)
    with open(dest / "resultados.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader(); w.writerows(filas)
    for strat, rid in out["runs"].items():
        decs = agent.store.all_decisions(rid)
        with open(dest / f"decisiones_{strat}.csv", "w", newline="") as f:
            cols = ["candle_time", "close", "forecast", "proposed_action", "final_action", "rejected",
                    "reject_reason", "justification", "llm_latency_ms", "tool_calls"]
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader(); w.writerows(decs)
        with open(dest / f"ordenes_{strat}.csv", "w", newline="") as f:
            ords = agent.store.all_orders(rid)
            if ords:
                w = csv.DictWriter(f, fieldnames=list(ords[0].keys()))
                w.writeheader(); w.writerows(ords)

    def pct(x):
        return "no definido" if x is None else f"{100 * x:+.2f}%"

    md = [f"# Evaluación: modelo sin LLM vs agente con LLM vs comprar y mantener\n",
          f"- Modo: **REPLAY** con velas históricas de MEXC (no es conexión en vivo).",
          f"- Periodo: {filas[0]['desde']} → {filas[0]['hasta']} ({filas[0]['ciclos']} ciclos de decisión por estrategia); "
          f"periodo de prueba independiente (TE) del modelo `{pred.version}`.",
          f"- Ejecución: decisión al cierre de t, ejecución a la apertura de t+1, valoración al cierre de t+1.",
          f"- Costos: comisión {100 * cfg.fee_rate:.2f}% y deslizamiento {cfg.slippage_bps} pb por operación; "
          f"capital inicial {cfg.initial_capital:,.0f} USDT; liquidación final al último cierre con costos.",
          f"- LLM: `{s.ollama_model}` local (Ollama {salud['ollama_version']}), duración total {dur / 60:.1f} min.\n",
          "| Estrategia | Rend. neto | Máx. drawdown | Operaciones (compras/ventas) | Win Rate | Costos (USDT) | Rechazadas | Latencia LLM media (ms) |",
          "|---|---|---|---|---|---|---|---|"]
    for f_ in filas:
        lat = "—" if f_["latencia_llm_ms_media"] is None else f"{f_['latencia_llm_ms_media']:.0f}"
        md.append(f"| {f_['estrategia']} | {pct(f_['rendimiento_neto'])} | {pct(f_['max_drawdown'])} | "
                  f"{f_['compras']}/{f_['ventas']} | {pct(f_['win_rate']) if f_['win_rate'] is not None else 'no definido'} | "
                  f"{f_['costos_totales']:,.2f} | {f_['acciones_rechazadas']} | {lat} |")
    (dest / "EVALUACION.md").write_text("\n".join(md) + "\n")
    (dest / "meta.json").write_text(json.dumps({"tag": out["tag"], "start": start, "end": end, "limit": a.limit,
                                                "risk": asdict(cfg), "llm": salud, "duration_s": dur}, indent=2, default=str))
    print("\n".join(md))


if __name__ == "__main__":
    main()
