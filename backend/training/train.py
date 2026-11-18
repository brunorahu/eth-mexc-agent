from __future__ import annotations

import argparse
import copy
import json
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from app import __version__ as APP_VERSION  # noqa: E402
from app.features import FEATURES, N_LAGS, supervised, validate_candles  # noqa: E402
from app.mexc import MexcClient  # noqa: E402
from app.model import build, file_sha256, to_input  # noqa: E402

MODEL_VERSION = "ethusdt-gru-1.0.0"
NEURONAS, LR, BATCH, MAX_EPOCAS, PACIENCIA = [16, 32, 64], 1e-3, 64, 300, 25
SEMILLAS = [42, 7, 2024]
DELTAS = [0.0001, 0.0005, 0.001, 0.002, 0.003]   # hasta cinco umbrales positivos predefinidos


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def seed_all(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False


def metricas(y, yhat) -> dict:
    e = yhat - y
    m = (np.sign(y) != 0) & (np.sign(yhat) != 0)
    return {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "prec_direccional": float((np.sign(y[m]) == np.sign(yhat[m])).mean()) if m.sum() else None, "n": int(len(y))}


def estrategia_umbral(pred, y, delta, fee=0.0) -> dict:
    """Largo 100% si pred > δ, efectivo en otro caso. Rendimiento neto con costo por cambio de posición."""
    pos = (pred > delta).astype(float)
    cambios = np.abs(np.diff(np.concatenate([[0.0], pos])))
    neto = pos * y - cambios * fee
    eq = np.cumprod(1 + neto)
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    return {"delta": delta, "rend_neto": float(eq[-1] - 1), "max_drawdown": dd,
            "operaciones": int(cambios.sum()), "dias_en_posicion": int(pos.sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="ETHUSDT")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--out", default=str(BACKEND / "models"))
    ap.add_argument("--data-dir", default=str(BACKEND / "data"))
    ap.add_argument("--require-t4", action="store_true")
    a = ap.parse_args()
    out, data_dir = Path(a.out), Path(a.data_dir)
    out.mkdir(parents=True, exist_ok=True); data_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- dispositivo
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    if a.require_t4 and (gpu is None or "T4" not in gpu):
        raise SystemExit("No hay una GPU T4 disponible: se detiene el entrenamiento profundo (protocolo, sección 4).")
    device = torch.device("cuda" if gpu else "cpu")
    try:
        nvsmi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=20).stdout
    except FileNotFoundError:
        nvsmi = ""
    log(f"Dispositivo: {device} {gpu or ''}")

    # ---------------------------------------------------------------- datos de MEXC
    ahora = datetime.now(timezone.utc)
    cli = MexcClient()
    info = cli.symbol_info(a.symbol)
    raw = cli.history(a.symbol, "1d", datetime.fromisoformat(a.start).replace(tzinfo=timezone.utc), ahora)
    n_desc = len(raw)
    raw = raw[raw["close_time"] <= ahora]                     # solo velas cerradas
    n_dup = int(raw["candle_date"].duplicated(keep=False).sum())
    raw = raw.sort_values("open_time").drop_duplicates("candle_date", keep="last").reset_index(drop=True)
    audit = validate_candles(raw)
    fechas = pd.to_datetime(raw["candle_date"])
    faltan = pd.date_range(fechas.min(), fechas.max(), freq="D").difference(fechas)
    audit.update({"descargadas": n_desc, "duplicadas_eliminadas(regla: conservar última)": n_dup,
                  "dias_sin_vela": len(faltan), "ejemplos_dias_sin_vela": [str(d.date()) for d in faltan[:10]],
                  "horas_apertura_distintas_de_00UTC": int((raw["open_time"].dt.hour != 0).sum()),
                  "volumen_cero": int((raw["volume"] == 0).sum()), "nota": "Calendario continuo; no se rellenan días sin vela."})
    raw_path = data_dir / f"{a.symbol.lower()}_1d_mexc.csv"
    raw.to_csv(raw_path, index=False)
    log(f"{len(raw)} velas diarias cerradas {audit['first']} → {audit['last']}; sin vela: {len(faltan)} días")
    if len(raw) < 1500:
        raise SystemExit(f"Solo {len(raw)} observaciones: el protocolo exige ≥ 1,500.")

    # ---------------------------------------------------------------- objetivo y particiones
    sup = supervised(raw)
    n = len(sup)
    i_va, i_te = int(round(n * 0.70)), int(round(n * 0.85))
    sup["conjunto"] = np.select([sup.index < i_va, sup.index < i_te], ["TR", "VA"], "TE")
    TR, VA, TE = (sup[sup.conjunto == s] for s in ("TR", "VA", "TE"))
    assert TR.fecha_objetivo.max() < VA.fecha_objetivo.min() and VA.fecha_objetivo.max() < TE.fecha_objetivo.min()
    particiones = {s: {"n": len(d), "emision": [d.fecha_emision.min(), d.fecha_emision.max()],
                       "objetivo": [d.fecha_objetivo.min(), d.fecha_objetivo.max()]}
                   for s, d in (("TR", TR), ("VA", VA), ("TE", TE))}
    log(f"Particiones: { {k: v['n'] for k, v in particiones.items()} }")

    Xtr, Xva, Xte = (d[FEATURES].to_numpy(float) for d in (TR, VA, TE))
    ytr, yva, yte = (d["y"].to_numpy(float) for d in (TR, VA, TE))
    esc_x = StandardScaler().fit(Xtr)
    y_mean, y_std = float(ytr.mean()), float(ytr.std())

    # ---------------------------------------------------------------- referencias
    lin = Pipeline([("esc", StandardScaler()), ("ols", LinearRegression())]).fit(Xtr, ytr)
    refs = {"cero": {s: metricas(y, np.zeros_like(y)) for s, y in (("TR", ytr), ("VA", yva))},
            "regresion_lineal": {s: metricas(y, lin.predict(X)) for s, X, y in (("TR", Xtr, ytr), ("VA", Xva, yva))},
            "media_TR": {s: metricas(y, np.full_like(y, y_mean)) for s, y in (("TR", ytr), ("VA", yva))}}

    # ---------------------------------------------------------------- entrenamiento
    def entrenar(tipo, h, semilla):
        seed_all(semilla)
        net = build(tipo, h).to(device)
        opt, crit = torch.optim.Adam(net.parameters(), lr=LR), nn.MSELoss()
        T = lambda X: torch.tensor(to_input(tipo, esc_x.transform(X)), device=device)  # noqa: E731
        xt, xv = T(Xtr), T(Xva)
        yt = torch.tensor(((ytr - y_mean) / y_std).astype(np.float32), device=device)
        yv = torch.tensor(((yva - y_mean) / y_std).astype(np.float32), device=device)
        g = torch.Generator().manual_seed(semilla)
        mejor, estado, mejor_ep, sin, hist = np.inf, None, 0, 0, []
        t0 = time.perf_counter()
        for ep in range(1, MAX_EPOCAS + 1):
            net.train()
            for idx in torch.randperm(len(xt), generator=g).split(BATCH):
                idx = idx.to(device)
                opt.zero_grad(); l = crit(net(xt[idx]), yt[idx]); l.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                ltr, lva = crit(net(xt), yt).item(), crit(net(xv), yv).item()
            if not (np.isfinite(ltr) and np.isfinite(lva)):
                raise SystemExit(f"Pérdida no finita en época {ep}: revisen escalas y tasa de aprendizaje.")
            hist.append((ep, ltr, lva))
            if lva < mejor - 1e-6:
                mejor, mejor_ep, sin, estado = lva, ep, 0, copy.deepcopy(net.state_dict())
            else:
                sin += 1
                if sin >= PACIENCIA:
                    break
        net.load_state_dict(estado); net.eval()
        pred = lambda X: net(torch.tensor(to_input(tipo, esc_x.transform(X)), device=device)).detach().cpu().numpy() * y_std + y_mean  # noqa: E731
        return net, {"h": h, "semilla": semilla, "epoca_mejor": mejor_ep, "epocas": ep,
                     "tiempo_s": round(time.perf_counter() - t0, 2), "historial": hist,
                     "TR": metricas(ytr, pred(Xtr)), "VA": metricas(yva, pred(Xva)),
                     "params": sum(p.numel() for p in net.parameters() if p.requires_grad)}, pred

    busqueda = []
    for h in NEURONAS:
        _, r, _ = entrenar("GRU", h, SEMILLAS[0])
        busqueda.append(r)
        log(f"GRU h={h}: RMSE VA={100*r['VA']['RMSE']:.4f}% (cero {100*refs['cero']['VA']['RMSE']:.4f}%) épocas={r['epocas']}")
    h_best = min(busqueda, key=lambda r: r["VA"]["RMSE"])["h"]
    semillas = []
    for s in SEMILLAS:
        net, r, pred = entrenar("GRU", h_best, s)
        semillas.append(r)
        if s == SEMILLAS[0]:
            net_final, pred_final = net, pred
    rm = [r["VA"]["RMSE"] for r in semillas]
    log(f"Mejor h={h_best}; RMSE VA 3 semillas = {100*np.mean(rm):.4f}% ± {100*np.std(rm, ddof=1):.4f}%")

    # ---------------------------------------------------------------- umbral δ en VA (costo 0% de MEXC)
    pva = pred_final(Xva)
    candidatos = [estrategia_umbral(pva, yva, d) for d in DELTAS]
    delta = sorted(candidatos, key=lambda c: (-c["rend_neto"], -c["max_drawdown"]))[0]["delta"]
    log(f"δ elegido en VA: {delta}")

    # ---------------------------------------------------------------- TE: una sola evaluación
    pte = pred_final(Xte)
    te = {"LSTM": metricas(yte, pte), "cero": metricas(yte, np.zeros_like(yte)),
          "regresion_lineal": metricas(yte, lin.predict(Xte)),
          "estrategia_umbral": estrategia_umbral(pte, yte, delta),
          "comprar_y_mantener": {"rend_neto": float(np.prod(1 + yte) - 1)}}
    log(f"TE: RMSE LSTM={100*te['LSTM']['RMSE']:.4f}% cero={100*te['cero']['RMSE']:.4f}%")

    # ---------------------------------------------------------------- artefactos versionados
    torch.save({k: v.cpu() for k, v in net_final.state_dict().items()}, out / "model.pt")
    joblib.dump({"esc_x": esc_x, "y_mean": y_mean, "y_std": y_std, "features": FEATURES}, out / "scaler.joblib")
    card = {
        "version": MODEL_VERSION, "app_version": APP_VERSION,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "symbol": a.symbol, "interval": "1d", "source": "MEXC Spot v3 /api/v3/klines (público)",
        "exchange_info": info, "features": FEATURES, "n_lags": N_LAGS,
        "target": "y_t = close_{t+1} / close_t - 1 (cierre diario UTC)",
        "architecture": {"type": "GRU", "hidden": h_best, "layers": 1, "dropout": 0.2,
                         "input_shape": [None, N_LAGS, 1], "params": semillas[0]["params"]},
        "training": {"loss": "MSE (y estandarizado con TR)", "optimizer": "Adam", "lr": LR, "batch": BATCH,
                     "max_epochs": MAX_EPOCAS, "patience": PACIENCIA, "seed": SEMILLAS[0],
                     "device": str(device), "gpu": gpu, "nvidia_smi": nvsmi[:2000] or None,
                     "torch": torch.__version__, "sklearn": sklearn.__version__, "python": platform.python_version()},
        "data": {"raw_file": raw_path.name, "raw_sha256": file_sha256(raw_path), "audit": audit},
        "splits": particiones,
        "selection": {"busqueda": [{k: v for k, v in r.items() if k != "historial"} for r in busqueda],
                      "semillas": [{k: v for k, v in r.items() if k != "historial"} for r in semillas],
                      "rmse_va_media": float(np.mean(rm)), "rmse_va_desv": float(np.std(rm, ddof=1)),
                      "referencias": refs, "umbral_candidatos_VA": candidatos},
        "signal_threshold": delta,
        "test": te,
        "artifacts": {"model_sha256": file_sha256(out / "model.pt"), "scaler_sha256": file_sha256(out / "scaler.joblib")},
        "notes": ("Evidencia T4: ejecutar training/train_colab.ipynb en Colab con --require-t4. "
                  "Si device=cpu, este artefacto es de desarrollo y no sustituye la evidencia de T4."),
    }
    (out / "model_card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False, default=str))
    pd.DataFrame([(r["h"], r["semilla"], *t) for r in busqueda + semillas for t in r["historial"]],
                 columns=["h", "semilla", "epoca", "perdida_TR", "perdida_VA"]).to_csv(out / "curvas.csv", index=False)
    log(f"Artefactos guardados en {out}")


if __name__ == "__main__":
    main()
