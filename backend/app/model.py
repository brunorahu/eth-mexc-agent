from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import joblib
import numpy as np

from .features import FEATURES, N_LAGS


class MLP(nn.Module):
    def __init__(self, h: int, p: float = 0.2):
        super().__init__()
        self.red = nn.Sequential(
            nn.Linear(N_LAGS, h), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h, h // 2), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h // 2, 1),
        )

    def forward(self, x):
        return self.red(x).squeeze(-1)


class RedLSTM(nn.Module):
    def __init__(self, h: int, p: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=h, num_layers=1, batch_first=True)
        self.drop, self.salida = nn.Dropout(p), nn.Linear(h, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.salida(self.drop(h_n[-1])).squeeze(-1)


class RedGRU(nn.Module):
    def __init__(self, h: int, p: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=h, num_layers=1, batch_first=True)
        self.drop, self.salida = nn.Dropout(p), nn.Linear(h, 1)

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.salida(self.drop(h_n[-1])).squeeze(-1)


def build(tipo: str, h: int) -> nn.Module:
    if tipo == "GRU":
        return RedGRU(h)
    if tipo == "LSTM":
        return RedLSTM(h)
    if tipo == "MLP":
        return MLP(h)
    raise ValueError(f"Tipo de modelo desconocido: {tipo}")


def to_input(tipo: str, X_scaled: np.ndarray) -> np.ndarray:
    """MLP: (n, 20). GRU/LSTM: (n, 20 pasos, 1 variable) del rendimiento más antiguo al más reciente."""
    if X_scaled.ndim != 2 or X_scaled.shape[1] != N_LAGS:
        raise ValueError(f"Forma esperada (n, {N_LAGS}); recibida {X_scaled.shape}")
    X = X_scaled.astype(np.float32)
    return np.ascontiguousarray(X[:, ::-1, None]) if tipo in ("LSTM", "GRU") else X


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Predictor:
    """Carga model.pt + scaler.joblib + model_card.json y valida el contrato de variables."""

    def __init__(self, model_dir: Path):
        self.dir = Path(model_dir)
        card_path = self.dir / "model_card.json"
        if not card_path.exists():
            raise FileNotFoundError(f"No existe {card_path}. Entrena el modelo primero.")
        self.card = json.loads(card_path.read_text())
        if self.card.get("features") != FEATURES:
            raise ValueError("El contrato de variables del modelo no coincide con app/features.py")
        sha = file_sha256(self.dir / "model.pt")
        if self.card.get("artifacts", {}).get("model_sha256") not in (None, sha):
            raise ValueError("model.pt no coincide con la huella registrada en model_card.json")
        self.tipo, self.h = self.card["architecture"]["type"], int(self.card["architecture"]["hidden"])
        self.net = build(self.tipo, self.h)
        self.net.load_state_dict(torch.load(self.dir / "model.pt", map_location="cpu", weights_only=True))
        self.net.eval()
        with warnings.catch_warnings(record=True) as avisos:
            warnings.simplefilter("always")
            sc = joblib.load(self.dir / "scaler.joblib")
        self.esc_x, self.y_mean, self.y_std = sc["esc_x"], float(sc["y_mean"]), float(sc["y_std"])
        mean, scale = np.asarray(self.esc_x.mean_, float), np.asarray(self.esc_x.scale_, float)
        if mean.shape != (N_LAGS,) or scale.shape != (N_LAGS,) or not (np.isfinite(mean).all() and (scale > 0).all()):
            raise ValueError("Escalador inválido: media/escala con forma o valores incorrectos")
        prueba = np.linspace(-0.05, 0.05, 2 * N_LAGS).reshape(2, N_LAGS)
        if not np.allclose(self.esc_x.transform(prueba), (prueba - mean) / scale):
            raise ValueError("El escalador no reproduce (x − media) / escala")
        self.scaler_version_warning = [str(a.message)[:160] for a in avisos]
        self.version = self.card.get("version", "desconocida")

    def predict(self, lags: np.ndarray) -> float:
        lags = np.asarray(lags, dtype=float).reshape(1, -1)
        if lags.shape != (1, N_LAGS) or not np.isfinite(lags).all():
            raise ValueError(f"Entrada inválida para el modelo: forma {lags.shape}")
        x = to_input(self.tipo, self.esc_x.transform(lags))
        with torch.no_grad():
            z = float(self.net(torch.from_numpy(x)).item())
        y = z * self.y_std + self.y_mean
        if not np.isfinite(y):
            raise ValueError("El modelo produjo un pronóstico no finito")
        return y