from __future__ import annotations

import math
from dataclasses import dataclass, field

from .config import RiskConfig

ACTIONS = ("BUY", "HOLD", "SELL")


@dataclass
class RiskResult:
    approved: bool
    action: str            # acción final: BUY / SELL / HOLD
    qty: float = 0.0
    reasons: list[str] = field(default_factory=list)
    halt: bool = False     # se alcanzó un límite que bloquea nuevas órdenes


def floor_step(x: float, step: float) -> float:
    return math.floor(x / step + 1e-9) * step


def evaluate(action: str, *, price: float, cash: float, qty: float, prev_equity: float | None,
             data_age_hours: float, blocked: bool, cfg: RiskConfig) -> RiskResult:
    """Valida la acción propuesta. Si se rechaza, la acción final es HOLD (no se opera)."""
    if action not in ACTIONS:
        return RiskResult(False, "HOLD", reasons=[f"Acción no permitida: {action!r}"])
    if not (price > 0 and math.isfinite(price)):
        return RiskResult(False, "HOLD", reasons=[f"Precio de referencia inválido: {price}"])
    if action == "HOLD":
        return RiskResult(True, "HOLD")

    motivos: list[str] = []
    if blocked:
        motivos.append("Órdenes bloqueadas: el agente está detenido o en pausa")
    if data_age_hours > cfg.max_data_age_hours:
        motivos.append(f"Dato atrasado: {data_age_hours:.1f} h > {cfg.max_data_age_hours} h")
    equity = cash + qty * price
    halt = False
    if prev_equity and prev_equity > 0 and equity / prev_equity - 1 <= -cfg.max_daily_loss_pct:
        halt = True
        motivos.append(f"Límite de pérdida diaria alcanzado: {100 * (equity / prev_equity - 1):.2f}% "
                       f"≤ −{100 * cfg.max_daily_loss_pct:.2f}%")
    if motivos:
        return RiskResult(False, "HOLD", reasons=motivos, halt=halt)

    slip = cfg.slippage_bps / 1e4
    if action == "BUY":
        espacio = equity * cfg.max_exposure_pct - qty * price          # límite de exposición
        presupuesto = min(cash, espacio)                               # nunca más que el efectivo: sin apalancamiento
        costo_unit = price * (1 + slip) * (1 + cfg.fee_rate)
        q = floor_step(max(presupuesto, 0.0) / costo_unit, cfg.qty_step)
        if q * price < cfg.min_notional:
            razon = ("Exposición máxima alcanzada" if espacio < cfg.min_notional and cash >= cfg.min_notional
                     else "Saldo insuficiente")
            return RiskResult(False, "HOLD", reasons=[f"{razon}: monto disponible {presupuesto:.2f} USDT "
                                                       f"< mínimo {cfg.min_notional} USDT"])
        return RiskResult(True, "BUY", qty=q)

    # SELL: solo se vende una posición existente (sin ventas en corto)
    if qty <= 0:
        return RiskResult(False, "HOLD", reasons=["Sin posición abierta: las ventas en corto no están permitidas"])
    q = floor_step(qty, cfg.qty_step)
    if q * price < cfg.min_notional:
        return RiskResult(False, "HOLD", reasons=[f"Posición menor al mínimo negociable ({q} unidades)"])
    return RiskResult(True, "SELL", qty=q)
