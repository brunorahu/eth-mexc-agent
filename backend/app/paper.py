from __future__ import annotations

from .config import RiskConfig
from .risk import floor_step


class InsufficientBalance(RuntimeError):
    pass


def fill(side: str, qty: float, ref_price: float, portfolio: dict, cfg: RiskConfig) -> tuple[dict, dict]:
    cash, pos, avg = portfolio["cash"], portfolio["qty"], portfolio["avg_price"]
    fees, slip_total = portfolio.get("fees", 0.0), portfolio.get("slippage", 0.0)
    slip = cfg.slippage_bps / 1e4
    qty = floor_step(qty, cfg.qty_step)
    if qty <= 0:
        raise InsufficientBalance("Cantidad nula después de redondear al paso del par")
    if side == "BUY":
        price = ref_price * (1 + slip)
        notional = qty * price
        fee = notional * cfg.fee_rate
        if notional + fee > cash + 1e-9:
            raise InsufficientBalance(f"Saldo insuficiente: se requieren {notional + fee:.2f} USDT y hay {cash:.2f}")
        new_qty = pos + qty
        new_avg = (avg * pos + notional + fee) / new_qty
        cash, pnl = cash - notional - fee, None
    elif side == "SELL":
        if qty > pos + 1e-12:
            raise InsufficientBalance(f"No hay posición suficiente: se intenta vender {qty} y hay {pos}")
        price = ref_price * (1 - slip)
        notional = qty * price
        fee = notional * cfg.fee_rate
        pnl = notional - fee - avg * qty
        new_qty = pos - qty
        new_avg = avg if new_qty > 1e-12 else 0.0
        new_qty = new_qty if new_qty > 1e-12 else 0.0
        cash = cash + notional - fee
    else:
        raise ValueError(f"Lado inválido: {side}")
    slippage_cost = abs(price - ref_price) * qty
    order = {"side": side, "qty": qty, "ref_price": ref_price, "price": price, "notional": notional,
             "fee": fee, "slippage_cost": slippage_cost, "realized_pnl": pnl}
    new_p = {"cash": cash, "qty": new_qty, "avg_price": new_avg, "fees": fees + fee, "slippage": slip_total + slippage_cost}
    return order, new_p
