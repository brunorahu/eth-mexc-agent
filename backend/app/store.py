from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs(
  id TEXT PRIMARY KEY, mode TEXT NOT NULL, strategy TEXT NOT NULL, created_at TEXT NOT NULL,
  params TEXT, status TEXT);
CREATE TABLE IF NOT EXISTS portfolio(
  run_id TEXT PRIMARY KEY, cash REAL NOT NULL, qty REAL NOT NULL, avg_price REAL NOT NULL DEFAULT 0,
  fees REAL NOT NULL DEFAULT 0, slippage REAL NOT NULL DEFAULT 0, initial_cash REAL NOT NULL, updated_at TEXT);
CREATE TABLE IF NOT EXISTS processed_candles(
  run_id TEXT NOT NULL, candle_time TEXT NOT NULL, processed_at TEXT NOT NULL,
  PRIMARY KEY(run_id, candle_time));
CREATE TABLE IF NOT EXISTS decisions(
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, candle_time TEXT, close REAL, forecast REAL,
  indicators TEXT, proposed_action TEXT, justification TEXT, llm_raw TEXT, tool_calls TEXT,
  llm_latency_ms REAL, final_action TEXT, rejected INTEGER, reject_reason TEXT, cycle_ms REAL, created_at TEXT);
CREATE TABLE IF NOT EXISTS orders(
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, decision_id INTEGER, candle_time TEXT, side TEXT,
  qty REAL, ref_price REAL, price REAL, notional REAL, fee REAL, slippage_cost REAL, realized_pnl REAL, ts TEXT);
CREATE TABLE IF NOT EXISTS equity(
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, candle_time TEXT, ts TEXT,
  cash REAL, qty REAL, price REAL, equity REAL);
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, run_id TEXT, level TEXT, kind TEXT, message TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS kv(key TEXT PRIMARY KEY, value TEXT);
CREATE INDEX IF NOT EXISTS ix_dec_run ON decisions(run_id, id);
CREATE INDEX IF NOT EXISTS ix_eq_run ON equity(run_id, id);
CREATE INDEX IF NOT EXISTS ix_ev_run ON events(run_id, id);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DuplicateCandle(Exception):
    """La vela ya fue procesada para esta corrida (evento duplicado)."""


class Store:
    def __init__(self, path: Path | str):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.con = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.con.row_factory = sqlite3.Row
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("PRAGMA foreign_keys=ON")
        self.con.executescript(SCHEMA)

    # ------------------------------------------------------------------ utilidades
    def _all(self, sql: str, args=()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self.con.execute(sql, args).fetchall()]

    def _one(self, sql: str, args=()) -> dict | None:
        with self._lock:
            r = self.con.execute(sql, args).fetchone()
            return dict(r) if r else None

    # ------------------------------------------------------------------ estado clave-valor
    def kv_get(self, key: str, default=None):
        r = self._one("SELECT value FROM kv WHERE key=?", (key,))
        return json.loads(r["value"]) if r else default

    def kv_set(self, key: str, value) -> None:
        with self._lock:
            self.con.execute("INSERT INTO kv(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                             (key, json.dumps(value)))

    # ------------------------------------------------------------------ corridas y portafolio
    def ensure_run(self, run_id: str, mode: str, strategy: str, initial_cash: float, params: dict | None = None) -> None:
        with self._lock:
            self.con.execute("INSERT OR IGNORE INTO runs(id,mode,strategy,created_at,params,status) VALUES(?,?,?,?,?,?)",
                             (run_id, mode, strategy, now_iso(), json.dumps(params or {}), "created"))
            self.con.execute("INSERT OR IGNORE INTO portfolio(run_id,cash,qty,avg_price,initial_cash,updated_at) "
                             "VALUES(?,?,?,?,?,?)", (run_id, initial_cash, 0.0, 0.0, initial_cash, now_iso()))

    def set_run_status(self, run_id: str, status: str) -> None:
        with self._lock:
            self.con.execute("UPDATE runs SET status=? WHERE id=?", (status, run_id))

    def portfolio(self, run_id: str) -> dict:
        p = self._one("SELECT * FROM portfolio WHERE run_id=?", (run_id,))
        if p is None:
            raise KeyError(f"No existe la corrida {run_id}")
        return p

    def is_processed(self, run_id: str, candle_time: str) -> bool:
        return self._one("SELECT 1 AS x FROM processed_candles WHERE run_id=? AND candle_time=?",
                         (run_id, candle_time)) is not None

    # ------------------------------------------------------------------ transacción de un ciclo
    def commit_cycle(self, run_id: str, candle_time: str, decision: dict, order: dict | None,
                     new_portfolio: dict, equity_row: dict) -> int:
        with self._lock:
            cur = self.con.cursor()
            cur.execute("BEGIN IMMEDIATE")
            try:
                cur.execute("INSERT INTO processed_candles(run_id,candle_time,processed_at) VALUES(?,?,?)",
                            (run_id, candle_time, now_iso()))
            except sqlite3.IntegrityError:
                cur.execute("ROLLBACK")
                raise DuplicateCandle(f"La vela {candle_time} ya fue procesada en {run_id}")
            try:
                cur.execute(
                    "INSERT INTO decisions(run_id,candle_time,close,forecast,indicators,proposed_action,justification,"
                    "llm_raw,tool_calls,llm_latency_ms,final_action,rejected,reject_reason,cycle_ms,created_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (run_id, candle_time, decision["close"], decision["forecast"], json.dumps(decision["indicators"]),
                     decision["proposed_action"], decision.get("justification"), decision.get("llm_raw"),
                     json.dumps(decision.get("tool_calls") or []), decision.get("llm_latency_ms"),
                     decision["final_action"], int(decision["rejected"]), decision.get("reject_reason"),
                     decision.get("cycle_ms"), now_iso()))
                dec_id = cur.lastrowid
                if order:
                    cur.execute("INSERT INTO orders(run_id,decision_id,candle_time,side,qty,ref_price,price,notional,fee,"
                                "slippage_cost,realized_pnl,ts) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                                (run_id, dec_id, candle_time, order["side"], order["qty"], order["ref_price"],
                                 order["price"], order["notional"], order["fee"], order["slippage_cost"],
                                 order.get("realized_pnl"), now_iso()))
                p = new_portfolio
                cur.execute("UPDATE portfolio SET cash=?,qty=?,avg_price=?,fees=?,slippage=?,updated_at=? WHERE run_id=?",
                            (p["cash"], p["qty"], p["avg_price"], p["fees"], p["slippage"], now_iso(), run_id))
                e = equity_row
                cur.execute("INSERT INTO equity(run_id,candle_time,ts,cash,qty,price,equity) VALUES(?,?,?,?,?,?,?)",
                            (run_id, e["candle_time"], now_iso(), e["cash"], e["qty"], e["price"], e["equity"]))
                cur.execute("COMMIT")
                return dec_id
            except Exception:
                cur.execute("ROLLBACK")
                raise

    # ------------------------------------------------------------------ eventos y consultas
    def log_event(self, level: str, kind: str, message: str, run_id: str | None = None, detail=None) -> None:
        with self._lock:
            self.con.execute("INSERT INTO events(ts,run_id,level,kind,message,detail) VALUES(?,?,?,?,?,?)",
                             (now_iso(), run_id, level, kind, message,
                              json.dumps(detail, default=str) if detail is not None else None))

    def decisions(self, run_id: str, limit: int = 50) -> list[dict]:
        return self._all("SELECT * FROM decisions WHERE run_id=? ORDER BY id DESC LIMIT ?", (run_id, limit))

    def orders(self, run_id: str, limit: int = 100) -> list[dict]:
        return self._all("SELECT * FROM orders WHERE run_id=? ORDER BY id DESC LIMIT ?", (run_id, limit))

    def equity(self, run_id: str, limit: int = 2000) -> list[dict]:
        return self._all("SELECT * FROM (SELECT * FROM equity WHERE run_id=? ORDER BY id DESC LIMIT ?) ORDER BY id",
                         (run_id, limit))

    def last_equity(self, run_id: str) -> dict | None:
        return self._one("SELECT * FROM equity WHERE run_id=? ORDER BY id DESC LIMIT 1", (run_id,))

    def events(self, run_id: str | None = None, limit: int = 100, level: str | None = None) -> list[dict]:
        sql, args = "SELECT * FROM events WHERE 1=1", []
        if run_id:
            sql += " AND (run_id=? OR run_id IS NULL)"; args.append(run_id)
        if level:
            sql += " AND level=?"; args.append(level)
        return self._all(sql + " ORDER BY id DESC LIMIT ?", (*args, limit))

    def runs(self) -> list[dict]:
        return self._all("SELECT * FROM runs ORDER BY created_at DESC")

    def all_orders(self, run_id: str) -> list[dict]:
        return self._all("SELECT * FROM orders WHERE run_id=? ORDER BY id", (run_id,))

    def all_decisions(self, run_id: str) -> list[dict]:
        return self._all("SELECT * FROM decisions WHERE run_id=? ORDER BY id", (run_id,))
