"""PnL 实时计算 + 策略聚合统计 + 基准对照。

公式（常规 A）：unrealized_pnl_pct = (current_price - entry_price) / entry_price * 100
跟踪期 5 天。平仓策略默认 Buy & Hold（exit=hold）。
下架处理：最后快照距今 >24h 标 delisted，PnL 用最后有效价。
"""
from __future__ import annotations
import os
import time
import requests
from datetime import datetime, timezone, timedelta
from typing import Optional
import db

CST = timezone(timedelta(hours=8))
FAPI = "https://fapi.binance.com"
TRACK_DAYS = 5
DELISTED_HOURS = 24


def _parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def fetch_live_price(symbol: str, timeout: float = 5) -> Optional[float]:
    """实时拉币安最新价，失败返回 None。"""
    try:
        r = requests.get(f"{FAPI}/fapi/v1/ticker/price",
                         params={"symbol": symbol}, timeout=timeout)
        if r.status_code == 200:
            return float(r.json()["price"])
    except Exception:
        pass
    return None


def compute_pnl(conn, signal, live: bool = True, exit: str = "hold") -> dict:
    """计算单条信号的 PnL。

    signal: db.get_signal 返回的 Row（dict-like）
    live: True 则实时拉币安最新价；False 用最新快照
    exit: 'hold'（默认，唯一实现），其他值预留接口
    """
    entry = float(signal["entry_price"])
    sym = signal["symbol"]
    st = signal["signal_time"]
    st_dt = _parse_iso(st)
    now = datetime.now(CST)

    # ── 当前价 ──
    current_price = None
    current_ts = None
    if live:
        p = fetch_live_price(sym)
        if p is not None:
            current_price, current_ts = p, now.isoformat(timespec="seconds")
    if current_price is None:
        snap = db.get_latest_snapshot(conn, sym)
        if snap:
            current_price, current_ts = snap["price"], snap["ts"]

    # ── 历史快照 ──
    snaps = db.get_snapshots_since(conn, sym, st)
    prices = [float(r["price"]) for r in snaps]
    max_price = max(prices) if prices else None
    min_price = min(prices) if prices else None

    # ── delisted 检测 ──
    delisted = False
    last_snap_ts = snaps[-1]["ts"] if snaps else None
    if last_snap_ts:
        gap = now - _parse_iso(last_snap_ts)
        if gap > timedelta(hours=DELISTED_HOURS):
            delisted = True

    # ── 未实现盈亏（公式 A）──
    unrealized = None
    if current_price is not None and entry > 0:
        unrealized = (current_price - entry) / entry * 100

    # ── pnl_Nd_pct：信号+N 天那一刻的快照价算 ──
    def pnl_at_offset(days):
        target = (st_dt + timedelta(days=days)).isoformat(timespec="seconds")
        s = db.get_snapshot_at(conn, sym, target)
        if not s:
            return None
        return (float(s["price"]) - entry) / entry * 100

    pnl_1d = pnl_at_offset(1)
    pnl_3d = pnl_at_offset(3)
    pnl_5d = pnl_at_offset(5)

    # ── 最大涨/跌（基于快照）──
    max_high_pct = (max_price - entry) / entry * 100 if max_price and entry > 0 else None
    max_low_pct = (min_price - entry) / entry * 100 if min_price and entry > 0 else None

    hold_hours = round((now - st_dt).total_seconds() / 3600, 1)

    # ── 曲线 ──
    curve = [
        {"ts": r["ts"], "price": float(r["price"]),
         "pnl_pct": (float(r["price"]) - entry) / entry * 100 if entry > 0 else None}
        for r in snaps
    ]

    return {
        "signal_id": signal["id"],
        "symbol": sym, "coin": signal["coin"], "strategy": signal["strategy"],
        "signal_time": st,
        "entry_price": entry,
        "current_price": current_price,
        "current_price_ts": current_ts,
        "current_source": "live" if (live and current_ts and current_ts.startswith(now.strftime("%Y-%m-%d"))) else "snapshot",
        "hold_hours": hold_hours,
        "unrealized_pnl_pct": unrealized,
        "pnl_1d_pct": pnl_1d,
        "pnl_3d_pct": pnl_3d,
        "pnl_5d_pct": pnl_5d,
        "max_high_pct": max_high_pct,
        "max_low_pct": max_low_pct,
        "delisted": delisted,
        "exit": exit,
        "curve": curve,
    }


def _pnl_for_stats(conn, signal, live: bool = False) -> Optional[dict]:
    """统计用：用快照算 PnL（不实时拉，避免限速）。"""
    return compute_pnl(conn, signal, live=live, exit="hold")


def compute_strategy_stats(conn, strategy: str, frm=None, to=None, live=False) -> dict:
    """某策略信号聚合统计。遍历所有信号算 PnL 再聚合。"""
    rows = db.get_signals(conn, strategy=strategy, frm=frm, to=to, limit=100000, offset=0)
    pnls = []
    delisted_n = 0
    for r in rows:
        p = _pnl_for_stats(conn, r, live=live)
        if p is None:
            continue
        if p["delisted"]:
            delisted_n += 1
        pnls.append(p)

    n = len(pnls)
    if n == 0:
        return {"strategy": strategy, "from": frm, "to": to,
                "total_signals": 0, "note": "no signals in range"}

    unrealized = [p["unrealized_pnl_pct"] for p in pnls if p["unrealized_pnl_pct"] is not None]
    max_highs = [p["max_high_pct"] for p in pnls if p["max_high_pct"] is not None]

    def median(xs):
        xs = sorted(xs)
        m = len(xs)
        return xs[m // 2] if m % 2 else (xs[m // 2 - 1] + xs[m // 2]) / 2

    def pct(xs, thr):
        if not xs:
            return None
        return sum(1 for x in xs if x >= thr) / len(xs)

    # PnL 分布分桶
    buckets = {"<-50%": 0, "-50~0%": 0, "0~50%": 0, "50~100%": 0, ">100%": 0}
    for x in unrealized:
        if x < -50: buckets["<-50%"] += 1
        elif x < 0: buckets["-50~0%"] += 1
        elif x < 50: buckets["0~50%"] += 1
        elif x < 100: buckets["50~100%"] += 1
        else: buckets[">100%"] += 1

    win = sum(1 for x in unrealized if x > 0)

    return {
        "strategy": strategy,
        "from": frm, "to": to,
        "total_signals": n,
        "with_unrealized": len(unrealized),
        "delisted_signals": delisted_n,
        "avg_unrealized_pnl_pct": round(sum(unrealized) / len(unrealized), 2) if unrealized else None,
        "median_unrealized_pnl_pct": round(median(unrealized), 2) if unrealized else None,
        "win_rate_pct": round(win / len(unrealized) * 100, 1) if unrealized else None,
        "hit_rate_50pct": round(pct(max_highs, 50) * 100, 1) if max_highs else None,
        "hit_rate_100pct": round(pct(max_highs, 100) * 100, 1) if max_highs else None,
        "hit_rate_300pct": round(pct(max_highs, 300) * 100, 1) if max_highs else None,
        "avg_max_high_pct": round(sum(max_highs) / len(max_highs), 2) if max_highs else None,
        "avg_max_low_pct": round(sum(p["max_low_pct"] for p in pnls if p["max_low_pct"] is not None) / max(1, sum(1 for p in pnls if p["max_low_pct"] is not None)), 2) if any(p["max_low_pct"] is not None for p in pnls) else None,
        "pnl_distribution": buckets,
    }


def compute_benchmark_stats(conn, live=False) -> dict:
    """基准组 PnL 统计：每个基准组内每个 symbol 用首次价为 entry、最新价为 current。"""
    groups = conn.execute(
        "SELECT DISTINCT cohort FROM benchmark_snapshots ORDER BY cohort"
    ).fetchall()
    pnls = []
    for c in groups:
        rows = db.benchmark_group_rows(conn, c["cohort"])
        by_sym = {}
        for r in rows:
            by_sym.setdefault(r["symbol"], []).append((r["ts"], float(r["price"])))
        for sym, snaps in by_sym.items():
            if len(snaps) < 2:
                continue  # 只有首次价，无后续 → PnL 未定义
            entry = snaps[0][1]
            current = snaps[-1][1]
            if entry > 0:
                pnls.append((current - entry) / entry * 100)
    n = len(pnls)
    if n == 0:
        return {"benchmark": True, "total": 0,
                "note": "no benchmark pairs yet (need 2+ samples per symbol)"}
    win = sum(1 for x in pnls if x > 0)
    buckets = {"<-50%": 0, "-50~0%": 0, "0~50%": 0, "50~100%": 0, ">100%": 0}
    for x in pnls:
        if x < -50: buckets["<-50%"] += 1
        elif x < 0: buckets["-50~0%"] += 1
        elif x < 50: buckets["0~50%"] += 1
        elif x < 100: buckets["50~100%"] += 1
        else: buckets[">100%"] += 1
    return {
        "benchmark": True,
        "total": n,
        "avg_unrealized_pnl_pct": round(sum(pnls) / n, 2),
        "win_rate_pct": round(win / n * 100, 1),
        "pnl_distribution": buckets,
    }