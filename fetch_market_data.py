#!/usr/bin/env python3
"""量价(日K) + OI 拉取脚本, 写入 DuckDB。

权重管理:
  - 日K(klines): 标准桶, weight=10(limit1500), 跟踪 x-mbx-used-weight-1m, 接近1800冷却到下一分钟。
    530币×10权重=5300, 需~3个分钟窗口。无WAF 403风险(标准桶)。
  - OI(openInterestHist): WAF敏感桶(同funding), 主动节流(每80次休30s)+1s间隔, 仅30天可得。
用法:
  python3 fetch_market_data.py --kind klines --days 130
  python3 fetch_market_data.py --kind oi
  python3 fetch_market_data.py --kind all --days 130
"""
import argparse
import logging
import time
from datetime import datetime, timezone
from binance_client import BinanceFuturesClient, get_perpetual_symbols
from duckdb_store import DuckDBStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fetch_market")


def _ms_to_dt(ms):
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).replace(tzinfo=None)


def fetch_klines_for(client, store, symbol, days):
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86400 * 1000
    # limit按需取(<=1500, weight按档: 100→1,500→2,1000→5,1500→10)
    klines = client.klines(symbol, interval="1d", limit=min(days + 5, 1500), start_ms=start_ms, end_ms=end_ms)
    if not klines:
        return 0
    rows = []
    last_ms = 0
    for k in klines:
        ot = int(k[0])
        if ot > last_ms:
            last_ms = ot
        rows.append((symbol, _ms_to_dt(ot),
                    float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]),
                    _ms_to_dt(int(k[6])), float(k[7]), int(k[8]), float(k[9]), float(k[10])))
    if rows:
        store.upsert_klines(rows)
        store.mark_fetch(symbol, "klines", last_ms, len(rows))
    return len(rows)


def fetch_oi_for(client, store, symbol):
    # openInterestHist 仅最近30天可得, 日周期
    data = client.open_interest_hist(symbol, period="1d", limit=30)
    if not data:
        return 0
    rows = []
    last_ms = 0
    for d in data:
        ts = int(d["timestamp"])
        if ts > last_ms:
            last_ms = ts
        rows.append((symbol, _ms_to_dt(ts),
                     float(d.get("sumOpenInterest", 0) or 0),
                     float(d.get("sumOpenInterestValue", 0) or 0)))
    if rows:
        store.upsert_oi(rows)
        store.mark_fetch(symbol, "oi", last_ms, len(rows))
    return len(rows)


def run(client, store, symbols, kind, days=None):
    log.info("=== %s: %d 币 ===", kind, len(symbols))
    total = 0; ok = 0; fail = []
    t0 = time.time()
    fetch_fn = fetch_klines_for if kind == "klines" else fetch_oi_for
    for i, s in enumerate(symbols):
        if kind == "klines":
            n = fetch_fn(client, store, s, days)
        else:
            n = fetch_fn(client, store, s)
        if n > 0:
            ok += 1; total += n
        else:
            fail.append(s)
        if (i + 1) % 25 == 0 or (i + 1) == len(symbols):
            rate = (i + 1) / (time.time() - t0)
            eta = (len(symbols) - i - 1) / max(rate, 0.01)
            log.info("进度 %d/%d | 成功%d 累积%d | 失败%d | 权重1m=%d | %.2f/s ETA%.0fs",
                     i + 1, len(symbols), ok, total, len(fail),
                     client.weight_used_1m, rate, eta)
    log.info("=== %s完成: 成功 %d/%d, 总 %d 条, 失败 %d: %s ===",
             kind, ok, len(symbols), total, len(fail), fail[:20])
    log.info("客户端统计: %s", client.stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["klines", "oi", "all"], default="all")
    ap.add_argument("--days", type=int, default=130, help="日K回看天数(默认130)")
    ap.add_argument("--db", default="data/market.duckdb")
    ap.add_argument("--symbols", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    client = BinanceFuturesClient()
    store = DuckDBStore(args.db)
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = get_perpetual_symbols(client)
        log.info("永续合约标的: %d", len(symbols))
    if args.limit:
        symbols = symbols[:args.limit]

    if args.kind in ("klines", "all"):
        run(client, store, symbols, "klines", args.days)
    if args.kind in ("oi", "all"):
        run(client, store, symbols, "oi")

    # 汇总验证
    log.info("=== DuckDB 汇总 ===")
    log.info("klines_daily: %d 行, %d 币", store.count("klines_daily"),
             con_rowcount(store, "SELECT COUNT(DISTINCT symbol) FROM klines_daily"))
    log.info("funding_rate: %d 行, %d 币", store.count("funding_rate"),
             con_rowcount(store, "SELECT COUNT(DISTINCT symbol) FROM funding_rate"))
    log.info("open_interest: %d 行, %d 币", store.count("open_interest"),
             con_rowcount(store, "SELECT COUNT(DISTINCT symbol) FROM open_interest"))


def con_rowcount(store, q):
    return store.con.execute(q).fetchone()[0]


if __name__ == "__main__":
    main()