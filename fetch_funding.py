#!/usr/bin/env python3
"""funding 数据拉取脚本 —— 严格权重管理，写入 DuckDB。

用法:
  python3 fetch_funding.py --days 400 --db data/market.duckdb
  python3 fetch_funding.py --symbols BTCUSDT,ETHUSDT --days 400

权重管理:
  - fundingRate 走 WAF 敏感桶，串行 + 0.45s 节流，403 指数冷却(60→600s)。
  - 标准桶权重跟踪 x-mbx-used-weight-1m，接近预算冷却到下一分钟窗口。
  - 绝不并发(并发是上次403事故根因)。
"""
import argparse
import logging
import time
import sys
from binance_client import BinanceFuturesClient, get_perpetual_symbols
from duckdb_store import DuckDBStore

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fetch_funding")


def fetch_funding_for(client, store, symbol, start_ms, end_ms):
    """拉单币 funding 全量(分页), 写入DuckDB。返回新增条数。"""
    batch = client.funding_rate_full(symbol, start_ms, end_ms)
    if not batch:
        return 0
    rows = []
    last_ms = 0
    for rec in batch:
        ft = int(rec["fundingTime"])
        if ft > last_ms:
            last_ms = ft
        try:
            fr = float(rec["fundingRate"])
        except (ValueError, TypeError):
            fr = None
        mp = None
        if rec.get("markPrice"):
            try:
                mp = float(rec["markPrice"])
            except (ValueError, TypeError):
                mp = None
        # 转DuckDB可接受的timestamp: 用datetime
        rows.append((symbol, _ms_to_dt(ft), fr, mp))
    if rows:
        store.upsert_funding(rows)
        store.mark_fetch(symbol, "funding", last_ms, len(rows))
    return len(rows)


def _ms_to_dt(ms):
    # DuckDB executemany 接受 python datetime
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).replace(tzinfo=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=400, help="回看天数(默认400)")
    ap.add_argument("--db", default="data/market.duckdb")
    ap.add_argument("--symbols", default=None, help="逗号分隔的symbol列表, 不传则全市场永续")
    ap.add_argument("--limit", type=int, default=None, help="最多拉多少币(调试用)")
    args = ap.parse_args()

    client = BinanceFuturesClient()
    store = DuckDBStore(args.db)

    # symbol universe
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        log.info("拉取 exchangeInfo 获取永续合约标的列表...")
        symbols = get_perpetual_symbols(client)
        log.info("永续合约标的: %d 个", len(symbols))
    if args.limit:
        symbols = symbols[:args.limit]

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - args.days * 86400 * 1000
    log.info("拉取 funding: %d 币, 回看 %d 天 (start=%s)", len(symbols), args.days,
             time.strftime("%Y-%m-%d", time.gmtime(start_ms / 1000)))

    total = 0
    ok = 0
    fail = []
    t0 = time.time()
    for i, s in enumerate(symbols):
        n = fetch_funding_for(client, store, s, start_ms, end_ms)
        if n > 0:
            ok += 1
            total += n
        else:
            fail.append(s)
        if (i + 1) % 25 == 0 or (i + 1) == len(symbols):
            rate = (i + 1) / (time.time() - t0)
            eta = (len(symbols) - i - 1) / max(rate, 0.01)
            log.info("进度 %d/%d | 成功%d 累积%d条 | 失败%d | 权重1m=%d | %.1f币/s ETA%.0fs",
                     i + 1, len(symbols), ok, total, len(fail),
                     client.weight_used_1m, rate, eta)

    log.info("=== funding 拉取完成 ===")
    log.info("成功 %d/%d 币, 总 %d 条, 失败 %d: %s", ok, len(symbols), total, len(fail), fail[:20])
    log.info("客户端统计: %s", client.stats)
    # 验证
    n_syms = len(store.funding_symbols())
    rng = store.funding_date_range()
    log.info("DuckDB funding_rate: 覆盖 %d 币, 时间范围 %s ~ %s, 总 %d 行",
             n_syms, rng[0], rng[1], rng[2])


if __name__ == "__main__":
    main()