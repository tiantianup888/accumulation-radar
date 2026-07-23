#!/usr/bin/env python3
"""构建 symbol_meta 表（universe-as-of 存活偏差修正核心）。

从当前 exchangeInfo 取所有 USDT PERPETUAL（TRADING + SETTLING），记录：
  symbol, status, contract_type, onboard_date, last_kline_date(从klines_daily回填), source

存活偏差修正：SETTLING 币 = 窗口期在交易、现在正在退市 → 必须纳入研究 universe。
  - status=TRADING: 530 个（当前可交易）
  - status=SETTLING: ~122 个（退市结算中，窗口期全程在交易）
退市币输出到 /tmp/settling_syms.txt 供 fetch 脚本拉取。

用法:
  python3 build_symbol_meta.py                 # 仅建meta
  python3 build_symbol_meta.py --backfill-last  # 拉取后回填 last_kline_date
"""
import argparse
from datetime import datetime, timezone
import requests
from duckdb_store import DuckDBStore


def get_usdt_perp_all():
    """取所有 USDT PERPETUAL（任意 status），区分 TRADING/SETTLING。"""
    r = requests.get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=60)
    r.raise_for_status()
    j = r.json()
    out = []
    for x in j["symbols"]:
        if x.get("quoteAsset") == "USDT" and x.get("contractType") == "PERPETUAL":
            out.append({
                "symbol": x["symbol"],
                "status": x.get("status"),  # TRADING / SETTLING / PENDING_TRADING
                "contract_type": x.get("contractType"),
                "onboard_date": datetime.fromtimestamp(
                    x["onboardDate"] / 1000, tz=timezone.utc).replace(tzinfo=None),
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/market.duckdb")
    ap.add_argument("--backfill-last", action="store_true",
                    help="从 klines_daily 回填 last_kline_date（拉取后执行）")
    args = ap.parse_args()

    store = DuckDBStore(args.db)
    syms = get_usdt_perp_all()
    trading = [s for s in syms if s["status"] == "TRADING"]
    settling = [s for s in syms if s["status"] == "SETTLING"]
    other = [s for s in syms if s["status"] not in ("TRADING", "SETTLING")]
    print(f"USDT PERPETUAL: TRADING={len(trading)} SETTLING={len(settling)} 其他={len(other)}")

    # 写 symbol_meta（last_kline_date 暂置 None，--backfill-last 后回填）
    rows = [(s["symbol"], s["status"], s["contract_type"],
             s["onboard_date"], None, "current_exchangeInfo") for s in syms]
    n = store.upsert_symbol_meta(rows)
    print(f"symbol_meta 写入 {n} 行")

    # 输出 SETTLING 列表供 fetch 脚本
    settling_list = sorted(s["symbol"] for s in settling)
    with open("/tmp/settling_syms.txt", "w") as f:
        f.write(",".join(settling_list))
    print(f"SETTLING 退市币 {len(settling_list)} 个 → /tmp/settling_syms.txt")
    print("  " + ", ".join(settling_list[:20]) + (" ..." if len(settling_list) > 20 else ""))

    if args.backfill_last:
        # 从 klines_daily 回填 last_kline_date（退市币=退市日近似）
        updated = store.con.execute("""
            UPDATE symbol_meta AS m
            SET last_kline_date = (SELECT MAX(open_time) FROM klines_daily k WHERE k.symbol = m.symbol)
            WHERE EXISTS (SELECT 1 FROM klines_daily k WHERE k.symbol = m.symbol)
        """).fetchall()
        # 报告: 无kline的meta(窗口前已退市/无数据)
        no_kline = store.con.execute("""
            SELECT symbol, status FROM symbol_meta
            WHERE last_kline_date IS NULL ORDER BY symbol""").fetchall()
        print(f"\n回填 last_kline_date 完成; 无日K的meta: {len(no_kline)}")
        for s, st in no_kline[:30]:
            print(f"  {s:16} {st}")
        # 全universe汇总
        tot = store.con.execute("SELECT COUNT(*) FROM symbol_meta").fetchone()[0]
        with_k = store.con.execute(
            "SELECT COUNT(*) FROM symbol_meta WHERE last_kline_date IS NOT NULL").fetchone()[0]
        print(f"\nsymbol_meta: {tot} 币, 有日K数据 {with_k} 币")
        # 窗口覆盖(03-15~07-22)
        inwin = store.con.execute("""
            SELECT status, COUNT(*) FROM symbol_meta
            WHERE last_kline_date >= '2026-07-22'
            GROUP BY status ORDER BY status""").fetchall()
        print(f"窗口末(07-22)仍有日K的: {inwin}")


if __name__ == "__main__":
    main()