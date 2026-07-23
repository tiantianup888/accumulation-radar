#!/usr/bin/env python3
"""建立庄家拉盘指纹库：扫描最近90天全市场，找出真实拉盘事件，提取起飞前14天指纹入库。

拉盘事件定义：某日涨幅 > +50%（日K收盘/开盘-1）。排除C超跌反弹假模式。
去重：同一币90天内只取涨幅最大的那次拉盘（避免一次行情多次入库）。
"""
import requests, time, statistics, os, sys
from datetime import datetime, timezone, timedelta
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fingerprint import extract_fingerprint, save_library, load_library, library_summary, Fingerprint, LIB_PATH

SPOT = "https://api.binance.com"
FUT = "https://fapi.binance.com"
CST = timezone(timedelta(hours=8))
sess = requests.Session()
DAYS = 104  # 90天+14天窗口余量


def get(url, params=None):
    for _ in range(3):
        try:
            r = sess.get(url, params=params, timeout=20)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(10)
        except Exception:
            time.sleep(2)
    return None


def build_map():
    si = get(f"{SPOT}/api/v3/exchangeInfo")
    fi = get(f"{FUT}/fapi/v1/exchangeInfo")
    spot = {s["symbol"] for s in si["symbols"] if s["quoteAsset"] == "USDT" and s["status"] == "TRADING"}
    fut = [s["symbol"] for s in fi["symbols"] if s["contractType"] == "PERPETUAL" and s["quoteAsset"] == "USDT" and s["status"] == "TRADING"]
    def m(fs):
        if fs.startswith("1000"):
            c = fs[3:]; return c if c in spot else None
        return fs if fs in spot else None
    return {fs: m(fs) for fs in fut}


def daily_klines(symbol, base, path, days):
    """拉日K, 返回升序 [{ts,o,h,l,c,v}]。"""
    out = []
    end = None
    need = days
    while need > 0:
        lim = min(1000, need)
        p = {"symbol": symbol, "interval": "1d", "limit": lim}
        if end:
            p["endTime"] = end - 1
        k = get(base + path, p)
        if not k:
            break
        bars = [{"ts": int(x[0]), "o": float(x[1]), "h": float(x[2]),
                 "l": float(x[3]), "c": float(x[4]), "v": float(x[5])} for x in k]
        out = bars + out
        end = k[0][0]
        need -= len(k)
        if len(k) < lim:
            break
        time.sleep(0.12)
    return out


def find_pump_events(daily, min_chg=30.0):
    """找日涨幅>min_chg的拉盘日索引(升序)。"""
    events = []
    for i in range(1, len(daily)):
        if daily[i]["o"] > 0:
            chg = (daily[i]["c"] / daily[i]["o"] - 1) * 100
            if chg >= min_chg:
                events.append((i, chg))
    return events


def main():
    m = build_map()
    cov = sorted([fs for fs in m if m[fs]])
    print(f"可映射 {len(cov)} 现货, 扫最近{DAYS}天日K找拉盘事件...")
    fps = []
    stats = defaultdict(int)
    per_coin_best = {}  # 币 -> (chg, idx) 90天内取最大
    for idx, fs in enumerate(cov):
        ss = m[fs]
        # 用现货日K(现货更反映真实吸筹；ESPORTS类无现货则用合约)
        dk = daily_klines(ss, SPOT, "/api/v3/klines", DAYS)
        if not dk or len(dk) < 20:
            # 回退合约
            dk = daily_klines(fs, FUT, "/fapi/v1/klines", DAYS)
        if not dk or len(dk) < 20:
            stats["no_data"] += 1
            continue
        events = find_pump_events(dk, 50.0)
        if not events:
            stats["no_pump"] += 1
            continue
        # 同币90天取最大涨幅那次
        best = max(events, key=lambda x: x[1])
        pi, chg = best
        stats["has_pump"] += 1
        fp = extract_fingerprint(ss, fs, dk, pi)
        if fp is None:
            stats["extract_fail"] += 1
            continue
        # 只入库真实庄家模式 A/B/D，排除C超跌反弹
        if fp.mode == "C":
            stats["mode_C_excluded"] += 1
            continue
        fps.append(fp)
        stats[f"mode_{fp.mode}"] += 1
        if (idx + 1) % 50 == 0:
            print(f"  {idx+1}/{len(cov)} 有拉盘{stats['has_pump']} 入库{len(fps)} " +
                  " ".join(f"{k}={v}" for k, v in stats.items() if k.startswith("mode_")), flush=True)
    print(f"\n扫描完成: {dict(stats)}")
    # 去重: 同币只保留涨幅最大的指纹(避免重复行情)
    by_coin = defaultdict(list)
    for fp in fps:
        by_coin[fp.source_coin].append(fp)
    dedup = [max(v, key=lambda x: x.pump_chg_24h) for v in by_coin.values()]
    print(f"去重后: {len(dedup)} 条指纹 (来自 {len(by_coin)} 币)")
    save_library(dedup)
    print(f"\n已保存到 {LIB_PATH}")
    print(library_summary(load_library()))


if __name__ == "__main__":
    main()