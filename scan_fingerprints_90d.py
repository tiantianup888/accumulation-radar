#!/usr/bin/env python3
"""庄家拉盘指纹库 — 最近90天全市场匹配扫描 + 详细报表数据生成。

对每个币拉104+天日K，滑动14日窗口(每3日一个快照)匹配指纹库，记录命中及5日前瞻PnL。
产出: /tmp/fp_report.json 供报表生成。
"""
import requests, time, statistics, os, sys, json
from datetime import datetime, timezone, timedelta
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fingerprint import load_library, match_window, extract_signatures, LIB_PATH

SPOT = "https://api.binance.com"; FUT = "https://fapi.binance.com"
CST = timezone(timedelta(hours=8))
sess = requests.Session()
DAYS = 120  # 90天回溯 + 14窗口 + 5前瞻 + 余量
THRESHOLD = 0.55
SNAPSHOT_STEP = 3  # 每3天一个快照
FORWARD = 5        # 5日前瞻PnL


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
    out = []; end = None; need = days
    while need > 0:
        lim = min(1000, need)
        p = {"symbol": symbol, "interval": "1d", "limit": lim}
        if end: p["endTime"] = end - 1
        k = get(base + path, p)
        if not k: break
        bars = [{"ts": int(x[0]), "o": float(x[1]), "h": float(x[2]),
                 "l": float(x[3]), "c": float(x[4]), "v": float(x[5])} for x in k]
        out = bars + out; end = k[0][0]; need -= len(k)
        if len(k) < lim: break
        time.sleep(0.12)
    return out


def main():
    lib = load_library()
    print(f"指纹库: {len(lib)} 条, 阈值{THRESHOLD}, 扫最近90天(每{SNAPSHOT_STEP}日快照, 5日前瞻)")
    lib_ids = {f.fp_id for f in lib}
    lib_coins = {f.source_coin for f in lib}
    m = build_map()
    cov = sorted([fs for fs in m if m[fs]])
    print(f"可映射 {len(cov)} 现货")
    all_matches = []
    in_sample = 0
    out_sample = 0
    scanned = 0
    for idx, fs in enumerate(cov):
        ss = m[fs]
        dk = daily_klines(ss, SPOT, "/api/v3/klines", DAYS)
        if not dk or len(dk) < 20:
            dk = daily_klines(fs, FUT, "/fapi/v1/klines", DAYS)
        if not dk or len(dk) < 20:
            continue
        scanned += 1
        n = len(dk)
        last_hit = -999
        # 滑动窗口: end 从 14 到 n-1-FORWARD (要5日前瞻)
        for end in range(14, n - FORWARD + 1, SNAPSHOT_STEP):
            if end - last_hit < FORWARD:  # 同币冷却5日
                continue
            window_bars = dk[end - 14:end + 1]  # 末日end作为窗口末
            # match_window 期望 daily_bars 末14日
            hits = match_window(dk[:end + 1], lib, 14, THRESHOLD)
            if not hits:
                continue
            best = hits[0]
            entry = dk[end]["c"]
            # 5日前瞻PnL
            f_end = min(end + FORWARD, n - 1)
            pnl = (dk[f_end]["c"] / entry - 1) * 100
            mh = (max(b["h"] for b in dk[end + 1:f_end + 1]) / entry - 1) * 100 if end + 1 <= f_end else 0
            ml = (min(b["l"] for b in dk[end + 1:f_end + 1]) / entry - 1) * 100 if end + 1 <= f_end else 0
            is_in = best["source"] in lib_coins and best["source"] == fs  # 命中自己来源
            rec = {
                "coin": fs, "spot": ss, "date": best["end_date"],
                "fp_id": best["fp_id"], "mode": best["mode"], "similarity": best["similarity"],
                "source": best["source"], "source_pump_chg": best["source_pump_chg"],
                "entry": entry, "pnl5d": round(pnl, 2), "max_high": round(mh, 2),
                "max_low": round(ml, 2), "in_sample": is_in,
                "signatures": best["signatures"],
            }
            all_matches.append(rec)
            if is_in: in_sample += 1
            else: out_sample += 1
            last_hit = end
        if (idx + 1) % 50 == 0:
            print(f"  {idx+1}/{len(cov)} 扫描{scanned} 命中{len(all_matches)} (样本内{in_sample}/外{out_sample})", flush=True)
    print(f"\n扫描完成: {scanned}币, 总命中{len(all_matches)} (样本内{in_sample}/外{out_sample})")
    # 统计
    def stats(matches, label):
        if not matches:
            print(f"{label}: 无命中"); return
        pnls = [x["pnl5d"] for x in matches]
        win = sum(1 for x in pnls if x > 0)
        big = sum(1 for x in pnls if x > 20)
        big_loss = sum(1 for x in pnls if x < -10)
        med_mh = statistics.median([x["max_high"] for x in matches])
        med_ml = statistics.median([x["max_low"] for x in matches])
        print(f"{label}: {len(pnls)}命中 均值{statistics.mean(pnls):+.2f}% 中位{statistics.median(pnls):+.2f}% "
              f"胜率{win/len(pnls)*100:.1f}% 涨>20%占{big/len(pnls)*100:.1f}% 跌>10%占{big_loss/len(pnls)*100:.1f}% "
              f"max高{statistics.median(med_mh if False else [x['max_high'] for x in matches]):+.1f}% max低{med_ml:+.1f}%")
    stats(all_matches, "全部")
    stats([x for x in all_matches if x["in_sample"]], "样本内(库来源币命中自己)")
    stats([x for x in all_matches if not x["in_sample"]], "样本外(新币命中)")
    for mode in ("A", "B", "D"):
        stats([x for x in all_matches if x["mode"] == mode], f"模式{mode}")
    # 保存
    with open("/tmp/fp_report.json", "w", encoding="utf-8") as f:
        json.dump({"library_count": len(lib), "scanned": scanned, "matches": all_matches,
                   "in_sample": in_sample, "out_sample": out_sample}, f, ensure_ascii=False, indent=2)
    print(f"\n已保存 {len(all_matches)} 条命中到 /tmp/fp_report.json")


if __name__ == "__main__":
    main()