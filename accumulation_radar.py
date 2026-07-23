#!/usr/bin/env python3
"""
庄家收筹雷达 v1 — 发现庄家横盘吸筹 + OI异动

核心逻辑（Patrick教的）：
1. 庄家拉盘前必须先收筹 → 长期横盘+低量 = 收筹中
2. OI暴涨 = 大资金进场建仓 = 即将拉盘
3. 两个信号叠加 = 最强信号

两个模块：
A. 横盘收筹标的池（每天扫一次）→ 找正在被庄家收筹的币
B. OI异动监控（每小时扫）→ 标的池内的币有OI异动立即报警

数据源：币安合约API（免费公开，零成本）
"""

import json
import os
import sys
import time
import requests
import sqlite3
import db as rdb
import fingerprint as fp_lib
from datetime import datetime, timezone, timedelta
from pathlib import Path

# === 加载 .env ===
env_file = Path(__file__).parent / ".env.oi"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

# === 配置 ===
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "")
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "")
FAPI = "https://fapi.binance.com"
DB_PATH = Path(os.environ.get("DB_PATH", str(Path(__file__).parent / "data" / "accumulation.db")))

# 收筹标的池参数
MIN_SIDEWAYS_DAYS = 45        # 至少横盘45天
MAX_RANGE_PCT = 80            # 横盘期价格波动<80%（宽松点，庄家盘波动可以大）
MAX_AVG_VOL_USD = 20_000_000  # 日均成交<$20M（低量才是收筹）
MIN_DATA_DAYS = 50            # 至少50天数据

# OI异动参数
MIN_OI_DELTA_PCT = 3.0        # OI变化至少3%
MIN_OI_USD = 2_000_000        # 最低OI门槛 $2M

# 放量突破参数
VOL_BREAKOUT_MULT = 3.0       # 当日Vol > 3x均值 = 放量


def api_get(endpoint, params=None):
    """币安API请求"""
    url = f"{FAPI}{endpoint}"
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                time.sleep(2)
            else:
                return None
        except:
            time.sleep(1)
    return None


def init_db():
    """初始化数据库（新 schema: signals/price_snapshots/pool_state/benchmark_snapshots）"""
    conn = rdb.connect(str(DB_PATH))
    rdb.init_db(conn)
    return conn

def get_all_perp_symbols():
    """获取所有USDT永续合约"""
    info = api_get("/fapi/v1/exchangeInfo")
    if not info:
        return []
    return [s["symbol"] for s in info["symbols"]
            if s["quoteAsset"] == "USDT" 
            and s["contractType"] == "PERPETUAL"
            and s["status"] == "TRADING"]


def analyze_accumulation(symbol, klines):
    """分析单个币的收筹特征"""
    if len(klines) < MIN_DATA_DAYS:
        return None
    
    data = []
    for k in klines:
        data.append({
            "ts": k[0],
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "vol": float(k[7]),  # quote volume (USDT)
        })
    
    coin = symbol.replace("USDT", "")
    
    # === 排除稳定币和指数 ===
    EXCLUDE = {"USDC", "USDP", "TUSD", "FDUSD", "BTCDOM", "DEFI", "USDM"}
    if coin in EXCLUDE:
        return None
    
    # === 排除已经暴涨过+崩盘的币 ===
    # 最近7天vs之前的均价，如果已经涨>300%就跳过（来不及了）
    recent_7d = data[-7:]
    prior = data[:-7]
    if not prior:
        return None
    
    recent_avg_px = sum(d["close"] for d in recent_7d) / len(recent_7d)
    prior_avg_px = sum(d["close"] for d in prior) / len(prior)
    
    if prior_avg_px > 0 and ((recent_avg_px - prior_avg_px) / prior_avg_px) > 3.0:
        return None  # 已经涨了300%+，来不及了
    
    # === 寻找横盘区间 ===
    # 从最近往回找，找最长的横盘期（价格波动<MAX_RANGE_PCT%）
    # 关键：必须是真横盘（斜率接近零），阴跌不算横盘！
    best_sideways = 0
    best_range = 0
    best_low = 0
    best_high = 0
    best_avg_vol = 0
    best_slope_pct = 0
    
    # 用滑动窗口从60天到全部
    for window in range(MIN_SIDEWAYS_DAYS, len(prior) + 1):
        window_data = prior[-window:]
        lows = [d["low"] for d in window_data]
        highs = [d["high"] for d in window_data]
        
        w_low = min(lows)
        w_high = max(highs)
        
        if w_low <= 0:
            continue
        
        range_pct = ((w_high - w_low) / w_low) * 100
        
        if range_pct <= MAX_RANGE_PCT:
            avg_vol = sum(d["vol"] for d in window_data) / len(window_data)
            if avg_vol <= MAX_AVG_VOL_USD:
                # 线性回归算斜率：阴跌/暴涨不算横盘
                closes = [d["close"] for d in window_data]
                n = len(closes)
                x_mean = (n - 1) / 2.0
                y_mean = sum(closes) / n
                num = sum((i - x_mean) * (c - y_mean) for i, c in enumerate(closes))
                den = sum((i - x_mean) ** 2 for i in range(n))
                slope = num / den if den > 0 else 0
                # 累计变化占起始价的百分比
                slope_pct = (slope * n / closes[0] * 100) if closes[0] > 0 else 0
                
                # 斜率过滤：累计变化超过±20%不算横盘
                if abs(slope_pct) > 20:
                    continue
                
                if window > best_sideways:
                    best_sideways = window
                    best_range = range_pct
                    best_low = w_low
                    best_high = w_high
                    best_avg_vol = avg_vol
                    best_slope_pct = slope_pct
    
    if best_sideways < MIN_SIDEWAYS_DAYS:
        return None
    
    # === 计算收筹评分 ===
    # 横盘越久越好（庄家需要时间吸筹）
    days_score = min(best_sideways / 90, 1.0) * 25  # 90天满分25
    
    # 区间越窄越好（控盘紧）
    range_score = max(0, (1 - best_range / MAX_RANGE_PCT)) * 20  # 越窄越高，满分20
    
    # 成交量越低越好（死水一潭 = 筹码集中）
    vol_score = max(0, (1 - best_avg_vol / MAX_AVG_VOL_USD)) * 20  # 越低越高，满分20
    
    # 最近是否开始放量？（放量是启动信号）
    recent_vol = sum(d["vol"] for d in recent_7d) / len(recent_7d)
    vol_breakout = recent_vol / best_avg_vol if best_avg_vol > 0 else 0
    breakout_score = min(vol_breakout / VOL_BREAKOUT_MULT, 1.0) * 15  # 放量加分，满分15
    
    # 市值越低空间越大（核心！Patrick: 低市值=大空间）
    # 用当前价格*日均成交量/换手率来粗估市值排名
    # 实际市值在推送时用CoinGecko补充
    est_mcap = data[-1]["close"] * best_avg_vol * 30  # 粗略估算
    if est_mcap > 0 and est_mcap < 50_000_000:
        mcap_score = 20  # <$50M 满分
    elif est_mcap < 100_000_000:
        mcap_score = 15
    elif est_mcap < 200_000_000:
        mcap_score = 10
    elif est_mcap < 500_000_000:
        mcap_score = 5
    else:
        mcap_score = 0
    
    total_score = days_score + range_score + vol_score + breakout_score + mcap_score
    
    # 横盘质量加分：斜率越接近零越好（真横盘bonus，满分+5）
    flatness_bonus = max(0, (1 - abs(best_slope_pct) / 20)) * 5
    total_score += flatness_bonus
    
    # 状态判断
    if vol_breakout >= VOL_BREAKOUT_MULT:
        status = "🔥放量启动"
    elif vol_breakout >= 1.5:
        status = "⚡开始放量"
    else:
        status = "💤收筹中"
    
    return {
        "symbol": symbol,
        "coin": coin,
        "sideways_days": best_sideways,
        "range_pct": best_range,
        "slope_pct": best_slope_pct,
        "low_price": best_low,
        "high_price": best_high,
        "avg_vol": best_avg_vol,
        "current_price": data[-1]["close"],
        "recent_vol": recent_vol,
        "vol_breakout": vol_breakout,
        "score": total_score,
        "status": status,
        "data_days": len(data),
    }


def scan_accumulation_pool():
    """扫描全市场，找正在被收筹的币"""
    print("📊 扫描全市场收筹标的...")
    
    symbols = get_all_perp_symbols()
    print(f"  共 {len(symbols)} 个合约")
    
    results = []
    
    for i, sym in enumerate(symbols):
        klines = api_get("/fapi/v1/klines", {
            "symbol": sym, "interval": "1d", "limit": 180
        })
        
        if klines and isinstance(klines, list):
            r = analyze_accumulation(sym, klines)
            if r:
                results.append(r)
        
        if (i + 1) % 10 == 0:
            time.sleep(0.5)
        if (i + 1) % 100 == 0:
            print(f"  进度: {i+1}/{len(symbols)}... 已发现{len(results)}个")
    
    results.sort(key=lambda x: x["score"], reverse=True)
    print(f"  ✅ 发现 {len(results)} 个收筹标的")
    return results


def scan_oi_changes(watchlist_symbols):
    """对标的池内的币扫描OI异动"""
    print(f"📊 扫描OI异动（{len(watchlist_symbols)}个标的）...")
    
    alerts = []
    
    for sym in watchlist_symbols:
        # OI历史
        oi_hist = api_get("/futures/data/openInterestHist", {
            "symbol": sym, "period": "1h", "limit": 3
        })
        
        if not oi_hist or len(oi_hist) < 2:
            continue
        
        prev_oi = float(oi_hist[-2]["sumOpenInterestValue"])
        curr_oi = float(oi_hist[-1]["sumOpenInterestValue"])
        
        if prev_oi <= 0 or curr_oi < MIN_OI_USD:
            continue
        
        delta_pct = ((curr_oi - prev_oi) / prev_oi) * 100
        
        if abs(delta_pct) >= MIN_OI_DELTA_PCT:
            # 拿当前价格
            ticker = api_get("/fapi/v1/ticker/24hr", {"symbol": sym})
            if not ticker:
                continue
            
            price = float(ticker["lastPrice"])
            vol_24h = float(ticker["quoteVolume"])
            px_chg = float(ticker["priceChangePercent"])
            
            # 拿费率
            funding = api_get("/fapi/v1/fundingRate", {"symbol": sym, "limit": 1})
            fr = float(funding[0]["fundingRate"]) if funding else 0
            
            coin = sym.replace("USDT", "")
            
            alerts.append({
                "symbol": sym,
                "coin": coin,
                "price": price,
                "oi_usd": curr_oi,
                "oi_delta_pct": delta_pct,
                "oi_delta_usd": curr_oi - prev_oi,
                "vol_24h": vol_24h,
                "px_chg_pct": px_chg,
                "funding_rate": fr,
            })
        
        time.sleep(0.3)
    
    alerts.sort(key=lambda x: abs(x["oi_delta_pct"]), reverse=True)
    print(f"  ✅ 发现 {len(alerts)} 个OI异动")
    return alerts


def format_usd(v):
    if v >= 1e9: return f"${v/1e9:.1f}B"
    if v >= 1e6: return f"${v/1e6:.1f}M"
    if v >= 1e3: return f"${v/1e3:.0f}K"
    return f"${v:.0f}"


def build_pool_report(results, top_n=25):
    """生成收筹标的池报告"""
    if not results:
        return ""
    
    now = datetime.now(timezone(timedelta(hours=8)))
    
    lines = [
        f"🏦 **庄家收筹雷达** — 标的池更新",
        f"⏰ {now.strftime('%Y-%m-%d %H:%M')} CST",
        f"━━━━━━━━━━━━━━━━━━",
        f"扫描 {len(results)} 个合约，发现标的：",
        "",
    ]
    
    # 分组：放量启动 > 开始放量 > 收筹中
    firing = [r for r in results if "放量启动" in r["status"]]
    warming = [r for r in results if "开始放量" in r["status"]]
    sleeping = [r for r in results if "收筹中" in r["status"]]
    
    if firing:
        lines.append(f"🔥 **放量启动** ({len(firing)}个) — 最高优先级！")
        for r in firing[:10]:
            lines.append(
                f"  🔥 **{r['coin']}** | 分:{r['score']:.0f} | "
                f"横盘{r['sideways_days']}天 | 波动{r['range_pct']:.0f}% | "
                f"Vol放大{r['vol_breakout']:.1f}x"
            )
            lines.append(
                f"     ${r['current_price']:.6f} | "
                f"区间: ${r['low_price']:.6f}~${r['high_price']:.6f} | "
                f"日均Vol: {format_usd(r['avg_vol'])}"
            )
        lines.append("")
    
    if warming:
        lines.append(f"⚡ **开始放量** ({len(warming)}个) — 关注中")
        for r in warming[:10]:
            lines.append(
                f"  ⚡ {r['coin']} | 分:{r['score']:.0f} | "
                f"横盘{r['sideways_days']}天 | 波动{r['range_pct']:.0f}% | "
                f"Vol{r['vol_breakout']:.1f}x"
            )
        lines.append("")
    
    if sleeping:
        lines.append(f"💤 **收筹中** ({len(sleeping)}个) — 持续监控")
        for r in sleeping[:15]:
            lines.append(
                f"  💤 {r['coin']} | 分:{r['score']:.0f} | "
                f"横盘{r['sideways_days']}天 | 波动{r['range_pct']:.0f}% | "
                f"日均Vol {format_usd(r['avg_vol'])}"
            )
    
    return "\n".join(lines)


def build_oi_alert_report(alerts, watchlist_coins):
    """生成OI异动报告（只报标的池内的）"""
    if not alerts:
        return ""
    
    now = datetime.now(timezone(timedelta(hours=8)))
    
    # 区分：池内 vs 池外
    in_pool = [a for a in alerts if a["symbol"] in watchlist_coins]
    out_pool = [a for a in alerts if a["symbol"] not in watchlist_coins]
    
    lines = [
        f"📊 **OI异动扫描** [收筹池]",
        f"⏰ {now.strftime('%Y-%m-%d %H:%M')} CST",
        f"━━━━━━━━━━━━━━━━━━",
        "",
    ]
    
    if in_pool:
        lines.append(f"🎯 **收筹池内异动** ({len(in_pool)}个) ⚠️ 重点关注!")
        for a in in_pool[:10]:
            emoji = "🟢" if a["oi_delta_pct"] > 0 else "🔴"
            lines.append(
                f"  {emoji} **{a['coin']}** | OI: {a['oi_delta_pct']:+.1f}% "
                f"({format_usd(a['oi_usd'])}) | 价格: {a['px_chg_pct']:+.1f}%"
            )
            # 信号解读
            if a["oi_delta_pct"] > 0 and abs(a["px_chg_pct"]) < 3:
                lines.append(f"     ⚡ 暗流涌动！OI涨但价格平 = 庄家建仓中")
            elif a["oi_delta_pct"] > 0 and a["px_chg_pct"] > 3:
                lines.append(f"     🚀 放量拉升！OI+价格同涨 = 启动中")
        lines.append("")
    
    if out_pool:
        lines.append(f"📋 池外异动 ({len(out_pool)}个)")
        for a in out_pool[:8]:
            emoji = "🟢" if a["oi_delta_pct"] > 0 else "🔴"
            lines.append(
                f"  {emoji} {a['coin']} | OI: {a['oi_delta_pct']:+.1f}% | "
                f"价格: {a['px_chg_pct']:+.1f}%"
            )
    
    return "\n".join(lines)


def send_telegram(text):
    """发送TG消息"""
    if not TG_BOT_TOKEN:
        print("\n[TG] No token, stdout:\n")
        print(text)
        return
    
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    
    # 分段发送（TG限制4096字）
    chunks = []
    current = ""
    for line in text.split("\n"):
        if len(current) + len(line) + 1 > 3800:
            chunks.append(current)
            current = line
        else:
            current += "\n" + line if current else line
    if current:
        chunks.append(current)
    
    for chunk in chunks:
        try:
            resp = requests.post(url, json={
                "chat_id": TG_CHAT_ID,
                "text": chunk,
                "parse_mode": "Markdown"
            }, timeout=10)
            if resp.status_code == 200:
                print(f"[TG] Sent ✓ ({len(chunk)} chars)")
            else:
                # Markdown失败就用纯文本
                resp2 = requests.post(url, json={
                    "chat_id": TG_CHAT_ID,
                    "text": chunk.replace("*", "").replace("_", ""),
                }, timeout=10)
                print(f"[TG] Sent plain ({'✓' if resp2.status_code == 200 else '✗'})")
        except Exception as e:
            print(f"[TG] Error: {e}")
        time.sleep(0.5)


def save_watchlist(conn, results):
    """保存收筹池快照到 pool_state 表"""
    snap_ts = rdb.now_iso()
    rdb.save_pool_state(conn, snap_ts, [
        {"symbol": r["symbol"], "sideways_days": r["sideways_days"],
         "range_pct": r["range_pct"], "avg_vol": r["avg_vol"],
         "pool_score": r["score"], "status": r["status"]}
        for r in results
    ])
    print(f"  💾 保存 {len(results)} 个标的到 pool_state @ {snap_ts}")

def load_watchlist_symbols(conn):
    """从 pool_state 加载最新一批标的池 symbol"""
    return [r["symbol"] for r in rdb.get_pool_state_latest(conn)]

def scan_short_fuel():
    """策略2: 空头燃料 — 涨了+费率负+OI大 = 庄家拉盘爆空单"""
    print("📊 扫描空头燃料（费率为负+在涨的币）...")
    
    tickers = api_get("/fapi/v1/ticker/24hr")
    premiums = api_get("/fapi/v1/premiumIndex")
    
    if not tickers or not premiums:
        return [], []
    
    funding_map = {p["symbol"]: float(p["lastFundingRate"]) 
                   for p in premiums if p["symbol"].endswith("USDT")}
    
    fuel_targets = []     # 已在涨+费率负 = 正在squeeze
    squeeze_targets = []  # 费率极负+还没大涨 = 潜在squeeze
    
    for t in tickers:
        sym = t["symbol"]
        if not sym.endswith("USDT"):
            continue
        
        px_chg = float(t["priceChangePercent"])
        vol = float(t["quoteVolume"])
        fr = funding_map.get(sym, 0)
        coin = sym.replace("USDT", "")
        price = float(t["lastPrice"])
        
        item = {
            "coin": coin, "symbol": sym,
            "px_chg": px_chg, "funding": fr,
            "vol": vol, "price": price,
        }
        
        # 正在squeeze: 涨>5% + 费率负 + Vol>$5M
        if px_chg > 5 and fr < -0.0003 and vol > 5_000_000:
            item["fuel_score"] = abs(fr) * 10000 * px_chg
            fuel_targets.append(item)
        
        # 潜在squeeze: 费率很负 + 还没大涨(<10%) + Vol>$2M
        elif fr < -0.0005 and px_chg < 10 and vol > 2_000_000:
            item["fuel_score"] = abs(fr) * 10000
            squeeze_targets.append(item)
    
    fuel_targets.sort(key=lambda x: x["fuel_score"], reverse=True)
    squeeze_targets.sort(key=lambda x: x["fuel_score"], reverse=True)
    
    print(f"  ✅ 正在squeeze: {len(fuel_targets)}个, 潜在squeeze: {len(squeeze_targets)}个")
    return fuel_targets, squeeze_targets


def build_fuel_report(fuel_targets, squeeze_targets):
    """生成空头燃料报告"""
    if not fuel_targets and not squeeze_targets:
        return ""
    
    now = datetime.now(timezone(timedelta(hours=8)))
    lines = [
        f"🔥 **空头燃料扫描**",
        f"⏰ {now.strftime('%Y-%m-%d %H:%M')} CST",
        f"━━━━━━━━━━━━━━━━━━",
        f"逻辑：费率负=大量做空，庄家拉盘爆空单+收资金费",
        "",
    ]
    
    if fuel_targets:
        lines.append(f"🚀 **正在Squeeze** ({len(fuel_targets)}个) — 涨了+空头还在扛")
        for t in fuel_targets[:8]:
            fr_pct = t["funding"] * 100
            flag = "🎯极度!" if fr_pct < -0.1 else "⚠️"
            lines.append(
                f"  {flag} **{t['coin']}** | 涨{t['px_chg']:+.1f}% | "
                f"费率🧊{fr_pct:.4f}% | Vol {format_usd(t['vol'])}"
            )
        lines.append("")
    
    if squeeze_targets:
        lines.append(f"🎯 **潜在Squeeze** ({len(squeeze_targets)}个) — 费率极负+还没大涨")
        for t in squeeze_targets[:8]:
            fr_pct = t["funding"] * 100
            lines.append(
                f"  🧊 {t['coin']} | 价格{t['px_chg']:+.1f}% | "
                f"费率{fr_pct:.4f}% | Vol {format_usd(t['vol'])}"
            )
    
    return "\n".join(lines)


def persist_signals(conn, candidates, scan_mode="oi"):
    """横盘暗筹信号 Top20 落库（策略 sideways_acc）。"""
    st = rdb.now_iso()
    n = 0
    for i, c in enumerate(candidates[:20]):
        n += _save_one(conn, st, scan_mode, "sideways_acc", i + 1, c["score"], c)
    print(f"  💾 落库 {n} 条信号 @ {st}")


def persist_fingerprint_signals(conn, matches, scan_mode="fp"):
    """指纹匹配信号落库（策略 fingerprint）。"""
    st = rdb.now_iso()
    n = 0
    for i, m in enumerate(matches):
        coin_name = m["coin"].replace("USDT", "")
        d = {"sym": m["coin"], "coin": coin_name, "price": m["entry_price"],
             "px_chg": None, "fr_pct": None, "fp_mode": m["mode"],
             "fp_similarity": m["similarity"], "fp_source": m["fp_id"],
             "fp_signatures": m["signatures"]}
        tags = [m["mode"], "src=" + m["source"]]
        n += _save_one(conn, st, scan_mode, "fingerprint", i + 1, m["similarity"] * 100, d, tags=tags)
    print(f"  💾 落库 {n} 条指纹信号 @ {st}")


def _save_one(conn, st, scan_mode, strategy, rank, score, d, tags=None):
    sig = {
        "signal_time": st, "scan_mode": scan_mode, "strategy": strategy,
        "rank_in_strategy": rank, "score": score,
        "symbol": d["sym"], "coin": d["coin"], "entry_price": d["price"],
        "px_chg_24h": d.get("px_chg"), "funding_rate": d.get("fr_pct"),
        "oi_usd": d.get("oi_usd"), "oi_delta_6h": d.get("d6h"),
        "oi_slope_5d": d.get("oi_slope"),
        "mcap": d.get("est_mcap"), "sideways_days": d.get("sw_days"),
        "in_pool": d.get("in_pool"), "heat": d.get("heat"),
        "amplitude_5d": d.get("amp"), "spot_vol_ratio": d.get("vol_ratio"),
        "spot_premium": d.get("prem"),
        "f_sc": d.get("f_sc"), "m_sc": d.get("m_sc"),
        "s_sc": d.get("s_sc"), "o_sc": d.get("o_sc"),
        "tags": tags,
        "fp_mode": d.get("fp_mode"), "fp_similarity": d.get("fp_similarity"),
        "fp_source": d.get("fp_source"), "fp_signatures": d.get("fp_signatures"),
        "ml_prob": d.get("ml_prob"), "ml_pos": d.get("ml_pos"),
        "ml_features": d.get("ml_features"),
    }
    rdb.insert_signal(conn, sig)
    return 1


def collect_price_snapshots(conn):
    """对所有活跃信号 symbol 拉最新价，落 price_snapshots。"""
    syms = rdb.active_signal_symbols(conn)
    if not syms:
        print("  📸 无活跃信号，跳过快照")
        return
    tickers = api_get("/fapi/v1/ticker/24hr")
    if not tickers:
        print("  ⚠️ 快照采集失败：拿不到行情")
        return
    px_map = {t["symbol"]: float(t["lastPrice"]) for t in tickers if t["symbol"].endswith("USDT")}
    ts = rdb.now_iso()
    n = 0
    for sym in syms:
        if sym in px_map:
            rdb.insert_snapshot(conn, sym, ts, px_map[sym])
            n += 1
    print(f"  📸 价格快照 {n}/{len(syms)} @ {ts}")


def collect_benchmark(conn):
    """基准采样：有活跃组则跟踪其 symbol，否则新建一组随机 50 币。"""
    import random
    tickers = api_get("/fapi/v1/ticker/24hr")
    if not tickers:
        print("  ⚠️ 基准采样失败：拿不到行情")
        return
    px_map = {t["symbol"]: float(t["lastPrice"]) for t in tickers if t["symbol"].endswith("USDT")}
    cohort = rdb.active_benchmark_group(conn)
    ts = rdb.now_iso()
    if cohort is None:
        info = api_get("/fapi/v1/exchangeInfo")
        if not info:
            print("  ⚠️ 基准新建失败：拿不到合约列表")
            return
        all_syms = [s["symbol"] for s in info["symbols"]
                    if s["quoteAsset"] == "USDT" and s["contractType"] == "PERPETUAL" and s["status"] == "TRADING"]
        sample = random.sample(all_syms, min(rdb.BENCHMARK_SAMPLE, len(all_syms)))
        cohort = ts
        n = 0
        for sym in sample:
            if sym in px_map:
                rdb.insert_benchmark(conn, sym, ts, px_map[sym], cohort)
                n += 1
        print(f"  🎯 基准新建组 {cohort} 采价 {n}")
    else:
        syms = rdb.benchmark_group_symbols(conn, cohort)
        n = 0
        for sym in syms:
            if sym in px_map:
                rdb.insert_benchmark(conn, sym, ts, px_map[sym], cohort)
                n += 1
        print(f"  🎯 基准跟踪组 {cohort} 采价 {n}/{len(syms)}")


# 现货 symbol 映射缓存
_SPOT_MAP = None

def get_spot_map():
    """合约永续 symbol → 现货 symbol 映射（有则返回现货名，无则 None）。"""
    global _SPOT_MAP
    if _SPOT_MAP is not None: return _SPOT_MAP
    si = api_get_full("https://api.binance.com/api/v3/exchangeInfo")
    fi = api_get("/fapi/v1/exchangeInfo")
    spot = {s["symbol"] for s in si["symbols"] if s["quoteAsset"]=="USDT" and s["status"]=="TRADING"} if si else set()
    fut = [s["symbol"] for s in fi["symbols"] if s["contractType"]=="PERPETUAL" and s["quoteAsset"]=="USDT" and s["status"]=="TRADING"] if fi else []
    def m(fs):
        if fs.startswith("1000"):
            c = fs[3:]; return c if c in spot else None
        return fs if fs in spot else None
    _SPOT_MAP = {fs: m(fs) for fs in fut}
    return _SPOT_MAP


def api_get_full(url):
    try:
        import requests as _r
        resp = _r.get(url, timeout=15)
        return resp.json() if resp.status_code==200 else None
    except Exception:
        return None


def scan_sideways_acc(conn):
    """信号① 横盘暗筹全市场扫描（重构后唯一策略）。
    回测验证：95天154信号、超额+0.79%、胜率48.7%，跑赢基准。"""
    import statistics
    smap = get_spot_map()
    cov = [fs for fs in smap if smap[fs]]
    print(f"  📐 全市场横盘暗筹扫描：{len(cov)} 个可映射现货")
    # 批量拿合约费率+markPrice、现货ticker、合约ticker
    premiums = api_get("/fapi/v1/premiumIndex") or []
    fut_tk = {t["symbol"]: t for t in (api_get("/fapi/v1/ticker/24hr") or []) if t["symbol"].endswith("USDT")}
    fr_map = {p["symbol"]: float(p["lastFundingRate"])*100 for p in premiums if p["symbol"].endswith("USDT")}
    mark_map = {p["symbol"]: float(p["markPrice"]) for p in premiums if p["symbol"].endswith("USDT")}
    # 真实市值（现货bapi）
    mcap_map = {}
    bj = api_get_full("https://www.binance.com/bapi/composite/v1/public/marketing/symbol/list")
    if bj:
        for item in bj.get("data", []):
            nm = item.get("name", ""); mc = item.get("marketCap", 0)
            if nm and mc: mcap_map[nm] = float(mc)
    candidates = []
    for i, fs in enumerate(cov):
        ss = smap[fs]
        k = api_get_full(f"https://api.binance.com/api/v3/klines?symbol={ss}&interval=1h&limit=168")
        if not k or len(k) < 168: continue
        highs = [float(x[2]) for x in k]; lows = [float(x[3]) for x in k]
        closes = [float(x[4]) for x in k]; vols = [float(x[5]) for x in k]
        amp = (max(highs)-min(lows))/statistics.mean(closes)*100
        chg5d = (closes[-1]/closes[0]-1)*100
        chg24 = (closes[-1]/closes[-24]-1)*100
        prev7 = statistics.mean(vols[-168:-24]); vol_ratio = statistics.mean(vols[-24:])/prev7 if prev7>0 else 0
        frate = fr_map.get(fs, 0)
        # 信号① 基本条件
        if not (amp < 8 and chg5d < 3 and vol_ratio > 1.1 and frate <= 0.01 and chg5d > -8):
            continue
        # 评分：振幅越小越好、量比越高越好、资金费越负越好
        amp_sc = 25 if amp<4 else 18 if amp<6 else 10
        vol_sc = 25 if vol_ratio>1.5 else 18 if vol_ratio>1.3 else 10
        fr_sc = 20 if frate<-0.05 else 12 if frate<-0.01 else 5
        # 溢价（bonus）
        sprice = closes[-1]; mprice = mark_map.get(fs, 0)
        prem = (sprice-mprice)/mprice*100 if mprice>0 else 0
        prem_sc = 8 if prem>0.05 else 0
        coin = fs.replace("USDT", "")
        est_mcap = mcap_map.get(coin, 0)
        d = {"sym": fs, "coin": coin, "price": sprice, "px_chg": chg24,
             "fr_pct": frate, "amp": amp, "vol_ratio": vol_ratio, "prem": prem,
             "est_mcap": est_mcap, "in_pool": False, "heat": 0, "sw_days": 0,
             "f_sc": fr_sc, "s_sc": amp_sc, "m_sc": 0, "o_sc": 0}
        # OI 缓增加分（实盘叠加，非必要）—仅对基本入选者拉 OI
        oi_sc = 0; oi_slope = None; oi_usd = 0; d6h = 0
        oi_hist = api_get("/futures/data/openInterestHist", {"symbol": fs, "period": "1h", "limit": 120})
        if oi_hist and len(oi_hist) >= 24:
            oi_vals = [float(x["sumOpenInterestValue"]) for x in oi_hist]
            oi_slope = (oi_vals[-1]/oi_vals[0]-1)*100 if oi_vals[0]>0 else 0
            d6h = (oi_vals[-1]/oi_vals[-7]-1)*100 if oi_vals[-7]>0 else 0
            oi_usd = oi_vals[-1]
            if oi_slope > 10: oi_sc = 15
            elif oi_slope > 3: oi_sc = 8
            elif oi_slope > 0: oi_sc = 4
        d["oi_slope"] = oi_slope; d["oi_usd"] = oi_usd; d["d6h"] = d6h; d["o_sc"] = oi_sc
        d["score"] = amp_sc + vol_sc + fr_sc + prem_sc + oi_sc
        candidates.append(d)
        time.sleep(0.12)
        if (i+1) % 40 == 0: print(f"    {i+1}/{len(cov)}... 入选 {len(candidates)}", flush=True)
    candidates.sort(key=lambda x: -x["score"])
    print(f"  ✅ 入选 {len(candidates)} 个横盘暗筹候选，Top20 落库")
    # Telegram 报告
    if candidates:
        now = datetime.now(timezone(timedelta(hours=8)))
        def mcap_str(v):
            if v>=1e9: return f"${v/1e9:.1f}B"
            if v>=1e6: return f"${v/1e6:.0f}M"
            return f"${v:.0f}"
        lines = [f"🏦 横盘暗筹雷达 (回测验证跑赢基准)", f"⏰ {now.strftime('%Y-%m-%d %H:%M')} CST",
                 f"📊 入选 {len(candidates)} 个（振幅<8%+量比>1.1+资金费中性）", ""]
        for s in candidates[:15]:
            tags = []
            if s.get("oi_slope") is not None and s["oi_slope"]>0: tags.append(f"⚡OI{s['oi_slope']:+.0f}%5d")
            if s["prem"]>0.05: tags.append(f"溢价{s['prem']:+.3f}%")
            if s["fr_pct"]<-0.01: tags.append(f"🧊{s['fr_pct']:.3f}%")
            lines.append(f"  {s['coin']:<8} {s['score']}分 振幅{s['amp']:.1f}% 量比{s['vol_ratio']:.2f} {' '.join(tags)}")
        send_telegram("\n".join(lines))
    persist_signals(conn, candidates, scan_mode="oi")
    collect_price_snapshots(conn)
    collect_benchmark(conn)


def scan_fingerprints(conn):
    """庄家拉盘指纹库匹配扫描（策略 fingerprint）。
    90天样本外回测验证：289命中 均值+3.11% 胜率52.9% 跌>10%仅8%。
    锁定配置: similarity>=0.65 + 收敛 + 止跌 + 低位(pos<0.3)。"""
    lib = fp_lib.load_library()
    if not lib:
        print("  ⚠️ 指纹库为空，请先运行 build_fingerprint_library.py")
        return
    smap = get_spot_map()
    cov = [fs for fs in smap if smap[fs]]
    print(f"  🧬 指纹库匹配扫描：{len(cov)}币 × {len(lib)}指纹")
    matches = []
    for i, fs in enumerate(cov):
        ss = smap[fs]
        # 拉近20天日K(现货优先, 回退合约)
        k = api_get_full(f"https://api.binance.com/api/v3/klines?symbol={ss}&interval=1d&limit=20")
        if not k or len(k) < 16:
            k = api_get_full(f"https://fapi.binance.com/fapi/v1/klines?symbol={fs}&interval=1d&limit=20")
        if not k or len(k) < 16:
            continue
        bars = [{"ts": int(x[0]), "o": float(x[1]), "h": float(x[2]),
                 "l": float(x[3]), "c": float(x[4]), "v": float(x[5])} for x in k]
        hits = fp_lib.match_window(bars, lib, 14, 0.65)
        for h in hits:
            sig = h["signatures"]
            # 锁定配置过滤: 收敛 + 止跌 + 低位
            if not (sig.get("converge") and sig.get("standstill") and sig.get("pos", 1) < 0.3):
                continue
            h["coin"] = fs
            matches.append(h)
        time.sleep(0.12)
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(cov)}... 命中 {len(matches)}", flush=True)
    # 同币只保留最高相似度
    dedup = {}
    for m in matches:
        c = m["coin"]
        if c not in dedup or m["similarity"] > dedup[c]["similarity"]:
            dedup[c] = m
    final = sorted(dedup.values(), key=lambda x: -x["similarity"])[:20]
    print(f"  ✅ 指纹命中 {len(matches)} → 去重 {len(dedup)} → Top20 落库")
    if final:
        now = datetime.now(timezone(timedelta(hours=8)))
        lines = ["🧬 庄家拉盘指纹雷达 (90天样本外期望+3.11%)",
                 "⏰ " + now.strftime("%Y-%m-%d %H:%M") + " CST",
                 "📊 锁定配置: 相似度>=0.65 + 收敛 + 止跌 + 低位", ""]
        for m in final[:15]:
            s = m["signatures"]
            lines.append("  " + m["coin"].replace("USDT", "") + " sim" + str(m["similarity"]) +
                         " [" + m["mode"] + "] 源" + m["source"].replace("USDT", "") +
                         " conv" + str(s.get("converge_ratio", 1)) +
                         " pos" + str(s.get("pos", 1)) + " 止跌" + str(s.get("standstill")))
        send_telegram("\n".join(lines))
    persist_fingerprint_signals(conn, final)
    collect_price_snapshots(conn)
    collect_benchmark(conn)


# ── ML 扫描模式 (LightGBM 拉盘预测, 唯一生产策略) ──────────────────────
_ML_CACHE = {"model": None, "cfg": None}

def _load_ml_model():
    """惰性加载 LightGBM 模型 + 配置 (缓存)。"""
    if _ML_CACHE["model"] is not None:
        return _ML_CACHE["model"], _ML_CACHE["cfg"]
    import lightgbm as lgb
    cfg_path = Path(os.environ.get("ML_CFG_PATH", str(Path(__file__).parent / "data" / "model_config.json")))
    model_path = Path(os.environ.get("ML_MODEL_PATH", str(Path(__file__).parent / "data" / "model.txt")))
    cfg = json.load(open(cfg_path))
    model = lgb.Booster(model_file=str(model_path))
    _ML_CACHE["model"] = model
    _ML_CACHE["cfg"] = cfg
    print(f"  🧠 加载 LightGBM 模型: {len(cfg['features'])}特征 best_iter={cfg['best_iteration']} SelectPR-AUC={cfg['select_pr_auc']:.4f}")
    return model, cfg

def scan_ml(conn):
    """ML扫描模式 (策略 ml) — LightGBM 预测庄家拉盘概率。
    数据源: market.duckdb (日K + funding, 需先 refresh 保持新鲜)。
    流程: 全市场打分 → pos<0.3低位护栏 → Top20落库 + TG推送。
    模型B配置(14特征含funding), 样本外 Select PR-AUC=0.2430, Precision@Top20+pos=0.30。"""
    import numpy as np, duckdb
    from feature_mining import build_features, load_klines_db, WINDOW
    from funding_features import load_funding, compute_fr
    model, cfg = _load_ml_model()
    FEATS = cfg["features"]; POS_F = cfg["pos_filter"]; BEST = cfg["best_iteration"]
    duck_path = os.environ.get("DUCKDB_PATH", str(Path(__file__).parent / "data" / "market.duckdb"))
    if not Path(duck_path).exists():
        print(f"  ⚠️ market.duckdb 不存在({duck_path}), 请先运行 refresh 模式")
        return
    dcon = duckdb.connect(duck_path, read_only=True)
    fr_map = load_funding(dcon)
    syms = [r[0] for r in dcon.execute(
        "SELECT k.symbol FROM klines_daily k "
        "JOIN symbol_meta m ON m.symbol=k.symbol "
        "WHERE m.status='TRADING' GROUP BY k.symbol ORDER BY k.symbol").fetchall()]
    print(f"  🧠 ML扫描: {len(syms)}币(TRADING) × LightGBM({len(FEATS)}特征) + pos<{POS_F}护栏")
    rows = []
    for i, sym in enumerate(syms):
        bars = load_klines_db(dcon, sym)
        if len(bars) < WINDOW:
            continue
        idx = len(bars) - 1
        f = build_features(bars, idx)
        t_dec = int(bars[idx]["ts"]) + 86400000
        fr = compute_fr(*fr_map.get(sym, (np.array([]), np.array([]))), t_dec) if sym in fr_map else None
        f["fr_min_early"] = fr["fr_min_early"] if fr else np.nan
        f["fr_late_val"] = fr["fr_late_val"] if fr else np.nan
        pos = f.get("pos", np.nan)
        if pos is None or (isinstance(pos, float) and np.isnan(pos)) or pos >= POS_F:
            continue  # 低位护栏: 只留 pos<0.3
        x = np.array([[f.get(k, np.nan) for k in FEATS]])
        prob = float(model.predict(x, num_iteration=BEST)[0])
        rows.append({"sym": sym, "coin": sym.replace("USDT", ""), "price": bars[idx]["c"],
                     "prob": prob, "pos": float(pos), "ret_14d": float(f.get("ret_14d", np.nan)),
                     "amp": float(f.get("amplitude_14d", np.nan)),
                     "fr_late": float(f["fr_late_val"]) if fr and not np.isnan(f["fr_late_val"]) else None,
                     "feats": {k: (None if (isinstance(f.get(k), float) and np.isnan(f.get(k))) else f.get(k))
                               for k in ["atr_14_pct", "ret_var", "fr_late_val", "ret_skew", "vol_ratio_var", "fr_min_early"]}})
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(syms)}... 过护栏 {len(rows)}", flush=True)
    dcon.close()
    rows.sort(key=lambda x: -x["prob"])
    final = rows[:20]
    print(f"  ✅ 过护栏 {len(rows)} 个低位候选 → Top20 落库")
    if final:
        now = datetime.now(timezone(timedelta(hours=8)))
        lines = ["🧠 庄家拉盘ML雷达 (LightGBM, Select PR-AUC 0.2430)",
                 "⏰ " + now.strftime("%Y-%m-%d %H:%M") + " CST",
                 f"📊 低位护栏 pos<{POS_F} · 全市场{len(syms)}币 → {len(rows)}候选 → Top20", ""]
        for r in final[:15]:
            lines.append(f"  {r['coin']:<10} {r['prob']*100:.1f}% pos{r['pos']:.2f} "
                         f"ret14d{r['ret_14d']*100:+.0f}% amp{r['amp']*100:.0f}%")
        send_telegram("\n".join(lines))
    persist_ml_signals(conn, final)
    collect_price_snapshots(conn)
    collect_benchmark(conn)

def persist_ml_signals(conn, rows):
    """ML信号落库(策略 ml)。"""
    st = rdb.now_iso()
    n = 0
    for i, r in enumerate(rows):
        d = {"sym": r["sym"], "coin": r["coin"], "price": r["price"],
             "px_chg": r["ret_14d"] * 100, "fr_pct": r["fr_late"],
             "ml_prob": r["prob"], "ml_pos": r["pos"], "ml_features": r["feats"]}
        tags = ["ml", f"pos{r['pos']:.2f}"]
        n += _save_one(conn, st, "ml", "ml", i + 1, r["prob"] * 100, d, tags=tags)
    print(f"  💾 落库 {n} 条ML信号 @ {st}")

def refresh_market_data():
    """刷新 market.duckdb: 拉取日K + funding (运行离线fetch脚本, 写入挂载卷)。"""
    import subprocess
    duck_path = os.environ.get("DUCKDB_PATH", str(Path(__file__).parent / "data" / "market.duckdb"))
    print(f"  🔄 刷新 {duck_path} (日K + funding)...")
    for script, days in (("fetch_market_data.py", "130"), ("fetch_funding.py", "90")):
        print(f"    -- {script} --")
        try:
            subprocess.run([sys.executable, script, "--days", days, "--db", duck_path],
                           timeout=3600, check=False)
        except Exception as e:
            print(f"    {script} error: {e}")
    print("  ✅ 刷新完成")


def run_loop():
    """守护模式：容器内运行，按时间调度 refresh/ml/snap/bench。
    调度: refresh+ml 每日01:30(数据刷新后打分), snap每小时(PnL追踪), bench每日。"""
    import subprocess
    print(f"🏦 雷达守护模式启动 @ {rdb.now_iso()}")
    last_ml_day = None
    last_snap_hour = None
    last_bench_day = None
    while True:
        now = datetime.now(rdb.CST)
        hh = now.hour; today = now.date()
        # ML扫描: 每日 01:30 (先refresh再ml, 同一子进程串行)
        if today != last_ml_day and hh == 1 and now.minute >= 30:
            print(f"\n[-- spawn refresh+ml {now} --]")
            try:
                subprocess.run([sys.executable, __file__, "refresh"], timeout=3600)
                subprocess.run([sys.executable, __file__, "ml"], timeout=1800)
            except Exception as e:
                print(f"ml spawn error: {e}")
            last_ml_day = today
        # 价格快照: 每小时 :05 (PnL实时追踪)
        if hh != last_snap_hour and now.minute >= 5:
            print(f"\n[-- spawn snap {now} --]")
            try:
                subprocess.run([sys.executable, __file__, "snap"], timeout=600)
            except Exception as e:
                print(f"snap spawn error: {e}")
            last_snap_hour = hh
        # 基准采样: 每日 02:00
        if today != last_bench_day and hh == 2 and now.minute >= 0:
            print(f"\n[-- spawn bench {now} --]")
            try:
                subprocess.run([sys.executable, __file__, "bench"], timeout=600)
            except Exception as e:
                print(f"bench spawn error: {e}")
            last_bench_day = today
        time.sleep(30)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "loop"
    
    print(f"🏦 庄家拉盘ML雷达 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   模式: {mode}\n")
    
    conn = init_db()
    if mode == "loop":
        run_loop(); return
    if mode == "snap":
        collect_price_snapshots(conn); conn.close(); print("\n✅ 完成"); return
    if mode == "bench":
        collect_benchmark(conn); conn.close(); print("\n✅ 完成"); return
    if mode == "refresh":
        refresh_market_data(); conn.close(); print("\n✅ 完成"); return
    if mode in ("ml", "full"):
        # 唯一生产策略: LightGBM 拉盘预测 + pos<0.3低位护栏
        scan_ml(conn)
    conn.close()
    print("\n✅ 完成")

if __name__ == "__main__":
    main()
