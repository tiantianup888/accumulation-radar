#!/usr/bin/env python3
"""庄家拉盘特征挖掘 — 第1阶段：构建特征矩阵（数据驱动，取代先验6签名）。

严格对齐 quant-ml-rules skill：
  - 时间因果：标签严格 forward，按 t_available 排序
  - t_available = 窗口末根日 i 收盘后；t_target=[i+1,i+5]；t_realized=i+5
  - 重叠标签 purge：窗口步长3日 < 前瞻5日，段间标签相交样本删除
  - Train/Select/Report 三角色按时间切，不相交

标签：forward_max_high_5d = max(h[i+1..i+5])/c[i] - 1  （连续值，评估时再切 θ）
特征池：连续值为主，不固化阈值。之前6签名的连续版也作为候选放入，留不留由数据决定。

产出：/tmp/feature_matrix.parquet
"""
import os, sys, time, statistics, math
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import duckdb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CST = timezone(timedelta(hours=8))
DB = "data/market.duckdb"
DAYS = 130          # 90天窗口+14日特征窗+5日前瞻+余量
WINDOW = 14         # 特征窗口长度
STEP = 3            # 窗口步长（<前瞻5日，需purge）
FORWARD = 5         # 前瞻天数
EMBARGO = 5         # 段边界禁运(=1标签窗, protocols §2)
TRAIN_FRAC, SELECT_FRAC = 0.60, 0.20  # report=0.20

OUT = "/tmp/feature_matrix.pkl"


def ema(arr, span):
    if len(arr) < span:
        return float("nan")
    k = 2 / (span + 1)
    e = arr[0]
    for x in arr[1:]:
        e = x * k + e * (1 - k)
    return e


def safe(x):
    return 0.0 if (x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))) else x


def build_features(bars, i):
    """bars 升序日K(含ts,o,h,l,c,v)，窗口=[i-13, i]（14根含末根i）。返回特征dict（连续值）。"""
    w = bars[i - WINDOW + 1 : i + 1]
    o = np.array([b["o"] for b in w], float)
    h = np.array([b["h"] for b in w], float)
    l = np.array([b["l"] for b in w], float)
    c = np.array([b["c"] for b in w], float)
    v = np.array([b["v"] for b in w], float)
    n = len(c)
    med_v = float(np.median(v)) if len(v) else 0.0
    f = {}

    # —— 收益统计 ——
    f["ret_1d"] = safe(c[-1] / c[-2] - 1) if n > 1 else 0
    f["ret_3d"] = safe(c[-1] / c[-4] - 1) if n > 4 else 0
    f["ret_5d"] = safe(c[-1] / c[-6] - 1) if n > 6 else 0
    f["ret_7d"] = safe(c[-1] / c[-8] - 1) if n > 8 else 0
    f["ret_14d"] = safe(c[-1] / c[0] - 1)
    dr = np.diff(c) / c[:-1] if n > 1 else np.array([0.0])
    f["ret_mean"] = safe(float(np.mean(dr)))
    f["ret_var"] = safe(float(np.var(dr)))
    f["ret_skew"] = safe(float(((dr - dr.mean()) ** 3).mean() / (dr.std() ** 3 + 1e-12))) if dr.std() > 0 else 0
    f["ret_kurt"] = safe(float(((dr - dr.mean()) ** 4).mean() / (dr.std() ** 4 + 1e-12) - 3)) if dr.std() > 0 else 0

    # —— 波动率 ——
    tr = np.array([max(h[k] - l[k], abs(h[k] - c[k - 1]), abs(l[k] - c[k - 1])) for k in range(1, n)], float)
    tr = np.concatenate([[h[0] - l[0]], tr])
    for span in (3, 5, 7, 14):
        if len(tr) >= span:
            f[f"atr_{span}_pct"] = safe(float(np.mean(tr[-span:])) / c[-1])
        else:
            f[f"atr_{span}_pct"] = 0.0
    amp_n = lambda s: (float(h[-s:].max()) - float(l[-s:].min())) / (float(np.mean(c[-s:])) + 1e-12)
    f["amplitude_14d"] = safe(amp_n(14))
    amp3 = amp_n(3) if n >= 3 else 0
    amp7 = amp_n(7) if n >= 7 else 0
    f["vol_conv_ratio"] = safe(amp3 / (amp7 + 1e-12))  # 收敛比连续值（近3/前7）
    f["vol_of_vol"] = safe(float(np.std(dr)) / (abs(float(np.mean(dr))) + 1e-12))

    # —— 量能 ——
    vr = v / (med_v + 1e-12) if med_v > 0 else np.zeros_like(v)
    f["vol_ratio_mean"] = safe(float(np.mean(vr)))
    f["vol_ratio_max"] = safe(float(np.max(vr)))
    f["vol_ratio_var"] = safe(float(np.var(vr)))
    x = np.arange(n, dtype=float)
    slope = float(np.polyfit(x, v, 1)[0]) if n >= 2 and np.var(v) > 0 else 0
    f["vol_trend_slope"] = safe(slope / (med_v + 1e-12))
    f["vol_price_corr"] = safe(float(np.corrcoef(c, v)[0, 1])) if np.std(c) > 0 and np.std(v) > 0 else 0
    f["heavy_vol_pct"] = safe(float(np.mean(v > 2 * med_v)) if med_v > 0 else 0)
    sign = np.sign(np.diff(c))
    obv = np.concatenate([[0.0], np.cumsum(sign * v[1:])]) if n > 1 else np.zeros(n)
    f["obv_slope"] = safe(float(np.polyfit(x, obv, 1)[0]) / (np.mean(np.abs(obv)) + 1e-12)) if np.std(obv) > 0 else 0

    # —— 位置 ——
    lo, hi = float(c.min()), float(c.max())
    f["pos"] = safe((c[-1] - lo) / (hi - lo + 1e-12))
    ma = float(np.mean(c))
    f["ma_dev"] = safe(c[-1] / (ma + 1e-12) - 1)
    sd = float(np.std(c))
    f["bb_width"] = safe(4 * sd / (ma + 1e-12))
    f["bb_pos"] = safe(np.clip((c[-1] - (ma - 2 * sd)) / (4 * sd + 1e-12), 0, 1)) if sd > 0 else 0.5

    # —— 形态 ——
    tp = 0
    for k in range(1, n - 1):
        if (c[k] > c[k - 1] and c[k] > c[k + 1]) or (c[k] < c[k - 1] and c[k] < c[k + 1]):
            tp += 1
    f["turning_points"] = tp
    f["up_days"] = int(np.sum(np.diff(c) > 0))
    f["down_days"] = int(np.sum(np.diff(c) < 0))
    run_max = np.maximum.accumulate(c)
    f["max_drawdown"] = safe(float(np.min((c - run_max) / run_max)))
    f["peak_valley_ratio"] = safe((hi - lo) / (lo + 1e-12))
    sl1 = float(np.polyfit(x[-7:], c[-7:], 1)[0]) if n >= 7 and np.std(c[-7:]) > 0 else 0
    sl2 = float(np.polyfit(x[:7], c[:7], 1)[0]) if n >= 7 and np.std(c[:7]) > 0 else 0
    f["accel"] = safe(sl1 - sl2)

    # —— 时序 ——
    f["autocorr_1"] = safe(float(np.corrcoef(dr[:-1], dr[1:])[0, 1])) if len(dr) > 2 and np.std(dr[:-1]) > 0 and np.std(dr[1:]) > 0 else 0
    # 简化 Hurst (R/S over window)
    if n >= 8:
        half = n // 2
        def rs(seg):
            m = np.mean(seg); acc = np.cumsum(seg - m); return (acc.max() - acc.min()) / (np.std(seg) + 1e-12)
        r1, r2 = rs(c[:half]), rs(c[half:])
        f["hurst_proxy"] = safe(math.log(max(r1, 1e-6) / max(r2, 1e-6)) / math.log(2) if r1 > 0 and r2 > 0 else 0)
    else:
        f["hurst_proxy"] = 0.0

    # —— 技术指标 ——
    gains = np.maximum(dr, 0); losses = np.maximum(-dr, 0)
    avg_g = float(np.mean(gains)) if len(gains) else 0
    avg_l = float(np.mean(losses)) if len(losses) else 0
    f["rsi_14"] = safe(100 - 100 / (1 + avg_g / (avg_l + 1e-12))) if avg_l > 0 else 100 if avg_g > 0 else 50
    ema_f = ema(c, 5); ema_s = ema(c, 10)
    f["macd_hist"] = safe((ema_f - ema_s) / (c[-1] + 1e-12)) if not math.isnan(ema_f) and not math.isnan(ema_s) else 0

    # —— 签名连续版（之前6签名的连续量，不固化阈值，留作候选）——
    f["test_pulse_count"] = 0
    for k in range(n):
        upper = (h[k] - max(o[k], c[k])) / (o[k] + 1e-12)
        chg = c[k] / o[k] - 1 if o[k] > 0 else 0
        if vr[k] > 2 and upper > 0.05 and -0.03 < chg < 0.03:
            f["test_pulse_count"] += 1
    f["washouts_count"] = int(np.sum((c / (o + 1e-12) - 1) < -0.08))
    f["washouts_count"] = f["washouts_count"]
    # 止跌连续版：前段跌幅 - 末段跌幅（正值=跌势收窄）
    prior_drop = float(np.mean(dr[:7]) - 1) if n >= 7 else 0  # 近似
    recent_drop = float(np.mean(dr[-3:])) if n >= 3 else 0
    f["standstill_score"] = safe(prior_drop - recent_drop)  # 先跌后收窄为正
    # 探底强度连续版：末日 量比 × 下影 × 创新低度
    last = w[-1]
    lower = (min(last["o"], last["c"]) - last["l"]) / (last["o"] + 1e-12)
    lo7 = float(np.min(l[-7:])) if n >= 7 else float(l.min())
    new_low_deg = max(0, (lo7 - last["l"]) / (lo7 + 1e-12))
    f["bottom_probe_strength"] = safe(vr[-1] * lower * (1 + new_low_deg))
    return f


def build_label(bars, i):
    """forward 5日最大涨幅（连续值，评估时切θ）。越界返回None。"""
    n = len(bars)
    if i + FORWARD >= n:
        return None
    entry = bars[i]["c"]
    fh = max(bars[k]["h"] for k in range(i + 1, i + 1 + FORWARD))
    fl = min(bars[k]["l"] for k in range(i + 1, i + 1 + FORWARD))
    fwd_max = fh / entry - 1
    fwd_close = bars[i + FORWARD]["c"] / entry - 1
    return {"fwd_max_high": fwd_max, "fwd_close": fwd_close,
            "fwd_max_low": fl / entry - 1}


def load_klines_db(con, symbol):
    """从 DuckDB 读真实日K(冻结K线已删), 返回升序bar dict。退市币只到真实末日→自动标签完整性。"""
    rows = con.execute(
        "SELECT open_time, open, high, low, close, volume "
        "FROM klines_daily WHERE symbol=? ORDER BY open_time", [symbol]).fetchall()
    return [{"ts": int(r[0].timestamp() * 1000), "o": r[1], "h": r[2],
             "l": r[3], "c": r[4], "v": r[5]} for r in rows]


def main():
    con = duckdb.connect(DB, read_only=True)
    # 可建模universe: 有真实日K的币(含退市SETTLING); status来自symbol_meta
    syms = con.execute("""SELECT k.symbol, m.status FROM klines_daily k
        LEFT JOIN symbol_meta m ON m.symbol=k.symbol
        GROUP BY k.symbol, m.status ORDER BY k.symbol""").fetchall()
    # 只取 TRADING/SETTLING(剔除 DELISTED_PREWINDOW/PENDING, 虽后者本就无真实K)
    cov = [(s, st) for s, st in syms if st in ("TRADING", "SETTLING")]
    print(f"可建模universe: {len(cov)} 币 (TRADING={sum(1 for _,s in cov if s=='TRADING')} "
          f"+ SETTLING退市={sum(1 for _,s in cov if s=='SETTLING')})，从DuckDB读真实日K...")
    rows = []
    for idx, (fs, status) in enumerate(cov):
        dk = load_klines_db(con, fs)
        if not dk or len(dk) < WINDOW + FORWARD:
            continue
        n = len(dk)
        # 窗口末根 i ∈ [WINDOW-1, n-1-FORWARD]，步长 STEP
        # 退市币 dk 只到真实末日 → i+FORWARD 越界自动跳过 = 标签完整性过滤(universe-as-of)
        for i in range(WINDOW - 1, n - FORWARD, STEP):
            feat = build_features(dk, i)
            lab = build_label(dk, i)
            if lab is None:
                continue
            t_avail = dk[i]["ts"]
            rows.append({"coin": fs, "status": status, "t_idx": i, "t_avail": t_avail,
                         "t_date": datetime.fromtimestamp(t_avail / 1000, tz=CST).strftime("%Y-%m-%d"),
                         **feat, **lab})
        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{len(cov)} 累积窗口 {len(rows)}", flush=True)
    con.close()
    print(f"\n总窗口: {len(rows)}")
    df = pd.DataFrame(rows)
    df = df.sort_values("t_avail").reset_index(drop=True)

    # —— 时间分段 + purge（重叠标签处理）——
    n = len(df)
    t1 = df["t_avail"].quantile(TRAIN_FRAC)
    t2 = df["t_avail"].quantile(TRAIN_FRAC + SELECT_FRAC)
    df["split_raw"] = np.where(df["t_avail"] < t1, "train",
                       np.where(df["t_avail"] < t2, "select", "report"))
    # purge: 训练样本标签窗口跨入选段边界的删除
    # 标签窗口 = [t_avail, t_avail+5日]。select/report 段边界处 purge train/select 中标签跨界的
    def label_end(ts):
        return ts + FORWARD * 86400 * 1000
    # train 中标签结尾 >= t1（跨入select）的 purge
    train_mask = df["split_raw"] == "train"
    purge_train = train_mask & (df["t_avail"].apply(label_end) >= t1)
    select_mask = df["split_raw"] == "select"
    purge_select = select_mask & (df["t_avail"].apply(label_end) >= t2)
    df["split"] = df["split_raw"]
    df.loc[purge_train, "split"] = "purged"
    df.loc[purge_select, "split"] = "purged"
    # embargo: 段边界后 embargo 的对侧训练样本也删(=1标签窗5日)
    emb = EMBARGO * 86400 * 1000
    df.loc[train_mask & (df["t_avail"] >= t1 - emb) & (df["t_avail"] < t1), "split"] = "purged"
    df.loc[select_mask & (df["t_avail"] >= t2 - emb) & (df["t_avail"] < t2), "split"] = "purged"

    # 标签: net_swing = fwd_max_high - |fwd_max_low| (净幅度, 排除双向波动噪音)
    # 阈值化 y_θ = 1{net_swing > θ} (θ在select定, 这里都存上供选)
    df["net_swing"] = df["fwd_max_high"] - df["fwd_max_low"].abs()
    for th in (0.05, 0.10, 0.15):
        df[f"y_{int(th*100)}"] = (df["net_swing"] > th).astype(int)

    df.to_pickle(OUT)
    cnt = df[df["split"] != "purged"]["split"].value_counts().to_dict()
    pos = {th: int(df[df["split"] == "select"][f"y_{th}"].sum()) for th in (5, 10, 15)}
    sel_n = int((df["split"] == "select").sum())
    print(f"分段(去purge后): {cnt}")
    print(f"select段正样本(net_swing): y5={pos[5]} y10={pos[10]} y15={pos[15]} / select总数={sel_n}")
    print(f"正样本率(select, y15): {pos[15]/max(sel_n,1)*100:.2f}%")
    # 退市币贡献
    if "status" in df.columns:
        settl = df[df["status"] == "SETTLING"]
        print(f"退市SETTLING币贡献窗口: {len(settl)} (其中有效非purge: {(settl['split']!='purged').sum()})")
    print(f"特征数: {len([c for c in df.columns if c not in ('coin','status','t_idx','t_avail','t_date','split','split_raw','fwd_max_high','fwd_close','fwd_max_low','net_swing','y_5','y_10','y_15')])}")
    print(f"\n已存 {OUT}  ({len(df)}行 × {len(df.columns)}列)")


if __name__ == "__main__":
    main()