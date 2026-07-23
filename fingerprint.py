"""庄家拉盘指纹库 — 指纹提取、相似匹配、库管理。

指纹定义：一次真实拉盘事件前14天的量价"编舞"，归一化为可跨币比较的形状向量。
核心思路（来自 ERA/ESPORTS/RE/BANK 四币实证）：
  - 不求一套参数适配所有庄家，而是采集足够多的指纹，下次见到相似形状即预警。
  - 庄家策略本质是低胜率高 payoff 的肥尾策略，评判用期望值/肥尾捕获率而非胜率。

指纹由日K聚合而成（14个交易日），包含：
  - price_shape: 14日收盘价 min-max 归一化到 [0,1]
  - vol_ratio_shape: 14日量 / 14日中位量
  - signatures: 关键行为签名（试盘脉冲/洗盘/收敛/止跌/探底）
  - mode: A低点吸筹 / B洗盘收敛 / D收敛爆发（C超跌反弹为假模式不入库）
  - provenance: 来源币、拉盘日、涨幅
"""
from __future__ import annotations
import json, os, statistics, math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import List, Optional

CST = timezone(timedelta(hours=8))
LIB_PATH = os.environ.get(
    "FP_LIB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "fingerprints.json"),
)


@dataclass
class Fingerprint:
    fp_id: str               # 形如 "ERAUSDT_20260721"
    mode: str                # A / B / D
    source_coin: str         # 来源合约 symbol
    source_spot: str         # 来源现货 symbol
    pump_date: str           # YYYY-MM-DD
    pump_chg_24h: float      # 起飞当日涨幅%
    pump_chg_7d: float       # 起飞前7日涨幅%
    price_shape: List[float] # 14日收盘归一化
    vol_ratio_shape: List[float]  # 14日量比
    signatures: dict         # 行为签名
    entry_price: float       # 指纹窗口末价（起飞前一日收盘）
    pre_low: float           # 窗口最低
    pre_high: float          # 窗口最高
    notes: str = ""


def _corr(a: List[float], b: List[float]) -> float:
    """皮尔逊相关，长度相同；方差为0返回0。"""
    n = len(a)
    if n != len(b) or n < 3:
        return 0.0
    ma = statistics.mean(a); mb = statistics.mean(b)
    da = [x - ma for x in a]; db = [x - mb for x in b]
    num = sum(x * y for x, y in zip(da, db))
    den = math.sqrt(sum(x * x for x in da) * sum(y * y for y in db))
    return num / den if den > 0 else 0.0


def extract_signatures(daily_bars: List[dict], window: int = 14) -> dict:
    """从14个日K(升序, 每个含 o,h,l,c,v,ts)提取行为签名。"""
    if len(daily_bars) < window:
        return {}
    bars = daily_bars[-window:]
    closes = [b["c"] for b in bars]
    vols = [b["v"] for b in bars]
    med_vol = statistics.median(vols) if vols else 0
    # 试盘脉冲: 量>2x中位 + 上影>5% + |日涨跌|<3%
    test_pulse = 0
    for b in bars:
        o, h, l, c = b["o"], b["h"], b["l"], b["c"]
        vr = b["v"] / med_vol if med_vol > 0 else 0
        upper = (h - max(o, c)) / o * 100 if o > 0 else 0
        chg = (c / o - 1) * 100 if o > 0 else 0
        if vr > 2 and upper > 5 and -3 < chg < 3:
            test_pulse += 1
    # 洗盘: 24h跌幅<-8% (用日chg近似)
    washouts = sum(1 for b in bars if b["o"] > 0 and (b["c"] / b["o"] - 1) < -0.08)
    # 波动率收敛: 近3日振幅 / 前7日振幅
    amp3 = (max(b["h"] for b in bars[-3:]) - min(b["l"] for b in bars[-3:])) / statistics.mean(b["c"] for b in bars[-3:]) * 100 if len(bars) >= 3 else 0
    amp7 = (max(b["h"] for b in bars[-7:]) - min(b["l"] for b in bars[-7:])) / statistics.mean(b["c"] for b in bars[-7:]) * 100 if len(bars) >= 7 else 0
    converge_ratio = amp3 / amp7 if amp7 > 0 else 1.0
    # 止跌: 末段某日跌幅收窄(>-1.5%)且前几日有过急跌
    standstill = False
    for i in range(max(1, len(bars) - 3), len(bars)):
        chg = (bars[i]["c"] / bars[i]["o"] - 1) * 100 if bars[i]["o"] > 0 else 0
        if -1.5 < chg < 1:
            prior_drop = any((bars[j]["c"] / bars[j]["o"] - 1) < -0.05 for j in range(max(0, i - 3), i) if bars[j]["o"] > 0)
            if prior_drop:
                standstill = True
    # 探底长下影: 末日量>2x + 下影>2% + 跌幅<6% + 创近7日新低
    last = bars[-1]
    lo7 = min(b["l"] for b in bars[-7:]) if len(bars) >= 7 else min(b["l"] for b in bars)
    vr_last = last["v"] / med_vol if med_vol > 0 else 0
    lower = (min(last["o"], last["c"]) - last["l"]) / last["o"] * 100 if last["o"] > 0 else 0
    chg_last = (last["c"] / last["o"] - 1) * 100 if last["o"] > 0 else 0
    bottom_probe = vr_last > 2 and lower > 2 and chg_last > -6 and last["l"] <= lo7 * 1.005
    # 低位: 末日收盘在窗口[低,高]区间的位置
    lo = min(closes); hi = max(closes)
    pos = (closes[-1] - lo) / (hi - lo) if hi > lo else 0.5
    return {
        "test_pulse": test_pulse,
        "washouts": washouts,
        "converge_ratio": round(converge_ratio, 3),
        "converge": converge_ratio < 0.7,
        "standstill": standstill,
        "bottom_probe": bottom_probe,
        "pos": round(pos, 3),
        "amp3": round(amp3, 2),
        "amp7": round(amp7, 2),
    }


def classify_mode(sig: dict, pump_chg_24h: float) -> str:
    """根据签名+涨幅分类庄家模式。C(超跌反弹)为假模式标记但不入库。"""
    if pump_chg_24h < 30:
        return "C"
    has_conv = sig.get("converge", False)
    has_wash = sig.get("washouts", 0) >= 1
    has_pulse = sig.get("test_pulse", 0) >= 1
    has_probe = sig.get("bottom_probe", False)
    # D 收敛爆发: 收敛 + 洗盘 + 探底
    if has_conv and has_wash:
        return "D"
    # B 洗盘收敛: 洗盘 + (脉冲或探底), 收敛较弱也算
    if has_wash and (has_pulse or has_probe):
        return "B"
    # A 低点吸筹: 脉冲 + 低位, 无收敛
    if has_pulse and sig.get("pos", 1) < 0.5:
        return "A"
    # C 超跌反弹: 无明显庄家签名
    return "C"


def extract_fingerprint(spot_symbol: str, fut_symbol: str, daily_bars: List[dict], pump_idx: int) -> Optional[Fingerprint]:
    """从拉盘事件提取指纹。daily_bars: 升序日K列表(含o,h,l,c,v,ts), pump_idx为拉盘日索引。
    指纹窗口 = pump_idx 前14日 (不含拉盘日)。"""
    if pump_idx < 14 or pump_idx >= len(daily_bars):
        return None
    window_bars = daily_bars[pump_idx - 14:pump_idx]
    closes = [b["c"] for b in window_bars]
    vols = [b["v"] for b in window_bars]
    med_vol = statistics.median(vols) if vols else 0
    lo = min(closes); hi = max(closes)
    price_shape = [round((c - lo) / (hi - lo), 4) if hi > lo else 0.5 for c in closes]
    vol_ratio_shape = [round(v / med_vol, 3) if med_vol > 0 else 0 for v in vols]
    sig = extract_signatures(daily_bars[:pump_idx], 14)
    pump_bar = daily_bars[pump_idx]
    pump_chg_24h = (pump_bar["c"] / pump_bar["o"] - 1) * 100 if pump_bar["o"] > 0 else 0
    pre7 = daily_bars[max(0, pump_idx - 7):pump_idx]
    pump_chg_7d = (closes[-1] / pre7[0]["c"] - 1) * 100 if pre7 and pre7[0]["c"] > 0 else 0
    mode = classify_mode(sig, pump_chg_24h)
    pump_date = datetime.fromtimestamp(pump_bar["ts"] / 1000, tz=CST).strftime("%Y-%m-%d")
    fp_id = f"{fut_symbol.split('USDT')[0]}_{pump_date.replace('-','')}"
    return Fingerprint(
        fp_id=fp_id, mode=mode, source_coin=fut_symbol, source_spot=spot_symbol,
        pump_date=pump_date, pump_chg_24h=round(pump_chg_24h, 2), pump_chg_7d=round(pump_chg_7d, 2),
        price_shape=price_shape, vol_ratio_shape=vol_ratio_shape, signatures=sig,
        entry_price=closes[-1], pre_low=lo, pre_high=hi,
    )


def similarity(coin_price_shape: List[float], coin_vol_shape: List[float],
               coin_sig: dict, fp: Fingerprint) -> float:
    """计算币窗口与指纹的相似度 ∈ [-1, 1]。价格形状相关性(0.4)+量比形状相关性(0.3)+签名匹配(0.3)。"""
    pcorr = _corr(coin_price_shape, fp.price_shape)
    vcorr = _corr(coin_vol_shape, fp.vol_ratio_shape)
    # 签名匹配: 关键签名是否一致
    keys = ["test_pulse", "washouts", "converge", "standstill", "bottom_probe"]
    match = 0; total = 0
    for k in keys:
        cv = coin_sig.get(k); fv = fp.signatures.get(k)
        if k in ("test_pulse", "washouts"):
            # 数量型: 都>0 或都==0
            if (cv or 0) > 0 and (fv or 0) > 0: match += 1
            elif (cv or 0) == 0 and (fv or 0) == 0: match += 1
        else:
            if bool(cv) == bool(fv): match += 1
        total += 1
    sig_match = match / total if total else 0
    return 0.4 * pcorr + 0.3 * vcorr + 0.3 * sig_match


def match_window(daily_bars: List[dict], library: List[Fingerprint],
                 window: int = 14, threshold: float = 0.55) -> List[dict]:
    """对一段日K的末日窗口做指纹匹配，返回命中列表。
    每个命中: {fp_id, mode, similarity, source, signatures, end_idx, entry_price}"""
    if len(daily_bars) < window:
        return []
    window_bars = daily_bars[-window:]
    closes = [b["c"] for b in window_bars]
    vols = [b["v"] for b in window_bars]
    med_vol = statistics.median(vols) if vols else 0
    lo = min(closes); hi = max(closes)
    pshape = [(c - lo) / (hi - lo) if hi > lo else 0.5 for c in closes]
    vshape = [v / med_vol if med_vol > 0 else 0 for v in vols]
    sig = extract_signatures(daily_bars, window)
    matches = []
    for fp in library:
        sim = similarity(pshape, vshape, sig, fp)
        if sim >= threshold:
            matches.append({
                "fp_id": fp.fp_id, "mode": fp.mode, "similarity": round(sim, 3),
                "source": fp.source_coin, "source_pump_date": fp.pump_date,
                "source_pump_chg": fp.pump_chg_24h, "signatures": sig,
                "entry_price": closes[-1], "end_date": datetime.fromtimestamp(
                    window_bars[-1]["ts"] / 1000, tz=CST).strftime("%Y-%m-%d"),
            })
    matches.sort(key=lambda x: -x["similarity"])
    return matches


# ---------- 库管理 ----------
def load_library(path: str = LIB_PATH) -> List[Fingerprint]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [Fingerprint(**d) for d in data.get("fingerprints", [])]


def save_library(fps: List[Fingerprint], path: str = LIB_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {"version": 1, "count": len(fps), "fingerprints": [asdict(f) for f in fps]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def library_summary(fps: List[Fingerprint]) -> str:
    if not fps:
        return "(空库)"
    from collections import Counter
    modes = Counter(f.mode for f in fps)
    lines = [f"指纹库: {len(fps)} 条", "模式分布: " + ", ".join(f"{m}={c}" for m, c in sorted(modes.items()))]
    lines.append("来源拉盘事件:")
    for f in sorted(fps, key=lambda x: -x.pump_chg_24h):
        lines.append(f"  [{f.mode}] {f.fp_id}  起飞{f.pump_chg_24h:+.1f}%  脉冲{f.signatures.get('test_pulse',0)} 洗盘{f.signatures.get('washouts',0)} 收敛{f.signatures.get('converge_ratio',1)} 探底{f.signatures.get('bottom_probe')}")
    return "\n".join(lines)


if __name__ == "__main__":
    fps = load_library()
    print(library_summary(fps))