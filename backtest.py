"""回测引擎 — 现货领先信号历史回测（无 look-ahead，退市币纳入）。

定位：快速过滤无效信号逻辑，通过者再进实盘确认。
"""
import requests, time, statistics, json, random
from datetime import datetime, timezone, timedelta
from collections import defaultdict

SPOT, FUT = "https://api.binance.com", "https://fapi.binance.com"
sess = requests.Session()
UTC = timezone.utc

def get(base, path, **kw):
    for _ in range(3):
        try:
            r = sess.get(base + path, timeout=20, **kw)
            if r.status_code == 200: return r.json()
            if r.status_code == 429: time.sleep(10)
            else: return None
        except Exception: time.sleep(2)
    return None

# ---------- symbol 映射 ----------
def build_map():
    si = get(SPOT, "/api/v3/exchangeInfo"); fi = get(FUT, "/fapi/v1/exchangeInfo")
    spot = {s["symbol"] for s in si["symbols"] if s["quoteAsset"]=="USDT" and s["status"]=="TRADING"}
    fut = [s["symbol"] for s in fi["symbols"] if s["contractType"]=="PERPETUAL" and s["quoteAsset"]=="USDT" and s["status"]=="TRADING"]
    def m(fs):
        if fs.startswith("1000"):
            c = fs[3:]; return c if c in spot else None
        return fs if fs in spot else None
    return {fs: m(fs) for fs in fut}

# ---------- 拉历史K线（翻页）----------
def klines(base, path_base, symbol, total_hours):
    """拉 total_hours 根 1h K线，翻页。返回 [(t_ms, o, h, l, c, v), ...] 升序。"""
    out = []
    end = None
    need = total_hours
    while need > 0:
        lim = min(1000, need)
        p = f"{path_base}?symbol={symbol}&interval=1h&limit={lim}"
        if end: p += f"&endTime={end-1}"
        k = get(base, p)
        if not k: break
        out = k + out
        end = k[0][0]
        need -= len(k)
        if len(k) < lim: break
        time.sleep(0.12)
    return out

def funding_hist(symbol, days):
    """资金费历史，返回 [(t_ms, rate)]，取 days 天。"""
    fr = get(FUT, f"/fapi/v1/fundingRate?symbol={symbol}&limit=1000")
    if not fr: return []
    cutoff = (datetime.now(UTC) - timedelta(days=days+5)).timestamp()*1000
    return [(f["fundingTime"], float(f["fundingRate"])) for f in fr if f["fundingTime"] >= cutoff]

# ---------- 信号检测 + PnL ----------
def run_signal1(spot_k, fut_k, fr_hist):
    """信号① 横盘暗筹：振幅小+量温和放大+资金费中性偏负（不依赖OI，OI缓增留实盘确认）。"""
    if len(spot_k) < 200 or len(fut_k) < 200: return []
    smap = {k[0]: k for k in spot_k}
    aligned = [k for k in fut_k if k[0] in smap]
    if len(aligned) < 200: return []
    spot_c = [float(smap[k[0]][4]) for k in aligned]
    spot_h = [float(smap[k[0]][2]) for k in aligned]
    spot_l = [float(smap[k[0]][3]) for k in aligned]
    spot_v = [float(smap[k[0]][5]) for k in aligned]
    fut_c  = [float(k[4]) for k in aligned]
    ts = [k[0] for k in aligned]; n = len(aligned)
    fr_idx = 0; sigs = []; last_trigger = -999
    for t in range(168, n-121):
        if t - last_trigger < 120: continue
        while fr_idx < len(fr_hist)-1 and fr_hist[fr_idx+1][0] <= ts[t]: fr_idx += 1
        frate = fr_hist[fr_idx][1]*100 if fr_idx < len(fr_hist) else 0
        vol_ratio = statistics.mean(spot_v[t-24:t]) / statistics.mean(spot_v[t-168:t-24]) if statistics.mean(spot_v[t-168:t-24])>0 else 0
        amp = (max(spot_h[t-168:t]) - min(spot_l[t-168:t])) / statistics.mean(spot_c[t-168:t]) * 100
        chg5d = (spot_c[t]/spot_c[t-168]-1)*100
        # 信号①：横盘(振幅<8) + 不涨(5日<3%) + 量温和放大(>1.1) + 资金费中性偏负(<=0.01)
        if amp < 8 and chg5d < 3 and vol_ratio > 1.1 and frate <= 0.01 and chg5d > -8:
            entry = spot_c[t+1]
            future = spot_c[t+1:t+121]
            fh = spot_h[t+1:t+121]; fl = spot_l[t+1:t+121]
            sigs.append((ts[t], entry, (future[-1]/entry-1)*100, (max(fh)/entry-1)*100, (min(fl)/entry-1)*100))
            last_trigger = t
    return sigs

def run_signal2(spot_k, fut_k, fr_hist):
    """信号② 现货异动压价。spot_k/fut_k: 升序 [(t_ms,o,h,l,c,v)]。
    返回 [(trigger_t, entry, pnl5d, max_high, max_low)]。
    """
    if len(spot_k) < 200 or len(fut_k) < 200: return []
    # 对齐时间
    smap = {k[0]: k for k in spot_k}
    aligned = [k for k in fut_k if k[0] in smap]
    if len(aligned) < 200: return []
    spot_c = [float(smap[k[0]][4]) for k in aligned]
    spot_h = [float(smap[k[0]][2]) for k in aligned]
    spot_l = [float(smap[k[0]][3]) for k in aligned]
    spot_v = [float(smap[k[0]][5]) for k in aligned]
    fut_c  = [float(k[4]) for k in aligned]
    ts     = [k[0] for k in aligned]
    n = len(aligned)
    fr_idx = 0
    sigs = []
    last_trigger = -999
    for t in range(168, n-121):  # 留5天(120h)给PnL
        if t - last_trigger < 120: continue  # 同币冷却5天
        # 资金费（最近的 <= t 时刻）
        while fr_idx < len(fr_hist)-1 and fr_hist[fr_idx+1][0] <= ts[t]: fr_idx += 1
        frate = fr_hist[fr_idx][1]*100 if fr_idx < len(fr_hist) else 0
        vol_ratio = statistics.mean(spot_v[t-24:t]) / statistics.mean(spot_v[t-168:t-24]) if statistics.mean(spot_v[t-168:t-24])>0 else 0
        chg5d = (spot_c[t]/spot_c[t-168]-1)*100
        chg24 = (spot_c[t]/spot_c[t-24]-1)*100
        prem = (spot_c[t]-fut_c[t])/fut_c[t]*100 if fut_c[t]>0 else 0
        # 信号②：量比>1.3 + 5日涨幅[-5,3] + 溢价>0 + 资金费<0
        if vol_ratio > 1.3 and -5 <= chg5d <= 3 and prem > 0 and frate < 0 and chg24 < 3:
            entry = spot_c[t+1]  # t+1h 进场
            future = spot_c[t+1:t+121]
            fh = spot_h[t+1:t+121]; fl = spot_l[t+1:t+121]
            pnl5d = (future[-1]/entry - 1)*100
            max_high = (max(fh)/entry - 1)*100
            max_low = (min(fl)/entry - 1)*100
            sigs.append((ts[t], entry, pnl5d, max_high, max_low))
            last_trigger = t
    return sigs

def benchmark(spot_k):
    """基准：同币随机时刻等量5天PnL。"""
    if len(spot_k) < 289: return []
    c = [float(k[4]) for k in spot_k]
    h = [float(k[2]) for k in spot_k]
    l = [float(k[3]) for k in spot_k]
    n = len(spot_k)
    rng = random.Random(42)
    idxs = rng.sample(range(168, n-121), min(30, n-121-168))
    out = []
    for t in idxs:
        entry = c[t+1]
        out.append((c[t+120]/entry-1)*100)
    return out

def run_signal_v2(spot_k, fut_k, fr_hist):
    """信号v2 庄家行为序列：量价背离脉冲(吸筹)+洗盘V型收回(强庄)+低位(未大涨)。
    基于ERA/ESPORTS/RE/BANK四币实证设计。返回 [(trigger_t, entry, pnl5d, max_high, max_low, score)]。
    """
    if len(spot_k) < 400 or len(fut_k) < 400: return []
    smap = {k[0]: k for k in spot_k}
    aligned = [k for k in fut_k if k[0] in smap]
    if len(aligned) < 400: return []
    spot_c = [float(smap[k[0]][4]) for k in aligned]
    spot_h = [float(smap[k[0]][2]) for k in aligned]
    spot_l = [float(smap[k[0]][3]) for k in aligned]
    spot_v = [float(smap[k[0]][5]) for k in aligned]
    ts = [k[0] for k in aligned]; n = len(aligned)
    fr_idx = 0; sigs = []; last_trigger = -999
    for t in range(336, n-121):
        if t - last_trigger < 120: continue  # 同币冷却5天
        while fr_idx < len(fr_hist)-1 and fr_hist[fr_idx+1][0] <= ts[t]: fr_idx += 1
        frate = fr_hist[fr_idx][1]*100 if fr_idx < len(fr_hist) else 0
        W = 336
        c = spot_c[t-W+1:t+1]; h = spot_h[t-W+1:t+1]; l = spot_l[t-W+1:t+1]; v = spot_v[t-W+1:t+1]
        # 按日聚合量价背离脉冲
        days = {}
        for j in range(len(c)):
            import datetime as _dt
            d = _dt.datetime.fromtimestamp(ts[t-W+1+j]/1000, tz=_dt.timezone(_dt.timedelta(hours=8))).date()
            days.setdefault(d, []).append(j)
        dvs = []; dcs = []
        for d, idxs in sorted(days.items()):
            dvs.append(sum(v[j] for j in idxs))
            dcs.append(c[idxs[-1]]/c[idxs[0]]-1)
        med_vol = statistics.median(dvs) if dvs else 0
        pulse_n = sum(1 for dv, dc in zip(dvs, dcs) if med_vol>0 and dv > med_vol*2 and -0.05 < dc < 0.03)
        # 洗盘V型收回
        wash_n = 0; wash_rec = 0
        for j in range(24, t+1):
            chg24 = spot_c[j]/spot_c[j-24]-1
            if chg24 < -0.08:
                wash_n += 1
                if any(spot_c[k] >= spot_c[j-24]*0.96 for k in range(j+1, min(j+72, t+1))):
                    wash_rec += 1
        wash_rate = wash_rec/wash_n if wash_n>0 else 0
        # 低位
        h14 = max(spot_h[t-W+1:t+1]); l14 = min(spot_l[t-W+1:t+1])
        pos = (spot_c[t]-l14)/(h14-l14) if h14>l14 else 0
        # 波动率收敛
        amp3 = (max(spot_h[t-71:t+1])-min(spot_l[t-71:t+1]))/statistics.mean(spot_c[t-71:t+1])*100 if statistics.mean(spot_c[t-71:t+1])>0 else 0
        amp7 = (max(spot_h[t-167:t+1])-min(spot_l[t-167:t+1]))/statistics.mean(spot_c[t-167:t+1])*100 if statistics.mean(spot_c[t-167:t+1])>0 else 0
        converge = amp3/amp7 if amp7>0 else 1
        chg24 = (spot_c[t]/spot_c[t-24]-1)*100
        chg5d = (spot_c[t]/spot_c[t-120]-1)*100 if t>=120 else 0
        # 门槛
        if pulse_n < 2: continue
        if wash_n < 1: continue
        if wash_rate < 0.5: continue
        if pos >= 0.3: continue
        if frate > 0.02: continue
        if chg24 >= 8: continue
        if chg5d >= 10: continue
        if chg5d <= -15: continue
        # 评分
        sc = 0
        sc += 30 if wash_rate>=0.8 else 20 if wash_rate>=0.6 else 10
        sc += 25 if pulse_n>=3 else 18 if pulse_n>=2 else 10
        sc += 20 if pos<0.2 else 13 if pos<0.3 else 7
        sc += 15 if converge<0.5 else 8 if converge<0.7 else 0
        sc += 10 if frate<-0.03 else 6 if frate<-0.01 else 3
        entry = spot_c[t+1]
        future = spot_c[t+1:t+121]
        fh = spot_h[t+1:t+121]; fl = spot_l[t+1:t+121]
        sigs.append((ts[t], entry, (future[-1]/entry-1)*100, (max(fh)/entry-1)*100, (min(fl)/entry-1)*100, sc))
        last_trigger = t
    return sigs

if __name__ == "__main__":
    import sys
    N = int(sys.argv[1]) if len(sys.argv)>1 else 40
    mode = sys.argv[2] if len(sys.argv)>2 else "v1v2"
    HOURS = 24*95
    print(f"=== 回测 {mode} | 样本={N}币 窗口≈95天 ===")
    m = build_map()
    cov = [fs for fs in m if m[fs]]
    print(f"可映射 {len(cov)} 现货，取前 {N}")
    syms = sorted(cov)[:N]
    s1=[]; s2=[]; sv=[]; bench=[]
    for i, fs in enumerate(syms):
        ss = m[fs]
        sk = klines(SPOT, "/api/v3/klines", ss, HOURS)
        fk = klines(FUT, "/fapi/v1/klines", fs, HOURS)
        fr = funding_hist(fs, 95)
        if not sk or not fk: continue
        if mode in ("v1v2","v1"): s1 += run_signal1(sk, fk, fr)
        if mode in ("v1v2","v2"): sv += run_signal_v2(sk, fk, fr)
        if mode == "v1v2": s2 += run_signal2(sk, fk, fr)
        bench += benchmark(sk)
        if (i+1) % 20 == 0: print(f"  {i+1}/{N}... v1{len(s1)} v2{len(sv)}", flush=True)
    if mode in ("v1v2","v1"):
        print(f"\n=== 信号①横盘暗筹v1 ===")
        if s1:
            pnls=[x[2] for x in s1]; mh=[x[3] for x in s1]; ml=[x[4] for x in s1]
            win=sum(1 for x in pnls if x>0)
            print(f"信号数:{len(pnls)} 5天PnL均值={statistics.mean(pnls):+.2f}% 中位={statistics.median(pnls):+.2f}% 胜率={win/len(pnls)*100:.1f}%")
    if mode in ("v1v2","v2"):
        print(f"\n=== 信号v2 庄家行为序列 ===")
        if sv:
            pnls=[x[2] for x in sv]; mh=[x[3] for x in sv]; ml=[x[4] for x in sv]
            win=sum(1 for x in pnls if x>0)
            print(f"信号数:{len(pnls)} 5天PnL均值={statistics.mean(pnls):+.2f}% 中位={statistics.median(pnls):+.2f}% 胜率={win/len(pnls)*100:.1f}%")
            print(f"  max_high中位={statistics.median(mh):+.2f}% max_low中位={statistics.median(ml):+.2f}%")
            # 高分信号子集
            hi=[x for x in sv if x[5]>=50]
            if hi:
                hp=[x[2] for x in hi]; hw=sum(1 for x in hp if x>0)
                print(f"  高分(≥50)子集: {len(hi)}信号 均值{statistics.mean(hp):+.2f}% 胜率{hw/len(hi)*100:.1f}%")
    print(f"\n=== 基准(随机) ===")
    if bench:
        win=sum(1 for x in bench if x>0)
        print(f"基准数:{len(bench)} 均值={statistics.mean(bench):+.2f}% 中位={statistics.median(bench):+.2f}% 胜率={win/len(bench)*100:.1f}%")
        if s1 and mode in ("v1v2","v1"): print(f"超额v1: {statistics.mean([x[2] for x in s1])-statistics.mean(bench):+.2f}%")
        if sv and mode in ("v1v2","v2"): print(f"超额v2: {statistics.mean([x[2] for x in sv])-statistics.mean(bench):+.2f}%")