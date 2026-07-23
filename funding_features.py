#!/usr/bin/env python3
"""阶段2 · funding 行为特征复核（全量 universe，PIT 对齐）。

对齐 quant-ml-rules：
  - PIT：每个窗口只用 funding_time < t_decision 的 funding 点（t_decision=t_avail+1日）。
    最后可用 funding = 决策日 D 的 16:00 结算（D+1 00:00 结算在决策时刻，保守排除）。
  - 14日 lookback（~42个8h点），前后半各21点。
  - 判别力只在 Select 段算；方向alpha = 向上skill>0.05 且 |方向性|>0.10；CI 排除0.5+稳定差<0.10。
  - 门 a：funding 行为特征 CI 排除0.5 + 稳定差<0.10 → 纳入建模；否则只用量价。
  - 门 b：量价10特征在全量数据下若不再显著 → 停。

特征（行为驱动，非静态均值）：
  fr_v_shape    = mean(后半) - mean(前半)      V型=空头挤压铺垫(后高前低)
  fr_min_early  = argmin 位置(0=最早,1=最晚)   前=洗盘蓄势(低→拉盘)
  fr_late_val   = mean(后7日)                  后期转正=空头平仓
  fr_neg_then_flat = (-mean前)×flatness         前负后平=蓄势
  fr_swing      = latest - min                  空头挤压势能
  fr_recovery   = mean(末3日) - min             空头平仓程度
  静态对照: fr_mean14/fr_latest/fr_max14/fr_cum14/fr_min14/fr_neg14(负占比)

产出：docs/funding行为复核.md + /tmp/feature_matrix_funding.pkl（含funding特征）
"""
import math
import numpy as np
import pandas as pd
import duckdb

DB = "data/market.duckdb"
MAT = "/tmp/feature_matrix.pkl"
OUT_MAT = "/tmp/feature_matrix_funding.pkl"
OUT_DOC = "docs/funding行为复核.md"
DAY_MS = 86400000
LOOKBACK_DAYS = 14
MIN_POINTS = 35      # ≈12天8h点，"完整覆盖"门槛（headline模型用）
MIN_POINTS_EVAL = 21  # ≥7天，判别力评估门槛（探索性覆盖更多）


def load_funding(con):
    """{symbol: (times_ms np.array, rates np.array)} 升序。"""
    df = con.execute("SELECT symbol, funding_time, funding_rate FROM funding_rate "
                     "ORDER BY symbol, funding_time").fetchdf()
    # datetime64[us] -> int64 微秒 -> //1000 = UTC ms (系统TZ=UTC, 与t_avail对齐)
    df["tms"] = df["funding_time"].astype("int64") // 1000
    out = {}
    for sym, g in df.groupby("symbol"):
        out[sym] = (g["tms"].to_numpy(), g["funding_rate"].to_numpy(float))
    return out


def compute_fr(times_ms, rates, t_decision_ms):
    """PIT: 用 funding_time < t_decision_ms 的点; 14日 lookback。返回特征dict or None。"""
    # PIT 切片: < t_decision (严格小于, 排除决策时刻那拍)
    mask = times_ms < t_decision_ms
    t = times_ms[mask]
    r = rates[mask]
    if len(t) < MIN_POINTS_EVAL:
        return None
    lo = t_decision_ms - LOOKBACK_DAYS * DAY_MS
    m = t >= lo
    t, r = t[m], r[m]
    if len(r) < MIN_POINTS_EVAL:
        return None
    n = len(r)
    half = n // 2
    first, second = r[:half], r[half:]
    mf = float(np.mean(first)) if len(first) else 0.0
    ms_ = float(np.mean(second)) if len(second) else 0.0
    mn = float(np.min(r))
    mx = float(np.max(r))
    latest = float(r[-1])
    argmin_pos = float(np.argmin(r)) / max(n - 1, 1)
    flatness = max(0.0, 1.0 - abs(ms_) / 0.0003)  # 后期均值|<0.03%|视为平
    f = {}
    f["fr_v_shape"] = ms_ - mf
    f["fr_min_early"] = argmin_pos
    f["fr_late_val"] = ms_
    f["fr_neg_then_flat"] = (-mf) * flatness if mf < 0 else 0.0
    f["fr_swing"] = latest - mn
    f["fr_recovery"] = float(np.mean(r[-9:])) - mn if n >= 9 else latest - mn  # 末3日≈9点
    # 静态对照
    f["fr_mean14"] = float(np.mean(r))
    f["fr_latest"] = latest
    f["fr_max14"] = mx
    f["fr_cum14"] = float(np.sum(r))
    f["fr_min14"] = mn
    f["fr_neg14"] = float(np.mean(r < 0))
    f["fr_n"] = n
    return f


def auc_score(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    ok = ~np.isnan(s); y, s = y[ok], s[ok]
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    r = pd.Series(s).rank().values
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def auc_ci(y, s, n_boot=300, seed=42):
    y = np.asarray(y, float); s = np.asarray(s, float)
    ok = ~np.isnan(s); y, s = y[ok], s[ok]
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 < 5 or n0 < 5:
        return 0.5, 0.5
    rng = np.random.default_rng(seed); idx = np.arange(len(y))
    aucs = [auc_score(y[idx2], s[idx2]) for idx2 in (rng.choice(idx, len(idx), replace=True) for _ in range(n_boot))]
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def stab(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    ok = ~np.isnan(s); y, s = y[ok], s[ok]
    mid = len(y) // 2
    if mid < 10:
        return 0.0
    return abs(auc_score(y[:mid], s[:mid]) - auc_score(y[mid:], s[mid:]))


FR_BEHAVIOR = ["fr_v_shape", "fr_min_early", "fr_late_val", "fr_neg_then_flat", "fr_swing", "fr_recovery"]
FR_STATIC = ["fr_mean14", "fr_latest", "fr_max14", "fr_cum14", "fr_min14", "fr_neg14"]
FR_ALL = FR_BEHAVIOR + FR_STATIC


def main():
    con = duckdb.connect(DB, read_only=True)
    fund = load_funding(con)
    con.close()
    df = pd.read_pickle(MAT)
    print(f"特征矩阵: {len(df)} 窗口, funding覆盖币: {len(fund)}")

    # 逐窗口算 funding 特征（按symbol分组，内存切片）
    res = {c: pd.Series(np.nan, index=df.index) for c in FR_ALL + ["fr_n"]}
    hit = 0
    for sym, g in df.groupby("coin"):
        if sym not in fund:
            continue
        tms, rts = fund[sym]
        for idx, ta in zip(g.index, g["t_avail"]):
            fr = compute_fr(tms, rts, int(ta) + DAY_MS)
            if fr is None:
                continue
            hit += 1
            for c in FR_ALL + ["fr_n"]:
                res[c].at[idx] = fr[c]
    for c in FR_ALL + ["fr_n"]:
        df[c] = res[c]
    print(f"funding特征可算窗口: {hit}/{len(df)} ({hit/len(df)*100:.1f}%)")

    # headline 完整覆盖门槛
    df["fr_complete"] = (df["fr_n"] >= MIN_POINTS).astype(int)
    df.to_pickle(OUT_MAT)
    print(f"已存 {OUT_MAT} ({len(df)}行, +{len(FR_ALL)+1}列)")

    # —— 判别力评估 (Select段) ——
    SEL = df[df["split"] == "select"].copy()
    sel_n = len(SEL)
    y_up = (SEL["net_swing"] > 0.15).astype(int).values
    y_dn = (SEL["net_swing"] < -0.15).astype(int).values
    n_up, n_dn = int(y_up.sum()), int(y_dn.sum())
    # funding覆盖(评估门槛)
    cov_eval = (~SEL["fr_v_shape"].isna()).sum()
    cov_complete = int(SEL["fr_complete"].sum())

    L = []
    L.append("# 阶段2 · funding 行为特征复核（全量 universe，PIT 对齐）\n")
    L.append("> **PIT**：每窗口只用 `funding_time < t_decision`（t_decision=t_avail+1日），最后可用=决策日16:00结算。")
    L.append("> **14日 lookback**（~42个8h点），前后半各21点。判别力只在 Select 段算。")
    L.append(f"> Select n={sel_n} | 净向上拉盘(>15%)={n_up} | 净向下砸盘(<-15%)={n_dn}")
    L.append(f"> funding可算窗口(≥{MIN_POINTS_EVAL}点≈7天): {cov_eval}/{sel_n} ({cov_eval/sel_n*100:.1f}%) | 完整覆盖(≥{MIN_POINTS}点≈12天, headline门槛): {cov_complete}/{sel_n} ({cov_complete/sel_n*100:.1f}%)\n")

    L.append("## 行为特征（6个，方向假设来自机制推演）\n")
    L.append("| 特征 | 含义 | 假设方向 |")
    L.append("|------|------|----------|")
    L.append("| fr_v_shape | 后期均值−前期均值 | V型(后高前低)=空头挤压铺垫→拉盘(+) |")
    L.append("| fr_min_early | 最低费率位置(0早/1晚) | 最低在前=洗盘蓄势→拉盘(−) |")
    L.append("| fr_late_val | 后7日均值 | 转正=空头平仓→拉盘(+) |")
    L.append("| fr_neg_then_flat | 前负后平 | 蓄势→拉盘(+) |")
    L.append("| fr_swing | latest−min | 空头挤压势能→拉盘(+) |")
    L.append("| fr_recovery | 末3日−min | 空头平仓程度→拉盘(+) |")
    L.append("")

    def eval_feat(name):
        s = SEL[name].values
        a_up = auc_score(y_up, s); a_dn = auc_score(y_dn, s)
        su, sd = abs(a_up - 0.5), abs(a_dn - 0.5)
        diff = a_up - a_dn
        lo, hi = auc_ci(y_up, s)
        sdd = stab(y_up, s)
        sig = lo > 0.5 or hi < 0.5
        typ = "方向alpha" if su > 0.05 and abs(diff) > 0.10 else ("双向噪音" if su > 0.05 and abs(diff) <= 0.10 else "无")
        pass_a = typ == "方向alpha" and sdd < 0.10 and sig
        return dict(feat=name, a_up=a_up, a_dn=a_dn, su=su, sd=sd, diff=diff, lo=lo, hi=hi, sdd=sdd, sig=sig, typ=typ, pass_a=pass_a)

    L.append("## 判别力（Select段，θ=15%）\n")
    L.append("| 特征 | 向上AUC | 向下AUC | 上技能 | 方向性 | CI下 | CI上 | 稳定差 | CI排除0.5 | 类型 | 门a |")
    L.append("|------|---------|---------|--------|--------|------|------|--------|-----------|------|-----|")
    results = []
    for f in FR_ALL:
        r = eval_feat(f)
        results.append(r)
        mark = "✅" if r["pass_a"] else ("⚠️" if r["typ"] == "方向alpha" else "")
        L.append(f"| {f} | {r['a_up']:.3f}{mark} | {r['a_dn']:.3f} | {r['su']:.3f} | {r['diff']:+.3f} | {r['lo']:.3f} | {r['hi']:.3f} | {r['sdd']:.3f} | {'是' if r['sig'] else '否'} | {r['typ']} | {'PASS' if r['pass_a'] else '—'} |")
    L.append("")

    pass_behavior = [r for r in results if r["feat"] in FR_BEHAVIOR and r["pass_a"]]
    pass_static = [r for r in results if r["feat"] in FR_STATIC and r["pass_a"]]
    L.append("## 门 a 判定\n")
    L.append(f"- 行为特征过门(CI排除0.5+稳定差<0.10+方向alpha): **{len(pass_behavior)}/6**")
    L.append(f"- 静态特征过门: **{len(pass_static)}/6**")
    L.append(f"- 过门行为特征: {[r['feat'] for r in pass_behavior] or '无'}")
    if pass_behavior:
        L.append(f"\n**门 a 结论：PASS** — {len(pass_behavior)} 个 funding 行为特征统计显著方向alpha → **纳入建模**（量价+funding两层）。")
        included = [r["feat"] for r in pass_behavior]
    else:
        # 退化：看是否有边缘(方向alpha但CI跨0.5或稳定差大)
        edge = [r for r in results if r["feat"] in FR_BEHAVIOR and r["typ"] == "方向alpha"]
        L.append(f"\n边缘(方向alpha但CI跨0.5或稳定差大): {[r['feat'] for r in edge] or '无'}")
        if edge:
            L.append(f"**门 a 结论：边缘** — {len(edge)} 个方向alpha但不够稳健 → 仅纳入最稳的作辅助，主力量价。")
            included = [r["feat"] for r in edge]
        else:
            L.append(f"**门 a 结论：FAIL** — funding 行为特征无统计显著方向alpha → **只用量价10特征建模**。")
            included = []
    L.append("")

    # 与量价10特征对比（门b复核）
    PV10 = ["amplitude_14d", "atr_14_pct", "bb_width", "washouts_count", "peak_valley_ratio",
            "max_drawdown", "ret_var", "atr_7_pct", "atr_5_pct"]
    L.append("## 门 b 复核（量价方向alpha在全量数据下是否仍显著）\n")
    L.append("| 特征 | 向上AUC | 方向性 | CI下 | CI上 | 稳定差 | CI排除0.5 | 门b |")
    L.append("|------|---------|--------|------|------|--------|-----------|-----|")
    pv_pass = 0
    for f in PV10:
        if f not in SEL.columns:
            continue
        r = eval_feat(f)
        pb = r["typ"] == "方向alpha" and r["sig"]
        if pb:
            pv_pass += 1
        L.append(f"| {f} | {r['a_up']:.3f} | {r['diff']:+.3f} | {r['lo']:.3f} | {r['hi']:.3f} | {r['sdd']:.3f} | {'是' if r['sig'] else '否'} | {'PASS' if pb else '—'} |")
    L.append(f"\n量价方向alpha过门: **{pv_pass}/{len(PV10)}**")
    L.append("")

    # 决策
    L.append("## 决策\n")
    if pv_pass >= 3:
        L.append(f"- 门 b PASS（{pv_pass}个量价方向alpha显著）→ **进入建模**。")
        if included:
            L.append(f"- 门 a PASS/边缘 → funding 特征 **{included}** 纳入建模特征池。")
            L.append(f"- 建模特征池 = 量价{pv_pass}个 + funding{len(included)}个。")
        else:
            L.append("- 门 a FAIL → 建模特征池 = 量价 only。")
        L.append("- 下一步：阶段3基线 → 阶段4 logistic → （若skill>0且非线性）LightGBM → 阶段5阈值 → 阶段6度量 → 阶段8经济回测 → 阶段9验证。")
    else:
        L.append(f"- 门 b FAIL（量价方向alpha<3个显著）→ **停，不建模**（按 skill 规矩）。")

    with open(OUT_DOC, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print(f"\n已生成 {OUT_DOC}")
    print(f"门a: 行为过门={len(pass_behavior)}/6, 静态过门={len(pass_static)}/6")
    print(f"门b: 量价过门={pv_pass}/{len(PV10)}")
    print("\n行为特征结果:")
    for r in results:
        if r["feat"] in FR_BEHAVIOR:
            print(f"  {r['feat']:18s} 上{r['a_up']:.3f} 下{r['a_dn']:.3f} 方向性{r['diff']:+.3f} 稳定{r['sdd']:.3f} {'PASS' if r['pass_a'] else r['typ']}")


if __name__ == "__main__":
    main()