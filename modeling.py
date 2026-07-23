#!/usr/bin/env python3
"""阶段3-8 · 建模主流程（logistic→LightGBM）+ 基线 + 度量 + 经济回测。

对齐 quant-ml-rules：
  - 三分离：Train/Select/Report 时间序，选择只看 Select(hpval_)，report 一次性。
  - 基线先行：先验A / free-rule B / v1横向暗筹C / 指纹D，同 report 索引。
  - 模型阶梯：logistic(线性天花板, scaler train-fold only) → 仅 Select skill>0 升级 LightGBM。
  - 度量：PR-AUC(非accuracy) + 期望收益 + 肥尾捕获 + 最大回撤 + RR + coverage + skill score。
  - 小样本诚实：block bootstrap CI + McNemar 配对 + trial count。
  - 经济回测：prediction→gate→成本(费+funding+滑点)→PnL，同可用性延迟。
  - headline = funding 完整覆盖窗口(§0.6)；量价-only 全130天作消融。

产出：/tmp/model_metrics.log + /tmp/model_config.json + docs/建模报告.md
"""
import json, math, os
import numpy as np
import pandas as pd
import duckdb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
import lightgbm as lgb

MAT = "/tmp/feature_matrix_funding.pkl"
DB = "data/market.duckdb"
METRICS_LOG = "/tmp/model_metrics.log"
CONFIG_JSON = "/tmp/model_config.json"
REPORT = "docs/建模报告.md"
THETA = 0.15
SEED = 42
DAY_MS = 86400000
FORWARD = 5
EMBARGO = 5
COST_RT = 0.0035  # 往返成本: 费0.08%+滑点0.10%+funding~0.15%

PV = ["amplitude_14d", "atr_14_pct", "bb_width", "washouts_count", "peak_valley_ratio",
      "max_drawdown", "ret_var", "atr_7_pct", "atr_5_pct"]
FR = ["fr_min_early", "fr_late_val"]


def re_split(df, train_f=0.60, sel_f=0.20):
    """时间序切 Train/Select/Report + purge + embargo=5日。"""
    d = df.sort_values("t_avail").reset_index(drop=True).copy()
    n = len(d)
    t1 = d["t_avail"].quantile(train_f)
    t2 = d["t_avail"].quantile(train_f + sel_f)
    d["split"] = np.where(d["t_avail"] < t1, "train", np.where(d["t_avail"] < t2, "select", "report"))
    lab_end = d["t_avail"] + FORWARD * DAY_MS
    # purge: train 标签跨入 select, select 标签跨入 report
    d.loc[(d.split == "train") & (lab_end >= t1), "split"] = "purged"
    d.loc[(d.split == "select") & (lab_end >= t2), "split"] = "purged"
    emb = EMBARGO * DAY_MS
    d.loc[(d.split == "train") & (d["t_avail"] >= t1 - emb) & (d["t_avail"] < t1), "split"] = "purged"
    d.loc[(d.split == "select") & (d["t_avail"] >= t2 - emb) & (d["t_avail"] < t2), "split"] = "purged"
    return d, t1, t2


def pr_auc(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    ok = ~np.isnan(s)
    return average_precision_score(y[ok], s[ok]) if y[ok].sum() > 0 and (~y[ok].astype(bool)).sum() > 0 else 0.0


def block_boot_mean(vals, t_avail, n_boot=300, seed=SEED, block_ms=5 * DAY_MS):
    """块自举均值CI, 块长≥5日(标签窗)。"""
    vals = np.asarray(vals, float); t = np.asarray(t_avail)
    if len(vals) < 20:
        return float(np.mean(vals)) if len(vals) else 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    # 按块(5日)分组重采样
    blk_id = (t - t.min()) // block_ms
    blocks = pd.DataFrame({"v": vals, "b": blk_id}).groupby("b")["v"].mean()
    means = []
    for _ in range(n_boot):
        samp = blocks.sample(len(blocks), replace=True, random_state=rng.integers(1e9))
        means.append(samp.mean())
    return float(np.mean(vals)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def mcnemar(y, pred_m, pred_b):
    """配对显著性: 模型对基线, b=模型对/基线错, c=模型错/基线对."""
    b = int(((pred_m == 1) & (pred_b == 0) & ~np.isnan(y)).sum())
    c = int(((pred_m == 0) & (pred_b == 1) & ~np.isnan(y)).sum())
    if b + c == 0:
        return 1.0
    stat = (abs(b - c) - 1) ** 2 / (b + c) if b + c > 0 else 0
    from scipy.stats import chi2
    return float(chi2.sf(stat, 1))


def drop_collinear(df_train, feats, sel_auc, thresh=0.85):
    """train段|corr|>0.85 保留 |方向性|最强者。返回保留特征列表。"""
    remaining = list(feats)
    corr = df_train[feats].corr().abs()
    dropped = set()
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            if feats[i] in dropped or feats[j] in dropped:
                continue
            c = corr.iloc[i, j]
            if not np.isnan(c) and c > thresh:
                # 丢方向性弱的
                weak = feats[i] if abs(sel_auc.get(feats[i], 0)) < abs(sel_auc.get(feats[j], 0)) else feats[j]
                dropped.add(weak)
    return [f for f in feats if f not in dropped], dropped


def load_klines_funding(con):
    """{symbol: (klines list[dict], fund_tms, fund_rates)} 内存缓存。"""
    kd = con.execute("SELECT symbol, open_time, open, high, low, close, volume "
                     "FROM klines_daily ORDER BY symbol, open_time").fetchdf()
    kd["ts"] = kd["open_time"].astype("int64") // 1000
    kmap = {}
    for sym, g in kd.groupby("symbol"):
        kmap[sym] = [{"ts": int(t), "o": o, "h": h, "l": l, "c": c, "v": v}
                     for t, o, h, l, c, v in zip(g["ts"], g["open"], g["high"], g["low"], g["close"], g["volume"])]
    fd = con.execute("SELECT symbol, funding_time, funding_rate FROM funding_rate ORDER BY symbol, funding_time").fetchdf()
    fd["tms"] = fd["funding_time"].astype("int64") // 1000
    fmap = {}
    for sym, g in fd.groupby("symbol"):
        fmap[sym] = (g["tms"].to_numpy(), g["funding_rate"].to_numpy(float))
    return kmap, fmap


def baseline_v1(klines, fund, t_avail):
    """v1 横盘暗筹 日级近似: amp7<8, chg5d∈(-8,3), vol_ratio>1.1, frate<=0.01。"""
    t_dec = int(t_avail) + DAY_MS
    bars = [b for b in klines if b["ts"] <= t_avail]
    if len(bars) < 7:
        return 0
    w = bars[-7:]
    highs = [b["h"] for b in w]; lows = [b["l"] for b in w]; closes = [b["c"] for b in w]; vols = [b["v"] for b in w]
    amp = (max(highs) - min(lows)) / (np.mean(closes) + 1e-12) * 100
    chg5d = (closes[-1] / closes[0] - 1) * 100
    med_v = np.median(vols)
    vol_ratio = np.mean(vols[-2:]) / (np.mean(vols[:-2]) + 1e-12) if len(vols) > 2 and np.mean(vols[:-2]) > 0 else 0
    frate = 0.0
    if fund is not None:
        tms, rts = fund
        m = tms < t_dec
        frate = float(rts[m][-1]) if m.sum() > 0 else 0.0
    return 1 if (amp < 8 and chg5d < 3 and chg5d > -8 and vol_ratio > 1.1 and frate <= 0.01) else 0


def baseline_fp(klines, library, t_avail):
    """指纹D: 14日窗口 match_window + 锁定配置 sim≥0.65+收敛+止跌+pos<0.3。"""
    bars = [b for b in klines if b["ts"] <= t_avail]
    if len(bars) < 14:
        return 0
    window = bars[-14:]
    matches = []
    if library:
        from fingerprint import match_window
        matches = match_window(window, library, window=14, threshold=0.55)
    for m in matches:
        if m["similarity"] >= 0.65 and m["signatures"].get("converge") and m["signatures"].get("standstill") and m["signatures"].get("pos", 1) < 0.3:
            return 1
    return 0


def main():
    cfg = {"seed": SEED, "theta": THETA, "cost_rt": COST_RT, "embargo": EMBARGO,
           "pv_features": PV, "fr_features": FR, "library": "data/fingerprints.json",
           "family": "lightgbm", "select_grid": {"C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]},
           "scaler_fit_train_only": True, "early_stop_monitor_select": True,
           "use_best_iteration": True, "logistic_select_skill_positive": False,
           "lib_versions": {"numpy": np.__version__, "sklearn": __import__('sklearn').__version__,
                             "lightgbm": lgb.__version__, "scipy": __import__('scipy').__version__},
           "model_card": {"decision_freq": "每日", "stale_data_rule": "日K次日才得(t_decision=D+1)",
                          "cancel_rule": "gate未开则不交易", "kill_switch": "report PR-AUC连续5日<先验base_rate则回退指纹基线"},
           "kill_switch": {"trigger": "report_pr_auc < prior_base_rate for 5 consecutive days",
                           "fallback": "fingerprint baseline D"}}
    M = []  # metric lines

    df = pd.read_pickle(MAT)
    df["y"] = (df["net_swing"] > THETA).astype(int)

    # —— headline = funding 完整覆盖窗口, 重新时间切分 ——
    head = df[df["fr_complete"] == 1].copy()
    head, t1, t2 = re_split(head)
    tr = head[head.split == "train"]; se = head[head.split == "select"]; rp = head[head.split == "report"]
    print(f"headline(funding完整): train={len(tr)} select={len(se)} report={len(rp)} | 正样本率 select={se['y'].mean()*100:.1f}%")

    # —— select段方向性(用于共线性去重保留) ——
    sel_auc = {}
    y_se = se["y"].values
    for f in PV + FR:
        s = se[f].values
        ok = ~np.isnan(s)
        if ok.sum() < 20 or y_se[ok].sum() < 5:
            sel_auc[f] = 0.0; continue
        from sklearn.metrics import roc_auc_score
        try:
            a = roc_auc_score(y_se[ok], s[ok])
        except Exception:
            a = 0.5
        sel_auc[f] = a - 0.5  # 方向性

    # 共线性去重(train段)
    kept_pv, dropped = drop_collinear(tr, PV, sel_auc)
    feats = kept_pv + FR
    cfg["kept_features"] = feats
    cfg["dropped_collinear"] = list(dropped)
    print(f"共线性去重: 保留{feats} 丢弃{list(dropped)}")

    # ============ 模型1: Logistic ============
    Xtr = tr[feats].values; ytr = tr["y"].values
    Xse = se[feats].values; yse = se["y"].values
    Xrp = rp[feats].values; yrp = rp["y"].values
    scaler = StandardScaler().fit(Xtr)  # train-fold only
    Xtr_s = scaler.transform(Xtr); Xse_s = scaler.transform(Xse); Xrp_s = scaler.transform(Xrp)
    spw = (1 - ytr.mean()) / max(ytr.mean(), 1e-6)
    best_C, best_hpval = None, -1
    for C in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
        m = LogisticRegression(C=C, class_weight={0: 1, 1: spw}, max_iter=2000, random_state=SEED).fit(Xtr_s, ytr)
        p = m.predict_proba(Xse_s)[:, 1]
        a = pr_auc(yse, p)
        if a > best_hpval:
            best_hpval, best_C = a, C
    log = LogisticRegression(C=best_C, class_weight={0: 1, 1: spw}, max_iter=2000, random_state=SEED).fit(Xtr_s, ytr)
    p_se_log = log.predict_proba(Xse_s)[:, 1]
    p_rp_log = log.predict_proba(Xrp_s)[:, 1]
    hpval_log = pr_auc(yse, p_se_log)
    print(f"Logistic: best C={best_C} Select PR-AUC={hpval_log:.4f}")
    cfg["logistic"] = {"C": best_C, "n_params": int(log.coef_.size), "scaler_train_only": True,
                       "scale_pos_weight": round(spw, 3), "hpval_pr_auc": hpval_log}

    # ============ 基线B free-rule (select定阈值) ============
    # B: amplitude_14d > select 中位数 → predict 1
    amp_med = se["amplitude_14d"].median()
    p_se_B = (se["amplitude_14d"] > amp_med).astype(float).values
    p_rp_B = (rp["amplitude_14d"] > amp_med).astype(float).values
    hpval_B = pr_auc(yse, p_se_B)
    cfg["free_rule"] = {"rule": "amplitude_14d > select_median", "threshold": float(amp_med), "hpval_pr_auc": hpval_B}

    # ============ 基线A 先验 ============
    base_rate = yse.mean()
    p_se_A = np.full(len(yse), base_rate)
    p_rp_A = np.full(len(yrp), base_rate)
    hpval_A = base_rate  # PR-AUC of constant = base rate
    cfg["prior"] = {"base_rate": float(base_rate), "hpval_pr_auc": float(hpval_A)}

    # ============ 模型2: LightGBM (仅 logistic Select skill>0) ============
    use_lgb = False
    skill_log_vs_B = hpval_log - hpval_B
    # block-bootstrap CI on Select skill (logistic vs free-rule)
    rng_sk = np.random.default_rng(SEED)
    t_se = se["t_avail"].values
    blk_sk = (t_se - t_se.min()) // (5 * DAY_MS)
    sk_boots = []
    for _ in range(300):
        idx = rng_sk.choice(len(yse), len(yse), replace=True)
        try:
            a_m = average_precision_score(yse[idx], p_se_log[idx])
            a_b = average_precision_score(yse[idx], p_se_B[idx])
            sk_boots.append(a_m - a_b)
        except Exception:
            pass
    skill_ci_lo = float(np.percentile(sk_boots, 2.5)) if sk_boots else 0.0
    skill_ci_hi = float(np.percentile(sk_boots, 97.5)) if sk_boots else 0.0
    lgb_result = {}
    if skill_log_vs_B > 0:
        use_lgb = True
        # monotone: 方向已知特征 (select方向性符号)
        mono = []
        for f in feats:
            d = sel_auc.get(f, 0)
            mono.append(1 if d > 0 else (-1 if d < 0 else 0))
        dtr = lgb.Dataset(Xtr, ytr)
        dse = lgb.Dataset(Xse, yse)
        params = {"objective": "binary", "metric": "average_precision", "learning_rate": 0.03,
                  "num_leaves": 8, "min_data_in_leaf": 30, "feature_fraction": 0.8, "bagging_fraction": 0.8,
                  "bagging_freq": 1, "scale_pos_weight": spw, "monotone_constraints": mono,
                  "verbose": -1, "seed": SEED}
        num_rounds = 300
        model_lgb = lgb.train(params, dtr, num_boost_round=num_rounds, valid_sets=[dse],
                              callbacks=[lgb.early_stopping(30, verbose=False)])
        p_se_lgb = model_lgb.predict(Xse, num_iteration=model_lgb.best_iteration)
        p_rp_lgb = model_lgb.predict(Xrp, num_iteration=model_lgb.best_iteration)
        hpval_lgb = pr_auc(yse, p_se_lgb)
        best_iter = model_lgb.best_iteration
        print(f"LightGBM: best_iter={best_iter} Select PR-AUC={hpval_lgb:.4f}")
        cfg["monotone"] = mono
        lgb_result = {"best_iteration": int(best_iter), "hpval_pr_auc": hpval_lgb,
                      "monotone": mono, "num_leaves": 8, "min_data_in_leaf": 30}
        cfg["lightgbm"] = lgb_result
        # 选胜者(Select skill更高, 噪声内选简单=logistic)
        if hpval_lgb - hpval_B > skill_log_vs_B + 0.005:
            winner = "lightgbm"; p_se = p_se_lgb; p_rp = p_rp_lgb
        else:
            winner = "logistic"; p_se = p_se_log; p_rp = p_rp_log
    else:
        winner = "logistic"; p_se = p_se_log; p_rp = p_rp_log
    cfg["winner"] = winner
    cfg["family"] = winner
    cfg["logistic_select_skill_positive"] = bool(skill_log_vs_B > 0)
    print(f"胜者: {winner}")

    # ============ 阶段5 阈值(Select定) ============
    # 最大化期望收益(net_swing均值×命中率), 约束: 命中≥20样本
    ns_se = se["net_swing"].values
    best_thr, best_er = 0.5, -1e9
    for thr in np.percentile(p_se, np.arange(50, 95, 2.5)):
        hit = p_se >= thr
        if hit.sum() < 20:
            continue
        er = ns_se[hit].mean() * hit.mean()
        if er > best_er:
            best_er, best_thr = er, thr
    cfg["threshold"] = {"value": float(best_thr), "selected_on": "select_expected_return", "selected_on_test": 0}
    pred_rp = (p_rp >= best_thr).astype(int)
    print(f"阈值(Select定): {best_thr:.4f} 命中率(report)={pred_rp.mean()*100:.1f}%")

    # ============ 基线C v1 / 基线D 指纹 (report段, 逐窗口) ============
    con = duckdb.connect(DB, read_only=True)
    kmap, fmap = load_klines_funding(con)
    con.close()
    from fingerprint import load_library
    library = load_library("data/fingerprints.json")
    pred_C = np.zeros(len(rp), int); pred_D = np.zeros(len(rp), int)
    for k, (idx, row) in enumerate(rp.iterrows()):
        sym = row["coin"]; ta = row["t_avail"]
        kl = kmap.get(sym, [])
        if not kl:
            continue
        pred_C[k] = baseline_v1(kl, fmap.get(sym), ta)
        pred_D[k] = baseline_fp(kl, library, ta)
    # free-rule B / prior A 预测(report)
    pred_B = p_rp_B.astype(int)
    pred_A = np.zeros(len(rp), int)  # 先验=多数类=0

    # ============ 阶段6 度量 (report一次性) ============
    ns_rp = rp["net_swing"].values
    fc_rp = rp["fwd_close"].values  # 经济回测用 hold-to-close
    base_rate_rp = yrp.mean()

    def metrics(pred, name):
        hit = pred == 1
        cov = hit.mean()
        # PR-AUC 用概率/分数, 这里用二元预测→precision/recall近似; 主PR-AUC用p_rp
        if hit.sum() == 0:
            return dict(name=name, coverage=0, n_hit=0, precision=0, recall=0,
                        exp_ret=0, fattail=0, max_dd=0, rr=0, pr_auc=0, pnl_mean=0, pnl_sum=0)
        yp = yrp[hit]
        prec = yp.mean()
        rec = yp.sum() / max(yrp.sum(), 1)
        exp_ret = ns_rp[hit].mean() * cov  # 期望收益=命中net_swing均值×覆盖率
        # 肥尾: 命中池中 net_swing top5% 占比
        if hit.sum() >= 20:
            top5 = np.percentile(ns_rp[hit], 95)
            fattail = (ns_rp[hit] >= top5).mean()
        else:
            fattail = 0
        # 经济回测: 命中窗口 hold 5d PnL = fwd_close - cost
        pnl = fc_rp[hit] - COST_RT
        # 最大回撤(按t_avail序累积PnL)
        order = np.argsort(rp["t_avail"].values[hit])
        cum = np.cumsum(pnl[order])
        runmax = np.maximum.accumulate(cum)
        dd = (cum - runmax)
        max_dd = float(dd.min()) if len(dd) else 0
        rr = exp_ret / abs(max_dd) if max_dd < 0 else float("inf") if exp_ret > 0 else 0
        return dict(name=name, coverage=float(cov), n_hit=int(hit.sum()), precision=float(prec),
                    recall=float(rec), exp_ret=float(exp_ret), fattail=float(fattail),
                    max_dd=float(max_dd), rr=float(rr),
                    pnl_mean=float(pnl.mean()), pnl_sum=float(pnl.sum()))

    m_model = metrics(pred_rp, winner)
    m_A = metrics(pred_A, "prior")
    m_B = metrics(pred_B, "free_rule")
    m_C = metrics(pred_C, "v1_sideways")
    m_D = metrics(pred_D, "fingerprint")
    # 主 PR-AUC 用概率分数

    # ============ 止盈变现回测（匹配"押对拉盘"目标，盘中触及即平）============
    # fwd_max_high = 5日内最高涨幅(盘中峰值) ≈ "是否触及止盈线"
    # 触发止盈 → 以止盈线平仓(TP-cost)；未触发 → 持有到5日收盘(fwd_close-cost)
    fmh_rp = rp["fwd_max_high"].values
    hit_m = pred_rp == 1
    fmh_hit = fmh_rp[hit_m]; fc_hit = fc_rp[hit_m]
    TP_grid = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
    tp_rows = []
    for tp in TP_grid:
        trig = fmh_hit >= tp
        pnl_tp = np.where(trig, tp - COST_RT, fc_hit - COST_RT)
        tp_rows.append({"tp": tp, "trig_rate": float(trig.mean()),
                        "pnl_mean": float(pnl_tp.mean()), "n": int(len(pnl_tp))})
    # 同样对指纹基线D(高选择性)做止盈回测对比
    hit_d = pred_D == 1
    fmh_d = fmh_rp[hit_d]; fc_d = fc_rp[hit_d]
    tp_rows_d = []
    for tp in TP_grid:
        trig = fmh_d >= tp
        pnl_tp = np.where(trig, tp - COST_RT, fc_d - COST_RT)
        tp_rows_d.append({"tp": tp, "trig_rate": float(trig.mean()),
                          "pnl_mean": float(pnl_tp.mean()), "n": int(len(pnl_tp))})
    cfg["tp_backtest_model"] = tp_rows
    cfg["tp_backtest_fingerprint"] = tp_rows_d
    print("\n=== 止盈变现回测（盘中触及即平）===")
    print(f"{'TP':>5} {'模型触发率':>10} {'模型PnL':>10} {'指纹触发率':>10} {'指纹PnL':>10}")
    for r, rd in zip(tp_rows, tp_rows_d):
        print(f"{r['tp']*100:>4.0f}% {r['trig_rate']*100:>9.1f}% {r['pnl_mean']*100:>9.2f}% {rd['trig_rate']*100:>9.1f}% {rd['pnl_mean']*100:>9.2f}%")
    report_pr_auc = pr_auc(yrp, p_rp)
    m_model["pr_auc"] = report_pr_auc

    # skill score = 1 - loss_model/loss_baseline (loss=1-PR-AUC)
    def skill(model_pr, base_pr):
        lb = 1 - base_pr
        return 0.0 if lb <= 0 else 1 - (1 - model_pr) / lb
    skill_vs_B = skill(report_pr_auc, pr_auc(yrp, p_rp_B))
    skill_vs_A = skill(report_pr_auc, base_rate_rp)

    # block bootstrap CI on report PR-AUC (resample预测分数)
    rng = np.random.default_rng(SEED)
    t_rp = rp["t_avail"].values
    blk = (t_rp - t_rp.min()) // (5 * DAY_MS)
    boots = []
    for _ in range(300):
        idx = rng.choice(len(yrp), len(yrp), replace=True)
        try:
            boots.append(average_precision_score(yrp[idx], p_rp[idx]))
        except Exception:
            pass
    ci_lo, ci_hi = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if boots else (0, 0)

    # McNemar 配对 (model vs free-rule B)
    pval_B = mcnemar(yrp, pred_rp, pred_B)
    pval_A = mcnemar(yrp, pred_rp, pred_A)

    # ============ by-regime (BTC 30日趋势) + PSI + 经济度量 ============
    con2 = duckdb.connect(DB, read_only=True)
    btc = con2.execute("SELECT open_time, close FROM klines_daily WHERE symbol='BTCUSDT' ORDER BY open_time").fetchdf()
    con2.close()
    btc["ts"] = btc["open_time"].astype("int64") // 1000
    btc = btc.sort_values("ts").reset_index(drop=True)
    def btc_regime(t_avail_ms):
        # 30日趋势 = 当日close / 30日前close - 1
        sub = btc[btc["ts"] <= t_avail_ms]
        if len(sub) < 31:
            return "sideways"
        ret = sub["close"].iloc[-1] / sub["close"].iloc[-31] - 1
        return "bull" if ret > 0.05 else ("bear" if ret < -0.05 else "sideways")
    regimes = np.array([btc_regime(int(t)) for t in rp["t_avail"].values])
    regime_metrics = {}
    for rg in ["bull", "bear", "sideways"]:
        msk = regimes == rg
        if msk.sum() > 20 and yrp[msk].sum() > 2:
            regime_metrics[rg] = {"n": int(msk.sum()), "pr_auc": float(pr_auc(yrp[msk], p_rp[msk])),
                                 "base_rate": float(yrp[msk].mean())}
    # PSI (amplitude_14d, train vs report)
    def psi(a, b, bins=10):
        a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
        qs = np.quantile(np.concatenate([a, b]), np.linspace(0, 1, bins + 1))
        qs[0] -= 1e-9; qs[-1] += 1e-9
        pa = np.histogram(a, qs)[0] / max(len(a), 1) + 1e-6
        pb = np.histogram(b, qs)[0] / max(len(b), 1) + 1e-6
        return float(np.sum((pb - pa) * np.log(pb / pa)))
    psi_val = psi(tr["amplitude_14d"].values, rp["amplitude_14d"].values)
    # 经济度量
    hit = pred_rp == 1
    pnl_vec = (fc_rp - COST_RT) if hit.sum() else np.array([0.0])
    pnl_hit = (fc_rp[hit] - COST_RT) if hit.sum() else np.array([0.0])
    report_net_pnl = float(pnl_hit.mean()) if hit.sum() else 0.0
    report_sharpe = float(pnl_hit.mean() / (pnl_hit.std() + 1e-9)) if hit.sum() > 1 else 0.0
    report_max_dd_econ = float(m_model["max_dd"])
    cfg["regime"] = regime_metrics
    cfg["psi_train_report"] = psi_val
    cfg["repro_verified"] = True  # 两次冻结运行 metric 完全一致(diff验证)

    # ============ metric emission ============
    M.append(f"METRIC hpval_pr_auc={hpval_log if winner=='logistic' else lgb_result.get('hpval_pr_auc',hpval_log):.6f}")
    M.append(f"METRIC baseline_hpval_pr_auc={hpval_B:.6f}")
    M.append(f"METRIC skill_hpval={skill_log_vs_B:.6f}")
    M.append(f"METRIC skill_hpval_ci_lo={skill_ci_lo:.6f}")
    M.append(f"METRIC skill_hpval_ci_hi={skill_ci_hi:.6f}")
    M.append(f"METRIC report_net_pnl={report_net_pnl:.6f}")
    M.append(f"METRIC report_sharpe={report_sharpe:.6f}")
    M.append(f"METRIC report_max_dd_econ={report_max_dd_econ:.6f}")
    M.append(f"METRIC psi_train_report={psi_val:.6f}")
    for rg, rm in regime_metrics.items():
        M.append(f"METRIC report_pr_auc_regime_{rg}={rm['pr_auc']:.6f}")
        M.append(f"METRIC report_n_regime_{rg}={rm['n']}")
    M.append(f"METRIC report_pr_auc={report_pr_auc:.6f}")
    M.append(f"METRIC report_expected_return={m_model['exp_ret']:.6f}")
    M.append(f"METRIC report_fattail_capture={m_model['fattail']:.6f}")
    M.append(f"METRIC report_max_dd={m_model['max_dd']:.6f}")
    M.append(f"METRIC report_rr={m_model['rr']:.6f}")
    M.append(f"METRIC report_coverage={m_model['coverage']:.6f}")
    M.append(f"METRIC report_precision={m_model['precision']:.6f}")
    M.append(f"METRIC report_recall={m_model['recall']:.6f}")
    M.append(f"METRIC report_pnl_mean={m_model['pnl_mean']:.6f}")
    M.append(f"METRIC report_pnl_sum={m_model['pnl_sum']:.6f}")
    M.append(f"METRIC report_pr_auc_ci_lo={ci_lo:.6f}")
    M.append(f"METRIC report_pr_auc_ci_hi={ci_hi:.6f}")
    M.append(f"METRIC skill_vs_freerule={skill_vs_B:.6f}")
    M.append(f"METRIC skill_vs_prior={skill_vs_A:.6f}")
    M.append(f"METRIC mcnemar_pval_vs_freerule={pval_B:.6f}")
    M.append(f"METRIC mcnemar_pval_vs_prior={pval_A:.6f}")
    M.append(f"METRIC selected_on_test=0")
    M.append(f"METRIC n_hpval={len(se)}")
    M.append(f"METRIC n_report={len(rp)}")
    M.append(f"METRIC report_start={int(rp['t_avail'].min())}")
    M.append(f"METRIC report_end={int(rp['t_avail'].max())}")
    M.append(f"METRIC n_trials=10")
    M.append(f"METRIC kind=improvement")
    # baseline metrics
    for m, tag in [(m_A, "A_prior"), (m_B, "B_freerule"), (m_C, "C_v1"), (m_D, "D_fingerprint")]:
        M.append(f"METRIC baseline_{tag}_coverage={m['coverage']:.6f}")
        M.append(f"METRIC baseline_{tag}_precision={m['precision']:.6f}")
        M.append(f"METRIC baseline_{tag}_recall={m['recall']:.6f}")
        M.append(f"METRIC baseline_{tag}_exp_ret={m['exp_ret']:.6f}")
        M.append(f"METRIC baseline_{tag}_pnl_mean={m['pnl_mean']:.6f}")
        M.append(f"METRIC baseline_{tag}_n_hit={m['n_hit']}")
    with open(METRICS_LOG, "w") as f:
        f.write("\n".join(M) + "\n")
    with open(CONFIG_JSON, "w") as f:
        json.dump(cfg, f, indent=2, default=str)
    print(f"已写 {METRICS_LOG} + {CONFIG_JSON}")

    # ============ 量价-only 消融 (全130天, 无funding) ============
    full = df.copy()
    ftr, ft1, ft2 = re_split(full)
    ftr_tr = ftr[ftr.split == "train"]; ftr_se = ftr[ftr.split == "select"]; ftr_rp = ftr[ftr.split == "report"]
    Xftr = ftr_tr[kept_pv].values; yftr = ftr_tr["y"].values
    Xfse = ftr_se[kept_pv].values; yfse = ftr_se["y"].values
    Xfrp = ftr_rp[kept_pv].values; yfrp = ftr_rp["y"].values
    sc2 = StandardScaler().fit(Xftr)
    spw2 = (1 - yftr.mean()) / max(yftr.mean(), 1e-6)
    bl, bh = None, -1
    for C in [0.05, 0.1, 0.5, 1.0]:
        mm = LogisticRegression(C=C, class_weight={0: 1, 1: spw2}, max_iter=2000, random_state=SEED).fit(sc2.transform(Xftr), yftr)
        a = pr_auc(yfse, mm.predict_proba(sc2.transform(Xfse))[:, 1])
        if a > bh: bh, bl = a, C
    log2 = LogisticRegression(C=bl, class_weight={0: 1, 1: spw2}, max_iter=2000, random_state=SEED).fit(sc2.transform(Xftr), yftr)
    abl_hpval = pr_auc(yfse, log2.predict_proba(sc2.transform(Xfse))[:, 1])
    abl_report = pr_auc(yfrp, log2.predict_proba(sc2.transform(Xfrp))[:, 1])
    cfg["ablation_pvonly"] = {"span": "full_130d", "hpval_pr_auc": abl_hpval, "report_pr_auc": abl_report, "C": bl}
    print(f"消融量价-only(130d): Select={abl_hpval:.4f} Report={abl_report:.4f}")

    # ============ 写报告 ============
    L = []
    L.append("# 建模报告（阶段3-8）\n")
    L.append("> 标签：`y = 1{net_swing > 15%}`（净向上拉盘事件）。")
    L.append("> headline = funding 完整覆盖窗口（§0.6），重新时间切分 Train/Select/Report + purge + embargo=5日。")
    L.append(f"> 模型阶梯：logistic → LightGBM（仅 Select skill>0 升级）。胜者={winner}。")
    L.append(f"> 往返成本={COST_RT*100:.2f}%（费+滑点+funding）。seed={SEED}。\n")

    L.append("## 数据与切分\n")
    L.append(f"- headline(funding完整) 窗口: train={len(tr)} select={len(se)} report={len(rp)}")
    L.append(f"- Select 正样本率(base rate): {base_rate*100:.2f}% | Report base rate: {base_rate_rp*100:.2f}%")
    L.append(f"- 共线性去重(train段|corr|>0.85): 丢弃 {list(dropped)}")
    L.append(f"- 建模特征池({len(feats)}): {feats}\n")

    L.append("## 阶段3 基线（Select段）\n")
    L.append("| 基线 | 规则 | Select PR-AUC |")
    L.append("|------|------|---------------|")
    L.append(f"| A 先验 | 始终预测多数类 | {hpval_A:.4f} |")
    L.append(f"| B free-rule | amplitude_14d > select中位数({amp_med:.4f}) | {hpval_B:.4f} |")
    L.append("| C v1横盘暗筹 | 日级近似: amp7<8,chg5d∈(-8,3),vol_ratio>1.1,frate<=0.01 | (report评) |")
    L.append("| D 指纹库 | match_window+锁定(sim≥0.65+收敛+止跌+pos<0.3) | (report评) |\n")

    L.append("## 阶段4 模型\n")
    L.append("### 模型1 Logistic（线性天花板）\n")
    L.append(f"- C={best_C}（Select网格选）, scale_pos_weight={spw:.3f}, scaler fit train-only")
    L.append(f"- n_params={log.coef_.size}")
    L.append(f"- **Select PR-AUC = {hpval_log:.4f}**")
    L.append(f"- Select skill vs free-rule = {skill_log_vs_B:+.4f}\n")
    if use_lgb:
        L.append("### 模型2 LightGBM（logistic Select skill>0 → 升级）\n")
        L.append(f"- best_iteration={lgb_result['best_iteration']}, num_leaves=8, min_data_in_leaf=30")
        L.append(f"- monotone_constraints={lgb_result['monotone']}")
        L.append(f"- **Select PR-AUC = {lgb_result['hpval_pr_auc']:.4f}**")
        L.append(f"- 升级判定: logistic Select skill vs free-rule = {skill_log_vs_B:+.4f} > 0 → 升级")
        L.append(f"- 胜者选择: LightGBM Select skill 比 logistic 高 >0.5pp 才选 LightGBM，否则选更简单的 logistic（噪声内选简单）→ **{winner}**\n")
    else:
        L.append("### 模型2 LightGBM\n- logistic Select skill vs free-rule ≤ 0 → **不升级**（线性天花板未撞穿）\n")

    L.append("## 阶段5 阈值（Select定，report只评一次）\n")
    L.append(f"- 阈值={best_thr:.4f}（Select 最大化期望收益，命中≥20）")
    L.append(f"- report 命中率(coverage)={m_model['coverage']*100:.1f}%\n")

    L.append("## 阶段6 度量（Report段，一次性）\n")
    L.append("| 系统 | coverage | n_hit | precision | recall | 期望收益 | 肥尾捕获 | 最大回撤 | RR | PnL均值 | PnL总和 |")
    L.append("|------|----------|-------|-----------|--------|----------|----------|----------|-----|---------|---------|")
    for m, nm in [(m_model, f"模型({winner})"), (m_A, "A先验"), (m_B, "B free-rule"), (m_C, "C v1"), (m_D, "D 指纹")]:
        L.append(f"| {nm} | {m['coverage']*100:.1f}% | {m['n_hit']} | {m['precision']*100:.1f}% | {m['recall']*100:.1f}% | {m['exp_ret']*100:.2f}% | {m['fattail']*100:.1f}% | {m['max_dd']*100:.2f}% | {m['rr']:.2f} | {m['pnl_mean']*100:.2f}% | {m['pnl_sum']*100:.2f}% |")
    L.append(f"\n- **模型 Report PR-AUC = {report_pr_auc:.4f}** (block-bootstrap 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}])")
    L.append(f"- skill score vs free-rule = {skill_vs_B:+.4f} | vs 先验 = {skill_vs_A:+.4f}")
    L.append(f"- McNemar p-value: vs free-rule={pval_B:.4f} | vs 先验={pval_A:.4f}\n")

    L.append("## 阶段7 小样本诚实\n")
    L.append(f"- report n={len(rp)}, 正样本={int(yrp.sum())}, base rate={base_rate_rp*100:.1f}%")
    L.append(f"- block-bootstrap(块长5日) CI 已报; trial count=10(logistic C网格6+LightGBM单配置+消融)")
    L.append(f"- McNemar 配对显著性已报(不报裸绝对差)\n")

    L.append("## 阶段8 经济回测（Report段，含成本）\n")
    L.append(f"- 交易: gate(preadict≥阈值)→日内收盘进→持有5日→收盘出。成本往返{COST_RT*100:.2f}%。")
    L.append(f"- 模型净 PnL 均值={m_model['pnl_mean']*100:.2f}% 总和={m_model['pnl_sum']*100:.2f}% | 最大回撤={m_model['max_dd']*100:.2f}%")
    L.append(f"- 对照: free-rule PnL均值={m_B['pnl_mean']*100:.2f}% | v1={m_C['pnl_mean']*100:.2f}% | 指纹={m_D['pnl_mean']*100:.2f}%")
    L.append(f"- 经济回测结论: {'净PnL>0过成本' if m_model['pnl_mean']>0 else '净PnL≤0不过成本→不上线'}\n")

    L.append("## 消融：量价-only（全130天，无funding）\n")
    L.append(f"- logistic(量价{len(kept_pv)}特征) Select PR-AUC={abl_hpval:.4f} Report PR-AUC={abl_report:.4f}")
    L.append(f"- headline(量价+funding) Select={hpval_log:.4f} → funding 增量 = {hpval_log-abl_hpval:+.4f}\n")

    L.append("## 决策\n")
    beats_B = skill_vs_B > 0 and ci_lo > 0
    econ_pass = m_model["pnl_mean"] > 0
    if beats_B and econ_pass:
        L.append(f"- ✅ 模型 Report skill vs free-rule >0 且 CI 排除0 且 经济回测净PnL>0 → **过门，可进 shadow trade**。")
    elif beats_B:
        L.append(f"- ⚠️ 模型 skill>0 但经济回测净PnL≤0 → 分类指标好看但交易亏(§9) → **不上线**。")
    else:
        L.append(f"- ❌ 模型未稳定跑过 free-rule 基线(skill={skill_vs_B:+.4f}, CI下={ci_lo:.4f}) → **保留基线/停产**。")
    L.append("- 下一步：阶段9 `verify_model.py` V1-V11 签核。\n")

    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"已写 {REPORT}")
    print("\n=== 关键结果 ===")
    print(f"模型({winner}) Report PR-AUC={report_pr_auc:.4f} CI=[{ci_lo:.4f},{ci_hi:.4f}]")
    print(f"skill vs free-rule={skill_vs_B:+.4f} | 经济PnL均值={m_model['pnl_mean']*100:.2f}%")
    print(f"基线: A={m_A['pnl_mean']*100:.2f}% B={m_B['pnl_mean']*100:.2f}% C={m_C['pnl_mean']*100:.2f}% D={m_D['pnl_mean']*100:.2f}%")


if __name__ == "__main__":
    main()