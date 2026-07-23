"""实验A(pos<0.3后置过滤) + 实验B(加量能特征重训)。
目标=找对可能庄家拉盘的币 → 全程分类指标(PR-AUC/Precision@TopK/Recall/Coverage), 不碰止盈/PnL。"""
import sys, numpy as np, pandas as pd, lightgbm as lgb, duckdb
from sklearn.metrics import average_precision_score
sys.path.insert(0, ".")
import modeling as M
from feature_mining import build_features, load_klines_db, WINDOW
from funding_features import load_funding, compute_fr
from fingerprint import load_library, match_window

DB = "data/market.duckdb"
MAT = "/tmp/feature_matrix_funding.pkl"
FEATS_A = ['atr_14_pct', 'bb_width', 'washouts_count', 'peak_valley_ratio',
           'max_drawdown', 'ret_var', 'atr_7_pct', 'atr_5_pct', 'fr_min_early', 'fr_late_val']
MONO_A = [1, 1, 1, 1, -1, 1, 1, 1, -1, 1]
POS_COL = "pos"

def auc(y, s):
    ok = ~np.isnan(s)
    if ok.sum() < 5 or y[ok].sum() < 2: return 0.0
    return float(average_precision_score(y[ok], s[ok]))

def eval_cls(y, prob, mask=None, topk=20):
    """分类评估: PR-AUC + Precision@TopK + Recall@TopK + Coverage@TopK"""
    if mask is not None:
        y, prob = y[mask], prob[mask]
    order = np.argsort(-prob)
    top = order[:topk]
    yt = y[top]
    prec = float(yt.mean()) if len(yt) else 0.0
    rec = float(yt.sum() / max(y.sum(), 1))
    pr = auc(y, prob)
    cov = float((prob >= 0.27978).mean())  # 参考阈值,仅示意coverage
    return dict(pr_auc=pr, precision_top20=prec, recall_top20=rec, coverage=cov, n=len(y), n_pos=int(y.sum()))

def train_lgb(Xtr, ytr, Xse, yse, mono):
    spw = (1 - ytr.mean()) / max(ytr.mean(), 1e-6)
    params = {"objective": "binary", "metric": "average_precision", "learning_rate": 0.03,
              "num_leaves": 8, "min_data_in_leaf": 30, "feature_fraction": 0.8, "bagging_fraction": 0.8,
              "bagging_freq": 1, "scale_pos_weight": spw, "monotone_constraints": mono,
              "verbose": -1, "seed": M.SEED}
    m = lgb.train(params, lgb.Dataset(Xtr, ytr), num_boost_round=300,
                  valid_sets=[lgb.Dataset(Xse, yse)], callbacks=[lgb.early_stopping(30, verbose=False)])
    return m

# ============ 加载 + 切分 ============
df = pd.read_pickle(MAT)
df["y"] = (df["net_swing"] > M.THETA).astype(int)
head = df[df["fr_complete"] == 1].copy()
head, _, _ = M.re_split(head)
tr, se, rp = head[head.split=="train"], head[head.split=="select"], head[head.split=="report"]
ytr, yse, yrp = tr["y"].values, se["y"].values, rp["y"].values
pos_rp = rp[POS_COL].values

print("="*70)
print("实验A: 保持当前LightGBM + pos<0.3后置过滤 (分类指标)")
print("="*70)
mA = train_lgb(tr[FEATS_A].values, ytr, se[FEATS_A].values, yse, MONO_A)
p_se_A = mA.predict(se[FEATS_A].values, num_iteration=mA.best_iteration)
p_rp_A = mA.predict(rp[FEATS_A].values, num_iteration=mA.best_iteration)
print(f"  Select PR-AUC = {auc(yse, p_se_A):.4f} (基线模型A)")

low = pos_rp < 0.3
print(f"\n  Report段: 全体 n={len(yrp)} 低位(pos<0.3) n={low.sum()} (占{low.mean()*100:.0f}%)")
print(f"  全体拉盘率={yrp.mean()*100:.1f}%  低位拉盘率={yrp[low].mean()*100:.1f}%")
print(f"\n  {'指标':<16}{'无过滤':>12}{'pos<0.3过滤':>14}{'差值':>10}")
print("  " + "-"*50)
e_all = eval_cls(yrp, p_rp_A)
e_low = eval_cls(yrp, p_rp_A, low)
for k in ["pr_auc","precision_top20","recall_top20","coverage"]:
    d = e_low[k] - e_all[k]
    print(f"  {k:<16}{e_all[k]:>12.4f}{e_low[k]:>14.4f}{d:>+10.4f}")
print(f"  {'n_pos(命中池)':<16}{e_all['n_pos']:>12}{e_low['n_pos']:>14}")

print("\n" + "="*70)
print("实验B: 加量能特征重训 (分类指标)")
print("="*70)
# 候选量能特征, 查共线性+方向性
cand = ["vol_ratio_var","vol_ratio_max","vol_ratio_mean","heavy_vol_pct","vol_price_corr","ret_skew","vol_trend_slope"]
print("  候选量能特征 select段方向性:")
from sklearn.metrics import roc_auc_score
keep_cand = []
for f in cand:
    s = se[f].values; ok=~np.isnan(s)
    if ok.sum()<20 or yse[ok].sum()<5: print(f"    {f}: 样本不足"); continue
    a = roc_auc_score(yse[ok], s[ok]); d = a-0.5
    # 共线性: 与FEATS_A的max|corr|
    maxcorr = 0
    for g in FEATS_A:
        c = np.corrcoef(se[f].values, se[g].values)[0,1]
        if abs(c) > maxcorr: maxcorr = abs(c)
    flag = "✅保留" if abs(d)>0.05 and maxcorr<0.85 else f"跳过(方向{d:+.3f}/共线{maxcorr:.2f})"
    print(f"    {f:<20} 方向性={d:+.3f} 与现特征max|corr|={maxcorr:.2f} {flag}")
    if abs(d)>0.05 and maxcorr<0.85: keep_cand.append(f)
# 候选之间去共线
final_cand = []
for f in keep_cand:
    dup = False
    for g in final_cand:
        if abs(np.corrcoef(se[f].values, se[g].values)[0,1]) > 0.85: dup=True; break
    if not dup: final_cand.append(f)
print(f"  → 最终加入特征: {final_cand}")
FEATS_B = FEATS_A + final_cand
# monotone: 新特征按select方向性
MONO_B = list(MONO_A)
for f in final_cand:
    s = se[f].values; ok=~np.isnan(s); d = roc_auc_score(yse[ok], s[ok])-0.5
    MONO_B.append(1 if d>0 else (-1 if d<0 else 0))
print(f"  特征池B({len(FEATS_B)}): {FEATS_B}")
print(f"  monotoneB: {MONO_B}")

mB = train_lgb(tr[FEATS_B].values, ytr, se[FEATS_B].values, yse, MONO_B)
p_se_B = mB.predict(se[FEATS_B].values, num_iteration=mB.best_iteration)
p_rp_B = mB.predict(rp[FEATS_B].values, num_iteration=mB.best_iteration)
print(f"\n  Select PR-AUC: A={auc(yse,p_se_A):.4f} → B={auc(yse,p_se_B):.4f} ({auc(yse,p_se_B)-auc(yse,p_se_A):+.4f})")
print(f"\n  {'指标':<16}{'A无过滤':>10}{'A+pos<0.3':>12}{'B无过滤':>10}{'B+pos<0.3':>12}")
print("  " + "-"*58)
eB_all = eval_cls(yrp, p_rp_B); eB_low = eval_cls(yrp, p_rp_B, low)
for k in ["pr_auc","precision_top20","recall_top20","coverage"]:
    print(f"  {k:<16}{e_all[k]:>10.4f}{e_low[k]:>12.4f}{eB_all[k]:>10.4f}{eB_low[k]:>12.4f}")

# ============ 最新决策日 top10 (A/B × 无过滤/pos<0.3) ============
print("\n" + "="*70)
print("最新决策日(2026-07-22) Top10 对比")
print("="*70)
con = duckdb.connect(DB, read_only=True)
syms = [r[0] for r in con.execute("SELECT DISTINCT symbol FROM klines_daily ORDER BY symbol").fetchall()]
fr_map = load_funding(con)
rows = []
for sym in syms:
    bars = load_klines_db(con, sym)
    if len(bars) < WINDOW: continue
    i = len(bars)-1; f = build_features(bars, i)
    t_dec = int(bars[i]["ts"]) + 86400000
    fr = compute_fr(*fr_map.get(sym, (np.array([]), np.array([]))), t_dec) if sym in fr_map else None
    f["fr_min_early"] = fr["fr_min_early"] if fr else np.nan
    f["fr_late_val"] = fr["fr_late_val"] if fr else np.nan
    for c in final_cand: pass  # 已在build_features
    row = {k: f.get(k, np.nan) for k in FEATS_B + ["amplitude_14d","pos","ret_14d"]}
    row["coin"]=sym; row["bars"]=bars
    rows.append(row)
con.close()
inf = pd.DataFrame(rows)
inf["pA"] = mA.predict(inf[FEATS_A].values, num_iteration=mA.best_iteration)
inf["pB"] = mB.predict(inf[FEATS_B].values, num_iteration=mB.best_iteration)

def show_top(df, pcol, filt, title, n=10):
    d = df if filt is None else df[df["pos"]<filt]
    t = d.sort_values(pcol, ascending=False).head(n)
    print(f"\n--- {title} ---")
    print(f"{'#':>2} {'币种':<13}{'概率':>7}{'pos':>6}{'ret_14d':>9}{'amp14d':>8}{'状态':<14}")
    for k,(_,r) in enumerate(t.iterrows()):
        st="高位已涨" if r["pos"]>0.5 and r["ret_14d"]>0.1 else ("低位蓄势" if r["pos"]<0.3 else "中位")
        print(f"{k+1:>2} {r['coin']:<13}{r[pcol]*100:>6.1f}%{r['pos']:>6.2f}{r['ret_14d']*100:>8.1f}%{r['amplitude_14d']*100:>7.1f}%  {st}")

show_top(inf,"pA",None,"A: 当前模型 Top10 (无过滤)")
show_top(inf,"pA",0.3,"A: 当前模型 Top10 (pos<0.3过滤)")
show_top(inf,"pB",None,"B: +量能特征 Top10 (无过滤)")
show_top(inf,"pB",0.3,"B: +量能特征 Top10 (pos<0.3过滤)")

# 与指纹锁定命中交叉
print("\n" + "="*70)
print("与指纹库锁定命中(同决策日)交叉")
print("="*70)
lib = load_library()
fp_set=set()
for _,r in inf.iterrows():
    for h in match_window(r["bars"], lib, 14, 0.65):
        s=h["signatures"]
        if s.get("converge") and s.get("standstill") and s.get("pos",1)<0.3:
            fp_set.add(r["coin"]); break
for pcol,filt,tag in [("pA",None,"A无过滤"),("pA",0.3,"A+pos<0.3"),("pB",None,"B无过滤"),("pB",0.3,"B+pos<0.3")]:
    d = inf if filt is None else inf[inf["pos"]<filt]
    top=set(d.sort_values(pcol,ascending=False).head(10)["coin"])
    print(f"  {tag:<12} Top10 ∩ 指纹锁定({len(fp_set)}币) = {len(top & fp_set)}  {sorted(top & fp_set) if top&fp_set else ''}")