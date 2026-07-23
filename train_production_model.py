"""训练生产 LightGBM 模型(B配置:14特征含funding)并保存artifact。
与 modeling.py 完全同参(seed=42, monotone, early_stopping)。"""
import sys, json, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import average_precision_score
sys.path.insert(0, ".")
import modeling as M

FEATS = ['atr_14_pct','bb_width','washouts_count','peak_valley_ratio','max_drawdown',
         'ret_var','atr_7_pct','atr_5_pct','fr_min_early','fr_late_val',
         'vol_ratio_var','heavy_vol_pct','vol_price_corr','ret_skew']
MONO  = [1,1,1,1,-1,1,1,1,-1,1, 1,1,1,1]
POS_FILTER = 0.3   # 后置护栏: pos<0.3 才开仓

def main():
    df = pd.read_pickle("/tmp/feature_matrix_funding.pkl")
    df["y"] = (df["net_swing"] > M.THETA).astype(int)
    head = df[df["fr_complete"] == 1].copy()
    head, t1, t2 = M.re_split(head)
    tr = head[head.split == "train"]; se = head[head.split == "select"]
    ytr, yse = tr["y"].values, se["y"].values
    spw = (1 - ytr.mean()) / max(ytr.mean(), 1e-6)
    params = {"objective":"binary","metric":"average_precision","learning_rate":0.03,
              "num_leaves":8,"min_data_in_leaf":30,"feature_fraction":0.8,
              "bagging_fraction":0.8,"bagging_freq":1,"scale_pos_weight":spw,
              "monotone_constraints":MONO,"verbose":-1,"seed":M.SEED}
    model = lgb.train(params, lgb.Dataset(tr[FEATS].values, ytr),
                      num_boost_round=300, valid_sets=[lgb.Dataset(se[FEATS].values, yse)],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
    # 验证
    p_se = model.predict(se[FEATS].values, num_iteration=model.best_iteration)
    p_tr = model.predict(tr[FEATS].values, num_iteration=model.best_iteration)
    sel_pr = average_precision_score(yse, p_se)
    print(f"Train n={len(tr)} Select n={len(se)}  best_iter={model.best_iteration}")
    print(f"Select PR-AUC = {sel_pr:.4f} (须≈0.2430 与 modeling.py 一致)")
    # 保存
    model.save_model("data/model.txt")
    cfg = {"features": FEATS, "monotone": MONO, "pos_filter": POS_FILTER,
           "theta": M.THETA, "seed": M.SEED, "best_iteration": model.best_iteration,
           "select_pr_auc": float(sel_pr), "window": 14, "forward": 5,
           "train_split_t1": int(t1), "select_split_t2": int(t2),
           "label": "net_swing>0.15", "model_family": "LightGBM",
           "note": "B配置:10基线(8量价+2funding)+4量能增强; pos<0.3后置护栏(非特征)"}
    with open("data/model_config.json","w") as f: json.dump(cfg, f, indent=2, ensure_ascii=False)
    print("已保存 data/model.txt + data/model_config.json")
    # 特征重要性
    imp = sorted(zip(FEATS, model.feature_importance(importance_type="gain")), key=lambda x:-x[1])
    print("\n特征重要性(gain):")
    for k,v in imp: print(f"  {k:<20} {v:.1f}")

if __name__ == "__main__":
    main()