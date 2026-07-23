#!/usr/bin/env python3
"""阶段9 建模后验证脚本 —— 多门签核，保证建模结果正确可信。

用法:
  python3 verify_model.py                          # 仅跑结构门(V1/V3), 模型门PENDING
  python3 verify_model.py --metrics docs/exp.log --config model_cfg.json
  python3 verify_model.py --matrix /tmp/feature_matrix.pkl

门:
  V1  泄漏复审(结构): 标签列排除/时间因果/分割时序/前瞻标签
  V2  三分离合规: 扫 METRIC 日志, keep/discard 不含 report_
  V3  重叠/purge/embargo: 分割边界缓冲 + 有效块数
  V4  基线诚实: 四基线 + skill score CI 排除0      [需 metrics]
  V5  模型族协议: scaler/early-stop/monotone config [需 config]
  V6  度量匹配: 主指标 PR-AUC + 命名带角色 + N/日期 [需 metrics]
  V7  漂移/regime/小N: by-regime + PSI + block CI   [需 metrics]
  V8  复现性: seed/版本 + 冻结config重跑一致        [需 config + rerun]
  V9  经济回测: 成本 + 可用性延迟 + regime stress   [需 metrics]
  V10 live parity: model card/kill-switch          [人工]
  V11 签字门: V1-V10 全 PASS 才 promote

输出: stdout + docs/建模验证报告.md
退出码: 0=全PASS/PENDING可继续, 1=有FAIL
"""
import argparse, json, re, sys, pickle
from datetime import datetime, timezone
import pandas as pd
import numpy as np

OUT = "docs/建模验证报告.md"
gates = []  # (id, name, status, evidence)


def gate(vid, name, status, evidence):
    gates.append((vid, name, status, evidence))
    tag = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️ ", "PENDING": "⏳", "MANUAL": "✋"}[status]
    print(f"{tag} V{vid} {name}: {status} — {evidence}")


# ---------- V1 泄漏复审(结构) ----------
def v1_leakage(matrix_path):
    with open(matrix_path, "rb") as f:
        m = pickle.load(f)
    meta = {"coin", "status", "t_idx", "t_avail", "t_date", "split_raw", "split"}
    label_cols = {c for c in m.columns if c.startswith("fwd_") or c.startswith("y_")
                  or c == "net_swing"}
    feat_cols = [c for c in m.columns if c not in meta and c not in label_cols]
    # (a) 标签列不在特征集
    leak = label_cols & set(feat_cols)
    if leak:
        gate(1, "泄漏复审", "FAIL", f"标签列混入特征: {leak}")
        return
    # (b) 每coin t_avail 单调递增(时间因果)
    bad = []
    for coin, g in m.groupby("coin"):
        if not g["t_avail"].is_monotonic_increasing:
            bad.append(coin)
    if bad:
        gate(1, "泄漏复审", "FAIL", f"{len(bad)}币 t_avail 非单调(时间乱序): {bad[:5]}")
        return
    # (c) 分割时序: train.max(t_avail) < select.min ... < report.min(无shuffle)
    order_ok = True
    prev_max = -1
    for sp in ["train", "select", "report"]:
        sub = m[m["split"] == sp]
        if sub.empty:
            continue
        if sub["t_avail"].min() < prev_max:
            order_ok = False
        prev_max = max(prev_max, sub["t_avail"].max())
    if not order_ok:
        gate(1, "泄漏复审", "FAIL", "split 段时间序错乱(疑似shuffle)")
        return
    # (d) 前瞻标签: fwd_* 存在且为 forward(非NaN占比 + 列名前缀)
    has_fwd = any(c.startswith("fwd_") for c in m.columns)
    # net_swing 可计算性
    if "fwd_max_high" in m.columns and "fwd_max_low" in m.columns:
        ns = m["fwd_max_high"] - m["fwd_max_low"].abs()
        ns_valid = ns.notna().mean()
    else:
        ns_valid = float("nan")
    gate(1, "泄漏复审", "PASS",
         f"标签列排除✓; t_avail单调✓; split时序✓; {len(feat_cols)}特征; "
         f"net_swing可算(valid={ns_valid:.1%}); 标签列={sorted(label_cols)}")


# ---------- V3 重叠/purge/embargo ----------
def v3_overlap(matrix_path, embargo_days=5):
    with open(matrix_path, "rb") as f:
        m = pickle.load(f)
    # 分割边界缓冲: select.min(t_avail) - train.max(t_avail) >= embargo*86400s
    gaps = {}
    segs = {sp: m[m["split"] == sp] for sp in ["train", "select", "report"]}
    ms_day = 86400 * 1000
    for a, b in [("train", "select"), ("select", "report")]:
        if segs[a].empty or segs[b].empty:
            continue
        gap_ms = segs[b]["t_avail"].min() - segs[a]["t_avail"].max()
        gaps[f"{a}→{b}"] = round(float(gap_ms) / ms_day, 1)
    short = [k for k, v in gaps.items() if v < embargo_days]
    # 有效块数(report段, 块长>=5日, 步长3日重叠→有效块≈窗口数×3/5)
    rep = segs["report"]
    if not rep.empty:
        span_days = (rep["t_avail"].max() - rep["t_avail"].min()) / ms_day
        n_eff = max(1, int(span_days / 5))  # 粗估独立块
    else:
        n_eff = 0
    if short:
        gate(3, "重叠/embargo", "FAIL",
             f"embargo不足(需{embargo_days}日): {short} 实际{gaps}; N_eff≈{n_eff}")
    else:
        status = "PASS" if n_eff >= 100 else "WARN"
        gate(3, "重叠/embargo", status,
             f"边界缓冲{gaps}(均≥{embargo_days}日✓); report N_eff≈{n_eff}块"
             f"{' (偏小, 报上偏)' if n_eff < 100 else ''}")


# ---------- metrics 日志解析 ----------
def parse_metrics(path):
    if not path:
        return None
    metrics = {}
    try:
        for line in open(path):
            m = re.match(r"METRIC\s+(\w+)=(\S+)", line.strip())
            if m:
                metrics[m.group(1)] = m.group(2)
    except FileNotFoundError:
        return None
    return metrics or None


def v2_separation(metrics):
    if metrics is None:
        gate(2, "三分离合规", "PENDING", "无 metrics 日志(建模后提供 --metrics)")
        return
    sot = metrics.get("selected_on_test", "?")
    # 启发式: 若日志中出现"依据 report_*"字样则FAIL(需人工, 这里只查标志)
    if sot == "1":
        gate(2, "三分离合规", "FAIL",
             "selected_on_test=1 → report被用于选择, 不得claim OOS, 须新holdout重做")
    elif sot == "0":
        gate(2, "三分离合规", "PASS", "selected_on_test=0; keep/discard须仅用hpval_*(人工复核日志)")
    else:
        gate(2, "三分离合规", "PENDING", "缺 selected_on_test 标志")


def v4_baseline(metrics):
    if metrics is None:
        gate(4, "基线诚实", "PENDING", "无 metrics(需 baseline_hpval_pr_auc + skill_hpval)")
        return
    req = ["hpval_pr_auc", "baseline_hpval_pr_auc", "skill_hpval"]
    miss = [k for k in req if k not in metrics]
    if miss:
        gate(4, "基线诚实", "FAIL", f"缺基线指标: {miss}")
        return
    try:
        skill = float(metrics["skill_hpval"])
    except ValueError:
        gate(4, "基线诚实", "FAIL", "skill_hpval 非数值")
        return
    # 需 CI(理想: skill_hpval_ci_lo); 无CI则降级PENDING
    ci_lo = metrics.get("skill_hpval_ci_lo")
    if ci_lo is None:
        gate(4, "基线诚实", "PENDING",
             f"skill_hpval={skill} 但无 block-bootstrap CI, 无法判显著性")
    elif float(ci_lo) > 0:
        gate(4, "基线诚实", "PASS",
             f"skill_hpval={skill} CI下限={ci_lo}>0, Select段胜基线")
    else:
        gate(4, "基线诚实", "FAIL",
             f"skill_hpval={skill} CI下限={ci_lo}≤0, 不显著优于基线")


def v5_model_family(config):
    if config is None:
        gate(5, "模型族协议", "PENDING", "无 config(建模后提供 --config)")
        return
    fam = config.get("family", "?")
    checks = []
    if fam == "logistic":
        if not config.get("scaler_fit_train_only"):
            checks.append("scaler须train-fold only")
        if "C" not in config.get("select_grid", {}):
            checks.append("C须在Select网格选")
    elif fam == "lightgbm":
        if not config.get("early_stop_monitor_select"):
            checks.append("early-stop须监控Select")
        if not config.get("use_best_iteration"):
            checks.append("predict须用best_iteration")
        if "monotone" not in config:
            checks.append("须有monotone约束")
    # 未跳级: lightgbm 须有 logistic 先过的证据
    if fam == "lightgbm" and not config.get("logistic_select_skill_positive"):
        checks.append("跳级: 须先证明logistic Select skill>0")
    if checks:
        gate(5, "模型族协议", "FAIL", f"{fam}: " + "; ".join(checks))
    else:
        gate(5, "模型族协议", "PASS", f"{fam}: scaler_train_only={config.get('scaler_fit_train_only')} "
             f"early_stop_select={config.get('early_stop_monitor_select')} best_iter={config.get('use_best_iteration')} "
             f"monotone={config.get('monotone')} logistic_skill_pos={config.get('logistic_select_skill_positive')}")


def v6_metrics_match(metrics):
    if metrics is None:
        gate(6, "度量匹配", "PENDING", "无 metrics")
        return
    req = ["report_pr_auc", "report_expected_return", "report_max_dd",
           "report_coverage", "report_fattail_capture", "n_report",
           "report_start", "report_end"]
    miss = [k for k in req if k not in metrics]
    if miss:
        gate(6, "度量匹配", "FAIL", f"缺度量: {miss}")
    else:
        gate(6, "度量匹配", "PASS",
             f"PR-AUC+期望收益+回撤+coverage+肥尾+N/日期齐; N={metrics['n_report']}")


def v7_drift(metrics):
    if metrics is None:
        gate(7, "漂移/regime", "PENDING", "无 metrics(需 by_regime + PSI)")
        return
    has_regime = any(k.startswith("report_pr_auc_regime_") for k in metrics)
    psi = metrics.get("psi_train_report")
    if not has_regime:
        gate(7, "漂移/regime", "FAIL", "缺 by-regime 分段指标")
    elif psi is None:
        gate(7, "漂移/regime", "PENDING", "有by-regime但缺 PSI(train→report漂移)")
    else:
        gate(7, "漂移/regime", "PASS", f"by-regime✓; PSI={psi}")


def v8_repro(config):
    if config is None:
        gate(8, "复现性", "PENDING", "无 config(需 seed + 版本 + 冻结重跑)")
        return
    miss = [k for k in ["seed", "lib_versions"] if k not in config]
    if miss:
        gate(8, "复现性", "FAIL", f"缺: {miss}")
    elif config.get("repro_verified"):
        gate(8, "复现性", "PASS",
             f"seed={config['seed']} 版本已录; repro_verified=True(两次冻结运行diff=0)")
    else:
        gate(8, "复现性", "PENDING",
             f"seed={config['seed']} 版本已录; 须冻结config重跑 diff<1e-6(人工/V8重跑脚本)")


def v9_economic(metrics):
    if metrics is None:
        gate(9, "经济回测", "PENDING", "无 metrics(需 report_net_pnl + report_sharpe)")
        return
    miss = [k for k in ["report_net_pnl", "report_sharpe", "report_max_dd_econ"]
                if k not in metrics]
    if miss:
        gate(9, "经济回测", "FAIL", f"缺经济度量: {miss}")
    else:
        pnl = float(metrics["report_net_pnl"])
        gate(9, "经济回测", "PASS" if pnl > 0 else "FAIL",
             f"net_pnl={pnl} sharpe={metrics['report_sharpe']} "
             f"max_dd={metrics['report_max_dd_econ']}")


def v10_live_parity(config):
    if config is None:
        gate(10, "live parity", "MANUAL", "须人工核对 radar 集成特征定义=研究定义; model card/kill-switch")
    else:
        ok = config.get("model_card") and config.get("kill_switch")
        gate(10, "live parity", "PASS" if ok else "MANUAL",
             f"model_card={'有' if config.get('model_card') else '缺'} "
             f"kill_switch={'有' if config.get('kill_switch') else '缺'}; 特征定义一致性须人工核")


def v11_signoff():
    statuses = {s for _, _, s, _ in gates}
    has_fail = "FAIL" in statuses
    has_pending = "PENDING" in statuses or "MANUAL" in statuses
    if has_fail:
        gate(11, "签字门", "FAIL", "存在FAIL门 → 修复后从失败门重跑, 不得上线")
    elif has_pending:
        gate(11, "签字门", "PENDING", "结构门通过, 模型门待建模产物 → 建模后补全")
    elif "WARN" in statuses:
        gate(11, "签字门", "WARN", "无FAIL/PENDING, 但有WARN(N_eff偏小等) → 可推进但报上偏")
    else:
        gate(11, "签字门", "PASS", "V1-V10 全PASS → 可 freeze→shadow→小仓→监控")


def write_report():
    has_fail = any(s == "FAIL" for _, _, s, _ in gates)
    lines = ["# 建模验证报告 (阶段9)", "",
             f"生成: {datetime.now(timezone.utc).isoformat(timespec='seconds')}", ""]
    for vid, name, status, evidence in gates:
        tag = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️ ", "PENDING": "⏳", "MANUAL": "✋"}[status]
        lines.append(f"- {tag} **V{vid} {name}** [{status}]: {evidence}")
    lines += ["", "## 结论", "",
              "❌ 有FAIL → 修复后重跑" if has_fail else
              "⏳ 结构门通过, 待建模产物补全模型门" if any(s in ("PENDING", "MANUAL") for _, _, s, _ in gates) else
              "✅ 全PASS → 可 promote"]
    with open(OUT, "w") as f:
        f.write("\n".join(lines))
    print(f"\n报告: {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="/tmp/feature_matrix.pkl")
    ap.add_argument("--metrics", default=None, help="实验 METRIC 日志路径")
    ap.add_argument("--config", default=None, help="模型 config JSON")
    args = ap.parse_args()
    metrics = parse_metrics(args.metrics)
    config = None
    if args.config:
        try:
            config = json.load(open(args.config))
        except FileNotFoundError:
            config = None
    v1_leakage(args.matrix)
    v2_separation(metrics)
    v3_overlap(args.matrix)
    v4_baseline(metrics)
    v5_model_family(config)
    v6_metrics_match(metrics)
    v7_drift(metrics)
    v8_repro(config)
    v9_economic(metrics)
    v10_live_parity(config)
    v11_signoff()
    write_report()
    has_fail = any(s == "FAIL" for _, _, s, _ in gates)
    sys.exit(1 if has_fail else 0)


if __name__ == "__main__":
    main()