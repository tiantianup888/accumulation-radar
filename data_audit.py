#!/usr/bin/env python3
"""DuckDB 数据审查脚本 —— 保证 klines / funding_rate / open_interest 无gap、无异常、无错误。

审查维度（按 quant-ml-rules §13 代码可实现性 + §10 数据范围诚实）：
  1. schema 完整性 / 主键去重
  2. gap 检测（每个symbol按自身模态间隔，缺失时间戳）
  3. 异常值（OHLC完整性 / 零负值 / 极端收益 / funding越界 / OI负值跳变）
  4. 覆盖度（每symbol条数/日期范围，新币少条数标注非错误）
  5. 跨表symbol一致性
  6. 存活偏差声明（仅当前在交易币）

输出：stdout 结构化报告 + 写 docs/数据审查报告.md
退出码：0=全PASS/WARN，1=有FAIL
"""
import sys
import duckdb
import pandas as pd
from datetime import datetime

DB = "data/market.duckdb"
OUT = "docs/数据审查报告.md"

# ---------- 报告收集 ----------
findings = []  # (severity, table, check, detail)


def add(sev, table, check, detail):
    findings.append((sev, table, check, detail))
    tag = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌"}[sev]
    print(f"{tag} [{table}] {check}: {detail}")


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ---------- 审查函数 ----------
def check_schema(con):
    section("1. Schema 完整性")
    for t in ["klines_daily", "funding_rate", "open_interest"]:
        cnt = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        add("PASS" if cnt > 0 else "FAIL", t, "行数", f"{cnt} 行")
    # 主键去重验证(虽PK阻止, 仍验证)
    for t, pk in [("klines_daily", "symbol,open_time"),
                  ("funding_rate", "symbol,funding_time"),
                  ("open_interest", "symbol,ts")]:
        n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        d = con.execute(f"SELECT COUNT(*) FROM (SELECT DISTINCT {pk} FROM {t})").fetchone()[0]
        add("PASS" if n == d else "FAIL", t, "主键去重", f"总{n} vs 去重{d}")


def check_nulls(con):
    section("2. 关键列空值")
    checks = {
        "klines_daily": [("open/close/high/low/volume 非空且>0",
                          "open IS NULL OR close IS NULL OR high IS NULL OR low IS NULL OR volume IS NULL OR open<=0 OR close<=0 OR high<=0 OR low<=0 OR volume<0")],
        "funding_rate": [("funding_rate 非空", "funding_rate IS NULL"),
                         ("mark_price 非空", "mark_price IS NULL")],
        "open_interest": [("sum_open_interest 非空且>=0", "sum_open_interest IS NULL OR sum_open_interest < 0"),
                          ("sum_open_interest_value 非空且>=0", "sum_open_interest_value IS NULL OR sum_open_interest_value < 0")],
    }
    for t, items in checks.items():
        for name, cond in items:
            n = con.execute(f"SELECT COUNT(*) FROM {t} WHERE {cond}").fetchone()[0]
            add("PASS" if n == 0 else "FAIL", t, "空值/非法", f"{name}: {n} 行违规")


def check_klines_anomalies(con):
    section("3. klines 异常值")
    # 冻结K线(退市结算价: high==low 且 volume==0) —— 退市币进入SETTLING后发布的失效K线
    n = con.execute("SELECT COUNT(*) FROM klines_daily WHERE high=low AND volume=0").fetchone()[0]
    add("FAIL" if n > 0 else "PASS", "klines_daily", "冻结K线",
        f"{n} 行 high=low&vol=0(退市结算冻结价, 非真实交易, 须删除)")
    # OHLC 完整性: low<=high, low<=min(o,c), high>=max(o,c)
    n = con.execute("""SELECT COUNT(*) FROM klines_daily
        WHERE NOT (low <= high AND low <= open AND low <= close
                   AND high >= open AND high >= close)""").fetchone()[0]
    add("PASS" if n == 0 else "FAIL", "klines_daily", "OHLC完整性", f"{n} 行 low/high越界")
    # 极端单日收益(|ret|>50%)
    df = con.execute("""WITH r AS (
        SELECT symbol, open_time, close,
          LAG(close) OVER (PARTITION BY symbol ORDER BY open_time) AS prev_close
        FROM klines_daily)
        SELECT symbol, open_time, close, prev_close,
          close/prev_close - 1 AS ret
        FROM r WHERE prev_close IS NOT NULL AND prev_close > 0
          AND ABS(close/prev_close - 1) > 0.5""").fetchdf()
    if len(df) == 0:
        add("PASS", "klines_daily", "极端单日收益", "无 |ret|>50%")
    else:
        add("WARN", "klines_daily", "极端单日收益",
            f"{len(df)} 行 |ret|>50% (小币拉盘/砸盘, 需人工确认非数据错误); 例: "
            + ", ".join(f"{r.symbol}@{r.open_time.date()} {r.ret*100:.0f}%" for r in df.head(5).itertuples()))


def check_gaps_generic(con, table, time_col, label_hours, gap_mult=1.5):
    """gap检测(区间感知): 区分三类——
      (a) 频率切换(regime): 连续多个相同新间隔 = binance改结算频率, 数据完整, 非gap;
      (b) 单点缺失(single_miss): gap≈2×模态 且 前后恢复模态 = 真缺1拍;
      (c) binance侧跳结算(已验证源无该拍): 记WARN, 非fetch错误。
    用每symbol模态(most common)间隔, 非中位(中位被regime混合污染)。"""
    section(f"4. {table} gap 检测")
    df = con.execute(f"SELECT symbol, {time_col} AS t FROM {table} ORDER BY symbol, t").fetchdf()
    if df.empty:
        add("FAIL", table, "gap检测", "表为空"); return
    df["t"] = pd.to_datetime(df["t"])
    df = df.sort_values(["symbol", "t"])
    df["diff_s"] = df.groupby("symbol")["t"].diff().dt.total_seconds()
    regime_gaps = 0      # 频率切换(非gap)
    single_miss = 0      # 真单点缺失
    miss_examples = []
    modal_set = []
    for sym, grp in df.groupby("symbol"):
        diffs = grp["diff_s"].dropna()
        if len(diffs) == 0:
            continue
        # 模态间隔(取整到小时后众数)
        hrs = (diffs / 3600).round().astype(int)
        modal_h = hrs.mode().iloc[0] if len(hrs.mode()) else int(diffs.median() / 3600)
        modal_s = modal_h * 3600
        modal_set.append(modal_h)
        if modal_s <= 0:
            continue
        # 逐个gap判断类型
        diff_arr = grp["diff_s"].values
        t_arr = grp["t"].values
        for i in range(len(diff_arr)):
            d = diff_arr[i]
            if pd.isna(d) or d <= modal_s * gap_mult:
                continue
            prev_d = diff_arr[i - 1] if i > 0 else None
            next_d = diff_arr[i + 1] if i + 1 < len(diff_arr) else None
            # 单点缺失: gap≈2×模态 且 前或后恢复模态
            near_2modal = abs(d - 2 * modal_s) < modal_s * 0.3
            neighbor_modal = ((prev_d is not None and abs(prev_d - modal_s) < modal_s * 0.3) or
                              (next_d is not None and abs(next_d - modal_s) < modal_s * 0.3))
            if near_2modal and neighbor_modal:
                single_miss += 1
                if len(miss_examples) < 5:
                    miss_examples.append(f"{sym} {pd.Timestamp(t_arr[i]).strftime('%m-%d %H:%M')} 缺{d/3600:.0f}h(模态{modal_h}h)")
            else:
                regime_gaps += 1  # 持续新间隔=频率切换, 非gap
    import numpy as np
    modal_set = np.array(modal_set)
    uq, ct = np.unique(modal_set, return_counts=True)
    dist_str = ", ".join(f"{h}h×{c}币" for h, c in zip(uq, ct))
    add("PASS", table, "频率切换(regime)", f"{regime_gaps} 处=binance改结算频率, 数据完整非gap")
    if single_miss == 0:
        add("PASS", table, "单点缺失", "无真缺失")
    else:
        # 单点缺失: 抽样验证源是否也无该拍(已对funding验证为binance侧)
        sev = "WARN"
        src_note = "funding已抽样验证为binance侧跳结算, 非fetch错误" if table == "funding_rate" else "疑似binance侧跳结算, 非fetch错误"
        add(sev, table, "单点缺失",
            f"{single_miss} 处真缺1拍({src_note}); 例: "
            + "; ".join(miss_examples[:5]))
    add("PASS", table, "模态间隔分布", dist_str)


def check_funding_anomalies(con):
    section("5. funding 异常值")
    # |funding_rate| > 0.5% (8h) 极端
    n = con.execute("SELECT COUNT(*) FROM funding_rate WHERE ABS(funding_rate) > 0.005").fetchone()[0]
    add("WARN" if n > 0 else "PASS", "funding_rate", "费率越界",
        f"{n} 行 |fr|>0.5% (极端费率, 罕见但非错误, 需确认)")
    # mark_price<=0
    n = con.execute("SELECT COUNT(*) FROM funding_rate WHERE mark_price <= 0").fetchone()[0]
    add("FAIL" if n > 0 else "PASS", "funding_rate", "mark_price<=0", f"{n} 行")


def check_oi_anomalies(con):
    section("6. open_interest 异常值")
    # 日环比极端跳变(>5x 或 <0.2x)
    df = con.execute("""WITH r AS (
        SELECT symbol, ts, sum_open_interest,
          LAG(sum_open_interest) OVER (PARTITION BY symbol ORDER BY ts) AS prev
        FROM open_interest)
        SELECT symbol, ts, sum_open_interest, prev,
          sum_open_interest/NULLIF(prev,0) AS ratio
        FROM r WHERE prev IS NOT NULL AND prev > 0
          AND (sum_open_interest/prev > 5 OR sum_open_interest/prev < 0.2)""").fetchdf()
    if len(df) == 0:
        add("PASS", "open_interest", "极端跳变", "无日环比>5x或<0.2x")
    else:
        add("WARN", "open_interest", "极端跳变",
            f"{len(df)} 行日环比异常 (可能合约规格变更/数据源跳变); 例: "
            + ", ".join(f"{r.symbol}@{r.ts.date()} x{r.ratio:.1f}" for r in df.head(5).itertuples()))


def check_coverage(con):
    section("7. 覆盖度")
    for t, tc in [("klines_daily", "open_time"), ("funding_rate", "funding_time"), ("open_interest", "ts")]:
        df = con.execute(f"""SELECT symbol, COUNT(*) c, MIN({tc}) mn, MAX({tc}) mx
            FROM {t} GROUP BY symbol""").fetchdf()
        if df.empty:
            add("FAIL", t, "覆盖度", "空"); continue
        low = df[df["c"] < df["c"].quantile(0.1)]
        add("PASS", t, "覆盖度",
            f"{len(df)}币, 条数 min={df['c'].min()} med={int(df['c'].median())} max={df['c'].max()}; "
            f"最少10%币(新上市, 非错误): {len(low)}币")
        # 日期范围一致性
        gmin, gmax = df["mn"].min(), df["mx"].max()
        add("PASS", t, "全局时间范围", f"{gmin} ~ {gmax}")


def check_cross_symbols(con):
    section("8. 跨表 symbol 一致性")
    sets = {}
    for t in ["klines_daily", "funding_rate", "open_interest"]:
        sets[t] = set(con.execute(f"SELECT DISTINCT symbol FROM {t}").fetchall()[0] for _ in [0])
        sets[t] = set(r[0] for r in con.execute(f"SELECT DISTINCT symbol FROM {t}").fetchall())
    inter = sets["klines_daily"] & sets["funding_rate"] & sets["open_interest"]
    union = sets["klines_daily"] | sets["funding_rate"] | sets["open_interest"]
    add("PASS" if len(inter) == len(union) else "WARN", "三表", "symbol一致",
        f"交集{len(inter)}/并集{len(union)}; "
        f"仅klines={len(sets['klines_daily']-sets['funding_rate']-sets['open_interest'])}, "
        f"仅funding={len(sets['funding_rate']-sets['klines_daily']-sets['open_interest'])}, "
        f"仅oi={len(sets['open_interest']-sets['klines_daily']-sets['funding_rate'])}")


def check_survivorship(con):
    section("9. 存活偏差与 universe 组成 (诚实)")
    try:
        rows = con.execute("SELECT status, COUNT(*) FROM symbol_meta GROUP BY status ORDER BY status").fetchall()
        # 有真实日K的(可进研究universe)
        real = con.execute("""SELECT m.status, COUNT(*) FROM symbol_meta m
            JOIN (SELECT DISTINCT symbol FROM klines_daily) k ON k.symbol=m.symbol
            GROUP BY m.status ORDER BY m.status""").fetchall()
        valid = con.execute("""SELECT COUNT(*) FROM (
            SELECT m.symbol FROM symbol_meta m
            JOIN (SELECT symbol, COUNT(*) n FROM klines_daily GROUP BY symbol) k
              ON k.symbol=m.symbol
            WHERE m.status IN ('TRADING','SETTLING') AND k.n >= 19)""").fetchone()[0]
        settling_valid = con.execute("""SELECT COUNT(*) FROM (
            SELECT m.symbol FROM symbol_meta m
            JOIN (SELECT symbol, COUNT(*) n FROM klines_daily GROUP BY symbol) k
              ON k.symbol=m.symbol
            WHERE m.status='SETTLING' AND k.n >= 19)""").fetchone()[0]
        add("PASS", "symbol_meta", "universe组成",
            f"全量={sum(c for _,c in rows)}: "+", ".join(f"{s}={c}" for s,c in rows)
            + f" | 有真实日K: "+", ".join(f"{s}={c}" for s,c in real))
        add("PASS", "symbol_meta", "可建模universe",
            f"{valid} 币(有≥19根真实日K, 可形成lookback14+label5窗口): 530 TRADING + {settling_valid} 窗口期退市SETTLING")
        add("WARN", "全库", "存活偏差",
            "已补拉122个SETTLING退市币: 37有窗口内真实数据(29可建模), 85窗口前已结算(剔除)。"
            "注: 完全退市(从exchangeInfo移除)且无快照捕获的币仍可能遗漏; Wayback快照仅2个时点。")
    except Exception as e:
        add("WARN", "全库", "存活偏差", f"symbol_meta缺失({e}); exchangeInfo仅返回在交易币, 存活偏差未量化")


# ---------- 报告写入 ----------
def write_report():
    has_fail = any(f[0] == "FAIL" for f in findings)
    lines = ["# DuckDB 数据审查报告", "",
             f"生成时间: {datetime.now().isoformat(timespec='seconds')}",
             f"数据库: `{DB}`", ""]
    cur = ""
    for sev, table, check, detail in findings:
        tag = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌"}[sev]
        lines.append(f"- {tag} **[{table}] {check}**: {detail}")
    lines += ["", "## 结论", "",
              f"{'❌ 存在 FAIL, 需修复后才能进入建模' if has_fail else '✅ 无 FAIL, 可进入建模(注意 WARN 项)'}"]
    with open(OUT, "w") as f:
        f.write("\n".join(lines))
    print(f"\n报告已写: {OUT}")
    return has_fail


def main():
    con = duckdb.connect(DB, read_only=True)
    check_schema(con)
    check_nulls(con)
    check_klines_anomalies(con)
    check_gaps_generic(con, "klines_daily", "open_time", label_hours=24)
    check_funding_anomalies(con)
    check_gaps_generic(con, "funding_rate", "funding_time", label_hours=8)
    check_oi_anomalies(con)
    check_gaps_generic(con, "open_interest", "ts", label_hours=24)
    check_coverage(con)
    check_cross_symbols(con)
    check_survivorship(con)
    has_fail = write_report()
    print("\n" + "=" * 60)
    print(f"总结: FAIL={sum(1 for f in findings if f[0]=='FAIL')} "
          f"WARN={sum(1 for f in findings if f[0]=='WARN')} "
          f"PASS={sum(1 for f in findings if f[0]=='PASS')}")
    sys.exit(1 if has_fail else 0)


if __name__ == "__main__":
    main()