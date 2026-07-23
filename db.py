"""共享数据库层 — SQLite schema + 读写 helper（radar 与 api 共用）。

设计要点：
- WAL 模式：api 读 / radar 写 并发安全
- 信号时刻数据快照不可变（entry_price 等），PnL 实时算
- 价格快照 UNIQUE(symbol, ts)，重跑幂等
- 跟踪期 5 天
"""
from __future__ import annotations
import json
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

CST = timezone(timedelta(hours=8))
TRACK_DAYS = 5
BENCHMARK_SAMPLE = 50  # 每小时随机抽样基准币数


def now_iso() -> str:
    return datetime.now(CST).isoformat(timespec="seconds")


def connect(db_path: str = "data/accumulation.db") -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30, isolation_level=None)  # autocommit
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """建表 + 索引（幂等）。"""
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS signals (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        signal_time     TEXT NOT NULL,
        scan_mode       TEXT NOT NULL,
        strategy        TEXT NOT NULL,
        rank_in_strategy INTEGER,
        score           REAL,
        symbol          TEXT NOT NULL,
        coin            TEXT NOT NULL,
        entry_price     REAL NOT NULL,
        px_chg_24h      REAL,
        funding_rate    REAL,
        funding_trend   TEXT,
        oi_usd          REAL,
        oi_delta_6h     REAL,
        oi_slope_5d     REAL,                 -- OI 5日斜率(实盘叠加加分)
        mcap            REAL,
        sideways_days   INTEGER,
        in_pool         INTEGER,
        amplitude_5d    REAL,                 -- 5日振幅%(信号①核心)
        spot_vol_ratio  REAL,                -- 现货量比(近24h/前7日)
        spot_premium    REAL,                -- 现货-合约溢价%
        heat            REAL,
        f_sc REAL, m_sc REAL, s_sc REAL, o_sc REAL,
        tags            TEXT,
        push_text       TEXT,
        created_at      TEXT DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_signals_time ON signals(signal_time DESC);
    CREATE INDEX IF NOT EXISTS idx_signals_strategy_time ON signals(strategy, signal_time DESC);
    CREATE INDEX IF NOT EXISTS idx_signals_symbol_time ON signals(symbol, signal_time DESC);

    CREATE TABLE IF NOT EXISTS price_snapshots (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol    TEXT NOT NULL,
        ts        TEXT NOT NULL,
        price     REAL NOT NULL,
        UNIQUE(symbol, ts)
    );
    CREATE INDEX IF NOT EXISTS idx_snap_symbol_ts ON price_snapshots(symbol, ts DESC);

    CREATE TABLE IF NOT EXISTS pool_state (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        snapshot_time TEXT NOT NULL,
        symbol        TEXT NOT NULL,
        sideways_days INTEGER, range_pct REAL, avg_vol REAL,
        pool_score    REAL, status TEXT,
        UNIQUE(snapshot_time, symbol)
    );

    CREATE TABLE IF NOT EXISTS benchmark_snapshots (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol    TEXT NOT NULL,
        ts        TEXT NOT NULL,
        price     REAL NOT NULL,
        cohort    TEXT NOT NULL,   -- 基准组标识，如 '2026-07-19T17:00'
        UNIQUE(symbol, ts)
    );
    CREATE INDEX IF NOT EXISTS idx_bench_symbol_ts ON benchmark_snapshots(symbol, ts DESC);
    CREATE INDEX IF NOT EXISTS idx_bench_cohort ON benchmark_snapshots(cohort);
    """)
    # 迁移：为旧库补齐重构后新增列（幂等）
    for col, decl in [
        ("oi_slope_5d", "REAL"), ("amplitude_5d", "REAL"),
        ("spot_vol_ratio", "REAL"), ("spot_premium", "REAL"),
        # 指纹库策略字段（庄家拉盘指纹匹配）
        ("fp_mode", "TEXT"),            # 命中的指纹模式 A/B/D
        ("fp_similarity", "REAL"),      # 与指纹的相似度
        ("fp_source", "TEXT"),          # 命中的来源指纹ID
        ("fp_signatures", "TEXT"),      # 命中时点行为签名(JSON)
    ]:
        try:
            conn.execute(f"ALTER TABLE signals ADD COLUMN {col} {decl}")
        except sqlite3.OperationalError:
            pass  # 列已存在


# ── 信号写入 ──────────────────────────────────────────────
def insert_signal(conn: sqlite3.Connection, s: dict) -> int:
    """插入一条信号，返回 id。s 字段见 schema。"""
    conn.execute(
        """INSERT INTO signals
        (signal_time, scan_mode, strategy, rank_in_strategy, score,
         symbol, coin, entry_price, px_chg_24h, funding_rate, funding_trend,
         oi_usd, oi_delta_6h, oi_slope_5d, mcap, sideways_days, in_pool,
         amplitude_5d, spot_vol_ratio, spot_premium, heat,
         f_sc, m_sc, s_sc, o_sc, tags, push_text,
         fp_mode, fp_similarity, fp_source, fp_signatures)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            s["signal_time"], s["scan_mode"], s["strategy"],
            s.get("rank_in_strategy"), s.get("score"),
            s["symbol"], s["coin"], s["entry_price"],
            s.get("px_chg_24h"), s.get("funding_rate"), s.get("funding_trend"),
            s.get("oi_usd"), s.get("oi_delta_6h"), s.get("oi_slope_5d"),
            s.get("mcap"), s.get("sideways_days"), int(bool(s.get("in_pool"))),
            s.get("amplitude_5d"), s.get("spot_vol_ratio"), s.get("spot_premium"),
            s.get("heat"),
            s.get("f_sc"), s.get("m_sc"), s.get("s_sc"), s.get("o_sc"),
            json.dumps(s["tags"], ensure_ascii=False) if s.get("tags") else None,
            s.get("push_text"),
            s.get("fp_mode"), s.get("fp_similarity"), s.get("fp_source"),
            json.dumps(s["fp_signatures"], ensure_ascii=False) if s.get("fp_signatures") else None,
        ),
    )
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


# ── 价格快照 ──────────────────────────────────────────────
def insert_snapshot(conn: sqlite3.Connection, symbol: str, ts: str, price: float) -> bool:
    """插入价格快照，幂等。返回是否新增。"""
    try:
        conn.execute(
            "INSERT OR IGNORE INTO price_snapshots(symbol, ts, price) VALUES (?,?,?)",
            (symbol, ts, price),
        )
        return conn.total_changes > 0
    except sqlite3.IntegrityError:
        return False


def insert_benchmark(conn, symbol, ts, price, cohort) -> bool:
    try:
        conn.execute(
            "INSERT OR IGNORE INTO benchmark_snapshots(symbol, ts, price, cohort) VALUES (?,?,?,?)",
            (symbol, ts, price, cohort),
        )
        return conn.total_changes > 0
    except sqlite3.IntegrityError:
        return False


def active_benchmark_group(conn) -> str | None:
    """返回当前活跃基准组 cohort（最早 entry 在跟踪期内），无则 None。"""
    cutoff = (datetime.now(CST) - timedelta(days=TRACK_DAYS)).isoformat(timespec="seconds")
    row = conn.execute(
        "SELECT cohort, MIN(ts) mints FROM benchmark_snapshots GROUP BY cohort ORDER BY cohort DESC LIMIT 1"
    ).fetchone()
    if row and row["mints"] and row["mints"] >= cutoff:
        return row["cohort"]
    return None


def benchmark_group_symbols(conn, cohort) -> list[str]:
    return [r["symbol"] for r in conn.execute(
        "SELECT DISTINCT symbol FROM benchmark_snapshots WHERE cohort=?", (cohort,)
    ).fetchall()]


def benchmark_group_rows(conn, cohort) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT symbol, ts, price FROM benchmark_snapshots WHERE cohort=? ORDER BY symbol, ts",
        (cohort,),
    ).fetchall()


# ── 查询 ──────────────────────────────────────────────────
def active_signal_symbols(conn: sqlite3.Connection) -> list[str]:
    """跟踪期内（5天）有信号的 symbol 去重列表，用于价格快照采集。"""
    cutoff = (datetime.now(CST) - timedelta(days=TRACK_DAYS)).isoformat(timespec="seconds")
    rows = conn.execute(
        "SELECT DISTINCT symbol FROM signals WHERE signal_time >= ? ORDER BY symbol",
        (cutoff,),
    ).fetchall()
    return [r["symbol"] for r in rows]


def get_signals(
    conn, strategy=None, frm=None, to=None, min_score=None, coin=None,
    limit=50, offset=0,
) -> list[sqlite3.Row]:
    sql = "SELECT * FROM signals WHERE 1=1"
    args = []
    if strategy:
        sql += " AND strategy=?"; args.append(strategy)
    if frm:
        sql += " AND signal_time>=?"; args.append(frm)
    if to:
        sql += " AND signal_time<=?"; args.append(to)
    if min_score is not None:
        sql += " AND score>=?"; args.append(min_score)
    if coin:
        sql += " AND coin=?"; args.append(coin)
    sql += " ORDER BY signal_time DESC, id DESC LIMIT ? OFFSET ?"
    args += [limit, offset]
    return conn.execute(sql, args).fetchall()


def get_signal(conn, signal_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM signals WHERE id=?", (signal_id,)).fetchone()


def get_signals_by_coin(conn, coin: str, limit=100) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM signals WHERE coin=? ORDER BY signal_time DESC, id DESC LIMIT ?",
        (coin, limit),
    ).fetchall()


def get_snapshots_since(conn, symbol: str, since_ts: str) -> list[sqlite3.Row]:
    """某信号之后该 symbol 的所有价格快照，按时间升序。"""
    return conn.execute(
        "SELECT ts, price FROM price_snapshots WHERE symbol=? AND ts>? ORDER BY ts ASC",
        (symbol, since_ts),
    ).fetchall()


def get_latest_snapshot(conn, symbol: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT ts, price FROM price_snapshots WHERE symbol=? ORDER BY ts DESC LIMIT 1",
        (symbol,),
    ).fetchone()


def get_snapshot_at(conn, symbol: str, target_ts: str) -> sqlite3.Row | None:
    """取 target_ts 时刻或之前最近的一条快照。"""
    return conn.execute(
        "SELECT ts, price FROM price_snapshots WHERE symbol=? AND ts<=? ORDER BY ts DESC LIMIT 1",
        (symbol, target_ts),
    ).fetchone()


def get_pool_state_latest(conn) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM pool_state WHERE snapshot_time=(SELECT MAX(snapshot_time) FROM pool_state)"
    ).fetchall()


def signal_count(conn) -> int:
    return conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]


def last_signal_time(conn) -> str | None:
    row = conn.execute("SELECT MAX(signal_time) FROM signals").fetchone()
    return row[0] if row else None


# ── 收筹池 ──────────────────────────────────────────────
def save_pool_state(conn, snapshot_time: str, rows: list[dict]) -> None:
    conn.executemany(
        """INSERT OR REPLACE INTO pool_state
        (snapshot_time, symbol, sideways_days, range_pct, avg_vol, pool_score, status)
        VALUES (?,?,?,?,?,?,?)""",
        [
            (snapshot_time, r["symbol"], r.get("sideways_days"), r.get("range_pct"),
             r.get("avg_vol"), r.get("pool_score"), r.get("status"))
            for r in rows
        ],
    )