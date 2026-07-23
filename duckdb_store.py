"""DuckDB 存储：量价(K线)/funding/oi 三类市场数据。

替代 SQLite 作为市场原始数据仓库。DuckDB 列存+压缩，适合分析型查询。
去重写入用 staging表+批量DELETE+INSERT(避免 INSERT OR REPLACE 逐行PK检查的性能陷阱)。
"""
import duckdb
import pandas as pd


class DuckDBStore:
    def __init__(self, path="data/market.duckdb"):
        self.path = path
        self.con = duckdb.connect(path)
        self._init_schema()

    def _init_schema(self):
        self.con.execute("""
        CREATE TABLE IF NOT EXISTS klines_daily(
            symbol VARCHAR NOT NULL, open_time TIMESTAMP NOT NULL,
            open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
            volume DOUBLE, close_time TIMESTAMP, quote_volume DOUBLE, trades BIGINT,
            taker_buy_volume DOUBLE, taker_buy_quote DOUBLE,
            PRIMARY KEY(symbol, open_time)
        )""")
        self.con.execute("""
        CREATE TABLE IF NOT EXISTS funding_rate(
            symbol VARCHAR NOT NULL, funding_time TIMESTAMP NOT NULL,
            funding_rate DOUBLE, mark_price DOUBLE,
            PRIMARY KEY(symbol, funding_time)
        )""")
        self.con.execute("""
        CREATE TABLE IF NOT EXISTS open_interest(
            symbol VARCHAR NOT NULL, ts TIMESTAMP NOT NULL,
            sum_open_interest DOUBLE, sum_open_interest_value DOUBLE,
            PRIMARY KEY(symbol, ts)
        )""")
        self.con.execute("""
        CREATE TABLE IF NOT EXISTS fetch_state(
            symbol VARCHAR, kind VARCHAR, last_fetched_ms BIGINT,
            last_run_ts TIMESTAMP, rows_added BIGINT,
            PRIMARY KEY(symbol, kind)
        )""")
        self.con.execute("""
        CREATE TABLE IF NOT EXISTS symbol_meta(
            symbol VARCHAR NOT NULL,
            status VARCHAR NOT NULL,            -- TRADING / SETTLING / DELISTED
            contract_type VARCHAR,
            onboard_date TIMESTAMP,           -- 上市时间(来自exchangeInfo)
            last_kline_date TIMESTAMP,        -- 最后一根日K(退市币=退市日近似)
            source VARCHAR,                   -- current_exchangeInfo / wayback_snap
            updated_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(symbol)
        )""")

    def _bulk_upsert(self, table, pk_cols, df):
        """通用批量幂等写入: staging表 -> 批量DELETE冲突PK -> 批量INSERT。"""
        if df is None or len(df) == 0:
            return 0
        # 注册为临时视图
        view = "_stg_" + table
        self.con.register(view, df)
        # 删除将被覆盖的PK
        pk_list = ", ".join(pk_cols)
        where_in = f"({pk_list}) IN (SELECT {pk_list} FROM {view})"
        self.con.execute(f"DELETE FROM {table} WHERE {where_in}")
        # 批量插入
        cols = ", ".join(df.columns)
        self.con.execute(f"INSERT INTO {table}({cols}) SELECT {cols} FROM {view}")
        self.con.unregister(view)
        return len(df)

    # ---- 量价日K ----
    def upsert_klines(self, rows):
        """rows: list of (symbol, open_time(datetime), o,h,l,c, vol, close_time(datetime), qv, trades, tbv, tbqv)"""
        if not rows:
            return 0
        df = pd.DataFrame(rows, columns=["symbol", "open_time", "open", "high", "low",
                                           "close", "volume", "close_time", "quote_volume",
                                           "trades", "taker_buy_volume", "taker_buy_quote"])
        return self._bulk_upsert("klines_daily", ["symbol", "open_time"], df)

    # ---- funding ----
    def upsert_funding(self, rows):
        """rows: list of (symbol, funding_time(datetime), funding_rate, mark_price)"""
        if not rows:
            return 0
        df = pd.DataFrame(rows, columns=["symbol", "funding_time", "funding_rate", "mark_price"])
        return self._bulk_upsert("funding_rate", ["symbol", "funding_time"], df)

    # ---- oi ----
    def upsert_oi(self, rows):
        """rows: list of (symbol, ts(datetime), sum_oi, sum_oi_value)"""
        if not rows:
            return 0
        df = pd.DataFrame(rows, columns=["symbol", "ts", "sum_open_interest", "sum_open_interest_value"])
        return self._bulk_upsert("open_interest", ["symbol", "ts"], df)

    def mark_fetch(self, symbol, kind, last_ms, rows_added):
        self.con.execute("""
        INSERT OR REPLACE INTO fetch_state VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?)
        """, [symbol, kind, last_ms, rows_added])

    # ---- symbol_meta (universe-as-of) ----
    def upsert_symbol_meta(self, rows):
        """rows: list of (symbol, status, contract_type, onboard_date, last_kline_date, source)"""
        if not rows:
            return 0
        df = pd.DataFrame(rows, columns=["symbol", "status", "contract_type",
                                           "onboard_date", "last_kline_date", "source"])
        return self._bulk_upsert("symbol_meta", ["symbol"], df)

    def universe_as_of(self, t_decision_ms, lookback_days=14, label_days=5):
        """返回在 t_decision 时合法可交易的symbol集合(§13 universe-as-of):
        onboard_date ≤ t_decision - lookback, 且 last_kline_date ≥ t_decision + label(标签可完整实现)。"""
        from datetime import datetime, timezone
        td = datetime.fromtimestamp(t_decision_ms / 1000, tz=timezone.utc).replace(tzinfo=None)
        return [r[0] for r in self.con.execute("""
            SELECT symbol FROM symbol_meta
            WHERE onboard_date <= ?
              AND last_kline_date >= ?
              AND status IN ('TRADING','SETTLING')
        """, [td, td]).fetchall()]

    # ---- 查询 ----
    def count(self, table):
        return self.con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    def funding_symbols(self):
        return [r[0] for r in self.con.execute(
            "SELECT DISTINCT symbol FROM funding_rate ORDER BY symbol").fetchall()]

    def funding_date_range(self, symbol=None):
        q = "SELECT MIN(funding_time), MAX(funding_time), COUNT(*) FROM funding_rate"
        params = []
        if symbol:
            q += " WHERE symbol = ?"; params = [symbol]
        return self.con.execute(q, params).fetchone()