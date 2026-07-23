"""FastAPI — 雷达信号查询 + PnL 实时计算 + 策略统计 + 基准对照。

PnL 公式（常规 A）：unrealized_pnl_pct = (current - entry) / entry * 100
平仓策略：Buy & Hold（exit=hold，唯一实现，其余预留）。
"""
import os
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, Query, HTTPException, Depends
import db
import pnl

DB_PATH = os.getenv("DB_PATH", "data/accumulation.db")
app = FastAPI(title="庄家收筹雷达 API", version="1.0")


def get_conn():
    conn = db.connect(DB_PATH)
    db.init_db(conn)
    return conn


# 连接池：每次请求开新连接（SQLite 轻量），用完关闭。
# 为简单起见用依赖注入。
def conn_dep():
    conn = get_conn()
    try:
        yield conn
    finally:
        conn.close()


def row_to_dict(r):
    d = dict(r)
    import json
    for k in ("tags", "fp_signatures", "ml_features"):
        if d.get(k):
            try:
                d[k] = json.loads(d[k])
            except Exception:
                pass
    return d


@app.get("/health")
def health(conn=Depends(conn_dep)):
    return {
        "status": "ok",
        "db": DB_PATH,
        "signal_count": db.signal_count(conn),
        "last_signal_time": db.last_signal_time(conn),
    }


@app.get("/signals")
def list_signals(
    strategy: Optional[str] = None,
    frm: Optional[str] = Query(None, alias="from"),
    to: Optional[str] = None,
    min_score: Optional[float] = None,
    coin: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    conn=Depends(conn_dep),
):
    rows = db.get_signals(conn, strategy=strategy, frm=frm, to=to,
                         min_score=min_score, coin=coin, limit=limit, offset=offset)
    return {"count": len(rows), "items": [row_to_dict(r) for r in rows]}


@app.get("/signals/{signal_id}")
def get_signal(signal_id: int, conn=Depends(conn_dep)):
    r = db.get_signal(conn, signal_id)
    if not r:
        raise HTTPException(404, "signal not found")
    return row_to_dict(r)


@app.get("/signals/{signal_id}/pnl")
def get_signal_pnl(signal_id: int, live: bool = True, exit: str = "hold", conn=Depends(conn_dep)):
    r = db.get_signal(conn, signal_id)
    if not r:
        raise HTTPException(404, "signal not found")
    return pnl.compute_pnl(conn, r, live=live, exit=exit)


@app.get("/strategies/{name}/stats")
def strategy_stats(
    name: str,
    frm: Optional[str] = Query(None, alias="from"),
    to: Optional[str] = None,
    live: bool = False,
    conn=Depends(conn_dep),
):
    return pnl.compute_strategy_stats(conn, name, frm=frm, to=to, live=live)


@app.get("/strategies/{name}/signals")
def strategy_signals(name: str, limit: int = 50, offset: int = 0, conn=Depends(conn_dep)):
    rows = db.get_signals(conn, strategy=name, limit=limit, offset=offset)
    return {"strategy": name, "count": len(rows), "items": [row_to_dict(r) for r in rows]}


@app.get("/coins/{coin}")
def coin_signals(coin: str, limit: int = 100, conn=Depends(conn_dep)):
    rows = db.get_signals_by_coin(conn, coin, limit=limit)
    out = []
    for r in rows:
        p = pnl.compute_pnl(conn, r, live=False, exit="hold")
        out.append({"signal": row_to_dict(r), "pnl": p})
    return {"coin": coin, "count": len(out), "items": out}


@app.get("/benchmark")
def benchmark_stats(conn=Depends(conn_dep)):
    return pnl.compute_benchmark_stats(conn, live=False)


@app.get("/fingerprints")
def fingerprint_library():
    """列出庄家拉盘指纹库内容。"""
    import fingerprint as fp_lib
    fps = fp_lib.load_library()
    return {
        "count": len(fps),
        "library": [
            {"fp_id": f.fp_id, "mode": f.mode, "source_coin": f.source_coin,
             "pump_date": f.pump_date, "pump_chg_24h": f.pump_chg_24h,
             "pump_chg_7d": f.pump_chg_7d, "signatures": f.signatures}
            for f in fps
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)