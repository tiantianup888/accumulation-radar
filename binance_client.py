"""Binance USDT-M Futures REST client with API weight management.

设计要点（来自实测响应头 + 403事故复盘）：
  1. 标准fapi端点(klines/premiumIndex/exchangeInfo)由 x-mbx-used-weight-1m 跟踪，限额~2400/分钟。
     klines权重随limit: 100→1, 500→2, 1000→5, 1500→10。
  2. fundingRate 与 /futures/data/*(openInterestHist/多空比) 不返回权重头，走独立WAF限制桶，
     并发或高频请求会触发 IP 级 403 封禁（本次项目的403事故根因）。
     → 这两类端点必须严格串行 + 节流，绝不能并发。
  3. 429/418 = 限流，按 Retry-After 退避；403 = WAF封禁，长冷却(指数退避,最多600s)。

权重管理策略：
  - 全局串行（max_concurrency=1），杜绝并发触发WAF。
  - 每端点组设最小间隔节流。
  - 跟踪响应头 x-mbx-used-weight-1m；接近预算时睡到下一分钟窗口。
  - 429/418: 指数退避 + Retry-After。
  - 403: 指数冷却(60→120→240→...→600)，并记录；连续403超过阈值则放弃该端点。
  - 400/404: 视为无数据，返回None不重试。
"""
import time
import logging
import requests

log = logging.getLogger("binance_client")


class BinanceFuturesClient:
    BASE = "https://fapi.binance.com"

    # 权重预算(留余量,真实限额约2400/分钟)
    WEIGHT_BUDGET_1M = 1800
    # 节流: 标准端点最小间隔; 敏感端点(funding/data)更长
    STD_MIN_DELAY = 0.20
    DATA_MIN_DELAY = 1.0   # fundingRate + /futures/data/*: WAF对持续>1req/s敏感, 用1s
    # 403 WAF冷却(指数): ban是transient(~1-2min自清), 但封禁期内快速重试会续期
    # 故base取120s, 翻倍至30min, 给足自清时间
    COOLDOWN_BASE = 120
    COOLDOWN_MAX = 1800
    MAX_403_STREAK = 5
    # 主动节流: 实测fundingRate连续~100次/2min触发WAF ban
    # 故每DATA_PAUSE_EVERY次data调用, 主动休息DATA_PAUSE_SEC秒, 避免触阈值
    DATA_PAUSE_EVERY = 80
    DATA_PAUSE_SEC = 30

    def __init__(self, weight_budget_1m=None, std_delay=None, data_delay=None):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "accumulation-radar/1.0"})
        self.weight_used_1m = 0
        self.weight_budget = weight_budget_1m or self.WEIGHT_BUDGET_1M
        self.std_delay = std_delay or self.STD_MIN_DELAY
        self.data_delay = data_delay or self.DATA_MIN_DELAY
        self.last_request_ts = 0.0
        self.minute_window_start = time.time()
        self.streak_403 = 0
        self.data_call_count = 0  # data端点累计调用, 用于主动节流
        self.stats = {"ok": 0, "429": 0, "403": 0, "empty": 0, "err": 0}

    # ---- 内部节流 ----
    def _throttle(self, is_data):
        delay = self.data_delay if is_data else self.std_delay
        elapsed = time.time() - self.last_request_ts
        if elapsed < delay:
            time.sleep(delay - elapsed)

    def _proactive_data_pause(self):
        """每DATA_PAUSE_EVERY次data调用后主动休息, 避免累积触发WAF ban阈值。"""
        self.data_call_count += 1
        if self.data_call_count % self.DATA_PAUSE_EVERY == 0:
            log.info("主动节流: 已发 %d 次data请求, 休息 %ss 避WAF", self.data_call_count, self.DATA_PAUSE_SEC)
            time.sleep(self.DATA_PAUSE_SEC)

    def _check_weight_budget(self):
        """接近权重预算时，睡到下一分钟窗口。"""
        if self.weight_used_1m >= self.weight_budget:
            # 不知精确重置点，保守睡到当前分钟结束+1s
            now = time.time()
            wait = 61 - (now - self.minute_window_start) % 60
            log.info("权重达到预算 %d/%d，冷却 %.0fs", self.weight_used_1m, self.weight_budget, wait)
            time.sleep(max(wait, 5))
            self.weight_used_1m = 0
            self.minute_window_start = time.time()

    def get(self, path, params=None, is_data=False, max_retries=6):
        """发起GET。is_data=True 对 fundingRate 与 /futures/data/* 必须传True(强节流)。
        返回 json(list/dict) 或 None(无数据/耗尽重试)。"""
        url = self.BASE + path
        params = params or {}
        for attempt in range(max_retries):
            if is_data:
                self._proactive_data_pause()
            self._throttle(is_data)
            self._check_weight_budget()
            try:
                r = self.session.get(url, params=params, timeout=20)
            except requests.RequestException as e:
                self.stats["err"] += 1
                log.warning("网络异常 %s: %s", path, e)
                time.sleep(min(30, 2 ** attempt))
                continue
            self.last_request_ts = time.time()
            # 更新权重跟踪
            w = r.headers.get("x-mbx-used-weight-1m")
            if w:
                try:
                    self.weight_used_1m = max(self.weight_used_1m, int(w))
                except ValueError:
                    pass
            # 处理状态码
            if r.status_code == 200:
                self.streak_403 = 0
                self.stats["ok"] += 1
                try:
                    return r.json()
                except ValueError:
                    self.stats["err"] += 1
                    log.warning("JSON解析失败 %s", path)
                    return None
            if r.status_code in (429, 418):
                self.stats["429"] += 1
                retry_after = r.headers.get("Retry-After")
                wait = int(retry_after) if retry_after and retry_after.isdigit() else min(60, 5 * (2 ** attempt))
                log.warning("限流 %s %s, 退避 %ss", path, r.status_code, wait)
                time.sleep(wait)
                continue
            if r.status_code == 403:
                self.stats["403"] += 1
                self.streak_403 += 1
                if self.streak_403 > self.MAX_403_STREAK:
                    log.error("连续403达 %d 次，放弃 %s", self.streak_403, path)
                    return None
                cooldown = min(self.COOLDOWN_MAX, self.COOLDOWN_BASE * (2 ** (self.streak_403 - 1)))
                log.warning("WAF封禁403 %s，冷却 %ss (连续第%d次)", path, cooldown, self.streak_403)
                time.sleep(cooldown)
                continue
            if r.status_code in (400, 404):
                self.stats["empty"] += 1
                return None
            # 其他5xx等
            log.warning("非预期状态 %s %s: %s", path, r.status_code, r.text[:120])
            self.stats["err"] += 1
            time.sleep(min(30, 2 ** attempt))
        return None

    # ---- 业务封装 ----
    def exchange_info(self):
        return self.get("/fapi/v1/exchangeInfo", is_data=False)

    def funding_rate(self, symbol, start_ms=None, end_ms=None, limit=1000):
        """资金费率历史(8h一条)。limit最大1000。可分页用startTime/endTime。"""
        p = {"symbol": symbol, "limit": limit}
        if start_ms is not None:
            p["startTime"] = start_ms
        if end_ms is not None:
            p["endTime"] = end_ms
        return self.get("/fapi/v1/fundingRate", params=p, is_data=True)

    def funding_rate_full(self, symbol, start_ms, end_ms):
        """分页拉取 [start_ms, end_ms] 全部资金费率。每页1000条(8h≈333天)。"""
        out = []
        cur = start_ms
        while cur < end_ms:
            batch = self.funding_rate(symbol, start_ms=cur, end_ms=end_ms, limit=1000)
            if not batch:
                break
            out.extend(batch)
            last_t = int(batch[-1]["fundingTime"])
            if len(batch) < 1000 or last_t >= end_ms:
                break
            cur = last_t + 1  # 下一页从最后一条之后
        return out

    def klines(self, symbol, interval="1d", limit=1000, start_ms=None, end_ms=None):
        p = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_ms is not None:
            p["startTime"] = start_ms
        if end_ms is not None:
            p["endTime"] = end_ms
        return self.get("/fapi/v1/klines", params=p, is_data=False)

    def open_interest_hist(self, symbol, period="1d", limit=30):
        return self.get("/futures/data/openInterestHist",
                        params={"symbol": symbol, "period": period, "limit": limit}, is_data=True)


def get_perpetual_symbols(client):
    """从exchangeInfo提取所有USDT永续合约标的。"""
    info = client.exchange_info()
    if not info:
        return []
    syms = []
    for s in info.get("symbols", []):
        if s.get("contractType") == "PERPETUAL" and s.get("quoteAsset") == "USDT" \
                and s.get("status") == "TRADING":
            syms.append(s["symbol"])
    return sorted(syms)