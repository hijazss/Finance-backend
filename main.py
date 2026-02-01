import os
import re
import time
import math
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import requests
import quiverquant
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware


# =========================================================
# App
# =========================================================
app = FastAPI(title="Finance Signals Backend", version="4.2.3")

ALLOWED_ORIGINS = [
    "https://hijazss.github.io",
    "http://localhost:5173",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

QUIVER_TOKEN = os.getenv("QUIVER_TOKEN")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "").strip()

_party_re = re.compile(r"/\s*([DR])\b", re.IGNORECASE)

UA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36",
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}


@app.get("/")
def root():
    return {"status": "ok", "version": "4.2.3"}


# =========================================================
# Small TTL cache (Render-friendly)
# =========================================================
_CACHE: Dict[str, Tuple[float, Any]] = {}


def cache_get(key: str) -> Optional[Any]:
    now = time.time()
    rec = _CACHE.get(key)
    if not rec:
        return None
    exp, val = rec
    if now > exp:
        _CACHE.pop(key, None)
        return None
    return val


def cache_set(key: str, val: Any, ttl_seconds: int = 120) -> Any:
    _CACHE[key] = (time.time() + ttl_seconds, val)
    return val


# =========================================================
# Provider cooldowns to avoid repeated 429 / timeouts
# =========================================================
_PROVIDER_COOLDOWN_UNTIL: Dict[str, float] = {}


def _cooldown(provider: str, seconds: int) -> None:
    _PROVIDER_COOLDOWN_UNTIL[provider] = time.time() + float(seconds)


def _is_cooled_down(provider: str) -> bool:
    until = _PROVIDER_COOLDOWN_UNTIL.get(provider, 0.0)
    return time.time() < until


# =========================================================
# Helpers
# =========================================================
def pick_first(row: dict, keys: List[str], default=""):
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return default


def parse_dt_any(v: Any) -> Optional[datetime]:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)

    s = str(v).strip()
    if not s:
        return None

    s = s.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        pass

    fmts = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%m/%d/%Y",
        "%m/%d/%y",
        "%m/%d/%Y %H:%M:%S",
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue

    return None


def iso_date_only(dt: Optional[datetime]) -> str:
    return dt.date().isoformat() if dt else ""


def norm_party_from_any(row: dict) -> str:
    for k in ["Party", "party"]:
        v = row.get(k)
        if v:
            s = str(v).strip().upper()
            if s.startswith("D"):
                return "D"
            if s.startswith("R"):
                return "R"

    for k in ["Politician", "politician", "Representative", "Senator", "Name", "name"]:
        v = row.get(k)
        if not v:
            continue
        s = str(v)
        m = _party_re.search(s)
        if m:
            return m.group(1).upper()

        s2 = s.strip().upper()
        if s2.endswith(" D"):
            return "D"
        if s2.endswith(" R"):
            return "R"

    return ""


def norm_ticker(row: dict) -> str:
    for k in ["Ticker", "ticker", "Stock", "stock", "Symbol", "symbol"]:
        v = row.get(k)
        if not v:
            continue
        s = str(v).strip().upper()
        first = s.split()[0]
        if 1 <= len(first) <= 12 and first.replace(".", "").replace("-", "").isalnum():
            return first
        return s
    return ""


def tx_text(row: dict) -> str:
    for k in ["Transaction", "transaction", "TransactionType", "Type", "type"]:
        v = row.get(k)
        if v:
            return str(v).strip()
    return ""


def is_buy(tx: str) -> bool:
    s = (tx or "").lower()
    return ("purchase" in s) or ("buy" in s)


def is_sell(tx: str) -> bool:
    s = (tx or "").lower()
    return ("sale" in s) or ("sell" in s) or ("sold" in s)


def row_best_dt(row: dict) -> Optional[datetime]:
    traded = pick_first(row, ["Traded", "traded", "TransactionDate", "transaction_date"], "")
    filed = pick_first(row, ["Filed", "filed", "ReportDate", "report_date", "Date", "date"], "")
    return parse_dt_any(traded) or parse_dt_any(filed)


def interleave(items: List[dict], limit: int = 140) -> List[dict]:
    d = [x for x in items if x.get("party") == "D"]
    r = [x for x in items if x.get("party") == "R"]
    out: List[dict] = []
    for i in range(0, max(len(d), len(r))):
        if i < len(d):
            out.append(d[i])
        if i < len(r):
            out.append(r[i])
        if len(out) >= limit:
            break
    return out


def to_card(x: dict, kind: str) -> dict:
    dem = 1 if x["party"] == "D" else 0
    rep = 1 if x["party"] == "R" else 0
    who = x.get("politician", "")
    last = x.get("filed") or x.get("traded") or ""
    return {
        "ticker": x.get("ticker", ""),
        "companyName": who,
        "demBuyers": dem if kind == "BUY" else 0,
        "repBuyers": rep if kind == "BUY" else 0,
        "demSellers": dem if kind == "SELL" else 0,
        "repSellers": rep if kind == "SELL" else 0,
        "lastFiledAt": last,
        "strength": kind,
        "chamber": x.get("chamber", ""),
        "amountRange": x.get("amountRange", ""),
        "traded": x.get("traded", ""),
        "filed": x.get("filed", ""),
        "description": x.get("description", ""),
    }


# =========================================================
# Crypto detection (disclosures)
# =========================================================
TOP_COINS = ["BTC", "ETH", "SOL", "LINK", "XRP", "ADA", "DOGE", "AVAX", "MATIC", "BNB"]

CRYPTO_ALIASES: Dict[str, List[str]] = {
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "eth", "ether"],
    "SOL": ["solana", "sol"],
    "LINK": ["chainlink", "link"],
    "XRP": ["xrp", "ripple"],
    "ADA": ["cardano", "ada"],
    "DOGE": ["dogecoin", "doge"],
    "AVAX": ["avalanche", "avax"],
    "MATIC": ["polygon", "matic"],
    "BNB": ["bnb", "binance coin"],
}

CRYPTO_RELATED_TICKERS = set([
    "IBIT", "FBTC", "ARKB", "BITB", "BTCO", "HODL", "GBTC", "BITO",
    "ETHE", "ETHA",
    "COIN", "MSTR", "RIOT", "MARA", "HUT", "CLSK",
])

TICKER_TO_COINS: Dict[str, List[str]] = {
    "IBIT": ["BTC"], "FBTC": ["BTC"], "ARKB": ["BTC"], "BITB": ["BTC"], "BTCO": ["BTC"], "HODL": ["BTC"],
    "GBTC": ["BTC"], "BITO": ["BTC"],
    "ETHE": ["ETH"], "ETHA": ["ETH"],
    "COIN": ["CRYPTO-LINKED"],
    "MSTR": ["BTC", "CRYPTO-LINKED"],
    "RIOT": ["BTC", "CRYPTO-LINKED"],
    "MARA": ["BTC", "CRYPTO-LINKED"],
    "HUT": ["BTC", "CRYPTO-LINKED"],
    "CLSK": ["BTC", "CRYPTO-LINKED"],
}

_CRYPTO_PATTERNS: Dict[str, List[re.Pattern]] = {}
for sym, words in CRYPTO_ALIASES.items():
    pats: List[re.Pattern] = []
    for w in words:
        if re.fullmatch(r"[a-z]{2,6}", w):
            pats.append(re.compile(rf"\b{re.escape(w)}\b", re.IGNORECASE))
        else:
            pats.append(re.compile(re.escape(w), re.IGNORECASE))
    _CRYPTO_PATTERNS[sym] = pats

_GENERIC_CRYPTO_HINTS = [
    re.compile(r"\bcryptocurrency\b", re.IGNORECASE),
    re.compile(r"\bdigital asset\b", re.IGNORECASE),
    re.compile(r"\bvirtual currency\b", re.IGNORECASE),
]


def collect_crypto_text(row: dict) -> str:
    fields = [
        "AssetDescription", "asset_description", "Description", "description",
        "Asset", "asset", "Name", "name", "Issuer", "issuer",
        "Ticker", "ticker", "Stock", "stock", "Symbol", "symbol",
        "Owner", "owner", "Type", "type",
    ]
    parts = []
    for f in fields:
        v = row.get(f)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            parts.append(s)
    return " | ".join(parts)


def detect_coins(text: str) -> List[str]:
    if not text:
        return []
    hits: List[str] = []
    for sym in TOP_COINS:
        for pat in _CRYPTO_PATTERNS.get(sym, []):
            if pat.search(text):
                hits.append(sym)
                break

    out: List[str] = []
    for sym in TOP_COINS:
        if sym in hits:
            out.append(sym)
    return out


def has_generic_crypto_hint(text: str) -> bool:
    if not text:
        return False
    return any(p.search(text) for p in _GENERIC_CRYPTO_HINTS)


def classify_crypto_trade(ticker: str, text: str) -> Tuple[bool, List[str], str]:
    t = (ticker or "").upper().strip()
    coins_from_text = detect_coins(text)
    is_related_ticker = bool(t and t in CRYPTO_RELATED_TICKERS)
    coins_from_ticker = TICKER_TO_COINS.get(t, []) if is_related_ticker else []

    coins = []
    merged = set(coins_from_text + coins_from_ticker)
    for sym in TOP_COINS:
        if sym in merged:
            coins.append(sym)
    if "CRYPTO-LINKED" in merged:
        coins.append("CRYPTO-LINKED")

    if coins_from_text:
        return True, coins, "direct"
    if is_related_ticker:
        return True, coins if coins else ["CRYPTO-LINKED"], "etf_or_proxy"
    if has_generic_crypto_hint(text):
        return True, coins if coins else ["CRYPTO"], "hint_only"

    return False, [], ""


def counts_to_list(m: Dict[str, int]) -> List[dict]:
    out = [{"symbol": k, "count": v} for k, v in m.items()]
    out.sort(key=lambda x: (-x["count"], x["symbol"]))
    return out


# =========================================================
# Market data providers (rate-limit safe)
# =========================================================
def _requests_get(url: str, params: Optional[dict] = None, timeout: int = 14) -> requests.Response:
    return requests.get(url, params=params, timeout=timeout, headers=UA_HEADERS)


def _yahoo_chart(symbol: str, range_str: str = "6mo", interval: str = "1d") -> dict:
    if _is_cooled_down("yahoo"):
        raise RuntimeError("Yahoo in cooldown (rate-limit backoff)")

    url = "https://query1.finance.yahoo.com/v8/finance/chart/" + quote_plus(symbol)
    params = {"range": range_str, "interval": interval}

    r = _requests_get(url, params=params, timeout=14)
    if r.status_code == 429:
        _cooldown("yahoo", 10 * 60)
        raise RuntimeError("Yahoo rate limited (HTTP 429)")
    r.raise_for_status()
    return r.json()


def _yahoo_closes(symbol: str, range_str: str = "6mo", interval: str = "1d") -> List[Tuple[datetime, float]]:
    j = _yahoo_chart(symbol, range_str=range_str, interval=interval)
    res = (j.get("chart") or {}).get("result") or []
    if not res:
        return []
    result = res[0]
    ts = result.get("timestamp") or []
    q = (result.get("indicators") or {}).get("quote") or []
    if not ts or not q:
        return []
    closes = (q[0] or {}).get("close") or []
    out: List[Tuple[datetime, float]] = []
    for t, c in zip(ts, closes):
        if c is None:
            continue
        dt = datetime.fromtimestamp(int(t), tz=timezone.utc)
        out.append((dt, float(c)))
    out.sort(key=lambda x: x[0])
    return out


def _stooq_daily_closes(symbol: str) -> List[Tuple[datetime, float]]:
    if _is_cooled_down("stooq"):
        raise RuntimeError("Stooq in cooldown")

    url = "https://stooq.com/q/d/l/"
    params = {"s": symbol, "i": "d"}

    last_err = None
    for attempt in range(3):
        try:
            r = _requests_get(url, params=params, timeout=12)
            if r.status_code >= 400:
                last_err = f"HTTP {r.status_code}"
                time.sleep(min(1.5, 0.5 * (attempt + 1)))
                continue

            lines = (r.text or "").strip().splitlines()
            if len(lines) < 3:
                last_err = "insufficient CSV rows"
                time.sleep(min(1.5, 0.5 * (attempt + 1)))
                continue

            out: List[Tuple[datetime, float]] = []
            for line in lines[1:]:
                parts = line.split(",")
                if len(parts) < 5:
                    continue
                d = parts[0].strip()
                c = parts[4].strip()
                if not d or not c or c.lower() == "null":
                    continue
                try:
                    dt = datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    out.append((dt, float(c)))
                except Exception:
                    continue

            out.sort(key=lambda x: x[0])
            if out:
                return out

            last_err = "parsed empty"
            time.sleep(min(1.5, 0.5 * (attempt + 1)))
        except requests.exceptions.Timeout as e:
            last_err = f"Timeout: {type(e).__name__}"
            time.sleep(min(1.5, 0.5 * (attempt + 1)))
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:160]}"
            time.sleep(min(1.5, 0.5 * (attempt + 1)))

    _cooldown("stooq", 90)
    raise RuntimeError(f"Stooq failed: {last_err or 'unknown'}")


def _finnhub_quote(symbol: str) -> dict:
    if not FINNHUB_API_KEY:
        raise RuntimeError("FINNHUB_API_KEY missing")
    if _is_cooled_down("finnhub"):
        raise RuntimeError("Finnhub in cooldown")

    url = "https://finnhub.io/api/v1/quote"
    params = {"symbol": symbol, "token": FINNHUB_API_KEY}
    r = _requests_get(url, params=params, timeout=10)
    if r.status_code == 429:
        _cooldown("finnhub", 5 * 60)
        raise RuntimeError("Finnhub rate limited (HTTP 429)")
    r.raise_for_status()
    return r.json() if r.text else {}


def _pct(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return 100.0 * (a / b - 1.0)


def _sma(vals: List[float], n: int) -> Optional[float]:
    if n <= 0 or len(vals) < n:
        return None
    return sum(vals[-n:]) / n


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


# =========================================================
# CNN Fear & Greed (best-effort)
# =========================================================
def _cnn_fear_greed_graphdata(date_str: Optional[str] = None) -> dict:
    d = date_str or datetime.now(timezone.utc).date().isoformat()
    url = f"https://production.dataviz.cnn.io/index/fearandgreed/graphdata/{d}"
    r = _requests_get(url, timeout=14)
    r.raise_for_status()
    return r.json()


@app.get("/market/fear-greed")
def market_fear_greed(date: Optional[str] = Query(default=None)):
    key = f"feargreed:{date or 'today'}"
    cached = cache_get(key)
    if cached is not None:
        return cached

    try:
        data = _cnn_fear_greed_graphdata(date)
        fg = (data or {}).get("fear_and_greed") or {}
        now_val = fg.get("now") or {}
        out = {
            "date": (data or {}).get("date") or (date or datetime.now(timezone.utc).date().isoformat()),
            "score": now_val.get("value"),
            "rating": now_val.get("valueText") or now_val.get("rating"),
            "raw": data,
        }
        return cache_set(key, out, ttl_seconds=600)
    except Exception as e:
        return cache_set(
            key,
            {
                "date": date or datetime.now(timezone.utc).date().isoformat(),
                "score": None,
                "rating": None,
                "error": f"{type(e).__name__}: {str(e)}",
            },
            ttl_seconds=120,
        )


# =========================================================
# Market Snapshot (cache + cooldown + fallbacks)
# =========================================================
def _compute_returns_from_closes(closes: List[Tuple[datetime, float]]) -> dict:
    if not closes or len(closes) < 2:
        return {"last": None, "ret1dPct": None, "ret5dPct": None, "ret1mPct": None}

    closes_sorted = sorted(closes, key=lambda x: x[0])
    vals = [c for _, c in closes_sorted]
    last = vals[-1]

    ret1d = _pct(vals[-1], vals[-2]) if len(vals) >= 2 else None
    ret5d = _pct(vals[-1], vals[-6]) if len(vals) >= 6 else None
    ret1m = _pct(vals[-1], vals[-22]) if len(vals) >= 22 else None

    return {
        "last": round(float(last), 4),
        "ret1dPct": round(float(ret1d), 4) if ret1d is not None else None,
        "ret5dPct": round(float(ret5d), 4) if ret5d is not None else None,
        "ret1mPct": round(float(ret1m), 4) if ret1m is not None else None,
    }


def _try_get_spy_closes(errors: List[str]) -> List[Tuple[datetime, float]]:
    if FINNHUB_API_KEY:
        try:
            q = _finnhub_quote("SPY")
            c = float(q.get("c") or 0.0)
            pc = float(q.get("pc") or 0.0)
            if c > 0 and pc > 0:
                now = datetime.now(timezone.utc)
                return [(now - timedelta(days=1), pc), (now, c)]
        except Exception as e:
            errors.append(f"Finnhub SPY: {type(e).__name__}: {str(e)}")

    try:
        return _yahoo_closes("SPY", range_str="3mo", interval="1d")
    except Exception as e:
        errors.append(f"Yahoo SPY: {type(e).__name__}: {str(e)}")

    try:
        return _stooq_daily_closes("spy.us")
    except Exception as e:
        errors.append(f"Stooq SPY: {type(e).__name__}: {str(e)}")

    return []


def _try_get_vix_closes(errors: List[str]) -> List[Tuple[datetime, float]]:
    if FINNHUB_API_KEY:
        try:
            q = _finnhub_quote("VIX")
            c = float(q.get("c") or 0.0)
            pc = float(q.get("pc") or 0.0)
            if c > 0 and pc > 0:
                now = datetime.now(timezone.utc)
                return [(now - timedelta(days=1), pc), (now, c)]
        except Exception as e:
            errors.append(f"Finnhub VIX: {type(e).__name__}: {str(e)}")

    try:
        return _yahoo_closes("^VIX", range_str="3mo", interval="1d")
    except Exception as e:
        errors.append(f"Yahoo VIX: {type(e).__name__}: {str(e)}")

    try:
        return _stooq_daily_closes("vix")
    except Exception as e:
        errors.append(f"Stooq VIX: {type(e).__name__}: {str(e)}")

    return []


@app.get("/market/snapshot")
def market_snapshot():
    key = "market:snapshot:v2"
    cached = cache_get(key)
    if cached is not None:
        return cached

    now = datetime.now(timezone.utc)
    out = {"date": now.date().isoformat(), "sp500": {}, "vix": {}, "fearGreed": {}, "errors": []}
    errors: List[str] = []

    spy_closes = _try_get_spy_closes(errors)
    spy_ret = _compute_returns_from_closes(spy_closes)
    out["sp500"] = {
        "symbol": "SPY",
        "last": spy_ret.get("last"),
        "ret1dPct": spy_ret.get("ret1dPct"),
        "ret5dPct": spy_ret.get("ret5dPct"),
        "ret1mPct": spy_ret.get("ret1mPct"),
    }

    vix_closes = _try_get_vix_closes(errors)
    if vix_closes and len(vix_closes) >= 2:
        closes = [c for _, c in sorted(vix_closes, key=lambda x: x[0])]
        out["vix"] = {
            "symbol": "^VIX",
            "last": round(closes[-1], 4),
            "chg1d": round(closes[-1] - closes[-2], 4),
        }
    else:
        out["vix"] = {"symbol": "^VIX", "last": None, "chg1d": None}

    try:
        fg = market_fear_greed(None)
        out["fearGreed"] = {"score": fg.get("score"), "rating": fg.get("rating")}
    except Exception as e:
        errors.append(f"FearGreed: {type(e).__name__}: {str(e)}")
        out["fearGreed"] = {"score": None, "rating": None}

    out["errors"] = errors

    last_good = cache_get("market:snapshot:last_good")
    if (out["sp500"].get("last") is None) and last_good:
        last_good = dict(last_good)
        last_good_errors = list(last_good.get("errors") or [])
        last_good_errors.extend(errors[:3])
        last_good["errors"] = last_good_errors
        return cache_set(key, last_good, ttl_seconds=90)

    if out["sp500"].get("last") is not None:
        cache_set("market:snapshot:last_good", out, ttl_seconds=3600)

    return cache_set(key, out, ttl_seconds=120)


# =========================================================
# Market Entry Index (caching + last-good fallback)
# =========================================================
@app.get("/market/entry")
def market_entry(window_days: int = Query(default=365, ge=30, le=365)):
    key = f"market:entry:v2:{window_days}"
    cached = cache_get(key)
    if cached is not None:
        return cached

    now = datetime.now(timezone.utc)
    err_notes: List[str] = []

    snap = market_snapshot()
    spy_last = (snap.get("sp500") or {}).get("last")
    vix_last = (snap.get("vix") or {}).get("last")

    spy: List[Tuple[datetime, float]] = []
    vix: List[Tuple[datetime, float]] = []

    try:
        spy = _yahoo_closes("SPY", range_str="1y", interval="1d")
    except Exception as e:
        err_notes.append(f"Yahoo SPY: {type(e).__name__}: {str(e)}")
        try:
            spy = _stooq_daily_closes("spy.us")
        except Exception as e2:
            err_notes.append(f"Stooq SPY: {type(e2).__name__}: {str(e2)}")
            spy = []

    try:
        vix = _yahoo_closes("^VIX", range_str="1y", interval="1d")
    except Exception as e:
        err_notes.append(f"Yahoo VIX: {type(e).__name__}: {str(e)}")
        try:
            vix = _stooq_daily_closes("vix")
        except Exception as e2:
            err_notes.append(f"Stooq VIX: {type(e2).__name__}: {str(e2)}")
            vix = []

    if len(spy) < 210 or len(vix) < 30:
        last_good = cache_get("market:entry:last_good")
        if last_good:
            last_good = dict(last_good)
            last_good["notes"] = (last_good.get("notes") or "") + " | " + (
                " | ".join(err_notes) if err_notes else "Insufficient market data."
            )
            last_good.setdefault("errors", [])
            last_good["errors"] = (last_good.get("errors") or []) + err_notes[:3]
            return cache_set(key, last_good, ttl_seconds=90)

        out = {
            "date": now.date().isoformat(),
            "score": 0,
            "regime": "NEUTRAL",
            "signal": "DATA UNAVAILABLE",
            "notes": " | ".join(err_notes) if err_notes else "Insufficient market data.",
            "components": {
                "spxTrend": 0.5,
                "vix": 0.5,
                "breadth": 0.5,
                "credit": 0.5,
                "rates": 0.5,
                "buffettProxy": 0.5,
            },
            "errors": err_notes,
        }
        return cache_set(key, out, ttl_seconds=90)

    _, spy_close = zip(*spy)
    _, vix_close = zip(*vix)

    price = float(spy_close[-1])
    if spy_last is not None:
        try:
            price = float(spy_last)
        except Exception:
            pass

    sma50 = _sma(list(spy_close), 50) or price
    sma200 = _sma(list(spy_close), 200) or price

    trend_cross = 1.0 if sma50 >= sma200 else 0.0
    price_vs_200 = _clamp01((price / sma200 - 0.90) / (1.10 - 0.90))
    spx_trend_01 = _clamp01(0.55 * trend_cross + 0.45 * price_vs_200)

    v = float(vix_close[-1])
    if vix_last is not None:
        try:
            v = float(vix_last)
        except Exception:
            pass

    vix_01 = _clamp01(1.0 - ((v - 12.0) / (35.0 - 12.0)))

    breadth_01 = spx_trend_01
    credit_01 = 0.55
    rates_01 = 0.50
    buffett_01 = 0.55

    score_01 = 0.65 * spx_trend_01 + 0.35 * vix_01
    score = int(round(100.0 * score_01))

    if score >= 75:
        regime = "RISK-ON"
        signal = "ACCUMULATE"
    elif score >= 55:
        regime = "NEUTRAL"
        signal = "ACCUMULATE SLOWLY"
    else:
        regime = "RISK-OFF"
        signal = "WAIT / SMALL DCA"

    notes = f"SPY={price:.2f} SMA50={sma50:.2f} SMA200={sma200:.2f} VIX={v:.2f}"
    if err_notes:
        notes = notes + " | " + " | ".join(err_notes)

    out = {
        "date": now.date().isoformat(),
        "score": score,
        "regime": regime,
        "signal": signal,
        "notes": notes,
        "components": {
            "spxTrend": float(spx_trend_01),
            "vix": float(vix_01),
            "breadth": float(breadth_01),
            "credit": float(credit_01),
            "rates": float(rates_01),
            "buffettProxy": float(buffett_01),
        },
        "errors": err_notes,
    }

    cache_set("market:entry:last_good", out, ttl_seconds=3600)
    return cache_set(key, out, ttl_seconds=180)


# =========================================================
# RSS + sentiment
# =========================================================
def _fetch_rss_items(url: str, timeout: int = 12, max_items: int = 30) -> List[dict]:
    """
    Robust RSS/Atom fetch with:
      - short retries for transient failures
      - tolerant XML parse
    """
    last_err: Optional[str] = None
    for attempt in range(2):
        try:
            r = _requests_get(url, timeout=timeout)
            if r.status_code in (429, 503, 502, 504):
                last_err = f"HTTP {r.status_code}"
                time.sleep(0.6 * (attempt + 1))
                continue
            r.raise_for_status()

            text = (r.text or "").strip()
            if not text:
                return []

            try:
                root = ET.fromstring(text)
            except Exception:
                # Some feeds include invalid control chars
                cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", text)
                root = ET.fromstring(cleaned)

            channel = root.find("channel")
            if channel is None:
                entries = root.findall("{http://www.w3.org/2005/Atom}entry")
                out = []
                for e in entries[:max_items]:
                    title = (e.findtext("{http://www.w3.org/2005/Atom}title") or "").strip()
                    link_el = e.find("{http://www.w3.org/2005/Atom}link")
                    link = (link_el.get("href") if link_el is not None else "") or ""
                    pub = (e.findtext("{http://www.w3.org/2005/Atom}updated") or "").strip()
                    out.append({"title": title, "link": link, "published": pub})
                return out

            out = []
            for item in channel.findall("item")[:max_items]:
                title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                pub = (item.findtext("pubDate") or "").strip()
                out.append({"title": title, "link": link, "published": pub})
            return out
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:200]}"
            time.sleep(0.6 * (attempt + 1))

    raise RuntimeError(last_err or "RSS fetch failed")


_POS_WORDS = set([
    "beat", "beats", "surge", "surges", "rally", "rallies", "gain", "gains", "up",
    "upgrade", "upgrades", "record", "strong", "bull", "bullish", "growth",
    "profit", "profits", "outperform", "buy", "wins", "win", "breakout"
])
_NEG_WORDS = set([
    "miss", "misses", "drop", "drops", "plunge", "plunges", "down",
    "downgrade", "downgrades", "warning", "weak", "bear", "bearish",
    "lawsuit", "probe", "investigation", "fraud", "loss", "losses",
    "cut", "cuts", "layoff", "layoffs", "recession", "crash"
])


def _headline_sentiment(headlines: List[str]) -> Dict[str, Any]:
    score = 0
    pos = 0
    neg = 0
    total = 0

    for h in headlines:
        if not h:
            continue
        total += 1
        tokens = re.findall(r"[a-zA-Z]+", h.lower())
        tset = set(tokens)
        p = len(tset & _POS_WORDS)
        n = len(tset & _NEG_WORDS)
        if p > n:
            pos += 1
            score += 1
        elif n > p:
            neg += 1
            score -= 1

    if total == 0:
        return {"label": "NEUTRAL", "score": 0, "pos": 0, "neg": 0, "total": 0}

    raw = score / max(1, total)
    gauge = int(round(50 + 50 * raw))
    gauge = max(0, min(100, gauge))

    if gauge >= 60:
        label = "POSITIVE"
    elif gauge <= 40:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"

    return {"label": label, "score": gauge, "pos": pos, "neg": neg, "total": total}


# =========================================================
# News briefing endpoints
# =========================================================
def _google_news_rss(query: str) -> str:
    q = quote_plus(query)
    return f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"


def _safe_news_last_good_key(tickers: str) -> str:
    t = ",".join([x.strip().upper() for x in (tickers or "").split(",") if x.strip()][:50])
    return f"news:brief:last_good:{t or 'default'}"


@app.get("/news/briefing")
def news_briefing(
    tickers: str = Query(default="SPY,QQQ,NVDA,AAPL,MSFT,AMZN,GOOGL,META,TSLA,BRK-B"),
    max_general: int = Query(default=55, ge=10, le=200),
    max_per_ticker: int = Query(default=6, ge=2, le=25),
):
    """
    Frontend expects:
      /news/briefing?tickers=...&max_general=55&max_per_ticker=6

    This returns a sector-like grouping using:
      - Google News RSS for general
      - Yahoo ticker RSS for watchlist tickers
    """
    key = f"news:brief:{tickers}:{max_general}:{max_per_ticker}"
    cached = cache_get(key)
    if cached is not None:
        return cached

    syms = [t.strip().upper() for t in (tickers or "").split(",") if t.strip()]
    syms = syms[:50]

    errors: Dict[str, List[str]] = {"general": [], "watchlist": []}

    # General pull
    general_queries = [
        "markets stocks macro inflation fed earnings",
        "AI semiconductors datacenter nvidia amd",
        "energy oil opec natural gas",
        "healthcare biotech fda clinical trial",
        "crypto bitcoin ethereum etf",
    ]

    general_items: List[dict] = []
    per_query = max(8, max_general // max(1, len(general_queries)))
    for q in general_queries:
        try:
            items = _fetch_rss_items(_google_news_rss(q), max_items=per_query)
            for x in items:
                x["bucket"] = "General"
            general_items.extend(items)
        except Exception as e:
            errors["general"].append(f"{type(e).__name__}: {str(e)}")

    # Dedup general
    seen = set()
    gen_ded = []
    for x in general_items:
        lk = (x.get("link") or "").strip()
        if not lk or lk in seen:
            continue
        seen.add(lk)
        gen_ded.append(x)
    gen_ded = gen_ded[:max_general]

    # Watchlist pull (Yahoo RSS)
    watch_items: List[dict] = []
    for t in syms:
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={quote_plus(t)}&region=US&lang=en-US"
        try:
            items = _fetch_rss_items(url, max_items=max_per_ticker)
            for x in items:
                x["bucket"] = "Watchlist"
                x["ticker"] = t
            watch_items.extend(items)
        except Exception as e:
            errors["watchlist"].append(f"{t}: {type(e).__name__}: {str(e)}")

    # Make simple "sector" groups by keyword buckets
    def bucketize(title: str) -> str:
        s = (title or "").lower()
        if any(w in s for w in ["nvidia", "amd", "chip", "semiconductor", "ai", "datacenter"]):
            return "AI"
        if any(w in s for w in ["biotech", "fda", "clinical", "medical", "health"]):
            return "Medical"
        if any(w in s for w in ["oil", "opec", "gas", "energy", "lng"]):
            return "Energy"
        if any(w in s for w in ["robot", "automation", "humanoid"]):
            return "Robotics"
        if any(w in s for w in ["crypto", "bitcoin", "ethereum", "etf", "blockchain"]):
            return "Crypto"
        return "General"

    sector_map: Dict[str, List[dict]] = {}
    for x in (gen_ded + watch_items):
        sec = bucketize(x.get("title", ""))
        sector_map.setdefault(sec, []).append(x)

    sectors_out = []
    total_headlines = 0
    for sec, items in sector_map.items():
        titles = [i.get("title", "") for i in items]
        sent = _headline_sentiment(titles)
        top = items[:12]
        total_headlines += len(items)
        sectors_out.append({
            "sector": sec,
            "count": len(items),
            "sentiment": sent,
            "summary": "",
            "implications_2_12_weeks": [],
            "headlines": top,
        })

    overall = _headline_sentiment([x.get("title", "") for x in (gen_ded[:20] + watch_items[:20])])

    out = {
        "date": datetime.now(timezone.utc).date().isoformat(),
        "overallSentiment": overall,
        "sectors": sorted(sectors_out, key=lambda x: (-int(x["count"]), x["sector"])),
        "sources": {
            "general": "Google News RSS (queries)",
            "watchlist": "Yahoo Finance RSS (per ticker)",
        },
        "errors": errors,
        "note": "Headline grouping for context. Not a prediction engine.",
    }

    # NEW: last-good fallback if feeds go empty (common when providers throttle or block)
    last_good_key = _safe_news_last_good_key(tickers)
    if total_headlines <= 0:
        last_good = cache_get(last_good_key)
        if last_good:
            lg = dict(last_good)
            lg_err = lg.get("errors") or {"general": [], "watchlist": []}
            try:
                lg_err = dict(lg_err)
            except Exception:
                lg_err = {"general": [], "watchlist": []}
            lg_err.setdefault("general", [])
            lg_err.setdefault("watchlist", [])
            lg_err["general"] = list(lg_err["general"]) + (errors.get("general") or [])[:2]
            lg_err["watchlist"] = list(lg_err["watchlist"]) + (errors.get("watchlist") or [])[:2]
            lg["errors"] = lg_err
            lg["note"] = (lg.get("note") or "") + " Using cached news briefing due to empty live pull."
            return cache_set(key, lg, ttl_seconds=120)

    # Save last good for 30 minutes if we have content
    if total_headlines > 0:
        cache_set(last_good_key, out, ttl_seconds=1800)

    return cache_set(key, out, ttl_seconds=180)


# =========================================================
# Congress endpoints your UI calls
# =========================================================
@app.get("/report/today")
def report_today(
    window_days: Optional[int] = Query(default=None, ge=1, le=365),
    horizon_days: Optional[int] = Query(default=None, ge=1, le=365),
):
    if not QUIVER_TOKEN:
        raise HTTPException(500, "QUIVER_TOKEN missing")

    days = window_days if window_days is not None else horizon_days if horizon_days is not None else 30

    now = datetime.now(timezone.utc)
    since = now - timedelta(days=days)

    q = quiverquant.quiver(QUIVER_TOKEN)
    df = q.congress_trading()

    if df is None or len(df) == 0:
        return {
            "date": now.date().isoformat(),
            "windowDays": days,
            "windowStart": since.date().isoformat(),
            "windowEnd": now.date().isoformat(),
            "convergence": [],
            "politicianBuys": [],
            "politicianSells": [],
            "crypto": {"buys": [], "sells": [], "rawBuys": [], "rawSells": [], "raw": []},
        }

    try:
        rows = df.to_dict(orient="records")
    except Exception:
        rows = list(df)

    buys: List[dict] = []
    sells: List[dict] = []

    crypto_counts_buy: Dict[str, int] = {}
    crypto_counts_sell: Dict[str, int] = {}
    crypto_raw: List[dict] = []
    crypto_raw_buys: List[dict] = []
    crypto_raw_sells: List[dict] = []

    for r in rows:
        best_dt = row_best_dt(r)
        if best_dt is None:
            continue
        if best_dt < since or best_dt > now:
            continue

        party = norm_party_from_any(r)
        if not party:
            continue

        tx = tx_text(r)
        kind = "BUY" if is_buy(tx) else "SELL" if is_sell(tx) else ""
        if not kind:
            continue

        ticker = norm_ticker(r)
        pol = str(pick_first(r, ["Politician", "politician", "Representative", "Senator", "Name", "name"], "")).strip()
        chamber = str(pick_first(r, ["Chamber", "chamber", "Office", "office"], "")).strip()
        amount = str(pick_first(r, ["Amount", "amount", "Range", "range", "AmountRange", "amount_range"], "")).strip()

        filed_dt = parse_dt_any(pick_first(r, ["Filed", "filed", "ReportDate", "report_date", "Date", "date"], ""))
        traded_dt = parse_dt_any(pick_first(r, ["Traded", "traded", "TransactionDate", "transaction_date"], ""))

        desc = str(pick_first(r, ["AssetDescription", "asset_description", "Description", "description", "Asset", "asset"], "")).strip()

        item = {
            "ticker": ticker,
            "party": party,
            "filed": iso_date_only(filed_dt),
            "traded": iso_date_only(traded_dt),
            "politician": pol,
            "chamber": chamber,
            "amountRange": amount,
            "best_dt": best_dt,
            "description": desc,
        }

        if ticker:
            if kind == "BUY":
                buys.append(item)
            else:
                sells.append(item)

        text_blob = collect_crypto_text(r)
        is_crypto, coins, crypto_kind = classify_crypto_trade(ticker, text_blob)

        if is_crypto:
            target_counts = crypto_counts_buy if kind == "BUY" else crypto_counts_sell
            for c in coins if coins else ["CRYPTO"]:
                target_counts[c] = target_counts.get(c, 0) + 1

            rec = {
                "kind": kind,
                "coins": coins if coins else ["CRYPTO"],
                "cryptoKind": crypto_kind,
                "ticker": (ticker or "").upper(),
                "description": desc,
                "party": party,
                "politician": pol,
                "chamber": chamber,
                "amountRange": amount,
                "traded": iso_date_only(traded_dt),
                "filed": iso_date_only(filed_dt),
            }
            crypto_raw.append(rec)
            if kind == "BUY":
                crypto_raw_buys.append(rec)
            else:
                crypto_raw_sells.append(rec)

    buys.sort(key=lambda x: x["best_dt"], reverse=True)
    sells.sort(key=lambda x: x["best_dt"], reverse=True)

    buy_cards = [to_card(x, "BUY") for x in interleave(buys, 140)]
    sell_cards = [to_card(x, "SELL") for x in interleave(sells, 140)]

    dem_buy = [x for x in buys if x["party"] == "D"]
    rep_buy = [x for x in buys if x["party"] == "R"]
    overlap = set(x["ticker"] for x in dem_buy if x["ticker"]) & set(x["ticker"] for x in rep_buy if x["ticker"])

    overlap_cards: List[dict] = []
    for t in overlap:
        dem_ct = sum(1 for x in dem_buy if x["ticker"] == t)
        rep_ct = sum(1 for x in rep_buy if x["ticker"] == t)
        latest_dt = None
        for x in buys:
            if x["ticker"] == t:
                if latest_dt is None or x["best_dt"] > latest_dt:
                    latest_dt = x["best_dt"]
        overlap_cards.append({
            "ticker": t,
            "companyName": "",
            "demBuyers": dem_ct,
            "repBuyers": rep_ct,
            "demSellers": 0,
            "repSellers": 0,
            "lastFiledAt": iso_date_only(latest_dt),
            "strength": "OVERLAP",
        })

    overlap_cards.sort(key=lambda x: (-(x["demBuyers"] + x["repBuyers"]), x["ticker"]))

    crypto_raw.sort(key=lambda x: (x.get("filed") or x.get("traded") or ""), reverse=True)
    crypto_raw_buys.sort(key=lambda x: (x.get("filed") or x.get("traded") or ""), reverse=True)
    crypto_raw_sells.sort(key=lambda x: (x.get("filed") or x.get("traded") or ""), reverse=True)

    crypto_payload = {
        "buys": counts_to_list(crypto_counts_buy),
        "sells": counts_to_list(crypto_counts_sell),
        "rawBuys": crypto_raw_buys[:250],
        "rawSells": crypto_raw_sells[:250],
        "raw": crypto_raw[:500],
        "topCoins": TOP_COINS,
    }

    return {
        "date": now.date().isoformat(),
        "windowDays": days,
        "windowStart": since.date().isoformat(),
        "windowEnd": now.date().isoformat(),
        "convergence": overlap_cards[:25],
        "politicianBuys": buy_cards,
        "politicianSells": sell_cards,
        "crypto": crypto_payload,
    }


@app.get("/report/holdings/common")
def report_holdings_common(
    window_days: int = Query(default=365, ge=30, le=365),
    top_n: int = Query(default=30, ge=5, le=100),
):
    if not QUIVER_TOKEN:
        raise HTTPException(500, "QUIVER_TOKEN missing")

    now = datetime.now(timezone.utc)
    since = now - timedelta(days=window_days)

    q = quiverquant.quiver(QUIVER_TOKEN)
    df = q.congress_trading()

    if df is None or len(df) == 0:
        return {
            "date": now.date().isoformat(),
            "windowDays": window_days,
            "windowStart": since.date().isoformat(),
            "windowEnd": now.date().isoformat(),
            "commonHoldings": [],
            "meta": {"uniquePoliticians": 0, "uniqueTickers": 0, "topN": top_n},
            "note": "No congress trading rows returned for this window.",
        }

    try:
        rows = df.to_dict(orient="records")
    except Exception:
        rows = list(df)

    per_ticker: Dict[str, set] = {}
    all_pol: set = set()

    for r in rows:
        best_dt = row_best_dt(r)
        if best_dt is None or best_dt < since or best_dt > now:
            continue

        ticker = norm_ticker(r)
        if not ticker:
            continue

        pol = str(pick_first(r, ["Politician", "politician", "Representative", "Senator", "Name", "name"], "")).strip()
        if not pol:
            continue

        t = ticker.upper()
        all_pol.add(pol)
        per_ticker.setdefault(t, set()).add(pol)

    common = [{"ticker": t, "companyName": "", "holders": len(pols)} for t, pols in per_ticker.items()]
    common.sort(key=lambda x: (-int(x["holders"]), x["ticker"]))
    common = common[:top_n]

    return {
        "date": now.date().isoformat(),
        "windowDays": window_days,
        "windowStart": since.date().isoformat(),
        "windowEnd": now.date().isoformat(),
        "commonHoldings": common,
        "meta": {"uniquePoliticians": len(all_pol), "uniqueTickers": len(per_ticker), "topN": top_n},
        "note": "Holdings are a proxy from disclosures: holders = unique members who disclosed activity in ticker within the window.",
    }


@app.get("/report/congress/daily")
def report_congress_daily(
    window_days: int = Query(default=14, ge=1, le=365),
    limit: int = Query(default=200, ge=50, le=1000),
):
    if not QUIVER_TOKEN:
        raise HTTPException(500, "QUIVER_TOKEN missing")

    now = datetime.now(timezone.utc)
    since = now - timedelta(days=window_days)

    q = quiverquant.quiver(QUIVER_TOKEN)
    df = q.congress_trading()

    if df is None or len(df) == 0:
        return {
            "date": now.date().isoformat(),
            "windowDays": window_days,
            "windowStart": since.date().isoformat(),
            "windowEnd": now.date().isoformat(),
            "days": [],
            "note": "No rows returned.",
        }

    try:
        rows = df.to_dict(orient="records")
    except Exception:
        rows = list(df)

    items: List[dict] = []
    for r in rows:
        best_dt = row_best_dt(r)
        if best_dt is None or best_dt < since or best_dt > now:
            continue

        party = norm_party_from_any(r)
        if not party:
            continue

        tx = tx_text(r)
        kind = "BUY" if is_buy(tx) else "SELL" if is_sell(tx) else ""
        if not kind:
            continue

        ticker = norm_ticker(r)
        pol = str(pick_first(r, ["Politician", "politician", "Representative", "Senator", "Name", "name"], "")).strip()
        chamber = str(pick_first(r, ["Chamber", "chamber", "Office", "office"], "")).strip()
        amount = str(pick_first(r, ["Amount", "amount", "Range", "range", "AmountRange", "amount_range"], "")).strip()

        filed_dt = parse_dt_any(pick_first(r, ["Filed", "filed", "ReportDate", "report_date", "Date", "date"], ""))
        traded_dt = parse_dt_any(pick_first(r, ["Traded", "traded", "TransactionDate", "transaction_date"], ""))

        desc = str(pick_first(r, ["AssetDescription", "asset_description", "Description", "description", "Asset", "asset"], "")).strip()

        cap_url = "https://www.capitoltrades.com/politicians/" + quote_plus(pol) if pol else ""
        ticker_url = "https://www.capitoltrades.com/trades?search=" + quote_plus(ticker) if ticker else ""

        items.append({
            "kind": kind,
            "ticker": (ticker or "").upper(),
            "politician": pol,
            "party": party,
            "chamber": chamber,
            "amountRange": amount,
            "traded": iso_date_only(traded_dt),
            "filed": iso_date_only(filed_dt),
            "description": desc,
            "bestDate": iso_date_only(best_dt),
            "links": {
                "capitoltrades_politician": cap_url,
                "capitoltrades_ticker": ticker_url,
                "quiver_congresstrading": "https://www.quiverquant.com/congresstrading/",
                "senate_efd_search": "https://efdsearch.senate.gov/search/home/",
                "house_disclosures": "https://disclosures-clerk.house.gov/",
            }
        })

    def sort_key(x):
        s = x.get("filed") or x.get("traded") or x.get("bestDate") or ""
        return s

    items.sort(key=sort_key, reverse=True)
    items = items[:limit]

    days: Dict[str, List[dict]] = {}
    for it in items:
        d = it.get("filed") or it.get("traded") or it.get("bestDate") or now.date().isoformat()
        days.setdefault(d, []).append(it)

    day_list = [{"date": d, "items": days[d]} for d in sorted(days.keys(), reverse=True)]

    return {
        "date": now.date().isoformat(),
        "windowDays": window_days,
        "windowStart": since.date().isoformat(),
        "windowEnd": now.date().isoformat(),
        "days": day_list,
        "note": "Grouped by filed date when available. Links are best-effort jump links.",
    }


# =========================================================
# Crypto News Briefing (endpoint your frontend calls)
# =========================================================
CRYPTO_OUTLETS = [
    {"name": "CoinDesk", "rss": "https://www.coindesk.com/arc/outboundfeeds/rss/"},
    {"name": "Cointelegraph", "rss": "https://cointelegraph.com/rss"},
    {"name": "Decrypt", "rss": "https://decrypt.co/feed"},
    {"name": "CryptoSlate", "rss": "https://cryptoslate.com/feed/"},
    {"name": "Bitcoin Magazine", "rss": "https://bitcoinmagazine.com/.rss/full/"},
]

COIN_CANON: Dict[str, Dict[str, Any]] = {
    "BTC": {"name": "Bitcoin", "strict_terms": [r"\bbitcoin\b", r"\bBTC\b"]},
    "ETH": {"name": "Ethereum", "strict_terms": [r"\bethereum\b", r"\bETH\b", r"\bether\b"]},
    "SOL": {"name": "Solana", "strict_terms": [r"\bsolana\b", r"\bSOL\b"]},
    "LINK": {"name": "Chainlink", "strict_terms": [r"\bchainlink\b", r"\bLINK\b"]},  # avoids generic "link"
    "XRP": {"name": "XRP", "strict_terms": [r"\bxrp\b", r"\bripple\b"]},
    "ADA": {"name": "Cardano", "strict_terms": [r"\bcardano\b", r"\bADA\b"]},
    "DOGE": {"name": "Dogecoin", "strict_terms": [r"\bdogecoin\b", r"\bDOGE\b"]},
    "AVAX": {"name": "Avalanche", "strict_terms": [r"\bavalanche\b", r"\bAVAX\b"]},
    "MATIC": {"name": "Polygon", "strict_terms": [r"\bpolygon\b", r"\bMATIC\b"]},
    "BNB": {"name": "BNB", "strict_terms": [r"\bBNB\b", r"\bbinance coin\b"]},
    "SHIB": {"name": "Shiba Inu", "strict_terms": [r"\bshib\b", r"\bshiba\b", r"\bshiba inu\b"]},
}

_CATALYST_KEYWORDS = [
    "ETF", "SEC", "lawsuit", "approval", "hack", "exploit", "upgrade",
    "fork", "mainnet", "airdrop", "staking", "rate", "Fed", "CPI",
    "FOMC", "earnings", "regulation", "ban",
]


def _clean_coin_list(coins: str, include_top_n: int) -> List[str]:
    raw = [c.strip().upper() for c in (coins or "").split(",") if c.strip()]
    out: List[str] = []
    for c in raw:
        if c in COIN_CANON and c not in out:
            out.append(c)

    if not out:
        base = list(COIN_CANON.keys())
        return base[: max(1, min(30, include_top_n))]

    if include_top_n and include_top_n > len(out):
        for c in list(COIN_CANON.keys()):
            if len(out) >= include_top_n:
                break
            if c not in out:
                out.append(c)

    return out[: max(1, min(30, include_top_n or 15))]


def _matches_coin(title: str, symbol: str) -> bool:
    t = (title or "").strip()
    if not t:
        return False
    canon = COIN_CANON.get(symbol)
    if not canon:
        return False
    for pat in canon["strict_terms"]:
        if re.search(pat, t, flags=re.IGNORECASE):
            return True
    return False


def _dedup_items(items: List[dict], max_items: int) -> List[dict]:
    seen = set()
    out = []
    for x in items:
        lk = (x.get("link") or "").strip()
        ttl = (x.get("title") or "").strip()
        key = lk or ttl
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(x)
        if len(out) >= max_items:
            break
    return out


def _summarize_titles(titles: List[str], coin_name: str) -> str:
    if not titles:
        return f"No major {coin_name} headlines in the current pull."
    blob = " ".join(titles[:12]).lower()
    themes = []
    if any(w in blob for w in ["etf", "sec", "approval", "lawsuit", "regulat"]):
        themes.append("regulatory headlines are in focus")
    if any(w in blob for w in ["hack", "exploit", "breach", "stolen"]):
        themes.append("security and incident risk is present")
    if any(w in blob for w in ["upgrade", "fork", "mainnet", "testnet", "staking"]):
        themes.append("protocol and network updates are being watched")
    if any(w in blob for w in ["whale", "flows", "exchange", "liquidation"]):
        themes.append("positioning and flows appear to be a driver")
    if not themes:
        themes = ["headlines are mixed and event-driven"]
    return f"{coin_name}: " + "; ".join(themes[:2]) + "."


def _crypto_last_good_key(coins_key: str) -> str:
    return f"crypto:brief:last_good:{coins_key or 'default'}"


@app.get("/crypto/news/briefing")
def crypto_news_briefing(
    coins: str = Query(default="BTC,ETH,LINK,SHIB"),
    include_top_n: int = Query(default=15, ge=5, le=30),
):
    """
    Returns JSON expected by your frontend:
    {
      date, note, errors, sources:{outlets:[...]},
      overallSentiment:{label,score},
      catalysts:[...],
      coins:[{symbol, summary, sentiment:{label,score}, headlines:[{title,link,published,source}]}]
    }
    """
    coin_list = _clean_coin_list(coins, include_top_n)
    coins_key = ",".join(coin_list)

    key = f"crypto:brief:{coins_key}:{include_top_n}"
    cached = cache_get(key)
    if cached is not None:
        return cached

    errors: List[str] = []
    all_items: List[dict] = []

    for outlet in CRYPTO_OUTLETS:
        try:
            items = _fetch_rss_items(outlet["rss"], max_items=50, timeout=12)
            for x in items:
                x["source"] = outlet["name"]
            all_items.extend(items)
        except Exception as e:
            errors.append(f"{outlet['name']}: {type(e).__name__}: {str(e)}")

    for sym in coin_list[: min(12, len(coin_list))]:
        try:
            nm = COIN_CANON.get(sym, {}).get("name", sym)
            q = f"{nm} {sym} crypto"
            g_items = _fetch_rss_items(_google_news_rss(q), max_items=18, timeout=12)
            for x in g_items:
                x["source"] = "Google News"
            all_items.extend(g_items)
        except Exception as e:
            errors.append(f"GoogleNews {sym}: {type(e).__name__}: {str(e)}")

    all_items = _dedup_items(all_items, max_items=260)

    coin_blocks = []
    all_titles_for_overall: List[str] = []
    catalysts: List[str] = []
    seen_cat = set()

    total_heads = 0
    for sym in coin_list:
        canon = COIN_CANON.get(sym, {"name": sym, "strict_terms": [rf"\b{re.escape(sym)}\b"]})
        nm = canon["name"]

        heads = []
        for x in all_items:
            title = x.get("title", "") or ""
            if _matches_coin(title, sym):
                heads.append({
                    "title": title,
                    "link": x.get("link", ""),
                    "published": x.get("published", ""),
                    "source": x.get("source", ""),
                    "bucket": x.get("source", ""),
                })

        heads = _dedup_items(heads, max_items=18)
        total_heads += len(heads)

        titles = [h["title"] for h in heads]
        all_titles_for_overall.extend(titles[:6])

        sent = _headline_sentiment(titles)
        summary = _summarize_titles(titles, nm)

        for t in titles[:10]:
            for kw in _CATALYST_KEYWORDS:
                if kw.lower() in (t or "").lower():
                    k = f"{sym}:{kw}"
                    if k not in seen_cat:
                        seen_cat.add(k)
                        catalysts.append(f"{sym}: watch {kw}-driven headlines (from current feed pull)")
                    break

        coin_blocks.append({
            "symbol": sym,
            "coin": nm,
            "summary": summary,
            "sentiment": {"label": sent.get("label", "NEUTRAL"), "score": int(sent.get("score", 50) or 50)},
            "headlines": heads,
        })

    overall = _headline_sentiment(all_titles_for_overall)

    out = {
        "date": datetime.now(timezone.utc).date().isoformat(),
        "note": "Crypto briefing is headline-based context. It highlights what could move price, but it is not predictive.",
        "errors": errors,
        "sources": {"outlets": [o["name"] for o in CRYPTO_OUTLETS] + ["Google News"]},
        "overallSentiment": {"label": overall.get("label", "NEUTRAL"), "score": int(overall.get("score", 50) or 50)},
        "catalysts": catalysts[:12],
        "coins": coin_blocks,
    }

    # NEW: last-good fallback if live pull goes empty
    lg_key = _crypto_last_good_key(coins_key)
    if total_heads <= 0:
        last_good = cache_get(lg_key)
        if last_good:
            lg = dict(last_good)
            lg["errors"] = list(lg.get("errors") or []) + errors[:3]
            lg["note"] = (lg.get("note") or "") + " Using cached crypto briefing due to empty live pull."
            return cache_set(key, lg, ttl_seconds=120)

    if total_heads > 0:
        cache_set(lg_key, out, ttl_seconds=1800)

    return cache_set(key, out, ttl_seconds=240)
