import os
import re
import time
import math
import json
import csv
import io
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
app = FastAPI(title="Finance Signals Backend", version="4.2.1")

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

_party_re = re.compile(r"/\s*([DR])\b", re.IGNORECASE)

UA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36",
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

# A single shared session helps with performance + connection reuse
SESSION = requests.Session()
SESSION.headers.update(UA_HEADERS)


@app.get("/")
def root():
    return {"status": "ok", "version": "4.2.1"}


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
# Crypto detection (unchanged)
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
# Market data (ROBUST): Stooq primary, Yahoo fallback, cached
# =========================================================
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


def _yahoo_chart(symbol: str, range_str: str = "6mo", interval: str = "1d") -> dict:
    url = "https://query1.finance.yahoo.com/v8/finance/chart/" + quote_plus(symbol)
    params = {"range": range_str, "interval": interval, "includePrePost": "false", "events": "div,splits"}
    r = SESSION.get(url, params=params, timeout=8)
    # If Yahoo rate limits, fail fast with a clear error
    if r.status_code == 429:
        raise RuntimeError("Yahoo rate limited (HTTP 429)")
    r.raise_for_status()
    return r.json()


def _yahoo_closes(symbol: str, range_str: str = "6mo", interval: str = "1d") -> List[Tuple[datetime, float]]:
    # Cache Yahoo per symbol+range+interval so the UI refresh button doesn't DOS Yahoo
    ckey = f"yahoo:{symbol}:{range_str}:{interval}"
    cached = cache_get(ckey)
    if cached is not None:
        return cached

    j = _yahoo_chart(symbol, range_str=range_str, interval=interval)
    res = (j.get("chart") or {}).get("result") or []
    if not res:
        return cache_set(ckey, [], ttl_seconds=300)

    result = res[0]
    ts = result.get("timestamp") or []
    q = (result.get("indicators") or {}).get("quote") or []
    if not ts or not q:
        return cache_set(ckey, [], ttl_seconds=300)

    closes = (q[0] or {}).get("close") or []
    out: List[Tuple[datetime, float]] = []
    for t, c in zip(ts, closes):
        if c is None:
            continue
        dt = datetime.fromtimestamp(int(t), tz=timezone.utc)
        out.append((dt, float(c)))
    out.sort(key=lambda x: x[0])
    return cache_set(ckey, out, ttl_seconds=600)


def _stooq_symbol(symbol: str) -> str:
    # Stooq uses lowercase. Indices use special mapping.
    s = (symbol or "").strip()
    if not s:
        return s
    if s.upper() == "^VIX":
        return "vix"
    # SPY is fine
    return s.lower()


def _stooq_closes(symbol: str, max_rows: int = 260) -> List[Tuple[datetime, float]]:
    """
    Stooq daily CSV (no key). Often more stable from cloud than Yahoo.
    Example: https://stooq.com/q/d/l/?s=spy.us&i=d
    """
    ckey = f"stooq:{symbol}"
    cached = cache_get(ckey)
    if cached is not None:
        return cached

    s = _stooq_symbol(symbol)
    if not s:
        return []

    # Stooq for US tickers uses .us. vix is index series.
    if s == "vix":
        url = "https://stooq.com/q/d/l/?s=vix&i=d"
    else:
        url = f"https://stooq.com/q/d/l/?s={quote_plus(s)}.us&i=d"

    r = SESSION.get(url, timeout=8)
    r.raise_for_status()

    text = r.text.strip()
    if not text or "No data" in text:
        return cache_set(ckey, [], ttl_seconds=600)

    out: List[Tuple[datetime, float]] = []
    f = io.StringIO(text)
    reader = csv.DictReader(f)
    for row in reader:
        d = row.get("Date") or row.get("date")
        c = row.get("Close") or row.get("close")
        if not d or not c:
            continue
        try:
            dt = datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            out.append((dt, float(c)))
        except Exception:
            continue

    out.sort(key=lambda x: x[0])
    if len(out) > max_rows:
        out = out[-max_rows:]

    return cache_set(ckey, out, ttl_seconds=600)


def _market_closes(symbol: str, preferred: str = "stooq", range_hint: str = "1y") -> Tuple[List[Tuple[datetime, float]], List[str]]:
    """
    Returns (closes, errors). Tries Stooq first then Yahoo (or vice versa).
    """
    errors: List[str] = []

    def try_stooq():
        try:
            return _stooq_closes(symbol), None
        except Exception as e:
            return [], f"Stooq: {type(e).__name__}: {str(e)}"

    def try_yahoo():
        try:
            # map range_hint to Yahoo ranges
            rmap = {"1y": "1y", "6mo": "6mo", "3mo": "3mo"}
            y_range = rmap.get(range_hint, "6mo")
            return _yahoo_closes(symbol, range_str=y_range, interval="1d"), None
        except Exception as e:
            return [], f"Yahoo: {type(e).__name__}: {str(e)}"

    if preferred == "yahoo":
        a, err = try_yahoo()
        if err:
            errors.append(err)
        if a:
            return a, errors
        b, err2 = try_stooq()
        if err2:
            errors.append(err2)
        return b, errors

    # default: stooq first
    a, err = try_stooq()
    if err:
        errors.append(err)
    if a:
        return a, errors
    b, err2 = try_yahoo()
    if err2:
        errors.append(err2)
    return b, errors


# =========================================================
# CNN Fear & Greed (robust: try recent dates)
# =========================================================
def _cnn_fear_greed_graphdata(date_str: str) -> dict:
    url = f"https://production.dataviz.cnn.io/index/fearandgreed/graphdata/{date_str}"
    r = SESSION.get(url, timeout=8)
    r.raise_for_status()
    return r.json()


@app.get("/market/fear-greed")
def market_fear_greed(date: Optional[str] = Query(default=None)):
    key = f"feargreed:{date or 'auto'}"
    cached = cache_get(key)
    if cached is not None:
        return cached

    # If date not provided, try today then prior few days (weekends/holiday issues happen)
    if date:
        try_dates = [date]
    else:
        today = datetime.now(timezone.utc).date()
        try_dates = [(today - timedelta(days=i)).isoformat() for i in range(0, 7)]

    last_err = None
    for d in try_dates:
        try:
            data = _cnn_fear_greed_graphdata(d)
            fg = (data or {}).get("fear_and_greed") or {}
            now_val = fg.get("now") or {}
            out = {
                "date": (data or {}).get("date") or d,
                "score": now_val.get("value"),
                "rating": now_val.get("valueText") or now_val.get("rating"),
                "raw": data,
            }
            return cache_set(key, out, ttl_seconds=600)
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)}"
            continue

    return cache_set(
        key,
        {"date": date or datetime.now(timezone.utc).date().isoformat(), "score": None, "rating": None, "error": last_err},
        ttl_seconds=120,
    )


@app.get("/market/snapshot")
def market_snapshot():
    """
    Quick market context for News tab:
    - SPY (proxy for S&P 500) level + 1D/5D/1M returns
    - VIX level
    - Fear & Greed score + label (best effort)
    """
    key = "market:snapshot"
    cached = cache_get(key)
    if cached is not None:
        return cached

    now = datetime.now(timezone.utc)
    out = {"date": now.date().isoformat(), "sp500": {}, "vix": {}, "fearGreed": {}, "errors": []}

    # SPY
    spy, errs = _market_closes("SPY", preferred="stooq", range_hint="3mo")
    out["errors"].extend(errs)
    if len(spy) >= 25:
        closes = [c for _, c in spy]
        out["sp500"] = {
            "symbol": "SPY",
            "last": round(closes[-1], 4),
            "ret1dPct": round(_pct(closes[-1], closes[-2]), 4) if len(closes) >= 2 else None,
            "ret5dPct": round(_pct(closes[-1], closes[-6]), 4) if len(closes) >= 6 else None,
            "ret1mPct": round(_pct(closes[-1], closes[-22]), 4) if len(closes) >= 22 else None,
            "source": "stooq->yahoo_fallback",
        }

    # VIX
    vix, errs2 = _market_closes("^VIX", preferred="stooq", range_hint="3mo")
    out["errors"].extend(errs2)
    if len(vix) >= 2:
        closes = [c for _, c in vix]
        out["vix"] = {
            "symbol": "^VIX",
            "last": round(closes[-1], 4),
            "chg1d": round(closes[-1] - closes[-2], 4),
            "source": "stooq->yahoo_fallback",
        }

    # Fear & Greed
    fg = market_fear_greed(None)
    out["fearGreed"] = {
        "score": fg.get("score"),
        "rating": fg.get("rating"),
        "date": fg.get("date"),
    }
    if fg.get("error"):
        out["errors"].append(f"FearGreed: {fg.get('error')}")

    return cache_set(key, out, ttl_seconds=300)


# =========================================================
# Market Entry Index (robust)
# =========================================================
@app.get("/market/entry")
def market_entry(window_days: int = Query(default=365, ge=30, le=365)):
    now = datetime.now(timezone.utc)

    errors: List[str] = []

    # Need ~200 trading days for SMA200. Pull as much as we can from stooq/yahoo.
    spy, errs = _market_closes("SPY", preferred="stooq", range_hint="1y")
    errors.extend(errs)
    vix, errs2 = _market_closes("^VIX", preferred="stooq", range_hint="6mo")
    errors.extend(errs2)

    if len(spy) < 210 or len(vix) < 30:
        return {
            "date": now.date().isoformat(),
            "score": 0,
            "regime": "NEUTRAL",
            "signal": "DATA UNAVAILABLE",
            "notes": " | ".join(errors) if errors else "Insufficient market data.",
            "components": {
                "spxTrend": 0.5,
                "vix": 0.5,
                "breadth": 0.5,
                "credit": 0.5,
                "rates": 0.5,
                "buffettProxy": 0.5,
            },
        }

    _, spy_close = zip(*spy)
    _, vix_close = zip(*vix)

    price = float(spy_close[-1])
    sma50 = _sma(list(spy_close), 50) or price
    sma200 = _sma(list(spy_close), 200) or price

    trend_cross = 1.0 if sma50 >= sma200 else 0.0
    price_vs_200 = _clamp01((price / sma200 - 0.90) / (1.10 - 0.90))
    spx_trend_01 = _clamp01(0.55 * trend_cross + 0.45 * price_vs_200)

    v = float(vix_close[-1])
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
    if errors:
        notes = notes + " | " + " | ".join(errors[:3])

    return {
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
    }


# =========================================================
# RSS + sentiment (your existing approach)
# =========================================================
def _fetch_rss_items(url: str, timeout: int = 12, max_items: int = 30) -> List[dict]:
    r = SESSION.get(url, timeout=timeout)
    r.raise_for_status()

    text = r.text.strip()
    root = ET.fromstring(text)

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


@app.get("/news/top")
def news_top(max_items: int = Query(default=30, ge=10, le=100)):
    feeds = [
        "https://www.yahoo.com/news/rss/finance",
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "https://www.marketwatch.com/rss/topstories",
    ]

    items: List[dict] = []
    errors: List[str] = []

    for f in feeds:
        try:
            it = _fetch_rss_items(f, max_items=max_items)
            for x in it:
                x["sourceFeed"] = f
            items.extend(it)
        except Exception as e:
            errors.append(f"{f} -> {type(e).__name__}: {str(e)}")

    seen = set()
    deduped = []
    for x in items:
        lk = (x.get("link") or "").strip()
        if not lk or lk in seen:
            continue
        seen.add(lk)
        deduped.append(x)

    deduped = deduped[:max_items]
    sentiment = _headline_sentiment([x.get("title", "") for x in deduped])

    return {
        "date": datetime.now(timezone.utc).date().isoformat(),
        "items": deduped,
        "sentiment": sentiment,
        "errors": errors,
        "note": "Sentiment is a lightweight headline-based gauge, not a price predictor.",
    }


@app.get("/news/watchlist")
def news_watchlist(
    tickers: str = Query(default="SPY,QQQ,NVDA,AAPL,MSFT,AMZN,GOOGL,META,TSLA,BRK-B"),
    max_items_per_ticker: int = Query(default=8, ge=3, le=25),
):
    syms = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    syms = syms[:40]

    all_items: Dict[str, List[dict]] = {}
    errors: List[str] = []

    for t in syms:
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={quote_plus(t)}&region=US&lang=en-US"
        try:
            it = _fetch_rss_items(url, max_items=max_items_per_ticker)
            for x in it:
                x["ticker"] = t
                x["sourceFeed"] = "yahoo_ticker_rss"
            all_items[t] = it
        except Exception as e:
            errors.append(f"{t}: {type(e).__name__}: {str(e)}")
            all_items[t] = []

    headlines = []
    for t in syms:
        headlines.extend([x.get("title", "") for x in all_items.get(t, [])])

    sentiment = _headline_sentiment(headlines)

    return {
        "date": datetime.now(timezone.utc).date().isoformat(),
        "tickers": syms,
        "itemsByTicker": all_items,
        "sentiment": sentiment,
        "errors": errors,
        "note": "Ticker news via Yahoo RSS. If a ticker has no feed items, it may be temporarily empty.",
    }


# =========================================================
# News briefing (your existing endpoint)
# =========================================================
DEFAULT_SECTORS = [
    "AI",
    "Medical",
    "Energy",
    "Robotics",
    "Infrastructure",
    "Semiconductors",
    "Cloud",
    "Cybersecurity",
    "Defense",
    "Financials",
    "Consumer",
]

SECTOR_QUERIES = {
    "AI": [
        "artificial intelligence OR AI chips OR foundation models OR inference OR datacenter AI",
        "NVIDIA OR NVDA OR AMD OR ARM OR hyperscalers AI",
    ],
    "Medical": [
        "biotech OR medtech OR medical devices OR FDA approval OR clinical trial",
        "healthcare AI OR radiology AI OR genomics",
    ],
    "Energy": [
        "oil OR OPEC OR crude OR LNG OR refinery",
        "solar OR wind OR nuclear OR grid OR battery storage",
    ],
    "Robotics": [
        "robotics OR humanoid robots OR automation OR industrial robots",
        "warehouse robots OR autonomous systems",
    ],
    "Infrastructure": [
        "infrastructure spending OR data center buildout OR power grid upgrade",
        "construction materials OR industrials",
    ],
    "Semiconductors": [
        "semiconductor OR chip shortage OR fab OR foundry OR lithography",
        "TSMC OR ASML OR Intel OR Samsung foundry",
    ],
    "Cloud": [
        "cloud computing OR AWS OR Azure OR Google Cloud OR enterprise software",
    ],
    "Cybersecurity": [
        "cybersecurity OR ransomware OR breach OR zero trust",
    ],
    "Defense": [
        "defense spending OR missiles OR drones OR aerospace defense",
    ],
    "Financials": [
        "banks OR credit OR regional banks OR liquidity OR net interest margin",
    ],
    "Consumer": [
        "retail OR consumer spending OR inflation impact OR pricing power",
    ],
}

def _google_news_rss(query: str) -> str:
    q = quote_plus(query)
    return f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"

_TICKER_MENTION_RE = re.compile(r"\b([A-Z]{1,5})\b")
_STOP_TICKERS = set(["A", "I", "AI", "US", "FED", "USA", "CEO", "EPS", "IPO", "ETF", "FDA", "SEC", "DOJ", "EU"])

def _extract_tickers_from_titles(titles: List[str], watchlist: List[str]) -> Dict[str, int]:
    wl = set([t.upper() for t in watchlist if t])
    counts: Dict[str, int] = {}
    for title in titles:
        for m in _TICKER_MENTION_RE.findall(title or ""):
            t = m.upper()
            if t in _STOP_TICKERS:
                continue
            if t in wl:
                counts[t] = counts.get(t, 0) + 1
    return counts

def _brief_paragraph_from_headlines(headlines: List[str], sector: str) -> str:
    if not headlines:
        return f"No major {sector} headlines in the current pull."
    h = " ".join(headlines[:10]).lower()
    themes = []
    def has_any(words):
        return any(w in h for w in words)

    if has_any(["earnings", "revenue", "guidance", "forecast", "margin"]):
        themes.append("earnings and guidance are a key driver")
    if has_any(["rates", "yield", "inflation", "fed", "cuts", "hike"]):
        themes.append("rates and inflation expectations remain influential")
    if has_any(["data center", "datacenter", "gpu", "chips", "semiconductor", "foundry"]):
        themes.append("compute supply chain and data center buildout are in focus")
    if has_any(["regulation", "doj", "sec", "antitrust", "probe", "lawsuit"]):
        themes.append("regulatory and legal risk is present")
    if has_any(["geopolitical", "china", "taiwan", "ukraine", "middle east"]):
        themes.append("geopolitical risk is affecting sentiment")
    if has_any(["capex", "spending", "buildout", "infrastructure", "grid"]):
        themes.append("capex and buildout signals are showing up")

    if not themes:
        themes = ["headlines are mixed and mostly event-driven"]

    s = "; ".join(themes[:3])
    return f"{sector}: {s}."

def _implications(sector: str, sentiment_score: int) -> str:
    if sentiment_score >= 60:
        return f"{sector}: headline tone is supportive. If it persists, the next 1 to 3 months often favor momentum and multiple expansion in the strongest names."
    if sentiment_score <= 40:
        return f"{sector}: headline tone is risk-off. If it persists, expect choppier price action and a preference for quality balance sheets and clear catalysts."
    return f"{sector}: headline tone is neutral. If the macro backdrop stays steady, relative winners tend to be those with near-term catalysts or strong guidance."

@app.get("/news/briefing")
def news_briefing(
    sectors: str = Query(default="AI,Medical,Energy,Robotics,Infrastructure,Semiconductors,Cloud,Cybersecurity,Defense,Financials,Consumer"),
    watchlist: str = Query(default="SPY,QQQ,NVDA,AAPL,MSFT,AMZN,GOOGL,META,TSLA,BRK-B"),
    max_items_per_sector: int = Query(default=12, ge=5, le=40),
):
    key = f"briefing:{sectors}:{watchlist}:{max_items_per_sector}"
    cached = cache_get(key)
    if cached is not None:
        return cached

    sector_list = [s.strip() for s in sectors.split(",") if s.strip()]
    if not sector_list:
        sector_list = DEFAULT_SECTORS

    wl = [t.strip().upper() for t in watchlist.split(",") if t.strip()]
    wl = wl[:60]

    market = market_snapshot()
    sector_tiles: List[dict] = []
    errors: List[str] = []

    for sec in sector_list:
        queries = SECTOR_QUERIES.get(sec, [sec])
        items: List[dict] = []
        for q in queries[:2]:
            url = _google_news_rss(q)
            try:
                got = _fetch_rss_items(url, max_items=max_items_per_sector)
                for x in got:
                    x["sourceFeed"] = "google_news_rss"
                    x["sector"] = sec
                items.extend(got)
            except Exception as e:
                errors.append(f"{sec}: {type(e).__name__}: {str(e)}")

        seen = set()
        ded = []
        for x in items:
            lk = (x.get("link") or "").strip()
            if not lk or lk in seen:
                continue
            seen.add(lk)
            ded.append(x)

        ded = ded[:max_items_per_sector]
        titles = [x.get("title", "") for x in ded]
        sent = _headline_sentiment(titles)

        ticker_mentions = _extract_tickers_from_titles(titles, wl)
        top_mentions = [{"ticker": k, "count": v} for k, v in sorted(ticker_mentions.items(), key=lambda kv: (-kv[1], kv[0]))][:6]

        tile = {
            "sector": sec,
            "sentiment": sent,
            "summary": _brief_paragraph_from_headlines(titles, sec),
            "implications": _implications(sec, int(sent.get("score", 50) or 50)),
            "topHeadlines": ded[:8],
            "watchlistMentions": top_mentions,
        }
        sector_tiles.append(tile)

    all_titles = []
    for t in sector_tiles:
        for h in t.get("topHeadlines", [])[:6]:
            all_titles.append(h.get("title", "") or "")
    overall_sent = _headline_sentiment(all_titles)

    out = {
        "date": datetime.now(timezone.utc).date().isoformat(),
        "market": market,
        "overallSentiment": overall_sent,
        "sectors": sector_tiles,
        "errors": errors,
        "note": "Briefing uses RSS headline signals. Treat as context, not prediction.",
    }
    return cache_set(key, out, ttl_seconds=180)


# =========================================================
# Congress: /report/today
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


# =========================================================
# Congress: holdings proxy endpoint your UI expects
# =========================================================
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


# =========================================================
# Congress: day-by-day activity feed for UI
# =========================================================
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

        cap_url = ""
        if pol:
            cap_url = "https://www.capitoltrades.com/politicians/" + quote_plus(pol)

        ticker_url = ""
        if ticker:
            ticker_url = "https://www.capitoltrades.com/trades?search=" + quote_plus(ticker)

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
# Signals (scored ideas) - your existing endpoint
# =========================================================
def _congress_ticker_agg(congress_payload: dict) -> Dict[str, dict]:
    agg: Dict[str, dict] = {}
    for arr_name, buy_mode in [("politicianBuys", True), ("politicianSells", False)]:
        arr = congress_payload.get(arr_name) or []
        if not isinstance(arr, list):
            continue
        for x in arr:
            t = str(x.get("ticker") or "").upper().strip()
            if not t:
                continue
            cur = agg.get(t) or {"ticker": t, "buy": 0, "sell": 0, "dem": 0, "rep": 0, "last": ""}
            dem = int(x.get("demBuyers") or 0) + int(x.get("demSellers") or 0)
            rep = int(x.get("repBuyers") or 0) + int(x.get("repSellers") or 0)
            cur["dem"] += dem
            cur["rep"] += rep
            if buy_mode:
                cur["buy"] += int(x.get("demBuyers") or 0) + int(x.get("repBuyers") or 0)
            else:
                cur["sell"] += int(x.get("demSellers") or 0) + int(x.get("repSellers") or 0)
            last = str(x.get("lastFiledAt") or "")
            if last and (not cur["last"] or last > cur["last"]):
                cur["last"] = last
            agg[t] = cur
    return agg


def _score_row(cong: dict, news_mentions: int, sector_mentions: int) -> dict:
    buy = int(cong.get("buy") or 0)
    sell = int(cong.get("sell") or 0)
    dem = int(cong.get("dem") or 0)
    rep = int(cong.get("rep") or 0)

    participation = buy + sell
    net = buy - sell
    bipartisan = min(dem, rep)

    score = 0.0
    score += 6.0 * participation
    score += 5.0 * max(0, net)
    score += 2.5 * bipartisan
    score += 2.0 * news_mentions
    score += 1.5 * sector_mentions

    score_100 = int(max(0, min(100, round(100.0 * (1.0 - math.exp(-score / 25.0))))))

    drivers = []
    if participation >= 4:
        drivers.append("high participation")
    if net >= 2:
        drivers.append("net accumulation")
    if bipartisan >= 2:
        drivers.append("bipartisan")
    if news_mentions >= 2:
        drivers.append("watchlist headlines")
    if sector_mentions >= 2:
        drivers.append("sector headlines")

    return {
        "ticker": cong.get("ticker"),
        "score": score_100,
        "congress": {
            "buys": buy,
            "sells": sell,
            "net": net,
            "bipartisan": bipartisan,
            "lastFiledAt": cong.get("last") or "",
        },
        "news": {
            "watchlistMentions": int(news_mentions),
            "sectorMentions": int(sector_mentions),
        },
        "drivers": drivers[:4],
    }


@app.get("/signals/ideas")
def signals_ideas(
    window_days: int = Query(default=30, ge=1, le=365),
    watchlist: str = Query(default="SPY,QQQ,NVDA,AAPL,MSFT,AMZN,GOOGL,META,TSLA,BRK-B"),
    sectors: str = Query(default="AI,Medical,Energy,Robotics,Infrastructure,Semiconductors,Cloud,Cybersecurity,Defense"),
    limit: int = Query(default=25, ge=5, le=100),
):
    key = f"ideas:{window_days}:{watchlist}:{sectors}:{limit}"
    cached = cache_get(key)
    if cached is not None:
        return cached

    wl = [t.strip().upper() for t in watchlist.split(",") if t.strip()]
    sec_str = sectors

    congress = report_today(window_days=window_days, horizon_days=None)

    cong_agg = _congress_ticker_agg(congress)

    wnews = news_watchlist(tickers=",".join(wl), max_items_per_ticker=8)
    watch_titles: List[str] = []
    for t, arr in (wnews.get("itemsByTicker") or {}).items():
        for x in (arr or [])[:8]:
            watch_titles.append(x.get("title", "") or "")

    briefing = news_briefing(sectors=sec_str, watchlist=",".join(wl), max_items_per_sector=12)
    sector_titles: List[str] = []
    for sec in briefing.get("sectors") or []:
        for x in (sec.get("topHeadlines") or [])[:10]:
            sector_titles.append(x.get("title", "") or "")

    watch_counts = _extract_tickers_from_titles(watch_titles, wl)
    sector_counts = _extract_tickers_from_titles(sector_titles, wl)

    tickers = set(cong_agg.keys()) | set(watch_counts.keys()) | set(sector_counts.keys())

    rows = []
    for t in tickers:
        cong = cong_agg.get(t) or {"ticker": t, "buy": 0, "sell": 0, "dem": 0, "rep": 0, "last": ""}
        rows.append(_score_row(cong, watch_counts.get(t, 0), sector_counts.get(t, 0)))

    rows.sort(key=lambda x: (-int(x.get("score") or 0), x.get("ticker") or ""))
    out = {
        "date": datetime.now(timezone.utc).date().isoformat(),
        "windowDays": window_days,
        "limit": limit,
        "ideas": rows[:limit],
        "note": "Score blends congress activity with relevance from watchlist + sector headlines.",
    }
    return cache_set(key, out, ttl_seconds=180)
