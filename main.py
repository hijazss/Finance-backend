import os
import re
import time
import math
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
app = FastAPI(title="Finance Signals Backend", version="4.3.0")

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


@app.get("/")
def root():
    return {"status": "ok", "version": "4.3.0"}


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
    "SHIB": ["shiba", "shib", "shiba inu"],
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
    for sym in list(_CRYPTO_PATTERNS.keys()):
        for pat in _CRYPTO_PATTERNS.get(sym, []):
            if pat.search(text):
                hits.append(sym)
                break

    out: List[str] = []
    for sym in list(_CRYPTO_PATTERNS.keys()):
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
    for sym in list(_CRYPTO_PATTERNS.keys()):
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
# Market data (Yahoo + Stooq fallback) with caching to reduce 429
# =========================================================
def _requests_get(url: str, params: Optional[dict] = None, timeout: int = 14) -> requests.Response:
    return requests.get(url, params=params, timeout=timeout, headers=UA_HEADERS)


def _yahoo_chart(symbol: str, range_str: str = "6mo", interval: str = "1d") -> dict:
    url = "https://query1.finance.yahoo.com/v8/finance/chart/" + quote_plus(symbol)
    params = {"range": range_str, "interval": interval}
    r = _requests_get(url, params=params, timeout=14)
    if r.status_code == 429:
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


def _stooq_closes(stooq_symbol: str) -> List[Tuple[datetime, float]]:
    # CSV: date,open,high,low,close,volume
    url = "https://stooq.com/q/d/l/"
    params = {"s": stooq_symbol, "i": "d"}
    r = _requests_get(url, params=params, timeout=10)
    r.raise_for_status()
    lines = [ln.strip() for ln in r.text.splitlines() if ln.strip()]
    if len(lines) < 3:
        return []
    out: List[Tuple[datetime, float]] = []
    for ln in lines[1:]:
        parts = ln.split(",")
        if len(parts) < 5:
            continue
        ds = parts[0].strip()
        cs = parts[4].strip()
        try:
            dt = datetime.strptime(ds, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            c = float(cs)
            out.append((dt, c))
        except Exception:
            continue
    out.sort(key=lambda x: x[0])
    return out


def _symbol_to_stooq(symbol: str) -> Optional[str]:
    s = (symbol or "").upper().strip()
    if s == "SPY":
        return "spy.us"
    if s == "^VIX":
        return "vix"
    return None


def _closes_best_effort(symbol: str, range_str: str, interval: str, errors: List[str]) -> List[Tuple[datetime, float]]:
    # Cache closes by symbol + range for a short time to reduce external calls
    ck = f"closes:{symbol}:{range_str}:{interval}"
    cached = cache_get(ck)
    if cached is not None:
        return cached

    try:
        data = _yahoo_closes(symbol, range_str=range_str, interval=interval)
        if data:
            return cache_set(ck, data, ttl_seconds=240)
    except Exception as e:
        errors.append(f"Yahoo: {type(e).__name__}: {str(e)}")

    stooq_sym = _symbol_to_stooq(symbol)
    if stooq_sym:
        try:
            data = _stooq_closes(stooq_sym)
            if data:
                return cache_set(ck, data, ttl_seconds=240)
        except Exception as e:
            errors.append(f"Stooq: {type(e).__name__}: {str(e)}")

    return cache_set(ck, [], ttl_seconds=60)


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
# Fear & Greed (CNN best-effort + optional alt sources)
# =========================================================
def _cnn_fear_greed_graphdata(date_str: Optional[str] = None) -> dict:
    d = date_str or datetime.now(timezone.utc).date().isoformat()
    url = f"https://production.dataviz.cnn.io/index/fearandgreed/graphdata/{d}"
    r = _requests_get(url, timeout=14)
    if r.status_code == 429:
        raise RuntimeError("CNN rate limited (HTTP 429)")
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
            "source": "cnn",
        }
        return cache_set(key, out, ttl_seconds=600)
    except Exception as e:
        return cache_set(
            key,
            {
                "date": date or datetime.now(timezone.utc).date().isoformat(),
                "score": None,
                "rating": None,
                "source": "cnn",
                "error": f"{type(e).__name__}: {str(e)}",
            },
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
    out = {"date": now.date().isoformat(), "sp500": {}, "vix": {}, "fearGreed": {}}
    errors: List[str] = []

    spy = _closes_best_effort("SPY", range_str="3mo", interval="1d", errors=errors)
    if len(spy) >= 25:
        closes = [c for _, c in spy]
        last = closes[-1]
        out["sp500"] = {
            "symbol": "SPY",
            "last": round(last, 4),
            "ret1dPct": round(_pct(closes[-1], closes[-2]), 4) if len(closes) >= 2 else None,
            "ret5dPct": round(_pct(closes[-1], closes[-6]), 4) if len(closes) >= 6 else None,
            "ret1mPct": round(_pct(closes[-1], closes[-22]), 4) if len(closes) >= 22 else None,
        }

    vix = _closes_best_effort("^VIX", range_str="3mo", interval="1d", errors=errors)
    if len(vix) >= 2:
        closes = [c for _, c in vix]
        out["vix"] = {
            "symbol": "^VIX",
            "last": round(closes[-1], 4),
            "chg1d": round(closes[-1] - closes[-2], 4),
        }

    fg = market_fear_greed(None)
    out["fearGreed"] = {
        "score": fg.get("score"),
        "rating": fg.get("rating"),
        "source": fg.get("source"),
        "error": fg.get("error"),
    }

    out["errors"] = errors
    # Longer cache to prevent rapid refresh 429s
    return cache_set(key, out, ttl_seconds=300)


# =========================================================
# Market Entry Index
# =========================================================
@app.get("/market/entry")
def market_entry(window_days: int = Query(default=365, ge=30, le=365)):
    key = f"market:entry:{window_days}"
    cached = cache_get(key)
    if cached is not None:
        return cached

    now = datetime.now(timezone.utc)
    errors: List[str] = []

    spy = _closes_best_effort("SPY", range_str="1y", interval="1d", errors=errors)
    vix = _closes_best_effort("^VIX", range_str="1y", interval="1d", errors=errors)

    if len(spy) < 210 or len(vix) < 30:
        out = {
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
        return cache_set(key, out, ttl_seconds=180)

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
        notes = notes + " | " + " | ".join(errors)

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
    }
    # Cache longer so rapid refresh does not hammer providers
    return cache_set(key, out, ttl_seconds=300)


# =========================================================
# RSS + sentiment (existing approach)
# =========================================================
def _fetch_rss_items(url: str, timeout: int = 12, max_items: int = 30) -> List[dict]:
    r = _requests_get(url, timeout=timeout)
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
    "profit", "profits", "outperform", "buy", "wins", "win", "breakout",
    "approval", "approved", "launch", "partnership", "adoption"
])
_NEG_WORDS = set([
    "miss", "misses", "drop", "drops", "plunge", "plunges", "down",
    "downgrade", "downgrades", "warning", "weak", "bear", "bearish",
    "lawsuit", "probe", "investigation", "fraud", "loss", "losses",
    "cut", "cuts", "layoff", "layoffs", "recession", "crash",
    "exploit", "hack", "breach", "ban", "banned"
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
# News briefing (sector-friendly daily summary)
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
    return cache_set(key, out, ttl_seconds=240)


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
# Congress: holdings proxy endpoint
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
# Congress: day-by-day activity feed
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
# Signals: scored list for Main tab
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
    for _, arr in (wnews.get("itemsByTicker") or {}).items():
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


# =========================================================
# NEW: Crypto news briefing for Crypto tab
# - Uses a mix of major outlet RSS feeds (best effort) + Google News RSS search per coin
# - Returns per-coin visual tiles: sentiment, summary, catalysts, top headlines
# =========================================================
COIN_KEYWORDS_DEFAULT = "BTC,ETH,LINK,SHIB"

# Best-effort major outlets. If any are blocked, Google News fallback still populates.
CRYPTO_OUTLET_FEEDS = [
    {"name": "Cointelegraph", "url": "https://cointelegraph.com/rss"},
    {"name": "Decrypt", "url": "https://decrypt.co/feed"},
    {"name": "CryptoSlate", "url": "https://cryptoslate.com/feed/"},
    {"name": "NewsBTC", "url": "https://www.newsbtc.com/feed/"},
    {"name": "Bitcoin Magazine", "url": "https://bitcoinmagazine.com/.rss"},
]

# Heuristic catalyst keywords
_CATALYST_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("ETF / approval decision", re.compile(r"\betf\b|\bapproval\b|\bapproved\b|\bsec\b", re.IGNORECASE)),
    ("Regulation / enforcement", re.compile(r"\bsec\b|\bdoj\b|\bfinra\b|\bregulat|\bban\b", re.IGNORECASE)),
    ("Upgrade / hard fork", re.compile(r"\bupgrade\b|\bfork\b|\bhard fork\b|\bmainnet\b|\btestnet\b", re.IGNORECASE)),
    ("Hack / exploit risk", re.compile(r"\bhack\b|\bexploit\b|\bbreach\b|\bdrain\b", re.IGNORECASE)),
    ("Macro risk (rates, CPI, Fed)", re.compile(r"\bfed\b|\bcpi\b|\binflation\b|\brates\b|\byields?\b", re.IGNORECASE)),
    ("Exchange / liquidity", re.compile(r"\bexchange\b|\bliquidat|\boutage\b|\bwithdraw", re.IGNORECASE)),
]


def _coingecko_top(n: int = 15) -> List[dict]:
    key = f"coingecko:top:{n}"
    cached = cache_get(key)
    if cached is not None:
        return cached

    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency": "usd", "order": "market_cap_desc", "per_page": n, "page": 1, "sparkline": "false"}
    r = _requests_get(url, params=params, timeout=14)
    if r.status_code == 429:
        # keep previous cache if exists, otherwise return empty
        return cache_set(key, [], ttl_seconds=120)
    r.raise_for_status()
    arr = r.json() if r.text else []
    out = []
    for x in arr[:n]:
        out.append({
            "id": x.get("id"),
            "symbol": str(x.get("symbol") or "").upper(),
            "name": x.get("name"),
            "marketCapRank": x.get("market_cap_rank"),
        })
    return cache_set(key, out, ttl_seconds=600)


def _coin_query_terms(symbol: str, name: str) -> List[str]:
    sym = (symbol or "").upper().strip()
    nm = (name or "").strip()
    base = []
    if nm and sym:
        base.append(f'"{nm}" OR {sym} crypto')
    elif nm:
        base.append(f'"{nm}" crypto')
    elif sym:
        base.append(f'{sym} crypto')
    # add "price" and "ETF/upgrade" angles
    if nm:
        base.append(f'"{nm}" (ETF OR SEC OR upgrade OR hack OR lawsuit OR partnership OR adoption)')
    else:
        base.append(f'{sym} (ETF OR SEC OR upgrade OR hack OR lawsuit OR partnership OR adoption)')
    return base


def _dedupe_by_link(items: List[dict], max_n: int) -> List[dict]:
    seen = set()
    out = []
    for x in items:
        lk = (x.get("link") or "").strip()
        if not lk or lk in seen:
            continue
        seen.add(lk)
        out.append(x)
        if len(out) >= max_n:
            break
    return out


def _match_coin(title: str, symbol: str, name: str) -> bool:
    t = (title or "")
    if not t:
        return False
    sym = (symbol or "").upper().strip()
    nm = (name or "").strip()
    # exact token match for symbol, and substring for name
    if sym and re.search(rf"\b{re.escape(sym)}\b", t, flags=re.IGNORECASE):
        return True
    if nm and re.search(re.escape(nm), t, flags=re.IGNORECASE):
        return True
    # special case: Shiba Inu often appears as "Shiba"
    if sym == "SHIB" and re.search(r"\bshiba\b", t, flags=re.IGNORECASE):
        return True
    return False


def _crypto_summary_and_catalysts(titles: List[str], coin_name: str) -> Tuple[str, List[str]]:
    if not titles:
        return f"No major {coin_name} headlines in the current pull.", []

    blob = " ".join(titles[:20])
    found = []
    for label, pat in _CATALYST_PATTERNS:
        if pat.search(blob):
            found.append(label)

    # simple thematic summary
    low = blob.lower()
    themes = []
    if any(w in low for w in ["etf", "sec", "approval", "filing"]):
        themes.append("regulatory and ETF narratives are prominent")
    if any(w in low for w in ["upgrade", "fork", "mainnet", "testnet", "release"]):
        themes.append("network upgrade or roadmap items are in focus")
    if any(w in low for w in ["hack", "exploit", "breach", "vulnerability"]):
        themes.append("security risk headlines are present")
    if any(w in low for w in ["institution", "custody", "blackrock", "fidelity", "spot"]):
        themes.append("institutional flow and product headlines are showing up")
    if any(w in low for w in ["inflation", "cpi", "fed", "rates", "yield"]):
        themes.append("macro data and rates could impact risk appetite")

    if not themes:
        themes = ["headlines look mixed and mostly event-driven"]

    summary = f"{coin_name}: " + "; ".join(themes[:3]) + "."
    return summary, found[:5]


@app.get("/crypto/top")
def crypto_top(limit: int = Query(default=15, ge=5, le=50)):
    return {
        "date": datetime.now(timezone.utc).date().isoformat(),
        "coins": _coingecko_top(limit),
        "source": "coingecko",
    }


@app.get("/crypto/news/briefing")
def crypto_news_briefing(
    coins: str = Query(default="BTC,ETH,LINK,SHIB"),
    include_top_n: int = Query(default=15, ge=0, le=30),
    max_items_per_outlet: int = Query(default=25, ge=10, le=60),
    max_items_per_coin: int = Query(default=14, ge=6, le=40),
):
    """
    Crypto page helper:
    - Pulls from major crypto outlets (RSS) + Google News RSS search per coin
    - Produces per-coin: sentiment, short summary, catalysts, top headlines
    """
    key = f"cryptoBrief:{coins}:{include_top_n}:{max_items_per_outlet}:{max_items_per_coin}"
    cached = cache_get(key)
    if cached is not None:
        return cached

    now = datetime.now(timezone.utc)

    wanted_syms = [c.strip().upper() for c in (coins or "").split(",") if c.strip()]
    wanted_syms = wanted_syms[:30]

    top = _coingecko_top(include_top_n) if include_top_n > 0 else []
    top_map = {x["symbol"]: x for x in top if x.get("symbol")}

    # Ensure requested coins exist, even if not in top list
    enriched: List[dict] = []
    for sym in wanted_syms:
        if sym in top_map:
            enriched.append({"symbol": sym, "name": top_map[sym].get("name") or sym, "rank": top_map[sym].get("marketCapRank")})
        else:
            enriched.append({"symbol": sym, "name": sym, "rank": None})

    # Add top coins (if not already requested)
    for x in top:
        sym = x.get("symbol")
        if sym and sym not in wanted_syms:
            enriched.append({"symbol": sym, "name": x.get("name") or sym, "rank": x.get("marketCapRank")})

    # Fetch outlet headlines once
    all_outlet_items: List[dict] = []
    outlet_errors: List[str] = []
    for f in CRYPTO_OUTLET_FEEDS:
        try:
            items = _fetch_rss_items(f["url"], timeout=12, max_items=max_items_per_outlet)
            for it in items:
                it["bucket"] = f["name"]
                it["sourceFeed"] = f["url"]
            all_outlet_items.extend(items)
        except Exception as e:
            outlet_errors.append(f'{f["name"]}: {type(e).__name__}: {str(e)}')

    all_outlet_items = _dedupe_by_link(all_outlet_items, max_n=400)

    # Build per-coin tiles with a Google News fallback query
    coin_tiles: List[dict] = []
    google_errors: List[str] = []

    for coin in enriched[:max(1, min(30, include_top_n + len(wanted_syms)))]:
        sym = coin["symbol"]
        nm = coin["name"]

        matched = [x for x in all_outlet_items if _match_coin(x.get("title", ""), sym, nm)]
        matched = matched[:max_items_per_coin]

        # If sparse, supplement with Google News RSS search
        google_items: List[dict] = []
        if len(matched) < max(6, max_items_per_coin // 2):
            for q in _coin_query_terms(sym, nm)[:2]:
                try:
                    url = _google_news_rss(q)
                    got = _fetch_rss_items(url, timeout=12, max_items=max_items_per_coin)
                    for it in got:
                        it["bucket"] = "Google News"
                        it["sourceFeed"] = "google_news_rss"
                    google_items.extend(got)
                except Exception as e:
                    google_errors.append(f"{sym}: {type(e).__name__}: {str(e)}")

        merged = matched + google_items
        merged = _dedupe_by_link(merged, max_n=max_items_per_coin)

        titles = [x.get("title", "") for x in merged if x.get("title")]
        sentiment = _headline_sentiment(titles)
        summary, catalysts = _crypto_summary_and_catalysts(titles, nm)

        coin_tiles.append({
            "symbol": sym,
            "name": nm,
            "rank": coin.get("rank"),
            "sentiment": sentiment,
            "summary": summary,
            "catalysts": catalysts,
            "headlines": merged,
        })

    # Overall rollup sentiment
    all_titles = []
    for t in coin_tiles:
        for h in t.get("headlines", [])[:4]:
            all_titles.append(h.get("title", "") or "")
    overall_sent = _headline_sentiment(all_titles)

    out = {
        "date": now.date().isoformat(),
        "overallSentiment": overall_sent,
        "coins": coin_tiles,
        "sources": {
            "outlets": [x["name"] for x in CRYPTO_OUTLET_FEEDS],
            "googleNewsFallback": True,
            "coingeckoTop": bool(include_top_n > 0),
        },
        "errors": {
            "outlets": outlet_errors,
            "google": google_errors[:25],
        },
        "note": "Crypto briefing is headline-based and best-effort. Catalysts are extracted by keyword, not a calendar feed.",
    }
    return cache_set(key, out, ttl_seconds=240)
