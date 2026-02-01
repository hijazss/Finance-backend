import os
import re
import time
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import requests
import quiverquant
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware


# =========================================================
# App
# =========================================================
app = FastAPI(title="Finance Signals Backend", version="4.4.0")

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
}

RSS_HEADERS = {
    "User-Agent": UA_HEADERS["User-Agent"],
    "Accept": "application/rss+xml, application/xml;q=0.9, text/xml;q=0.9, */*;q=0.8",
    "Accept-Language": UA_HEADERS["Accept-Language"],
}

SESSION = requests.Session()


@app.get("/")
def root():
    return {"status": "ok", "version": "4.4.0"}


# =========================================================
# Render-safe stale cache
# =========================================================
_CACHE: Dict[str, Tuple[float, float, Any]] = {}


def cache_get(key: str, allow_stale: bool = False):
    now = time.time()
    rec = _CACHE.get(key)
    if not rec:
        return None
    fresh_until, stale_until, val = rec
    if now <= fresh_until:
        return val
    if allow_stale and now <= stale_until:
        return val
    return None


def cache_set(key: str, val: Any, ttl: int = 180, stale: int = 1800):
    now = time.time()
    _CACHE[key] = (now + ttl, now + stale, val)
    return val


# =========================================================
# Provider cooldown isolation
# =========================================================
_PROVIDER_COOLDOWN: Dict[str, float] = {}


def cooldown(provider: str, seconds: int):
    _PROVIDER_COOLDOWN[provider] = time.time() + seconds


def is_cool(provider: str) -> bool:
    return time.time() < _PROVIDER_COOLDOWN.get(provider, 0)


# =========================================================
# Utility helpers
# =========================================================
def parse_date(v):
    if not v:
        return None
    try:
        if isinstance(v, datetime):
            return v
        return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except Exception:
        return None


def iso(d):
    return d.date().isoformat() if d else ""


def pick(row, keys):
    for k in keys:
        if row.get(k):
            return row[k]
    return ""


def is_buy(txt):
    return "buy" in (txt or "").lower() or "purchase" in (txt or "").lower()


def is_sell(txt):
    t = (txt or "").lower()
    return "sell" in t or "sale" in t


def norm_party(row):
    for k in ["Party", "party"]:
        if row.get(k):
            return str(row[k])[0].upper()
    for k in ["Politician", "Name", "Representative", "Senator"]:
        v = str(row.get(k, ""))
        m = _party_re.search(v)
        if m:
            return m.group(1).upper()
    return ""


def norm_ticker(row):
    for k in ["Ticker", "ticker", "Symbol", "symbol", "Stock"]:
        if row.get(k):
            return str(row[k]).upper().split()[0]
    return ""


# =========================================================
# Yahoo chart (isolated from RSS)
# =========================================================
def yahoo_chart(symbol: str):
    if is_cool("yahoo_chart"):
        raise RuntimeError("Yahoo chart cooldown")

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote_plus(symbol)}"
    r = SESSION.get(url, timeout=12, headers=UA_HEADERS)

    if r.status_code == 429:
        cooldown("yahoo_chart", 600)
        raise RuntimeError("Yahoo chart 429")

    r.raise_for_status()
    return r.json()


def yahoo_closes(symbol: str):
    j = yahoo_chart(symbol)
    result = (j.get("chart") or {}).get("result") or []
    if not result:
        return []

    r0 = result[0]
    ts = r0.get("timestamp") or []
    closes = (r0.get("indicators") or {}).get("quote", [{}])[0].get("close") or []

    out = []
    for t, c in zip(ts, closes):
        if c:
            out.append((datetime.fromtimestamp(t, tz=timezone.utc), float(c)))
    return out


# =========================================================
# Stooq fallback
# =========================================================
def stooq(symbol: str):
    url = "https://stooq.com/q/d/l/"
    r = SESSION.get(url, params={"s": symbol, "i": "d"}, timeout=12)
    r.raise_for_status()

    rows = r.text.splitlines()
    out = []
    for ln in rows[1:]:
        p = ln.split(",")
        if len(p) >= 5:
            try:
                d = datetime.strptime(p[0], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                out.append((d, float(p[4])))
            except Exception:
                pass
    return out


# =========================================================
# CNN Fear & Greed
# =========================================================
@app.get("/market/fear-greed")
def fear_greed():
    key = "fg"
    cached = cache_get(key, True)
    if cached:
        return cached

    url = f"https://production.dataviz.cnn.io/index/fearandgreed/graphdata/{datetime.utcnow().date().isoformat()}"
    r = SESSION.get(url, timeout=12)
    r.raise_for_status()
    j = r.json()

    now = j.get("fear_and_greed", {}).get("now", {})
    out = {
        "score": now.get("value"),
        "rating": now.get("valueText"),
    }

    return cache_set(key, out, 900, 21600)


# =========================================================
# Market Snapshot (429 immune)
# =========================================================
@app.get("/market/snapshot")
def snapshot():
    key = "snapshot"
    cached = cache_get(key, True)
    if cached:
        return cached

    errors = []

    try:
        spy = stooq("spy.us")
    except Exception as e:
        errors.append("Stooq SPY failed")
        try:
            spy = yahoo_closes("SPY")
        except Exception:
            spy = []

    try:
        vix = stooq("vix")
    except Exception:
        try:
            vix = yahoo_closes("^VIX")
        except Exception:
            vix = []

    def returns(vals):
        if len(vals) < 6:
            return {}
        p = [v for _, v in vals]
        return {
            "last": round(p[-1], 2),
            "ret1dPct": round((p[-1] / p[-2] - 1) * 100, 2),
            "ret5dPct": round((p[-1] / p[-6] - 1) * 100, 2),
        }

    out = {
        "date": datetime.utcnow().date().isoformat(),
        "sp500": returns(spy),
        "vix": returns(vix),
        "errors": errors,
    }

    return cache_set(key, out, 180, 3600)


# =========================================================
# Market Entry Index
# =========================================================
@app.get("/market/entry")
def market_entry():
    key = "entry"
    cached = cache_get(key, True)
    if cached:
        return cached

    try:
        spy = stooq("spy.us")
    except Exception:
        spy = yahoo_closes("SPY")

    try:
        vix = stooq("vix")
    except Exception:
        vix = yahoo_closes("^VIX")

    if len(spy) < 210 or len(vix) < 30:
        return cache_set(
            key,
            {"score": 0, "signal": "DATA UNAVAILABLE"},
            120,
            900,
        )

    prices = [c for _, c in spy]
    sma50 = sum(prices[-50:]) / 50
    sma200 = sum(prices[-200:]) / 200

    trend = 1 if sma50 > sma200 else 0
    v = vix[-1][1]

    vix_score = max(0, min(1, 1 - (v - 12) / 25))
    score = int(round((0.65 * trend + 0.35 * vix_score) * 100))

    signal = "ACCUMULATE" if score > 65 else "WAIT"

    return cache_set(
        key,
        {
            "score": score,
            "signal": signal,
            "regime": "RISK-ON" if score > 70 else "NEUTRAL",
        },
        300,
        3600,
    )


# =========================================================
# NEWS SYSTEM STARTS BELOW
# =========================================================
# =========================================================
# RSS helpers
# =========================================================

def fetch_rss(url: str, limit: int = 25):
    key = f"rss:{url}"
    cached = cache_get(key, True)
    if cached:
        return cached

    try:
        r = SESSION.get(url, timeout=10, headers=RSS_HEADERS)
        if r.status_code == 429:
            raise RuntimeError("429")
        r.raise_for_status()
    except Exception:
        return cached or []

    try:
        root = ET.fromstring(r.text)
    except Exception:
        return []

    items = []
    for it in root.findall(".//item")[:limit]:
        items.append({
            "title": (it.findtext("title") or "").strip(),
            "link": (it.findtext("link") or "").strip(),
            "published": (it.findtext("pubDate") or "").strip(),
        })

    return cache_set(key, items, 240, 10800)


# =========================================================
# Google News RSS
# =========================================================

def google_news(query: str):
    q = quote_plus(query)
    return f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"


# =========================================================
# Headline sentiment
# =========================================================

POS = {"surge","beat","beats","rally","gain","upgrade","record","strong"}
NEG = {"miss","drop","downgrade","lawsuit","probe","fraud","crash","cut","layoff"}

def sentiment(titles):
    pos = neg = 0
    for t in titles:
        words = set(re.findall(r"[a-z]+", t.lower()))
        if words & POS:
            pos += 1
        if words & NEG:
            neg += 1

    total = max(1, pos + neg)
    score = int(50 + (pos - neg) / total * 50)

    if score >= 60:
        label = "POSITIVE"
    elif score <= 40:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"

    return {"label": label, "score": max(0, min(100, score))}


# =========================================================
# NEWS BRIEFING ENDPOINT
# =========================================================

@app.get("/news/briefing")
def news_briefing(
    sectors: str = "AI,Medical,Energy,Crypto",
    watchlist: str = "SPY,QQQ,NVDA,AAPL,MSFT",
    limit: int = 12,
):
    key = f"brief:{sectors}:{watchlist}"
    cached = cache_get(key, True)
    if cached:
        return cached

    sector_list = [s.strip() for s in sectors.split(",")]
    tickers = [t.strip().upper() for t in watchlist.split(",")]

    sectors_out = []
    errors = []

    for sec in sector_list:
        try:
            url = google_news(f"{sec} stock market")
            items = fetch_rss(url, limit * 2)

            titles = [i["title"] for i in items]
            sent = sentiment(titles)

            sectors_out.append({
                "sector": sec,
                "sentiment": sent,
                "topHeadlines": items[:limit],
            })
        except Exception as e:
            errors.append(str(e))

    out = {
        "date": datetime.utcnow().date().isoformat(),
        "overallSentiment": sentiment(
            [h["title"] for s in sectors_out for h in s["topHeadlines"][:3]]
        ),
        "sectors": sectors_out,
        "errors": errors,
        "note": "Yahoo throttling isolated. Google News always active.",
    }

    return cache_set(key, out, 300, 10800)


# =========================================================
# CRYPTO NEWS
# =========================================================

CRYPTO_SOURCES = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
]

COINS = {
    "BTC": [r"\bbitcoin\b", r"\bbtc\b"],
    "ETH": [r"\bethereum\b", r"\beth\b"],
    "LINK": [r"\bchainlink\b", r"\bLINK\b"],
    "SOL": [r"\bsolana\b", r"\bSOL\b"],
    "XRP": [r"\bxrp\b", r"\bripple\b"],
    "SHIB": [r"\bshiba\b", r"\bshib\b"],
}


@app.get("/crypto/news/briefing")
def crypto_news():
    key = "crypto:news"
    cached = cache_get(key, True)
    if cached:
        return cached

    all_items = []

    for src in CRYPTO_SOURCES:
        items = fetch_rss(src, 50)
        for x in items:
            x["source"] = src
        all_items.extend(items)

    coins_out = []

    for sym, patterns in COINS.items():
        matched = []
        for it in all_items:
            title = it["title"]
            if any(re.search(p, title, re.I) for p in patterns):
                matched.append(it)

        coins_out.append({
            "symbol": sym,
            "headlines": matched[:15],
            "sentiment": sentiment([x["title"] for x in matched]),
        })

    return cache_set(
        key,
        {
            "date": datetime.utcnow().date().isoformat(),
            "coins": coins_out,
            "sources": ["CoinDesk", "CoinTelegraph", "Decrypt"],
        },
        300,
        10800,
    )


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "time": datetime.utcnow().isoformat(),
        "cacheKeys": len(_CACHE),
    }
