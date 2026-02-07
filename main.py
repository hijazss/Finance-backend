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
app = FastAPI(title="Finance Signals Backend", version="4.12.1")

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

QUIVER_TOKEN = os.getenv("QUIVER_TOKEN", "").strip()
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "").strip()

UA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36",
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

RSS_HEADERS = {
    "User-Agent": UA_HEADERS["User-Agent"],
    "Accept": "application/rss+xml, application/xml;q=0.9, text/xml;q=0.9, */*;q=0.8",
    "Accept-Language": UA_HEADERS["Accept-Language"],
    "Connection": "keep-alive",
}

NASDAQ_HEADERS = {
    "User-Agent": UA_HEADERS["User-Agent"],
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": UA_HEADERS["Accept-Language"],
    "Connection": "keep-alive",
    "Referer": "https://www.nasdaq.com/",
    "Origin": "https://www.nasdaq.com",
}

YAHOO_HEADERS = {
    "User-Agent": UA_HEADERS["User-Agent"],
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": UA_HEADERS["Accept-Language"],
    "Connection": "keep-alive",
    "Referer": "https://finance.yahoo.com/",
}

SESSION = requests.Session()


@app.get("/")
def root():
    return {"status": "ok", "version": "4.12.1"}


@app.get("/health")
def health():
    return {
        "ok": True,
        "version": "4.12.1",
        "hasQuiverToken": bool(QUIVER_TOKEN),
        "hasFinnhubKey": bool(FINNHUB_API_KEY),
        "utc": datetime.now(timezone.utc).isoformat(),
    }


# =========================================================
# Simple stale-while-revalidate cache
# key -> (fresh_until_epoch, stale_until_epoch, value)
# =========================================================
_CACHE: Dict[str, Tuple[float, float, Any]] = {}


def cache_get(key: str, allow_stale: bool = False) -> Optional[Any]:
    now = time.time()
    rec = _CACHE.get(key)
    if not rec:
        return None
    fresh_until, stale_until, val = rec
    if now <= fresh_until:
        return val
    if allow_stale and now <= stale_until:
        return val
    _CACHE.pop(key, None)
    return None


def cache_set(key: str, val: Any, ttl_seconds: int = 120, stale_ttl_seconds: int = 900) -> Any:
    now = time.time()
    _CACHE[key] = (now + float(ttl_seconds), now + float(stale_ttl_seconds), val)
    return val


# =========================================================
# HTTP helpers
# =========================================================
def _requests_get(
    url: str,
    params: Optional[dict] = None,
    timeout: int = 16,
    headers: Optional[dict] = None
) -> requests.Response:
    return SESSION.get(url, params=params, timeout=timeout, headers=headers or UA_HEADERS)


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace("$", "").replace(",", "")
    m = re.search(r"-?\d+(\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def _pct(a: float, b: float) -> float:
    if not b:
        return 0.0
    return 100.0 * (a / b - 1.0)


def _ret_from_series(vals: List[float], offset: int) -> Optional[float]:
    if len(vals) < (offset + 1):
        return None
    a = float(vals[-1])
    b = float(vals[-1 - offset])
    return _pct(a, b)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _sma(vals: List[float], n: int) -> Optional[float]:
    if n <= 0 or len(vals) < n:
        return None
    return sum(vals[-n:]) / n


# =========================================================
# Nasdaq quote (best effort last price fallback)
# =========================================================
def _nasdaq_assetclass_for_symbol(symbol: str) -> str:
    s = (symbol or "").upper().strip()
    if s in ["SPY", "QQQ", "DIA", "IWM"]:
        return "etf"
    if s in ["VIX", "^VIX", "NDX", "^NDX", "SPX", "^SPX", "TNX", "^TNX"]:
        return "index"
    return "stocks"


def _nasdaq_symbol_normalize(symbol: str) -> str:
    s = (symbol or "").strip().upper()
    if s == "^VIX":
        return "VIX"
    if s == "^SPX":
        return "SPX"
    if s == "^NDX":
        return "NDX"
    if s == "^TNX":
        return "TNX"
    return s


def _nasdaq_quote(symbol: str, assetclass: Optional[str] = None) -> dict:
    sym = _nasdaq_symbol_normalize(symbol)
    ac = assetclass or _nasdaq_assetclass_for_symbol(sym)

    url = f"https://api.nasdaq.com/api/quote/{quote_plus(sym)}/info"
    r = _requests_get(url, params={"assetclass": ac}, timeout=14, headers=NASDAQ_HEADERS)
    if r.status_code == 429:
        raise RuntimeError("Nasdaq rate limited (429)")
    r.raise_for_status()
    return r.json() if r.text else {}


def _nasdaq_last_and_prev(symbol: str) -> Tuple[Optional[float], Optional[float]]:
    j = _nasdaq_quote(symbol)
    data = (j or {}).get("data") or {}
    primary = data.get("primaryData") or {}
    secondary = data.get("secondaryData") or {}
    key_stats = data.get("keyStats") or {}

    last = _safe_float(primary.get("lastSalePrice") or primary.get("lastSale") or primary.get("last"))
    prev = _safe_float(primary.get("previousClose") or secondary.get("previousClose") or key_stats.get("PreviousClose"))
    if prev is None:
        prev = _safe_float(key_stats.get("previousClose"))
    return last, prev


# =========================================================
# Yahoo Finance chart (primary daily series source)
# =========================================================
def _yahoo_chart_daily_closes(symbol: str, range_str: str = "1y") -> List[Tuple[datetime, float]]:
    """
    Uses Yahoo chart endpoint:
      https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=1y
    """
    sym = (symbol or "").strip()
    if not sym:
        return []

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote_plus(sym)}"
    params = {"interval": "1d", "range": range_str}

    r = _requests_get(url, params=params, timeout=16, headers=YAHOO_HEADERS)
    if r.status_code == 429:
        raise RuntimeError("Yahoo rate limited (429)")
    r.raise_for_status()

    j = r.json() if r.text else {}
    chart = (j or {}).get("chart") or {}
    res = (chart.get("result") or [None])[0] or {}
    ts = res.get("timestamp") or []
    ind = (res.get("indicators") or {}).get("quote") or []
    q0 = ind[0] if ind else {}
    closes = q0.get("close") or []

    out: List[Tuple[datetime, float]] = []
    for t, c in zip(ts, closes):
        if t is None or c is None:
            continue
        try:
            dt = datetime.fromtimestamp(int(t), tz=timezone.utc)
            out.append((dt, float(c)))
        except Exception:
            continue

    out.sort(key=lambda x: x[0])
    return out


def _series_daily(symbol: str, yahoo_symbol: str) -> Tuple[List[Tuple[datetime, float]], List[str]]:
    """
    Returns (series, errors). Series is list of (dt, close).
    """
    errors: List[str] = []
    try:
        s = _yahoo_chart_daily_closes(yahoo_symbol, "1y")
        if s:
            return s, errors
        errors.append(f"Yahoo {symbol}: empty series")
    except Exception as e:
        errors.append(f"Yahoo {symbol}: {type(e).__name__}: {str(e)[:160]}")

    # Last-price fallback only (no history) via Nasdaq
    try:
        last, prev = _nasdaq_last_and_prev(symbol)
        if last is not None:
            now = datetime.now(timezone.utc)
            series = [(now - timedelta(days=1), float(prev))] if prev is not None else []
            series.append((now, float(last)))
            return series, errors
        errors.append(f"Nasdaq {symbol}: empty last")
    except Exception as e:
        errors.append(f"Nasdaq {symbol}: {type(e).__name__}: {str(e)[:160]}")

    return [], errors


# =========================================================
# Market analytics helpers
# =========================================================
def _realized_vol_pct(vals: List[float], n: int = 20) -> Optional[float]:
    if len(vals) < (n + 1):
        return None
    rets = []
    for i in range(-n, 0):
        p0 = float(vals[i - 1])
        p1 = float(vals[i])
        if p0 <= 0 or p1 <= 0:
            continue
        rets.append(math.log(p1 / p0))
    if len(rets) < max(5, int(0.75 * n)):
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / max(1, (len(rets) - 1))
    sd = math.sqrt(var)
    ann = sd * math.sqrt(252.0)
    return float(ann * 100.0)


def _risk_appetite_score(spy_vals: List[float], vix_vals: List[float]) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    """
    0..100 composite:
      - Trend: SPY SMA50 vs SMA200 + price vs SMA200
      - Vol: VIX level normalized
      - Realized vol: SPY RV20
    If VIX is unavailable, return None values so frontend can hide it.
    """
    if not spy_vals or not vix_vals:
        return None, None, None

    price = float(spy_vals[-1])
    vix = float(vix_vals[-1])

    sma50 = _sma(spy_vals, 50) or price
    sma200 = _sma(spy_vals, 200) or price

    trend_cross = 1.0 if sma50 >= sma200 else 0.0
    price_vs_200 = _clamp01((price / sma200 - 0.92) / (1.08 - 0.92))
    trend = _clamp01(0.55 * trend_cross + 0.45 * price_vs_200)

    vix_norm = _clamp01(1.0 - ((vix - 12.0) / (35.0 - 12.0)))
    rv20 = _realized_vol_pct(spy_vals, 20)
    rv_norm = 0.5
    if isinstance(rv20, (int, float)):
        rv_norm = _clamp01(1.0 - ((float(rv20) - 10.0) / (35.0 - 10.0)))

    score01 = 0.55 * trend + 0.30 * vix_norm + 0.15 * rv_norm
    score = int(round(100.0 * score01))

    if score >= 70:
        label = "RISK-ON"
    elif score <= 40:
        label = "RISK-OFF"
    else:
        label = "NEUTRAL"

    notes = f"SPY trend + VIX level + SPY RV20 composite. Score={score}."
    return score, label, notes


# =========================================================
# Daily market index dashboard (frontend uses this)
# =========================================================
@app.get("/market/index")
def market_index():
    key = "market:index:v4121"
    cached = cache_get(key, allow_stale=True)
    if cached is not None:
        return cached

    now = datetime.now(timezone.utc)
    errors: List[str] = []

    # Core market proxies
    # Yahoo symbols: SPY, QQQ, IWM, DIA, ^VIX, ^TNX
    # Nasdaq fallback uses SPY/QQQ/IWM/DIA/VIX/TNX where possible
    series_spy, e1 = _series_daily("SPY", "SPY")
    series_qqq, e2 = _series_daily("QQQ", "QQQ")
    series_iwm, e3 = _series_daily("IWM", "IWM")
    series_dia, e4 = _series_daily("DIA", "DIA")
    series_vix, e5 = _series_daily("^VIX", "^VIX")
    series_tnx, e6 = _series_daily("^TNX", "^TNX")

    errors.extend(e1 + e2 + e3 + e4 + e5 + e6)

    def _vals(s: List[Tuple[datetime, float]]) -> List[float]:
        return [c for _, c in s]

    spy_vals = _vals(series_spy)
    qqq_vals = _vals(series_qqq)
    iwm_vals = _vals(series_iwm)
    dia_vals = _vals(series_dia)
    vix_vals = _vals(series_vix)
    tnx_vals = _vals(series_tnx)

    def _pack(symbol: str, vals: List[float]) -> dict:
        last = float(vals[-1]) if vals else None
        return {
            "symbol": symbol,
            "last": last,
            "ret1dPct": _ret_from_series(vals, 1),
            "ret5dPct": _ret_from_series(vals, 5),
            "ret1mPct": _ret_from_series(vals, 21),
        }

    out: Dict[str, Any] = {
        "date": now.date().isoformat(),
        "indices": {
            "spy": _pack("SPY", spy_vals),
            "qqq": _pack("QQQ", qqq_vals),
            "iwm": _pack("IWM", iwm_vals),
            "dia": _pack("DIA", dia_vals),
            "vix": _pack("^VIX", vix_vals),
            "tnx": _pack("^TNX", tnx_vals),  # 10Y yield index level, not a percent here
        },
        "volatility": None,
        "riskAppetite": None,
        "errors": errors,
        "note": "Market index uses Yahoo Finance daily chart as primary source; Nasdaq quote is fallback. If VIX unavailable, volatility and risk appetite are omitted.",
    }

    # Volatility block: only if we truly have VIX history
    if len(vix_vals) >= 22:
        vix_rv = _realized_vol_pct(vix_vals, 20)
        out["volatility"] = {
            "vixLevel": float(vix_vals[-1]) if vix_vals else None,
            "vixRet1dPct": _ret_from_series(vix_vals, 1),
            "vixRealizedVol20dPct": vix_rv,
        }

    # Risk appetite proxy: only if SPY + VIX exist
    score, label, notes = _risk_appetite_score(spy_vals, vix_vals)
    if score is not None:
        out["riskAppetite"] = {"score": score, "label": label, "notes": notes}

    return cache_set(key, out, ttl_seconds=180, stale_ttl_seconds=1800)


# Backward compatibility (if your frontend still calls /market/read anywhere)
@app.get("/market/read")
def market_read():
    idx = market_index()
    indices = idx.get("indices") or {}
    spy = indices.get("spy") or {}
    qqq = indices.get("qqq") or {}
    vix = indices.get("vix") or {}

    parts: List[str] = []
    if spy.get("last") is not None:
        parts.append(f"SPY {spy['last']:.2f} (1D {spy.get('ret1dPct') if spy.get('ret1dPct') is not None else '—'}%).")
    if qqq.get("last") is not None:
        parts.append(f"QQQ {qqq['last']:.2f} (1D {qqq.get('ret1dPct') if qqq.get('ret1dPct') is not None else '—'}%).")
    if vix.get("last") is not None:
        parts.append(f"VIX {vix['last']:.2f}.")

    return {
        "date": idx.get("date"),
        "summary": " ".join(parts).strip() or "Market index loaded.",
        "sp500": spy,
        "nasdaq": qqq,
        "vix": vix,
        "fearGreed": None,
        "errors": idx.get("errors") or [],
        "note": "Deprecated: use /market/index. Kept for older frontends.",
    }


# =========================================================
# RSS helpers (unchanged)
# =========================================================
def _google_news_rss(query: str) -> str:
    q = quote_plus(query)
    return f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _strip_html(s: str) -> str:
    if not s:
        return ""
    s2 = _TAG_RE.sub(" ", s)
    s2 = s2.replace("&nbsp;", " ").replace("&amp;", "&").replace("&quot;", '"').replace("&#39;", "'")
    s2 = _WS_RE.sub(" ", s2).strip()
    return s2


def _parse_rss_date(s: str) -> Optional[datetime]:
    if not s:
        return None
    s = s.strip()
    fmts = [
        "%a, %d %b %Y %H:%M:%S %Z",
        "%a, %d %b %Y %H:%M:%S %z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d",
    ]
    for f in fmts:
        try:
            dt = datetime.strptime(s, f)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            continue
    try:
        s2 = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s2)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _within_days(pub: str, max_age_days: int) -> bool:
    dt = _parse_rss_date(pub)
    if dt is None:
        return True
    return dt >= (datetime.now(timezone.utc) - timedelta(days=max_age_days))


def _fetch_rss_items_uncached(url: str, timeout: int = 10, max_items: int = 25, max_age_days: int = 30) -> List[dict]:
    r = _requests_get(url, timeout=timeout, headers=RSS_HEADERS)
    if r.status_code == 429:
        raise requests.HTTPError("429 Too Many Requests", response=r)
    r.raise_for_status()

    text = (r.text or "").strip()
    if not text:
        return []
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff").strip()

    try:
        root = ET.fromstring(text)
    except Exception:
        return []

    out: List[dict] = []

    channel = root.find("channel")
    if channel is None:
        entries = root.findall("{http://www.w3.org/2005/Atom}entry")
        for e in entries[: max_items * 2]:
            title = (e.findtext("{http://www.w3.org/2005/Atom}title") or "").strip()
            link_el = e.find("{http://www.w3.org/2005/Atom}link")
            link = (link_el.get("href") if link_el is not None else "") or ""
            pub = (e.findtext("{http://www.w3.org/2005/Atom}updated") or "").strip()
            summ = (e.findtext("{http://www.w3.org/2005/Atom}summary") or "").strip()

            if not title:
                continue
            if not _within_days(pub, max_age_days):
                continue

            summary = _strip_html(summ)
            out.append({"title": title, "link": link, "published": pub, "rawSummary": summary})
            if len(out) >= max_items:
                break
        return out

    for item in channel.findall("item")[: max_items * 3]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        desc = (item.findtext("description") or "").strip()

        if not title:
            continue
        if not _within_days(pub, max_age_days):
            continue

        summary = _strip_html(desc)
        out.append({"title": title, "link": link, "published": pub, "rawSummary": summary})
        if len(out) >= max_items:
            break

    return out


def _fetch_rss_items(
    url: str,
    timeout: int = 10,
    max_items: int = 25,
    ttl_seconds: int = 240,
    stale_ttl_seconds: int = 6 * 3600,
    max_age_days: int = 30,
) -> List[dict]:
    key = f"rss:{url}:age{max_age_days}:n{max_items}"
    fresh = cache_get(key, allow_stale=False)
    if fresh is not None:
        return fresh
    stale = cache_get(key, allow_stale=True)
    try:
        items = _fetch_rss_items_uncached(url, timeout=timeout, max_items=max_items, max_age_days=max_age_days)
        return cache_set(key, items, ttl_seconds=ttl_seconds, stale_ttl_seconds=stale_ttl_seconds)
    except Exception:
        if stale is not None:
            return stale
        raise


def _normalize_title_key(t: str) -> str:
    s = (t or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s\-\.\:]", "", s)
    return s[:220]


def _normalize_link_key(link: str) -> str:
    s = (link or "").strip()
    return s[:140]


def _dedup_items(items: List[dict], max_items: int) -> List[dict]:
    seen = set()
    out = []
    for x in items:
        title = str(x.get("title") or "").strip()
        pub = str(x.get("published") or "").strip()
        day = ""
        dt = _parse_rss_date(pub)
        if dt is not None:
            day = dt.date().isoformat()

        tkey = _normalize_title_key(title)
        lkey = _normalize_link_key(str(x.get("link") or ""))

        k = f"{day}|{tkey}" if tkey else f"{day}|{lkey}"
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(x)
        if len(out) >= max_items:
            break
    return out


def _headline_summary_safe(title: str, raw_summary: str) -> str:
    t = (title or "").strip()
    s = (raw_summary or "").strip()
    if not s:
        return ""

    t_low = t.lower()
    s_low = s.lower()

    if t_low and (s_low.startswith(t_low) or t_low in s_low[: max(60, len(t_low) + 10)]):
        return ""

    t_words = {w for w in re.findall(r"[a-zA-Z]{3,}", t_low)}
    s_words = {w for w in re.findall(r"[a-zA-Z]{3,}", s_low)}
    if t_words and s_words:
        overlap = len(t_words & s_words) / max(1, len(t_words))
        if overlap >= 0.75:
            return ""

    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > 240:
        s = s[:240].rsplit(" ", 1)[0].strip() + "…"
    return s


# =========================================================
# Lightweight sentiment + summaries from titles (unchanged)
# =========================================================
_STOPWORDS = {
    "the","a","an","and","or","to","of","in","on","for","with","as","at","by","from","into",
    "over","after","before","than","is","are","was","were","be","been","being","it","its",
    "this","that","these","those","you","your","they","their","we","our","us","will","may",
    "new","today","latest","live","update","reports","report","says","say","said"
}

_TICKER_RE = re.compile(r"\b[A-Z]{1,5}\b")

_POS_WORDS = {
    "beats","beat","surge","soar","record","upgrade","strong","growth","profit","raises",
    "rally","bullish","wins","approval","partnership","acquisition","buyback"
}
_NEG_WORDS = {
    "miss","misses","slump","falls","drop","downgrade","weak","layoff","cuts","probe",
    "lawsuit","ban","halt","recall","fraud","warning"
}

def _extract_tickers_from_titles(titles: List[str]) -> List[str]:
    bad = {"A","I","AN","THE","AND","OR","TO","OF","IN","ON","US","AI"}
    out = []
    for t in titles:
        for x in _TICKER_RE.findall((t or "").upper()):
            if x in bad:
                continue
            out.append(x)
    seen = set()
    uniq = []
    for x in out:
        if x in seen:
            continue
        seen.add(x)
        uniq.append(x)
    return uniq[:30]

def _top_terms(titles: List[str], k: int = 12) -> List[str]:
    counts: Dict[str, int] = {}
    for t in titles:
        s = (t or "").lower()
        words = re.findall(r"[a-zA-Z]{3,}", s)
        for w in words:
            if w in _STOPWORDS:
                continue
            counts[w] = counts.get(w, 0) + 1
    ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    return [w for w, _c in ranked[:k]]

def _sentiment_from_titles(titles: List[str]) -> Tuple[int, str]:
    if not titles:
        return 50, "NEUTRAL"

    pos = 0
    neg = 0
    for t in titles:
        s = (t or "").lower()
        pos += sum(1 for w in _POS_WORDS if w in s)
        neg += sum(1 for w in _NEG_WORDS if w in s)

    ratio = (pos + 1.0) / (neg + 1.0)
    score = int(round(50 + 18 * (ratio - 1.0)))
    score = max(15, min(85, score))
    label = "BULLISH" if score >= 62 else "BEARISH" if score <= 38 else "NEUTRAL"
    return score, label


# =========================================================
# News briefing (kept as you had it)
# =========================================================
@app.get("/news/briefing")
def news_briefing(
    tickers: str = Query(default=""),
    max_general: int = Query(default=60, ge=10, le=200),
    max_per_ticker: int = Query(default=6, ge=1, le=30),
    sectors: str = Query(default="AI,Medical,Energy,Robotics,Infrastructure,Semiconductors,Cloud,Cybersecurity,Defense,Financials,Consumer"),
    max_items_per_sector: int = Query(default=12, ge=5, le=30),
    max_age_days: int = Query(default=30, ge=7, le=45),
):
    key = f"news:brief:v4121:{tickers}:{max_general}:{max_per_ticker}:{sectors}:{max_items_per_sector}:{max_age_days}"
    cached = cache_get(key, allow_stale=True)
    if cached is not None:
        return cached

    watch = [x.strip().upper() for x in (tickers or "").split(",") if x.strip()]
    sector_list = [s.strip() for s in (sectors or "").split(",") if s.strip()]
    if not sector_list:
        sector_list = ["General"]

    errors: List[str] = []
    all_items: List[dict] = []

    jobs: List[Tuple[str, str]] = []
    for sec in sector_list[:18]:
        jobs.append((sec, _google_news_rss(f"{sec} stocks markets")))

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [(sec, ex.submit(_fetch_rss_items, url, 10, max_items_per_sector * 3, 240, 6 * 3600, max_age_days)) for sec, url in jobs]
        for sec, fut in futs:
            try:
                items = fut.result()
                for x in items:
                    x["sector"] = sec
                    x["source"] = "Google News"
                all_items.extend(items)
            except Exception as e:
                errors.append(f"{sec}: {type(e).__name__}: {str(e)}")

    if watch:
        ticker_jobs: List[Tuple[str, str]] = []
        for t in watch[:40]:
            ticker_jobs.append((t, _google_news_rss(f"{t} stock")))
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs2 = [(t, ex.submit(_fetch_rss_items, url, 10, max_per_ticker * 2, 240, 6 * 3600, max_age_days)) for t, url in ticker_jobs]
            for t, fut in futs2:
                try:
                    items = fut.result()
                    for x in items:
                        x["sector"] = "Watchlist"
                        x["source"] = "Google News"
                        x["ticker"] = t
                    all_items.extend(items)
                except Exception as e:
                    errors.append(f"{t}: {type(e).__name__}: {str(e)}")

    all_items = _dedup_items(all_items, 1500)

    sectors_out: List[dict] = []

    def _simple_sector_summary(sec_name: str, titles: List[str]) -> Tuple[str, List[str]]:
        if not titles:
            return (f"{sec_name}: No fresh items in the last {max_age_days} days.", [])
        terms = _top_terms(titles, k=10)[:6]
        tickers = _extract_tickers_from_titles(titles)[:10]
        s = []
        if terms:
            s.append(f"Main themes: {', '.join(terms)}.")
        if tickers:
            s.append(f"Tickers referenced: {', '.join(tickers)}.")
        return (" ".join(s).strip() or f"{sec_name}: Headlines loaded."), []

    # Watchlist
    if watch:
        wl_items = [x for x in all_items if x.get("sector") == "Watchlist"][:max_general]
        wl_items = _dedup_items(wl_items, max_general)
        wl_titles = [x.get("title", "") for x in wl_items]
        wl_score, wl_label = _sentiment_from_titles(wl_titles)
        wl_sum, _ = _simple_sector_summary("Watchlist", wl_titles)

        sectors_out.append({
            "sector": "Watchlist",
            "sentiment": {"label": wl_label, "score": wl_score},
            "summary": wl_sum,
            "topHeadlines": [
                {
                    "title": x.get("title", ""),
                    "link": x.get("link", ""),
                    "published": x.get("published", ""),
                    "sourceFeed": x.get("source", "Google News"),
                    "sector": "Watchlist",
                    "summary": _headline_summary_safe(x.get("title", ""), x.get("rawSummary", "")),
                }
                for x in wl_items
            ],
        })

    # Other sectors
    for sec in sector_list:
        sec_items = [x for x in all_items if x.get("sector") == sec][:max_items_per_sector]
        sec_titles = [x.get("title", "") for x in sec_items]
        score, label = _sentiment_from_titles(sec_titles)
        sec_sum, _ = _simple_sector_summary(sec, sec_titles)

        sectors_out.append({
            "sector": sec,
            "sentiment": {"label": label, "score": score},
            "summary": sec_sum,
            "topHeadlines": [
                {
                    "title": x.get("title", ""),
                    "link": x.get("link", ""),
                    "published": x.get("published", ""),
                    "sourceFeed": x.get("source", "Google News"),
                    "sector": sec,
                    "summary": _headline_summary_safe(x.get("title", ""), x.get("rawSummary", "")),
                }
                for x in sec_items
            ],
        })

    all_titles = []
    for s in sectors_out:
        all_titles.extend([h.get("title", "") for h in (s.get("topHeadlines") or [])])

    overall_score, overall_label = _sentiment_from_titles(all_titles)

    out = {
        "date": datetime.now(timezone.utc).date().isoformat(),
        "overallSentiment": {"label": overall_label, "score": overall_score},
        "sectors": sectors_out,
        "errors": errors,
        "note": f"Google News RSS. Headlines limited to last {max_age_days} days.",
    }
    return cache_set(key, out, ttl_seconds=300, stale_ttl_seconds=3 * 3600)


# =========================================================
# Crypto briefing endpoint (kept)
# =========================================================
@app.get("/crypto/news/briefing")
def crypto_news_briefing(
    coins: str = Query(default="BTC,ETH,LINK,SHIB"),
    include_top_n: int = Query(default=15, ge=5, le=50),
    max_age_days: int = Query(default=30, ge=7, le=45),
):
    key = f"crypto:news:v4121:{coins}:{include_top_n}:{max_age_days}"
    cached = cache_get(key, allow_stale=True)
    if cached is not None:
        return cached

    now = datetime.now(timezone.utc)
    errors: List[str] = []
    coin_list = [c.strip().upper() for c in (coins or "").split(",") if c.strip()]
    if not coin_list:
        coin_list = ["BTC", "ETH"]

    all_titles: List[str] = []
    coins_out: List[dict] = []

    for c in coin_list[:25]:
        try:
            items = _fetch_rss_items(
                _google_news_rss(f"{c} crypto"),
                timeout=10,
                max_items=max(10, include_top_n),
                ttl_seconds=240,
                stale_ttl_seconds=6 * 3600,
                max_age_days=max_age_days,
            )
            items = _dedup_items(items, include_top_n)
            titles = [x.get("title", "") for x in items]

            score, label = _sentiment_from_titles(titles)
            all_titles.extend(titles)

            coins_out.append({
                "symbol": c,
                "sentiment": {"label": label, "score": score},
                "summary": f"{c}: {len(items)} headlines.",
                "headlines": [
                    {
                        "title": x.get("title", ""),
                        "link": x.get("link", ""),
                        "published": x.get("published", ""),
                        "source": "Google News",
                        "summary": _headline_summary_safe(x.get("title", ""), x.get("rawSummary", "")),
                    }
                    for x in items
                ]
            })
        except Exception as e:
            errors.append(f"{c}: {type(e).__name__}: {str(e)}")
            coins_out.append({
                "symbol": c,
                "sentiment": {"label": "NEUTRAL", "score": 50},
                "summary": f"{c}: No fresh items.",
                "headlines": []
            })

    overall_score, overall_label = _sentiment_from_titles(all_titles)

    out = {
        "date": now.date().isoformat(),
        "overallSentiment": {"label": overall_label, "score": overall_score},
        "coins": coins_out,
        "errors": errors,
        "note": f"Google News RSS. Headlines limited to last {max_age_days} days.",
    }
    return cache_set(key, out, ttl_seconds=300, stale_ttl_seconds=3 * 3600)


# =========================================================
# Congress endpoints (unchanged)
# =========================================================
_party_re = re.compile(r"/\s*([DR])\b", re.IGNORECASE)


def _pick_first(row: dict, keys: List[str], default=""):
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return default


def _parse_dt_any(v: Any) -> Optional[datetime]:
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
    for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d %H:%M:%S"]:
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return None


def _iso_date_only(dt: Optional[datetime]) -> str:
    return dt.date().isoformat() if dt else ""


def _norm_party(row: dict) -> str:
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


def _norm_ticker(row: dict) -> str:
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


def _tx_text(row: dict) -> str:
    for k in ["Transaction", "transaction", "TransactionType", "Type", "type"]:
        v = row.get(k)
        if v:
            return str(v).strip()
    return ""


def _is_buy(tx: str) -> bool:
    s = (tx or "").lower()
    return ("purchase" in s) or ("buy" in s)


def _is_sell(tx: str) -> bool:
    s = (tx or "").lower()
    return ("sale" in s) or ("sell" in s) or ("sold" in s)


def _row_best_dt(row: dict) -> Optional[datetime]:
    traded = _pick_first(row, ["Traded", "traded", "TransactionDate", "transaction_date"], "")
    filed = _pick_first(row, ["Filed", "filed", "ReportDate", "report_date", "Date", "date"], "")
    return _parse_dt_any(traded) or _parse_dt_any(filed)


def _row_chamber(row: dict) -> str:
    for k in ["Chamber", "chamber", "House", "house"]:
        v = row.get(k)
        if v:
            s = str(v).strip()
            if s:
                return s
    name = str(_pick_first(row, ["Politician", "politician", "Representative", "Senator"], "")).lower()
    if "sen" in name:
        return "Senate"
    if "rep" in name or "house" in name:
        return "House"
    return ""


def _amount_range(row: dict) -> str:
    for k in ["Amount", "amount", "Range", "range", "AmountRange", "amount_range"]:
        v = row.get(k)
        if v:
            s = str(v).strip()
            if s:
                return s
    return ""


@app.get("/report/holdings/common")
def holdings_common(window_days: int = Query(default=365, ge=30, le=365), top_n: int = Query(default=30, ge=5, le=200)):
    if not QUIVER_TOKEN:
        raise HTTPException(500, "QUIVER_TOKEN missing")

    key = f"holdings:common:v4121:{window_days}:{top_n}"
    cached = cache_get(key, allow_stale=True)
    if cached is not None:
        return cached

    now = datetime.now(timezone.utc)
    since = now - timedelta(days=window_days)

    q = quiverquant.quiver(QUIVER_TOKEN)
    df = q.congress_trading()

    if df is None or len(df) == 0:
        out = {"date": now.date().isoformat(), "windowDays": window_days, "commonHoldings": []}
        return cache_set(key, out, ttl_seconds=300, stale_ttl_seconds=3600)

    rows = df.to_dict(orient="records") if hasattr(df, "to_dict") else list(df)

    holders_by_ticker: Dict[str, set] = {}

    for r in rows:
        best_dt = _row_best_dt(r)
        if best_dt is None or best_dt < since or best_dt > now:
            continue

        ticker = _norm_ticker(r)
        if not ticker:
            continue

        pol = str(_pick_first(r, ["Politician", "politician", "Representative", "Senator", "Name", "name"], "")).strip()
        if not pol:
            continue

        holders_by_ticker.setdefault(ticker.upper(), set()).add(pol)

    items = [{"ticker": t, "holders": len(pols)} for t, pols in holders_by_ticker.items()]
    items.sort(key=lambda x: (-int(x["holders"]), str(x["ticker"])))
    out = {"date": now.date().isoformat(), "windowDays": window_days, "commonHoldings": items[:top_n]}
    return cache_set(key, out, ttl_seconds=300, stale_ttl_seconds=3600)


@app.get("/report/congress/daily")
def congress_daily(window_days: int = Query(default=30, ge=1, le=365), limit: int = Query(default=250, ge=50, le=1000)):
    if not QUIVER_TOKEN:
        raise HTTPException(500, "QUIVER_TOKEN missing")

    key = f"congress:daily:v4121:{window_days}:{limit}"
    cached = cache_get(key, allow_stale=True)
    if cached is not None:
        return cached

    now = datetime.now(timezone.utc)
    since = now - timedelta(days=window_days)

    q = quiverquant.quiver(QUIVER_TOKEN)
    df = q.congress_trading()

    if df is None or len(df) == 0:
        out = {"date": now.date().isoformat(), "windowDays": window_days, "days": []}
        return cache_set(key, out, ttl_seconds=120, stale_ttl_seconds=900)

    rows = df.to_dict(orient="records") if hasattr(df, "to_dict") else list(df)

    items: List[dict] = []
    for r in rows:
        best_dt = _row_best_dt(r)
        if best_dt is None or best_dt < since or best_dt > now:
            continue

        tx = _tx_text(r)
        kind = "BUY" if _is_buy(tx) else "SELL" if _is_sell(tx) else ""
        if not kind:
            continue

        ticker = _norm_ticker(r)
        if not ticker:
            continue

        party = _norm_party(r)
        pol = str(_pick_first(r, ["Politician", "politician", "Representative", "Senator", "Name", "name"], "")).strip()
        filed_dt = _parse_dt_any(_pick_first(r, ["Filed", "filed", "ReportDate", "report_date", "Date", "date"], ""))
        traded_dt = _parse_dt_any(_pick_first(r, ["Traded", "traded", "TransactionDate", "transaction_date"], ""))

        chamber = _row_chamber(r)
        amt = _amount_range(r)
        desc = str(_pick_first(r, ["Description", "description", "AssetDescription", "asset_description"], "")).strip()

        items.append({
            "kind": kind,
            "ticker": ticker.upper(),
            "politician": pol,
            "party": party,
            "chamber": chamber,
            "amountRange": amt,
            "traded": _iso_date_only(traded_dt),
            "filed": _iso_date_only(filed_dt),
            "description": desc,
            "_best_dt": best_dt,
        })

    items.sort(key=lambda x: x.get("_best_dt") or datetime(1970, 1, 1, tzinfo=timezone.utc), reverse=True)
    items = items[:limit]

    by_day: Dict[str, List[dict]] = {}
    for it in items:
        d = (it.get("filed") or it.get("traded") or _iso_date_only(it.get("_best_dt")) or now.date().isoformat())
        it.pop("_best_dt", None)
        by_day.setdefault(d, []).append(it)

    day_list = [{"date": d, "items": by_day[d]} for d in sorted(by_day.keys(), reverse=True)]
    out = {"date": now.date().isoformat(), "windowDays": window_days, "days": day_list}
    return cache_set(key, out, ttl_seconds=120, stale_ttl_seconds=900)
