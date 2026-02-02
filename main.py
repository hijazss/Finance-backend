import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus, urlsplit, urlunsplit
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import requests
import quiverquant
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware


# =========================================================
# App
# =========================================================
app = FastAPI(title="Finance Signals Backend", version="4.9.0")

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

SESSION = requests.Session()


@app.get("/")
def root():
    return {"status": "ok", "version": "4.9.0"}


@app.get("/health")
def health():
    return {
        "ok": True,
        "version": "4.9.0",
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
# Provider cooldowns
# =========================================================
_PROVIDER_COOLDOWN_UNTIL: Dict[str, float] = {}


def _cooldown(provider: str, seconds: int) -> None:
    _PROVIDER_COOLDOWN_UNTIL[provider] = time.time() + float(seconds)


def _is_cooled_down(provider: str) -> bool:
    return time.time() < _PROVIDER_COOLDOWN_UNTIL.get(provider, 0.0)


# =========================================================
# HTTP helpers
# =========================================================
def _requests_get(url: str, params: Optional[dict] = None, timeout: int = 16, headers: Optional[dict] = None) -> requests.Response:
    return SESSION.get(url, params=params, timeout=timeout, headers=headers or UA_HEADERS, allow_redirects=True)


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


def _sma(vals: List[float], n: int) -> Optional[float]:
    if n <= 0 or len(vals) < n:
        return None
    return sum(vals[-n:]) / n


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


# =========================================================
# Nasdaq quote (best effort)
# =========================================================
def _nasdaq_assetclass_for_symbol(symbol: str) -> str:
    s = (symbol or "").upper().strip()
    if s in ["SPY", "QQQ", "DIA", "IWM"]:
        return "etf"
    if s in ["VIX", "^VIX", "NDX", "^NDX", "SPX", "^SPX"]:
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
    return s


def _nasdaq_quote(symbol: str, assetclass: Optional[str] = None) -> dict:
    if _is_cooled_down("nasdaq"):
        raise RuntimeError("Nasdaq in cooldown")

    sym = _nasdaq_symbol_normalize(symbol)
    ac = assetclass or _nasdaq_assetclass_for_symbol(sym)

    url = f"https://api.nasdaq.com/api/quote/{quote_plus(sym)}/info"
    r = _requests_get(url, params={"assetclass": ac}, timeout=14, headers=NASDAQ_HEADERS)
    if r.status_code == 429:
        _cooldown("nasdaq", 10 * 60)
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
# Stooq history (stable)
# =========================================================
def _stooq_daily_closes(symbol: str) -> List[Tuple[datetime, float]]:
    if _is_cooled_down("stooq"):
        raise RuntimeError("Stooq in cooldown")

    url = "https://stooq.com/q/d/l/"
    params = {"s": symbol, "i": "d"}

    last_err = None
    for attempt in range(4):
        try:
            r = _requests_get(url, params=params, timeout=22, headers=UA_HEADERS)
            if r.status_code >= 400:
                last_err = f"HTTP {r.status_code}"
                time.sleep(min(2.0, 0.6 * (attempt + 1)))
                continue

            lines = (r.text or "").strip().splitlines()
            if len(lines) < 3:
                last_err = "insufficient CSV rows"
                time.sleep(min(2.0, 0.6 * (attempt + 1)))
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
            time.sleep(min(2.0, 0.6 * (attempt + 1)))
        except requests.exceptions.Timeout as e:
            last_err = f"Timeout: {type(e).__name__}"
            time.sleep(min(2.0, 0.6 * (attempt + 1)))
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:160]}"
            time.sleep(min(2.0, 0.6 * (attempt + 1)))

    _cooldown("stooq", 120)
    raise RuntimeError(f"Stooq failed: {last_err or 'unknown'}")


def _ret_from_series(vals: List[float], offset: int) -> Optional[float]:
    if len(vals) < (offset + 1):
        return None
    a = float(vals[-1])
    b = float(vals[-1 - offset])
    return _pct(a, b)


# =========================================================
# CNN Fear & Greed (best effort)
# =========================================================
def _cnn_fear_greed_graphdata(date_str: Optional[str] = None) -> dict:
    d = date_str or datetime.now(timezone.utc).date().isoformat()
    url = f"https://production.dataviz.cnn.io/index/fearandgreed/graphdata/{d}"
    r = _requests_get(url, timeout=16, headers=UA_HEADERS)
    r.raise_for_status()
    return r.json()


@app.get("/market/fear-greed")
def market_fear_greed(date: Optional[str] = Query(default=None)):
    key = f"feargreed:{date or 'today'}"
    cached = cache_get(key, allow_stale=True)
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
        }
        return cache_set(key, out, ttl_seconds=900, stale_ttl_seconds=6 * 3600)
    except Exception as e:
        stale = cache_get(key, allow_stale=True)
        if stale is not None:
            return stale
        return cache_set(
            key,
            {
                "date": date or datetime.now(timezone.utc).date().isoformat(),
                "score": None,
                "rating": None,
                "error": f"{type(e).__name__}: {str(e)}",
            },
            ttl_seconds=120,
            stale_ttl_seconds=900,
        )


# =========================================================
# Daily market read
# =========================================================
@app.get("/market/read")
def market_read():
    key = "market:read:v490"
    cached = cache_get(key, allow_stale=True)
    if cached is not None:
        return cached

    now = datetime.now(timezone.utc)
    errors: List[str] = []

    spy_vals: List[float] = []
    qqq_vals: List[float] = []
    vix_vals: List[float] = []

    try:
        spy = _stooq_daily_closes("spy.us")
        spy_vals = [c for _, c in spy]
    except Exception as e:
        errors.append(f"Stooq SPY: {type(e).__name__}: {str(e)}")

    try:
        qqq = _stooq_daily_closes("qqq.us")
        qqq_vals = [c for _, c in qqq]
    except Exception as e:
        errors.append(f"Stooq QQQ: {type(e).__name__}: {str(e)}")

    try:
        vix = _stooq_daily_closes("vix")
        vix_vals = [c for _, c in vix]
    except Exception as e:
        errors.append(f"Stooq VIX: {type(e).__name__}: {str(e)}")

    def _fallback_last_prev(symbol: str) -> Tuple[Optional[float], Optional[float]]:
        try:
            last, prev = _nasdaq_last_and_prev(symbol)
            return (float(last) if last else None), (float(prev) if prev else None)
        except Exception as e:
            errors.append(f"Nasdaq {symbol}: {type(e).__name__}: {str(e)}")
            return None, None

    spy_last = float(spy_vals[-1]) if spy_vals else None
    qqq_last = float(qqq_vals[-1]) if qqq_vals else None
    vix_last = float(vix_vals[-1]) if vix_vals else None

    if spy_last is None:
        last, _prev = _fallback_last_prev("SPY")
        spy_last = last
    if qqq_last is None:
        last, _prev = _fallback_last_prev("QQQ")
        qqq_last = last
    if vix_last is None:
        last, _prev = _fallback_last_prev("VIX")
        vix_last = last

    spy_1d = _ret_from_series(spy_vals, 1)
    spy_5d = _ret_from_series(spy_vals, 5)
    spy_1m = _ret_from_series(spy_vals, 21)

    qqq_1d = _ret_from_series(qqq_vals, 1)
    qqq_5d = _ret_from_series(qqq_vals, 5)
    qqq_1m = _ret_from_series(qqq_vals, 21)

    fg = {"score": None, "rating": None}
    try:
        fgd = market_fear_greed(None)
        fg = {"score": fgd.get("score"), "rating": fgd.get("rating")}
    except Exception as e:
        errors.append(f"FearGreed: {type(e).__name__}: {str(e)}")

    def _fmt_pct(x: Optional[float]) -> str:
        if x is None:
            return "—"
        s = "+" if x > 0 else ""
        return f"{s}{x:.2f}%"

    parts: List[str] = []
    if spy_last is not None:
        parts.append(f"SPY {spy_last:.2f} (1D {_fmt_pct(spy_1d)}, 5D {_fmt_pct(spy_5d)}, 1M {_fmt_pct(spy_1m)}).")
    else:
        parts.append("SPY unavailable.")

    if qqq_last is not None:
        parts.append(f"QQQ {qqq_last:.2f} (1D {_fmt_pct(qqq_1d)}, 5D {_fmt_pct(qqq_5d)}, 1M {_fmt_pct(qqq_1m)}).")
    else:
        parts.append("QQQ unavailable.")

    if vix_last is not None:
        parts.append(f"VIX {vix_last:.2f}.")
    if isinstance(fg.get("score"), (int, float)):
        parts.append(f"Fear & Greed {fg.get('score')} ({fg.get('rating') or '—'}).")

    out = {
        "date": now.date().isoformat(),
        "summary": " ".join(parts).strip(),
        "sp500": {"symbol": "SPY", "last": spy_last, "ret1dPct": spy_1d, "ret5dPct": spy_5d, "ret1mPct": spy_1m},
        "nasdaq": {"symbol": "QQQ", "last": qqq_last, "ret1dPct": qqq_1d, "ret5dPct": qqq_5d, "ret1mPct": qqq_1m},
        "vix": {"symbol": "^VIX", "last": vix_last},
        "fearGreed": fg,
        "errors": errors,
        "note": "Uses Stooq daily history first; falls back to Nasdaq quote when needed.",
    }
    return cache_set(key, out, ttl_seconds=180, stale_ttl_seconds=1800)


# =========================================================
# RSS + freshness + dedupe helpers
# =========================================================
def _google_news_rss(query: str) -> str:
    q = quote_plus(query)
    return f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"


def _normalize_url_for_dedupe(url: str) -> str:
    if not url:
        return ""
    try:
        u = url.strip()
        sp = urlsplit(u)
        # Drop query/fragment so Google tracking variations do not create duplicates
        return urlunsplit((sp.scheme, sp.netloc, sp.path, "", ""))
    except Exception:
        return url.strip()


def _parse_rss_datetime(s: str) -> Optional[datetime]:
    if not s:
        return None
    txt = s.strip()
    # Atom updated often ISO
    try:
        dt = datetime.fromisoformat(txt.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        pass

    # Common RSS: "Wed, 31 Jan 2026 15:04:05 GMT"
    fmts = [
        "%a, %d %b %Y %H:%M:%S %Z",
        "%a, %d %b %Y %H:%M:%S %z",
        "%d %b %Y %H:%M:%S %Z",
        "%Y-%m-%d",
    ]
    for f in fmts:
        try:
            dt = datetime.strptime(txt, f)
            return dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return None


def _fetch_rss_items_uncached(url: str, timeout: int = 10, max_items: int = 25) -> List[dict]:
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

    channel = root.find("channel")
    if channel is None:
        entries = root.findall("{http://www.w3.org/2005/Atom}entry")
        out = []
        for e in entries[:max_items]:
            title = (e.findtext("{http://www.w3.org/2005/Atom}title") or "").strip()
            link_el = e.find("{http://www.w3.org/2005/Atom}link")
            link = (link_el.get("href") if link_el is not None else "") or ""
            pub = (e.findtext("{http://www.w3.org/2005/Atom}updated") or "").strip()
            if title:
                out.append({"title": title, "link": link, "published": pub})
        return out

    out = []
    for item in channel.findall("item")[:max_items]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        if title:
            out.append({"title": title, "link": link, "published": pub})
    return out


def _fetch_rss_items(url: str, timeout: int = 10, max_items: int = 25, ttl_seconds: int = 240, stale_ttl_seconds: int = 6 * 3600) -> List[dict]:
    key = f"rss:{url}"
    fresh = cache_get(key, allow_stale=False)
    if fresh is not None:
        return fresh
    stale = cache_get(key, allow_stale=True)
    try:
        items = _fetch_rss_items_uncached(url, timeout=timeout, max_items=max_items)
        return cache_set(key, items, ttl_seconds=ttl_seconds, stale_ttl_seconds=stale_ttl_seconds)
    except Exception:
        if stale is not None:
            return stale
        raise


def _is_within_days(published: str, max_age_days: int) -> bool:
    if not published:
        return True  # some feeds omit, keep but it will be deduped
    dt = _parse_rss_datetime(published)
    if not dt:
        return True
    now = datetime.now(timezone.utc)
    return dt >= (now - timedelta(days=max_age_days))


def _dedup_items(items: List[dict], max_items: int) -> List[dict]:
    seen = set()
    out = []
    for x in items:
        lk = _normalize_url_for_dedupe((x.get("link") or "").strip())
        ttl = (x.get("title") or "").strip()
        ttl_key = re.sub(r"\s+", " ", ttl.lower()).strip()

        # Combine to catch "same headline, different Google wrapper URL"
        k = lk or ttl_key
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(x)
        if len(out) >= max_items:
            break
    return out


def _dedup_within_list_by_title(items: List[dict]) -> List[dict]:
    seen = set()
    out = []
    for x in items:
        ttl = re.sub(r"\s+", " ", str(x.get("title") or "").lower()).strip()
        if not ttl or ttl in seen:
            continue
        seen.add(ttl)
        out.append(x)
    return out


# =========================================================
# Headline summary extraction (lazy)
# =========================================================
_TAG_RE = re.compile(r"<[^>]+>")
_META_OG_DESC = re.compile(r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\']([^"\']+)["\']', re.IGNORECASE)
_META_DESC = re.compile(r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']+)["\']', re.IGNORECASE)
_META_TW_DESC = re.compile(r'<meta[^>]+name=["\']twitter:description["\'][^>]+content=["\']([^"\']+)["\']', re.IGNORECASE)


def _clean_text(s: str) -> str:
    s2 = (s or "").replace("&amp;", "&").replace("&quot;", '"').replace("&#39;", "'").strip()
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2


def _html_to_text(html: str) -> str:
    if not html:
        return ""
    # Remove script/style quickly
    html = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
    txt = _TAG_RE.sub(" ", html)
    txt = _clean_text(txt)
    return txt


def _extract_description_from_html(html: str) -> str:
    if not html:
        return ""
    for rx in (_META_OG_DESC, _META_DESC, _META_TW_DESC):
        m = rx.search(html)
        if m and m.group(1):
            return _clean_text(m.group(1))
    return ""


def _headline_summary_from_url(url: str) -> dict:
    if not url:
        return {"summary": "", "finalUrl": "", "error": "missing url"}

    key = f"sum:{_normalize_url_for_dedupe(url)}"
    cached = cache_get(key, allow_stale=True)
    if cached is not None:
        return cached

    try:
        r = _requests_get(url, timeout=14, headers=UA_HEADERS)
        final_url = getattr(r, "url", url)
        html = (r.text or "")[:350_000]

        desc = _extract_description_from_html(html)
        if desc and len(desc) >= 50:
            out = {"summary": desc[:420], "finalUrl": final_url, "error": ""}
            return cache_set(key, out, ttl_seconds=12 * 3600, stale_ttl_seconds=48 * 3600)

        # fallback: take first short chunk of visible text
        txt = _html_to_text(html)
        # crude "first sentences"
        snippet = txt[:900]
        # trim at sentence-ish boundary
        cut = re.split(r"(?<=[.!?])\s+", snippet)
        summary = " ".join(cut[:2]).strip()
        summary = summary[:420]

        out = {"summary": summary, "finalUrl": final_url, "error": ""}
        return cache_set(key, out, ttl_seconds=6 * 3600, stale_ttl_seconds=48 * 3600)

    except Exception as e:
        out = {"summary": "", "finalUrl": url, "error": f"{type(e).__name__}: {str(e)[:200]}"}
        return cache_set(key, out, ttl_seconds=600, stale_ttl_seconds=3600)


@app.get("/news/article/summary")
def news_article_summary(url: str = Query(..., min_length=5)):
    return _headline_summary_from_url(url)


@app.get("/crypto/article/summary")
def crypto_article_summary(url: str = Query(..., min_length=5)):
    return _headline_summary_from_url(url)


# =========================================================
# Sentiment + implications (unchanged)
# =========================================================
_TICKER_RE = re.compile(r"\b[A-Z]{1,5}\b")


def _extract_tickers_from_text(text: str) -> List[str]:
    if not text:
        return []
    bad = {"A", "I", "AN", "THE", "AND", "OR", "TO", "OF", "IN", "ON", "US", "AI"}
    out = []
    for t in _TICKER_RE.findall(text.upper()):
        if t in bad:
            continue
        out.append(t)
    seen = set()
    uniq = []
    for t in out:
        if t in seen:
            continue
        seen.add(t)
        uniq.append(t)
    return uniq[:12]


_POS_WORDS = {
    "beats", "beat", "surge", "soar", "record", "upgrade", "strong", "growth", "profit", "raises",
    "rally", "bullish", "wins", "approval", "partnership", "acquisition", "buyback"
}
_NEG_WORDS = {
    "miss", "misses", "slump", "falls", "drop", "downgrade", "weak", "layoff", "cuts", "probe",
    "lawsuit", "ban", "halt", "recall", "fraud", "warning"
}


def _sentiment_from_titles(titles: List[str]) -> Tuple[int, str]:
    if not titles:
        return 50, "NEUTRAL"
    score = 50
    for t in titles:
        s = (t or "").lower()
        pos = sum(1 for w in _POS_WORDS if w in s)
        neg = sum(1 for w in _NEG_WORDS if w in s)
        score += 4 * pos
        score -= 5 * neg
    score = int(max(0, min(100, score)))
    label = "BULLISH" if score >= 62 else "BEARISH" if score <= 38 else "NEUTRAL"
    return score, label


def _flowing_summary(sector: str, titles: List[str], watchlist: List[str]) -> Tuple[str, List[str], List[str]]:
    picked = []
    seen = set()
    for t in titles:
        tt = (t or "").strip()
        if not tt:
            continue
        key = tt.lower()
        if key in seen:
            continue
        seen.add(key)
        picked.append(tt)
        if len(picked) >= 6:
            break

    tickers = []
    for t in picked:
        tickers.extend(_extract_tickers_from_text(t))

    wl_upper = [w.upper() for w in watchlist if w]
    mention = [t for t in tickers if t in wl_upper][:8]

    if not picked:
        return "", [], []

    lead = f"Today in {sector}: "
    clauses = [t.rstrip(".") for t in picked[:4]]
    summary = lead + "; ".join(clauses) + "."

    implications: List[str] = []
    if mention:
        implications.append(f"Watchlist names showing up in headlines: {', '.join(mention)}.")
    if tickers:
        extra = [t for t in tickers if t not in mention][:8]
        if extra:
            implications.append(f"Other tickers mentioned: {', '.join(extra)}.")
    implications.append("If this sector stays headline-heavy, prefer scaling in (small adds) over chasing gaps up.")
    implications.append("If you see repeated negative headlines around guidance, consider waiting for a better entry or sizing smaller.")

    return summary, implications, (mention or tickers)[:12]


# =========================================================
# News briefing (recency + dedupe improvements)
# =========================================================
@app.get("/news/briefing")
def news_briefing(
    tickers: str = Query(default=""),
    max_general: int = Query(default=60, ge=10, le=200),
    max_per_ticker: int = Query(default=6, ge=1, le=30),
    sectors: str = Query(default="AI,Medical,Energy,Robotics,Infrastructure,Semiconductors,Cloud,Cybersecurity,Defense,Financials,Consumer"),
    max_items_per_sector: int = Query(default=12, ge=5, le=30),
    max_age_days: int = Query(default=30, ge=7, le=60),
):
    key = f"news:brief:v490:{tickers}:{max_general}:{max_per_ticker}:{sectors}:{max_items_per_sector}:{max_age_days}"
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
        futs = [(sec, ex.submit(_fetch_rss_items, url, 10, max_items_per_sector * 3, 240, 6 * 3600)) for sec, url in jobs]
        for sec, fut in futs:
            try:
                items = fut.result()
                items = [x for x in items if _is_within_days(str(x.get("published") or ""), max_age_days)]
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
            futs2 = [(t, ex.submit(_fetch_rss_items, url, 10, max_per_ticker * 3, 240, 6 * 3600)) for t, url in ticker_jobs]
            for t, fut in futs2:
                try:
                    items = fut.result()
                    items = [x for x in items if _is_within_days(str(x.get("published") or ""), max_age_days)]
                    for x in items:
                        x["sector"] = "Watchlist"
                        x["source"] = "Google News"
                        x["ticker"] = t
                    all_items.extend(items)
                except Exception as e:
                    errors.append(f"{t}: {type(e).__name__}: {str(e)}")

    all_items = _dedup_items(all_items, 1200)

    sectors_out = []
    for sec in sector_list:
        sec_items = [x for x in all_items if x.get("sector") == sec]
        sec_items = _dedup_within_list_by_title(sec_items)[:max_items_per_sector]

        titles = [x.get("title", "") for x in sec_items]
        score, label = _sentiment_from_titles(titles)
        summary, implications, tickers_mentioned = _flowing_summary(sec, titles, watch)

        watch_mentions = []
        if watch:
            for t in watch:
                if any(re.search(rf"\b{re.escape(t)}\b", (ttl or "").upper()) for ttl in titles):
                    watch_mentions.append(t)
        watch_mentions = list(dict.fromkeys(watch_mentions))[:12]

        sectors_out.append({
            "sector": sec,
            "sentiment": {"label": label, "score": score},
            "summary": summary,
            "implications": implications,
            "topHeadlines": [
                {
                    "title": x.get("title", ""),
                    "link": x.get("link", ""),
                    "published": x.get("published", ""),
                    "sourceFeed": x.get("source", "Google News"),
                    "sector": sec
                    # summary is lazy-loaded via /news/article/summary
                }
                for x in sec_items
            ],
            "watchlistMentions": watch_mentions,
            "tickersMentioned": tickers_mentioned,
        })

    if watch:
        wl_items = [x for x in all_items if x.get("sector") == "Watchlist"]
        wl_items = _dedup_within_list_by_title(wl_items)[:max_general]

        wl_titles = [x.get("title", "") for x in wl_items]
        wl_score, wl_label = _sentiment_from_titles(wl_titles)
        wl_summary, wl_imp, wl_tickers = _flowing_summary("Your watchlist", wl_titles, watch)

        sectors_out.insert(0, {
            "sector": "Watchlist",
            "sentiment": {"label": wl_label, "score": wl_score},
            "summary": wl_summary,
            "implications": wl_imp + ([
                "Names to watch closely: " + ", ".join(watch[:12]) + "."
            ] if watch else []),
            "topHeadlines": [
                {
                    "title": x.get("title", ""),
                    "link": x.get("link", ""),
                    "published": x.get("published", ""),
                    "sourceFeed": x.get("source", "Google News"),
                    "sector": "Watchlist"
                }
                for x in wl_items
            ],
            "watchlistMentions": watch[:25],
            "tickersMentioned": wl_tickers,
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
        "note": f"Briefing uses Google News RSS. Headlines limited to {max_age_days} days. Per-headline summary is lazy via /news/article/summary.",
    }
    return cache_set(key, out, ttl_seconds=300, stale_ttl_seconds=3 * 3600)


# =========================================================
# Congress endpoints (unchanged from your file)
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

    key = f"holdings:common:v490:{window_days}:{top_n}"
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

    key = f"congress:daily:v490:{window_days}:{limit}"
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
            "party": _norm_party(r),
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


# =========================================================
# Crypto briefing (recency + dedupe improvements)
# =========================================================
@app.get("/crypto/news/briefing")
def crypto_news_briefing(
    coins: str = Query(default="BTC,ETH,LINK,SHIB"),
    include_top_n: int = Query(default=15, ge=5, le=50),
    max_age_days: int = Query(default=30, ge=7, le=60),
):
    key = f"crypto:news:v490:{coins}:{include_top_n}:{max_age_days}"
    cached = cache_get(key, allow_stale=True)
    if cached is not None:
        return cached

    now = datetime.now(timezone.utc)
    errors: List[str] = []
    coin_list = [c.strip().upper() for c in (coins or "").split(",") if c.strip()]
    if not coin_list:
        coin_list = ["BTC", "ETH"]

    all_titles: List[str] = []
    coin_blocks: List[dict] = []

    for c in coin_list[:25]:
        try:
            items = _fetch_rss_items(
                _google_news_rss(f"{c} crypto"),
                timeout=10,
                max_items=max(12, include_top_n * 3),
                ttl_seconds=240,
                stale_ttl_seconds=6 * 3600
            )
            items = [x for x in items if _is_within_days(str(x.get("published") or ""), max_age_days)]
            items = _dedup_items(items, 500)
            items = _dedup_within_list_by_title(items)[:include_top_n]

            titles = [x.get("title", "") for x in items]
            score, label = _sentiment_from_titles(titles)
            summary, implications, _tks = _flowing_summary(c, titles, [])
            all_titles.extend(titles)

            coin_blocks.append({
                "symbol": c,
                "sentiment": {"label": label, "score": score},
                "summary": summary or (implications[0] if implications else ""),
                "headlines": [
                    {
                        "title": x.get("title", ""),
                        "link": x.get("link", ""),
                        "published": x.get("published", ""),
                        "source": "Google News"
                        # summary is lazy via /crypto/article/summary
                    }
                    for x in items
                ]
            })
        except Exception as e:
            errors.append(f"{c}: {type(e).__name__}: {str(e)}")
            coin_blocks.append({
                "symbol": c,
                "sentiment": {"label": "NEUTRAL", "score": 50},
                "summary": "",
                "headlines": []
            })

    overall_score, overall_label = _sentiment_from_titles(all_titles)

    out = {
        "date": now.date().isoformat(),
        "overallSentiment": {"label": overall_label, "score": overall_score},
        "catalysts": [
            "Macro risk-on/risk-off can dominate crypto short-term.",
            "Watch for ETF flows, exchange outages, major protocol upgrades, and large unlock schedules (when relevant).",
        ],
        "coins": coin_blocks,
        "sources": {"outlets": ["Google News RSS"]},
        "errors": errors,
        "note": f"Crypto briefing is headline-based. Headlines limited to {max_age_days} days. Per-headline summary is lazy via /crypto/article/summary.",
    }
    return cache_set(key, out, ttl_seconds=300, stale_ttl_seconds=3 * 3600)
