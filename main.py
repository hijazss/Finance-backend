import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus, urlparse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import requests
import quiverquant
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware


# =========================================================
# App
# =========================================================
app = FastAPI(title="Finance Signals Backend", version="4.11.1")

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
    return {"status": "ok", "version": "4.11.1"}


@app.get("/health")
def health():
    return {
        "ok": True,
        "version": "4.11.1",
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
def _requests_get(
    url: str,
    params: Optional[dict] = None,
    timeout: int = 16,
    headers: Optional[dict] = None,
    allow_redirects: bool = True,
) -> requests.Response:
    return SESSION.get(
        url,
        params=params,
        timeout=timeout,
        headers=headers or UA_HEADERS,
        allow_redirects=allow_redirects,
    )


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
# Daily market read (matches your frontend expectations)
# =========================================================
@app.get("/market/read")
def market_read():
    key = "market:read:v4111"
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

    def _fallback_last(symbol: str) -> Optional[float]:
        try:
            last, _prev = _nasdaq_last_and_prev(symbol)
            return float(last) if last else None
        except Exception as e:
            errors.append(f"Nasdaq {symbol}: {type(e).__name__}: {str(e)}")
            return None

    spy_last = float(spy_vals[-1]) if spy_vals else _fallback_last("SPY")
    qqq_last = float(qqq_vals[-1]) if qqq_vals else _fallback_last("QQQ")
    vix_last = float(vix_vals[-1]) if vix_vals else _fallback_last("VIX")

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
# RSS helpers
# =========================================================
def _google_news_rss(query: str) -> str:
    q = quote_plus(query)
    return f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_META_DESC_RE = re.compile(
    r'<meta[^>]+(?:name="description"|property="og:description"|name="twitter:description")[^>]+content="([^"]+)"',
    re.IGNORECASE,
)
_P_RE = re.compile(r"<p[^>]*>(.*?)</p>", re.IGNORECASE | re.DOTALL)


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

            raw_summary = _strip_html(summ)
            out.append({"title": title, "link": link, "published": pub, "rawSummary": raw_summary})
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

        raw_summary = _strip_html(desc)
        out.append({"title": title, "link": link, "published": pub, "rawSummary": raw_summary})
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
    if len(s) > 320:
        s = s[:320].rsplit(" ", 1)[0].strip() + "…"
    return s


def _is_probably_google_news_redirect(url: str) -> bool:
    try:
        host = urlparse(url).netloc.lower()
        return "news.google.com" in host or "google.com" in host
    except Exception:
        return False


def _extract_meta_description(html: str) -> str:
    if not html:
        return ""
    m = _META_DESC_RE.search(html)
    if not m:
        return ""
    return _strip_html(m.group(1) or "")


def _extract_first_paragraph(html: str) -> str:
    if not html:
        return ""
    # Pick the first non-trivial paragraph
    for m in _P_RE.finditer(html):
        txt = _strip_html(m.group(1) or "")
        if len(txt) >= 80:
            return txt
    return ""


def _fetch_article_snippet(url: str, timeout: int = 10) -> str:
    """
    Best-effort "key takeaways" from the article page:
      - meta description / og:description
      - otherwise first substantial paragraph
    Cached to reduce load.
    """
    if not url:
        return ""

    ck = f"artsum:{_normalize_link_key(url)}"
    cached = cache_get(ck, allow_stale=True)
    if cached is not None:
        return str(cached or "")

    try:
        r = _requests_get(url, timeout=timeout, headers=UA_HEADERS, allow_redirects=True)
        if r.status_code >= 400:
            return cache_set(ck, "", ttl_seconds=900, stale_ttl_seconds=6 * 3600)
        html = (r.text or "")[:350000]  # hard cap
        desc = _extract_meta_description(html)
        if not desc:
            desc = _extract_first_paragraph(html)
        desc = re.sub(r"\s+", " ", (desc or "")).strip()
        if len(desc) > 360:
            desc = desc[:360].rsplit(" ", 1)[0].strip() + "…"
        return cache_set(ck, desc, ttl_seconds=6 * 3600, stale_ttl_seconds=24 * 3600)
    except Exception:
        return cache_set(ck, "", ttl_seconds=900, stale_ttl_seconds=6 * 3600)


def _best_headline_summary(title: str, raw_summary: str, link: str, allow_fetch: bool) -> str:
    # Prefer RSS rawSummary if it is genuinely additive.
    s1 = _headline_summary_safe(title, raw_summary)
    if s1:
        return s1
    if not allow_fetch:
        return ""
    # If RSS summary is empty or just repeats title, try article metadata.
    s2 = _fetch_article_snippet(link, timeout=10)
    s2 = _headline_summary_safe(title, s2)
    return s2


# =========================================================
# AI-style summarization from titles (richer Watchlist)
# =========================================================
_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "as", "at", "by", "from", "into",
    "over", "after", "before", "than", "is", "are", "was", "were", "be", "been", "being", "it", "its",
    "this", "that", "these", "those", "you", "your", "they", "their", "we", "our", "us", "will", "may",
    "new", "today", "latest", "live", "update", "reports", "report", "says", "say", "said"
}

_TICKER_RE = re.compile(r"\b[A-Z]{1,5}\b")


def _extract_tickers_from_titles(titles: List[str]) -> List[str]:
    bad = {"A", "I", "AN", "THE", "AND", "OR", "TO", "OF", "IN", "ON", "US", "AI"}
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


def _count_tickers(titles: List[str], watch: List[str]) -> List[Tuple[str, int]]:
    wl = [w.strip().upper() for w in watch if w.strip()]
    wl_set = set(wl)
    counts: Dict[str, int] = {w: 0 for w in wl}
    for t in titles:
        hits = set()
        for x in _TICKER_RE.findall((t or "").upper()):
            if x in wl_set:
                hits.add(x)
        for h in hits:
            counts[h] = counts.get(h, 0) + 1
    ranked = [(k, v) for k, v in counts.items() if v > 0]
    ranked.sort(key=lambda x: (-x[1], x[0]))
    return ranked[:10]


_CATALYST_BUCKETS = [
    ("earnings/guidance", ["earnings", "guidance", "revenue", "profit", "margin", "forecast", "outlook", "eps", "beats", "miss"]),
    ("deals/partnerships", ["acquisition", "acquire", "merger", "deal", "partnership", "joint", "buyout", "stake"]),
    ("regulation/legal", ["sec", "doj", "ftc", "probe", "lawsuit", "settlement", "ban", "court", "regulator", "antitrust"]),
    ("fda/clinical", ["fda", "trial", "phase", "clinical", "approval", "label", "safety", "recall", "medicare", "cms"]),
    ("ai/product", ["ai", "model", "chip", "gpu", "product", "launch", "upgrade", "roadmap"]),
    ("security/outage", ["hack", "breach", "cyber", "outage", "leak", "ransom", "vulnerability"]),
    ("macro/rates", ["fed", "rates", "inflation", "jobs", "cpi", "yield", "treasury", "recession"]),
]


def _bucket_counts(titles: List[str]) -> List[Tuple[str, int]]:
    counts: Dict[str, int] = {name: 0 for name, _ in _CATALYST_BUCKETS}
    for t in titles:
        s = (t or "").lower()
        for name, keys in _CATALYST_BUCKETS:
            if any(k in s for k in keys):
                counts[name] += 1
    ranked = [(k, v) for k, v in counts.items() if v > 0]
    ranked.sort(key=lambda x: (-x[1], x[0]))
    return ranked[:6]


def _freshness_counts(items: List[dict]) -> Dict[str, int]:
    now = datetime.now(timezone.utc)
    c7 = 0
    c30 = 0
    for x in items:
        dt = _parse_rss_date(str(x.get("published") or ""))
        if dt is None:
            c30 += 1
            continue
        if dt >= now - timedelta(days=7):
            c7 += 1
        if dt >= now - timedelta(days=30):
            c30 += 1
    return {"last7d": c7, "last30d": c30}


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


def _sector_ai_summary(sector: str, titles: List[str]) -> Tuple[str, List[str]]:
    if not titles:
        return (f"{sector}: No fresh items in the last 30 days.", ["No key takeaways available."])

    tickers = _extract_tickers_from_titles(titles)[:12]
    terms = _top_terms(titles, k=10)[:6]
    buckets = _bucket_counts(titles)

    parts: List[str] = []
    if terms:
        parts.append(f"Main themes: {', '.join(terms)}.")
    if tickers:
        parts.append(f"Tickers referenced: {', '.join(tickers[:8])}.")
    if buckets:
        b_str = ", ".join([f"{b} ({n})" for b, n in buckets[:4]])
        parts.append(f"Dominant catalysts: {b_str}.")
    parts.append("Net: use this as context, then validate the repeated catalyst in the linked articles before sizing up.")

    bullets: List[str] = []
    if terms:
        bullets.append(f"Theme cluster: {', '.join(terms[:4])}.")
    if buckets and any(b == "earnings/guidance" for b, _ in buckets):
        bullets.append("Earnings and guidance are a primary driver. Watch revisions and forward commentary.")
    if buckets and any(b == "regulation/legal" for b, _ in buckets):
        bullets.append("Legal and regulatory headlines can be noisy. Track follow-on updates before reacting.")
    bullets.append("Triage: prioritize the newest items plus repeated names and repeated catalyst buckets.")
    return " ".join(parts).strip(), bullets[:6]


def _watchlist_ai_summary(watch_items: List[dict], watchlist: List[str]) -> Tuple[str, List[str]]:
    titles = [str(x.get("title") or "") for x in watch_items if str(x.get("title") or "").strip()]
    if not titles:
        return ("Watchlist: No fresh items in the last 30 days.", ["No key takeaways available."])

    fresh = _freshness_counts(watch_items)
    top_tickers = _count_tickers(titles, watchlist)
    buckets = _bucket_counts(titles)
    terms = _top_terms(titles, k=12)
    score, label = _sentiment_from_titles(titles)

    lines: List[str] = []
    lines.append(f"Watchlist scan: {len(titles)} unique headlines (last 7d: {fresh['last7d']}, last 30d: {fresh['last30d']}).")
    lines.append(f"Headline tone gauge: {label} ({score}/100), based on headline language (not price action).")
    if top_tickers:
        top_str = ", ".join([f"{t} ({n})" for t, n in top_tickers[:6]])
        lines.append(f"Most mentioned watchlist names: {top_str}.")
    if buckets:
        b_str = ", ".join([f"{b} ({n})" for b, n in buckets[:5]])
        lines.append(f"Dominant catalyst buckets: {b_str}.")
    if terms:
        lines.append(f"Recurring themes: {', '.join(terms[:7])}.")
    lines.append("Use: this is a triage summary. Open the top repeated names and confirm the catalyst details before acting.")

    narrative = " ".join(lines).strip()

    bullets: List[str] = []
    if top_tickers:
        bullets.append("Triage first: " + ", ".join([f"{t} ({n} headlines)" for t, n in top_tickers[:8]]) + ".")
    if buckets:
        bullets.append("Catalyst mix: " + ", ".join([f"{b} ({n})" for b, n in buckets[:6]]) + ".")
    if fresh["last7d"] >= max(3, int(0.35 * max(1, fresh["last30d"]))):
        bullets.append("Freshness: many headlines are from the last week, so catalysts may still be active.")
    else:
        bullets.append("Freshness: headlines are spread across the month. Prioritize the most recent plus repeats.")
    if any(b == "fda/clinical" for b, _ in buckets):
        bullets.append("Clinical and regulatory items can drive outsized volatility. Confirm stage, timeline, and scope.")
    if any(b == "earnings/guidance" for b, _ in buckets):
        bullets.append("Earnings and guidance: watch revisions, margins, and forward commentary, not the print alone.")
    if any(b == "regulation/legal" for b, _ in buckets):
        bullets.append("Regulatory and legal: separate headline risk from operational impact. Look for follow-on actions.")
    bullets.append("Risk control: if items are mostly one-off, consider smaller adds or wait for confirmation.")
    return narrative, bullets[:10]


def _enrich_headline_summaries(
    items: List[dict],
    max_fetch: int,
) -> None:
    """
    Mutates items in place, adding "summary" for each.
    Uses RSS rawSummary first; fetches article snippets for a limited subset.
    """
    if not items:
        return

    # Decide which items are worth fetching: those with empty or title-like rawSummary.
    candidates = []
    for idx, x in enumerate(items):
        title = str(x.get("title") or "")
        raw = str(x.get("rawSummary") or "")
        link = str(x.get("link") or "")
        safe = _headline_summary_safe(title, raw)
        if safe:
            x["summary"] = safe
            continue
        x["summary"] = ""
        if link:
            candidates.append(idx)

    if not candidates or max_fetch <= 0:
        return

    # Limit fetch count
    candidates = candidates[: max_fetch]

    def _job(i: int) -> Tuple[int, str]:
        x = items[i]
        title = str(x.get("title") or "")
        raw = str(x.get("rawSummary") or "")
        link = str(x.get("link") or "")
        # allow_fetch True
        s = _best_headline_summary(title, raw, link, allow_fetch=True)
        return i, s

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(_job, i) for i in candidates]
        for fut in futs:
            try:
                i, s = fut.result()
                if s:
                    items[i]["summary"] = s
            except Exception:
                continue


# =========================================================
# News briefing
# =========================================================
@app.get("/news/briefing")
def news_briefing(
    tickers: str = Query(default=""),
    max_general: int = Query(default=60, ge=10, le=200),
    max_per_ticker: int = Query(default=6, ge=1, le=30),
    sectors: str = Query(default="AI,Medical,Energy,Robotics,Infrastructure,Semiconductors,Cloud,Cybersecurity,Defense,Financials,Consumer"),
    max_items_per_sector: int = Query(default=12, ge=5, le=30),
    max_age_days: int = Query(default=30, ge=7, le=45),
    fetch_article_summaries: bool = Query(default=True),
):
    """
    Returns sectors + watchlist blocks.
    Each headline has a best-effort "summary":
      - RSS description if it is additive
      - else meta description / first paragraph (limited fetches)
    """
    key = f"news:brief:v4111:{tickers}:{max_general}:{max_per_ticker}:{sectors}:{max_items_per_sector}:{max_age_days}:{int(fetch_article_summaries)}"
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
        futs = [
            (sec, ex.submit(_fetch_rss_items, url, 10, max_items_per_sector * 3, 240, 6 * 3600, max_age_days))
            for sec, url in jobs
        ]
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
            futs2 = [
                (t, ex.submit(_fetch_rss_items, url, 10, max_per_ticker * 2, 240, 6 * 3600, max_age_days))
                for t, url in ticker_jobs
            ]
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

    # Big dedup across everything
    all_items = _dedup_items(all_items, 2000)

    sectors_out: List[dict] = []

    # Watchlist sector first
    if watch:
        wl_items = [x for x in all_items if x.get("sector") == "Watchlist"][:max_general]
        wl_items = _dedup_items(wl_items, max_general)

        # Add per-headline summaries (fetch limited subset)
        if fetch_article_summaries:
            _enrich_headline_summaries(wl_items, max_fetch=min(24, len(wl_items)))
        else:
            _enrich_headline_summaries(wl_items, max_fetch=0)

        wl_titles = [x.get("title", "") for x in wl_items]
        wl_score, wl_label = _sentiment_from_titles(wl_titles)
        wl_ai, wl_bullets = _watchlist_ai_summary(wl_items, watch)

        sectors_out.append({
            "sector": "Watchlist",
            "sentiment": {"label": wl_label, "score": wl_score},
            "aiSummary": wl_ai,
            "summary": wl_ai,
            "implications": wl_bullets,
            "topHeadlines": [
                {
                    "title": x.get("title", ""),
                    "link": x.get("link", ""),
                    "published": x.get("published", ""),
                    "sourceFeed": x.get("source", "Google News"),
                    "sector": "Watchlist",
                    "summary": x.get("summary", "") or "",
                }
                for x in wl_items
            ],
            "watchlistMentions": watch[:25],
            "tickersMentioned": _extract_tickers_from_titles(wl_titles)[:12],
        })

    # Each sector
    for sec in sector_list:
        sec_items = [x for x in all_items if x.get("sector") == sec][:max_items_per_sector]
        sec_items = _dedup_items(sec_items, max_items_per_sector)

        # Add per-headline summaries (fetch limited subset)
        if fetch_article_summaries:
            _enrich_headline_summaries(sec_items, max_fetch=min(10, len(sec_items)))
        else:
            _enrich_headline_summaries(sec_items, max_fetch=0)

        sec_titles = [x.get("title", "") for x in sec_items]
        score, label = _sentiment_from_titles(sec_titles)
        ai_sum, bullets = _sector_ai_summary(sec, sec_titles)

        sectors_out.append({
            "sector": sec,
            "sentiment": {"label": label, "score": score},
            "aiSummary": ai_sum,
            "summary": ai_sum,
            "implications": bullets,
            "topHeadlines": [
                {
                    "title": x.get("title", ""),
                    "link": x.get("link", ""),
                    "published": x.get("published", ""),
                    "sourceFeed": x.get("source", "Google News"),
                    "sector": sec,
                    "summary": x.get("summary", "") or "",
                }
                for x in sec_items
            ],
            "watchlistMentions": [],
            "tickersMentioned": _extract_tickers_from_titles(sec_titles)[:12],
        })

    all_titles: List[str] = []
    for s in sectors_out:
        all_titles.extend([h.get("title", "") for h in (s.get("topHeadlines") or [])])

    overall_score, overall_label = _sentiment_from_titles(all_titles)

    out = {
        "date": datetime.now(timezone.utc).date().isoformat(),
        "overallSentiment": {"label": overall_label, "score": overall_score},
        "sectors": sectors_out,
        "errors": errors,
        "note": f"Google News RSS. Headlines limited to last {max_age_days} days. Headline summaries use RSS text when available, otherwise article metadata (limited fetch).",
    }
    return cache_set(key, out, ttl_seconds=300, stale_ttl_seconds=3 * 3600)


# =========================================================
# Crypto briefing endpoint
# =========================================================
@app.get("/crypto/news/briefing")
def crypto_news_briefing(
    coins: str = Query(default="BTC,ETH,LINK,SHIB"),
    include_top_n: int = Query(default=15, ge=5, le=50),
    max_age_days: int = Query(default=30, ge=7, le=45),
    fetch_article_summaries: bool = Query(default=True),
):
    key = f"crypto:news:v4111:{coins}:{include_top_n}:{max_age_days}:{int(fetch_article_summaries)}"
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

            if fetch_article_summaries:
                _enrich_headline_summaries(items, max_fetch=min(10, len(items)))
            else:
                _enrich_headline_summaries(items, max_fetch=0)

            titles = [x.get("title", "") for x in items]
            score, label = _sentiment_from_titles(titles)
            ai_sum, bullets = _sector_ai_summary(c, titles)

            all_titles.extend(titles)

            coins_out.append({
                "symbol": c,
                "sentiment": {"label": label, "score": score},
                "aiSummary": ai_sum,
                "summary": ai_sum,
                "implications": bullets,
                "headlines": [
                    {
                        "title": x.get("title", ""),
                        "link": x.get("link", ""),
                        "published": x.get("published", ""),
                        "source": "Google News",
                        "summary": x.get("summary", "") or "",
                    }
                    for x in items
                ],
            })
        except Exception as e:
            errors.append(f"{c}: {type(e).__name__}: {str(e)}")
            coins_out.append({
                "symbol": c,
                "sentiment": {"label": "NEUTRAL", "score": 50},
                "aiSummary": f"{c}: No fresh items.",
                "summary": f"{c}: No fresh items.",
                "implications": ["No key takeaways available."],
                "headlines": [],
            })

    overall_score, overall_label = _sentiment_from_titles(all_titles)

    out = {
        "date": now.date().isoformat(),
        "overallSentiment": {"label": overall_label, "score": overall_score},
        "coins": coins_out,
        "errors": errors,
        "note": f"Google News RSS. Headlines limited to last {max_age_days} days. Headline summaries use RSS text when available, otherwise article metadata (limited fetch).",
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

    key = f"holdings:common:v4111:{window_days}:{top_n}"
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

    key = f"congress:daily:v4111:{window_days}:{limit}"
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
