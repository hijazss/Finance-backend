import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus, urlparse, parse_qs
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

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

SESSION = requests.Session()

# =========================================================
# Summary guardrails (prevents Render timeouts)
# =========================================================
# Max publisher-page fetches per request (everything else uses RSS snippets)
MAX_PUBLISHER_FETCHES_PER_REQUEST = int(os.getenv("MAX_PUBLISHER_FETCHES_PER_REQUEST", "40"))
# Max seconds allowed for the whole bullets phase
BULLETS_TOTAL_BUDGET_SECONDS = float(os.getenv("BULLETS_TOTAL_BUDGET_SECONDS", "18"))
# Per-article request timeout
ARTICLE_FETCH_TIMEOUT_SECONDS = int(os.getenv("ARTICLE_FETCH_TIMEOUT_SECONDS", "6"))
# Parallelism for article fetching
ARTICLE_FETCH_WORKERS = int(os.getenv("ARTICLE_FETCH_WORKERS", "10"))


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
def _requests_get(url: str, params: Optional[dict] = None, timeout: int = 16, headers: Optional[dict] = None, allow_redirects: bool = True) -> requests.Response:
    return SESSION.get(url, params=params, timeout=timeout, headers=headers or UA_HEADERS, allow_redirects=allow_redirects)


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
# Daily market read
# =========================================================
@app.get("/market/read")
def market_read():
    key = "market:read:v4121"
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

    def _fmt_pct(x: Optional[float]) -> str:
        if x is None:
            return "—"
        s = "+" if x > 0 else ""
        return f"{s}{x:.2f}%"

    parts: List[str] = []
    parts.append(f"SPY {spy_last:.2f} (1D {_fmt_pct(spy_1d)}, 5D {_fmt_pct(spy_5d)}, 1M {_fmt_pct(spy_1m)})." if spy_last is not None else "SPY unavailable.")
    parts.append(f"QQQ {qqq_last:.2f} (1D {_fmt_pct(qqq_1d)}, 5D {_fmt_pct(qqq_5d)}, 1M {_fmt_pct(qqq_1m)})." if qqq_last is not None else "QQQ unavailable.")
    if vix_last is not None:
        parts.append(f"VIX {vix_last:.2f}.")

    out = {
        "date": now.date().isoformat(),
        "summary": " ".join(parts).strip(),
        "sp500": {"symbol": "SPY", "last": spy_last, "ret1dPct": spy_1d, "ret5dPct": spy_5d, "ret1mPct": spy_1m},
        "nasdaq": {"symbol": "QQQ", "last": qqq_last, "ret1dPct": qqq_1d, "ret5dPct": qqq_5d, "ret1mPct": qqq_1m},
        "vix": {"symbol": "^VIX", "last": vix_last},
        "fearGreed": {"score": None, "rating": None},
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
    r'<meta[^>]+(?:name="description"|property="og:description")[^>]+content="([^"]+)"',
    re.IGNORECASE
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


# =========================================================
# Article summary extraction (best effort, guarded)
# =========================================================
def _maybe_decode_google_news_redirect(url: str) -> str:
    try:
        u = urlparse(url)
        q = parse_qs(u.query or "")
        if "url" in q and q["url"]:
            return q["url"][0]
    except Exception:
        pass
    return url


def _extract_meta_description(html: str) -> str:
    if not html:
        return ""
    m = _META_DESC_RE.search(html)
    if not m:
        return ""
    return _strip_html(m.group(1))


def _extract_first_sentences_from_paragraphs(html: str, max_sentences: int = 3) -> str:
    if not html:
        return ""
    paras = _P_RE.findall(html)
    text_bits: List[str] = []
    for p in paras[:8]:
        clean = _strip_html(p)
        if len(clean) < 60:
            continue
        text_bits.append(clean)
        if len(" ".join(text_bits)) > 900:
            break
    blob = " ".join(text_bits).strip()
    if not blob:
        return ""
    sents = re.split(r"(?<=[\.\!\?])\s+", blob)
    sents = [s.strip() for s in sents if s.strip()]
    return " ".join(sents[:max_sentences]).strip()


def _fetch_article_snippet(url: str) -> str:
    url0 = (url or "").strip()
    if not url0:
        return ""
    url0 = _maybe_decode_google_news_redirect(url0)

    key = f"art:snip:{url0[:280]}"
    cached = cache_get(key, allow_stale=True)
    if cached is not None:
        return str(cached or "")

    try:
        r = _requests_get(url0, timeout=ARTICLE_FETCH_TIMEOUT_SECONDS, headers=UA_HEADERS, allow_redirects=True)
        if r.status_code >= 400:
            return cache_set(key, "", ttl_seconds=180, stale_ttl_seconds=3600)
        html = (r.text or "")[:250000]

        desc = _extract_meta_description(html)
        if desc and len(desc) >= 60:
            return cache_set(key, desc, ttl_seconds=900, stale_ttl_seconds=6 * 3600)

        p = _extract_first_sentences_from_paragraphs(html, max_sentences=3)
        if p and len(p) >= 80:
            return cache_set(key, p, ttl_seconds=900, stale_ttl_seconds=6 * 3600)

        return cache_set(key, "", ttl_seconds=180, stale_ttl_seconds=3600)
    except Exception:
        return cache_set(key, "", ttl_seconds=180, stale_ttl_seconds=3600)


def _title_is_repeating(title: str, text: str) -> bool:
    t = (title or "").strip().lower()
    s = (text or "").strip().lower()
    if not t or not s:
        return False
    if s.startswith(t):
        return True
    t_words = {w for w in re.findall(r"[a-zA-Z]{3,}", t)}
    s_words = {w for w in re.findall(r"[a-zA-Z]{3,}", s)}
    if not t_words:
        return False
    overlap = len(t_words & s_words) / max(1, len(t_words))
    return overlap >= 0.80


def _to_bullets(text: str, max_bullets: int = 3) -> List[str]:
    s = _strip_html(text or "")
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return []

    sents = re.split(r"(?<=[\.\!\?])\s+", s)
    sents = [x.strip() for x in sents if x.strip()]

    bullets: List[str] = []
    for sent in sents:
        if len(sent) < 35:
            continue
        if len(sent) > 180:
            sent = sent[:180].rsplit(" ", 1)[0].strip() + "…"
        bullets.append(sent)
        if len(bullets) >= max_bullets:
            break

    if len(bullets) < 2:
        chunks = re.split(r"\s*;\s*|\s+but\s+|\s+while\s+", s)
        chunks = [c.strip() for c in chunks if len(c.strip()) >= 40]
        for c in chunks[:max_bullets]:
            if len(bullets) >= max_bullets:
                break
            if len(c) > 180:
                c = c[:180].rsplit(" ", 1)[0].strip() + "…"
            if c not in bullets:
                bullets.append(c)

    return bullets[:max_bullets]


def _headline_bullets_from_rss(title: str, raw_summary: str) -> List[str]:
    rs = _strip_html(raw_summary or "")
    if rs and not _title_is_repeating(title, rs):
        b = _to_bullets(rs, max_bullets=3)
        if b:
            return b
    return []


def _headline_bullets_best_effort(title: str, raw_summary: str, link: str, allow_publisher_fetch: bool) -> List[str]:
    if allow_publisher_fetch:
        snip = _fetch_article_snippet(link)
        if snip and not _title_is_repeating(title, snip):
            b = _to_bullets(snip, max_bullets=3)
            if b:
                return b
    return _headline_bullets_from_rss(title, raw_summary)


# =========================================================
# Lightweight sentiment and summaries
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


def _sector_ai_summary(sector: str, titles: List[str]) -> Tuple[str, List[str]]:
    if not titles:
        return (f"{sector}: No fresh items in the last 30 days.", [])
    terms = _top_terms(titles, k=10)[:6]
    tickers = _extract_tickers_from_titles(titles)[:10]
    parts: List[str] = []
    if terms:
        parts.append(f"Main themes: {', '.join(terms)}.")
    if tickers:
        parts.append(f"Tickers referenced: {', '.join(tickers[:8])}.")
    parts.append("Use this as context and confirm via the linked sources.")
    return " ".join(parts).strip(), []


# =========================================================
# News briefing (FIXED: bounded + parallel bullet extraction)
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

    # ---------- bounded + parallel bullets ----------
    bullets_deadline = time.time() + BULLETS_TOTAL_BUDGET_SECONDS
    publisher_budget = MAX_PUBLISHER_FETCHES_PER_REQUEST

    def build_headlines_payload(items: List[dict], sec: str) -> List[dict]:
        nonlocal publisher_budget

        # Prepare jobs: only for first N items overall that still have budget and time
        payload: List[dict] = []
        tasks: Dict[int, dict] = {}

        for idx, x in enumerate(items):
            title = x.get("title", "") or ""
            link = x.get("link", "") or ""
            pub = x.get("published", "") or ""
            raw = x.get("rawSummary", "") or ""

            allow_fetch = False
            if publisher_budget > 0 and time.time() < bullets_deadline and link:
                allow_fetch = True
                publisher_budget -= 1

            payload.append({
                "title": title,
                "link": link,
                "published": pub,
                "sourceFeed": x.get("source", "Google News"),
                "sector": sec,
                "_rawSummary": raw,
                "_allowFetch": allow_fetch,
                "summary": "",
                "summaryBullets": [],
            })

            if allow_fetch:
                tasks[idx] = payload[idx]

        # Run publisher fetches in parallel, but stop honoring results if we hit deadline
        if tasks:
            with ThreadPoolExecutor(max_workers=ARTICLE_FETCH_WORKERS) as ex:
                future_map = {}
                for idx, row in tasks.items():
                    future_map[ex.submit(_headline_bullets_best_effort, row["title"], row["_rawSummary"], row["link"], True)] = idx

                for fut in as_completed(future_map):
                    if time.time() >= bullets_deadline:
                        break
                    idx = future_map[fut]
                    try:
                        b = fut.result(timeout=0)
                        payload[idx]["summaryBullets"] = b or []
                        payload[idx]["summary"] = (b[0] if b else "")
                    except Exception:
                        # leave empty for now, will fall back to RSS below
                        pass

        # Fill any missing bullets from RSS immediately
        for row in payload:
            if not row["summaryBullets"]:
                b = _headline_bullets_from_rss(row["title"], row["_rawSummary"])
                row["summaryBullets"] = b or []
                row["summary"] = (b[0] if b else "")

            # cleanup internal keys
            row.pop("_rawSummary", None)
            row.pop("_allowFetch", None)

        return payload

    sectors_out: List[dict] = []

    # Watchlist sector first
    if watch:
        wl_items = [x for x in all_items if x.get("sector") == "Watchlist"][:max_general]
        wl_items = _dedup_items(wl_items, max_general)

        wl_titles = [x.get("title", "") for x in wl_items]
        wl_score, wl_label = _sentiment_from_titles(wl_titles)

        sectors_out.append({
            "sector": "Watchlist",
            "sentiment": {"label": wl_label, "score": wl_score},
            "summary": "Per-headline takeaways shown under each item.",
            "aiSummary": "Per-headline takeaways shown under each item.",
            "implications": [],
            "topHeadlines": build_headlines_payload(wl_items, "Watchlist"),
            "watchlistMentions": watch[:25],
            "tickersMentioned": _extract_tickers_from_titles(wl_titles)[:12],
        })

    # Each sector
    for sec in sector_list:
        sec_items = [x for x in all_items if x.get("sector") == sec][:max_items_per_sector]
        sec_titles = [x.get("title", "") for x in sec_items]

        score, label = _sentiment_from_titles(sec_titles)
        ai_sum, _unused = _sector_ai_summary(sec, sec_titles)

        sectors_out.append({
            "sector": sec,
            "sentiment": {"label": label, "score": score},
            "summary": ai_sum,
            "aiSummary": ai_sum,
            "implications": [],
            "topHeadlines": build_headlines_payload(sec_items, sec),
            "watchlistMentions": [],
            "tickersMentioned": _extract_tickers_from_titles(sec_titles)[:12],
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
        "note": f"Google News RSS. Headlines limited to last {max_age_days} days. Bullets: bounded + parallel publisher fetch, RSS fallback.",
    }
    return cache_set(key, out, ttl_seconds=300, stale_ttl_seconds=3 * 3600)


# =========================================================
# Crypto briefing endpoint (also bounded bullets)
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
            ai_sum, _unused = _sector_ai_summary(c, titles)

            all_titles.extend(titles)

            heads = []
            for x in items:
                title = x.get("title", "") or ""
                link = x.get("link", "") or ""
                pub = x.get("published", "") or ""
                raw = x.get("rawSummary", "") or ""

                # Crypto uses RSS-first (fast) to avoid timeouts
                bullets = _headline_bullets_from_rss(title, raw)

                heads.append({
                    "title": title,
                    "link": link,
                    "published": pub,
                    "source": "Google News",
                    "summary": bullets[0] if bullets else "",
                    "summaryBullets": bullets,
                })

            coins_out.append({
                "symbol": c,
                "sentiment": {"label": label, "score": score},
                "summary": ai_sum,
                "aiSummary": ai_sum,
                "implications": [],
                "headlines": heads
            })
        except Exception as e:
            errors.append(f"{c}: {type(e).__name__}: {str(e)}")
            coins_out.append({
                "symbol": c,
                "sentiment": {"label": "NEUTRAL", "score": 50},
                "summary": f"{c}: No fresh items.",
                "aiSummary": f"{c}: No fresh items.",
                "implications": [],
                "headlines": []
            })

    overall_score, overall_label = _sentiment_from_titles(all_titles)

    out = {
        "date": now.date().isoformat(),
        "overallSentiment": {"label": overall_label, "score": overall_score},
        "coins": coins_out,
        "errors": errors,
        "note": f"Google News RSS. Headlines limited to last {max_age_days} days. Bullets: RSS-first for speed.",
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
