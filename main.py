import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

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
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "").strip()  # kept for compatibility

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

HTML_HEADERS = {
    "User-Agent": UA_HEADERS["User-Agent"],
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": UA_HEADERS["Accept-Language"],
    "Connection": "keep-alive",
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


def _sma(vals: List[float], n: int) -> Optional[float]:
    if n <= 0 or len(vals) < n:
        return None
    return sum(vals[-n:]) / n


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _clean_text(s: str) -> str:
    if not s:
        return ""
    s2 = re.sub(r"<[^>]+>", " ", s)  # strip HTML tags
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2


# =========================================================
# Nasdaq (quotes only)
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
# Stooq (history, best-effort)
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


def _stooq_last_prev_and_returns(symbol: str) -> Dict[str, Optional[float]]:
    closes = _stooq_daily_closes(symbol)
    vals = [c for _, c in closes if c is not None]
    if len(vals) < 2:
        return {"last": None, "prev": None, "ret1dPct": None, "ret5dPct": None, "ret1mPct": None}

    last = float(vals[-1])
    prev = float(vals[-2])

    ret1d = _pct(last, prev) if prev else None

    ret5d = None
    if len(vals) >= 6 and vals[-6]:
        ret5d = _pct(last, float(vals[-6]))

    ret1m = None
    if len(vals) >= 22 and vals[-22]:
        ret1m = _pct(last, float(vals[-22]))

    return {"last": last, "prev": prev, "ret1dPct": ret1d, "ret5dPct": ret5d, "ret1mPct": ret1m}


# =========================================================
# CNN Fear & Greed (best-effort, weekend fallback)
# =========================================================
def _cnn_fear_greed_graphdata(date_str: Optional[str] = None) -> dict:
    d0 = date_str or datetime.now(timezone.utc).date().isoformat()
    start_date = datetime.fromisoformat(d0).date()
    last_err: Optional[str] = None

    for back in range(0, 11):
        d = (start_date - timedelta(days=back)).isoformat()
        url = f"https://production.dataviz.cnn.io/index/fearandgreed/graphdata/{d}"
        try:
            r = _requests_get(url, timeout=16, headers=UA_HEADERS)
            if r.status_code == 404:
                last_err = "HTTP 404"
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:160]}"
            continue

    raise RuntimeError(f"CNN fear/greed failed for {d0} (and fallback): {last_err or 'unknown'}")


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
# Market Snapshot
# =========================================================
@app.get("/market/snapshot")
def market_snapshot():
    key = "market:snapshot:v490"
    cached = cache_get(key, allow_stale=True)
    if cached is not None:
        return cached

    now = datetime.now(timezone.utc)
    errors: List[str] = []

    spy_hist = {"last": None, "prev": None, "ret1dPct": None, "ret5dPct": None, "ret1mPct": None}
    try:
        spy_hist = _stooq_last_prev_and_returns("spy.us")
    except Exception as e:
        errors.append(f"Stooq SPY: {type(e).__name__}: {str(e)}")

    vix_hist_last = vix_hist_prev = None
    try:
        vix_hist = _stooq_last_prev_and_returns("vix")
        vix_hist_last = vix_hist.get("last")
        vix_hist_prev = vix_hist.get("prev")
    except Exception as e:
        errors.append(f"Stooq VIX: {type(e).__name__}: {str(e)}")

    spy_last = spy_prev = None
    try:
        spy_last, spy_prev = _nasdaq_last_and_prev("SPY")
    except Exception as e:
        errors.append(f"Nasdaq SPY: {type(e).__name__}: {str(e)}")

    vix_last = vix_prev = None
    try:
        vix_last, vix_prev = _nasdaq_last_and_prev("VIX")
    except Exception as e:
        errors.append(f"Nasdaq VIX: {type(e).__name__}: {str(e)}")

    spy_last_final = spy_hist.get("last") if spy_hist.get("last") is not None else spy_last
    spy_prev_final = spy_hist.get("prev") if spy_hist.get("prev") is not None else spy_prev

    ret1d = spy_hist.get("ret1dPct")
    if ret1d is None and spy_last_final is not None and spy_prev_final is not None and spy_prev_final:
        ret1d = _pct(float(spy_last_final), float(spy_prev_final))

    ret5d = spy_hist.get("ret5dPct")
    ret1m = spy_hist.get("ret1mPct")

    vix_last_final = vix_last if vix_last is not None else vix_hist_last
    vix_prev_final = vix_prev if vix_prev is not None else vix_hist_prev

    fg = {"score": None, "rating": None}
    try:
        fgd = market_fear_greed(None)
        fg = {"score": fgd.get("score"), "rating": fgd.get("rating")}
    except Exception as e:
        errors.append(f"FearGreed: {type(e).__name__}: {str(e)}")

    out = {
        "date": now.date().isoformat(),
        "sp500": {
            "symbol": "SPY",
            "last": spy_last_final,
            "ret1dPct": ret1d,
            "ret5dPct": ret5d,
            "ret1mPct": ret1m,
        },
        "vix": {
            "symbol": "^VIX",
            "last": vix_last_final,
            "chg1d": (float(vix_last_final) - float(vix_prev_final)) if (vix_last_final is not None and vix_prev_final is not None) else None,
        },
        "fearGreed": fg,
        "errors": errors,
        "note": "Snapshot uses Stooq for 1D/5D/1M returns and CNN for Fear & Greed, with weekend fallbacks. Nasdaq is best-effort for quotes.",
    }

    if out["sp500"]["last"] is not None:
        cache_set("market:snapshot:last_good", out, ttl_seconds=3600, stale_ttl_seconds=24 * 3600)
    else:
        lg = cache_get("market:snapshot:last_good", allow_stale=True)
        if lg:
            merged = dict(lg)
            merged["errors"] = (list(merged.get("errors") or []) + errors)[:12]
            merged["note"] = "Serving last_good snapshot due to provider errors."
            return cache_set(key, merged, ttl_seconds=90, stale_ttl_seconds=1800)

    return cache_set(key, out, ttl_seconds=180, stale_ttl_seconds=1800)


# =========================================================
# Market Entry Index
# =========================================================
@app.get("/market/entry")
def market_entry(window_days: int = Query(default=365, ge=30, le=365)):
    key = f"market:entry:v490:{window_days}"
    cached = cache_get(key, allow_stale=True)
    if cached is not None:
        return cached

    now = datetime.now(timezone.utc)
    errors: List[str] = []

    spy: List[Tuple[datetime, float]] = []
    vix: List[Tuple[datetime, float]] = []
    try:
        spy = _stooq_daily_closes("spy.us")
    except Exception as e:
        errors.append(f"Stooq SPY: {type(e).__name__}: {str(e)}")

    try:
        vix = _stooq_daily_closes("vix")
    except Exception as e:
        errors.append(f"Stooq VIX: {type(e).__name__}: {str(e)}")

    if len(spy) < 40:
        try:
            last, prev = _nasdaq_last_and_prev("SPY")
            if last:
                spy = [(now - timedelta(days=1), float(prev) if prev else float(last)), (now, float(last))]
        except Exception as e:
            errors.append(f"Nasdaq SPY quote fallback: {type(e).__name__}: {str(e)}")

    if len(vix) < 10:
        try:
            last, prev = _nasdaq_last_and_prev("VIX")
            if last:
                vix = [(now - timedelta(days=1), float(prev) if prev else float(last)), (now, float(last))]
        except Exception as e:
            errors.append(f"Nasdaq VIX quote fallback: {type(e).__name__}: {str(e)}")

    if len(spy) < 2 or len(vix) < 2:
        lg = cache_get("market:entry:last_good", allow_stale=True)
        if lg:
            merged = dict(lg)
            merged["errors"] = (list(merged.get("errors") or []) + errors)[:12]
            merged["notes"] = (merged.get("notes") or "") + " | serving last_good"
            return cache_set(key, merged, ttl_seconds=120, stale_ttl_seconds=3600)

        out = {
            "date": now.date().isoformat(),
            "score": 50,
            "regime": "NEUTRAL",
            "signal": "DATA LIMITED",
            "notes": "Insufficient market data. " + (" | ".join(errors) if errors else ""),
            "components": {"spxTrend": 0.5, "vix": 0.5},
            "errors": errors,
        }
        return cache_set(key, out, ttl_seconds=120, stale_ttl_seconds=1800)

    spy_sorted = sorted(spy, key=lambda x: x[0])
    vix_sorted = sorted(vix, key=lambda x: x[0])
    spy_vals = [c for _, c in spy_sorted if c is not None]
    vix_vals = [c for _, c in vix_sorted if c is not None]

    price = float(spy_vals[-1])
    v = float(vix_vals[-1])

    if len(spy_vals) >= 210:
        fast_n, slow_n = 50, 200
    elif len(spy_vals) >= 120:
        fast_n, slow_n = 20, 100
    elif len(spy_vals) >= 70:
        fast_n, slow_n = 20, 60
    else:
        fast_n, slow_n = 10, 30

    sma_fast = _sma(spy_vals, fast_n) or price
    sma_slow = _sma(spy_vals, slow_n) or price

    trend_cross = 1.0 if sma_fast >= sma_slow else 0.0
    price_vs_slow = _clamp01((price / sma_slow - 0.92) / (1.08 - 0.92))
    spx_trend_01 = _clamp01(0.55 * trend_cross + 0.45 * price_vs_slow)

    vix_01 = _clamp01(1.0 - ((v - 12.0) / (35.0 - 12.0)))

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

    notes = f"SPY={price:.2f} SMA{fast_n}={sma_fast:.2f} SMA{slow_n}={sma_slow:.2f} VIX={v:.2f}"
    if errors:
        notes += " | " + " | ".join(errors[:6])

    out = {
        "date": now.date().isoformat(),
        "score": score,
        "regime": regime,
        "signal": signal,
        "notes": notes,
        "components": {"spxTrend": float(spx_trend_01), "vix": float(vix_01)},
        "errors": errors,
    }

    cache_set("market:entry:last_good", out, ttl_seconds=3600, stale_ttl_seconds=24 * 3600)
    return cache_set(key, out, ttl_seconds=300, stale_ttl_seconds=1800)


# =========================================================
# RSS helpers + News briefing with flowing reads + dedupe + actionable implications
# =========================================================
def _google_news_rss(query: str) -> str:
    q = quote_plus(query)
    return f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"


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
            summary = (e.findtext("{http://www.w3.org/2005/Atom}summary") or "").strip()
            if title:
                out.append({"title": title, "link": link, "published": pub, "description": _clean_text(summary)})
        return out

    out = []
    for item in channel.findall("item")[:max_items]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        desc = (item.findtext("description") or "").strip()
        if title:
            out.append({"title": title, "link": link, "published": pub, "description": _clean_text(desc)})
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


def _norm_key(title: str, link: str) -> str:
    t = (title or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^a-z0-9 \-:/]", "", t)[:220]
    lk = (link or "").strip()
    # Google News often wraps URLs and can vary params. Key on title primarily, then link.
    return f"{t}|{lk[:160]}"


def _dedup_items(items: List[dict], max_items: int, global_seen: Optional[set] = None) -> List[dict]:
    seen = global_seen if global_seen is not None else set()
    out = []
    for x in items:
        lk = (x.get("link") or "").strip()
        ttl = (x.get("title") or "").strip()
        k = _norm_key(ttl, lk)
        if not ttl or k in seen:
            continue
        seen.add(k)
        out.append(x)
        if len(out) >= max_items:
            break
    return out


# Optional deep snippet extraction (meta description / og:description)
_META_DESC_RE = re.compile(r'<meta[^>]+(?:name="description"|property="og:description")[^>]+content="([^"]+)"', re.IGNORECASE)
_TITLE_TAG_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)


def _fetch_meta_description(url: str, timeout: int = 8) -> str:
    if not url:
        return ""
    ck = f"meta_desc:{url}"
    cached = cache_get(ck, allow_stale=True)
    if cached is not None:
        return str(cached or "")

    try:
        r = _requests_get(url, timeout=timeout, headers=HTML_HEADERS)
        if r.status_code >= 400:
            return cache_set(ck, "", ttl_seconds=600, stale_ttl_seconds=6 * 3600)
        html = (r.text or "")[:250_000]
        m = _META_DESC_RE.search(html)
        if m:
            desc = _clean_text(m.group(1))
            return cache_set(ck, desc[:420], ttl_seconds=1800, stale_ttl_seconds=12 * 3600)

        m2 = _TITLE_TAG_RE.search(html)
        if m2:
            t = _clean_text(m2.group(1))
            return cache_set(ck, t[:260], ttl_seconds=900, stale_ttl_seconds=6 * 3600)

        return cache_set(ck, "", ttl_seconds=600, stale_ttl_seconds=6 * 3600)
    except Exception:
        return cache_set(ck, "", ttl_seconds=300, stale_ttl_seconds=3 * 3600)


# Scoring and theme detection
POS_WORDS = {
    "beat", "beats", "surge", "surges", "soar", "soars", "record", "profit", "profits", "growth",
    "upgrade", "upgraded", "strong", "stronger", "guidance raised", "raises guidance", "rally", "rallies",
    "partnership", "contract win", "wins contract", "backlog", "bullish", "outperform", "breakthrough",
    "margin expansion", "re-accelerate", "reaccelerate",
}
NEG_WORDS = {
    "miss", "misses", "plunge", "plunges", "drop", "drops", "slump", "slumps", "loss", "losses",
    "downgrade", "downgraded", "weak", "weaker", "guidance cut", "cuts guidance", "layoffs",
    "probe", "investigation", "lawsuit", "recall", "ban", "antitrust", "fine", "breach", "hack",
    "data leak", "outage", "warning", "cuts forecast", "lowers forecast",
}
INTENSIFIERS = {
    "sec", "doj", "ftc", "antitrust", "sanction", "ban", "export controls", "rates", "inflation", "cpi", "fomc",
    "earnings", "guidance", "forecast", "quarter", "merger", "acquisition", "bankruptcy", "strike",
    "ransomware", "breach", "hack", "outage",
}

THEME_BUCKETS = [
    ("earnings and guidance", ["earnings", "guidance", "forecast", "revenue", "margin", "profit", "results", "quarter"]),
    ("regulation and policy", ["sec", "doj", "ftc", "antitrust", "ban", "sanction", "export", "controls", "regulator", "law"]),
    ("chips and supply chain", ["chip", "gpu", "semiconductor", "tsmc", "asml", "supply", "fab", "foundry", "hbm", "packaging"]),
    ("product and platform launches", ["launch", "release", "rollout", "product", "platform", "model", "copilot", "gemini", "llama", "agent"]),
    ("cyber incidents", ["breach", "hack", "ransomware", "outage", "security", "leak"]),
    ("m&a and partnerships", ["acquire", "acquisition", "merge", "merger", "buyout", "partnership", "deal", "contract"]),
    ("macro and rates", ["cpi", "inflation", "rates", "fomc", "fed", "yield", "recession", "soft landing"]),
]

SECTOR_WATCHLIST = {
    "AI": ["NVDA", "MSFT", "GOOGL", "AMZN", "META", "AAPL", "AMD", "AVGO", "TSM", "ASML", "SMCI", "PLTR", "NOW", "ORCL"],
    "Semiconductors": ["NVDA", "AMD", "AVGO", "TSM", "ASML", "AMAT", "LRCX", "KLAC", "MU", "INTC", "QCOM"],
    "Cloud": ["MSFT", "AMZN", "GOOGL", "ORCL", "NOW", "CRM", "SNOW", "DDOG", "MDB"],
    "Cybersecurity": ["CRWD", "PANW", "ZS", "NET", "OKTA", "S", "FTNT"],
    "Defense": ["LMT", "NOC", "RTX", "GD", "BA", "HII"],
    "Medical": ["JNJ", "PFE", "LLY", "NVO", "MRK", "ABBV", "ISRG", "SYK", "MDT", "BSX"],
    "Energy": ["XOM", "CVX", "SLB", "COP", "EOG", "OXY", "NEE", "ENPH"],
    "Robotics": ["TSLA", "ABB", "ROK", "TER", "SYM"],
    "Infrastructure": ["CAT", "DE", "URI", "VMC", "MLM", "CSX", "UNP", "ETN"],
    "Financials": ["JPM", "BAC", "WFC", "GS", "MS", "BLK", "SCHW"],
    "Consumer": ["AMZN", "WMT", "COST", "TGT", "NKE", "SBUX", "MCD"],
    "General": ["SPY", "QQQ", "IWM"],
}

# Ticker extraction: $NVDA, (NVDA), NVDA, but avoid common words
_TICKER_RE = re.compile(r"(?:\$(?P<t1>[A-Z]{1,5})|\b(?P<t2>[A-Z]{1,5})\b)")
_TICKER_BLACKLIST = {
    "AI", "ETF", "USD", "CEO", "CFO", "SEC", "DOJ", "FTC", "FDA", "FOMC", "CPI", "US", "UK", "EU",
    "SPY", "QQQ",  # allow these via watchlist if needed, but blacklist in extraction to reduce noise
}


def _extract_tickers(text: str) -> List[str]:
    if not text:
        return []
    out: List[str] = []
    for m in _TICKER_RE.finditer(text):
        t = (m.group("t1") or m.group("t2") or "").upper().strip()
        if not t:
            continue
        if t in _TICKER_BLACKLIST:
            continue
        # very short ones often false positives, but allow 2-5. Allow 1 char only if it is a real watchlist match later.
        if len(t) == 1:
            continue
        out.append(t)
    # dedup preserve order
    seen = set()
    out2 = []
    for t in out:
        if t in seen:
            continue
        seen.add(t)
        out2.append(t)
    return out2


def _score_items(items: List[dict]) -> Tuple[int, str]:
    if not items:
        return 50, "NEUTRAL"

    pos = 0.0
    neg = 0.0
    intensity = 0.0

    for x in items[:18]:
        t = (x.get("title") or "").lower()
        d = (x.get("description") or "").lower()
        text = (t + " " + d).strip()

        for w in POS_WORDS:
            if w in text:
                pos += 1.0
        for w in NEG_WORDS:
            if w in text:
                neg += 1.0
        for w in INTENSIFIERS:
            if w in text:
                intensity += 0.35

        if "earnings" in text or "guidance" in text or "forecast" in text:
            intensity += 0.8
        if "sec" in text or "doj" in text or "ftc" in text:
            intensity += 0.9

    raw = 50.0 + 7.5 * (pos - neg)
    raw = raw + 3.0 * intensity

    score = int(round(max(0.0, min(100.0, raw))))
    if score >= 62:
        label = "POSITIVE"
    elif score <= 38:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"
    return score, label


def _themes_from_items(items: List[dict], max_themes: int = 3) -> List[Tuple[str, List[dict]]]:
    buckets: Dict[str, List[dict]] = {name: [] for name, _ in THEME_BUCKETS}
    other: List[dict] = []

    for x in items[:18]:
        text = ((x.get("title") or "") + " " + (x.get("description") or "")).lower()
        matched = False
        for name, kws in THEME_BUCKETS:
            if any(kw in text for kw in kws):
                buckets[name].append(x)
                matched = True
                break
        if not matched:
            other.append(x)

    ranked = [(k, v) for k, v in buckets.items() if v]
    ranked.sort(key=lambda t: len(t[1]), reverse=True)

    out = ranked[:max_themes]
    if len(out) < max_themes and other:
        out.append(("other notable headlines", other[:6]))
    return out[:max_themes]


def _fmt_sentiment_line(score: int, label: str) -> str:
    if label == "POSITIVE":
        return f"Tone is constructive ({score}/100)."
    if label == "NEGATIVE":
        return f"Tone is risk-heavy ({score}/100)."
    return f"Tone is mixed ({score}/100)."


def _build_today_read(sector: str, items: List[dict], score: int, label: str) -> str:
    """
    Flowing narrative. Avoids repeating the same headline by using themed picks.
    """
    if not items:
        return "No notable headlines returned in this sector."

    themes = _themes_from_items(items, max_themes=3)

    # Pick 1 representative story per theme, and stitch into a short paragraph
    picks: List[dict] = []
    for _, lst in themes:
        if lst:
            picks.append(lst[0])

    # Create a smooth narrative with minimal snippet use
    lines: List[str] = []
    lines.append(_fmt_sentiment_line(score, label))

    # Intro
    if sector.lower() in ["ai", "semiconductors", "cloud", "cybersecurity"]:
        lines.append("The main drivers today are a mix of product/platform momentum and catalyst risk.")
    else:
        lines.append("The tape today is being shaped by a handful of headline catalysts and follow-through risk.")

    # Body: 2-3 key updates
    used = 0
    for x in picks[:3]:
        title = (x.get("title") or "").strip()
        desc = (x.get("description") or "").strip()

        if not title:
            continue

        # Keep it readable: 1 headline + optional short clause from description
        if desc:
            desc2 = desc
            desc2 = re.sub(r"\s+", " ", desc2).strip()
            if len(desc2) > 220:
                desc2 = desc2[:220].rstrip() + "…"
            lines.append(f"{title}. {desc2}")
        else:
            lines.append(f"{title}.")
        used += 1

    if used == 0:
        lines.append("No clean summary could be generated from the returned headlines.")

    # Close with a practical framing
    if label == "POSITIVE":
        lines.append("Bottom line: favor adds on pullbacks, and prioritize the strongest operators in the group.")
    elif label == "NEGATIVE":
        lines.append("Bottom line: stay selective, reduce chasing, and keep sizing tighter until the catalyst clears.")
    else:
        lines.append("Bottom line: stay patient, and let price action confirm which narrative wins.")

    return " ".join(lines)


def _is_positive_text(text: str) -> bool:
    t = (text or "").lower()
    return any(w in t for w in POS_WORDS)


def _is_negative_text(text: str) -> bool:
    t = (text or "").lower()
    return any(w in t for w in NEG_WORDS)


def _is_risk_catalyst(text: str) -> bool:
    t = (text or "").lower()
    # catalyst risk: regulation, guidance/earnings, cyber, bans/export controls
    risk_kws = ["sec", "doj", "ftc", "antitrust", "ban", "sanction", "export", "controls", "guidance", "forecast", "earnings", "breach", "hack", "outage", "probe", "investigation", "lawsuit", "recall"]
    return any(k in t for k in risk_kws)


def _rank_unique(items: List[Tuple[str, str]], limit: int = 10) -> List[dict]:
    # items: [(ticker, reason)]
    seen = set()
    out = []
    for t, r in items:
        if not t or t in seen:
            continue
        seen.add(t)
        out.append({"ticker": t, "reason": r})
        if len(out) >= limit:
            break
    return out


def _build_actionable_implications(sector: str, score: int, label: str, items: List[dict]) -> Dict[str, List[dict]]:
    """
    Returns:
      - goAfter: names worth accumulating (positive bias)
      - watchClosely: names with active catalysts or headline risk
    """
    watchlist = SECTOR_WATCHLIST.get(sector, SECTOR_WATCHLIST.get("General", []))

    extracted: List[Tuple[str, str, dict]] = []  # (ticker, reason, item)
    for x in items[:18]:
        text = ((x.get("title") or "") + " " + (x.get("description") or "")).strip()
        for t in _extract_tickers(text):
            extracted.append((t, "Mentioned in headlines", x))

    # Score tickers based on context
    go_after: List[Tuple[str, str]] = []
    watch_close: List[Tuple[str, str]] = []

    # Prefer tickers mentioned explicitly first
    for t, _, x in extracted:
        text = ((x.get("title") or "") + " " + (x.get("description") or "")).strip()
        pos = _is_positive_text(text)
        neg = _is_negative_text(text)
        risk = _is_risk_catalyst(text)

        if pos and not neg and label in ["POSITIVE", "NEUTRAL"] and score >= 50:
            go_after.append((t, "Positive catalyst in headlines"))
        if risk or neg:
            watch_close.append((t, "Active catalyst risk in headlines"))

    # Add watchlist names as candidates if extraction is sparse
    # goAfter: only when sector tone is positive
    if label == "POSITIVE" and score >= 62:
        for t in watchlist[:10]:
            go_after.append((t, "Sector leader to accumulate on pullbacks"))

    # watchClosely: when sector tone is negative or very high intensity
    if label == "NEGATIVE" or score <= 38:
        for t in watchlist[:12]:
            watch_close.append((t, "Sector exposure to monitor closely"))

    # If we still have too little, keep it useful
    if len(go_after) < 4 and label != "NEGATIVE":
        for t in watchlist[:8]:
            go_after.append((t, "High-quality name to keep on the short list"))

    if len(watch_close) < 4:
        for t in watchlist[:10]:
            watch_close.append((t, "Watchlist name"))

    return {
        "goAfter": _rank_unique(go_after, limit=10),
        "watchClosely": _rank_unique(watch_close, limit=12),
    }


def _build_global_today_read(sector_blocks: List[dict]) -> str:
    """
    Combine the most important sector reads into one morning-style briefing without duplication.
    """
    if not sector_blocks:
        return "No sector reads available."

    # pick the 4 most "notable" sectors by distance from neutral
    def _notability(s: dict) -> int:
        sc = int((s.get("sentiment") or {}).get("score") or 50)
        return abs(sc - 50)

    ranked = sorted(sector_blocks, key=_notability, reverse=True)
    picks = ranked[:4] if len(ranked) >= 4 else ranked

    parts: List[str] = []
    parts.append("Here is the clean read for today across your sectors.")

    for s in picks:
        sec = s.get("sector", "Sector")
        sent = s.get("sentiment") or {}
        sc = int(sent.get("score") or 50)
        lab = sent.get("label") or "NEUTRAL"
        tr = (s.get("todayRead") or "").strip()
        # One short section per sector
        if tr:
            parts.append(f"{sec}: {lab} ({sc}/100). {tr}")

    parts.append("If you want this even tighter, reduce sectors or set max_items_per_sector lower.")
    return "\n\n".join(parts)


@app.get("/news/briefing")
def news_briefing(
    sectors: str = Query(default="AI,Medical,Energy,Robotics,Infrastructure,Semiconductors,Cloud,Cybersecurity,Defense,Financials,Consumer"),
    max_items_per_sector: int = Query(default=12, ge=5, le=30),
    deep: int = Query(default=0, ge=0, le=1),
):
    """
    Returns sector headlines plus:
      - sentiment score that varies
      - todayRead: a flowing narrative you can read quickly
      - implications: goAfter vs watchClosely
    deep=1: attempts to fetch story URLs and pull meta descriptions (best-effort).
    """
    key = f"news:brief:v490:{sectors}:{max_items_per_sector}:deep{deep}"
    cached = cache_get(key, allow_stale=True)
    if cached is not None:
        return cached

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
                for x in items:
                    x["sector"] = sec
                    x["source"] = "Google News"
                all_items.extend(items)
            except Exception as e:
                errors.append(f"{sec}: {type(e).__name__}: {str(e)}")

    # Global dedupe to reduce cross-sector repeats
    global_seen: set = set()
    all_items = _dedup_items(all_items, 1200, global_seen=global_seen)

    # Optional deep enrichment: pull meta descriptions for top items overall
    if deep == 1:
        to_fetch: List[dict] = [x for x in all_items[:360] if x.get("link")]
        with ThreadPoolExecutor(max_workers=10) as ex:
            futures = {ex.submit(_fetch_meta_description, x["link"], 8): x for x in to_fetch[:140]}
            for f in as_completed(futures):
                x = futures[f]
                try:
                    md = f.result() or ""
                    if md:
                        if not x.get("description") or len(str(x.get("description") or "")) < 40:
                            x["description"] = md
                        x["metaDescription"] = md
                except Exception:
                    continue

    sectors_out: List[dict] = []

    # Per-sector: local dedupe as well (using shared global_seen would over-prune; use a local set)
    for sec in sector_list:
        sec_raw = [x for x in all_items if x.get("sector") == sec]
        sec_seen: set = set()
        sec_items = _dedup_items(sec_raw, max_items_per_sector, global_seen=sec_seen)

        score, label = _score_items(sec_items)
        today_read = _build_today_read(sec, sec_items, score, label)
        implications = _build_actionable_implications(sec, score, label, sec_items)

        sectors_out.append({
            "sector": sec,
            "sentiment": {"label": label, "score": score},
            "todayRead": today_read,
            "implications": implications,  # now structured: goAfter + watchClosely
            "topHeadlines": [
                {
                    "title": x.get("title", ""),
                    "link": x.get("link", ""),
                    "published": x.get("published", ""),
                    "sourceFeed": x.get("source", "Google News"),
                    "sector": sec,
                    "description": x.get("description", ""),
                }
                for x in sec_items
            ],
        })

    scores = [int(s.get("sentiment", {}).get("score", 50)) for s in sectors_out if s.get("sentiment")]
    overall_score = int(round(sum(scores) / len(scores))) if scores else 50
    if overall_score >= 62:
        overall_label = "POSITIVE"
    elif overall_score <= 38:
        overall_label = "NEGATIVE"
    else:
        overall_label = "NEUTRAL"

    out = {
        "date": datetime.now(timezone.utc).date().isoformat(),
        "overallSentiment": {"label": overall_label, "score": overall_score},
        "todayRead": _build_global_today_read(sectors_out),
        "sectors": sectors_out,
        "errors": errors,
        "note": "todayRead is heuristic and based on titles + snippets (and optional meta descriptions if deep=1). Implications are watchlist and catalyst based, not financial advice.",
    }
    return cache_set(key, out, ttl_seconds=300, stale_ttl_seconds=3 * 3600)


# =========================================================
# Congress endpoint you’re using (kept compatible)
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

    rows = df.to_dict(orient="records") if hasattr(df, "to_dict") else list(df)

    buys: List[dict] = []
    sells: List[dict] = []

    for r in rows:
        best_dt = _row_best_dt(r)
        if best_dt is None or best_dt < since or best_dt > now:
            continue

        party = _norm_party(r)
        if not party:
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

        card = {
            "ticker": ticker,
            "companyName": pol,
            "demBuyers": 1 if (kind == "BUY" and party == "D") else 0,
            "repBuyers": 1 if (kind == "BUY" and party == "R") else 0,
            "demSellers": 1 if (kind == "SELL" and party == "D") else 0,
            "repSellers": 1 if (kind == "SELL" and party == "R") else 0,
            "lastFiledAt": _iso_date_only(filed_dt) or _iso_date_only(traded_dt),
            "strength": kind,
            "traded": _iso_date_only(traded_dt),
            "filed": _iso_date_only(filed_dt),
        }

        if kind == "BUY":
            buys.append(card)
        else:
            sells.append(card)

    return {
        "date": now.date().isoformat(),
        "windowDays": days,
        "windowStart": since.date().isoformat(),
        "windowEnd": now.date().isoformat(),
        "convergence": [],
        "politicianBuys": buys[:200],
        "politicianSells": sells[:200],
        "crypto": {"buys": [], "sells": [], "rawBuys": [], "rawSells": [], "raw": []},
    }
