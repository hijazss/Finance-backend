import os
import re
import time
import html
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from email.utils import parsedate_to_datetime

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


def _requests_head(url: str, timeout: int = 12, headers: Optional[dict] = None) -> requests.Response:
    return SESSION.head(url, timeout=timeout, headers=headers or UA_HEADERS, allow_redirects=True)


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
    # CNN sometimes 404s for "today" early in the day or if a date is missing
    # We try date_str, then yesterday.
    d = date_str or datetime.now(timezone.utc).date().isoformat()
    for attempt_date in [d, (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()]:
        url = f"https://production.dataviz.cnn.io/index/fearandgreed/graphdata/{attempt_date}"
        r = _requests_get(url, timeout=16, headers=UA_HEADERS)
        if r.status_code == 404:
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("Fear & Greed graphdata not available for today or yesterday")


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
# Daily market read (MATCHES YOUR HTML EXPECTATIONS)
# =========================================================
@app.get("/market/read")
def market_read():
    """
    Returns what your frontend expects:
      {
        date,
        summary,
        sp500: {symbol,last,ret1dPct,ret5dPct,ret1mPct},
        nasdaq:{symbol,last,ret1dPct,ret5dPct,ret1mPct},
        fearGreed:{score,rating},
        errors:[...]
      }
    """
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
# Market Entry Index (kept)
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
# RSS helpers + headline summaries (NEW)
# =========================================================
def _google_news_rss(query: str) -> str:
    q = quote_plus(query)
    return f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"


def _parse_pubdate(pub: str) -> Optional[datetime]:
    s = (pub or "").strip()
    if not s:
        return None
    # RSS pubDate is usually RFC822
    try:
        dt = parsedate_to_datetime(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        pass
    # Some Atom feeds use ISO timestamps in "published"/"updated"
    try:
        s2 = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s2)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _normalize_title_for_dedup(title: str) -> str:
    t = (title or "").strip()
    if not t:
        return ""
    # Google News often appends " - Outlet"
    t = re.sub(r"\s+-\s+[^-]{2,80}$", "", t).strip()
    return t.lower()


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

    out: List[dict] = []

    channel = root.find("channel")
    if channel is None:
        # Atom
        entries = root.findall("{http://www.w3.org/2005/Atom}entry")
        for e in entries[:max_items]:
            title = (e.findtext("{http://www.w3.org/2005/Atom}title") or "").strip()
            link_el = e.find("{http://www.w3.org/2005/Atom}link")
            link = (link_el.get("href") if link_el is not None else "") or ""
            pub = (e.findtext("{http://www.w3.org/2005/Atom}updated") or e.findtext("{http://www.w3.org/2005/Atom}published") or "").strip()
            if title:
                dt = _parse_pubdate(pub)
                out.append({"title": title, "link": link, "published": pub, "published_dt": dt.isoformat() if dt else ""})
        return out

    # RSS
    for item in channel.findall("item")[:max_items]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        if title:
            dt = _parse_pubdate(pub)
            out.append({"title": title, "link": link, "published": pub, "published_dt": dt.isoformat() if dt else ""})
    return out


def _fetch_rss_items(
    url: str,
    timeout: int = 10,
    max_items: int = 25,
    ttl_seconds: int = 240,
    stale_ttl_seconds: int = 6 * 3600
) -> List[dict]:
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


def _dedup_items(items: List[dict], max_items: int) -> List[dict]:
    seen = set()
    out = []
    for x in items:
        ttl = (x.get("title") or "").strip()
        lk = (x.get("link") or "").strip()

        k_title = _normalize_title_for_dedup(ttl)
        k = k_title or lk
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(x)
        if len(out) >= max_items:
            break
    return out


def _filter_recent_items(items: List[dict], max_age_days: int = 30) -> List[dict]:
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=max_age_days)
    out = []
    for x in items:
        dt = None
        if x.get("published_dt"):
            try:
                dt = datetime.fromisoformat(str(x.get("published_dt"))).astimezone(timezone.utc)
            except Exception:
                dt = None
        if dt is None:
            dt = _parse_pubdate(str(x.get("published") or ""))
        # If missing date, keep it (some feeds omit, but Google usually has it)
        if dt is None or dt >= cutoff:
            out.append(x)
    return out


def _resolve_google_news_link(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return ""
    # Cache resolves for 24h
    key = f"resolve:{u}"
    cached = cache_get(key, allow_stale=True)
    if cached:
        return str(cached)

    try:
        # HEAD is cheaper; many sites block HEAD, so we fall back to GET.
        try:
            r = _requests_head(u, timeout=10, headers=UA_HEADERS)
            final_url = r.url or u
        except Exception:
            r = _requests_get(u, timeout=12, headers=UA_HEADERS)
            final_url = r.url or u

        final_url = str(final_url)
        return cache_set(key, final_url, ttl_seconds=6 * 3600, stale_ttl_seconds=48 * 3600)
    except Exception:
        return cache_set(key, u, ttl_seconds=900, stale_ttl_seconds=6 * 3600)


_META_DESC_RE = re.compile(
    r'<meta\s+(?:property|name)=["\'](?:og:description|description)["\']\s+content=["\']([^"\']{1,500})["\']\s*/?>',
    re.IGNORECASE
)


def _extract_meta_description(html_text: str) -> str:
    if not html_text:
        return ""
    m = _META_DESC_RE.search(html_text)
    if not m:
        return ""
    desc = m.group(1) or ""
    desc = html.unescape(desc).strip()
    desc = re.sub(r"\s+", " ", desc).strip()
    return desc


def _headline_summary_from_link(link: str) -> str:
    lk = (link or "").strip()
    if not lk:
        return ""

    key = f"hs:{lk}"
    cached = cache_get(key, allow_stale=True)
    if cached is not None:
        return str(cached)

    try:
        final_url = lk
        if "news.google.com" in lk:
            final_url = _resolve_google_news_link(lk)

        r = _requests_get(final_url, timeout=14, headers=UA_HEADERS)
        text = r.text or ""
        desc = _extract_meta_description(text)

        # Defensive trimming
        if desc:
            if len(desc) > 260:
                desc = desc[:257].rstrip() + "..."
            return cache_set(key, desc, ttl_seconds=24 * 3600, stale_ttl_seconds=72 * 3600)

        return cache_set(key, "", ttl_seconds=6 * 3600, stale_ttl_seconds=24 * 3600)
    except Exception:
        return cache_set(key, "", ttl_seconds=900, stale_ttl_seconds=6 * 3600)


def _attach_summaries(items: List[dict], max_summaries: int = 8) -> List[dict]:
    """
    Adds item["summary"] (short meta description) for the first N items only.
    Everything else remains unchanged, so the UI can show summaries when present.
    """
    if not items:
        return items

    n = min(max_summaries, len(items))
    idxs = list(range(n))

    def work(i: int) -> Tuple[int, str]:
        lk = str(items[i].get("link") or "")
        return i, _headline_summary_from_link(lk)

    try:
        with ThreadPoolExecutor(max_workers=8) as ex:
            results = list(ex.map(work, idxs))
        for i, summ in results:
            if summ:
                items[i]["summary"] = summ
    except Exception:
        # If anything fails, just return without summaries
        return items

    return items


# =========================================================
# Ticker extraction + sentiment (tuned to avoid constant 100s)
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
    "rally", "wins", "approval", "partnership", "acquisition", "buyback"
}
_NEG_WORDS = {
    "miss", "misses", "slump", "falls", "drop", "downgrade", "weak", "layoff", "cuts", "probe",
    "lawsuit", "ban", "halt", "recall", "fraud", "warning"
}


def _sentiment_from_titles(titles: List[str]) -> Tuple[int, str]:
    # More stable: normalize by number of titles so it does not pin at 100 easily.
    if not titles:
        return 50, "NEUTRAL"

    pos = 0
    neg = 0
    for t in titles:
        s = (t or "").lower()
        pos += sum(1 for w in _POS_WORDS if w in s)
        neg += sum(1 for w in _NEG_WORDS if w in s)

    n = max(1, len(titles))
    # Net impact per title, scaled
    net = (pos - neg) / float(n)
    score = int(round(50.0 + 25.0 * net))  # each +1 net per title moves about +25
    score = max(0, min(100, score))

    label = "BULLISH" if score >= 62 else "BEARISH" if score <= 38 else "NEUTRAL"
    return score, label


def _flowing_summary(sector: str, titles: List[str], watchlist: List[str]) -> Tuple[str, List[str], List[str]]:
    picked = []
    seen = set()
    for t in titles:
        tt = (t or "").strip()
        if not tt:
            continue
        key = _normalize_title_for_dedup(tt)
        if not key or key in seen:
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
# News briefing (adds summary per headline + filters >30d)
# =========================================================
@app.get("/news/briefing")
def news_briefing(
    tickers: str = Query(default=""),
    max_general: int = Query(default=60, ge=10, le=200),
    max_per_ticker: int = Query(default=6, ge=1, le=30),
    sectors: str = Query(default="AI,Medical,Energy,Robotics,Infrastructure,Semiconductors,Cloud,Cybersecurity,Defense,Financials,Consumer"),
    max_items_per_sector: int = Query(default=12, ge=5, le=30),
):
    key = f"news:brief:v490:{tickers}:{max_general}:{max_per_ticker}:{sectors}:{max_items_per_sector}"
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
                items = _filter_recent_items(items, max_age_days=30)
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
            futs2 = [(t, ex.submit(_fetch_rss_items, url, 10, max_per_ticker * 2, 240, 6 * 3600)) for t, url in ticker_jobs]
            for t, fut in futs2:
                try:
                    items = fut.result()
                    items = _filter_recent_items(items, max_age_days=30)
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
        sec_items = [x for x in all_items if x.get("sector") == sec][:max_items_per_sector]
        sec_items = _attach_summaries(sec_items, max_summaries=min(8, max_items_per_sector))
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
                    "sector": sec,
                    "summary": x.get("summary", ""),
                }
                for x in sec_items
            ],
            "watchlistMentions": watch_mentions,
            "tickersMentioned": tickers_mentioned,
        })

    if watch:
        wl_items = [x for x in all_items if x.get("sector") == "Watchlist"][:max_general]
        wl_items = _attach_summaries(wl_items, max_summaries=min(10, max_general))
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
                    "sector": "Watchlist",
                    "summary": x.get("summary", ""),
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
        "note": "Briefing uses Google News RSS. Headline summaries are best-effort from article meta descriptions. Items older than 30 days are filtered out.",
    }
    return cache_set(key, out, ttl_seconds=300, stale_ttl_seconds=3 * 3600)


# =========================================================
# Congress helpers + endpoints
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

        party = _norm_party(r)
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
            "party": party,
            "chamber": chamber,
            "amountRange": amt,
            "traded": _iso_date_only(traded_dt),
            "filed": _iso_date_only(filed_dt),
            "description": desc,
            "links": {"capitoltrades_politician": "", "capitoltrades_ticker": ""},
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
# Crypto briefing endpoint (adds summary per headline + filters >30d)
# =========================================================
@app.get("/crypto/news/briefing")
def crypto_news_briefing(
    coins: str = Query(default="BTC,ETH,LINK,SHIB"),
    include_top_n: int = Query(default=15, ge=5, le=50),
):
    key = f"crypto:news:v490:{coins}:{include_top_n}"
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
                max_items=max(10, include_top_n * 2),
                ttl_seconds=240,
                stale_ttl_seconds=6 * 3600
            )
            items = _filter_recent_items(items, max_age_days=30)
            items = _dedup_items(items, include_top_n)
            items = _attach_summaries(items, max_summaries=min(8, include_top_n))

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
                        "source": "Google News",
                        "summary": x.get("summary", ""),
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
        "note": "Crypto briefing is headline-based (Google News RSS). Headline summaries are best-effort from article meta descriptions. Items older than 30 days are filtered out.",
    }
    return cache_set(key, out, ttl_seconds=300, stale_ttl_seconds=3 * 3600)
