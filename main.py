import os
import re
import time
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
app = FastAPI(title="Finance Signals Backend", version="4.7.1")

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
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "").strip()  # kept for compatibility, not required

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
    return {"status": "ok", "version": "4.7.1"}


@app.get("/health")
def health():
    return {
        "ok": True,
        "version": "4.7.1",
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
    """
    Returns last, prev, ret1dPct, ret5dPct, ret1mPct computed from trading days.
    5D uses last vs close 5 trading days ago (index -6).
    1M uses last vs close 21 trading days ago (index -22).
    """
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
# CNN Fear & Greed (best-effort, with weekend fallback)
# =========================================================
def _cnn_fear_greed_graphdata(date_str: Optional[str] = None) -> dict:
    d0 = date_str or datetime.now(timezone.utc).date().isoformat()

    # CNN sometimes 404s on weekends/holidays. Walk back up to 10 days.
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
# Market Snapshot (now includes 1D/5D/1M via Stooq, with fallbacks)
# =========================================================
@app.get("/market/snapshot")
def market_snapshot():
    key = "market:snapshot:v471"
    cached = cache_get(key, allow_stale=True)
    if cached is not None:
        return cached

    now = datetime.now(timezone.utc)
    errors: List[str] = []

    # Prefer Stooq for returns because it gives consistent history
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

    # Nasdaq quote as optional supplement / fallback for last+prev if Stooq was empty
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

    # Decide SPY values: prefer Stooq last/prev if present, else Nasdaq, else None
    spy_last_final = spy_hist.get("last") if spy_hist.get("last") is not None else spy_last
    spy_prev_final = spy_hist.get("prev") if spy_hist.get("prev") is not None else spy_prev

    # Returns: prefer Stooq computed values; if missing, compute 1D from last/prev if we can
    ret1d = spy_hist.get("ret1dPct")
    if ret1d is None and spy_last_final is not None and spy_prev_final is not None and spy_prev_final:
        ret1d = _pct(float(spy_last_final), float(spy_prev_final))

    ret5d = spy_hist.get("ret5dPct")
    ret1m = spy_hist.get("ret1mPct")

    # Decide VIX values: prefer Nasdaq for "live-ish" if present, else Stooq
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

    # Keep a last_good
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
    key = f"market:entry:v471:{window_days}"
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
# RSS helpers + News briefing
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


def _dedup_items(items: List[dict], max_items: int) -> List[dict]:
    seen = set()
    out = []
    for x in items:
        lk = (x.get("link") or "").strip()
        ttl = (x.get("title") or "").strip()
        k = lk or ttl
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(x)
        if len(out) >= max_items:
            break
    return out


@app.get("/news/briefing")
def news_briefing(
    sectors: str = Query(default="AI,Medical,Energy,Robotics,Infrastructure,Semiconductors,Cloud,Cybersecurity,Defense,Financials,Consumer"),
    max_items_per_sector: int = Query(default=12, ge=5, le=30),
):
    key = f"news:brief:v471:{sectors}:{max_items_per_sector}"
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
        futs = [(sec, ex.submit(_fetch_rss_items, url, 10, max_items_per_sector * 2, 240, 6 * 3600)) for sec, url in jobs]
        for sec, fut in futs:
            try:
                items = fut.result()
                for x in items:
                    x["sector"] = sec
                    x["source"] = "Google News"
                all_items.extend(items)
            except Exception as e:
                errors.append(f"{sec}: {type(e).__name__}: {str(e)}")

    all_items = _dedup_items(all_items, 600)

    sectors_out = []
    for sec in sector_list:
        sec_items = [x for x in all_items if x.get("sector") == sec][:max_items_per_sector]
        sectors_out.append({
            "sector": sec,
            "sentiment": {"label": "NEUTRAL", "score": 50},
            "summary": "",
            "implications": [],
            "topHeadlines": [
                {"title": x.get("title", ""), "link": x.get("link", ""), "published": x.get("published", ""), "sourceFeed": x.get("source", "Google News"), "sector": sec}
                for x in sec_items
            ],
            "watchlistMentions": [],
        })

    out = {
        "date": datetime.now(timezone.utc).date().isoformat(),
        "overallSentiment": {"label": "NEUTRAL", "score": 50},
        "sectors": sectors_out,
        "errors": errors,
        "note": "News briefing uses Google News RSS only.",
    }
    return cache_set(key, out, ttl_seconds=300, stale_ttl_seconds=3 * 3600)


# =========================================================
# Crypto News Briefing (for your Crypto tab)
# =========================================================
def _split_csv(s: str) -> List[str]:
    return [x.strip().upper() for x in (s or "").split(",") if x.strip()]


@app.get("/crypto/news/briefing")
def crypto_news_briefing(
    coins: str = Query(default="BTC,ETH,LINK,SHIB"),
    include_top_n: int = Query(default=15, ge=5, le=30),
):
    key = f"crypto:news:v471:{coins}:{include_top_n}"
    cached = cache_get(key, allow_stale=True)
    if cached is not None:
        return cached

    coin_list = _split_csv(coins)
    if not coin_list:
        coin_list = ["BTC", "ETH"]

    errors: List[str] = []
    now = datetime.now(timezone.utc).date().isoformat()

    catalysts = [
        "Watch major macro prints (CPI, FOMC, jobs) for risk-on/risk-off shifts.",
        "ETF/flows headlines can move BTC and ETH quickly.",
        "Regulatory headlines (SEC, CFTC, EU MiCA updates) can change sentiment fast.",
    ]

    def _coin_query(sym: str) -> str:
        if sym in ["BTC", "BITCOIN"]:
            return "Bitcoin"
        if sym in ["ETH", "ETHEREUM"]:
            return "Ethereum"
        return sym

    coins_out: List[dict] = []
    for sym in coin_list[:12]:
        try:
            url = _google_news_rss(f"{_coin_query(sym)} crypto")
            items = _fetch_rss_items(url, timeout=10, max_items=include_top_n, ttl_seconds=240, stale_ttl_seconds=6 * 3600)
            items = _dedup_items(items, include_top_n)

            headlines = [
                {
                    "title": x.get("title", ""),
                    "link": x.get("link", ""),
                    "published": x.get("published", ""),
                    "source": "Google News",
                    "bucket": "Google News",
                }
                for x in items
            ]

            coins_out.append({
                "symbol": sym,
                "sentiment": {"label": "NEUTRAL", "score": 50},
                "summary": "",
                "headlines": headlines,
            })
        except Exception as e:
            errors.append(f"{sym}: {type(e).__name__}: {str(e)}")
            coins_out.append({
                "symbol": sym,
                "sentiment": {"label": "NEUTRAL", "score": 50},
                "summary": "",
                "headlines": [],
            })

    out = {
        "date": now,
        "note": "Crypto news uses Google News RSS only.",
        "errors": errors,
        "sources": {"outlets": ["Google News RSS"]},
        "overallSentiment": {"label": "NEUTRAL", "score": 50},
        "catalysts": catalysts,
        "coins": coins_out,
    }
    return cache_set(key, out, ttl_seconds=300, stale_ttl_seconds=3 * 3600)


# =========================================================
# Congress parsing helpers
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


def _row_filed_dt(row: dict) -> Optional[datetime]:
    filed = _pick_first(row, ["Filed", "filed", "ReportDate", "report_date", "Date", "date"], "")
    return _parse_dt_any(filed)


def _row_traded_dt(row: dict) -> Optional[datetime]:
    traded = _pick_first(row, ["Traded", "traded", "TransactionDate", "transaction_date"], "")
    return _parse_dt_any(traded)


def _norm_politician(row: dict) -> str:
    return str(_pick_first(row, ["Politician", "politician", "Representative", "Senator", "Name", "name"], "")).strip()


def _norm_chamber(row: dict) -> str:
    v = _pick_first(row, ["Chamber", "chamber", "HouseSenate", "house_senate"], "")
    s = str(v).strip()
    if not s:
        return ""
    up = s.upper()
    if "HOUSE" in up:
        return "House"
    if "SENATE" in up:
        return "Senate"
    return s


def _norm_amount_range(row: dict) -> str:
    v = _pick_first(row, ["Amount", "amount", "AmountRange", "amountRange", "Range", "range"], "")
    return str(v).strip()


def _capitoltrades_links(politician: str, ticker: str) -> Dict[str, str]:
    pol_slug = quote_plus((politician or "").strip())
    t = quote_plus((ticker or "").strip().upper())
    return {
        "capitoltrades_politician": f"https://www.capitoltrades.com/politicians?search={pol_slug}" if pol_slug else "",
        "capitoltrades_ticker": f"https://www.capitoltrades.com/trades?search={t}" if t else "",
    }


# =========================================================
# Quiver access helper (cached)
# =========================================================
def _get_congress_df_cached(ttl_seconds: int = 180) -> Any:
    key = "quiver:congress_df"
    cached = cache_get(key, allow_stale=True)
    if cached is not None:
        return cached

    if not QUIVER_TOKEN:
        raise HTTPException(500, "QUIVER_TOKEN missing")

    q = quiverquant.quiver(QUIVER_TOKEN)
    df = q.congress_trading()
    return cache_set(key, df, ttl_seconds=ttl_seconds, stale_ttl_seconds=6 * 3600)


# =========================================================
# /report/today (aggregated tickers + convergence)
# =========================================================
@app.get("/report/today")
def report_today(
    window_days: Optional[int] = Query(default=None, ge=1, le=365),
    horizon_days: Optional[int] = Query(default=None, ge=1, le=365),
):
    days = window_days if window_days is not None else horizon_days if horizon_days is not None else 30

    cache_key = f"report:today:v471:{days}"
    cached = cache_get(cache_key, allow_stale=True)
    if cached is not None:
        return cached

    now = datetime.now(timezone.utc)
    since = now - timedelta(days=days)

    df = _get_congress_df_cached(ttl_seconds=180)
    if df is None or len(df) == 0:
        out = {
            "date": now.date().isoformat(),
            "windowDays": days,
            "windowStart": since.date().isoformat(),
            "windowEnd": now.date().isoformat(),
            "convergence": [],
            "politicianBuys": [],
            "politicianSells": [],
            "crypto": {"buys": [], "sells": [], "rawBuys": [], "rawSells": [], "raw": []},
        }
        return cache_set(cache_key, out, ttl_seconds=120, stale_ttl_seconds=1800)

    rows = df.to_dict(orient="records") if hasattr(df, "to_dict") else list(df)

    agg: Dict[str, Dict[str, Any]] = {}
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
        ticker = ticker.upper()

        filed_dt = _row_filed_dt(r)
        traded_dt = _row_traded_dt(r)
        last_dt = filed_dt or traded_dt or best_dt

        cur = agg.get(ticker)
        if not cur:
            cur = {
                "ticker": ticker,
                "companyName": "",
                "demBuyers": 0,
                "repBuyers": 0,
                "demSellers": 0,
                "repSellers": 0,
                "lastFiledAt": "",
                "_last_dt": last_dt,
            }
            agg[ticker] = cur

        if kind == "BUY":
            if party == "D":
                cur["demBuyers"] += 1
            elif party == "R":
                cur["repBuyers"] += 1
        else:
            if party == "D":
                cur["demSellers"] += 1
            elif party == "R":
                cur["repSellers"] += 1

        if last_dt and (cur.get("_last_dt") is None or last_dt > cur["_last_dt"]):
            cur["_last_dt"] = last_dt

    all_cards = list(agg.values())
    for c in all_cards:
        c["lastFiledAt"] = _iso_date_only(c.get("_last_dt"))
        c.pop("_last_dt", None)

    buys = [c for c in all_cards if (c.get("demBuyers", 0) + c.get("repBuyers", 0)) > 0]
    sells = [c for c in all_cards if (c.get("demSellers", 0) + c.get("repSellers", 0)) > 0]

    convergence = [
        c for c in buys
        if (c.get("demBuyers", 0) > 0 and c.get("repBuyers", 0) > 0)
    ]

    def _strength(c: Dict[str, Any]) -> int:
        return int(c.get("demBuyers", 0) + c.get("repBuyers", 0) + c.get("demSellers", 0) + c.get("repSellers", 0))

    def _sort_key(c: Dict[str, Any]) -> Tuple[int, str]:
        return (_strength(c), str(c.get("ticker", "")))

    buys.sort(key=_sort_key, reverse=True)
    sells.sort(key=_sort_key, reverse=True)
    convergence.sort(key=_sort_key, reverse=True)

    out = {
        "date": now.date().isoformat(),
        "windowDays": days,
        "windowStart": since.date().isoformat(),
        "windowEnd": now.date().isoformat(),
        "convergence": convergence[:120],
        "politicianBuys": buys[:220],
        "politicianSells": sells[:220],
        "crypto": {"buys": [], "sells": [], "rawBuys": [], "rawSells": [], "raw": []},
    }
    return cache_set(cache_key, out, ttl_seconds=120, stale_ttl_seconds=1800)


# =========================================================
# /report/holdings/common
# =========================================================
@app.get("/report/holdings/common")
def holdings_common(
    window_days: int = Query(default=365, ge=1, le=365),
    top_n: int = Query(default=30, ge=5, le=100),
):
    cache_key = f"report:holdings:common:v471:{window_days}:{top_n}"
    cached = cache_get(cache_key, allow_stale=True)
    if cached is not None:
        return cached

    now = datetime.now(timezone.utc)
    since = now - timedelta(days=window_days)

    df = _get_congress_df_cached(ttl_seconds=240)
    rows = df.to_dict(orient="records") if hasattr(df, "to_dict") else list(df)

    holders_by_ticker: Dict[str, set] = {}
    for r in rows:
        dt = _row_best_dt(r)
        if dt is None or dt < since or dt > now:
            continue
        ticker = _norm_ticker(r)
        if not ticker:
            continue
        pol = _norm_politician(r)
        if not pol:
            continue
        t = ticker.upper()
        if t not in holders_by_ticker:
            holders_by_ticker[t] = set()
        holders_by_ticker[t].add(pol)

    common = [{"ticker": t, "holders": len(pols)} for t, pols in holders_by_ticker.items()]
    common.sort(key=lambda x: (int(x["holders"]), str(x["ticker"])), reverse=True)

    out = {
        "date": now.date().isoformat(),
        "windowDays": window_days,
        "commonHoldings": common[:top_n],
        "note": "Holders = unique politicians with activity in the ticker during the window.",
    }
    return cache_set(cache_key, out, ttl_seconds=300, stale_ttl_seconds=3 * 3600)


# =========================================================
# /report/congress/daily
# =========================================================
@app.get("/report/congress/daily")
def congress_daily(
    window_days: int = Query(default=30, ge=1, le=365),
    limit: int = Query(default=250, ge=50, le=1000),
):
    cache_key = f"report:congress:daily:v471:{window_days}:{limit}"
    cached = cache_get(cache_key, allow_stale=True)
    if cached is not None:
        return cached

    now = datetime.now(timezone.utc)
    since = now - timedelta(days=window_days)

    df = _get_congress_df_cached(ttl_seconds=180)
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

        ticker = _norm_ticker(r).upper()
        if not ticker:
            continue

        politician = _norm_politician(r)
        chamber = _norm_chamber(r)
        filed_dt = _row_filed_dt(r)
        traded_dt = _row_traded_dt(r)
        amount_range = _norm_amount_range(r)

        group_dt = filed_dt or traded_dt or best_dt

        desc = str(_pick_first(r, ["Description", "description", "AssetDescription", "asset_description"], "")).strip()

        items.append({
            "_group_dt": group_dt,
            "kind": kind,
            "party": party,
            "ticker": ticker,
            "politician": politician,
            "chamber": chamber,
            "amountRange": amount_range,
            "traded": _iso_date_only(traded_dt),
            "filed": _iso_date_only(filed_dt),
            "description": desc,
            "links": _capitoltrades_links(politician, ticker),
        })

    items.sort(key=lambda x: (x.get("_group_dt") or datetime(1970, 1, 1, tzinfo=timezone.utc), x.get("ticker", "")), reverse=True)
    items = items[:limit]

    by_day: Dict[str, List[dict]] = {}
    for it in items:
        d = _iso_date_only(it.get("_group_dt"))
        it.pop("_group_dt", None)
        by_day.setdefault(d, []).append(it)

    days_sorted = sorted(by_day.keys(), reverse=True)
    days_out = [{"date": d, "items": by_day[d]} for d in days_sorted]

    out = {
        "date": now.date().isoformat(),
        "windowDays": window_days,
        "days": days_out,
        "note": "Grouped by filed date when present, else traded date.",
    }
    return cache_set(cache_key, out, ttl_seconds=120, stale_ttl_seconds=1800)
