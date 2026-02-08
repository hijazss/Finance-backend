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
app = FastAPI(title="Finance Signals Backend", version="4.12.2")

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
    return {"status": "ok", "version": "4.12.2"}


@app.get("/health")
def health():
    return {
        "ok": True,
        "version": "4.12.2",
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


def _ret_from_series(vals: List[float], offset: int) -> Optional[float]:
    if len(vals) < (offset + 1):
        return None
    a = float(vals[-1])
    b = float(vals[-1 - offset])
    return _pct(a, b)


# =========================================================
# Nasdaq quote (primary)
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
# Stooq history (optional, never required)
# =========================================================
def _stooq_daily_closes(symbol: str) -> List[Tuple[datetime, float]]:
    if _is_cooled_down("stooq"):
        raise RuntimeError("Stooq in cooldown")

    url = "https://stooq.com/q/d/l/"
    params = {"s": symbol, "i": "d"}

    last_err = None
    for attempt in range(3):
        try:
            r = _requests_get(url, params=params, timeout=18, headers=UA_HEADERS)
            if r.status_code >= 400:
                last_err = f"HTTP {r.status_code}"
                time.sleep(0.6 * (attempt + 1))
                continue

            lines = (r.text or "").strip().splitlines()
            if len(lines) < 3:
                last_err = "insufficient CSV rows"
                time.sleep(0.6 * (attempt + 1))
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
            time.sleep(0.6 * (attempt + 1))
        except requests.exceptions.Timeout as e:
            last_err = f"Timeout: {type(e).__name__}"
            time.sleep(0.6 * (attempt + 1))
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:160]}"
            time.sleep(0.6 * (attempt + 1))

    _cooldown("stooq", 120)
    raise RuntimeError(f"Stooq failed: {last_err or 'unknown'}")


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


def _risk_label(score: int) -> str:
    if score >= 70:
        return "RISK-ON"
    if score <= 40:
        return "RISK-OFF"
    return "NEUTRAL"


def _simple_risk_proxy(spy_ret1d: Optional[float], qqq_ret1d: Optional[float], vix_last: Optional[float]) -> Dict[str, Any]:
    """
    Fully daily, no CNN, no Yahoo.
    Uses: SPY 1D, QQQ 1D, VIX level.
    Returns a 0..100 score + label.
    """
    score = 50.0
    notes: List[str] = []

    if isinstance(spy_ret1d, (int, float)):
        score += max(-8.0, min(8.0, float(spy_ret1d))) * 1.2
        notes.append(f"SPY 1D={spy_ret1d:.2f}%")
    if isinstance(qqq_ret1d, (int, float)):
        score += max(-10.0, min(10.0, float(qqq_ret1d))) * 0.9
        notes.append(f"QQQ 1D={qqq_ret1d:.2f}%")
    if isinstance(vix_last, (int, float)):
        # lower VIX -> higher score
        v = float(vix_last)
        if v <= 14:
            score += 10
        elif v <= 18:
            score += 5
        elif v >= 28:
            score -= 12
        elif v >= 22:
            score -= 7
        notes.append(f"VIX={v:.2f}")

    score_i = int(round(max(0.0, min(100.0, score))))
    return {"score": score_i, "label": _risk_label(score_i), "notes": "; ".join(notes)}


# =========================================================
# NEW: Daily market index dashboard
# =========================================================
@app.get("/market/index")
def market_index():
    key = "market:index:v4122"
    cached = cache_get(key, allow_stale=True)
    if cached is not None:
        return cached

    now = datetime.now(timezone.utc)
    errors: List[str] = []

    # Primary daily quotes from Nasdaq
    def q(symbol: str) -> Tuple[Optional[float], Optional[float]]:
        try:
            return _nasdaq_last_and_prev(symbol)
        except Exception as e:
            errors.append(f"Nasdaq {symbol}: {type(e).__name__}: {str(e)}")
            return None, None

    spy_last, spy_prev = q("SPY")
    qqq_last, qqq_prev = q("QQQ")
    vix_last, vix_prev = q("VIX")

    spy_ret1d = _pct(spy_last, spy_prev) if (spy_last is not None and spy_prev) else None
    qqq_ret1d = _pct(qqq_last, qqq_prev) if (qqq_last is not None and qqq_prev) else None

    # Optional: history (for 5D/1M + realized vol). Never required.
    spy_vals: List[float] = []
    qqq_vals: List[float] = []
    vix_vals: List[float] = []

    try:
        spy_hist = _stooq_daily_closes("spy.us")
        spy_vals = [c for _, c in spy_hist]
    except Exception as e:
        errors.append(f"Stooq SPY: {type(e).__name__}: {str(e)}")

    try:
        qqq_hist = _stooq_daily_closes("qqq.us")
        qqq_vals = [c for _, c in qqq_hist]
    except Exception as e:
        errors.append(f"Stooq QQQ: {type(e).__name__}: {str(e)}")

    try:
        vix_hist = _stooq_daily_closes("vix")
        vix_vals = [c for _, c in vix_hist]
    except Exception as e:
        errors.append(f"Stooq VIX: {type(e).__name__}: {str(e)}")

    spy_ret5d = _ret_from_series(spy_vals, 5)
    spy_ret1m = _ret_from_series(spy_vals, 21)

    qqq_ret5d = _ret_from_series(qqq_vals, 5)
    qqq_ret1m = _ret_from_series(qqq_vals, 21)

    spy_rv20 = _realized_vol_pct(spy_vals, 20)

    risk_proxy = _simple_risk_proxy(spy_ret1d, qqq_ret1d, vix_last)

    out = {
        "date": now.date().isoformat(),
        "indices": {
            "SPY": {"last": spy_last, "ret1dPct": spy_ret1d, "ret5dPct": spy_ret5d, "ret1mPct": spy_ret1m},
            "QQQ": {"last": qqq_last, "ret1dPct": qqq_ret1d, "ret5dPct": qqq_ret5d, "ret1mPct": qqq_ret1m},
            "VIX": {"last": vix_last, "ret1dPct": (_pct(vix_last, vix_prev) if (vix_last is not None and vix_prev) else None)},
        },
        "volatility": {
            "spyRealizedVol20dPct": spy_rv20,
        },
        "riskAppetite": risk_proxy,
        "errors": errors,
        "note": "Primary quotes via Nasdaq. 5D/1M/realized vol use Stooq when available. Endpoint never depends on Yahoo.",
    }

    return cache_set(key, out, ttl_seconds=180, stale_ttl_seconds=1800)


# =========================================================
# (Keep your existing RSS + news/briefing + crypto + congress endpoints below)
# NOTE: I am not repeating the entire rest of your file here because you only asked
# to fix the daily market index provider issue. Paste this file over your existing
# backend only if you want /market/index added safely.
# =========================================================

# IMPORTANT:
# If you want me to repaste the full backend including your existing /news/briefing,
/*  /crypto/news/briefing, /report/holdings/common, /report/congress/daily, etc.,
    tell me "repaste full backend including all endpoints" and I will provide the full file
    with this /market/index integrated. */
