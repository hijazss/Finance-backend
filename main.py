import os
import re
import json
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
app = FastAPI(title="Finance Signals Backend", version="4.0.0")

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


@app.get("/")
def root():
    return {"status": "ok", "version": "4.0.0"}


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
# Crypto detection
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
    for sym in TOP_COINS:
        for pat in _CRYPTO_PATTERNS.get(sym, []):
            if pat.search(text):
                hits.append(sym)
                break

    out: List[str] = []
    for sym in TOP_COINS:
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
    for sym in TOP_COINS:
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
# Congress: holdings proxy endpoint your UI expects
# GET /report/holdings/common?window_days=365
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
# Congress: day-by-day activity feed for UI
# GET /report/congress/daily?window_days=14
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
# Market Entry Index
# Uses Yahoo Finance chart API first, then falls back to Stooq
# =========================================================
def _yahoo_chart_daily_close(symbol: str, range_str: str = "1y", interval: str = "1d") -> List[Tuple[datetime, float]]:
    url = "https://query1.finance.yahoo.com/v8/finance/chart/" + quote_plus(symbol)
    params = {"range": range_str, "interval": interval}
    r = requests.get(url, params=params, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    j = r.json()
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


def _stooq_daily_close(symbol: str, lookback_days: int = 260) -> List[Tuple[datetime, float]]:
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()

    lines = [ln.strip() for ln in r.text.splitlines() if ln.strip()]
    if len(lines) < 5:
        return []

    header = lines[0].lower().split(",")
    try:
        date_i = header.index("date")
        close_i = header.index("close")
    except ValueError:
        return []

    out: List[Tuple[datetime, float]] = []
    for ln in lines[1:]:
        parts = ln.split(",")
        if len(parts) <= max(date_i, close_i):
            continue
        try:
            dt = datetime.strptime(parts[date_i], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            cl = float(parts[close_i])
            out.append((dt, cl))
        except Exception:
            continue

    out.sort(key=lambda x: x[0])
    if lookback_days and len(out) > lookback_days:
        out = out[-lookback_days:]
    return out


def _sma(vals: List[float], n: int) -> Optional[float]:
    if n <= 0 or len(vals) < n:
        return None
    return sum(vals[-n:]) / n


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


@app.get("/market/entry")
def market_entry(window_days: int = Query(default=365, ge=30, le=365)):
    now = datetime.now(timezone.utc)

    spy = []
    vix = []
    err_notes = []

    try:
        spy = _yahoo_chart_daily_close("SPY", range_str="1y", interval="1d")
        vix = _yahoo_chart_daily_close("^VIX", range_str="1y", interval="1d")
    except Exception as e:
        err_notes.append(f"Yahoo chart failed: {type(e).__name__}: {str(e)}")

    if len(spy) < 210 or len(vix) < 30:
        try:
            spy = _stooq_daily_close("spy.us", lookback_days=260)
            vix = _stooq_daily_close("^vix", lookback_days=260)
        except Exception as e:
            err_notes.append(f"Stooq failed: {type(e).__name__}: {str(e)}")

    if len(spy) < 210 or len(vix) < 30:
        return {
            "date": now.date().isoformat(),
            "score": 0,
            "regime": "NEUTRAL",
            "signal": "DATA UNAVAILABLE",
            "notes": " | ".join(err_notes) if err_notes else "Insufficient market data.",
            "components": {
                "spxTrend": 0.5,
                "vix": 0.5,
                "breadth": 0.5,
                "credit": 0.5,
                "rates": 0.5,
                "buffettProxy": 0.5,
            },
        }

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
    if err_notes:
        notes = notes + " | " + " | ".join(err_notes)

    return {
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


# =========================================================
# News + Sentiment (RSS)
# =========================================================
def _fetch_rss_items(url: str, timeout: int = 12, max_items: int = 30) -> List[dict]:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
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
    "profit", "profits", "outperform", "buy", "wins", "win", "breakout"
])
_NEG_WORDS = set([
    "miss", "misses", "drop", "drops", "plunge", "plunges", "down",
    "downgrade", "downgrades", "warning", "weak", "bear", "bearish",
    "lawsuit", "probe", "investigation", "fraud", "loss", "losses",
    "cut", "cuts", "layoff", "layoffs", "recession", "crash"
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
def news_top(max_items: int = Query(default=30, ge=10, le=150)):
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
# News briefing: sector summaries + implications
# GET /news/briefing?watchlist=...&max_items=60
# =========================================================
_SECTOR_KEYWORDS: Dict[str, List[str]] = {
    "AI": [
        "ai", "artificial intelligence", "machine learning", "ml", "llm", "chatgpt",
        "nvidia", "gpu", "data center", "datacenter", "semiconductor", "chip",
        "openai", "anthropic", "google", "microsoft", "amazon", "meta",
        "cloud", "hyperscaler"
    ],
    "Medical": [
        "medical", "health", "healthcare", "biotech", "pharma", "drug", "fda",
        "medtech", "device", "hospital", "diagnostic", "imaging", "mri", "ct",
        "robotic surgery", "surgery", "clinical trial"
    ],
    "Energy": [
        "energy", "oil", "crude", "gas", "lng", "opec", "refinery",
        "renewable", "solar", "wind", "nuclear", "uranium",
        "battery", "lithium", "grid", "power"
    ],
    "Robotics": [
        "robot", "robotics", "automation", "industrial", "factory",
        "humanoid", "drone", "autonomous", "self-driving"
    ],
    "Infrastructure": [
        "infrastructure", "construction", "transport", "rail", "bridge",
        "semiconductor fab", "fab", "foundry", "data center build",
        "utilities", "water", "broadband", "5g", "telecom"
    ],
}

def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def _tag_sectors(title: str) -> List[str]:
    t = _norm_text(title)
    hits: List[str] = []
    for sector, kws in _SECTOR_KEYWORDS.items():
        for kw in kws:
            k = _norm_text(kw)
            if not k:
                continue
            if re.search(rf"\b{re.escape(k)}\b", t) if re.fullmatch(r"[a-z0-9\- ]+", k) else (k in t):
                hits.append(sector)
                break
    return hits

def _build_sector_summary(headlines: List[str]) -> str:
    # very lightweight: highlight recurring themes by keyword
    blob = " ".join(_norm_text(h) for h in headlines if h)
    if not blob:
        return "No meaningful sector headlines in the sample."

    themes = []
    theme_map = [
        ("earnings/guidance", ["earnings", "guidance", "forecast", "revenue", "margin"]),
        ("rates/liquidity", ["fed", "rates", "yields", "inflation", "treasury"]),
        ("regulation/legal", ["sec", "doj", "ftc", "lawsuit", "probe", "regulation", "ban"]),
        ("capex/expansion", ["capex", "build", "data center", "factory", "plant", "expansion"]),
        ("m&a/partnerships", ["acquire", "acquisition", "merger", "deal", "partnership"]),
        ("supply chain", ["supply", "shortage", "shipments", "inventory"]),
    ]
    for name, kws in theme_map:
        if any(k in blob for k in kws):
            themes.append(name)

    if not themes:
        themes = ["sector-specific developments"]

    return "Themes: " + ", ".join(themes) + "."

def _build_implications(sector: str, headlines: List[str], sentiment_score: int) -> List[str]:
    # generic but useful, bounded to 2–4 bullets
    out: List[str] = []
    if sector == "AI":
        out.append("If cloud and chip demand headlines stay strong, AI infrastructure leaders can keep momentum.")
        out.append("Watch valuation sensitivity: higher yields or weaker guidance can trigger fast multiple compression.")
        out.append("Regulatory scrutiny or export controls can create sudden dispersion inside the sector.")
    elif sector == "Medical":
        out.append("FDA, trial readouts, and reimbursement headlines can dominate near-term moves more than macro.")
        out.append("If rates stay elevated, early-stage and high cash-burn names may remain volatile.")
        out.append("Device utilization trends can be a tell for procedure volumes and hospital capex appetite.")
    elif sector == "Energy":
        out.append("Oil, gas, and LNG are headline-driven: supply shocks and geopolitics can overwhelm fundamentals short term.")
        out.append("Grid and power buildout headlines can support electrification suppliers even if crude drifts.")
        out.append("If inflation re-accelerates, energy can act as a hedge but tends to stay choppy.")
    elif sector == "Robotics":
        out.append("Robotics tends to track industrial demand and capex cycles: watch PMI, orders, and guidance.")
        out.append("If labor tightness persists, automation adoption can stay durable even in slower growth.")
        out.append("AI coupling is key: perception and autonomy headlines often re-rate winners quickly.")
    elif sector == "Infrastructure":
        out.append("Infrastructure names can benefit from multi-quarter backlogs, but are sensitive to rates and funding pace.")
        out.append("Data center and grid buildout headlines can create persistent demand pockets.")
        out.append("Cost inflation in labor and materials can pressure margins unless contracts pass-through.")
    else:
        out.append("Macro headlines can set the tape: rates, inflation, and earnings guidance matter most over weeks to months.")
        out.append("If sentiment turns risk-off, high-beta growth themes typically underperform.")
    # sentiment tilt
    if sentiment_score >= 60:
        out.insert(0, "Headlines skew constructive; dips may be bought if liquidity stays supportive.")
    elif sentiment_score <= 40:
        out.insert(0, "Headlines skew cautious; expect higher volatility and more selectivity.")
    return out[:4]

@app.get("/news/briefing")
def news_briefing(
    watchlist: str = Query(default="SPY,QQQ,NVDA,AAPL,MSFT,AMZN,GOOGL,META,TSLA,BRK-B"),
    max_items: int = Query(default=60, ge=20, le=200),
    max_watch_items_per_ticker: int = Query(default=6, ge=2, le=20),
):
    # Pull both general + watchlist RSS (reuse the same feed logic)
    general_feeds = [
        "https://www.yahoo.com/news/rss/finance",
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "https://www.marketwatch.com/rss/topstories",
    ]

    general_items: List[dict] = []
    general_errors: List[str] = []
    for f in general_feeds:
        try:
            it = _fetch_rss_items(f, max_items=max_items)
            for x in it:
                x["sourceFeed"] = f
                x["source"] = "general"
            general_items.extend(it)
        except Exception as e:
            general_errors.append(f"{f} -> {type(e).__name__}: {str(e)}")

    syms = [t.strip().upper() for t in (watchlist or "").split(",") if t.strip()][:40]
    watch_items_by: Dict[str, List[dict]] = {}
    watch_errors: List[str] = []
    for t in syms:
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={quote_plus(t)}&region=US&lang=en-US"
        try:
            it = _fetch_rss_items(url, max_items=max_watch_items_per_ticker)
            for x in it:
                x["sourceFeed"] = "yahoo_ticker_rss"
                x["source"] = "watchlist"
                x["ticker"] = t
            watch_items_by[t] = it
        except Exception as e:
            watch_errors.append(f"{t}: {type(e).__name__}: {str(e)}")
            watch_items_by[t] = []

    merged: List[dict] = []
    merged.extend(general_items)

    for t in syms:
        merged.extend(watch_items_by.get(t, []))

    # de-dupe by link
    seen = set()
    deduped: List[dict] = []
    for x in merged:
        lk = (x.get("link") or "").strip()
        if not lk or lk in seen:
            continue
        seen.add(lk)
        deduped.append(x)

    # tag sectors
    for x in deduped:
        title = str(x.get("title") or "")
        x["sectors"] = _tag_sectors(title)
        x["bucket"] = "sector" if x["sectors"] else "general"

    overall_sent = _headline_sentiment([str(x.get("title") or "") for x in deduped])

    # Build sector payloads
    sector_rows: List[dict] = []
    for sector in ["AI", "Medical", "Energy", "Robotics", "Infrastructure"]:
        sector_items = [x for x in deduped if sector in (x.get("sectors") or [])]
        titles = [str(x.get("title") or "") for x in sector_items]
        sent = _headline_sentiment(titles)
        summary = _build_sector_summary(titles[:20])
        implications = _build_implications(sector, titles[:20], int(sent.get("score") or 50))

        sector_rows.append({
            "sector": sector,
            "count": len(sector_items),
            "sentiment": sent,
            "summary": summary,
            "implications_2_12_weeks": implications,
            "headlines": sector_items[:40],
        })

    # General / leftover headlines
    general_left = [x for x in deduped if not (x.get("sectors") or [])]
    general_titles = [str(x.get("title") or "") for x in general_left]
    general_sent = _headline_sentiment(general_titles)
    general_summary = _build_sector_summary(general_titles[:20])
    general_imp = _build_implications("General", general_titles[:20], int(general_sent.get("score") or 50))

    # Sort sectors by count descending
    sector_rows.sort(key=lambda x: (-int(x.get("count") or 0), x.get("sector") or ""))

    return {
        "date": datetime.now(timezone.utc).date().isoformat(),
        "overallSentiment": overall_sent,
        "sectors": sector_rows,
        "general": {
            "count": len(general_left),
            "sentiment": general_sent,
            "summary": general_summary,
            "implications_2_12_weeks": general_imp,
            "headlines": general_left[:50],
        },
        "sources": {
            "general_feeds": general_feeds,
            "watchlist": syms,
        },
        "errors": {
            "general": general_errors,
            "watchlist": watch_errors,
        },
        "note": "Briefing uses RSS headlines + keyword sector tagging + lightweight sentiment. It is not investment advice.",
    }
