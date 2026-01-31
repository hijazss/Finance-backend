"""
Finance Signals Backend (single-file)

What this backend provides:
  - GET /                       health check
  - GET /report/today           congress buys/sells + bipartisan overlap + crypto detection
  - GET /report/crypto          crypto-only view (from /report/today)
  - GET /report/holdings/common common holdings proxy (frontend-friendly shape)
  - GET /report/congress-holdings richer holdings proxy (topHoldings + breadth + scoring)
  - GET /market/entry           market entry score (SPY + VIX) with Stooq primary + Yahoo fallback

Deploy notes (Render):
  - Set env var: QUIVER_TOKEN
  - Start command: uvicorn main:app --host 0.0.0.0 --port $PORT
  - Runtime: python-3.11.x recommended
"""

import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
import quiverquant
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# --------------------------
# App + CORS
# --------------------------
app = FastAPI(title="Finance Signals Backend", version="3.0.0")

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
    return {"status": "ok", "version": app.version}


# --------------------------
# HTTP session with retries
# --------------------------
def _http_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update(
        {"User-Agent": "Mozilla/5.0 (compatible; FinanceSignals/1.0; +https://hijazss.github.io)"}
    )
    return s


_SESSION = _http_session()


# --------------------------
# Helpers
# --------------------------
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


# --------------------------
# Crypto detection
# --------------------------
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

CRYPTO_RELATED_TICKERS = set(
    [
        "IBIT",
        "FBTC",
        "ARKB",
        "BITB",
        "BTCO",
        "HODL",
        "GBTC",
        "BITO",
        "ETHE",
        "ETHA",
        "COIN",
        "MSTR",
        "RIOT",
        "MARA",
        "HUT",
        "CLSK",
    ]
)

TICKER_TO_COINS: Dict[str, List[str]] = {
    "IBIT": ["BTC"],
    "FBTC": ["BTC"],
    "ARKB": ["BTC"],
    "BITB": ["BTC"],
    "BTCO": ["BTC"],
    "HODL": ["BTC"],
    "GBTC": ["BTC"],
    "BITO": ["BTC"],
    "ETHE": ["ETH"],
    "ETHA": ["ETH"],
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
        "AssetDescription",
        "asset_description",
        "Description",
        "description",
        "Asset",
        "asset",
        "Name",
        "name",
        "Issuer",
        "issuer",
        "Ticker",
        "ticker",
        "Stock",
        "stock",
        "Symbol",
        "symbol",
        "Owner",
        "owner",
        "Type",
        "type",
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


# --------------------------
# Quiver fetch (single place)
# --------------------------
def _get_congress_rows() -> List[dict]:
    if not QUIVER_TOKEN:
        raise HTTPException(500, "QUIVER_TOKEN missing")

    q = quiverquant.quiver(QUIVER_TOKEN)
    df = q.congress_trading()

    if df is None or len(df) == 0:
        return []

    try:
        return df.to_dict(orient="records")
    except Exception:
        return list(df)


# --------------------------
# Congress: today
# --------------------------
@app.get("/report/today")
def report_today(
    window_days: Optional[int] = Query(default=None, ge=1, le=365),
    horizon_days: Optional[int] = Query(default=None, ge=1, le=365),
):
    days = window_days if window_days is not None else horizon_days if horizon_days is not None else 30

    now = datetime.now(timezone.utc)
    since = now - timedelta(days=days)

    rows = _get_congress_rows()
    if not rows:
        return {
            "date": now.date().isoformat(),
            "windowDays": days,
            "windowStart": since.date().isoformat(),
            "windowEnd": now.date().isoformat(),
            "convergence": [],
            "politicianBuys": [],
            "politicianSells": [],
            "crypto": {"buys": [], "sells": [], "rawBuys": [], "rawSells": [], "raw": [], "topCoins": TOP_COINS},
        }

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
        amount = str(
            pick_first(r, ["Amount", "amount", "Range", "range", "AmountRange", "amount_range"], "")
        ).strip()

        filed_dt = parse_dt_any(pick_first(r, ["Filed", "filed", "ReportDate", "report_date", "Date", "date"], ""))
        traded_dt = parse_dt_any(pick_first(r, ["Traded", "traded", "TransactionDate", "transaction_date"], ""))

        desc = str(
            pick_first(r, ["AssetDescription", "asset_description", "Description", "description", "Asset", "asset"], "")
        ).strip()

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
        overlap_cards.append(
            {
                "ticker": t,
                "companyName": "",
                "demBuyers": dem_ct,
                "repBuyers": rep_ct,
                "demSellers": 0,
                "repSellers": 0,
                "lastFiledAt": iso_date_only(latest_dt),
                "strength": "OVERLAP",
            }
        )

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


@app.get("/report/crypto")
def report_crypto(
    window_days: Optional[int] = Query(default=None, ge=1, le=365),
    horizon_days: Optional[int] = Query(default=None, ge=1, le=365),
):
    payload = report_today(window_days=window_days, horizon_days=horizon_days)
    return {
        "date": payload["date"],
        "windowDays": payload["windowDays"],
        "windowStart": payload["windowStart"],
        "windowEnd": payload["windowEnd"],
        "crypto": payload["crypto"],
    }


# --------------------------
# Holdings proxy (two endpoints)
# --------------------------
@app.get("/report/holdings/common")
def report_holdings_common(
    window_days: int = Query(default=365, ge=30, le=365),
    top_n: int = Query(default=30, ge=5, le=100),
):
    """
    Frontend-friendly holdings shape:
      {
        "date": "...",
        "windowDays": 365,
        "commonHoldings": [
          {"ticker":"NVDA","companyName":"","holders":42},
          ...
        ],
        "meta": {...}
      }

    holders = unique politicians who disclosed activity in ticker within the window
    """
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=window_days)

    rows = _get_congress_rows()
    if not rows:
        return {
            "date": now.date().isoformat(),
            "windowDays": window_days,
            "windowStart": since.date().isoformat(),
            "windowEnd": now.date().isoformat(),
            "commonHoldings": [],
            "meta": {"uniquePoliticians": 0, "uniqueTickers": 0, "topN": top_n},
            "note": "No congress trading rows returned for this window.",
        }

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
        if t not in per_ticker:
            per_ticker[t] = set()
        per_ticker[t].add(pol)

    common = [
        {"ticker": t, "companyName": "", "holders": len(pols)}
        for t, pols in per_ticker.items()
    ]
    common.sort(key=lambda x: (-int(x["holders"]), x["ticker"]))
    common = common[:top_n]

    return {
        "date": now.date().isoformat(),
        "windowDays": window_days,
        "windowStart": since.date().isoformat(),
        "windowEnd": now.date().isoformat(),
        "commonHoldings": common,
        "meta": {"uniquePoliticians": len(all_pol), "uniqueTickers": len(per_ticker), "topN": top_n},
        "note": "Holdings are a proxy from disclosures: holders = unique members with activity in ticker within window.",
    }


@app.get("/report/congress-holdings")
def report_congress_holdings(
    window_days: int = Query(default=365, ge=30, le=365),
    top_n: int = Query(default=30, ge=5, le=100),
):
    """
    Rich holdings proxy:
      - per ticker: holderCount, demHolders, repHolders, breadthPct, holdingsScore, tradeCount, lastActivity
      - topHoldings = first top_n sorted by holderCount
    """
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=window_days)

    rows = _get_congress_rows()
    if not rows:
        return {
            "date": now.date().isoformat(),
            "windowDays": window_days,
            "windowStart": since.date().isoformat(),
            "windowEnd": now.date().isoformat(),
            "breadth": {"uniquePoliticians": 0, "uniqueTickers": 0},
            "topHoldings": [],
            "holdings": [],
            "note": "No congress trading rows returned for this window.",
        }

    pol_set_all = set()
    per: Dict[str, dict] = {}

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

        party = norm_party_from_any(r)

        pol_set_all.add(pol)
        t = ticker.upper()

        cur = per.get(t)
        if not cur:
            cur = {
                "ticker": t,
                "politicians": set(),
                "dems": set(),
                "reps": set(),
                "lastSeen": None,
                "tradeCount": 0,
            }
            per[t] = cur

        cur["politicians"].add(pol)
        if party == "D":
            cur["dems"].add(pol)
        elif party == "R":
            cur["reps"].add(pol)

        cur["tradeCount"] += 1
        if cur["lastSeen"] is None or best_dt > cur["lastSeen"]:
            cur["lastSeen"] = best_dt

    total_unique_pol = len(pol_set_all)

    def holdings_score(holder_count: int, dem_count: int, rep_count: int) -> int:
        if total_unique_pol <= 0:
            return 0
        breadth_pct = holder_count / total_unique_pol
        base = breadth_pct * 85.0
        bipartisan = 0.0
        if dem_count > 0 and rep_count > 0:
            balance = min(dem_count, rep_count) / max(dem_count, rep_count)
            bipartisan = 15.0 * balance
        return int(round(min(100.0, base + bipartisan)))

    holdings: List[dict] = []
    for t, cur in per.items():
        holder_count = len(cur["politicians"])
        dem_count = len(cur["dems"])
        rep_count = len(cur["reps"])
        last_seen = cur["lastSeen"]
        breadth_pct = (holder_count / total_unique_pol) if total_unique_pol else 0.0

        holdings.append(
            {
                "ticker": t,
                "holderCount": holder_count,
                "demHolders": dem_count,
                "repHolders": rep_count,
                "breadthPct": round(breadth_pct * 100.0, 2),
                "holdingsScore": holdings_score(holder_count, dem_count, rep_count),
                "tradeCount": int(cur["tradeCount"]),
                "lastActivity": iso_date_only(last_seen),
            }
        )

    holdings.sort(key=lambda x: (-int(x["holderCount"]), -int(x["holdingsScore"]), x["ticker"]))
    top_holdings = holdings[: int(top_n or 30)]

    return {
        "date": now.date().isoformat(),
        "windowDays": window_days,
        "windowStart": since.date().isoformat(),
        "windowEnd": now.date().isoformat(),
        "breadth": {"uniquePoliticians": total_unique_pol, "uniqueTickers": len(holdings)},
        "topHoldings": top_holdings,
        "holdings": holdings,
        "note": (
            "Holdings are a proxy computed from disclosed congress trading activity: "
            "holderCount = unique members with activity in ticker within the window."
        ),
        "scoring": {
            "holdingsScoreMeaning": "0-100. Mostly breadth (holderCount/uniquePoliticians), with a bipartisan boost when both parties participate.",
            "maxScore": 100,
        },
    }


# --------------------------
# Market Entry Index (Stooq primary, Yahoo fallback)
# --------------------------
def _parse_date_yyyy_mm_dd(s: str) -> Optional[datetime]:
    try:
        return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _stooq_daily_close(symbol: str, lookback_days: int = 260) -> List[Tuple[datetime, float]]:
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    r = _SESSION.get(url, timeout=(8, 20))
    if r.status_code != 200:
        raise RuntimeError(f"Stooq HTTP {r.status_code}")

    lines = [ln.strip() for ln in r.text.splitlines() if ln.strip()]
    if len(lines) < 5:
        return []

    header = lines[0].lower().split(",")
    if "date" not in header or "close" not in header:
        return []

    date_i = header.index("date")
    close_i = header.index("close")

    out: List[Tuple[datetime, float]] = []
    for ln in lines[1:]:
        parts = ln.split(",")
        if len(parts) <= max(date_i, close_i):
            continue
        dt = _parse_date_yyyy_mm_dd(parts[date_i])
        if not dt:
            continue
        try:
            cl = float(parts[close_i])
        except Exception:
            continue
        out.append((dt, cl))

    out.sort(key=lambda x: x[0])
    if lookback_days and len(out) > lookback_days:
        out = out[-lookback_days:]
    return out


def _yahoo_daily_close(ticker: str, lookback_days: int = 260) -> List[Tuple[datetime, float]]:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"range": "2y", "interval": "1d"}
    r = _SESSION.get(url, params=params, timeout=(8, 20))
    if r.status_code != 200:
        raise RuntimeError(f"Yahoo HTTP {r.status_code}")

    j = r.json()
    result = (((j or {}).get("chart") or {}).get("result") or [])
    if not result:
        return []

    r0 = result[0]
    ts = r0.get("timestamp") or []
    ind = ((r0.get("indicators") or {}).get("quote") or [])
    if not ind:
        return []

    closes = ind[0].get("close") or []
    out: List[Tuple[datetime, float]] = []
    for t, cl in zip(ts, closes):
        if cl is None:
            continue
        try:
            dt = datetime.fromtimestamp(int(t), tz=timezone.utc)
            out.append((dt, float(cl)))
        except Exception:
            continue

    out.sort(key=lambda x: x[0])
    if lookback_days and len(out) > lookback_days:
        out = out[-lookback_days:]
    return out


def _daily_close_with_fallback(stooq_symbol: str, yahoo_symbol: str, lookback_days: int = 260) -> List[Tuple[datetime, float]]:
    try:
        data = _stooq_daily_close(stooq_symbol, lookback_days=lookback_days)
        if data:
            return data
    except Exception:
        pass
    return _yahoo_daily_close(yahoo_symbol, lookback_days=lookback_days)


def _sma(vals: List[float], n: int) -> Optional[float]:
    if n <= 0 or len(vals) < n:
        return None
    return sum(vals[-n:]) / n


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


@app.get("/market/entry")
def market_entry(window_days: int = Query(default=365, ge=30, le=365)):
    """
    Entry score 0-100 using:
      - SPY trend: SMA50 vs SMA200 and price vs SMA200
      - VIX: lower VIX gives higher score
    Stooq is primary, Yahoo is fallback.

    Returns:
      {
        "date":"YYYY-MM-DD",
        "score": int,
        "regime":"RISK-ON|NEUTRAL|RISK-OFF",
        "signal":"ACCUMULATE|ACCUMULATE SLOWLY|WAIT / SMALL DCA",
        "notes":"...",
        "components": { spxTrend, vix, breadth, credit, rates, buffettProxy }
      }
    """
    now = datetime.now(timezone.utc)

    try:
        spy = _daily_close_with_fallback("spy.us", "SPY", lookback_days=260)
        vix = _daily_close_with_fallback("^vix", "^VIX", lookback_days=260)
    except Exception as e:
        return {
            "date": now.date().isoformat(),
            "score": 0,
            "regime": "NEUTRAL",
            "signal": "DATA UNAVAILABLE",
            "notes": f"Market data fetch failed: {str(e)}",
            "components": {
                "spxTrend": 0.5,
                "vix": 0.5,
                "breadth": 0.5,
                "credit": 0.5,
                "rates": 0.5,
                "buffettProxy": 0.5,
            },
        }

    if len(spy) < 210 or len(vix) < 30:
        return {
            "date": now.date().isoformat(),
            "score": 0,
            "regime": "NEUTRAL",
            "signal": "INSUFFICIENT DATA",
            "notes": "Not enough history returned for SPY/VIX.",
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

    return {
        "date": now.date().isoformat(),
        "score": score,
        "regime": regime,
        "signal": signal,
        "notes": f"SPY={price:.2f} SMA50={sma50:.2f} SMA200={sma200:.2f} VIX={v:.2f}",
        "components": {
            "spxTrend": float(spx_trend_01),
            "vix": float(vix_01),
            "breadth": float(breadth_01),
            "credit": float(credit_01),
            "rates": float(rates_01),
            "buffettProxy": float(buffett_01),
        },
    }
