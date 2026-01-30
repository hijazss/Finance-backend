import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import quiverquant

app = FastAPI(title="Finance Signals Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://hijazss.github.io"],
    allow_methods=["*"],
    allow_headers=["*"],
)

QUIVER_TOKEN = os.getenv("QUIVER_TOKEN")

SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "FinanceApp/1.0 (contact: hijazss@gmail.com)")
SEC_TIMEOUT = 20

_party_re = re.compile(r"/\s*([DR])\b", re.IGNORECASE)

# --------------------------
# Root
# --------------------------

@app.get("/")
def root():
    return {"status": "ok"}


# --------------------------
# Generic helpers
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
        if 1 <= len(first) <= 10 and first.replace(".", "").replace("-", "").isalnum():
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
# Goal: catch both
# - crypto ETFs/proxies via ticker (IBIT, FBTC, GBTC, ETHE, etc.)
# - "direct crypto reporting" via text in asset description/name fields (Bitcoin, Ethereum, SOL, LINK, etc.)
#
# We do stricter token matching to reduce false positives.

TOP_COINS = [
    "BTC", "ETH", "SOL", "LINK", "XRP", "ADA", "DOGE", "AVAX", "MATIC", "BNB"
]

# Words/aliases per coin. We add boundaries for symbols to avoid partial matches.
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

# Known crypto-related tickers (ETFs, trusts, miners, exchanges, proxies)
CRYPTO_RELATED_TICKERS = set([
    # spot BTC ETFs and products
    "IBIT", "FBTC", "ARKB", "BITB", "BTCO", "HODL", "GBTC", "BITO",
    # ETH products
    "ETHE", "ETHA",
    # proxies / companies
    "COIN", "MSTR", "RIOT", "MARA", "HUT", "CLSK",
])

# Map tickers to coin tags where reasonable
TICKER_TO_COINS: Dict[str, List[str]] = {
    "IBIT": ["BTC"], "FBTC": ["BTC"], "ARKB": ["BTC"], "BITB": ["BTC"], "BTCO": ["BTC"], "HODL": ["BTC"],
    "GBTC": ["BTC"], "BITO": ["BTC"],
    "ETHE": ["ETH"], "ETHA": ["ETH"],
    # proxies are crypto-linked rather than single coin, keep as CRYPTO-LINKED unless text adds coins
    "COIN": ["CRYPTO-LINKED"], "MSTR": ["BTC", "CRYPTO-LINKED"], "RIOT": ["BTC", "CRYPTO-LINKED"],
    "MARA": ["BTC", "CRYPTO-LINKED"], "HUT": ["BTC", "CRYPTO-LINKED"], "CLSK": ["BTC", "CRYPTO-LINKED"],
}

# Precompile regex patterns
# For symbols like "SOL" we require word boundaries.
# For words like "bitcoin" we just search case-insensitive.
_CRYPTO_PATTERNS: Dict[str, List[re.Pattern]] = {}
for sym, words in CRYPTO_ALIASES.items():
    pats: List[re.Pattern] = []
    for w in words:
        # If it's an all-caps style symbol alias (btc, eth, sol, link, etc.), do word boundary.
        # Otherwise plain contains match is usually safe (bitcoin, ethereum, chainlink).
        if re.fullmatch(r"[a-z]{2,6}", w):
            pats.append(re.compile(rf"\b{re.escape(w)}\b", re.IGNORECASE))
        else:
            pats.append(re.compile(re.escape(w), re.IGNORECASE))
    _CRYPTO_PATTERNS[sym] = pats

# Some additional generic hints that often show up in disclosures
_GENERIC_CRYPTO_HINTS = [
    re.compile(r"\bcryptocurrency\b", re.IGNORECASE),
    re.compile(r"\bdigital asset\b", re.IGNORECASE),
    re.compile(r"\bvirtual currency\b", re.IGNORECASE),
]


def collect_crypto_text(row: dict) -> str:
    """
    Pull every likely text field from Quiver rows.
    If Quiver changes column names, this still catches most.
    """
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

    # stable unique
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
    """
    Returns:
      - is_crypto_related: bool
      - coins: list of coin tags
      - crypto_kind: "direct" | "etf_or_proxy" | "hint_only"
    """
    t = (ticker or "").upper().strip()
    coins_from_text = detect_coins(text)
    is_related_ticker = bool(t and t in CRYPTO_RELATED_TICKERS)
    coins_from_ticker = TICKER_TO_COINS.get(t, []) if is_related_ticker else []

    coins = []
    # Merge, but keep TOP_COINS ordering and then CRYPTO-LINKED
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
# Endpoint: Congress rolling window (30/180/365)
# Accept both window_days and horizon_days so your frontend works.
# --------------------------

@app.get("/report/today")
def report_today(
    window_days: Optional[int] = Query(default=None, ge=1, le=365),
    horizon_days: Optional[int] = Query(default=None, ge=1, le=365),
):
    if not QUIVER_TOKEN:
        raise HTTPException(500, "QUIVER_TOKEN missing")

    # Frontend uses horizon_days, backend used window_days. Support both.
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

    # crypto details
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

        # Normal equity-like trades bucket only if ticker exists
        if ticker:
            if kind == "BUY":
                buys.append(item)
            else:
                sells.append(item)

        # Crypto detection pass uses full text
        text_blob = collect_crypto_text(r)
        is_crypto, coins, crypto_kind = classify_crypto_trade(ticker, text_blob)

        if is_crypto:
            # count each coin tag
            target_counts = crypto_counts_buy if kind == "BUY" else crypto_counts_sell
            for c in coins if coins else ["CRYPTO"]:
                target_counts[c] = target_counts.get(c, 0) + 1

            rec = {
                "kind": kind,
                "coins": coins if coins else ["CRYPTO"],
                "cryptoKind": crypto_kind,  # direct | etf_or_proxy | hint_only
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

    # Convergence = overlap tickers on BUY inside window
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

    # Crypto sorts
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
# 2-year performance (kept same, but accept both names)
# --------------------------

def stooq_symbol(ticker: str) -> str:
    t = (ticker or "").strip().lower()
    return f"{t}.us" if t else ""


def fetch_close_stooq(ticker: str, start: datetime, end: datetime) -> List[Tuple[datetime, float]]:
    sym = stooq_symbol(ticker)
    if not sym:
        return []
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    r = requests.get(url, timeout=15)
    if r.status_code != 200 or "Date,Open" not in r.text:
        return []
    lines = r.text.strip().splitlines()
    out = []
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) < 5:
            continue
        d = parse_dt_any(parts[0])
        if not d:
            continue
        d = d.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        if d < start or d > end:
            continue
        try:
            close = float(parts[4])
        except Exception:
            continue
        out.append((d, close))
    out.sort(key=lambda x: x[0])
    return out


def nearest_close(series: List[Tuple[datetime, float]], target: datetime) -> Optional[float]:
    if not series:
        return None
    for d, c in series:
        if d >= target:
            return c
    return series[-1][1]


@app.get("/report/performance-2y")
def performance_2y(
    horizon_days: Optional[int] = Query(default=None, ge=7, le=365),
    window_days: Optional[int] = Query(default=None, ge=7, le=365),
):
    if not QUIVER_TOKEN:
        raise HTTPException(500, "QUIVER_TOKEN missing")

    # Support both params, frontend uses horizon_days
    days = horizon_days if horizon_days is not None else window_days if window_days is not None else 30

    now = datetime.now(timezone.utc)
    since2y = now - timedelta(days=730)

    q = quiverquant.quiver(QUIVER_TOKEN)
    df = q.congress_trading()

    if df is None or len(df) == 0:
        return {"date": now.date().isoformat(), "horizonDays": days, "leaders": []}

    try:
        rows = df.to_dict(orient="records")
    except Exception:
        rows = list(df)

    trades: List[dict] = []
    tickers_needed = set()

    for r in rows:
        best_dt = row_best_dt(r)
        if not best_dt or best_dt < since2y or best_dt > now:
            continue

        party = norm_party_from_any(r)
        if not party:
            continue

        tx = tx_text(r)
        kind = "BUY" if is_buy(tx) else "SELL" if is_sell(tx) else ""
        if not kind:
            continue

        ticker = norm_ticker(r)
        if not ticker:
            continue

        pol = str(pick_first(r, ["Politician", "politician", "Representative", "Senator", "Name", "name"], "")).strip() or "Unknown"
        chamber = str(pick_first(r, ["Chamber", "chamber", "Office", "office"], "")).strip()

        traded_dt = parse_dt_any(pick_first(r, ["Traded", "traded", "TransactionDate", "transaction_date"], ""))
        filed_dt = parse_dt_any(pick_first(r, ["Filed", "filed", "ReportDate", "report_date", "Date", "date"], ""))

        base_dt = traded_dt or filed_dt
        if not base_dt:
            continue

        base_dt = base_dt.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        end_dt = base_dt + timedelta(days=days)
        if end_dt > now:
            continue

        tickers_needed.add(ticker.upper())
        trades.append({
            "politician": pol,
            "party": party,
            "chamber": chamber,
            "ticker": ticker.upper(),
            "kind": kind,
            "base_dt": base_dt,
            "end_dt": end_dt,
            "traded": iso_date_only(traded_dt),
            "filed": iso_date_only(filed_dt),
        })

    if not trades:
        return {"date": now.date().isoformat(), "horizonDays": days, "leaders": []}

    price_cache: Dict[str, List[Tuple[datetime, float]]] = {}
    min_start = since2y.replace(hour=0, minute=0, second=0, microsecond=0)
    max_end = now.replace(hour=0, minute=0, second=0, microsecond=0)

    for t in sorted(tickers_needed):
        try:
            price_cache[t] = fetch_close_stooq(t, min_start, max_end)
        except Exception:
            price_cache[t] = []

    stats: Dict[str, dict] = {}
    for tr in trades:
        t = tr["ticker"]
        series = price_cache.get(t) or []
        p0 = nearest_close(series, tr["base_dt"])
        p1 = nearest_close(series, tr["end_dt"])
        if p0 is None or p1 is None or p0 <= 0:
            continue

        ret = (p1 - p0) / p0
        if tr["kind"] == "SELL":
            ret = -ret

        key = tr["politician"]
        cur = stats.get(key) or {
            "name": key,
            "party": tr["party"],
            "chamber": tr["chamber"],
            "tradeCount": 0,
            "sumReturn": 0.0,
            "avgReturn": 0.0,
            "sample": [],
        }
        cur["tradeCount"] += 1
        cur["sumReturn"] += ret
        cur["avgReturn"] = cur["sumReturn"] / max(cur["tradeCount"], 1)

        if len(cur["sample"]) < 5:
            cur["sample"].append({
                "ticker": t,
                "kind": tr["kind"],
                "traded": tr["traded"],
                "filed": tr["filed"],
                "scorePct": round(ret * 100, 2),
            })

        stats[key] = cur

    leaders = list(stats.values())
    leaders.sort(key=lambda x: (-(x["avgReturn"]), -x["tradeCount"], x["name"]))

    for x in leaders:
        x["avgReturnPct"] = round(x["avgReturn"] * 100, 2)

    return {
        "date": now.date().isoformat(),
        "windowStart": since2y.date().isoformat(),
        "windowEnd": now.date().isoformat(),
        "horizonDays": days,
        "leaders": leaders[:60],
        "note": "2Y performance score. BUY uses forward return. SELL uses negative forward return over the selected horizon.",
    }


# --------------------------
# SEC EDGAR endpoints (unchanged placeholders)
# Keeping them because you already had them, but they are not needed for option A.
# --------------------------

def sec_get_json(url: str) -> dict:
    headers = {"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate", "Host": "data.sec.gov"}
    r = requests.get(url, headers=headers, timeout=SEC_TIMEOUT)
    if r.status_code != 200:
        raise HTTPException(502, f"SEC fetch failed HTTP {r.status_code} for {url}")
    return r.json()


def sec_get_text(url: str) -> str:
    headers = {"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}
    r = requests.get(url, headers=headers, timeout=SEC_TIMEOUT)
    if r.status_code != 200:
        raise HTTPException(502, f"SEC fetch failed HTTP {r.status_code} for {url}")
    return r.text


def cik10(cik: str) -> str:
    s = re.sub(r"\D", "", str(cik).strip())
    return s.zfill(10)


def accession_no_dashes(acc: str) -> str:
    return (acc or "").replace("-", "").strip()


def build_filing_index_url(cik: str, accession: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_no_dashes(accession)}/{accession}-index.html"


def build_primary_doc_url(cik: str, accession: str, primary_doc: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_no_dashes(accession)}/{primary_doc}"


def pull_recent_filings_for_cik(cik: str) -> dict:
    url = f"https://data.sec.gov/submissions/CIK{cik10(cik)}.json"
    return sec_get_json(url)


def extract_latest_filings(submissions: dict, forms_allow: List[str], max_items: int = 20) -> List[dict]:
    filings = (submissions.get("filings") or {}).get("recent") or {}
    forms = filings.get("form") or []
    accs = filings.get("accessionNumber") or []
    primary_docs = filings.get("primaryDocument") or []
    filed_dates = filings.get("filingDate") or []

    out = []
    n = min(len(forms), len(accs), len(primary_docs), len(filed_dates))
    for i in range(n):
        form = str(forms[i])
        if form not in forms_allow:
            continue
        out.append({
            "form": form,
            "accession": str(accs[i]),
            "primaryDocument": str(primary_docs[i]),
            "filingDate": str(filed_dates[i]),
        })

    out.sort(key=lambda x: x.get("filingDate", ""), reverse=True)
    return out[:max_items]


def parse_form4_transactions(xml_text: str) -> List[dict]:
    out: List[dict] = []
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return out

    issuer = ""
    symbol = ""

    for node in root.iter():
        tag = node.tag.lower()
        if tag.endswith("issuername"):
            issuer = (node.text or "").strip()
        elif tag.endswith("issuertradingsymbol"):
            symbol = (node.text or "").strip().upper()

    for tx in root.iter():
        if not tx.tag.lower().endswith("nonderivativetransaction"):
            continue

        code = ""
        date = ""
        shares = ""
        price = ""

        for node in tx.iter():
            t = node.tag.lower()
            if t.endswith("transactioncode"):
                code = (node.text or "").strip().upper()

        for node in tx.iter():
            t = node.tag.lower()
            if t.endswith("transactiondate"):
                for c in node.iter():
                    if c.tag.lower().endswith("value"):
                        date = (c.text or "").strip()
            elif t.endswith("transactionshares"):
                for c in node.iter():
                    if c.tag.lower().endswith("value"):
                        shares = (c.text or "").strip()
            elif t.endswith("transactionpricepershare"):
                for c in node.iter():
                    if c.tag.lower().endswith("value"):
                        price = (c.text or "").strip()

        if code in ["P", "S"]:
            out.append({
                "issuerName": issuer,
                "ticker": symbol,
                "code": code,
                "transactionDate": date,
                "shares": shares,
                "price": price,
            })

    return out


@app.get("/report/public-leaders")
def report_public_leaders():
    now = datetime.now(timezone.utc)
    since365 = now - timedelta(days=365)

    leaders = [
        {"key": "musk", "name": "Elon Musk", "cik": "1494730", "forms": ["4", "4/A"]},
        {"key": "berkshire", "name": "Berkshire Hathaway", "cik": "1067983", "forms": ["13F-HR", "13F-HR/A"]},
        {"key": "bridgewater", "name": "Bridgewater Associates", "cik": "1350694", "forms": ["13F-HR", "13F-HR/A"]},
    ]

    results = []
    for leader in leaders:
        cik = leader["cik"]
        subs = pull_recent_filings_for_cik(cik)
        filings = extract_latest_filings(subs, leader["forms"], max_items=12)

        events = []
        for f in filings:
            filed_dt = parse_dt_any(f.get("filingDate"))
            if filed_dt is not None and filed_dt < since365:
                continue

            accession = f["accession"]
            primary_doc = f["primaryDocument"]
            form = f["form"]
            filing_date = f["filingDate"]

            index_url = build_filing_index_url(cik, accession)
            primary_url = build_primary_doc_url(cik, accession, primary_doc)

            label = "FILED"
            details = {}

            if form.startswith("4"):
                try:
                    xml_text = sec_get_text(primary_url)
                    txs = parse_form4_transactions(xml_text)
                    buys = [t for t in txs if (t.get("code") or "").upper() == "P"]
                    sells = [t for t in txs if (t.get("code") or "").upper() == "S"]
                    if buys and not sells:
                        label = "BUY"
                    elif sells and not buys:
                        label = "SELL"
                    elif buys and sells:
                        label = "MIXED"
                    else:
                        label = "FORM4"
                    details = {"transactions": txs[:20], "buyCount": len(buys), "sellCount": len(sells)}
                except Exception:
                    label = "FORM4"

            if form.startswith("13F"):
                label = "13F FILED"
                details = {"note": "13F is quarterly and not a direct trade feed."}

            events.append({
                "form": form,
                "label": label,
                "filingDate": filing_date,
                "indexUrl": index_url,
                "primaryUrl": primary_url,
                "details": details,
            })

            if len(events) >= 4:
                break

            time.sleep(0.2)

        results.append({"name": leader["name"], "cik": cik10(cik), "events": events})

    return {
        "date": now.date().isoformat(),
        "windowDays": 365,
        "windowStart": since365.date().isoformat(),
        "windowEnd": now.date().isoformat(),
        "leaders": results,
        "note": "This endpoint is separate from congress crypto extraction (Option A uses Quiver only).",
    }
