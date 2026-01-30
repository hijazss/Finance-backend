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
# Crypto detection helpers
# --------------------------

CRYPTO_MAP = {
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "eth", "ether"],
    "SOL": ["solana", "sol"],
    "LINK": ["chainlink", "link"],
    "XRP": ["xrp", "ripple"],
    "ADA": ["cardano", "ada"],
    "DOGE": ["dogecoin", "doge"],
    "AVAX": ["avalanche", "avax"],
    "MATIC": ["polygon", "matic"],
    "BNB": ["bnb", "binance coin", "binance"],
}

CRYPTO_LINKED_TICKERS = set([
    "GBTC","ETHE","BITO","IBIT","FBTC","ARKB","BTCO","HODL",
    "ETHA","ETHW","COIN","MSTR","RIOT","MARA","HUT","CLSK"
])


def detect_crypto_from_text(desc: str) -> List[str]:
    if not desc:
        return []
    d = desc.lower()
    hits = []
    for sym, words in CRYPTO_MAP.items():
        for w in words:
            if w in d:
                hits.append(sym)
                break
    # de-dupe, stable order by CRYPTO_MAP insertion
    out = []
    for sym in CRYPTO_MAP.keys():
        if sym in hits:
            out.append(sym)
    return out


# --------------------------
# SEC EDGAR helpers (leaders)
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

    # Collect transaction rows
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


def aggregate_form4_by_ticker(txs: List[dict]) -> Tuple[List[dict], List[dict]]:
    buy_map: Dict[str, int] = {}
    sell_map: Dict[str, int] = {}
    last_date: Dict[str, str] = {}

    for t in txs:
        sym = (t.get("ticker") or "").upper()
        code = (t.get("code") or "").upper()
        dt = (t.get("transactionDate") or "").strip()
        if not sym:
            continue

        if dt:
            last_date[sym] = max(last_date.get(sym, ""), dt)

        if code == "P":
            buy_map[sym] = buy_map.get(sym, 0) + 1
        elif code == "S":
            sell_map[sym] = sell_map.get(sym, 0) + 1

    buys = [{"ticker": k, "count": v, "lastTrade": last_date.get(k, "")} for k, v in buy_map.items()]
    sells = [{"ticker": k, "count": v, "lastTrade": last_date.get(k, "")} for k, v in sell_map.items()]

    buys.sort(key=lambda x: (-x["count"], x["ticker"]))
    sells.sort(key=lambda x: (-x["count"], x["ticker"]))

    return buys, sells


# --------------------------
# Endpoint: Congress rolling window (30, 180, 365)
# --------------------------

@app.get("/report/today")
def report_today(window_days: int = Query(30, ge=1, le=365)):
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
            "convergence": [],
            "politicianBuys": [],
            "politicianSells": [],
            "crypto": {"buys": [], "sells": [], "raw": []},
            "ethers": {"buys": [], "sells": [], "raw": []},
        }

    try:
        rows = df.to_dict(orient="records")
    except Exception:
        rows = list(df)

    buys: List[dict] = []
    sells: List[dict] = []

    crypto_raw: List[dict] = []
    crypto_coin_counts = {"BUY": {}, "SELL": {}}

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

        # Normal equity trades bucket (only if ticker present)
        if ticker:
            if kind == "BUY":
                buys.append(item)
            else:
                sells.append(item)

        # Crypto detection: from either ticker (crypto-linked tickers) OR description (direct coin mentions)
        coins = detect_crypto_from_text(desc)
        is_linked = (ticker.upper() in CRYPTO_LINKED_TICKERS) if ticker else False

        if coins or is_linked:
            # If linked but no coin keyword, label as "CRYPTO-LINKED"
            tag_coins = coins if coins else (["CRYPTO-LINKED"] if is_linked else [])
            for c in tag_coins:
                m = crypto_coin_counts[kind]
                m[c] = m.get(c, 0) + 1

            crypto_raw.append({
                "kind": kind,
                "coins": tag_coins,
                "ticker": ticker,
                "description": desc,
                "party": party,
                "politician": pol,
                "chamber": chamber,
                "amountRange": amount,
                "traded": iso_date_only(traded_dt),
                "filed": iso_date_only(filed_dt),
            })

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

    # Build crypto summary lists
    def counts_to_list(m: Dict[str, int]) -> List[dict]:
        out = [{"symbol": k, "count": v} for k, v in m.items()]
        out.sort(key=lambda x: (-x["count"], x["symbol"]))
        return out

    crypto_summary = {
        "buys": counts_to_list(crypto_coin_counts["BUY"]),
        "sells": counts_to_list(crypto_coin_counts["SELL"]),
        "raw": crypto_raw[:250],
    }

    # ETH slice
    eth_raw = [r for r in crypto_raw if ("ETH" in r.get("coins", []) or "CRYPTO-LINKED" in r.get("coins", [])) and ("eth" in (r.get("description") or "").lower() or r.get("ticker","").upper() in {"ETHE","ETHA"})]
    eth_buy_ct: Dict[str, int] = {}
    eth_sell_ct: Dict[str, int] = {}
    for r in eth_raw:
        for c in r.get("coins", []) or ["ETH"]:
            if r["kind"] == "BUY":
                eth_buy_ct[c] = eth_buy_ct.get(c, 0) + 1
            else:
                eth_sell_ct[c] = eth_sell_ct.get(c, 0) + 1

    ethers_summary = {
        "buys": counts_to_list(eth_buy_ct),
        "sells": counts_to_list(eth_sell_ct),
        "raw": eth_raw[:250],
    }

    return {
        "date": now.date().isoformat(),
        "windowDays": window_days,
        "windowStart": since.date().isoformat(),
        "windowEnd": now.date().isoformat(),
        "convergence": overlap_cards[:25],
        "politicianBuys": buy_cards,
        "politicianSells": sell_cards,
        "crypto": crypto_summary,
        "ethers": ethers_summary,
    }


@app.get("/report/crypto")
def report_crypto(window_days: int = Query(30, ge=1, le=365)):
    payload = report_today(window_days=window_days)
    return {
        "date": payload["date"],
        "windowDays": payload["windowDays"],
        "windowStart": payload["windowStart"],
        "windowEnd": payload["windowEnd"],
        "crypto": payload["crypto"],
    }


@app.get("/report/ethers")
def report_ethers(window_days: int = Query(30, ge=1, le=365)):
    payload = report_today(window_days=window_days)
    return {
        "date": payload["date"],
        "windowDays": payload["windowDays"],
        "windowStart": payload["windowStart"],
        "windowEnd": payload["windowEnd"],
        "ethers": payload["ethers"],
    }


# --------------------------
# Endpoint: 2-year performance
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
def performance_2y(horizon_days: int = Query(30, ge=7, le=365)):
    if not QUIVER_TOKEN:
        raise HTTPException(500, "QUIVER_TOKEN missing")

    now = datetime.now(timezone.utc)
    since2y = now - timedelta(days=730)

    q = quiverquant.quiver(QUIVER_TOKEN)
    df = q.congress_trading()

    if df is None or len(df) == 0:
        return {"date": now.date().isoformat(), "horizonDays": horizon_days, "leaders": []}

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
        end_dt = base_dt + timedelta(days=horizon_days)
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
        return {"date": now.date().isoformat(), "horizonDays": horizon_days, "leaders": []}

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
        "horizonDays": horizon_days,
        "leaders": leaders[:60],
        "note": "2Y performance score. BUY uses forward return. SELL uses negative forward return over the selected horizon.",
    }


# --------------------------
# Endpoint: Public leaders (SEC EDGAR)
# --------------------------

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
                    buys_by_ticker, sells_by_ticker = aggregate_form4_by_ticker(txs)

                    buy_ct = sum(x["count"] for x in buys_by_ticker)
                    sell_ct = sum(x["count"] for x in sells_by_ticker)

                    if buy_ct and not sell_ct:
                        label = "BUY"
                    elif sell_ct and not buy_ct:
                        label = "SELL"
                    elif buy_ct and sell_ct:
                        label = "MIXED"
                    else:
                        label = "FORM4"

                    details = {
                        "buyTickers": buys_by_ticker[:15],
                        "sellTickers": sells_by_ticker[:15],
                        "buyCount": buy_ct,
                        "sellCount": sell_ct,
                    }
                except Exception:
                    label = "FORM4"

            if form.startswith("13F"):
                label = "13F FILED"
                details = {
                    "note": "13F is quarterly. This endpoint shows filing events. Computing true buys/sells requires comparing consecutive 13F tables.",
                }

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

        results.append({
            "name": leader["name"],
            "cik": cik10(cik),
            "events": events,
        })

    return {
        "date": now.date().isoformat(),
        "windowDays": 365,
        "windowStart": since365.date().isoformat(),
        "windowEnd": now.date().isoformat(),
        "leaders": results,
        "note": "Form 4 includes buy/sell tickers aggregated by transaction count. 13F is shown as filing events.",
    }
