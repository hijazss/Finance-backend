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

# SEC requires identifying User-Agent (include contact email)
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "FinanceApp/1.0 (contact: hijazss@gmail.com)")
SEC_TIMEOUT = 20

_party_re = re.compile(r"/\s*([DR])\b", re.IGNORECASE)

# --------------------------
# Generic helpers
# --------------------------

def pick_first(row: dict, keys: List[str], default=""):
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return default


def parse_dt_any(v: Any) -> Optional[datetime]:
    """
    Robust date parsing for Quiver rows:
    supports ISO, 'YYYY-MM-DD', 'YYYY-MM-DDTHH:MM:SS', and common US formats like 'MM/DD/YYYY'.
    """
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


def interleave(items: List[dict], limit: int = 120) -> List[dict]:
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
        "funds": [],
        "lastFiledAt": last,
        "strength": kind,
        "chamber": x.get("chamber", ""),
        "amountRange": x.get("amountRange", ""),
        "traded": x.get("traded", ""),
        "filed": x.get("filed", ""),
    }


# --------------------------
# SEC EDGAR helpers
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
    for i in range(min(len(forms), len(accs), len(primary_docs), len(filed_dates))):
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
    """
    Parse Form 4 XML into per-transaction rows with:
    - issuerName, issuerTradingSymbol
    - transactionDate
    - code P/S
    - shares, price
    """
    out: List[dict] = []
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return out

    def txt(path_endswith: str) -> str:
        for node in root.iter():
            if node.tag.lower().endswith(path_endswith.lower()):
                return (node.text or "").strip()
        return ""

    issuer = ""
    symbol = ""
    # issuerName and issuerTradingSymbol appear under issuer
    for node in root.iter():
        if node.tag.lower().endswith("issuername"):
            issuer = (node.text or "").strip()
        if node.tag.lower().endswith("issuertradingsymbol"):
            symbol = (node.text or "").strip().upper()

    # Non-derivative transactions
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
            elif t.endswith("transactiondate"):
                # within transactionDate there is a <value>
                pass
            elif t.endswith("value") and node.text:
                # try to map by parent tag name
                parent = node.getparent() if hasattr(node, "getparent") else None

        # More reliable: just search within tx for known endpoints
        for node in tx.iter():
            t = node.tag.lower()
            if t.endswith("transactiondate"):
                # child value
                for c in node.iter():
                    if c.tag.lower().endswith("value"):
                        date = (c.text or "").strip()
            if t.endswith("transactionshares"):
                for c in node.iter():
                    if c.tag.lower().endswith("value"):
                        shares = (c.text or "").strip()
            if t.endswith("transactionpricepershare"):
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

    # If we found nothing above, fall back to a simpler “any transactionCode”
    if not out:
        buy = 0
        sell = 0
        for node in root.iter():
            if node.tag.lower().endswith("transactioncode"):
                c = (node.text or "").strip().upper()
                if c == "P":
                    buy += 1
                elif c == "S":
                    sell += 1
        if buy or sell:
            out.append({"issuerName": issuer, "ticker": symbol, "code": "MIXED", "buyTxCount": buy, "sellTxCount": sell})

    return out


def parse_13f_info_table(xml_text: str) -> List[dict]:
    """
    Parse 13F 'information table' XML to get holdings.
    Many 13Fs provide an XML info table containing:
      - nameOfIssuer, cusip, value, sshPrnamt, sshPrnamtType, putCall, investmentDiscretion, votingAuthority
    We'll return a list of holdings.
    """
    out: List[dict] = []
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return out

    # Information table entries often use <infoTable> nodes
    for node in root.iter():
        if node.tag.lower().endswith("infotable"):
            rec = {
                "nameOfIssuer": "",
                "cusip": "",
                "value": "",
                "sshPrnamt": "",
            }
            for c in node.iter():
                t = c.tag.lower()
                if t.endswith("nameofissuer"):
                    rec["nameOfIssuer"] = (c.text or "").strip()
                elif t.endswith("cusip"):
                    rec["cusip"] = (c.text or "").strip()
                elif t.endswith("value"):
                    rec["value"] = (c.text or "").strip()
                elif t.endswith("sshprnamt"):
                    rec["sshPrnamt"] = (c.text or "").strip()
            if rec["nameOfIssuer"] or rec["cusip"]:
                out.append(rec)

    # Sort by value desc if numeric
    def val_num(x):
        try:
            return float(str(x.get("value", "")).replace(",", ""))
        except Exception:
            return 0.0

    out.sort(key=val_num, reverse=True)
    return out


# --------------------------
# Prices for performance
# --------------------------

def stooq_symbol(ticker: str) -> str:
    # Stooq uses lower case and ".us" for US equities
    t = (ticker or "").strip().lower()
    if not t:
        return ""
    return f"{t}.us"


def fetch_close_stooq(ticker: str, start: datetime, end: datetime) -> List[Tuple[datetime, float]]:
    """
    Fetch daily closes from stooq as CSV.
    Returns list of (date, close).
    """
    sym = stooq_symbol(ticker)
    if not sym:
        return []
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    r = requests.get(url, timeout=15)
    if r.status_code != 200 or "Date,Open" not in r.text:
        return []

    lines = r.text.strip().splitlines()
    # Date,Open,High,Low,Close,Volume
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
    # choose first close on/after target; else last before
    for d, c in series:
        if d >= target:
            return c
    return series[-1][1]


# --------------------------
# Root
# --------------------------

@app.get("/")
def root():
    return {"status": "ok"}


# --------------------------
# Endpoint: Congress rolling window (30 or 180)
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
            "cryptoDirect": [],
            "cryptoLinked": [],
        }

    try:
        rows = df.to_dict(orient="records")
    except Exception:
        rows = list(df)

    buys: List[dict] = []
    sells: List[dict] = []
    crypto_direct: List[dict] = []
    crypto_linked: List[dict] = []

    # simple crypto detection from description
    crypto_words = ["bitcoin", "btc", "ethereum", "eth", "solana", "sol", "dogecoin", "doge", "litecoin", "ltc", "xrp", "ripple", "avax", "polygon", "matic", "bnb", "binance", "cardano", "ada"]

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
        desc_l = desc.lower()

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

        # collect congress equity-like trades
        if ticker:
            if kind == "BUY":
                buys.append(item)
            else:
                sells.append(item)

            # crypto-linked tickers bucket (trusts/ETFs often show here)
            if ticker in ["GBTC", "ETHE", "BTC", "IBIT", "FBTC", "ARKB", "BITO", "BTF", "HODL", "BTCO", "ETHA", "ETH", "MSTR", "COIN"]:
                crypto_linked.append(item)

        # attempt to find direct crypto mentions even when no ticker
        if desc and any(w in desc_l for w in crypto_words):
            # normalize a "symbol" for UI
            sym = ""
            if "bitcoin" in desc_l or "btc" in desc_l:
                sym = "BTC"
            elif "ethereum" in desc_l or "eth" in desc_l:
                sym = "ETH"
            elif "solana" in desc_l or "sol" in desc_l:
                sym = "SOL"
            elif "dogecoin" in desc_l or "doge" in desc_l:
                sym = "DOGE"
            elif "xrp" in desc_l or "ripple" in desc_l:
                sym = "XRP"
            elif "cardano" in desc_l or "ada" in desc_l:
                sym = "ADA"

            crypto_direct.append({
                "crypto": sym or "CRYPTO",
                "description": desc,
                "party": party,
                "politician": pol,
                "chamber": chamber,
                "amountRange": amount,
                "traded": iso_date_only(traded_dt),
                "filed": iso_date_only(filed_dt),
                "kind": kind,
            })

    buys.sort(key=lambda x: x["best_dt"], reverse=True)
    sells.sort(key=lambda x: x["best_dt"], reverse=True)

    buy_cards = [to_card(x, "BUY") for x in interleave(buys, 120)]
    sell_cards = [to_card(x, "SELL") for x in interleave(sells, 120)]

    # Convergence = overlapping tickers on BUY within window
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
            "funds": [],
            "lastFiledAt": iso_date_only(latest_dt),
            "strength": "OVERLAP",
        })

    # sort convergence by total overlap participation desc then ticker
    overlap_cards.sort(key=lambda x: (-(x["demBuyers"] + x["repBuyers"]), x["ticker"]))

    # compact crypto outputs
    crypto_direct.sort(key=lambda x: (x.get("filed") or ""), reverse=True)
    crypto_linked.sort(key=lambda x: x.get("best_dt") or datetime(1970,1,1,tzinfo=timezone.utc), reverse=True)

    return {
        "date": now.date().isoformat(),
        "windowDays": window_days,
        "windowStart": since.date().isoformat(),
        "windowEnd": now.date().isoformat(),
        "convergence": overlap_cards[:25],
        "politicianBuys": buy_cards,
        "politicianSells": sell_cards,
        "cryptoDirect": crypto_direct[:200],
        "cryptoLinked": crypto_linked[:200],
    }


# --------------------------
# Endpoint: 2-year performance (with horizon window)
# --------------------------

@app.get("/report/performance-2y")
def performance_2y(horizon_days: int = Query(30, ge=7, le=365)):
    """
    Computes a simple signal score per politician from the past 2 years:
      - For BUY: forward return from traded date to traded+horizon_days
      - For SELL: negative forward return (selling before upside is "bad", before downside is "good")
    Equal-weight per trade (no sizing), requires price data via Stooq.
    """
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
        if not best_dt:
            continue
        if best_dt < since2y or best_dt > now:
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

        pol = str(pick_first(r, ["Politician", "politician", "Representative", "Senator", "Name", "name"], "")).strip()
        chamber = str(pick_first(r, ["Chamber", "chamber", "Office", "office"], "")).strip()

        traded_dt = parse_dt_any(pick_first(r, ["Traded", "traded", "TransactionDate", "transaction_date"], ""))
        filed_dt = parse_dt_any(pick_first(r, ["Filed", "filed", "ReportDate", "report_date", "Date", "date"], ""))

        # traded date is required for pricing; fallback to filed if missing
        base_dt = traded_dt or filed_dt
        if not base_dt:
            continue

        base_dt = base_dt.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        end_dt = base_dt + timedelta(days=horizon_days)

        if end_dt > now:
            continue

        tickers_needed.add(ticker.upper())
        trades.append({
            "politician": pol or "Unknown",
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

    # Fetch price series per ticker once (bounded by 2y)
    price_cache: Dict[str, List[Tuple[datetime, float]]] = {}
    min_start = since2y.replace(hour=0, minute=0, second=0, microsecond=0)
    max_end = now.replace(hour=0, minute=0, second=0, microsecond=0)

    for t in sorted(tickers_needed):
        try:
            price_cache[t] = fetch_close_stooq(t, min_start, max_end)
        except Exception:
            price_cache[t] = []

    # Score per politician
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
            "avgReturn": 0.0,
            "sumReturn": 0.0,
            "sample": [],
        }
        cur["tradeCount"] += 1
        cur["sumReturn"] += ret
        cur["avgReturn"] = cur["sumReturn"] / max(cur["tradeCount"], 1)

        # keep a small sample
        if len(cur["sample"]) < 5:
            cur["sample"].append({
                "ticker": t,
                "kind": tr["kind"],
                "traded": tr["traded"],
                "filed": tr["filed"],
                "horizonDays": horizon_days,
                "score": round(ret * 100, 2),
            })

        stats[key] = cur

    leaders = list(stats.values())
    leaders.sort(key=lambda x: (-(x["avgReturn"]), -x["tradeCount"], x["name"]))

    # format returns in percent for UI
    for x in leaders:
        x["avgReturnPct"] = round(x["avgReturn"] * 100, 2)

    return {
        "date": now.date().isoformat(),
        "windowStart": since2y.date().isoformat(),
        "windowEnd": now.date().isoformat(),
        "horizonDays": horizon_days,
        "leaders": leaders[:50],
        "note": "Performance is a simple equal-weight score from 2y of trades. BUY uses forward return; SELL uses negative forward return over the selected horizon.",
    }


# --------------------------
# Endpoint: Public leaders (SEC EDGAR) enriched
# --------------------------

@app.get("/report/public-leaders")
def report_public_leaders():
    """
    Returns recent SEC filings for curated public figures/funds.
    - Form 4: parses XML to list tickers and buy/sell transactions.
    - 13F: attempts to parse info table and compare to previous 13F to infer increases/decreases by CUSIP.
    """
    now = datetime.now(timezone.utc)
    since365 = now - timedelta(days=365)

    leaders = [
        # Note: "Donald Trump" does not have a single universal investment CIK like a fund;
        # for SEC tracking we focus on entities with consistent SEC filer identity.
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
        last_13f_holdings = None
        prev_13f_holdings = None
        last_13f_date = None
        prev_13f_date = None

        # For 13F, grab the newest two filings within 365d so we can compute deltas
        if any(f.startswith("13F") for f in leader["forms"]):
            thirteen = [f for f in filings if str(f.get("form","")).startswith("13F")]
            if len(thirteen) >= 1:
                last_13f_date = thirteen[0].get("filingDate","")
                acc = thirteen[0]["accession"]
                primary = thirteen[0]["primaryDocument"]
                primary_url = build_primary_doc_url(cik, acc, primary)
                try:
                    xml_text = sec_get_text(primary_url)
                    last_13f_holdings = parse_13f_info_table(xml_text)
                except Exception:
                    last_13f_holdings = None
                time.sleep(0.2)

            if len(thirteen) >= 2:
                prev_13f_date = thirteen[1].get("filingDate","")
                acc = thirteen[1]["accession"]
                primary = thirteen[1]["primaryDocument"]
                primary_url = build_primary_doc_url(cik, acc, primary)
                try:
                    xml_text = sec_get_text(primary_url)
                    prev_13f_holdings = parse_13f_info_table(xml_text)
                except Exception:
                    prev_13f_holdings = None
                time.sleep(0.2)

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

                    # summarize
                    buys = [t for t in txs if t.get("code") == "P"]
                    sells = [t for t in txs if t.get("code") == "S"]

                    label = "BUY" if buys and not sells else "SELL" if sells and not buys else "MIXED" if buys and sells else "FORM4"
                    details = {
                        "transactions": txs[:20],
                        "buyCount": len(buys),
                        "sellCount": len(sells),
                    }
                except Exception:
                    label = "FORM4"

            if form.startswith("13F"):
                label = "13F FILED"
                # Provide top holdings and inferred deltas if we have both
                holdings_top = (last_13f_holdings or [])[:12] if last_13f_holdings else []
                deltas = []
                if last_13f_holdings and prev_13f_holdings:
                    prev_map = {h.get("cusip",""): h for h in prev_13f_holdings if h.get("cusip")}
                    for h in last_13f_holdings[:200]:
                        cusip = h.get("cusip","")
                        if not cusip:
                            continue
                        v_now = float(str(h.get("value","0")).replace(",","") or 0)
                        v_prev = float(str((prev_map.get(cusip) or {}).get("value","0")).replace(",","") or 0)
                        dv = v_now - v_prev
                        if abs(dv) > 0:
                            deltas.append({
                                "nameOfIssuer": h.get("nameOfIssuer",""),
                                "cusip": cusip,
                                "valueNow": v_now,
                                "valuePrev": v_prev,
                                "deltaValue": dv,
                            })
                    deltas.sort(key=lambda x: abs(x.get("deltaValue",0)), reverse=True)
                    deltas = deltas[:12]

                details = {
                    "latest13FDate": last_13f_date,
                    "prev13FDate": prev_13f_date,
                    "topHoldings": holdings_top,
                    "topDeltas": deltas,
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
        "note": "Form 4 shows actual issuer tickers + buy/sell transactions. 13F shows holdings and (when possible) deltas vs the prior 13F filing within the last year.",
    }
