import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import quiverquant
import xml.etree.ElementTree as ET


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

WINDOW_30_DAYS = 30
WINDOW_180_DAYS = 180
WINDOW_365_DAYS = 365

_party_re = re.compile(r"/\s*([DR])\b", re.IGNORECASE)


@app.get("/")
def root():
    return {"status": "ok"}


# --------------------------
# Helpers: Quiver (Congress)
# --------------------------

def pick_first(row: dict, keys: List[str], default=""):
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return default


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
    ]
    for fmt in fmts:
        try:
            # Safe slice length for cases where string has extra parts
            cut = min(len(s), len(datetime.now().strftime(fmt)))
            dt = datetime.strptime(s[:cut], fmt)
            return dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue

    return None


def row_best_dt(row: dict) -> Optional[datetime]:
    traded = pick_first(row, ["Traded", "traded", "TransactionDate", "transaction_date"], "")
    filed = pick_first(row, ["Filed", "filed", "ReportDate", "report_date", "Date", "date"], "")
    return parse_dt_any(traded) or parse_dt_any(filed)


def iso_date_only(dt: Optional[datetime]) -> str:
    return dt.date().isoformat() if dt else ""


def interleave(items: List[dict], limit: int = 80) -> List[dict]:
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
        "ticker": x["ticker"],
        "companyName": who,
        "demBuyers": dem if kind == "BUY" else 0,
        "repBuyers": rep if kind == "BUY" else 0,
        "demSellers": dem if kind == "SELL" else 0,
        "repSellers": rep if kind == "SELL" else 0,
        "funds": [],
        "lastFiledAt": last,
        "strength": kind,
    }


def compute_congress_window(
    rows: List[dict],
    horizon_days: int,
    ticker_allowlist: Optional[set] = None,
) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=horizon_days)

    buys: List[dict] = []
    sells: List[dict] = []

    for r in rows:
        best_dt = row_best_dt(r)
        if best_dt is None:
            continue
        if best_dt < since or best_dt > now:
            continue

        ticker = norm_ticker(r)
        if not ticker:
            continue
        if ticker_allowlist is not None and ticker not in ticker_allowlist:
            continue

        party = norm_party_from_any(r)
        if not party:
            continue

        tx = tx_text(r)
        kind = "BUY" if is_buy(tx) else "SELL" if is_sell(tx) else ""
        if not kind:
            continue

        filed_dt = parse_dt_any(pick_first(r, ["Filed", "filed", "ReportDate", "report_date", "Date", "date"], ""))
        traded_dt = parse_dt_any(pick_first(r, ["Traded", "traded", "TransactionDate", "transaction_date"], ""))
        pol = str(pick_first(r, ["Politician", "politician", "Representative", "Senator", "Name", "name"], "")).strip()

        item = {
            "ticker": ticker,
            "party": party,
            "filed": iso_date_only(filed_dt),
            "traded": iso_date_only(traded_dt),
            "politician": pol,
            "best_dt": best_dt,
        }

        if kind == "BUY":
            buys.append(item)
        else:
            sells.append(item)

    buys.sort(key=lambda x: x["best_dt"], reverse=True)
    sells.sort(key=lambda x: x["best_dt"], reverse=True)

    buy_mixed = interleave(
        [{"ticker": x["ticker"], "party": x["party"], "filed": x["filed"], "traded": x["traded"], "politician": x["politician"]} for x in buys],
        120,
    )
    sell_mixed = interleave(
        [{"ticker": x["ticker"], "party": x["party"], "filed": x["filed"], "traded": x["traded"], "politician": x["politician"]} for x in sells],
        120,
    )

    buy_cards = [to_card(x, "BUY") for x in buy_mixed]
    sell_cards = [to_card(x, "SELL") for x in sell_mixed]

    # Convergence (overlap) is computed from ALL buys in the window, not the interleaved slice.
    dem_buy = [x for x in buys if x["party"] == "D"]
    rep_buy = [x for x in buys if x["party"] == "R"]
    overlap_tickers = sorted(set(x["ticker"] for x in dem_buy).intersection(set(x["ticker"] for x in rep_buy)))

    overlap_cards: List[dict] = []
    for t in overlap_tickers:
        dem_ct = sum(1 for x in dem_buy if x["ticker"] == t)
        rep_ct = sum(1 for x in rep_buy if x["ticker"] == t)

        latest_dt: Optional[datetime] = None
        for x in buys:
            if x["ticker"] == t and (latest_dt is None or x["best_dt"] > latest_dt):
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

    # Sort overlap by participation (descending), then ticker
    overlap_cards.sort(key=lambda x: (-(x["demBuyers"] + x["repBuyers"]), x["ticker"]))

    return {
        "date": now.date().isoformat(),
        "windowDays": horizon_days,
        "windowStart": since.date().isoformat(),
        "windowEnd": now.date().isoformat(),
        "convergence": overlap_cards[:25],
        "politicianBuys": buy_cards[:120],
        "politicianSells": sell_cards[:120],
    }


# --------------------------
# SEC EDGAR helpers (Leaders)
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
    s = str(cik).strip()
    s = re.sub(r"\D", "", s)
    return s.zfill(10)


def accession_no_dashes(acc: str) -> str:
    return (acc or "").replace("-", "").strip()


def build_filing_index_url(cik: str, accession: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_no_dashes(accession)}/{accession}-index.html"


def build_primary_doc_url(cik: str, accession: str, primary_doc: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_no_dashes(accession)}/{primary_doc}"


def parse_form4_buy_sell(xml_text: str) -> Tuple[int, int]:
    buy = 0
    sell = 0
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return (0, 0)

    for node in root.iter():
        if node.tag.lower().endswith("transactioncode"):
            code = (node.text or "").strip().upper()
            if code == "P":
                buy += 1
            elif code == "S":
                sell += 1

    return (buy, sell)


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


# --------------------------
# Endpoint: Congress main (rolling 30D)
# --------------------------

@app.get("/report/today")
def report_today():
    if not QUIVER_TOKEN:
        raise HTTPException(500, "QUIVER_TOKEN missing")

    q = quiverquant.quiver(QUIVER_TOKEN)
    df = q.congress_trading()
    if df is None or len(df) == 0:
        now = datetime.now(timezone.utc)
        since = now - timedelta(days=WINDOW_30_DAYS)
        return {
            "date": now.date().isoformat(),
            "windowDays": WINDOW_30_DAYS,
            "windowStart": since.date().isoformat(),
            "windowEnd": now.date().isoformat(),
            "convergence": [],
            "bipartisanTickers": [],
            "politicianBuys": [],
            "politicianSells": [],
            "fundSignals": [],
        }

    try:
        rows = df.to_dict(orient="records")
    except Exception:
        rows = list(df)

    payload = compute_congress_window(rows, WINDOW_30_DAYS)

    # Keep old key for your front-end fallback
    payload["bipartisanTickers"] = payload["politicianBuys"][:60]
    payload["fundSignals"] = []

    return payload


# --------------------------
# NEW Endpoint: Congress crypto (rolling 30/180/365)
# --------------------------

def crypto_allowlist() -> set:
    """
    Congress disclosures sometimes appear as:
    - Spot crypto tickers (BTC, ETH) depending on data normalization
    - Crypto ETFs/trusts: IBIT, FBTC, BITB, ARKB, GBTC, ETHE, BITO, etc.
    We include both. You can expand anytime.
    """
    return {
        # Spot-style tickers
        "BTC", "ETH", "SOL", "ADA", "XRP", "DOT", "AVAX", "LINK", "MATIC",

        # Common crypto funds / ETFs / trusts (examples, not exhaustive)
        "GBTC", "ETHE", "BITO",
        "IBIT", "FBTC", "ARKB", "BITB", "HODL", "BTCO", "BRRR",
        "EZBC", "DEFI",  # some crypto themed tickers may appear
    }


@app.get("/report/crypto")
def report_crypto(
    horizon_days: int = Query(30, ge=7, le=365),
):
    if not QUIVER_TOKEN:
        raise HTTPException(500, "QUIVER_TOKEN missing")

    if horizon_days not in (30, 180, 365):
        # Keep it strict so your UI matches expected toggles
        raise HTTPException(400, "horizon_days must be 30, 180, or 365")

    q = quiverquant.quiver(QUIVER_TOKEN)
    df = q.congress_trading()
    if df is None or len(df) == 0:
        now = datetime.now(timezone.utc)
        since = now - timedelta(days=horizon_days)
        return {
            "date": now.date().isoformat(),
            "windowDays": horizon_days,
            "windowStart": since.date().isoformat(),
            "windowEnd": now.date().isoformat(),
            "convergence": [],
            "politicianBuys": [],
            "politicianSells": [],
            "note": "No congress trading rows returned from Quiver.",
        }

    try:
        rows = df.to_dict(orient="records")
    except Exception:
        rows = list(df)

    allow = crypto_allowlist()
    payload = compute_congress_window(rows, horizon_days, ticker_allowlist=allow)
    payload["note"] = "Crypto view filters congress trades to a crypto allowlist (spot tickers + common crypto ETFs/trusts)."

    return payload


# --------------------------
# Endpoint: Public filings “Leaders”
# --------------------------

@app.get("/report/public-leaders")
def report_public_leaders():
    now = datetime.now(timezone.utc)
    since365 = now - timedelta(days=WINDOW_365_DAYS)

    # Curated SEC entities (you can extend later)
    leaders = [
        {"key": "musk", "name": "Elon Musk", "cik": "1494730", "forms": ["4", "4/A"]},
        {"key": "berkshire", "name": "Berkshire Hathaway", "cik": "1067983", "forms": ["13F-HR", "13F-HR/A"]},
        {"key": "bridgewater", "name": "Bridgewater Associates", "cik": "1350694", "forms": ["13F-HR", "13F-HR/A"]},
    ]

    results = []

    for leader in leaders:
        cik = leader["cik"]
        subs = pull_recent_filings_for_cik(cik)
        filings = extract_latest_filings(subs, leader["forms"], max_items=10)

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
            buy_ct = 0
            sell_ct = 0

            if form.startswith("4"):
                try:
                    xml_text = sec_get_text(primary_url)
                    buy_ct, sell_ct = parse_form4_buy_sell(xml_text)
                    if buy_ct > 0 and sell_ct == 0:
                        label = "BUY"
                    elif sell_ct > 0 and buy_ct == 0:
                        label = "SELL"
                    elif buy_ct > 0 and sell_ct > 0:
                        label = "MIXED"
                    else:
                        label = "FORM4"
                except Exception:
                    label = "FORM4"

            if form.startswith("13F"):
                label = "13F FILED"

            events.append({
                "form": form,
                "label": label,
                "filingDate": filing_date,
                "indexUrl": index_url,
                "primaryUrl": primary_url,
                "buyTxCount": buy_ct,
                "sellTxCount": sell_ct,
            })

            if len(events) >= 3:
                break

            time.sleep(0.15)

        results.append({
            "name": leader["name"],
            "cik": cik10(cik),
            "events": events,
        })

    return {
        "date": now.date().isoformat(),
        "windowDays": WINDOW_365_DAYS,
        "windowStart": since365.date().isoformat(),
        "windowEnd": now.date().isoformat(),
        "leaders": results,
        "note": "Form 4 infers BUY/SELL from transaction codes (P/S). 13F is quarterly and shown as filing events.",
    }
