# main.py
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

SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "FinanceApp/1.0 (contact: hijazss@gmail.com)")
SEC_TIMEOUT = 20

WINDOW_30_DAYS = 30
WINDOW_180_DAYS = 180
WINDOW_365_DAYS = 365
WINDOW_2Y_DAYS = 730

_party_re = re.compile(r"/\s*([DR])\b", re.IGNORECASE)


@app.get("/")
def root():
    return {"status": "ok"}


# --------------------------
# Helpers: common parsing
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


def parse_dt_any(v: Any) -> Optional[datetime]:
    """
    Robust date parser.
    Handles:
      - ISO: 2026-01-30, 2026-01-30T12:34:56Z
      - US: 01/30/2026, 1/30/26
      - With time: 01/30/2026 13:22:00
    """
    if v is None:
        return None
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)

    s = str(v).strip()
    if not s:
        return None

    s = s.replace("Z", "+00:00")

    # Try ISO first
    try:
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        pass

    # Common formats (IMPORTANT: includes MM/DD/YYYY)
    fmts = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%m/%d/%Y",
        "%m/%d/%y",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%y %H:%M:%S",
    ]

    # Normalize double spaces
    s2 = re.sub(r"\s+", " ", s)

    for fmt in fmts:
        try:
            dt = datetime.strptime(s2, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue

    # Last resort: try to extract something that looks like a date
    m = re.search(r"(\d{1,2}/\d{1,2}/\d{2,4})", s2)
    if m:
        for fmt in ("%m/%d/%Y", "%m/%d/%y"):
            try:
                dt = datetime.strptime(m.group(1), fmt)
                return dt.replace(tzinfo=timezone.utc)
            except Exception:
                continue

    return None


def iso_date_only(dt: Optional[datetime]) -> str:
    return dt.date().isoformat() if dt else ""


def norm_politician_name(row: dict) -> str:
    return str(pick_first(row, ["Politician", "politician", "Representative", "Senator", "Name", "name"], "")).strip()


def norm_chamber(row: dict) -> str:
    for k in ["Chamber", "chamber"]:
        v = row.get(k)
        if not v:
            continue
        s = str(v).strip().lower()
        if "sen" in s:
            return "Senate"
        if "hou" in s:
            return "House"

    p = norm_politician_name(row).lower()
    if p.startswith("sen") or " sen" in p:
        return "Senate"
    if p.startswith("rep") or " rep" in p:
        return "House"
    return ""


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
    dem = 1 if x.get("party") == "D" else 0
    rep = 1 if x.get("party") == "R" else 0
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
    }


# --------------------------
# Safe price lookup (Stooq)
# --------------------------

_PRICE_CACHE: Dict[str, List[Tuple[datetime, float]]] = {}


def _stooq_symbol(ticker: str) -> str:
    t = (ticker or "").strip().lower()
    if not t:
        return ""
    if not t.endswith(".us"):
        t = f"{t}.us"
    return t


def _fetch_stooq_daily_closes(ticker: str) -> List[Tuple[datetime, float]]:
    sym = _stooq_symbol(ticker)
    if not sym:
        return []

    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"

    try:
        r = requests.get(url, timeout=12)
        if r.status_code != 200:
            return []

        text = (r.text or "").strip()
        if not text or "Date,Open" not in text:
            return []

        lines = text.splitlines()
        if len(lines) < 2:
            return []

        out: List[Tuple[datetime, float]] = []
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) < 5:
                continue

            dt = parse_dt_any(parts[0])
            if not dt:
                continue

            try:
                close = float(parts[4])
            except Exception:
                continue

            out.append((dt.replace(hour=0, minute=0, second=0, microsecond=0), close))

        out.sort(key=lambda x: x[0])
        return out

    except Exception:
        return []


def get_close_on_or_after(ticker: str, dt: datetime) -> Optional[float]:
    try:
        t = (ticker or "").upper().strip()
        if not t:
            return None

        if t not in _PRICE_CACHE:
            _PRICE_CACHE[t] = _fetch_stooq_daily_closes(t)

        series = _PRICE_CACHE.get(t) or []
        if not series:
            return None

        target = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        for d, price in series:
            if d >= target:
                return price

        return None
    except Exception:
        return None


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
        out.append(
            {
                "form": form,
                "accession": str(accs[i]),
                "primaryDocument": str(primary_docs[i]),
                "filingDate": str(filed_dates[i]),
            }
        )

    out.sort(key=lambda x: x.get("filingDate", ""), reverse=True)
    return out[:max_items]


# --------------------------
# Core Congress window builder
# --------------------------

def build_congress_window(window_days: int) -> dict:
    if not QUIVER_TOKEN:
        raise HTTPException(500, "QUIVER_TOKEN missing")

    now = datetime.now(timezone.utc)
    since = now - timedelta(days=window_days)

    q = quiverquant.quiver(QUIVER_TOKEN)
    df = q.congress_trading()

    if df is None or len(df) == 0:
        return {
            "windowDays": window_days,
            "windowStart": since.date().isoformat(),
            "windowEnd": now.date().isoformat(),
            "convergence": [],
            "politicianBuys": [],
            "politicianSells": [],
        }

    try:
        rows = df.to_dict(orient="records")
    except Exception:
        rows = list(df)

    buys: List[dict] = []
    sells: List[dict] = []

    for r in rows:
        ticker = norm_ticker(r)
        party = norm_party_from_any(r)
        pol = norm_politician_name(r)

        if not ticker or not party or not pol:
            continue

        tx = tx_text(r)
        kind = "BUY" if is_buy(tx) else "SELL" if is_sell(tx) else ""
        if not kind:
            continue

        filed_dt = parse_dt_any(pick_first(r, ["Filed", "filed", "ReportDate", "report_date", "Date", "date"], ""))
        traded_dt = parse_dt_any(pick_first(r, ["Traded", "traded", "TransactionDate", "transaction_date"], ""))

        # IMPORTANT CHANGE:
        # Include if EITHER traded OR filed is within the rolling window.
        in_window = False
        if traded_dt and since <= traded_dt <= now:
            in_window = True
        if filed_dt and since <= filed_dt <= now:
            in_window = True
        if not in_window:
            continue

        # For ordering, prefer traded date, else filed date
        best_dt = traded_dt or filed_dt
        if not best_dt:
            continue

        item = {
            "ticker": ticker.upper(),
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

    # Convergence = overlapping tickers on BUY in this window
    dem_buy = [x for x in buys if x["party"] == "D"]
    rep_buy = [x for x in buys if x["party"] == "R"]
    overlap = sorted(set(x["ticker"] for x in dem_buy).intersection(set(x["ticker"] for x in rep_buy)))

    overlap_cards: List[dict] = []
    for t in overlap:
        dem_ct = sum(1 for x in dem_buy if x["ticker"] == t)
        rep_ct = sum(1 for x in rep_buy if x["ticker"] == t)

        latest_dt: Optional[datetime] = None
        for x in buys:
            if x["ticker"] != t:
                continue
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
                "funds": [],
                "lastFiledAt": iso_date_only(latest_dt),
                "strength": "OVERLAP",
            }
        )

    overlap_cards.sort(key=lambda x: (-(x["demBuyers"] + x["repBuyers"]), x["ticker"]))

    return {
        "windowDays": window_days,
        "windowStart": since.date().isoformat(),
        "windowEnd": now.date().isoformat(),
        "convergence": overlap_cards[:50],
        "politicianBuys": buy_cards[:120],
        "politicianSells": sell_cards[:120],
    }


# --------------------------
# Endpoint: Congress rolling 30D + include 180D overlap payload
# --------------------------

@app.get("/report/today")
def report_today():
    now = datetime.now(timezone.utc)

    data30 = build_congress_window(WINDOW_30_DAYS)
    data180 = build_congress_window(WINDOW_180_DAYS)

    return {
        "date": now.date().isoformat(),
        "windowDays": WINDOW_30_DAYS,
        "windowStart": data30["windowStart"],
        "windowEnd": data30["windowEnd"],

        # Main 30D payload (what your UI currently uses)
        "convergence": data30["convergence"][:25],
        "politicianBuys": data30["politicianBuys"][:120],
        "politicianSells": data30["politicianSells"][:120],

        # Backward compatibility
        "bipartisanTickers": data30["politicianBuys"][:60],
        "fundSignals": [],

        # Extra: 180D overlap (so UI can show "30D overlap" and "180D overlap")
        "overlap180": {
            "windowDays": WINDOW_180_DAYS,
            "windowStart": data180["windowStart"],
            "windowEnd": data180["windowEnd"],
            "convergence": data180["convergence"][:50],
        },
    }


# --------------------------
# Endpoint: Congress performance over last 2 years
# --------------------------

@app.get("/report/performance-2y")
def report_performance_2y(
    horizon_days: int = Query(30, ge=7, le=365),
    limit: int = Query(25, ge=5, le=200),
):
    if not QUIVER_TOKEN:
        raise HTTPException(500, "QUIVER_TOKEN missing")

    now = datetime.now(timezone.utc)
    since2y = now - timedelta(days=WINDOW_2Y_DAYS)

    q = quiverquant.quiver(QUIVER_TOKEN)
    df = q.congress_trading()

    if df is None or len(df) == 0:
        return {
            "date": now.date().isoformat(),
            "lookbackDays": WINDOW_2Y_DAYS,
            "horizonDays": horizon_days,
            "leaders": [],
            "note": "No congress data returned from Quiver.",
        }

    try:
        rows = df.to_dict(orient="records")
    except Exception:
        rows = list(df)

    trades: List[dict] = []
    for r in rows:
        ticker = norm_ticker(r)
        party = norm_party_from_any(r)
        pol = norm_politician_name(r)
        chamber = norm_chamber(r)

        if not ticker or not party or not pol:
            continue

        tx = tx_text(r)
        kind = "BUY" if is_buy(tx) else "SELL" if is_sell(tx) else ""
        if not kind:
            continue

        filed_dt = parse_dt_any(pick_first(r, ["Filed", "filed", "ReportDate", "report_date", "Date", "date"], ""))
        traded_dt = parse_dt_any(pick_first(r, ["Traded", "traded", "TransactionDate", "transaction_date"], ""))

        event_dt = traded_dt or filed_dt
        if not event_dt:
            continue
        if not (since2y <= event_dt <= now):
            continue

        amount = str(pick_first(r, ["Range", "range", "Amount", "amount", "TransactionAmount", "transaction_amount"], "")).strip()

        trades.append(
            {
                "politician": pol,
                "party": party,
                "chamber": chamber,
                "ticker": ticker.upper(),
                "kind": kind,
                "tradedDate": iso_date_only(traded_dt),
                "filedDate": iso_date_only(filed_dt),
                "event_dt": event_dt,
                "amount": amount,
            }
        )

    if not trades:
        return {
            "date": now.date().isoformat(),
            "lookbackDays": WINDOW_2Y_DAYS,
            "horizonDays": horizon_days,
            "leaders": [],
            "note": "No scorable trades found in last 2 years.",
        }

    horizon_delta = timedelta(days=horizon_days)

    agg: Dict[str, dict] = {}

    def ensure(pol_key: str, party: str, chamber: str) -> dict:
        if pol_key not in agg:
            agg[pol_key] = {
                "name": pol_key,
                "party": party,
                "chamber": chamber or "",
                "buyCount": 0,
                "sellCount": 0,
                "scoredBuy": 0,
                "scoredSell": 0,
                "avgBuyReturn": 0.0,
                "avgSellAlpha": 0.0,
                "score": 0.0,
                "examples": [],
            }
        return agg[pol_key]

    price_pair_cache: Dict[Tuple[str, str, int], Optional[Tuple[float, float]]] = {}

    for tr in trades:
        ticker = tr["ticker"]
        event_dt = tr["event_dt"]
        horizon_dt = event_dt + horizon_delta

        key = (ticker, iso_date_only(event_dt), horizon_days)
        if key not in price_pair_cache:
            p0 = get_close_on_or_after(ticker, event_dt)
            p1 = get_close_on_or_after(ticker, horizon_dt)
            if p0 is None or p1 is None or p0 <= 0:
                price_pair_cache[key] = None
            else:
                price_pair_cache[key] = (p0, p1)

        pair = price_pair_cache.get(key)
        if not pair:
            continue

        p0, p1 = pair
        fwd_ret = (p1 - p0) / p0

        rec = ensure(tr["politician"], tr["party"], tr["chamber"])

        if len(rec["examples"]) < 6:
            rec["examples"].append(
                {
                    "ticker": ticker,
                    "kind": tr["kind"],
                    "tradedDate": tr["tradedDate"],
                    "filedDate": tr["filedDate"],
                    "amount": tr["amount"],
                    "horizonReturn": round(fwd_ret, 4),
                }
            )

        if tr["kind"] == "BUY":
            rec["buyCount"] += 1
            rec["avgBuyReturn"] += fwd_ret
            rec["scoredBuy"] += 1
        else:
            rec["sellCount"] += 1
            rec["avgSellAlpha"] += (-fwd_ret)
            rec["scoredSell"] += 1

    leaders = list(agg.values())

    for r in leaders:
        if r["scoredBuy"] > 0:
            r["avgBuyReturn"] = r["avgBuyReturn"] / r["scoredBuy"]
        else:
            r["avgBuyReturn"] = 0.0

        if r["scoredSell"] > 0:
            r["avgSellAlpha"] = r["avgSellAlpha"] / r["scoredSell"]
        else:
            r["avgSellAlpha"] = 0.0

        sample = r["scoredBuy"] + r["scoredSell"]
        size_boost = min(1.0, sample / 20.0)
        r["score"] = (r["avgBuyReturn"] + r["avgSellAlpha"]) * size_boost

        r["avgBuyReturn"] = round(r["avgBuyReturn"], 4)
        r["avgSellAlpha"] = round(r["avgSellAlpha"], 4)
        r["score"] = round(r["score"], 4)

    leaders.sort(key=lambda x: (x["score"], x["scoredBuy"] + x["scoredSell"]), reverse=True)
    leaders = [x for x in leaders if (x["scoredBuy"] + x["scoredSell"]) >= 3]

    return {
        "date": now.date().isoformat(),
        "lookbackDays": WINDOW_2Y_DAYS,
        "horizonDays": horizon_days,
        "note": "Scores use Stooq daily closes. Buys use forward return. Sells use negative forward return (price down after sell is positive).",
        "leaders": leaders[:limit],
    }


# --------------------------
# Endpoint: Public filings “Leaders” (SEC EDGAR)
# --------------------------

@app.get("/report/public-leaders")
def report_public_leaders():
    now = datetime.now(timezone.utc)
    since365 = now - timedelta(days=WINDOW_365_DAYS)

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

            events.append(
                {
                    "form": form,
                    "label": label,
                    "filingDate": filing_date,
                    "indexUrl": index_url,
                    "primaryUrl": primary_url,
                    "buyTxCount": buy_ct,
                    "sellTxCount": sell_ct,
                }
            )

            if len(events) >= 3:
                break

            time.sleep(0.2)

        results.append({"name": leader["name"], "cik": cik10(cik), "events": events})

    return {
        "date": now.date().isoformat(),
        "windowDays": WINDOW_365_DAYS,
        "windowStart": since365.date().isoformat(),
        "windowEnd": now.date().isoformat(),
        "leaders": results,
        "note": "Form 4 infers BUY/SELL from transaction codes (P/S). 13F is quarterly and shown as filing events only.",
    }
