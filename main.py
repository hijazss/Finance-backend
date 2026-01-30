import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
import xml.etree.ElementTree as ET

from fastapi import FastAPI, HTTPException
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
        if 1 <= len(first) <= 8 and first.replace(".", "").replace("-", "").isalnum():
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
            dt = datetime.strptime(s[: len(fmt.replace("%f", "000000"))], fmt)
            return dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue

    return None


def iso_date_only(dt: Optional[datetime]) -> str:
    return dt.date().isoformat() if dt else ""


def interleave(items: List[dict], limit: int = 80) -> List[dict]:
    d = [x for x in items if x["party"] == "D"]
    r = [x for x in items if x["party"] == "R"]
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


def score_total(x: dict) -> int:
    return int(x.get("demBuyers", 0)) + int(x.get("repBuyers", 0)) + int(x.get("demSellers", 0)) + int(x.get("repSellers", 0))


def score_overlap_buy(x: dict) -> int:
    return int(x.get("demBuyers", 0)) + int(x.get("repBuyers", 0))


def build_window_view(rows: List[dict], now: datetime, days: int) -> dict:
    """
    Build a view for a rolling window.
    We prioritize FILED date for window filtering (disclosures), fallback to TRADED if filed missing.
    """
    since = now - timedelta(days=days)

    buys: List[dict] = []
    sells: List[dict] = []

    for r in rows:
        ticker = norm_ticker(r)
        party = norm_party_from_any(r)
        if not ticker or not party:
            continue

        tx = tx_text(r)
        kind = "BUY" if is_buy(tx) else "SELL" if is_sell(tx) else ""
        if not kind:
            continue

        filed_dt = parse_dt_any(pick_first(r, ["Filed", "filed", "ReportDate", "report_date", "Date", "date"], ""))
        traded_dt = parse_dt_any(pick_first(r, ["Traded", "traded", "TransactionDate", "transaction_date"], ""))

        best_dt = filed_dt or traded_dt
        if best_dt is None:
            continue

        if best_dt < since or best_dt > now:
            continue

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

    buy_cards = [to_card(x, "BUY") for x in interleave(buys, 120)]
    sell_cards = [to_card(x, "SELL") for x in interleave(sells, 120)]

    # Overlap for BUY only, using unique politicians per party per ticker
    dem_by_ticker: Dict[str, set] = {}
    rep_by_ticker: Dict[str, set] = {}

    for x in buys:
        t = x["ticker"]
        p = x.get("politician", "").strip()
        if x["party"] == "D":
            dem_by_ticker.setdefault(t, set()).add(p or "(unknown)")
        elif x["party"] == "R":
            rep_by_ticker.setdefault(t, set()).add(p or "(unknown)")

    overlap = sorted(set(dem_by_ticker.keys()).intersection(set(rep_by_ticker.keys())))

    overlap_cards: List[dict] = []
    for t in overlap:
        dem_ct = len(dem_by_ticker.get(t, set()))
        rep_ct = len(rep_by_ticker.get(t, set()))

        latest_dt: Optional[datetime] = None
        for x in buys:
            if x["ticker"] != t:
                continue
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

    overlap_cards.sort(key=lambda x: (-(x["demBuyers"] + x["repBuyers"]), x["ticker"]))

    # Dashboard summary explicitly scoped to this window
    total_buy = sum((c["demBuyers"] + c["repBuyers"]) for c in buy_cards)
    total_sell = sum((c["demSellers"] + c["repSellers"]) for c in sell_cards)
    overlap_count = len(overlap_cards)

    top_overlap = overlap_cards[:10]
    top_buys = sorted(buy_cards, key=lambda x: (-(x["demBuyers"] + x["repBuyers"]), x["ticker"]))[:20]
    top_sells = sorted(sell_cards, key=lambda x: (-(x["demSellers"] + x["repSellers"]), x["ticker"]))[:20]

    return {
        "windowDays": days,
        "windowStart": since.date().isoformat(),
        "windowEnd": now.date().isoformat(),
        "summary": {
            "totalBuyParticipation": total_buy,
            "totalSellParticipation": total_sell,
            "overlapTickerCount": overlap_count,
        },
        "convergence": top_overlap,
        "politicianBuys": top_buys,
        "politicianSells": top_sells,
    }


# --------------------------
# Endpoint: Congress report (dual windows)
# --------------------------

@app.get("/report/today")
def report_today():
    if not QUIVER_TOKEN:
        raise HTTPException(500, "QUIVER_TOKEN missing")

    now = datetime.now(timezone.utc)
    q = quiverquant.quiver(QUIVER_TOKEN)
    df = q.congress_trading()

    if df is None or len(df) == 0:
        empty30 = {
            "windowDays": WINDOW_30_DAYS,
            "windowStart": (now - timedelta(days=WINDOW_30_DAYS)).date().isoformat(),
            "windowEnd": now.date().isoformat(),
            "summary": {"totalBuyParticipation": 0, "totalSellParticipation": 0, "overlapTickerCount": 0},
            "convergence": [],
            "politicianBuys": [],
            "politicianSells": [],
        }
        empty180 = {
            "windowDays": WINDOW_180_DAYS,
            "windowStart": (now - timedelta(days=WINDOW_180_DAYS)).date().isoformat(),
            "windowEnd": now.date().isoformat(),
            "summary": {"totalBuyParticipation": 0, "totalSellParticipation": 0, "overlapTickerCount": 0},
            "convergence": [],
            "politicianBuys": [],
            "politicianSells": [],
        }
        return {
            "date": now.date().isoformat(),
            "window30": empty30,
            "window180": empty180,
            "fundSignals": [],
        }

    try:
        rows = df.to_dict(orient="records")
    except Exception:
        rows = list(df)

    view30 = build_window_view(rows, now, WINDOW_30_DAYS)
    view180 = build_window_view(rows, now, WINDOW_180_DAYS)

    # Backward-compat keys
    # Keep old "convergence" = 30-day convergence
    # Keep old "politicianBuys/Sells" = 30-day lists
    return {
        "date": now.date().isoformat(),

        "window30": view30,
        "window180": view180,

        "convergence": view30["convergence"],
        "politicianBuys": view30["politicianBuys"],
        "politicianSells": view30["politicianSells"],
        "bipartisanTickers": view30["politicianBuys"],

        "fundSignals": [],
    }


# --------------------------
# SEC EDGAR helpers + Leaders endpoint (unchanged)
# --------------------------

def sec_get_text(url: str) -> str:
    headers = {"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}
    r = requests.get(url, headers=headers, timeout=SEC_TIMEOUT)
    if r.status_code != 200:
        raise HTTPException(502, f"SEC fetch failed HTTP {r.status_code} for {url}")
    return r.text


def sec_get_json(url: str) -> dict:
    headers = {"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}
    r = requests.get(url, headers=headers, timeout=SEC_TIMEOUT)
    if r.status_code != 200:
        raise HTTPException(502, f"SEC fetch failed HTTP {r.status_code} for {url}")
    return r.json()


def cik10(cik: str) -> str:
    s = re.sub(r"\D", "", str(cik))
    return s.zfill(10)


def accession_no_dashes(acc: str) -> str:
    return (acc or "").replace("-", "").strip()


def build_filing_index_url(cik: str, accession: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_no_dashes(accession)}/{accession}-index.html"


def build_primary_doc_url(cik: str, accession: str, primary_doc: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_no_dashes(accession)}/{primary_doc}"


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def parse_form4_buy_sell(xml_text: str) -> Tuple[int, int]:
    buy = 0
    sell = 0
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return (0, 0)

    for node in root.iter():
        if _local_name(node.tag).lower() == "transactioncode":
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


@app.get("/report/public-leaders")
def report_public_leaders():
    now = datetime.now(timezone.utc)
    since365 = now - timedelta(days=WINDOW_365_DAYS)

    leaders = [
        {"name": "Elon Musk", "cik": "1494730", "forms": ["4", "4/A"]},
        {"name": "Berkshire Hathaway", "cik": "1067983", "forms": ["13F-HR", "13F-HR/A"]},
        {"name": "Bridgewater Associates", "cik": "1350694", "forms": ["13F-HR", "13F-HR/A"]},
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

            time.sleep(0.2)

        results.append({"name": leader["name"], "cik": cik10(cik), "events": events})

    return {
        "date": now.date().isoformat(),
        "windowDays": WINDOW_365_DAYS,
        "windowStart": since365.date().isoformat(),
        "windowEnd": now.date().isoformat(),
        "leaders": results,
        "note": "Form 4 BUY/SELL uses transaction codes P/S. 13F is shown as filing events.",
    }
