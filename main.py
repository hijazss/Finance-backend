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

    # ISO-ish
    try:
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        pass

    # Common non-ISO formats that often break overlap counts
    fmts = [
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%m/%d/%y",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(s[: len(datetime.now().strftime(fmt))], fmt)
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


# --------------------------
# Price data helpers (Stooq)
# --------------------------

_price_cache: Dict[str, Tuple[float, Dict[str, float]]] = {}
# cache: symbol -> (timestamp_epoch, { "YYYY-MM-DD": close })

def stooq_symbol(ticker: str) -> str:
    # Stooq US equities typically are: aapl.us, nvda.us
    # ETFs like SPY: spy.us
    t = ticker.lower()
    t = t.replace("-", ".")  # stooq uses dots in some symbols
    return f"{t}.us"


def fetch_stooq_daily_closes(ticker: str) -> Dict[str, float]:
    """
    Returns dict date->close for ticker from Stooq.
    """
    now_ts = time.time()
    key = ticker.upper()

    cached = _price_cache.get(key)
    if cached and (now_ts - cached[0] < 6 * 60 * 60):
        return cached[1]

    sym = stooq_symbol(ticker)
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    r = requests.get(url, timeout=20)
    if r.status_code != 200:
        raise HTTPException(502, f"Price fetch failed HTTP {r.status_code} for {ticker}")

    lines = r.text.strip().splitlines()
    if len(lines) < 2:
        raise HTTPException(502, f"No price data for {ticker}")

    # CSV header: Date,Open,High,Low,Close,Volume
    out: Dict[str, float] = {}
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) < 5:
            continue
        d = parts[0].strip()
        c = parts[4].strip()
        try:
            out[d] = float(c)
        except Exception:
            continue

    _price_cache[key] = (now_ts, out)
    return out


def closest_trading_close(closes: Dict[str, float], target_date: datetime, direction: str) -> Optional[Tuple[str, float]]:
    """
    Find close on or after (direction='forward') or on or before ('backward') target_date.
    """
    if not closes:
        return None
    d0 = target_date.date().isoformat()
    dates = sorted(closes.keys())
    if direction == "forward":
        for d in dates:
            if d >= d0:
                return (d, closes[d])
        return None
    else:
        for d in reversed(dates):
            if d <= d0:
                return (d, closes[d])
        return None


def forward_return(ticker: str, start_dt: datetime, horizon_days: int) -> Optional[Tuple[float, str, str]]:
    """
    Compute close-to-close forward return over horizon_days trading days approximated by calendar days.
    Returns (ret, start_date_used, end_date_used).
    """
    closes = fetch_stooq_daily_closes(ticker)
    s = closest_trading_close(closes, start_dt, "forward")
    if not s:
        return None
    s_date, s_close = s
    end_dt = start_dt + timedelta(days=horizon_days)
    e = closest_trading_close(closes, end_dt, "forward")
    if not e:
        return None
    e_date, e_close = e
    if s_close <= 0:
        return None
    return ((e_close / s_close) - 1.0, s_date, e_date)


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
# Endpoint: Congress (rolling N days)
# --------------------------

@app.get("/report/today")
def report_today(days: int = Query(30, ge=1, le=365)):
    """
    Rolling congress report. Use days=30 or days=180 from the frontend.
    """
    if not QUIVER_TOKEN:
        raise HTTPException(500, "QUIVER_TOKEN missing")

    # safety: only allow your two windows for now
    if days not in (30, 180):
        raise HTTPException(400, "days must be 30 or 180")

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
            "bipartisanTickers": [],
            "politicianBuys": [],
            "politicianSells": [],
            "fundSignals": [],
        }

    try:
        rows = df.to_dict(orient="records")
    except Exception:
        rows = list(df)

    buys: List[dict] = []
    sells: List[dict] = []

    for r in rows:
        best_dt = row_best_dt(r)
        if best_dt is None:
            continue
        if best_dt < since or best_dt > now:
            continue

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
        100
    )
    sell_mixed = interleave(
        [{"ticker": x["ticker"], "party": x["party"], "filed": x["filed"], "traded": x["traded"], "politician": x["politician"]} for x in sells],
        100
    )

    buy_cards = [to_card(x, "BUY") for x in buy_mixed]
    sell_cards = [to_card(x, "SELL") for x in sell_mixed]

    # Convergence = overlapping tickers on BUY in the window, not limited by interleave
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

    # Sort overlap by most bipartisan participation
    overlap_cards.sort(key=lambda x: (-(x["demBuyers"] + x["repBuyers"]), x["ticker"]))

    return {
        "date": now.date().isoformat(),
        "windowDays": days,
        "windowStart": since.date().isoformat(),
        "windowEnd": now.date().isoformat(),
        "convergence": overlap_cards[:25],
        "bipartisanTickers": buy_cards[:60],  # backward compatibility
        "politicianBuys": buy_cards[:100],
        "politicianSells": sell_cards[:100],
        "fundSignals": [],
    }


# --------------------------
# NEW Endpoint: Performance (2Y) leaderboard
# --------------------------

@app.get("/report/performance-2y")
def performance_2y(horizon_days: int = Query(30, ge=7, le=180)):
    """
    Two-year performance leaderboard for politicians:
    - BUY: forward horizon_days return
    - SELL: negative forward horizon_days return
    - Score: excess return vs SPY for same windows
    """
    if not QUIVER_TOKEN:
        raise HTTPException(500, "QUIVER_TOKEN missing")

    now = datetime.now(timezone.utc)
    since = now - timedelta(days=365 * 2)

    q = quiverquant.quiver(QUIVER_TOKEN)
    df = q.congress_trading()

    if df is None or len(df) == 0:
        return {
            "date": now.date().isoformat(),
            "windowDays": 730,
            "windowStart": since.date().isoformat(),
            "windowEnd": now.date().isoformat(),
            "horizonDays": horizon_days,
            "leaders": [],
            "note": "No congress trading rows returned.",
        }

    try:
        rows = df.to_dict(orient="records")
    except Exception:
        rows = list(df)

    # Preload SPY closes once (caches anyway)
    _ = fetch_stooq_daily_closes("SPY")

    stats: Dict[str, Dict[str, Any]] = {}

    for r in rows:
        best_dt = row_best_dt(r)
        if best_dt is None:
            continue
        if best_dt < since or best_dt > now:
            continue

        ticker = norm_ticker(r)
        party = norm_party_from_any(r)
        if not ticker or not party:
            continue

        tx = tx_text(r)
        kind = "BUY" if is_buy(tx) else "SELL" if is_sell(tx) else ""
        if not kind:
            continue

        pol = str(pick_first(r, ["Politician", "politician", "Representative", "Senator", "Name", "name"], "")).strip()
        if not pol:
            continue

        # Use traded date if possible, otherwise filed date, for performance timestamp
        traded_dt = parse_dt_any(pick_first(r, ["Traded", "traded", "TransactionDate", "transaction_date"], ""))
        filed_dt = parse_dt_any(pick_first(r, ["Filed", "filed", "ReportDate", "report_date", "Date", "date"], ""))
        perf_dt = traded_dt or filed_dt or best_dt

        # Compute forward returns
        try:
            stock_ret = forward_return(ticker, perf_dt, horizon_days)
            spy_ret = forward_return("SPY", perf_dt, horizon_days)
        except Exception:
            continue

        if not stock_ret or not spy_ret:
            continue

        stock_r, stock_s, stock_e = stock_ret
        spy_r, spy_s, spy_e = spy_ret

        # BUY positive is good, SELL negative forward is good
        signed_stock = stock_r if kind == "BUY" else -stock_r
        signed_spy = spy_r if kind == "BUY" else -spy_r
        excess = signed_stock - signed_spy

        key = pol
        if key not in stats:
            stats[key] = {
                "name": pol,
                "party": party,
                "tradesScored": 0,
                "winTrades": 0,
                "avgExcessReturn": 0.0,
                "sumExcessReturn": 0.0,
                "sample": [],
            }

        stats[key]["tradesScored"] += 1
        stats[key]["sumExcessReturn"] += excess
        if excess > 0:
            stats[key]["winTrades"] += 1

        # Keep a small sample of recent trades for UI drilldown
        if len(stats[key]["sample"]) < 6:
            stats[key]["sample"].append({
                "ticker": ticker.upper(),
                "kind": kind,
                "tradeDateUsed": iso_date_only(perf_dt),
                "windowStartUsed": stock_s,
                "windowEndUsed": stock_e,
                "stockReturn": round(signed_stock * 100.0, 2),
                "spyReturn": round(signed_spy * 100.0, 2),
                "excessReturn": round(excess * 100.0, 2),
            })

    leaders = []
    for _, v in stats.items():
        n = v["tradesScored"]
        if n <= 0:
            continue
        v["avgExcessReturn"] = v["sumExcessReturn"] / n
        v["winRate"] = (v["winTrades"] / n) if n else 0.0
        leaders.append(v)

    # Sort: best avg excess first, then more trades
    leaders.sort(key=lambda x: (-(x["avgExcessReturn"]), -(x["tradesScored"]), x["name"]))

    # Return top 25 for UI
    out = []
    for x in leaders[:25]:
        out.append({
            "name": x["name"],
            "party": x["party"],
            "tradesScored": x["tradesScored"],
            "winRate": round(x["winRate"] * 100.0, 1),
            "avgExcessReturnPct": round(x["avgExcessReturn"] * 100.0, 2),
            "sample": x["sample"],
        })

    return {
        "date": now.date().isoformat(),
        "windowDays": 730,
        "windowStart": since.date().isoformat(),
        "windowEnd": now.date().isoformat(),
        "horizonDays": horizon_days,
        "leaders": out,
        "note": "Performance uses forward returns over horizonDays and compares to SPY on same dates. SELL is scored as -forward return.",
    }


# --------------------------
# Endpoint: Public filings “Leaders” (SEC EDGAR)
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
        "note": "Form 4 events infer BUY/SELL from transaction codes (P/S). 13F is quarterly and shown as filing events.",
    }
