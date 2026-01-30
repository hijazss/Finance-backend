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

# Windows
WINDOW_30_DAYS = 30
WINDOW_2Y_DAYS = 730
WINDOW_365_DAYS = 365

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

    # Common date only
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            dt = datetime.strptime(s[:10], fmt)
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


# --------------------------
# SEC EDGAR helpers (Leaders tab)
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
# Price helper (Performance tab)
# Uses Stooq daily closes: https://stooq.com/q/d/l/?s=nvda.us&i=d
# --------------------------

_PRICE_CACHE: Dict[str, List[Tuple[datetime, float]]] = {}


def _stooq_symbol(ticker: str) -> str:
    # Stooq uses lowercase and .us for most US equities
    t = (ticker or "").strip().lower()
    return f"{t}.us"


def _fetch_stooq_daily_closes(ticker: str) -> List[Tuple[datetime, float]]:
    sym = _stooq_symbol(ticker)
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    r = requests.get(url, timeout=12)
    if r.status_code != 200:
        return []

    lines = (r.text or "").strip().splitlines()
    if len(lines) < 2:
        return []

    # header: Date,Open,High,Low,Close,Volume
    out: List[Tuple[datetime, float]] = []
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) < 5:
            continue
        dt = parse_dt_any(parts[0])
        try:
            close = float(parts[4])
        except Exception:
            continue
        if dt:
            out.append((dt.replace(hour=0, minute=0, second=0, microsecond=0), close))

    out.sort(key=lambda x: x[0])
    return out


def get_close_on_or_after(ticker: str, dt: datetime) -> Optional[float]:
    t = (ticker or "").upper().strip()
    if not t:
        return None

    if t not in _PRICE_CACHE:
        _PRICE_CACHE[t] = _fetch_stooq_daily_closes(t)

    series = _PRICE_CACHE.get(t) or []
    if not series:
        return None

    target = dt.replace(hour=0, minute=0, second=0, microsecond=0)

    # linear scan is OK because series is daily and we keep limited calls,
    # but do a simple binary search pattern for speed.
    lo, hi = 0, len(series) - 1
    ans_idx = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if series[mid][0] >= target:
            ans_idx = mid
            hi = mid - 1
        else:
            lo = mid + 1

    if ans_idx is None:
        return None
    return series[ans_idx][1]


def forward_return(ticker: str, traded_dt: datetime, horizon_days: int) -> Optional[float]:
    p0 = get_close_on_or_after(ticker, traded_dt)
    if p0 is None or p0 <= 0:
        return None
    p1 = get_close_on_or_after(ticker, traded_dt + timedelta(days=horizon_days))
    if p1 is None or p1 <= 0:
        return None
    return (p1 / p0) - 1.0


# --------------------------
# Endpoint: Congress rolling 30D (Main tab)
# --------------------------

@app.get("/report/today")
def report_today():
    if not QUIVER_TOKEN:
        raise HTTPException(500, "QUIVER_TOKEN missing")

    now = datetime.now(timezone.utc)
    since30 = now - timedelta(days=WINDOW_30_DAYS)

    q = quiverquant.quiver(QUIVER_TOKEN)
    df = q.congress_trading()

    if df is None or len(df) == 0:
        return {
            "date": now.date().isoformat(),
            "windowDays": WINDOW_30_DAYS,
            "windowStart": since30.date().isoformat(),
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
        if best_dt < since30 or best_dt > now:
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
        80,
    )
    sell_mixed = interleave(
        [{"ticker": x["ticker"], "party": x["party"], "filed": x["filed"], "traded": x["traded"], "politician": x["politician"]} for x in sells],
        80,
    )

    buy_cards = [to_card(x, "BUY") for x in buy_mixed]
    sell_cards = [to_card(x, "SELL") for x in sell_mixed]

    # Convergence: overlapping tickers on BUY in last 30D
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

    overlap_cards.sort(key=lambda x: (-(x["demBuyers"] + x["repBuyers"]), x["ticker"]))

    return {
        "date": now.date().isoformat(),
        "windowDays": WINDOW_30_DAYS,
        "windowStart": since30.date().isoformat(),
        "windowEnd": now.date().isoformat(),
        "convergence": overlap_cards[:10],
        "bipartisanTickers": buy_cards[:60],  # backward compatibility
        "politicianBuys": buy_cards[:80],
        "politicianSells": sell_cards[:80],
        "fundSignals": [],
    }


# --------------------------
# Endpoint: Performance 2Y (new "Performance at two years" tab)
# Query param: horizon_days (default 30)
# --------------------------

@app.get("/report/performance-2y")
def report_performance_2y(
    horizon_days: int = Query(default=30, ge=7, le=180),
    min_trades: int = Query(default=6, ge=2, le=50),
    max_trades_evaluated: int = Query(default=600, ge=100, le=2000),
):
    """
    Ranks politicians over the last 2 years using a forward-return score.

    - BUY score: forward % return over horizon_days
    - SELL score: negative forward return (selling before a decline scores positive)

    Output: top performers by average score, plus summary stats.
    """
    if not QUIVER_TOKEN:
        raise HTTPException(500, "QUIVER_TOKEN missing")

    now = datetime.now(timezone.utc)
    since2y = now - timedelta(days=WINDOW_2Y_DAYS)

    q = quiverquant.quiver(QUIVER_TOKEN)
    df = q.congress_trading()
    if df is None or len(df) == 0:
        return {
            "date": now.date().isoformat(),
            "windowDays": WINDOW_2Y_DAYS,
            "windowStart": since2y.date().isoformat(),
            "windowEnd": now.date().isoformat(),
            "horizonDays": horizon_days,
            "leaders": [],
            "note": "No congress trading rows returned.",
        }

    try:
        rows = df.to_dict(orient="records")
    except Exception:
        rows = list(df)

    # Collect eligible trades with reliable traded date
    trades: List[dict] = []
    for r in rows:
        best_dt = row_best_dt(r)
        if best_dt is None or best_dt < since2y or best_dt > now:
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
        chamber = str(pick_first(r, ["Chamber", "chamber", "House", "Senate"], "")).strip()

        # Prefer traded date for performance calculations
        traded_dt = parse_dt_any(pick_first(r, ["Traded", "traded", "TransactionDate", "transaction_date"], ""))
        filed_dt = parse_dt_any(pick_first(r, ["Filed", "filed", "ReportDate", "report_date", "Date", "date"], ""))

        base_dt = traded_dt or filed_dt or best_dt

        # Some rows can be missing politician name; skip
        if not pol:
            continue

        trades.append({
            "politician": pol,
            "party": party,
            "chamber": chamber,
            "ticker": ticker,
            "kind": kind,
            "traded_dt": base_dt,
            "filed_dt": filed_dt,
        })

    # Sort newest first and cap work to avoid timeouts
    trades.sort(key=lambda x: x["traded_dt"], reverse=True)
    trades = trades[:max_trades_evaluated]

    # Score each trade using cached price lookups
    scored: List[dict] = []
    for t in trades:
        r = forward_return(t["ticker"], t["traded_dt"], horizon_days)
        if r is None:
            continue

        # SELL: if price goes down after selling, that is "good" timing, so invert
        score = r if t["kind"] == "BUY" else (-r)

        scored.append({
            **t,
            "forward_return": r,
            "score": score,
        })

        # Tiny pacing to be polite to upstream if cache misses are happening
        if len(scored) % 25 == 0:
            time.sleep(0.05)

    # Aggregate per politician
    by_pol: Dict[str, dict] = {}
    for s in scored:
        name = s["politician"]
        cur = by_pol.get(name)
        if cur is None:
            cur = {
                "name": name,
                "party": s["party"],
                "chamber": s["chamber"] or "",
                "trades": 0,
                "buys": 0,
                "sells": 0,
                "avgScore": 0.0,
                "avgBuyReturn": 0.0,
                "avgSellAlpha": 0.0,
                "recentExamples": [],
            }
            by_pol[name] = cur

        cur["trades"] += 1
        cur["avgScore"] += float(s["score"])

        if s["kind"] == "BUY":
            cur["buys"] += 1
            cur["avgBuyReturn"] += float(s["forward_return"])
        else:
            cur["sells"] += 1
            # sell alpha = -forward_return
            cur["avgSellAlpha"] += float(-s["forward_return"])

        # keep a few examples
        if len(cur["recentExamples"]) < 4:
            cur["recentExamples"].append({
                "ticker": s["ticker"],
                "kind": s["kind"],
                "tradeDate": iso_date_only(s["traded_dt"]),
                "filedDate": iso_date_only(s["filed_dt"]),
                "forwardReturn": round(float(s["forward_return"]), 4),
                "score": round(float(s["score"]), 4),
            })

    # Finalize averages
    out: List[dict] = []
    for pol, cur in by_pol.items():
        if cur["trades"] < min_trades:
            continue

        cur["avgScore"] = cur["avgScore"] / max(1, cur["trades"])
        if cur["buys"] > 0:
            cur["avgBuyReturn"] = cur["avgBuyReturn"] / cur["buys"]
        else:
            cur["avgBuyReturn"] = 0.0

        if cur["sells"] > 0:
            cur["avgSellAlpha"] = cur["avgSellAlpha"] / cur["sells"]
        else:
            cur["avgSellAlpha"] = 0.0

        # round for UI
        cur["avgScore"] = round(cur["avgScore"], 4)
        cur["avgBuyReturn"] = round(cur["avgBuyReturn"], 4)
        cur["avgSellAlpha"] = round(cur["avgSellAlpha"], 4)

        out.append(cur)

    # Sort best performers
    out.sort(key=lambda x: (x["avgScore"], x["trades"]), reverse=True)

    return {
        "date": now.date().isoformat(),
        "windowDays": WINDOW_2Y_DAYS,
        "windowStart": since2y.date().isoformat(),
        "windowEnd": now.date().isoformat(),
        "horizonDays": horizon_days,
        "minTrades": min_trades,
        "maxTradesEvaluated": max_trades_evaluated,
        "leaders": out[:25],
        "note": "Scores use Stooq daily closes. BUY score = forward return over horizon. SELL score = negative forward return.",
    }


# --------------------------
# Endpoint: SEC leaders (Leaders tab)
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
        "note": "Form 4 infers BUY/SELL from transaction codes (P/S). 13F is shown as filing events.",
    }
