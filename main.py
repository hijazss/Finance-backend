import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, HTTPException
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

# Rolling windows
WINDOW_30_DAYS = 30
WINDOW_180_DAYS = 180
WINDOW_2Y_DAYS = 730
WINDOW_365_DAYS = 365

_party_re = re.compile(r"/\s*([DR])\b", re.IGNORECASE)

# --------------------------
# Small in-memory caches
# --------------------------
_CACHE: Dict[str, Tuple[float, Any]] = {}
CACHE_TTL_SECONDS = 60 * 60 * 6  # 6 hours


def cache_get(key: str):
    v = _CACHE.get(key)
    if not v:
        return None
    ts, payload = v
    if time.time() - ts > CACHE_TTL_SECONDS:
        _CACHE.pop(key, None)
        return None
    return payload


def cache_set(key: str, payload: Any):
    _CACHE[key] = (time.time(), payload)


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
    try:
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        pass

    fmts = ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"]
    for fmt in fmts:
        try:
            dt = datetime.strptime(s[:10], "%Y-%m-%d")
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


# --------------------------
# Price helper (fast + cached)
# Uses Stooq free CSV. For US tickers, use {ticker}.us
# --------------------------
def stooq_symbol(ticker: str) -> str:
    t = (ticker or "").strip().lower()
    if not t:
        return ""
    # crude but good enough for US equities
    if t.endswith(".us") or t in ["spy.us", "^spx"]:
        return t
    # handle BRK.B -> brk.b.us
    return f"{t}.us"


def fetch_stooq_history_close(ticker: str) -> List[Tuple[datetime, float]]:
    """
    Returns list of (date_utc, close) sorted ascending.
    Cached aggressively.
    """
    t = ticker.upper().strip()
    if not t:
        return []

    cache_key = f"stooq_hist::{t}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    sym = stooq_symbol(t)
    if not sym:
        return []

    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            cache_set(cache_key, [])
            return []
        text = r.text.strip()
        if not text or "Date,Open,High,Low,Close,Volume" not in text:
            cache_set(cache_key, [])
            return []

        out: List[Tuple[datetime, float]] = []
        lines = text.splitlines()
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) < 5:
                continue
            d = parts[0].strip()
            c = parts[4].strip()
            try:
                dt = datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                close = float(c)
                out.append((dt, close))
            except Exception:
                continue

        out.sort(key=lambda x: x[0])
        cache_set(cache_key, out)
        return out
    except Exception:
        cache_set(cache_key, [])
        return []


def nearest_close_on_or_after(series: List[Tuple[datetime, float]], dt: datetime) -> Optional[Tuple[datetime, float]]:
    if not series:
        return None
    target = dt.date()
    for d, c in series:
        if d.date() >= target:
            return (d, c)
    return None


def forward_return(series: List[Tuple[datetime, float]], dt: datetime, horizon_days: int) -> Optional[float]:
    """
    Return % change from close at/after dt to close at/after dt+horizon.
    """
    if not series:
        return None
    p0 = nearest_close_on_or_after(series, dt)
    if not p0:
        return None
    dt2 = dt + timedelta(days=horizon_days)
    p1 = nearest_close_on_or_after(series, dt2)
    if not p1:
        return None
    _, c0 = p0
    _, c1 = p1
    if c0 <= 0:
        return None
    return (c1 - c0) / c0


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
# Core extraction from Quiver with correct rolling window behavior
# IMPORTANT: filter on FILED date first, then TRADED date if filed missing
# --------------------------
def extract_congress_rows(window_days: int) -> Dict[str, Any]:
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
        if not ticker or not party:
            continue

        tx = tx_text(r)
        kind = "BUY" if is_buy(tx) else "SELL" if is_sell(tx) else ""
        if not kind:
            continue

        filed_dt = parse_dt_any(pick_first(r, ["Filed", "filed", "ReportDate", "report_date", "Date", "date"], ""))
        traded_dt = parse_dt_any(pick_first(r, ["Traded", "traded", "TransactionDate", "transaction_date"], ""))

        # KEY FIX: prefer filed date for window filtering, fall back to traded only if filed missing
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

    # Convergence: overlap tickers on BUY
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
        "date": now.date().isoformat(),
        "windowDays": window_days,
        "windowStart": since.date().isoformat(),
        "windowEnd": now.date().isoformat(),
        "convergence": overlap_cards[:20],
        "politicianBuys": buy_cards[:120],
        "politicianSells": sell_cards[:120],
    }


# --------------------------
# Endpoint: Congress (30D main + includes 180D block)
# --------------------------
@app.get("/report/today")
def report_today():
    # 30D primary
    main30 = extract_congress_rows(WINDOW_30_DAYS)

    # 180D secondary (for your 30D/180D toggle)
    main180 = extract_congress_rows(WINDOW_180_DAYS)

    return {
        **main30,
        "windows": {
            "d30": {
                "windowDays": main30["windowDays"],
                "windowStart": main30["windowStart"],
                "windowEnd": main30["windowEnd"],
                "convergence": main30["convergence"],
            },
            "d180": {
                "windowDays": main180["windowDays"],
                "windowStart": main180["windowStart"],
                "windowEnd": main180["windowEnd"],
                "convergence": main180["convergence"],
            },
        },
        # Backward compatible keys some older frontends used
        "bipartisanTickers": main30.get("politicianBuys", [])[:60],
        "fundSignals": [],
    }


# --------------------------
# Endpoint: Performance 2Y (fast, capped, cached)
# Score: BUY gets forward horizon return minus SPY. SELL gets negative forward return minus SPY.
# Uses filed date first (same logic as above).
# --------------------------
@app.get("/report/performance-2y")
def report_performance_2y(horizon_days: int = 30, top_n: int = 5):
    if horizon_days < 5 or horizon_days > 180:
        raise HTTPException(400, "horizon_days must be between 5 and 180")

    cache_key = f"perf2y::{horizon_days}::{top_n}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    if not QUIVER_TOKEN:
        raise HTTPException(500, "QUIVER_TOKEN missing")

    now = datetime.now(timezone.utc)
    since = now - timedelta(days=WINDOW_2Y_DAYS)

    q = quiverquant.quiver(QUIVER_TOKEN)
    df = q.congress_trading()
    if df is None or len(df) == 0:
        payload = {
            "date": now.date().isoformat(),
            "windowDays": WINDOW_2Y_DAYS,
            "windowStart": since.date().isoformat(),
            "windowEnd": now.date().isoformat(),
            "horizonDays": horizon_days,
            "leaders": [],
            "note": "No congress trades returned from Quiver.",
        }
        cache_set(cache_key, payload)
        return payload

    try:
        rows = df.to_dict(orient="records")
    except Exception:
        rows = list(df)

    # Build a list of scored trade records (capped for speed)
    # We cap number of rows processed to avoid Render timeout.
    MAX_TRADES_TO_SCORE = 600  # keep it fast
    trades: List[dict] = []

    for r in rows:
        if len(trades) >= MAX_TRADES_TO_SCORE:
            break

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
        if best_dt is None or best_dt < since or best_dt > now:
            continue

        pol = str(pick_first(r, ["Politician", "politician", "Representative", "Senator", "Name", "name"], "")).strip()
        chamber = str(pick_first(r, ["Chamber", "chamber"], "")).strip()
        amount = str(pick_first(r, ["Range", "range", "Amount", "amount"], "")).strip()

        trades.append(
            {
                "politician": pol,
                "party": party,
                "chamber": chamber,
                "ticker": ticker.upper(),
                "kind": kind,
                "dt": best_dt,
                "filed": iso_date_only(filed_dt),
                "traded": iso_date_only(traded_dt),
                "amount": amount,
            }
        )

    # Group by politician, score vs SPY
    spy_series = fetch_stooq_history_close("SPY")
    if not spy_series:
        raise HTTPException(502, "Could not fetch SPY price history (needed for performance scoring)")

    score_map: Dict[str, dict] = {}

    # Cache price histories for tickers used
    ticker_series_cache: Dict[str, List[Tuple[datetime, float]]] = {}

    for tr in trades:
        t = tr["ticker"]
        if t not in ticker_series_cache:
            ticker_series_cache[t] = fetch_stooq_history_close(t)

        series = ticker_series_cache[t]
        if not series:
            continue

        r_ticker = forward_return(series, tr["dt"], horizon_days)
        r_spy = forward_return(spy_series, tr["dt"], horizon_days)
        if r_ticker is None or r_spy is None:
            continue

        # BUY: +return; SELL: penalize if it goes up after they sold, reward if it drops
        # Score is excess over SPY on same dates.
        excess = (r_ticker - r_spy)
        if tr["kind"] == "SELL":
            excess = (-r_ticker) - (-r_spy)  # equivalent to (r_spy - r_ticker)

        key = tr["politician"] or "Unknown"
        cur = score_map.get(key)
        if not cur:
            cur = {
                "name": key,
                "party": tr["party"],
                "chamber": tr["chamber"],
                "tradesScored": 0,
                "score": 0.0,
                "examples": [],
            }
            score_map[key] = cur

        cur["tradesScored"] += 1
        cur["score"] += float(excess)

        # keep a few examples for UI
        if len(cur["examples"]) < 4:
            cur["examples"].append(
                {
                    "ticker": t,
                    "kind": tr["kind"],
                    "filed": tr["filed"],
                    "traded": tr["traded"],
                    "amount": tr["amount"],
                    "excessVsSPY": round(excess, 4),
                }
            )

    leaders = list(score_map.values())
    # Normalize: require minimum number of scored trades so we don’t rank noise
    MIN_SCORED = 3
    leaders = [x for x in leaders if x["tradesScored"] >= MIN_SCORED]
    leaders.sort(key=lambda x: x["score"], reverse=True)
    leaders = leaders[: max(1, min(int(top_n), 20))]

    payload = {
        "date": now.date().isoformat(),
        "windowDays": WINDOW_2Y_DAYS,
        "windowStart": since.date().isoformat(),
        "windowEnd": now.date().isoformat(),
        "horizonDays": horizon_days,
        "leaders": leaders,
        "note": "Performance is scored vs SPY using forward horizon returns on filed date (or traded date if filed missing). BUY adds excess return; SELL rewards stocks that fall after selling. Capped to keep response fast.",
    }
    cache_set(cache_key, payload)
    return payload


# --------------------------
# Endpoint: Public filings “Leaders” (SEC EDGAR, rolling 365D)
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
        "note": "Form 4 events infer BUY/SELL from transaction codes (P/S). 13F is quarterly and shown as filing events unless holdings deltas are computed.",
    }
