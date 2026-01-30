import os
import re
from datetime import datetime, timezone
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

@app.get("/")
def root():
    return {"status": "ok"}

_party_re = re.compile(r"/\s*([DR])\b", re.IGNORECASE)

def norm_party_from_any(row: dict) -> str:
    # 1) Direct Party column if present
    for k in ["Party", "party"]:
        v = row.get(k)
        if v:
            s = str(v).strip().upper()
            if s.startswith("D"):
                return "D"
            if s.startswith("R"):
                return "R"

    # 2) Parse from Politician field like "Gilbert Cisneros House / D"
    for k in ["Politician", "politician", "Representative", "Senator", "Name", "name"]:
        v = row.get(k)
        if not v:
            continue
        s = str(v)
        m = _party_re.search(s)
        if m:
            return m.group(1).upper()

        # fallback: endswith " D" / " R"
        s2 = s.strip().upper()
        if s2.endswith(" D"):
            return "D"
        if s2.endswith(" R"):
            return "R"

    return ""

def norm_ticker(row: dict) -> str:
    # Quiver python-api shows congress_trading() exists; fields vary.  [oai_citation:1‡GitHub](https://github.com/Quiver-Quantitative/python-api)
    for k in ["Ticker", "ticker", "Stock", "stock", "Symbol", "symbol"]:
        v = row.get(k)
        if not v:
            continue
        s = str(v).strip().upper()

        # If "Stock" is a combined field, try to grab leading ticker token
        # Example rows often show "APG API GROUP..." where first token is ticker.  [oai_citation:2‡Quiver Quantitative](https://www.quiverquant.com/congresstrading/stock/APG?utm_source=chatgpt.com)
        first = s.split()[0]
        if 1 <= len(first) <= 6 and first.isalnum():
            return first
        return s
    return ""

def is_purchase(row: dict) -> bool:
    for k in ["Transaction", "transaction", "TransactionType", "Type", "type"]:
        v = row.get(k)
        if not v:
            continue
        s = str(v).lower()
        if "purchase" in s or "buy" in s:
            return True
    return False

def pick_first(row: dict, keys: list[str], default=""):
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return default

@app.get("/report/today")
def report_today():
    if not QUIVER_TOKEN:
        raise HTTPException(500, "QUIVER_TOKEN missing")

    q = quiverquant.quiver(QUIVER_TOKEN)
    df = q.congress_trading()  # should return recent trades  [oai_citation:3‡GitHub](https://github.com/Quiver-Quantitative/python-api)

    if df is None or len(df) == 0:
        return {
            "date": datetime.now(timezone.utc).date().isoformat(),
            "convergence": [],
            "bipartisanTickers": [],
            "fundSignals": [],
        }

    # dataframe -> list[dict]
    try:
        rows = df.to_dict(orient="records")
    except Exception:
        rows = list(df)

    items = []
    for r in rows:
        ticker = norm_ticker(r)
        party = norm_party_from_any(r)
        if not ticker or not party:
            continue
        if not is_purchase(r):
            continue

        filed = str(pick_first(r, ["Filed", "filed", "ReportDate", "report_date", "Date", "date"], "")).strip()
        traded = str(pick_first(r, ["Traded", "traded", "TransactionDate", "transaction_date"], "")).strip()
        pol = str(pick_first(r, ["Politician", "politician", "Representative", "Senator", "Name", "name"], "")).strip()
        desc = str(pick_first(r, ["Description", "description"], "")).strip()

        items.append({
            "ticker": ticker,
            "party": party,          # D or R
            "filed": filed,
            "traded": traded,
            "politician": pol,
            "description": desc,
        })

    # If Quiver returned data but we filtered too hard, surface that (helps debugging)
    if not items:
        return {
            "date": datetime.now(timezone.utc).date().isoformat(),
            "convergence": [],
            "bipartisanTickers": [],
            "fundSignals": [],
            "note": "No purchases matched after parsing. Check raw columns via /debug/columns and /debug/sample."
        }

    # Sort by filed (string sort works fine for many formats, otherwise it’s still “recent-ish”)
    items.sort(key=lambda x: x["filed"], reverse=True)

    dem = [x for x in items if x["party"] == "D"]
    rep = [x for x in items if x["party"] == "R"]

    # Build cards for frontend
    def card_from(x):
        return {
            "ticker": x["ticker"],
            "companyName": x["politician"],  # shows who made the trade (useful immediately)
            "demBuyers": 1 if x["party"] == "D" else 0,
            "repBuyers": 1 if x["party"] == "R" else 0,
            "funds": [],
            "lastFiledAt": x["filed"] or x["traded"],
            "strength": "BUY",
        }

    # Interleave Dem + Rep so it feels “both sides”
    mixed = []
    for i in range(0, 30):
        if i < len(dem):
            mixed.append(card_from(dem[i]))
        if i < len(rep):
            mixed.append(card_from(rep[i]))
        if len(mixed) >= 60:
            break

    # Overlap tickers (bonus)
    dem_t = set(x["ticker"] for x in dem)
    rep_t = set(x["ticker"] for x in rep)
    overlap = sorted(list(dem_t.intersection(rep_t)))

    overlap_cards = []
    for t in overlap[:20]:
        dem_ct = sum(1 for x in dem if x["ticker"] == t)
        rep_ct = sum(1 for x in rep if x["ticker"] == t)
        last = ""
        for x in items:
            if x["ticker"] == t and (x["filed"] or x["traded"]):
                last = x["filed"] or x["traded"]
                break
        overlap_cards.append({
            "ticker": t,
            "companyName": "",
            "demBuyers": dem_ct,
            "repBuyers": rep_ct,
            "funds": [],
            "lastFiledAt": last,
            "strength": "OVERLAP",
        })

    return {
        "date": datetime.now(timezone.utc).date().isoformat(),
        "convergence": overlap_cards,
        "bipartisanTickers": mixed,
        "fundSignals": [],
    }

# Optional debug helpers (no frontend changes needed)
@app.get("/debug/columns")
def debug_columns():
    if not QUIVER_TOKEN:
        raise HTTPException(500, "QUIVER_TOKEN missing")
    q = quiverquant.quiver(QUIVER_TOKEN)
    df = q.congress_trading()
    try:
        cols = list(df.columns)
    except Exception:
        cols = []
    return {"columns": cols}

@app.get("/debug/sample")
def debug_sample():
    if not QUIVER_TOKEN:
        raise HTTPException(500, "QUIVER_TOKEN missing")
    q = quiverquant.quiver(QUIVER_TOKEN)
    df = q.congress_trading()
    try:
        rows = df.to_dict(orient="records")[:5]
    except Exception:
        rows = []
    return {"sample": rows}
