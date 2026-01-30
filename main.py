import os
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import quiverquant

app = FastAPI(title="Finance Signals Backend")

# Allow your GitHub Pages frontend
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

def _norm_party(p: str) -> str:
    s = (p or "").strip().lower()
    if "dem" in s:
        return "D"
    if "rep" in s:
        return "R"
    return ""

def _is_purchase(tx: str) -> bool:
    s = (tx or "").strip().lower()
    return ("purchase" in s) or ("buy" in s)

def _first_existing(d: dict, keys: list[str], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

def _to_float_range(amount_str: str):
    # Quiver often provides ranges like "$1,001 - $15,000"
    # We will keep it simple and just return the raw string.
    return (amount_str or "").strip()

@app.get("/report/today")
def report_today():
    if not QUIVER_TOKEN:
        raise HTTPException(500, "QUIVER_TOKEN missing")

    q = quiverquant.quiver(QUIVER_TOKEN)

    # Pull the latest congress trading table
    df = q.congress_trading()
    if df is None or len(df) == 0:
        return {
            "date": datetime.now(timezone.utc).date().isoformat(),
            "convergence": [],
            "bipartisanTickers": [],
            "fundSignals": [],
        }

    # Convert dataframe rows to dicts, without depending on pandas explicitly
    try:
        rows = df.to_dict(orient="records")
    except Exception:
        # Fallback if df isn't a pandas DF for some reason
        rows = list(df)

    # Normalize columns across possible Quiver formats
    normalized = []
    for r in rows:
        ticker = str(_first_existing(r, ["Ticker", "ticker"], "") or "").upper().strip()
        party = _norm_party(str(_first_existing(r, ["Party", "party"], "") or ""))
        tx = str(_first_existing(r, ["Transaction", "TransactionType", "Type", "transaction"], "") or "")
        is_buy = _is_purchase(tx)

        # Date columns vary
        filed = _first_existing(r, ["ReportDate", "report_date", "Date", "date", "TransactionDate", "transaction_date"], "")
        # Amount columns vary
        amount = _first_existing(r, ["Amount", "amount", "Range", "range"], "")

        # Optional politician name
        pol = _first_existing(r, ["Representative", "Senator", "Politician", "Name", "name"], "")

        if not ticker or not party:
            continue
        if not is_buy:
            continue

        normalized.append({
            "ticker": ticker,
            "party": party,  # "D" or "R"
            "filed": str(filed or "").strip(),
            "amount": _to_float_range(str(amount or "")),
            "politician": str(pol or "").strip(),
        })

    # Sort newest first if we have a usable date; otherwise keep order
    def sort_key(x):
        # Try ISO dates first, fall back to string
        return x["filed"] or ""

    normalized.sort(key=sort_key, reverse=True)

    dem_buys = [x for x in normalized if x["party"] == "D"]
    rep_buys = [x for x in normalized if x["party"] == "R"]

    # Build "latest buys" cards for your frontend
    def card_from(x):
        # Keep fields your frontend already knows
        return {
            "ticker": x["ticker"],
            "companyName": "",
            "demBuyers": 1 if x["party"] == "D" else 0,
            "repBuyers": 1 if x["party"] == "R" else 0,
            "funds": [],
            "lastFiledAt": x["filed"],
            "strength": "LIVE",
        }

    # Build overlap tickers (bonus section)
    dem_tickers = set(x["ticker"] for x in dem_buys)
    rep_tickers = set(x["ticker"] for x in rep_buys)
    overlap = sorted(list(dem_tickers.intersection(rep_tickers)))

    overlap_cards = []
    for t in overlap[:25]:
        # Count number of buys by each side for that ticker in this pull
        dem_ct = sum(1 for x in dem_buys if x["ticker"] == t)
        rep_ct = sum(1 for x in rep_buys if x["ticker"] == t)
        last = ""
        # Get latest filed date we saw for this ticker
        for x in normalized:
            if x["ticker"] == t and x["filed"]:
                last = x["filed"]
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

    # Option A output mapping:
    # - convergence: show the OVERLAP tickers (when they exist)
    # - bipartisanTickers: show latest buys from BOTH sides (mixed list)
    latest_mixed = []
    # Interleave D and R so the list looks "both sides" on screen
    for i in range(0, 20):
        if i < len(dem_buys):
            latest_mixed.append(card_from(dem_buys[i]))
        if i < len(rep_buys):
            latest_mixed.append(card_from(rep_buys[i]))
        if len(latest_mixed) >= 40:
            break

    return {
        "date": datetime.now(timezone.utc).date().isoformat(),
        "convergence": overlap_cards[:10],
        "bipartisanTickers": latest_mixed[:40],
        "fundSignals": [],
    }
