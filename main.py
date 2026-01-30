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

def pick_first(row: dict, keys: list[str], default=""):
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return default

def to_card(x, kind: str):
    # kind: "BUY" or "SELL"
    dem = 1 if x["party"] == "D" else 0
    rep = 1 if x["party"] == "R" else 0
    return {
        "ticker": x["ticker"],
        "companyName": x["politician"],  # show who traded
        "demBuyers": dem if kind == "BUY" else 0,
        "repBuyers": rep if kind == "BUY" else 0,
        "demSellers": dem if kind == "SELL" else 0,
        "repSellers": rep if kind == "SELL" else 0,
        "funds": [],
        "lastFiledAt": x["filed"] or x["traded"],
        "strength": kind,
    }

@app.get("/report/today")
def report_today():
    if not QUIVER_TOKEN:
        raise HTTPException(500, "QUIVER_TOKEN missing")

    q = quiverquant.quiver(QUIVER_TOKEN)
    df = q.congress_trading()

    if df is None or len(df) == 0:
        return {
            "date": datetime.now(timezone.utc).date().isoformat(),
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

    buys = []
    sells = []

    for r in rows:
        ticker = norm_ticker(r)
        party = norm_party_from_any(r)
        if not ticker or not party:
            continue

        tx = tx_text(r)
        kind = "BUY" if is_buy(tx) else "SELL" if is_sell(tx) else ""
        if not kind:
            continue

        filed = str(pick_first(r, ["Filed", "filed", "ReportDate", "report_date", "Date", "date"], "")).strip()
        traded = str(pick_first(r, ["Traded", "traded", "TransactionDate", "transaction_date"], "")).strip()
        pol = str(pick_first(r, ["Politician", "politician", "Representative", "Senator", "Name", "name"], "")).strip()

        item = {
            "ticker": ticker,
            "party": party,
            "filed": filed,
            "traded": traded,
            "politician": pol,
        }

        if kind == "BUY":
            buys.append(item)
        else:
            sells.append(item)

    # Sort newest first by filed string
    buys.sort(key=lambda x: x["filed"], reverse=True)
    sells.sort(key=lambda x: x["filed"], reverse=True)

    # Interleave D and R for readability
    def interleave(items, limit=60):
        d = [x for x in items if x["party"] == "D"]
        r = [x for x in items if x["party"] == "R"]
        out = []
        for i in range(0, max(len(d), len(r))):
            if i < len(d): out.append(d[i])
            if i < len(r): out.append(r[i])
            if len(out) >= limit: break
        return out

    buy_mixed = interleave(buys, 80)
    sell_mixed = interleave(sells, 80)

    buy_cards = [to_card(x, "BUY") for x in buy_mixed]
    sell_cards = [to_card(x, "SELL") for x in sell_mixed]

    # Overlap tickers for BUY side only
    dem_buy = [x for x in buys if x["party"] == "D"]
    rep_buy = [x for x in buys if x["party"] == "R"]
    overlap = sorted(list(set(x["ticker"] for x in dem_buy).intersection(set(x["ticker"] for x in rep_buy))))

    overlap_cards = []
    for t in overlap[:25]:
        dem_ct = sum(1 for x in dem_buy if x["ticker"] == t)
        rep_ct = sum(1 for x in rep_buy if x["ticker"] == t)
        last = ""
        for x in buys:
            if x["ticker"] == t and (x["filed"] or x["traded"]):
                last = x["filed"] or x["traded"]
                break
        overlap_cards.append({
            "ticker": t,
            "companyName": "",
            "demBuyers": dem_ct,
            "repBuyers": rep_ct,
            "demSellers": 0,
            "repSellers": 0,
            "funds": [],
            "lastFiledAt": last,
            "strength": "OVERLAP",
        })

    return {
        "date": datetime.now(timezone.utc).date().isoformat(),
        "convergence": overlap_cards[:10],

        # Keep existing key for backward compatibility (will show buys)
        "bipartisanTickers": buy_cards[:60],

        # New keys
        "politicianBuys": buy_cards[:80],
        "politicianSells": sell_cards[:80],

        "fundSignals": [],
    }
