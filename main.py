import os
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import quiverquant  # Quiver's python client

QUIVER_TOKEN = os.getenv("QUIVER_TOKEN")

app = FastAPI(title="Finance Signals Backend")

# Allow your GitHub Pages site to call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://hijazss.github.io"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def require_token():
    if not QUIVER_TOKEN:
        raise HTTPException(500, "Missing QUIVER_TOKEN env var")

@app.get("/health")
def health():
    return {"ok": True, "ts": datetime.now(timezone.utc).isoformat()}

@app.get("/report/today")
def report_today():
    """
    Returns a JSON payload shaped like your PWA UI expects.
    For now: "live" = latest available from Quiver (still subject to filing delays).
    """
    require_token()

    q = quiverquant.quiver(QUIVER_TOKEN)

    # Pull latest congress trades (Quiver method shown in their README)
    # You can also filter by ticker: q.congress_trading("NVDA")
    df = q.congress_trading()

    # Defensive: if empty
    if df is None or len(df) == 0:
        return {
            "date": datetime.now(timezone.utc).date().isoformat(),
            "convergence": [],
            "bipartisanTickers": [],
            "fundSignals": [],
        }

    # Normalize columns (Quiver returns a dataframe; columns may vary by dataset updates)
    # We'll try common column names used in Quiver congress trading outputs.
    # If a column is missing, fill with None.
    def col(name, default=None):
        return df[name] if name in df.columns else default

    # Make a simple bipartisan “same ticker bought by both parties” grouping for recent filings.
    # You can tighten this later (time windows, sizes, etc.)
    df2 = df.copy()
    if "Ticker" in df2.columns:
        df2["Ticker"] = df2["Ticker"].astype(str).str.upper()

    # Party column names differ in some exports; try a few possibilities
    party_series = None
    for pn in ["Party", "party", "PoliticianParty"]:
        if pn in df2.columns:
            party_series = df2[pn].astype(str)
            break
    if party_series is None:
        party_series = None

    # Transaction type (Purchase/Sale)
    tx_series = None
    for tn in ["Transaction", "TransactionType", "Type"]:
        if tn in df2.columns:
            tx_series = df2[tn].astype(str)
            break

    # Filter to purchases if we can identify them
    if tx_series is not None:
        buy_mask = tx_series.str.contains("purchase", case=False, na=False) | tx_series.str.contains("buy", case=False, na=False)
        df_buy = df2[buy_mask].copy()
    else:
        df_buy = df2.copy()

    # Build bipartisan list: tickers with >=1 Dem and >=1 Rep purchase in recent filings
    bipartisan = []
    if party_series is not None and "Ticker" in df_buy.columns:
        df_buy["_party"] = party_series.loc[df_buy.index].str.lower()
        grouped = df_buy.groupby("Ticker")["_party"].agg(list)
        for ticker, parties in grouped.items():
            has_dem = any("dem" in p for p in parties)
            has_rep = any("rep" in p for p in parties)
            if has_dem and has_rep:
                bipartisan.append(ticker)

    # Build cards
    def make_card(ticker: str):
        sub = df_buy[df_buy["Ticker"] == ticker] if "Ticker" in df_buy.columns else df_buy
        dem = 0
        rep = 0
        if party_series is not None:
            parties = sub["_party"].tolist() if "_party" in sub.columns else party_series.loc[sub.index].astype(str).str.lower().tolist()
            dem = sum(1 for p in parties if "dem" in p)
            rep = sum(1 for p in parties if "rep" in p)

        return {
            "ticker": ticker,
            "companyName": "",
            "demBuyers": dem,
            "repBuyers": rep,
            "funds": [],
            "lastFiledAt": "",
            "strength": "LIVE",
        }

    bipartisan_cards = [make_card(t) for t in bipartisan[:25]]

    return {
        "date": datetime.now(timezone.utc).date().isoformat(),
        "convergence": bipartisan_cards[:10],      # for now, treat bipartisan overlap as "convergence"
        "bipartisanTickers": bipartisan_cards[:25],
        "fundSignals": [],                         # we’ll add SEC 13D/13G + 13F next
    }
