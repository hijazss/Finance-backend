import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

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

_party_re = re.compile(r"/\s*([DR])\b", re.IGNORECASE)


@app.get("/")
def root():
    return {"status": "ok"}


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
            dt = datetime.strptime(s[:len(fmt.replace("%f", "000000"))], fmt)
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


def agg_by_ticker(trades: List[dict]) -> Dict[str, dict]:
    """
    Aggregate trades into per-ticker counts suitable for Yearly tab:
    demBuyers/repBuyers/demSellers/repSellers as counts of rows,
    and lastFiledAt as latest dt.
    """
    out: Dict[str, dict] = {}
    for x in trades:
        t = x["ticker"]
        cur = out.get(t)
        if not cur:
            cur = {
                "ticker": t,
                "companyName": "",
                "demBuyers": 0,
                "repBuyers": 0,
                "demSellers": 0,
                "repSellers": 0,
                "funds": [],
                "lastFiledAt": "",
                "strength": "",
                "_latest": None,
            }
            out[t] = cur

        party = x["party"]
        kind = x["kind"]
        if kind == "BUY":
            if party == "D":
                cur["demBuyers"] += 1
            elif party == "R":
                cur["repBuyers"] += 1
        elif kind == "SELL":
            if party == "D":
                cur["demSellers"] += 1
            elif party == "R":
                cur["repSellers"] += 1

        dt = x.get("best_dt")
        if dt is not None:
            if cur["_latest"] is None or dt > cur["_latest"]:
                cur["_latest"] = dt
                cur["lastFiledAt"] = iso_date_only(dt)

    # remove internal
    for v in out.values():
        v.pop("_latest", None)

    return out


@app.get("/report/today")
def report_today():
    if not QUIVER_TOKEN:
        raise HTTPException(500, "QUIVER_TOKEN missing")

    # Windows
    now = datetime.now(timezone.utc)
    window30 = 30
    window365 = 365
    since30 = now - timedelta(days=window30)
    since365 = now - timedelta(days=window365)

    q = quiverquant.quiver(QUIVER_TOKEN)
    df = q.congress_trading()

    if df is None or len(df) == 0:
        return {
            "date": now.date().isoformat(),
            "windowDays": window30,
            "windowStart": since30.date().isoformat(),
            "windowEnd": now.date().isoformat(),
            "convergence": [],
            "bipartisanTickers": [],
            "politicianBuys": [],
            "politicianSells": [],
            "yearly": {"windowDays": window365, "windowStart": since365.date().isoformat(), "windowEnd": now.date().isoformat(), "universe": []},
            "fundSignals": [],
        }

    try:
        rows = df.to_dict(orient="records")
    except Exception:
        rows = list(df)

    trades_30: List[dict] = []
    trades_365: List[dict] = []

    for r in rows:
        best_dt = row_best_dt(r)
        if best_dt is None:
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

        base = {
            "ticker": ticker,
            "party": party,
            "filed": iso_date_only(filed_dt),
            "traded": iso_date_only(traded_dt),
            "politician": pol,
            "best_dt": best_dt,
            "kind": kind,
        }

        if since365 <= best_dt <= now:
            trades_365.append(base)

        if since30 <= best_dt <= now:
            trades_30.append(base)

    # 30D buy/sell cards (mixed)
    buys_30 = [x for x in trades_30 if x["kind"] == "BUY"]
    sells_30 = [x for x in trades_30 if x["kind"] == "SELL"]

    buys_30.sort(key=lambda x: x["best_dt"], reverse=True)
    sells_30.sort(key=lambda x: x["best_dt"], reverse=True)

    buy_mixed = interleave([{"ticker": x["ticker"], "party": x["party"], "filed": x["filed"], "traded": x["traded"], "politician": x["politician"]} for x in buys_30], 80)
    sell_mixed = interleave([{"ticker": x["ticker"], "party": x["party"], "filed": x["filed"], "traded": x["traded"], "politician": x["politician"]} for x in sells_30], 80)

    buy_cards = [to_card(x, "BUY") for x in buy_mixed]
    sell_cards = [to_card(x, "SELL") for x in sell_mixed]

    # 30D convergence (overlap tickers on BUY)
    dem_buy = [x for x in buys_30 if x["party"] == "D"]
    rep_buy = [x for x in buys_30 if x["party"] == "R"]
    overlap = sorted(set(x["ticker"] for x in dem_buy).intersection(set(x["ticker"] for x in rep_buy)))

    overlap_cards: List[dict] = []
    for t in overlap:
        dem_ct = sum(1 for x in dem_buy if x["ticker"] == t)
        rep_ct = sum(1 for x in rep_buy if x["ticker"] == t)

        latest_dt: Optional[datetime] = None
        for x in buys_30:
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

    # Yearly universe aggregation (per ticker totals in 1Y window)
    yearly_map = agg_by_ticker(trades_365)
    yearly_universe = list(yearly_map.values())

    # Sort yearly universe by total participation then ticker
    def total_participation(v: dict) -> int:
        return int(v.get("demBuyers", 0)) + int(v.get("repBuyers", 0)) + int(v.get("demSellers", 0)) + int(v.get("repSellers", 0))

    yearly_universe.sort(key=lambda v: (-total_participation(v), v.get("ticker", "")))

    return {
        "date": now.date().isoformat(),

        # 30D metadata
        "windowDays": window30,
        "windowStart": since30.date().isoformat(),
        "windowEnd": now.date().isoformat(),

        # 30D data
        "convergence": overlap_cards[:10],
        "bipartisanTickers": buy_cards[:60],  # backward compatible key
        "politicianBuys": buy_cards[:80],
        "politicianSells": sell_cards[:80],

        # 1Y data for Yearly tab
        "yearly": {
            "windowDays": window365,
            "windowStart": since365.date().isoformat(),
            "windowEnd": now.date().isoformat(),
            "universe": yearly_universe[:200],  # keep payload sane
        },

        "fundSignals": [],
    }
