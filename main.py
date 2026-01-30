import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

import requests
from fastapi import FastAPI, HTTPException, Query
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
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY", "")
SOLSCAN_API_KEY = os.getenv("SOLSCAN_API_KEY", "")

SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "FinanceApp/1.0 (contact: hijazss@gmail.com)")
SEC_TIMEOUT = 20

_party_re = re.compile(r"/\s*([DR])\b", re.IGNORECASE)


@app.get("/")
def root():
    return {"status": "ok"}


# --------------------------
# Generic helpers
# --------------------------

def pick_first(row: dict, keys: List[str], default=""):
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return default


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
        "%m/%d/%Y",
        "%m/%d/%y",
        "%m/%d/%Y %H:%M:%S",
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return None


def iso_date_only(dt: Optional[datetime]) -> str:
    return dt.date().isoformat() if dt else ""


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


def row_best_dt(row: dict) -> Optional[datetime]:
    traded = pick_first(row, ["Traded", "traded", "TransactionDate", "transaction_date"], "")
    filed = pick_first(row, ["Filed", "filed", "ReportDate", "report_date", "Date", "date"], "")
    return parse_dt_any(traded) or parse_dt_any(filed)


def interleave(items: List[dict], limit: int = 120) -> List[dict]:
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
        "ticker": x.get("ticker", ""),
        "companyName": who,
        "demBuyers": dem if kind == "BUY" else 0,
        "repBuyers": rep if kind == "BUY" else 0,
        "demSellers": dem if kind == "SELL" else 0,
        "repSellers": rep if kind == "SELL" else 0,
        "funds": [],
        "lastFiledAt": last,
        "strength": kind,
        "chamber": x.get("chamber", ""),
        "amountRange": x.get("amountRange", ""),
        "traded": x.get("traded", ""),
        "filed": x.get("filed", ""),
        "description": x.get("description", ""),
    }


# --------------------------
# Congress rolling window
# Supports 30 / 180 / 365 by query param
# --------------------------

@app.get("/report/today")
def report_today(window_days: int = Query(30, ge=1, le=365)):
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
            "cryptoDisclosures": {"direct": [], "linked": []},
        }

    try:
        rows = df.to_dict(orient="records")
    except Exception:
        rows = list(df)

    buys: List[dict] = []
    sells: List[dict] = []
    crypto_direct: List[dict] = []
    crypto_linked: List[dict] = []

    crypto_words = [
        "bitcoin", "btc",
        "ethereum", "eth",
        "solana", "sol",
        "chainlink", "link",
        "dogecoin", "doge",
        "litecoin", "ltc",
        "xrp", "ripple",
        "avalanche", "avax",
        "polygon", "matic",
        "bnb", "binance",
        "cardano", "ada",
    ]

    crypto_linked_tickers = {
        "GBTC","ETHE","IBIT","FBTC","ARKB","BITO","BTCO","HODL","BTF",
        "MSTR","COIN"
    }

    for r in rows:
        best_dt = row_best_dt(r)
        if best_dt is None:
            continue
        if best_dt < since or best_dt > now:
            continue

        party = norm_party_from_any(r)
        if not party:
            continue

        tx = tx_text(r)
        kind = "BUY" if is_buy(tx) else "SELL" if is_sell(tx) else ""
        if not kind:
            continue

        ticker = norm_ticker(r)
        pol = str(pick_first(r, ["Politician","politician","Representative","Senator","Name","name"], "")).strip()
        chamber = str(pick_first(r, ["Chamber","chamber","Office","office"], "")).strip()
        amount = str(pick_first(r, ["Amount","amount","Range","range","AmountRange","amount_range"], "")).strip()

        filed_dt = parse_dt_any(pick_first(r, ["Filed","filed","ReportDate","report_date","Date","date"], ""))
        traded_dt = parse_dt_any(pick_first(r, ["Traded","traded","TransactionDate","transaction_date"], ""))

        desc = str(pick_first(r, ["AssetDescription","asset_description","Description","description","Asset","asset"], "")).strip()
        desc_l = desc.lower()

        item = {
            "ticker": ticker,
            "party": party,
            "filed": iso_date_only(filed_dt),
            "traded": iso_date_only(traded_dt),
            "politician": pol,
            "chamber": chamber,
            "amountRange": amount,
            "best_dt": best_dt,
            "description": desc,
        }

        if ticker:
            if kind == "BUY":
                buys.append(item)
            else:
                sells.append(item)

            if ticker in crypto_linked_tickers:
                crypto_linked.append({**item, "kind": kind})

        if desc and any(w in desc_l for w in crypto_words):
            sym = ""
            if "bitcoin" in desc_l or "btc" in desc_l:
                sym = "BTC"
            elif "ethereum" in desc_l or "eth" in desc_l:
                sym = "ETH"
            elif "solana" in desc_l or "sol" in desc_l:
                sym = "SOL"
            elif "chainlink" in desc_l or "link" in desc_l:
                sym = "LINK"

            crypto_direct.append({
                "crypto": sym or "CRYPTO",
                "description": desc,
                "party": party,
                "politician": pol,
                "chamber": chamber,
                "amountRange": amount,
                "traded": iso_date_only(traded_dt),
                "filed": iso_date_only(filed_dt),
                "kind": kind,
            })

    buys.sort(key=lambda x: x["best_dt"], reverse=True)
    sells.sort(key=lambda x: x["best_dt"], reverse=True)

    buy_cards = [to_card(x, "BUY") for x in interleave(buys, 120)]
    sell_cards = [to_card(x, "SELL") for x in interleave(sells, 120)]

    dem_buy = [x for x in buys if x["party"] == "D" and x.get("ticker")]
    rep_buy = [x for x in buys if x["party"] == "R" and x.get("ticker")]
    overlap = set(x["ticker"] for x in dem_buy) & set(x["ticker"] for x in rep_buy)

    overlap_cards: List[dict] = []
    for t in overlap:
        dem_ct = sum(1 for x in dem_buy if x["ticker"] == t)
        rep_ct = sum(1 for x in rep_buy if x["ticker"] == t)
        latest_dt = None
        for x in buys:
            if x.get("ticker") == t:
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

    crypto_direct.sort(key=lambda x: (x.get("filed") or ""), reverse=True)
    crypto_linked.sort(key=lambda x: x.get("best_dt") or datetime(1970,1,1,tzinfo=timezone.utc), reverse=True)

    return {
        "date": now.date().isoformat(),
        "windowDays": window_days,
        "windowStart": since.date().isoformat(),
        "windowEnd": now.date().isoformat(),
        "convergence": overlap_cards[:25],
        "politicianBuys": buy_cards,
        "politicianSells": sell_cards,
        "cryptoDisclosures": {
            "direct": crypto_direct[:300],
            "linked": crypto_linked[:300],
        },
    }


# --------------------------
# Top-10 crypto list (programmatic)
# --------------------------

def fetch_top10_crypto() -> List[dict]:
    """
    Uses CoinGecko markets endpoint for a simple top-10 by market cap.
    No key required for basic usage.
    """
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 10,
        "page": 1,
        "sparkline": "false",
    }
    r = requests.get(url, params=params, timeout=15)
    if r.status_code != 200:
        return []
    data = r.json()
    out = []
    for x in data:
        out.append({
            "id": x.get("id",""),
            "symbol": str(x.get("symbol","")).upper(),
            "name": x.get("name",""),
            "marketCap": x.get("market_cap", None),
        })
    return out


# --------------------------
# On-chain: BTC via mempool.space
# --------------------------

def btc_address_txs(address: str) -> List[dict]:
    address = address.strip()
    if not address:
        return []
    url = f"https://mempool.space/api/address/{address}/txs"
    r = requests.get(url, timeout=20)
    if r.status_code != 200:
        return []
    txs = r.json()
    out = []
    for tx in txs:
        status = tx.get("status", {}) or {}
        block_time = status.get("block_time")
        ts = None
        if block_time:
            ts = datetime.fromtimestamp(int(block_time), tz=timezone.utc)
        out.append({
            "hash": tx.get("txid",""),
            "timestamp": ts.isoformat() if ts else "",
            "confirmed": bool(status.get("confirmed", False)),
        })
    return out


# --------------------------
# On-chain: ETH + ERC20 via Etherscan
# --------------------------

def etherscan_get(params: Dict[str, str]) -> dict:
    if not ETHERSCAN_API_KEY:
        raise HTTPException(500, "ETHERSCAN_API_KEY missing")
    base = "https://api.etherscan.io/api"
    p = dict(params)
    p["apikey"] = ETHERSCAN_API_KEY
    r = requests.get(base, params=p, timeout=25)
    if r.status_code != 200:
        raise HTTPException(502, f"Etherscan HTTP {r.status_code}")
    return r.json()


def eth_normal_txs(address: str) -> List[dict]:
    js = etherscan_get({
        "module": "account",
        "action": "txlist",
        "address": address,
        "startblock": "0",
        "endblock": "99999999",
        "sort": "desc",
    })
    if str(js.get("status")) != "1":
        return []
    out = []
    for t in js.get("result", [])[:1000]:
        ts = datetime.fromtimestamp(int(t.get("timeStamp","0")), tz=timezone.utc)
        out.append({
            "hash": t.get("hash",""),
            "timestamp": ts.isoformat(),
            "from": t.get("from",""),
            "to": t.get("to",""),
            "valueWei": t.get("value","0"),
            "isError": t.get("isError","0"),
        })
    return out


def eth_erc20_txs(address: str, contract: Optional[str] = None) -> List[dict]:
    params = {
        "module": "account",
        "action": "tokentx",
        "address": address,
        "page": "1",
        "offset": "1000",
        "sort": "desc",
    }
    if contract:
        params["contractaddress"] = contract
    js = etherscan_get(params)
    if str(js.get("status")) != "1":
        return []
    out = []
    for t in js.get("result", []):
        ts = datetime.fromtimestamp(int(t.get("timeStamp","0")), tz=timezone.utc)
        out.append({
            "hash": t.get("hash",""),
            "timestamp": ts.isoformat(),
            "from": t.get("from",""),
            "to": t.get("to",""),
            "tokenSymbol": t.get("tokenSymbol",""),
            "tokenName": t.get("tokenName",""),
            "contract": t.get("contractAddress",""),
            "value": t.get("value",""),
            "decimals": t.get("tokenDecimal",""),
        })
    return out


# Chainlink ERC-20 contract on Ethereum mainnet
LINK_CONTRACT = "0x514910771AF9Ca656af840dff83E8264EcF986CA"


# --------------------------
# On-chain: SOL (Solscan Pro if key, else Solana JSON-RPC)
# --------------------------

def solana_rpc(method: str, params: list) -> dict:
    url = "https://api.mainnet-beta.solana.com"
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params,
    }
    r = requests.post(url, json=payload, timeout=25)
    if r.status_code != 200:
        return {}
    return r.json()


def sol_signatures(address: str, limit: int = 50) -> List[dict]:
    js = solana_rpc("getSignaturesForAddress", [address, {"limit": limit}])
    res = (js.get("result") or [])
    out = []
    for x in res:
        bt = x.get("blockTime")
        ts = ""
        if bt:
            ts = datetime.fromtimestamp(int(bt), tz=timezone.utc).isoformat()
        out.append({
            "signature": x.get("signature",""),
            "timestamp": ts,
            "err": x.get("err", None),
        })
    return out


# --------------------------
# On-chain aggregation endpoint
# --------------------------

def filter_window(items: List[dict], window_start: datetime, ts_key: str) -> List[dict]:
    out = []
    for x in items:
        ts = x.get(ts_key, "")
        dt = parse_dt_any(ts)
        if dt and dt >= window_start:
            out.append(x)
    return out


@app.get("/report/crypto-onchain")
def crypto_onchain(
    window_days: int = Query(30, ge=1, le=365),
    btc_addresses: str = Query("", description="Comma-separated BTC addresses"),
    eth_addresses: str = Query("", description="Comma-separated ETH addresses"),
    sol_addresses: str = Query("", description="Comma-separated SOL addresses"),
):
    """
    True on-chain wallet monitor.
    You must provide address lists. This does NOT infer politician wallets.
    Returns activity in the last N days for:
      - BTC transactions (mempool.space)
      - ETH normal tx + LINK ERC20 transfers (Etherscan)
      - SOL recent signatures (Solana RPC)
    """
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=window_days)

    btc_list = [x.strip() for x in (btc_addresses or "").split(",") if x.strip()]
    eth_list = [x.strip() for x in (eth_addresses or "").split(",") if x.strip()]
    sol_list = [x.strip() for x in (sol_addresses or "").split(",") if x.strip()]

    result = {
        "date": now.date().isoformat(),
        "windowDays": window_days,
        "windowStart": since.date().isoformat(),
        "windowEnd": now.date().isoformat(),
        "top10": fetch_top10_crypto(),
        "btc": [],
        "eth": [],
        "sol": [],
        "note": "On-chain monitoring requires wallet addresses. Congress disclosures do not provide wallet addresses.",
    }

    # BTC
    for addr in btc_list[:25]:
        txs = btc_address_txs(addr)
        txs = filter_window(txs, since, "timestamp")
        result["btc"].append({"address": addr, "txs": txs[:200]})
        time.sleep(0.15)

    # ETH + LINK
    for addr in eth_list[:25]:
        normal = eth_normal_txs(addr)
        normal = filter_window(normal, since, "timestamp")

        link = eth_erc20_txs(addr, LINK_CONTRACT)
        link = filter_window(link, since, "timestamp")

        result["eth"].append({
            "address": addr,
            "normalTxs": normal[:300],
            "linkTransfers": link[:300],
        })
        time.sleep(0.2)

    # SOL
    for addr in sol_list[:25]:
        sigs = sol_signatures(addr, limit=100)
        sigs = filter_window(sigs, since, "timestamp")
        result["sol"].append({"address": addr, "signatures": sigs[:200]})
        time.sleep(0.15)

    return result
