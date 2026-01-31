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

SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "FinanceApp/1.0 (contact: hijazss@gmail.com)")
SEC_TIMEOUT = 25

_party_re = re.compile(r"/\s*([DR])\b", re.IGNORECASE)

# --------------------------
# Root
# --------------------------

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


def interleave(items: List[dict], limit: int = 140) -> List[dict]:
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
        "lastFiledAt": last,
        "strength": kind,
        "chamber": x.get("chamber", ""),
        "amountRange": x.get("amountRange", ""),
        "traded": x.get("traded", ""),
        "filed": x.get("filed", ""),
        "description": x.get("description", ""),
    }


# --------------------------
# Crypto detection (unchanged)
# --------------------------

TOP_COINS = [
    "BTC", "ETH", "SOL", "LINK", "XRP", "ADA", "DOGE", "AVAX", "MATIC", "BNB"
]

CRYPTO_ALIASES: Dict[str, List[str]] = {
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "eth", "ether"],
    "SOL": ["solana", "sol"],
    "LINK": ["chainlink", "link"],
    "XRP": ["xrp", "ripple"],
    "ADA": ["cardano", "ada"],
    "DOGE": ["dogecoin", "doge"],
    "AVAX": ["avalanche", "avax"],
    "MATIC": ["polygon", "matic"],
    "BNB": ["bnb", "binance coin"],
}

CRYPTO_RELATED_TICKERS = set([
    "IBIT", "FBTC", "ARKB", "BITB", "BTCO", "HODL", "GBTC", "BITO",
    "ETHE", "ETHA",
    "COIN", "MSTR", "RIOT", "MARA", "HUT", "CLSK",
])

TICKER_TO_COINS: Dict[str, List[str]] = {
    "IBIT": ["BTC"], "FBTC": ["BTC"], "ARKB": ["BTC"], "BITB": ["BTC"], "BTCO": ["BTC"], "HODL": ["BTC"],
    "GBTC": ["BTC"], "BITO": ["BTC"],
    "ETHE": ["ETH"], "ETHA": ["ETH"],
    "COIN": ["CRYPTO-LINKED"],
    "MSTR": ["BTC", "CRYPTO-LINKED"],
    "RIOT": ["BTC", "CRYPTO-LINKED"],
    "MARA": ["BTC", "CRYPTO-LINKED"],
    "HUT": ["BTC", "CRYPTO-LINKED"],
    "CLSK": ["BTC", "CRYPTO-LINKED"],
}

_CRYPTO_PATTERNS: Dict[str, List[re.Pattern]] = {}
for sym, words in CRYPTO_ALIASES.items():
    pats: List[re.Pattern] = []
    for w in words:
        if re.fullmatch(r"[a-z]{2,6}", w):
            pats.append(re.compile(rf"\b{re.escape(w)}\b", re.IGNORECASE))
        else:
            pats.append(re.compile(re.escape(w), re.IGNORECASE))
    _CRYPTO_PATTERNS[sym] = pats

_GENERIC_CRYPTO_HINTS = [
    re.compile(r"\bcryptocurrency\b", re.IGNORECASE),
    re.compile(r"\bdigital asset\b", re.IGNORECASE),
    re.compile(r"\bvirtual currency\b", re.IGNORECASE),
]


def collect_crypto_text(row: dict) -> str:
    fields = [
        "AssetDescription", "asset_description", "Description", "description",
        "Asset", "asset", "Name", "name", "Issuer", "issuer",
        "Ticker", "ticker", "Stock", "stock", "Symbol", "symbol",
        "Owner", "owner", "Type", "type",
    ]
    parts = []
    for f in fields:
        v = row.get(f)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            parts.append(s)
    return " | ".join(parts)


def detect_coins(text: str) -> List[str]:
    if not text:
        return []
    hits: List[str] = []
    for sym in TOP_COINS:
        for pat in _CRYPTO_PATTERNS.get(sym, []):
            if pat.search(text):
                hits.append(sym)
                break

    out: List[str] = []
    for sym in TOP_COINS:
        if sym in hits:
            out.append(sym)
    return out


def has_generic_crypto_hint(text: str) -> bool:
    if not text:
        return False
    return any(p.search(text) for p in _GENERIC_CRYPTO_HINTS)


def classify_crypto_trade(ticker: str, text: str) -> Tuple[bool, List[str], str]:
    t = (ticker or "").upper().strip()
    coins_from_text = detect_coins(text)
    is_related_ticker = bool(t and t in CRYPTO_RELATED_TICKERS)
    coins_from_ticker = TICKER_TO_COINS.get(t, []) if is_related_ticker else []

    coins = []
    merged = set(coins_from_text + coins_from_ticker)
    for sym in TOP_COINS:
        if sym in merged:
            coins.append(sym)
    if "CRYPTO-LINKED" in merged:
        coins.append("CRYPTO-LINKED")

    if coins_from_text:
        return True, coins, "direct"
    if is_related_ticker:
        return True, coins if coins else ["CRYPTO-LINKED"], "etf_or_proxy"
    if has_generic_crypto_hint(text):
        return True, coins if coins else ["CRYPTO"], "hint_only"

    return False, [], ""


def counts_to_list(m: Dict[str, int]) -> List[dict]:
    out = [{"symbol": k, "count": v} for k, v in m.items()]
    out.sort(key=lambda x: (-x["count"], x["symbol"]))
    return out


# --------------------------
# Endpoint: Congress rolling window (30/180/365)
# --------------------------

@app.get("/report/today")
def report_today(
    window_days: Optional[int] = Query(default=None, ge=1, le=365),
    horizon_days: Optional[int] = Query(default=None, ge=1, le=365),
):
    if not QUIVER_TOKEN:
        raise HTTPException(500, "QUIVER_TOKEN missing")

    days = window_days if window_days is not None else horizon_days if horizon_days is not None else 30

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
            "politicianBuys": [],
            "politicianSells": [],
            "crypto": {"buys": [], "sells": [], "rawBuys": [], "rawSells": [], "raw": []},
        }

    try:
        rows = df.to_dict(orient="records")
    except Exception:
        rows = list(df)

    buys: List[dict] = []
    sells: List[dict] = []

    crypto_counts_buy: Dict[str, int] = {}
    crypto_counts_sell: Dict[str, int] = {}
    crypto_raw: List[dict] = []
    crypto_raw_buys: List[dict] = []
    crypto_raw_sells: List[dict] = []

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
        pol = str(pick_first(r, ["Politician", "politician", "Representative", "Senator", "Name", "name"], "")).strip()
        chamber = str(pick_first(r, ["Chamber", "chamber", "Office", "office"], "")).strip()
        amount = str(pick_first(r, ["Amount", "amount", "Range", "range", "AmountRange", "amount_range"], "")).strip()

        filed_dt = parse_dt_any(pick_first(r, ["Filed", "filed", "ReportDate", "report_date", "Date", "date"], ""))
        traded_dt = parse_dt_any(pick_first(r, ["Traded", "traded", "TransactionDate", "transaction_date"], ""))

        desc = str(pick_first(r, ["AssetDescription", "asset_description", "Description", "description", "Asset", "asset"], "")).strip()

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

        text_blob = collect_crypto_text(r)
        is_crypto, coins, crypto_kind = classify_crypto_trade(ticker, text_blob)

        if is_crypto:
            target_counts = crypto_counts_buy if kind == "BUY" else crypto_counts_sell
            for c in coins if coins else ["CRYPTO"]:
                target_counts[c] = target_counts.get(c, 0) + 1

            rec = {
                "kind": kind,
                "coins": coins if coins else ["CRYPTO"],
                "cryptoKind": crypto_kind,
                "ticker": (ticker or "").upper(),
                "description": desc,
                "party": party,
                "politician": pol,
                "chamber": chamber,
                "amountRange": amount,
                "traded": iso_date_only(traded_dt),
                "filed": iso_date_only(filed_dt),
            }
            crypto_raw.append(rec)
            if kind == "BUY":
                crypto_raw_buys.append(rec)
            else:
                crypto_raw_sells.append(rec)

    buys.sort(key=lambda x: x["best_dt"], reverse=True)
    sells.sort(key=lambda x: x["best_dt"], reverse=True)

    buy_cards = [to_card(x, "BUY") for x in interleave(buys, 140)]
    sell_cards = [to_card(x, "SELL") for x in interleave(sells, 140)]

    dem_buy = [x for x in buys if x["party"] == "D"]
    rep_buy = [x for x in buys if x["party"] == "R"]
    overlap = set(x["ticker"] for x in dem_buy if x["ticker"]) & set(x["ticker"] for x in rep_buy if x["ticker"])

    overlap_cards: List[dict] = []
    for t in overlap:
        dem_ct = sum(1 for x in dem_buy if x["ticker"] == t)
        rep_ct = sum(1 for x in rep_buy if x["ticker"] == t)
        latest_dt = None
        for x in buys:
            if x["ticker"] == t:
                if latest_dt is None or x["best_dt"] > latest_dt:
                    latest_dt = x["best_dt"]
        overlap_cards.append({
            "ticker": t,
            "companyName": "",
            "demBuyers": dem_ct,
            "repBuyers": rep_ct,
            "demSellers": 0,
            "repSellers": 0,
            "lastFiledAt": iso_date_only(latest_dt),
            "strength": "OVERLAP",
        })

    overlap_cards.sort(key=lambda x: (-(x["demBuyers"] + x["repBuyers"]), x["ticker"]))

    crypto_raw.sort(key=lambda x: (x.get("filed") or x.get("traded") or ""), reverse=True)
    crypto_raw_buys.sort(key=lambda x: (x.get("filed") or x.get("traded") or ""), reverse=True)
    crypto_raw_sells.sort(key=lambda x: (x.get("filed") or x.get("traded") or ""), reverse=True)

    crypto_payload = {
        "buys": counts_to_list(crypto_counts_buy),
        "sells": counts_to_list(crypto_counts_sell),
        "rawBuys": crypto_raw_buys[:250],
        "rawSells": crypto_raw_sells[:250],
        "raw": crypto_raw[:500],
        "topCoins": TOP_COINS,
    }

    return {
        "date": now.date().isoformat(),
        "windowDays": days,
        "windowStart": since.date().isoformat(),
        "windowEnd": now.date().isoformat(),
        "convergence": overlap_cards[:25],
        "politicianBuys": buy_cards,
        "politicianSells": sell_cards,
        "crypto": crypto_payload,
    }


@app.get("/report/crypto")
def report_crypto(
    window_days: Optional[int] = Query(default=None, ge=1, le=365),
    horizon_days: Optional[int] = Query(default=None, ge=1, le=365),
):
    payload = report_today(window_days=window_days, horizon_days=horizon_days)
    return {
        "date": payload["date"],
        "windowDays": payload["windowDays"],
        "windowStart": payload["windowStart"],
        "windowEnd": payload["windowEnd"],
        "crypto": payload["crypto"],
    }


# --------------------------
# SEC EDGAR helpers
# --------------------------

def sec_get_json(url: str) -> dict:
    headers = {
        "User-Agent": SEC_USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov",
    }
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
    s = re.sub(r"\D", "", str(cik).strip())
    return s.zfill(10)


def accession_no_dashes(acc: str) -> str:
    return (acc or "").replace("-", "").strip()


def build_filing_index_url(cik: str, accession: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_no_dashes(accession)}/{accession}-index.html"


def build_primary_doc_url(cik: str, accession: str, primary_doc: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_no_dashes(accession)}/{primary_doc}"


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


# --------------------------
# 13F parsing and delta logic
# --------------------------

def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _find_first_text(node: ET.Element, want: List[str]) -> str:
    want_set = set(want)
    for ch in node.iter():
        if _strip_ns(ch.tag) in want_set and ch.text:
            return ch.text.strip()
    return ""


def parse_13f_info_table(xml_text: str) -> List[dict]:
    """
    Parses SEC 13F information table XML.
    Returns holdings as list of dicts with:
      issuer, cusip, value (int, in thousands), shares (int), type (SH/PRN), putCall
    """
    out: List[dict] = []
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return out

    # Find all <infoTable> nodes (namespaced or not)
    for node in root.iter():
        if _strip_ns(node.tag).lower() != "infotable":
            continue

        issuer = _find_first_text(node, ["nameOfIssuer", "nameofissuer"])
        cusip = _find_first_text(node, ["cusip"])
        title = _find_first_text(node, ["titleOfClass", "titleofclass"])
        value = _find_first_text(node, ["value"])
        sh_amt = ""
        sh_type = ""
        put_call = _find_first_text(node, ["putCall", "putcall"])

        # shares are nested under shrsOrPrnAmt
        for ch in node.iter():
            if _strip_ns(ch.tag).lower() == "shrsorprnamt":
                sh_amt = _find_first_text(ch, ["sshPrnamt", "sshprnamt"])
                sh_type = _find_first_text(ch, ["sshPrnamtType", "sshprnamttype"])
                break

        try:
            v_int = int(float(value)) if value else 0
        except Exception:
            v_int = 0
        try:
            s_int = int(float(sh_amt)) if sh_amt else 0
        except Exception:
            s_int = 0

        key = (cusip or issuer or title).strip()
        if not key:
            continue

        out.append({
            "issuer": issuer,
            "titleClass": title,
            "cusip": cusip,
            "valueK": v_int,      # SEC 13F "value" is typically in thousands of dollars
            "shares": s_int,
            "shareType": sh_type,
            "putCall": put_call,
        })

    return out


def extract_links_from_index_html(index_html: str) -> List[str]:
    # Simple href extraction, good enough for SEC filing index pages
    hrefs = re.findall(r'href="([^"]+)"', index_html, flags=re.IGNORECASE)
    return hrefs


def resolve_sec_archive_url(index_url: str, href: str) -> str:
    # SEC index uses relative links like "xslForm13F_X01/infotable.xml" or full paths
    if href.startswith("http"):
        return href
    if href.startswith("/Archives/"):
        return "https://www.sec.gov" + href
    base = index_url.rsplit("/", 1)[0]
    return base + "/" + href.lstrip("./")


def find_best_13f_info_table_url(index_url: str) -> Tuple[str, str]:
    """
    Returns (info_table_url, raw_index_html_url)
    """
    html = sec_get_text(index_url)

    # Pull all candidate links
    links = extract_links_from_index_html(html)
    # Prefer .xml links that look like information tables
    xmls = [x for x in links if x.lower().endswith(".xml")]
    preferred = []
    for x in xmls:
        lx = x.lower()
        if "infotable" in lx or "informationtable" in lx or "form13f" in lx or "xslform13f" in lx:
            preferred.append(x)
    candidates = preferred or xmls

    for href in candidates:
        full = resolve_sec_archive_url(index_url, href)
        # quick filter: SEC archive xml often accessible
        return full, html

    # If none found, return empty
    return "", html


def holdings_map(holdings: List[dict]) -> Dict[str, dict]:
    """
    key by CUSIP if available, else issuer+title
    """
    m: Dict[str, dict] = {}
    for h in holdings:
        cusip = (h.get("cusip") or "").strip()
        issuer = (h.get("issuer") or "").strip()
        title = (h.get("titleClass") or "").strip()
        key = cusip if cusip else (issuer + "|" + title).strip("|")
        if not key:
            continue
        m[key] = h
    return m


def compute_holdings_delta(newer: List[dict], older: List[dict], top_n: int = 20) -> dict:
    """
    Compute adds/trims/new/exits based on share delta and value delta.
    """
    m_new = holdings_map(newer)
    m_old = holdings_map(older)

    keys = set(m_new.keys()) | set(m_old.keys())
    rows = []
    for k in keys:
        a = m_new.get(k)
        b = m_old.get(k)

        new_sh = int((a or {}).get("shares") or 0)
        old_sh = int((b or {}).get("shares") or 0)
        d_sh = new_sh - old_sh

        new_v = int((a or {}).get("valueK") or 0)
        old_v = int((b or {}).get("valueK") or 0)
        d_v = new_v - old_v

        issuer = (a or b or {}).get("issuer") or ""
        cusip = (a or b or {}).get("cusip") or ""
        title = (a or b or {}).get("titleClass") or ""

        row = {
            "issuer": issuer,
            "cusip": cusip,
            "titleClass": title,
            "oldShares": old_sh,
            "newShares": new_sh,
            "deltaShares": d_sh,
            "oldValueK": old_v,
            "newValueK": new_v,
            "deltaValueK": d_v,
        }
        rows.append(row)

    new_positions = [r for r in rows if r["oldShares"] == 0 and r["newShares"] > 0]
    exits = [r for r in rows if r["oldShares"] > 0 and r["newShares"] == 0]
    adds = [r for r in rows if r["deltaShares"] > 0 and r["oldShares"] > 0 and r["newShares"] > 0]
    trims = [r for r in rows if r["deltaShares"] < 0 and r["newShares"] > 0]

    # Sort primarily by absolute value change, then abs shares change
    def key_abs(r):
        return (abs(int(r.get("deltaValueK") or 0)), abs(int(r.get("deltaShares") or 0)))

    new_positions.sort(key=key_abs, reverse=True)
    exits.sort(key=key_abs, reverse=True)
    adds.sort(key=key_abs, reverse=True)
    trims.sort(key=key_abs, reverse=True)

    return {
        "summary": {
            "positionsNew": len(new_positions),
            "positionsExited": len(exits),
            "positionsAdded": len(adds),
            "positionsTrimmed": len(trims),
        },
        "topNew": new_positions[:top_n],
        "topExited": exits[:top_n],
        "topAdds": adds[:top_n],
        "topTrims": trims[:top_n],
    }


# --------------------------
# Form 4 parsing (kept)
# --------------------------

def parse_form4_transactions(xml_text: str) -> List[dict]:
    out: List[dict] = []
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return out

    issuer = ""
    symbol = ""

    for node in root.iter():
        tag = node.tag.lower()
        if tag.endswith("issuername"):
            issuer = (node.text or "").strip()
        elif tag.endswith("issuertradingsymbol"):
            symbol = (node.text or "").strip().upper()

    for tx in root.iter():
        if not tx.tag.lower().endswith("nonderivativetransaction"):
            continue

        code = ""
        date = ""
        shares = ""
        price = ""

        for node in tx.iter():
            t = node.tag.lower()
            if t.endswith("transactioncode"):
                code = (node.text or "").strip().upper()

        for node in tx.iter():
            t = node.tag.lower()
            if t.endswith("transactiondate"):
                for c in node.iter():
                    if c.tag.lower().endswith("value"):
                        date = (c.text or "").strip()
            elif t.endswith("transactionshares"):
                for c in node.iter():
                    if c.tag.lower().endswith("value"):
                        shares = (c.text or "").strip()
            elif t.endswith("transactionpricepershare"):
                for c in node.iter():
                    if c.tag.lower().endswith("value"):
                        price = (c.text or "").strip()

        if code in ["P", "S"]:
            out.append({
                "issuerName": issuer,
                "ticker": symbol,
                "code": code,
                "transactionDate": date,
                "shares": shares,
                "price": price,
            })

    return out


# --------------------------
# Leaders endpoint: now includes 13F holdings deltas
# --------------------------

@app.get("/report/public-leaders")
def report_public_leaders():
    now = datetime.now(timezone.utc)
    since365 = now - timedelta(days=365)

    leaders = [
        {"key": "musk", "name": "Elon Musk", "cik": "1494730", "forms": ["4", "4/A"]},
        {"key": "berkshire", "name": "Berkshire Hathaway", "cik": "1067983", "forms": ["13F-HR", "13F-HR/A"]},
        {"key": "bridgewater", "name": "Bridgewater Associates", "cik": "1350694", "forms": ["13F-HR", "13F-HR/A"]},
        {"key": "blackrock", "name": "BlackRock, Inc.", "cik": "1364742", "forms": ["13F-HR", "13F-HR/A"]},
    ]

    results = []
    for leader in leaders:
        cik = leader["cik"]
        subs = pull_recent_filings_for_cik(cik10(cik))

        filings = extract_latest_filings(subs, leader["forms"], max_items=20)

        # Filter to within 365d when possible, but keep enough 13Fs to compute delta
        filtered = []
        for f in filings:
            filed_dt = parse_dt_any(f.get("filingDate"))
            if filed_dt is not None and filed_dt < since365 and not f["form"].startswith("13F"):
                continue
            filtered.append(f)

        events = []

        # If 13F leader: compute delta between latest two 13F filings
        if any(f["form"].startswith("13F") for f in filtered):
            f13 = [f for f in filtered if f["form"].startswith("13F")]
            f13.sort(key=lambda x: x.get("filingDate", ""), reverse=True)

            # Try to compute delta using the latest two 13F filings
            delta = None
            info_urls = {"newerInfoTable": "", "olderInfoTable": ""}

            if len(f13) >= 2:
                newer = f13[0]
                older = f13[1]

                newer_index = build_filing_index_url(cik10(cik), newer["accession"])
                older_index = build_filing_index_url(cik10(cik), older["accession"])

                try:
                    newer_info_url, _ = find_best_13f_info_table_url(newer_index)
                    older_info_url, _ = find_best_13f_info_table_url(older_index)

                    info_urls["newerInfoTable"] = newer_info_url
                    info_urls["olderInfoTable"] = older_info_url

                    if newer_info_url and older_info_url:
                        newer_xml = sec_get_text(newer_info_url)
                        older_xml = sec_get_text(older_info_url)
                        newer_holdings = parse_13f_info_table(newer_xml)
                        older_holdings = parse_13f_info_table(older_xml)
                        delta = compute_holdings_delta(newer_holdings, older_holdings, top_n=20)
                except Exception:
                    delta = None

            # Add a compact 13F event card with delta
            if f13:
                latest = f13[0]
                latest_index_url = build_filing_index_url(cik10(cik), latest["accession"])
                latest_primary_url = build_primary_doc_url(cik10(cik), latest["accession"], latest["primaryDocument"])

                events.append({
                    "form": latest["form"],
                    "label": "13F HOLDINGS DELTA" if delta else "13F FILED",
                    "filingDate": latest.get("filingDate", ""),
                    "indexUrl": latest_index_url,
                    "primaryUrl": latest_primary_url,
                    "details": {
                        "note": "13F shows holdings, not real-time trades. Values are SEC 'value' in thousands of dollars (valueK).",
                        "infoTableUrls": info_urls,
                        "delta": delta,
                    },
                })

        # Add latest Form 4 events as before
        for f in filtered:
            form = str(f.get("form") or "")
            if not form.startswith("4"):
                continue

            filing_date = f.get("filingDate", "")
            filed_dt = parse_dt_any(filing_date)
            if filed_dt is not None and filed_dt < since365:
                continue

            accession = f["accession"]
            primary_doc = f["primaryDocument"]

            index_url = build_filing_index_url(cik10(cik), accession)
            primary_url = build_primary_doc_url(cik10(cik), accession, primary_doc)

            label = "FORM4"
            details = {}

            try:
                xml_text = sec_get_text(primary_url)
                txs = parse_form4_transactions(xml_text)
                buys = [t for t in txs if (t.get("code") or "").upper() == "P"]
                sells = [t for t in txs if (t.get("code") or "").upper() == "S"]
                if buys and not sells:
                    label = "BUY"
                elif sells and not buys:
                    label = "SELL"
                elif buys and sells:
                    label = "MIXED"
                details = {"transactions": txs[:40], "buyCount": len(buys), "sellCount": len(sells)}
            except Exception:
                label = "FORM4"

            events.append({
                "form": form,
                "label": label,
                "filingDate": filing_date,
                "indexUrl": index_url,
                "primaryUrl": primary_url,
                "details": details,
            })

            if len(events) >= 6:
                break

            time.sleep(0.15)

        results.append({"name": leader["name"], "cik": cik10(cik), "events": events})

    return {
        "date": now.date().isoformat(),
        "windowDays": 365,
        "windowStart": since365.date().isoformat(),
        "windowEnd": now.date().isoformat(),
        "leaders": results,
        "note": "Leaders now includes 13F holdings delta (adds/trims/new/exits). 13F holdings do not include ticker symbols, only issuer name + CUSIP + shares + value.",
    }
