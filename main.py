<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
  <title>Convergence Signals</title>

  <meta name="theme-color" content="#0b0b0f" />
  <style>
    :root{
      --bg:#0b0b0f;
      --card:#151522;
      --text:#ffffff;
      --muted:rgba(255,255,255,.70);
      --border:rgba(255,255,255,.10);

      --dem:#4aa3ff;
      --rep:#ff5a5a;
      --both:#b38bff;

      --ok:#7cffc0;
      --err:#ff8a8a;

      --pos:#7cffc0;
      --neg:#ff8a8a;

      --sec:#ffd27d;
    }

    *{box-sizing:border-box}
    body{
      margin:0;
      background:var(--bg);
      color:var(--text);
      font-family:-apple-system,BlinkMacSystemFont,sans-serif;
      font-size:12px;
    }
    a{color:inherit;text-decoration:none}

    .header{
      padding:12px 14px 8px;
      display:flex;
      justify-content:space-between;
      align-items:flex-start;
      gap:12px;
      max-width:1100px;
      margin:0 auto;
    }
    .h1{font-size:16px;font-weight:900;letter-spacing:.01em}
    .subtle{font-size:12px;color:var(--muted);margin-top:2px}
    .pillBtn{
      border:1px solid var(--border);
      padding:7px 10px;
      border-radius:999px;
      background:rgba(255,255,255,.05);
      color:#fff;
      font-size:12px;
      cursor:pointer;
    }

    .container{
      max-width:1100px;
      margin:0 auto;
      padding:8px 14px 110px;
    }

    .status{
      border:1px solid var(--border);
      background:rgba(255,255,255,.03);
      border-radius:12px;
      padding:9px 10px;
      margin:6px 0 10px;
      font-size:12px;
      color:var(--muted);
      line-height:1.25;
    }
    .status strong{color:#fff}
    .ok{color:var(--ok)}
    .err{color:var(--err)}
    .mono{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace}

    .tabRow{
      display:flex;
      gap:8px;
      flex-wrap:wrap;
      margin:8px 0 8px;
      align-items:center;
    }
    .tabBtn{
      border:1px solid var(--border);
      background:rgba(255,255,255,.04);
      padding:7px 9px;
      border-radius:999px;
      font-size:12px;
      color:var(--muted);
      cursor:pointer;
    }
    .tabBtnActive{
      color:#fff;
      background:rgba(255,255,255,.10);
    }

    .rightTools{
      margin-left:auto;
      display:flex;
      gap:8px;
      flex-wrap:wrap;
      align-items:center;
    }
    .seg{
      display:flex;
      border:1px solid var(--border);
      border-radius:999px;
      overflow:hidden;
      background:rgba(255,255,255,.03);
    }
    .seg button{
      border:0;
      background:transparent;
      color:var(--muted);
      font-size:12px;
      padding:7px 10px;
      cursor:pointer;
    }
    .seg .on{
      background:rgba(255,255,255,.10);
      color:#fff;
    }

    .sectionTitle{
      margin:14px 0 8px;
      font-size:11px;
      color:var(--muted);
      letter-spacing:.08em;
      text-transform:uppercase;
    }

    .grid2{
      display:grid;
      grid-template-columns:1fr;
      gap:10px;
    }
    @media (min-width: 820px){
      .grid2{grid-template-columns:1fr 1fr;}
    }

    .grid3{
      display:grid;
      grid-template-columns:1fr;
      gap:10px;
    }
    @media (min-width: 980px){
      .grid3{grid-template-columns:repeat(3,1fr);}
    }
    @media (min-width: 760px) and (max-width: 979px){
      .grid3{grid-template-columns:repeat(2,1fr);}
    }

    .kpi{
      background:var(--card);
      border:1px solid var(--border);
      border-radius:14px;
      padding:10px;
    }
    .kpiTop{
      display:flex;
      justify-content:space-between;
      gap:10px;
      align-items:flex-start;
      margin-bottom:6px;
    }
    .kpiTitle{
      font-size:11px;
      color:var(--muted);
      letter-spacing:.08em;
      text-transform:uppercase;
    }
    .kpiValue{
      font-size:16px;
      font-weight:900;
      letter-spacing:.02em;
      margin-top:4px;
    }
    .kpiSub{
      font-size:12px;
      color:rgba(255,255,255,.82);
      line-height:1.25;
    }
    .kpiList{
      margin-top:8px;
      display:flex;
      flex-direction:column;
      gap:6px;
    }
    .kpiRow{
      display:flex;
      justify-content:space-between;
      gap:8px;
      font-size:12px;
      color:rgba(255,255,255,.90);
      align-items:center;
    }
    .kpiBadge{
      font-size:10px;
      padding:4px 7px;
      border-radius:999px;
      border:1px solid var(--border);
      background:rgba(255,255,255,.04);
      white-space:nowrap;
      color:rgba(255,255,255,.90);
    }
    .badgePos{border-color:rgba(124,255,192,.45); color:var(--pos)}
    .badgeNeg{border-color:rgba(255,138,138,.45); color:var(--neg)}
    .badgeNeu{border-color:rgba(255,255,255,.18); color:rgba(255,255,255,.85)}
    .badgeBoth{border-color:rgba(179,139,255,.45); color:var(--both)}

    .card{
      background:var(--card);
      border:1px solid var(--border);
      border-radius:14px;
      padding:10px 10px 9px;
      margin-bottom:8px;
    }
    .cardTop{
      display:flex;
      justify-content:space-between;
      align-items:flex-start;
      gap:10px;
    }

    .tickerLine{
      display:flex;
      align-items:baseline;
      gap:8px;
      flex-wrap:wrap;
    }
    .ticker{
      font-size:13px;
      font-weight:900;
      letter-spacing:.03em;
    }
    .who{
      font-size:12px;
      color:var(--muted);
      max-width:720px;
      overflow:hidden;
      text-overflow:ellipsis;
      white-space:nowrap;
    }

    .tags{
      display:flex;
      gap:6px;
      flex-wrap:wrap;
      justify-content:flex-end;
    }
    .tag{
      font-size:10px;
      padding:4px 7px;
      border-radius:999px;
      border:1px solid var(--border);
      background:rgba(255,255,255,.04);
      white-space:nowrap;
    }
    .tagDem{border-color:rgba(74,163,255,.5); color:var(--dem)}
    .tagRep{border-color:rgba(255,90,90,.5); color:var(--rep)}
    .tagBoth{border-color:rgba(179,139,255,.55); color:var(--both)}
    .tagBuy{border-color:rgba(124,255,192,.45); color:var(--ok)}
    .tagSell{border-color:rgba(255,138,138,.45); color:var(--err)}
    .tagSEC{border-color:rgba(255,210,125,.40); color:var(--sec)}

    .metaRow{
      margin-top:7px;
      font-size:12px;
      color:var(--muted);
      display:flex;
      justify-content:space-between;
      gap:10px;
      flex-wrap:wrap;
      line-height:1.25;
    }

    .divider{height:1px;background:rgba(255,255,255,.08);margin:8px 0 7px}

    .miniLinks{
      display:flex;
      gap:8px;
      flex-wrap:wrap;
      margin-top:8px;
    }
    .miniLink{
      border:1px solid var(--border);
      background:rgba(255,255,255,.03);
      padding:6px 9px;
      border-radius:999px;
      font-size:11px;
      color:rgba(255,255,255,.92);
    }

    .panel{
      border:1px solid var(--border);
      background:rgba(255,255,255,.03);
      border-radius:14px;
      padding:10px;
      margin:10px 0 10px;
    }

    iframe{
      width:100%;
      height:360px;
      border:0;
      border-radius:12px;
      background:#000;
      margin-top:8px;
    }

    .tabsBottom{
      position:fixed;
      bottom:0;
      left:0;
      right:0;
      background:#151522f0;
      border-top:1px solid var(--border);
      backdrop-filter:blur(12px);
    }
    .tabBarBottom{
      max-width:1100px;
      margin:0 auto;
      display:grid;
      grid-template-columns:repeat(4,1fr);
    }
    .tabBottom{
      text-align:center;
      padding:11px 10px 16px;
      font-size:11px;
      color:var(--muted);
      user-select:none;
      cursor:pointer;
    }
    .tabBottomActive{color:#fff}

    .holdingsGrid{
      display:grid;
      grid-template-columns:1fr;
      gap:10px;
      margin-top:10px;
    }
    @media (min-width: 980px){
      .holdingsGrid{grid-template-columns:1fr 1fr;}
    }
    .holdingsItem{
      border:1px solid rgba(255,255,255,.10);
      background:rgba(255,255,255,.02);
      border-radius:12px;
      padding:9px;
    }

    .input{
      width:100%;
      border:1px solid var(--border);
      background:rgba(255,255,255,.03);
      color:#fff;
      border-radius:12px;
      padding:10px;
      font-size:12px;
      outline:none;
    }

    details.briefingDetails{
      border:1px solid rgba(255,255,255,.10);
      background:rgba(255,255,255,.02);
      border-radius:12px;
      padding:9px;
      margin-top:10px;
    }
    details.briefingDetails > summary{
      cursor:pointer;
      list-style:none;
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap:10px;
      user-select:none;
    }
    details.briefingDetails > summary::-webkit-details-marker{display:none}
    .summaryLeft{
      display:flex;
      flex-direction:column;
      gap:3px;
      min-width:0;
    }
    .summaryTitle{
      font-size:13px;
      font-weight:900;
      letter-spacing:.02em;
    }
    .summaryMeta{
      font-size:12px;
      color:var(--muted);
      overflow:hidden;
      text-overflow:ellipsis;
      white-space:nowrap;
      max-width:740px;
    }
    .bulletList{
      margin:8px 0 0;
      padding-left:16px;
      color:rgba(255,255,255,.86);
      line-height:1.3;
    }
    .headlineList{
      margin-top:10px;
      display:flex;
      flex-direction:column;
      gap:8px;
    }
    .headlineItem{
      border:1px solid rgba(255,255,255,.10);
      background:rgba(255,255,255,.02);
      border-radius:12px;
      padding:9px;
    }
    .headlineTitle{
      font-size:12px;
      color:rgba(255,255,255,.92);
      line-height:1.25;
    }
    .headlineMeta{
      margin-top:6px;
      font-size:11px;
      color:var(--muted);
      display:flex;
      gap:8px;
      flex-wrap:wrap;
    }
    .headlineSummary{
      margin-top:6px;
      font-size:12px;
      color:rgba(255,255,255,.84);
      line-height:1.25;
    }
  </style>
</head>

<body>
  <header class="header">
    <div>
      <div class="h1" id="pageTitle">News</div>
      <div class="subtle" id="reportDate"></div>
    </div>
    <button class="pillBtn" id="refreshBtn">Refresh</button>
  </header>

  <main class="container">
    <div class="status" id="statusBox">
      <strong>Status:</strong> <span id="statusText">Loading…</span><br/>
      <span class="mono" id="statusDetail"></span>
    </div>

    <div class="tabRow">
      <button class="tabBtn tabBtnActive" id="tabNewsBtn">News</button>
      <button class="tabBtn" id="tabCongressBtn">Congress</button>
      <button class="tabBtn" id="tabCryptoBtn">Crypto</button>
      <button class="tabBtn" id="tabPerfBtn">Performance</button>

      <div class="rightTools" id="mainWindowTools" style="display:none;">
        <div class="seg" aria-label="window toggle">
          <button id="btn30" class="on">30D</button>
          <button id="btn180">180D</button>
          <button id="btn365">365D</button>
        </div>
      </div>
    </div>

    <div id="stockPanel" style="display:none;"></div>

    <!-- NEWS TAB -->
    <section id="viewNews">
      <div class="sectionTitle">Daily market read</div>
      <div class="grid3">
        <div class="kpi" id="readKpiSPY"></div>
        <div class="kpi" id="readKpiQQQ"></div>
        <div class="kpi" id="readKpiFG"></div>
      </div>
      <div class="panel" style="margin-top:10px;">
        <div class="kpiSub" id="readSummary">Loading…</div>
        <div class="subtle" id="readHint" style="margin-top:8px;"></div>
        <div class="miniLinks" style="margin-top:10px;">
          <button class="pillBtn" id="readRefreshBtn">Refresh market read</button>
        </div>
      </div>

      <div class="sectionTitle">Daily briefing</div>
      <div class="grid3">
        <div class="kpi" id="briefKpiSentiment"></div>
        <div class="kpi" id="briefKpiScope"></div>
        <div class="kpi" id="briefKpiNotes"></div>
      </div>

      <div class="sectionTitle">Your watchlist</div>
      <div class="panel">
        <div class="kpiSub">
          Comma-separated tickers. Example: SPY,QQQ,NVDA,AAPL
        </div>
        <div style="margin-top:10px;">
          <input class="input" id="watchInput" value="SPY,QQQ,NVDA,AAPL,MSFT,AMZN,GOOGL,META,TSLA,BRK-B" />
        </div>
        <div class="miniLinks" style="margin-top:10px;">
          <button class="pillBtn" id="watchApplyBtn">Refresh briefing</button>
        </div>
      </div>

      <div class="sectionTitle">Sectors</div>
      <div id="briefingSectors"></div>
      <div class="subtle" id="briefHint" style="margin-top:12px;"></div>
    </section>

    <!-- CONGRESS TAB -->
    <section id="viewCongress" style="display:none;">
      <div class="sectionTitle">Common holdings (simple)</div>
      <div class="panel">
        <div class="kpiSub">
          Ticker + how many unique politicians disclosed activity in that ticker during the selected window.
        </div>

        <div class="divider"></div>

        <div class="kpi" id="holdingsKpiSimple"></div>

        <div class="divider"></div>

        <div class="sectionTitle" style="margin-top:0;">Ranked by holder count</div>
        <div id="holdingsList" class="holdingsGrid"></div>

        <div class="subtle" id="holdingsHint" style="margin-top:10px;"></div>
      </div>

      <div class="sectionTitle">Day-by-day trade feed</div>
      <div class="panel">
        <div class="kpiSub">
          Grouped by date. Tap a ticker for chart.
        </div>
        <div class="divider"></div>
        <div id="dailyTradeList"></div>
        <div class="subtle" id="dailyHint" style="margin-top:10px;"></div>
      </div>
    </section>

    <!-- CRYPTO TAB -->
    <section id="viewCrypto" style="display:none;">
      <div class="sectionTitle">Crypto briefing</div>
      <div class="panel">
        <div class="kpiSub">
          Daily crypto headlines and a simple sentiment gauge.
        </div>

        <div class="divider"></div>

        <div class="kpi" id="cryptoKpiOverall"></div>

        <div class="divider"></div>

        <div class="kpiSub">Coins (comma-separated):</div>
        <div style="margin-top:10px;">
          <input class="input" id="coinInput" value="BTC,ETH,LINK,SHIB" />
        </div>
        <div class="miniLinks" style="margin-top:10px;">
          <button class="pillBtn" id="cryptoApplyBtn">Refresh crypto</button>
        </div>
      </div>

      <div id="cryptoCoins"></div>
      <div class="subtle" id="cryptoHint" style="margin-top:12px;"></div>
    </section>

    <!-- PERFORMANCE TAB -->
    <section id="viewPerf" style="display:none;">
      <div class="sectionTitle">Politician performance</div>
      <div class="panel">
        <div class="kpiSub">
          Portfolio performance inside the app needs a backend integration.
          Use these trackers plus the Congress day-by-day feed.
        </div>

        <div class="miniLinks" style="margin-top:10px;">
          <a class="miniLink" target="_blank" rel="noopener" href="https://www.capitoltrades.com/">Capitol Trades</a>
          <a class="miniLink" target="_blank" rel="noopener" href="https://www.quiverquant.com/congresstrading/">QuiverQuant Congress Trading</a>
          <a class="miniLink" target="_blank" rel="noopener" href="https://efdsearch.senate.gov/search/home/">Senate EFD Search</a>
          <a class="miniLink" target="_blank" rel="noopener" href="https://disclosures-clerk.house.gov/">House Disclosures</a>
        </div>
      </div>
    </section>
  </main>

  <nav class="tabsBottom">
    <div class="tabBarBottom">
      <div class="tabBottom tabBottomActive" id="navNews">News</div>
      <div class="tabBottom" id="navCongress">Congress</div>
      <div class="tabBottom" id="navCrypto">Crypto</div>
      <div class="tabBottom" id="navPerf">Performance</div>
    </div>
  </nav>

  <script>
    const BACKEND_BASE = "https://finance-backend-1-bbi9.onrender.com";

    const MARKET_READ_URL = () => `${BACKEND_BASE}/market/read`;
    const NEWS_BRIEF_URL = (watchlist) => {
      const wl = String(watchlist || "");
      return `${BACKEND_BASE}/news/briefing?tickers=${encodeURIComponent(wl)}&max_general=60&max_per_ticker=6`;
    };

    const HOLDINGS_URL = (windowDays) =>
      `${BACKEND_BASE}/report/holdings/common?window_days=${encodeURIComponent(String(windowDays))}&top_n=30`;

    const DAILY_URL = (windowDays) =>
      `${BACKEND_BASE}/report/congress/daily?window_days=${encodeURIComponent(String(windowDays))}&limit=250`;

    const CRYPTO_URL = (coins) => {
      const c = String(coins || "");
      return `${BACKEND_BASE}/crypto/news/briefing?coins=${encodeURIComponent(c)}&include_top_n=15`;
    };

    const el = (id) => document.getElementById(id);

    let ACTIVE_TAB = "News";
    let WINDOW_DAYS = 30;

    let CACHE_READ = null;
    let CACHE_BRIEFING = new Map();
    let CACHE_HOLDINGS = new Map();
    let CACHE_DAILY = new Map();
    let CACHE_CRYPTO = new Map();

    function setStatus(kind, text, detail){
      const st = el("statusText");
      const sd = el("statusDetail");
      st.textContent = text || "";
      st.className = kind === "ok" ? "ok" : kind === "err" ? "err" : "";
      sd.textContent = detail || "";
    }

    function setTitleForTab(tab){
      el("pageTitle").textContent = tab;
    }

    async function fetchJson(url){
      const res = await fetch(url, { cache: "no-store" });
      const data = await res.json().catch(() => ({}));
      if (!res.ok){
        const msg = (data && (data.detail || data.error || data.message)) ? (data.detail || data.error || data.message) : ("HTTP " + res.status);
        const err = new Error(msg);
        err.status = res.status;
        err.data = data;
        throw err;
      }
      return data;
    }

    function setStockPanel(ticker){
      const t = String(ticker||"").toUpperCase();
      if (!t){
        el("stockPanel").style.display = "none";
        el("stockPanel").innerHTML = "";
        return;
      }
      const symbol = "NASDAQ:" + t;
      const yf = `https://finance.yahoo.com/quote/${encodeURIComponent(t)}`;
      const tv = `https://www.tradingview.com/symbols/${encodeURIComponent(symbol)}/`;

      el("stockPanel").style.display = "";
      el("stockPanel").innerHTML = `
        <div class="panel">
          <div class="cardTop">
            <div class="tickerLine">
              <div class="ticker">${t}</div>
              <div class="who">Chart</div>
            </div>
            <button class="pillBtn" id="closeStockBtn">Close</button>
          </div>
          <iframe
            title="TradingView Chart"
            src="https://s.tradingview.com/widgetembed/?symbol=${encodeURIComponent(symbol)}&interval=D&hidesidetoolbar=0&symboledit=1&saveimage=1&toolbarbg=%23151522&theme=dark&style=1&timezone=Etc%2FUTC&withdateranges=1&hideideas=1"
            loading="lazy">
          </iframe>
          <div class="miniLinks">
            <a class="miniLink" href="${yf}" target="_blank" rel="noopener">Open Yahoo</a>
            <a class="miniLink" href="${tv}" target="_blank" rel="noopener">Open TradingView</a>
          </div>
        </div>
      `;
      const btn = document.getElementById("closeStockBtn");
      if (btn) btn.addEventListener("click", () => {
        history.pushState("", document.title, window.location.pathname + window.location.search);
        setStockPanel("");
      });
    }

    function handleHash(){
      const h = window.location.hash || "";
      if (h.startsWith("#stock=")){
        const t = decodeURIComponent(h.replace("#stock=","")).trim();
        setStockPanel(t);
        return;
      }
      setStockPanel("");
    }

    function tickerLinksHTML(ticker){
      const t = String(ticker||"").toUpperCase();
      const yf = `https://finance.yahoo.com/quote/${encodeURIComponent(t)}`;
      const tvChart = `https://www.tradingview.com/chart/?symbol=${encodeURIComponent("NASDAQ:" + t)}`;
      return `
        <div class="miniLinks">
          <a class="miniLink" href="#stock=${t}">Chart</a>
          <a class="miniLink" href="${yf}" target="_blank" rel="noopener">Yahoo</a>
          <a class="miniLink" href="${tvChart}" target="_blank" rel="noopener">TV</a>
        </div>
      `;
    }

    function fmtPct(x){
      if (x === null || x === undefined || isNaN(Number(x))) return "—";
      const n = Number(x);
      const s = n > 0 ? "+" : "";
      return `${s}${n.toFixed(2)}%`;
    }

    function renderMarketRead(payload){
      const sp = payload?.sp500 || {};
      const nq = payload?.nasdaq || {};
      const fg = payload?.fearGreed || {};
      const errs = Array.isArray(payload?.errors) ? payload.errors : [];

      el("readKpiSPY").innerHTML = `
        <div class="kpiTop">
          <div>
            <div class="kpiTitle">S&P proxy</div>
            <div class="kpiValue">${sp?.last ? sp.last : "—"}</div>
          </div>
          <span class="kpiBadge badgeNeu">${sp?.symbol || "SPY"}</span>
        </div>
        <div class="kpiSub">
          1D: <strong>${fmtPct(sp?.ret1dPct)}</strong> · 5D: <strong>${fmtPct(sp?.ret5dPct)}</strong> · 1M: <strong>${fmtPct(sp?.ret1mPct)}</strong>
        </div>
      `;

      el("readKpiQQQ").innerHTML = `
        <div class="kpiTop">
          <div>
            <div class="kpiTitle">Nasdaq proxy</div>
            <div class="kpiValue">${nq?.last ? nq.last : "—"}</div>
          </div>
          <span class="kpiBadge badgeNeu">${nq?.symbol || "QQQ"}</span>
        </div>
        <div class="kpiSub">
          1D: <strong>${fmtPct(nq?.ret1dPct)}</strong> · 5D: <strong>${fmtPct(nq?.ret5dPct)}</strong> · 1M: <strong>${fmtPct(nq?.ret1mPct)}</strong>
        </div>
      `;

      const fgScore = fg?.score;
      const fgRating = fg?.rating;
      const fgBadge = (typeof fgScore === "number")
        ? (fgScore >= 60 ? "badgePos" : fgScore <= 40 ? "badgeNeg" : "badgeNeu")
        : "badgeNeu";

      el("readKpiFG").innerHTML = `
        <div class="kpiTop">
          <div>
            <div class="kpiTitle">Fear & greed</div>
            <div class="kpiValue">${(typeof fgScore === "number") ? fgScore : "—"}</div>
          </div>
          <span class="kpiBadge ${fgBadge}">${fgRating || "External index"}</span>
        </div>
        <div class="kpiSub">${(typeof fgScore === "number") ? "Loaded." : "Not available right now."}</div>
      `;

      el("readSummary").textContent = String(payload?.summary || "—");
      el("readHint").textContent = errs.length ? ("Market read issues: " + errs.slice(0,2).join(" | ")) : "Market read loaded.";
    }

    async function loadMarketRead(force){
      if (CACHE_READ && !force){
        renderMarketRead(CACHE_READ);
        return;
      }
      const url = MARKET_READ_URL();
      el("readHint").textContent = `Loading from: ${url}`;
      try{
        const data = await fetchJson(url);
        CACHE_READ = data;
        renderMarketRead(data);
      }catch(err){
        el("readHint").textContent = `Could not load market read. ${err?.message || ""}`;
        el("readSummary").textContent = "—";
      }
    }

    function sentimentBadge(s){
      const score = Number(s?.score ?? 50);
      const label = String(s?.label ?? "NEUTRAL");
      if (score >= 60) return { cls: "badgePos", label: label, score };
      if (score <= 40) return { cls: "badgeNeg", label: label, score };
      return { cls: "badgeNeu", label: label, score };
    }

    function normalizeNewsErrors(err){
      if (!err) return [];
      if (Array.isArray(err)) return err.filter(Boolean).map(String);
      if (typeof err === "object"){
        const out = [];
        for (const k of Object.keys(err)){
          const v = err[k];
          if (Array.isArray(v)) out.push(...v.filter(Boolean).map(String));
          else if (v) out.push(String(v));
        }
        return out;
      }
      return [String(err)];
    }

    function normalizeSectorHeadlines(s){
      if (!s || typeof s !== "object") return [];
      if (Array.isArray(s.headlines)) return s.headlines;
      if (Array.isArray(s.topHeadlines)) return s.topHeadlines;
      return [];
    }

    function headlineHTML(h){
      const title = String(h?.title || "").trim();
      const link = String(h?.link || "").trim();
      const pub = String(h?.published || "").trim();
      const bucket = String(h?.bucket || h?.sourceFeed || h?.source || "").trim();
      const summary = String(h?.summary || "").trim();

      return `
        <div class="headlineItem">
          <div class="headlineTitle">${link ? `<a href="${link}" target="_blank" rel="noopener">${title}</a>` : title}</div>
          ${summary ? `<div class="headlineSummary">${summary}</div>` : ""}
          <div class="headlineMeta">
            ${pub ? `<span>Published: <strong>${pub}</strong></span>` : ""}
            ${bucket ? `<span>Source: <strong>${bucket}</strong></span>` : ""}
          </div>
        </div>
      `;
    }

    function sectorPanelHTML(s){
      const sector = String(s?.sector || "General");
      const sent = sentimentBadge(s?.sentiment);
      const summary = String(s?.summary || "").trim();
      const headlines = normalizeSectorHeadlines(s);

      const meta = [];
      meta.push(`${headlines.length} headlines`);
      meta.push(`Sentiment ${sent.score}/100`);
      const metaLine = meta.join(" · ");

      return `
        <details class="briefingDetails">
          <summary>
            <div class="summaryLeft">
              <div class="summaryTitle">${sector}</div>
              <div class="summaryMeta">${metaLine}${summary ? " · " + summary : ""}</div>
            </div>
            <div class="tags">
              <span class="kpiBadge badgeNeu">${headlines.length}</span>
              <span class="kpiBadge ${sent.cls}">${sent.label} · ${sent.score}</span>
            </div>
          </summary>

          <div class="divider"></div>

          <div class="kpiSub">
            <strong>Sector scan:</strong> ${summary || "No summary available."}
          </div>

          <div class="divider"></div>

          <div class="kpiSub"><strong>Headlines:</strong></div>
          <div class="headlineList">
            ${headlines.length ? headlines.map(headlineHTML).join("") : "<div class='subtle'>No headlines in this sector.</div>"}
          </div>
        </details>
      `;
    }

    function renderBriefing(data, watchlist){
      const overall = sentimentBadge(data?.overallSentiment);
      const sectors = Array.isArray(data?.sectors) ? data.sectors : [];
      const date = String(data?.date || "");
      const note = String(data?.note || "");

      const total = sectors.reduce((a, s) => a + normalizeSectorHeadlines(s).length, 0);

      el("briefKpiSentiment").innerHTML = `
        <div class="kpiTop">
          <div>
            <div class="kpiTitle">Overall sentiment</div>
            <div class="kpiValue">${overall.score}/100</div>
          </div>
          <span class="kpiBadge ${overall.cls}">${overall.label}</span>
        </div>
        <div class="kpiSub">Headline-based context gauge.</div>
        <div class="kpiList">
          <div class="kpiRow"><span>Date</span><span class="kpiBadge badgeNeu">${date || "—"}</span></div>
        </div>
      `;

      el("briefKpiScope").innerHTML = `
        <div class="kpiTop">
          <div>
            <div class="kpiTitle">Briefing scope</div>
            <div class="kpiValue">${total} headlines</div>
          </div>
          <span class="kpiBadge badgeNeu">Sectors</span>
        </div>
        <div class="kpiSub">
          Watchlist: ${String(watchlist||"").split(",").filter(x => x.trim()).length} tickers
        </div>
        <div class="kpiList">
          <div class="kpiRow"><span>Sectors</span><span class="kpiBadge badgeNeu">${sectors.length}</span></div>
        </div>
      `;

      el("briefKpiNotes").innerHTML = `
        <div class="kpiTop">
          <div>
            <div class="kpiTitle">Notes</div>
            <div class="kpiValue">RSS</div>
          </div>
          <span class="kpiBadge badgeNeu">Backend</span>
        </div>
        <div class="kpiSub">${note ? note : "Daily briefing."}</div>
      `;

      el("briefingSectors").innerHTML = sectors.length
        ? sectors.map(sectorPanelHTML).join("")
        : "<div class='subtle'>No sectors returned.</div>";

      const errAll = normalizeNewsErrors(data?.errors);
      el("briefHint").textContent = errAll.length
        ? ("Some feeds failed: " + errAll.slice(0, 3).join(" | "))
        : "Daily briefing loaded.";
    }

    async function loadBriefing(watchlist, force){
      const key = String(watchlist || "").trim();
      if (CACHE_BRIEFING.has(key) && !force){
        renderBriefing(CACHE_BRIEFING.get(key), key);
        return;
      }

      const url = NEWS_BRIEF_URL(key);
      el("briefingSectors").innerHTML = "<div class='subtle'>Loading daily briefing…</div>";
      el("briefHint").textContent = `Loading from: ${url}`;

      try{
        const data = await fetchJson(url);
        CACHE_BRIEFING.set(key, data);
        renderBriefing(data, key);
      }catch(err){
        el("briefingSectors").innerHTML = "<div class='subtle'>Could not load briefing.</div>";
        el("briefHint").textContent = `Backend /news/briefing failed. ${err?.message || ""}`;
        el("briefKpiSentiment").innerHTML = "";
        el("briefKpiScope").innerHTML = "";
        el("briefKpiNotes").innerHTML = "";
      }
    }

    function holdingsRowHTML(item){
      const t = String(item.ticker || "").toUpperCase();
      const holders = Number(item.holders ?? 0);
      return `
        <div class="holdingsItem">
          <div class="cardTop">
            <div>
              <div class="tickerLine">
                <a class="ticker" href="#stock=${encodeURIComponent(t)}">${t}</a>
                <span class="kpiBadge badgeNeu">Holders: <strong>${holders}</strong></span>
              </div>
              <div class="who">Popularity list (idea discovery)</div>
            </div>
            <div class="tags">
              <span class="tag tagSEC">HOLDINGS</span>
            </div>
          </div>
          ${tickerLinksHTML(t)}
        </div>
      `;
    }

    function renderHoldingsCommon(payload){
      const win = Number(payload?.windowDays || WINDOW_DAYS);
      const list = Array.isArray(payload?.commonHoldings) ? payload.commonHoldings : [];

      if (!list.length){
        el("holdingsKpiSimple").innerHTML = `
          <div class="kpiTop">
            <div>
              <div class="kpiTitle">Holdings</div>
              <div class="kpiValue">No data</div>
            </div>
            <span class="kpiBadge badgeNeu">${win}D</span>
          </div>
          <div class="kpiSub">
            If you see this, check backend <span class="mono">/report/holdings/common</span> and that QUIVER_TOKEN exists.
          </div>
        `;
        el("holdingsList").innerHTML = "<div class='subtle'>No holdings data returned.</div>";
        el("holdingsHint").textContent = "If empty, try another window or check backend health.";
        return;
      }

      const sorted = list.slice()
        .map(x => ({ ticker: x.ticker, holders: Number(x.holders||0) }))
        .filter(x => x.ticker)
        .sort((a,b) => (b.holders - a.holders) || String(a.ticker).localeCompare(String(b.ticker)));

      const top = sorted[0];

      el("holdingsKpiSimple").innerHTML = `
        <div class="kpiTop">
          <div>
            <div class="kpiTitle">Top holding</div>
            <div class="kpiValue">${top ? String(top.ticker).toUpperCase() : "—"}</div>
          </div>
          <span class="kpiBadge badgeNeu">${win}D</span>
        </div>
        <div class="kpiSub">Idea discovery. Not a timing signal.</div>
        <div class="kpiList">
          ${sorted.slice(0,5).map(x => `
            <div class="kpiRow">
              <a href="#stock=${encodeURIComponent(String(x.ticker).toUpperCase())}"><strong>${String(x.ticker).toUpperCase()}</strong></a>
              <span class="kpiBadge badgeNeu">${Number(x.holders||0)} holders</span>
            </div>
          `).join("")}
        </div>
      `;

      el("holdingsList").innerHTML = sorted.map(holdingsRowHTML).join("");
      el("holdingsHint").textContent = "Simple view: ticker + holders.";
    }

    async function loadHoldingsCommon(windowDays, force){
      const key = String(windowDays || 365);
      if (CACHE_HOLDINGS.has(key) && !force){
        renderHoldingsCommon(CACHE_HOLDINGS.get(key));
        return;
      }

      const url = HOLDINGS_URL(windowDays || 365);
      el("holdingsList").innerHTML = "<div class='subtle'>Loading holdings…</div>";
      el("holdingsHint").textContent = `Loading from: ${url}`;

      try{
        const data = await fetchJson(url);
        CACHE_HOLDINGS.set(key, data);
        renderHoldingsCommon(data);
      }catch(err){
        renderHoldingsCommon({ windowDays: windowDays || 365, commonHoldings: [] });
        el("holdingsHint").textContent = `Could not load holdings. ${err?.message || ""} (Most common cause: missing QUIVER_TOKEN on backend.)`;
      }
    }

    function dailyItemHTML(it){
      const kind = String(it.kind||"").toUpperCase();
      const t = String(it.ticker||"").toUpperCase();
      const who = String(it.politician||"").trim();
      const chamber = String(it.chamber||"").trim();
      const amt = String(it.amountRange||"").trim();
      const traded = String(it.traded||"");
      const filed = String(it.filed||"");
      const desc = String(it.description||"").trim();

      return `
        <div class="card">
          <div class="cardTop">
            <div>
              <div class="tickerLine">
                ${t ? `<a class="ticker" href="#stock=${encodeURIComponent(t)}">${t}</a>` : `<div class="ticker">N/A</div>`}
                <div class="who">${who}${chamber ? " · " + chamber : ""}</div>
              </div>
            </div>
            <div class="tags">
              ${kind === "BUY" ? `<span class="tag tagBuy">BUY</span>` : `<span class="tag tagSell">SELL</span>`}
              <span class="tag tagSEC">DISCLOSURE</span>
            </div>
          </div>

          <div class="divider"></div>

          <div class="metaRow">
            ${amt ? `<div>Amount: <strong>${amt}</strong></div>` : ""}
            ${traded ? `<div>Traded: <strong>${traded}</strong></div>` : ""}
            ${filed ? `<div>Filed: <strong>${filed}</strong></div>` : ""}
          </div>

          ${desc ? `
            <div class="metaRow">
              <div style="max-width:900px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
                ${desc}
              </div>
            </div>
          ` : ""}

          ${tickerLinksHTML(t)}
        </div>
      `;
    }

    function renderDaily(payload){
      const days = Array.isArray(payload?.days) ? payload.days : [];
      if (!days.length){
        el("dailyTradeList").innerHTML = "<div class='subtle'>No daily trades returned for this window.</div>";
        el("dailyHint").textContent = "If empty, try 180D or 365D. If always empty, check backend QUIVER_TOKEN.";
        return;
      }

      el("dailyTradeList").innerHTML = days.map(d => {
        const items = Array.isArray(d.items) ? d.items : [];
        return `
          <div class="sectionTitle">${d.date}</div>
          ${items.map(dailyItemHTML).join("")}
        `;
      }).join("");

      el("dailyHint").textContent = "Day-by-day activity feed loaded.";
    }

    async function loadDaily(windowDays, force){
      const key = String(windowDays || 30);
      if (CACHE_DAILY.has(key) && !force){
        renderDaily(CACHE_DAILY.get(key));
        return;
      }

      const url = DAILY_URL(windowDays || 30);
      el("dailyTradeList").innerHTML = "<div class='subtle'>Loading daily feed…</div>";
      el("dailyHint").textContent = `Loading from: ${url}`;

      try{
        const data = await fetchJson(url);
        CACHE_DAILY.set(key, data);
        renderDaily(data);
      }catch(err){
        renderDaily({ days: [] });
        el("dailyHint").textContent = `Could not load daily feed. ${err?.message || ""} (Most common cause: missing QUIVER_TOKEN on backend.)`;
      }
    }

    function renderCrypto(payload){
      const overall = sentimentBadge(payload?.overallSentiment);
      const date = String(payload?.date || "");
      const note = String(payload?.note || "");
      const errs = Array.isArray(payload?.errors) ? payload.errors : [];

      el("cryptoKpiOverall").innerHTML = `
        <div class="kpiTop">
          <div>
            <div class="kpiTitle">Overall sentiment</div>
            <div class="kpiValue">${overall.score}/100</div>
          </div>
          <span class="kpiBadge ${overall.cls}">${overall.label}</span>
        </div>
        <div class="kpiSub">Headline-based gauge.</div>
        <div class="kpiList">
          <div class="kpiRow"><span>Date</span><span class="kpiBadge badgeNeu">${date || "—"}</span></div>
        </div>
      `;

      const coins = Array.isArray(payload?.coins) ? payload.coins : [];
      if (!coins.length){
        el("cryptoCoins").innerHTML = "<div class='subtle'>No coins returned.</div>";
      } else {
        el("cryptoCoins").innerHTML = coins.map(c => {
          const symRaw = String(c?.symbol || "").toUpperCase();
          const sym = (symRaw === "LINK") ? "Chainlink" : symRaw;
          const sent = sentimentBadge(c?.sentiment);
          const sum = String(c?.summary || "").trim();
          const heads = Array.isArray(c?.headlines) ? c.headlines : [];

          return `
            <details class="briefingDetails">
              <summary>
                <div class="summaryLeft">
                  <div class="summaryTitle">${sym}</div>
                  <div class="summaryMeta">${heads.length} headlines · Sentiment ${sent.score}/100${sum ? " · " + sum : ""}</div>
                </div>
                <div class="tags">
                  <span class="kpiBadge badgeNeu">${heads.length}</span>
                  <span class="kpiBadge ${sent.cls}">${sent.label} · ${sent.score}</span>
                </div>
              </summary>

              <div class="divider"></div>

              <div class="kpiSub"><strong>Scan:</strong> ${sum || "No summary available."}</div>

              <div class="divider"></div>

              <div class="kpiSub"><strong>Headlines:</strong></div>
              <div class="headlineList">
                ${heads.length ? heads.map(h => headlineHTML({
                  title: h.title,
                  link: h.link,
                  published: h.published,
                  sourceFeed: h.source,
                  summary: h.summary
                })).join("") : "<div class='subtle'>No headlines.</div>"}
              </div>
            </details>
          `;
        }).join("");
      }

      el("cryptoHint").textContent = errs.length
        ? ("Some feeds failed: " + errs.slice(0, 3).join(" | "))
        : (note || "Crypto briefing loaded.");
    }

    async function loadCrypto(coins, force){
      const key = String(coins || "").trim().toUpperCase();
      if (CACHE_CRYPTO.has(key) && !force){
        renderCrypto(CACHE_CRYPTO.get(key));
        return;
      }

      const url = CRYPTO_URL(key);
      el("cryptoCoins").innerHTML = "<div class='subtle'>Loading crypto briefing…</div>";
      el("cryptoHint").textContent = `Loading from: ${url}`;

      try{
        const data = await fetchJson(url);
        CACHE_CRYPTO.set(key, data);
        renderCrypto(data);
      }catch(err){
        el("cryptoCoins").innerHTML = "<div class='subtle'>Could not load crypto.</div>";
        el("cryptoHint").textContent = `Backend /crypto/news/briefing failed. ${err?.message || ""}`;
        el("cryptoKpiOverall").innerHTML = "";
      }
    }

    function showTab(tab){
      ACTIVE_TAB = tab;
      setTitleForTab(tab);

      el("viewNews").style.display = tab === "News" ? "" : "none";
      el("viewCongress").style.display = tab === "Congress" ? "" : "none";
      el("viewCrypto").style.display = tab === "Crypto" ? "" : "none";
      el("viewPerf").style.display = tab === "Performance" ? "" : "none";

      el("tabNewsBtn").classList.toggle("tabBtnActive", tab === "News");
      el("tabCongressBtn").classList.toggle("tabBtnActive", tab === "Congress");
      el("tabCryptoBtn").classList.toggle("tabBtnActive", tab === "Crypto");
      el("tabPerfBtn").classList.toggle("tabBtnActive", tab === "Performance");

      el("navNews").classList.toggle("tabBottomActive", tab === "News");
      el("navCongress").classList.toggle("tabBottomActive", tab === "Congress");
      el("navCrypto").classList.toggle("tabBottomActive", tab === "Crypto");
      el("navPerf").classList.toggle("tabBottomActive", tab === "Performance");

      el("mainWindowTools").style.display = (tab === "Congress") ? "" : "none";

      if (tab === "News"){
        loadMarketRead(false);
        loadBriefing(el("watchInput").value, false);
      }
      if (tab === "Congress"){
        loadHoldingsCommon(WINDOW_DAYS, false);
        loadDaily(WINDOW_DAYS, false);
      }
      if (tab === "Crypto"){
        loadCrypto(el("coinInput").value, false);
      }

      handleHash();
    }

    function setWindow(days){
      WINDOW_DAYS = days;
      el("btn30").classList.toggle("on", WINDOW_DAYS === 30);
      el("btn180").classList.toggle("on", WINDOW_DAYS === 180);
      el("btn365").classList.toggle("on", WINDOW_DAYS === 365);

      if (ACTIVE_TAB === "Congress"){
        loadHoldingsCommon(WINDOW_DAYS, true);
        loadDaily(WINDOW_DAYS, true);
      }
    }

    function handleRefresh(){
      CACHE_READ = null;
      CACHE_BRIEFING.clear();
      CACHE_HOLDINGS.clear();
      CACHE_DAILY.clear();
      CACHE_CRYPTO.clear();

      setStatus("", "Refreshing…", "");
      if (ACTIVE_TAB === "News"){
        loadMarketRead(true);
        loadBriefing(el("watchInput").value, true);
      } else if (ACTIVE_TAB === "Congress"){
        loadHoldingsCommon(WINDOW_DAYS, true);
        loadDaily(WINDOW_DAYS, true);
      } else if (ACTIVE_TAB === "Crypto"){
        loadCrypto(el("coinInput").value, true);
      }
    }

    el("btn30").addEventListener("click", () => setWindow(30));
    el("btn180").addEventListener("click", () => setWindow(180));
    el("btn365").addEventListener("click", () => setWindow(365));

    el("tabNewsBtn").addEventListener("click", () => showTab("News"));
    el("tabCongressBtn").addEventListener("click", () => showTab("Congress"));
    el("tabCryptoBtn").addEventListener("click", () => showTab("Crypto"));
    el("tabPerfBtn").addEventListener("click", () => showTab("Performance"));

    el("navNews").addEventListener("click", () => showTab("News"));
    el("navCongress").addEventListener("click", () => showTab("Congress"));
    el("navCrypto").addEventListener("click", () => showTab("Crypto"));
    el("navPerf").addEventListener("click", () => showTab("Performance"));

    el("refreshBtn").addEventListener("click", handleRefresh);
    window.addEventListener("hashchange", handleHash);

    el("watchApplyBtn").addEventListener("click", () => {
      loadBriefing(el("watchInput").value, true);
    });

    el("readRefreshBtn").addEventListener("click", () => {
      loadMarketRead(true);
    });

    el("cryptoApplyBtn").addEventListener("click", () => {
      loadCrypto(el("coinInput").value, true);
    });

    showTab("News");
    setStatus("ok", "Ready ✅", "");
    el("reportDate").textContent = "";
  </script>
</body>
</html>
