#!/usr/bin/env python3
"""
Nifty 50 Options Strategy Dashboard — GitHub Pages Generator
Aurora Borealis Theme · OI: Sample 3 Ticker Table · Strikes +-10 from Spot
pip install curl_cffi pandas numpy yfinance pytz
"""

import os, json, time, warnings, pytz
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from curl_cffi import requests as curl_requests
import yfinance as yf

warnings.filterwarnings("ignore")

# =================================================================
#  SECTION 1 -- NSE OPTION CHAIN FETCHER
# =================================================================

class NSEOptionChain:
    def __init__(self):
        self.symbol = "NIFTY"

    def _make_session(self):
        headers = {
            "authority": "www.nseindia.com",
            "accept": "application/json, text/plain, */*",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
            "referer": "https://www.nseindia.com/option-chain",
            "accept-language": "en-US,en;q=0.9",
        }
        session = curl_requests.Session()
        try:
            session.get("https://www.nseindia.com/", headers=headers, impersonate="chrome", timeout=15)
            time.sleep(1.5)
            session.get("https://www.nseindia.com/option-chain", headers=headers, impersonate="chrome", timeout=15)
            time.sleep(1)
        except Exception as e:
            print(f"  WARNING  Session warm-up: {e}")
        return session, headers

    def _upcoming_tuesday(self):
        ist_tz    = pytz.timezone("Asia/Kolkata")
        today_ist = datetime.now(ist_tz).date()
        weekday   = today_ist.weekday()
        days_ahead = 7 if weekday == 1 else (1 - weekday) % 7 or 7
        return (today_ist + timedelta(days=days_ahead)).strftime("%d-%b-%Y")

    def _fetch_available_expiries(self, session, headers):
        try:
            url  = f"https://www.nseindia.com/api/option-chain-v3?type=Indices&symbol={self.symbol}"
            resp = session.get(url, headers=headers, impersonate="chrome", timeout=20)
            if resp.status_code == 200:
                expiries = resp.json().get("records", {}).get("expiryDates", [])
                if expiries:
                    print(f"  Available expiries: {expiries[:5]}")
                    return expiries[0]
        except Exception as e:
            print(f"  WARNING  Expiry fetch: {e}")
        return None

    def _fetch_for_expiry(self, session, headers, expiry):
        api_url = (
            f"https://www.nseindia.com/api/option-chain-v3"
            f"?type=Indices&symbol={self.symbol}&expiry={expiry}"
        )
        for attempt in range(1, 3):
            try:
                print(f"    Attempt {attempt}: expiry={expiry}")
                resp = session.get(api_url, headers=headers, impersonate="chrome", timeout=30)
                print(f"    HTTP {resp.status_code}")
                if resp.status_code != 200:
                    time.sleep(2)
                    continue
                json_data  = resp.json()
                data       = json_data.get("records", {}).get("data", [])
                if not data:
                    return None

                underlying = json_data.get("records", {}).get("underlyingValue", 0)
                atm_strike = round(underlying / 50) * 50
                lower_bound = underlying - 500
                upper_bound = underlying + 500

                rows = []
                for item in data:
                    strike = item.get("strikePrice")
                    if strike is None or not (lower_bound <= strike <= upper_bound):
                        continue
                    ce = item.get("CE", {})
                    pe = item.get("PE", {})
                    rows.append({
                        "Strike":       strike,
                        "CE_LTP":       ce.get("lastPrice", 0),
                        "CE_OI":        ce.get("openInterest", 0),
                        "CE_Vol":       ce.get("totalTradedVolume", 0),
                        "PE_LTP":       pe.get("lastPrice", 0),
                        "PE_OI":        pe.get("openInterest", 0),
                        "PE_Vol":       pe.get("totalTradedVolume", 0),
                        "CE_OI_Change": ce.get("changeinOpenInterest", 0),
                        "PE_OI_Change": pe.get("changeinOpenInterest", 0),
                        "CE_IV":        ce.get("impliedVolatility", 0),
                        "PE_IV":        pe.get("impliedVolatility", 0),
                        "CE_Delta":     ce.get("delta", 0),
                        "PE_Delta":     pe.get("delta", 0),
                        "CE_Theta":     ce.get("theta", 0),
                        "PE_Theta":     pe.get("theta", 0),
                    })

                df = pd.DataFrame(rows).sort_values("Strike").reset_index(drop=True)
                print(f"    OK {len(df)} strikes | Spot={underlying:.0f} ATM={atm_strike} Range=[{lower_bound:.0f}-{upper_bound:.0f}]")
                return {"expiry": expiry, "df": df, "underlying": underlying, "atm_strike": atm_strike}
            except Exception as e:
                print(f"    FAIL Attempt {attempt}: {e}")
                time.sleep(2)
        return None

    def fetch(self):
        session, headers = self._make_session()
        expiry = self._upcoming_tuesday()
        print(f"  Fetching option chain for: {expiry}")
        result = self._fetch_for_expiry(session, headers, expiry)
        if result is None:
            real_expiry = self._fetch_available_expiries(session, headers)
            if real_expiry and real_expiry != expiry:
                print(f"  Retrying with NSE expiry: {real_expiry}")
                result = self._fetch_for_expiry(session, headers, real_expiry)
        if result is None:
            print("  ERROR Option chain fetch failed.")
        return result, session, headers


# =================================================================
#  SECTION 1B -- INDIA VIX FETCHER
# =================================================================

def fetch_india_vix(nse_session=None, nse_headers=None):
    if nse_session and nse_headers:
        try:
            resp = nse_session.get(
                "https://www.nseindia.com/api/allIndices",
                headers=nse_headers, impersonate="chrome", timeout=15)
            if resp.status_code == 200:
                for item in resp.json().get("data", []):
                    if "INDIA VIX" in item.get("index", "").upper():
                        last  = float(item.get("last", 0))
                        prev  = float(item.get("previousClose", last))
                        chg   = round(last - prev, 2)
                        chg_p = round((chg / prev * 100), 2) if prev else 0
                        print(f"  India VIX (allIndices): {last:.2f}  {chg:+.2f} ({chg_p:+.2f}%)")
                        return {"value": round(last,2), "prev_close": round(prev,2),
                                "change": chg, "change_pct": chg_p,
                                "high": float(item.get("high", last)),
                                "low":  float(item.get("low",  last)), "status": "live"}
        except Exception as e:
            print(f"  WARNING VIX source1: {e}")

    if nse_session and nse_headers:
        try:
            resp = nse_session.get(
                "https://www.nseindia.com/api/option-chain-v3?type=Indices&symbol=NIFTY",
                headers=nse_headers, impersonate="chrome", timeout=15)
            if resp.status_code == 200:
                v = float(resp.json().get("records", {}).get("vixClose") or 0)
                if v > 0:
                    print(f"  India VIX (vixClose): {v:.2f}")
                    return {"value": round(v,2), "prev_close": round(v,2),
                            "change": 0.0, "change_pct": 0.0,
                            "high": round(v,2), "low": round(v,2), "status": "snapshot"}
        except Exception as e:
            print(f"  WARNING VIX source2: {e}")

    try:
        import yfinance as yf
        hist = yf.Ticker("^INDIAVIX").history(period="2d")
        if not hist.empty:
            last = float(hist.iloc[-1]["Close"])
            prev = float(hist.iloc[-2]["Close"]) if len(hist) > 1 else last
            chg  = round(last - prev, 2)
            chg_p = round((chg / prev * 100), 2) if prev else 0
            print(f"  India VIX (yfinance): {last:.2f}  {chg:+.2f} ({chg_p:+.2f}%)")
            return {"value": round(last,2), "prev_close": round(prev,2),
                    "change": chg, "change_pct": chg_p,
                    "high": float(hist.iloc[-1].get("High", last)),
                    "low":  float(hist.iloc[-1].get("Low",  last)), "status": "live"}
    except Exception as e:
        print(f"  WARNING VIX source3 (yfinance): {e}")

    print("  ERROR: All VIX sources failed.")
    return None


def vix_label(v):
    if   v < 12:  return "Extremely Low",  "#00c896", "rgba(0,200,150,.12)",  "rgba(0,200,150,.35)",  "Sell Premium"
    elif v < 15:  return "Low",            "#4de8b8", "rgba(0,200,150,.08)",  "rgba(0,200,150,.25)",  "Sell Premium"
    elif v < 20:  return "Normal",         "#6480ff", "rgba(100,128,255,.1)", "rgba(100,128,255,.3)", "Balanced"
    elif v < 25:  return "Elevated",       "#ffd166", "rgba(255,209,102,.1)", "rgba(255,209,102,.3)", "Buy Premium"
    elif v < 30:  return "High",           "#ff9a3c", "rgba(255,154,60,.1)",  "rgba(255,154,60,.3)",  "Buy Straddles"
    else:         return "Extreme Fear",   "#ff6b6b", "rgba(255,107,107,.1)", "rgba(255,107,107,.3)", "Buy Puts/Hedge"

# =================================================================
#  SECTION 2 -- OPTION CHAIN ANALYSIS
# =================================================================

def analyze_option_chain(oc_data):
    if not oc_data:
        return None
    df           = oc_data["df"]
    total_ce_oi  = df["CE_OI"].sum()
    total_pe_oi  = df["PE_OI"].sum()
    total_ce_vol = df["CE_Vol"].sum()
    total_pe_vol = df["PE_Vol"].sum()
    pcr_oi       = total_pe_oi  / total_ce_oi  if total_ce_oi  > 0 else 0
    pcr_vol      = total_pe_vol / total_ce_vol if total_ce_vol > 0 else 0
    ce_chg       = int(df["CE_OI_Change"].sum())
    pe_chg       = int(df["PE_OI_Change"].sum())
    net_chg      = pe_chg + ce_chg

    if   ce_chg > 0 and pe_chg < 0:
        oi_dir, oi_sig, oi_icon, oi_cls = "Strong Bearish", "Call Build-up + Put Unwinding", "RED",    "bearish"
    elif ce_chg < 0 and pe_chg > 0:
        oi_dir, oi_sig, oi_icon, oi_cls = "Strong Bullish", "Put Build-up + Call Unwinding", "GREEN",  "bullish"
    elif ce_chg > 0 and pe_chg > 0:
        if   pe_chg > ce_chg * 1.5:
            oi_dir, oi_sig, oi_icon, oi_cls = "Bullish",            "Put Build-up Dominant",      "GREEN",  "bullish"
        elif ce_chg > pe_chg * 1.5:
            oi_dir, oi_sig, oi_icon, oi_cls = "Bearish",            "Call Build-up Dominant",     "RED",    "bearish"
        else:
            oi_dir, oi_sig, oi_icon, oi_cls = "Neutral (High Vol)", "Both Calls & Puts Building", "YELLOW", "neutral"
    elif ce_chg < 0 and pe_chg < 0:
        oi_dir, oi_sig, oi_icon, oi_cls = "Neutral (Unwinding)", "Both Calls & Puts Unwinding", "YELLOW", "neutral"
    else:
        if   net_chg > 0: oi_dir, oi_sig, oi_icon, oi_cls = "Moderately Bullish", "Net Put Accumulation",  "GREEN",  "bullish"
        elif net_chg < 0: oi_dir, oi_sig, oi_icon, oi_cls = "Moderately Bearish", "Net Call Accumulation", "RED",    "bearish"
        else:              oi_dir, oi_sig, oi_icon, oi_cls = "Neutral",            "Balanced OI Changes",   "YELLOW", "neutral"

    max_ce_row   = df.loc[df["CE_OI"].idxmax()]
    max_pe_row   = df.loc[df["PE_OI"].idxmax()]
    df["pain"]   = abs(df["CE_OI"] - df["PE_OI"])
    max_pain_row = df.loc[df["pain"].idxmin()]
    top_ce = df.nlargest(5, "CE_OI")[["Strike", "CE_OI", "CE_LTP"]].to_dict("records")
    top_pe = df.nlargest(5, "PE_OI")[["Strike", "PE_OI", "PE_LTP"]].to_dict("records")

    atm  = oc_data["atm_strike"]
    spot = oc_data["underlying"]

    strikes_data = []
    for _, row in df.iterrows():
        strikes_data.append({
            "strike": int(row["Strike"]),
            "ce_ltp": float(row["CE_LTP"]),
            "pe_ltp": float(row["PE_LTP"]),
            "ce_iv":  float(row.get("CE_IV", 15)),
            "pe_iv":  float(row.get("PE_IV", 15)),
            "ce_oi":  int(row["CE_OI"]),
            "pe_oi":  int(row["PE_OI"]),
        })

    # Bull/bear force for gauge
    bull_force = (abs(pe_chg) if pe_chg > 0 else 0) + (abs(ce_chg) if ce_chg < 0 else 0)
    bear_force = (abs(ce_chg) if ce_chg > 0 else 0) + (abs(pe_chg) if pe_chg < 0 else 0)
    total_f    = bull_force + bear_force or 1
    bull_pct   = round(bull_force / total_f * 100)
    bear_pct   = 100 - bull_pct

    return {
        "expiry":        oc_data["expiry"],
        "underlying":    oc_data["underlying"],
        "atm_strike":    oc_data["atm_strike"],
        "pcr_oi":        round(pcr_oi, 3),
        "pcr_vol":       round(pcr_vol, 3),
        "total_ce_oi":   int(total_ce_oi),
        "total_pe_oi":   int(total_pe_oi),
        "max_ce_strike": int(max_ce_row["Strike"]),
        "max_ce_oi":     int(max_ce_row["CE_OI"]),
        "max_pe_strike": int(max_pe_row["Strike"]),
        "max_pe_oi":     int(max_pe_row["PE_OI"]),
        "max_pain":      int(max_pain_row["Strike"]),
        "ce_chg":        ce_chg,
        "pe_chg":        pe_chg,
        "net_chg":       net_chg,
        "oi_dir":        oi_dir,
        "oi_sig":        oi_sig,
        "oi_icon":       oi_icon,
        "oi_cls":        oi_cls,
        "top_ce":        top_ce,
        "top_pe":        top_pe,
        "strikes_data":  strikes_data,
        "bull_pct":      bull_pct,
        "bear_pct":      bear_pct,
        "bull_force":    bull_force,
        "bear_force":    bear_force,
        "df":            df,
    }


# =================================================================
#  SECTION 3 -- TECHNICAL ANALYSIS
# =================================================================

def get_technical_data():
    try:
        print("  Fetching technical data from yfinance...")
        nifty = yf.Ticker("^NSEI")
        df    = nifty.history(period="1y")
        if df.empty:
            return None
        df["SMA_20"]  = df["Close"].rolling(20).mean()
        df["SMA_50"]  = df["Close"].rolling(50).mean()
        df["SMA_200"] = df["Close"].rolling(200).mean()
        delta = df["Close"].diff()
        gain  = delta.where(delta > 0, 0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df["RSI"]    = 100 - (100 / (1 + gain / loss))
        df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
        df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"]   = df["EMA_12"] - df["EMA_26"]
        df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        latest = df.iloc[-1]
        cp     = latest["Close"]

        s1 = s2 = r1 = r2 = None
        try:
            df_1h = nifty.history(interval="1h", period="60d")
            if not df_1h.empty:
                recent_1h = df_1h.tail(120)
                highs = sorted(recent_1h["High"].values)
                lows  = sorted(recent_1h["Low"].values)
                res_c = [h for h in highs if cp < h <= cp + 200]
                sup_c = [l for l in lows  if cp - 200 <= l < cp]
                if len(res_c) >= 4:
                    r1 = round(float(np.percentile(res_c, 40)) / 25) * 25
                    r2 = round(float(np.percentile(res_c, 80)) / 25) * 25
                if len(sup_c) >= 4:
                    s1 = round(float(np.percentile(sup_c, 70)) / 25) * 25
                    s2 = round(float(np.percentile(sup_c, 20)) / 25) * 25
                if r1 and r1 <= cp:         r1 = round((cp + 50) / 25) * 25
                if r2 and r1 and r2 <= r1:  r2 = r1 + 75
                if s1 and s1 >= cp:         s1 = round((cp - 50) / 25) * 25
                if s2 and s1 and s2 >= s1:  s2 = s1 - 75
        except Exception as e:
            print(f"  WARNING 1H data: {e}")

        recent_d   = df.tail(60)
        resistance = r1 if r1 else recent_d["High"].quantile(0.90)
        support    = s1 if s1 else recent_d["Low"].quantile(0.10)
        strong_res = r2 if r2 else resistance + 100
        strong_sup = s2 if s2 else support - 100

        df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))
        hv = df["log_ret"].tail(20).std() * np.sqrt(252) * 100

        print(f"  Technical OK | CMP={cp:.2f} RSI={latest['RSI']:.1f} MACD={latest['MACD']:.2f} HV={hv:.1f}%")
        return {
            "price":       cp,
            "sma20":       latest["SMA_20"],
            "sma50":       latest["SMA_50"],
            "sma200":      latest["SMA_200"],
            "rsi":         latest["RSI"],
            "macd":        latest["MACD"],
            "signal_line": latest["Signal"],
            "support":     support,
            "resistance":  resistance,
            "strong_sup":  strong_sup,
            "strong_res":  strong_res,
            "hv":          round(float(hv), 2),
        }
    except Exception as e:
        print(f"  ERROR Technical: {e}")
        return None


# =================================================================
#  SECTION 4 -- MARKET DIRECTION SCORING
# =================================================================

def compute_market_direction(tech, oc_analysis):
    if not tech:
        return {"bias": "UNKNOWN", "confidence": "LOW", "bull": 0, "bear": 0, "diff": 0}

    cp   = tech["price"]
    bull = bear = 0

    for sma in ["sma20", "sma50", "sma200"]:
        if cp > tech[sma]: bull += 1
        else:               bear += 1

    rsi = tech["rsi"]
    if   rsi > 70: bear += 1
    elif rsi < 30: bull += 2

    if tech["macd"] > tech["signal_line"]: bull += 1
    else:                                   bear += 1

    if oc_analysis:
        pcr = oc_analysis["pcr_oi"]
        if   pcr > 1.2: bull += 2
        elif pcr < 0.7: bear += 2
        mp = oc_analysis["max_pain"]
        if   cp > mp + 100: bear += 1
        elif cp < mp - 100: bull += 1

    diff = bull - bear
    print(f"  Score -> Bullish:{bull}  Bearish:{bear}  Diff:{diff}")

    if   diff >= 3:  bias, bias_cls = "BULLISH",  "bullish"; confidence = "HIGH" if diff >= 4 else "MEDIUM"
    elif diff <= -3: bias, bias_cls = "BEARISH",  "bearish"; confidence = "HIGH" if diff <= -4 else "MEDIUM"
    else:            bias, bias_cls = "SIDEWAYS", "neutral"; confidence = "MEDIUM"

    return {"bias": bias, "bias_cls": bias_cls, "confidence": confidence,
            "bull": bull, "bear": bear, "diff": diff}


# =================================================================
#  SECTION 5 -- HTML SECTION BUILDERS
# =================================================================

def _cls_color(cls):
    return "#00c896" if cls == "bullish" else ("#ff6b6b" if cls == "bearish" else "#6480ff")

def _cls_bg(cls):
    return ("rgba(0,200,150,.08)"   if cls == "bullish" else
            "rgba(255,107,107,.08)" if cls == "bearish" else "rgba(100,128,255,.08)")

def _cls_bdr(cls):
    return ("rgba(0,200,150,.22)"   if cls == "bullish" else
            "rgba(255,107,107,.22)" if cls == "bearish" else "rgba(100,128,255,.22)")


# =================================================================
#  SECTION 5A -- COMPACT DUAL GAUGE HERO
# =================================================================

def build_dual_gauge_hero(oc, tech, md, ts):
    """
    Compact single-row hero:
      [Bull Gauge] | [Signal + Bars] | [Bear Gauge] | [5 Stats + Bias/Ts]
    Fixed height ~92px so it never dominates the page.
    """
    # ── Pull data ──────────────────────────────────────────────
    if oc:
        ce_chg     = oc["ce_chg"]
        pe_chg     = oc["pe_chg"]
        net_chg    = oc["net_chg"]
        bull_pct   = oc["bull_pct"]
        bear_pct   = oc["bear_pct"]
        pcr        = oc["pcr_oi"]
        oi_sig     = oc["oi_sig"]
        oi_dir     = oc["oi_dir"]
        oi_cls     = oc["oi_cls"]
        expiry     = oc["expiry"]
        underlying = oc["underlying"]
        atm        = oc["atm_strike"]
        max_pain   = oc["max_pain"]
        def fmt_oi(n):
            if abs(n) >= 1_000_000: return f"{'+' if n>0 else ''}{n/1_000_000:.1f}M"
            if abs(n) >= 1_000:     return f"{'+' if n>0 else ''}{n/1_000:.0f}K"
            return f"{n:+,}"
        bull_label = fmt_oi(pe_chg) if pe_chg > 0 else fmt_oi(abs(ce_chg) if ce_chg < 0 else net_chg)
        bear_label = fmt_oi(ce_chg) if ce_chg > 0 else fmt_oi(abs(pe_chg) if pe_chg < 0 else -net_chg)
    else:
        ce_chg = pe_chg = net_chg = 0
        bull_pct = bear_pct = 50
        pcr = 1.0
        oi_sig = "No Data"; oi_dir = "UNKNOWN"; oi_cls = "neutral"
        expiry = "N/A"; underlying = 0; atm = 0; max_pain = 0
        bull_label = "N/A"; bear_label = "N/A"

    cp   = tech["price"] if tech else 0
    bias = md["bias"]
    conf = md["confidence"]
    bull_sc = md["bull"]
    bear_sc = md["bear"]
    diff    = md["diff"]

    dir_col = _cls_color(oi_cls)
    pcr_col = "#00c896" if pcr > 1.2 else ("#ff6b6b" if pcr < 0.7 else "#6480ff")
    b_col   = _cls_color(md.get("bias_cls", "neutral"))
    b_bg    = _cls_bg(md.get("bias_cls", "neutral"))
    b_bdr   = _cls_bdr(md.get("bias_cls", "neutral"))

    # SVG gauge — r=29, circumference=182.2
    C = 194.8
    def clamp(v, lo=3, hi=97): return max(lo, min(hi, v))
    bull_offset = C * (1 - clamp(bull_pct) / 100)
    bear_offset = C * (1 - clamp(bear_pct) / 100)
    oi_bar_w    = clamp(bull_pct)
    bear_bar_w  = clamp(bear_pct)
    pcr_bar_w   = clamp(min(pcr / 2.0 * 100, 97))

    b_arrow = "▲" if bias == "BULLISH" else ("▼" if bias == "BEARISH" else "◆")

    return f"""
<div class="hero">

  <!-- ① BULL GAUGE -->
  <div class="h-bull">
    <div class="gauge-wrap">
      <svg width="76" height="76" viewBox="0 0 76 76">
        <circle cx="38" cy="38" r="31" fill="none" stroke="rgba(255,255,255,.05)" stroke-width="6"/>
        <circle cx="38" cy="38" r="31" fill="none"
          stroke="url(#bull-g)" stroke-width="6" stroke-linecap="round"
          stroke-dasharray="{C}" stroke-dashoffset="{bull_offset:.1f}"
          style="transform:rotate(-90deg);transform-origin:38px 38px;
                 transition:stroke-dashoffset 1s ease;"/>
        <defs>
          <linearGradient id="bull-g" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="#00c896"/>
            <stop offset="100%" stop-color="#4de8b8"/>
          </linearGradient>
        </defs>
      </svg>
      <div class="gauge-inner">
        <div class="g-val" style="color:#00c896;">{bull_label}</div>
        <div class="g-lbl">OI BULL</div>
      </div>
    </div>
  </div>

  <!-- ② MIDDLE: signal + pill bars -->
  <div class="h-mid">
    <div class="h-eyebrow">OI NET SIGNAL &middot; {expiry} &middot; SPOT &#8377;{underlying:,.0f}</div>
    <div class="h-signal" style="color:{dir_col};">{oi_dir.upper()}</div>
    <div class="h-sub">{oi_sig} &middot; PCR&nbsp;<span style="color:{pcr_col};font-weight:700;">{pcr:.3f}</span></div>
    <div class="h-divider"></div>
    <div class="pill-bars">
      <div class="pill-row">
        <div class="pill-dot" style="background:#00c896;box-shadow:0 0 5px rgba(0,200,150,.5);"></div>
        <div class="pill-lbl">BULL STRENGTH</div>
        <div class="pill-track-wrap">
          <div class="pill-track">
            <div class="pill-fill" style="width:{oi_bar_w}%;background:linear-gradient(90deg,#00c896,#4de8b8);"></div>
          </div>
          <div class="pill-num" style="color:#00c896;">{bull_pct}%</div>
        </div>
      </div>
      <div class="pill-row">
        <div class="pill-dot" style="background:#ff6b6b;box-shadow:0 0 5px rgba(255,107,107,.4);"></div>
        <div class="pill-lbl">BEAR STRENGTH</div>
        <div class="pill-track-wrap">
          <div class="pill-track">
            <div class="pill-fill" style="width:{bear_bar_w}%;background:linear-gradient(90deg,#ff6b6b,#ff9090);"></div>
          </div>
          <div class="pill-num" style="color:#ff6b6b;">{bear_pct}%</div>
        </div>
      </div>
    </div>
  </div>

  <!-- ③ BEAR GAUGE -->
  <div class="h-bear">
    <div class="gauge-wrap">
      <svg width="76" height="76" viewBox="0 0 76 76">
        <circle cx="38" cy="38" r="31" fill="none" stroke="rgba(255,255,255,.05)" stroke-width="6"/>
        <circle cx="38" cy="38" r="31" fill="none"
          stroke="url(#bear-g)" stroke-width="6" stroke-linecap="round"
          stroke-dasharray="{C}" stroke-dashoffset="{bear_offset:.1f}"
          style="transform:rotate(-90deg);transform-origin:38px 38px;
                 transition:stroke-dashoffset 1s ease;"/>
        <defs>
          <linearGradient id="bear-g" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="#ff6b6b"/>
            <stop offset="100%" stop-color="#ff9090"/>
          </linearGradient>
        </defs>
      </svg>
      <div class="gauge-inner">
        <div class="g-val" style="color:#ff6b6b;">{bear_label}</div>
        <div class="g-lbl">OI BEAR</div>
      </div>
    </div>
  </div>

  <!-- ④ RIGHT STAT PANEL -->
  <div class="h-stats">
    <!-- Top: 5 key stat cells -->
    <div class="h-stat-row">
      <div class="h-stat">
        <div class="h-stat-lbl">NIFTY SPOT</div>
        <div class="h-stat-val" style="color:rgba(255,255,255,.85);">&#8377;{cp:,.2f}</div>
      </div>
      <div class="h-stat">
        <div class="h-stat-lbl">ATM STRIKE</div>
        <div class="h-stat-val" style="color:#00c896;">&#8377;{atm:,}</div>
      </div>
      <div class="h-stat">
        <div class="h-stat-lbl">EXPIRY</div>
        <div class="h-stat-val" style="color:#00c8e0;">{expiry}</div>
      </div>
      <div class="h-stat">
        <div class="h-stat-lbl">PCR (OI)</div>
        <div class="h-stat-val" style="color:{pcr_col};">{pcr:.3f}</div>
      </div>
      <div class="h-stat">
        <div class="h-stat-lbl">MAX PAIN</div>
        <div class="h-stat-val" style="color:#ffd166;">&#8377;{max_pain:,}</div>
      </div>
    </div>
    <!-- Bottom: bias chip + confidence + score + timestamp -->
    <div class="h-stat-bottom">
      <div class="h-bias-row">
        <span class="h-chip" style="background:{b_bg};color:{b_col};border:1px solid {b_bdr};">
          {b_arrow}&nbsp;{bias}
        </span>
        <span class="h-chip" style="background:rgba(255,209,102,.1);color:#ffd166;
          border:1px solid rgba(255,209,102,.3);">{conf}&nbsp;CONF</span>
        <span class="h-score">Bull&nbsp;{bull_sc} &middot; Bear&nbsp;{bear_sc} &middot; Diff&nbsp;{diff:+d}</span>
      </div>
      <div class="h-ts">{ts}</div>
    </div>
  </div>

</div>
"""


def build_oi_html(oc):
    ce       = oc["ce_chg"]
    pe       = oc["pe_chg"]
    net      = oc["net_chg"]
    expiry   = oc["expiry"]
    oi_dir   = oc["oi_dir"]
    oi_sig   = oc["oi_sig"]
    oi_cls   = oc["oi_cls"]
    pcr      = oc["pcr_oi"]
    total_ce = oc["total_ce_oi"]
    total_pe = oc["total_pe_oi"]
    max_ce_s = oc["max_ce_strike"]
    max_pe_s = oc["max_pe_strike"]
    max_pain = oc["max_pain"]
    underlying = oc["underlying"]

    dir_col  = _cls_color(oi_cls)
    dir_bg   = _cls_bg(oi_cls)
    dir_bdr  = _cls_bdr(oi_cls)

    bull_force = (abs(pe) if pe > 0 else 0) + (abs(ce) if ce < 0 else 0)
    bear_force = (abs(ce) if ce > 0 else 0) + (abs(pe) if pe < 0 else 0)
    total_f    = bull_force + bear_force or 1
    bull_pct   = round(bull_force / total_f * 100)
    bear_pct   = 100 - bull_pct

    pcr_col  = "#00c896" if pcr > 1.2 else ("#ff6b6b" if pcr < 0.7 else "#6480ff")
    ce_col   = "#00c896" if ce  < 0 else "#ff6b6b"
    pe_col   = "#00c896" if pe  > 0 else "#ff6b6b"
    net_col  = "#00c896" if net > 0 else "#ff6b6b"

    strength_bar = (
        f"<div style=\"height:3px;background:rgba(255,255,255,.06);border-radius:2px;"
        f"overflow:hidden;width:80px;display:inline-block;vertical-align:middle;margin:0 6px;\">"
        f"<div style=\"width:{bull_pct}%;height:100%;"
        f"background:linear-gradient(90deg,#00c896,#6480ff);border-radius:2px;"
        f"box-shadow:0 0 6px rgba(0,200,150,.4);\"></div></div>"
    )

    return (
        f"<div class=\"section\">"
        f"<div class=\"sec-title\">OPEN INTEREST DASHBOARD"
        f"<span class=\"sec-sub\">Spot &#177;500 pts (&#177;10 strikes) &middot; Expiry: {expiry} &middot; Spot: &#8377;{underlying:,.2f}</span></div>"

        f"<div style=\"display:flex;align-items:center;gap:14px;flex-wrap:wrap;"
        f"padding:12px 18px;border-radius:12px;margin-bottom:14px;"
        f"background:{dir_bg};border:1px solid {dir_bdr};\">"
        f"<div>"
        f"<div style=\"font-size:9px;letter-spacing:2px;color:rgba(255,255,255,.3);text-transform:uppercase;margin-bottom:4px;\">OI DIRECTION</div>"
        f"<div style=\"font-size:22px;font-weight:700;color:{dir_col};line-height:1;\">{oi_dir}</div>"
        f"<div style=\"font-size:11px;color:{dir_col};opacity:.7;margin-top:3px;\">{oi_sig}</div>"
        f"</div>"
        f"<div style=\"margin-left:auto;text-align:right;\">"
        f"<div style=\"font-size:10px;color:rgba(255,255,255,.3);margin-bottom:6px;\">"
        f"<span style=\"color:#00c896;font-weight:700;\">{bull_pct}% Bull</span>"
        f"{strength_bar}"
        f"<span style=\"color:#ff6b6b;font-weight:700;\">{bear_pct}% Bear</span>"
        f"</div>"
        f"<span style=\"display:inline-block;padding:4px 14px;border-radius:20px;font-size:10px;font-weight:700;"
        f"color:{dir_col};background:{dir_bg};border:1px solid {dir_bdr};\">{oi_dir}</span>"
        f"</div></div>"

        f"<div class=\"oi-ticker-table\">"

        f"<div class=\"oi-ticker-hdr\" style=\"background:rgba(0,200,150,.05);border-bottom:1px solid rgba(0,200,150,.1);\">"
        f"<div class=\"oi-ticker-hdr-label\" style=\"color:rgba(0,200,150,.8);\">&#9651; CHANGE IN OI</div>"
        f"<div class=\"oi-ticker-hdr-cell\">Call OI Change</div>"
        f"<div class=\"oi-ticker-hdr-cell\">Put OI Change</div>"
        f"<div class=\"oi-ticker-hdr-cell\">Net OI Change</div>"
        f"<div class=\"oi-ticker-hdr-cell\">Interpretation</div>"
        f"<div class=\"oi-ticker-hdr-cell\">Signal</div>"
        f"</div>"
        f"<div class=\"oi-ticker-row\">"
        f"<div class=\"oi-ticker-metric\">OI Change</div>"
        f"<div class=\"oi-ticker-cell\" style=\"color:{ce_col};font-family:'DM Mono',monospace;font-weight:700;font-size:15px;\">{ce:+,}</div>"
        f"<div class=\"oi-ticker-cell\" style=\"color:{pe_col};font-family:'DM Mono',monospace;font-weight:700;font-size:15px;\">{pe:+,}</div>"
        f"<div class=\"oi-ticker-cell\" style=\"color:{net_col};font-family:'DM Mono',monospace;font-weight:700;font-size:15px;\">{net:+,}</div>"
        f"<div class=\"oi-ticker-cell\" style=\"font-size:11px;color:rgba(255,255,255,.5);\">{oi_sig}</div>"
        f"<div class=\"oi-ticker-cell\">"
        f"<span style=\"padding:4px 12px;border-radius:6px;font-size:11px;font-weight:700;"
        f"color:{dir_col};background:{dir_bg};border:1px solid {dir_bdr};\">{oi_dir}</span>"
        f"</div></div>"

        f"<div class=\"oi-ticker-hdr\" style=\"background:rgba(100,128,255,.05);border-top:1px solid rgba(255,255,255,.04);border-bottom:1px solid rgba(100,128,255,.1);\">"
        f"<div class=\"oi-ticker-hdr-label\" style=\"color:rgba(100,128,255,.8);\">&#9632; OPEN INTEREST</div>"
        f"<div class=\"oi-ticker-hdr-cell\">Total CE OI</div>"
        f"<div class=\"oi-ticker-hdr-cell\">Total PE OI</div>"
        f"<div class=\"oi-ticker-hdr-cell\">PCR (OI)</div>"
        f"<div class=\"oi-ticker-hdr-cell\">Max CE Strike</div>"
        f"<div class=\"oi-ticker-hdr-cell\">Max PE Strike</div>"
        f"</div>"
        f"<div class=\"oi-ticker-row\">"
        f"<div class=\"oi-ticker-metric\">Snapshot</div>"
        f"<div class=\"oi-ticker-cell\" style=\"color:#ff6b6b;font-family:'DM Mono',monospace;font-weight:700;font-size:15px;\">{total_ce:,}</div>"
        f"<div class=\"oi-ticker-cell\" style=\"color:#00c896;font-family:'DM Mono',monospace;font-weight:700;font-size:15px;\">{total_pe:,}</div>"
        f"<div class=\"oi-ticker-cell\" style=\"color:{pcr_col};font-family:'DM Mono',monospace;font-weight:700;font-size:15px;\">{pcr:.3f}</div>"
        f"<div class=\"oi-ticker-cell\" style=\"color:#ff6b6b;font-family:'DM Mono',monospace;font-weight:700;font-size:15px;\">&#8377;{max_ce_s:,}</div>"
        f"<div class=\"oi-ticker-cell\" style=\"color:#00c896;font-family:'DM Mono',monospace;font-weight:700;font-size:15px;\">&#8377;{max_pe_s:,}</div>"
        f"</div>"

        f"<div style=\"display:flex;align-items:center;justify-content:space-between;"
        f"padding:10px 18px;border-top:1px solid rgba(255,255,255,.04);background:rgba(100,128,255,.03);flex-wrap:wrap;gap:10px;\">"
        f"<div style=\"display:flex;align-items:center;gap:10px;\">"
        f"<span style=\"font-size:9px;letter-spacing:1.5px;text-transform:uppercase;color:rgba(255,255,255,.3);\">MAX PAIN</span>"
        f"<span style=\"font-family:'DM Mono',monospace;font-size:18px;font-weight:700;color:#6480ff;\">&#8377;{max_pain:,}</span>"
        f"<span style=\"font-size:10px;color:rgba(100,128,255,.6);\">Option writers&apos; target</span>"
        f"</div>"
        f"<div style=\"display:flex;gap:16px;flex-wrap:wrap;font-size:10px;color:rgba(255,255,255,.3);\">"
        f"<span>Call OI+ &rarr; <b style=\"color:#ff6b6b;\">Bearish</b></span>"
        f"<span>Call OI&minus; &rarr; <b style=\"color:#00c896;\">Bullish</b></span>"
        f"<span>Put OI+ &rarr; <b style=\"color:#00c896;\">Bullish</b></span>"
        f"<span>Put OI&minus; &rarr; <b style=\"color:#ff6b6b;\">Bearish</b></span>"
        f"<span>PCR&gt;1.2 &rarr; <b style=\"color:#00c896;\">Bullish</b></span>"
        f"<span>PCR&lt;0.7 &rarr; <b style=\"color:#ff6b6b;\">Bearish</b></span>"
        f"</div></div>"
        f"</div></div>"
    )


def build_key_levels_html(tech, oc):
    cp  = tech["price"]
    ss  = tech["strong_sup"]
    s1  = tech["support"]
    r1  = tech["resistance"]
    sr  = tech["strong_res"]
    rng = sr - ss or 1

    def pct(v):
        return round(max(3, min(97, (v - ss) / rng * 100)), 1)

    cp_pct = pct(cp)
    pts_r  = int(r1 - cp)
    pts_s  = int(cp - s1)

    mp_html = ""
    if oc:
        mp_p     = pct(oc["max_pain"])
        max_pain = oc["max_pain"]
        mp_html  = (
            f"<div class=\"kl-node\" style=\"left:{mp_p}%;top:0;transform:translateX(-50%);\">"
            f"<div class=\"kl-dot\" style=\"background:#6480ff;box-shadow:0 0 8px rgba(100,128,255,.5);margin:0 auto 4px;\"></div>"
            f"<div class=\"kl-lbl\" style=\"color:#6480ff;\">Max Pain</div>"
            f"<div class=\"kl-val\" style=\"color:#8aa0ff;\">&#8377;{max_pain:,}</div>"
            f"</div>"
        )

    return (
        f"<div class=\"section\"><div class=\"sec-title\">KEY LEVELS"
        f"<span class=\"sec-sub\">1H Candles &middot; Last 120 bars &middot; Rounded to 25</span></div>"
        f"<div class=\"kl-zone-labels\">"
        f"<span style=\"color:#00c896;\">SUPPORT ZONE</span>"
        f"<span style=\"color:#ff6b6b;\">RESISTANCE ZONE</span>"
        f"</div>"
        f"<div style=\"position:relative;height:58px;\">"
        f"<div class=\"kl-node\" style=\"left:3%;bottom:0;transform:translateX(-50%);\">"
        f"<div class=\"kl-lbl\" style=\"color:#00a07a;\">Strong Sup</div>"
        f"<div class=\"kl-val\" style=\"color:#00c896;\">&#8377;{ss:,.0f}</div>"
        f"<div class=\"kl-dot\" style=\"background:#00a07a;margin:5px auto 0;\"></div></div>"
        f"<div class=\"kl-node\" style=\"left:22%;bottom:0;transform:translateX(-50%);\">"
        f"<div class=\"kl-lbl\" style=\"color:#00c896;\">Support</div>"
        f"<div class=\"kl-val\" style=\"color:#4de8b8;\">&#8377;{s1:,.0f}</div>"
        f"<div class=\"kl-dot\" style=\"background:#00c896;box-shadow:0 0 8px rgba(0,200,150,.5);margin:5px auto 0;\"></div></div>"
        f"<div style=\"position:absolute;left:{cp_pct}%;bottom:6px;transform:translateX(-50%);"
        f"background:linear-gradient(90deg,#00c896,#6480ff);color:#fff;font-size:11px;font-weight:700;"
        f"padding:3px 14px;border-radius:20px;white-space:nowrap;"
        f"box-shadow:0 2px 14px rgba(0,200,150,.35);z-index:10;\">NOW &#8377;{cp:,.0f}</div>"
        f"<div class=\"kl-node\" style=\"left:75%;bottom:0;transform:translateX(-50%);\">"
        f"<div class=\"kl-lbl\" style=\"color:#ff6b6b;\">Resistance</div>"
        f"<div class=\"kl-val\" style=\"color:#ff9090;\">&#8377;{r1:,.0f}</div>"
        f"<div class=\"kl-dot\" style=\"background:#ff6b6b;box-shadow:0 0 8px rgba(255,107,107,.5);margin:5px auto 0;\"></div></div>"
        f"<div class=\"kl-node\" style=\"left:95%;bottom:0;transform:translateX(-50%);\">"
        f"<div class=\"kl-lbl\" style=\"color:#cc4040;\">Strong Res</div>"
        f"<div class=\"kl-val\" style=\"color:#ff6b6b;\">&#8377;{sr:,.0f}</div>"
        f"<div class=\"kl-dot\" style=\"background:#cc4040;margin:5px auto 0;\"></div></div>"
        f"</div>"
        f"<div class=\"kl-gradient-bar\"><div class=\"kl-price-tick\" style=\"left:{cp_pct}%;\"></div></div>"
        f"<div style=\"position:relative;height:54px;\">{mp_html}</div>"
        f"<div class=\"kl-dist-row\">"
        f"<div class=\"kl-dist-box\" style=\"border-color:rgba(255,107,107,.18);\">"
        f"<span style=\"color:var(--muted);\">To Resistance</span>"
        f"<span style=\"color:#ff6b6b;font-weight:700;\">+{pts_r:,} pts</span></div>"
        f"<div class=\"kl-dist-box\" style=\"border-color:rgba(0,200,150,.18);\">"
        f"<span style=\"color:var(--muted);\">To Support</span>"
        f"<span style=\"color:#00c896;font-weight:700;\">-{pts_s:,} pts</span></div>"
        f"</div></div>"
    )


def build_strikes_html(oc):
    if not oc or (not oc["top_ce"] and not oc["top_pe"]):
        return ""

    def ce_rows(rows):
        out = ""
        for i, r in enumerate(rows, 1):
            out += (
                f"<tr><td>{i}</td><td><b>&#8377;{int(r['Strike']):,}</b></td>"
                f"<td>{int(r['CE_OI']):,}</td>"
                f"<td style=\"color:#00c8e0;font-weight:700;\">&#8377;{r['CE_LTP']:.2f}</td></tr>"
            )
        return out

    def pe_rows(rows):
        out = ""
        for i, r in enumerate(rows, 1):
            out += (
                f"<tr><td>{i}</td><td><b>&#8377;{int(r['Strike']):,}</b></td>"
                f"<td>{int(r['PE_OI']):,}</td>"
                f"<td style=\"color:#ff6b6b;font-weight:700;\">&#8377;{r['PE_LTP']:.2f}</td></tr>"
            )
        return out

    return (
        f"<div class=\"section\"><div class=\"sec-title\">TOP 5 STRIKES BY OPEN INTEREST"
        f"<span class=\"sec-sub\">Spot &#177;500 pts &middot; Top 5 CE + PE</span></div>"
        f"<div class=\"strikes-wrap\">"
        f"<div><div class=\"strikes-head\" style=\"color:#00c8e0;\">&#9651; CALL Options (CE)</div>"
        f"<table class=\"s-table\"><thead><tr><th>#</th><th>Strike</th><th>OI</th><th>LTP</th></tr></thead>"
        f"<tbody>{ce_rows(oc['top_ce'])}</tbody></table></div>"
        f"<div><div class=\"strikes-head\" style=\"color:#ff6b6b;\">&#9661; PUT Options (PE)</div>"
        f"<table class=\"s-table\"><thead><tr><th>#</th><th>Strike</th><th>OI</th><th>LTP</th></tr></thead>"
        f"<tbody>{pe_rows(oc['top_pe'])}</tbody></table></div>"
        f"</div></div>"
    )


# =================================================================
#  SECTION 6 -- STRATEGIES WITH PAYOFF CHARTS (SVG-based)
# =================================================================

def make_payoff_svg(shape, bull_color="#00c896", bear_color="#ff6b6b"):
    w, h = 80, 50
    mid  = h // 2
    pad  = 8

    def pts(*coords):
        return " ".join(f"{x},{y}" for x, y in coords)

    shapes = {
        "long_call":         {"profit": [(pad,mid),(40,mid),(72,h-pad)],           "loss": [(pad,mid),(40,mid)]},
        "short_put":         {"profit": [(pad,h-pad),(40,mid),(72,mid)],           "loss": [(pad,mid),(40,mid)]},
        "bull_call_spread":  {"profit": [(pad,mid),(30,mid),(55,h-pad),(72,h-pad)],"loss": [(pad,mid),(30,mid)]},
        "bull_put_spread":   {"profit": [(pad,h-pad),(30,h-pad),(55,mid),(72,mid)],"loss": [(pad,mid),(72,mid)]},
        "call_ratio_back":   {"profit": [(pad,h-pad),(25,mid),(50,mid),(72,h-pad)],"loss": [(pad,mid),(25,mid),(50,mid)]},
        "long_synthetic":    {"profit": [(pad,mid),(72,h-pad)],                    "loss": [(pad,pad),(40,mid)]},
        "range_forward":     {"profit": [(pad,mid),(30,mid),(55,h-pad),(72,h-pad)],"loss": [(pad,pad),(30,pad),(55,mid)]},
        "bull_butterfly":    {"profit": [(pad,mid),(36,h-pad),(54,mid)],           "loss": [(pad,mid),(36,mid),(54,mid),(72,mid)]},
        "bull_condor":       {"profit": [(pad,mid),(28,mid),(36,h-pad),(50,h-pad),(58,mid),(72,mid)],"loss": [(pad,mid),(72,mid)]},
        "short_call":        {"profit": [(pad,mid),(40,mid),(72,pad)],             "loss": [(pad,mid),(40,mid)]},
        "long_put":          {"profit": [(pad,h-pad),(40,mid),(72,mid)],           "loss": [(pad,mid),(40,mid)]},
        "bear_call_spread":  {"profit": [(pad,h-pad),(30,h-pad),(55,mid),(72,mid)],"loss": [(pad,mid),(72,mid)]},
        "bear_put_spread":   {"profit": [(pad,mid),(30,mid),(55,h-pad),(72,h-pad)],"loss": [(pad,mid),(30,mid)]},
        "put_ratio_back":    {"profit": [(pad,h-pad),(25,mid),(50,mid),(72,pad)],  "loss": [(pad,mid),(25,mid),(50,mid)]},
        "short_synthetic":   {"profit": [(pad,mid),(72,pad)],                     "loss": [(pad,h-pad),(40,mid)]},
        "risk_reversal":     {"profit": [(pad,h-pad),(36,mid),(72,pad)],           "loss": [(pad,mid),(36,mid)]},
        "bear_butterfly":    {"profit": [(pad,mid),(36,pad),(54,mid)],             "loss": [(pad,mid),(36,mid),(54,mid),(72,mid)]},
        "bear_condor":       {"profit": [(pad,mid),(28,mid),(36,pad),(50,pad),(58,mid),(72,mid)], "loss": [(pad,mid),(72,mid)]},
        "long_straddle":     {"profit": [(pad,h-pad),(36,mid),(54,mid),(72,h-pad)],"loss": [(pad,mid),(36,mid),(54,mid),(72,mid)]},
        "short_straddle":    {"profit": [(pad,mid),(36,mid),(54,mid),(72,mid)],    "loss": [(pad,h-pad),(36,mid),(54,mid),(72,h-pad)]},
        "long_strangle":     {"profit": [(pad,h-pad),(30,mid),(50,mid),(72,h-pad)],"loss": [(pad,mid),(30,mid),(50,mid),(72,mid)]},
        "short_strangle":    {"profit": [(pad,mid),(30,mid),(50,mid),(72,mid)],    "loss": [(pad,h-pad),(30,mid),(50,mid),(72,h-pad)]},
        "jade_lizard":       {"profit": [(pad,pad),(30,mid),(55,mid),(72,mid)],    "loss": [(pad,mid),(30,mid)]},
        "reverse_jade":      {"profit": [(pad,mid),(30,mid),(55,h-pad),(72,h-pad)],"loss": [(pad,pad),(30,mid)]},
        "call_ratio_spread": {"profit": [(pad,mid),(36,h-pad),(72,mid)],           "loss": [(pad,mid),(36,mid),(72,pad)]},
        "put_ratio_spread":  {"profit": [(pad,mid),(36,h-pad),(72,mid)],           "loss": [(pad,pad),(36,mid),(72,mid)]},
        "batman":            {"profit": [(pad,mid),(20,h-pad),(36,mid),(54,mid),(68,h-pad),(72,mid)],"loss": [(pad,mid),(72,mid)]},
        "long_iron_fly":     {"profit": [(pad,pad),(36,mid),(54,mid),(72,pad)],    "loss": [(pad,mid),(36,h-pad),(54,h-pad),(72,mid)]},
        "short_iron_fly":    {"profit": [(pad,mid),(36,h-pad),(54,h-pad),(72,mid)],"loss": [(pad,pad),(36,mid),(54,mid),(72,pad)]},
        "double_fly":        {"profit": [(pad,mid),(20,h-pad),(36,mid),(54,mid),(68,h-pad),(72,mid)],"loss": [(pad,mid),(72,mid)]},
        "long_iron_condor":  {"profit": [(pad,pad),(24,mid),(36,mid),(54,mid),(66,pad),(72,pad)],   "loss": [(pad,mid),(24,mid),(66,mid),(72,mid)]},
        "short_iron_condor": {"profit": [(pad,mid),(24,h-pad),(36,h-pad),(54,h-pad),(66,mid),(72,mid)],"loss": [(pad,pad),(72,pad)]},
        "double_condor":     {"profit": [(pad,mid),(20,h-pad),(36,mid),(54,mid),(68,h-pad),(72,mid)],"loss": [(pad,mid),(72,mid)]},
        "call_calendar":     {"profit": [(pad,mid),(36,h-pad),(54,mid),(72,mid)],  "loss": [(pad,mid),(36,mid),(54,mid),(72,mid)]},
        "put_calendar":      {"profit": [(pad,mid),(36,h-pad),(54,mid),(72,mid)],  "loss": [(pad,mid),(36,mid),(54,mid),(72,mid)]},
        "diagonal_calendar": {"profit": [(pad,mid),(30,h-pad),(55,mid),(72,mid)],  "loss": [(pad,mid),(30,mid),(55,mid),(72,mid)]},
        "call_butterfly":    {"profit": [(pad,mid),(36,h-pad),(54,mid),(72,mid)],  "loss": [(pad,mid),(72,mid)]},
        "put_butterfly":     {"profit": [(pad,mid),(36,h-pad),(54,mid),(72,mid)],  "loss": [(pad,mid),(72,mid)]},
    }

    s = shapes.get(shape, {"profit": [(pad,mid),(72,mid)], "loss": []})
    profit_pts = pts(*s["profit"])
    loss_pts   = pts(*s["loss"]) if s["loss"] else ""

    def area_path(coords, is_profit):
        if not coords: return ""
        base_y = mid
        col    = bull_color if is_profit else bear_color
        path_d = f"M {coords[0][0]},{base_y} "
        for x, y in coords:
            path_d += f"L {x},{y} "
        path_d += f"L {coords[-1][0]},{base_y} Z"
        return f'<path d="{path_d}" fill="{col}" fill-opacity="0.18"/>'

    profit_area = area_path(s["profit"], True)
    loss_area   = area_path(s["loss"],   False)

    svg = (
        f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">'
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h-pad}" stroke="rgba(255,255,255,.15)" stroke-width="1"/>'
        f'<line x1="{pad}" y1="{mid}" x2="{w-pad}" y2="{mid}" stroke="rgba(255,255,255,.15)" stroke-width="1"/>'
        f'{profit_area}{loss_area}'
        f'<polyline points="{profit_pts}" fill="none" stroke="{bull_color}" stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round"/>'
    )
    if loss_pts:
        svg += f'<polyline points="{loss_pts}" fill="none" stroke="{bear_color}" stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round" stroke-dasharray="3,2"/>'
    svg += '</svg>'
    return svg


STRATEGIES_DATA = {
    "bullish": [
        {"name": "Long Call",         "shape": "long_call",        "risk": "Limited",  "reward": "Unlimited", "legs": "BUY CALL (ATM)",                         "desc": "Buy a call option. Profits as market rises above strike. Risk is limited to premium paid.", "mp": "Unlimited", "ml": "Premium Paid", "be": "Strike + Premium", "lot_size": 75, "margin_mult": 1.0},
        {"name": "Short Put",         "shape": "short_put",        "risk": "Moderate", "reward": "Limited",   "legs": "SELL PUT (OTM)",                          "desc": "Sell a put option below market. Collect premium. Profit if market stays above strike.", "mp": "Premium Received", "ml": "Strike - Premium", "be": "Strike - Premium", "lot_size": 75, "margin_mult": 5.0},
        {"name": "Bull Call Spread",  "shape": "bull_call_spread", "risk": "Limited",  "reward": "Limited",   "legs": "BUY CALL (Low) · SELL CALL (High)",       "desc": "Buy lower call, sell higher call. Reduces cost; caps profit at upper strike.", "mp": "Spread - Debit", "ml": "Net Debit", "be": "Lower Strike + Debit", "lot_size": 75, "margin_mult": 1.5},
        {"name": "Bull Put Spread",   "shape": "bull_put_spread",  "risk": "Limited",  "reward": "Limited",   "legs": "SELL PUT (High) · BUY PUT (Low)",          "desc": "Sell higher put, buy lower put. Credit received upfront. Profit if market stays above higher strike.", "mp": "Net Credit", "ml": "Spread - Credit", "be": "Higher Strike - Credit", "lot_size": 75, "margin_mult": 2.0},
        {"name": "Call Ratio Back Spread","shape":"call_ratio_back","risk": "Limited",  "reward": "Unlimited", "legs": "SELL 1 CALL (Low) · BUY 2 CALLS (High)",  "desc": "Sell fewer calls, buy more higher calls. Benefits from a big upside move.", "mp": "Unlimited", "ml": "Limited", "be": "Varies", "lot_size": 75, "margin_mult": 2.5},
        {"name": "Long Synthetic",    "shape": "long_synthetic",   "risk": "High",     "reward": "Unlimited", "legs": "BUY CALL (ATM) · SELL PUT (ATM)",         "desc": "Replicates owning the underlying. Unlimited profit potential with high risk.", "mp": "Unlimited", "ml": "Unlimited", "be": "ATM Strike", "lot_size": 75, "margin_mult": 6.0},
        {"name": "Range Forward",     "shape": "range_forward",    "risk": "Limited",  "reward": "Limited",   "legs": "BUY CALL (High) · SELL PUT (Low)",        "desc": "Collar-like structure. Profit in a range. Used to hedge existing positions.", "mp": "Capped", "ml": "Capped", "be": "Varies", "lot_size": 75, "margin_mult": 2.0},
        {"name": "Bull Butterfly",    "shape": "bull_butterfly",   "risk": "Limited",  "reward": "Limited",   "legs": "BUY Low CALL · SELL 2 Mid CALL · BUY High CALL", "desc": "Max profit at middle strike. Low cost strategy for moderate bullish view.", "mp": "Mid - Low - Debit", "ml": "Net Debit", "be": "Low+Debit / High-Debit", "lot_size": 75, "margin_mult": 1.2},
        {"name": "Bull Condor",       "shape": "bull_condor",      "risk": "Limited",  "reward": "Limited",   "legs": "BUY Low · SELL Mid-Low · SELL Mid-High · BUY High", "desc": "Four-leg bullish strategy. Profit in a range above current price.", "mp": "Net Credit", "ml": "Spread - Credit", "be": "Lower strikes + Credit", "lot_size": 75, "margin_mult": 1.8},
    ],
    "bearish": [
        {"name": "Short Call",        "shape": "short_call",       "risk": "Unlimited","reward": "Limited",   "legs": "SELL CALL (ATM/OTM)",                     "desc": "Sell a call option above market. Collect premium. Profit if market falls or stays below strike.", "mp": "Premium Received", "ml": "Unlimited", "be": "Strike + Premium", "lot_size": 75, "margin_mult": 5.0},
        {"name": "Long Put",          "shape": "long_put",         "risk": "Limited",  "reward": "High",      "legs": "BUY PUT (ATM)",                           "desc": "Buy a put option. Profits as market falls below strike. Risk is limited to premium paid.", "mp": "Strike - Premium", "ml": "Premium Paid", "be": "Strike - Premium", "lot_size": 75, "margin_mult": 1.0},
        {"name": "Bear Call Spread",  "shape": "bear_call_spread", "risk": "Limited",  "reward": "Limited",   "legs": "SELL CALL (Low) · BUY CALL (High)",       "desc": "Sell lower call, buy higher call. Credit received. Profit if market stays below lower strike.", "mp": "Net Credit", "ml": "Spread - Credit", "be": "Lower Strike + Credit", "lot_size": 75, "margin_mult": 2.0},
        {"name": "Bear Put Spread",   "shape": "bear_put_spread",  "risk": "Limited",  "reward": "Limited",   "legs": "BUY PUT (High) · SELL PUT (Low)",          "desc": "Buy higher put, sell lower put. Cheaper bearish bet with capped profit.", "mp": "Spread - Debit", "ml": "Net Debit", "be": "Higher Strike - Debit", "lot_size": 75, "margin_mult": 1.5},
        {"name": "Put Ratio Back Spread","shape":"put_ratio_back", "risk": "Limited",  "reward": "High",      "legs": "SELL 1 PUT (High) · BUY 2 PUTS (Low)",    "desc": "Sell fewer puts, buy more lower puts. Benefits from a big downside move.", "mp": "High", "ml": "Limited", "be": "Varies", "lot_size": 75, "margin_mult": 2.5},
        {"name": "Short Synthetic",   "shape": "short_synthetic",  "risk": "High",     "reward": "High",      "legs": "SELL CALL (ATM) · BUY PUT (ATM)",         "desc": "Replicates shorting the underlying. Profit as market falls. High risk.", "mp": "Unlimited", "ml": "Unlimited", "be": "ATM Strike", "lot_size": 75, "margin_mult": 6.0},
        {"name": "Risk Reversal",     "shape": "risk_reversal",    "risk": "High",     "reward": "High",      "legs": "BUY PUT (Low) · SELL CALL (High)",        "desc": "Protect downside while giving up upside. Common hedging structure.", "mp": "High", "ml": "High", "be": "Varies", "lot_size": 75, "margin_mult": 3.0},
        {"name": "Bear Butterfly",    "shape": "bear_butterfly",   "risk": "Limited",  "reward": "Limited",   "legs": "BUY Low PUT · SELL 2 Mid PUT · BUY High PUT", "desc": "Max profit at middle strike. Low cost strategy for moderate bearish view.", "mp": "Mid - Low - Debit", "ml": "Net Debit", "be": "Low+Debit / High-Debit", "lot_size": 75, "margin_mult": 1.2},
        {"name": "Bear Condor",       "shape": "bear_condor",      "risk": "Limited",  "reward": "Limited",   "legs": "BUY High · SELL Mid-High · SELL Mid-Low · BUY Low", "desc": "Four-leg bearish strategy. Profit in a range below current price.", "mp": "Net Credit", "ml": "Spread - Credit", "be": "Upper strikes - Credit", "lot_size": 75, "margin_mult": 1.8},
    ],
    "nondirectional": [
        {"name": "Long Straddle",     "shape": "long_straddle",    "risk": "Limited",  "reward": "Unlimited", "legs": "BUY CALL (ATM) + BUY PUT (ATM)",          "desc": "Buy both ATM call and put. Profit from big move in either direction. Best before events.", "mp": "Unlimited", "ml": "Total Premium", "be": "Strike +/- Total Premium", "lot_size": 75, "margin_mult": 1.0},
        {"name": "Short Straddle",    "shape": "short_straddle",   "risk": "Unlimited","reward": "Limited",   "legs": "SELL CALL (ATM) + SELL PUT (ATM)",        "desc": "Sell both ATM call and put. Profit from low volatility. High risk unlimited loss.", "mp": "Total Premium", "ml": "Unlimited", "be": "Strike +/- Total Premium", "lot_size": 75, "margin_mult": 8.0},
        {"name": "Long Strangle",     "shape": "long_strangle",    "risk": "Limited",  "reward": "Unlimited", "legs": "BUY OTM CALL + BUY OTM PUT",              "desc": "Buy OTM call and put. Cheaper than straddle. Needs bigger move to profit.", "mp": "Unlimited", "ml": "Total Premium", "be": "Strikes +/- Premium", "lot_size": 75, "margin_mult": 1.0},
        {"name": "Short Strangle",    "shape": "short_strangle",   "risk": "Unlimited","reward": "Limited",   "legs": "SELL OTM CALL + SELL OTM PUT",            "desc": "Sell OTM call and put. Wider profit range than short straddle. Still high risk.", "mp": "Total Premium", "ml": "Unlimited", "be": "Strikes +/- Premium", "lot_size": 75, "margin_mult": 7.0},
        {"name": "Jade Lizard",       "shape": "jade_lizard",      "risk": "Limited",  "reward": "Limited",   "legs": "SELL OTM PUT + SELL CALL SPREAD",         "desc": "No upside risk. Collect premium. Bearish but risk-defined.", "mp": "Total Credit", "ml": "Below Put Strike", "be": "Put Strike - Credit", "lot_size": 75, "margin_mult": 3.0},
        {"name": "Reverse Jade Lizard","shape":"reverse_jade",     "risk": "Limited",  "reward": "Limited",   "legs": "SELL OTM CALL + SELL PUT SPREAD",         "desc": "No downside risk. Collect premium. Bullish but risk-defined.", "mp": "Total Credit", "ml": "Above Call Strike", "be": "Call Strike + Credit", "lot_size": 75, "margin_mult": 3.0},
        {"name": "Call Ratio Spread", "shape": "call_ratio_spread","risk": "Unlimited","reward": "Limited",   "legs": "BUY 1 CALL (Low) · SELL 2 CALLS (High)", "desc": "Sell more calls than bought. Credit or debit. Risk if big upside move occurs.", "mp": "Net Credit + Spread", "ml": "Unlimited", "be": "Varies", "lot_size": 75, "margin_mult": 4.0},
        {"name": "Put Ratio Spread",  "shape": "put_ratio_spread", "risk": "Unlimited","reward": "Limited",   "legs": "BUY 1 PUT (High) · SELL 2 PUTS (Low)",   "desc": "Sell more puts than bought. Risk if big downside move occurs.", "mp": "Net Credit + Spread", "ml": "Unlimited", "be": "Varies", "lot_size": 75, "margin_mult": 4.0},
        {"name": "Batman Strategy",   "shape": "batman",           "risk": "Limited",  "reward": "Limited",   "legs": "BUY 2 CALLS + SELL 4 CALLS + BUY 2 CALLS","desc": "Double butterfly. Two profit peaks. Complex strategy for range-bound markets.", "mp": "Limited", "ml": "Net Debit", "be": "Multiple", "lot_size": 75, "margin_mult": 2.0},
        {"name": "Long Iron Fly",     "shape": "long_iron_fly",    "risk": "Limited",  "reward": "Limited",   "legs": "BUY CALL · BUY PUT · SELL ATM CALL · SELL ATM PUT", "desc": "Debit iron fly. Profit from a big move. Max loss if price stays at ATM.", "mp": "Wings - Debit", "ml": "Net Debit", "be": "ATM +/- Debit", "lot_size": 75, "margin_mult": 1.5},
        {"name": "Short Iron Fly",    "shape": "short_iron_fly",   "risk": "Limited",  "reward": "Limited",   "legs": "SELL CALL · SELL PUT · BUY OTM CALL · BUY OTM PUT", "desc": "Credit iron fly. Max profit at ATM. Common non-directional strategy.", "mp": "Net Credit", "ml": "Wings - Credit", "be": "ATM +/- Credit", "lot_size": 75, "margin_mult": 3.0},
        {"name": "Double Fly",        "shape": "double_fly",       "risk": "Limited",  "reward": "Limited",   "legs": "TWO BUTTERFLY SPREADS",                   "desc": "Two butterfly spreads at different strikes. Two profit peaks.", "mp": "Limited", "ml": "Net Debit", "be": "Multiple", "lot_size": 75, "margin_mult": 2.0},
        {"name": "Long Iron Condor",  "shape": "long_iron_condor", "risk": "Limited",  "reward": "Limited",   "legs": "BUY CALL SPREAD + BUY PUT SPREAD",        "desc": "Debit condor. Profit from a big move. Opposite of short iron condor.", "mp": "Spreads - Debit", "ml": "Net Debit", "be": "Inner strikes +/- Debit", "lot_size": 75, "margin_mult": 1.5},
        {"name": "Short Iron Condor", "shape": "short_iron_condor","risk": "Limited",  "reward": "Limited",   "legs": "SELL CALL SPREAD + SELL PUT SPREAD",       "desc": "Collect premium from both sides. Profit if price stays in a range.", "mp": "Net Credit", "ml": "Spreads - Credit", "be": "Short strikes +/- Credit", "lot_size": 75, "margin_mult": 3.5},
        {"name": "Double Condor",     "shape": "double_condor",    "risk": "Limited",  "reward": "Limited",   "legs": "TWO CONDOR SPREADS",                      "desc": "Two condor spreads. Wider profit range. Complex multi-leg strategy.", "mp": "Limited", "ml": "Net Debit/Credit", "be": "Multiple", "lot_size": 75, "margin_mult": 2.5},
        {"name": "Call Calendar",     "shape": "call_calendar",    "risk": "Limited",  "reward": "Limited",   "legs": "SELL NEAR-TERM CALL · BUY FAR-TERM CALL", "desc": "Profit from time decay difference. Best when price stays near strike.", "mp": "Limited", "ml": "Net Debit", "be": "Near strike", "lot_size": 75, "margin_mult": 2.0},
        {"name": "Put Calendar",      "shape": "put_calendar",     "risk": "Limited",  "reward": "Limited",   "legs": "SELL NEAR-TERM PUT · BUY FAR-TERM PUT",   "desc": "Profit from time decay. Best when price stays near strike on expiry.", "mp": "Limited", "ml": "Net Debit", "be": "Near strike", "lot_size": 75, "margin_mult": 2.0},
        {"name": "Diagonal Calendar", "shape": "diagonal_calendar","risk": "Limited",  "reward": "Limited",   "legs": "SELL NEAR CALL/PUT · BUY FAR DIFF STRIKE","desc": "Calendar spread with different strikes. Combines time and price movement.", "mp": "Limited", "ml": "Net Debit", "be": "Varies", "lot_size": 75, "margin_mult": 2.0},
        {"name": "Call Butterfly",    "shape": "call_butterfly",   "risk": "Limited",  "reward": "Limited",   "legs": "BUY Low CALL · SELL 2 Mid CALL · BUY High CALL", "desc": "Max profit at middle strike using calls only. Low net debit strategy.", "mp": "Mid - Low - Debit", "ml": "Net Debit", "be": "Low+Debit / High-Debit", "lot_size": 75, "margin_mult": 1.2},
        {"name": "Put Butterfly",     "shape": "put_butterfly",    "risk": "Limited",  "reward": "Limited",   "legs": "BUY High PUT · SELL 2 Mid PUT · BUY Low PUT",   "desc": "Max profit at middle strike using puts only. Low net debit strategy.", "mp": "High - Mid - Debit", "ml": "Net Debit", "be": "High-Debit / Low+Debit", "lot_size": 75, "margin_mult": 1.2},
    ],
}


def build_strategies_html(oc_analysis):
    if oc_analysis:
        spot        = oc_analysis["underlying"]
        atm         = oc_analysis["atm_strike"]
        pcr         = oc_analysis["pcr_oi"]
        max_pain    = oc_analysis["max_pain"]
        strikes_json = json.dumps(oc_analysis.get("strikes_data", []))
    else:
        spot        = 23000
        atm         = 23000
        pcr         = 1.0
        max_pain    = 23000
        strikes_json = "[]"

    def render_cards(strats, cat):
        cards = ""
        for idx, s in enumerate(strats):
            svg     = make_payoff_svg(s["shape"])
            rc      = "#00c896" if s["risk"]   in ("Limited","Low") else ("#ff6b6b" if s["risk"] in ("Unlimited","High") else "#6480ff")
            rwc     = "#00c896" if s["reward"] == "Unlimited" else "#6480ff"
            card_id = f"sc_{cat}_{idx}"
            cards += (
                f'<div class="sc-card" data-cat="{cat}" data-shape="{s["shape"]}" '
                f'data-name="{s["name"]}" data-legs="{s["legs"]}" '
                f'data-risk="{s["risk"]}" data-reward="{s["reward"]}" '
                f'data-margin-mult="{s.get("margin_mult", 1.0)}" '
                f'data-lot-size="{s.get("lot_size", 75)}" '
                f'id="{card_id}">'
                f'<div class="sc-pop-badge" id="pop_{card_id}">—%</div>'
                f'<div class="sc-svg">{svg}</div>'
                f'<div class="sc-body">'
                f'<div class="sc-name">{s["name"]}</div>'
                f'<div class="sc-legs">{s["legs"]}</div>'
                f'<div class="sc-tags">'
                f'<span class="sc-tag" style="color:{rc};border-color:{rc}40;">Risk: {s["risk"]}</span>'
                f'<span class="sc-tag" style="color:{rwc};border-color:{rwc}40;">Reward: {s["reward"]}</span>'
                f'</div>'
                f'</div>'
                f'<div class="sc-detail" id="detail_{card_id}">'
                f'<div class="sc-desc">{s["desc"]}</div>'
                f'<div class="sc-metrics-live" id="metrics_{card_id}">'
                f'<div class="sc-loading">&#9685; Calculating metrics...</div>'
                f'</div>'
                f'</div>'
                f'</div>'
            )
        return cards

    bull_cards  = render_cards(STRATEGIES_DATA["bullish"],       "bullish")
    bear_cards  = render_cards(STRATEGIES_DATA["bearish"],       "bearish")
    nd_cards    = render_cards(STRATEGIES_DATA["nondirectional"],"nondirectional")

    return f"""
<div class="section" id="strat">
  <div class="sec-title">STRATEGIES REFERENCE
    <span class="sec-sub">Live metrics from option chain &middot; Click to expand &middot; PoP = Probability of Profit</span>
  </div>

  <div class="sc-tabs">
    <button class="sc-tab active" onclick="filterStrat('bullish',this)"
      style="border-color:#00c896;color:#00c896;background:rgba(0,200,150,.12);">
      &#9650; BULLISH <span class="sc-cnt" style="background:#00c896;">9</span>
    </button>
    <button class="sc-tab" onclick="filterStrat('bearish',this)"
      style="border-color:rgba(255,255,255,.15);color:rgba(255,255,255,.5);">
      &#9660; BEARISH <span class="sc-cnt" style="background:#ff6b6b;">9</span>
    </button>
    <button class="sc-tab" onclick="filterStrat('nondirectional',this)"
      style="border-color:rgba(255,255,255,.15);color:rgba(255,255,255,.5);">
      &#8596; NON-DIRECTIONAL <span class="sc-cnt" style="background:#6480ff;">20</span>
    </button>
  </div>

  <div class="sc-grid" id="sc-grid">
    {bull_cards}{bear_cards}{nd_cards}
  </div>
</div>

<script>
const OC = {{
  spot:     {spot:.2f},
  atm:      {atm},
  pcr:      {pcr:.3f},
  maxPain:  {max_pain},
  strikes:  {strikes_json},
  lotSize:  75,
}};

const STRIKE_MAP = {{}};
OC.strikes.forEach(s => {{ STRIKE_MAP[s.strike] = s; }});

function normCDF(x) {{
  const a1=0.254829592, a2=-0.284496736, a3=1.421413741,
        a4=-1.453152027, a5=1.061405429, p=0.3275911;
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x);
  const t = 1 / (1 + p * x);
  const y = 1 - (((((a5*t+a4)*t)+a3)*t+a2)*t+a1)*t*Math.exp(-x*x);
  return 0.5 * (1 + sign * y);
}}

function bsDelta(spot, strike, iv, T, isCall) {{
  if (iv <= 0 || T <= 0) return isCall ? 0.5 : -0.5;
  const r = 0.065;
  const d1 = (Math.log(spot/strike) + (r + 0.5*iv*iv)*T) / (iv * Math.sqrt(T));
  if (isCall) return normCDF(d1);
  return normCDF(d1) - 1;
}}

function getATMLTP(type) {{
  const atm = OC.atm;
  const row = STRIKE_MAP[atm] || OC.strikes.reduce((best, s) => {{
    return Math.abs(s.strike - atm) < Math.abs(best.strike - atm) ? s : best;
  }}, OC.strikes[0] || {{strike: atm, ce_ltp: 0, pe_ltp: 0, ce_iv: 15, pe_iv: 15}});
  return type === 'ce' ? row.ce_ltp : row.pe_ltp;
}}

function getOTM(type, offset) {{
  const targetStrike = type === 'ce' ? OC.atm + offset*50 : OC.atm - offset*50;
  const row = STRIKE_MAP[targetStrike] || OC.strikes.reduce((best, s) => {{
    return Math.abs(s.strike - targetStrike) < Math.abs(best.strike - targetStrike) ? s : best;
  }}, OC.strikes[0] || {{strike: OC.atm, ce_ltp: 0, pe_ltp: 0, ce_iv: 15, pe_iv: 15}});
  return {{strike: row.strike || targetStrike, ltp: type === 'ce' ? row.ce_ltp : row.pe_ltp, iv: type === 'ce' ? row.ce_iv : row.pe_iv}};
}}

function getPCRAdjust() {{
  if (OC.pcr > 1.3) return 0.05;
  if (OC.pcr > 1.1) return 0.02;
  if (OC.pcr < 0.7) return -0.05;
  if (OC.pcr < 0.9) return -0.02;
  return 0;
}}

function daysToExpiry() {{ return 5; }}

function calcMetrics(shape) {{
  const spot   = OC.spot;
  const atm    = OC.atm;
  const T      = daysToExpiry() / 365;
  const pcrAdj = getPCRAdjust();
  const lotSz  = OC.lotSize;

  const ce_atm_ltp = getATMLTP('ce');
  const pe_atm_ltp = getATMLTP('pe');
  const ce_otm1    = getOTM('ce', 1);
  const ce_otm2    = getOTM('ce', 2);
  const pe_otm1    = getOTM('pe', 1);
  const pe_otm2    = getOTM('pe', 2);

  const atmIV    = (getOTM('ce',0) && STRIKE_MAP[atm]) ? (STRIKE_MAP[atm].ce_iv||15)/100 : 0.15;
  const otm1IV   = (ce_otm1.iv || 15) / 100;

  let pop=50, mp=0, ml=0, be=[], nc=0, margin=0, pnl=0, rrRatio=0;

  switch(shape) {{
    case 'long_call': {{
      const prem = ce_atm_ltp || 150;
      const delta = bsDelta(spot, atm, atmIV, T, true);
      pop  = Math.round((1 - delta + pcrAdj) * 100);
      mp   = 999999;
      ml   = prem * lotSz;
      be   = [atm + prem];
      nc   = -prem * lotSz;
      margin = prem * lotSz;
      pnl  = Math.max(spot - atm - prem, -prem) * lotSz;
      rrRatio = 0;
      break;
    }}
    case 'long_put': {{
      const prem = pe_atm_ltp || 150;
      const delta = bsDelta(spot, atm, atmIV, T, false);
      pop  = Math.round((Math.abs(delta) + pcrAdj) * 100);
      mp   = 999999;
      ml   = prem * lotSz;
      be   = [atm - prem];
      nc   = -prem * lotSz;
      margin = prem * lotSz;
      pnl  = Math.max(atm - spot - prem, -prem) * lotSz;
      rrRatio = 0;
      break;
    }}
    case 'short_put': {{
      const prem = pe_atm_ltp || 150;
      const delta = bsDelta(spot, atm, atmIV, T, false);
      pop  = Math.round((1 - Math.abs(delta) + pcrAdj) * 100);
      mp   = prem * lotSz;
      ml   = (atm - prem) * lotSz;
      be   = [atm - prem];
      nc   = prem * lotSz;
      margin = atm * lotSz * 0.15;
      pnl  = Math.min(prem, Math.max(spot - atm + prem, -atm + prem)) * lotSz;
      rrRatio = ((atm - prem) / prem).toFixed(2);
      break;
    }}
    case 'short_call': {{
      const prem = ce_atm_ltp || 150;
      const delta = bsDelta(spot, atm, atmIV, T, true);
      pop  = Math.round((1 - delta - pcrAdj) * 100);
      mp   = prem * lotSz;
      ml   = 999999;
      be   = [atm + prem];
      nc   = prem * lotSz;
      margin = atm * lotSz * 0.15;
      pnl  = 0;
      rrRatio = 0;
      break;
    }}
    case 'bull_call_spread': {{
      const buyStrike  = atm;
      const sellStrike = ce_otm1.strike;
      const buyPrem    = ce_atm_ltp || 150;
      const sellPrem   = ce_otm1.ltp || 80;
      const netDebit   = buyPrem - sellPrem;
      const spread     = sellStrike - buyStrike;
      pop  = Math.round((0.45 + pcrAdj) * 100);
      mp   = (spread - netDebit) * lotSz;
      ml   = netDebit * lotSz;
      be   = [buyStrike + netDebit];
      nc   = -netDebit * lotSz;
      margin = netDebit * lotSz;
      pnl  = 0;
      rrRatio = ((spread - netDebit) / netDebit).toFixed(2);
      break;
    }}
    case 'bull_put_spread': {{
      const sellStrike = atm;
      const buyStrike  = pe_otm1.strike;
      const sellPrem   = pe_atm_ltp || 150;
      const buyPrem    = pe_otm1.ltp || 80;
      const netCredit  = sellPrem - buyPrem;
      const spread     = sellStrike - buyStrike;
      pop  = Math.round((0.55 + pcrAdj) * 100);
      mp   = netCredit * lotSz;
      ml   = (spread - netCredit) * lotSz;
      be   = [sellStrike - netCredit];
      nc   = netCredit * lotSz;
      margin = spread * lotSz;
      pnl  = 0;
      rrRatio = (netCredit / (spread - netCredit)).toFixed(2);
      break;
    }}
    case 'bear_call_spread': {{
      const sellStrike = atm;
      const buyStrike  = ce_otm1.strike;
      const sellPrem   = ce_atm_ltp || 150;
      const buyPrem    = ce_otm1.ltp || 80;
      const netCredit  = sellPrem - buyPrem;
      const spread     = buyStrike - sellStrike;
      pop  = Math.round((0.55 - pcrAdj) * 100);
      mp   = netCredit * lotSz;
      ml   = (spread - netCredit) * lotSz;
      be   = [sellStrike + netCredit];
      nc   = netCredit * lotSz;
      margin = spread * lotSz;
      pnl  = 0;
      rrRatio = (netCredit / (spread - netCredit)).toFixed(2);
      break;
    }}
    case 'bear_put_spread': {{
      const buyStrike  = atm;
      const sellStrike = pe_otm1.strike;
      const buyPrem    = pe_atm_ltp || 150;
      const sellPrem   = pe_otm1.ltp || 80;
      const netDebit   = buyPrem - sellPrem;
      const spread     = buyStrike - sellStrike;
      pop  = Math.round((0.45 - pcrAdj) * 100);
      mp   = (spread - netDebit) * lotSz;
      ml   = netDebit * lotSz;
      be   = [buyStrike - netDebit];
      nc   = -netDebit * lotSz;
      margin = netDebit * lotSz;
      pnl  = 0;
      rrRatio = ((spread - netDebit) / netDebit).toFixed(2);
      break;
    }}
    case 'long_straddle': {{
      const ce_prem = ce_atm_ltp || 150;
      const pe_prem = pe_atm_ltp || 150;
      const totalPrem = ce_prem + pe_prem;
      pop  = Math.round((0.35 + Math.abs(pcrAdj)) * 100);
      mp   = 999999;
      ml   = totalPrem * lotSz;
      be   = [atm - totalPrem, atm + totalPrem];
      nc   = -totalPrem * lotSz;
      margin = totalPrem * lotSz;
      pnl  = (Math.abs(spot - atm) - totalPrem) * lotSz;
      rrRatio = 0;
      break;
    }}
    case 'short_straddle': {{
      const ce_prem = ce_atm_ltp || 150;
      const pe_prem = pe_atm_ltp || 150;
      const totalPrem = ce_prem + pe_prem;
      pop  = Math.round((0.65 - Math.abs(pcrAdj)) * 100);
      mp   = totalPrem * lotSz;
      ml   = 999999;
      be   = [atm - totalPrem, atm + totalPrem];
      nc   = totalPrem * lotSz;
      margin = atm * lotSz * 0.25;
      pnl  = (totalPrem - Math.abs(spot - atm)) * lotSz;
      rrRatio = 0;
      break;
    }}
    case 'long_strangle': {{
      const ce_prem = ce_otm1.ltp || 100;
      const pe_prem = pe_otm1.ltp || 100;
      const totalPrem = ce_prem + pe_prem;
      pop  = Math.round((0.30 + Math.abs(pcrAdj)) * 100);
      mp   = 999999;
      ml   = totalPrem * lotSz;
      be   = [pe_otm1.strike - totalPrem, ce_otm1.strike + totalPrem];
      nc   = -totalPrem * lotSz;
      margin = totalPrem * lotSz;
      pnl  = 0;
      rrRatio = 0;
      break;
    }}
    case 'short_strangle': {{
      const ce_prem = ce_otm1.ltp || 100;
      const pe_prem = pe_otm1.ltp || 100;
      const totalPrem = ce_prem + pe_prem;
      pop  = Math.round((0.68 - Math.abs(pcrAdj)) * 100);
      mp   = totalPrem * lotSz;
      ml   = 999999;
      be   = [pe_otm1.strike - totalPrem, ce_otm1.strike + totalPrem];
      nc   = totalPrem * lotSz;
      margin = atm * lotSz * 0.20;
      pnl  = (totalPrem - Math.max(0, spot - ce_otm1.strike) - Math.max(0, pe_otm1.strike - spot)) * lotSz;
      rrRatio = 0;
      break;
    }}
    case 'short_iron_condor': {{
      const sellCE = ce_otm1.ltp || 100;
      const buyCE  = ce_otm2.ltp || 50;
      const sellPE = pe_otm1.ltp || 100;
      const buyPE  = pe_otm2.ltp || 50;
      const netCredit = sellCE - buyCE + sellPE - buyPE;
      const wingWidth = 50;
      pop  = Math.round((0.65 + pcrAdj) * 100);
      mp   = netCredit * lotSz;
      ml   = (wingWidth - netCredit) * lotSz;
      be   = [pe_otm1.strike - netCredit, ce_otm1.strike + netCredit];
      nc   = netCredit * lotSz;
      margin = wingWidth * lotSz * 2;
      pnl  = 0;
      rrRatio = (netCredit / (wingWidth - netCredit)).toFixed(2);
      break;
    }}
    case 'long_iron_condor': {{
      const sellCE = ce_otm1.ltp || 100;
      const buyCE  = ce_otm2.ltp || 50;
      const sellPE = pe_otm1.ltp || 100;
      const buyPE  = pe_otm2.ltp || 50;
      const netDebit = buyCE - sellCE + buyPE - sellPE;
      const wingWidth = 50;
      pop  = Math.round((0.33 - pcrAdj) * 100);
      mp   = (wingWidth - Math.abs(netDebit)) * lotSz;
      ml   = Math.abs(netDebit) * lotSz;
      be   = [pe_otm1.strike - Math.abs(netDebit), ce_otm1.strike + Math.abs(netDebit)];
      nc   = netDebit * lotSz;
      margin = Math.abs(netDebit) * lotSz;
      pnl  = 0;
      rrRatio = ((wingWidth - Math.abs(netDebit)) / Math.abs(netDebit)).toFixed(2);
      break;
    }}
    case 'short_iron_fly': {{
      const ce_prem = ce_atm_ltp || 150;
      const pe_prem = pe_atm_ltp || 150;
      const wing_ce = ce_otm1.ltp || 80;
      const wing_pe = pe_otm1.ltp || 80;
      const netCredit = ce_prem + pe_prem - wing_ce - wing_pe;
      const wingWidth = 50;
      pop  = Math.round((0.60 + pcrAdj) * 100);
      mp   = netCredit * lotSz;
      ml   = (wingWidth - netCredit) * lotSz;
      be   = [atm - netCredit, atm + netCredit];
      nc   = netCredit * lotSz;
      margin = wingWidth * lotSz * 2;
      pnl  = 0;
      rrRatio = (netCredit / (wingWidth - netCredit)).toFixed(2);
      break;
    }}
    case 'long_iron_fly': {{
      const ce_prem = ce_atm_ltp || 150;
      const pe_prem = pe_atm_ltp || 150;
      const wing_ce = ce_otm1.ltp || 80;
      const wing_pe = pe_otm1.ltp || 80;
      const netDebit = wing_ce + wing_pe - ce_prem - pe_prem;
      const wingWidth = 50;
      pop  = Math.round((0.38 - pcrAdj) * 100);
      mp   = (wingWidth - Math.abs(netDebit)) * lotSz;
      ml   = Math.abs(netDebit) * lotSz;
      be   = [atm - Math.abs(netDebit), atm + Math.abs(netDebit)];
      nc   = -Math.abs(netDebit) * lotSz;
      margin = Math.abs(netDebit) * lotSz;
      pnl  = 0;
      rrRatio = ((wingWidth - Math.abs(netDebit)) / Math.abs(netDebit)).toFixed(2);
      break;
    }}
    case 'call_ratio_back': {{
      const sell_prem = ce_atm_ltp || 150;
      const buy_prem  = ce_otm1.ltp || 80;
      const netDebit  = 2 * buy_prem - sell_prem;
      pop  = Math.round((0.40 + pcrAdj) * 100);
      mp   = 999999;
      ml   = netDebit > 0 ? netDebit * lotSz : 0;
      be   = [ce_otm1.strike + buy_prem];
      nc   = -netDebit * lotSz;
      margin = ce_otm1.strike * lotSz * 0.15;
      pnl  = 0;
      rrRatio = 0;
      break;
    }}
    case 'put_ratio_back': {{
      const sell_prem = pe_atm_ltp || 150;
      const buy_prem  = pe_otm1.ltp || 80;
      const netDebit  = 2 * buy_prem - sell_prem;
      pop  = Math.round((0.40 - pcrAdj) * 100);
      mp   = 999999;
      ml   = netDebit > 0 ? netDebit * lotSz : 0;
      be   = [pe_otm1.strike - buy_prem];
      nc   = -netDebit * lotSz;
      margin = pe_otm1.strike * lotSz * 0.15;
      pnl  = 0;
      rrRatio = 0;
      break;
    }}
    case 'long_synthetic': {{
      const ce_prem = ce_atm_ltp || 150;
      const pe_prem = pe_atm_ltp || 150;
      const netDebit = ce_prem - pe_prem;
      pop  = Math.round((0.50 + pcrAdj) * 100);
      mp   = 999999;
      ml   = 999999;
      be   = [atm + netDebit];
      nc   = -Math.abs(netDebit) * lotSz;
      margin = atm * lotSz * 0.30;
      pnl  = (spot - atm - netDebit) * lotSz;
      rrRatio = 0;
      break;
    }}
    case 'short_synthetic': {{
      const ce_prem = ce_atm_ltp || 150;
      const pe_prem = pe_atm_ltp || 150;
      const netCredit = ce_prem - pe_prem;
      pop  = Math.round((0.50 - pcrAdj) * 100);
      mp   = 999999;
      ml   = 999999;
      be   = [atm + netCredit];
      nc   = Math.abs(netCredit) * lotSz;
      margin = atm * lotSz * 0.30;
      pnl  = (atm - spot + netCredit) * lotSz;
      rrRatio = 0;
      break;
    }}
    case 'call_butterfly': case 'bull_butterfly': {{
      const low_prem  = ce_atm_ltp || 150;
      const mid_prem  = ce_otm1.ltp || 80;
      const high_prem = ce_otm2.ltp || 40;
      const netDebit  = low_prem - 2*mid_prem + high_prem;
      const spread    = 50;
      pop  = Math.round((0.55 + pcrAdj) * 100);
      mp   = (spread - netDebit) * lotSz;
      ml   = netDebit * lotSz;
      be   = [atm + netDebit, ce_otm2.strike - netDebit];
      nc   = -netDebit * lotSz;
      margin = netDebit * lotSz;
      pnl  = 0;
      rrRatio = ((spread - netDebit) / netDebit).toFixed(2);
      break;
    }}
    case 'put_butterfly': case 'bear_butterfly': {{
      const high_prem = pe_atm_ltp || 150;
      const mid_prem  = pe_otm1.ltp || 80;
      const low_prem  = pe_otm2.ltp || 40;
      const netDebit  = high_prem - 2*mid_prem + low_prem;
      const spread    = 50;
      pop  = Math.round((0.55 - pcrAdj) * 100);
      mp   = (spread - netDebit) * lotSz;
      ml   = netDebit * lotSz;
      be   = [pe_otm2.strike + netDebit, atm - netDebit];
      nc   = -netDebit * lotSz;
      margin = netDebit * lotSz;
      pnl  = 0;
      rrRatio = ((spread - netDebit) / netDebit).toFixed(2);
      break;
    }}
    case 'jade_lizard': {{
      const pe_prem  = pe_otm1.ltp || 100;
      const ce_sell  = ce_otm1.ltp || 80;
      const ce_buy   = ce_otm2.ltp || 40;
      const netCredit = pe_prem + ce_sell - ce_buy;
      pop  = Math.round((0.60 + pcrAdj) * 100);
      mp   = netCredit * lotSz;
      ml   = (pe_otm1.strike - netCredit) * lotSz;
      be   = [pe_otm1.strike - netCredit];
      nc   = netCredit * lotSz;
      margin = pe_otm1.strike * lotSz * 0.15;
      pnl  = 0;
      rrRatio = (netCredit / (pe_otm1.strike - netCredit)).toFixed(2);
      break;
    }}
    case 'reverse_jade': {{
      const ce_prem  = ce_otm1.ltp || 100;
      const pe_sell  = pe_otm1.ltp || 80;
      const pe_buy   = pe_otm2.ltp || 40;
      const netCredit = ce_prem + pe_sell - pe_buy;
      pop  = Math.round((0.60 - pcrAdj) * 100);
      mp   = netCredit * lotSz;
      ml   = (ce_otm1.strike - netCredit) * lotSz;
      be   = [ce_otm1.strike + netCredit];
      nc   = netCredit * lotSz;
      margin = ce_otm1.strike * lotSz * 0.15;
      pnl  = 0;
      rrRatio = (netCredit / (ce_otm1.strike - netCredit)).toFixed(2);
      break;
    }}
    case 'bull_condor': case 'bear_condor': {{
      const s1 = shape === 'bull_condor' ? ce_atm_ltp : pe_atm_ltp;
      const s2 = shape === 'bull_condor' ? ce_otm1.ltp : pe_otm1.ltp;
      const s3 = shape === 'bull_condor' ? (ce_otm2.ltp * 0.7) : (pe_otm2.ltp * 0.7);
      const s4 = shape === 'bull_condor' ? (ce_otm2.ltp * 0.4) : (pe_otm2.ltp * 0.4);
      const netCredit = (s1 - s2) - (s3 - s4);
      const adj = shape === 'bull_condor' ? pcrAdj : -pcrAdj;
      pop  = Math.round((0.55 + adj) * 100);
      mp   = netCredit * lotSz;
      ml   = (50 - netCredit) * lotSz;
      be   = [atm + netCredit];
      nc   = netCredit * lotSz;
      margin = 100 * lotSz;
      pnl  = 0;
      rrRatio = (netCredit / (50 - netCredit)).toFixed(2);
      break;
    }}
    default: {{
      const prem = ce_atm_ltp || 150;
      pop  = Math.round((0.50 + pcrAdj) * 100);
      mp   = prem * lotSz * 0.5;
      ml   = prem * lotSz * 0.3;
      be   = [atm];
      nc   = -prem * 0.3 * lotSz;
      margin = prem * lotSz;
      pnl  = 0;
      rrRatio = 1.5;
    }}
  }}

  pop = Math.min(95, Math.max(5, pop));

  let strikeStr = '';
  switch(shape) {{
    case 'long_call': case 'short_put': case 'long_put': case 'short_call':
    case 'long_straddle': case 'short_straddle': case 'long_synthetic': case 'short_synthetic':
      strikeStr = '₹' + atm.toLocaleString('en-IN') + ' (ATM)'; break;
    case 'bull_call_spread':
      strikeStr = 'Buy ₹' + atm.toLocaleString('en-IN') + ' · Sell ₹' + ce_otm1.strike.toLocaleString('en-IN'); break;
    case 'bull_put_spread':
      strikeStr = 'Sell ₹' + atm.toLocaleString('en-IN') + ' · Buy ₹' + pe_otm1.strike.toLocaleString('en-IN'); break;
    case 'bear_call_spread':
      strikeStr = 'Sell ₹' + atm.toLocaleString('en-IN') + ' · Buy ₹' + ce_otm1.strike.toLocaleString('en-IN'); break;
    case 'bear_put_spread':
      strikeStr = 'Buy ₹' + atm.toLocaleString('en-IN') + ' · Sell ₹' + pe_otm1.strike.toLocaleString('en-IN'); break;
    case 'long_strangle': case 'short_strangle':
      strikeStr = 'CE ₹' + ce_otm1.strike.toLocaleString('en-IN') + ' · PE ₹' + pe_otm1.strike.toLocaleString('en-IN'); break;
    case 'short_iron_condor': case 'long_iron_condor':
      strikeStr = 'PE ₹' + pe_otm2.strike.toLocaleString('en-IN') + '/' + pe_otm1.strike.toLocaleString('en-IN')
               + ' · CE ₹' + ce_otm1.strike.toLocaleString('en-IN') + '/' + ce_otm2.strike.toLocaleString('en-IN'); break;
    case 'short_iron_fly': case 'long_iron_fly':
      strikeStr = 'Wings ₹' + pe_otm1.strike.toLocaleString('en-IN') + ' / ₹' + ce_otm1.strike.toLocaleString('en-IN')
               + ' · ATM ₹' + atm.toLocaleString('en-IN'); break;
    case 'call_butterfly': case 'bull_butterfly':
      strikeStr = '₹' + atm.toLocaleString('en-IN') + ' / ₹' + ce_otm1.strike.toLocaleString('en-IN')
               + ' / ₹' + ce_otm2.strike.toLocaleString('en-IN'); break;
    case 'put_butterfly': case 'bear_butterfly':
      strikeStr = '₹' + pe_otm2.strike.toLocaleString('en-IN') + ' / ₹' + pe_otm1.strike.toLocaleString('en-IN')
               + ' / ₹' + atm.toLocaleString('en-IN'); break;
    case 'call_ratio_back':
      strikeStr = 'Sell ₹' + atm.toLocaleString('en-IN') + ' · Buy 2x ₹' + ce_otm1.strike.toLocaleString('en-IN'); break;
    case 'put_ratio_back':
      strikeStr = 'Sell ₹' + atm.toLocaleString('en-IN') + ' · Buy 2x ₹' + pe_otm1.strike.toLocaleString('en-IN'); break;
    case 'jade_lizard':
      strikeStr = 'PE ₹' + pe_otm1.strike.toLocaleString('en-IN')
               + ' · CE ₹' + ce_otm1.strike.toLocaleString('en-IN') + '/' + ce_otm2.strike.toLocaleString('en-IN'); break;
    case 'reverse_jade':
      strikeStr = 'CE ₹' + ce_otm1.strike.toLocaleString('en-IN')
               + ' · PE ₹' + pe_otm1.strike.toLocaleString('en-IN') + '/' + pe_otm2.strike.toLocaleString('en-IN'); break;
    default:
      strikeStr = 'ATM ₹' + atm.toLocaleString('en-IN');
  }}

  const beStr = be.map(v => '₹' + Math.round(v).toLocaleString('en-IN')).join(' / ');
  const mpStr = mp === 999999 ? 'Unlimited' : '₹' + Math.round(mp).toLocaleString('en-IN');
  const mlStr = ml === 999999 ? 'Unlimited' : '₹' + Math.round(ml).toLocaleString('en-IN');
  const ncStr = (nc >= 0 ? '+ ' : '- ') + '₹' + Math.abs(Math.round(nc)).toLocaleString('en-IN');
  const marginStr = '₹' + Math.round(margin).toLocaleString('en-IN');
  const pnlStr   = pnl === 0 ? '₹0' : (pnl >= 0 ? '+ ' : '- ') + '₹' + Math.abs(Math.round(pnl)).toLocaleString('en-IN');
  const rrStr    = rrRatio === 0 ? '∞' : ('1:' + Math.abs(rrRatio));
  const mpPct    = mp === 999999 ? '∞' : (ml > 0 ? (mp/ml*100).toFixed(0) + '%' : '—');

  return {{
    pop, mpStr, mlStr, rrStr, beStr, ncStr, marginStr, pnlStr, mpPct, strikeStr,
    mpRaw: mp, mlRaw: ml, ncRaw: Math.round(nc), pnlPositive: pnl >= 0, ncPositive: nc >= 0,
  }};
}}

function renderMetrics(m, cardEl) {{
  const probColor = m.pop >= 60 ? '#00c896' : (m.pop >= 45 ? '#6480ff' : '#ff6b6b');
  const ncColor   = m.ncPositive  ? '#00c896' : '#ff6b6b';
  const pnlColor  = m.pnlPositive ? '#00c896' : '#ff6b6b';
  return `
    <div class="metric-row metric-strike">
      <span class="metric-lbl">Strike Price</span>
      <span class="metric-val" style="color:#ffd166;font-size:11px;text-align:right;max-width:160px;line-height:1.4;">${{m.strikeStr}}</span>
    </div>
    <div class="metric-row">
      <span class="metric-lbl">Prob. of Profit</span>
      <span class="metric-val" style="color:${{probColor}};font-weight:800;font-size:15px;">${{m.pop}}%</span>
    </div>
    <div class="metric-row">
      <span class="metric-lbl">Max. Profit</span>
      <span class="metric-val" style="color:#00c896;">${{m.mpStr}} <small style="opacity:.5;">${{m.mpPct}}</small></span>
    </div>
    <div class="metric-row">
      <span class="metric-lbl">Max. Loss</span>
      <span class="metric-val" style="color:#ff6b6b;">${{m.mlStr}}</span>
    </div>
    <div class="metric-row">
      <span class="metric-lbl">Max RR Ratio</span>
      <span class="metric-val" style="color:#6480ff;">${{m.rrStr}}</span>
    </div>
    <div class="metric-row">
      <span class="metric-lbl">Breakevens</span>
      <span class="metric-val" style="color:#00c8e0;font-size:11px;">${{m.beStr}}</span>
    </div>
    <div class="metric-row">
      <span class="metric-lbl">Total PNL (est.)</span>
      <span class="metric-val" style="color:${{pnlColor}};">${{m.pnlStr}}</span>
    </div>
    <div class="metric-row">
      <span class="metric-lbl">Net Credit / Debit</span>
      <span class="metric-val" style="color:${{ncColor}};">${{m.ncStr}}</span>
    </div>
    <div class="metric-row" style="border-bottom:none;">
      <span class="metric-lbl">Est. Margin/Premium</span>
      <span class="metric-val" style="color:#8aa0ff;">${{m.marginStr}}</span>
    </div>
  `;
}}

function popBadgeStyle(pop) {{
  if (pop >= 65) return 'background:rgba(0,200,150,.2);color:#00c896;border-color:rgba(0,200,150,.4);';
  if (pop >= 50) return 'background:rgba(100,128,255,.2);color:#8aa0ff;border-color:rgba(100,128,255,.4);';
  return 'background:rgba(255,107,107,.2);color:#ff6b6b;border-color:rgba(255,107,107,.4);';
}}

function initAllCards() {{
  document.querySelectorAll('.sc-card').forEach(card => {{
    const shape  = card.dataset.shape;
    const cardId = card.id;
    const badge  = document.getElementById('pop_' + cardId);
    try {{
      const m = calcMetrics(shape);
      card.dataset.pop = m.pop;
      if (badge) {{
        badge.textContent = m.pop + '%';
        badge.setAttribute('style', badge.getAttribute('style') + ';' + popBadgeStyle(m.pop));
      }}
    }} catch(e) {{
      card.dataset.pop = 0;
      if (badge) badge.textContent = '—%';
    }}
  }});
}}

function sortGridByPoP(cat) {{
  const grid  = document.getElementById('sc-grid');
  if (!grid) return;
  const allCards = {{
    bullish:        Array.from(grid.querySelectorAll('.sc-card[data-cat="bullish"]')),
    bearish:        Array.from(grid.querySelectorAll('.sc-card[data-cat="bearish"]')),
    nondirectional: Array.from(grid.querySelectorAll('.sc-card[data-cat="nondirectional"]')),
  }};
  allCards[cat].sort((a, b) => parseInt(b.dataset.pop || 0) - parseInt(a.dataset.pop || 0));
  ['bullish', 'bearish', 'nondirectional'].forEach(c => {{
    allCards[c].forEach(card => grid.appendChild(card));
  }});
}}

window.addEventListener('load', function() {{
  initAllCards();
  sortGridByPoP('bullish');
  sortGridByPoP('bearish');
  sortGridByPoP('nondirectional');
  filterStrat('bullish', document.querySelector('.sc-tab'));
}});
</script>
"""


# =================================================================
#  SECTION 5B -- SCROLLING TICKER BAR (with highlighted labels)
# =================================================================

def rsi_label(rsi):
    if   rsi >= 70: return "Overbought", "#ff6b6b", "bearish"
    elif rsi >= 60: return "Strong",     "#ffd166", "neutral"
    elif rsi >= 40: return "Neutral",    "#6480ff", "neutral"
    elif rsi >= 30: return "Weak",       "#ffd166", "neutral"
    else:           return "Oversold",   "#00c896", "bullish"

def macd_label(macd, signal):
    diff = macd - signal
    if   diff >  0.5: return "Bullish",         "#00c896", "bullish"
    elif diff >  0:   return "Mildly Bullish",   "#4de8b8", "bullish"
    elif diff > -0.5: return "Mildly Bearish",   "#ffd166", "neutral"
    else:             return "Bearish",          "#ff6b6b", "bearish"

def pcr_label(pcr):
    if   pcr > 1.3: return "Very Bullish",  "#00c896", "bullish"
    elif pcr > 1.1: return "Bullish",       "#4de8b8", "bullish"
    elif pcr > 0.9: return "Neutral",       "#6480ff", "neutral"
    elif pcr > 0.7: return "Bearish",       "#ffd166", "neutral"
    else:           return "Very Bearish",  "#ff6b6b", "bearish"


def build_ticker_bar(tech, oc, vix_data):
    """
    Scrolling ticker bar — label names are now highlighted with a
    coloured background pill so they stand out clearly against the
    dark strip. Values and signal badges follow after.
    """
    items = []

    # ── INDIA VIX ────────────────────────────────────────────────
    if vix_data:
        v     = vix_data["value"]
        chg   = vix_data["change"]
        chg_p = vix_data["change_pct"]
        lbl, col, bg, bdr, sig = vix_label(v)
        chg_str  = f"{chg:+.2f} ({chg_p:+.2f}%)"
        chg_col  = "#ff6b6b" if chg > 0 else ("#00c896" if chg < 0 else "#6480ff")
        items.append(
            f'<div class="tk-item">'
            # ← highlighted label name
            f'<span class="tk-name" style="background:rgba(0,200,220,.15);color:#00c8e0;'
            f'border:1px solid rgba(0,200,220,.3);">&#9650;&nbsp;INDIA VIX</span>'
            f'<span class="tk-val" style="color:{col};">{v:.2f}</span>'
            f'<span class="tk-sub" style="color:{chg_col};">{chg_str}</span>'
            f'<span class="tk-badge" style="background:{bg};color:{col};border:1px solid {bdr};">'
            f'{lbl} &middot; {sig}</span>'
            f'</div>'
        )
    else:
        items.append(
            f'<div class="tk-item">'
            f'<span class="tk-name" style="background:rgba(100,128,255,.15);color:#6480ff;'
            f'border:1px solid rgba(100,128,255,.3);">&#9650;&nbsp;INDIA VIX</span>'
            f'<span class="tk-val" style="color:#6480ff;">N/A</span>'
            f'<span class="tk-badge" style="background:rgba(100,128,255,.1);color:#6480ff;'
            f'border:1px solid rgba(100,128,255,.3);">Unavailable</span>'
            f'</div>'
        )

    # ── RSI ──────────────────────────────────────────────────────
    if tech:
        rsi = tech["rsi"]
        lbl, col, cls = rsi_label(rsi)
        bg  = f"rgba({'0,200,150' if cls=='bullish' else ('255,107,107' if cls=='bearish' else '100,128,255')},.1)"
        bdr = f"rgba({'0,200,150' if cls=='bullish' else ('255,107,107' if cls=='bearish' else '100,128,255')},.3)"
        name_bg  = f"rgba({'0,200,150' if cls=='bullish' else ('255,107,107' if cls=='bearish' else '100,128,255')},.18)"
        name_col = col
        name_bdr = f"rgba({'0,200,150' if cls=='bullish' else ('255,107,107' if cls=='bearish' else '100,128,255')},.35)"
        items.append(
            f'<div class="tk-item">'
            f'<span class="tk-name" style="background:{name_bg};color:{name_col};border:1px solid {name_bdr};">'
            f'&#9643;&nbsp;RSI (14)</span>'
            f'<span class="tk-val" style="color:{col};">{rsi:.1f}</span>'
            f'<span class="tk-badge" style="background:{bg};color:{col};border:1px solid {bdr};">{lbl}</span>'
            f'</div>'
        )

    # ── MACD ─────────────────────────────────────────────────────
    if tech:
        macd   = tech["macd"]
        signal = tech["signal_line"]
        lbl, col, cls = macd_label(macd, signal)
        bg  = f"rgba({'0,200,150' if cls=='bullish' else ('255,107,107' if cls=='bearish' else '100,128,255')},.1)"
        bdr = f"rgba({'0,200,150' if cls=='bullish' else ('255,107,107' if cls=='bearish' else '100,128,255')},.3)"
        name_bg  = f"rgba({'0,200,150' if cls=='bullish' else ('255,107,107' if cls=='bearish' else '100,128,255')},.18)"
        name_col = col
        name_bdr = f"rgba({'0,200,150' if cls=='bullish' else ('255,107,107' if cls=='bearish' else '100,128,255')},.35)"
        diff = macd - signal
        items.append(
            f'<div class="tk-item">'
            f'<span class="tk-name" style="background:{name_bg};color:{name_col};border:1px solid {name_bdr};">'
            f'&#9654;&nbsp;MACD</span>'
            f'<span class="tk-val" style="color:{col};">{macd:.2f}</span>'
            f'<span class="tk-sub" style="color:rgba(255,255,255,.4);">Sig:&nbsp;{signal:.2f} &nbsp;Hist:&nbsp;{diff:+.2f}</span>'
            f'<span class="tk-badge" style="background:{bg};color:{col};border:1px solid {bdr};">{lbl}</span>'
            f'</div>'
        )

    # ── PCR ──────────────────────────────────────────────────────
    if oc:
        pcr = oc["pcr_oi"]
        lbl, col, cls = pcr_label(pcr)
        bg  = f"rgba({'0,200,150' if cls=='bullish' else ('255,107,107' if cls=='bearish' else '100,128,255')},.1)"
        bdr = f"rgba({'0,200,150' if cls=='bullish' else ('255,107,107' if cls=='bearish' else '100,128,255')},.3)"
        name_bg  = f"rgba({'0,200,150' if cls=='bullish' else ('255,107,107' if cls=='bearish' else '100,128,255')},.18)"
        name_col = col
        name_bdr = f"rgba({'0,200,150' if cls=='bullish' else ('255,107,107' if cls=='bearish' else '100,128,255')},.35)"
        items.append(
            f'<div class="tk-item">'
            f'<span class="tk-name" style="background:{name_bg};color:{name_col};border:1px solid {name_bdr};">'
            f'&#9670;&nbsp;PCR (OI)</span>'
            f'<span class="tk-val" style="color:{col};">{pcr:.3f}</span>'
            f'<span class="tk-badge" style="background:{bg};color:{col};border:1px solid {bdr};">{lbl}</span>'
            f'</div>'
        )

    # Triplicate so the loop scrolls seamlessly
    track_inner = "".join(items) * 3

    return f'''
<div class="ticker-wrap">
  <div class="ticker-label">LIVE&nbsp;&#9654;</div>
  <div class="ticker-viewport">
    <div class="ticker-track" id="tkTrack">
      {track_inner}
    </div>
  </div>
</div>
'''


# =================================================================
#  SECTION 7 -- CSS
# =================================================================

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=DM+Mono:wght@300;400;500&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#06080f;--surf:#080b14;--card:#0c1020;
  --bdr:rgba(255,255,255,.07);--bdr2:rgba(255,255,255,.12);
  --aurora1:#00c896;--aurora2:#6480ff;--aurora3:#00c8e0;
  --bull:#00c896;--bear:#ff6b6b;--neut:#6480ff;
  --text:rgba(255,255,255,.9);--muted:rgba(255,255,255,.45);--muted2:rgba(255,255,255,.28);
  --fh:'Sora',sans-serif;--fm:'DM Mono',monospace;
}
html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--text);font-family:var(--fh);font-size:13px;line-height:1.6;min-height:100vh}
body::before{
  content:'';position:fixed;inset:0;
  background-image:
    radial-gradient(ellipse at 15% 0%,rgba(0,200,150,.10) 0%,transparent 50%),
    radial-gradient(ellipse at 85% 10%,rgba(100,128,255,.10) 0%,transparent 50%),
    radial-gradient(ellipse at 50% 90%,rgba(0,200,220,.06) 0%,transparent 50%),
    radial-gradient(ellipse at 10% 80%,rgba(0,200,150,.05) 0%,transparent 40%),
    radial-gradient(ellipse at 90% 60%,rgba(100,128,255,.05) 0%,transparent 40%);
  pointer-events:none;z-index:0;
}
.app{position:relative;z-index:1;display:grid;grid-template-rows:auto auto auto 1fr auto;min-height:100vh}
/* header / ticker / hero / main / footer */

/* ── HEADER ── */
header{display:flex;align-items:center;justify-content:space-between;padding:14px 32px;
  background:rgba(6,8,15,.85);backdrop-filter:blur(16px);
  border-bottom:1px solid rgba(255,255,255,.07);position:sticky;top:0;z-index:200;
  box-shadow:0 1px 0 rgba(0,200,150,.1)}
.logo{font-family:var(--fh);font-size:20px;font-weight:700;
  background:linear-gradient(90deg,#00c896,#6480ff);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  filter:drop-shadow(0 0 12px rgba(0,200,150,.3))}
.hdr-meta{display:flex;align-items:center;gap:14px;font-size:11px;color:var(--muted);font-family:var(--fm)}
.live-dot{width:7px;height:7px;border-radius:50%;background:#00c896;box-shadow:0 0 10px #00c896;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.2}}

/* ══════════════════════════════════════════════
   COMPACT HERO  (fixed 92px height single row)
══════════════════════════════════════════════ */
.hero{
  display:flex;align-items:stretch;
  background:linear-gradient(135deg,rgba(0,200,150,.055) 0%,rgba(100,128,255,.055) 100%);
  border-bottom:1px solid rgba(255,255,255,.07);
  overflow:hidden;position:relative;
  height:97px;
}
.hero::before{
  content:'';position:absolute;top:-50px;left:-50px;
  width:200px;height:200px;border-radius:50%;
  background:radial-gradient(circle,rgba(0,200,150,.10),transparent 70%);
  pointer-events:none;
}
.hero::after{
  content:'';position:absolute;bottom:-50px;right:350px;
  width:200px;height:200px;border-radius:50%;
  background:radial-gradient(circle,rgba(255,107,107,.07),transparent 70%);
  pointer-events:none;
}

/* ① Bull gauge col */
.h-bull{
  flex-shrink:0;display:flex;align-items:center;justify-content:center;
  padding:0 11px 0 17px;
}
/* ③ Bear gauge col */
.h-bear{
  flex-shrink:0;display:flex;align-items:center;justify-content:center;
  padding:0 13px 0 9px;
  border-left:1px solid rgba(255,255,255,.05);
}
/* Gauge shared */
.gauge-wrap{position:relative;width:76px;height:76px;}
.gauge-wrap svg{display:block;}
.gauge-inner{
  position:absolute;inset:0;
  display:flex;flex-direction:column;align-items:center;justify-content:center;
}
.g-val{font-family:'DM Mono',monospace;font-size:13px;font-weight:700;line-height:1;}
.g-lbl{font-size:7.5px;letter-spacing:1.5px;text-transform:uppercase;
  color:rgba(255,255,255,.28);margin-top:2px;}

/* ② Middle: signal + bars */
.h-mid{
  flex:1;min-width:0;
  display:flex;flex-direction:column;justify-content:center;
  padding:0 15px 0 13px;
  border-left:1px solid rgba(255,255,255,.05);
}
.h-eyebrow{font-size:8px;font-weight:700;letter-spacing:2px;text-transform:uppercase;
  color:rgba(255,255,255,.22);margin-bottom:2px;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.h-signal{font-size:18px;font-weight:800;letter-spacing:-.3px;line-height:1.1;margin-bottom:2px;}
.h-sub{font-size:9.5px;color:rgba(255,255,255,.32);margin-bottom:0;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
/* pill strength bars — track capped at 35% width */
.h-divider{height:1px;background:rgba(255,255,255,.05);margin:5px 0;}
.pill-bars{display:flex;flex-direction:column;gap:5px;}
.pill-row{display:flex;align-items:center;gap:8px;}
.pill-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0;}
.pill-lbl{font-size:8px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;
  color:rgba(255,255,255,.35);width:96px;flex-shrink:0;}
.pill-track-wrap{flex:1;display:flex;align-items:center;}
.pill-track{width:35%;height:5px;background:rgba(255,255,255,.07);border-radius:3px;overflow:hidden;}
.pill-fill{height:100%;border-radius:3px;}
.pill-num{font-family:'DM Mono',monospace;font-size:10px;font-weight:700;
  margin-left:8px;flex-shrink:0;}

/* ④ Right stat panel */
.h-stats{
  flex-shrink:0;min-width:360px;
  display:flex;flex-direction:column;
  border-left:1px solid rgba(255,255,255,.07);
  background:rgba(255,255,255,.015);
}
.h-stat-row{
  display:flex;align-items:stretch;flex:1;
  border-bottom:1px solid rgba(255,255,255,.05);
}
.h-stat{
  flex:1;display:flex;flex-direction:column;justify-content:center;
  padding:5px 10px;text-align:center;
  border-right:1px solid rgba(255,255,255,.04);
}
.h-stat:last-child{border-right:none;}
.h-stat-lbl{font-size:7.5px;font-weight:700;letter-spacing:1.8px;text-transform:uppercase;
  color:rgba(255,255,255,.22);margin-bottom:3px;white-space:nowrap;}
.h-stat-val{font-family:'DM Mono',monospace;font-size:13px;font-weight:700;
  line-height:1;white-space:nowrap;}

/* Bottom: bias + conf + score + timestamp */
.h-stat-bottom{
  display:flex;align-items:center;justify-content:space-between;
  padding:4px 10px;
}
.h-bias-row{display:flex;align-items:center;gap:6px;}
.h-chip{
  font-size:9px;font-weight:700;letter-spacing:.8px;text-transform:uppercase;
  padding:2px 9px;border-radius:20px;white-space:nowrap;
}
.h-score{font-family:'DM Mono',monospace;font-size:8px;
  color:rgba(255,255,255,.22);letter-spacing:.5px;}
.h-ts{font-family:'DM Mono',monospace;font-size:8px;
  color:rgba(255,255,255,.18);letter-spacing:.5px;white-space:nowrap;}

/* ── LAYOUT ── */
.main{display:grid;grid-template-columns:268px 1fr;min-height:0}

/* ── SIDEBAR ── */
.sidebar{background:rgba(8,11,20,.7);backdrop-filter:blur(12px);
  border-right:1px solid rgba(255,255,255,.06);
  position:sticky;top:57px;height:calc(100vh - 57px);overflow-y:auto}
.sidebar::-webkit-scrollbar{width:3px}
.sidebar::-webkit-scrollbar-thumb{background:rgba(255,255,255,.1);border-radius:2px}
.sb-sec{padding:16px 12px 8px}
.sb-lbl{font-size:9px;font-weight:700;letter-spacing:.15em;text-transform:uppercase;
  color:var(--aurora1);margin-bottom:8px;padding:0 0 0 8px;border-left:2px solid var(--aurora1)}
.sb-btn{display:flex;align-items:center;gap:8px;width:100%;padding:9px 12px;
  border-radius:8px;border:1px solid transparent;cursor:pointer;
  background:transparent;color:var(--muted);font-family:var(--fh);font-size:12px;text-align:left;transition:all .15s}
.sb-btn:hover{background:rgba(0,200,150,.08);color:rgba(255,255,255,.8);border-color:rgba(0,200,150,.2)}
.sb-btn.active{background:rgba(0,200,150,.1);border-color:rgba(0,200,150,.25);color:#00c896;font-weight:600}
.sb-badge{font-size:10px;margin-left:auto;font-weight:700}
.sig-card{margin:12px 10px 8px;padding:18px 14px;
  background:linear-gradient(135deg,rgba(0,200,150,.12),rgba(100,128,255,.12));
  border-radius:14px;border:1px solid rgba(0,200,150,.2);text-align:center;
  box-shadow:0 4px 24px rgba(0,200,150,.1),inset 0 1px 0 rgba(255,255,255,.05)}
.sig-arrow{font-family:var(--fh);font-size:38px;font-weight:700;line-height:1;margin-bottom:4px;
  background:linear-gradient(135deg,#00c896,#6480ff);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.sig-bias{font-family:var(--fh);font-size:18px;font-weight:700;color:rgba(255,255,255,.9)}
.sig-meta{font-size:10px;color:var(--muted);margin-top:4px}

/* ── CONTENT ── */
.content{overflow-y:auto}
.section{padding:26px 28px;border-bottom:1px solid rgba(255,255,255,.05);background:transparent;position:relative}
.section:nth-child(odd){background:rgba(255,255,255,.015)}
.sec-title{font-family:var(--fh);font-size:11px;font-weight:700;letter-spacing:2.5px;
  color:var(--aurora1);text-transform:uppercase;
  display:flex;align-items:center;gap:10px;flex-wrap:wrap;
  margin-bottom:20px;padding-bottom:12px;border-bottom:1px solid rgba(0,200,150,.15)}
.sec-sub{font-size:11px;color:var(--muted2);font-weight:400;letter-spacing:.5px;text-transform:none;margin-left:auto}

/* ── OI TICKER TABLE ── */
.oi-ticker-table{border:1px solid rgba(255,255,255,.07);border-radius:14px;overflow:hidden}
.oi-ticker-hdr{display:grid;grid-template-columns:130px repeat(5,1fr);padding:9px 18px;align-items:center;gap:6px}
.oi-ticker-hdr-label{font-size:9px;font-weight:700;letter-spacing:2px;text-transform:uppercase}
.oi-ticker-hdr-cell{font-size:9px;letter-spacing:1.5px;text-transform:uppercase;color:rgba(255,255,255,.35);text-align:center}
.oi-ticker-row{display:grid;grid-template-columns:130px repeat(5,1fr);padding:15px 18px;
  border-top:1px solid rgba(255,255,255,.04);align-items:center;gap:6px;transition:background .15s}
.oi-ticker-row:hover{background:rgba(255,255,255,.03)}
.oi-ticker-metric{font-size:10px;font-weight:600;letter-spacing:1px;text-transform:uppercase;color:rgba(255,255,255,.35)}
.oi-ticker-cell{text-align:center}

/* ── KEY LEVELS ── */
.kl-zone-labels{display:flex;justify-content:space-between;margin-bottom:6px;font-size:11px;font-weight:700}
.kl-node{position:absolute;text-align:center}
.kl-lbl{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.5px;line-height:1.3;white-space:nowrap}
.kl-val{font-size:12px;font-weight:700;color:rgba(255,255,255,.7);white-space:nowrap;margin-top:2px}
.kl-dot{width:11px;height:11px;border-radius:50%;border:2px solid var(--bg)}
.kl-gradient-bar{position:relative;height:6px;border-radius:3px;
  background:linear-gradient(90deg,#00a07a 0%,#00c896 25%,#6480ff 55%,#ff6b6b 80%,#cc4040 100%);
  box-shadow:0 0 12px rgba(0,200,150,.2)}
.kl-price-tick{position:absolute;top:50%;transform:translate(-50%,-50%);
  width:3px;height:18px;background:#fff;border-radius:2px;
  box-shadow:0 0 12px rgba(255,255,255,.6);z-index:10}
.kl-dist-row{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:4px}
.kl-dist-box{background:rgba(255,255,255,.03);border:1px solid;border-radius:10px;
  padding:10px 14px;display:flex;justify-content:space-between;align-items:center}

/* ── STRIKES TABLE ── */
.strikes-head{font-weight:700;margin-bottom:10px;font-size:13px}
.strikes-wrap{display:grid;grid-template-columns:1fr 1fr;gap:20px}
.s-table{width:100%;border-collapse:collapse;border-radius:10px;overflow:hidden}
.s-table th{background:linear-gradient(90deg,rgba(0,200,150,.15),rgba(100,128,255,.15));
  color:rgba(255,255,255,.7);padding:10px 12px;font-size:11px;font-weight:600;text-align:left;
  letter-spacing:.5px;border-bottom:1px solid rgba(255,255,255,.08)}
.s-table td{padding:10px 12px;border-bottom:1px solid rgba(255,255,255,.05);
  font-size:12px;color:rgba(255,255,255,.8);background:rgba(255,255,255,.02)}
.s-table tr:last-child td{border-bottom:none}
.s-table tr:hover td{background:rgba(0,200,150,.05)}

/* ── STRATEGY CARDS ── */
.sc-tabs{display:flex;gap:10px;margin-bottom:20px;flex-wrap:wrap}
.sc-tab{padding:8px 20px;border-radius:24px;border:1px solid;cursor:pointer;
  font-family:var(--fh);font-size:12px;font-weight:600;transition:all .2s;
  display:flex;align-items:center;gap:8px;background:transparent}
.sc-tab:hover{opacity:.85}
.sc-cnt{font-size:10px;padding:1px 7px;border-radius:10px;color:#fff;font-weight:700}
.sc-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px}
.sc-card{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08);
  border-radius:14px;overflow:hidden;cursor:pointer;
  transition:all .2s;display:flex;flex-direction:column;position:relative;}
.sc-card:hover{border-color:rgba(0,200,150,.3);transform:translateY(-3px);
  box-shadow:0 8px 28px rgba(0,200,150,.1)}
.sc-card.hidden{display:none}
.sc-card.expanded .sc-detail{display:block}
.sc-card.expanded{border-color:rgba(0,200,150,.35);
  box-shadow:0 0 0 1px rgba(0,200,150,.2),0 12px 32px rgba(0,200,150,.12)}
.sc-pop-badge{
  position:absolute;top:8px;right:8px;
  font-family:'DM Mono',monospace;font-size:10px;font-weight:700;
  padding:3px 8px;border-radius:20px;border:1px solid rgba(255,255,255,.15);
  background:rgba(255,255,255,.08);color:rgba(255,255,255,.5);
  z-index:5;letter-spacing:.5px;transition:all .3s;
  min-width:38px;text-align:center;
}
.sc-svg{display:flex;align-items:center;justify-content:center;
  padding:14px 0 6px;background:rgba(255,255,255,.02)}
.sc-body{padding:10px 12px 12px}
.sc-name{font-family:var(--fh);font-size:12px;font-weight:700;
  color:rgba(255,255,255,.9);margin-bottom:4px;line-height:1.3;padding-right:48px}
.sc-legs{font-family:var(--fm);font-size:9px;color:rgba(0,200,220,.7);
  margin-bottom:8px;letter-spacing:.3px;line-height:1.4}
.sc-tags{display:flex;flex-direction:column;gap:4px}
.sc-tag{font-size:9px;padding:2px 8px;border-radius:6px;
  border:1px solid;background:rgba(0,0,0,.2);display:inline-block;width:fit-content}
.sc-detail{display:none;border-top:1px solid rgba(255,255,255,.06);
  background:rgba(0,200,150,.03)}
.sc-desc{font-size:11px;color:rgba(255,255,255,.5);line-height:1.7;
  padding:12px 12px 8px;border-bottom:1px solid rgba(255,255,255,.05);}
.sc-metrics-live{padding:0}
.sc-loading{padding:14px 12px;font-size:11px;color:rgba(255,255,255,.3);text-align:center;font-family:'DM Mono',monospace}
.metric-row{
  display:flex;justify-content:space-between;align-items:center;
  padding:8px 12px;border-bottom:1px solid rgba(255,255,255,.04);
  transition:background .15s;
}
.metric-row:hover{background:rgba(255,255,255,.03)}
.metric-strike{background:rgba(255,209,102,.04);border-bottom:1px solid rgba(255,209,102,.12) !important;}
.metric-lbl{font-size:10px;color:rgba(255,255,255,.35);letter-spacing:.5px;
  text-transform:uppercase;font-family:'DM Mono',monospace;}
.metric-val{font-family:'DM Mono',monospace;font-size:12px;font-weight:600;text-align:right;}

/* ══════════════════════════════════════════════
   SCROLLING TICKER BAR  — highlighted names
══════════════════════════════════════════════ */
.ticker-wrap{
  display:flex;align-items:center;
  background:rgba(4,6,12,.97);
  border-bottom:1px solid rgba(255,255,255,.07);
  height:46px;overflow:hidden;position:relative;z-index:190;
  box-shadow:0 2px 20px rgba(0,0,0,.5);
}
.ticker-label{
  flex-shrink:0;padding:0 16px;
  font-family:var(--fm);font-size:9px;font-weight:700;
  letter-spacing:3px;color:#00c896;text-transform:uppercase;
  border-right:1px solid rgba(0,200,150,.2);height:100%;
  display:flex;align-items:center;
  background:rgba(0,200,150,.07);white-space:nowrap;
}
.ticker-viewport{flex:1;overflow:hidden;height:100%}
.ticker-track{
  display:flex;align-items:center;gap:0;
  height:100%;white-space:nowrap;
  animation:ticker-scroll 38s linear infinite;
  will-change:transform;
}
.ticker-track:hover{animation-play-state:paused}
@keyframes ticker-scroll{
  0%   {transform:translateX(0)}
  100% {transform:translateX(-33.333%)}
}
.tk-item{
  display:inline-flex;align-items:center;gap:10px;
  padding:0 20px;height:100%;
  border-right:1px solid rgba(255,255,255,.04);
  flex-shrink:0;
}
/* ── Highlighted label name pill ── */
.tk-name{
  font-family:var(--fm);font-size:10px;font-weight:700;
  letter-spacing:1.5px;text-transform:uppercase;
  padding:3px 10px;border-radius:6px;
  white-space:nowrap;flex-shrink:0;
  /* default — overridden inline for each item */
  background:rgba(255,255,255,.08);
  color:rgba(255,255,255,.5);
  border:1px solid rgba(255,255,255,.1);
}
.tk-val{
  font-family:var(--fm);font-size:18px;font-weight:700;
  line-height:1;white-space:nowrap;
}
.tk-sub{
  font-family:var(--fm);font-size:10px;
  color:rgba(255,255,255,.35);white-space:nowrap;
}
.tk-badge{
  font-family:var(--fh);font-size:10px;font-weight:700;
  padding:3px 10px;border-radius:20px;
  white-space:nowrap;letter-spacing:.3px;
}

/* ── FOOTER ── */
footer{padding:16px 32px;border-top:1px solid rgba(255,255,255,.06);
  background:rgba(6,8,15,.9);backdrop-filter:blur(12px);
  display:flex;justify-content:space-between;
  font-size:11px;color:var(--muted2);font-family:var(--fm)}

/* ── RESPONSIVE ── */
@media(max-width:1200px){
  .h-stats{min-width:280px;}
}
@media(max-width:1024px){
  .main{grid-template-columns:1fr}
  .sidebar{position:static;height:auto;border-right:none;border-bottom:1px solid rgba(255,255,255,.06)}
  .hero{height:auto;flex-wrap:wrap;}
  .h-stats{min-width:100%;border-left:none;border-top:1px solid rgba(255,255,255,.07);}
  .h-stat-row{flex-wrap:wrap;}
  .h-stat{min-width:33%;}
  .oi-ticker-hdr,.oi-ticker-row{grid-template-columns:100px repeat(3,1fr)}
  .oi-ticker-hdr-cell:nth-child(n+5),.oi-ticker-cell:nth-child(n+5){display:none}
  .strikes-wrap{grid-template-columns:1fr}
  .sc-grid{grid-template-columns:repeat(auto-fill,minmax(160px,1fr))}
}
@media(max-width:640px){
  header{padding:12px 16px}
  .hero{height:auto;}
  .h-bull,.h-bear{padding:8px 12px;}
  .gauge-wrap{width:60px;height:60px;}
  .h-signal{font-size:14px;}
  .h-stats{min-width:100%;}
  .h-stat{min-width:50%;}
  .section{padding:18px 16px}
  .oi-ticker-hdr,.oi-ticker-row{grid-template-columns:90px repeat(2,1fr)}
  .oi-ticker-hdr-cell:nth-child(n+4),.oi-ticker-cell:nth-child(n+4){display:none}
  .kl-dist-row{grid-template-columns:1fr}
  footer{flex-direction:column;gap:6px}
  .sc-grid{grid-template-columns:repeat(auto-fill,minmax(150px,1fr))}
}
"""

# =================================================================
#  SECTION 8 -- MASTER HTML ASSEMBLER
# =================================================================

def generate_html(tech, oc, md, ts, vix_data=None):
    cp     = tech["price"]    if tech else 0
    expiry = oc["expiry"]     if oc   else "N/A"
    atm    = oc["atm_strike"] if oc   else (round(cp / 50) * 50 if cp else 0)
    pcr    = oc["pcr_oi"]     if oc   else 0
    mp     = oc["max_pain"]   if oc   else 0

    bias   = md["bias"]
    conf   = md["confidence"]
    bull   = md["bull"]
    bear   = md["bear"]
    diff   = md["diff"]
    bias_cls = md.get("bias_cls", "neutral")

    b_arrow = "&#9650;" if bias == "BULLISH" else ("&#9660;" if bias == "BEARISH" else "&#8596;")

    oi_html      = build_oi_html(oc)                          if oc   else ""
    kl_html      = build_key_levels_html(tech, oc)            if tech else ""
    strat_html   = build_strategies_html(oc)
    strikes_html = build_strikes_html(oc)
    ticker_html  = build_ticker_bar(tech, oc, vix_data)
    # NEW — dual gauge hero replaces old .hero block
    gauge_html   = build_dual_gauge_hero(oc, tech, md, ts)

    sig_card = (
        f"<div class=\"sb-sec\"><div class=\"sb-lbl\">TODAY'S SIGNAL</div>"
        f"<div class=\"sig-card\">"
        f"<div class=\"sig-arrow\">{b_arrow}</div>"
        f"<div class=\"sig-bias\">{bias}</div>"
        f"<div class=\"sig-meta\">{conf} CONFIDENCE</div>"
        f"<div class=\"sig-meta\" style=\"margin-top:4px;\">Bull {bull} pts &nbsp;&middot;&nbsp; Bear {bear} pts</div>"
        f"</div></div>"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Nifty 50 Options Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet">
<style>{CSS}</style>
</head>
<body>
<div class="app">

<header>
  <div class="logo">NIFTYCRAFT</div>
  <div class="hdr-meta">
    <div class="live-dot"></div>
    <span>NSE Options Dashboard</span>
    <span style="color:rgba(255,255,255,.15);">|</span>
    <span>{ts}</span>
  </div>
</header>

{ticker_html}

{gauge_html}

<div class="main">
  <aside class="sidebar">
    {sig_card}
    <div class="sb-sec">
      <div class="sb-lbl">LIVE ANALYSIS</div>
      <button class="sb-btn active" onclick="go('oi',this)">OI Dashboard</button>
      <button class="sb-btn"       onclick="go('kl',this)">Key Levels</button>
    </div>
    <div class="sb-sec">
      <div class="sb-lbl">STRATEGIES</div>
      <button class="sb-btn" onclick="go('strat',this);filterStrat('bullish',null)">&#9650; Bullish <span class="sb-badge" style="color:var(--bull);">9</span></button>
      <button class="sb-btn" onclick="go('strat',this);filterStrat('bearish',null)">&#9660; Bearish <span class="sb-badge" style="color:var(--bear);">9</span></button>
      <button class="sb-btn" onclick="go('strat',this);filterStrat('nondirectional',null)">&#8596; Non-Directional <span class="sb-badge" style="color:var(--neut);">20</span></button>
    </div>
    <div class="sb-sec">
      <div class="sb-lbl">OPTION CHAIN</div>
      <button class="sb-btn" onclick="go('strikes',this)">Top 5 Strikes</button>
    </div>
  </aside>

  <main class="content">
    <div id="oi">{oi_html}</div>
    <div id="kl">{kl_html}</div>
    {strat_html}
    <div id="strikes">{strikes_html}</div>
    <div class="section">
      <div style="background:rgba(100,128,255,.06);border:1px solid rgba(100,128,255,.18);
                  border-left:3px solid #6480ff;border-radius:12px;padding:16px 18px;
                  font-size:13px;color:rgba(255,255,255,.5);line-height:1.8;">
        <strong style="color:rgba(255,255,255,.7);">DISCLAIMER</strong><br>
        This dashboard is for EDUCATIONAL purposes only &mdash; NOT financial advice.<br>
        Always use stop losses. Consult a SEBI-registered investment advisor before trading.
      </div>
    </div>
  </main>
</div>

<footer>
  <span>NiftyCraft &middot; NSE Options Dashboard</span>
  <span>For Educational Purposes Only &middot; &copy; 2025</span>
</footer>
</div>

<script>
function go(id, btn) {{
  const el = document.getElementById(id);
  if (el) el.scrollIntoView({{ behavior:"smooth", block:"start" }});
  if (btn) {{
    document.querySelectorAll(".sb-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
  }}
}}

function filterStrat(cat, btn) {{
  document.querySelectorAll(".sc-card").forEach(c => {{
    c.classList.toggle("hidden", c.dataset.cat !== cat);
  }});
  const colors = {{ bullish:"#00c896", bearish:"#ff6b6b", nondirectional:"#6480ff" }};
  const col = colors[cat] || "#00c896";
  document.querySelectorAll(".sc-tab").forEach(t => {{
    t.style.borderColor = "rgba(255,255,255,.15)";
    t.style.color       = "rgba(255,255,255,.5)";
    t.style.background  = "transparent";
  }});
  if (btn) {{
    btn.style.borderColor = col;
    btn.style.color       = col;
    btn.style.background  = col + "20";
  }} else {{
    document.querySelectorAll(".sc-tab").forEach(t => {{
      if ((cat === "bullish" && t.textContent.includes("BULLISH")) ||
          (cat === "bearish" && t.textContent.includes("BEARISH")) ||
          (cat === "nondirectional" && t.textContent.includes("NON"))) {{
        t.style.borderColor = col;
        t.style.color       = col;
        t.style.background  = col + "20";
      }}
    }});
  }}
}}

document.addEventListener("click", function(e) {{
  const card = e.target.closest(".sc-card");
  if (card) {{
    const wasExpanded = card.classList.contains("expanded");
    document.querySelectorAll(".sc-card.expanded").forEach(c => c.classList.remove("expanded"));
    if (!wasExpanded) {{
      card.classList.add("expanded");
      const metricsEl = card.querySelector('.sc-metrics-live');
      if (metricsEl && metricsEl.querySelector('.sc-loading')) {{
        const shape = card.dataset.shape;
        try {{
          const m = calcMetrics(shape);
          metricsEl.innerHTML = renderMetrics(m, card);
        }} catch(err) {{
          metricsEl.innerHTML = '<div class="sc-loading">Could not calculate metrics</div>';
        }}
      }}
    }}
  }}
}});
</script>
</body>
</html>"""


# =================================================================
#  SECTION 9 -- MAIN RUNNER
# =================================================================

def main():
    ist_tz = pytz.timezone("Asia/Kolkata")
    ts     = datetime.now(ist_tz).strftime("%d-%b-%Y %H:%M IST")
    print("=" * 65)
    print("  NIFTY 50 OPTIONS DASHBOARD — Aurora Theme v7 (Gauge Hero)")
    print(f"  {ts}")
    print("=" * 65)

    print("\n[1/4] Fetching NSE Option Chain (Spot +-500 pts)...")
    nse_fetcher = NSEOptionChain()
    oc_raw, nse_session, nse_headers = nse_fetcher.fetch()
    oc_analysis = analyze_option_chain(oc_raw) if oc_raw else None
    if oc_analysis:
        print(f"  OK  Spot={oc_analysis['underlying']:.2f}  ATM={oc_analysis['atm_strike']}  PCR={oc_analysis['pcr_oi']:.3f}")
    else:
        print("  WARNING  Option chain unavailable — technical-only mode")

    print("\n[2/4] Fetching India VIX...")
    vix_data = fetch_india_vix(nse_session, nse_headers)
    if vix_data:
        lbl, col, bg, bdr, sig = vix_label(vix_data['value'])
        print(f"  OK  India VIX={vix_data['value']}  Change={vix_data['change']:+.2f}  Status: {lbl}")
    else:
        print("  WARNING  India VIX unavailable — dashboard will show N/A")

    print("\n[3/4] Fetching Technical Indicators...")
    tech = get_technical_data()

    print("\n[4/4] Scoring Market Direction...")
    md = compute_market_direction(tech, oc_analysis)
    print(f"  OK  {md['bias']} ({md['confidence']} confidence)  Bull={md['bull']} Bear={md['bear']}")

    print("\nGenerating HTML dashboard...")
    html = generate_html(tech, oc_analysis, md, ts, vix_data=vix_data)

    os.makedirs("docs", exist_ok=True)
    out = os.path.join("docs", "index.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Saved: {out}  ({len(html)/1024:.1f} KB)")

    meta = {
        "timestamp":  ts,
        "bias":       md["bias"],
        "confidence": md["confidence"],
        "bull":       md["bull"],
        "bear":       md["bear"],
        "diff":       md["diff"],
        "price":      round(tech["price"], 2) if tech else None,
        "expiry":     oc_analysis["expiry"]   if oc_analysis else None,
        "pcr":        oc_analysis["pcr_oi"]   if oc_analysis else None,
        "oi_dir":     oc_analysis["oi_dir"]   if oc_analysis else None,
        "india_vix":  vix_data["value"]       if vix_data    else None,
    }
    with open(os.path.join("docs", "latest.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("  Saved: docs/latest.json")

    print("\n" + "=" * 65)
    print(f"  DONE  |  Bias: {md['bias']}  |  Confidence: {md['confidence']}")
    print("  Push to GitHub to deploy to GitHub Pages")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
