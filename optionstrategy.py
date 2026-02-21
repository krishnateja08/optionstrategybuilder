#!/usr/bin/env python3
"""
Nifty 50 Options Strategy Dashboard — GitHub Pages Generator
Fetches live data from NSE -> runs analysis -> writes docs/index.html
Triggered by GitHub Actions on every push / scheduled run.

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
                rows = []
                for item in data:
                    strike = item.get("strikePrice")
                    ce     = item.get("CE", {})
                    pe     = item.get("PE", {})
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
                    })
                df_full    = pd.DataFrame(rows).sort_values("Strike").reset_index(drop=True)
                underlying = json_data.get("records", {}).get("underlyingValue", 0)
                atm_strike = round(underlying / 50) * 50
                all_strikes = sorted(df_full["Strike"].unique())
                if atm_strike in all_strikes:
                    atm_idx = all_strikes.index(atm_strike)
                else:
                    atm_idx    = min(range(len(all_strikes)), key=lambda i: abs(all_strikes[i] - underlying))
                    atm_strike = all_strikes[atm_idx]
                lo = max(0, atm_idx - 10)
                hi = min(len(all_strikes) - 1, atm_idx + 10)
                df = df_full[df_full["Strike"].isin(all_strikes[lo:hi + 1])].reset_index(drop=True)
                print(f"    OK {len(df_full)} strikes -> ATM+-10 -> {len(df)} rows")
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
        return result


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
            oi_dir, oi_sig, oi_icon, oi_cls = "Bullish",           "Put Build-up Dominant",      "GREEN",  "bullish"
        elif ce_chg > pe_chg * 1.5:
            oi_dir, oi_sig, oi_icon, oi_cls = "Bearish",           "Call Build-up Dominant",     "RED",    "bearish"
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
                print(f"  1H Levels: S2={s2} S1={s1} CMP={cp:.0f} R1={r1} R2={r2}")
        except Exception as e:
            print(f"  WARNING 1H data: {e}")

        recent_d   = df.tail(60)
        resistance = r1 if r1 else recent_d["High"].quantile(0.90)
        support    = s1 if s1 else recent_d["Low"].quantile(0.10)
        strong_res = r2 if r2 else resistance + 100
        strong_sup = s2 if s2 else support - 100

        rsi_val  = latest['RSI']
        macd_val = latest['MACD']
        print(f"  Technical OK | CMP={cp:.2f} RSI={rsi_val:.1f} MACD={macd_val:.2f}")
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
#  SECTION 5 -- PAYOFF SVG GENERATOR
# =================================================================

def get_payoff_svg(payoff_type, width=80, height=50):
    w, h = width, height
    mid_x, mid_y = w / 2, h / 2
    G, R = "#0fa86b", "#c0392b"
    BG_G, BG_R = "rgba(15,168,107,0.15)", "rgba(192,57,43,0.15)"
    AXIS = f'<line x1="0" y1="{mid_y}" x2="{w}" y2="{mid_y}" stroke="rgba(100,120,140,0.35)" stroke-width="1"/>'

    diagrams = {
        # BULLISH
        "bullish_ramp":    lambda: f'<polygon points="0,{h*.75} {mid_x},{h*.75} {w},{h*.05}" fill="{BG_G}"/><polyline points="0,{h*.75} {mid_x},{h*.75} {w},{h*.05}" fill="none" stroke="{G}" stroke-width="2"/>',
        "short_put":       lambda: f'<polygon points="0,{h*.8} {mid_x*.6},{h*.75} {w},{h*.25}" fill="{BG_G}"/><polyline points="0,{h*.8} {mid_x*.6},{h*.75} {w},{h*.25}" fill="none" stroke="{G}" stroke-width="2"/>',
        "bull_spread":     lambda: f'<polygon points="0,{h*.75} {mid_x*.8},{h*.75} {mid_x*1.2},{h*.25} {w},{h*.25}" fill="{BG_G}"/><polyline points="0,{h*.75} {mid_x*.8},{h*.75} {mid_x*1.2},{h*.25} {w},{h*.25}" fill="none" stroke="{G}" stroke-width="2"/>',
        "bull_put_spread":  lambda: f'<polygon points="0,{h*.75} {mid_x*.8},{h*.75} {mid_x*1.2},{h*.25} {w},{h*.25}" fill="{BG_G}"/><polyline points="0,{h*.75} {mid_x*.8},{h*.75} {mid_x*1.2},{h*.25} {w},{h*.25}" fill="none" stroke="{G}" stroke-width="2"/>',
        "call_ratio_back": lambda: f'<polygon points="0,{h*.5} {mid_x*.7},{h*.65} {mid_x},{h*.65} {w},{h*.05}" fill="{BG_G}"/><polyline points="0,{h*.5} {mid_x*.7},{h*.65} {mid_x},{h*.65} {w},{h*.05}" fill="none" stroke="{G}" stroke-width="2"/>',
        "long_synthetic":  lambda: f'<polygon points="0,{h*.9} {mid_x},{h*.5} {w},{h*.1}" fill="{BG_G}" opacity="0.5"/><polyline points="0,{h*.9} {mid_x},{h*.5} {w},{h*.1}" fill="none" stroke="{G}" stroke-width="2"/>',
        "range_forward":   lambda: f'<polygon points="{mid_x*.5},{h*.5} {mid_x*1.5},{h*.5} {w},{h*.15}" fill="{BG_G}"/><polyline points="0,{h*.5} {mid_x*.5},{h*.5} {mid_x*1.5},{h*.5} {w},{h*.15}" fill="none" stroke="{G}" stroke-width="2"/>',
        "bull_butterfly":  lambda: f'<polygon points="{mid_x*.5},{h*.75} {mid_x},{h*.2} {mid_x*1.5},{h*.75}" fill="{BG_G}"/><polyline points="0,{h*.8} {mid_x*.5},{h*.75} {mid_x},{h*.2} {mid_x*1.5},{h*.75} {w},{h*.8}" fill="none" stroke="{G}" stroke-width="2"/>',
        "bull_condor":     lambda: f'<polygon points="{mid_x*.3},{h*.75} {mid_x*.7},{h*.25} {mid_x*1.3},{h*.25} {mid_x*1.7},{h*.75}" fill="{BG_G}"/><polyline points="0,{h*.8} {mid_x*.3},{h*.75} {mid_x*.7},{h*.25} {mid_x*1.3},{h*.25} {mid_x*1.7},{h*.75} {w},{h*.8}" fill="none" stroke="{G}" stroke-width="2"/>',
        # BEARISH
        "bearish_ramp":    lambda: f'<polygon points="0,{h*.05} {mid_x},{h*.75} {w},{h*.75}" fill="{BG_R}"/><polyline points="0,{h*.05} {mid_x},{h*.75} {w},{h*.75}" fill="none" stroke="{R}" stroke-width="2"/>',
        "short_call":      lambda: f'<polygon points="0,{h*.25} {mid_x*1.4},{h*.25} {w},{h*.85}" fill="{BG_G}" opacity="0.5"/><polyline points="0,{h*.25} {mid_x*1.4},{h*.25} {w},{h*.85}" fill="none" stroke="{R}" stroke-width="2"/>',
        "bear_put_spread":  lambda: f'<polygon points="0,{h*.25} {mid_x*.8},{h*.25} {mid_x*1.2},{h*.75} {w},{h*.75}" fill="{BG_R}"/><polyline points="0,{h*.25} {mid_x*.8},{h*.25} {mid_x*1.2},{h*.75} {w},{h*.75}" fill="none" stroke="{R}" stroke-width="2"/>',
        "bear_call_spread": lambda: f'<polygon points="0,{h*.25} {mid_x*.8},{h*.25} {mid_x*1.2},{h*.75} {w},{h*.75}" fill="{BG_R}"/><polyline points="0,{h*.25} {mid_x*.8},{h*.25} {mid_x*1.2},{h*.75} {w},{h*.75}" fill="none" stroke="{R}" stroke-width="2"/>',
        "put_ratio_back":  lambda: f'<polygon points="0,{h*.05} {mid_x},{h*.65} {mid_x*1.3},{h*.65} {w},{h*.5}" fill="{BG_R}"/><polyline points="0,{h*.05} {mid_x},{h*.65} {mid_x*1.3},{h*.65} {w},{h*.5}" fill="none" stroke="{R}" stroke-width="2"/>',
        "short_synthetic": lambda: f'<polygon points="0,{h*.1} {mid_x},{h*.5} {w},{h*.9}" fill="{BG_R}" opacity="0.5"/><polyline points="0,{h*.1} {mid_x},{h*.5} {w},{h*.9}" fill="none" stroke="{R}" stroke-width="2"/>',
        "bear_butterfly":  lambda: f'<polygon points="{mid_x*.5},{h*.25} {mid_x},{h*.8} {mid_x*1.5},{h*.25}" fill="{BG_R}"/><polyline points="0,{h*.2} {mid_x*.5},{h*.25} {mid_x},{h*.8} {mid_x*1.5},{h*.25} {w},{h*.2}" fill="none" stroke="{R}" stroke-width="2"/>',
        "bear_condor":     lambda: f'<polygon points="{mid_x*.3},{h*.25} {mid_x*.7},{h*.75} {mid_x*1.3},{h*.75} {mid_x*1.7},{h*.25}" fill="{BG_R}"/><polyline points="0,{h*.2} {mid_x*.3},{h*.25} {mid_x*.7},{h*.75} {mid_x*1.3},{h*.75} {mid_x*1.7},{h*.25} {w},{h*.2}" fill="none" stroke="{R}" stroke-width="2"/>',
        # NON-DIRECTIONAL
        "long_straddle":      lambda: f'<polygon points="0,{h*.1} {mid_x},{h*.85} {w},{h*.1}" fill="{BG_G}"/><polyline points="0,{h*.1} {mid_x},{h*.85} {w},{h*.1}" fill="none" stroke="{G}" stroke-width="2"/>',
        "short_straddle":     lambda: f'<polygon points="0,{h*.9} {mid_x},{h*.15} {w},{h*.9}" fill="{BG_R}"/><polyline points="0,{h*.9} {mid_x},{h*.15} {w},{h*.9}" fill="none" stroke="{R}" stroke-width="2"/>',
        "long_strangle":      lambda: f'<polygon points="0,{h*.15} {mid_x*.6},{h*.8} {mid_x*1.4},{h*.8} {w},{h*.15}" fill="{BG_G}"/><polyline points="0,{h*.15} {mid_x*.6},{h*.8} {mid_x*1.4},{h*.8} {w},{h*.15}" fill="none" stroke="{G}" stroke-width="2"/>',
        "short_strangle":     lambda: f'<polygon points="0,{h*.85} {mid_x*.6},{h*.2} {mid_x*1.4},{h*.2} {w},{h*.85}" fill="{BG_R}"/><polyline points="0,{h*.85} {mid_x*.6},{h*.2} {mid_x*1.4},{h*.2} {w},{h*.85}" fill="none" stroke="{R}" stroke-width="2"/>',
        "jade_lizard":        lambda: f'<polygon points="0,{h*.75} {mid_x*.5},{h*.3} {mid_x*1.4},{h*.3} {w},{h*.65}" fill="{BG_G}"/><polyline points="0,{h*.75} {mid_x*.5},{h*.3} {mid_x*1.4},{h*.3} {w},{h*.65}" fill="none" stroke="{G}" stroke-width="2"/>',
        "rev_jade_lizard":    lambda: f'<polygon points="0,{h*.65} {mid_x*.6},{h*.3} {mid_x*1.5},{h*.3} {w},{h*.75}" fill="{BG_G}"/><polyline points="0,{h*.65} {mid_x*.6},{h*.3} {mid_x*1.5},{h*.3} {w},{h*.75}" fill="none" stroke="{G}" stroke-width="2"/>',
        "call_ratio":         lambda: f'<polygon points="0,{h*.5} {mid_x*.6},{h*.5} {mid_x},{h*.2} {mid_x*1.4},{h*.5} {w},{h*.9}" fill="{BG_G}" opacity="0.5"/><polyline points="0,{h*.5} {mid_x*.6},{h*.5} {mid_x},{h*.2} {mid_x*1.4},{h*.5} {w},{h*.9}" fill="none" stroke="{G}" stroke-width="2"/>',
        "put_ratio":          lambda: f'<polygon points="0,{h*.9} {mid_x*.6},{h*.5} {mid_x},{h*.2} {mid_x*1.4},{h*.5} {w},{h*.5}" fill="{BG_R}" opacity="0.5"/><polyline points="0,{h*.9} {mid_x*.6},{h*.5} {mid_x},{h*.2} {mid_x*1.4},{h*.5} {w},{h*.5}" fill="none" stroke="{R}" stroke-width="2"/>',
        "batman":             lambda: f'<polygon points="0,{h*.75} {mid_x*.3},{h*.75} {mid_x*.5},{h*.35} {mid_x*.7},{h*.2} {mid_x*1.3},{h*.2} {mid_x*1.5},{h*.35} {mid_x*1.7},{h*.75} {w},{h*.75}" fill="{BG_G}"/><polyline points="0,{h*.8} {mid_x*.3},{h*.75} {mid_x*.5},{h*.35} {mid_x*.7},{h*.2} {mid_x*1.3},{h*.2} {mid_x*1.5},{h*.35} {mid_x*1.7},{h*.75} {w},{h*.8}" fill="none" stroke="{G}" stroke-width="2"/>',
        "long_iron_fly":      lambda: f'<polygon points="0,{h*.2} {mid_x},{h*.8} {w},{h*.2}" fill="{BG_G}"/><polyline points="0,{h*.2} {mid_x},{h*.8} {w},{h*.2}" fill="none" stroke="{G}" stroke-width="2"/>',
        "short_iron_fly":     lambda: f'<polygon points="{mid_x*.5},{h*.75} {mid_x},{h*.2} {mid_x*1.5},{h*.75}" fill="{BG_R}"/><polyline points="0,{h*.75} {mid_x*.5},{h*.75} {mid_x},{h*.2} {mid_x*1.5},{h*.75} {w},{h*.75}" fill="none" stroke="{R}" stroke-width="2"/>',
        "double_fly":         lambda: f'<polygon points="{mid_x*.3},{h*.7} {mid_x*.7},{h*.25} {mid_x},{h*.55} {mid_x*1.3},{h*.25} {mid_x*1.7},{h*.7}" fill="{BG_G}"/><polyline points="0,{h*.75} {mid_x*.3},{h*.7} {mid_x*.7},{h*.25} {mid_x},{h*.55} {mid_x*1.3},{h*.25} {mid_x*1.7},{h*.7} {w},{h*.75}" fill="none" stroke="{G}" stroke-width="2"/>',
        "long_iron_condor":   lambda: f'<polygon points="0,{h*.2} {mid_x*.4},{h*.75} {mid_x*1.6},{h*.75} {w},{h*.2}" fill="{BG_G}"/><polyline points="0,{h*.2} {mid_x*.4},{h*.75} {mid_x*1.6},{h*.75} {w},{h*.2}" fill="none" stroke="{G}" stroke-width="2"/>',
        "short_iron_condor":  lambda: f'<polygon points="{mid_x*.3},{h*.75} {mid_x*.7},{h*.25} {mid_x*1.3},{h*.25} {mid_x*1.7},{h*.75}" fill="{BG_R}"/><polyline points="0,{h*.8} {mid_x*.3},{h*.75} {mid_x*.7},{h*.25} {mid_x*1.3},{h*.25} {mid_x*1.7},{h*.75} {w},{h*.8}" fill="none" stroke="{R}" stroke-width="2"/>',
        "double_condor":      lambda: f'<polygon points="{mid_x*.2},{h*.75} {mid_x*.5},{h*.25} {mid_x*1.5},{h*.25} {mid_x*1.8},{h*.75}" fill="{BG_R}"/><polyline points="0,{h*.8} {mid_x*.2},{h*.75} {mid_x*.5},{h*.25} {mid_x*1.5},{h*.25} {mid_x*1.8},{h*.75} {w},{h*.8}" fill="none" stroke="{G}" stroke-width="2"/>',
        "call_calendar":      lambda: f'<polygon points="{mid_x*.6},{h*.75} {mid_x},{h*.15} {mid_x*1.4},{h*.75}" fill="{BG_G}"/><polyline points="0,{h*.65} {mid_x*.6},{h*.75} {mid_x},{h*.15} {mid_x*1.4},{h*.75} {w},{h*.65}" fill="none" stroke="{G}" stroke-width="2"/>',
        "put_calendar":       lambda: f'<polygon points="{mid_x*.6},{h*.75} {mid_x},{h*.15} {mid_x*1.4},{h*.75}" fill="{BG_R}"/><polyline points="0,{h*.65} {mid_x*.6},{h*.75} {mid_x},{h*.15} {mid_x*1.4},{h*.75} {w},{h*.65}" fill="none" stroke="{R}" stroke-width="2"/>',
        "diagonal_calendar":  lambda: f'<polygon points="{mid_x*.4},{h*.7} {mid_x*.9},{h*.2} {mid_x*1.5},{h*.7}" fill="{BG_G}"/><polyline points="0,{h*.7} {mid_x*.4},{h*.7} {mid_x*.9},{h*.2} {mid_x*1.5},{h*.7} {w},{h*.7}" fill="none" stroke="{G}" stroke-width="2"/>',
        "call_butterfly":     lambda: f'<polygon points="{mid_x*.5},{h*.75} {mid_x},{h*.2} {mid_x*1.5},{h*.75}" fill="{BG_G}"/><polyline points="0,{h*.8} {mid_x*.5},{h*.75} {mid_x},{h*.2} {mid_x*1.5},{h*.75} {w},{h*.8}" fill="none" stroke="{G}" stroke-width="2"/>',
        "put_butterfly":      lambda: f'<polygon points="{mid_x*.5},{h*.75} {mid_x},{h*.2} {mid_x*1.5},{h*.75}" fill="{BG_R}"/><polyline points="0,{h*.8} {mid_x*.5},{h*.75} {mid_x},{h*.2} {mid_x*1.5},{h*.75} {w},{h*.8}" fill="none" stroke="{R}" stroke-width="2"/>',
    }
    fn = diagrams.get(payoff_type)
    inner = fn() if fn else f'<line x1="0" y1="{mid_y}" x2="{w}" y2="{mid_y}" stroke="#7b6cf6" stroke-width="2"/>'
    return (f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">'
            f'{AXIS}{inner}</svg>')


# =================================================================
#  SECTION 6 -- STRATEGY DEFINITIONS (FULL LIBRARY)
# =================================================================

ALL_STRATEGIES = {
    "bullish": {
        "label": "Bullish", "color": "#0fa86b",
        "items": [
            {"name": "Long Call",            "risk": "Limited",   "reward": "Unlimited", "legs": "BUY CALL (ATM)",                                        "desc": "Buy a call option. Profits as market rises. Risk limited to premium paid.",                              "mp": "Unlimited",             "ml": "Premium Paid",        "be": "Strike + Premium",        "payoff": "bullish_ramp"},
            {"name": "Short Put",            "risk": "High",      "reward": "Limited",   "legs": "SELL PUT (OTM)",                                         "desc": "Sell a put option. Collect premium; profits if market stays above strike.",                               "mp": "Premium Received",      "ml": "Strike - Premium",    "be": "Strike - Premium",        "payoff": "short_put"},
            {"name": "Bull Call Spread",     "risk": "Limited",   "reward": "Limited",   "legs": "BUY CALL (Low) · SELL CALL (High)",                     "desc": "Buy lower call, sell higher call. Reduces cost; caps profit at upper strike.",                            "mp": "Spread Width - Debit",  "ml": "Net Debit",           "be": "Lower Strike + Debit",    "payoff": "bull_spread"},
            {"name": "Bull Put Spread",      "risk": "Limited",   "reward": "Limited",   "legs": "SELL PUT (High) · BUY PUT (Low)",                       "desc": "Sell higher put, buy lower put. Receive credit; profit if market stays above sold strike.",               "mp": "Net Credit",            "ml": "Spread - Credit",     "be": "Higher Strike - Credit",  "payoff": "bull_put_spread"},
            {"name": "Call Ratio Back Spread","risk": "Limited",  "reward": "Unlimited", "legs": "SELL 1 CALL (Low) · BUY 2 CALLS (High)",                "desc": "Sell fewer lower calls, buy more higher calls. Profits from big upward move.",                           "mp": "Unlimited",             "ml": "Net Debit/Credit",    "be": "Varies",                  "payoff": "call_ratio_back"},
            {"name": "Long Synthetic",       "risk": "High",      "reward": "Unlimited", "legs": "BUY CALL (ATM) · SELL PUT (ATM)",                       "desc": "Mimics owning the underlying. Unlimited upside, large downside risk.",                                   "mp": "Unlimited",             "ml": "Large",               "be": "ATM Strike",              "payoff": "long_synthetic"},
            {"name": "Range Forward",        "risk": "Moderate",  "reward": "Moderate",  "legs": "BUY CALL (High) · SELL PUT (Low)",                      "desc": "Participate in upside above call strike; protected against downside below put strike.",                   "mp": "Above Call Strike",     "ml": "Below Put Strike",    "be": "Between Strikes",         "payoff": "range_forward"},
            {"name": "Bullish Butterfly",    "risk": "Limited",   "reward": "Limited",   "legs": "BUY 1 Low CALL · SELL 2 Mid CALL · BUY 1 High CALL",   "desc": "Profit from moderate rise to middle strike. Very low cost, defined risk.",                                "mp": "At Middle Strike",      "ml": "Net Debit",           "be": "Low+Debit & High-Debit", "payoff": "bull_butterfly"},
            {"name": "Bullish Condor",       "risk": "Limited",   "reward": "Limited",   "legs": "BUY Call(A) · SELL Call(B) · SELL Call(C) · BUY Call(D)","desc": "Four-strike call spread. Profit when market moves moderately higher.",                                   "mp": "Net Credit (B-C range)","ml": "Spread - Credit",     "be": "A+Debit & D-Debit",      "payoff": "bull_condor"},
        ]
    },
    "bearish": {
        "label": "Bearish", "color": "#c0392b",
        "items": [
            {"name": "Long Put",             "risk": "Limited",   "reward": "High",      "legs": "BUY PUT (ATM)",                                          "desc": "Buy a put option. Profits as market falls. Risk limited to premium paid.",                               "mp": "Strike - Premium",      "ml": "Premium Paid",        "be": "Strike - Premium",        "payoff": "bearish_ramp"},
            {"name": "Short Call",           "risk": "Unlimited", "reward": "Limited",   "legs": "SELL CALL (OTM)",                                        "desc": "Sell a call option. Collect premium; profits if market stays below strike.",                              "mp": "Premium Received",      "ml": "Unlimited",           "be": "Strike + Premium",        "payoff": "short_call"},
            {"name": "Bear Put Spread",      "risk": "Limited",   "reward": "Limited",   "legs": "BUY PUT (High) · SELL PUT (Low)",                       "desc": "Buy higher put, sell lower put. Cheaper bearish bet with capped profit.",                                "mp": "Spread - Debit",        "ml": "Net Debit",           "be": "Higher Strike - Debit",   "payoff": "bear_put_spread"},
            {"name": "Bear Call Spread",     "risk": "Limited",   "reward": "Limited",   "legs": "SELL CALL (Low) · BUY CALL (High)",                     "desc": "Sell lower call, buy higher call. Credit received; profit if stock stays below lower strike.",            "mp": "Net Credit",            "ml": "Spread - Credit",     "be": "Lower Strike + Credit",   "payoff": "bear_call_spread"},
            {"name": "Put Ratio Back Spread","risk": "Limited",   "reward": "High",      "legs": "SELL 1 PUT (High) · BUY 2 PUTS (Low)",                  "desc": "Sell fewer higher puts, buy more lower puts. Profits from big downward move.",                          "mp": "Large on big drop",     "ml": "Net Debit/Credit",    "be": "Varies",                  "payoff": "put_ratio_back"},
            {"name": "Short Synthetic",      "risk": "Unlimited", "reward": "High",      "legs": "SELL CALL (ATM) · BUY PUT (ATM)",                       "desc": "Mimics being short the underlying. Large downside profit, unlimited upside risk.",                       "mp": "Large",                 "ml": "Unlimited",           "be": "ATM Strike",              "payoff": "short_synthetic"},
            {"name": "Bearish Butterfly",    "risk": "Limited",   "reward": "Limited",   "legs": "BUY 1 High PUT · SELL 2 Mid PUT · BUY 1 Low PUT",       "desc": "Profit from moderate fall to middle strike. Very low cost, defined risk.",                               "mp": "At Middle Strike",      "ml": "Net Debit",           "be": "High-Debit & Low+Debit",  "payoff": "bear_butterfly"},
            {"name": "Bearish Condor",       "risk": "Limited",   "reward": "Limited",   "legs": "BUY Put(D) · SELL Put(C) · SELL Put(B) · BUY Put(A)",   "desc": "Four-strike put spread. Profit when market moves moderately lower.",                                    "mp": "Net Credit (B-C range)","ml": "Spread - Credit",     "be": "D-Debit & A+Debit",      "payoff": "bear_condor"},
        ]
    },
    "nondirectional": {
        "label": "Non-Directional", "color": "#7b6cf6",
        "items": [
            {"name": "Long Straddle",        "risk": "Limited",   "reward": "Unlimited", "legs": "BUY CALL + BUY PUT (ATM)",                               "desc": "Buy ATM call and put. Profit from large move in either direction.",                                      "mp": "Unlimited both sides",  "ml": "Total Premium",       "be": "Strike ± Total Premium",  "payoff": "long_straddle"},
            {"name": "Short Straddle",       "risk": "Unlimited", "reward": "Limited",   "legs": "SELL CALL + SELL PUT (ATM)",                             "desc": "Sell ATM call and put. Profit if market stays near strike. High risk.",                                 "mp": "Net Credit",            "ml": "Unlimited both sides","be": "Strike ± Credit",          "payoff": "short_straddle"},
            {"name": "Long Strangle",        "risk": "Limited",   "reward": "Unlimited", "legs": "BUY OTM CALL + BUY OTM PUT",                             "desc": "Buy OTM call and put. Cheaper than straddle; needs bigger move to profit.",                             "mp": "Unlimited both sides",  "ml": "Total Premium",       "be": "Strikes ± Total Premium", "payoff": "long_strangle"},
            {"name": "Short Strangle",       "risk": "Unlimited", "reward": "Limited",   "legs": "SELL OTM CALL + SELL OTM PUT",                           "desc": "Sell OTM call and put. Wider range than short straddle. High risk.",                                   "mp": "Net Credit",            "ml": "Unlimited both sides","be": "Strikes ± Credit",         "payoff": "short_strangle"},
            {"name": "Jade Lizard",          "risk": "Limited",   "reward": "Limited",   "legs": "SELL OTM CALL SPREAD + SELL OTM PUT",                    "desc": "No upside risk if total credit > call spread width. Complex income strategy.",                          "mp": "Net Credit",            "ml": "Put Strike - Credit", "be": "Put Strike - Credit",     "payoff": "jade_lizard"},
            {"name": "Reverse Jade Lizard",  "risk": "Limited",   "reward": "Limited",   "legs": "SELL OTM PUT SPREAD + SELL OTM CALL",                    "desc": "No downside risk. Opposite of Jade Lizard. Credit strategy.",                                          "mp": "Net Credit",            "ml": "Varies",              "be": "Varies",                  "payoff": "rev_jade_lizard"},
            {"name": "Call Ratio Spread",    "risk": "Unlimited", "reward": "Limited",   "legs": "BUY 1 CALL (Low) · SELL 2 CALLS (High)",                 "desc": "Profit if market stays below higher strike. Risk on big upside move.",                                 "mp": "At Higher Strike",      "ml": "Unlimited above top", "be": "Varies",                  "payoff": "call_ratio"},
            {"name": "Put Ratio Spread",     "risk": "High",      "reward": "Limited",   "legs": "BUY 1 PUT (High) · SELL 2 PUTS (Low)",                   "desc": "Profit if market stays above lower strike. Risk on big downside move.",                                "mp": "At Lower Strike",       "ml": "Large below bottom",  "be": "Varies",                  "payoff": "put_ratio"},
            {"name": "Batman Strategy",      "risk": "Limited",   "reward": "Limited",   "legs": "Complex 8-leg Spread",                                   "desc": "Wide profit zone. Combination of spreads creating a batman payoff shape.",                               "mp": "Wide middle zone",      "ml": "Net Debit",           "be": "Outer strikes ± Debit",   "payoff": "batman"},
            {"name": "Long Iron Fly",        "risk": "Limited",   "reward": "Limited",   "legs": "BUY ATM CALL+PUT · SELL OTM CALL+PUT",                   "desc": "Profit from large move. Inverse of short iron fly. Buy the wings.",                                    "mp": "Large move either side","ml": "Net Debit",           "be": "ATM ± Debit",             "payoff": "long_iron_fly"},
            {"name": "Short Iron Fly",       "risk": "Limited",   "reward": "Limited",   "legs": "SELL ATM CALL+PUT · BUY OTM CALL+PUT",                   "desc": "Profit if market stays near ATM. Sell the body, buy the wings.",                                      "mp": "Net Credit",            "ml": "Wing Width - Credit", "be": "ATM ± Credit",            "payoff": "short_iron_fly"},
            {"name": "Double Fly",           "risk": "Limited",   "reward": "Limited",   "legs": "TWO BUTTERFLY SPREADS at different centers",              "desc": "Two butterflies combined for a wider profit zone. Complex but defined risk.",                           "mp": "At either center",      "ml": "Net Debit",           "be": "Outer edges ± Debit",     "payoff": "double_fly"},
            {"name": "Long Iron Condor",     "risk": "Limited",   "reward": "Limited",   "legs": "BUY OTM CALL+PUT · SELL Further OTM CALL+PUT",           "desc": "Profit from large move in either direction. Debit strategy.",                                          "mp": "Big move either side",  "ml": "Net Debit",           "be": "Inner strikes ± Debit",   "payoff": "long_iron_condor"},
            {"name": "Short Iron Condor",    "risk": "Limited",   "reward": "Limited",   "legs": "SELL OTM CALL+PUT · BUY Further OTM CALL+PUT",           "desc": "Profit if market stays within a range. Most popular income strategy.",                                 "mp": "Net Credit",            "ml": "Wing Width - Credit", "be": "Short strikes ± Credit",  "payoff": "short_iron_condor"},
            {"name": "Double Condor",        "risk": "Limited",   "reward": "Limited",   "legs": "TWO CONDOR SPREADS combined",                            "desc": "Extended profit zone of condor. Very wide range strategy.",                                            "mp": "Net Credit",            "ml": "Wing - Credit",       "be": "Outer short strikes ± Credit", "payoff": "double_condor"},
            {"name": "Call Calendar",        "risk": "Limited",   "reward": "Limited",   "legs": "SELL Near-Term CALL · BUY Far-Term CALL (Same Strike)",   "desc": "Profit from time decay. Near-term decays faster than far-term.",                                       "mp": "At strike at near expiry","ml": "Net Debit",          "be": "Near strike area",        "payoff": "call_calendar"},
            {"name": "Put Calendar",         "risk": "Limited",   "reward": "Limited",   "legs": "SELL Near-Term PUT · BUY Far-Term PUT (Same Strike)",     "desc": "Same as call calendar but using puts. Profit from time decay.",                                        "mp": "At strike at near expiry","ml": "Net Debit",          "be": "Near strike area",        "payoff": "put_calendar"},
            {"name": "Diagonal Calendar",    "risk": "Limited",   "reward": "Limited",   "legs": "SELL Near-Term Option · BUY Far-Term Option (Diff Strike)","desc": "Like calendar spread but with different strikes. More flexibility.",                                   "mp": "Varies",                "ml": "Net Debit",           "be": "Varies",                  "payoff": "diagonal_calendar"},
            {"name": "Call Butterfly",       "risk": "Limited",   "reward": "Limited",   "legs": "BUY 1 Low CALL · SELL 2 Mid CALL · BUY 1 High CALL",    "desc": "Classic butterfly using calls. Maximum profit at middle strike.",                                       "mp": "At Middle Strike",      "ml": "Net Debit",           "be": "Low+Debit & High-Debit",  "payoff": "call_butterfly"},
            {"name": "Put Butterfly",        "risk": "Limited",   "reward": "Limited",   "legs": "BUY 1 High PUT · SELL 2 Mid PUT · BUY 1 Low PUT",        "desc": "Classic butterfly using puts. Maximum profit at middle strike.",                                        "mp": "At Middle Strike",      "ml": "Net Debit",           "be": "High-Debit & Low+Debit",  "payoff": "put_butterfly"},
        ]
    }
}


# =================================================================
#  SECTION 7 -- STRATEGY RECOMMENDATIONS
# =================================================================

def recommend_strategies(bias, atm_strike, oi_dir):
    atm = atm_strike
    tech_map = {
        "BULLISH": [
            {"name": "Bull Call Spread",    "legs": f"Buy {atm} CE - Sell {atm+200} CE",                                                                           "type": "Debit",  "risk": "Moderate"},
            {"name": "Long Call",           "legs": f"Buy {atm} CE",                                                                                                "type": "Debit",  "risk": "High"},
            {"name": "Bull Put Spread",     "legs": f"Sell {atm-100} PE - Buy {atm-200} PE",                                                                        "type": "Credit", "risk": "Moderate"},
        ],
        "BEARISH": [
            {"name": "Bear Put Spread",     "legs": f"Buy {atm} PE - Sell {atm-200} PE",                                                                           "type": "Debit",  "risk": "Moderate"},
            {"name": "Long Put",            "legs": f"Buy {atm} PE",                                                                                                "type": "Debit",  "risk": "High"},
            {"name": "Bear Call Spread",    "legs": f"Sell {atm+100} CE - Buy {atm+200} CE",                                                                        "type": "Credit", "risk": "Moderate"},
        ],
        "SIDEWAYS": [
            {"name": "Iron Condor",         "legs": f"Sell {atm+100} CE / Buy {atm+200} CE / Sell {atm-100} PE / Buy {atm-200} PE",                                "type": "Credit", "risk": "Low"},
            {"name": "Iron Butterfly",      "legs": f"Sell {atm} CE / Sell {atm} PE / Buy {atm+100} CE / Buy {atm-100} PE",                                         "type": "Credit", "risk": "Low"},
            {"name": "Short Straddle",      "legs": f"Sell {atm} CE - Sell {atm} PE",                                                                              "type": "Credit", "risk": "Very High"},
        ],
    }
    oi_map = {
        "Strong Bullish":      {"name": "Long Call",      "legs": f"Buy {atm} CE",                                 "signal": "Put build-up - bullish momentum"},
        "Bullish":             {"name": "Long Call",      "legs": f"Buy {atm} CE",                                 "signal": "Put build-up dominant"},
        "Strong Bearish":      {"name": "Long Put",       "legs": f"Buy {atm} PE",                                 "signal": "Call build-up - bearish momentum"},
        "Bearish":             {"name": "Long Put",       "legs": f"Buy {atm} PE",                                 "signal": "Call build-up dominant"},
        "Neutral (High Vol)":  {"name": "Long Straddle",  "legs": f"Buy {atm} CE + {atm} PE",                      "signal": "Both building - big move expected"},
        "Neutral (Unwinding)": {"name": "Iron Butterfly", "legs": f"Sell {atm} CE+PE, Buy {atm+100}+{atm-100}",   "signal": "Unwinding - range bound"},
    }
    tech_strats = tech_map.get(bias, tech_map["SIDEWAYS"])
    oi_strat    = oi_map.get(oi_dir, {"name": "Vertical Spread", "legs": f"Near {atm}", "signal": "Mixed signals"})
    return tech_strats, oi_strat


# =================================================================
#  SECTION 8 -- HTML SECTION BUILDERS
# =================================================================

def build_oi_html(oc):
    ce  = oc["ce_chg"]
    pe  = oc["pe_chg"]
    net = oc["net_chg"]
    bull_force = (abs(pe) if pe > 0 else 0) + (abs(ce) if ce < 0 else 0)
    bear_force = (abs(ce) if ce > 0 else 0) + (abs(pe) if pe < 0 else 0)
    total_f    = bull_force + bear_force or 1
    bull_pct   = round(bull_force / total_f * 100)
    bear_pct   = 100 - bull_pct

    def oi_card(lbl, val, is_bull, sub):
        col = "#0fa86b" if is_bull else "#c0392b"
        sig = "Bullish Signal" if is_bull else "Bearish Signal"
        bg  = "rgba(15,168,107,.07)"  if is_bull else "rgba(192,57,43,.07)"
        bdr = "rgba(15,168,107,.25)"  if is_bull else "rgba(192,57,43,.25)"
        return (
            f"<div class=\"oi-card\">"
            f"<div class=\"oi-lbl\">{lbl}</div>"
            f"<div class=\"oi-val\" style=\"color:{col};\">{val:+,}</div>"
            f"<div class=\"oi-sub\">{sub}</div>"
            f"<div class=\"oi-sig\" style=\"color:{col};background:{bg};border:1px solid {bdr};\">{sig}</div>"
            f"</div>"
        )

    oi_cls = oc["oi_cls"]
    oi_dir = oc["oi_dir"]
    oi_sig = oc["oi_sig"]
    expiry = oc["expiry"]
    dir_col = "#0fa86b" if oi_cls == "bullish" else ("#c0392b" if oi_cls == "bearish" else "#d4a017")
    dir_bg  = ("rgba(15,168,107,.06)"  if oi_cls == "bullish" else
               "rgba(192,57,43,.06)"   if oi_cls == "bearish" else "rgba(212,160,23,.06)")
    dir_bdr = ("rgba(15,168,107,.22)"  if oi_cls == "bullish" else
               "rgba(192,57,43,.22)"   if oi_cls == "bearish" else "rgba(212,160,23,.22)")

    return (
        f"<div class=\"section\"><div class=\"sec-title\">CHANGE IN OPEN INTEREST"
        f"<span class=\"sec-sub\">ATM +-10 strikes only &middot; Expiry: {expiry}</span></div>"
        f"<div class=\"oi-dir-box\" style=\"background:{dir_bg};border:1px solid {dir_bdr};\">"
        f"<div class=\"oi-dir-tag\">OI DIRECTION</div>"
        f"<div class=\"oi-dir-name\" style=\"color:{dir_col};\">{oi_dir}</div>"
        f"<div class=\"oi-dir-sig\" style=\"color:{dir_col};opacity:.75;\">{oi_sig}</div>"
        f"<div class=\"oi-meter-wrap\">"
        f"<div class=\"oi-meter-lbl\">BULL STRENGTH</div>"
        f"<div class=\"oi-meter-track\"><div style=\"width:{bull_pct}%;height:100%;"
        f"background:linear-gradient(90deg,#0d8a9e,#26d0a0);border-radius:4px;\"></div></div>"
        f"<div class=\"oi-meter-pct\" style=\"color:#0d8a9e;\">{bull_pct}% Bull &middot; {bear_pct}% Bear</div>"
        f"</div></div>"
        f"<div class=\"oi-cards\">"
        f"{oi_card('CALL OI CHANGE', ce,  ce < 0,  'CE open interest delta')}"
        f"{oi_card('PUT OI CHANGE',  pe,  pe > 0,  'PE open interest delta')}"
        f"{oi_card('NET OI CHANGE',  net, net > 0, 'PE delta + CE delta')}"
        f"</div>"
        f"<div class=\"oi-legend\">"
        f"<span>Call OI + = Writers selling calls <b style=\"color:#c0392b;\">Bearish</b></span>"
        f"<span>Call OI - = Unwinding <b style=\"color:#0fa86b;\">Bullish</b></span>"
        f"<span>Put OI + = Writers selling puts <b style=\"color:#0fa86b;\">Bullish</b></span>"
        f"<span>Put OI - = Unwinding <b style=\"color:#c0392b;\">Bearish</b></span>"
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
        max_pain = oc['max_pain']
        mp_html  = (
            f"<div class=\"kl-node\" style=\"left:{mp_p}%;top:0;transform:translateX(-50%);\">"
            f"<div class=\"kl-dot\" style=\"background:#d4a017;box-shadow:0 0 8px #d4a01760;margin:0 auto 4px;\"></div>"
            f"<div class=\"kl-lbl\" style=\"color:#d4a017;\">Max Pain</div>"
            f"<div class=\"kl-val\" style=\"color:#b8880e;\">Rs{max_pain:,}</div>"
            f"</div>"
        )

    return (
        f"<div class=\"section\"><div class=\"sec-title\">KEY LEVELS"
        f"<span class=\"sec-sub\">1H Candles &middot; Last 120 bars &middot; ATM +-200 pts &middot; Rounded to 25</span></div>"
        f"<div class=\"kl-zone-labels\">"
        f"<span style=\"color:#0d8a9e;\">SUPPORT ZONE</span>"
        f"<span style=\"color:#c0392b;\">RESISTANCE ZONE</span>"
        f"</div>"
        f"<div style=\"position:relative;height:58px;\">"
        f"<div class=\"kl-node\" style=\"left:3%;bottom:0;transform:translateX(-50%);\">"
        f"<div class=\"kl-lbl\" style=\"color:#0d6e80;\">Strong Sup</div>"
        f"<div class=\"kl-val\" style=\"color:#0d4a5a;\">Rs{ss:,.0f}</div>"
        f"<div class=\"kl-dot\" style=\"background:#0d6e80;margin:5px auto 0;\"></div></div>"
        f"<div class=\"kl-node\" style=\"left:22%;bottom:0;transform:translateX(-50%);\">"
        f"<div class=\"kl-lbl\" style=\"color:#0d8a9e;\">Support</div>"
        f"<div class=\"kl-val\" style=\"color:#0d5c6e;\">Rs{s1:,.0f}</div>"
        f"<div class=\"kl-dot\" style=\"background:#0d8a9e;box-shadow:0 0 8px #0d8a9e70;margin:5px auto 0;\"></div></div>"
        f"<div style=\"position:absolute;left:{cp_pct}%;bottom:6px;transform:translateX(-50%);"
        f"background:#1a9aad;color:#fff;font-size:11px;font-weight:700;"
        f"padding:3px 12px;border-radius:6px;white-space:nowrap;"
        f"box-shadow:0 2px 10px rgba(26,154,173,.45);z-index:10;\">NOW Rs{cp:,.0f}</div>"
        f"<div class=\"kl-node\" style=\"left:75%;bottom:0;transform:translateX(-50%);\">"
        f"<div class=\"kl-lbl\" style=\"color:#c0392b;\">Resistance</div>"
        f"<div class=\"kl-val\" style=\"color:#8b2318;\">Rs{r1:,.0f}</div>"
        f"<div class=\"kl-dot\" style=\"background:#c0392b;box-shadow:0 0 8px #c0392b70;margin:5px auto 0;\"></div></div>"
        f"<div class=\"kl-node\" style=\"left:95%;bottom:0;transform:translateX(-50%);\">"
        f"<div class=\"kl-lbl\" style=\"color:#96281b;\">Strong Res</div>"
        f"<div class=\"kl-val\" style=\"color:#6b1c12;\">Rs{sr:,.0f}</div>"
        f"<div class=\"kl-dot\" style=\"background:#96281b;margin:5px auto 0;\"></div></div>"
        f"</div>"
        f"<div class=\"kl-gradient-bar\"><div class=\"kl-price-tick\" style=\"left:{cp_pct}%;\"></div></div>"
        f"<div style=\"position:relative;height:54px;\">{mp_html}</div>"
        f"<div class=\"kl-dist-row\">"
        f"<div class=\"kl-dist-box\" style=\"border-color:rgba(192,57,43,.18);\">"
        f"<span style=\"color:#5a6a7a;\">To Resistance</span>"
        f"<span style=\"color:#c0392b;font-weight:700;\">+{pts_r:,} pts</span></div>"
        f"<div class=\"kl-dist-box\" style=\"border-color:rgba(13,138,158,.18);\">"
        f"<span style=\"color:#5a6a7a;\">To Support</span>"
        f"<span style=\"color:#0d8a9e;font-weight:700;\">-{pts_s:,} pts</span></div>"
        f"</div></div>"
    )


def build_strategy_selector_html(md, tech, oc):
    """Builds the full strategy selector with tab switcher + payoff diagram cards."""
    bias   = md["bias"]
    atm    = oc["atm_strike"] if oc else (round(tech["price"] / 50) * 50 if tech else 25000)
    oi_dir = oc["oi_dir"] if oc else "Neutral"

    tech_strats, oi_strat = recommend_strategies(bias, atm, oi_dir)

    bc = "#0fa86b" if bias == "BULLISH" else ("#c0392b" if bias == "BEARISH" else "#d4a017")

    # --- Recommended banner ---
    rec_cards = ""
    for s in tech_strats:
        rec_cards += (
            f"<div class=\"rec-card\" style=\"border-color:{bc}28;\">"
            f"<div class=\"rec-name\">{s['name']}</div>"
            f"<div class=\"rec-legs\">{s['legs']}</div>"
            f"<div style=\"display:flex;gap:6px;margin-top:8px;\">"
            f"<span class=\"rec-tag\">{s['type']}</span>"
            f"<span class=\"rec-tag\">{s['risk']} Risk</span></div></div>"
        )

    rec_block = (
        f"<div class=\"rec-banner\" style=\"border-color:{bc}30;background:{bc}05;\">"
        f"<div class=\"rec-title\" style=\"color:{bc};\">TODAY'S RECOMMENDED STRATEGIES &mdash; {bias}</div>"
        f"<div class=\"rec-grid\">{rec_cards}</div>"
        f"<div class=\"rec-oi-box\">"
        f"<span class=\"rec-oi-lbl\">OI Signal Strategy:</span>"
        f"<span class=\"rec-oi-name\">{oi_strat['name']}</span>"
        f"<span class=\"rec-oi-legs\">{oi_strat['legs']}</span>"
        f"<span class=\"rec-oi-sig\">{oi_strat['signal']}</span>"
        f"</div></div>"
    )

    # --- Tab buttons ---
    tab_defs = [
        ("bullish",        "BULLISH",         "#0fa86b", "9"),
        ("bearish",        "BEARISH",         "#c0392b", "8"),
        ("nondirectional", "NON-DIRECTIONAL", "#7b6cf6", "20"),
    ]
    tabs_html = '<div class="stab-wrap">'
    for key, label, color, count in tab_defs:
        active = "stab-active" if key == "bullish" else ""
        tabs_html += (
            f'<button class="stab {active}" data-tab="{key}" '
            f'onclick="switchTab(\'{key}\')" style="--tc:{color};">'
            f'{label} <span class="stab-count">{count}</span></button>'
        )
    tabs_html += "</div>"

    # --- Strategy cards per category ---
    panels_html = ""
    for cat_key, cat_data in ALL_STRATEGIES.items():
        col   = cat_data["color"]
        items = cat_data["items"]
        visible = "block" if cat_key == "bullish" else "none"
        cards_html = '<div class="strat-grid">'
        for s in items:
            svg     = get_payoff_svg(s.get("payoff", ""), 80, 50)
            rc      = "#0fa86b" if s["risk"] in ("Limited",) else ("#d4a017" if s["risk"] == "Moderate" else "#c0392b")
            rwc     = "#0fa86b" if s["reward"] == "Unlimited" else "#d4a017" if s["reward"] == "High" else "#8a8a8a"
            cards_html += (
                f'<div class="strat-card" onclick="openModal(this)" '
                f'data-name="{s["name"]}" data-legs="{s["legs"]}" data-desc="{s["desc"]}" '
                f'data-mp="{s["mp"]}" data-ml="{s["ml"]}" data-be="{s["be"]}" '
                f'data-risk="{s["risk"]}" data-reward="{s["reward"]}" data-color="{col}">'
                f'<div class="sc-svg">{svg}</div>'
                f'<div class="sc-name">{s["name"]}</div>'
                f'<div class="sc-legs">{s["legs"]}</div>'
                f'<div class="sc-badges">'
                f'<span class="sc-badge" style="color:{rc};">Risk: {s["risk"]}</span>'
                f'<span class="sc-badge" style="color:{rwc};">Reward: {s["reward"]}</span>'
                f'</div></div>'
            )
        cards_html += "</div>"
        panels_html += f'<div id="tab-{cat_key}" class="stab-panel" style="display:{visible};">{cards_html}</div>'

    # --- Detail modal (shown on card click) ---
    modal_html = """
<div id="strat-modal" class="smodal-overlay" onclick="if(event.target===this)closeModal()">
  <div class="smodal-box">
    <div class="smodal-header">
      <div>
        <div class="smodal-name" id="modal-name">Strategy Name</div>
        <div class="smodal-legs" id="modal-legs">Legs</div>
      </div>
      <button class="smodal-close" onclick="closeModal()">&#x2715;</button>
    </div>
    <div class="smodal-body">
      <div class="smodal-svg-wrap" id="modal-svg"></div>
      <p class="smodal-desc" id="modal-desc"></p>
      <div class="smodal-metrics">
        <div class="smodal-metric"><span class="smodal-ml">Max Profit</span><span class="smodal-mv" id="modal-mp" style="color:#0fa86b;"></span></div>
        <div class="smodal-metric"><span class="smodal-ml">Max Loss</span><span class="smodal-mv" id="modal-ml" style="color:#c0392b;"></span></div>
        <div class="smodal-metric"><span class="smodal-ml">Breakeven</span><span class="smodal-mv" id="modal-be" style="color:#d4a017;"></span></div>
      </div>
    </div>
  </div>
</div>"""

    js = """
<script>
function switchTab(key) {
  document.querySelectorAll('.stab-panel').forEach(p => p.style.display = 'none');
  document.querySelectorAll('.stab').forEach(b => b.classList.remove('stab-active'));
  const panel = document.getElementById('tab-' + key);
  if (panel) panel.style.display = 'block';
  const btn = document.querySelector('[data-tab="' + key + '"]');
  if (btn) btn.classList.add('stab-active');
}
function openModal(card) {
  const m = document.getElementById('strat-modal');
  document.getElementById('modal-name').textContent = card.dataset.name;
  document.getElementById('modal-legs').textContent = card.dataset.legs;
  document.getElementById('modal-desc').textContent = card.dataset.desc;
  document.getElementById('modal-mp').textContent   = card.dataset.mp;
  document.getElementById('modal-ml').textContent   = card.dataset.ml;
  document.getElementById('modal-be').textContent   = card.dataset.be;
  // Clone svg from card for modal
  const svgEl = card.querySelector('.sc-svg');
  const svgBox = document.getElementById('modal-svg');
  svgBox.innerHTML = svgEl ? svgEl.innerHTML.replace('width="80" height="50"','width="220" height="120"').replace('viewBox="0 0 80 50"','viewBox="0 0 80 50"') : '';
  m.style.display = 'flex';
}
function closeModal() {
  document.getElementById('strat-modal').style.display = 'none';
}
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });
</script>"""

    return (
        f"<div class=\"section\"><div class=\"sec-title\">STRATEGY RECOMMENDATIONS &amp; BUILDER</div>"
        f"{rec_block}"
        f"<div class=\"sec-title\" style=\"border:none;padding:0;margin:24px 0 14px;font-size:12px;\">ALL STRATEGIES REFERENCE</div>"
        f"{tabs_html}"
        f"{panels_html}"
        f"{modal_html}"
        f"{js}"
        f"</div>"
    )


def build_strikes_html(oc):
    if not oc or (not oc["top_ce"] and not oc["top_pe"]):
        return ""

    def ce_rows(rows):
        out = ""
        for i, r in enumerate(rows, 1):
            out += (f"<tr><td>{i}</td><td><b>Rs{int(r['Strike']):,}</b></td>"
                    f"<td>{int(r['CE_OI']):,}</td>"
                    f"<td style=\"color:#0d8a9e;font-weight:700;\">Rs{r['CE_LTP']:.2f}</td></tr>")
        return out

    def pe_rows(rows):
        out = ""
        for i, r in enumerate(rows, 1):
            out += (f"<tr><td>{i}</td><td><b>Rs{int(r['Strike']):,}</b></td>"
                    f"<td>{int(r['PE_OI']):,}</td>"
                    f"<td style=\"color:#c0392b;font-weight:700;\">Rs{r['PE_LTP']:.2f}</td></tr>")
        return out

    return (
        f"<div class=\"section\"><div class=\"sec-title\">TOP 5 STRIKES BY OPEN INTEREST"
        f"<span class=\"sec-sub\">ATM +-10 only</span></div>"
        f"<div class=\"strikes-wrap\">"
        f"<div><div style=\"color:#0d8a9e;font-weight:700;margin-bottom:10px;\">CALL Options (CE)</div>"
        f"<table class=\"s-table\"><thead><tr><th>#</th><th>Strike</th><th>OI</th><th>LTP</th></tr></thead>"
        f"<tbody>{ce_rows(oc['top_ce'])}</tbody></table></div>"
        f"<div><div style=\"color:#c0392b;font-weight:700;margin-bottom:10px;\">PUT Options (PE)</div>"
        f"<table class=\"s-table\"><thead><tr><th>#</th><th>Strike</th><th>OI</th><th>LTP</th></tr></thead>"
        f"<tbody>{pe_rows(oc['top_pe'])}</tbody></table></div>"
        f"</div></div>"
    )


# =================================================================
#  SECTION 9 -- CSS
# =================================================================

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=DM+Mono:wght@300;400;500&display=swap');
*,*::before,*::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg: #e8edf2; --surf: #dde4ec; --card: #d0d9e4; --bdr: #bfcad8; --bdr2: #a8b8cc;
  --teal-dark: #0d5c6e; --teal: #1a9aad; --teal-light: #26d0a0; --teal-pale: #c8d8e4;
  --text: #0d2d38; --muted: #4a5a6a; --muted2: #6a7a8a;
  --bull: #0fa86b; --bear: #c0392b; --neut: #d4a017; --accent: #0d8a9e;
  --fh: 'Sora', sans-serif; --fm: 'DM Mono', monospace;
}
html { scroll-behavior: smooth; }
body { background: var(--bg); color: var(--text); font-family: var(--fh); font-size: 13px; line-height: 1.6; min-height: 100vh; }
body::before {
  content: ''; position: fixed; inset: 0;
  background-image: radial-gradient(circle at 15% 25%, rgba(26,154,173,.05) 0%, transparent 45%),
    radial-gradient(circle at 85% 75%, rgba(13,92,110,.05) 0%, transparent 45%),
    linear-gradient(rgba(13,92,110,.025) 1px, transparent 1px),
    linear-gradient(90deg, rgba(13,92,110,.025) 1px, transparent 1px);
  background-size: 100% 100%, 100% 100%, 36px 36px, 36px 36px;
  pointer-events: none; z-index: 0;
}
.app { position: relative; z-index: 1; display: grid; grid-template-rows: auto auto 1fr auto; min-height: 100vh; }
header { display: flex; align-items: center; justify-content: space-between; padding: 14px 32px; border-bottom: 2px solid rgba(255,255,255,.15); background: linear-gradient(135deg, #0d4a5a, #0d6e80); position: sticky; top: 0; z-index: 200; box-shadow: 0 2px 12px rgba(13,74,90,.3); }
.logo { font-family: var(--fh); font-size: 20px; font-weight: 700; color: #fff; letter-spacing: .5px; }
.logo span { color: #a8eee0; }
.hdr-meta { display: flex; align-items: center; gap: 14px; font-size: 11px; color: rgba(255,255,255,.55); font-family: var(--fm); }
.live-dot { width: 7px; height: 7px; border-radius: 50%; background: #26d0a0; box-shadow: 0 0 8px #26d0a0; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100% { opacity:1 } 50% { opacity:.3 } }
.hero { padding: 26px 32px; background: linear-gradient(135deg, #0d4a5a 0%, #0d6e80 50%, #1a9aad 100%); border-bottom: 1px solid rgba(255,255,255,.1); display: flex; align-items: center; justify-content: space-between; gap: 24px; flex-wrap: wrap; box-shadow: 0 3px 16px rgba(13,74,90,.2); }
.hero-dir { font-family: var(--fh); font-size: 52px; font-weight: 700; line-height: 1; letter-spacing: -2px; color: #fff; }
.hero-sub { font-size: 13px; color: rgba(255,255,255,.55); margin-top: 5px; }
.hero-conf { display: inline-block; margin-top: 10px; font-size: 11px; font-weight: 600; padding: 4px 16px; border-radius: 20px; letter-spacing: 1px; border: 1px solid rgba(255,255,255,.25); background: rgba(255,255,255,.1); color: #a8eee0; }
.hero-stats { display: flex; gap: 28px; align-items: center; flex-wrap: wrap; }
.hstat { text-align: center; }
.hstat-lbl { font-size: 10px; color: rgba(255,255,255,.45); letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 3px; }
.hstat-val { font-family: var(--fm); font-size: 18px; font-weight: 600; color: #fff; }
.main { display: grid; grid-template-columns: 268px 1fr; min-height: 0; }
.sidebar { border-right: 1px solid var(--bdr2); background: linear-gradient(180deg, #d0dae6 0%, #c8d4e2 100%); position: sticky; top: 57px; height: calc(100vh - 57px); overflow-y: auto; }
.sidebar::-webkit-scrollbar { width: 3px; }
.sidebar::-webkit-scrollbar-thumb { background: var(--bdr2); border-radius: 2px; }
.sb-sec { padding: 16px 12px 8px; }
.sb-lbl { font-size: 9px; font-weight: 700; letter-spacing: .15em; text-transform: uppercase; color: var(--teal); margin-bottom: 8px; padding: 0 0 0 8px; border-left: 3px solid var(--teal); }
.sb-btn { display: flex; align-items: center; gap: 8px; width: 100%; padding: 9px 12px; border-radius: 8px; border: 1px solid transparent; cursor: pointer; background: transparent; color: var(--muted); font-family: var(--fh); font-size: 12px; text-align: left; transition: all .15s; }
.sb-btn:hover { background: rgba(13,138,158,.1); color: var(--teal-dark); border-color: var(--bdr); }
.sb-btn.active { background: rgba(13,138,158,.14); border-color: var(--bdr2); color: var(--teal-dark); font-weight: 600; }
.sb-badge { font-size: 10px; margin-left: auto; font-weight: 700; }
.sig-card { margin: 12px 10px 8px; padding: 16px 14px; background: linear-gradient(135deg, #0d4a5a, #0d8a9e); border-radius: 12px; border: 1px solid rgba(255,255,255,.1); text-align: center; box-shadow: 0 4px 16px rgba(13,74,90,.25); }
.sig-arrow { font-family: var(--fh); font-size: 38px; font-weight: 700; line-height: 1; margin-bottom: 4px; color: #a8eee0; }
.sig-bias  { font-family: var(--fh); font-size: 18px; font-weight: 700; color: #fff; }
.sig-meta  { font-size: 10px; color: rgba(255,255,255,.5); margin-top: 4px; }
.content { overflow-y: auto; }
.section { padding: 24px 28px; border-bottom: 1px solid var(--bdr); background: var(--bg); }
.section:nth-child(odd) { background: var(--surf); }
.sec-title { font-family: var(--fh); font-size: 12px; font-weight: 700; letter-spacing: 2px; color: var(--teal-dark); text-transform: uppercase; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 18px; padding-bottom: 12px; border-bottom: 2px solid var(--teal-pale); }
.sec-sub { font-size: 11px; color: var(--muted2); font-weight: 400; letter-spacing: .5px; text-transform: none; margin-left: auto; }
.oi-dir-box { border-radius: 12px; padding: 18px 20px; margin-bottom: 16px; }
.oi-dir-tag { font-size: 10px; letter-spacing: 2px; color: var(--muted2); margin-bottom: 6px; text-transform: uppercase; }
.oi-dir-name { font-family: var(--fh); font-size: 26px; font-weight: 700; margin-bottom: 4px; }
.oi-dir-sig { font-size: 13px; }
.oi-meter-wrap { margin-top: 16px; }
.oi-meter-lbl { font-size: 10px; color: var(--muted2); letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 5px; }
.oi-meter-track { height: 8px; background: rgba(13,92,110,.1); border-radius: 4px; overflow: hidden; width: 100%; max-width: 320px; }
.oi-meter-pct { font-size: 12px; margin-top: 4px; }
.oi-cards { display: grid; grid-template-columns: repeat(3,1fr); gap: 12px; margin-bottom: 12px; }
.oi-card { background: var(--card); border: 1px solid var(--bdr); border-radius: 12px; padding: 16px; box-shadow: 0 1px 6px rgba(13,60,80,.07); }
.oi-lbl { font-size: 9px; letter-spacing: 2px; color: var(--muted2); text-transform: uppercase; margin-bottom: 8px; }
.oi-val { font-family: var(--fm); font-size: 24px; font-weight: 600; margin-bottom: 4px; }
.oi-sub { font-size: 10px; color: var(--muted2); margin-bottom: 12px; }
.oi-sig { display: block; padding: 7px 12px; border-radius: 6px; text-align: center; font-size: 12px; font-weight: 600; }
.oi-legend { display: flex; flex-wrap: wrap; gap: 10px 20px; font-size: 11px; color: var(--muted); padding: 10px 0; }
.kl-zone-labels { display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 11px; font-weight: 700; }
.kl-node { position: absolute; text-align: center; }
.kl-lbl { font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: .5px; line-height: 1.3; white-space: nowrap; }
.kl-val { font-size: 12px; font-weight: 700; color: var(--text); white-space: nowrap; margin-top: 2px; }
.kl-dot { width: 11px; height: 11px; border-radius: 50%; border: 2px solid var(--surf); }
.kl-gradient-bar { position: relative; height: 8px; border-radius: 4px; background: linear-gradient(90deg,#0d6e80 0%,#0d8a9e 20%,#26d0a0 40%,#d4a017 58%,#c0392b 80%,#96281b 100%); }
.kl-price-tick { position: absolute; top: 50%; transform: translate(-50%,-50%); width: 4px; height: 20px; background: var(--teal-dark); border-radius: 2px; box-shadow: 0 0 10px rgba(13,92,110,.5); z-index: 10; }
.kl-dist-row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 4px; }
.kl-dist-box { background: var(--card); border: 1px solid; border-radius: 8px; padding: 9px 14px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 1px 4px rgba(13,60,80,.06); }
.rec-banner { border: 1px solid; border-radius: 14px; padding: 18px 20px; margin-bottom: 20px; }
.rec-title { font-family: var(--fh); font-size: 15px; font-weight: 700; margin-bottom: 14px; }
.rec-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 10px; margin-bottom: 14px; }
.rec-card { background: var(--card); border: 1px solid; border-radius: 10px; padding: 14px; box-shadow: 0 1px 6px rgba(13,60,80,.07); }
.rec-name { font-family: var(--fh); font-size: 14px; font-weight: 700; color: var(--teal-dark); margin-bottom: 5px; }
.rec-legs { font-family: var(--fm); font-size: 11px; color: var(--muted); }
.rec-tag { font-size: 10px; padding: 2px 8px; border-radius: 4px; background: rgba(13,138,158,.07); color: var(--teal); border: 1px solid var(--bdr); }
.rec-oi-box { background: var(--card); border: 1px solid var(--bdr); border-radius: 8px; padding: 12px 14px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; font-size: 12px; box-shadow: 0 1px 4px rgba(13,60,80,.06); }
.rec-oi-lbl { color: var(--muted2); font-size: 10px; text-transform: uppercase; letter-spacing: 1px; }
.rec-oi-name { font-weight: 700; color: var(--teal-dark); }
.rec-oi-legs { font-family: var(--fm); font-size: 11px; color: var(--muted); }
.rec-oi-sig { margin-left: auto; font-style: italic; color: var(--muted2); }

/* ---- Strategy Tabs ---- */
.stab-wrap { display: flex; gap: 6px; margin-bottom: 18px; flex-wrap: wrap; }
.stab {
  padding: 8px 18px; border-radius: 8px; border: 1.5px solid var(--tc, #0d8a9e);
  background: transparent; color: var(--tc, #0d8a9e); font-family: var(--fh);
  font-size: 12px; font-weight: 600; cursor: pointer; transition: all .18s; letter-spacing: .5px;
}
.stab:hover { background: color-mix(in srgb, var(--tc, #0d8a9e) 12%, transparent); }
.stab-active { background: var(--tc, #0d8a9e) !important; color: #fff !important; box-shadow: 0 3px 12px color-mix(in srgb, var(--tc,#0d8a9e) 30%, transparent); }
.stab-count { font-size: 10px; background: rgba(255,255,255,.25); padding: 1px 6px; border-radius: 10px; margin-left: 4px; }
.stab-panel { animation: fadeIn .2s ease; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: none; } }

/* ---- Strategy Grid Cards ---- */
.strat-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(170px,1fr)); gap: 12px; }
.strat-card {
  background: var(--card); border: 1px solid var(--bdr); border-radius: 12px;
  padding: 14px 12px 12px; cursor: pointer; transition: all .2s;
  box-shadow: 0 1px 6px rgba(13,60,80,.07);
}
.strat-card:hover { border-color: var(--teal); transform: translateY(-2px); box-shadow: 0 6px 20px rgba(13,92,110,.13); }
.sc-svg { margin-bottom: 8px; display: flex; justify-content: center; }
.sc-name { font-family: var(--fh); font-size: 12px; font-weight: 700; color: var(--teal-dark); margin-bottom: 4px; line-height: 1.3; }
.sc-legs { font-family: var(--fm); font-size: 9px; color: var(--muted2); margin-bottom: 8px; line-height: 1.5; }
.sc-badges { display: flex; flex-wrap: wrap; gap: 4px; }
.sc-badge { font-size: 9px; padding: 1px 6px; border-radius: 4px; background: rgba(13,60,80,.05); border: 1px solid var(--bdr); }

/* ---- Strategy Modal ---- */
.smodal-overlay {
  display: none; position: fixed; inset: 0; z-index: 9000;
  background: rgba(13,30,40,.65); backdrop-filter: blur(4px);
  justify-content: center; align-items: center; padding: 20px;
}
.smodal-box {
  background: var(--surf); border: 1px solid var(--bdr2); border-radius: 18px;
  width: 100%; max-width: 480px; box-shadow: 0 24px 60px rgba(0,0,0,.3);
  overflow: hidden;
}
.smodal-header {
  display: flex; justify-content: space-between; align-items: flex-start;
  padding: 20px 22px 14px; border-bottom: 1px solid var(--bdr);
  background: linear-gradient(135deg, #0d4a5a, #0d8a9e);
}
.smodal-name { font-family: var(--fh); font-size: 20px; font-weight: 700; color: #fff; }
.smodal-legs { font-family: var(--fm); font-size: 11px; color: rgba(255,255,255,.6); margin-top: 4px; }
.smodal-close { background: rgba(255,255,255,.15); border: none; color: #fff; width: 30px; height: 30px; border-radius: 50%; cursor: pointer; font-size: 14px; display: flex; align-items: center; justify-content: center; transition: background .15s; }
.smodal-close:hover { background: rgba(255,255,255,.28); }
.smodal-body { padding: 20px 22px; }
.smodal-svg-wrap { display: flex; justify-content: center; margin-bottom: 14px; background: var(--card); border-radius: 10px; padding: 14px; border: 1px solid var(--bdr); }
.smodal-desc { font-size: 13px; color: var(--muted); margin-bottom: 16px; line-height: 1.7; }
.smodal-metrics { display: grid; grid-template-columns: repeat(3,1fr); gap: 10px; }
.smodal-metric { background: var(--card); border: 1px solid var(--bdr); border-radius: 8px; padding: 10px 12px; }
.smodal-ml { display: block; font-size: 9px; color: var(--muted2); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
.smodal-mv { font-family: var(--fm); font-size: 12px; font-weight: 600; }

.strikes-wrap { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.s-table { width: 100%; border-collapse: collapse; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 10px rgba(13,60,80,.09); }
.s-table th { background: linear-gradient(135deg, #0d4a5a, #0d8a9e); color: #fff; padding: 10px 12px; font-size: 11px; font-weight: 600; text-align: left; letter-spacing: .5px; }
.s-table td { padding: 10px 12px; border-bottom: 1px solid var(--bdr); font-size: 12px; color: var(--text); background: var(--card); }
.s-table tr:last-child td { border-bottom: none; }
.s-table tr:hover td { background: var(--surf); }
footer { padding: 16px 32px; border-top: 2px solid rgba(255,255,255,.12); background: linear-gradient(135deg, #0d4a5a, #0d6e80); display: flex; justify-content: space-between; font-size: 11px; color: rgba(255,255,255,.45); font-family: var(--fm); }
@media(max-width:1024px) { .main { grid-template-columns: 1fr; } .sidebar { position: static; height: auto; border-right: none; border-bottom: 1px solid var(--bdr); } .hero-dir { font-size: 36px; } .oi-cards { grid-template-columns: 1fr; } .rec-grid { grid-template-columns: 1fr; } .strikes-wrap { grid-template-columns: 1fr; } }
@media(max-width:640px) { header { padding: 12px 16px; } .hero { padding: 16px; flex-direction: column; } .hero-dir { font-size: 30px; } .section { padding: 16px; } .smodal-metrics { grid-template-columns: 1fr; } .kl-dist-row { grid-template-columns: 1fr; } footer { flex-direction: column; gap: 6px; } }
"""


# =================================================================
#  SECTION 10 -- MASTER HTML ASSEMBLER
# =================================================================

def generate_html(tech, oc, md, ts):
    cp     = tech["price"]    if tech else 0
    expiry = oc["expiry"]     if oc   else "N/A"
    atm    = oc["atm_strike"] if oc   else (round(cp / 50) * 50 if cp else 0)
    pcr    = oc["pcr_oi"]     if oc   else 0
    mp     = oc["max_pain"]   if oc   else 0

    bias = md["bias"]
    conf = md["confidence"]
    bull = md["bull"]
    bear = md["bear"]
    diff = md["diff"]

    b_arrow = "UP" if bias == "BULLISH" else ("DOWN" if bias == "BEARISH" else "SIDEWAYS")
    bc      = "#0d8a9e" if bias == "BULLISH" else ("#c0392b" if bias == "BEARISH" else "#d4a017")
    pcr_col = "var(--bull)" if pcr > 1.2 else ("var(--bear)" if pcr < 0.7 else "var(--neut)")

    oi_html      = build_oi_html(oc)                          if oc   else ""
    kl_html      = build_key_levels_html(tech, oc)           if tech else ""
    strat_html   = build_strategy_selector_html(md, tech, oc)
    strikes_html = build_strikes_html(oc)

    sig_card = (
        f"<div class=\"sb-sec\">"
        f"<div class=\"sb-lbl\">TODAY'S SIGNAL</div>"
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
  <div class="logo">NIFTY<span>CRAFT</span></div>
  <div class="hdr-meta">
    <div class="live-dot"></div>
    <span>NSE Options Dashboard</span>
    <span style="color:rgba(255,255,255,.2);">|</span>
    <span>{ts}</span>
  </div>
</header>
<div class="hero">
  <div>
    <div class="hero-dir">{bias}</div>
    <div class="hero-sub">Market Direction &middot; {conf} Confidence</div>
    <span class="hero-conf">Bull {bull} pt &nbsp;&middot;&nbsp; Bear {bear} pt &nbsp;&middot;&nbsp; Diff {diff:+d}</span>
  </div>
  <div class="hero-stats">
    <div class="hstat"><div class="hstat-lbl">NIFTY Spot</div><div class="hstat-val">&#8377;{cp:,.2f}</div></div>
    <div class="hstat"><div class="hstat-lbl">ATM Strike</div><div class="hstat-val" style="color:#a8eee0;">&#8377;{atm:,}</div></div>
    <div class="hstat"><div class="hstat-lbl">Expiry</div><div class="hstat-val" style="color:#f5d77a;">{expiry}</div></div>
    <div class="hstat"><div class="hstat-lbl">PCR (OI)</div><div class="hstat-val" style="color:{pcr_col};">{pcr:.3f}</div></div>
    <div class="hstat"><div class="hstat-lbl">Max Pain</div><div class="hstat-val" style="color:#f5d77a;">&#8377;{mp:,}</div></div>
  </div>
</div>
<div class="main">
  <aside class="sidebar">
    {sig_card}
    <div class="sb-sec">
      <div class="sb-lbl">LIVE ANALYSIS</div>
      <button class="sb-btn active" onclick="go('oi',this)">OI Change</button>
      <button class="sb-btn"       onclick="go('kl',this)">Key Levels</button>
    </div>
    <div class="sb-sec">
      <div class="sb-lbl">STRATEGIES</div>
      <button class="sb-btn" onclick="go('strat',this);switchTab('bullish')">Bullish <span class="sb-badge" style="color:var(--bull);">9</span></button>
      <button class="sb-btn" onclick="go('strat',this);switchTab('bearish')">Bearish <span class="sb-badge" style="color:var(--bear);">8</span></button>
      <button class="sb-btn" onclick="go('strat',this);switchTab('nondirectional')">Non-Directional <span class="sb-badge" style="color:#7b6cf6;">20</span></button>
    </div>
    <div class="sb-sec">
      <div class="sb-lbl">OPTION CHAIN</div>
      <button class="sb-btn" onclick="go('strikes',this)">Top 5 Strikes</button>
    </div>
  </aside>
  <main class="content">
    <div id="oi">{oi_html}</div>
    <div id="kl">{kl_html}</div>
    <div id="strat">{strat_html}</div>
    <div id="strikes">{strikes_html}</div>
    <div class="section">
      <div style="background:rgba(212,160,23,.06);border:1px solid rgba(212,160,23,.22);border-left:3px solid #d4a017;border-radius:10px;padding:16px 18px;font-size:13px;color:#7a5c00;line-height:1.8;">
        <strong>DISCLAIMER</strong><br>
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
  if (el) el.scrollIntoView({{ behavior: "smooth", block: "start" }});
  document.querySelectorAll(".sb-btn").forEach(b => b.classList.remove("active"));
  if (btn) btn.classList.add("active");
}}
</script>
</body>
</html>"""


# =================================================================
#  SECTION 11 -- MAIN RUNNER
# =================================================================

def main():
    ist_tz = pytz.timezone("Asia/Kolkata")
    ts     = datetime.now(ist_tz).strftime("%d-%b-%Y %H:%M IST")
    print("=" * 65)
    print("  NIFTY 50 OPTIONS DASHBOARD -- GitHub Pages Generator")
    print(f"  {ts}")
    print("=" * 65)

    print("\n[1/3] Fetching NSE Option Chain...")
    oc_raw      = NSEOptionChain().fetch()
    oc_analysis = analyze_option_chain(oc_raw) if oc_raw else None
    if oc_analysis:
        print(f"  OK  Spot={oc_analysis['underlying']:.2f}  ATM={oc_analysis['atm_strike']}  PCR={oc_analysis['pcr_oi']:.3f}")
    else:
        print("  WARNING  Option chain unavailable -- technical-only mode")

    print("\n[2/3] Fetching Technical Indicators...")
    tech = get_technical_data()

    print("\n[3/3] Scoring Market Direction...")
    md   = compute_market_direction(tech, oc_analysis)
    print(f"  OK  {md['bias']} ({md['confidence']} confidence)  Bull={md['bull']} Bear={md['bear']}")

    print("\nGenerating HTML dashboard...")
    html = generate_html(tech, oc_analysis, md, ts)
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
