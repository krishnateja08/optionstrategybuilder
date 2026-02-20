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
                json_data   = resp.json()
                data        = json_data.get("records", {}).get("data", [])
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
                df_full     = pd.DataFrame(rows).sort_values("Strike").reset_index(drop=True)
                underlying  = json_data.get("records", {}).get("underlyingValue", 0)
                atm_strike  = round(underlying / 50) * 50
                all_strikes = sorted(df_full["Strike"].unique())
                if atm_strike in all_strikes:
                    atm_idx = all_strikes.index(atm_strike)
                else:
                    atm_idx    = min(range(len(all_strikes)), key=lambda i: abs(all_strikes[i] - underlying))
                    atm_strike = all_strikes[atm_idx]
                lo  = max(0, atm_idx - 10)
                hi  = min(len(all_strikes) - 1, atm_idx + 10)
                df  = df_full[df_full["Strike"].isin(all_strikes[lo:hi + 1])].reset_index(drop=True)
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
            oi_dir, oi_sig, oi_icon, oi_cls = "Bullish",          "Put Build-up Dominant",    "GREEN",  "bullish"
        elif ce_chg > pe_chg * 1.5:
            oi_dir, oi_sig, oi_icon, oi_cls = "Bearish",          "Call Build-up Dominant",   "RED",    "bearish"
        else:
            oi_dir, oi_sig, oi_icon, oi_cls = "Neutral (High Vol)","Both Calls & Puts Building","YELLOW","neutral"
    elif ce_chg < 0 and pe_chg < 0:
        oi_dir, oi_sig, oi_icon, oi_cls = "Neutral (Unwinding)","Both Calls & Puts Unwinding","YELLOW","neutral"
    else:
        if   net_chg > 0: oi_dir, oi_sig, oi_icon, oi_cls = "Moderately Bullish","Net Put Accumulation",  "GREEN", "bullish"
        elif net_chg < 0: oi_dir, oi_sig, oi_icon, oi_cls = "Moderately Bearish","Net Call Accumulation", "RED",   "bearish"
        else:              oi_dir, oi_sig, oi_icon, oi_cls = "Neutral",           "Balanced OI Changes",   "YELLOW","neutral"

    max_ce_row   = df.loc[df["CE_OI"].idxmax()]
    max_pe_row   = df.loc[df["PE_OI"].idxmax()]
    df["pain"]   = abs(df["CE_OI"] - df["PE_OI"])
    max_pain_row = df.loc[df["pain"].idxmin()]
    top_ce = df.nlargest(5, "CE_OI")[["Strike","CE_OI","CE_LTP"]].to_dict("records")
    top_pe = df.nlargest(5, "PE_OI")[["Strike","PE_OI","PE_LTP"]].to_dict("records")

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
                if r1 and r1 <= cp:           r1 = round((cp + 50) / 25) * 25
                if r2 and r1 and r2 <= r1:    r2 = r1 + 75
                if s1 and s1 >= cp:           s1 = round((cp - 50) / 25) * 25
                if s2 and s1 and s2 >= s1:    s2 = s1 - 75
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
#  SECTION 4 -- MARKET DIRECTION SCORING  (internal, drives strategies)
# =================================================================

def compute_market_direction(tech, oc_analysis):
    if not tech:
        return {"bias": "UNKNOWN", "confidence": "LOW", "bull": 0, "bear": 0, "diff": 0}

    cp   = tech["price"]
    bull = bear = 0

    for sma in ["sma20","sma50","sma200"]:
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
#  SECTION 5 -- STRATEGY DEFINITIONS & RECOMMENDATIONS
# =================================================================

ALL_STRATEGIES = {
    "bullish": {
        "label": "Bullish", "color": "#00d4a0",
        "items": [
            {"name": "Long Call",
             "risk": "Limited",  "reward": "Unlimited",
             "legs": "BUY CALL (ATM)",
             "desc": "Buy a call option. Profits as stock rises. Risk limited to premium paid.",
             "mp": "Unlimited", "ml": "Premium Paid", "be": "Strike + Premium"},
            {"name": "Covered Call",
             "risk": "Moderate", "reward": "Limited",
             "legs": "OWN STOCK . SELL CALL (OTM)",
             "desc": "Own shares and sell a call against them. Generates income; caps upside.",
             "mp": "Strike - Cost + Premium", "ml": "Cost - Premium", "be": "Stock Cost - Premium"},
            {"name": "Bull Call Spread",
             "risk": "Limited",  "reward": "Limited",
             "legs": "BUY CALL (Low) . SELL CALL (High)",
             "desc": "Buy lower call, sell higher call. Reduces cost; caps profit at upper strike.",
             "mp": "Spread Width - Debit", "ml": "Net Debit", "be": "Lower Strike + Debit"},
            {"name": "Cash-Secured Put",
             "risk": "Moderate", "reward": "Limited",
             "legs": "SELL PUT (OTM/ATM)",
             "desc": "Sell a put holding enough cash. Collect premium; buy shares at discount if assigned.",
             "mp": "Premium Received", "ml": "Strike - Premium", "be": "Strike - Premium"},
        ]
    },
    "bearish": {
        "label": "Bearish", "color": "#ff4560",
        "items": [
            {"name": "Long Put",
             "risk": "Limited",  "reward": "High",
             "legs": "BUY PUT (ATM)",
             "desc": "Buy a put option. Profits as stock falls. Risk limited to premium paid.",
             "mp": "Strike - Premium", "ml": "Premium Paid", "be": "Strike - Premium"},
            {"name": "Bear Put Spread",
             "risk": "Limited",  "reward": "Limited",
             "legs": "BUY PUT (High) . SELL PUT (Low)",
             "desc": "Buy higher put, sell lower put. Cheaper bearish bet with capped profit.",
             "mp": "Spread - Debit", "ml": "Net Debit", "be": "Higher Strike - Debit"},
            {"name": "Bear Call Spread",
             "risk": "Limited",  "reward": "Limited",
             "legs": "SELL CALL (Low) . BUY CALL (High)",
             "desc": "Sell lower call, buy higher call. Credit received; profit if stock stays below lower strike.",
             "mp": "Net Credit", "ml": "Spread - Credit", "be": "Lower Strike + Credit"},
        ]
    },
    "neutral": {
        "label": "Neutral / Volatility", "color": "#f0b429",
        "items": [
            {"name": "Iron Condor",
             "risk": "Limited",  "reward": "Limited",
             "legs": "SELL OTM PUT+CALL SPREADS",
             "desc": "Sell OTM put spread + OTM call spread. Profit if stock stays in a defined range.",
             "mp": "Net Credit", "ml": "Spread - Credit", "be": "Short strikes +- Credit"},
            {"name": "Straddle",
             "risk": "Limited",  "reward": "Unlimited",
             "legs": "BUY CALL + PUT (ATM)",
             "desc": "Buy ATM call and put. Profit from a large move in either direction.",
             "mp": "Unlimited (both sides)", "ml": "Total Premium", "be": "Strike +- Total Premium"},
            {"name": "Strangle",
             "risk": "Limited",  "reward": "Unlimited",
             "legs": "BUY OTM CALL + OTM PUT",
             "desc": "Buy OTM call and OTM put. Cheaper than straddle; needs a bigger move to profit.",
             "mp": "Unlimited (both sides)", "ml": "Total Premium", "be": "Strikes +- Total Premium"},
            {"name": "Butterfly Spread",
             "risk": "Limited",  "reward": "Limited",
             "legs": "BUY Low . SELL 2xMid . BUY High",
             "desc": "Three strike combo. Maximum profit when stock lands exactly at middle strike.",
             "mp": "Mid - Low - Debit", "ml": "Net Debit", "be": "Low+Debit and High-Debit"},
        ]
    }
}


def recommend_strategies(bias, atm_strike, oi_dir):
    atm = atm_strike
    tech_map = {
        "BULLISH": [
            {"name": "Bull Call Spread", "legs": f"Buy {atm} CE - Sell {atm+200} CE",                    "type": "Debit",  "risk": "Moderate"},
            {"name": "Long Call",        "legs": f"Buy {atm} CE",                                         "type": "Debit",  "risk": "High"},
            {"name": "Bull Put Spread",  "legs": f"Sell {atm-100} PE - Buy {atm-200} PE",                 "type": "Credit", "risk": "Moderate"},
        ],
        "BEARISH": [
            {"name": "Bear Put Spread",  "legs": f"Buy {atm} PE - Sell {atm-200} PE",                     "type": "Debit",  "risk": "Moderate"},
            {"name": "Long Put",         "legs": f"Buy {atm} PE",                                         "type": "Debit",  "risk": "High"},
            {"name": "Bear Call Spread", "legs": f"Sell {atm+100} CE - Buy {atm+200} CE",                 "type": "Credit", "risk": "Moderate"},
        ],
        "SIDEWAYS": [
            {"name": "Iron Condor",    "legs": f"Sell {atm+100} CE / Buy {atm+200} CE / Sell {atm-100} PE / Buy {atm-200} PE", "type": "Credit", "risk": "Low"},
            {"name": "Iron Butterfly", "legs": f"Sell {atm} CE / Sell {atm} PE / Buy {atm+100} CE / Buy {atm-100} PE",          "type": "Credit", "risk": "Low"},
            {"name": "Short Straddle", "legs": f"Sell {atm} CE - Sell {atm} PE",                                                "type": "Credit", "risk": "Very High"},
        ],
    }
    oi_map = {
        "Strong Bullish":     {"name": "Long Call",      "legs": f"Buy {atm} CE",                              "signal": "Put build-up - bullish momentum"},
        "Bullish":            {"name": "Long Call",      "legs": f"Buy {atm} CE",                              "signal": "Put build-up dominant"},
        "Strong Bearish":     {"name": "Long Put",       "legs": f"Buy {atm} PE",                              "signal": "Call build-up - bearish momentum"},
        "Bearish":            {"name": "Long Put",       "legs": f"Buy {atm} PE",                              "signal": "Call build-up dominant"},
        "Neutral (High Vol)": {"name": "Long Straddle",  "legs": f"Buy {atm} CE + {atm} PE",                   "signal": "Both building - big move expected"},
        "Neutral (Unwinding)":{"name": "Iron Butterfly", "legs": f"Sell {atm} CE+PE, Buy {atm+100}+{atm-100}", "signal": "Unwinding - range bound"},
    }
    tech_strats = tech_map.get(bias, tech_map["SIDEWAYS"])
    oi_strat    = oi_map.get(oi_dir, {"name": "Vertical Spread", "legs": f"Near {atm}", "signal": "Mixed signals"})
    return tech_strats, oi_strat


# =================================================================
#  SECTION 6 -- HTML SECTION BUILDERS
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
        col = "#34d399" if is_bull else "#fb7185"
        sig = "Bullish Signal" if is_bull else "Bearish Signal"
        bg  = "rgba(16,185,129,.1)" if is_bull else "rgba(239,68,68,.1)"
        bdr = "rgba(16,185,129,.4)" if is_bull else "rgba(239,68,68,.4)"
        return (f"<div class=\"oi-card\">"
                f"<div class=\"oi-lbl\">{lbl}</div>"
                f"<div class=\"oi-val\" style=\"color:{col};\">{val:+,}</div>"
                f"<div class=\"oi-sub\">{sub}</div>"
                f"<div class=\"oi-sig\" style=\"color:{col};background:{bg};border:1px solid {bdr};\">{sig}</div>"
                f"</div>")

    dir_col = "#34d399" if oc["oi_cls"] == "bullish" else ("#fb7185" if oc["oi_cls"] == "bearish" else "#fbbf24")
    dir_bg  = ("rgba(10,30,20,.9)"   if oc["oi_cls"] == "bullish" else
               "rgba(30,10,14,.9)"   if oc["oi_cls"] == "bearish" else "rgba(20,20,10,.9)")
    dir_bdr = ("rgba(16,185,129,.35)" if oc["oi_cls"] == "bullish" else
               "rgba(239,68,68,.35)"  if oc["oi_cls"] == "bearish" else "rgba(251,191,36,.3)")

    expiry  = oc['expiry']
    oi_dir  = oc['oi_dir']
    oi_sig  = oc['oi_sig']

    return (
        f"<div class=\"section\"><div class=\"sec-title\">CHANGE IN OPEN INTEREST"
        f"<span class=\"sec-sub\">ATM +-10 strikes only &middot; Expiry: {expiry}</span></div>"
        f"<div class=\"oi-dir-box\" style=\"background:{dir_bg};border:1px solid {dir_bdr};\">"
        f"<div class=\"oi-dir-tag\">OI DIRECTION</div>"
        f"<div class=\"oi-dir-name\" style=\"color:{dir_col};\">{oi_dir}</div>"
        f"<div class=\"oi-dir-sig\" style=\"color:{dir_col}80;\">{oi_sig}</div>"
        f"<div class=\"oi-meter-wrap\">"
        f"<div class=\"oi-meter-lbl\">BULL STRENGTH</div>"
        f"<div class=\"oi-meter-track\"><div style=\"width:{bull_pct}%;height:100%;background:linear-gradient(90deg,#10b981,#34d399);border-radius:4px;\"></div></div>"
        f"<div class=\"oi-meter-pct\" style=\"color:#34d399;\">{bull_pct}% Bull &middot; {bear_pct}% Bear</div>"
        f"</div></div>"
        f"<div class=\"oi-cards\">"
        f"{oi_card('CALL OI CHANGE', ce,  ce < 0,  'CE open interest delta')}"
        f"{oi_card('PUT OI CHANGE',  pe,  pe > 0,  'PE open interest delta')}"
        f"{oi_card('NET OI CHANGE',  net, net > 0, 'PE delta + CE delta')}"
        f"</div>"
        f"<div class=\"oi-legend\">"
        f"<span>Call OI + = Writers selling calls <b style=\"color:#fb7185;\">Bearish</b></span>"
        f"<span>Call OI - = Unwinding <b style=\"color:#34d399;\">Bullish</b></span>"
        f"<span>Put OI + = Writers selling puts <b style=\"color:#34d399;\">Bullish</b></span>"
        f"<span>Put OI - = Unwinding <b style=\"color:#fb7185;\">Bearish</b></span>"
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

    cp_pct  = pct(cp)
    pts_r   = int(r1 - cp)
    pts_s   = int(cp - s1)

    mp_html = ""
    if oc:
        mp_p      = pct(oc["max_pain"])
        max_pain  = oc['max_pain']
        mp_html = (
            f"<div class=\"kl-node\" style=\"left:{mp_p}%;top:0;transform:translateX(-50%);\">"
            f"<div class=\"kl-dot\" style=\"background:#ffb74d;box-shadow:0 0 8px #ffb74d;margin:0 auto 4px;\"></div>"
            f"<div class=\"kl-lbl\" style=\"color:#ffb74d;\">Max Pain</div>"
            f"<div class=\"kl-val\" style=\"color:#ffb74d;\">Rs{max_pain:,}</div>"
            f"</div>"
        )

    return (
        f"<div class=\"section\"><div class=\"sec-title\">KEY LEVELS"
        f"<span class=\"sec-sub\">1H Candles &middot; Last 120 bars &middot; ATM +-200 pts &middot; Rounded to 25</span></div>"
        f"<div class=\"kl-zone-labels\">"
        f"<span style=\"color:#26c6da;\">SUPPORT ZONE</span>"
        f"<span style=\"color:#f44336;\">RESISTANCE ZONE</span>"
        f"</div>"
        f"<div style=\"position:relative;height:58px;\">"
        f"<div class=\"kl-node\" style=\"left:3%;bottom:0;transform:translateX(-50%);\">"
        f"<div class=\"kl-lbl\" style=\"color:#26c6da;\">Strong Sup</div>"
        f"<div class=\"kl-val\" style=\"color:#26c6da;\">Rs{ss:,.0f}</div>"
        f"<div class=\"kl-dot\" style=\"background:#26c6da;margin:5px auto 0;\"></div></div>"
        f"<div class=\"kl-node\" style=\"left:22%;bottom:0;transform:translateX(-50%);\">"
        f"<div class=\"kl-lbl\" style=\"color:#00bcd4;\">Support</div>"
        f"<div class=\"kl-val\" style=\"color:#00bcd4;\">Rs{s1:,.0f}</div>"
        f"<div class=\"kl-dot\" style=\"background:#00bcd4;box-shadow:0 0 8px #00bcd4;margin:5px auto 0;\"></div></div>"
        f"<div style=\"position:absolute;left:{cp_pct}%;bottom:6px;transform:translateX(-50%);"
        f"background:#4fc3f7;color:#000;font-size:11px;font-weight:700;"
        f"padding:3px 12px;border-radius:6px;white-space:nowrap;"
        f"box-shadow:0 0 14px rgba(79,195,247,.7);z-index:10;\">NOW Rs{cp:,.0f}</div>"
        f"<div class=\"kl-node\" style=\"left:75%;bottom:0;transform:translateX(-50%);\">"
        f"<div class=\"kl-lbl\" style=\"color:#ff7043;\">Resistance</div>"
        f"<div class=\"kl-val\" style=\"color:#ff7043;\">Rs{r1:,.0f}</div>"
        f"<div class=\"kl-dot\" style=\"background:#ff7043;box-shadow:0 0 8px #ff7043;margin:5px auto 0;\"></div></div>"
        f"<div class=\"kl-node\" style=\"left:95%;bottom:0;transform:translateX(-50%);\">"
        f"<div class=\"kl-lbl\" style=\"color:#f44336;\">Strong Res</div>"
        f"<div class=\"kl-val\" style=\"color:#f44336;\">Rs{sr:,.0f}</div>"
        f"<div class=\"kl-dot\" style=\"background:#f44336;margin:5px auto 0;\"></div></div>"
        f"</div>"
        f"<div class=\"kl-gradient-bar\"><div class=\"kl-price-tick\" style=\"left:{cp_pct}%;\"></div></div>"
        f"<div style=\"position:relative;height:54px;\">{mp_html}</div>"
        f"<div class=\"kl-dist-row\">"
        f"<div class=\"kl-dist-box\" style=\"border-color:rgba(244,67,54,.25);\">"
        f"<span style=\"color:#b0bec5;\">To Resistance</span>"
        f"<span style=\"color:#f44336;font-weight:700;\">+{pts_r:,} pts</span></div>"
        f"<div class=\"kl-dist-box\" style=\"border-color:rgba(0,188,212,.25);\">"
        f"<span style=\"color:#b0bec5;\">To Support</span>"
        f"<span style=\"color:#00bcd4;font-weight:700;\">-{pts_s:,} pts</span></div>"
        f"</div></div>"
    )


def build_strategies_html(md, tech, oc):
    bias   = md["bias"]
    atm    = oc["atm_strike"] if oc else (round(tech["price"] / 50) * 50 if tech else 25000)
    oi_dir = oc["oi_dir"] if oc else "Neutral"

    tech_strats, oi_strat = recommend_strategies(bias, atm, oi_dir)

    bc = "#00d4a0" if bias == "BULLISH" else ("#ff4560" if bias == "BEARISH" else "#f0b429")

    rec_cards = ""
    for s in tech_strats:
        s_name = s['name']
        s_legs = s['legs']
        s_type = s['type']
        s_risk = s['risk']
        rec_cards += (
            f"<div class=\"rec-card\" style=\"border-color:{bc}30;\">"
            f"<div class=\"rec-name\">{s_name}</div>"
            f"<div class=\"rec-legs\">{s_legs}</div>"
            f"<div style=\"display:flex;gap:6px;margin-top:8px;\">"
            f"<span class=\"rec-tag\">{s_type}</span>"
            f"<span class=\"rec-tag\">{s_risk} Risk</span></div></div>"
        )

    oi_name = oi_strat['name']
    oi_legs = oi_strat['legs']
    oi_sig  = oi_strat['signal']

    rec_block = (
        f"<div class=\"rec-banner\" style=\"border-color:{bc}40;background:{bc}08;\">"
        f"<div class=\"rec-title\" style=\"color:{bc};\">TODAY'S RECOMMENDED STRATEGIES &mdash; {bias}</div>"
        f"<div class=\"rec-grid\">{rec_cards}</div>"
        f"<div class=\"rec-oi-box\">"
        f"<span class=\"rec-oi-lbl\">OI Signal Strategy:</span>"
        f"<span class=\"rec-oi-name\">{oi_name}</span>"
        f"<span class=\"rec-oi-legs\">{oi_legs}</span>"
        f"<span class=\"rec-oi-sig\">{oi_sig}</span>"
        f"</div></div>"
    )

    dir_html = "<div class=\"strat-dir\">"
    for direction, info in ALL_STRATEGIES.items():
        col   = info["color"]
        label = info["label"]
        emoji = "UP" if direction == "bullish" else ("DOWN" if direction == "bearish" else "SIDEWAYS")
        dir_html += f"<div class=\"strat-group\"><div class=\"strat-group-title\" style=\"color:{col};\">{emoji} {label} Strategies</div>"
        for s in info["items"]:
            s_risk   = s["risk"]
            s_reward = s["reward"]
            s_name   = s["name"]
            s_desc   = s["desc"]
            s_legs   = s["legs"]
            s_mp     = s["mp"]
            s_ml     = s["ml"]
            s_be     = s["be"]
            rc  = "#34d399" if s_risk == "Limited" else "#fb7185" if s_risk == "High" else "#fbbf24"
            rwc = "#34d399" if s_reward == "Unlimited" else "#fbbf24"
            dir_html += (
                f"<div class=\"strat-card\">"
                f"<div class=\"strat-top\">"
                f"<div class=\"strat-name\">{s_name}</div>"
                f"<div style=\"display:flex;gap:5px;flex-wrap:wrap;\">"
                f"<span style=\"font-size:10px;padding:2px 8px;border-radius:4px;color:{rc};background:{rc}15;border:1px solid {rc}30;\">Risk: {s_risk}</span>"
                f"<span style=\"font-size:10px;padding:2px 8px;border-radius:4px;color:{rwc};background:{rwc}15;border:1px solid {rwc}30;\">Reward: {s_reward}</span>"
                f"</div></div>"
                f"<div class=\"strat-desc\">{s_desc}</div>"
                f"<div class=\"strat-legs\">{s_legs}</div>"
                f"<div class=\"strat-metrics\">"
                f"<div><span class=\"sm-lbl\">Max Profit</span><span style=\"color:#34d399;\">{s_mp}</span></div>"
                f"<div><span class=\"sm-lbl\">Max Loss</span><span style=\"color:#fb7185;\">{s_ml}</span></div>"
                f"<div><span class=\"sm-lbl\">Breakeven</span><span style=\"color:#fbbf24;\">{s_be}</span></div>"
                f"</div></div>"
            )
        dir_html += "</div>"
    dir_html += "</div>"

    return (
        f"<div class=\"section\"><div class=\"sec-title\">STRATEGY RECOMMENDATIONS &amp; BUILDER</div>"
        f"{rec_block}"
        f"<div class=\"sec-title\" style=\"border:none;padding:0;margin:24px 0 14px;font-size:12px;\">ALL STRATEGIES REFERENCE</div>"
        f"{dir_html}</div>"
    )


def build_strikes_html(oc):
    if not oc or (not oc["top_ce"] and not oc["top_pe"]):
        return ""

    def ce_rows(rows):
        out = ""
        for i, r in enumerate(rows, 1):
            strike = int(r['Strike'])
            ce_oi  = int(r['CE_OI'])
            ce_ltp = r['CE_LTP']
            out += (f"<tr><td>{i}</td><td><b>Rs{strike:,}</b></td>"
                    f"<td>{ce_oi:,}</td>"
                    f"<td style=\"color:#00bcd4;font-weight:700;\">Rs{ce_ltp:.2f}</td></tr>")
        return out

    def pe_rows(rows):
        out = ""
        for i, r in enumerate(rows, 1):
            strike = int(r['Strike'])
            pe_oi  = int(r['PE_OI'])
            pe_ltp = r['PE_LTP']
            out += (f"<tr><td>{i}</td><td><b>Rs{strike:,}</b></td>"
                    f"<td>{pe_oi:,}</td>"
                    f"<td style=\"color:#f44336;font-weight:700;\">Rs{pe_ltp:.2f}</td></tr>")
        return out

    top_ce = oc['top_ce']
    top_pe = oc['top_pe']

    return (
        f"<div class=\"section\"><div class=\"sec-title\">TOP 5 STRIKES BY OPEN INTEREST"
        f"<span class=\"sec-sub\">ATM +-10 only</span></div>"
        f"<div class=\"strikes-wrap\">"
        f"<div><div style=\"color:#00bcd4;font-weight:700;margin-bottom:10px;\">CALL Options (CE)</div>"
        f"<table class=\"s-table\"><thead><tr><th>#</th><th>Strike</th><th>OI</th><th>LTP</th></tr></thead>"
        f"<tbody>{ce_rows(top_ce)}</tbody></table></div>"
        f"<div><div style=\"color:#f44336;font-weight:700;margin-bottom:10px;\">PUT Options (PE)</div>"
        f"<table class=\"s-table\"><thead><tr><th>#</th><th>Strike</th><th>OI</th><th>LTP</th></tr></thead>"
        f"<tbody>{pe_rows(top_pe)}</tbody></table></div>"
        f"</div></div>"
    )


# =================================================================
#  SECTION 7 -- MASTER HTML ASSEMBLER
# =================================================================

CSS = """
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
:root{
  --bg:#080c10;--surf:#0e1318;--card:#111820;--bdr:#1a2530;--bdr2:#243040;
  --text:#c8d8e0;--muted:#5a7080;
  --bull:#00d4a0;--bear:#ff4560;--neut:#f0b429;--accent:#3d9eff;
  --fh:'Syne',sans-serif;--fm:'JetBrains Mono',monospace;--fb:'Rajdhani',sans-serif;
}
html{scroll-behavior:smooth;}
body{background:var(--bg);color:var(--text);font-family:var(--fb);font-size:13px;line-height:1.6;min-height:100vh;}
body::before{content:'';position:fixed;inset:0;
  background-image:linear-gradient(rgba(255,255,255,.012) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.012) 1px,transparent 1px);
  background-size:40px 40px;pointer-events:none;z-index:0;}
.app{position:relative;z-index:1;display:grid;grid-template-rows:auto auto 1fr auto;min-height:100vh;}

/* Header */
header{display:flex;align-items:center;justify-content:space-between;padding:14px 32px;
  border-bottom:1px solid var(--bdr);background:rgba(8,12,16,.97);
  backdrop-filter:blur(16px);position:sticky;top:0;z-index:200;}
.logo{font-family:var(--fh);font-size:18px;font-weight:800;color:#fff;letter-spacing:-.5px;}
.logo span{color:var(--bull);}
.hdr-meta{display:flex;align-items:center;gap:14px;font-size:11px;color:var(--muted);font-family:var(--fm);}
.live-dot{width:7px;height:7px;border-radius:50%;background:var(--bull);box-shadow:0 0 8px var(--bull);animation:pulse 2s infinite;}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}

/* Hero banner */
.hero{padding:28px 32px;background:linear-gradient(135deg,rgba(14,19,24,.97),rgba(8,12,16,1));
  border-bottom:1px solid var(--bdr);display:flex;align-items:center;
  justify-content:space-between;gap:24px;flex-wrap:wrap;}
.hero-dir{font-family:var(--fh);font-size:48px;font-weight:800;line-height:1;letter-spacing:-1.5px;}
.hero-sub{font-size:13px;color:var(--muted);margin-top:5px;}
.hero-conf{display:inline-block;margin-top:10px;font-size:11px;font-weight:700;
  padding:4px 14px;border-radius:20px;letter-spacing:1px;border:1px solid;}
.hero-stats{display:flex;gap:28px;align-items:center;flex-wrap:wrap;}
.hstat{text-align:center;}
.hstat-lbl{font-size:10px;color:var(--muted);letter-spacing:1.5px;text-transform:uppercase;margin-bottom:3px;}
.hstat-val{font-family:var(--fm);font-size:18px;font-weight:600;}

/* Main grid */
.main{display:grid;grid-template-columns:272px 1fr;min-height:0;}

/* Sidebar */
.sidebar{border-right:1px solid var(--bdr);background:var(--surf);
  position:sticky;top:57px;height:calc(100vh - 57px);overflow-y:auto;}
.sidebar::-webkit-scrollbar{width:3px;}
.sidebar::-webkit-scrollbar-thumb{background:var(--bdr2);border-radius:2px;}
.sb-sec{padding:14px 12px 6px;}
.sb-lbl{font-size:10px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;
  color:var(--muted);margin-bottom:8px;padding:0 4px;}
.sb-btn{display:flex;align-items:center;gap:8px;width:100%;padding:9px 12px;
  border-radius:8px;border:1px solid transparent;cursor:pointer;
  background:transparent;color:var(--muted);font-family:var(--fm);
  font-size:12px;text-align:left;transition:all .15s;}
.sb-btn:hover{background:var(--card);color:var(--text);}
.sb-btn.active{background:var(--card);border-color:var(--bdr2);color:var(--text);}
.sb-badge{font-size:10px;margin-left:auto;font-weight:700;}

/* Signal card in sidebar */
.sig-card{margin:0 8px 10px;padding:14px 12px;background:var(--card);
  border-radius:10px;border:1px solid;text-align:center;}
.sig-arrow{font-family:var(--fh);font-size:34px;font-weight:900;line-height:1;margin-bottom:4px;}
.sig-bias{font-family:var(--fh);font-size:17px;font-weight:800;}
.sig-meta{font-size:10px;color:var(--muted);margin-top:5px;}

/* Content */
.content{overflow-y:auto;}

/* Section */
.section{padding:24px 28px;border-bottom:1px solid var(--bdr);}
.sec-title{font-family:var(--fh);font-size:13px;font-weight:700;letter-spacing:2px;
  color:var(--accent);text-transform:uppercase;display:flex;align-items:center;
  gap:10px;flex-wrap:wrap;margin-bottom:18px;padding-bottom:12px;
  border-bottom:1px solid var(--bdr);}
.sec-sub{font-size:11px;color:var(--muted);font-weight:400;
  letter-spacing:.5px;text-transform:none;margin-left:auto;}

/* OI Section */
.oi-dir-box{border-radius:12px;padding:18px 20px;margin-bottom:16px;}
.oi-dir-tag{font-size:10px;letter-spacing:2px;color:rgba(148,163,184,.5);margin-bottom:6px;}
.oi-dir-name{font-family:var(--fh);font-size:26px;font-weight:700;margin-bottom:4px;}
.oi-dir-sig{font-size:13px;}
.oi-meter-wrap{margin-top:16px;}
.oi-meter-lbl{font-size:10px;color:rgba(148,163,184,.5);letter-spacing:2px;margin-bottom:5px;}
.oi-meter-track{height:8px;background:rgba(0,0,0,.4);border-radius:4px;overflow:hidden;width:100%;max-width:320px;}
.oi-meter-pct{font-size:12px;margin-top:3px;}
.oi-cards{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:12px;}
.oi-card{background:rgba(20,28,45,.9);border:1px solid rgba(79,195,247,.12);border-radius:12px;padding:16px;}
.oi-lbl{font-size:9px;letter-spacing:2px;color:rgba(148,163,184,.5);text-transform:uppercase;margin-bottom:8px;}
.oi-val{font-family:var(--fh);font-size:26px;font-weight:700;margin-bottom:4px;}
.oi-sub{font-size:10px;color:rgba(100,116,139,.7);margin-bottom:12px;}
.oi-sig{display:block;padding:7px 12px;border-radius:6px;text-align:center;font-size:12px;font-weight:700;}
.oi-legend{display:flex;flex-wrap:wrap;gap:10px 20px;font-size:11px;color:rgba(176,190,197,.6);padding:10px 0;}

/* Key Levels */
.kl-zone-labels{display:flex;justify-content:space-between;margin-bottom:6px;font-size:11px;font-weight:700;}
.kl-node{position:absolute;text-align:center;}
.kl-lbl{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;line-height:1.3;white-space:nowrap;}
.kl-val{font-size:12px;font-weight:700;color:#fff;white-space:nowrap;margin-top:2px;}
.kl-dot{width:11px;height:11px;border-radius:50%;border:2px solid rgba(8,12,16,.9);}
.kl-gradient-bar{position:relative;height:8px;border-radius:4px;
  background:linear-gradient(90deg,#26c6da 0%,#00bcd4 20%,#4fc3f7 40%,#ffb74d 58%,#ff7043 76%,#f44336 100%);}
.kl-price-tick{position:absolute;top:50%;transform:translate(-50%,-50%);
  width:4px;height:20px;background:#fff;border-radius:2px;box-shadow:0 0 14px #fff;z-index:10;}
.kl-dist-row{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:4px;}
.kl-dist-box{background:rgba(255,255,255,.03);border:1px solid;border-radius:8px;
  padding:9px 14px;display:flex;justify-content:space-between;align-items:center;}

/* Strategies */
.rec-banner{border:1px solid;border-radius:14px;padding:18px 20px;margin-bottom:20px;}
.rec-title{font-family:var(--fh);font-size:15px;font-weight:700;margin-bottom:14px;}
.rec-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:14px;}
.rec-card{background:rgba(255,255,255,.04);border:1px solid;border-radius:10px;padding:14px;}
.rec-name{font-family:var(--fh);font-size:14px;font-weight:700;color:#fff;margin-bottom:5px;}
.rec-legs{font-family:var(--fm);font-size:11px;color:var(--muted);}
.rec-tag{font-size:10px;padding:2px 8px;border-radius:4px;
  background:rgba(255,255,255,.06);color:var(--muted);border:1px solid var(--bdr2);}
.rec-oi-box{background:rgba(255,255,255,.03);border:1px solid var(--bdr);
  border-radius:8px;padding:12px 14px;display:flex;align-items:center;gap:10px;flex-wrap:wrap;font-size:12px;}
.rec-oi-lbl{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:1px;}
.rec-oi-name{font-weight:700;color:#fff;}
.rec-oi-legs{font-family:var(--fm);font-size:11px;color:var(--muted);}
.rec-oi-sig{margin-left:auto;font-style:italic;color:var(--muted);}

.strat-dir{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;}
.strat-group{display:flex;flex-direction:column;gap:10px;}
.strat-group-title{font-family:var(--fh);font-size:13px;font-weight:700;margin-bottom:4px;}
.strat-card{background:var(--card);border:1px solid var(--bdr);border-radius:10px;padding:14px;transition:all .2s;}
.strat-card:hover{border-color:var(--bdr2);transform:translateY(-2px);box-shadow:0 6px 24px rgba(0,0,0,.3);}
.strat-top{display:flex;justify-content:space-between;align-items:flex-start;gap:8px;margin-bottom:8px;}
.strat-name{font-family:var(--fh);font-size:14px;font-weight:700;color:#fff;}
.strat-desc{font-size:11px;color:var(--muted);line-height:1.7;margin-bottom:8px;}
.strat-legs{font-family:var(--fm);font-size:10px;color:rgba(61,158,255,.8);margin-bottom:10px;letter-spacing:.5px;}
.strat-metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;}
.strat-metrics>div{background:rgba(255,255,255,.03);border:1px solid var(--bdr);border-radius:6px;padding:6px 8px;}
.sm-lbl{display:block;font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:2px;}

/* Strikes table */
.strikes-wrap{display:grid;grid-template-columns:1fr 1fr;gap:20px;}
.s-table{width:100%;border-collapse:collapse;border-radius:8px;overflow:hidden;}
.s-table th{background:linear-gradient(135deg,#4fc3f7,#26c6da);color:#000;padding:10px;font-size:11px;font-weight:700;text-align:left;}
.s-table td{padding:10px;border-bottom:1px solid rgba(79,195,247,.06);font-size:12px;color:var(--muted);}
.s-table tr:hover{background:rgba(79,195,247,.04);}

footer{padding:16px 32px;border-top:1px solid var(--bdr);background:var(--surf);
  display:flex;justify-content:space-between;font-size:11px;color:var(--muted);font-family:var(--fm);}

/* Responsive */
@media(max-width:1024px){
  .main{grid-template-columns:1fr;}
  .sidebar{position:static;height:auto;border-right:none;border-bottom:1px solid var(--bdr);}
  .hero-dir{font-size:34px;}.oi-cards{grid-template-columns:1fr;}
  .strat-dir{grid-template-columns:1fr;}.rec-grid{grid-template-columns:1fr;}.strikes-wrap{grid-template-columns:1fr;}
}
@media(max-width:640px){
  header{padding:12px 16px;}.hero{padding:16px;flex-direction:column;}.hero-dir{font-size:28px;}
  .section{padding:16px;}.strat-metrics{grid-template-columns:1fr;}
  .kl-dist-row{grid-template-columns:1fr;}
  footer{flex-direction:column;gap:6px;}
}
"""


def generate_html(tech, oc, md, ts):
    cp     = tech["price"]     if tech else 0
    expiry = oc["expiry"]      if oc   else "N/A"
    atm    = oc["atm_strike"]  if oc   else (round(cp / 50) * 50 if cp else 0)
    pcr    = oc["pcr_oi"]      if oc   else 0
    mp     = oc["max_pain"]    if oc   else 0

    bias   = md["bias"]
    conf   = md["confidence"]
    bull   = md["bull"]
    bear   = md["bear"]
    diff   = md["diff"]

    bc     = "#00d4a0" if bias == "BULLISH" else ("#ff4560" if bias == "BEARISH" else "#f0b429")
    b_grad = ("linear-gradient(135deg,#00d4a0,#2ecc8a)" if bias == "BULLISH" else
              "linear-gradient(135deg,#ff4560,#cc1133)" if bias == "BEARISH" else
              "linear-gradient(135deg,#f0b429,#f7931e)")
    b_arrow  = "UP" if bias == "BULLISH" else ("DOWN" if bias == "BEARISH" else "SIDEWAYS")
    pcr_col  = "var(--bull)" if pcr > 1.2 else ("var(--bear)" if pcr < 0.7 else "var(--neut)")

    oi_html      = build_oi_html(oc)               if oc   else ""
    kl_html      = build_key_levels_html(tech, oc) if tech else ""
    strat_html   = build_strategies_html(md, tech, oc)
    strikes_html = build_strikes_html(oc)

    sig_card = (
        f"<div class=\"sb-sec\">"
        f"<div class=\"sb-lbl\">TODAY'S SIGNAL</div>"
        f"<div class=\"sig-card\" style=\"border-color:{bc}40;\">"
        f"<div class=\"sig-arrow\" style=\"background:{b_grad};-webkit-background-clip:text;"
        f"-webkit-text-fill-color:transparent;background-clip:text;\">{b_arrow}</div>"
        f"<div class=\"sig-bias\" style=\"color:{bc};\">{bias}</div>"
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
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500;600&family=Rajdhani:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>{CSS}</style>
</head>
<body>
<div class="app">

<header>
  <div class="logo">NIFTY<span>CRAFT</span></div>
  <div class="hdr-meta">
    <div class="live-dot"></div>
    <span>NSE Options Dashboard</span>
    <span style="color:var(--bdr2)">|</span>
    <span>{ts}</span>
  </div>
</header>

<div class="hero">
  <div>
    <div class="hero-dir" style="background:{b_grad};-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">{bias}</div>
    <div class="hero-sub">Market Direction &middot; {conf} Confidence</div>
    <span class="hero-conf" style="color:{bc};border-color:{bc}50;background:{bc}12;">
      Bull {bull} pt &nbsp;&middot;&nbsp; Bear {bear} pt &nbsp;&middot;&nbsp; Diff {diff:+d}
    </span>
  </div>
  <div class="hero-stats">
    <div class="hstat"><div class="hstat-lbl">NIFTY Spot</div><div class="hstat-val" style="color:#fff;">&#8377;{cp:,.2f}</div></div>
    <div class="hstat"><div class="hstat-lbl">ATM Strike</div><div class="hstat-val" style="color:var(--accent);">&#8377;{atm:,}</div></div>
    <div class="hstat"><div class="hstat-lbl">Expiry</div><div class="hstat-val" style="color:var(--neut);">{expiry}</div></div>
    <div class="hstat"><div class="hstat-lbl">PCR (OI)</div><div class="hstat-val" style="color:{pcr_col};">{pcr:.3f}</div></div>
    <div class="hstat"><div class="hstat-lbl">Max Pain</div><div class="hstat-val" style="color:var(--neut);">&#8377;{mp:,}</div></div>
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
      <button class="sb-btn" onclick="go('strat',this)">Recommendations</button>
      <button class="sb-btn" onclick="go('strat',this)">Bullish <span class="sb-badge" style="color:var(--bull);">4</span></button>
      <button class="sb-btn" onclick="go('strat',this)">Bearish <span class="sb-badge" style="color:var(--bear);">3</span></button>
      <button class="sb-btn" onclick="go('strat',this)">Neutral <span class="sb-badge" style="color:var(--neut);">4</span></button>
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
      <div style="background:rgba(255,183,77,.06);border:1px solid rgba(255,183,77,.2);border-left:3px solid #ffb74d;border-radius:10px;padding:16px 18px;font-size:13px;color:#ffb74d;line-height:1.8;">
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
function go(id,btn){{
  const el=document.getElementById(id);
  if(el) el.scrollIntoView({{behavior:"smooth",block:"start"}});
  document.querySelectorAll(".sb-btn").forEach(b=>b.classList.remove("active"));
  if(btn) btn.classList.add("active");
}}
</script>
</body>
</html>"""


# =================================================================
#  SECTION 8 -- MAIN RUNNER
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
        underlying = oc_analysis['underlying']
        atm_strike = oc_analysis['atm_strike']
        pcr_oi     = oc_analysis['pcr_oi']
        print(f"  OK  Spot={underlying:.2f}  ATM={atm_strike}  PCR={pcr_oi:.3f}")
    else:
        print("  WARNING  Option chain unavailable -- technical-only mode")

    print("\n[2/3] Fetching Technical Indicators...")
    tech = get_technical_data()

    print("\n[3/3] Scoring Market Direction...")
    md = compute_market_direction(tech, oc_analysis)
    bias = md['bias']
    conf = md['confidence']
    bull = md['bull']
    bear = md['bear']
    print(f"  OK  {bias} ({conf} confidence)  Bull={bull} Bear={bear}")

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
