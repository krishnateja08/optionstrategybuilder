#!/usr/bin/env python3
"""
Nifty 50 Options Strategy Dashboard — GitHub Pages Generator
Aurora Borealis Theme · v16 · Option Greeks panel · Dynamic Strike Dropdown
pip install curl_cffi pandas numpy yfinance pytz scipy
"""

import os, json, time, warnings, pytz
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from curl_cffi import requests as curl_requests
import yfinance as yf
from math import log, sqrt, exp
from scipy.stats import norm as _norm

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
                underlying  = json_data.get("records", {}).get("underlyingValue", 0)
                atm_strike  = round(underlying / 50) * 50
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
                        "CE_Gamma":     ce.get("gamma", 0),
                        "PE_Gamma":     pe.get("gamma", 0),
                        "CE_Vega":      ce.get("vega", 0),
                        "PE_Vega":      pe.get("vega", 0),
                    })
                df = pd.DataFrame(rows).sort_values("Strike").reset_index(drop=True)
                print(f"    OK {len(df)} strikes | Spot={underlying:.0f} ATM={atm_strike}")
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
        hist = yf.Ticker("^INDIAVIX").history(period="2d")
        if not hist.empty:
            last  = float(hist.iloc[-1]["Close"])
            prev  = float(hist.iloc[-2]["Close"]) if len(hist) > 1 else last
            chg   = round(last - prev, 2)
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
#  SECTION 1C -- BLACK-SCHOLES CALCULATOR (FIXED & ROBUST)
# =================================================================

def _bs_greeks(S, K, T, r, sigma, option_type="CE"):
    """
    Compute Black-Scholes Greeks.
    S     = spot price
    K     = strike price
    T     = time to expiry in years (e.g. 7/365)
    r     = risk-free rate (e.g. 0.065 for 6.5%)
    sigma = implied volatility as decimal (e.g. 0.15 for 15%)
    Returns dict with delta, gamma, theta (per day), vega (per 1% IV move).
    Always returns a valid dict — never None.
    """
    try:
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0}

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        nd1   = _norm.pdf(d1)
        gamma = nd1 / (S * sigma * np.sqrt(T))
        vega  = (S * nd1 * np.sqrt(T)) / 100   # per 1% IV change

        if option_type == "CE":
            delta         = _norm.cdf(d1)
            theta_annual  = (-(S * nd1 * sigma) / (2 * np.sqrt(T))
                             - r * K * np.exp(-r * T) * _norm.cdf(d2))
        else:  # PE
            delta         = _norm.cdf(d1) - 1
            theta_annual  = (-(S * nd1 * sigma) / (2 * np.sqrt(T))
                             + r * K * np.exp(-r * T) * _norm.cdf(-d2))

        return {
            "delta": round(float(delta), 4),
            "gamma": round(float(gamma), 6),
            "theta": round(float(theta_annual / 365.0), 4),   # daily theta
            "vega":  round(float(vega), 4),
        }
    except Exception:
        return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0}


def _days_to_expiry(expiry_str):
    """Parse NSE expiry string like '27-Feb-2026' → days remaining (min 1)."""
    try:
        ist_tz  = pytz.timezone("Asia/Kolkata")
        today   = datetime.now(ist_tz).date()
        exp_dt  = datetime.strptime(expiry_str, "%d-%b-%Y").date()
        days    = (exp_dt - today).days
        return max(days, 1)
    except Exception:
        return 7   # fallback: 1 week


def _compute_greeks_for_row(r, spot, expiry_str, risk_free=0.065, vix=18.0):
    """
    DEFINITIVE FIX: Always compute Greeks via Black-Scholes for every strike.
    NSE data is ONLY used as the IV input — never for Delta/Theta/Vega/Gamma directly.
    The old health-check approach was wrong: NSE returns non-zero delta/theta for ALL
    strikes (not just ATM), causing every row to return the same NSE ATM Greeks.
    BS with per-strike IV guarantees unique, mathematically correct values per strike.

    IV priority per side: NSE own IV > NSE opposite side IV > India VIX > 18%
    """
    T = _days_to_expiry(expiry_str) / 365.0
    K = float(r["Strike"])

    ce_iv_nse = float(r.get("CE_IV", 0) or 0)
    pe_iv_nse = float(r.get("PE_IV", 0) or 0)

    # CE IV to use for BS
    if ce_iv_nse > 0.5:
        ce_iv = ce_iv_nse
    elif pe_iv_nse > 0.5:
        ce_iv = pe_iv_nse
    else:
        ce_iv = vix

    # PE IV to use for BS
    if pe_iv_nse > 0.5:
        pe_iv = pe_iv_nse
    elif ce_iv_nse > 0.5:
        pe_iv = ce_iv_nse
    else:
        pe_iv = vix

    # Black-Scholes gives unique Greeks for every strike based on K vs spot
    ce_g = _bs_greeks(spot, K, T, risk_free, ce_iv / 100.0, "CE")
    pe_g = _bs_greeks(spot, K, T, risk_free, pe_iv / 100.0, "PE")

    return ce_g, pe_g


def extract_atm_greeks(df, atm_strike, underlying=None, expiry_str="", vix=18.0):
    """
    Extract Greeks for ALL strikes in df (for dropdown) plus ATM row.
    Greeks are taken from NSE if valid, otherwise computed via Black-Scholes.
    vix parameter is passed through to _compute_greeks_for_row.
    """
    spot = underlying or float(atm_strike)
    greeks_rows = []

    for _, r in df.iterrows():
        strike    = int(r["Strike"])
        is_atm    = strike == int(atm_strike)
        ce_iv_raw = round(float(r.get("CE_IV",  0) or 0), 2)
        pe_iv_raw = round(float(r.get("PE_IV",  0) or 0), 2)
        ce_ltp    = round(float(r.get("CE_LTP", 0) or 0), 2)
        pe_ltp    = round(float(r.get("PE_LTP", 0) or 0), 2)

        # FIXED: pass vix into the computation
        ce_g, pe_g = _compute_greeks_for_row(r, spot, expiry_str, vix=vix)

        greeks_rows.append({
            "strike":   strike,
            "is_atm":   is_atm,
            "ce_iv":    ce_iv_raw,
            "pe_iv":    pe_iv_raw,
            "ce_delta": ce_g["delta"],
            "pe_delta": pe_g["delta"],
            "ce_theta": ce_g["theta"],
            "pe_theta": pe_g["theta"],
            "ce_gamma": ce_g["gamma"],
            "pe_gamma": pe_g["gamma"],
            "ce_vega":  ce_g["vega"],
            "pe_vega":  pe_g["vega"],
            "ce_ltp":   ce_ltp,
            "pe_ltp":   pe_ltp,
        })

    greeks_rows.sort(key=lambda x: x["strike"])

    atm_row = next((g for g in greeks_rows if g["is_atm"]),
                   greeks_rows[len(greeks_rows)//2] if greeks_rows else {})

    atm_idx = next((i for i, g in enumerate(greeks_rows) if g["is_atm"]),
                   len(greeks_rows)//2)
    lo      = max(0, atm_idx - 2)
    hi      = min(len(greeks_rows), atm_idx + 3)
    table_5 = greeks_rows[lo:hi]

    return {
        "atm_greeks":   atm_row,
        "greeks_table": table_5,
        "all_strikes":  greeks_rows,
    }


# =================================================================
#  SECTION 2 -- OPTION CHAIN ANALYSIS
# =================================================================

def analyze_option_chain(oc_data, vix=18.0):
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

    raw_total = int(total_pe_oi) + int(total_ce_oi)
    raw_total = raw_total if raw_total > 0 else 1
    bull_pct  = round(int(total_pe_oi) / raw_total * 100)
    bear_pct  = 100 - bull_pct

    if   pcr_oi > 1.5:
        raw_oi_dir = "STRONG BULLISH"; raw_oi_sig = "Heavy Put Writing — Strong Support Floor"; raw_oi_cls = "bullish"
    elif pcr_oi > 1.2:
        raw_oi_dir = "BULLISH";        raw_oi_sig = "Put OI > Call OI — Bulls in Control";      raw_oi_cls = "bullish"
    elif pcr_oi < 0.5:
        raw_oi_dir = "STRONG BEARISH"; raw_oi_sig = "Heavy Call Writing — Strong Resistance Cap"; raw_oi_cls = "bearish"
    elif pcr_oi < 0.7:
        raw_oi_dir = "BEARISH";        raw_oi_sig = "Call OI > Put OI — Bears in Control";      raw_oi_cls = "bearish"
    else:
        if int(total_pe_oi) >= int(total_ce_oi):
            raw_oi_dir = "CAUTIOUSLY BULLISH"; raw_oi_sig = "Balanced OI — Slight Put Dominance"; raw_oi_cls = "bullish"
        else:
            raw_oi_dir = "CAUTIOUSLY BEARISH"; raw_oi_sig = "Balanced OI — Slight Call Dominance"; raw_oi_cls = "bearish"

    chg_bull_force = (abs(pe_chg) if pe_chg > 0 else 0) + (abs(ce_chg) if ce_chg < 0 else 0)
    chg_bear_force = (abs(ce_chg) if ce_chg > 0 else 0) + (abs(pe_chg) if pe_chg < 0 else 0)

    atm_strike = oc_data["atm_strike"]
    # FIXED: pass vix to extract_atm_greeks
    greeks = extract_atm_greeks(df, atm_strike,
                                underlying=oc_data["underlying"],
                                expiry_str=oc_data["expiry"],
                                vix=vix)

    return {
        "expiry":          oc_data["expiry"],
        "underlying":      oc_data["underlying"],
        "atm_strike":      atm_strike,
        "pcr_oi":          round(pcr_oi, 3),
        "pcr_vol":         round(pcr_vol, 3),
        "total_ce_oi":     int(total_ce_oi),
        "total_pe_oi":     int(total_pe_oi),
        "max_ce_strike":   int(max_ce_row["Strike"]),
        "max_ce_oi":       int(max_ce_row["CE_OI"]),
        "max_pe_strike":   int(max_pe_row["Strike"]),
        "max_pe_oi":       int(max_pe_row["PE_OI"]),
        "max_pain":        int(max_pain_row["Strike"]),
        "ce_chg":          ce_chg,
        "pe_chg":          pe_chg,
        "net_chg":         net_chg,
        "oi_dir":          oi_dir,
        "oi_sig":          oi_sig,
        "oi_icon":         oi_icon,
        "oi_cls":          oi_cls,
        "top_ce":          top_ce,
        "top_pe":          top_pe,
        "strikes_data":    strikes_data,
        "bull_pct":        bull_pct,
        "bear_pct":        bear_pct,
        "bull_force":      int(total_pe_oi),
        "bear_force":      int(total_ce_oi),
        "raw_oi_dir":      raw_oi_dir,
        "raw_oi_sig":      raw_oi_sig,
        "raw_oi_cls":      raw_oi_cls,
        "chg_bull_force":  chg_bull_force,
        "chg_bear_force":  chg_bear_force,
        "atm_greeks":      greeks["atm_greeks"],
        "greeks_table":    greeks["greeks_table"],
        "all_strikes":     greeks["all_strikes"],
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
#  SECTION 5 -- HELPERS
# =================================================================

def _cls_color(cls):
    return "#00c896" if cls == "bullish" else ("#ff6b6b" if cls == "bearish" else "#6480ff")

def _cls_bg(cls):
    return ("rgba(0,200,150,.08)"   if cls == "bullish" else
            "rgba(255,107,107,.08)" if cls == "bearish" else "rgba(100,128,255,.08)")

def _cls_bdr(cls):
    return ("rgba(0,200,150,.22)"   if cls == "bullish" else
            "rgba(255,107,107,.22)" if cls == "bearish" else "rgba(100,128,255,.22)")

def _fmt_oi(n):
    abs_n = abs(n); sign = "+" if n > 0 else ("-" if n < 0 else "")
    if abs_n >= 1_00_00_000:  return f"{sign}{abs_n / 1_00_00_000:.1f}Cr"
    if abs_n >= 1_00_000:     return f"{sign}{abs_n / 1_00_000:.0f}L"
    if abs_n >= 1_000:        return f"{sign}{abs_n / 1_000:.0f}K"
    return f"{n:+,}"

def _fmt_chg_oi(n):
    if abs(n) >= 1_000_000: return f"{'+' if n > 0 else ''}{n / 1_000_000:.1f}M"
    if abs(n) >= 1_000:     return f"{'+' if n > 0 else ''}{n / 1_000:.0f}K"
    return f"{n:+,}"


# =================================================================
#  SECTION 5A -- OPTION GREEKS PANEL (FULLY FIXED)
# =================================================================

def _delta_bar_html(delta_val, is_ce=True):
    pct = abs(delta_val) * 100
    col = "#00c896" if is_ce else "#ff6b6b"
    return (
        f'<div style="display:flex;align-items:center;gap:5px;">'
        f'<div style="width:34px;height:3px;background:rgba(255,255,255,.10);border-radius:2px;overflow:hidden;">'
        f'<div style="width:{pct:.0f}%;height:100%;background:{col};border-radius:2px;"></div></div>'
        f'<span style="font-family:\'DM Mono\',monospace;font-size:11px;font-weight:700;color:{col};">'
        f'{delta_val:+.3f}</span></div>'
    )


def build_greeks_sidebar_html(oc_analysis):
    """
    FIXED Option Greeks panel for the sidebar.
    - Title renamed to 'OPTION GREEKS'
    - Golden strike-price dropdown
    - Selecting any strike updates ALL greek values correctly
    - Per-strike BS-computed Greeks ensure unique values for every strike
    """
    if not oc_analysis:
        return ""

    g    = oc_analysis.get("atm_greeks", {})
    atm  = oc_analysis.get("atm_strike", 0)
    spot = oc_analysis.get("underlying", 0)
    exp  = oc_analysis.get("expiry", "N/A")
    all_rows = oc_analysis.get("all_strikes", oc_analysis.get("greeks_table", []))

    if not g:
        return ""

    ce_iv    = g.get("ce_iv",    15.0)
    pe_iv    = g.get("pe_iv",    15.0)
    ce_delta = g.get("ce_delta", 0.5)
    pe_delta = g.get("pe_delta", -0.5)
    ce_theta = g.get("ce_theta", 0.0)
    pe_theta = g.get("pe_theta", 0.0)
    ce_vega  = g.get("ce_vega",  0.0)
    pe_vega  = g.get("pe_vega",  0.0)
    ce_ltp   = g.get("ce_ltp",   0.0)
    pe_ltp   = g.get("pe_ltp",   0.0)

    iv_skew  = round(pe_iv - ce_iv, 2)
    skew_col = "#ff6b6b" if iv_skew > 1.5 else ("#00c896" if iv_skew < -1.5 else "#6480ff")
    skew_txt = f"PE Skew +{iv_skew:.1f}" if iv_skew > 0 else f"CE Skew {iv_skew:.1f}"

    iv_avg    = (ce_iv + pe_iv) / 2
    iv_pct    = min(100, max(0, (iv_avg / 60) * 100))
    iv_col    = "#ff6b6b" if iv_avg > 25 else "#ffd166" if iv_avg > 18 else "#00c896"
    iv_regime = "High IV · Buy Premium" if iv_avg > 25 else "Normal IV · Balanced" if iv_avg > 15 else "Low IV · Sell Premium"

    def tfmt(t): return f"&#8377;{abs(t):.2f}" if abs(t) >= 0.01 else f"{t:.4f}"
    def vfmt(v): return f"{v:.4f}" if abs(v) >= 0.0001 else "&#8212;"

    # Build dropdown — option values are always clean integers
    otm_ce_opts = ""
    atm_opt     = ""
    otm_pe_opts = ""
    for row in all_rows:
        s      = int(row["strike"])   # force int
        is_atm = row["is_atm"]
        dist   = abs(s - atm) // 50
        if is_atm:
            label   = f"\u2605  ATM  \u20b9{s:,}"
            atm_opt = f'<option value="{s}" selected>{label}</option>\n'
        elif s > atm:
            label       = f"\u25b2  CE+{dist}  \u20b9{s:,}"
            otm_ce_opts += f'<option value="{s}">{label}</option>\n'
        else:
            label       = f"\u25bc  PE-{dist}  \u20b9{s:,}"
            otm_pe_opts += f'<option value="{s}">{label}</option>\n'

    dropdown_options = (
        f'<optgroup label="\u2500 OTM CALLS (CE) \u2500">{otm_ce_opts}</optgroup>'
        f'<optgroup label="\u2500 ATM \u2500">{atm_opt}</optgroup>'
        f'<optgroup label="\u2500 OTM PUTS (PE) \u2500">{otm_pe_opts}</optgroup>'
    )

    # Build per-strike JSON — keys are always clean integers, no floats
    strikes_json_parts = []
    for row in all_rows:
        s = int(row["strike"])   # force int — prevents "25900.0" float keys
        strikes_json_parts.append(
            f'"{s}":{{' +
            f'"ce_ltp":{round(float(row["ce_ltp"]),2)},' +
            f'"pe_ltp":{round(float(row["pe_ltp"]),2)},' +
            f'"ce_delta":{round(float(row["ce_delta"]),4)},' +
            f'"pe_delta":{round(float(row["pe_delta"]),4)},' +
            f'"ce_iv":{round(float(row["ce_iv"]),2)},' +
            f'"pe_iv":{round(float(row["pe_iv"]),2)},' +
            f'"ce_theta":{round(float(row["ce_theta"]),4)},' +
            f'"pe_theta":{round(float(row["pe_theta"]),4)},' +
            f'"ce_vega":{round(float(row["ce_vega"]),4)},' +
            f'"pe_vega":{round(float(row["pe_vega"]),4)}' +
            f'}}'
        )
    strikes_json = "{" + ",".join(strikes_json_parts) + "}"

    return f"""
<div class="greeks-panel" id="greeksPanel">
  <div class="greeks-title">
    &#9652; OPTION GREEKS
    <span class="greeks-expiry-tag">{exp}</span>
  </div>

  <!-- GOLDEN STRIKE DROPDOWN -->
  <div class="greeks-strike-wrap">
    <select class="greeks-strike-select" id="greeksStrikeSelect"
            onchange="greeksUpdateStrike(this.value)">
      {dropdown_options}
    </select>
  </div>

  <!-- Strike badge — type label (ATM / CE+N / PE-N) + selected strike + LTPs -->
  <div class="greeks-atm-badge" id="greeksAtmBadge">
    <span style="font-size:8.5px;font-weight:700;color:rgba(138,160,255,.9);" id="greeksStrikeTypeLabel">ATM</span>
    <span class="greeks-atm-strike" id="greeksStrikeLabel">&#8377;{atm:,}</span>
    <span style="font-size:8px;color:rgba(255,255,255,.2);">|</span>
    <span style="font-size:8.5px;color:rgba(0,200,220,.8);" id="greeksCeLtp">CE &#8377;{ce_ltp:.1f}</span>
    <span style="font-size:8px;color:rgba(255,255,255,.25);">/</span>
    <span style="font-size:8.5px;color:rgba(255,107,107,.8);" id="greeksPeLtp">PE &#8377;{pe_ltp:.1f}</span>
  </div>

  <!-- Δ Delta -->
  <div class="greeks-row">
    <div style="display:flex;flex-direction:column;">
      <span class="greek-name">&#916; Delta</span>
      <span class="greek-sub">CE / PE</span>
    </div>
    <div style="display:flex;flex-direction:column;gap:3px;align-items:flex-end;" id="greeksDeltaWrap">
      {_delta_bar_html(ce_delta, True)}
      {_delta_bar_html(pe_delta, False)}
    </div>
  </div>

  <!-- σ IV -->
  <div class="greeks-row">
    <div style="display:flex;flex-direction:column;">
      <span class="greek-name">&#963; IV</span>
      <span class="greek-sub" id="greeksSkewLbl" style="color:{skew_col};font-weight:700;">{skew_txt}</span>
    </div>
    <div style="display:flex;flex-direction:column;align-items:flex-end;gap:3px;">
      <div style="display:flex;align-items:center;gap:6px;">
        <span style="font-size:8.5px;color:rgba(0,200,220,.85);">CE</span>
        <span style="font-family:'DM Mono',monospace;font-size:13px;font-weight:700;color:#00c8e0;" id="greeksIvCe">{ce_iv:.1f}%</span>
      </div>
      <div style="display:flex;align-items:center;gap:6px;">
        <span style="font-size:8.5px;color:rgba(255,144,144,.85);">PE</span>
        <span style="font-family:'DM Mono',monospace;font-size:13px;font-weight:700;color:#ff9090;" id="greeksIvPe">{pe_iv:.1f}%</span>
      </div>
    </div>
  </div>

  <!-- Θ Theta -->
  <div class="greeks-row">
    <div style="display:flex;flex-direction:column;">
      <span class="greek-name">&#920; Theta</span>
      <span class="greek-sub">per day</span>
    </div>
    <div style="display:flex;flex-direction:column;align-items:flex-end;gap:3px;">
      <div style="display:flex;align-items:center;gap:6px;">
        <span style="font-size:8.5px;color:rgba(0,200,220,.85);">CE</span>
        <span style="font-family:'DM Mono',monospace;font-size:11px;font-weight:700;color:#ff9090;" id="greeksThetaCe">{tfmt(ce_theta)}</span>
      </div>
      <div style="display:flex;align-items:center;gap:6px;">
        <span style="font-size:8.5px;color:rgba(255,144,144,.85);">PE</span>
        <span style="font-family:'DM Mono',monospace;font-size:11px;font-weight:700;color:#ff9090;" id="greeksThetaPe">{tfmt(pe_theta)}</span>
      </div>
    </div>
  </div>

  <!-- ν Vega -->
  <div class="greeks-row">
    <div style="display:flex;flex-direction:column;">
      <span class="greek-name">&#957; Vega</span>
      <span class="greek-sub">per 1% IV</span>
    </div>
    <div style="display:flex;flex-direction:column;align-items:flex-end;gap:3px;">
      <div style="display:flex;align-items:center;gap:6px;">
        <span style="font-size:8.5px;color:rgba(0,200,220,.85);">CE</span>
        <span style="font-family:'DM Mono',monospace;font-size:11px;font-weight:700;color:#8aa0ff;" id="greeksVegaCe">{vfmt(ce_vega)}</span>
      </div>
      <div style="display:flex;align-items:center;gap:6px;">
        <span style="font-size:8.5px;color:rgba(255,144,144,.85);">PE</span>
        <span style="font-family:'DM Mono',monospace;font-size:11px;font-weight:700;color:#8aa0ff;" id="greeksVegaPe">{vfmt(pe_vega)}</span>
      </div>
    </div>
  </div>

  <!-- IV regime bar -->
  <div class="iv-bar-wrap">
    <span class="iv-bar-label">IV Avg</span>
    <div class="iv-bar-track">
      <div class="iv-bar-fill" id="greeksIvBar"
           style="width:{iv_pct:.1f}%;background:{iv_col};box-shadow:0 0 6px {iv_col}88;"></div>
    </div>
    <span class="iv-bar-num" id="greeksIvAvg" style="color:{iv_col};">{iv_avg:.1f}%</span>
  </div>
  <div style="font-size:8.5px;text-align:center;margin-top:6px;font-weight:700;letter-spacing:.5px;color:{iv_col};"
       id="greeksIvRegime">{iv_regime}</div>
</div>

"""


def build_greeks_table_html(oc_analysis):
    """Full 5-strike CE/PE Greeks table (ATM ±2) shown below OI section."""
    if not oc_analysis:
        return ""

    rows = oc_analysis.get("greeks_table", [])
    atm  = oc_analysis.get("atm_strike", 0)
    exp  = oc_analysis.get("expiry", "N/A")
    spot = oc_analysis.get("underlying", 0)

    if not rows:
        return ""

    def atm_tag_html(is_atm):
        if not is_atm: return ""
        return '<span style="font-size:8px;background:rgba(100,128,255,.25);color:#8aa0ff;padding:1px 5px;border-radius:4px;margin-left:4px;font-weight:700;">ATM</span>'

    def delta_iv_row(g):
        is_atm   = g["is_atm"]
        row_cls  = 'style="background:rgba(100,128,255,.06);border-left:3px solid rgba(100,128,255,.45);"' if is_atm else ""
        sc       = 'style="color:#8aa0ff;"' if is_atm else ""
        ce_d_col = "#00c896" if g["ce_delta"] >= 0.5 else "#ffd166" if g["ce_delta"] >= 0.35 else "#ff9090"
        pe_d_col = "#ff6b6b" if g["pe_delta"] <= -0.5 else "#ffd166" if g["pe_delta"] <= -0.35 else "#4de8b8"
        ce_iv_c  = "#ff6b6b" if g["ce_iv"] > 25 else "#ffd166" if g["ce_iv"] > 18 else "#00c896"
        pe_iv_c  = "#ff6b6b" if g["pe_iv"] > 25 else "#ffd166" if g["pe_iv"] > 18 else "#00c896"
        return (
            f'<div class="greeks-tbl-row" {row_cls}>'
            f'<div class="greeks-tbl-strike" {sc}>&#8377;{g["strike"]:,}{atm_tag_html(is_atm)}</div>'
            f'<div class="greeks-tbl-cell" style="color:{ce_d_col};">{g["ce_delta"]:+.3f}</div>'
            f'<div class="greeks-tbl-cell" style="color:{pe_d_col};">{g["pe_delta"]:+.3f}</div>'
            f'<div class="greeks-tbl-cell" style="color:{ce_iv_c};">{g["ce_iv"]:.1f}%</div>'
            f'<div class="greeks-tbl-cell" style="color:{pe_iv_c};">{g["pe_iv"]:.1f}%</div>'
            f'</div>'
        )

    def theta_vega_row(g):
        is_atm   = g["is_atm"]
        row_cls  = 'style="background:rgba(100,128,255,.06);border-left:3px solid rgba(100,128,255,.45);"' if is_atm else ""
        sc       = 'style="color:#8aa0ff;"' if is_atm else ""
        tfmt = lambda t: f"&#8377;{abs(t):.2f}" if abs(t) >= 0.01 else f"{t:.4f}"
        vfmt = lambda v: f"{v:.4f}" if abs(v) >= 0.0001 else "&#8212;"
        return (
            f'<div class="greeks-tbl-row" {row_cls}>'
            f'<div class="greeks-tbl-strike" {sc}>&#8377;{g["strike"]:,}{atm_tag_html(is_atm)}</div>'
            f'<div class="greeks-tbl-cell" style="color:#ff9090;">{tfmt(g["ce_theta"])}</div>'
            f'<div class="greeks-tbl-cell" style="color:#ff9090;">{tfmt(g["pe_theta"])}</div>'
            f'<div class="greeks-tbl-cell" style="color:#8aa0ff;">{vfmt(g["ce_vega"])}</div>'
            f'<div class="greeks-tbl-cell" style="color:#8aa0ff;">{vfmt(g["pe_vega"])}</div>'
            f'</div>'
        )

    dv_rows = "".join(delta_iv_row(g) for g in rows)
    tv_rows = "".join(theta_vega_row(g) for g in rows)

    return f"""
<div class="greeks-table-section" id="greeksTable">
  <div class="sec-title" style="color:#8aa0ff;border-color:rgba(100,128,255,.18);">
    OPTION GREEKS &nbsp;&middot;&nbsp; ATM &#177;2 STRIKES
    <span class="sec-sub">Live from NSE &middot; {exp} &middot; Spot &#8377;{spot:,.0f}</span>
  </div>

  <div class="greeks-table-wrap">

    <!-- Δ Delta + σ IV -->
    <div>
      <div style="font-size:9.5px;font-weight:700;color:rgba(100,128,255,.75);
        margin-bottom:10px;letter-spacing:1.5px;text-transform:uppercase;
        display:flex;align-items:center;gap:8px;">
        <span style="width:3px;height:14px;background:#6480ff;border-radius:2px;display:inline-block;"></span>
        &#916; DELTA &amp; &#963; IMPLIED VOLATILITY
      </div>
      <div class="greeks-tbl">
        <div class="greeks-tbl-head">
          <div class="greeks-tbl-head-label" style="text-align:left;">Strike</div>
          <div class="greeks-tbl-head-label" style="color:rgba(0,200,150,.65);">CE &#916;</div>
          <div class="greeks-tbl-head-label" style="color:rgba(255,107,107,.65);">PE &#916;</div>
          <div class="greeks-tbl-head-label" style="color:rgba(0,200,220,.65);">CE IV</div>
          <div class="greeks-tbl-head-label" style="color:rgba(255,144,144,.65);">PE IV</div>
        </div>
        {dv_rows}
      </div>
      <div style="font-size:9px;color:rgba(255,255,255,.25);margin-top:7px;padding:0 4px;line-height:1.6;">
        &#9432; CE &#916;&ge;0.5 = deep ITM &middot; CE &#916;&asymp;0.5 = ATM &middot; CE &#916;&le;0.3 = OTM
      </div>
    </div>

    <!-- Θ Theta + ν Vega -->
    <div>
      <div style="font-size:9.5px;font-weight:700;color:rgba(255,107,107,.75);
        margin-bottom:10px;letter-spacing:1.5px;text-transform:uppercase;
        display:flex;align-items:center;gap:8px;">
        <span style="width:3px;height:14px;background:#ff6b6b;border-radius:2px;display:inline-block;"></span>
        &#920; THETA (Day Decay) &amp; &#957; VEGA
      </div>
      <div class="greeks-tbl">
        <div class="greeks-tbl-head">
          <div class="greeks-tbl-head-label" style="text-align:left;">Strike</div>
          <div class="greeks-tbl-head-label" style="color:rgba(0,200,220,.65);">CE &#920;</div>
          <div class="greeks-tbl-head-label" style="color:rgba(255,144,144,.65);">PE &#920;</div>
          <div class="greeks-tbl-head-label" style="color:rgba(138,160,255,.65);">CE &#957;</div>
          <div class="greeks-tbl-head-label" style="color:rgba(138,160,255,.65);">PE &#957;</div>
        </div>
        {tv_rows}
      </div>
      <div style="font-size:9px;color:rgba(255,255,255,.25);margin-top:7px;padding:0 4px;line-height:1.6;">
        &#9432; Theta = &#8377; lost per day &middot; Vega = &#8377; change per 1% IV move
      </div>
    </div>

  </div>
</div>
"""


# =================================================================
#  SECTION 5B -- HERO
# =================================================================

def build_dual_gauge_hero(oc, tech, md, ts):
    if oc:
        total_pe_oi = oc["total_pe_oi"]; total_ce_oi = oc["total_ce_oi"]
        bull_pct = oc["bull_pct"]; bear_pct = oc["bear_pct"]; pcr = oc["pcr_oi"]
        oi_dir = oc["raw_oi_dir"]; oi_sig = oc["raw_oi_sig"]; oi_cls = oc["raw_oi_cls"]
        bull_label = _fmt_oi(total_pe_oi); bear_label = _fmt_oi(total_ce_oi)
        expiry = oc["expiry"]; underlying = oc["underlying"]; atm = oc["atm_strike"]; max_pain = oc["max_pain"]
    else:
        total_pe_oi = total_ce_oi = 0; bull_pct = bear_pct = 50; pcr = 1.0
        oi_sig = "No Data"; oi_dir = "UNKNOWN"; oi_cls = "neutral"
        expiry = "N/A"; underlying = 0; atm = 0; max_pain = 0
        bull_label = "N/A"; bear_label = "N/A"

    cp = tech["price"] if tech else 0
    bias = md["bias"]; conf = md["confidence"]; bull_sc = md["bull"]; bear_sc = md["bear"]; diff = md["diff"]
    dir_col = _cls_color(oi_cls)
    pcr_col = "#00c896" if pcr > 1.2 else ("#ff6b6b" if pcr < 0.7 else "#6480ff")
    b_col = _cls_color(md.get("bias_cls", "neutral"))
    b_bg = _cls_bg(md.get("bias_cls", "neutral")); b_bdr = _cls_bdr(md.get("bias_cls", "neutral"))
    C = 194.8
    def clamp(v, lo=10, hi=97): return max(lo, min(hi, v))
    bull_offset = C * (1 - clamp(bull_pct) / 100); bear_offset = C * (1 - clamp(bear_pct) / 100)
    oi_bar_w = clamp(bull_pct); bear_bar_w = clamp(bear_pct)
    b_arrow = "▲" if bias == "BULLISH" else ("▼" if bias == "BEARISH" else "◆")
    glow_rgb = ("0,200,150" if dir_col == "#00c896" else "255,107,107" if dir_col == "#ff6b6b" else "100,128,255")

    return f"""
<div class="hero" id="heroWidget">
  <div class="h-gauges">
    <div class="gauge-wrap">
      <svg width="76" height="76" viewBox="0 0 76 76">
        <circle cx="38" cy="38" r="31" fill="none" stroke="rgba(255,255,255,.18)" stroke-width="6"/>
        <circle cx="38" cy="38" r="31" fill="none" stroke="url(#bull-g)" stroke-width="6"
          stroke-linecap="round" stroke-dasharray="{C}" stroke-dashoffset="{bull_offset:.1f}"
          style="transform:rotate(-90deg);transform-origin:38px 38px;transition:stroke-dashoffset 1s ease;"/>
        <defs><linearGradient id="bull-g" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stop-color="#00c896"/><stop offset="100%" stop-color="#4de8b8"/>
        </linearGradient></defs>
      </svg>
      <div class="gauge-inner">
        <div class="g-val" style="color:#00c896;">{bull_label}</div>
        <div class="g-lbl">OI BULL</div>
      </div>
    </div>
    <div class="gauge-sep"></div>
    <div class="gauge-wrap">
      <svg width="76" height="76" viewBox="0 0 76 76">
        <circle cx="38" cy="38" r="31" fill="none" stroke="rgba(255,255,255,.18)" stroke-width="6"/>
        <circle cx="38" cy="38" r="31" fill="none" stroke="url(#bear-g)" stroke-width="6"
          stroke-linecap="round" stroke-dasharray="{C}" stroke-dashoffset="{bear_offset:.1f}"
          style="transform:rotate(-90deg);transform-origin:38px 38px;transition:stroke-dashoffset 1s ease;"/>
        <defs><linearGradient id="bear-g" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stop-color="#ff6b6b"/><stop offset="100%" stop-color="#ff9090"/>
        </linearGradient></defs>
      </svg>
      <div class="gauge-inner">
        <div class="g-val" style="color:#ff6b6b;">{bear_label}</div>
        <div class="g-lbl">OI BEAR</div>
      </div>
    </div>
  </div>
  <div class="h-mid">
    <div class="h-eyebrow">OI NET SIGNAL · {expiry} · SPOT ₹{underlying:,.0f}</div>
    <div class="h-signal" style="color:{dir_col};text-shadow:0 0 20px rgba({glow_rgb},.6),0 0 40px rgba({glow_rgb},.3);font-size:22px;font-weight:900;letter-spacing:1px;">{oi_dir}</div>
    <div class="h-sub">{oi_sig} · PCR <span style="color:{pcr_col};font-weight:700;">{pcr:.3f}</span></div>
    <div class="h-divider"></div>
    <div class="pill-row">
      <div class="pill-dot" style="background:#00c896;box-shadow:0 0 5px rgba(0,200,150,.5);"></div>
      <div class="pill-lbl">BULL STRENGTH</div>
      <div class="pill-track"><div class="pill-fill" style="width:{oi_bar_w}%;background:linear-gradient(90deg,#00c896,#4de8b8);"></div></div>
      <div class="pill-num" style="color:#00c896;">{bull_pct}%</div>
    </div>
    <div class="pill-row">
      <div class="pill-dot" style="background:#ff6b6b;box-shadow:0 0 5px rgba(255,107,107,.4);"></div>
      <div class="pill-lbl">BEAR STRENGTH</div>
      <div class="pill-track"><div class="pill-fill" style="width:{bear_bar_w}%;background:linear-gradient(90deg,#ff6b6b,#ff9090);"></div></div>
      <div class="pill-num" style="color:#ff6b6b;">{bear_pct}%</div>
    </div>
  </div>
  <div class="h-stats">
    <div class="h-stat-row">
      <div class="h-stat"><div class="h-stat-lbl">NIFTY SPOT</div><div class="h-stat-val" style="color:rgba(255,255,255,.85);">&#8377;{cp:,.2f}</div></div>
      <div class="h-stat"><div class="h-stat-lbl">ATM STRIKE</div><div class="h-stat-val" style="color:#00c896;">&#8377;{atm:,}</div></div>
      <div class="h-stat"><div class="h-stat-lbl">EXPIRY</div><div class="h-stat-val" style="color:#00c8e0;">{expiry}</div></div>
      <div class="h-stat"><div class="h-stat-lbl">PCR (OI)</div><div class="h-stat-val" style="color:{pcr_col};">{pcr:.3f}</div></div>
      <div class="h-stat"><div class="h-stat-lbl">MAX PAIN</div><div class="h-stat-val" style="color:#ffd166;">&#8377;{max_pain:,}</div></div>
    </div>
    <div class="h-stat-bottom">
      <div class="h-bias-row">
        <span class="h-chip" style="background:{b_bg};color:{b_col};border:1px solid {b_bdr};">{b_arrow}&nbsp;{bias}</span>
        <span class="h-chip" style="background:rgba(255,209,102,.1);color:#ffd166;border:1px solid rgba(255,209,102,.3);">{conf}&nbsp;CONF</span>
        <span class="h-score">Bull&nbsp;{bull_sc} · Bear&nbsp;{bear_sc} · Diff&nbsp;{diff:+d}</span>
      </div>
      <div class="h-ts" id="lastUpdatedTs">{ts}</div>
    </div>
  </div>
</div>
"""


# =================================================================
#  SECTION 5C -- OI DASHBOARD
# =================================================================

def build_oi_html(oc):
    ce  = oc["ce_chg"]; pe = oc["pe_chg"]; net = oc["net_chg"]
    expiry = oc["expiry"]; oi_dir = oc["oi_dir"]; oi_sig = oc["oi_sig"]; oi_cls = oc["oi_cls"]
    pcr = oc["pcr_oi"]; total_ce = oc["total_ce_oi"]; total_pe = oc["total_pe_oi"]
    max_ce_s = oc["max_ce_strike"]; max_pe_s = oc["max_pe_strike"]; max_pain = oc["max_pain"]
    underlying = oc["underlying"]

    dir_col = _cls_color(oi_cls); dir_bg = _cls_bg(oi_cls); dir_bdr = _cls_bdr(oi_cls)
    pcr_col = "#00c896" if pcr > 1.2 else ("#ff6b6b" if pcr < 0.7 else "#6480ff")

    ce_col = "#00c896" if ce < 0 else "#ff6b6b"
    ce_label = "Call Unwinding ↓ (Bullish)" if ce < 0 else "Call Build-up ↑ (Bearish)"
    ce_fmt = _fmt_chg_oi(ce)
    pe_col = "#00c896" if pe > 0 else "#ff6b6b"
    pe_label = "Put Build-up ↑ (Bullish)" if pe > 0 else "Put Unwinding ↓ (Bearish)"
    pe_fmt = _fmt_chg_oi(pe)
    bull_force_pre = (abs(pe) if pe > 0 else 0) + (abs(ce) if ce < 0 else 0)
    bear_force_pre = (abs(ce) if ce > 0 else 0) + (abs(pe) if pe < 0 else 0)
    net_is_bullish = bull_force_pre >= bear_force_pre
    net_col = "#00c896" if net_is_bullish else "#ff6b6b"
    net_label = "Net Bullish Flow" if net_is_bullish else "Net Bearish Flow"
    net_fmt = _fmt_chg_oi(bull_force_pre) if net_is_bullish else _fmt_chg_oi(-bear_force_pre)
    total_abs = abs(ce) + abs(pe) or 1
    ce_pct = round(abs(ce) / total_abs * 100); ce_bullish = ce < 0
    ce_pct_display = f"+{ce_pct}%" if ce_bullish else f"−{ce_pct}%"
    ce_bar_col = "#00c896" if ce_bullish else "#ff6b6b"
    pe_pct = round(abs(pe) / total_abs * 100); pe_bullish = pe > 0
    pe_pct_display = f"+{pe_pct}%" if pe_bullish else f"−{pe_pct}%"
    pe_bar_col = "#00c896" if pe_bullish else "#ff6b6b"
    total_f = bull_force_pre + bear_force_pre or 1
    bull_pct = round(bull_force_pre / total_f * 100); bear_pct = 100 - bull_pct
    net_pct = bull_pct if net_is_bullish else bear_pct
    net_bar_col = "#00c896" if net_is_bullish else "#ff6b6b"
    net_pct_display = f"+{net_pct}%" if net_is_bullish else f"−{net_pct}%"

    dir_card = f"""
<div style="display:flex;align-items:stretch;border:1px solid {dir_bdr};border-radius:14px;
  background:{dir_bg};overflow:hidden;margin-bottom:16px;">
  <div style="padding:18px 24px;min-width:200px;border-right:1px solid rgba(255,255,255,.07);
    display:flex;flex-direction:column;justify-content:center;flex-shrink:0;">
    <div style="font-size:8.5px;letter-spacing:2px;text-transform:uppercase;color:rgba(255,255,255,.28);margin-bottom:7px;">OI CHANGE DIRECTION</div>
    <div style="font-size:21px;font-weight:700;color:{dir_col};line-height:1.1;margin-bottom:5px;">{oi_dir}</div>
    <div style="font-size:10.5px;color:{dir_col};opacity:.7;">{oi_sig}</div>
    <div style="margin-top:10px;font-family:'DM Mono',monospace;font-size:10px;color:rgba(255,255,255,.3);">PCR &nbsp;<span style="color:{pcr_col};font-weight:700;">{pcr:.3f}</span></div>
  </div>
  <div style="display:flex;flex:1;align-items:stretch;">
    <div style="flex:1;display:flex;flex-direction:column;justify-content:center;padding:16px 20px;border-right:1px solid rgba(255,255,255,.05);gap:5px;">
      <div style="font-size:8.5px;letter-spacing:1.8px;text-transform:uppercase;color:rgba(255,255,255,.28);white-space:nowrap;">CE OI Change</div>
      <div style="font-family:'DM Mono',monospace;font-size:22px;font-weight:700;color:{ce_col};line-height:1;">{ce_fmt}</div>
      <div style="font-size:10px;color:rgba(255,255,255,.3);white-space:nowrap;">{ce_label}</div>
      <div style="display:flex;align-items:center;gap:8px;margin-top:3px;">
        <div style="flex:1;height:5px;background:rgba(255,255,255,.07);border-radius:3px;overflow:hidden;"><div style="width:{ce_pct}%;height:100%;border-radius:3px;background:{ce_bar_col};"></div></div>
        <div style="font-family:'DM Mono',monospace;font-size:10px;font-weight:700;color:{ce_bar_col};min-width:38px;text-align:right;">{ce_pct_display}</div>
      </div>
    </div>
    <div style="flex:1;display:flex;flex-direction:column;justify-content:center;padding:16px 20px;border-right:1px solid rgba(255,255,255,.05);gap:5px;">
      <div style="font-size:8.5px;letter-spacing:1.8px;text-transform:uppercase;color:rgba(255,255,255,.28);white-space:nowrap;">PE OI Change</div>
      <div style="font-family:'DM Mono',monospace;font-size:22px;font-weight:700;color:{pe_col};line-height:1;">{pe_fmt}</div>
      <div style="font-size:10px;color:rgba(255,255,255,.3);white-space:nowrap;">{pe_label}</div>
      <div style="display:flex;align-items:center;gap:8px;margin-top:3px;">
        <div style="flex:1;height:5px;background:rgba(255,255,255,.07);border-radius:3px;overflow:hidden;"><div style="width:{pe_pct}%;height:100%;border-radius:3px;background:{pe_bar_col};"></div></div>
        <div style="font-family:'DM Mono',monospace;font-size:10px;font-weight:700;color:{pe_bar_col};min-width:38px;text-align:right;">{pe_pct_display}</div>
      </div>
    </div>
    <div style="flex:1;display:flex;flex-direction:column;justify-content:center;padding:16px 20px;gap:5px;">
      <div style="font-size:8.5px;letter-spacing:1.8px;text-transform:uppercase;color:rgba(255,255,255,.28);white-space:nowrap;">Net OI Change</div>
      <div style="font-family:'DM Mono',monospace;font-size:22px;font-weight:700;color:{net_col};line-height:1;">{net_fmt}</div>
      <div style="font-size:10px;color:rgba(255,255,255,.3);white-space:nowrap;">{net_label}</div>
      <div style="display:flex;align-items:center;gap:8px;margin-top:3px;">
        <div style="flex:1;height:5px;background:rgba(255,255,255,.07);border-radius:3px;overflow:hidden;"><div style="width:{net_pct}%;height:100%;border-radius:3px;background:{net_bar_col};box-shadow:0 0 8px {net_bar_col}66;"></div></div>
        <div style="font-family:'DM Mono',monospace;font-size:10px;font-weight:700;color:{net_bar_col};min-width:38px;text-align:right;">{net_pct_display}</div>
      </div>
    </div>
  </div>
</div>"""

    snapshot_table = (
        f'<div class="oi-ticker-table">'
        f'<div class="oi-ticker-hdr" style="background:rgba(100,128,255,.05);border-bottom:1px solid rgba(100,128,255,.1);">'
        f'<div class="oi-ticker-hdr-label" style="color:rgba(100,128,255,.8);">&#9632; OI SNAPSHOT</div>'
        f'<div class="oi-ticker-hdr-cell">Total CE OI</div><div class="oi-ticker-hdr-cell">Total PE OI</div>'
        f'<div class="oi-ticker-hdr-cell">PCR (OI)</div><div class="oi-ticker-hdr-cell">Max CE</div>'
        f'<div class="oi-ticker-hdr-cell">Max PE</div></div>'
        f'<div class="oi-ticker-row">'
        f'<div class="oi-ticker-metric">Snapshot</div>'
        f'<div class="oi-ticker-cell" style="color:#ff6b6b;font-family:\'DM Mono\',monospace;font-weight:700;font-size:15px;">{total_ce:,}</div>'
        f'<div class="oi-ticker-cell" style="color:#00c896;font-family:\'DM Mono\',monospace;font-weight:700;font-size:15px;">{total_pe:,}</div>'
        f'<div class="oi-ticker-cell" style="color:{pcr_col};font-family:\'DM Mono\',monospace;font-weight:700;font-size:15px;">{pcr:.3f}</div>'
        f'<div class="oi-ticker-cell" style="color:#ff6b6b;font-family:\'DM Mono\',monospace;font-weight:700;font-size:15px;">&#8377;{max_ce_s:,}</div>'
        f'<div class="oi-ticker-cell" style="color:#00c896;font-family:\'DM Mono\',monospace;font-weight:700;font-size:15px;">&#8377;{max_pe_s:,}</div>'
        f'</div>'
        f'<div style="display:flex;align-items:center;justify-content:space-between;padding:10px 18px;border-top:1px solid rgba(255,255,255,.04);flex-wrap:wrap;gap:10px;">'
        f'<div style="display:flex;align-items:center;gap:10px;">'
        f'<span style="font-size:9px;letter-spacing:1.5px;text-transform:uppercase;color:rgba(255,255,255,.3);">MAX PAIN</span>'
        f'<span style="font-family:\'DM Mono\',monospace;font-size:18px;font-weight:700;color:#6480ff;">&#8377;{max_pain:,}</span>'
        f'<span style="font-size:10px;color:rgba(100,128,255,.6);">Option writers\' target</span></div>'
        f'</div></div>'
    )

    return (
        f'<div class="section"><div class="sec-title">OPEN INTEREST DASHBOARD'
        f'<span class="sec-sub">Spot ±500 pts · Expiry: {expiry} · Spot: &#8377;{underlying:,.2f}</span></div>'
        + dir_card + snapshot_table + "</div>"
    )


def build_key_levels_html(tech, oc):
    cp = tech["price"]; ss = tech["strong_sup"]; s1 = tech["support"]
    r1 = tech["resistance"]; sr = tech["strong_res"]; rng = sr - ss or 1
    def pct(v): return round(max(3, min(97, (v - ss) / rng * 100)), 1)
    cp_pct = pct(cp); pts_r = int(r1 - cp); pts_s = int(cp - s1)
    mp_html = ""
    if oc:
        mp_p = pct(oc["max_pain"]); mp = oc["max_pain"]
        mp_html = (f'<div class="kl-node" style="left:{mp_p}%;top:0;transform:translateX(-50%);">'
                   f'<div class="kl-dot" style="background:#6480ff;box-shadow:0 0 8px rgba(100,128,255,.5);margin:0 auto 4px;"></div>'
                   f'<div class="kl-lbl" style="color:#6480ff;">Max Pain</div>'
                   f'<div class="kl-val" style="color:#8aa0ff;">&#8377;{mp:,}</div></div>')
    return (
        f'<div class="section"><div class="sec-title">KEY LEVELS'
        f'<span class="sec-sub">1H Candles · Last 120 bars · Rounded to 25</span></div>'
        f'<div class="kl-zone-labels"><span style="color:#00c896;">SUPPORT ZONE</span><span style="color:#ff6b6b;">RESISTANCE ZONE</span></div>'
        f'<div style="position:relative;height:58px;">'
        f'<div class="kl-node" style="left:3%;bottom:0;transform:translateX(-50%);"><div class="kl-lbl" style="color:#00a07a;">Strong Sup</div><div class="kl-val" style="color:#00c896;">&#8377;{ss:,.0f}</div><div class="kl-dot" style="background:#00a07a;margin:5px auto 0;"></div></div>'
        f'<div class="kl-node" style="left:22%;bottom:0;transform:translateX(-50%);"><div class="kl-lbl" style="color:#00c896;">Support</div><div class="kl-val" style="color:#4de8b8;">&#8377;{s1:,.0f}</div><div class="kl-dot" style="background:#00c896;box-shadow:0 0 8px rgba(0,200,150,.5);margin:5px auto 0;"></div></div>'
        f'<div style="position:absolute;left:{cp_pct}%;bottom:6px;transform:translateX(-50%);background:linear-gradient(90deg,#00c896,#6480ff);color:#fff;font-size:11px;font-weight:700;padding:3px 14px;border-radius:20px;white-space:nowrap;box-shadow:0 2px 14px rgba(0,200,150,.35);z-index:10;">NOW &#8377;{cp:,.0f}</div>'
        f'<div class="kl-node" style="left:75%;bottom:0;transform:translateX(-50%);"><div class="kl-lbl" style="color:#ff6b6b;">Resistance</div><div class="kl-val" style="color:#ff9090;">&#8377;{r1:,.0f}</div><div class="kl-dot" style="background:#ff6b6b;box-shadow:0 0 8px rgba(255,107,107,.5);margin:5px auto 0;"></div></div>'
        f'<div class="kl-node" style="left:95%;bottom:0;transform:translateX(-50%);"><div class="kl-lbl" style="color:#cc4040;">Strong Res</div><div class="kl-val" style="color:#ff6b6b;">&#8377;{sr:,.0f}</div><div class="kl-dot" style="background:#cc4040;margin:5px auto 0;"></div></div>'
        f'</div>'
        f'<div class="kl-gradient-bar"><div class="kl-price-tick" style="left:{cp_pct}%;"></div></div>'
        f'<div style="position:relative;height:54px;">{mp_html}</div>'
        f'<div class="kl-dist-row">'
        f'<div class="kl-dist-box" style="border-color:rgba(255,107,107,.18);"><span style="color:var(--muted);">To Resistance</span><span style="color:#ff6b6b;font-weight:700;">+{pts_r:,} pts</span></div>'
        f'<div class="kl-dist-box" style="border-color:rgba(0,200,150,.18);"><span style="color:var(--muted);">To Support</span><span style="color:#00c896;font-weight:700;">-{pts_s:,} pts</span></div>'
        f'</div></div>'
    )


def build_strikes_html(oc):
    if not oc or (not oc["top_ce"] and not oc["top_pe"]): return ""
    def ce_rows(rows):
        o = ""
        for i, r in enumerate(rows, 1):
            o += (f"<tr><td>{i}</td><td><b>&#8377;{int(r['Strike']):,}</b></td>"
                  f"<td>{int(r['CE_OI']):,}</td>"
                  f"<td style=\"color:#00c8e0;font-weight:700;\">&#8377;{r['CE_LTP']:.2f}</td></tr>")
        return o
    def pe_rows(rows):
        o = ""
        for i, r in enumerate(rows, 1):
            o += (f"<tr><td>{i}</td><td><b>&#8377;{int(r['Strike']):,}</b></td>"
                  f"<td>{int(r['PE_OI']):,}</td>"
                  f"<td style=\"color:#ff6b6b;font-weight:700;\">&#8377;{r['PE_LTP']:.2f}</td></tr>")
        return o
    return (
        f'<div class="section"><div class="sec-title">TOP 5 STRIKES BY OPEN INTEREST'
        f'<span class="sec-sub">Spot ±500 pts · Top 5 CE + PE</span></div>'
        f'<div class="strikes-wrap">'
        f'<div><div class="strikes-head" style="color:#00c8e0;">▲ CALL Options (CE)</div>'
        f'<table class="s-table"><thead><tr><th>#</th><th>Strike</th><th>OI</th><th>LTP</th></tr></thead>'
        f'<tbody>{ce_rows(oc["top_ce"])}</tbody></table></div>'
        f'<div><div class="strikes-head" style="color:#ff6b6b;">▼ PUT Options (PE)</div>'
        f'<table class="s-table"><thead><tr><th>#</th><th>Strike</th><th>OI</th><th>LTP</th></tr></thead>'
        f'<tbody>{pe_rows(oc["top_pe"])}</tbody></table></div>'
        f'</div></div>'
    )


# =================================================================
#  SECTION 6 -- STRATEGIES (placeholder)
# =================================================================

def build_strategies_html(oc_analysis):
    return '<div class="section" id="strat"><div class="sec-title">STRATEGIES REFERENCE</div><p style="color:rgba(255,255,255,.4);padding:20px;">Paste your v13 build_strategies_html() here.</p></div>'


# =================================================================
#  SECTION 7 -- TICKER BAR
# =================================================================

def rsi_label(rsi):
    if   rsi >= 70: return "Overbought","#ff6b6b","bearish"
    elif rsi >= 60: return "Strong",    "#ffd166","neutral"
    elif rsi >= 40: return "Neutral",   "#6480ff","neutral"
    elif rsi >= 30: return "Weak",      "#ffd166","neutral"
    else:           return "Oversold",  "#00c896","bullish"

def macd_label(macd, signal):
    d = macd - signal
    if   d >  0.5: return "Bullish",       "#00c896","bullish"
    elif d >  0:   return "Mildly Bullish","#4de8b8","bullish"
    elif d > -0.5: return "Mildly Bearish","#ffd166","neutral"
    else:          return "Bearish",       "#ff6b6b","bearish"

def pcr_label(pcr):
    if   pcr > 1.3: return "Very Bullish","#00c896","bullish"
    elif pcr > 1.1: return "Bullish",     "#4de8b8","bullish"
    elif pcr > 0.9: return "Neutral",     "#6480ff","neutral"
    elif pcr > 0.7: return "Bearish",     "#ffd166","neutral"
    else:           return "Very Bearish","#ff6b6b","bearish"


def build_ticker_bar(tech, oc, vix_data):
    items = []
    if vix_data:
        v=vix_data["value"]; chg=vix_data["change"]; chg_p=vix_data["change_pct"]
        lbl,col,bg,bdr,sig=vix_label(v)
        chg_col="#ff6b6b" if chg>0 else ("#00c896" if chg<0 else "#6480ff")
        items.append(
            f'<div class="tk-item">'
            f'<span class="tk-name" style="background:rgba(0,200,220,.15);color:#00c8e0;border:1px solid rgba(0,200,220,.3);">&#9650;&nbsp;INDIA VIX</span>'
            f'<span class="tk-val" style="color:{col};">{v:.2f}</span>'
            f'<span class="tk-sub" style="color:{chg_col};">{chg:+.2f} ({chg_p:+.2f}%)</span>'
            f'<span class="tk-badge" style="background:{bg};color:{col};border:1px solid {bdr};">{lbl} · {sig}</span>'
            f'</div>'
        )
    if tech:
        rsi=tech["rsi"]; lbl,col,cls=rsi_label(rsi)
        rgb="0,200,150" if cls=="bullish" else ("255,107,107" if cls=="bearish" else "100,128,255")
        items.append(
            f'<div class="tk-item">'
            f'<span class="tk-name" style="background:rgba({rgb},.18);color:{col};border:1px solid rgba({rgb},.35);">&#9643;&nbsp;RSI (14)</span>'
            f'<span class="tk-val" style="color:{col};">{rsi:.1f}</span>'
            f'<span class="tk-badge" style="background:rgba({rgb},.1);color:{col};border:1px solid rgba({rgb},.3);">{lbl}</span>'
            f'</div>'
        )
        macd=tech["macd"]; sig2=tech["signal_line"]; lbl,col,cls=macd_label(macd,sig2)
        rgb2="0,200,150" if cls=="bullish" else ("255,107,107" if cls=="bearish" else "100,128,255")
        diff=macd-sig2
        items.append(
            f'<div class="tk-item">'
            f'<span class="tk-name" style="background:rgba({rgb2},.18);color:{col};border:1px solid rgba({rgb2},.35);">&#9654;&nbsp;MACD</span>'
            f'<span class="tk-val" style="color:{col};">{macd:.2f}</span>'
            f'<span class="tk-sub" style="color:rgba(255,255,255,.4);">Sig:&nbsp;{sig2:.2f}&nbsp;Hist:&nbsp;{diff:+.2f}</span>'
            f'<span class="tk-badge" style="background:rgba({rgb2},.1);color:{col};border:1px solid rgba({rgb2},.3);">{lbl}</span>'
            f'</div>'
        )
    if oc:
        pcr=oc["pcr_oi"]; lbl,col,cls=pcr_label(pcr)
        rgb3="0,200,150" if cls=="bullish" else ("255,107,107" if cls=="bearish" else "100,128,255")
        items.append(
            f'<div class="tk-item">'
            f'<span class="tk-name" style="background:rgba({rgb3},.18);color:{col};border:1px solid rgba({rgb3},.35);">&#9670;&nbsp;PCR (OI)</span>'
            f'<span class="tk-val" style="color:{col};">{pcr:.3f}</span>'
            f'<span class="tk-badge" style="background:rgba({rgb3},.1);color:{col};border:1px solid rgba({rgb3},.3);">{lbl}</span>'
            f'</div>'
        )
    track = "".join(items) * 3
    return f'''<div class="ticker-wrap">
  <div class="ticker-label">LIVE&nbsp;&#9654;</div>
  <div class="ticker-viewport"><div class="ticker-track" id="tkTrack">{track}</div></div>
</div>'''


# =================================================================
#  SECTION 8 -- CSS
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
  --gold:#f5c518;--gold-dim:rgba(245,197,24,.45);--gold-bg:rgba(245,197,24,.10);
}
html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--text);font-family:var(--fh);font-size:13px;line-height:1.6;min-height:100vh}
body::before{content:'';position:fixed;inset:0;
  background-image:
    radial-gradient(ellipse at 15% 0%,rgba(0,200,150,.10) 0%,transparent 50%),
    radial-gradient(ellipse at 85% 10%,rgba(100,128,255,.10) 0%,transparent 50%),
    radial-gradient(ellipse at 50% 90%,rgba(0,200,220,.06) 0%,transparent 50%);
  pointer-events:none;z-index:0;}
.app{position:relative;z-index:1;display:grid;grid-template-rows:auto auto auto 1fr auto;min-height:100vh}
header{display:flex;align-items:center;justify-content:space-between;padding:14px 32px;
  background:rgba(6,8,15,.85);backdrop-filter:blur(16px);
  border-bottom:1px solid rgba(255,255,255,.07);position:sticky;top:0;z-index:200;
  box-shadow:0 1px 0 rgba(0,200,150,.1)}
.logo-wrap{position:relative;height:28px;overflow:hidden;min-width:280px;}
.logo-slide{position:absolute;top:0;left:0;width:100%;font-family:var(--fh);font-size:20px;font-weight:700;
  background:linear-gradient(90deg,#00c896,#6480ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;
  filter:drop-shadow(0 0 12px rgba(0,200,150,.3));opacity:0;transform:translateY(20px);
  transition:opacity .5s ease, transform .5s ease;white-space:nowrap;}
.logo-slide.active{opacity:1;transform:translateY(0);}
.logo-slide.exit{opacity:0;transform:translateY(-20px);}
.hdr-meta{display:flex;align-items:center;gap:14px;font-size:11px;color:var(--muted);font-family:var(--fm)}
.live-dot{width:7px;height:7px;border-radius:50%;background:#00c896;box-shadow:0 0 10px #00c896;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.2}}
.refresh-countdown{display:flex;align-items:center;gap:8px;background:rgba(0,200,150,.07);
  border:1px solid rgba(0,200,150,.18);border-radius:20px;padding:4px 12px;font-family:var(--fm);font-size:11px;}
.countdown-arc-wrap{position:relative;width:18px;height:18px;flex-shrink:0;}
.countdown-arc-wrap svg{display:block;}
.countdown-num{font-family:var(--fm);font-size:12px;font-weight:700;color:#00c896;min-width:20px;text-align:center;transition:color .3s;}
.countdown-num.urgent{color:#ff6b6b;}
.countdown-num.halfway{color:#ffd166;}
.countdown-lbl{font-size:10px;color:rgba(255,255,255,.3);letter-spacing:.5px;}
.refresh-ring{display:none;width:14px;height:14px;border-radius:50%;border:2px solid rgba(0,200,150,.2);border-top-color:#00c896;animation:spin 0.8s linear infinite;}
.refresh-ring.active{display:inline-block;}
@keyframes spin{to{transform:rotate(360deg)}}
#refreshStatus{font-size:10px;color:rgba(255,255,255,.35);transition:color .3s;letter-spacing:.3px;}
#refreshStatus.updated{color:#00c896;font-weight:600;}
.hero{display:flex;align-items:stretch;background:linear-gradient(135deg,rgba(0,200,150,.055) 0%,rgba(100,128,255,.055) 100%);border-bottom:1px solid rgba(255,255,255,.07);overflow:hidden;position:relative;height:97px;}
.hero::before{content:'';position:absolute;top:-50px;left:-50px;width:200px;height:200px;border-radius:50%;background:radial-gradient(circle,rgba(0,200,150,.10),transparent 70%);pointer-events:none;}
.h-gauges{flex-shrink:0;display:flex;align-items:center;gap:10px;padding:0 16px 0 18px;}
.gauge-sep{width:1px;height:56px;background:rgba(255,255,255,.08);flex-shrink:0;}
.gauge-wrap{position:relative;width:76px;height:76px;}
.gauge-wrap svg{display:block;}
.gauge-inner{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;}
.g-val{font-family:'DM Mono',monospace;font-size:13px;font-weight:700;line-height:1;}
.g-lbl{font-size:7.5px;letter-spacing:1.5px;text-transform:uppercase;color:rgba(255,255,255,.28);margin-top:2px;}
.h-mid{flex:1;min-width:0;display:flex;flex-direction:column;justify-content:center;padding:0 15px 0 13px;border-left:1px solid rgba(255,255,255,.05);}
.h-eyebrow{font-size:8px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:rgba(255,255,255,.22);margin-bottom:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.h-signal{font-size:22px;font-weight:900;letter-spacing:1px;line-height:1.1;margin-bottom:2px;}
.h-sub{font-size:9.5px;color:rgba(255,255,255,.32);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.h-divider{height:1px;background:rgba(255,255,255,.05);margin:5px 0;}
.pill-row{display:flex;align-items:center;gap:8px;margin-bottom:4px;}
.pill-row:last-child{margin-bottom:0;}
.pill-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0;}
.pill-lbl{font-size:8px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:rgba(255,255,255,.35);width:96px;flex-shrink:0;}
.pill-track{width:120px;height:5px;background:rgba(255,255,255,.07);border-radius:3px;overflow:hidden;flex-shrink:0;}
.pill-fill{height:100%;border-radius:3px;}
.pill-num{font-family:'DM Mono',monospace;font-size:10px;font-weight:700;margin-left:8px;flex-shrink:0;}
.h-stats{flex-shrink:0;min-width:360px;display:flex;flex-direction:column;border-left:1px solid rgba(255,255,255,.07);background:rgba(255,255,255,.015);}
.h-stat-row{display:flex;align-items:stretch;flex:1;border-bottom:1px solid rgba(255,255,255,.05);}
.h-stat{flex:1;display:flex;flex-direction:column;justify-content:center;padding:5px 10px;text-align:center;border-right:1px solid rgba(255,255,255,.04);}
.h-stat:last-child{border-right:none;}
.h-stat-lbl{font-size:7.5px;font-weight:700;letter-spacing:1.8px;text-transform:uppercase;color:rgba(255,255,255,.22);margin-bottom:3px;white-space:nowrap;}
.h-stat-val{font-family:'DM Mono',monospace;font-size:13px;font-weight:700;line-height:1;white-space:nowrap;}
.h-stat-bottom{display:flex;align-items:center;justify-content:space-between;padding:4px 10px;}
.h-bias-row{display:flex;align-items:center;gap:6px;}
.h-chip{font-size:9px;font-weight:700;letter-spacing:.8px;text-transform:uppercase;padding:2px 9px;border-radius:20px;white-space:nowrap;}
.h-score{font-family:'DM Mono',monospace;font-size:8px;color:rgba(255,255,255,.22);letter-spacing:.5px;}
.h-ts{font-family:'DM Mono',monospace;font-size:8px;color:rgba(255,255,255,.18);letter-spacing:.5px;white-space:nowrap;}
.main{display:grid;grid-template-columns:268px 1fr;min-height:0}
.sidebar{background:rgba(8,11,20,.7);backdrop-filter:blur(12px);border-right:1px solid rgba(255,255,255,.06);position:sticky;top:57px;height:calc(100vh - 57px);overflow-y:auto;display:flex;flex-direction:column;}
.sidebar-sticky-top{position:sticky;top:0;z-index:50;background:rgba(8,11,20,.95);backdrop-filter:blur(16px);border-bottom:1px solid rgba(100,128,255,.15);padding-bottom:4px;}
.sidebar-scroll{flex:1;overflow-y:auto;}
.sidebar::-webkit-scrollbar{width:3px}
.sidebar::-webkit-scrollbar-thumb{background:rgba(255,255,255,.1);border-radius:2px}
.sidebar-scroll::-webkit-scrollbar{width:3px}
.sidebar-scroll::-webkit-scrollbar-thumb{background:rgba(255,255,255,.1);border-radius:2px}
.sb-sec{padding:16px 12px 8px}
.sb-lbl{font-size:9px;font-weight:700;letter-spacing:.15em;text-transform:uppercase;color:var(--aurora1);margin-bottom:8px;padding:0 0 0 8px;border-left:2px solid var(--aurora1)}
.sb-btn{display:flex;align-items:center;gap:8px;width:100%;padding:9px 12px;border-radius:8px;border:1px solid transparent;cursor:pointer;background:transparent;color:var(--muted);font-family:var(--fh);font-size:12px;text-align:left;transition:all .15s}
.sb-btn:hover{background:rgba(0,200,150,.08);color:rgba(255,255,255,.8);border-color:rgba(0,200,150,.2)}
.sb-btn.active{background:rgba(0,200,150,.1);border-color:rgba(0,200,150,.25);color:#00c896;font-weight:600}
.sb-badge{font-size:10px;margin-left:auto;font-weight:700}
.content{overflow-y:auto}
.section{padding:26px 28px;border-bottom:1px solid rgba(255,255,255,.05);background:transparent;position:relative}
.section:nth-child(odd){background:rgba(255,255,255,.015)}
.sec-title{font-family:var(--fh);font-size:11px;font-weight:700;letter-spacing:2.5px;color:var(--aurora1);text-transform:uppercase;display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:20px;padding-bottom:12px;border-bottom:1px solid rgba(0,200,150,.15)}
.sec-sub{font-size:11px;color:var(--muted2);font-weight:400;letter-spacing:.5px;text-transform:none;margin-left:auto}
.oi-ticker-table{border:1px solid rgba(255,255,255,.07);border-radius:14px;overflow:hidden}
.oi-ticker-hdr{display:grid;grid-template-columns:130px repeat(5,1fr);padding:9px 18px;align-items:center;gap:6px}
.oi-ticker-hdr-label{font-size:9px;font-weight:700;letter-spacing:2px;text-transform:uppercase}
.oi-ticker-hdr-cell{font-size:9px;letter-spacing:1.5px;text-transform:uppercase;color:rgba(255,255,255,.35);text-align:center}
.oi-ticker-row{display:grid;grid-template-columns:130px repeat(5,1fr);padding:15px 18px;border-top:1px solid rgba(255,255,255,.04);align-items:center;gap:6px;transition:background .15s}
.oi-ticker-row:hover{background:rgba(255,255,255,.03)}
.oi-ticker-metric{font-size:10px;font-weight:600;letter-spacing:1px;text-transform:uppercase;color:rgba(255,255,255,.35)}
.oi-ticker-cell{text-align:center}
.kl-zone-labels{display:flex;justify-content:space-between;margin-bottom:6px;font-size:11px;font-weight:700}
.kl-node{position:absolute;text-align:center}
.kl-lbl{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.5px;line-height:1.3;white-space:nowrap}
.kl-val{font-size:12px;font-weight:700;color:rgba(255,255,255,.7);white-space:nowrap;margin-top:2px}
.kl-dot{width:11px;height:11px;border-radius:50%;border:2px solid var(--bg)}
.kl-gradient-bar{position:relative;height:6px;border-radius:3px;background:linear-gradient(90deg,#00a07a 0%,#00c896 25%,#6480ff 55%,#ff6b6b 80%,#cc4040 100%);box-shadow:0 0 12px rgba(0,200,150,.2)}
.kl-price-tick{position:absolute;top:50%;transform:translate(-50%,-50%);width:3px;height:18px;background:#fff;border-radius:2px;box-shadow:0 0 12px rgba(255,255,255,.6);z-index:10}
.kl-dist-row{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:4px}
.kl-dist-box{background:rgba(255,255,255,.03);border:1px solid;border-radius:10px;padding:10px 14px;display:flex;justify-content:space-between;align-items:center}
.strikes-head{font-weight:700;margin-bottom:10px;font-size:13px}
.strikes-wrap{display:grid;grid-template-columns:1fr 1fr;gap:20px}
.s-table{width:100%;border-collapse:collapse;border-radius:10px;overflow:hidden}
.s-table th{background:linear-gradient(90deg,rgba(0,200,150,.15),rgba(100,128,255,.15));color:rgba(255,255,255,.7);padding:10px 12px;font-size:11px;font-weight:600;text-align:left;letter-spacing:.5px;border-bottom:1px solid rgba(255,255,255,.08)}
.s-table td{padding:10px 12px;border-bottom:1px solid rgba(255,255,255,.05);font-size:12px;color:rgba(255,255,255,.8);background:rgba(255,255,255,.02)}
.s-table tr:last-child td{border-bottom:none}
.s-table tr:hover td{background:rgba(0,200,150,.05)}
.ticker-wrap{display:flex;align-items:center;background:rgba(4,6,12,.97);border-bottom:1px solid rgba(255,255,255,.07);height:46px;overflow:hidden;position:relative;z-index:190;box-shadow:0 2px 20px rgba(0,0,0,.5);}
.ticker-label{flex-shrink:0;padding:0 16px;font-family:var(--fm);font-size:9px;font-weight:700;letter-spacing:3px;color:#00c896;text-transform:uppercase;border-right:1px solid rgba(0,200,150,.2);height:100%;display:flex;align-items:center;background:rgba(0,200,150,.07);white-space:nowrap;}
.ticker-viewport{flex:1;overflow:hidden;height:100%}
.ticker-track{display:flex;align-items:center;height:100%;white-space:nowrap;animation:ticker-scroll 38s linear infinite;will-change:transform;}
.ticker-track:hover{animation-play-state:paused}
@keyframes ticker-scroll{0%{transform:translateX(0)}100%{transform:translateX(-33.333%)}}
.tk-item{display:inline-flex;align-items:center;gap:10px;padding:0 20px;height:100%;border-right:1px solid rgba(255,255,255,.04);flex-shrink:0;}
.tk-name{font-family:var(--fm);font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;padding:3px 10px;border-radius:6px;white-space:nowrap;flex-shrink:0;background:rgba(255,255,255,.08);color:rgba(255,255,255,.5);border:1px solid rgba(255,255,255,.1);}
.tk-val{font-family:var(--fm);font-size:18px;font-weight:700;line-height:1;white-space:nowrap;}
.tk-sub{font-family:var(--fm);font-size:10px;color:rgba(255,255,255,.35);white-space:nowrap;}
.tk-badge{font-family:var(--fh);font-size:10px;font-weight:700;padding:3px 10px;border-radius:20px;white-space:nowrap;letter-spacing:.3px;}
footer{padding:16px 32px;border-top:1px solid rgba(255,255,255,.06);background:rgba(6,8,15,.9);backdrop-filter:blur(12px);display:flex;justify-content:space-between;font-size:11px;color:var(--muted2);font-family:var(--fm)}

/* ══ OPTION GREEKS PANEL (sidebar) ══ */
.greeks-panel{margin:10px 10px 6px;padding:14px 12px;background:linear-gradient(135deg,rgba(100,128,255,.12),rgba(0,200,220,.10));border-radius:14px;border:1px solid rgba(100,128,255,.28);box-shadow:0 4px 20px rgba(100,128,255,.1),inset 0 1px 0 rgba(255,255,255,.06);}
.greeks-title{font-size:9px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:rgba(138,160,255,1.0);margin-bottom:10px;padding-bottom:8px;border-bottom:1px solid rgba(100,128,255,.25);display:flex;align-items:center;justify-content:space-between;}
.greeks-expiry-tag{font-size:8.5px;color:rgba(255,255,255,.5);font-weight:400;letter-spacing:.5px;text-transform:none;}
/* GOLDEN DROPDOWN */
.greeks-strike-wrap{position:relative;margin-bottom:10px;}
.greeks-strike-wrap::after{content:'▼';position:absolute;right:10px;top:50%;transform:translateY(-50%);font-size:8px;color:var(--gold);pointer-events:none;z-index:2;}
.greeks-strike-select{width:100%;appearance:none;-webkit-appearance:none;background:linear-gradient(135deg,rgba(245,197,24,.12),rgba(200,155,10,.06));border:1px solid var(--gold-dim);border-radius:8px;color:var(--gold);font-family:'DM Mono',monospace;font-size:11px;font-weight:700;padding:7px 28px 7px 10px;cursor:pointer;outline:none;letter-spacing:.5px;transition:border-color .2s,background .2s,box-shadow .2s;}
.greeks-strike-select:hover{border-color:rgba(245,197,24,.75);background:linear-gradient(135deg,rgba(245,197,24,.18),rgba(200,155,10,.10));box-shadow:0 0 10px rgba(245,197,24,.18);}
.greeks-strike-select:focus{border-color:var(--gold);box-shadow:0 0 0 2px rgba(245,197,24,.25);}
.greeks-strike-select option{background:#0e1225;color:var(--gold);font-weight:700;}
/* Greek name labels — BRIGHT */
.greek-name{font-family:'DM Mono',monospace;font-size:11px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:rgba(255,255,255,.92);}
.greek-sub{font-size:8px;color:rgba(255,255,255,.55);margin-top:1px;}
.greeks-row{display:flex;align-items:center;justify-content:space-between;padding:7px 0;border-bottom:1px solid rgba(255,255,255,.06);}
.greeks-row:last-child{border-bottom:none;}
.greeks-atm-badge{display:flex;align-items:center;justify-content:center;gap:6px;background:rgba(100,128,255,.1);border:1px solid rgba(100,128,255,.25);border-radius:8px;padding:5px 8px;margin-bottom:10px;font-family:'DM Mono',monospace;font-size:11px;flex-wrap:wrap;}
.greeks-atm-strike{font-weight:700;color:#8aa0ff;}
.iv-bar-wrap{display:flex;align-items:center;gap:6px;margin-top:8px;padding-top:8px;border-top:1px solid rgba(255,255,255,.06);}
.iv-bar-label{font-size:8px;color:rgba(255,255,255,.7);letter-spacing:1px;text-transform:uppercase;font-weight:600;width:42px;flex-shrink:0;}
.iv-bar-track{flex:1;height:4px;background:rgba(255,255,255,.08);border-radius:2px;overflow:hidden;}
.iv-bar-fill{height:100%;border-radius:2px;transition:width .6s ease;}
.iv-bar-num{font-family:'DM Mono',monospace;font-size:11px;font-weight:700;min-width:38px;text-align:right;}
/* GREEKS TABLE (main content) */
.greeks-table-section{padding:22px 28px;border-bottom:1px solid rgba(255,255,255,.05);}
.greeks-table-wrap{display:grid;grid-template-columns:1fr 1fr;gap:16px;}
.greeks-tbl{border:1px solid rgba(255,255,255,.07);border-radius:12px;overflow:hidden;}
.greeks-tbl-head{display:grid;grid-template-columns:90px repeat(4,1fr);background:rgba(255,255,255,.04);padding:8px 14px;border-bottom:1px solid rgba(255,255,255,.06);gap:4px;}
.greeks-tbl-head-label{font-size:8.5px;letter-spacing:1.5px;text-transform:uppercase;color:rgba(255,255,255,.3);text-align:center;}
.greeks-tbl-row{display:grid;grid-template-columns:90px repeat(4,1fr);padding:9px 14px;border-bottom:1px solid rgba(255,255,255,.04);align-items:center;gap:4px;transition:background .15s;}
.greeks-tbl-row:last-child{border-bottom:none;}
.greeks-tbl-row:hover{background:rgba(255,255,255,.03);}
.greeks-tbl-strike{font-family:'DM Mono',monospace;font-size:12px;font-weight:700;color:rgba(255,255,255,.8);}
.greeks-tbl-cell{font-family:'DM Mono',monospace;font-size:11px;font-weight:600;text-align:center;color:rgba(255,255,255,.65);}

@media(max-width:1200px){.h-stats{min-width:280px;}.logo-wrap{min-width:220px;}}
@media(max-width:1024px){
  .main{grid-template-columns:1fr}.sidebar{position:static;height:auto;border-right:none;border-bottom:1px solid rgba(255,255,255,.06)}
  .hero{height:auto;flex-wrap:wrap;}.h-gauges{padding:12px 18px;}.h-stats{min-width:100%;border-left:none;border-top:1px solid rgba(255,255,255,.07);}
  .oi-ticker-hdr,.oi-ticker-row{grid-template-columns:100px repeat(3,1fr)}
  .oi-ticker-hdr-cell:nth-child(n+5),.oi-ticker-cell:nth-child(n+5){display:none}
  .strikes-wrap{grid-template-columns:1fr}
  .greeks-table-wrap{grid-template-columns:1fr}
  .greeks-tbl-head,.greeks-tbl-row{grid-template-columns:80px repeat(4,1fr)}
}
@media(max-width:640px){
  header{padding:12px 16px}.section{padding:18px 16px}
  .oi-ticker-hdr,.oi-ticker-row{grid-template-columns:90px repeat(2,1fr)}
  .oi-ticker-hdr-cell:nth-child(n+4),.oi-ticker-cell:nth-child(n+4){display:none}
  .kl-dist-row{grid-template-columns:1fr}footer{flex-direction:column;gap:6px}
  .logo-wrap{min-width:160px;}.refresh-countdown{display:none;}
  .greeks-table-section{padding:16px;}
  .greeks-tbl-head,.greeks-tbl-row{grid-template-columns:70px repeat(3,1fr)}
  .greeks-tbl-head-label:nth-child(5),.greeks-tbl-cell:nth-child(5){display:none}
}
"""

ANIMATED_JS = """
<script>
(function() {
  const NAMES = ['NIFTYCRAFT','Nifty Option Strategy Builder','OI Signal Dashboard','Options Analytics Hub','PCR &amp; Max Pain Tracker'];
  const wrap = document.getElementById('logoWrap');
  if (!wrap) return;
  NAMES.forEach((name, i) => {
    const el = document.createElement('div');
    el.className = 'logo-slide' + (i === 0 ? ' active' : '');
    el.innerHTML = name;
    wrap.appendChild(el);
  });
  let cur = 0;
  setInterval(() => {
    const slides = wrap.querySelectorAll('.logo-slide');
    slides[cur].classList.remove('active'); slides[cur].classList.add('exit');
    setTimeout(() => { slides[cur].classList.remove('exit'); }, 600);
    cur = (cur + 1) % slides.length; slides[cur].classList.add('active');
  }, 4000);
})();

(function() {
  const TOTAL = 30, R = 7, C = 2 * Math.PI * R;
  let remaining = TOTAL, countdownTimer = null;
  const numEl = document.getElementById('cdNum'), arcEl = document.getElementById('cdArc');
  function updateCountdown(secs) {
    if (!numEl || !arcEl) return;
    numEl.textContent = secs;
    numEl.className = 'countdown-num' + (secs <= 5 ? ' urgent' : secs <= 15 ? ' halfway' : '');
    arcEl.style.strokeDashoffset = (C * (1 - secs / TOTAL)).toFixed(2);
    arcEl.style.stroke = secs <= 5 ? '#ff6b6b' : secs <= 15 ? '#ffd166' : '#00c896';
  }
  function startCountdown(from) {
    clearInterval(countdownTimer); remaining = from; updateCountdown(remaining);
    countdownTimer = setInterval(() => { remaining = Math.max(0, remaining - 1); updateCountdown(remaining); }, 1000);
  }
  window.__resetCountdown = function() { startCountdown(TOTAL); };
  startCountdown(TOTAL);
})();

(function() {
  const INTERVAL_MS = 30000;
  let _lastBias = null, _lastPCR = null, _refreshTimer = null;
  function showSpinner(on) {
    const ring = document.getElementById('refreshRing'), txt = document.getElementById('refreshStatus');
    if (ring) ring.classList.toggle('active', on);
    if (txt) txt.textContent = on ? 'Refreshing\u2026' : '';
  }
  function flashUpdated() {
    const txt = document.getElementById('refreshStatus');
    if (!txt) return;
    txt.textContent = 'Updated \u2713'; txt.classList.add('updated');
    setTimeout(() => { txt.textContent = ''; txt.classList.remove('updated'); }, 2500);
  }
  function patchEl(cur, neo) {
    if (cur && neo && cur.innerHTML !== neo.innerHTML) { cur.innerHTML = neo.innerHTML; return true; }
    return false;
  }
  function applyNewDoc(html) {
    const parser = new DOMParser(), newDoc = parser.parseFromString(html, 'text/html');
    let changed = false;
    const curHero = document.getElementById('heroWidget'), neoHero = newDoc.getElementById('heroWidget');
    if (curHero && neoHero && curHero.outerHTML !== neoHero.outerHTML) { curHero.outerHTML = neoHero.outerHTML; changed = true; }
    ['oi','kl','strikes','greeksTable','greeksPanel'].forEach(id => { changed |= patchEl(document.getElementById(id), newDoc.getElementById(id)); });
    changed |= patchEl(document.getElementById('tkTrack'), newDoc.getElementById('tkTrack'));
    const curTs = document.getElementById('lastUpdatedTs'), neoTs = newDoc.getElementById('lastUpdatedTs');
    if (curTs && neoTs && curTs.textContent !== neoTs.textContent) { curTs.textContent = neoTs.textContent; changed = true; }
    return changed;
  }
  function silentRefresh() {
    fetch('latest.json?_=' + Date.now())
      .then(r => { if (!r.ok) throw new Error('json'); return r.json(); })
      .then(data => {
        if (window.__resetCountdown) window.__resetCountdown();
        if (_lastBias !== null && data.bias === _lastBias && String(data.pcr) === String(_lastPCR)) { schedule(); return; }
        _lastBias = data.bias; _lastPCR = String(data.pcr);
        showSpinner(true);
        fetch('index.html?_=' + Date.now())
          .then(r => { if (!r.ok) throw new Error('html'); return r.text(); })
          .then(html => { const changed = applyNewDoc(html); showSpinner(false); if (changed) flashUpdated(); schedule(); })
          .catch(() => { showSpinner(false); schedule(); });
      })
      .catch(() => { if (window.__resetCountdown) window.__resetCountdown(); schedule(); });
  }
  function schedule() { clearTimeout(_refreshTimer); _refreshTimer = setTimeout(silentRefresh, INTERVAL_MS); }
  window.addEventListener('load', function() { schedule(); });
})();
</script>
"""



def build_greeks_script_html(oc_analysis):
    """
    Returns the <script> block for the Option Greeks panel.
    Kept OUTSIDE greeksPanel div so innerHTML-patching during auto-refresh
    does NOT kill the greeksUpdateStrike function.
    """
    if not oc_analysis:
        return ""

    all_rows = oc_analysis.get("all_strikes", oc_analysis.get("greeks_table", []))
    atm      = int(oc_analysis.get("atm_strike", 0))

    if not all_rows:
        return ""

    strikes_json_parts = []
    for row in all_rows:
        s = int(row["strike"])
        strikes_json_parts.append(
            f'"{s}":{{'
            + f'"ce_ltp":{round(float(row["ce_ltp"]),2)},'
            + f'"pe_ltp":{round(float(row["pe_ltp"]),2)},'
            + f'"ce_delta":{round(float(row["ce_delta"]),4)},'
            + f'"pe_delta":{round(float(row["pe_delta"]),4)},'
            + f'"ce_iv":{round(float(row["ce_iv"]),2)},'
            + f'"pe_iv":{round(float(row["pe_iv"]),2)},'
            + f'"ce_theta":{round(float(row["ce_theta"]),4)},'
            + f'"pe_theta":{round(float(row["pe_theta"]),4)},'
            + f'"ce_vega":{round(float(row["ce_vega"]),4)},'
            + f'"pe_vega":{round(float(row["pe_vega"]),4)}'
            + f'}}')
    strikes_json = "{" + ",".join(strikes_json_parts) + "}"

    return f"""<script>
/* Greeks script lives OUTSIDE #greeksPanel so auto-refresh innerHTML patch
   cannot destroy greeksUpdateStrike. Re-initialises on every full page load. */
(function() {{
  var _gData = {strikes_json};
  var _atm   = {atm};

  function _initGreeks() {{
    var sel = document.getElementById('greeksStrikeSelect');
    if (sel) {{
      sel.addEventListener('change', function() {{ greeksUpdateStrike(this.value); }});
      greeksUpdateStrike(sel.value);
    }}
  }}

  window.greeksUpdateStrike = function(strike) {{
    var key = String(parseInt(strike, 10));
    var d   = _gData[key] || _gData[String(parseFloat(strike))] || _gData[String(strike)];
    if (!d) {{
      console.warn('greeksUpdateStrike: key miss for', strike, 'keys:', Object.keys(_gData).slice(0,5));
      return;
    }}

    /* Fade */
    var ids = ['greeksStrikeTypeLabel','greeksStrikeLabel','greeksCeLtp','greeksPeLtp',
               'greeksDeltaWrap','greeksSkewLbl',
               'greeksIvCe','greeksIvPe',
               'greeksThetaCe','greeksThetaPe',
               'greeksVegaCe','greeksVegaPe',
               'greeksIvBar','greeksIvAvg','greeksIvRegime'];
    ids.forEach(function(id) {{
      var el = document.getElementById(id);
      if (el) {{ el.style.transition = 'opacity .18s'; el.style.opacity = '0.15'; }}
    }});

    setTimeout(function() {{
      var sel  = parseInt(strike, 10);
      var dist = Math.round(Math.abs(sel - _atm) / 50);
      var lbl  = sel === _atm ? 'ATM' : (sel > _atm ? 'CE+' + dist : 'PE-' + dist);

      document.getElementById('greeksStrikeTypeLabel').textContent = lbl;
      document.getElementById('greeksStrikeLabel').innerHTML = '₹' + sel.toLocaleString('en-IN');
      document.getElementById('greeksCeLtp').innerHTML = 'CE ₹' + (d.ce_ltp||0).toFixed(1);
      document.getElementById('greeksPeLtp').innerHTML = 'PE ₹' + (d.pe_ltp||0).toFixed(1);

      /* Delta bars */
      var ceCol='#00c896', peCol='#ff6b6b';
      var cePct=Math.min(100,Math.abs(d.ce_delta)*100).toFixed(0);
      var pePct=Math.min(100,Math.abs(d.pe_delta)*100).toFixed(0);
      document.getElementById('greeksDeltaWrap').innerHTML =
        '<div style="display:flex;align-items:center;gap:5px;">' +
          '<div style="width:34px;height:3px;background:rgba(255,255,255,.10);border-radius:2px;overflow:hidden;">' +
            '<div style="width:'+cePct+'%;height:100%;background:'+ceCol+';border-radius:2px;"></div></div>' +
          '<span style="font-family:DM Mono,monospace;font-size:11px;font-weight:700;color:'+ceCol+';">' +
               (d.ce_delta>=0?'+':'')+d.ce_delta.toFixed(3)+'</span></div>' +
        '<div style="display:flex;align-items:center;gap:5px;margin-top:3px;">' +
          '<div style="width:34px;height:3px;background:rgba(255,255,255,.10);border-radius:2px;overflow:hidden;">' +
            '<div style="width:'+pePct+'%;height:100%;background:'+peCol+';border-radius:2px;"></div></div>' +
          '<span style="font-family:DM Mono,monospace;font-size:11px;font-weight:700;color:'+peCol+';">' +
               (d.pe_delta>=0?'+':'')+d.pe_delta.toFixed(3)+'</span></div>';

      /* IV */
      document.getElementById('greeksIvCe').textContent = (d.ce_iv||0).toFixed(1)+'%';
      document.getElementById('greeksIvPe').textContent = (d.pe_iv||0).toFixed(1)+'%';

      /* IV Skew */
      var skew=((d.pe_iv||0)-(d.ce_iv||0)).toFixed(1);
      var skewEl=document.getElementById('greeksSkewLbl');
      skewEl.textContent = parseFloat(skew)>0?'PE Skew +'+skew:'CE Skew '+skew;
      skewEl.style.color = parseFloat(skew)>1.5?'#ff6b6b':(parseFloat(skew)<-1.5?'#00c896':'#6480ff');

      /* Theta */
      function tfmt(t){{ return Math.abs(t)>=0.01?'₹'+Math.abs(t).toFixed(2):t.toFixed(4); }}
      document.getElementById('greeksThetaCe').innerHTML = tfmt(d.ce_theta||0);
      document.getElementById('greeksThetaPe').innerHTML = tfmt(d.pe_theta||0);

      /* Vega */
      function vfmt(v){{ return Math.abs(v)>=0.0001?v.toFixed(4):'—'; }}
      document.getElementById('greeksVegaCe').innerHTML = vfmt(d.ce_vega||0);
      document.getElementById('greeksVegaPe').innerHTML = vfmt(d.pe_vega||0);

      /* IV bar */
      var ivAvg=((d.ce_iv||0)+(d.pe_iv||0))/2;
      var ivCol=ivAvg>25?'#ff6b6b':(ivAvg>18?'#ffd166':'#00c896');
      var ivReg=ivAvg>25?'High IV · Buy Premium':(ivAvg>15?'Normal IV · Balanced':'Low IV · Sell Premium');
      var ivPct=Math.min(100,Math.max(0,(ivAvg/60)*100)).toFixed(1);
      var barEl=document.getElementById('greeksIvBar');
      barEl.style.width=ivPct+'%'; barEl.style.background=ivCol; barEl.style.boxShadow='0 0 6px '+ivCol+'88';
      var avgEl=document.getElementById('greeksIvAvg');
      avgEl.textContent=ivAvg.toFixed(1)+'%'; avgEl.style.color=ivCol;
      var regEl=document.getElementById('greeksIvRegime');
      regEl.textContent=ivReg; regEl.style.color=ivCol;

      /* Fade in */
      ids.forEach(function(id){{ var el=document.getElementById(id); if(el) el.style.opacity='1'; }});
    }}, 180);
  }};

  if (document.readyState === 'loading') {{
    document.addEventListener('DOMContentLoaded', _initGreeks);
  }} else {{
    setTimeout(_initGreeks, 50);
  }}
}})();
</script>"""


# =================================================================
#  SECTION 10 -- HTML ASSEMBLER
# =================================================================

def generate_html(tech, oc, md, ts, vix_data=None):
    cp   = tech["price"]    if tech else 0
    bias = md["bias"]; conf = md["confidence"]
    bull = md["bull"]; bear = md["bear"]; diff = md["diff"]

    oi_html          = build_oi_html(oc)               if oc   else ""
    kl_html          = build_key_levels_html(tech, oc) if tech else ""
    strat_html       = build_strategies_html(oc)
    strikes_html     = build_strikes_html(oc)
    ticker_html      = build_ticker_bar(tech, oc, vix_data)
    gauge_html       = build_dual_gauge_hero(oc, tech, md, ts)
    greeks_sidebar   = build_greeks_sidebar_html(oc)
    greeks_script    = build_greeks_script_html(oc)
    greeks_table     = build_greeks_table_html(oc)

    C = 2 * 3.14159 * 7

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Nifty 50 Options Dashboard v16</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet">
<style>{CSS}</style>
</head>
<body>
<div class="app">
<header>
  <div class="logo-wrap" id="logoWrap"></div>
  <div class="hdr-meta">
    <div class="live-dot"></div>
    <span>NSE Options Dashboard</span>
    <span style="color:rgba(255,255,255,.15);">|</span>
    <span>{ts}</span>
    <span style="color:rgba(255,255,255,.15);">|</span>
    <div class="refresh-countdown">
      <div class="countdown-arc-wrap">
        <svg width="18" height="18" viewBox="0 0 18 18">
          <circle cx="9" cy="9" r="7" fill="none" stroke="rgba(255,255,255,.1)" stroke-width="2"/>
          <circle id="cdArc" cx="9" cy="9" r="7" fill="none" stroke="#00c896" stroke-width="2"
            stroke-linecap="round" stroke-dasharray="{C:.2f}" stroke-dashoffset="0"
            style="transform:rotate(-90deg);transform-origin:9px 9px;transition:stroke-dashoffset 1s linear,stroke .3s;"/>
        </svg>
      </div>
      <span class="countdown-num" id="cdNum">30</span>
      <span class="countdown-lbl" id="cdLbl">s</span>
      <div class="refresh-ring" id="refreshRing"></div>
      <span id="refreshStatus"></span>
    </div>
  </div>
</header>
{ticker_html}
{gauge_html}
<div class="main">
  <aside class="sidebar">
    <div class="sidebar-sticky-top">
      <div id="greeksPanel">{greeks_sidebar}</div>
    </div>
    <div class="sidebar-scroll">
    <div class="sb-sec">
      <div class="sb-lbl">LIVE ANALYSIS</div>
      <button class="sb-btn active" onclick="go('oi',this)">OI Dashboard</button>
      <button class="sb-btn"        onclick="go('greeksTable',this)">&#9652; Option Greeks</button>
      <button class="sb-btn"        onclick="go('kl',this)">Key Levels</button>
    </div>
    <div class="sb-sec">
      <div class="sb-lbl">STRATEGIES</div>
      <button class="sb-btn" onclick="go('strat',this)">&#9650; Bullish</button>
      <button class="sb-btn" onclick="go('strat',this)">&#9660; Bearish</button>
      <button class="sb-btn" onclick="go('strat',this)">&#8596; Non-Directional</button>
    </div>
    <div class="sb-sec">
      <div class="sb-lbl">OPTION CHAIN</div>
      <button class="sb-btn" onclick="go('strikes',this)">Top 5 Strikes</button>
    </div>
    </div>
  </aside>
  <main class="content">
    <div id="oi">{oi_html}</div>
    {greeks_table}
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
  <span>NiftyCraft · Nifty Option Strategy Builder · v16</span>
  <span>Option Greeks · OI Dashboard · 30s Silent Refresh · Educational Only · &copy; 2025</span>
</footer>
</div>

<script>
function go(id,btn){{
  const el=document.getElementById(id);
  if(el)el.scrollIntoView({{behavior:"smooth",block:"start"}});
  if(btn){{document.querySelectorAll(".sb-btn").forEach(b=>b.classList.remove("active"));btn.classList.add("active");}}
}}
</script>
{greeks_script}
{ANIMATED_JS}
</body>
</html>"""


# =================================================================
#  SECTION 11 -- MAIN
# =================================================================

def main():
    ist_tz = pytz.timezone("Asia/Kolkata")
    ts     = datetime.now(ist_tz).strftime("%d-%b-%Y %H:%M IST")
    print("=" * 65)
    print("  NIFTY 50 OPTIONS DASHBOARD — Aurora Theme v16")
    print(f"  {ts}")
    print("  FIXES:")
    print("  + Option Greeks (renamed from ATM Greeks)")
    print("  + BS Greeks computed for EVERY strike — dropdown always updates")
    print("  + VIX wired through full call chain as IV fallback")
    print("  + _bs_greeks always returns valid dict (never None)")
    print("=" * 65)

    print("\n[1/4] Fetching NSE Option Chain...")
    nse = NSEOptionChain()
    oc_raw, nse_session, nse_headers = nse.fetch()

    print("\n[2/4] Fetching India VIX...")
    vix_data = fetch_india_vix(nse_session, nse_headers)
    live_vix = vix_data["value"] if vix_data else 18.0
    print(f"  VIX for BS fallback: {live_vix}")

    # FIXED: pass live_vix into analyze_option_chain
    oc_analysis = analyze_option_chain(oc_raw, vix=live_vix) if oc_raw else None
    if oc_analysis:
        g = oc_analysis.get("atm_greeks", {})
        n_strikes = len(oc_analysis.get("all_strikes", []))
        print(f"\n  OK  Spot={oc_analysis['underlying']:.2f}  ATM={oc_analysis['atm_strike']}")
        print(f"      Greeks computed for {n_strikes} strikes (all unique via BS)")
        print(f"      ATM CE Δ={g.get('ce_delta',0):.3f}  IV={g.get('ce_iv',0):.1f}%  θ={g.get('ce_theta',0):.4f}")
        print(f"      ATM PE Δ={g.get('pe_delta',0):.3f}  IV={g.get('pe_iv',0):.1f}%  θ={g.get('pe_theta',0):.4f}")

    print("\n[3/4] Fetching Technical Indicators...")
    tech = get_technical_data()

    print("\n[4/4] Scoring Market Direction...")
    md = compute_market_direction(tech, oc_analysis)
    print(f"  OK  {md['bias']} ({md['confidence']} confidence)")

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
        "expiry":     oc_analysis["expiry"]    if oc_analysis else None,
        "pcr":        oc_analysis["pcr_oi"]    if oc_analysis else None,
        "oi_dir":     oc_analysis["oi_dir"]    if oc_analysis else None,
        "raw_oi_dir": oc_analysis["raw_oi_dir"] if oc_analysis else None,
        "india_vix":  vix_data["value"]         if vix_data   else None,
        "atm_greeks": oc_analysis.get("atm_greeks") if oc_analysis else None,
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
