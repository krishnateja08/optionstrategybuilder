#!/usr/bin/env python3
"""
Nifty 50 Options Strategy Dashboard — GitHub Pages Generator
Aurora Borealis Theme · v19.0 · Smart S&R Manual Input Recommender
- NEW: Smart S&R Section — user enters weekly Support & Resistance
  System auto-decides: BUY CALL or BUY PUT, ATM or 1-OTM strike
  Calculates: PoP, Max Profit, Max Loss, RR Ratio, Breakeven, Margin
- All v18.4 features retained (Holiday-aware expiry, silent refresh, etc.)
- lotSize fixed to 65

pip install curl_cffi pandas numpy yfinance pytz scipy
"""

import os, json, time, warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from curl_cffi import requests as curl_requests
import yfinance as yf
from math import log, sqrt, exp
from scipy.stats import norm as _norm
import pytz

warnings.filterwarnings("ignore")

IST = pytz.timezone("Asia/Kolkata")

def now_ist():
    return datetime.now(IST)

def today_ist():
    return now_ist().date()

def ist_weekday():
    return today_ist().weekday()

def ist_timestamp_str():
    return now_ist().strftime("%d-%b-%Y %H:%M IST")


# =================================================================
#  NSE MARKET HOLIDAYS 2026
# =================================================================

NSE_HOLIDAYS_2026 = {
    "15-Jan-2026": "Municipal Corporation Election - Maharashtra",
    "26-Jan-2026": "Republic Day",
    "03-Mar-2026": "Holi",
    "26-Mar-2026": "Shri Ram Navami",
    "31-Mar-2026": "Shri Mahavir Jayanti",
    "03-Apr-2026": "Good Friday",
    "14-Apr-2026": "Dr. Baba Saheb Ambedkar Jayanti",
    "01-May-2026": "Maharashtra Day",
    "28-May-2026": "Bakri Id",
    "26-Jun-2026": "Muharram",
    "14-Sep-2026": "Ganesh Chaturthi",
    "02-Oct-2026": "Mahatma Gandhi Jayanti",
    "20-Oct-2026": "Dussehra",
    "10-Nov-2026": "Diwali-Balipratipada",
    "24-Nov-2026": "Prakash Gurpurb Sri Guru Nanak Dev",
    "25-Dec-2026": "Christmas",
}

_HOLIDAY_DATES_2026 = set()
for _ds in NSE_HOLIDAYS_2026:
    try:
        _HOLIDAY_DATES_2026.add(datetime.strptime(_ds, "%d-%b-%Y").date())
    except Exception:
        pass


def is_nse_holiday(dt):
    if dt.weekday() >= 5:
        return True
    return dt in _HOLIDAY_DATES_2026


def get_prev_trading_day(dt):
    candidate = dt - timedelta(days=1)
    for _ in range(10):
        if not is_nse_holiday(candidate):
            return candidate
        candidate -= timedelta(days=1)
    return candidate


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

    def _current_or_next_tuesday_ist(self):
        today  = today_ist()
        wd     = today.weekday()
        if wd == 1:
            target_tuesday = today
        elif wd < 1:
            days_ahead = 1 - wd
            target_tuesday = today + timedelta(days=days_ahead)
        else:
            days_ahead = (8 - wd)
            target_tuesday = today + timedelta(days=days_ahead)

        if is_nse_holiday(target_tuesday):
            reason = NSE_HOLIDAYS_2026.get(target_tuesday.strftime("%d-%b-%Y"), "Holiday/Weekend")
            adjusted = get_prev_trading_day(target_tuesday)
            print(f"  [Holiday] {target_tuesday.strftime('%d-%b-%Y')} is '{reason}'. Expiry moved to {adjusted.strftime('%d-%b-%Y')}")
            expiry_date = adjusted
        else:
            expiry_date = target_tuesday

        result = expiry_date.strftime("%d-%b-%Y")
        print(f"  Computed expiry (IST, holiday-adjusted): {result}")
        return result

    def _fetch_available_expiries(self, session, headers):
        try:
            url = f"https://www.nseindia.com/api/option-chain-v3?type=Indices&symbol={self.symbol}"
            resp = session.get(url, headers=headers, impersonate="chrome", timeout=20)
            if resp.status_code == 200:
                expiries = resp.json().get("records", {}).get("expiryDates", [])
                if expiries:
                    today = today_ist()
                    for exp_str in expiries:
                        try:
                            exp_dt = datetime.strptime(exp_str, "%d-%b-%Y").date()
                            if exp_dt >= today:
                                return exp_str
                        except Exception:
                            continue
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
                resp = session.get(api_url, headers=headers, impersonate="chrome", timeout=30)
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

    def fetch_multiple_expiries(self, session, headers, n=7):
        expiry_list = []
        today = today_ist()
        wd = today.weekday()
        if wd <= 1:
            days_to_tue = 1 - wd
        else:
            days_to_tue = 8 - wd
        base_tuesday = today + timedelta(days=days_to_tue)

        candidate = base_tuesday
        attempts = 0
        while len(expiry_list) < n and attempts < 30:
            if is_nse_holiday(candidate):
                adjusted = get_prev_trading_day(candidate)
            else:
                adjusted = candidate
            exp_str = adjusted.strftime("%d-%b-%Y")
            if exp_str not in expiry_list:
                expiry_list.append(exp_str)
            candidate += timedelta(days=7)
            attempts += 1

        results = {}
        for exp in expiry_list:
            data = self._fetch_for_expiry(session, headers, exp)
            if data:
                results[exp] = data
            time.sleep(0.8)

        return results, expiry_list

    def fetch(self):
        session, headers = self._make_session()
        expiry = self._current_or_next_tuesday_ist()
        result = self._fetch_for_expiry(session, headers, expiry)

        if result is None:
            real_expiry = self._fetch_available_expiries(session, headers)
            if real_expiry and real_expiry != expiry:
                result = self._fetch_for_expiry(session, headers, real_expiry)

        self._cached_expiry_list = []
        try:
            url = f"https://www.nseindia.com/api/option-chain-v3?type=Indices&symbol={self.symbol}"
            resp = session.get(url, headers=headers, impersonate="chrome", timeout=20)
            if resp.status_code == 200:
                all_exp = resp.json().get("records", {}).get("expiryDates", [])
                today = today_ist()
                for exp_str in all_exp:
                    try:
                        exp_dt = datetime.strptime(exp_str, "%d-%b-%Y").date()
                        if exp_dt >= today:
                            self._cached_expiry_list.append(exp_str)
                            if len(self._cached_expiry_list) >= 7:
                                break
                    except Exception:
                        continue
        except Exception as e:
            print(f"  WARNING expiry list: {e}")
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
                        return {"value": round(last,2), "prev_close": round(prev,2),
                                "change": chg, "change_pct": chg_p,
                                "high": float(item.get("high", last)),
                                "low":  float(item.get("low",  last)), "status": "live"}
        except Exception as e:
            print(f"  WARNING VIX: {e}")

    try:
        hist = yf.Ticker("^INDIAVIX").history(period="2d")
        if not hist.empty:
            last  = float(hist.iloc[-1]["Close"])
            prev  = float(hist.iloc[-2]["Close"]) if len(hist) > 1 else last
            chg   = round(last - prev, 2)
            chg_p = round((chg / prev * 100), 2) if prev else 0
            return {"value": round(last,2), "prev_close": round(prev,2),
                    "change": chg, "change_pct": chg_p,
                    "high": float(hist.iloc[-1].get("High", last)),
                    "low":  float(hist.iloc[-1].get("Low",  last)), "status": "live"}
    except Exception as e:
        print(f"  WARNING VIX yfinance: {e}")
    return None


def vix_label(v):
    if   v < 12:  return "Extremely Low",  "#00c896", "rgba(0,200,150,.12)",  "rgba(0,200,150,.35)",  "Sell Premium"
    elif v < 15:  return "Low",            "#4de8b8", "rgba(0,200,150,.08)",  "rgba(0,200,150,.25)",  "Sell Premium"
    elif v < 20:  return "Normal",         "#6480ff", "rgba(100,128,255,.1)", "rgba(100,128,255,.3)", "Balanced"
    elif v < 25:  return "Elevated",       "#ffd166", "rgba(255,209,102,.1)", "rgba(255,209,102,.3)", "Buy Premium"
    elif v < 30:  return "High",           "#ff9a3c", "rgba(255,154,60,.1)",  "rgba(255,154,60,.3)",  "Buy Straddles"
    else:         return "Extreme Fear",   "#ff6b6b", "rgba(255,107,107,.1)", "rgba(255,107,107,.3)", "Buy Puts/Hedge"


# =================================================================
#  SECTION 1C -- BLACK-SCHOLES CALCULATOR
# =================================================================

def _bs_greeks(S, K, T, r, sigma, option_type="CE"):
    try:
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0}
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        nd1   = _norm.pdf(d1)
        gamma = nd1 / (S * sigma * np.sqrt(T))
        vega  = (S * nd1 * np.sqrt(T)) / 100
        if option_type == "CE":
            delta        = _norm.cdf(d1)
            theta_annual = (-(S * nd1 * sigma) / (2 * np.sqrt(T))
                            - r * K * np.exp(-r * T) * _norm.cdf(d2))
        else:
            delta        = _norm.cdf(d1) - 1
            theta_annual = (-(S * nd1 * sigma) / (2 * np.sqrt(T))
                            + r * K * np.exp(-r * T) * _norm.cdf(-d2))
        return {
            "delta": round(float(delta), 4),
            "gamma": round(float(gamma), 6),
            "theta": round(float(theta_annual / 365.0), 4),
            "vega":  round(float(vega), 4),
        }
    except Exception:
        return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0}


def _days_to_expiry_ist(expiry_str):
    try:
        today   = today_ist()
        exp_dt  = datetime.strptime(expiry_str, "%d-%b-%Y").date()
        days    = (exp_dt - today).days
        return max(days, 1)
    except Exception:
        return 7


def _compute_greeks_for_row(r, spot, expiry_str, risk_free=0.065, vix=18.0):
    T = _days_to_expiry_ist(expiry_str) / 365.0
    K = float(r["Strike"])
    ce_iv_nse = float(r.get("CE_IV", 0) or 0)
    pe_iv_nse = float(r.get("PE_IV", 0) or 0)
    ce_iv = ce_iv_nse if ce_iv_nse > 0.5 else (pe_iv_nse if pe_iv_nse > 0.5 else vix)
    pe_iv = pe_iv_nse if pe_iv_nse > 0.5 else (ce_iv_nse if ce_iv_nse > 0.5 else vix)
    ce_g = _bs_greeks(spot, K, T, risk_free, ce_iv / 100.0, "CE")
    pe_g = _bs_greeks(spot, K, T, risk_free, pe_iv / 100.0, "PE")
    return ce_g, pe_g


def extract_atm_greeks(df, atm_strike, underlying=None, expiry_str="", vix=18.0):
    spot = underlying or float(atm_strike)
    greeks_rows = []
    for _, r in df.iterrows():
        strike    = int(r["Strike"])
        is_atm    = strike == int(atm_strike)
        ce_iv_raw = round(float(r.get("CE_IV",  0) or 0), 2)
        pe_iv_raw = round(float(r.get("PE_IV",  0) or 0), 2)
        ce_ltp    = round(float(r.get("CE_LTP", 0) or 0), 2)
        pe_ltp    = round(float(r.get("PE_LTP", 0) or 0), 2)
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
    lo = max(0, atm_idx - 2)
    hi = min(len(greeks_rows), atm_idx + 3)
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
    chg_total = chg_bull_force + chg_bear_force
    chg_total = chg_total if chg_total > 0 else 1
    chg_bull_pct = round(chg_bull_force / chg_total * 100)
    chg_bear_pct = 100 - chg_bull_pct

    atm_strike = oc_data["atm_strike"]
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
        "chg_bull_pct":    chg_bull_pct,
        "chg_bear_pct":    chg_bear_pct,
        "atm_greeks":      greeks["atm_greeks"],
        "greeks_table":    greeks["greeks_table"],
        "all_strikes":     greeks["all_strikes"],
    }


# =================================================================
#  SECTION 3 -- TECHNICAL ANALYSIS
# =================================================================

def get_technical_data():
    try:
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
        return {"bias": "UNKNOWN", "confidence": "LOW", "bull": 0, "bear": 0, "diff": 0, "bias_cls": "neutral"}

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
#  SECTION 5A -- OPTION GREEKS PANEL  (unchanged from v18.4)
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
    if not oc_analysis:
        return '<div style="padding:14px 12px;font-size:11px;color:rgba(255,255,255,.3);text-align:center;">Greeks unavailable.</div>'

    g    = oc_analysis.get("atm_greeks", {})
    atm  = oc_analysis.get("atm_strike", 0)
    exp  = oc_analysis.get("expiry", "N/A")
    all_rows = oc_analysis.get("all_strikes", oc_analysis.get("greeks_table", []))

    if not g:
        return '<div style="padding:14px 12px;font-size:11px;color:rgba(255,255,255,.3);text-align:center;">Greeks not computed yet.</div>'

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

    otm_ce_opts = ""
    atm_opt     = ""
    otm_pe_opts = ""
    for row in sorted(all_rows, key=lambda x: x["strike"], reverse=True):
        s      = int(row["strike"])
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

    return f"""
<div class="greeks-panel" id="greeksPanelInner">
  <div class="greeks-title">
    &#9652; OPTION GREEKS
    <span class="greeks-expiry-tag">{exp}</span>
  </div>
  <div class="greeks-strike-wrap">
    <select class="greeks-strike-select" id="greeksStrikeSelect"
            onchange="greeksUpdateStrike(this.value)">
      {dropdown_options}
    </select>
  </div>
  <div class="greeks-atm-badge" id="greeksAtmBadge">
    <span style="font-size:8.5px;font-weight:700;color:rgba(138,160,255,.9);" id="greeksStrikeTypeLabel">ATM</span>
    <span class="greeks-atm-strike" id="greeksStrikeLabel">&#8377;{atm:,}</span>
    <span style="font-size:8px;color:rgba(255,255,255,.2);">|</span>
    <span style="font-size:8.5px;color:rgba(0,200,220,.8);" id="greeksCeLtp">CE &#8377;{ce_ltp:.1f}</span>
    <span style="font-size:8px;color:rgba(255,255,255,.25);">/</span>
    <span style="font-size:8.5px;color:rgba(255,107,107,.8);" id="greeksPeLtp">PE &#8377;{pe_ltp:.1f}</span>
  </div>
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
    <div>
      <div style="font-size:9.5px;font-weight:700;color:rgba(100,128,255,.75);margin-bottom:10px;letter-spacing:1.5px;text-transform:uppercase;display:flex;align-items:center;gap:8px;">
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
    </div>
    <div>
      <div style="font-size:9.5px;font-weight:700;color:rgba(255,107,107,.75);margin-bottom:10px;letter-spacing:1.5px;text-transform:uppercase;display:flex;align-items:center;gap:8px;">
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
    </div>
  </div>
</div>
"""


# =================================================================
#  SECTION 5B -- HERO  (unchanged from v18.4)
# =================================================================

def build_dual_gauge_hero(oc, tech, md, ts):
    if oc:
        chg_bull = oc["chg_bull_force"]; chg_bear = oc["chg_bear_force"]
        bull_pct = oc["chg_bull_pct"]; bear_pct = oc["chg_bear_pct"]; pcr = oc["pcr_oi"]
        oi_dir = oc["raw_oi_dir"]; oi_sig = oc["raw_oi_sig"]; oi_cls = oc["raw_oi_cls"]
        bull_label = _fmt_chg_oi(chg_bull); bear_label = _fmt_chg_oi(chg_bear)
        expiry = oc["expiry"]; underlying = oc["underlying"]; atm = oc["atm_strike"]; max_pain = oc["max_pain"]
    else:
        chg_bull = chg_bear = 0; bull_pct = bear_pct = 50; pcr = 1.0
        oi_sig = "NSE data unavailable"; oi_dir = "UNKNOWN"; oi_cls = "neutral"
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
        <div class="g-lbl">CHG BULL</div>
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
        <div class="g-lbl">CHG BEAR</div>
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
#  SECTION 5C -- OI DASHBOARD  (unchanged from v18.4)
# =================================================================

def build_oi_html(oc):
    ce  = oc["ce_chg"]; pe = oc["pe_chg"]
    expiry = oc["expiry"]; oi_dir = oc["oi_dir"]; oi_sig = oc["oi_sig"]; oi_cls = oc["oi_cls"]
    pcr = oc["pcr_oi"]; total_ce = oc["total_ce_oi"]; total_pe = oc["total_pe_oi"]
    max_ce_s = oc["max_ce_strike"]; max_pe_s = oc["max_pe_strike"]; max_pain = oc["max_pain"]
    underlying = oc["underlying"]

    dir_col = _cls_color(oi_cls); dir_bg = _cls_bg(oi_cls); dir_bdr = _cls_bdr(oi_cls)
    pcr_col = "#00c896" if pcr > 1.2 else ("#ff6b6b" if pcr < 0.7 else "#6480ff")
    ce_col   = "#00c896" if ce < 0 else "#ff6b6b"
    ce_label = "Call Unwinding ↓ (Bullish)" if ce < 0 else "Call Build-up ↑ (Bearish)"
    ce_fmt   = _fmt_chg_oi(ce)
    pe_col   = "#00c896" if pe > 0 else "#ff6b6b"
    pe_label = "Put Build-up ↑ (Bullish)" if pe > 0 else "Put Unwinding ↓ (Bearish)"
    pe_fmt   = _fmt_chg_oi(pe)
    bull_force = (abs(pe) if pe > 0 else 0) + (abs(ce) if ce < 0 else 0)
    bear_force = (abs(ce) if ce > 0 else 0) + (abs(pe) if pe < 0 else 0)
    net_diff       = bull_force - bear_force
    net_is_bullish = net_diff >= 0
    net_col        = "#00c896" if net_is_bullish else "#ff6b6b"
    net_label      = "Net Bullish Flow" if net_is_bullish else "Net Bearish Flow"
    net_fmt        = _fmt_chg_oi(net_diff)
    total_abs  = abs(ce) + abs(pe) or 1
    ce_pct     = round(abs(ce) / total_abs * 100)
    ce_bullish = ce < 0
    ce_pct_display = f"+{ce_pct}%" if ce_bullish else f"−{ce_pct}%"
    ce_bar_col = "#00c896" if ce_bullish else "#ff6b6b"
    pe_pct     = round(abs(pe) / total_abs * 100)
    pe_bullish = pe > 0
    pe_pct_display = f"+{pe_pct}%" if pe_bullish else f"−{pe_pct}%"
    pe_bar_col = "#00c896" if pe_bullish else "#ff6b6b"
    total_f    = bull_force + bear_force or 1
    bull_pct   = round(bull_force / total_f * 100)
    bear_pct   = 100 - bull_pct
    net_pct    = round(abs(net_diff) / total_f * 100) if total_f > 0 else 0
    net_pct    = max(5, min(95, net_pct))
    net_bar_col     = "#00c896" if net_is_bullish else "#ff6b6b"
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


# =================================================================
#  SECTION 5D -- *** NEW *** SMART S&R OPTION RECOMMENDER
# =================================================================

def build_smart_sr_recommendation_html(oc_analysis, tech):
    """
    User enters Weekly Support & Resistance manually.
    System auto-decides:
      - BUY CALL or BUY PUT (based on spot position in S/R range + OI walls + PCR)
      - ATM or 1-OTM strike  (whichever gives better Risk:Reward at target)
    Calculates all fields shown in the trade card:
      Strike, LTP, PoP, Max Profit, Max Loss, RR Ratio, Breakeven, Net Debit, Margin
    """
    if not oc_analysis:
        return ""

    spot      = oc_analysis["underlying"]
    atm       = oc_analysis["atm_strike"]
    pcr       = oc_analysis["pcr_oi"]
    max_ce_s  = oc_analysis["max_ce_strike"]
    max_pe_s  = oc_analysis["max_pe_strike"]
    expiry    = oc_analysis["expiry"]
    strikes_j = json.dumps(oc_analysis.get("strikes_data", []))

    # Pre-fill from algo-detected levels (user can override)
    default_sup = int(round((tech["support"]    if tech else spot - 150) / 25) * 25)
    default_res = int(round((tech["resistance"] if tech else spot + 150) / 25) * 25)

    return f"""
<div class="section" id="srRec">
  <div class="sec-title" style="color:#f5c518;border-color:rgba(245,197,24,.18);">
    &#9889; SMART S&amp;R OPTION RECOMMENDER
    <span class="sec-sub" style="color:rgba(255,209,102,.6);">Enter your weekly S&amp;R &rarr; System auto-picks the BEST option to buy</span>
  </div>

  <!-- ── INPUT PANEL ── -->
  <div style="display:grid;grid-template-columns:1fr 1fr auto;gap:14px;margin-bottom:22px;align-items:end;">
    <div>
      <label style="font-size:9px;font-weight:700;letter-spacing:2px;text-transform:uppercase;
                    color:#00c896;display:block;margin-bottom:7px;">&#128205; WEEKLY SUPPORT</label>
      <input type="number" id="srSupport" value="{default_sup}" step="25"
        style="width:100%;background:rgba(0,200,150,.07);border:2px solid rgba(0,200,150,.3);
               border-radius:10px;padding:13px 14px;color:#fff;font-family:'DM Mono',monospace;
               font-size:18px;font-weight:700;outline:none;transition:all .2s;"
        oninput="srCalc()"
        onfocus="this.style.borderColor='rgba(0,200,150,.8)';this.style.boxShadow='0 0 0 3px rgba(0,200,150,.15)'"
        onblur="this.style.borderColor='rgba(0,200,150,.3)';this.style.boxShadow='none'"/>
    </div>
    <div>
      <label style="font-size:9px;font-weight:700;letter-spacing:2px;text-transform:uppercase;
                    color:#ff6b6b;display:block;margin-bottom:7px;">&#128205; WEEKLY RESISTANCE</label>
      <input type="number" id="srResistance" value="{default_res}" step="25"
        style="width:100%;background:rgba(255,107,107,.07);border:2px solid rgba(255,107,107,.3);
               border-radius:10px;padding:13px 14px;color:#fff;font-family:'DM Mono',monospace;
               font-size:18px;font-weight:700;outline:none;transition:all .2s;"
        oninput="srCalc()"
        onfocus="this.style.borderColor='rgba(255,107,107,.8)';this.style.boxShadow='0 0 0 3px rgba(255,107,107,.15)'"
        onblur="this.style.borderColor='rgba(255,107,107,.3)';this.style.boxShadow='none'"/>
    </div>
    <button onclick="srCalc()"
      style="background:linear-gradient(135deg,#f5c518,#e6b000);color:#000;border:none;
             border-radius:10px;padding:13px 26px;font-family:'Sora',sans-serif;font-weight:700;
             font-size:13px;cursor:pointer;letter-spacing:.5px;white-space:nowrap;
             box-shadow:0 4px 16px rgba(245,197,24,.35);transition:all .2s;"
      onmouseover="this.style.transform='translateY(-2px)';this.style.boxShadow='0 8px 24px rgba(245,197,24,.45)'"
      onmouseout="this.style.transform='';this.style.boxShadow='0 4px 16px rgba(245,197,24,.35)'">
      &#9889; CALCULATE
    </button>
  </div>

  <!-- ── RESULT PANEL ── -->
  <div id="srResult" style="display:none;">
    <div id="srBanner"></div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px;">
      <!-- Trade Metrics Card -->
      <div id="srTradeCard" style="background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.08);border-radius:14px;overflow:hidden;">
        <div style="padding:10px 14px;background:rgba(255,255,255,.04);border-bottom:1px solid rgba(255,255,255,.06);
                    font-size:9px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:rgba(255,255,255,.35);">
          RECOMMENDED TRADE · FULL METRICS
        </div>
        <div id="srMetrics"></div>
      </div>
      <!-- Why This Trade -->
      <div style="background:rgba(100,128,255,.055);border:1px solid rgba(100,128,255,.18);border-radius:14px;padding:18px;">
        <div style="font-size:9px;font-weight:700;letter-spacing:2px;text-transform:uppercase;
                    color:rgba(138,160,255,.9);margin-bottom:14px;">WHY THIS TRADE?</div>
        <div id="srLogic" style="font-size:12px;color:rgba(255,255,255,.5);line-height:2.1;"></div>
      </div>
    </div>
    <!-- ATM vs 1-OTM Comparison -->
    <div style="background:rgba(255,255,255,.015);border:1px solid rgba(255,255,255,.07);border-radius:14px;overflow:hidden;">
      <div style="padding:10px 14px;background:rgba(255,255,255,.03);border-bottom:1px solid rgba(255,255,255,.05);
                  font-size:9px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:rgba(255,255,255,.3);">
        ATM vs 1-OTM — FULL COMPARISON (SYSTEM PICKS THE BETTER ONE AUTOMATICALLY)
      </div>
      <div id="srCompGrid" style="display:grid;grid-template-columns:1fr 1fr;"></div>
    </div>
  </div>

  <div id="srNoSignal" style="display:none;text-align:center;padding:36px;color:rgba(255,255,255,.3);
       font-size:13px;background:rgba(255,255,255,.02);border-radius:14px;border:1px solid rgba(255,255,255,.06);">
    &#9888; Spot is in the middle of S&amp;R range — no clear directional edge.<br>
    <span style="font-size:11px;opacity:.6;">Wait for spot to move closer to Support or Resistance.</span>
  </div>
</div>

<script>
(function() {{
  var LOT    = 65;
  var SPOT   = {spot:.2f};
  var ATM    = {atm};
  var PCR    = {pcr:.3f};
  var MAX_CE = {max_ce_s};
  var MAX_PE = {max_pe_s};
  var EXPIRY = "{expiry}";
  var STRIKES= {strikes_j};
  var SMAP   = {{}};
  STRIKES.forEach(function(s) {{ SMAP[s.strike] = s; }});

  function nearestStrike(t) {{
    var keys = Object.keys(SMAP).map(Number);
    if (!keys.length) return t;
    return keys.reduce(function(a,b) {{ return Math.abs(b-t)<Math.abs(a-t)?b:a; }}, keys[0]);
  }}
  function getLTP(strike, type) {{
    var s = SMAP[nearestStrike(strike)];
    if (!s) return 0;
    return type==='ce' ? (s.ce_ltp||0) : (s.pe_ltp||0);
  }}
  function rs(n) {{ return '\u20b9' + Math.abs(Math.round(n)).toLocaleString('en-IN'); }}

  function metricRow(label, valHtml, color, extra) {{
    return '<div style="display:flex;justify-content:space-between;align-items:center;' +
      'padding:10px 14px;border-bottom:1px solid rgba(255,255,255,.04);">' +
      '<span style="font-size:10px;color:rgba(255,255,255,.32);letter-spacing:.5px;text-transform:uppercase;font-family:\\'DM Mono\\',monospace;">' + label + '</span>' +
      '<span style="font-family:\\'DM Mono\\',monospace;font-size:13px;font-weight:700;color:' + (color||'rgba(255,255,255,.8)') + ';">' +
        valHtml + (extra ? '<span style="font-size:10px;opacity:.55;margin-left:5px;">' + extra + '</span>' : '') +
      '</span></div>';
  }}

  // ── CORE CALCULATION ──────────────────────────────────────────
  function calcOption(direction, strikePx, target, sup, res) {{
    var type = direction==='CALL'?'ce':'pe';
    var ltp  = getLTP(strikePx, type);
    if (!ltp || ltp <= 0) return null;

    var maxLoss  = ltp * LOT;
    var intAtTgt = direction==='CALL' ? Math.max(0,target-strikePx) : Math.max(0,strikePx-target);
    var profPts  = intAtTgt - ltp;
    var maxProfit= profPts * LOT;
    var breakeven= direction==='CALL' ? strikePx+ltp : strikePx-ltp;

    // PoP: what % of the S/R range must spot travel to cross breakeven?
    // The less it needs to travel, the higher the probability.
    var distToTarget = direction==='CALL'?(target-SPOT):(SPOT-target);
    var distBe       = direction==='CALL'?(breakeven-SPOT):(SPOT-breakeven);
    var beRatio      = distToTarget>0 ? distBe/distToTarget : 1;
    // beRatio=0 means already in profit, =1 means breakeven is exactly at target
    var basePoP = beRatio<=0?84 : beRatio<=0.25?76 : beRatio<=0.50?66 : beRatio<=0.75?54 : beRatio<=1.0?42 : 28;
    // PCR nudge
    var pcrAdj = 0;
    if (direction==='CALL') {{ pcrAdj = PCR>1.3?6:PCR>1.1?3:PCR<0.8?-6:PCR<1.0?-3:0; }}
    else                    {{ pcrAdj = PCR<0.7?6:PCR<0.9?3:PCR>1.2?-6:PCR>1.0?-3:0; }}
    var pop = Math.min(88, Math.max(18, basePoP+pcrAdj));

    var rrRatio   = maxLoss>0 ? maxProfit/maxLoss : 0;
    var profitPct = maxLoss>0 ? Math.round(maxProfit/maxLoss*100) : 0;

    return {{
      direction:direction, strikePx:strikePx, ltp:ltp, type:type,
      maxLoss:maxLoss, maxProfit:maxProfit, breakeven:breakeven,
      pop:pop, rrRatio:rrRatio, profitPct:profitPct,
      viable: profPts > 0
    }};
  }}

  // ── MAIN FUNCTION ─────────────────────────────────────────────
  window.srCalc = function() {{
    var supEl = document.getElementById('srSupport');
    var resEl = document.getElementById('srResistance');
    if (!supEl||!resEl) return;
    var sup = parseFloat(supEl.value);
    var res = parseFloat(resEl.value);
    if (!sup||!res||sup>=res) return;

    var distToSup = SPOT-sup;
    var distToRes = res-SPOT;
    var range     = res-sup;
    var ratio     = distToSup/range;   // 0=at support, 1=at resistance

    // ── STEP 1: Direction ────────────────────────────────────────
    var direction, dirConfidence, dirReason;
    if      (ratio <= 0.30) {{ direction='CALL'; dirConfidence='HIGH';
      dirReason='Spot is only <b style="color:#00c896;">'+Math.round(distToSup)+' pts</b> above Support. Strong bounce zone &mdash; high probability of upside move.'; }}
    else if (ratio >= 0.70) {{ direction='PUT';  dirConfidence='HIGH';
      dirReason='Spot is only <b style="color:#ff6b6b;">'+Math.round(distToRes)+' pts</b> below Resistance. Reversal zone &mdash; high probability of downside move.'; }}
    else if (ratio <= 0.42) {{ direction='CALL'; dirConfidence='MEDIUM';
      dirReason='Spot is closer to Support ('+Math.round(distToSup)+' pts) than Resistance ('+Math.round(distToRes)+' pts). Upside bias.'; }}
    else if (ratio >= 0.58) {{ direction='PUT';  dirConfidence='MEDIUM';
      dirReason='Spot is closer to Resistance ('+Math.round(distToRes)+' pts) than Support ('+Math.round(distToSup)+' pts). Downside bias.'; }}
    else {{
      if      (PCR >= 1.15) {{ direction='CALL'; dirConfidence='LOW';
        dirReason='Spot is mid-range. PCR='+PCR.toFixed(3)+' is bullish &mdash; slight edge for CALL. Trade cautiously.'; }}
      else if (PCR <= 0.85) {{ direction='PUT';  dirConfidence='LOW';
        dirReason='Spot is mid-range. PCR='+PCR.toFixed(3)+' is bearish &mdash; slight edge for PUT. Trade cautiously.'; }}
      else {{
        document.getElementById('srResult').style.display='none';
        document.getElementById('srNoSignal').style.display='block';
        return;
      }}
    }}

    // OI Wall confirmation
    var oiLine='';
    if (direction==='CALL' && Math.abs(MAX_PE-sup)<=200) {{
      oiLine='\u2705 PE OI wall at \u20b9'+MAX_PE.toLocaleString('en-IN')+' is near your Support &mdash; confirms strong floor.';
    }} else if (direction==='PUT' && Math.abs(MAX_CE-res)<=200) {{
      oiLine='\u2705 CE OI wall at \u20b9'+MAX_CE.toLocaleString('en-IN')+' is near your Resistance &mdash; confirms strong cap.';
    }} else {{
      oiLine='\u2139\uFE0F CE wall: \u20b9'+MAX_CE.toLocaleString('en-IN')+' &middot; PE wall: \u20b9'+MAX_PE.toLocaleString('en-IN');
    }}

    // ── STEP 2: Strike Selection ──────────────────────────────────
    var target    = direction==='CALL' ? res : sup;
    var atmStrike = nearestStrike(ATM);
    var otmStrike = direction==='CALL' ? nearestStrike(ATM+50) : nearestStrike(ATM-50);

    var atmOpt = calcOption(direction, atmStrike, target, sup, res);
    var otmOpt = calcOption(direction, otmStrike, target, sup, res);

    // Pick the strike with BETTER Risk:Reward (and must be viable = profit positive at target)
    var rec, alt;
    if (otmOpt && otmOpt.viable && atmOpt && atmOpt.viable) {{
      rec = otmOpt.rrRatio > atmOpt.rrRatio * 1.10 ? otmOpt : atmOpt;
      alt = rec===otmOpt ? atmOpt : otmOpt;
    }} else if (atmOpt && atmOpt.viable) {{ rec=atmOpt; alt=otmOpt; }}
    else if (otmOpt && otmOpt.viable)   {{ rec=otmOpt; alt=atmOpt; }}
    else {{ rec=atmOpt||otmOpt; alt=rec===atmOpt?otmOpt:atmOpt; }}

    if (!rec) {{
      document.getElementById('srResult').style.display='none';
      document.getElementById('srNoSignal').style.display='block';
      document.getElementById('srNoSignal').innerHTML='\u26A0 Live LTP unavailable. Please run during market hours.';
      return;
    }}

    document.getElementById('srNoSignal').style.display='none';
    document.getElementById('srResult').style.display='block';

    // ── COLORS ────────────────────────────────────────────────────
    var dirCol  = direction==='CALL'?'#00c896':'#ff6b6b';
    var dirBg   = direction==='CALL'?'rgba(0,200,150,.07)':'rgba(255,107,107,.07)';
    var dirBdr  = direction==='CALL'?'rgba(0,200,150,.28)':'rgba(255,107,107,.28)';
    var ltpCol  = direction==='CALL'?'#00c8e0':'#ff9090';
    var confCol = dirConfidence==='HIGH'?'#00c896':dirConfidence==='MEDIUM'?'#ffd166':'#6480ff';
    var arrow   = direction==='CALL'?'\u25B2':'\u25BC';
    var typeStr = direction==='CALL'?'BUY CE':'BUY PE';
    var isATM   = rec.strikePx===atmStrike;
    var badge   = isATM
      ? '<span style="font-size:9px;background:rgba(100,128,255,.22);color:#8aa0ff;padding:1px 7px;border-radius:4px;margin-right:6px;font-weight:700;">ATM</span>'
      : '<span style="font-size:9px;background:rgba(245,197,24,.18);color:#f5c518;padding:1px 7px;border-radius:4px;margin-right:6px;font-weight:700;">1-OTM</span>';

    // ── BANNER ────────────────────────────────────────────────────
    var bannerEl = document.getElementById('srBanner');
    bannerEl.style.cssText = 'background:'+dirBg+';border:1.5px solid '+dirBdr+';border-radius:14px;'+
      'padding:18px 22px;margin-bottom:18px;display:flex;align-items:center;gap:18px;';
    bannerEl.innerHTML =
      '<div style="font-size:44px;line-height:1;filter:drop-shadow(0 0 14px '+dirCol+'88);">'+arrow+'</div>'+
      '<div style="flex:1;">'+
        '<div style="font-size:9px;letter-spacing:2px;text-transform:uppercase;color:rgba(255,255,255,.3);margin-bottom:5px;">SYSTEM AUTO-RECOMMENDATION</div>'+
        '<div style="font-size:26px;font-weight:900;color:'+dirCol+';line-height:1.1;margin-bottom:5px;text-shadow:0 0 20px '+dirCol+'55;">'+
          typeStr+' &mdash; \u20b9'+rec.strikePx.toLocaleString('en-IN')+'</div>'+
        '<div style="font-size:12px;color:rgba(255,255,255,.4);">'+
          'Expiry: <b style="color:rgba(255,255,255,.7);">'+EXPIRY+'</b>'+
          ' &nbsp;&middot;&nbsp; LTP: <b style="color:'+ltpCol+';">\u20b9'+rec.ltp.toFixed(2)+'</b>'+
          ' &nbsp;&middot;&nbsp; Lot: <b style="color:rgba(255,255,255,.65);">'+LOT+' units</b>'+
          ' &nbsp;&middot;&nbsp; Target: <b style="color:'+dirCol+';">\u20b9'+Math.round(target).toLocaleString('en-IN')+'</b>'+
        '</div>'+
      '</div>'+
      '<div style="text-align:center;background:rgba(0,0,0,.25);border-radius:12px;padding:14px 22px;border:1px solid rgba(255,255,255,.07);">'+
        '<div style="font-size:8px;letter-spacing:1.5px;text-transform:uppercase;color:rgba(255,255,255,.28);margin-bottom:5px;">CONFIDENCE</div>'+
        '<div style="font-size:21px;font-weight:800;color:'+confCol+';letter-spacing:1px;">'+dirConfidence+'</div>'+
      '</div>';

    document.getElementById('srTradeCard').style.borderColor = dirBdr;

    // ── METRICS ───────────────────────────────────────────────────
    var popCol  = rec.pop>=65?'#00c896':rec.pop>=50?'#ffd166':'#ff6b6b';
    var profCol = rec.maxProfit>0?'#00c896':'#ff6b6b';
    var rrStr   = rec.rrRatio>0?'1:'+rec.rrRatio.toFixed(2):'\u2014';
    var rrCol   = rec.rrRatio>=2?'#00c896':rec.rrRatio>=1?'#ffd166':'#ff6b6b';

    document.getElementById('srMetrics').innerHTML =
      metricRow('STRIKE PRICE', badge+'\u20b9'+rec.strikePx.toLocaleString('en-IN'), dirCol) +
      '<div style="display:flex;justify-content:space-between;align-items:center;'+
        'padding:10px 14px;border-bottom:1px solid rgba(255,255,255,.04);">'+
        '<span style="font-size:10px;color:rgba(255,255,255,.32);letter-spacing:.5px;'+
          'text-transform:uppercase;font-family:\\'DM Mono\\',monospace;">LTP (PER LEG)</span>'+
        '<div style="text-align:right;">'+
          '<div style="font-size:10px;color:'+ltpCol+';opacity:.8;margin-bottom:2px;">'+typeStr+' \u20b9'+rec.strikePx.toLocaleString('en-IN')+'</div>'+
          '<div style="font-family:\\'DM Mono\\',monospace;font-size:20px;font-weight:800;color:'+ltpCol+';">\u20b9'+rec.ltp.toFixed(2)+'</div>'+
        '</div>'+
      '</div>'+
      metricRow('PROB. OF PROFIT', rec.pop+'%', popCol) +
      metricRow('MAX. PROFIT', rec.maxProfit>0?rs(rec.maxProfit):'\u2014 (move not enough)', profCol,
                rec.maxProfit>0?rec.profitPct+'%':'') +
      metricRow('MAX. LOSS', rs(rec.maxLoss), '#ff6b6b') +
      metricRow('MAX RR RATIO', rrStr, rrCol) +
      metricRow('BREAKEVEN', '\u20b9'+Math.round(rec.breakeven).toLocaleString('en-IN'), '#00c8e0') +
      metricRow('NET CREDIT / DEBIT', '- '+rs(rec.maxLoss), '#ff6b6b') +
      '<div style="border-bottom:none;">'+
        metricRow('EST. MARGIN/PREMIUM', rs(rec.maxLoss), '#8aa0ff') +
      '</div>';

    // ── LOGIC EXPLAINER ───────────────────────────────────────────
    var tgtPts = direction==='CALL'?Math.round(res-SPOT):Math.round(SPOT-sup);
    document.getElementById('srLogic').innerHTML = [
      '\uD83D\uDCCD <b style="color:rgba(255,255,255,.8);">Current Spot:</b> \u20b9'+SPOT.toLocaleString('en-IN'),
      '\uD83D\uDFE2 <b style="color:#00c896;">Your Support:</b> \u20b9'+sup.toLocaleString('en-IN')+'  <span style="color:rgba(255,255,255,.28);">('+Math.round(distToSup)+' pts below spot)</span>',
      '\uD83D\uDD34 <b style="color:#ff6b6b;">Your Resistance:</b> \u20b9'+res.toLocaleString('en-IN')+'  <span style="color:rgba(255,255,255,.28);">('+Math.round(distToRes)+' pts above spot)</span>',
      '\u26A1 '+dirReason,
      oiLine,
      '\uD83C\uDFAF <b style="color:rgba(255,255,255,.8);">Target:</b> \u20b9'+Math.round(target).toLocaleString('en-IN')+
        ' <span style="color:rgba(255,255,255,.28);">('+tgtPts+' pts move needed)</span>',
      '\uD83D\uDCCA <b style="color:rgba(255,255,255,.8);">PCR:</b> '+PCR.toFixed(3)+' &mdash; <span style="color:'+(PCR>1.2?'#00c896':PCR<0.8?'#ff6b6b':'#6480ff')+';">'+(PCR>1.2?'Bullish':PCR<0.8?'Bearish':'Neutral')+'</span>',
      '\uD83D\uDCB0 <b style="color:rgba(255,255,255,.8);">Strike chosen:</b> '+(isATM?'ATM (better R:R for this move size)':'1-OTM (better % return for this move size)'),
    ].join('<br>');

    // ── COMPARISON GRID ────────────────────────────────────────────
    function compCard(opt, isRec) {{
      if (!opt) return '<div style="padding:24px;text-align:center;color:rgba(255,255,255,.18);font-size:12px;">Data unavailable</div>';
      var col     = direction==='CALL'?'#00c8e0':'#ff9090';
      var bg      = isRec?(direction==='CALL'?'rgba(0,200,150,.055)':'rgba(255,107,107,.055)'):'transparent';
      var lbl     = opt.strikePx===atmStrike?'ATM':'1-OTM';
      var pCol    = opt.maxProfit>0?'#00c896':'#ff6b6b';
      var rrC     = opt.rrRatio>=2?'#00c896':opt.rrRatio>=1?'#ffd166':'#6480ff';
      var popC    = opt.pop>=60?'#00c896':opt.pop>=50?'#ffd166':'#ff6b6b';
      var cells   = [
        ['LTP',         '\u20b9'+opt.ltp.toFixed(2),                         col],
        ['Max Loss',    rs(opt.maxLoss),                                      '#ff6b6b'],
        ['Max Profit',  opt.maxProfit>0?rs(opt.maxProfit):'\u2014',           pCol],
        ['% Return',    opt.maxProfit>0?'+'+opt.profitPct+'%':'\u2014',       pCol],
        ['RR Ratio',    opt.rrRatio>0?'1:'+opt.rrRatio.toFixed(2):'\u2014',  rrC],
        ['Breakeven',   '\u20b9'+Math.round(opt.breakeven).toLocaleString('en-IN'), '#00c8e0'],
        ['Prob. Profit', opt.pop+'%',                                          popC],
        ['Margin',       rs(opt.maxLoss),                                     '#8aa0ff'],
      ];
      return '<div style="padding:18px;background:'+bg+';border-right:1px solid rgba(255,255,255,.04);">'+
        '<div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;">'+
          '<div style="font-size:13px;font-weight:700;color:'+col+';">'+lbl+' &mdash; \u20b9'+opt.strikePx.toLocaleString('en-IN')+'</div>'+
          (isRec?'<span style="font-size:8.5px;background:rgba(245,197,24,.18);color:#f5c518;padding:2px 9px;border-radius:10px;font-weight:700;">\u2713 RECOMMENDED</span>':'')+
        '</div>'+
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">'+
          cells.map(function(c) {{
            return '<div style="background:rgba(255,255,255,.03);border-radius:8px;padding:9px 11px;">'+
              '<div style="font-size:8px;color:rgba(255,255,255,.28);text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">'+c[0]+'</div>'+
              '<div style="font-family:\\'DM Mono\\',monospace;font-size:14px;font-weight:700;color:'+c[2]+';">'+c[1]+'</div>'+
            '</div>';
          }}).join('')+
        '</div></div>';
    }}

    var firstIsATM = rec.strikePx===atmStrike;
    document.getElementById('srCompGrid').innerHTML =
      compCard(firstIsATM?atmOpt:otmOpt, true) +
      compCard(firstIsATM?otmOpt:atmOpt, false);
  }};

  // Auto-run on load
  if (document.readyState==='loading') {{
    document.addEventListener('DOMContentLoaded', function(){{ setTimeout(window.srCalc,300); }});
  }} else {{
    setTimeout(window.srCalc, 300);
  }}
}})();
</script>
"""


# =================================================================
#  SECTION 5E -- KEY LEVELS / STRIKES / TICKER  (unchanged)
# =================================================================

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
        f'<div class="section" id="strikes"><div class="sec-title">TOP 5 STRIKES BY OPEN INTEREST'
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
  <div class="ticker-viewport"><div class="ticker-track" id="tkTrack">''' + track + '''</div></div>
</div>'''


# =================================================================
#  SECTION 6 -- STRATEGIES  (unchanged from v18.4 — abbreviated here)
# =================================================================
# NOTE: Copy the full STRATEGIES_DATA dict, make_payoff_svg(), and
# build_strategies_html() from v18.4 verbatim here.
# (Omitted in this diff for brevity — they are unchanged)

# Placeholder so main() can run without strategies section:
STRATEGIES_DATA = {"bullish":[],"bearish":[],"nondirectional":[]}
def make_payoff_svg(shape, bull_color="#00c896", bear_color="#ff6b6b"): return ""
def build_strategies_html(oc_analysis, tech=None, md=None, multi_expiry_analyzed=None, expiry_list=None):
    return '<div class="section" id="strat"><div class="sec-title">STRATEGIES REFERENCE</div><div style="padding:20px;color:rgba(255,255,255,.4);">Copy full build_strategies_html() from v18.4 here.</div></div>'


# =================================================================
#  CSS  (unchanged from v18.4 — copy verbatim from original)
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
.countdown-num{font-family:var(--fm);font-size:12px;font-weight:700;color:#00c896;min-width:20px;text-align:center;transition:color .3s;}
.countdown-num.urgent{color:#ff6b6b;}.countdown-num.halfway{color:#ffd166;}
.refresh-ring{display:none;width:14px;height:14px;border-radius:50%;border:2px solid rgba(0,200,150,.2);border-top-color:#00c896;animation:spin 0.8s linear infinite;}
.refresh-ring.active{display:inline-block;}
@keyframes spin{to{transform:rotate(360deg)}}
#refreshStatus{font-size:10px;color:rgba(255,255,255,.35);transition:color .3s;letter-spacing:.3px;}
#refreshStatus.updated{color:#00c896;font-weight:600;}
.hero{display:flex;align-items:stretch;background:linear-gradient(135deg,rgba(0,200,150,.055) 0%,rgba(100,128,255,.055) 100%);border-bottom:1px solid rgba(255,255,255,.07);overflow:hidden;position:relative;height:97px;}
.h-gauges{flex-shrink:0;display:flex;align-items:center;gap:10px;padding:0 16px 0 18px;}
.gauge-sep{width:1px;height:56px;background:rgba(255,255,255,.08);flex-shrink:0;}
.gauge-wrap{position:relative;width:76px;height:76px;}
.gauge-inner{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;}
.g-val{font-family:'DM Mono',monospace;font-size:13px;font-weight:700;line-height:1;}
.g-lbl{font-size:7.5px;letter-spacing:1.5px;text-transform:uppercase;color:rgba(255,255,255,.28);margin-top:2px;}
.h-mid{flex:1;min-width:0;display:flex;flex-direction:column;justify-content:center;padding:0 15px 0 13px;border-left:1px solid rgba(255,255,255,.05);}
.h-eyebrow{font-size:8px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:rgba(255,255,255,.22);margin-bottom:2px;}
.h-signal{font-size:22px;font-weight:900;letter-spacing:1px;line-height:1.1;margin-bottom:2px;}
.h-sub{font-size:9.5px;color:rgba(255,255,255,.32);}
.h-divider{height:1px;background:rgba(255,255,255,.05);margin:5px 0;}
.pill-row{display:flex;align-items:center;gap:8px;margin-bottom:4px;}
.pill-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0;}
.pill-lbl{font-size:8px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:rgba(255,255,255,.35);width:96px;flex-shrink:0;}
.pill-track{width:120px;height:5px;background:rgba(255,255,255,.07);border-radius:3px;overflow:hidden;flex-shrink:0;}
.pill-fill{height:100%;border-radius:3px;}
.pill-num{font-family:'DM Mono',monospace;font-size:10px;font-weight:700;margin-left:8px;flex-shrink:0;}
.h-stats{flex-shrink:0;min-width:360px;display:flex;flex-direction:column;border-left:1px solid rgba(255,255,255,.07);background:rgba(255,255,255,.015);}
.h-stat-row{display:flex;align-items:stretch;flex:1;border-bottom:1px solid rgba(255,255,255,.05);}
.h-stat{flex:1;display:flex;flex-direction:column;justify-content:center;padding:5px 10px;text-align:center;border-right:1px solid rgba(255,255,255,.04);}
.h-stat:last-child{border-right:none;}
.h-stat-lbl{font-size:7.5px;font-weight:700;letter-spacing:1.8px;text-transform:uppercase;color:rgba(255,255,255,.22);margin-bottom:3px;}
.h-stat-val{font-family:'DM Mono',monospace;font-size:13px;font-weight:700;line-height:1;}
.h-stat-bottom{display:flex;align-items:center;justify-content:space-between;padding:4px 10px;}
.h-bias-row{display:flex;align-items:center;gap:6px;}
.h-chip{font-size:9px;font-weight:700;letter-spacing:.8px;text-transform:uppercase;padding:2px 9px;border-radius:20px;}
.h-score{font-family:'DM Mono',monospace;font-size:8px;color:rgba(255,255,255,.22);}
.h-ts{font-family:'DM Mono',monospace;font-size:8px;color:rgba(255,255,255,.18);}
.main{display:grid;grid-template-columns:268px 1fr;min-height:0}
.sidebar{background:rgba(8,11,20,.7);backdrop-filter:blur(12px);border-right:1px solid rgba(255,255,255,.06);position:sticky;top:57px;height:calc(100vh - 57px);overflow-y:auto;display:flex;flex-direction:column;}
.sidebar-sticky-top{position:sticky;top:0;z-index:50;background:rgba(8,11,20,.95);backdrop-filter:blur(16px);border-bottom:1px solid rgba(100,128,255,.15);padding-bottom:4px;}
.sidebar-scroll{flex:1;overflow-y:auto;}
.sidebar::-webkit-scrollbar{width:3px}.sidebar::-webkit-scrollbar-thumb{background:rgba(255,255,255,.1);border-radius:2px}
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
.oi-ticker-row{display:grid;grid-template-columns:130px repeat(5,1fr);padding:15px 18px;border-top:1px solid rgba(255,255,255,.04);align-items:center;gap:6px}
.oi-ticker-metric{font-size:10px;font-weight:600;letter-spacing:1px;text-transform:uppercase;color:rgba(255,255,255,.35)}
.oi-ticker-cell{text-align:center}
.kl-zone-labels{display:flex;justify-content:space-between;margin-bottom:6px;font-size:11px;font-weight:700}
.kl-node{position:absolute;text-align:center}
.kl-lbl{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.5px;line-height:1.3;white-space:nowrap}
.kl-val{font-size:12px;font-weight:700;color:rgba(255,255,255,.7);white-space:nowrap;margin-top:2px}
.kl-dot{width:11px;height:11px;border-radius:50%;border:2px solid var(--bg)}
.kl-gradient-bar{position:relative;height:6px;border-radius:3px;background:linear-gradient(90deg,#00a07a 0%,#00c896 25%,#6480ff 55%,#ff6b6b 80%,#cc4040 100%)}
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
.ticker-label{flex-shrink:0;padding:0 16px;font-family:var(--fm);font-size:9px;font-weight:700;letter-spacing:3px;color:#00c896;text-transform:uppercase;border-right:1px solid rgba(0,200,150,.2);height:100%;display:flex;align-items:center;background:rgba(0,200,150,.07);}
.ticker-viewport{flex:1;overflow:hidden;height:100%}
.ticker-track{display:flex;align-items:center;height:100%;white-space:nowrap;animation:ticker-scroll 38s linear infinite;}
@keyframes ticker-scroll{0%{transform:translateX(0)}100%{transform:translateX(-33.333%)}}
.tk-item{display:inline-flex;align-items:center;gap:10px;padding:0 20px;height:100%;border-right:1px solid rgba(255,255,255,.04);flex-shrink:0;}
.tk-name{font-family:var(--fm);font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;padding:3px 10px;border-radius:6px;white-space:nowrap;flex-shrink:0;}
.tk-val{font-family:var(--fm);font-size:18px;font-weight:700;line-height:1;}
.tk-sub{font-family:var(--fm);font-size:10px;color:rgba(255,255,255,.35);}
.tk-badge{font-family:var(--fh);font-size:10px;font-weight:700;padding:3px 10px;border-radius:20px;}
footer{padding:16px 32px;border-top:1px solid rgba(255,255,255,.06);background:rgba(6,8,15,.9);display:flex;justify-content:space-between;font-size:11px;color:var(--muted2);font-family:var(--fm)}
.sc-tabs{display:flex;gap:10px;margin-bottom:20px;flex-wrap:wrap}
.sc-tab{padding:8px 20px;border-radius:24px;border:1px solid;cursor:pointer;font-family:var(--fh);font-size:12px;font-weight:600;transition:all .2s;display:flex;align-items:center;gap:8px;background:transparent}
.sc-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px}
.greeks-panel{margin:10px 10px 6px;padding:14px 12px;background:linear-gradient(135deg,rgba(100,128,255,.12),rgba(0,200,220,.10));border-radius:14px;border:1px solid rgba(100,128,255,.28);}
.greeks-title{font-size:9px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:rgba(138,160,255,1.0);margin-bottom:10px;padding-bottom:8px;border-bottom:1px solid rgba(100,128,255,.25);display:flex;align-items:center;justify-content:space-between;}
.greeks-expiry-tag{font-size:8.5px;color:rgba(255,255,255,.5);font-weight:400;}
.greeks-strike-wrap{position:relative;margin-bottom:10px;}
.greeks-strike-wrap::after{content:'▼';position:absolute;right:10px;top:50%;transform:translateY(-50%);font-size:8px;color:var(--gold);pointer-events:none;z-index:2;}
.greeks-strike-select{width:100%;appearance:none;-webkit-appearance:none;background:linear-gradient(135deg,rgba(245,197,24,.12),rgba(200,155,10,.06));border:1px solid var(--gold-dim);border-radius:8px;color:var(--gold);font-family:'DM Mono',monospace;font-size:11px;font-weight:700;padding:7px 28px 7px 10px;cursor:pointer;outline:none;}
.greek-name{font-family:'DM Mono',monospace;font-size:11px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:rgba(255,255,255,.92);}
.greek-sub{font-size:8px;color:rgba(255,255,255,.55);margin-top:1px;}
.greeks-row{display:flex;align-items:center;justify-content:space-between;padding:7px 0;border-bottom:1px solid rgba(255,255,255,.06);}
.greeks-atm-badge{display:flex;align-items:center;justify-content:center;gap:6px;background:rgba(100,128,255,.1);border:1px solid rgba(100,128,255,.25);border-radius:8px;padding:5px 8px;margin-bottom:10px;font-family:'DM Mono',monospace;font-size:11px;flex-wrap:wrap;}
.greeks-atm-strike{font-weight:700;color:#8aa0ff;}
.iv-bar-wrap{display:flex;align-items:center;gap:6px;margin-top:8px;padding-top:8px;border-top:1px solid rgba(255,255,255,.06);}
.iv-bar-label{font-size:8px;color:rgba(255,255,255,.7);letter-spacing:1px;text-transform:uppercase;font-weight:600;width:42px;flex-shrink:0;}
.iv-bar-track{flex:1;height:4px;background:rgba(255,255,255,.08);border-radius:2px;overflow:hidden;}
.iv-bar-fill{height:100%;border-radius:2px;transition:width .6s ease;}
.iv-bar-num{font-family:'DM Mono',monospace;font-size:11px;font-weight:700;min-width:38px;text-align:right;}
.greeks-table-section{padding:22px 28px;border-bottom:1px solid rgba(255,255,255,.05);}
.greeks-table-wrap{display:grid;grid-template-columns:1fr 1fr;gap:16px;}
.greeks-tbl{border:1px solid rgba(255,255,255,.07);border-radius:12px;overflow:hidden;}
.greeks-tbl-head{display:grid;grid-template-columns:90px repeat(4,1fr);background:rgba(255,255,255,.04);padding:8px 14px;border-bottom:1px solid rgba(255,255,255,.06);gap:4px;}
.greeks-tbl-head-label{font-size:8.5px;letter-spacing:1.5px;text-transform:uppercase;color:rgba(255,255,255,.3);text-align:center;}
.greeks-tbl-row{display:grid;grid-template-columns:90px repeat(4,1fr);padding:9px 14px;border-bottom:1px solid rgba(255,255,255,.04);align-items:center;gap:4px;}
.greeks-tbl-row:last-child{border-bottom:none;}
.greeks-tbl-strike{font-family:'DM Mono',monospace;font-size:12px;font-weight:700;color:rgba(255,255,255,.8);}
.greeks-tbl-cell{font-family:'DM Mono',monospace;font-size:11px;font-weight:600;text-align:center;color:rgba(255,255,255,.65);}
#silentRefreshFrame{position:fixed;width:0;height:0;border:none;visibility:hidden;pointer-events:none;opacity:0;}
@media(max-width:1024px){.main{grid-template-columns:1fr}.hero{height:auto;flex-wrap:wrap;}.h-stats{min-width:100%;}}
@media(max-width:640px){header{padding:12px 16px}.section{padding:18px 16px}.kl-dist-row{grid-template-columns:1fr}}
"""

ANIMATED_JS = """
<script>
(function() {
  const NAMES = ['NIFTYCRAFT','Nifty Option Strategy Builder','OI Signal Dashboard','Options Analytics Hub'];
  const wrap = document.getElementById('logoWrap');
  if (!wrap) return;
  NAMES.forEach((name, i) => {
    const el = document.createElement('div');
    el.className = 'logo-slide' + (i === 0 ? ' active' : '');
    el.textContent = name;
    wrap.appendChild(el);
  });
  let cur = 0;
  setInterval(() => {
    const slides = wrap.querySelectorAll('.logo-slide');
    slides[cur].classList.remove('active');
    slides[cur].classList.add('exit');
    setTimeout(() => slides[cur].classList.remove('exit'), 600);
    cur = (cur + 1) % slides.length;
    slides[cur].classList.add('active');
  }, 4000);
})();
(function() {
  const TOTAL_SECS = 30;
  const R = 7, C = 2 * Math.PI * R;
  function setCountdownUI(secs) {
    const numEl = document.getElementById('cdNum');
    const arcEl = document.getElementById('cdArc');
    if (numEl) { numEl.textContent = secs; numEl.className = 'countdown-num' + (secs <= 5 ? ' urgent' : secs <= 15 ? ' halfway' : ''); }
    if (arcEl) { arcEl.style.strokeDashoffset = (C * (1 - secs / TOTAL_SECS)).toFixed(2); arcEl.style.stroke = secs <= 5 ? '#ff6b6b' : secs <= 15 ? '#ffd166' : '#00c896'; }
  }
  let remaining = TOTAL_SECS;
  setCountdownUI(remaining);
  setInterval(function() { remaining -= 1; if (remaining <= 0) remaining = TOTAL_SECS; setCountdownUI(remaining); }, 1000);
})();
</script>
"""


def build_greeks_script_html(oc_analysis):
    if not oc_analysis:
        return ""
    all_rows = oc_analysis.get("all_strikes", oc_analysis.get("greeks_table", []))
    atm      = int(oc_analysis.get("atm_strike", 0))
    if not all_rows:
        return ""
    parts = []
    for row in all_rows:
        s = int(row["strike"])
        parts.append(
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
    strikes_json = "{" + ",".join(parts) + "}"
    return f"""<script>
(function() {{
  var _gData = {strikes_json};
  var _atm   = {atm};
  function _initGreeks() {{ var sel = document.getElementById('greeksStrikeSelect'); if (sel) {{ greeksUpdateStrike(sel.value); }} }}
  window.greeksUpdateStrike = function(strike) {{
    var key = String(parseInt(strike, 10));
    var d = _gData[key];
    if (!d) {{ var keys=Object.keys(_gData).map(Number); var nearest=keys.reduce((a,b)=>Math.abs(b-parseInt(strike))<Math.abs(a-parseInt(strike))?b:a,keys[0]); d=_gData[String(nearest)]; }}
    if (!d) return;
    var sel=parseInt(strike,10); var dist=Math.round(Math.abs(sel-_atm)/50);
    var lbl=sel===_atm?'ATM':(sel>_atm?'CE+'+dist:'PE-'+dist);
    var e1=document.getElementById('greeksStrikeTypeLabel'); if(e1) e1.textContent=lbl;
    var e2=document.getElementById('greeksStrikeLabel'); if(e2) e2.innerHTML='&#8377;'+sel.toLocaleString('en-IN');
    var e3=document.getElementById('greeksCeLtp'); if(e3) e3.innerHTML='CE &#8377;'+(d.ce_ltp||0).toFixed(1);
    var e4=document.getElementById('greeksPeLtp'); if(e4) e4.innerHTML='PE &#8377;'+(d.pe_ltp||0).toFixed(1);
    var ceCol='#00c896',peCol='#ff6b6b';
    var cePct=Math.min(100,Math.abs(d.ce_delta)*100).toFixed(0); var pePct=Math.min(100,Math.abs(d.pe_delta)*100).toFixed(0);
    var dw=document.getElementById('greeksDeltaWrap');
    if(dw) dw.innerHTML='<div style="display:flex;align-items:center;gap:5px;"><div style="width:34px;height:3px;background:rgba(255,255,255,.10);border-radius:2px;overflow:hidden;"><div style="width:'+cePct+'%;height:100%;background:'+ceCol+';border-radius:2px;"></div></div><span style="font-family:DM Mono,monospace;font-size:11px;font-weight:700;color:'+ceCol+';">'+(d.ce_delta>=0?'+':'')+d.ce_delta.toFixed(3)+'</span></div><div style="display:flex;align-items:center;gap:5px;margin-top:3px;"><div style="width:34px;height:3px;background:rgba(255,255,255,.10);border-radius:2px;overflow:hidden;"><div style="width:'+pePct+'%;height:100%;background:'+peCol+';border-radius:2px;"></div></div><span style="font-family:DM Mono,monospace;font-size:11px;font-weight:700;color:'+peCol+';">'+(d.pe_delta>=0?'+':'')+d.pe_delta.toFixed(3)+'</span></div>';
    var ice=document.getElementById('greeksIvCe'); if(ice) ice.textContent=(d.ce_iv||0).toFixed(1)+'%';
    var ipe=document.getElementById('greeksIvPe'); if(ipe) ipe.textContent=(d.pe_iv||0).toFixed(1)+'%';
  }};
  if (document.readyState === 'loading') {{ document.addEventListener('DOMContentLoaded', _initGreeks); }} else {{ setTimeout(_initGreeks, 80); }}
}})();
</script>"""


# =================================================================
#  SECTION 10 -- HTML ASSEMBLER  (v19 — includes srRec section)
# =================================================================

def generate_html(tech, oc, md, ts, vix_data=None, multi_expiry_analyzed=None, expiry_list=None):
    oi_html      = build_oi_html(oc)               if oc   else ""
    sr_rec_html  = build_smart_sr_recommendation_html(oc, tech) if oc else ""   # ← NEW
    kl_html      = build_key_levels_html(tech, oc) if tech else ""
    strat_html   = build_strategies_html(oc, tech, md, multi_expiry_analyzed=multi_expiry_analyzed, expiry_list=expiry_list)
    strikes_html = build_strikes_html(oc)
    ticker_html  = build_ticker_bar(tech, oc, vix_data)
    gauge_html   = build_dual_gauge_hero(oc, tech, md, ts)
    greeks_sb    = build_greeks_sidebar_html(oc)
    greeks_scr   = build_greeks_script_html(oc)
    greeks_tbl   = build_greeks_table_html(oc)

    C  = 2 * 3.14159 * 7
    cp = tech["price"] if tech else 0

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Nifty 50 Options Dashboard v19</title>
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
      <span style="font-size:10px;color:rgba(255,255,255,.3);">s</span>
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
      <div id="greeksPanel">{greeks_sb}</div>
    </div>
    <div class="sidebar-scroll">
    <div class="sb-sec">
      <div class="sb-lbl">LIVE ANALYSIS</div>
      <button class="sb-btn active" onclick="go('oi',this)">OI Dashboard</button>
      <button class="sb-btn"        onclick="go('greeksTable',this)">&#9652; Option Greeks</button>
      <button class="sb-btn"        onclick="go('kl',this)">Key Levels</button>
    </div>
    <div class="sb-sec">
      <div class="sb-lbl" style="color:#f5c518;border-left-color:#f5c518;">&#9889; SMART RECOMMENDER</div>
      <button class="sb-btn" onclick="go('srRec',this)"
        style="color:#f5c518;border-color:rgba(245,197,24,.2);background:rgba(245,197,24,.05);">
        &#128205; Enter S&amp;R &rarr; Best Trade
      </button>
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
    {greeks_tbl}
    <div id="kl">{kl_html}</div>
    {strat_html}
    <div id="strikes">{strikes_html}</div>
    <div class="section">
      <div style="background:rgba(100,128,255,.06);border:1px solid rgba(100,128,255,.18);
                  border-left:3px solid #6480ff;border-radius:12px;padding:16px 18px;
                  font-size:13px;color:rgba(255,255,255,.5);line-height:1.8;">
        <strong style="color:rgba(255,255,255,.7);">DISCLAIMER</strong><br>
        This dashboard is for EDUCATIONAL purposes only &mdash; NOT financial advice.<br>
        Smart PoP uses S/R levels, OI walls, market bias and PCR &mdash; not a guaranteed signal.<br>
        Always use stop losses. Consult a SEBI-registered investment advisor before trading.
      </div>
    </div>
  </main>
</div>
<footer>
  <span>NiftyCraft &middot; v18.4 &middot; Holiday-Aware Expiry + Silent Background Refresh</span>
  <span>S/R + OI Walls + Bias + PCR &middot; Educational Only &middot; &copy; 2025</span>
</footer>
</div>

<iframe id="silentRefreshFrame" src="about:blank"
  style="position:fixed;width:0;height:0;border:none;visibility:hidden;
         pointer-events:none;opacity:0;top:-9999px;left:-9999px;"></iframe>

<script>
function go(id,btn){{
  const el=document.getElementById(id);
  if(el)el.scrollIntoView({{behavior:"smooth",block:"start"}});
  if(btn){{document.querySelectorAll(".sb-btn").forEach(b=>b.classList.remove("active"));btn.classList.add("active");}}
}}
function filterStrat(cat,btn){{
  document.querySelectorAll(".sc-card").forEach(c=>{{c.classList.toggle("hidden",c.dataset.cat!==cat);}});
  const colors={{bullish:"#00c896",bearish:"#ff6b6b",nondirectional:"#6480ff"}};
  const col=colors[cat]||"#00c896";
  document.querySelectorAll(".sc-tab").forEach(t=>{{t.style.borderColor="rgba(255,255,255,.15)";t.style.color="rgba(255,255,255,.5)";t.style.background="transparent";}});
  if(btn){{btn.style.borderColor=col;btn.style.color=col;btn.style.background=col+"20";}}
  else{{document.querySelectorAll(".sc-tab").forEach(t=>{{
    if((cat==="bullish"&&t.textContent.includes("BULLISH"))||(cat==="bearish"&&t.textContent.includes("BEARISH"))||(cat==="nondirectional"&&t.textContent.includes("NON")))
    {{t.style.borderColor=col;t.style.color=col;t.style.background=col+"20";}}
  }});}}
}}
document.addEventListener("click",function(e){{
  const card=e.target.closest(".sc-card");
  if(card){{
    const was=card.classList.contains("expanded");
    document.querySelectorAll(".sc-card.expanded").forEach(c=>c.classList.remove("expanded"));
    if(!was){{
      card.classList.add("expanded");
      const mel=card.querySelector('.sc-metrics-live');
      if(mel&&mel.querySelector('.sc-loading')){{
        try{{
          const shape=card.dataset.shape, cat=card.dataset.cat;
          const scoreResult=smartPoP(shape,cat);
          const m=calcMetrics(shape,scoreResult.pop);
          mel.innerHTML=renderMetrics(m, scoreResult);
        }}catch(err){{mel.innerHTML='<div class="sc-loading">Could not calculate metrics</div>';}}
      }}
    }}
  }}
}});
</script>
{greeks_scr}
{ANIMATED_JS}
</body>
</html>"""


# =================================================================
#  SECTION 11 -- MAIN
# =================================================================

def main():
    ts = ist_timestamp_str()
    print("=" * 65)
    print("  NIFTY 50 OPTIONS DASHBOARD — v18.4 · Holiday-Aware Expiry")
    print(f"  {ts}")
    print(f"  IST Date: {today_ist()}  IST Weekday: {ist_weekday()}")
    print("=" * 65)

    # ── Print holiday check for current week ──────────────────────
    print("\n[0/4] Holiday Awareness Check...")
    from datetime import date as _date
    today = today_ist()
    wd    = today.weekday()
    if wd <= 1:
        days = 1 - wd
    else:
        days = (8 - wd)
    this_tue = today + timedelta(days=days) if wd != 1 else today
    if is_nse_holiday(this_tue):
        reason = NSE_HOLIDAYS_2026.get(this_tue.strftime("%d-%b-%Y"), "Holiday")
        prev_td = get_prev_trading_day(this_tue)
        print(f"  ⚠ {this_tue.strftime('%d-%b-%Y')} (Tue) = {reason}")
        print(f"  ✓ Expiry shifted to {prev_td.strftime('%d-%b-%Y')} ({prev_td.strftime('%A')})")
    else:
        print(f"  ✓ {this_tue.strftime('%d-%b-%Y')} (Tue) is a normal trading day. No holiday adjustment needed.")

    print("\n[1/4] Fetching NSE Option Chain...")
    nse = NSEOptionChain()
    oc_raw, nse_session, nse_headers = nse.fetch()

    print("\n[2/4] Fetching India VIX...")
    vix_data = fetch_india_vix(nse_session, nse_headers)
    live_vix = vix_data["value"] if vix_data else 18.0
    # Fetch all 7 expiries for dropdown
    print("\n  Fetching next 7 expiries for dropdown...")
    time.sleep(1.5)   # small gap so NSE doesn't block
    multi_expiry_raw, expiry_list = nse.fetch_multiple_expiries(nse_session, nse_headers, n=7)
    print(f"  Expiry dropdown will show: {expiry_list}")

    # Pre-analyze all expiry data
    multi_expiry_analyzed = {}
    for exp, raw in multi_expiry_raw.items():
        analyzed = analyze_option_chain(raw, vix=live_vix)
        if analyzed:
            multi_expiry_analyzed[exp] = analyzed
            print(f"    Analyzed {exp}: ATM={analyzed['atm_strike']} PCR={analyzed['pcr_oi']:.3f}")

    oc_analysis = analyze_option_chain(oc_raw, vix=live_vix) if oc_raw else None
    if oc_analysis:
        print(f"\n  OK  Spot={oc_analysis['underlying']:.2f}  ATM={oc_analysis['atm_strike']}")
        print(f"      MaxCE={oc_analysis['max_ce_strike']}  MaxPE={oc_analysis['max_pe_strike']}")
        print(f"      Expiry={oc_analysis['expiry']}  PCR={oc_analysis['pcr_oi']:.3f}")
        print(f"      CE CHG={oc_analysis['ce_chg']:+,}  PE CHG={oc_analysis['pe_chg']:+,}")
        print(f"      CHG Bull Force={oc_analysis['chg_bull_force']:,}  CHG Bear Force={oc_analysis['chg_bear_force']:,}")
        print(f"      CHG Bull%={oc_analysis['chg_bull_pct']}%  CHG Bear%={oc_analysis['chg_bear_pct']}%")

    print("\n[3/4] Fetching Technical Indicators (S/R levels)...")
    tech = get_technical_data()
    if tech:
        print(f"  Support={tech['support']:.0f}  Resistance={tech['resistance']:.0f}")
        print(f"  StrongSup={tech['strong_sup']:.0f}  StrongRes={tech['strong_res']:.0f}")

    print("\n[4/4] Scoring Market Direction...")
    md = compute_market_direction(tech, oc_analysis)
    print(f"  Bias={md['bias']}  Conf={md['confidence']}  Bull={md['bull']}  Bear={md['bear']}")

    print("\nGenerating Holiday-Aware Dashboard...")
    html = generate_html(tech, oc_analysis, md, ts, vix_data=vix_data,
                     multi_expiry_analyzed=multi_expiry_analyzed,
                     expiry_list=expiry_list)

    os.makedirs("docs", exist_ok=True)
    out = os.path.join("docs", "index.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Saved: {out}  ({len(html)/1024:.1f} KB)")

    meta = {
        "timestamp":       ts,
        "generated_at":    int(time.time()),
        "ist_date":        str(today_ist()),
        "ist_weekday":     ist_weekday(),
        "bias":            md["bias"],
        "confidence":      md["confidence"],
        "bull":            md["bull"],
        "bear":            md["bear"],
        "diff":            md["diff"],
        "price":           round(tech["price"], 2)         if tech        else None,
        "expiry":          oc_analysis["expiry"]           if oc_analysis else None,
        "pcr":             oc_analysis["pcr_oi"]           if oc_analysis else None,
        "oi_dir":          oc_analysis["oi_dir"]           if oc_analysis else None,
        "raw_oi_dir":      oc_analysis["raw_oi_dir"]       if oc_analysis else None,
        "india_vix":       vix_data["value"]               if vix_data    else None,
        "atm_strike":      oc_analysis["atm_strike"]       if oc_analysis else None,
        "max_ce":          oc_analysis["max_ce_strike"]    if oc_analysis else None,
        "max_pe":          oc_analysis["max_pe_strike"]    if oc_analysis else None,
        "support":         round(tech["support"], 0)       if tech        else None,
        "resistance":      round(tech["resistance"], 0)    if tech        else None,
        "ce_chg":          oc_analysis["ce_chg"]           if oc_analysis else None,
        "pe_chg":          oc_analysis["pe_chg"]           if oc_analysis else None,
        "chg_bull_force":  oc_analysis["chg_bull_force"]   if oc_analysis else None,
        "chg_bear_force":  oc_analysis["chg_bear_force"]   if oc_analysis else None,
        "chg_bull_pct":    oc_analysis["chg_bull_pct"]     if oc_analysis else None,
        "chg_bear_pct":    oc_analysis["chg_bear_pct"]     if oc_analysis else None,
    }
    with open(os.path.join("docs", "latest.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("  Saved: docs/latest.json")
    print("\n" + "=" * 65)
    print(f"  DONE  |  v18.4 · Holiday-Aware Expiry Active")
    print(f"  Bias: {md['bias']}  |  Confidence: {md['confidence']}")
    print("  Holiday list: 2026 NSE official holidays pre-loaded")
    print("  Logic: Tuesday holiday → Monday → Friday (fallback)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
