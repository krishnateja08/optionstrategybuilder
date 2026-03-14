#!/usr/bin/env python3
"""
Nifty 50 Options Strategy Dashboard — GitHub Pages Generator
Aurora Borealis Theme · v22.0 · Smart Dynamic PoP Engine + Intraday P&L Simulator
- PoP now reflects: Market Bias + Support/Resistance + Max CE/PE OI walls + PCR
- lotSize fixed to 65
- Strategies ranked by smart PoP — highest PoP = best trade right now
- FIXED v18.1: All strategy legs now show actual strike prices (3-4 leg strategies)
- FIXED v18.2: Gauges now show OI CHANGE data (chg_bull_force / chg_bear_force)
- FIXED v18.3: Silent background auto-refresh — no flicker, no layout shift
               Works on both file:// and http:// protocols using hidden iframe trick
- FIXED v18.4: Holiday-aware expiry — if Tuesday is NSE holiday, expiry moves to
               previous trading day (Monday, then Friday if Monday also holiday)
- NEW v18.5: Intraday P&L Simulator — per-strategy Today P&L (Delta+Theta+Vega)
             3-tab panel: Scenarios Table · Greeks Breakdown · Live Slider
- NEW v20.0: Professional Enhancements
  1. EdgeScore + True PoP (IV-based N(d2)) shown side-by-side on every expanded card
  2. Asymmetric IV move in P&L sim — down moves spike IV 3x harder than up moves
  3. Slippage deduction on Net Credit/Debit (0.3–0.7% by leg count)
  4. Skew-adjusted Vega already correct in BS (CE uses CE_IV, PE uses PE_IV)
- NEW v21.0: Three Professional Corrections
  1. Skew-aware IV fallback — missing OTM IVs now use ATM skew proxy instead of flat VIX
  2. GEX Gamma Flip strike — added to OI analysis and displayed in Key Levels
  3. Exhaustion override in bias scoring — RSI>68 at CE wall caps bias at SIDEWAYS/CAUTION
- NEW v22.0: Three Professional Strategy Metrics on Every Card
  1. IV Percentile (IVP) — 90-day VIX percentile baked into smartPoP stratAdj
     Short premium penalised −15 when IVP<20; rewarded +8 when IVP>70
  2. Theta/Vega Ratio — displayed in Greeks tab; T/V≥0.10=well compensated
  3. Theoretical EV — (TruePop×MaxProfit)−((1−TruePop)×MaxLoss) per lot

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
#  Source: NSE India official holiday list
#  If Tuesday expiry falls on a holiday → move to Monday
#  If Monday also holiday → move to Friday
# =================================================================

NSE_HOLIDAYS_2026 = {
    # date-string: description
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

# Convert to a set of date objects for fast lookup
_HOLIDAY_DATES_2026 = set()
for _ds in NSE_HOLIDAYS_2026:
    try:
        _HOLIDAY_DATES_2026.add(datetime.strptime(_ds, "%d-%b-%Y").date())
    except Exception:
        pass


def is_nse_holiday(dt):
    """Return True if the given date is an NSE trading holiday or weekend."""
    if dt.weekday() >= 5:   # Saturday=5, Sunday=6
        return True
    return dt in _HOLIDAY_DATES_2026


def get_prev_trading_day(dt):
    """Return the nearest previous trading day (not holiday, not weekend)."""
    candidate = dt - timedelta(days=1)
    for _ in range(10):           # safety: max 10 look-back days
        if not is_nse_holiday(candidate):
            return candidate
        candidate -= timedelta(days=1)
    return candidate              # fallback (should never reach here)


# =================================================================
#  SECTION 1 -- NSE OPTION CHAIN FETCHER
# =================================================================

class NSEOptionChain:
    COOKIE_FILE    = os.path.join("docs", "nse_cookies.json")
    COOKIE_MAX_AGE = 3600 * 4      # reuse cookies for up to 4 hours

    def __init__(self):
        self.symbol = "NIFTY"

    # ── Cookie persistence helpers ────────────────────────────────
    def _save_cookies(self, session):
        """Persist session cookies to disk so the next run skips warm-up."""
        try:
            os.makedirs("docs", exist_ok=True)
            cookies = {name: value for name, value in session.cookies.items()}
            with open(self.COOKIE_FILE, "w") as f:
                json.dump({"cookies": cookies, "saved_at": time.time()}, f)
            print(f"  Cookies saved ({len(cookies)} values)")
        except Exception as e:
            print(f"  WARNING: Could not save cookies: {e}")

    def _load_cookies(self, session):
        """
        Load cookies from disk into an existing session.
        Returns True if cookies are fresh and loaded; False if stale/missing.
        """
        try:
            if not os.path.exists(self.COOKIE_FILE):
                return False
            with open(self.COOKIE_FILE, "r") as f:
                data = json.load(f)
            age = time.time() - data.get("saved_at", 0)
            if age > self.COOKIE_MAX_AGE:
                print(f"  Cached cookies are {age / 3600:.1f}h old — will refresh session")
                return False
            for name, value in data.get("cookies", {}).items():
                session.cookies.set(name, value)
            print(f"  Loaded cached cookies (age: {age / 60:.0f} min) — skipping warm-up")
            return True
        except Exception as e:
            print(f"  WARNING: Could not load cookies: {e}")
            return False

    def _make_session(self):
        headers = {
            "authority": "www.nseindia.com",
            "accept": "application/json, text/plain, */*",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
            "referer": "https://www.nseindia.com/option-chain",
            "accept-language": "en-US,en;q=0.9",
        }
        session = curl_requests.Session()

        # Try loading saved cookies first — avoids expensive warm-up on every run
        if self._load_cookies(session):
            return session, headers

        # Full warm-up: hit homepage + option-chain page to get fresh cookies
        try:
            session.get("https://www.nseindia.com/", headers=headers, impersonate="chrome", timeout=15)
            time.sleep(1.5)
            session.get("https://www.nseindia.com/option-chain", headers=headers, impersonate="chrome", timeout=15)
            time.sleep(1)
            self._save_cookies(session)   # persist so next run skips this
        except Exception as e:
            print(f"  WARNING  Session warm-up: {e}")
        return session, headers

    def _current_or_next_tuesday_ist(self):
        """
        Find the current/next Tuesday and apply holiday adjustment:
        - If Tuesday is an NSE holiday  → move to previous trading day
        - Prints a clear log of the adjustment made
        """
        today  = today_ist()
        wd     = today.weekday()        # Mon=0 … Sun=6

        # ── Find the target Tuesday ──────────────────────────────
        if wd == 1:                     # today IS Tuesday
            target_tuesday = today
        elif wd < 1:                    # Sunday/Monday — next day is Tuesday
            days_ahead = 1 - wd
            target_tuesday = today + timedelta(days=days_ahead)
        else:                           # Wed–Sat — skip to next week's Tuesday
            days_ahead = (8 - wd)       # e.g. Wed(2): 8-2=6 days → next Tue
            target_tuesday = today + timedelta(days=days_ahead)

        # ── Holiday adjustment ────────────────────────────────────
        if is_nse_holiday(target_tuesday):
            reason = NSE_HOLIDAYS_2026.get(
                target_tuesday.strftime("%d-%b-%Y"), "Holiday/Weekend"
            )
            adjusted = get_prev_trading_day(target_tuesday)
            print(
                f"  [Holiday] {target_tuesday.strftime('%d-%b-%Y')} is '{reason}'. "
                f"Expiry moved to {adjusted.strftime('%d-%b-%Y')} ({adjusted.strftime('%A')})"
            )
            expiry_date = adjusted
        else:
            expiry_date = target_tuesday

        result = expiry_date.strftime("%d-%b-%Y")
        print(f"  Computed expiry (IST, holiday-adjusted): {result}")
        return result

    def _fetch_available_expiries(self, session, headers):
        """Fallback: fetch actual expiry list from NSE and pick nearest upcoming."""
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
                                print(f"  Fallback expiry from NSE API: {exp_str}")
                                return exp_str
                        except Exception:
                            continue
                    return expiries[0]
        except Exception as e:
            print(f"  WARNING  Expiry fetch: {e}")
        return None

    def _fetch_for_expiry(self, session, headers, expiry, vix=18.0):
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

                # Dynamic strike range based on VIX — during high volatility NIFTY can
                # move 500+ pts in a session so a fixed ±500 misses the true OI walls.
                # The vix param is passed in from the caller when available.
                if vix >= 28:
                    half_range = 1000   # extreme fear: capture all meaningful strikes
                elif vix >= 20:
                    half_range = 800    # elevated vol: expand to catch OTM walls
                else:
                    half_range = 500    # normal: ±500 is sufficient

                lower_bound = underlying - half_range
                upper_bound = underlying + half_range
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
                print(f"    OK {len(df)} strikes | Spot={underlying:.0f} ATM={atm_strike} Range=±{half_range}")
                return {"expiry": expiry, "df": df, "underlying": underlying, "atm_strike": atm_strike}
            except Exception as e:
                print(f"    FAIL Attempt {attempt}: {e}")
                time.sleep(2)
        return None

    def fetch_multiple_expiries(self, session, headers, n=7, vix=18.0):
        """Fetch next n expiry dates directly from NSE API (includes weekly,
           monthly, and quarterly expiries exactly as NSE lists them)."""

        # ── Step 1: Get actual expiry list directly from NSE ───────
        expiry_list = []
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
                            expiry_list.append(exp_str)
                        if len(expiry_list) >= n:
                            break
                    except Exception:
                        continue
                print(f"  Expiry list from NSE API ({len(expiry_list)} expiries): {expiry_list}")
            else:
                print(f"  WARNING: NSE expiry list returned status {resp.status_code}")
        except Exception as e:
            print(f"  WARNING expiry list fetch: {e}")

        # Fallback: if NSE API failed, generate Tuesdays as before
        if not expiry_list:
            print("  Falling back to generated Tuesday expiry list...")
            today = today_ist()
            wd = today.weekday()
            days_to_tue = 1 - wd if wd <= 1 else 8 - wd
            candidate = today + timedelta(days=days_to_tue)
            attempts = 0
            while len(expiry_list) < n and attempts < 30:
                adjusted = get_prev_trading_day(candidate) if is_nse_holiday(candidate) else candidate
                exp_str = adjusted.strftime("%d-%b-%Y")
                if exp_str not in expiry_list:
                    expiry_list.append(exp_str)
                candidate += timedelta(days=7)
                attempts += 1

        print(f"  Final expiry list: {expiry_list}")

        # ── Step 2: Fetch option chain for each expiry ─────────────
        results = {}
        for exp in expiry_list:
            print(f"    Fetching expiry: {exp}")
            data = self._fetch_for_expiry(session, headers, exp, vix=vix)
            if data:
                results[exp] = data
                print(f"      OK: {exp}")
            else:
                print(f"      SKIP: {exp} not found on NSE (may be monthly/not listed yet)")
            time.sleep(0.8)

        print(f"  Successfully fetched {len(results)} of {len(expiry_list)} expiries")
        return results, expiry_list

    def fetch(self):
        session, headers = self._make_session()

        # ── Step 1: compute holiday-adjusted expiry ───────────────
        expiry = self._current_or_next_tuesday_ist()

        # ── Step 2: try to fetch with computed expiry ─────────────
        result = self._fetch_for_expiry(session, headers, expiry)

        # ── Step 3: fallback — ask NSE for actual expiry list ─────
        if result is None:
            print(f"  Computed expiry {expiry} not found on NSE. Trying API fallback...")
            real_expiry = self._fetch_available_expiries(session, headers)
            if real_expiry and real_expiry != expiry:
                result = self._fetch_for_expiry(session, headers, real_expiry)

        if result is None:
            print("  ERROR: Option chain fetch failed for all expiries.")
        # NOTE: _cached_expiry_list removed — expiry list is fetched inside
        # fetch_multiple_expiries() which is always called right after fetch().
        # Fetching it here too was a redundant NSE API call.
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
            print(f"  WARNING VIX source1: {e}")

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
        print(f"  WARNING VIX source3 (yfinance): {e}")
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
        # Vanna = ∂Delta/∂σ — how much delta shifts when IV moves by 1 point
        # Explains why NIFTY rallies into expiry even without news (dealers hedge vanna)
        vanna = float(-nd1 * d2 / sigma) if sigma > 0 else 0.0

        # Charm = ∂Delta/∂t (per calendar day) — delta decay due purely to time passing
        # Explains why deep ITM options lose delta faster near expiry
        _charm_denom = 2.0 * T * sigma * np.sqrt(T)
        if _charm_denom > 1e-10:
            _charm_annual = -nd1 * (2.0 * r * T - d2 * sigma * np.sqrt(T)) / _charm_denom
            charm = float(_charm_annual / 365.0)   # per calendar day
        else:
            charm = 0.0

        return {
            "delta": round(float(delta), 4),
            "gamma": round(float(gamma), 6),
            "theta": round(float(theta_annual / 365.0), 4),
            "vega":  round(float(vega), 4),
            "vanna": round(vanna, 6),
            "charm": round(charm, 6),
        }
    except Exception:
        return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "vanna": 0, "charm": 0}




# =================================================================
#  SECTION 1C-2 -- TRUE STATISTICAL POP (IV-BASED)
#  Uses Black-Scholes N(d2) — the probability that the option expires
#  out-of-the-money (worthless), which is the textbook definition of
#  Probability of Profit for a short option position.
#
#  This is SEPARATE from the EdgeScore (which is a sentiment/bias
#  composite). True PoP answers: "what does the bell curve say?"
#  EdgeScore answers: "what does the market setup say?"
#  Both are shown side-by-side on every expanded strategy card.
# =================================================================

def _true_pop_from_iv(spot, strike, T, iv_pct, option_type="PE", r=0.065):
    """
    True statistical PoP = P(option expires OTM) via Black-Scholes N(d2).

    For a SHORT PE at `strike`:  PoP = P(spot > strike at expiry) = N(d2)
    For a SHORT CE at `strike`:  PoP = P(spot < strike at expiry) = N(-d2)
    For spreads / multi-leg, this gives the PoP of the short strike leg,
    which is the binding constraint for the strategy to expire profitable.

    Returns integer percent, e.g. 68.
    """
    try:
        if T <= 0 or iv_pct <= 0 or spot <= 0 or strike <= 0:
            return 50
        sigma = iv_pct / 100.0
        d1 = (np.log(spot / strike) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == "PE":
            # Short PE profits when spot stays ABOVE strike → N(d2)
            return int(round(float(_norm.cdf(d2)) * 100))
        else:
            # Short CE profits when spot stays BELOW strike → N(-d2)
            return int(round(float(_norm.cdf(-d2)) * 100))
    except Exception:
        return 50


def _compute_true_pop_map(oc_analysis, vix=18.0):
    """
    Pre-computes True PoP (IV-based) for every strategy shape and embeds
    it in the HTML as the JS constant TRUE_POP_MAP.

    For directional strategies the relevant short strike is:
      - Bullish: short strike is OTM PE (below spot) → use PE PoP
      - Bearish: short strike is OTM CE (above spot) → use CE PoP
      - Spreads: use the short leg of the spread
      - Non-directional (straddle/condor): average of CE and PE PoP at the
        relevant short strikes
    """
    if not oc_analysis:
        return {}

    spot       = float(oc_analysis["underlying"])
    atm        = int(oc_analysis["atm_strike"])
    expiry_str = oc_analysis["expiry"]
    T          = _days_to_expiry_ist(expiry_str) / 365.0

    # Pull ATM and OTM IVs from the strikes_data list.
    # ATM IVs are computed first via a simple raw lookup so the full get_iv()
    # can reference them as the skew fallback for missing OTM IVs.
    strikes_data = oc_analysis.get("strikes_data", [])

    def _raw_strike_iv(strike, side):
        """Simple raw lookup — returns 0.0 if missing, no fallback."""
        key = "ce_iv" if side == "ce" else "pe_iv"
        for s in strikes_data:
            if s["strike"] == strike:
                val = float(s.get(key, 0) or 0)
                return val if val > 0.5 else 0.0
        return 0.0

    atm_ce_iv = _raw_strike_iv(atm, "ce") or vix
    atm_pe_iv = _raw_strike_iv(atm, "pe") or vix

    def get_iv(strike, side):
        """
        Return IV% for a given strike and side ('ce' or 'pe').

        Fallback hierarchy when NSE IV is missing or zero:
          1. Use the actual per-strike IV from strikes_data (best — skew-correct)
          2. Scale ATM IV by a simple skew approximation (wing proxy):
               OTM puts: add ~0.5 IV pts per 50-pt OTM distance  (put skew)
               OTM calls: subtract ~0.2 IV pts per 50-pt OTM distance (call skew flatter)
          3. Only use flat VIX as last resort (worst — ignores skew entirely)
        This ensures OTM puts are not underpriced in True PoP, which would
        overstate the probability of a bull put spread expiring OTM.
        """
        key = "ce_iv" if side == "ce" else "pe_iv"
        for s in strikes_data:
            if s["strike"] == strike:
                val = float(s.get(key, 0) or 0)
                if val > 0.5:
                    return val
                # Strike found but IV missing — use skew-scaled ATM IV
                atm_ref = atm_ce_iv if side == "ce" else atm_pe_iv
                if atm_ref > 0.5:
                    dist_steps = abs(strike - atm) / 50.0
                    if side == "pe":
                        # Puts have steeper skew — IV rises as we go further OTM
                        skew_adj = dist_steps * 0.5
                    else:
                        # Calls have shallower skew — IV slightly falls OTM
                        skew_adj = -dist_steps * 0.2
                    return round(max(atm_ref + skew_adj, 5.0), 2)
                return vix
        # Strike not found at all — use skew-scaled ATM IV if available
        atm_ref = atm_ce_iv if side == "ce" else atm_pe_iv
        if atm_ref > 0.5:
            dist_steps = abs(strike - atm) / 50.0
            skew_adj = dist_steps * 0.5 if side == "pe" else -dist_steps * 0.2
            return round(max(atm_ref + skew_adj, 5.0), 2)
        return vix

    def otm_strike(side, offset):
        """ATM ± offset*50, rounded to nearest 50."""
        return atm + offset * 50 if side == "ce" else atm - offset * 50

    otm1_ce   = otm_strike("ce", 1)
    otm1_pe   = otm_strike("pe", 1)
    otm2_ce   = otm_strike("ce", 2)
    otm2_pe   = otm_strike("pe", 2)
    iv_co1    = get_iv(otm1_ce, "ce")
    iv_po1    = get_iv(otm1_pe, "pe")
    iv_co2    = get_iv(otm2_ce, "ce")
    iv_po2    = get_iv(otm2_pe, "pe")

    def tp(strike, side):
        iv = get_iv(strike, side)
        opt = "CE" if side == "ce" else "PE"
        return _true_pop_from_iv(spot, strike, T, iv, opt)

    pop_map = {
        # ── Bullish ──────────────────────────────────────────────
        "long_call":        tp(atm,    "ce"),   # buyer: PoP = P(expires ITM) = 1 - N(d2)
        "short_put":        tp(atm,    "pe"),
        "bull_call_spread": tp(atm,    "pe"),   # debit spread: PoP ~ P(above lower CE)
        "bull_put_spread":  tp(atm,    "pe"),   # short higher PE: N(d2 of ATM PE)
        "call_ratio_back":  tp(otm1_ce,"ce"),
        "long_synthetic":   tp(atm,    "pe"),
        "range_forward":    tp(otm1_pe,"pe"),
        "bull_butterfly":   tp(atm,    "pe"),
        "bull_condor":      tp(atm,    "pe"),
        # ── Bearish ──────────────────────────────────────────────
        "short_call":       tp(atm,    "ce"),
        "long_put":         tp(atm,    "pe"),   # buyer: PoP = P(expires ITM)
        "bear_call_spread": tp(atm,    "ce"),
        "bear_put_spread":  tp(atm,    "ce"),
        "put_ratio_back":   tp(otm1_pe,"pe"),
        "short_synthetic":  tp(atm,    "ce"),
        "risk_reversal":    tp(otm1_pe,"pe"),
        "bear_butterfly":   tp(atm,    "ce"),
        "bear_condor":      tp(atm,    "ce"),
        # ── Non-directional ──────────────────────────────────────
        "long_straddle":    50,   # purely vol play, no directional PoP
        "short_straddle":   (tp(atm, "ce") + tp(atm, "pe")) // 2,
        "long_strangle":    50,
        "short_strangle":   (tp(otm1_ce, "ce") + tp(otm1_pe, "pe")) // 2,
        "jade_lizard":      tp(otm1_pe, "pe"),
        "reverse_jade":     tp(otm1_ce, "ce"),
        "call_ratio_spread":tp(otm1_ce, "ce"),
        "put_ratio_spread": tp(otm1_pe, "pe"),
        "batman":           (tp(atm, "ce") + tp(atm, "pe")) // 2,
        "long_iron_fly":    50,
        "short_iron_fly":   (tp(atm, "ce") + tp(atm, "pe")) // 2,
        "double_fly":       (tp(atm, "ce") + tp(atm, "pe")) // 2,
        "long_iron_condor": 50,
        "short_iron_condor":(tp(otm1_ce, "ce") + tp(otm1_pe, "pe")) // 2,
        "double_condor":    (tp(otm1_ce, "ce") + tp(otm1_pe, "pe")) // 2,
        "call_calendar":    tp(atm, "ce"),
        "put_calendar":     tp(atm, "pe"),
        "diagonal_calendar":tp(otm1_ce, "ce"),
        "call_butterfly":   (tp(atm, "ce") + tp(otm2_ce, "ce")) // 2,
        "put_butterfly":    (tp(atm, "pe") + tp(otm2_pe, "pe")) // 2,
    }

    # For long options, PoP = P(expires ITM) = 100 - short_PoP
    for shape in ("long_call", "long_put", "long_straddle", "long_strangle",
                  "long_iron_fly", "long_iron_condor"):
        if shape in pop_map and pop_map[shape] != 50:
            pop_map[shape] = 100 - pop_map[shape]

    return pop_map


# =================================================================
#  SECTION 1D -- NEWTON-RAPHSON IMPLIED VOLATILITY SOLVER
#  Used as VIX fallback: market-implied IV is always more accurate
#  than the VIX index for Black-Scholes Greek calculations.
# =================================================================

def _calc_implied_vol_nr(S, K, T, r, market_price, option_type="CE",
                         max_iter=100, tol=1e-7):
    """
    Newton-Raphson solver for implied volatility.
    Returns IV as a percentage (e.g., 18.5) or None if it fails to converge.
    Uses Brenner-Subrahmanyam approximation as the initial guess.
    """
    if T <= 0 or market_price <= 0 or S <= 0 or K <= 0:
        return None
    intrinsic = max(0.0, (S - K) if option_type == "CE" else (K - S))
    if market_price <= intrinsic + 1e-6:
        return None  # price at or below intrinsic — no time value to solve for

    # Brenner-Subrahmanyam initial guess
    sigma = float((market_price / S) * np.sqrt(2.0 * np.pi / T))
    sigma = max(0.01, min(sigma, 5.0))

    for _ in range(max_iter):
        try:
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            nd1 = _norm.pdf(d1)
            if option_type == "CE":
                price = S * _norm.cdf(d1) - K * np.exp(-r * T) * _norm.cdf(d2)
            else:
                price = K * np.exp(-r * T) * _norm.cdf(-d2) - S * _norm.cdf(-d1)
            vega_nr = S * nd1 * np.sqrt(T)        # BS vega (no /100 here)
            if vega_nr < 1e-10:
                break
            diff = price - market_price
            if abs(diff) < tol:
                break
            sigma -= diff / vega_nr
            sigma = max(0.001, min(sigma, 5.0))    # clamp 0.1%–500%
        except Exception:
            break

    if 0.001 < sigma < 4.99:
        return round(sigma * 100.0, 2)             # return as % e.g. 18.5
    return None


def _derive_atm_iv_fallback(oc_raw, risk_free=0.065):
    """
    Derive ATM implied volatility from live ATM CE + PE prices using
    Newton-Raphson when India VIX cannot be fetched.
    Returns average of CE and PE implied vols as a percentage, or None.
    """
    try:
        df          = oc_raw["df"]
        S           = float(oc_raw["underlying"])
        atm         = int(oc_raw["atm_strike"])
        expiry_str  = oc_raw["expiry"]
        T           = _days_to_expiry_ist(expiry_str) / 365.0
        if T <= 0:
            return None
        atm_row = df[df["Strike"] == atm]
        if atm_row.empty:
            # Try nearest strike
            df["_dist"] = abs(df["Strike"] - atm)
            atm_row = df.loc[df["_dist"].idxmin():df["_dist"].idxmin()]
        if atm_row.empty:
            return None
        row      = atm_row.iloc[0]
        ce_price = float(row.get("CE_LTP", 0) or 0)
        pe_price = float(row.get("PE_LTP", 0) or 0)
        ivs = []
        if ce_price > 0.5:
            iv_ce = _calc_implied_vol_nr(S, atm, T, risk_free, ce_price, "CE")
            if iv_ce:
                ivs.append(iv_ce)
        if pe_price > 0.5:
            iv_pe = _calc_implied_vol_nr(S, atm, T, risk_free, pe_price, "PE")
            if iv_pe:
                ivs.append(iv_pe)
        if ivs:
            return round(sum(ivs) / len(ivs), 2)
    except Exception as e:
        print(f"  WARNING NR-IV fallback: {e}")
    return None


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
            "ce_vanna": ce_g.get("vanna", 0.0),
            "pe_vanna": pe_g.get("vanna", 0.0),
            "ce_charm": ce_g.get("charm", 0.0),
            "pe_charm": pe_g.get("charm", 0.0),
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

def _dealer_hedge_alert(atm_greeks, days_to_expiry, atm_strike):
    """
    Detects "Dealer Pinning" — when both Vanna and Charm are elevated near expiry.

    In NIFTY specifically: IV drops when the market goes UP (negative IV-spot correlation).
    Positive Vanna means as the market rises and IV drops, dealers MUST BUY the underlying
    to re-hedge — this mechanical buying is why NIFTY drifts up into expiry with no news.

    When BOTH Vanna and Charm are high:
    - Vanna: IV is falling → dealers re-hedge delta by buying
    - Charm: time is passing → dealers re-hedge delta by buying
    Both forces push dealers to buy, causing price to "pin" near the ATM strike.
    Expiry pin is most powerful in the last 2 days.
    """
    if not atm_greeks:
        return {"active": False, "reason": "No ATM greeks available"}

    ce_vanna = abs(atm_greeks.get("ce_vanna", 0))
    pe_vanna = abs(atm_greeks.get("pe_vanna", 0))
    ce_charm = abs(atm_greeks.get("ce_charm", 0))
    pe_charm = abs(atm_greeks.get("pe_charm", 0))

    # Use the higher of CE/PE for each greek (whichever side is more active)
    vanna = max(ce_vanna, pe_vanna)
    charm = max(ce_charm, pe_charm)

    # Thresholds: Vanna > 0.05 and Charm > 0.001 are meaningful near expiry
    VANNA_THRESHOLD = 0.05
    CHARM_THRESHOLD = 0.001

    vanna_high = vanna > VANNA_THRESHOLD
    charm_high = charm > CHARM_THRESHOLD
    near_expiry = days_to_expiry <= 2   # last 2 days: pin is strongest

    if vanna_high and charm_high and near_expiry:
        severity = "STRONG"
        msg = (f"Vanna={vanna:.4f} & Charm={charm:.4f} both elevated with {days_to_expiry}d left. "
               f"Dealers are mechanically buying near ₹{atm_strike:,} — expect price to stay pinned.")
    elif vanna_high and charm_high:
        severity = "MODERATE"
        msg = (f"Vanna={vanna:.4f} & Charm={charm:.4f} elevated — dealer hedging building. "
               f"Pin force grows as expiry approaches ₹{atm_strike:,}.")
    elif vanna_high or charm_high:
        severity = "WATCH"
        which = "Vanna" if vanna_high else "Charm"
        val   = vanna if vanna_high else charm
        msg = (f"{which}={val:.4f} elevated. Only partial dealer-pin signal — "
               f"watch for the other greek to confirm near ₹{atm_strike:,}.")
    else:
        return {"active": False, "vanna": round(vanna, 4), "charm": round(charm, 4),
                "days_to_expiry": days_to_expiry, "reason": "Vanna and Charm below pin thresholds"}

    return {
        "active":         True,
        "severity":       severity,
        "message":        msg,
        "vanna":          round(vanna, 4),
        "charm":          round(charm, 4),
        "days_to_expiry": days_to_expiry,
        "pin_strike":     atm_strike,
        "nifty_note":     "NIFTY IV drops on rallies — positive Vanna causes dealers to BUY on the way up",
    }


def _detect_oi_spike(ce_chg, pe_chg):
    """
    Detects a sudden OI spike that may signal a gamma squeeze or stop-loss cascade.

    The script runs every N minutes via crontab. Between two consecutive runs, if a
    large portion of the day's total CE OI change appeared in one window, it means
    a single burst of positioning — characteristic of a gamma squeeze or forced
    stop-loss from option writers.

    Threshold: if the current run's CE/PE CHG jumped by ≥50% relative to the
    prior run's value in a single interval, flag it as a spike.
    """
    prev_ce_chg = None
    prev_pe_chg = None
    try:
        meta_path = os.path.join("docs", "latest.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                prev = json.load(f)
                prev_ce_chg = prev.get("ce_chg")
                prev_pe_chg = prev.get("pe_chg")
    except Exception:
        pass

    if prev_ce_chg is None or prev_pe_chg is None:
        return {"spike": False, "reason": "No prior run data for comparison"}

    # How much did each side change since last run?
    ce_delta = ce_chg - prev_ce_chg   # positive = more call writing this window
    pe_delta = pe_chg - prev_pe_chg   # positive = more put writing this window

    # A spike = one side moved ≥ 50% of its TOTAL day value in a single interval
    SPIKE_RATIO = 0.50
    alerts = []

    if abs(ce_chg) > 0 and abs(ce_delta) >= abs(ce_chg) * SPIKE_RATIO:
        direction = "bearish" if ce_delta > 0 else "bullish"
        alerts.append({
            "side": "CE",
            "delta": ce_delta,
            "direction": direction,
            "message": (f"CE OI spiked by {ce_delta:+,} in one window "
                        f"({abs(ce_delta)/max(abs(ce_chg),1)*100:.0f}% of day's total) — "
                        f"{'gamma squeeze / stop-loss trigger' if abs(ce_delta) > 200_000 else 'sudden positioning shift'}"),
        })

    if abs(pe_chg) > 0 and abs(pe_delta) >= abs(pe_chg) * SPIKE_RATIO:
        direction = "bullish" if pe_delta > 0 else "bearish"
        alerts.append({
            "side": "PE",
            "delta": pe_delta,
            "direction": direction,
            "message": (f"PE OI spiked by {pe_delta:+,} in one window "
                        f"({abs(pe_delta)/max(abs(pe_chg),1)*100:.0f}% of day's total) — "
                        f"{'gamma squeeze / stop-loss trigger' if abs(pe_delta) > 200_000 else 'sudden positioning shift'}"),
        })

    if alerts:
        return {
            "spike":       True,
            "alerts":      alerts,
            "prev_ce_chg": prev_ce_chg,
            "prev_pe_chg": prev_pe_chg,
        }
    return {
        "spike":       False,
        "prev_ce_chg": prev_ce_chg,
        "prev_pe_chg": prev_pe_chg,
        "reason":      "No spike detected — OI changed gradually",
    }


def _price_oi_divergence(current_price, net_oi_change):
    """
    Cross-checks price direction vs OI change to classify market participation.
    Loads previous price from docs/latest.json (saved by prior run).

    4 patterns:
      Price↑ + OI↑ = Long Buildup    (Strong Bullish — fresh money, sustainable)
      Price↑ + OI↓ = Short Covering  (Explosive — violent but unsustainable, no fresh longs)
      Price↓ + OI↑ = Short Buildup   (Strong Bearish — fresh shorts, sustainable)
      Price↓ + OI↓ = Long Unwinding  (Fragile Fall — longs exiting, no fresh sellers)
    """
    prev_price = None
    try:
        meta_path = os.path.join("docs", "latest.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                prev_price = json.load(f).get("price")
    except Exception:
        pass

    if prev_price is None or prev_price == 0:
        return {
            "label": "N/A", "signal": "No prior run data yet",
            "strength": "neutral", "cls": "neutral", "prev_price": None,
        }

    price_up = current_price > prev_price
    oi_up    = net_oi_change > 0

    if price_up and oi_up:
        return {"label": "Long Buildup",
                "signal": "Price↑ + OI↑ — Fresh longs entering, trend is strong and sustainable",
                "strength": "Strong Bullish", "cls": "bullish",
                "prev_price": round(float(prev_price), 2)}
    elif price_up and not oi_up:
        # Short covering is NOT weak — it produces the most violent, fastest rallies.
        # It is however UNSUSTAINABLE because there are no fresh longs behind it.
        # Once all shorts have covered, buying stops. Trade it hard but don't hold.
        return {"label": "Short Covering",
                "signal": "Price↑ + OI↓ — Shorts closing, explosive move but NO fresh longs (exit fast)",
                "strength": "Explosive / Unsustainable", "cls": "bullish",
                "prev_price": round(float(prev_price), 2)}
    elif not price_up and oi_up:
        return {"label": "Short Buildup",
                "signal": "Price↓ + OI↑ — Fresh shorts entering, downtrend is confirmed and sustainable",
                "strength": "Strong Bearish", "cls": "bearish",
                "prev_price": round(float(prev_price), 2)}
    else:
        # Long unwinding is a fragile, seller-less fall — longs just giving up.
        # No conviction from bears, can snap back quickly.
        return {"label": "Long Unwinding",
                "signal": "Price↓ + OI↓ — Longs exiting, NO fresh sellers (snap-back risk, shallow fall)",
                "strength": "Fragile / Unsustainable", "cls": "bearish",
                "prev_price": round(float(prev_price), 2)}

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

    # Price-OI Divergence: cross-check price direction vs net OI change
    net_oi_change = ce_chg + pe_chg
    price_oi_div  = _price_oi_divergence(oc_data["underlying"], net_oi_change)

    # OI Spike detection: did a large chunk of today's OI move happen in one window?
    oi_spike = _detect_oi_spike(ce_chg, pe_chg)

    atm_strike = oc_data["atm_strike"]
    greeks = extract_atm_greeks(df, atm_strike,
                                underlying=oc_data["underlying"],
                                expiry_str=oc_data["expiry"],
                                vix=vix)

    # Merge BS-computed greeks into strikes_data so JS STRIKE_MAP has delta/theta/vega
    _g_by_strike = {g["strike"]: g for g in greeks["all_strikes"]}
    for _sd in strikes_data:
        _g = _g_by_strike.get(_sd["strike"], {})
        _sd["ce_delta"] = round(_g.get("ce_delta", 0.50),  4)
        _sd["pe_delta"] = round(_g.get("pe_delta", -0.50), 4)
        _sd["ce_theta"] = round(_g.get("ce_theta", 0.0),   4)
        _sd["pe_theta"] = round(_g.get("pe_theta", 0.0),   4)
        _sd["ce_vega"]  = round(_g.get("ce_vega",  0.0),   4)
        _sd["pe_vega"]  = round(_g.get("pe_vega",  0.0),   4)

    # Dealer hedging alert: check if Vanna + Charm are pinning price near ATM
    days_left    = _days_to_expiry_ist(oc_data["expiry"])
    dealer_alert = _dealer_hedge_alert(greeks["atm_greeks"], days_left, atm_strike)

    # ── GEX Gamma Flip Strike ──────────────────────────────────────────────
    # GEX = Σ (OI × Gamma × Spot² × 0.01 × LotSize)
    # Convention: Call GEX is positive (dealers sold calls → they are short gamma
    # → must buy on rises). Put GEX is negative (dealers sold puts → short gamma
    # → must sell on drops). The flip strike is where total GEX crosses zero.
    # Above the flip: dealers are net long gamma → they dampen moves (mean reversion).
    # Below the flip: dealers are net short gamma → they amplify moves (trending).
    # NOTE: We assume MMs are net short all options (standard for weeklies).
    # This assumption holds well for NIFTY retail-dominated flow but is approximate.
    LOT_SIZE = 65
    gex_rows = []
    for _, row in df.iterrows():
        strike  = float(row["Strike"])
        ce_oi   = float(row["CE_OI"])
        pe_oi   = float(row["PE_OI"])
        g_row   = _g_by_strike.get(int(strike), {})
        ce_gam  = float(g_row.get("ce_gamma", 0) or 0)
        pe_gam  = float(g_row.get("pe_gamma", 0) or 0)
        spot_sq = oc_data["underlying"] ** 2 * 0.01
        ce_gex  =  ce_oi * ce_gam * spot_sq * LOT_SIZE   # positive
        pe_gex  = -pe_oi * pe_gam * spot_sq * LOT_SIZE   # negative
        gex_rows.append({"strike": int(strike), "gex": ce_gex + pe_gex})

    # Sort by strike and find the flip point (where cumulative GEX sign changes)
    gex_rows.sort(key=lambda x: x["strike"])
    gex_flip_strike = None
    total_gex = sum(r["gex"] for r in gex_rows)
    # Walk from ATM upward and downward to find crossing
    atm_idx = next((i for i, r in enumerate(gex_rows) if r["strike"] >= atm_strike), len(gex_rows) // 2)
    # Cumulative GEX from lowest strike to each level — flip is where sign changes
    cum = 0.0
    prev_cum = 0.0
    for i, r in enumerate(gex_rows):
        cum += r["gex"]
        if i > 0 and prev_cum * cum < 0:   # sign change
            gex_flip_strike = r["strike"]
            break
        prev_cum = cum
    # Fallback: use the strike with minimum absolute cumulative GEX
    if gex_flip_strike is None and gex_rows:
        cum = 0.0
        min_abs = float("inf")
        for r in gex_rows:
            cum += r["gex"]
            if abs(cum) < min_abs:
                min_abs = abs(cum)
                gex_flip_strike = r["strike"]

    gex_regime = "positive"  # above flip = dealers long gamma = dampen moves
    if gex_flip_strike and oc_data["underlying"] < gex_flip_strike:
        gex_regime = "negative"  # below flip = dealers short gamma = amplify moves

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
        "price_oi_div":    price_oi_div,
        "dealer_alert":    dealer_alert,
        "oi_spike":        oi_spike,
        "gex_flip_strike": gex_flip_strike,
        "gex_regime":      gex_regime,
        "total_gex":       round(total_gex / 1e7, 2),  # in crores for display
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

        # ── IV Percentile (IVP) ───────────────────────────────────────────
        # IVP = % of days in the last 90 trading days where India VIX was
        # LOWER than today's VIX.  IVP > 70 = IV expensive (good to sell).
        # IVP < 20 = IV cheap (avoid selling premium).
        # We use ^INDIAVIX daily close as the IV proxy.
        ivp = 50   # safe default if fetch fails
        try:
            vix_hist = yf.Ticker("^INDIAVIX").history(period="6mo")
            if not vix_hist.empty and len(vix_hist) >= 10:
                vix_closes = vix_hist["Close"].dropna().values
                today_vix  = float(vix_closes[-1])
                window     = vix_closes[-90:] if len(vix_closes) >= 90 else vix_closes
                days_below = int(np.sum(window < today_vix))
                ivp        = round(days_below / len(window) * 100)
        except Exception as e:
            print(f"  WARNING IVP calc: {e}")

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
            "ivp":         ivp,
        }
    except Exception as e:
        print(f"  ERROR Technical: {e}")
        return None


# =================================================================
#  SECTION 4 -- MARKET DIRECTION SCORING
# =================================================================

def compute_market_direction(tech, oc_analysis, live_vix=18.0):
    if not tech:
        return {"bias": "UNKNOWN", "confidence": "LOW", "bull": 0, "bear": 0, "diff": 0,
                "bias_cls": "neutral", "vix_regime": "normal", "vix_high": False,
                "vix_low": False, "sma200_filter_active": False}

    cp  = tech["price"]
    bull = bear = 0

    # ── Structural trend filter ────────────────────────────────────
    # If price is below SMA200, we are in a structural bear market.
    # Bullish signals from oscillators (RSI, MACD) are unreliable here
    # because they often produce "dead cat bounce" false positives.
    is_below_sma200 = cp < tech.get("sma200", cp)

    # ── SMA scoring (structural weight) ───────────────────────────
    for sma in ["sma20", "sma50", "sma200"]:
        if cp > tech[sma]: bull += 1
        else:               bear += 1

    # ── RSI scoring (penalized below SMA200) ──────────────────────
    rsi = tech["rsi"]
    if rsi > 70:
        bear += 1                           # overbought — always penalise
    elif rsi < 30:
        if is_below_sma200:
            bull += 1                       # oversold in downtrend = bounce risk, not reversal
        else:
            bull += 2                       # oversold in uptrend = high-confidence long signal

    # ── MACD scoring (ignored in structural bear) ─────────────────
    if tech["macd"] > tech["signal_line"]:
        if not is_below_sma200:
            bull += 1                       # MACD crossover in structural bear = unreliable
        # else: crossover below SMA200 gets 0 points — could be fake
    else:
        bear += 1

    # ── OI / PCR scoring ──────────────────────────────────────────
    if oc_analysis:
        pcr = oc_analysis["pcr_oi"]
        if   pcr > 1.2: bull += 2
        elif pcr < 0.7: bear += 2

        # ── Max Pain scoring — time-weighted by days to expiry ────
        # Max Pain is almost meaningless 5+ days before expiry (market hasn't
        # started pinning yet). It becomes highly predictive in the last 48h.
        # Weight: 0 pts (early week) → 1 pt (Mon/Tue of expiry week).
        mp           = oc_analysis["max_pain"]
        days_left    = _days_to_expiry_ist(oc_analysis["expiry"])
        mp_threshold = 100   # pts away from Max Pain to count as a signal

        if   days_left <= 1: mp_weight = 2   # expiry day / day before: strongest pin
        elif days_left <= 2: mp_weight = 2   # Mon/Tue of expiry week
        elif days_left <= 4: mp_weight = 1   # Thu/Fri prior week: moderate
        else:                mp_weight = 0   # early week: Max Pain not reliable yet

        if mp_weight > 0:
            if   cp > mp + mp_threshold: bear += mp_weight
            elif cp < mp - mp_threshold: bull += mp_weight

        # ── Max Pain shift detection (leading indicator) ──────────
        # If Max Pain has moved UP since last run while bias is neutral/sideways,
        # that is a leading bullish signal (writers repositioning higher).
        # If Max Pain moved DOWN, it is a leading bearish signal.
        prev_max_pain = None
        try:
            meta_path = os.path.join("docs", "latest.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    prev_max_pain = json.load(f).get("max_pain")
        except Exception:
            pass

        max_pain_shift = None
        if prev_max_pain and prev_max_pain != mp:
            shift = mp - prev_max_pain
            if   shift > 0:
                max_pain_shift = {"direction": "up",   "pts": shift,
                                  "signal": f"Max Pain shifted UP +{shift} pts — leading bullish signal"}
                bull += 1   # writers repositioning higher = subtle bull lean
            elif shift < 0:
                max_pain_shift = {"direction": "down", "pts": abs(shift),
                                  "signal": f"Max Pain shifted DOWN −{abs(shift)} pts — leading bearish signal"}
                bear += 1   # writers repositioning lower = subtle bear lean
    else:
        mp_weight      = 0
        max_pain_shift = None

    diff = bull - bear

    if   diff >= 3:  bias, bias_cls = "BULLISH",  "bullish"; confidence = "HIGH" if diff >= 4 else "MEDIUM"
    elif diff <= -3: bias, bias_cls = "BEARISH",  "bearish"; confidence = "HIGH" if diff <= -4 else "MEDIUM"
    else:            bias, bias_cls = "SIDEWAYS", "neutral"; confidence = "MEDIUM"

    # ── Exhaustion override ────────────────────────────────────────────────
    # The raw scoring can output BULLISH even when RSI is stretched and spot
    # is pressing against a heavy OI wall — which is an exhaustion setup, not
    # a continuation setup. We cap the bias at SIDEWAYS and flag caution.
    #
    # Rule 1 — CE Wall Exhaustion:
    #   RSI > 68 AND spot within 50pts of max CE OI strike (resistance wall)
    #   → bias capped at SIDEWAYS, flag "Exhausted / CE Wall Resistance"
    #   Rationale: heavy call writing at that strike creates a ceiling; overbought
    #   RSI confirms buyers are losing momentum right at the sellers' zone.
    #
    # Rule 2 — PE Wall Exhaustion (mirror image):
    #   RSI < 32 AND spot within 50pts of max PE OI strike (support wall)
    #   → bias capped at SIDEWAYS, flag "Exhausted / PE Wall Support"
    #   Rationale: heavy put writing = strong floor; oversold RSI = sellers
    #   exhausted right at where put writers are defending.
    exhaustion_flag = None
    if oc_analysis:
        max_ce_s = oc_analysis["max_ce_strike"]
        max_pe_s = oc_analysis["max_pe_strike"]
        rsi      = tech["rsi"]

        if rsi > 68 and 0 <= (max_ce_s - cp) <= 50:
            if bias == "BULLISH":
                bias      = "SIDEWAYS"
                bias_cls  = "neutral"
                confidence = "LOW"
                bear += 1   # adjust score to reflect caution
            exhaustion_flag = {
                "type":    "CE_WALL",
                "signal":  f"RSI {rsi:.1f} overbought at CE wall ₹{max_ce_s:,} — exhaustion risk",
                "rsi":     round(rsi, 1),
                "wall":    max_ce_s,
                "warning": "Bullish momentum likely stalling — avoid fresh long entries",
            }

        elif rsi < 32 and 0 <= (cp - max_pe_s) <= 50:
            if bias == "BEARISH":
                bias      = "SIDEWAYS"
                bias_cls  = "neutral"
                confidence = "LOW"
                bull += 1   # adjust score to reflect caution
            exhaustion_flag = {
                "type":    "PE_WALL",
                "signal":  f"RSI {rsi:.1f} oversold at PE wall ₹{max_pe_s:,} — snap-back risk",
                "rsi":     round(rsi, 1),
                "wall":    max_pe_s,
                "warning": "Bearish momentum likely stalling — avoid fresh short entries",
            }

    # ── VIX regime flags (used by JS PoP engine for strategy scoring) ─
    # High VIX (>22): favour Long Volatility (Straddles, Backspreads)
    #                 penalise Non-Directional premium-selling (Iron Condors)
    # Low  VIX (<15): favour premium-selling strategies (Iron Condors, Short Straddles)
    vix_regime = "high" if live_vix > 22 else "low" if live_vix < 15 else "normal"

    return {
        "bias":                 bias,
        "bias_cls":             bias_cls,
        "confidence":           confidence,
        "bull":                 bull,
        "bear":                 bear,
        "diff":                 diff,
        "sma200_filter_active": is_below_sma200,
        "vix_regime":           vix_regime,
        "vix_high":             live_vix > 22,
        "vix_low":              live_vix < 15,
        "max_pain_shift":       max_pain_shift,
        "mp_weight":            mp_weight,
        "exhaustion_flag":      exhaustion_flag,
    }


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

def _fmt_chg_oi(n):
    if abs(n) >= 1_000_000: return f"{'+' if n > 0 else ''}{n / 1_000_000:.1f}M"
    if abs(n) >= 1_000:     return f"{'+' if n > 0 else ''}{n / 1_000:.0f}K"
    return f"{n:+,}"


# =================================================================
#  SECTION 5A -- OPTION GREEKS PANEL
# =================================================================

def _delta_bar_html(delta_val, is_ce=True):
    pct = abs(delta_val) * 100
    col = "#00c896" if is_ce else "#ff6b6b"
    return (
        f'<div style="display:flex;align-items:center;gap:5px;">'
        f'<div style="width:34px;height:3px;background:rgba(255,255,255,.10);border-radius:2px;overflow:hidden;">'
        f'<div style="width:{pct:.0f}%;height:100%;background:{col};border-radius:2px;"></div></div>'
        f'<span style="font-family:\'DM Mono\',monospace;font-size:15.9px;font-weight:700;color:{col};">'
        f'{delta_val:+.3f}</span></div>'
    )


def build_greeks_sidebar_html(oc_analysis):
    if not oc_analysis:
        return '<div style="padding:14px 12px;font-size:15.9px;color:rgba(255,255,255,.68);text-align:center;">Greeks unavailable.</div>'

    g    = oc_analysis.get("atm_greeks", {})
    atm  = oc_analysis.get("atm_strike", 0)
    exp  = oc_analysis.get("expiry", "N/A")
    all_rows = oc_analysis.get("all_strikes", oc_analysis.get("greeks_table", []))

    if not g:
        return '<div style="padding:14px 12px;font-size:15.9px;color:rgba(255,255,255,.68);text-align:center;">Greeks not computed yet.</div>'

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
    <span style="font-size:12.3px;font-weight:700;color:rgba(138,160,255,.9);" id="greeksStrikeTypeLabel">ATM</span>
    <span class="greeks-atm-strike" id="greeksStrikeLabel">&#8377;{atm:,}</span>
    <span style="font-size:11.6px;color:rgba(255,255,255,.2);">|</span>
    <span style="font-size:12.3px;color:rgba(0,200,220,.8);" id="greeksCeLtp">CE &#8377;{ce_ltp:.1f}</span>
    <span style="font-size:11.6px;color:rgba(255,255,255,.62);">/</span>
    <span style="font-size:12.3px;color:rgba(255,107,107,.8);" id="greeksPeLtp">PE &#8377;{pe_ltp:.1f}</span>
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
        <span style="font-size:12.3px;color:rgba(0,200,220,.85);">CE</span>
        <span style="font-family:'DM Mono',monospace;font-size:18.8px;font-weight:700;color:#00c8e0;" id="greeksIvCe">{ce_iv:.1f}%</span>
      </div>
      <div style="display:flex;align-items:center;gap:6px;">
        <span style="font-size:12.3px;color:rgba(255,144,144,.85);">PE</span>
        <span style="font-family:'DM Mono',monospace;font-size:18.8px;font-weight:700;color:#ff9090;" id="greeksIvPe">{pe_iv:.1f}%</span>
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
        <span style="font-size:12.3px;color:rgba(0,200,220,.85);">CE</span>
        <span style="font-family:'DM Mono',monospace;font-size:15.9px;font-weight:700;color:#ff9090;" id="greeksThetaCe">{tfmt(ce_theta)}</span>
      </div>
      <div style="display:flex;align-items:center;gap:6px;">
        <span style="font-size:12.3px;color:rgba(255,144,144,.85);">PE</span>
        <span style="font-family:'DM Mono',monospace;font-size:15.9px;font-weight:700;color:#ff9090;" id="greeksThetaPe">{tfmt(pe_theta)}</span>
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
        <span style="font-size:12.3px;color:rgba(0,200,220,.85);">CE</span>
        <span style="font-family:'DM Mono',monospace;font-size:15.9px;font-weight:700;color:#8aa0ff;" id="greeksVegaCe">{vfmt(ce_vega)}</span>
      </div>
      <div style="display:flex;align-items:center;gap:6px;">
        <span style="font-size:12.3px;color:rgba(255,144,144,.85);">PE</span>
        <span style="font-family:'DM Mono',monospace;font-size:15.9px;font-weight:700;color:#8aa0ff;" id="greeksVegaPe">{vfmt(pe_vega)}</span>
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
  <div style="font-size:12.3px;text-align:center;margin-top:6px;font-weight:700;letter-spacing:.5px;color:{iv_col};"
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
        return '<span style="font-size:11.6px;background:rgba(100,128,255,.25);color:#8aa0ff;padding:1px 5px;border-radius:4px;margin-left:4px;font-weight:700;">ATM</span>'

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
      <div style="font-size:13.8px;font-weight:700;color:rgba(100,128,255,.75);margin-bottom:10px;letter-spacing:1.5px;text-transform:uppercase;display:flex;align-items:center;gap:8px;">
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
      <div style="font-size:13.8px;font-weight:700;color:rgba(255,107,107,.75);margin-bottom:10px;letter-spacing:1.5px;text-transform:uppercase;display:flex;align-items:center;gap:8px;">
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
#  SECTION 5B -- HERO
# =================================================================

def _build_exhaustion_banner_html(md):
    """
    Renders an amber warning strip below the hero widget when the
    exhaustion override is active. Returns empty string when inactive.
    """
    ef = md.get("exhaustion_flag")
    if not ef:
        return ""
    is_ce = ef["type"] == "CE_WALL"
    col   = "#ffd166"
    bg    = "rgba(255,209,102,.07)"
    bdr   = "rgba(255,209,102,.25)"
    icon  = "&#9651;" if is_ce else "&#9661;"   # up/down triangle
    return (
        f'<div style="background:{bg};border-bottom:1px solid {bdr};'
        f'padding:8px 28px;display:flex;align-items:center;gap:14px;flex-wrap:wrap;">'
        f'<span style="font-size:17.4px;color:{col};">{icon}</span>'
        f'<span style="font-family:\'DM Mono\',monospace;font-size:13px;font-weight:700;'
        f'letter-spacing:1.5px;text-transform:uppercase;color:{col};">EXHAUSTION CAUTION</span>'
        f'<span style="font-size:15.2px;color:rgba(255,209,102,.85);">{ef["signal"]}</span>'
        f'<span style="font-size:14.5px;color:rgba(255,255,255,.50);margin-left:auto;">'
        f'{ef["warning"]}</span>'
        f'</div>'
    )


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
    C = 263.9
    def clamp(v, lo=10, hi=97): return max(lo, min(hi, v))
    bull_offset = C * (1 - clamp(bull_pct) / 100); bear_offset = C * (1 - clamp(bear_pct) / 100)
    oi_bar_w = clamp(bull_pct); bear_bar_w = clamp(bear_pct)
    b_arrow = "▲" if bias == "BULLISH" else ("▼" if bias == "BEARISH" else "◆")
    glow_rgb = ("0,200,150" if dir_col == "#00c896" else "255,107,107" if dir_col == "#ff6b6b" else "100,128,255")

    return f"""
<div class="hero" id="heroWidget">
  <div class="h-gauges">
    <div class="gauge-wrap">
      <svg width="100" height="100" viewBox="0 0 100 100">
        <circle cx="50" cy="50" r="42" fill="none" stroke="rgba(255,255,255,.18)" stroke-width="7"/>
        <circle cx="50" cy="50" r="42" fill="none" stroke="url(#bull-g)" stroke-width="7"
          stroke-linecap="round" stroke-dasharray="{C}" stroke-dashoffset="{bull_offset:.1f}"
          style="transform:rotate(-90deg);transform-origin:50px 50px;transition:stroke-dashoffset 1s ease;"/>
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
      <svg width="100" height="100" viewBox="0 0 100 100">
        <circle cx="50" cy="50" r="42" fill="none" stroke="rgba(255,255,255,.18)" stroke-width="7"/>
        <circle cx="50" cy="50" r="42" fill="none" stroke="url(#bear-g)" stroke-width="7"
          stroke-linecap="round" stroke-dasharray="{C}" stroke-dashoffset="{bear_offset:.1f}"
          style="transform:rotate(-90deg);transform-origin:50px 50px;transition:stroke-dashoffset 1s ease;"/>
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
    <div class="h-signal" style="color:{dir_col};text-shadow:0 0 20px rgba({glow_rgb},.6),0 0 40px rgba({glow_rgb},.3);font-size:31.9px;font-weight:900;letter-spacing:1px;">{oi_dir}</div>
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
{_build_exhaustion_banner_html(md)}
"""


# =================================================================
#  SECTION 5C -- OI DASHBOARD
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
    <div style="font-size:12.3px;letter-spacing:2px;text-transform:uppercase;color:rgba(255,255,255,.65);margin-bottom:7px;">OI CHANGE DIRECTION</div>
    <div style="font-size:30.4px;font-weight:700;color:{dir_col};line-height:1.1;margin-bottom:5px;">{oi_dir}</div>
    <div style="font-size:15.2px;color:{dir_col};opacity:.7;">{oi_sig}</div>
    <div style="margin-top:10px;font-family:'DM Mono',monospace;font-size:14.5px;color:rgba(255,255,255,.68);">PCR &nbsp;<span style="color:{pcr_col};font-weight:700;">{pcr:.3f}</span></div>
  </div>
  <div style="display:flex;flex:1;align-items:stretch;">
    <div style="flex:1;display:flex;flex-direction:column;justify-content:center;padding:16px 20px;border-right:1px solid rgba(255,255,255,.05);gap:5px;">
      <div style="font-size:12.3px;letter-spacing:1.8px;text-transform:uppercase;color:rgba(255,255,255,.65);white-space:nowrap;">CE OI Change</div>
      <div style="font-family:'DM Mono',monospace;font-size:31.9px;font-weight:700;color:{ce_col};line-height:1;">{ce_fmt}</div>
      <div style="font-size:14.5px;color:rgba(255,255,255,.68);white-space:nowrap;">{ce_label}</div>
      <div style="display:flex;align-items:center;gap:8px;margin-top:3px;">
        <div style="flex:1;height:5px;background:rgba(255,255,255,.07);border-radius:3px;overflow:hidden;"><div style="width:{ce_pct}%;height:100%;border-radius:3px;background:{ce_bar_col};"></div></div>
        <div style="font-family:'DM Mono',monospace;font-size:14.5px;font-weight:700;color:{ce_bar_col};min-width:38px;text-align:right;">{ce_pct_display}</div>
      </div>
    </div>
    <div style="flex:1;display:flex;flex-direction:column;justify-content:center;padding:16px 20px;border-right:1px solid rgba(255,255,255,.05);gap:5px;">
      <div style="font-size:12.3px;letter-spacing:1.8px;text-transform:uppercase;color:rgba(255,255,255,.65);white-space:nowrap;">PE OI Change</div>
      <div style="font-family:'DM Mono',monospace;font-size:31.9px;font-weight:700;color:{pe_col};line-height:1;">{pe_fmt}</div>
      <div style="font-size:14.5px;color:rgba(255,255,255,.68);white-space:nowrap;">{pe_label}</div>
      <div style="display:flex;align-items:center;gap:8px;margin-top:3px;">
        <div style="flex:1;height:5px;background:rgba(255,255,255,.07);border-radius:3px;overflow:hidden;"><div style="width:{pe_pct}%;height:100%;border-radius:3px;background:{pe_bar_col};"></div></div>
        <div style="font-family:'DM Mono',monospace;font-size:14.5px;font-weight:700;color:{pe_bar_col};min-width:38px;text-align:right;">{pe_pct_display}</div>
      </div>
    </div>
    <div style="flex:1;display:flex;flex-direction:column;justify-content:center;padding:16px 20px;gap:5px;">
      <div style="font-size:12.3px;letter-spacing:1.8px;text-transform:uppercase;color:rgba(255,255,255,.65);white-space:nowrap;">Net OI Change</div>
      <div style="font-family:'DM Mono',monospace;font-size:31.9px;font-weight:700;color:{net_col};line-height:1;">{net_fmt}</div>
      <div style="font-size:14.5px;color:rgba(255,255,255,.68);white-space:nowrap;">{net_label}</div>
      <div style="display:flex;align-items:center;gap:8px;margin-top:3px;">
        <div style="flex:1;height:5px;background:rgba(255,255,255,.07);border-radius:3px;overflow:hidden;"><div style="width:{net_pct}%;height:100%;border-radius:3px;background:{net_bar_col};box-shadow:0 0 8px {net_bar_col}66;"></div></div>
        <div style="font-family:'DM Mono',monospace;font-size:14.5px;font-weight:700;color:{net_bar_col};min-width:38px;text-align:right;">{net_pct_display}</div>
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
        f'<div class="oi-ticker-cell" style="color:#ff6b6b;font-family:\'DM Mono\',monospace;font-weight:700;font-size:21.8px;">{total_ce:,}</div>'
        f'<div class="oi-ticker-cell" style="color:#00c896;font-family:\'DM Mono\',monospace;font-weight:700;font-size:21.8px;">{total_pe:,}</div>'
        f'<div class="oi-ticker-cell" style="color:{pcr_col};font-family:\'DM Mono\',monospace;font-weight:700;font-size:21.8px;">{pcr:.3f}</div>'
        f'<div class="oi-ticker-cell" style="color:#ff6b6b;font-family:\'DM Mono\',monospace;font-weight:700;font-size:21.8px;">&#8377;{max_ce_s:,}</div>'
        f'<div class="oi-ticker-cell" style="color:#00c896;font-family:\'DM Mono\',monospace;font-weight:700;font-size:21.8px;">&#8377;{max_pe_s:,}</div>'
        f'</div>'
        f'<div style="display:flex;align-items:center;justify-content:space-between;padding:10px 18px;border-top:1px solid rgba(255,255,255,.04);flex-wrap:wrap;gap:10px;">'
        f'<div style="display:flex;align-items:center;gap:10px;">'
        f'<span style="font-size:13px;letter-spacing:1.5px;text-transform:uppercase;color:rgba(255,255,255,.68);">MAX PAIN</span>'
        f'<span style="font-family:\'DM Mono\',monospace;font-size:26.1px;font-weight:700;color:#6480ff;">&#8377;{max_pain:,}</span>'
        f'<span style="font-size:14.5px;color:rgba(100,128,255,.6);">Option writers\' target</span></div>'
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

    # Max Pain node
    mp_html = ""
    if oc:
        mp_p = pct(oc["max_pain"]); mp = oc["max_pain"]
        mp_html = (f'<div class="kl-node" style="left:{mp_p}%;top:0;transform:translateX(-50%);">'
                   f'<div class="kl-dot" style="background:#6480ff;box-shadow:0 0 8px rgba(100,128,255,.5);margin:0 auto 4px;"></div>'
                   f'<div class="kl-lbl" style="color:#6480ff;">Max Pain</div>'
                   f'<div class="kl-val" style="color:#8aa0ff;">&#8377;{mp:,}</div></div>')

    # GEX Gamma Flip node — shown in a second row below max pain
    gex_html = ""
    if oc and oc.get("gex_flip_strike"):
        gfs      = oc["gex_flip_strike"]
        regime   = oc.get("gex_regime", "positive")
        gfp      = pct(gfs)
        gex_col  = "#00c896" if regime == "positive" else "#ff6b6b"
        gex_bg   = "rgba(0,200,150,.18)" if regime == "positive" else "rgba(255,107,107,.18)"
        gex_lbl  = "GEX Flip ▲ Dampen" if regime == "positive" else "GEX Flip ▼ Amplify"
        spot_side = "above" if cp > gfs else "below"
        gex_html = (
            f'<div style="margin-top:8px;padding:10px 16px;border-radius:10px;'
            f'background:{gex_bg};border:1px solid {gex_col}44;'
            f'display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">'
            f'<div style="display:flex;align-items:center;gap:10px;">'
            f'<span style="font-size:13px;font-weight:700;letter-spacing:1.5px;'
            f'text-transform:uppercase;color:{gex_col};">&#9650; GEX GAMMA FLIP</span>'
            f'<span style="font-family:\'DM Mono\',monospace;font-size:22px;font-weight:700;'
            f'color:{gex_col};">&#8377;{gfs:,}</span>'
            f'<span style="font-size:14px;color:rgba(255,255,255,.65);">{gex_lbl}</span>'
            f'</div>'
            f'<div style="font-size:13px;color:rgba(255,255,255,.55);">'
            f'Spot is <b style="color:{gex_col};">{spot_side}</b> flip · '
            f'{"Dealers long gamma → mean-revert moves" if regime == "positive" else "Dealers short gamma → trending moves"}'
            f'</div></div>'
        )

    return (
        f'<div class="section"><div class="sec-title">KEY LEVELS'
        f'<span class="sec-sub">1H Candles · Last 120 bars · Rounded to 25</span></div>'
        f'<div class="kl-zone-labels"><span style="color:#00c896;">SUPPORT ZONE</span><span style="color:#ff6b6b;">RESISTANCE ZONE</span></div>'
        f'<div style="position:relative;height:58px;">'
        f'<div class="kl-node" style="left:3%;bottom:0;transform:translateX(-50%);"><div class="kl-lbl" style="color:#00a07a;">Strong Sup</div><div class="kl-val" style="color:#00c896;">&#8377;{ss:,.0f}</div><div class="kl-dot" style="background:#00a07a;margin:5px auto 0;"></div></div>'
        f'<div class="kl-node" style="left:22%;bottom:0;transform:translateX(-50%);"><div class="kl-lbl" style="color:#00c896;">Support</div><div class="kl-val" style="color:#4de8b8;">&#8377;{s1:,.0f}</div><div class="kl-dot" style="background:#00c896;box-shadow:0 0 8px rgba(0,200,150,.5);margin:5px auto 0;"></div></div>'
        f'<div style="position:absolute;left:{cp_pct}%;bottom:6px;transform:translateX(-50%);background:linear-gradient(90deg,#00c896,#6480ff);color:#fff;font-size:15.9px;font-weight:700;padding:3px 14px;border-radius:20px;white-space:nowrap;box-shadow:0 2px 14px rgba(0,200,150,.35);z-index:10;">NOW &#8377;{cp:,.0f}</div>'
        f'<div class="kl-node" style="left:75%;bottom:0;transform:translateX(-50%);"><div class="kl-lbl" style="color:#ff6b6b;">Resistance</div><div class="kl-val" style="color:#ff9090;">&#8377;{r1:,.0f}</div><div class="kl-dot" style="background:#ff6b6b;box-shadow:0 0 8px rgba(255,107,107,.5);margin:5px auto 0;"></div></div>'
        f'<div class="kl-node" style="left:95%;bottom:0;transform:translateX(-50%);"><div class="kl-lbl" style="color:#cc4040;">Strong Res</div><div class="kl-val" style="color:#ff6b6b;">&#8377;{sr:,.0f}</div><div class="kl-dot" style="background:#cc4040;margin:5px auto 0;"></div></div>'
        f'</div>'
        f'<div class="kl-gradient-bar"><div class="kl-price-tick" style="left:{cp_pct}%;"></div></div>'
        f'<div style="position:relative;height:54px;">{mp_html}</div>'
        f'{gex_html}'
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
#  SECTION 6 -- STRATEGIES WITH SMART POP ENGINE
# =================================================================

def make_payoff_svg(shape, bull_color="#00c896", bear_color="#ff6b6b"):
    w, h = 80, 50; mid = h // 2; pad = 8
    def pts(*coords): return " ".join(f"{x},{y}" for x, y in coords)
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
        "bear_condor":       {"profit": [(pad,mid),(28,mid),(36,pad),(50,pad),(58,mid),(72,mid)],"loss": [(pad,mid),(72,mid)]},
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
    profit_pts = pts(*s["profit"]); loss_pts = pts(*s["loss"]) if s["loss"] else ""
    def area(coords, is_p):
        if not coords: return ""
        col = bull_color if is_p else bear_color
        d = f"M {coords[0][0]},{mid} " + " ".join(f"L {x},{y}" for x, y in coords) + f" L {coords[-1][0]},{mid} Z"
        return f'<path d="{d}" fill="{col}" fill-opacity="0.18"/>'
    svg = (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">'
           f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h-pad}" stroke="rgba(255,255,255,.15)" stroke-width="1"/>'
           f'<line x1="{pad}" y1="{mid}" x2="{w-pad}" y2="{mid}" stroke="rgba(255,255,255,.15)" stroke-width="1"/>'
           f'{area(s["profit"],True)}{area(s["loss"],False)}'
           f'<polyline points="{profit_pts}" fill="none" stroke="{bull_color}" stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round"/>')
    if loss_pts:
        svg += f'<polyline points="{loss_pts}" fill="none" stroke="{bear_color}" stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round" stroke-dasharray="3,2"/>'
    return svg + '</svg>'


STRATEGIES_DATA = {
    "bullish": [
        {"name":"Long Call","shape":"long_call","risk":"Limited","reward":"Unlimited","legs":"BUY CALL (ATM)","desc":"Buy a call option. Profits as market rises above strike. Risk is limited to premium paid."},
        {"name":"Short Put","shape":"short_put","risk":"Moderate","reward":"Limited","legs":"SELL PUT (OTM)","desc":"Sell a put option below market. Collect premium. Profit if market stays above strike."},
        {"name":"Bull Call Spread","shape":"bull_call_spread","risk":"Limited","reward":"Limited","legs":"BUY CALL (Low) · SELL CALL (High)","desc":"Buy lower call, sell higher call. Reduces cost; caps profit at upper strike."},
        {"name":"Bull Put Spread","shape":"bull_put_spread","risk":"Limited","reward":"Limited","legs":"SELL PUT (High) · BUY PUT (Low)","desc":"Sell higher put, buy lower put. Credit received upfront. Profit if market stays above higher strike."},
        {"name":"Call Ratio Back Spread","shape":"call_ratio_back","risk":"Limited","reward":"Unlimited","legs":"SELL 1 CALL (Low) · BUY 2 CALLS (High)","desc":"Sell fewer calls, buy more higher calls. Benefits from a big upside move."},
        {"name":"Long Synthetic","shape":"long_synthetic","risk":"High","reward":"Unlimited","legs":"BUY CALL (ATM) · SELL PUT (ATM)","desc":"Replicates owning the underlying. Unlimited profit potential with high risk."},
        {"name":"Range Forward","shape":"range_forward","risk":"Limited","reward":"Limited","legs":"BUY CALL (High) · SELL PUT (Low)","desc":"Collar-like structure. Profit in a range. Used to hedge existing positions."},
        {"name":"Bull Butterfly","shape":"bull_butterfly","risk":"Limited","reward":"Limited","legs":"BUY Low CALL · SELL 2 Mid CALL · BUY High CALL","desc":"Max profit at middle strike. Low cost strategy for moderate bullish view."},
        {"name":"Bull Condor","shape":"bull_condor","risk":"Limited","reward":"Limited","legs":"BUY Low · SELL Mid-Low · SELL Mid-High · BUY High","desc":"Four-leg bullish strategy. Profit in a range above current price."},
    ],
    "bearish": [
        {"name":"Short Call","shape":"short_call","risk":"Unlimited","reward":"Limited","legs":"SELL CALL (ATM/OTM)","desc":"Sell a call option above market. Collect premium. Profit if market falls or stays below strike."},
        {"name":"Long Put","shape":"long_put","risk":"Limited","reward":"High","legs":"BUY PUT (ATM)","desc":"Buy a put option. Profits as market falls below strike. Risk is limited to premium paid."},
        {"name":"Bear Call Spread","shape":"bear_call_spread","risk":"Limited","reward":"Limited","legs":"SELL CALL (Low) · BUY CALL (High)","desc":"Sell lower call, buy higher call. Credit received. Profit if market stays below lower strike."},
        {"name":"Bear Put Spread","shape":"bear_put_spread","risk":"Limited","reward":"Limited","legs":"BUY PUT (High) · SELL PUT (Low)","desc":"Buy higher put, sell lower put. Cheaper bearish bet with capped profit."},
        {"name":"Put Ratio Back Spread","shape":"put_ratio_back","risk":"Limited","reward":"High","legs":"SELL 1 PUT (High) · BUY 2 PUTS (Low)","desc":"Sell fewer puts, buy more lower puts. Benefits from a big downside move."},
        {"name":"Short Synthetic","shape":"short_synthetic","risk":"High","reward":"High","legs":"SELL CALL (ATM) · BUY PUT (ATM)","desc":"Replicates shorting the underlying. Profit as market falls. High risk."},
        {"name":"Risk Reversal","shape":"risk_reversal","risk":"High","reward":"High","legs":"BUY PUT (Low) · SELL CALL (High)","desc":"Protect downside while giving up upside. Common hedging structure."},
        {"name":"Bear Butterfly","shape":"bear_butterfly","risk":"Limited","reward":"Limited","legs":"BUY Low PUT · SELL 2 Mid PUT · BUY High PUT","desc":"Max profit at middle strike. Low cost strategy for moderate bearish view."},
        {"name":"Bear Condor","shape":"bear_condor","risk":"Limited","reward":"Limited","legs":"BUY High · SELL Mid-High · SELL Mid-Low · BUY Low","desc":"Four-leg bearish strategy. Profit in a range below current price."},
    ],
    "nondirectional": [
        {"name":"Long Straddle","shape":"long_straddle","risk":"Limited","reward":"Unlimited","legs":"BUY CALL (ATM) + BUY PUT (ATM)","desc":"Buy both ATM call and put. Profit from big move in either direction. Best before events."},
        {"name":"Short Straddle","shape":"short_straddle","risk":"Unlimited","reward":"Limited","legs":"SELL CALL (ATM) + SELL PUT (ATM)","desc":"Sell both ATM call and put. Profit from low volatility. High risk unlimited loss."},
        {"name":"Long Strangle","shape":"long_strangle","risk":"Limited","reward":"Unlimited","legs":"BUY OTM CALL + BUY OTM PUT","desc":"Buy OTM call and put. Cheaper than straddle. Needs bigger move to profit."},
        {"name":"Short Strangle","shape":"short_strangle","risk":"Unlimited","reward":"Limited","legs":"SELL OTM CALL + SELL OTM PUT","desc":"Sell OTM call and put. Wider profit range than short straddle. Still high risk."},
        {"name":"Jade Lizard","shape":"jade_lizard","risk":"Limited","reward":"Limited","legs":"SELL OTM PUT + SELL CALL SPREAD","desc":"No upside risk. Collect premium. Bearish but risk-defined."},
        {"name":"Reverse Jade Lizard","shape":"reverse_jade","risk":"Limited","reward":"Limited","legs":"SELL OTM CALL + SELL PUT SPREAD","desc":"No downside risk. Collect premium. Bullish but risk-defined."},
        {"name":"Call Ratio Spread","shape":"call_ratio_spread","risk":"Unlimited","reward":"Limited","legs":"BUY 1 CALL (Low) · SELL 2 CALLS (High)","desc":"Sell more calls than bought. Credit or debit. Risk if big upside move occurs."},
        {"name":"Put Ratio Spread","shape":"put_ratio_spread","risk":"Unlimited","reward":"Limited","legs":"BUY 1 PUT (High) · SELL 2 PUTS (Low)","desc":"Sell more puts than bought. Risk if big downside move occurs."},
        {"name":"Batman Strategy","shape":"batman","risk":"Limited","reward":"Limited","legs":"BUY 2 CALLS + SELL 4 CALLS + BUY 2 CALLS","desc":"Double butterfly. Two profit peaks. Complex strategy for range-bound markets."},
        {"name":"Long Iron Fly","shape":"long_iron_fly","risk":"Limited","reward":"Limited","legs":"BUY CALL · BUY PUT · SELL ATM CALL · SELL ATM PUT","desc":"Debit iron fly. Profit from a big move. Max loss if price stays at ATM."},
        {"name":"Short Iron Fly","shape":"short_iron_fly","risk":"Limited","reward":"Limited","legs":"SELL CALL · SELL PUT · BUY OTM CALL · BUY OTM PUT","desc":"Credit iron fly. Max profit at ATM. Common non-directional strategy."},
        {"name":"Double Fly","shape":"double_fly","risk":"Limited","reward":"Limited","legs":"TWO BUTTERFLY SPREADS","desc":"Two butterfly spreads at different strikes. Two profit peaks."},
        {"name":"Long Iron Condor","shape":"long_iron_condor","risk":"Limited","reward":"Limited","legs":"BUY CALL SPREAD + BUY PUT SPREAD","desc":"Debit condor. Profit from a big move. Opposite of short iron condor."},
        {"name":"Short Iron Condor","shape":"short_iron_condor","risk":"Limited","reward":"Limited","legs":"SELL CALL SPREAD + SELL PUT SPREAD","desc":"Collect premium from both sides. Profit if price stays in a range."},
        {"name":"Double Condor","shape":"double_condor","risk":"Limited","reward":"Limited","legs":"TWO CONDOR SPREADS","desc":"Two condor spreads. Wider profit range. Complex multi-leg strategy."},
        {"name":"Call Calendar","shape":"call_calendar","risk":"Limited","reward":"Limited","legs":"SELL NEAR-TERM CALL · BUY FAR-TERM CALL","desc":"Profit from time decay difference. Best when price stays near strike."},
        {"name":"Put Calendar","shape":"put_calendar","risk":"Limited","reward":"Limited","legs":"SELL NEAR-TERM PUT · BUY FAR-TERM PUT","desc":"Profit from time decay. Best when price stays near strike on expiry."},
        {"name":"Diagonal Calendar","shape":"diagonal_calendar","risk":"Limited","reward":"Limited","legs":"SELL NEAR CALL/PUT · BUY FAR DIFF STRIKE","desc":"Calendar spread with different strikes. Combines time and price movement."},
        {"name":"Call Butterfly","shape":"call_butterfly","risk":"Limited","reward":"Limited","legs":"BUY Low CALL · SELL 2 Mid CALL · BUY High CALL","desc":"Max profit at middle strike using calls only. Low net debit strategy."},
        {"name":"Put Butterfly","shape":"put_butterfly","risk":"Limited","reward":"Limited","legs":"BUY High PUT · SELL 2 Mid PUT · BUY Low PUT","desc":"Max profit at middle strike using puts only. Low net debit strategy."},
    ],
}


def build_strategies_html(oc_analysis, tech=None, md=None, multi_expiry_analyzed=None,
                          expiry_list=None, true_pop_map=None):
    spot       = oc_analysis["underlying"]   if oc_analysis else 23000
    atm        = oc_analysis["atm_strike"]   if oc_analysis else 23000
    expiry     = oc_analysis["expiry"]       if oc_analysis else "17-Mar-2026"
    pcr        = oc_analysis["pcr_oi"]       if oc_analysis else 1.0
    mp         = oc_analysis["max_pain"]     if oc_analysis else 23000
    max_ce_s   = oc_analysis["max_ce_strike"] if oc_analysis else atm + 200
    max_pe_s   = oc_analysis["max_pe_strike"] if oc_analysis else atm - 200
    strikes_json  = json.dumps(oc_analysis.get("strikes_data", [])) if oc_analysis else "[]"
    true_pop_json = json.dumps(true_pop_map or {})

    # Build multi-expiry data for dropdown
    all_expiry_js = {}
    if multi_expiry_analyzed and expiry_list:
        for exp in expiry_list:
            oc_e = multi_expiry_analyzed.get(exp)
            if not oc_e:
                continue
            all_expiry_js[exp] = {
                "spot":        round(oc_e["underlying"], 2),
                "atm":         oc_e["atm_strike"],
                "expiry":      exp,
                "pcr":         round(oc_e["pcr_oi"], 3),
                "maxCeStrike": oc_e["max_ce_strike"],
                "maxPeStrike": oc_e["max_pe_strike"],
                "support":     round(tech["support"], 2) if tech else spot - 150,
                "resistance":  round(tech["resistance"], 2) if tech else spot + 150,
                "strongSup":   round(tech["strong_sup"], 2) if tech else spot - 300,
                "strongRes":   round(tech["strong_res"], 2) if tech else spot + 300,
                "strikes":     oc_e.get("strikes_data", []),
            }
    all_expiry_json = json.dumps(all_expiry_js)
    expiry_opts_html = ""
    if expiry_list:
        first_with_data = True
        for exp in expiry_list:
            has_data = exp in (all_expiry_js or {})
            sel = "selected" if (has_data and first_with_data) else ""
            if has_data and first_with_data:
                first_with_data = False
            # Show all expiries; disable ones with no data
            if has_data:
                expiry_opts_html += f'<option value="{exp}" {sel}>{exp}</option>\n'
            else:
                expiry_opts_html += f'<option value="{exp}" disabled style="color:rgba(255,255,255,0.3);">{exp} (no data)</option>\n'
    else:
        expiry_opts_html = f'<option value="">{oc_analysis["expiry"] if oc_analysis else "N/A"}</option>'
    
    support     = tech["support"]    if tech else spot - 150
    resistance  = tech["resistance"] if tech else spot + 150
    strong_sup  = tech["strong_sup"] if tech else spot - 300
    strong_res  = tech["strong_res"] if tech else spot + 300
    ivp         = tech["ivp"]        if tech else 50   # IV Percentile (0–100)

    bias        = md["bias"]       if md else "SIDEWAYS"
    conf        = md["confidence"] if md else "MEDIUM"
    bull_sc     = md["bull"]       if md else 4
    bear_sc     = md["bear"]       if md else 4

    def render_cards(strats, cat):
        cards = ""
        for idx, s in enumerate(strats):
            svg  = make_payoff_svg(s["shape"])
            rc   = "#00c896" if s["risk"]   in ("Limited","Low") else ("#ff6b6b" if s["risk"] in ("Unlimited","High") else "#6480ff")
            rwc  = "#00c896" if s["reward"] == "Unlimited" else "#6480ff"
            cid  = f"sc_{cat}_{idx}"
            cards += (
                f'<div class="sc-card" data-cat="{cat}" data-shape="{s["shape"]}" '
                f'data-name="{s["name"]}" data-legs="{s["legs"]}" '
                f'data-risk="{s["risk"]}" data-reward="{s["reward"]}" '
                f'data-margin-mult="{s.get("margin_mult",1.0)}" data-lot-size="{s.get("lot_size",65)}" id="{cid}">'
                f'<div class="sc-pop-badge" id="pop_{cid}">—%</div>'
                f'<div class="sc-svg">{svg}</div>'
                f'<div class="sc-body">'
                f'<div class="sc-name">{s["name"]}</div>'
                f'<div class="sc-legs">{s["legs"]}</div>'
                f'<div class="sc-tags">'
                f'<span class="sc-tag" style="color:{rc};border-color:{rc}40;">Risk: {s["risk"]}</span>'
                f'<span class="sc-tag" style="color:{rwc};border-color:{rwc}40;">Reward: {s["reward"]}</span>'
                f'</div></div>'
                f'<div class="sc-detail" id="detail_{cid}">'
                f'<div class="sc-desc">{s["desc"]}</div>'
                f'<div class="sc-metrics-live" id="metrics_{cid}">'
                f'<div class="sc-loading">&#9685; Calculating metrics...</div>'
                f'</div></div>'
                f'<div class="sc-payoff" id="payoff_{cid}">'
                f'<div class="sc-payoff-inner" id="payoff_canvas_{cid}"></div>'
                f'</div></div>'
            )
        return cards

    bull_cards = render_cards(STRATEGIES_DATA["bullish"],       "bullish")
    bear_cards = render_cards(STRATEGIES_DATA["bearish"],       "bearish")
    nd_cards   = render_cards(STRATEGIES_DATA["nondirectional"],"nondirectional")

    return f"""
<div class="section" id="strat">
  <div class="sec-title">STRATEGIES REFERENCE
    <span class="sec-sub">Smart PoP · Live S/R + OI Walls + Market Bias · Click to expand</span>
  </div>

  <div id="smartPopLegend" style="
    background:linear-gradient(135deg,rgba(100,128,255,.08),rgba(0,200,150,.06));
    border:1px solid rgba(100,128,255,.2);border-radius:14px;padding:14px 18px;
    margin-bottom:18px;display:flex;flex-wrap:wrap;gap:14px;align-items:flex-start;">
    <div style="flex:0 0 auto;">
      <div style="font-size:13px;font-weight:700;letter-spacing:2px;color:#8aa0ff;text-transform:uppercase;margin-bottom:8px;">&#9889; EDGE SCORE + TRUE POP ENGINE</div>
      <div style="font-size:14.5px;color:rgba(255,255,255,.78);line-height:1.8;">
        Edge Score = Base 50% + Bias + S/R + OI Walls + PCR &nbsp;|&nbsp; True PoP = N(d2) via IV
      </div>
    </div>
    <div style="display:flex;gap:10px;flex-wrap:wrap;flex:1;">
      <div id="legendBias" style="background:rgba(0,0,0,.2);border-radius:8px;padding:8px 12px;min-width:130px;">
        <div style="font-size:11.6px;color:rgba(255,255,255,.68);letter-spacing:1.5px;text-transform:uppercase;margin-bottom:4px;">A · BIAS</div>
        <div id="legendBiasVal" style="font-family:'DM Mono',monospace;font-size:17.4px;font-weight:700;color:#8aa0ff;">—</div>
      </div>
      <div id="legendSR" style="background:rgba(0,0,0,.2);border-radius:8px;padding:8px 12px;min-width:160px;">
        <div style="font-size:11.6px;color:rgba(255,255,255,.68);letter-spacing:1.5px;text-transform:uppercase;margin-bottom:4px;">B · S/R ZONE</div>
        <div id="legendSRVal" style="font-family:'DM Mono',monospace;font-size:17.4px;font-weight:700;color:#ffd166;">—</div>
      </div>
      <div id="legendOI" style="background:rgba(0,0,0,.2);border-radius:8px;padding:8px 12px;min-width:160px;">
        <div style="font-size:11.6px;color:rgba(255,255,255,.68);letter-spacing:1.5px;text-transform:uppercase;margin-bottom:4px;">C · OI WALLS</div>
        <div id="legendOIVal" style="font-family:'DM Mono',monospace;font-size:17.4px;font-weight:700;color:#00c8e0;">—</div>
      </div>
      <div id="legendPCR" style="background:rgba(0,0,0,.2);border-radius:8px;padding:8px 12px;min-width:120px;">
        <div style="font-size:11.6px;color:rgba(255,255,255,.68);letter-spacing:1.5px;text-transform:uppercase;margin-bottom:4px;">D · PCR</div>
        <div id="legendPCRVal" style="font-family:'DM Mono',monospace;font-size:17.4px;font-weight:700;color:#00c896;">—</div>
      </div>
      <div id="legendRec" style="background:rgba(0,200,150,.06);border:1px solid rgba(0,200,150,.2);border-radius:8px;padding:8px 12px;min-width:180px;">
        <div style="font-size:11.6px;color:rgba(0,200,150,.6);letter-spacing:1.5px;text-transform:uppercase;margin-bottom:4px;">&#9733; TOP STRATEGY</div>
        <div id="legendRecVal" style="font-size:15.9px;font-weight:700;color:#00c896;">Calculating...</div>
      </div>
    </div>
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
    <div style="margin-left:auto;display:flex;align-items:center;gap:8px;">
      <span style="font-size:13px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;
                   color:rgba(255,209,102,.7);">&#128197; EXPIRY DATE</span>
      <select id="expiryDropdown" onchange="switchExpiry(this.value)"
        style="appearance:none;-webkit-appearance:none;
               background:linear-gradient(135deg,rgba(245,197,24,.12),rgba(200,155,10,.06));
               border:1px solid rgba(245,197,24,.45);border-radius:8px;
               color:#ffd166;font-family:'DM Mono',monospace;font-size:15.9px;font-weight:700;
               padding:7px 28px 7px 12px;cursor:pointer;outline:none;letter-spacing:.5px;">
        {expiry_opts_html}
      </select>
    </div>
  </div>
  <div class="sc-grid" id="sc-grid">
    {bull_cards}{bear_cards}{nd_cards}
  </div>
</div>

<script>
const OC={{
  spot:        {spot:.2f},
  atm:         {atm},
  expiry:      "{expiry}",
  pcr:         {pcr:.3f},
  maxPain:     {mp},
  maxCeStrike: {max_ce_s},
  maxPeStrike: {max_pe_s},
  support:     {support:.2f},
  resistance:  {resistance:.2f},
  strongSup:   {strong_sup:.2f},
  strongRes:   {strong_res:.2f},
  bias:        "{bias}",
  biasConf:    "{conf}",
  bullScore:   {bull_sc},
  bearScore:   {bear_sc},
  strikes:     {strikes_json},
  lotSize:     65,
  ivp:         {ivp}
}};

// TRUE_POP_MAP — IV-based N(d2) probability, pre-computed in Python.
// Answers "what does the Black-Scholes bell curve say?" (independent of bias/OI).
const TRUE_POP_MAP = {true_pop_json};

// IVP — India VIX percentile over last 90 trading days.
// IVP > 70: IV expensive → favour premium selling.
// IVP < 20: IV cheap → avoid short premium, prefer long vega strategies.
const IVP = {ivp};

const STRIKE_MAP={{}};
OC.strikes.forEach(s=>{{ STRIKE_MAP[s.strike]=s; }});

function smartPoP(shape, cat) {{
  const spot=OC.spot, pcr=OC.pcr;
  const sup=OC.support, res=OC.resistance;
  const ssup=OC.strongSup, sres=OC.strongRes;
  const maxCE=OC.maxCeStrike, maxPE=OC.maxPeStrike;
  const bias=OC.bias, conf=OC.biasConf;
  const rangeSize = res - sup || 200;
  const confMult = conf==="HIGH" ? 1.25 : conf==="LOW" ? 0.6 : 1.0;
  let biasAdj = 0;
  if (cat === "bullish") {{ biasAdj = bias==="BULLISH" ? 15 : bias==="BEARISH" ? -15 : 0; }}
  else if (cat === "bearish") {{ biasAdj = bias==="BEARISH" ? 15 : bias==="BULLISH" ? -15 : 0; }}
  else {{ biasAdj = bias==="SIDEWAYS" ? 8 : (OC.bullScore===OC.bearScore ? 5 : -5); }}
  biasAdj = biasAdj * confMult;
  let srAdj = 0;
  const distToSup = spot - sup; const distToRes = res - spot;
  if (cat === "bullish") {{
    if (distToSup >= 0 && distToSup <= rangeSize * 0.25) {{ srAdj = 10; }}
    else if (distToSup >= 0 && distToSup <= rangeSize * 0.5) {{ srAdj = 5; }}
    else if (distToRes >= 0 && distToRes <= rangeSize * 0.2) {{ srAdj = -10; }}
    else if (spot > res) {{ srAdj = -8; }} else {{ srAdj = 2; }}
  }} else if (cat === "bearish") {{
    if (distToRes >= 0 && distToRes <= rangeSize * 0.25) {{ srAdj = 10; }}
    else if (distToRes >= 0 && distToRes <= rangeSize * 0.5) {{ srAdj = 5; }}
    else if (distToSup >= 0 && distToSup <= rangeSize * 0.2) {{ srAdj = -10; }}
    else if (spot < sup) {{ srAdj = -8; }} else {{ srAdj = 2; }}
  }} else {{
    const midRange = (sup + res) / 2; const distFromMid = Math.abs(spot - midRange); const halfRange = rangeSize / 2;
    if (distFromMid <= halfRange * 0.3) {{ srAdj = 10; }} else if (distFromMid <= halfRange * 0.6) {{ srAdj = 5; }} else {{ srAdj = -5; }}
  }}
  srAdj = srAdj * confMult;
  let oiAdj = 0;
  const distAboveMaxPE = spot - maxPE; const distBelowMaxCE = maxCE - spot;
  if (cat === "bullish") {{
    if (distAboveMaxPE > 0 && distAboveMaxPE < 150) {{ oiAdj += 8; }} else if (distAboveMaxPE > 150) {{ oiAdj += 4; }} else {{ oiAdj -= 8; }}
    if (distBelowMaxCE > 200) {{ oiAdj += 5; }} else if (distBelowMaxCE < 100) {{ oiAdj -= 7; }}
  }} else if (cat === "bearish") {{
    if (distBelowMaxCE > 0 && distBelowMaxCE < 150) {{ oiAdj += 8; }} else if (distBelowMaxCE > 150) {{ oiAdj += 4; }} else {{ oiAdj -= 8; }}
    if (distAboveMaxPE > 200) {{ oiAdj += 5; }} else if (distAboveMaxPE < 100) {{ oiAdj -= 7; }}
  }} else {{
    if (distAboveMaxPE > 0 && distBelowMaxCE > 0) {{
      const oiRange = maxCE - maxPE || 200; const midOI = (maxPE + maxCE) / 2; const distFromOIMid = Math.abs(spot - midOI);
      if (distFromOIMid < oiRange * 0.3) {{ oiAdj = 10; }} else {{ oiAdj = 4; }}
    }} else {{ oiAdj = -5; }}
  }}
  let pcrAdj = 0;
  if (cat === "bullish") {{ pcrAdj = pcr > 1.5 ? 8 : pcr > 1.2 ? 6 : pcr > 1.0 ? 3 : pcr < 0.7 ? -8 : pcr < 0.9 ? -4 : 0; }}
  else if (cat === "bearish") {{ pcrAdj = pcr < 0.5 ? 8 : pcr < 0.7 ? 6 : pcr < 0.9 ? 3 : pcr > 1.3 ? -8 : pcr > 1.1 ? -4 : 0; }}
  else {{ pcrAdj = (pcr >= 0.85 && pcr <= 1.15) ? 6 : (pcr >= 0.7 && pcr <= 1.3) ? 3 : -4; }}

  // ── IVP-based strategy adjustment ────────────────────────────────────
  // Short premium strategies (condors, straddles, spreads) are penalised
  // when IV is cheap (IVP < 20) — you are selling low and the trade is
  // structurally poor regardless of bias alignment.
  // Long vol strategies (straddles, backspreads) are rewarded when IV is cheap.
  // Short premium is rewarded when IVP > 70 — IV is expensive relative to history.
  const ivp = IVP;
  const isShortPremium = ['short_straddle','short_strangle','short_iron_condor',
    'short_iron_fly','short_put','short_call','bear_call_spread','bull_put_spread',
    'jade_lizard','reverse_jade','call_ratio_spread','put_ratio_spread'].includes(shape);
  const isLongVol = ['long_straddle','long_strangle','long_iron_condor',
    'long_iron_fly','call_ratio_back','put_ratio_back','long_call','long_put'].includes(shape);
  let ivpAdj = 0;
  if (isShortPremium) {{
    if (ivp < 20)      ivpAdj = -15;  // IV cheap — never sell; hard penalty
    else if (ivp < 35) ivpAdj = -8;   // IV below average — cautious
    else if (ivp > 70) ivpAdj = +8;   // IV expensive — premium selling ideal
    else if (ivp > 55) ivpAdj = +4;   // IV moderately high — slight boost
  }}
  if (isLongVol) {{
    if (ivp < 20)      ivpAdj = +10;  // IV cheap — long vol is a bargain
    else if (ivp < 35) ivpAdj = +5;
    else if (ivp > 70) ivpAdj = -8;   // IV expensive — long vol is overpriced
    else if (ivp > 55) ivpAdj = -4;
  }}

  let stratAdj = 0;
  if (shape.includes('spread') || shape.includes('condor') || shape.includes('butterfly')) {{ stratAdj = 2; }}
  if (shape === 'short_straddle' || shape === 'short_strangle') {{ stratAdj = bias === 'SIDEWAYS' ? 8 : -10; }}
  if (shape === 'long_straddle' || shape === 'long_strangle') {{ stratAdj = bias === 'SIDEWAYS' ? -8 : 8; }}
  if ((shape === 'short_iron_condor' || shape === 'short_iron_fly') && bias === 'SIDEWAYS') {{ stratAdj = 10; }}

  const rawPoP = 50 + biasAdj + srAdj + oiAdj + pcrAdj + stratAdj + ivpAdj;
  return {{ edgeScore: Math.min(95, Math.max(5, Math.round(rawPoP))),
            biasAdj: Math.round(biasAdj), srAdj: Math.round(srAdj),
            oiAdj: Math.round(oiAdj), pcrAdj: Math.round(pcrAdj),
            stratAdj: Math.round(stratAdj), ivpAdj: Math.round(ivpAdj) }};
}}

function normCDF(x) {{
  const a1=0.254829592,a2=-0.284496736,a3=1.421413741,a4=-1.453152027,a5=1.061405429,p=0.3275911;
  const sign=x<0?-1:1; x=Math.abs(x);
  const t=1/(1+p*x); const y=1-(((((a5*t+a4)*t)+a3)*t+a2)*t+a1)*t*Math.exp(-x*x);
  return 0.5*(1+sign*y);
}}
function getATMLTP(type) {{
  const row=STRIKE_MAP[OC.atm]||OC.strikes.reduce((b,s)=>Math.abs(s.strike-OC.atm)<Math.abs(b.strike-OC.atm)?s:b,OC.strikes[0]||{{strike:OC.atm,ce_ltp:0,pe_ltp:0,ce_iv:15,pe_iv:15}});
  return type==='ce'?row.ce_ltp:row.pe_ltp;
}}
function getOTM(type,offset) {{
  const t=type==='ce'?OC.atm+offset*50:OC.atm-offset*50;
  const row=STRIKE_MAP[t]||OC.strikes.reduce((b,s)=>Math.abs(s.strike-t)<Math.abs(b.strike-t)?s:b,OC.strikes[0]||{{strike:OC.atm,ce_ltp:0,pe_ltp:0,ce_iv:15,pe_iv:15}});
  return {{strike:row.strike||t,ltp:type==='ce'?row.ce_ltp:row.pe_ltp,iv:type==='ce'?row.ce_iv:row.pe_iv}};
}}

function getGreeks(type, strike) {{
  // Returns {{delta, theta, vega}} for a given strike and option type (ce/pe)
  // Uses STRIKE_MAP which now includes BS-computed greeks from Python
  const row = STRIKE_MAP[strike] || OC.strikes.reduce(
    (b, s) => Math.abs(s.strike - strike) < Math.abs(b.strike - strike) ? s : b,
    OC.strikes[0] || {{strike: OC.atm, ce_delta:0.5, pe_delta:-0.5, ce_theta:-0.1, pe_theta:-0.1, ce_vega:0.05, pe_vega:0.05}}
  );
  if (type === 'ce') return {{ delta: row.ce_delta||0.5,  theta: row.ce_theta||0, vega: row.ce_vega||0,  gamma: row.ce_gamma||0 }};
  else               return {{ delta: row.pe_delta||-0.5, theta: row.pe_theta||0, vega: row.pe_vega||0, gamma: row.pe_gamma||0 }};
}}

function calcMetrics(shape, edgeScore) {{
  const spot   = OC.spot, atm = OC.atm;
  const lotSz  = OC.lotSize;
  const ce_atm = getATMLTP('ce'), pe_atm = getATMLTP('pe');

  const co1 = getOTM('ce', 1), co2 = getOTM('ce', 2), co3 = getOTM('ce', 3);
  const po1 = getOTM('pe', 1), po2 = getOTM('pe', 2), po3 = getOTM('pe', 3);

  const ceWing1 = co1.strike - atm;
  const peWing1 = atm - po1.strike;

  // Greeks helpers
  const gCeAtm = getGreeks('ce', atm);
  const gPeAtm = getGreeks('pe', atm);
  const gCo1   = getGreeks('ce', co1.strike);
  const gCo2   = getGreeks('ce', co2.strike);
  const gCo3   = getGreeks('ce', co3.strike);
  const gPo1   = getGreeks('pe', po1.strike);
  const gPo2   = getGreeks('pe', po2.strike);
  const gPo3   = getGreeks('pe', po3.strike);

  let es = edgeScore || 50;
  let mp = 0, ml = 0, be = [], nc = 0, margin = 0, rrRatio = 0;
  let ltpParts = [];
  // Net greeks for intraday simulator (per lot)
  let netDelta = 0, netTheta = 0, netVega = 0, netGamma = 0;

  switch (shape) {{

    case 'long_call': {{
      const p = ce_atm || 150;
      mp = 999999; ml = p * lotSz; be = [atm + p];
      nc = -p * lotSz; margin = p * lotSz;
      ltpParts = [{{ l: 'BUY CE \u20b9' + atm.toLocaleString('en-IN'), v: p, c: '#00c8e0' }}];
      netDelta = gCeAtm.delta * lotSz; netTheta = gCeAtm.theta * lotSz; netVega = gCeAtm.vega * lotSz;
      break;
    }}
    case 'long_put': {{
      const p = pe_atm || 150;
      mp = 999999; ml = p * lotSz; be = [atm - p];
      nc = -p * lotSz; margin = p * lotSz;
      ltpParts = [{{ l: 'BUY PE \u20b9' + atm.toLocaleString('en-IN'), v: p, c: '#ff9090' }}];
      netDelta = gPeAtm.delta * lotSz; netTheta = gPeAtm.theta * lotSz; netVega = gPeAtm.vega * lotSz;
      break;
    }}
    case 'short_put': {{
      const p = pe_atm || 150;
      mp = p * lotSz; ml = (atm - p) * lotSz; be = [atm - p];
      nc = p * lotSz; margin = atm * lotSz * 0.15;
      rrRatio = ((atm - p) / p).toFixed(2);
      ltpParts = [{{ l: 'SELL PE \u20b9' + atm.toLocaleString('en-IN'), v: p, c: '#ff9090' }}];
      netDelta = -gPeAtm.delta * lotSz; netTheta = -gPeAtm.theta * lotSz; netVega = -gPeAtm.vega * lotSz;
      break;
    }}
    case 'short_call': {{
      const p = ce_atm || 150;
      mp = p * lotSz; ml = 999999; be = [atm + p];
      nc = p * lotSz; margin = atm * lotSz * 0.15;
      ltpParts = [{{ l: 'SELL CE \u20b9' + atm.toLocaleString('en-IN'), v: p, c: '#00c8e0' }}];
      netDelta = -gCeAtm.delta * lotSz; netTheta = -gCeAtm.theta * lotSz; netVega = -gCeAtm.vega * lotSz;
      break;
    }}

    case 'bull_call_spread': {{
      const bp = ce_atm || 150, sp = co1.ltp || 80;
      const nd = bp - sp, sw = ceWing1;
      mp = (sw - nd) * lotSz; ml = nd * lotSz; be = [atm + nd];
      nc = -nd * lotSz; margin = nd * lotSz;
      rrRatio = ((sw - nd) / nd).toFixed(2);
      ltpParts = [
        {{ l: 'BUY CE \u20b9' + atm.toLocaleString('en-IN'), v: bp, c: '#00c8e0' }},
        {{ l: 'SELL CE \u20b9' + co1.strike.toLocaleString('en-IN'), v: sp, c: '#00c896' }}
      ];
      netDelta=(gCeAtm.delta - gCo1.delta)*lotSz; netTheta=(gCeAtm.theta - gCo1.theta)*lotSz; netVega=(gCeAtm.vega - gCo1.vega)*lotSz; netGamma=(gCeAtm.gamma - gCo1.gamma)*lotSz;
      break;
    }}
    case 'bull_put_spread': {{
      const sp = pe_atm || 150, bp = po1.ltp || 80;
      const nc2 = sp - bp, sw = peWing1;
      mp = nc2 * lotSz; ml = (sw - nc2) * lotSz; be = [atm - nc2];
      nc = nc2 * lotSz; margin = sw * lotSz;
      rrRatio = (nc2 / (sw - nc2)).toFixed(2);
      ltpParts = [
        {{ l: 'SELL PE \u20b9' + atm.toLocaleString('en-IN'), v: sp, c: '#00c896' }},
        {{ l: 'BUY PE \u20b9' + po1.strike.toLocaleString('en-IN'), v: bp, c: '#ff9090' }}
      ];
      netDelta=(-gPeAtm.delta + gPo1.delta)*lotSz; netTheta=(-gPeAtm.theta + gPo1.theta)*lotSz; netVega=(-gPeAtm.vega + gPo1.vega)*lotSz; netGamma=(-gPeAtm.gamma + gPo1.gamma)*lotSz;
      break;
    }}
    case 'bear_call_spread': {{
      const sp = ce_atm || 150, bp = co1.ltp || 80;
      const nc2 = sp - bp, sw = ceWing1;
      mp = nc2 * lotSz; ml = (sw - nc2) * lotSz; be = [atm + nc2];
      nc = nc2 * lotSz; margin = sw * lotSz;
      rrRatio = (nc2 / (sw - nc2)).toFixed(2);
      ltpParts = [
        {{ l: 'SELL CE \u20b9' + atm.toLocaleString('en-IN'), v: sp, c: '#00c896' }},
        {{ l: 'BUY CE \u20b9' + co1.strike.toLocaleString('en-IN'), v: bp, c: '#00c8e0' }}
      ];
      netDelta=(-gCeAtm.delta + gCo1.delta)*lotSz; netTheta=(-gCeAtm.theta + gCo1.theta)*lotSz; netVega=(-gCeAtm.vega + gCo1.vega)*lotSz; netGamma=(-gCeAtm.gamma + gCo1.gamma)*lotSz;
      break;
    }}
    case 'bear_put_spread': {{
      const bp = pe_atm || 150, sp = po1.ltp || 80;
      const nd = bp - sp, sw = peWing1;
      mp = (sw - nd) * lotSz; ml = nd * lotSz; be = [atm - nd];
      nc = -nd * lotSz; margin = nd * lotSz;
      rrRatio = ((sw - nd) / nd).toFixed(2);
      ltpParts = [
        {{ l: 'BUY PE \u20b9' + atm.toLocaleString('en-IN'), v: bp, c: '#ff9090' }},
        {{ l: 'SELL PE \u20b9' + po1.strike.toLocaleString('en-IN'), v: sp, c: '#00c896' }}
      ];
      netDelta=(gPeAtm.delta - gPo1.delta)*lotSz; netTheta=(gPeAtm.theta - gPo1.theta)*lotSz; netVega=(gPeAtm.vega - gPo1.vega)*lotSz; netGamma=(gPeAtm.gamma - gPo1.gamma)*lotSz;
      break;
    }}

    case 'long_straddle': {{
      const cp2 = ce_atm || 150, pp = pe_atm || 150, tp = cp2 + pp;
      mp = 999999; ml = tp * lotSz; be = [atm - tp, atm + tp];
      nc = -tp * lotSz; margin = tp * lotSz;
      ltpParts = [
        {{ l: 'BUY CE \u20b9' + atm.toLocaleString('en-IN'), v: cp2, c: '#00c8e0' }},
        {{ l: 'BUY PE \u20b9' + atm.toLocaleString('en-IN'), v: pp, c: '#ff9090' }}
      ];
      netDelta=(gCeAtm.delta + gPeAtm.delta)*lotSz; netTheta=(gCeAtm.theta + gPeAtm.theta)*lotSz; netVega=(gCeAtm.vega + gPeAtm.vega)*lotSz; netGamma=(gCeAtm.gamma + gPeAtm.gamma)*lotSz;
      break;
    }}
    case 'short_straddle': {{
      const cp2 = ce_atm || 150, pp = pe_atm || 150, tp = cp2 + pp;
      mp = tp * lotSz; ml = 999999; be = [atm - tp, atm + tp];
      nc = tp * lotSz; margin = Math.round(0.155 * OC.spot * OC.lotSize); // SPAN straddle: both ATM naked ~15.5% notional
      ltpParts = [
        {{ l: 'SELL CE \u20b9' + atm.toLocaleString('en-IN'), v: cp2, c: '#00c8e0' }},
        {{ l: 'SELL PE \u20b9' + atm.toLocaleString('en-IN'), v: pp, c: '#ff9090' }}
      ];
      netDelta=-(gCeAtm.delta + gPeAtm.delta)*lotSz; netTheta=-(gCeAtm.theta + gPeAtm.theta)*lotSz; netVega=-(gCeAtm.vega + gPeAtm.vega)*lotSz; netGamma=-(gCeAtm.gamma + gPeAtm.gamma)*lotSz;
      break;
    }}
    case 'long_strangle': {{
      const cp2 = co1.ltp || 100, pp = po1.ltp || 100, tp = cp2 + pp;
      mp = 999999; ml = tp * lotSz;
      be = [po1.strike - tp, co1.strike + tp];
      nc = -tp * lotSz; margin = tp * lotSz;
      ltpParts = [
        {{ l: 'BUY CE \u20b9' + co1.strike.toLocaleString('en-IN'), v: cp2, c: '#00c8e0' }},
        {{ l: 'BUY PE \u20b9' + po1.strike.toLocaleString('en-IN'), v: pp, c: '#ff9090' }}
      ];
      netDelta=(gCo1.delta + gPo1.delta)*lotSz; netTheta=(gCo1.theta + gPo1.theta)*lotSz; netVega=(gCo1.vega + gPo1.vega)*lotSz; netGamma=(gCo1.gamma + gPo1.gamma)*lotSz;
      break;
    }}
    case 'short_strangle': {{
      const cp2 = co1.ltp || 100, pp = po1.ltp || 100, tp = cp2 + pp;
      mp = tp * lotSz; ml = 999999;
      be = [po1.strike - tp, co1.strike + tp];
      nc = tp * lotSz; margin = Math.round((Math.max(0.117*OC.spot,0.075*co1.strike)*OC.lotSize > Math.max(0.117*OC.spot,0.075*po1.strike)*OC.lotSize ? Math.max(0.117*OC.spot,0.075*co1.strike)*OC.lotSize*0.85 + Math.max(0.117*OC.spot,0.075*po1.strike)*OC.lotSize*0.15 : Math.max(0.117*OC.spot,0.075*po1.strike)*OC.lotSize*0.85 + Math.max(0.117*OC.spot,0.075*co1.strike)*OC.lotSize*0.15)); // SPAN strangle: 85/15 netting
      ltpParts = [
        {{ l: 'SELL CE \u20b9' + co1.strike.toLocaleString('en-IN'), v: cp2, c: '#00c8e0' }},
        {{ l: 'SELL PE \u20b9' + po1.strike.toLocaleString('en-IN'), v: pp, c: '#ff9090' }}
      ];
      netDelta=-(gCo1.delta + gPo1.delta)*lotSz; netTheta=-(gCo1.theta + gPo1.theta)*lotSz; netVega=-(gCo1.vega + gPo1.vega)*lotSz; netGamma=-(gCo1.gamma + gPo1.gamma)*lotSz;
      break;
    }}

    case 'short_iron_fly': {{
      const cp2 = ce_atm || 150, pp = pe_atm || 150;
      const wc = co1.ltp || 80, wp = po1.ltp || 80;
      const nc2 = cp2 + pp - wc - wp;
      mp = nc2 * lotSz; ml = (ceWing1 - nc2) * lotSz;
      be = [atm - nc2, atm + nc2];
      nc = nc2 * lotSz; margin = ceWing1 * lotSz * 2;
      rrRatio = (nc2 / (ceWing1 - nc2)).toFixed(2);
      ltpParts = [
        {{ l: 'SELL CE \u20b9' + atm.toLocaleString('en-IN'), v: cp2, c: '#00c8e0' }},
        {{ l: 'SELL PE \u20b9' + atm.toLocaleString('en-IN'), v: pp, c: '#ff9090' }},
        {{ l: 'BUY CE \u20b9' + co1.strike.toLocaleString('en-IN'), v: wc, c: '#00c8e0' }},
        {{ l: 'BUY PE \u20b9' + po1.strike.toLocaleString('en-IN'), v: wp, c: '#ff9090' }}
      ];
      netDelta=-(gCeAtm.delta+gPeAtm.delta-gCo1.delta-gPo1.delta)*lotSz; netTheta=-(gCeAtm.theta+gPeAtm.theta-gCo1.theta-gPo1.theta)*lotSz; netVega=-(gCeAtm.vega+gPeAtm.vega-gCo1.vega-gPo1.vega)*lotSz; netGamma=-(gCeAtm.gamma+gPeAtm.gamma-gCo1.gamma-gPo1.gamma)*lotSz;
      break;
    }}
    case 'long_iron_fly': {{
      const cp2 = ce_atm || 150, pp = pe_atm || 150;
      const wc = co1.ltp || 80, wp = po1.ltp || 80;
      const nd = wc + wp - cp2 - pp;
      mp = (ceWing1 - Math.abs(nd)) * lotSz; ml = Math.abs(nd) * lotSz;
      be = [atm - Math.abs(nd), atm + Math.abs(nd)];
      nc = -Math.abs(nd) * lotSz; margin = Math.abs(nd) * lotSz;
      rrRatio = ((ceWing1 - Math.abs(nd)) / Math.abs(nd)).toFixed(2);
      ltpParts = [
        {{ l: 'BUY CE \u20b9' + atm.toLocaleString('en-IN'), v: cp2, c: '#00c8e0' }},
        {{ l: 'BUY PE \u20b9' + atm.toLocaleString('en-IN'), v: pp, c: '#ff9090' }},
        {{ l: 'SELL CE \u20b9' + co1.strike.toLocaleString('en-IN'), v: wc, c: '#00c8e0' }},
        {{ l: 'SELL PE \u20b9' + po1.strike.toLocaleString('en-IN'), v: wp, c: '#ff9090' }}
      ];
      netDelta=(gCeAtm.delta+gPeAtm.delta-gCo1.delta-gPo1.delta)*lotSz; netTheta=(gCeAtm.theta+gPeAtm.theta-gCo1.theta-gPo1.theta)*lotSz; netVega=(gCeAtm.vega+gPeAtm.vega-gCo1.vega-gPo1.vega)*lotSz; netGamma=(gCeAtm.gamma+gPeAtm.gamma-gCo1.gamma-gPo1.gamma)*lotSz;
      break;
    }}
    case 'short_iron_condor': {{
      const sc = co1.ltp || 100, bc = co2.ltp || 50;
      const sp = po1.ltp || 100, bp = po2.ltp || 50;
      const nc2 = sc - bc + sp - bp;
      mp = nc2 * lotSz; ml = (ceWing1 - nc2) * lotSz;
      be = [po1.strike - nc2, co1.strike + nc2];
      nc = nc2 * lotSz; margin = ceWing1 * lotSz * 2;
      rrRatio = (nc2 / (ceWing1 - nc2)).toFixed(2);
      ltpParts = [
        {{ l: 'SELL CE \u20b9' + co1.strike.toLocaleString('en-IN'), v: sc, c: '#00c8e0' }},
        {{ l: 'BUY CE \u20b9' + co2.strike.toLocaleString('en-IN'), v: bc, c: '#00c8e0' }},
        {{ l: 'SELL PE \u20b9' + po1.strike.toLocaleString('en-IN'), v: sp, c: '#ff9090' }},
        {{ l: 'BUY PE \u20b9' + po2.strike.toLocaleString('en-IN'), v: bp, c: '#ff9090' }}
      ];
      netDelta=-(gCo1.delta+gPo1.delta-gCo2.delta-gPo2.delta)*lotSz; netTheta=-(gCo1.theta+gPo1.theta-gCo2.theta-gPo2.theta)*lotSz; netVega=-(gCo1.vega+gPo1.vega-gCo2.vega-gPo2.vega)*lotSz; netGamma=-(gCo1.gamma+gPo1.gamma-gCo2.gamma-gPo2.gamma)*lotSz;
      break;
    }}
    case 'long_iron_condor': {{
      const sc = co1.ltp || 100, bc = co2.ltp || 50;
      const sp = po1.ltp || 100, bp = po2.ltp || 50;
      const nd = bc - sc + bp - sp;
      mp = (ceWing1 - Math.abs(nd)) * lotSz; ml = Math.abs(nd) * lotSz;
      be = [po1.strike - Math.abs(nd), co1.strike + Math.abs(nd)];
      nc = nd * lotSz; margin = Math.abs(nd) * lotSz;
      rrRatio = ((ceWing1 - Math.abs(nd)) / Math.abs(nd)).toFixed(2);
      ltpParts = [
        {{ l: 'SELL CE \u20b9' + co1.strike.toLocaleString('en-IN'), v: sc, c: '#00c8e0' }},
        {{ l: 'BUY CE \u20b9' + co2.strike.toLocaleString('en-IN'), v: bc, c: '#00c8e0' }},
        {{ l: 'SELL PE \u20b9' + po1.strike.toLocaleString('en-IN'), v: sp, c: '#ff9090' }},
        {{ l: 'BUY PE \u20b9' + po2.strike.toLocaleString('en-IN'), v: bp, c: '#ff9090' }}
      ];
      netDelta=(gCo1.delta+gPo1.delta-gCo2.delta-gPo2.delta)*lotSz; netTheta=(gCo1.theta+gPo1.theta-gCo2.theta-gPo2.theta)*lotSz; netVega=(gCo1.vega+gPo1.vega-gCo2.vega-gPo2.vega)*lotSz; netGamma=(gCo1.gamma+gPo1.gamma-gCo2.gamma-gPo2.gamma)*lotSz;
      break;
    }}

    case 'call_butterfly':
    case 'bull_butterfly': {{
      const lp = ce_atm || 150, mid = co1.ltp || 80, hp = co2.ltp || 40;
      const nd = lp - 2 * mid + hp;
      mp = (ceWing1 - nd) * lotSz; ml = nd * lotSz;
      be = [atm + nd, co2.strike - nd];
      nc = -nd * lotSz; margin = nd * lotSz;
      rrRatio = ((ceWing1 - nd) / nd).toFixed(2);
      ltpParts = [
        {{ l: 'BUY CE \u20b9' + atm.toLocaleString('en-IN'), v: lp, c: '#00c8e0' }},
        {{ l: 'SELL 2x CE \u20b9' + co1.strike.toLocaleString('en-IN'), v: mid, c: '#00c896' }},
        {{ l: 'BUY CE \u20b9' + co2.strike.toLocaleString('en-IN'), v: hp, c: '#00c8e0' }}
      ];
      netDelta=(gCeAtm.delta - 2*gCo1.delta + gCo2.delta)*lotSz; netTheta=(gCeAtm.theta - 2*gCo1.theta + gCo2.theta)*lotSz; netVega=(gCeAtm.vega - 2*gCo1.vega + gCo2.vega)*lotSz; netGamma=(gCeAtm.gamma - 2*gCo1.gamma + gCo2.gamma)*lotSz;
      break;
    }}
    case 'put_butterfly':
    case 'bear_butterfly': {{
      const hp = pe_atm || 150, mid = po1.ltp || 80, lp = po2.ltp || 40;
      const nd = hp - 2 * mid + lp;
      mp = (peWing1 - nd) * lotSz; ml = nd * lotSz;
      be = [po2.strike + nd, atm - nd];
      nc = -nd * lotSz; margin = nd * lotSz;
      rrRatio = ((peWing1 - nd) / nd).toFixed(2);
      ltpParts = [
        {{ l: 'BUY PE \u20b9' + atm.toLocaleString('en-IN'), v: hp, c: '#ff9090' }},
        {{ l: 'SELL 2x PE \u20b9' + po1.strike.toLocaleString('en-IN'), v: mid, c: '#00c896' }},
        {{ l: 'BUY PE \u20b9' + po2.strike.toLocaleString('en-IN'), v: lp, c: '#ff9090' }}
      ];
      netDelta=(gPeAtm.delta - 2*gPo1.delta + gPo2.delta)*lotSz; netTheta=(gPeAtm.theta - 2*gPo1.theta + gPo2.theta)*lotSz; netVega=(gPeAtm.vega - 2*gPo1.vega + gPo2.vega)*lotSz; netGamma=(gPeAtm.gamma - 2*gPo1.gamma + gPo2.gamma)*lotSz;
      break;
    }}

    case 'call_ratio_back': {{
      const sp = ce_atm || 150, bp = co1.ltp || 80, nd = 2 * bp - sp;
      mp = 999999; ml = nd > 0 ? nd * lotSz : 0;
      be = [co1.strike + Math.abs(nd)];
      nc = -nd * lotSz; margin = Math.round((0.117 - 0.01) * OC.spot * OC.lotSize); // SPAN: naked(11.7%) - 1 BUY hedge(1%)
      ltpParts = [
        {{ l: 'SELL CE \u20b9' + atm.toLocaleString('en-IN'), v: sp, c: '#00c896' }},
        {{ l: 'BUY 2x CE \u20b9' + co1.strike.toLocaleString('en-IN'), v: bp, c: '#00c8e0' }}
      ];
      netDelta=(-gCeAtm.delta + 2*gCo1.delta)*lotSz; netTheta=(-gCeAtm.theta + 2*gCo1.theta)*lotSz; netVega=(-gCeAtm.vega + 2*gCo1.vega)*lotSz; netGamma=(-gCeAtm.gamma + 2*gCo1.gamma)*lotSz;
      break;
    }}
    case 'put_ratio_back': {{
      const sp = pe_atm || 150, bp = po1.ltp || 80, nd = 2 * bp - sp;
      mp = 999999; ml = nd > 0 ? nd * lotSz : 0;
      be = [po1.strike - Math.abs(nd)];
      nc = -nd * lotSz; margin = Math.round((0.117 - 0.01) * OC.spot * OC.lotSize); // SPAN: naked(11.7%) - 1 BUY hedge(1%)
      ltpParts = [
        {{ l: 'SELL PE \u20b9' + atm.toLocaleString('en-IN'), v: sp, c: '#00c896' }},
        {{ l: 'BUY 2x PE \u20b9' + po1.strike.toLocaleString('en-IN'), v: bp, c: '#ff9090' }}
      ];
      netDelta=(-gPeAtm.delta + 2*gPo1.delta)*lotSz; netTheta=(-gPeAtm.theta + 2*gPo1.theta)*lotSz; netVega=(-gPeAtm.vega + 2*gPo1.vega)*lotSz; netGamma=(-gPeAtm.gamma + 2*gPo1.gamma)*lotSz;
      break;
    }}

    case 'call_ratio_spread': {{
      const bp = ce_atm || 150, sp = co1.ltp || 80;
      const nc2 = 2 * sp - bp;
      const maxProfitPts = ceWing1 + nc2;
      mp = maxProfitPts * lotSz; ml = 999999;
      be = nc2 >= 0
        ? [2 * co1.strike - atm + nc2]
        : [atm - nc2, 2 * co1.strike - atm + nc2];
      nc = nc2 * lotSz; margin = co1.strike * lotSz * 0.15;
      ltpParts = [
        {{ l: 'BUY CE \u20b9' + atm.toLocaleString('en-IN'), v: bp, c: '#00c8e0' }},
        {{ l: 'SELL 2x CE \u20b9' + co1.strike.toLocaleString('en-IN'), v: sp, c: '#00c896' }}
      ];
      netDelta=(gCeAtm.delta - 2*gCo1.delta)*lotSz; netTheta=(gCeAtm.theta - 2*gCo1.theta)*lotSz; netVega=(gCeAtm.vega - 2*gCo1.vega)*lotSz; netGamma=(gCeAtm.gamma - 2*gCo1.gamma)*lotSz;
      break;
    }}
    case 'put_ratio_spread': {{
      const bp = pe_atm || 150, sp = po1.ltp || 80;
      const nc2 = 2 * sp - bp;
      const maxProfitPts = peWing1 + nc2;
      mp = maxProfitPts * lotSz; ml = 999999;
      be = nc2 >= 0
        ? [2 * po1.strike - atm - nc2]
        : [2 * po1.strike - atm - nc2, atm + nc2];
      nc = nc2 * lotSz; margin = po1.strike * lotSz * 0.15;
      ltpParts = [
        {{ l: 'BUY PE \u20b9' + atm.toLocaleString('en-IN'), v: bp, c: '#ff9090' }},
        {{ l: 'SELL 2x PE \u20b9' + po1.strike.toLocaleString('en-IN'), v: sp, c: '#00c896' }}
      ];
      netDelta=(gPeAtm.delta - 2*gPo1.delta)*lotSz; netTheta=(gPeAtm.theta - 2*gPo1.theta)*lotSz; netVega=(gPeAtm.vega - 2*gPo1.vega)*lotSz; netGamma=(gPeAtm.gamma - 2*gPo1.gamma)*lotSz;
      break;
    }}

    case 'long_synthetic': {{
      const cp2 = ce_atm || 150, pp = pe_atm || 150, nd = cp2 - pp;
      mp = 999999; ml = 999999;
      be = [atm + nd]; nc = -Math.abs(nd) * lotSz;
      margin = atm * lotSz * 0.30;
      ltpParts = [
        {{ l: 'BUY CE \u20b9' + atm.toLocaleString('en-IN'), v: cp2, c: '#00c8e0' }},
        {{ l: 'SELL PE \u20b9' + atm.toLocaleString('en-IN'), v: pp, c: '#ff9090' }}
      ];
      netDelta=(gCeAtm.delta - gPeAtm.delta)*lotSz; netTheta=(gCeAtm.theta - gPeAtm.theta)*lotSz; netVega=(gCeAtm.vega - gPeAtm.vega)*lotSz; netGamma=(gCeAtm.gamma - gPeAtm.gamma)*lotSz;
      break;
    }}
    case 'short_synthetic': {{
      const cp2 = ce_atm || 150, pp = pe_atm || 150, nc2 = cp2 - pp;
      mp = 999999; ml = 999999;
      be = [atm + nc2]; nc = Math.abs(nc2) * lotSz;
      margin = atm * lotSz * 0.30;
      ltpParts = [
        {{ l: 'SELL CE \u20b9' + atm.toLocaleString('en-IN'), v: cp2, c: '#00c8e0' }},
        {{ l: 'BUY PE \u20b9' + atm.toLocaleString('en-IN'), v: pp, c: '#ff9090' }}
      ];
      netDelta=(-gCeAtm.delta + gPeAtm.delta)*lotSz; netTheta=(-gCeAtm.theta + gPeAtm.theta)*lotSz; netVega=(-gCeAtm.vega + gPeAtm.vega)*lotSz; netGamma=(-gCeAtm.gamma + gPeAtm.gamma)*lotSz;
      break;
    }}

    case 'risk_reversal': {{
      const bp = po1.ltp || 100, sc = co1.ltp || 100;
      const nd = bp - sc;
      mp = 999999; ml = 999999;
      be = [po1.strike - Math.abs(nd), co1.strike + Math.abs(nd)];
      nc = -nd * lotSz; margin = atm * lotSz * 0.20;
      ltpParts = [
        {{ l: 'BUY PE \u20b9' + po1.strike.toLocaleString('en-IN'), v: bp, c: '#ff9090' }},
        {{ l: 'SELL CE \u20b9' + co1.strike.toLocaleString('en-IN'), v: sc, c: '#00c8e0' }}
      ];
      netDelta=(gPo1.delta - gCo1.delta)*lotSz; netTheta=(gPo1.theta - gCo1.theta)*lotSz; netVega=(gPo1.vega - gCo1.vega)*lotSz; netGamma=(gPo1.gamma - gCo1.gamma)*lotSz;
      break;
    }}
    case 'range_forward': {{
      const bc = co1.ltp || 100, sp = po1.ltp || 100;
      const nd = bc - sp;
      mp = 999999; ml = 999999;
      be = [po1.strike - Math.abs(nd), co1.strike + Math.abs(nd)];
      nc = -nd * lotSz; margin = atm * lotSz * 0.20;
      ltpParts = [
        {{ l: 'BUY CE \u20b9' + co1.strike.toLocaleString('en-IN'), v: bc, c: '#00c8e0' }},
        {{ l: 'SELL PE \u20b9' + po1.strike.toLocaleString('en-IN'), v: sp, c: '#ff9090' }}
      ];
      netDelta=(gCo1.delta - gPo1.delta)*lotSz; netTheta=(gCo1.theta - gPo1.theta)*lotSz; netVega=(gCo1.vega - gPo1.vega)*lotSz; netGamma=(gCo1.gamma - gPo1.gamma)*lotSz;
      break;
    }}

    case 'jade_lizard': {{
      const pp = po1.ltp || 100, cs = co1.ltp || 80, cb = co2.ltp || 40;
      const nc2 = pp + cs - cb;
      mp = nc2 * lotSz; ml = (po1.strike - nc2) * lotSz;
      be = [po1.strike - nc2];
      nc = nc2 * lotSz; margin = po1.strike * lotSz * 0.15;
      ltpParts = [
        {{ l: 'SELL PE \u20b9' + po1.strike.toLocaleString('en-IN'), v: pp, c: '#ff9090' }},
        {{ l: 'SELL CE \u20b9' + co1.strike.toLocaleString('en-IN'), v: cs, c: '#00c8e0' }},
        {{ l: 'BUY CE \u20b9' + co2.strike.toLocaleString('en-IN'), v: cb, c: '#00c8e0' }}
      ];
      netDelta=(-gPo1.delta - gCo1.delta + gCo2.delta)*lotSz; netTheta=(-gPo1.theta - gCo1.theta + gCo2.theta)*lotSz; netVega=(-gPo1.vega - gCo1.vega + gCo2.vega)*lotSz; netGamma=(-gPo1.gamma - gCo1.gamma + gCo2.gamma)*lotSz;
      break;
    }}
    case 'reverse_jade': {{
      const cp2 = co1.ltp || 100, ps = po1.ltp || 80, pb = po2.ltp || 40;
      const nc2 = cp2 + ps - pb;
      mp = nc2 * lotSz; ml = (po1.strike - nc2) * lotSz;
      be = [po1.strike - nc2];
      nc = nc2 * lotSz; margin = co1.strike * lotSz * 0.15;
      ltpParts = [
        {{ l: 'SELL CE \u20b9' + co1.strike.toLocaleString('en-IN'), v: cp2, c: '#00c8e0' }},
        {{ l: 'SELL PE \u20b9' + po1.strike.toLocaleString('en-IN'), v: ps, c: '#ff9090' }},
        {{ l: 'BUY PE \u20b9' + po2.strike.toLocaleString('en-IN'), v: pb, c: '#ff9090' }}
      ];
      netDelta=(-gCo1.delta - gPo1.delta + gPo2.delta)*lotSz; netTheta=(-gCo1.theta - gPo1.theta + gPo2.theta)*lotSz; netVega=(-gCo1.vega - gPo1.vega + gPo2.vega)*lotSz; netGamma=(-gCo1.gamma - gPo1.gamma + gPo2.gamma)*lotSz;
      break;
    }}

    case 'bull_condor': {{
      const s1 = ce_atm || 150;
      const s2 = co1.ltp || 100;
      const s3 = co2.ltp || 60;
      const s4 = co3.ltp || 30;
      const nd = s1 - s2 - s3 + s4;
      mp = (ceWing1 - nd) * lotSz; ml = nd * lotSz;
      be = [atm + nd, co3.strike - nd];
      nc = -nd * lotSz; margin = nd * lotSz;
      rrRatio = ((ceWing1 - nd) / nd).toFixed(2);
      ltpParts = [
        {{ l: 'BUY CE \u20b9'  + atm.toLocaleString('en-IN'),        v: s1, c: '#00c8e0' }},
        {{ l: 'SELL CE \u20b9' + co1.strike.toLocaleString('en-IN'), v: s2, c: '#00c896' }},
        {{ l: 'SELL CE \u20b9' + co2.strike.toLocaleString('en-IN'), v: s3, c: '#00c896' }},
        {{ l: 'BUY CE \u20b9'  + co3.strike.toLocaleString('en-IN'), v: s4, c: '#00c8e0' }}
      ];
      netDelta=(gCeAtm.delta - gCo1.delta - gCo2.delta + gCo3.delta)*lotSz; netTheta=(gCeAtm.theta - gCo1.theta - gCo2.theta + gCo3.theta)*lotSz; netVega=(gCeAtm.vega - gCo1.vega - gCo2.vega + gCo3.vega)*lotSz; netGamma=(gCeAtm.gamma - gCo1.gamma - gCo2.gamma + gCo3.gamma)*lotSz;
      break;
    }}
    case 'bear_condor': {{
      const s1 = pe_atm || 150;
      const s2 = po1.ltp || 100;
      const s3 = po2.ltp || 60;
      const s4 = po3.ltp || 30;
      const nd = s1 - s2 - s3 + s4;
      mp = (peWing1 - nd) * lotSz; ml = nd * lotSz;
      be = [po3.strike + nd, atm - nd];
      nc = -nd * lotSz; margin = nd * lotSz;
      rrRatio = ((peWing1 - nd) / nd).toFixed(2);
      ltpParts = [
        {{ l: 'BUY PE \u20b9'  + atm.toLocaleString('en-IN'),        v: s1, c: '#ff9090' }},
        {{ l: 'SELL PE \u20b9' + po1.strike.toLocaleString('en-IN'), v: s2, c: '#00c896' }},
        {{ l: 'SELL PE \u20b9' + po2.strike.toLocaleString('en-IN'), v: s3, c: '#00c896' }},
        {{ l: 'BUY PE \u20b9'  + po3.strike.toLocaleString('en-IN'), v: s4, c: '#ff9090' }}
      ];
      netDelta=(gPeAtm.delta - gPo1.delta - gPo2.delta + gPo3.delta)*lotSz; netTheta=(gPeAtm.theta - gPo1.theta - gPo2.theta + gPo3.theta)*lotSz; netVega=(gPeAtm.vega - gPo1.vega - gPo2.vega + gPo3.vega)*lotSz; netGamma=(gPeAtm.gamma - gPo1.gamma - gPo2.gamma + gPo3.gamma)*lotSz;
      break;
    }}

    case 'batman': {{
      const s1 = ce_atm || 150, s2 = co1.ltp || 80, s3 = co2.ltp || 40;
      const ndPerUnit = s1 - 2 * s2 + s3;
      const totalNd = 2 * ndPerUnit;
      mp = 2 * (ceWing1 - ndPerUnit) * lotSz; ml = totalNd * lotSz;
      be = [atm + ndPerUnit, co2.strike - ndPerUnit];
      nc = -totalNd * lotSz; margin = totalNd * lotSz;
      ltpParts = [
        {{ l: 'BUY 2x CE \u20b9' + atm.toLocaleString('en-IN'), v: s1, c: '#00c8e0' }},
        {{ l: 'SELL 4x CE \u20b9' + co1.strike.toLocaleString('en-IN'), v: s2, c: '#00c896' }},
        {{ l: 'BUY 2x CE \u20b9' + co2.strike.toLocaleString('en-IN'), v: s3, c: '#00c8e0' }}
      ];
      netDelta=2*(gCeAtm.delta - 2*gCo1.delta + gCo2.delta)*lotSz; netTheta=2*(gCeAtm.theta - 2*gCo1.theta + gCo2.theta)*lotSz; netVega=2*(gCeAtm.vega - 2*gCo1.vega + gCo2.vega)*lotSz; netGamma=2*(gCeAtm.gamma - 2*gCo1.gamma + gCo2.gamma)*lotSz;
      break;
    }}

    case 'double_fly': {{
      const cNd = ce_atm - 2 * co1.ltp + (co2.ltp || 40);
      const pNd = pe_atm - 2 * po1.ltp + (po2.ltp || 40);
      const totalNd = cNd + pNd;
      mp = ((ceWing1 - cNd) + (peWing1 - pNd)) * lotSz;
      ml = totalNd * lotSz;
      be = [atm - totalNd * 0.5, atm + totalNd * 0.5];
      nc = -totalNd * lotSz; margin = totalNd * lotSz;
      ltpParts = [
        {{ l: 'CALL FLY: BUY/SELL/BUY CE', v: ce_atm || 150, c: '#00c8e0' }},
        {{ l: 'PUT FLY:  BUY/SELL/BUY PE', v: pe_atm || 150, c: '#ff9090' }}
      ];
      netDelta=((gCeAtm.delta-2*gCo1.delta+gCo2.delta)+(gPeAtm.delta-2*gPo1.delta+gPo2.delta))*lotSz; netTheta=((gCeAtm.theta-2*gCo1.theta+gCo2.theta)+(gPeAtm.theta-2*gPo1.theta+gPo2.theta))*lotSz; netVega=((gCeAtm.vega-2*gCo1.vega+gCo2.vega)+(gPeAtm.vega-2*gPo1.vega+gPo2.vega))*lotSz; netGamma=((gCeAtm.gamma-2*gCo1.gamma+gCo2.gamma)+(gPeAtm.gamma-2*gPo1.gamma+gPo2.gamma))*lotSz;
      break;
    }}

    case 'double_condor': {{
      const cNd = ce_atm - co1.ltp - co2.ltp + (co3.ltp || 30);
      const pNd = pe_atm - po1.ltp - po2.ltp + (po3.ltp || 30);
      const totalNd = cNd + pNd;
      mp = ((ceWing1 - cNd) + (peWing1 - pNd)) * lotSz;
      ml = totalNd * lotSz;
      be = [atm - totalNd, atm + totalNd];
      nc = -totalNd * lotSz; margin = totalNd * lotSz;
      ltpParts = [
        {{ l: 'CE CONDOR \u20b9' + atm.toLocaleString('en-IN') + ' \u2192 \u20b9' + co3.strike.toLocaleString('en-IN'), v: ce_atm || 150, c: '#00c8e0' }},
        {{ l: 'PE CONDOR \u20b9' + atm.toLocaleString('en-IN') + ' \u2192 \u20b9' + po3.strike.toLocaleString('en-IN'), v: pe_atm || 150, c: '#ff9090' }}
      ];
      netDelta=((gCeAtm.delta-gCo1.delta-gCo2.delta+gCo3.delta)+(gPeAtm.delta-gPo1.delta-gPo2.delta+gPo3.delta))*lotSz; netTheta=((gCeAtm.theta-gCo1.theta-gCo2.theta+gCo3.theta)+(gPeAtm.theta-gPo1.theta-gPo2.theta+gPo3.theta))*lotSz; netVega=((gCeAtm.vega-gCo1.vega-gCo2.vega+gCo3.vega)+(gPeAtm.vega-gPo1.vega-gPo2.vega+gPo3.vega))*lotSz; netGamma=((gCeAtm.gamma-gCo1.gamma-gCo2.gamma+gCo3.gamma)+(gPeAtm.gamma-gPo1.gamma-gPo2.gamma+gPo3.gamma))*lotSz;
      break;
    }}

    case 'call_calendar': {{
      const farLTP  = ce_atm || 150;
      const nearLTP = Math.round(farLTP * 0.55);
      const nd = farLTP - nearLTP;
      mp = Math.round(nd * 0.80) * lotSz; ml = nd * lotSz;
      be = [atm - Math.round(nd * 0.6), atm + Math.round(nd * 0.6)];
      nc = -nd * lotSz; margin = nd * lotSz;
      ltpParts = [
        {{ l: 'SELL NEAR CE \u20b9' + atm.toLocaleString('en-IN') + ' (~est)', v: nearLTP, c: '#00c896' }},
        {{ l: 'BUY FAR CE \u20b9'   + atm.toLocaleString('en-IN'),             v: farLTP,  c: '#00c8e0' }}
      ];
      netDelta=gCeAtm.delta*0.1*lotSz; netTheta=Math.abs(gCeAtm.theta)*0.45*lotSz; netVega=gCeAtm.vega*0.3*lotSz; netGamma=gCeAtm.gamma*0.1*lotSz;
      break;
    }}
    case 'put_calendar': {{
      const farLTP  = pe_atm || 150;
      const nearLTP = Math.round(farLTP * 0.55);
      const nd = farLTP - nearLTP;
      mp = Math.round(nd * 0.80) * lotSz; ml = nd * lotSz;
      be = [atm - Math.round(nd * 0.6), atm + Math.round(nd * 0.6)];
      nc = -nd * lotSz; margin = nd * lotSz;
      ltpParts = [
        {{ l: 'SELL NEAR PE \u20b9' + atm.toLocaleString('en-IN') + ' (~est)', v: nearLTP, c: '#00c896' }},
        {{ l: 'BUY FAR PE \u20b9'   + atm.toLocaleString('en-IN'),             v: farLTP,  c: '#ff9090' }}
      ];
      netDelta=gPeAtm.delta*0.1*lotSz; netTheta=Math.abs(gPeAtm.theta)*0.45*lotSz; netVega=gPeAtm.vega*0.3*lotSz; netGamma=gPeAtm.gamma*0.1*lotSz;
      break;
    }}
    case 'diagonal_calendar': {{
      const farLTP  = ce_atm || 150;
      const nearLTP = Math.round((co1.ltp || 80) * 0.55);
      const nd = farLTP - nearLTP;
      mp = Math.round(nd * 0.70) * lotSz; ml = nd * lotSz;
      be = [atm + Math.round(nd * 0.25), atm + Math.round(nd * 1.4)];
      nc = -nd * lotSz; margin = nd * lotSz;
      ltpParts = [
        {{ l: 'SELL NEAR CE \u20b9' + co1.strike.toLocaleString('en-IN') + ' (~est)', v: nearLTP, c: '#00c896' }},
        {{ l: 'BUY FAR CE \u20b9'   + atm.toLocaleString('en-IN'),                   v: farLTP,  c: '#00c8e0' }}
      ];
      netDelta=(gCeAtm.delta - gCo1.delta*0.5)*0.1*lotSz; netTheta=Math.abs(gCeAtm.theta)*0.35*lotSz; netVega=(gCeAtm.vega-gCo1.vega*0.5)*0.3*lotSz; netGamma=(gCeAtm.gamma - gCo1.gamma*0.5)*0.1*lotSz;
      break;
    }}

    default: {{
      const p = ce_atm || 150;
      mp = p * lotSz * 0.5; ml = p * lotSz * 0.3;
      be = [atm]; nc = -p * 0.3 * lotSz;
      margin = p * lotSz; rrRatio = 1.5;
      ltpParts = [{{ l: 'ATM \u20b9' + atm.toLocaleString('en-IN'), v: p, c: '#00c8e0' }}];
    }}
  }}

  const beStr     = be.map(v => '\u20b9' + Math.round(v).toLocaleString('en-IN')).join(' / ');
  const mpStr     = mp === 999999 ? 'Unlimited' : '\u20b9' + Math.round(mp).toLocaleString('en-IN');
  const mlStr     = ml === 999999 ? 'Unlimited' : '\u20b9' + Math.round(ml).toLocaleString('en-IN');

  // ── Slippage deduction on net credit/debit ──────────────────────────
  // Realistic fill price is worse than LTP. Slippage grows with leg count
  // because each leg needs to cross the bid-ask spread individually.
  // 1 leg: 0.3%  |  2 legs: 0.5%  |  3–4 legs: 0.7%
  const legCount     = ltpParts.length;
  const slipPct      = legCount >= 3 ? 0.007 : legCount === 2 ? 0.005 : 0.003;
  const slipAmt      = Math.round(Math.abs(nc) * slipPct);
  const ncAdjusted   = nc >= 0 ? nc - slipAmt : nc - slipAmt;  // credit gets less, debit pays more
  const ncStr        = (ncAdjusted >= 0 ? '+ ' : '- ') + '\u20b9' + Math.abs(Math.round(ncAdjusted)).toLocaleString('en-IN');
  const slipNote     = slipAmt > 0
    ? `<div style="font-size:10px;color:rgba(255,185,0,.5);padding:2px 0 0;font-family:DM Mono,monospace;">
         ~\u20b9${{slipAmt.toLocaleString('en-IN')}} slippage est. (${{(slipPct*100).toFixed(1)}}% \u00d7 ${{legCount}} leg${{legCount>1?'s':''}})
       </div>` : '';

  const marginStr = '\u20b9' + Math.round(margin).toLocaleString('en-IN');
  const rrStr     = rrRatio === 0 ? '\u221e' : ('1:' + Math.abs(rrRatio));
  const mpPct     = mp === 999999 ? '\u221e' : (ml > 0 ? (mp / ml * 100).toFixed(0) + '%' : '\u2014');
  const ltpStr    = ltpParts.map(x =>
    `<span style="display:inline-flex;align-items:center;gap:4px;margin-bottom:2px;">
      <span style="font-size:12.3px;color:rgba(255,255,255,.70);">${{x.l}}</span>
      <span style="font-family:'DM Mono',monospace;font-weight:700;color:${{x.c}};">\u20b9${{x.v.toFixed(2)}}</span>
    </span>`
  ).join('<br>');
  const strikeStr = 'ATM \u20b9' + atm.toLocaleString('en-IN');

  // True PoP from Python-computed IV-based N(d2) map
  const truePop = TRUE_POP_MAP[shape] !== undefined ? TRUE_POP_MAP[shape] : null;

  // ── Theta/Vega Ratio ────────────────────────────────────────────────
  // Answers: "for each ₹1 of daily theta collected, how much IV-spike risk am I carrying?"
  // T/V = |netTheta| / |netVega|
  // > 0.10 = acceptable compensation (earn back vega exposure in ~10 days of theta)
  // < 0.05 = poorly compensated (a 1% IV spike costs 20+ days of theta)
  // N/A for strategies with near-zero vega (synthetics, calendars have their own metric)
  const absVega  = Math.abs(Math.round(netVega * 100) / 100);
  const absTheta = Math.abs(Math.round(netTheta * 100) / 100);
  const tvRatio  = absVega > 0.5 ? Math.round((absTheta / absVega) * 1000) / 1000 : null;

  // ── Theoretical Expected Value ──────────────────────────────────────
  // EV = (truePop/100 × maxProfit) − ((1 − truePop/100) × maxLoss)
  // Positive EV = trade has mathematical edge. Negative EV = avoid.
  // Important: this is MODEL-DEPENDENT — assumes log-normal returns and
  // accurate IV. Fat tails (NIFTY events) can make actual EV worse.
  // "Unlimited" max loss/profit = EV not meaningful → null.
  let evRaw = null;
  const tp_ = TRUE_POP_MAP[shape];
  if (tp_ !== undefined && mp < 999990 && ml < 999990 && mp > 0 && ml > 0) {{
    const pWin  = tp_ / 100;
    const pLose = 1 - pWin;
    evRaw = Math.round(pWin * mp - pLose * ml);
  }}
  const evStr = evRaw === null ? null
    : (evRaw >= 0 ? '+' : '') + '\u20b9' + Math.abs(evRaw).toLocaleString('en-IN');
  const evCol = evRaw === null ? '#888'
    : evRaw >= 500 ? '#38d888' : evRaw >= 0 ? '#ffcc00' : evRaw >= -500 ? '#ffaa00' : '#f04050';
  const evLabel = evRaw === null ? 'N/A (unlimited leg)'
    : evRaw >= 500  ? 'Positive edge — trade has merit'
    : evRaw >= 0    ? 'Marginal — thin edge, watch slippage'
    : evRaw >= -500 ? 'Negative EV — poor R:R vs PoP'
    :                 'Avoid — strongly negative EV';

  return {{edgeScore:es, truePop, tvRatio, evRaw, evStr, evCol, evLabel,
           mpStr,mlStr,rrStr,beStr,ncStr,slipNote,marginStr,mpPct,strikeStr,ltpStr,
           mpRaw:mp,mlRaw:ml,ncRaw:Math.round(ncAdjusted),ncPositive:ncAdjusted>=0,
           netDelta:Math.round(netDelta*100)/100,
           netTheta:Math.round(netTheta*100)/100,
           netVega:Math.round(netVega*100)/100,
           netGamma:Math.round(netGamma*10000)/10000,
           mlRawVal:ml}};
}}

function renderMetrics(m, scoreBreakdown) {{
  const es  = m.edgeScore || 50;
  const tp  = m.truePop;
  const esc = es>=70?'#38d888': es>=55?'#ffcc00': es>=45?'#ffaa00':'#f04050';
  const tpc = tp===null ? '#888' : tp>=70?'#38d888': tp>=55?'#ffcc00': tp>=45?'#ffaa00':'#f04050';
  const nc  = m.ncPositive?'#38d888':'#f04050';

  // ── EdgeScore breakdown strip ──────────────────────────────────────
  const sbHtml = scoreBreakdown ? `
    <div style="background:rgba(255,185,0,.04);border-top:1px solid rgba(255,185,0,.12);padding:9px 12px 11px;">
      <div style="font-size:11px;letter-spacing:1.8px;text-transform:uppercase;color:rgba(255,210,0,.85);margin-bottom:7px;font-family:DM Mono,monospace;font-weight:700;">EDGE SCORE BREAKDOWN</div>
      <div style="display:flex;flex-wrap:wrap;gap:5px;">
        <span style="font-size:13px;background:rgba(0,0,0,.3);padding:3px 9px;border-radius:4px;border:1px solid rgba(255,185,0,.25);color:rgba(255,210,0,.85);font-family:DM Mono,monospace;">Base <b style="color:#ffcc00;">50%</b></span>
        <span style="font-size:13px;background:rgba(0,0,0,.3);padding:3px 9px;border-radius:4px;border:1px solid rgba(255,185,0,.25);color:rgba(255,210,0,.85);font-family:DM Mono,monospace;">Bias <b style="color:${{scoreBreakdown.biasAdj>=0?'#38d888':'#f04050'}};">${{scoreBreakdown.biasAdj>=0?'+':''}}${{scoreBreakdown.biasAdj}}</b></span>
        <span style="font-size:13px;background:rgba(0,0,0,.3);padding:3px 9px;border-radius:4px;border:1px solid rgba(255,185,0,.25);color:rgba(255,210,0,.85);font-family:DM Mono,monospace;">S/R <b style="color:${{scoreBreakdown.srAdj>=0?'#38d888':'#f04050'}};">${{scoreBreakdown.srAdj>=0?'+':''}}${{scoreBreakdown.srAdj}}</b></span>
        <span style="font-size:13px;background:rgba(0,0,0,.3);padding:3px 9px;border-radius:4px;border:1px solid rgba(255,185,0,.25);color:rgba(255,210,0,.85);font-family:DM Mono,monospace;">OI <b style="color:${{scoreBreakdown.oiAdj>=0?'#38d888':'#f04050'}};">${{scoreBreakdown.oiAdj>=0?'+':''}}${{scoreBreakdown.oiAdj}}</b></span>
        <span style="font-size:13px;background:rgba(0,0,0,.3);padding:3px 9px;border-radius:4px;border:1px solid rgba(255,185,0,.25);color:rgba(255,210,0,.85);font-family:DM Mono,monospace;">PCR <b style="color:${{scoreBreakdown.pcrAdj>=0?'#38d888':'#f04050'}};">${{scoreBreakdown.pcrAdj>=0?'+':''}}${{scoreBreakdown.pcrAdj}}</b></span>
        <span style="font-size:13px;background:rgba(0,0,0,.3);padding:3px 9px;border-radius:4px;border:1px solid rgba(255,185,0,.25);color:rgba(255,210,0,.85);font-family:DM Mono,monospace;">Strat <b style="color:${{scoreBreakdown.stratAdj>=0?'#38d888':'#f04050'}};">${{scoreBreakdown.stratAdj>=0?'+':''}}${{scoreBreakdown.stratAdj}}</b></span>
        <span style="font-size:13px;background:rgba(0,0,0,.3);padding:3px 9px;border-radius:4px;border:1px solid rgba(255,185,0,.25);color:rgba(255,210,0,.85);font-family:DM Mono,monospace;">IVP ${{IVP}}% <b style="color:${{(scoreBreakdown.ivpAdj||0)>=0?'#38d888':'#f04050'}};">${{(scoreBreakdown.ivpAdj||0)>=0?'+':''}}${{scoreBreakdown.ivpAdj||0}}</b></span>
      </div>
    </div>` : '';

  return `
  <div style="display:flex;justify-content:space-between;align-items:center;padding:8px 12px;border-bottom:1px solid rgba(255,185,0,.12);background:rgba(255,185,0,.05);">
    <span style="font-family:DM Mono,monospace;font-size:11px;color:rgba(255,210,0,.85);letter-spacing:1.2px;text-transform:uppercase;font-weight:700;">Strike Price</span>
    <span style="font-family:DM Mono,monospace;font-size:15px;font-weight:700;text-align:right;color:#ffcc00;">${{m.strikeStr}}</span>
  </div>
  <div style="display:flex;justify-content:space-between;align-items:center;padding:8px 12px;border-bottom:1px solid rgba(255,185,0,.1);background:rgba(255,185,0,.03);">
    <span style="font-family:DM Mono,monospace;font-size:11px;color:rgba(255,210,0,.9);letter-spacing:1.2px;text-transform:uppercase;font-weight:700;">LTP (per leg)</span>
    <span style="font-family:DM Mono,monospace;font-size:13px;font-weight:700;text-align:right;line-height:1.8;display:flex;flex-direction:column;align-items:flex-end;">${{m.ltpStr}}</span>
  </div>

  <!-- ── EdgeScore + True PoP side-by-side ─────────────────────── -->
  <div style="display:grid;grid-template-columns:1fr 1fr;border-bottom:1px solid rgba(255,185,0,.1);">
    <div style="padding:11px 12px;border-right:1px solid rgba(255,185,0,.1);">
      <div style="font-family:DM Mono,monospace;font-size:11px;color:rgba(255,210,0,.85);letter-spacing:1.2px;text-transform:uppercase;margin-bottom:3px;font-weight:700;">Edge Score</div>
      <div style="font-family:DM Mono,monospace;font-size:26px;font-weight:800;color:${{esc}};">${{es}}%</div>
      <div style="font-family:DM Mono,monospace;font-size:10px;color:rgba(255,255,255,.42);margin-top:2px;">Bias + S/R + OI + PCR + IVP</div>
    </div>
    <div style="padding:11px 12px;">
      <div style="display:flex;align-items:center;justify-content:space-between;">
        <div style="font-family:DM Mono,monospace;font-size:11px;color:rgba(255,210,0,.85);letter-spacing:1.2px;text-transform:uppercase;margin-bottom:3px;font-weight:700;">True PoP (IV)</div>
        <div style="font-family:DM Mono,monospace;font-size:10px;font-weight:700;padding:1px 6px;border-radius:4px;
          background:${{IVP>=70?'rgba(56,216,136,.15)':IVP>=35?'rgba(255,185,0,.12)':IVP<20?'rgba(240,64,80,.18)':'rgba(100,128,255,.12)'}};
          color:${{IVP>=70?'#38d888':IVP>=35?'#ffcc00':IVP<20?'#f04050':'#8aa0ff'}};
          border:1px solid ${{IVP>=70?'rgba(56,216,136,.3)':IVP>=35?'rgba(255,185,0,.3)':IVP<20?'rgba(240,64,80,.3)':'rgba(100,128,255,.3)'}};
          white-space:nowrap;">
          IVP ${{IVP}}%
        </div>
      </div>
      <div style="font-family:DM Mono,monospace;font-size:26px;font-weight:800;color:${{tpc}};">${{tp !== null ? tp+'%' : '—'}}</div>
      <div style="font-family:DM Mono,monospace;font-size:10px;color:rgba(255,255,255,.42);margin-top:2px;">N(d2) · P(expire OTM)</div>
    </div>
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr;border-bottom:1px solid rgba(255,185,0,.1);">
    <div style="padding:11px 12px;border-right:1px solid rgba(255,185,0,.1);">
      <div style="font-family:DM Mono,monospace;font-size:11px;color:rgba(255,210,0,.85);letter-spacing:1.2px;text-transform:uppercase;margin-bottom:5px;font-weight:700;">Max Profit</div>
      <div style="font-family:DM Mono,monospace;font-size:20px;font-weight:700;color:#38d888;">${{m.mpStr}} <span style="font-size:12px;opacity:.7;">${{m.mpPct}}</span></div>
    </div>
    <div style="padding:11px 12px;">
      <div style="font-family:DM Mono,monospace;font-size:11px;color:rgba(255,210,0,.85);letter-spacing:1.2px;text-transform:uppercase;margin-bottom:5px;font-weight:700;">Max Loss</div>
      <div style="font-family:DM Mono,monospace;font-size:20px;font-weight:700;color:#f04050;">${{m.mlStr}}</div>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;border-bottom:1px solid rgba(255,185,0,.1);">
    <div style="padding:11px 12px;border-right:1px solid rgba(255,185,0,.1);">
      <div style="font-family:DM Mono,monospace;font-size:11px;color:rgba(255,210,0,.85);letter-spacing:1.2px;text-transform:uppercase;margin-bottom:5px;font-weight:700;">R/R Ratio</div>
      <div style="font-family:DM Mono,monospace;font-size:20px;font-weight:700;color:#ffcc00;">${{m.rrStr}}</div>
    </div>
    <div style="padding:11px 12px;">
      <div style="font-family:DM Mono,monospace;font-size:11px;color:rgba(255,210,0,.85);letter-spacing:1.2px;text-transform:uppercase;margin-bottom:5px;font-weight:700;">Breakeven</div>
      <div style="font-family:DM Mono,monospace;font-size:14px;font-weight:700;color:#fff8e0;">${{m.beStr}}</div>
    </div>
  </div>

  <!-- ── Theoretical EV row ─────────────────────────────────────── -->
  <div style="padding:10px 12px;border-bottom:1px solid rgba(255,185,0,.1);background:rgba(0,0,0,.2);">
    <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;">
      <div>
        <div style="font-family:DM Mono,monospace;font-size:11px;color:rgba(255,210,0,.85);letter-spacing:1.2px;text-transform:uppercase;margin-bottom:3px;font-weight:700;">
          Theoretical EV
          <span style="font-size:9px;font-weight:400;color:rgba(255,255,255,.35);letter-spacing:.5px;text-transform:none;margin-left:5px;">model-based · per lot</span>
        </div>
        <div style="font-family:DM Mono,monospace;font-size:22px;font-weight:800;color:${{m.evCol || '#888'}};">
          ${{m.evStr || '—'}}
        </div>
      </div>
      <div style="font-family:DM Mono,monospace;font-size:12px;color:${{m.evCol || 'rgba(255,255,255,.4)'}};
           padding:4px 10px;border-radius:6px;
           background:${{m.evRaw === null ? 'rgba(255,255,255,.04)' : m.evRaw >= 0 ? 'rgba(56,216,136,.08)' : 'rgba(240,64,80,.08)'}};
           border:1px solid ${{m.evRaw === null ? 'rgba(255,255,255,.08)' : m.evRaw >= 0 ? 'rgba(56,216,136,.25)' : 'rgba(240,64,80,.25)'}};
           white-space:nowrap;max-width:260px;text-align:right;">
        ${{m.evLabel || 'N/A'}}
      </div>
    </div>
    <div style="font-family:DM Mono,monospace;font-size:10px;color:rgba(255,255,255,.28);margin-top:5px;line-height:1.5;">
      EV = (TruePoP × MaxProfit) − ((1−TruePoP) × MaxLoss) · assumes log-normal · events inflate actual tail risk
    </div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;border-bottom:1px solid rgba(255,185,0,.1);">
    <div style="padding:9px 10px;border-right:1px solid rgba(255,185,0,.08);">
      <div style="font-family:DM Mono,monospace;font-size:10px;color:rgba(255,205,60,.82);letter-spacing:1px;text-transform:uppercase;margin-bottom:4px;font-weight:700;">Net CR/DR (after slip)</div>
      <div style="font-family:DM Mono,monospace;font-size:13px;font-weight:700;color:${{nc}};">${{m.ncStr}}</div>
      ${{m.slipNote}}
    </div>
    <div style="padding:9px 10px;">
      <div style="font-family:DM Mono,monospace;font-size:10px;color:rgba(255,205,60,.82);letter-spacing:1px;text-transform:uppercase;margin-bottom:4px;font-weight:700;">Margin Required</div>
      <div style="font-family:DM Mono,monospace;font-size:13px;font-weight:700;color:#fff8e0;">${{m.marginStr}}</div>
    </div>
  </div>
  ${{sbHtml}}
  ${{buildIntradaySim(m)}}`;
}}
// ── Intraday P&L Simulator ───────────────────────────────────────────────────
function buildIntradaySim(m) {{
  const lotSz   = OC.lotSize;
  const maxL    = m.mlRawVal === 999999 ? null : m.mlRawVal;
  const maxP    = m.mpRaw    === 999999 ? null : m.mpRaw;
  const nd      = m.netDelta;
  const nt      = m.netTheta;   // ₹ per day
  const nv      = m.netVega;
  const ng      = m.netGamma;

  // ── Days to expiry (IST) ────────────────────────────────────────────
  function calcDaysToExpiry() {{
    try {{
      const expStr = OC.expiry;
      const parts  = expStr.split('-');
      const months = {{Jan:0,Feb:1,Mar:2,Apr:3,May:4,Jun:5,Jul:6,Aug:7,Sep:8,Oct:9,Nov:10,Dec:11}};
      const expDate = new Date(Date.UTC(parseInt(parts[2]), months[parts[1]], parseInt(parts[0])));
      const nowUtc  = Date.now() + (new Date().getTimezoneOffset() * 60000);
      const nowIst  = new Date(nowUtc + 5.5 * 3600000);
      const todayIst = new Date(Date.UTC(nowIst.getUTCFullYear(), nowIst.getUTCMonth(), nowIst.getUTCDate()));
      return Math.max(1, Math.round((expDate - todayIst) / 86400000));
    }} catch(e) {{ return 4; }}
  }}
  const maxDays = calcDaysToExpiry();

  const moves = [-500,-400,-300,-200,-150,-100,-50,0,50,100,150,200,300,400,500];

  function calcPnl(movePts, days) {{
    // Asymmetric IV response (leverage effect):
    // Down moves spike IV harder — historically ~3x vs up moves for NIFTY.
    // Up moves compress IV softly because IV floors at ~10 on big rallies.
    const ivEst = movePts < 0
      ? -(movePts / OC.spot) * 600   // down: IV spikes (e.g. -200pt → ~+3.0% IV)
      : -(movePts / OC.spot) * 200;  // up:   IV compresses softly (~-1.0% IV)
    let pnl = nd * movePts + 0.5 * ng * movePts * movePts + nv * ivEst + (nt * days);
    if (maxL !== null) pnl = Math.max(-maxL, pnl);
    if (maxP !== null) pnl = Math.min(maxP * 0.9, pnl);
    return Math.round(pnl);
  }}

  function buildRows(days) {{
    return moves.map(mv => {{
      const pnl   = calcPnl(mv, days);
      const col   = pnl > 100 ? '#38d888' : pnl > 0 ? '#ffcc00' : pnl > -200 ? '#ffaa00' : '#f04050';
      const mvcol = mv > 0 ? '#38d888' : mv < 0 ? '#f04050' : '#ffcc00';
      const mvbg  = mv > 0 ? 'rgba(56,216,136,.12)' : mv < 0 ? 'rgba(240,64,80,.18)' : 'rgba(255,185,0,.18)';
      const mvlbl = mv > 0 ? '+' + mv : mv === 0 ? 'Flat' : String(mv);
      const pctmp = maxP ? ((pnl / maxP) * 100).toFixed(0) + '%' : '\u2014';
      const rowBg = mv === 0 ? 'background:rgba(255,185,0,.05);' : '';
      return `<tr style="${{rowBg}}">
        <td style="padding:6px 10px;white-space:nowrap;">
          <span style="font-family:'DM Mono',monospace;font-size:13px;font-weight:700;padding:4px 8px;border-radius:4px;background:${{mvbg}};color:${{mvcol}};white-space:nowrap;display:inline-block;min-width:56px;text-align:center;">${{mvlbl}}${{mv!==0?'p':''}}</span>
        </td>
        <td style="padding:6px 8px;font-family:'DM Mono',monospace;font-size:13px;color:rgba(255,200,80,.82);white-space:nowrap;text-align:left;">${{(OC.spot+mv).toLocaleString('en-IN')}}</td>
        <td style="padding:6px 8px;font-family:'DM Mono',monospace;font-weight:700;font-size:15px;color:${{col}};white-space:nowrap;text-align:right;">${{pnl>=0?'+':''}}\u20b9${{Math.abs(pnl).toLocaleString('en-IN')}}</td>
        <td style="padding:6px 6px;font-family:'DM Mono',monospace;font-size:13px;font-weight:700;color:${{col}};text-align:right;white-space:nowrap;">${{pctmp}}</td>
      </tr>`;
    }}).join('');
  }}

  const ntCol  = nt >= 0 ? '#38d888' : '#f04050';
  const ntSign = nt >= 0 ? '+' : '';
  const ndStr  = (nd >= 0 ? '+' : '') + '\u20b9' + Math.abs(nd).toFixed(2);
  const ntStr  = ntSign + '\u20b9' + Math.abs(Math.round(nt));
  const nvStr  = (nv >= 0 ? '+' : '') + '\u20b9' + Math.abs(nv).toFixed(2);
  const flatPnl = Math.round(nd * 0 + nt);   // P&L at flat (0 move), 1 day
  const flatCol = flatPnl >= 0 ? '#38d888' : '#f04050';

  // Day selector buttons
  const dayBtnsHtml = Array.from({{length: maxDays}}, (_,i)=>i+1).map(d=>{{
    const isExp = d === maxDays;
    const lbl   = isExp ? d+'D \u2605 Expiry' : d+'D';
    const act   = d === 1;
    return `<button id="SIDBTN_${{d}}"
      onclick="SIDSEL(${{d}})"
      style="padding:4px 10px;font-family:DM Mono,monospace;font-size:12px;font-weight:700;
        border-radius:4px;cursor:pointer;white-space:nowrap;transition:all .15s;
        border:1px solid ${{act?'#ffcc00':'rgba(255,185,0,.3)'}};
        color:${{act?'#ffcc00':'rgba(255,200,80,.5)'}};
        background:${{act?'rgba(255,185,0,.15)':'transparent'}};">${{lbl}}</button>`;
  }}).join('');

  const slMin = Math.round((OC.spot - 400) / 25) * 25;
  const slMax = Math.round((OC.spot + 400) / 25) * 25;
  const simId = 'sim_' + Math.random().toString(36).slice(2,7);

  const dayBtns = dayBtnsHtml.replace(/SIDBTN/g, 'daybtn_'+simId).replace(/SIDSEL/g, 'selDay_'+simId);

  return `
<div style="border-top:1px solid rgba(255,185,0,.2);background:rgba(11,8,0,.6);" onclick="event.stopPropagation()">
  <!-- Sub-tabs -->
  <div style="display:flex;border-bottom:1px solid rgba(255,185,0,.15);">
    <button id="${{simId}}_t1" onclick="simTab('${{simId}}','t1')" style="flex:1;padding:10px 4px;font-family:DM Mono,monospace;font-size:13px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;cursor:pointer;border:none;border-bottom:2px solid #ffcc00;color:#ffcc00;background:rgba(255,185,0,.07);transition:all .2s;">📊 SCENARIOS</button>
    <button id="${{simId}}_t2" onclick="simTab('${{simId}}','t2')" style="flex:1;padding:10px 4px;font-family:DM Mono,monospace;font-size:13px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;cursor:pointer;border:none;border-bottom:2px solid transparent;color:rgba(255,185,0,.5);background:transparent;transition:all .2s;">Δ GREEKS</button>
    <button id="${{simId}}_t3" onclick="simTab('${{simId}}','t3')" style="flex:1;padding:10px 4px;font-family:DM Mono,monospace;font-size:13px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;cursor:pointer;border:none;border-bottom:2px solid transparent;color:rgba(255,185,0,.5);background:transparent;transition:all .2s;">⟺ SLIDER</button>
  </div>

  <!-- TAB 1: Scenarios -->
  <div id="${{simId}}_c1">

    <!-- Day Selector -->
    <div style="display:flex;align-items:center;gap:7px;padding:8px 12px;border-bottom:1px solid rgba(255,185,0,.1);background:rgba(0,0,0,.25);flex-wrap:wrap;">
      <span style="font-family:DM Mono,monospace;font-size:11px;font-weight:700;letter-spacing:1px;color:rgba(255,200,60,.7);text-transform:uppercase;white-space:nowrap;">📅 DAYS TO EXPIRY:</span>
      ${{dayBtns}}
      <span style="font-family:DM Mono,monospace;font-size:10px;color:rgba(255,185,0,.35);margin-left:2px;">Max ${{maxDays}}D · IST</span>
    </div>

    <div style="display:flex;align-items:center;justify-content:space-between;padding:8px 12px 6px;border-bottom:1px solid rgba(255,185,0,.12);">
      <div id="${{simId}}_hdr" style="font-family:DM Mono,monospace;font-size:12px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;color:rgba(255,210,0,.95);">📋 1 DAY P&amp;L SCENARIOS</div>
      <div style="font-family:DM Mono,monospace;font-size:11px;color:rgba(255,200,60,.7);background:rgba(0,0,0,.25);padding:2px 8px;border-radius:4px;">Δ + ½Γ + ν(IV) + Θ×days</div>
    </div>
    <table style="width:100%;border-collapse:collapse;">
      <thead>
        <tr style="background:rgba(0,0,0,.3);">
          <th style="padding:6px 10px;font-family:DM Mono,monospace;font-size:11px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;color:rgba(255,205,60,.85);text-align:left;border-bottom:1px solid rgba(255,185,0,.08);">MOVE</th>
          <th id="${{simId}}_col2" style="padding:6px 8px;font-family:DM Mono,monospace;font-size:11px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;color:rgba(255,205,60,.85);text-align:left;border-bottom:1px solid rgba(255,185,0,.08);">1 DAY P&amp;L</th>
          <th style="padding:6px 8px;font-family:DM Mono,monospace;font-size:11px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;color:rgba(255,205,60,.85);text-align:right;border-bottom:1px solid rgba(255,185,0,.08);">VS MAX</th>
          <th style="padding:6px 6px;border-bottom:1px solid rgba(255,185,0,.08);"></th>
        </tr>
      </thead>
      <tbody id="${{simId}}_tbody">${{buildRows(1)}}</tbody>
    </table>
    <div style="padding:9px 12px;font-family:DM Mono,monospace;font-size:11px;color:rgba(255,200,70,.75);background:rgba(0,0,0,.25);border-top:1px solid rgba(255,185,0,.1);line-height:1.7;">
      P&amp;L = <span style="color:rgba(255,215,0,.9);">Δ×move + ½Γ×move\u00b2 + \u03bd×\u0394IV + \u0398\u00d7<span id="${{simId}}_daylbl">1</span> day(s)</span>. Max profit ${{m.mpStr}} at expiry only.
    </div>
  </div>


  <!-- TAB 2: Greeks Breakdown -->
  <div id="${{simId}}_c2" style="display:none;">
    <div style="display:flex;align-items:center;justify-content:space-between;padding:10px 12px 8px;border-bottom:1px solid rgba(255,185,0,.12);">
      <div style="font-family:DM Mono,monospace;font-size:12px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;color:rgba(255,210,0,.95);">🔬 NET GREEKS (per lot)</div>
      <div style="font-family:DM Mono,monospace;font-size:11px;color:rgba(255,200,60,.7);background:rgba(0,0,0,.25);padding:2px 8px;border-radius:4px;">values per lot · today</div>
    </div>
    <div style="padding:10px 12px;display:flex;flex-direction:column;gap:5px;">
      <div style="display:flex;align-items:center;gap:10px;background:rgba(56,216,136,.06);border:1px solid rgba(56,216,136,.18);border-radius:8px;padding:9px 12px;">
        <div style="width:32px;height:32px;border-radius:6px;background:rgba(56,216,136,.15);border:1px solid rgba(56,216,136,.28);display:flex;align-items:center;justify-content:center;font-size:18px;font-weight:700;color:#38d888;flex-shrink:0;">Δ</div>
        <div style="flex:1;min-width:0;">
          <div style="font-family:DM Mono,monospace;font-size:12px;font-weight:700;letter-spacing:1.2px;color:rgba(56,216,136,.9);text-transform:uppercase;">DELTA</div>
          <div style="font-family:DM Mono,monospace;font-size:11px;color:rgba(255,200,80,.95);">per 1pt move</div>
        </div>
        <div style="font-family:DM Mono,monospace;font-size:20px;font-weight:700;color:${{nd>=0?'#38d888':'#f04050'}};white-space:nowrap;">${{ndStr}}</div>
      </div>
      <div style="display:flex;align-items:center;gap:10px;background:rgba(240,64,80,.06);border:1px solid rgba(240,64,80,.18);border-radius:8px;padding:9px 12px;">
        <div style="width:32px;height:32px;border-radius:6px;background:rgba(240,64,80,.15);border:1px solid rgba(240,64,80,.28);display:flex;align-items:center;justify-content:center;font-size:18px;font-weight:700;color:#f04050;flex-shrink:0;">Θ</div>
        <div style="flex:1;min-width:0;">
          <div style="font-family:DM Mono,monospace;font-size:12px;font-weight:700;letter-spacing:1.2px;color:rgba(240,64,80,.9);text-transform:uppercase;">THETA</div>
          <div style="font-family:DM Mono,monospace;font-size:11px;color:rgba(255,200,80,.95);">decay / day</div>
        </div>
        <div style="font-family:DM Mono,monospace;font-size:20px;font-weight:700;color:${{ntCol}};white-space:nowrap;">${{ntSign}}\u20b9${{Math.abs(Math.round(nt))}}</div>
      </div>
      <div style="display:flex;align-items:center;gap:10px;background:rgba(255,185,0,.06);border:1px solid rgba(255,185,0,.2);border-radius:8px;padding:9px 12px;">
        <div style="width:32px;height:32px;border-radius:6px;background:rgba(255,185,0,.15);border:1px solid rgba(255,185,0,.28);display:flex;align-items:center;justify-content:center;font-size:18px;font-weight:700;color:#ffcc00;flex-shrink:0;">ν</div>
        <div style="flex:1;min-width:0;">
          <div style="font-family:DM Mono,monospace;font-size:12px;font-weight:700;letter-spacing:1.2px;color:rgba(255,204,0,.9);text-transform:uppercase;">VEGA</div>
          <div style="font-family:DM Mono,monospace;font-size:11px;color:rgba(255,200,80,.95);">per 1% IV</div>
        </div>
        <div style="font-family:DM Mono,monospace;font-size:20px;font-weight:700;color:#ffcc00;white-space:nowrap;">${{nvStr}}</div>
      </div>
      ${{(()=>{{
        // ── Theta / Vega ratio ─────────────────────────────────────────
        // T/V = |netTheta| / |netVega| per lot.
        // Tells you: "for each ₹1 of daily theta, how much IV-spike risk?"
        // ≥ 0.10 = well compensated  |  0.05–0.10 = marginal  |  < 0.05 = poorly compensated
        // N/A when vega ≈ 0 (pure directional strategies, long synthetics).
        const tv = m.tvRatio;
        if (tv === null) {{
          return `<div style="display:flex;align-items:center;gap:10px;background:rgba(100,128,255,.05);border:1px solid rgba(100,128,255,.15);border-radius:8px;padding:9px 12px;">
            <div style="width:32px;height:32px;border-radius:6px;background:rgba(100,128,255,.12);border:1px solid rgba(100,128,255,.2);display:flex;align-items:center;justify-content:center;font-size:14px;font-weight:700;color:#8aa0ff;flex-shrink:0;">T/V</div>
            <div style="flex:1;min-width:0;">
              <div style="font-family:DM Mono,monospace;font-size:12px;font-weight:700;letter-spacing:1.2px;color:rgba(138,160,255,.9);text-transform:uppercase;">THETA / VEGA</div>
              <div style="font-family:DM Mono,monospace;font-size:11px;color:rgba(255,200,80,.95);">compensation ratio</div>
            </div>
            <div style="font-family:DM Mono,monospace;font-size:16px;font-weight:700;color:rgba(255,255,255,.4);">N/A</div>
          </div>`;
        }}
        const tvCol  = tv >= 0.10 ? '#38d888' : tv >= 0.05 ? '#ffcc00' : '#f04050';
        const tvLbl  = tv >= 0.10 ? 'Well compensated' : tv >= 0.05 ? 'Marginal — monitor' : 'Poorly compensated';
        const tvNote = tv >= 0.10
          ? `Earn back full vega in ~${{Math.round(1/tv)}} days theta`
          : tv >= 0.05
          ? `A 1% IV spike costs ~${{Math.round(1/tv)}} days theta`
          : `A 1% IV spike costs ${{Math.round(1/tv)}}+ days theta — high risk`;
        return `<div style="display:flex;align-items:center;gap:10px;background:rgba(100,128,255,.05);border:1px solid rgba(100,128,255,.15);border-radius:8px;padding:9px 12px;">
          <div style="width:32px;height:32px;border-radius:6px;background:rgba(100,128,255,.12);border:1px solid rgba(100,128,255,.2);display:flex;align-items:center;justify-content:center;font-size:14px;font-weight:700;color:#8aa0ff;flex-shrink:0;">T/V</div>
          <div style="flex:1;min-width:0;">
            <div style="font-family:DM Mono,monospace;font-size:12px;font-weight:700;letter-spacing:1.2px;color:rgba(138,160,255,.9);text-transform:uppercase;">THETA / VEGA RATIO</div>
            <div style="font-family:DM Mono,monospace;font-size:11px;color:rgba(255,200,80,.95);">${{tvNote}}</div>
          </div>
          <div style="text-align:right;">
            <div style="font-family:DM Mono,monospace;font-size:20px;font-weight:700;color:${{tvCol}};white-space:nowrap;">${{tv.toFixed(3)}}</div>
            <div style="font-family:DM Mono,monospace;font-size:10px;color:${{tvCol}};opacity:.8;">${{tvLbl}}</div>
          </div>
        </div>`;
      }})()}}
    </div>
    <div style="border-top:1px solid rgba(255,185,0,.12);margin:0 12px;"></div>
    <div style="padding:10px 12px 10px;">
      <div style="font-family:DM Mono,monospace;font-size:11px;color:rgba(255,210,0,.85);letter-spacing:1.2px;text-transform:uppercase;margin-bottom:8px;font-weight:700;">TODAY'S P&amp;L IF MARKET IS FLAT</div>
      <div style="display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:8px;padding:12px;background:rgba(0,0,0,.25);border-radius:8px;border:1px solid rgba(255,185,0,.15);">
        <div style="text-align:center;">
          <div style="font-family:DM Mono,monospace;font-size:11px;color:rgba(255,205,60,.85);letter-spacing:1px;text-transform:uppercase;margin-bottom:5px;font-weight:700;">THETA DRAG</div>
          <div style="font-family:DM Mono,monospace;font-size:24px;font-weight:700;color:${{ntCol}};line-height:1;">${{ntSign}}\u20b9${{Math.abs(Math.round(nt)).toLocaleString('en-IN')}}</div>
        </div>
        <div style="font-size:20px;color:rgba(255,255,255,.2);text-align:center;">=</div>
        <div style="text-align:center;">
          <div style="font-family:DM Mono,monospace;font-size:11px;color:rgba(255,205,60,.85);letter-spacing:1px;text-transform:uppercase;margin-bottom:5px;font-weight:700;">FLAT P&amp;L TODAY</div>
          <div style="font-family:DM Mono,monospace;font-size:24px;font-weight:700;color:${{flatCol}};line-height:1;">${{flatPnl>=0?'+':''}}\u20b9${{Math.abs(flatPnl).toLocaleString('en-IN')}}</div>
        </div>
      </div>
      <div style="margin-top:9px;font-family:DM Mono,monospace;font-size:12px;color:rgba(255,200,70,.75);line-height:1.7;">
        ${{nt < 0 ? '🔴 <span style="color:#f04050;font-weight:700;">Theta negative</span> — you pay \u20b9' + Math.abs(Math.round(nt)).toLocaleString("en-IN") + '/day for holding. Need Nifty to move <span style="color:#ffcc00;font-weight:700;">' + Math.ceil(Math.abs(nt)/Math.abs(nd||1)) + ' pts</span> just to break even today.' : '🟢 <span style="color:#38d888;font-weight:700;">Theta positive</span> — you earn \u20b9' + Math.abs(Math.round(nt)).toLocaleString("en-IN") + '/day time decay. Premium selling strategy benefits from flat market.'}}
      </div>
    </div>
  </div>

  <!-- TAB 3: Live Slider -->
  <div id="${{simId}}_c3" style="display:none;">
    <div style="display:flex;align-items:center;justify-content:space-between;padding:10px 12px 8px;border-bottom:1px solid rgba(255,185,0,.12);">
      <div style="font-family:DM Mono,monospace;font-size:12px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;color:rgba(255,210,0,.95);">⟺ LIVE SCENARIO SLIDER</div>
    </div>
    <div style="padding:11px 12px;">
      <div style="display:flex;justify-content:space-between;margin-bottom:7px;font-family:DM Mono,monospace;font-size:12px;">
        <span style="color:#ffcc00;">\u20b9${{slMin.toLocaleString('en-IN')}}</span>
        <span id="${{simId}}_slv" style="font-weight:700;color:#ffcc00;background:rgba(255,185,0,.12);border:1px solid rgba(255,185,0,.35);border-radius:4px;padding:2px 10px;">Spot: \u20b9${{OC.spot.toLocaleString('en-IN')}}</span>
        <span style="color:#ffcc00;">\u20b9${{slMax.toLocaleString('en-IN')}}</span>
      </div>
      <input type="range" id="${{simId}}_sl" min="${{slMin}}" max="${{slMax}}" value="${{OC.spot}}" step="25"
        style="width:100%;height:5px;border-radius:3px;outline:none;border:none;-webkit-appearance:none;cursor:pointer;background:linear-gradient(90deg,#ffcc00 50%,rgba(255,185,0,.2) 50%);"
        onclick="event.stopPropagation()" onmousedown="event.stopPropagation()" ontouchstart="event.stopPropagation()" oninput="simSlide('${{simId}}', this.value, ${{slMin}}, ${{slMax}}, ${{nd}}, ${{nt}}, ${{nv}}, ${{maxL===null?'null':maxL}}, ${{maxP===null?'null':maxP}})">
    </div>
    <div id="${{simId}}_sr" style="padding:4px 12px 14px;text-align:center;">
      <div style="background:rgba(0,0,0,.3);border-radius:10px;padding:16px;border:1px solid rgba(255,185,0,.2);">
        <div style="font-family:DM Mono,monospace;font-size:11px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:rgba(255,205,60,.88);margin-bottom:7px;">ESTIMATED EXIT P&amp;L TODAY</div>
        <div id="${{simId}}_spnl" style="font-family:DM Mono,monospace;font-size:38px;font-weight:700;color:${{ntCol}};">${{flatPnl>=0?'+':''}}\u20b9${{Math.abs(flatPnl).toLocaleString('en-IN')}}</div>
        <div id="${{simId}}_snote" style="font-family:DM Mono,monospace;font-size:12px;color:rgba(255,200,70,.92);margin-top:5px;">Flat market — theta drag only</div>
        <div style="display:flex;gap:12px;justify-content:center;margin-top:11px;padding-top:11px;border-top:1px solid rgba(255,185,0,.12);">
          <div style="text-align:center;">
            <div style="font-family:DM Mono,monospace;font-size:11px;color:rgba(255,205,60,.88);letter-spacing:1px;text-transform:uppercase;margin-bottom:3px;font-weight:700;">Delta P&amp;L</div>
            <div id="${{simId}}_sdelta" style="font-family:DM Mono,monospace;font-size:17px;font-weight:700;color:#38d888;">\u20b90</div>
          </div>
          <div style="text-align:center;">
            <div style="font-family:DM Mono,monospace;font-size:11px;color:rgba(255,205,60,.88);letter-spacing:1px;text-transform:uppercase;margin-bottom:3px;font-weight:700;">Theta Cost</div>
            <div style="font-family:DM Mono,monospace;font-size:17px;font-weight:700;color:${{ntCol}};">${{ntSign}}\u20b9${{Math.abs(Math.round(nt)).toLocaleString('en-IN')}}</div>
          </div>
          <div style="text-align:center;">
            <div style="font-family:DM Mono,monospace;font-size:11px;color:rgba(255,205,60,.88);letter-spacing:1px;text-transform:uppercase;margin-bottom:3px;font-weight:700;">% of Max</div>
            <div id="${{simId}}_spct" style="font-family:DM Mono,monospace;font-size:17px;font-weight:700;color:#ffcc00;">—</div>
          </div>
        </div>
      </div>
    </div>
  </div>

</div>`;
}}

function simTab(simId, tab) {{
  ['t1','t2','t3'].forEach(t => {{
    const btn = document.getElementById(simId + '_' + t);
    const con = document.getElementById(simId + '_c' + t.slice(1));
    if (!btn || !con) return;
    const isActive = t === tab;
    con.style.display = isActive ? 'block' : 'none';
    btn.style.borderBottomColor = isActive ? '#ffcc00' : 'transparent';
    btn.style.color = isActive ? '#ffcc00' : 'rgba(255,185,0,.5)';
    btn.style.background = isActive ? 'rgba(255,185,0,.07)' : 'transparent';
  }});
}}

function simSlide(simId, val, slMin, slMax, nd, nt, nv, maxL, maxP) {{
  const spot = parseInt(val);
  const move = spot - OC.spot;
  const pct  = ((val - slMin) / (slMax - slMin) * 100);
  const sl   = document.getElementById(simId + '_sl');
  if (sl) sl.style.background = `linear-gradient(90deg,#ffcc00 ${{pct}}%,rgba(255,185,0,.2) ${{pct}}%)`;
  const slv = document.getElementById(simId + '_slv');
  if (slv) slv.textContent = 'Spot: \u20b9' + spot.toLocaleString('en-IN');
  // Asymmetric IV: down moves spike IV 3x harder than up moves compress it
  const ivEst = move < 0 ? -(move / OC.spot) * 600 : -(move / OC.spot) * 200;
  let pnl = nd * move + nv * ivEst + nt;
  if (maxL !== null) pnl = Math.max(-maxL, pnl);
  if (maxP !== null) pnl = Math.min(maxP * 0.9, pnl);
  pnl = Math.round(pnl);
  const col = pnl > 100 ? '#38d888' : pnl > 0 ? '#ffcc00' : pnl > -200 ? '#ffaa00' : '#f04050';
  const pEl = document.getElementById(simId + '_spnl');
  if (pEl) {{ pEl.textContent = (pnl>=0?'+':'') + '\u20b9' + Math.abs(pnl).toLocaleString('en-IN'); pEl.style.color = col; }}
  const nEl = document.getElementById(simId + '_snote');
  if (nEl) nEl.textContent = move > 0 ? `Nifty up ${{move}} pts` : move < 0 ? `Nifty down ${{Math.abs(move)}} pts` : 'Flat market — theta drag only';
  const deltaPnl = Math.round(nd * move);
  const dEl = document.getElementById(simId + '_sdelta');
  if (dEl) {{ dEl.textContent = (deltaPnl>=0?'+':'') + '\u20b9' + Math.abs(deltaPnl).toLocaleString('en-IN'); dEl.style.color = deltaPnl >= 0 ? '#38d888' : '#f04050'; }}
  const pctEl = document.getElementById(simId + '_spct');
  if (pctEl) {{ const pPct = maxP ? ((pnl / maxP) * 100).toFixed(1) + '%' : '—'; pctEl.textContent = pPct; pctEl.style.color = pnl >= 0 ? '#ffcc00' : '#f04050'; }}
}}

function popBadgeStyle(es) {{
  if(es>=70) return 'background:rgba(0,200,150,.25);color:#00c896;border-color:rgba(0,200,150,.5);font-weight:800;';
  if(es>=60) return 'background:rgba(77,232,184,.2);color:#4de8b8;border-color:rgba(77,232,184,.4);';
  if(es>=50) return 'background:rgba(100,128,255,.2);color:#8aa0ff;border-color:rgba(100,128,255,.4);';
  return 'background:rgba(255,107,107,.2);color:#ff6b6b;border-color:rgba(255,107,107,.4);';
}}

function initAllCards() {{
  let topScore=0, topName='', topCat='';
  const bullEx = smartPoP('bull_put_spread','bullish');
  const el_b = document.getElementById('legendBiasVal');
  if(el_b) {{ el_b.textContent=OC.bias+' ('+OC.biasConf+')'; el_b.style.color=OC.bias==='BULLISH'?'#00c896':OC.bias==='BEARISH'?'#ff6b6b':'#6480ff'; }}
  const srPts = bullEx.srAdj;
  const el_sr = document.getElementById('legendSRVal');
  if(el_sr) {{ const srLabel = srPts>5?'Near Support ✓':srPts<-5?'Near Resistance ✗':'Mid Range'; el_sr.textContent=srLabel+' ('+(srPts>=0?'+':'')+srPts+')'; el_sr.style.color=srPts>=0?'#00c896':'#ff6b6b'; }}
  const oiPts = bullEx.oiAdj;
  const el_oi = document.getElementById('legendOIVal');
  if(el_oi) {{ const oiLabel = OC.spot>OC.maxPeStrike?'Above PE Wall ✓':'Below PE Wall ✗'; el_oi.textContent=oiLabel+' ('+(oiPts>=0?'+':'')+oiPts+')'; el_oi.style.color=oiPts>=0?'#00c896':'#ff6b6b'; }}
  const el_pcr = document.getElementById('legendPCRVal');
  if(el_pcr) {{ const pcrLabel = OC.pcr>1.2?'Bullish PCR ':OC.pcr<0.8?'Bearish PCR ':'Neutral PCR '; el_pcr.textContent=pcrLabel+OC.pcr.toFixed(3); el_pcr.style.color=OC.pcr>1.2?'#00c896':OC.pcr<0.8?'#ff6b6b':'#6480ff'; }}
  document.querySelectorAll('.sc-card').forEach(card=>{{
    const shape=card.dataset.shape, cat=card.dataset.cat;
    const badge=document.getElementById('pop_'+card.id);
    try {{
      const result = smartPoP(shape, cat);
      const es = result.edgeScore;
      card.dataset.pop=es;
      card.dataset.scoreBreakdown=JSON.stringify(result);
      if(badge) {{
        const tp = TRUE_POP_MAP[shape];
        const tpStr = tp !== undefined ? ' / '+tp+'%' : '';
        badge.textContent = es+'%'+tpStr;
        badge.setAttribute('style', popBadgeStyle(es));
        badge.title = 'Edge Score: '+es+'%  |  True PoP (IV): '+(tp !== undefined ? tp+'%' : 'N/A');
      }}
      if(es>topScore) {{ topScore=es; topName=card.dataset.name; topCat=cat; }}
    }}catch(e){{card.dataset.pop=0;if(badge)badge.textContent='—%';}}
  }});
  const el_rec = document.getElementById('legendRecVal');
  if(el_rec && topName) {{
    const recCol = topCat==='bullish'?'#00c896':topCat==='bearish'?'#ff6b6b':'#6480ff';
    const tp = TRUE_POP_MAP[document.querySelector('.sc-card[data-name="'+topName+'"]')?.dataset?.shape];
    const tpStr = tp !== undefined ? ' · True PoP '+tp+'%' : '';
    el_rec.innerHTML=`<span style="color:${{recCol}};">${{topName}}</span> <span style="color:rgba(255,255,255,.75);font-size:13px;">${{topScore}}% Edge${{tpStr}}</span>`;
  }}
}}

function sortGridByPoP(cat) {{
  const grid=document.getElementById('sc-grid'); if(!grid)return;
  const cards=Array.from(grid.querySelectorAll(`.sc-card[data-cat="${{cat}}"]`));
  cards.sort((a,b)=>parseInt(b.dataset.pop||0)-parseInt(a.dataset.pop||0));
  cards.forEach(c=>grid.appendChild(c));
}}

window.addEventListener('load',function(){{
  initAllCards();
  ['bullish','bearish','nondirectional'].forEach(sortGridByPoP);
  filterStrat('bullish',document.querySelector('.sc-tab'));
}});

// ── Multi-Expiry Switcher ─────────────────────────────────────
const ALL_EXPIRY_DATA = {all_expiry_json};

window.switchExpiry = function(exp) {{
  let d = ALL_EXPIRY_DATA[exp];
  if (!d) {{
    const sel = document.getElementById('expiryDropdown');
    if (sel) {{
      const opts = Array.from(sel.options);
      const curIdx = opts.findIndex(o => o.value === exp);
      for (let i = curIdx + 1; i < opts.length; i++) {{
        const nextExp = opts[i].value;
        if (ALL_EXPIRY_DATA[nextExp]) {{
          sel.value = nextExp;
          exp = nextExp;
          d = ALL_EXPIRY_DATA[nextExp];
          break;
        }}
      }}
    }}
    if (!d) return;
  }}
  // Update OC object with selected expiry's data
  OC.spot        = d.spot;
  OC.atm         = d.atm;
  OC.expiry      = exp;
  OC.pcr         = d.pcr;
  OC.maxCeStrike = d.maxCeStrike;
  OC.maxPeStrike = d.maxPeStrike;
  OC.support     = d.support;
  OC.resistance  = d.resistance;
  OC.strongSup   = d.strongSup;
  OC.strongRes   = d.strongRes;
  OC.strikes     = d.strikes;
  // Rebuild strike map
  Object.keys(STRIKE_MAP).forEach(k => delete STRIKE_MAP[k]);
  OC.strikes.forEach(s => {{ STRIKE_MAP[s.strike] = s; }});
  // Collapse all expanded cards first
  document.querySelectorAll('.sc-card.expanded').forEach(c => c.classList.remove('expanded'));
  // Reset all metrics panels
  document.querySelectorAll('.sc-metrics-live').forEach(m => {{
    m.innerHTML = '<div class="sc-loading">&#9685; Calculating metrics...</div>';
  }});
  // Re-run PoP + sort
  initAllCards();
  ['bullish','bearish','nondirectional'].forEach(sortGridByPoP);
  // Flash indicator
  const sel = document.getElementById('expiryDropdown');
  if (sel) {{
    sel.style.borderColor = '#00c896';
    sel.style.boxShadow = '0 0 10px rgba(0,200,150,.3)';
    setTimeout(() => {{
      sel.style.borderColor = 'rgba(245,197,24,.45)';
      sel.style.boxShadow = 'none';
    }}, 800);
  }}
}};
</script>
"""


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
            f'<span class="tk-sub" style="color:rgba(255,255,255,.75);">Sig:&nbsp;{sig2:.2f}&nbsp;Hist:&nbsp;{diff:+.2f}</span>'
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
    track = "".join(items) * 2
    return f'''<div class="ticker-wrap">
  <div class="ticker-label">LIVE&nbsp;&#9654;</div>
  <div class="ticker-viewport"><div class="ticker-track" id="tkTrack">''' + track + '''</div></div>
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
  --text:rgba(255,255,255,.9);--muted:rgba(255,255,255,.75);--muted2:rgba(255,255,255,.55);
  --fh:'Sora',sans-serif;--fm:'DM Mono',monospace;
  --gold:#f5c518;--gold-dim:rgba(245,197,24,.45);--gold-bg:rgba(245,197,24,.10);
}
html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--text);font-family:var(--fh);font-size:18.8px;line-height:1.6;min-height:100vh;overflow-x:hidden;}
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
.logo-wrap{position:relative;height:42px;overflow:hidden;min-width:400px;}
.logo-slide{position:absolute;top:0;left:0;width:100%;font-family:var(--fh);font-size:29px;font-weight:700;
  background:linear-gradient(90deg,#00c896,#6480ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;
  filter:drop-shadow(0 0 12px rgba(0,200,150,.3));opacity:0;transform:translateY(20px);
  transition:opacity .5s ease, transform .5s ease;white-space:nowrap;}
.logo-slide.active{opacity:1;transform:translateY(0);}
.logo-slide.exit{opacity:0;transform:translateY(-20px);}
.hdr-meta{display:flex;align-items:center;gap:14px;font-size:15.9px;color:var(--muted);font-family:var(--fm)}
.live-dot{width:7px;height:7px;border-radius:50%;background:#00c896;box-shadow:0 0 10px #00c896;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.2}}
.refresh-countdown{display:flex;align-items:center;gap:8px;background:rgba(0,200,150,.07);
  border:1px solid rgba(0,200,150,.18);border-radius:20px;padding:4px 12px;font-family:var(--fm);font-size:15.9px;}
.countdown-arc-wrap{position:relative;width:18px;height:18px;flex-shrink:0;}
.countdown-arc-wrap svg{display:block;}
.countdown-num{font-family:var(--fm);font-size:17.4px;font-weight:700;color:#00c896;min-width:20px;text-align:center;transition:color .3s;}
.countdown-num.urgent{color:#ff6b6b;}
.countdown-num.halfway{color:#ffd166;}
.countdown-lbl{font-size:14.5px;color:rgba(255,255,255,.68);letter-spacing:.5px;}
.refresh-ring{display:none;width:14px;height:14px;border-radius:50%;border:2px solid rgba(0,200,150,.2);border-top-color:#00c896;animation:spin 0.8s linear infinite;}
.refresh-ring.active{display:inline-block;}
@keyframes spin{to{transform:rotate(360deg)}}
#refreshStatus{font-size:14.5px;color:rgba(255,255,255,.70);transition:color .3s;letter-spacing:.3px;}
#refreshStatus.updated{color:#00c896;font-weight:600;}
.hero{display:flex;align-items:stretch;background:linear-gradient(135deg,rgba(0,200,150,.055) 0%,rgba(100,128,255,.055) 100%);border-bottom:1px solid rgba(255,255,255,.07);overflow:hidden;position:relative;height:130px;}
.hero::before{content:'';position:absolute;top:-50px;left:-50px;width:200px;height:200px;border-radius:50%;background:radial-gradient(circle,rgba(0,200,150,.10),transparent 70%);pointer-events:none;}
.h-gauges{flex-shrink:0;display:flex;align-items:center;gap:10px;padding:0 16px 0 18px;}
.gauge-sep{width:1px;height:56px;background:rgba(255,255,255,.08);flex-shrink:0;}
.gauge-wrap{position:relative;width:100px;height:100px;}
.gauge-wrap svg{display:block;}
.gauge-inner{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;}
.g-val{font-family:'DM Mono',monospace;font-size:18.8px;font-weight:700;line-height:1;}
.g-lbl{font-size:10.9px;letter-spacing:1.5px;text-transform:uppercase;color:rgba(255,255,255,.65);margin-top:2px;}
.h-mid{flex:1;min-width:0;display:flex;flex-direction:column;justify-content:center;padding:0 15px 0 13px;border-left:1px solid rgba(255,255,255,.05);}
.h-eyebrow{font-size:11.6px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:rgba(255,255,255,.60);margin-bottom:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.h-signal{font-size:31.9px;font-weight:900;letter-spacing:1px;line-height:1.1;margin-bottom:2px;}
.h-sub{font-size:13.8px;color:rgba(255,255,255,.68);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.h-divider{height:1px;background:rgba(255,255,255,.05);margin:5px 0;}
.pill-row{display:flex;align-items:center;gap:8px;margin-bottom:4px;}
.pill-row:last-child{margin-bottom:0;}
.pill-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0;}
.pill-lbl{font-size:11.6px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:rgba(255,255,255,.70);width:96px;flex-shrink:0;}
.pill-track{width:120px;height:5px;background:rgba(255,255,255,.07);border-radius:3px;overflow:hidden;flex-shrink:0;}
.pill-fill{height:100%;border-radius:3px;}
.pill-num{font-family:'DM Mono',monospace;font-size:14.5px;font-weight:700;margin-left:8px;flex-shrink:0;}
.h-stats{flex-shrink:0;min-width:360px;display:flex;flex-direction:column;border-left:1px solid rgba(255,255,255,.07);background:rgba(255,255,255,.015);}
.h-stat-row{display:flex;align-items:stretch;flex:1;border-bottom:1px solid rgba(255,255,255,.05);}
.h-stat{flex:1;display:flex;flex-direction:column;justify-content:center;padding:5px 10px;text-align:center;border-right:1px solid rgba(255,255,255,.04);}
.h-stat:last-child{border-right:none;}
.h-stat-lbl{font-size:10.9px;font-weight:700;letter-spacing:1.8px;text-transform:uppercase;color:rgba(255,255,255,.60);margin-bottom:3px;white-space:nowrap;}
.h-stat-val{font-family:'DM Mono',monospace;font-size:18.8px;font-weight:700;line-height:1;white-space:nowrap;}
.h-stat-bottom{display:flex;align-items:center;justify-content:space-between;padding:4px 10px;}
.h-bias-row{display:flex;align-items:center;gap:6px;}
.h-chip{font-size:13px;font-weight:700;letter-spacing:.8px;text-transform:uppercase;padding:2px 9px;border-radius:20px;white-space:nowrap;}
.h-score{font-family:'DM Mono',monospace;font-size:11.6px;color:rgba(255,255,255,.60);letter-spacing:.5px;}
.h-ts{font-family:'DM Mono',monospace;font-size:11.6px;color:rgba(255,255,255,.18);letter-spacing:.5px;white-space:nowrap;}
.main{display:grid;grid-template-columns:268px 1fr;min-height:0}
.sidebar{background:rgba(8,11,20,.7);backdrop-filter:blur(12px);border-right:1px solid rgba(255,255,255,.06);position:sticky;top:57px;height:calc(100vh - 57px);overflow-y:auto;display:flex;flex-direction:column;}
.sidebar-sticky-top{position:sticky;top:0;z-index:50;background:rgba(8,11,20,.95);backdrop-filter:blur(16px);border-bottom:1px solid rgba(100,128,255,.15);padding-bottom:4px;}
.sidebar-scroll{flex:1;overflow-y:auto;}
.sidebar::-webkit-scrollbar{width:3px}
.sidebar::-webkit-scrollbar-thumb{background:rgba(255,255,255,.1);border-radius:2px}
.sidebar-scroll::-webkit-scrollbar{width:3px}
.sidebar-scroll::-webkit-scrollbar-thumb{background:rgba(255,255,255,.1);border-radius:2px}
.sb-sec{padding:16px 12px 8px}
.sb-lbl{font-size:13px;font-weight:700;letter-spacing:.15em;text-transform:uppercase;color:var(--aurora1);margin-bottom:8px;padding:0 0 0 8px;border-left:2px solid var(--aurora1)}
.sb-btn{display:flex;align-items:center;gap:8px;width:100%;padding:9px 12px;border-radius:8px;border:1px solid transparent;cursor:pointer;background:transparent;color:var(--muted);font-family:var(--fh);font-size:17.4px;text-align:left;transition:all .15s}
.sb-btn:hover{background:rgba(0,200,150,.08);color:rgba(255,255,255,.8);border-color:rgba(0,200,150,.2)}
.sb-btn.active{background:rgba(0,200,150,.1);border-color:rgba(0,200,150,.25);color:#00c896;font-weight:600}
.sb-badge{font-size:14.5px;margin-left:auto;font-weight:700}
.content{overflow-y:auto}
.section{padding:26px 28px;border-bottom:1px solid rgba(255,255,255,.05);background:transparent;position:relative}
.section:nth-child(odd){background:rgba(255,255,255,.015)}
.sec-title{font-family:var(--fh);font-size:15.9px;font-weight:700;letter-spacing:2.5px;color:var(--aurora1);text-transform:uppercase;display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:20px;padding-bottom:12px;border-bottom:1px solid rgba(0,200,150,.15)}
.sec-sub{font-size:15.9px;color:var(--muted2);font-weight:400;letter-spacing:.5px;text-transform:none;margin-left:auto}
.oi-ticker-table{border:1px solid rgba(255,255,255,.07);border-radius:14px;overflow:hidden}
.oi-ticker-hdr{display:grid;grid-template-columns:130px repeat(5,1fr);padding:9px 18px;align-items:center;gap:6px}
.oi-ticker-hdr-label{font-size:13px;font-weight:700;letter-spacing:2px;text-transform:uppercase}
.oi-ticker-hdr-cell{font-size:13px;letter-spacing:1.5px;text-transform:uppercase;color:rgba(255,255,255,.70);text-align:center}
.oi-ticker-row{display:grid;grid-template-columns:130px repeat(5,1fr);padding:15px 18px;border-top:1px solid rgba(255,255,255,.04);align-items:center;gap:6px;transition:background .15s}
.oi-ticker-row:hover{background:rgba(255,255,255,.03)}
.oi-ticker-metric{font-size:14.5px;font-weight:600;letter-spacing:1px;text-transform:uppercase;color:rgba(255,255,255,.70)}
.oi-ticker-cell{text-align:center}
.kl-zone-labels{display:flex;justify-content:space-between;margin-bottom:6px;font-size:15.9px;font-weight:700}
.kl-node{position:absolute;text-align:center}
.kl-lbl{font-size:14.5px;font-weight:600;text-transform:uppercase;letter-spacing:.5px;line-height:1.3;white-space:nowrap}
.kl-val{font-size:17.4px;font-weight:700;color:rgba(255,255,255,.7);white-space:nowrap;margin-top:2px}
.kl-dot{width:11px;height:11px;border-radius:50%;border:2px solid var(--bg)}
.kl-gradient-bar{position:relative;height:6px;border-radius:3px;background:linear-gradient(90deg,#00a07a 0%,#00c896 25%,#6480ff 55%,#ff6b6b 80%,#cc4040 100%);box-shadow:0 0 12px rgba(0,200,150,.2)}
.kl-price-tick{position:absolute;top:50%;transform:translate(-50%,-50%);width:3px;height:18px;background:#fff;border-radius:2px;box-shadow:0 0 12px rgba(255,255,255,.6);z-index:10}
.kl-dist-row{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:4px}
.kl-dist-box{background:rgba(255,255,255,.03);border:1px solid;border-radius:10px;padding:10px 14px;display:flex;justify-content:space-between;align-items:center}
.strikes-head{font-weight:700;margin-bottom:10px;font-size:18.8px}
.strikes-wrap{display:grid;grid-template-columns:1fr 1fr;gap:20px}
.s-table{width:100%;border-collapse:collapse;border-radius:10px;overflow:hidden}
.s-table th{background:linear-gradient(90deg,rgba(0,200,150,.15),rgba(100,128,255,.15));color:rgba(255,255,255,.7);padding:10px 12px;font-size:15.9px;font-weight:600;text-align:left;letter-spacing:.5px;border-bottom:1px solid rgba(255,255,255,.08)}
.s-table td{padding:10px 12px;border-bottom:1px solid rgba(255,255,255,.05);font-size:17.4px;color:rgba(255,255,255,.8);background:rgba(255,255,255,.02)}
.s-table tr:last-child td{border-bottom:none}
.s-table tr:hover td{background:rgba(0,200,150,.05)}
.ticker-wrap{display:flex;align-items:center;background:rgba(4,6,12,.97);border-bottom:1px solid rgba(255,255,255,.07);height:46px;overflow:hidden;position:relative;z-index:190;box-shadow:0 2px 20px rgba(0,0,0,.5);}
.ticker-label{flex-shrink:0;padding:0 16px;font-family:var(--fm);font-size:13px;font-weight:700;letter-spacing:3px;color:#00c896;text-transform:uppercase;border-right:1px solid rgba(0,200,150,.2);height:100%;display:flex;align-items:center;background:rgba(0,200,150,.07);white-space:nowrap;}
.ticker-viewport{flex:1;overflow:hidden;height:100%}
.ticker-track{display:flex;align-items:center;height:100%;white-space:nowrap;animation:ticker-scroll 38s linear infinite;will-change:transform;}
.ticker-track:hover{animation-play-state:paused}
@keyframes ticker-scroll{0%{transform:translateX(0)}100%{transform:translateX(-50%)}}
.tk-item{display:inline-flex;align-items:center;gap:10px;padding:0 20px;height:100%;border-right:1px solid rgba(255,255,255,.04);flex-shrink:0;}
.tk-name{font-family:var(--fm);font-size:14.5px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;padding:3px 10px;border-radius:6px;white-space:nowrap;flex-shrink:0;background:rgba(255,255,255,.08);color:rgba(255,255,255,.5);border:1px solid rgba(255,255,255,.1);}
.tk-val{font-family:var(--fm);font-size:26.1px;font-weight:700;line-height:1;white-space:nowrap;}
.tk-sub{font-family:var(--fm);font-size:14.5px;color:rgba(255,255,255,.70);white-space:nowrap;}
.tk-badge{font-family:var(--fh);font-size:14.5px;font-weight:700;padding:3px 10px;border-radius:20px;white-space:nowrap;letter-spacing:.3px;}

/* ── Main Tab Bar ─────────────────────────────────── */
.main-tabs{display:flex;gap:8px;padding:16px 28px 0;border-bottom:1px solid rgba(255,255,255,.07);background:rgba(4,6,12,.6);position:sticky;top:0;z-index:100;}
.main-tab{padding:10px 22px;font-family:var(--fh);font-size:13px;font-weight:700;letter-spacing:1px;text-transform:uppercase;border:none;border-bottom:3px solid transparent;background:transparent;color:rgba(255,255,255,.55);cursor:pointer;transition:all .2s;border-radius:6px 6px 0 0;}
.main-tab:hover{color:rgba(255,255,255,.85);background:rgba(255,255,255,.04);}
.main-tab.active{color:#00c896;border-bottom:3px solid #00c896;background:rgba(0,200,150,.07);}
footer{padding:16px 32px;border-top:1px solid rgba(255,255,255,.06);background:rgba(6,8,15,.9);backdrop-filter:blur(12px);display:flex;justify-content:space-between;font-size:15.9px;color:var(--muted2);font-family:var(--fm)}
.sc-tabs{display:flex;gap:10px;margin-bottom:20px;flex-wrap:wrap}
.sc-tab{padding:8px 20px;border-radius:24px;border:1px solid;cursor:pointer;font-family:var(--fh);font-size:17.4px;font-weight:600;transition:all .2s;display:flex;align-items:center;gap:8px;background:transparent}
.sc-tab:hover{opacity:.85}
.sc-cnt{font-size:14.5px;padding:1px 7px;border-radius:10px;color:#fff;font-weight:700}
.sc-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px}
.sc-card{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08);border-radius:14px;overflow:hidden;cursor:pointer;transition:all .2s;display:flex;flex-direction:column;position:relative;}
.sc-card:hover{border-color:rgba(0,200,150,.3);transform:translateY(-3px);box-shadow:0 8px 28px rgba(0,200,150,.1)}
.sc-card.hidden{display:none}
.sc-card.expanded{grid-column:1 / -1 !important;flex-direction:row !important;align-items:stretch;border-color:rgba(0,200,150,.35);box-shadow:0 0 0 1px rgba(0,200,150,.2),0 12px 32px rgba(0,200,150,.12);overflow:visible;}
.sc-card.expanded .sc-detail{display:block;flex:0 0 400px;width:400px;border-top:none;border-left:1px solid rgba(0,229,160,.15);overflow:visible;background:rgba(5,13,26,.8);}
.sc-card.expanded .sc-summary{flex:1;min-width:180px;}
.sc-card.expanded:hover{transform:none;}
.sc-payoff{display:none;}
.sc-card.expanded .sc-payoff{display:flex;flex:1;min-width:0;flex-direction:column;border-left:1px solid rgba(0,229,160,.15);background:rgba(3,10,22,.9);padding:0;}
.sc-payoff-inner{width:100%;height:100%;display:flex;flex-direction:column;}
.sc-pop-badge{position:absolute;top:8px;right:8px;font-family:'DM Mono',monospace;font-size:14.5px;font-weight:700;padding:3px 8px;border-radius:20px;border:1px solid rgba(255,255,255,.15);background:rgba(255,255,255,.08);color:rgba(255,255,255,.5);z-index:5;letter-spacing:.5px;transition:all .3s;min-width:38px;text-align:center;}
.sc-svg{display:flex;align-items:center;justify-content:center;padding:14px 0 6px;background:rgba(255,255,255,.02)}
.sc-body{padding:10px 12px 12px}
.sc-name{font-family:var(--fh);font-size:17.4px;font-weight:700;color:rgba(255,255,255,.9);margin-bottom:4px;line-height:1.3;padding-right:48px}
.sc-legs{font-family:var(--fm);font-size:13px;color:rgba(0,200,220,.7);margin-bottom:8px;letter-spacing:.3px;line-height:1.4}
.sc-tags{display:flex;flex-direction:column;gap:4px}
.sc-tag{font-size:13px;padding:2px 8px;border-radius:6px;border:1px solid;background:rgba(0,0,0,.2);display:inline-block;width:fit-content}
.sc-detail{display:none;border-top:1px solid rgba(255,255,255,.06);background:rgba(0,200,150,.03)}
.sc-desc{font-size:15.9px;color:rgba(255,255,255,.5);line-height:1.7;padding:12px 12px 8px;border-bottom:1px solid rgba(255,255,255,.05);}
.sc-metrics-live{padding:0}
.sc-loading{padding:14px 12px;font-size:15.9px;color:rgba(255,255,255,.68);text-align:center;font-family:'DM Mono',monospace}
.metric-row{display:flex;justify-content:space-between;align-items:center;padding:8px 12px;border-bottom:1px solid rgba(255,255,255,.04);transition:background .15s;}
.metric-row:hover{background:rgba(255,255,255,.03)}
.metric-strike{background:rgba(255,209,102,.04);border-bottom:1px solid rgba(255,209,102,.12) !important;}
.metric-lbl{font-size:14.5px;color:rgba(255,255,255,.70);letter-spacing:.5px;text-transform:uppercase;font-family:'DM Mono',monospace;}
.metric-val{font-family:'DM Mono',monospace;font-size:17.4px;font-weight:600;text-align:right;}
.greeks-panel{margin:10px 10px 6px;padding:14px 12px;background:linear-gradient(135deg,rgba(100,128,255,.12),rgba(0,200,220,.10));border-radius:14px;border:1px solid rgba(100,128,255,.28);box-shadow:0 4px 20px rgba(100,128,255,.1),inset 0 1px 0 rgba(255,255,255,.06);}
.greeks-title{font-size:13px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:rgba(138,160,255,1.0);margin-bottom:10px;padding-bottom:8px;border-bottom:1px solid rgba(100,128,255,.25);display:flex;align-items:center;justify-content:space-between;}
.greeks-expiry-tag{font-size:12.3px;color:rgba(255,255,255,.5);font-weight:400;letter-spacing:.5px;text-transform:none;}
.greeks-strike-wrap{position:relative;margin-bottom:10px;}
.greeks-strike-wrap::after{content:'▼';position:absolute;right:10px;top:50%;transform:translateY(-50%);font-size:11.6px;color:var(--gold);pointer-events:none;z-index:2;}
.greeks-strike-select{width:100%;appearance:none;-webkit-appearance:none;background:linear-gradient(135deg,rgba(245,197,24,.12),rgba(200,155,10,.06));border:1px solid var(--gold-dim);border-radius:8px;color:var(--gold);font-family:'DM Mono',monospace;font-size:15.9px;font-weight:700;padding:7px 28px 7px 10px;cursor:pointer;outline:none;letter-spacing:.5px;transition:border-color .2s,background .2s,box-shadow .2s;}
.greeks-strike-select:hover{border-color:rgba(245,197,24,.75);background:linear-gradient(135deg,rgba(245,197,24,.18),rgba(200,155,10,.10));box-shadow:0 0 10px rgba(245,197,24,.18);}
.greeks-strike-select:focus{border-color:var(--gold);box-shadow:0 0 0 2px rgba(245,197,24,.25);}
.greeks-strike-select option{background:#0e1225;color:var(--gold);font-weight:700;}
.greek-name{font-family:'DM Mono',monospace;font-size:15.9px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:rgba(255,255,255,.92);}
.greek-sub{font-size:11.6px;color:rgba(255,255,255,.55);margin-top:1px;}
.greeks-row{display:flex;align-items:center;justify-content:space-between;padding:7px 0;border-bottom:1px solid rgba(255,255,255,.06);}
.greeks-row:last-child{border-bottom:none;}
.greeks-atm-badge{display:flex;align-items:center;justify-content:center;gap:6px;background:rgba(100,128,255,.1);border:1px solid rgba(100,128,255,.25);border-radius:8px;padding:5px 8px;margin-bottom:10px;font-family:'DM Mono',monospace;font-size:15.9px;flex-wrap:wrap;}
.greeks-atm-strike{font-weight:700;color:#8aa0ff;}
.iv-bar-wrap{display:flex;align-items:center;gap:6px;margin-top:8px;padding-top:8px;border-top:1px solid rgba(255,255,255,.06);}
.iv-bar-label{font-size:11.6px;color:rgba(255,255,255,.7);letter-spacing:1px;text-transform:uppercase;font-weight:600;width:42px;flex-shrink:0;}
.iv-bar-track{flex:1;height:4px;background:rgba(255,255,255,.08);border-radius:2px;overflow:hidden;}
.iv-bar-fill{height:100%;border-radius:2px;transition:width .6s ease;}
.iv-bar-num{font-family:'DM Mono',monospace;font-size:15.9px;font-weight:700;min-width:38px;text-align:right;}
.greeks-table-section{padding:22px 28px;border-bottom:1px solid rgba(255,255,255,.05);}
.greeks-table-wrap{display:grid;grid-template-columns:1fr 1fr;gap:16px;}
.greeks-tbl{border:1px solid rgba(255,255,255,.07);border-radius:12px;overflow:hidden;}
.greeks-tbl-head{display:grid;grid-template-columns:90px repeat(4,1fr);background:rgba(255,255,255,.04);padding:8px 14px;border-bottom:1px solid rgba(255,255,255,.06);gap:4px;}
.greeks-tbl-head-label{font-size:12.3px;letter-spacing:1.5px;text-transform:uppercase;color:rgba(255,255,255,.68);text-align:center;}
.greeks-tbl-row{display:grid;grid-template-columns:90px repeat(4,1fr);padding:9px 14px;border-bottom:1px solid rgba(255,255,255,.04);align-items:center;gap:4px;transition:background .15s;}
.greeks-tbl-row:last-child{border-bottom:none;}
.greeks-tbl-row:hover{background:rgba(255,255,255,.03);}
.greeks-tbl-strike{font-family:'DM Mono',monospace;font-size:17.4px;font-weight:700;color:rgba(255,255,255,.8);}
.greeks-tbl-cell{font-family:'DM Mono',monospace;font-size:15.9px;font-weight:600;text-align:center;color:rgba(255,255,255,.65);}
/* Hidden refresh iframe — zero footprint */
#silentRefreshFrame{position:fixed;width:0;height:0;border:none;visibility:hidden;pointer-events:none;opacity:0;}
@media(max-width:1024px){
  .main{grid-template-columns:1fr}
  .sidebar{position:static;height:auto;border-right:none;border-bottom:1px solid rgba(255,255,255,.06)}
  .hero{height:auto;flex-wrap:wrap;}
  .h-gauges{padding:12px 18px;}
  .h-stats{min-width:100%;border-left:none;border-top:1px solid rgba(255,255,255,.07);}
  .strikes-wrap{grid-template-columns:1fr}
  .greeks-table-wrap{grid-template-columns:1fr}
  .sc-grid{grid-template-columns:repeat(auto-fill,minmax(160px,1fr))}
  .main-tabs{padding:12px 16px 0;gap:6px;flex-wrap:wrap;}
  .main-tab{font-size:11px;padding:8px 14px;}
  .hdr-meta{flex-wrap:wrap;gap:8px;}
  .logo-wrap{min-width:0;max-width:280px;}
  .h-eyebrow{white-space:normal;overflow:visible;}
  .h-sub{white-space:normal;overflow:visible;}
  .sc-card.expanded{flex-direction:column !important;}
  .sc-card.expanded .sc-detail{flex:none !important;width:100% !important;border-left:none;border-top:1px solid rgba(0,229,160,.15);}
}
@media(max-width:768px){
  body{font-size:16px;}
  .h-stat-row{flex-wrap:wrap;gap:8px;}
  .h-stat{min-width:calc(50% - 8px);}
  .main-tabs{overflow-x:auto;flex-wrap:nowrap;}
  .main-tab{white-space:nowrap;}
  header{flex-wrap:wrap;gap:8px;padding:10px 14px;}
  .logo-wrap{min-width:0;max-width:200px;height:36px;}
  .logo-slide{font-size:22px;}
  .hdr-meta{width:100%;font-size:13px;}
  .oi-ticker-table{overflow-x:auto;-webkit-overflow-scrolling:touch;}
  .oi-ticker-hdr,.oi-ticker-row{min-width:560px;}
  .greeks-tbl{overflow-x:auto;-webkit-overflow-scrolling:touch;}
  .h-mid{padding:8px 12px;}
  .h-signal{font-size:24px;}
  .h-eyebrow{font-size:10px;}
  .h-sub{font-size:12px;}
  .pill-track{width:80px;}
  .sec-sub{margin-left:0;width:100%;}
  .greeks-table-section{padding:16px 14px;}
  .sc-tabs{flex-direction:column;gap:8px;}
  .sc-tabs>div[style]{margin-left:0!important;width:100%;}
  #expiryDropdown{width:100%;font-size:14px;}
}
@media(max-width:640px){
  header{padding:10px 12px}
  .section{padding:14px 12px}
  .kl-dist-row{grid-template-columns:1fr}
  footer{flex-direction:column;gap:6px}
  .logo-wrap{min-width:0;max-width:180px;height:32px;}
  .logo-slide{font-size:20px;}
  .refresh-countdown{display:none;}
  .sidebar{display:none;}
  .sc-grid{grid-template-columns:repeat(auto-fill,minmax(140px,1fr))}
  .main-tab{font-size:10px;padding:6px 10px;}
  .h-gauges{padding:10px 12px;gap:8px;}
  .gauge-wrap{width:76px;height:76px;}
  .hdr-meta{font-size:11px;gap:6px;}
  .greeks-table-section{padding:14px 12px;}
  .greeks-tbl-head,.greeks-tbl-row{grid-template-columns:70px repeat(4,1fr);padding:7px 8px;gap:2px;}
  .greeks-tbl-head-label,.greeks-tbl-cell{font-size:11px;}
  .greeks-tbl-strike{font-size:13px;}
  .sec-title{font-size:13px;letter-spacing:1.5px;margin-bottom:14px;}
  .sec-sub{font-size:11px;margin-left:0;width:100%;}
  .h-stat-val{font-size:15px;}
  .h-stat-lbl{font-size:9px;letter-spacing:1px;}
  .sc-tabs{gap:6px;margin-bottom:14px;flex-direction:column;}
  .sc-tabs>div[style]{margin-left:0!important;width:100%!important;}
  #expiryDropdown{width:100%!important;font-size:13px;padding:6px 10px;}
  .sc-tab{padding:6px 14px;font-size:15px;}
  .sc-name{font-size:15px;}
  .sb-btn{font-size:15px;padding:7px 10px;}
  .strikes-head{font-size:16px;}
  .s-table th,.s-table td{padding:8px 10px;font-size:14px;}
  .oi-ticker-hdr,.oi-ticker-row{min-width:520px;}
  .metric-lbl{font-size:13px;}
  .metric-val{font-size:15px;}
  .h-mid{width:100%;border-left:none;border-top:1px solid rgba(255,255,255,.05);}
}
@media(max-width:480px){
  .sc-grid{grid-template-columns:1fr 1fr;}
  .gauge-wrap{width:62px;height:62px;}
  .g-val{font-size:13px;}
  .g-lbl{font-size:9px;letter-spacing:1px;}
  .gauge-sep{display:none;}
  .h-signal{font-size:19px;letter-spacing:.5px;}
  .h-gauges{gap:4px;padding:8px 10px;}
  .h-mid{padding:6px 10px;}
  .h-sub{font-size:11px;}
  .greeks-tbl-head,.greeks-tbl-row{grid-template-columns:56px repeat(4,1fr);padding:6px 6px;gap:2px;}
  .greeks-tbl-head-label,.greeks-tbl-cell{font-size:10px;}
  .greeks-tbl-strike{font-size:11px;}
  .section{padding:12px 10px;}
  .sec-title{font-size:12px;letter-spacing:1px;}
  .h-stat{min-width:calc(50% - 4px);}
  .h-stat-val{font-size:14px;}
  .h-stat-lbl{font-size:8px;letter-spacing:.8px;}
  .pill-lbl{width:68px;font-size:9px;letter-spacing:1px;}
  .pill-num{font-size:12px;margin-left:4px;}
  .pill-track{width:64px;}
  .pill-dot{width:6px;height:6px;}
  .sb-btn{font-size:14px;padding:6px 10px;}
  .sb-lbl{font-size:11px;}
  .sb-badge{font-size:12px;}
  .main-tabs{padding:8px 10px 0;gap:4px;}
  .main-tab{font-size:9px;padding:5px 8px;}
  .strikes-wrap{gap:10px;}
  .s-table th,.s-table td{padding:7px 8px;font-size:13px;}
  footer{padding:12px 10px;font-size:13px;}
  .sc-name{font-size:14px;padding-right:36px;}
  .sc-legs{font-size:11px;}
  .sc-tag{font-size:11px;}
  .sc-pop-badge{font-size:12px;padding:2px 6px;}
  .kl-lbl{font-size:12px;}
  .kl-val{font-size:14px;}
  .kl-dist-box{padding:8px 10px;font-size:14px;}
  .greeks-panel{margin:6px 6px 4px;padding:10px 10px;}
  .greeks-row{padding:5px 0;}
  .greek-name{font-size:13px;}
  .greek-sub{font-size:10px;}
  .greeks-atm-badge{font-size:13px;padding:4px 6px;}
  .iv-bar-label{font-size:10px;width:34px;}
  .iv-bar-num{font-size:13px;min-width:32px;}
  .sc-desc{font-size:13px;padding:10px 10px 6px;}
  .metric-lbl{font-size:12px;}
  .metric-val{font-size:14px;}
  .oi-ticker-hdr,.oi-ticker-row{min-width:480px;}
  .sc-card.expanded .sc-detail{width:100% !important;}
  .logo-wrap{min-width:0;max-width:160px;height:30px;}
  .logo-slide{font-size:18px;}
  #smartPopLegend{flex-direction:column;}
  .hdr-meta>span:not(:first-child):not(:last-child){display:none;}
}
"""

# =================================================================
#  SECTION 9 -- ANIMATED JS  (v18.3 — SILENT BACKGROUND REFRESH)
# =================================================================

ANIMATED_JS = """
<script>
// ── Logo rotator ────────────────────────────────────────────────
(function() {
  const NAMES = ['NIFTYCRAFT','Nifty Option Strategy Builder','OI Signal Dashboard','Options Analytics Hub','PCR & Max Pain Tracker'];
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

// ── Silent Background Refresh Engine (v18.3) ───────────────────
(function() {
  const TOTAL_SECS = 30;
  const R = 7, C = 2 * Math.PI * R;

  function setCountdownUI(secs) {
    const numEl = document.getElementById('cdNum');
    const arcEl = document.getElementById('cdArc');
    if (numEl) {
      numEl.textContent = secs;
      numEl.className = 'countdown-num' +
        (secs <= 5 ? ' urgent' : secs <= 15 ? ' halfway' : '');
    }
    if (arcEl) {
      arcEl.style.strokeDashoffset = (C * (1 - secs / TOTAL_SECS)).toFixed(2);
      arcEl.style.stroke = secs <= 5 ? '#ff6b6b' : secs <= 15 ? '#ffd166' : '#00c896';
    }
  }

  function showSpinner(on) {
    const ring = document.getElementById('refreshRing');
    if (ring) ring.classList.toggle('active', on);
  }

  function flashUpdated() {
    const txt = document.getElementById('refreshStatus');
    if (!txt) return;
    txt.textContent = 'Updated \u2713';
    txt.classList.add('updated');
    setTimeout(() => { txt.textContent = ''; txt.classList.remove('updated'); }, 2500);
  }

  const PATCH_IDS = [
    'heroWidget','oi','kl','strikes','greeksTable','greeksPanel','tkTrack','lastUpdatedTs','mainPanelStrat'
  ];

  function microDiff(newDoc) {
    let changed = false;
    PATCH_IDS.forEach(function(id) {
      const curEl = document.getElementById(id);
      const newEl = newDoc.getElementById(id);
      if (!curEl || !newEl) return;
      if (curEl.innerHTML !== newEl.innerHTML) {
        const content = document.querySelector('.content');
        const scrollTop = content ? content.scrollTop : 0;
        curEl.innerHTML = newEl.innerHTML;
        if (content) content.scrollTop = scrollTop;
        changed = true;
      }
    });
    return changed;
  }

  let iframe = document.getElementById('silentRefreshFrame');
  if (!iframe) {
    iframe = document.createElement('iframe');
    iframe.id = 'silentRefreshFrame';
    iframe.style.cssText = 'position:fixed;width:0;height:0;border:none;' +
                            'visibility:hidden;pointer-events:none;opacity:0;' +
                            'top:-9999px;left:-9999px;';
    document.body.appendChild(iframe);
  }

  let _lastTimestamp  = null;
  let _refreshing     = false;

  function doSilentRefresh() {
    if (_refreshing) return;
    _refreshing = true;
    showSpinner(true);
    iframe.src = 'index.html?_=' + Date.now();
  }

  iframe.addEventListener('load', function() {
    if (!iframe.src || iframe.src === 'about:blank') {
      _refreshing = false; showSpinner(false); return;
    }
    try {
      const newDoc = iframe.contentDocument || iframe.contentWindow.document;
      if (!newDoc || !newDoc.body) throw new Error('empty doc');
      const newTsEl = newDoc.getElementById('lastUpdatedTs');
      const newTs   = newTsEl ? newTsEl.textContent.trim() : '';
      if (_lastTimestamp !== null && newTs === _lastTimestamp) {
        showSpinner(false); _refreshing = false; return;
      }
      _lastTimestamp = newTs;
      const changed = microDiff(newDoc);
      showSpinner(false); _refreshing = false;
      if (changed) {
        flashUpdated();
        setTimeout(function() {
          try {
            if (typeof initAllCards === 'function') {
              initAllCards();
              ['bullish','bearish','nondirectional'].forEach(function(c) {
                if (typeof sortGridByPoP === 'function') sortGridByPoP(c);
              });
            }
            if (typeof greeksUpdateStrike === 'function') {
              var sel = document.getElementById('greeksStrikeSelect');
              if (sel) greeksUpdateStrike(sel.value);
            }
          } catch(e) {}
        }, 60);
      }
    } catch(e) {
      showSpinner(false); _refreshing = false;
    }
    setTimeout(function() { try { iframe.src = 'about:blank'; } catch(e) {} }, 500);
  });

  let remaining = TOTAL_SECS;
  setCountdownUI(remaining);

  setInterval(function() {
    remaining -= 1;
    if (remaining <= 0) {
      remaining = TOTAL_SECS;
      setCountdownUI(remaining);
      doSilentRefresh();
    } else {
      setCountdownUI(remaining);
    }
  }, 1000);

  window.addEventListener('load', function() {
    setTimeout(doSilentRefresh, 2000);
  });
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
(function() {{
  var _gData = {strikes_json};
  var _atm   = {atm};
  function _initGreeks() {{
    var sel = document.getElementById('greeksStrikeSelect');
    if (sel) {{ greeksUpdateStrike(sel.value); }}
  }}
  window.greeksUpdateStrike = function(strike) {{
    var key = String(parseInt(strike, 10));
    var d   = _gData[key];
    if (!d) {{
      var keys = Object.keys(_gData).map(Number);
      var nearest = keys.reduce((a,b) => Math.abs(b-parseInt(strike))<Math.abs(a-parseInt(strike))?b:a, keys[0]);
      d = _gData[String(nearest)];
    }}
    if (!d) return;
    var sel  = parseInt(strike, 10);
    var dist = Math.round(Math.abs(sel - _atm) / 50);
    var lbl  = sel === _atm ? 'ATM' : (sel > _atm ? 'CE+' + dist : 'PE-' + dist);
    var e1 = document.getElementById('greeksStrikeTypeLabel'); if(e1) e1.textContent = lbl;
    var e2 = document.getElementById('greeksStrikeLabel'); if(e2) e2.innerHTML = '&#8377;' + sel.toLocaleString('en-IN');
    var e3 = document.getElementById('greeksCeLtp'); if(e3) e3.innerHTML = 'CE &#8377;' + (d.ce_ltp||0).toFixed(1);
    var e4 = document.getElementById('greeksPeLtp'); if(e4) e4.innerHTML = 'PE &#8377;' + (d.pe_ltp||0).toFixed(1);
    var ceCol='#00c896', peCol='#ff6b6b';
    var cePct=Math.min(100,Math.abs(d.ce_delta)*100).toFixed(0);
    var pePct=Math.min(100,Math.abs(d.pe_delta)*100).toFixed(0);
    var dw = document.getElementById('greeksDeltaWrap');
    if(dw) dw.innerHTML =
      '<div style="display:flex;align-items:center;gap:5px;">' +
        '<div style="width:34px;height:3px;background:rgba(255,255,255,.10);border-radius:2px;overflow:hidden;">' +
          '<div style="width:'+cePct+'%;height:100%;background:'+ceCol+';border-radius:2px;"></div></div>' +
        '<span style="font-family:DM Mono,monospace;font-size:15.9px;font-weight:700;color:'+ceCol+';">' +
             (d.ce_delta>=0?'+':'')+d.ce_delta.toFixed(3)+'</span></div>' +
      '<div style="display:flex;align-items:center;gap:5px;margin-top:3px;">' +
        '<div style="width:34px;height:3px;background:rgba(255,255,255,.10);border-radius:2px;overflow:hidden;">' +
          '<div style="width:'+pePct+'%;height:100%;background:'+peCol+';border-radius:2px;"></div></div>' +
        '<span style="font-family:DM Mono,monospace;font-size:15.9px;font-weight:700;color:'+peCol+';">' +
             (d.pe_delta>=0?'+':'')+d.pe_delta.toFixed(3)+'</span></div>';
    var ice = document.getElementById('greeksIvCe'); if(ice) ice.textContent = (d.ce_iv||0).toFixed(1)+'%';
    var ipe = document.getElementById('greeksIvPe'); if(ipe) ipe.textContent = (d.pe_iv||0).toFixed(1)+'%';
    var skew=((d.pe_iv||0)-(d.ce_iv||0)).toFixed(1);
    var skewEl=document.getElementById('greeksSkewLbl');
    if(skewEl) {{ skewEl.textContent = parseFloat(skew)>0?'PE Skew +'+skew:'CE Skew '+skew; skewEl.style.color = parseFloat(skew)>1.5?'#ff6b6b':(parseFloat(skew)<-1.5?'#00c896':'#6480ff'); }}
    function tfmt(t){{ return Math.abs(t)>=0.01?'&#8377;'+Math.abs(t).toFixed(2):t.toFixed(4); }}
    var tc = document.getElementById('greeksThetaCe'); if(tc) tc.innerHTML = tfmt(d.ce_theta||0);
    var tp = document.getElementById('greeksThetaPe'); if(tp) tp.innerHTML = tfmt(d.pe_theta||0);
    function vfmt(v){{ return Math.abs(v)>=0.0001?v.toFixed(4):'&mdash;'; }}
    var vc = document.getElementById('greeksVegaCe'); if(vc) vc.innerHTML = vfmt(d.ce_vega||0);
    var vp = document.getElementById('greeksVegaPe'); if(vp) vp.innerHTML = vfmt(d.pe_vega||0);
    var ivAvg=((d.ce_iv||0)+(d.pe_iv||0))/2;
    var ivCol=ivAvg>25?'#ff6b6b':(ivAvg>18?'#ffd166':'#00c896');
    var ivReg=ivAvg>25?'High IV \u00b7 Buy Premium':(ivAvg>15?'Normal IV \u00b7 Balanced':'Low IV \u00b7 Sell Premium');
    var ivPct=Math.min(100,Math.max(0,(ivAvg/60)*100)).toFixed(1);
    var barEl=document.getElementById('greeksIvBar');
    if(barEl) {{barEl.style.width=ivPct+'%'; barEl.style.background=ivCol; barEl.style.boxShadow='0 0 6px '+ivCol+'88';}}
    var avgEl=document.getElementById('greeksIvAvg');
    if(avgEl) {{avgEl.textContent=ivAvg.toFixed(1)+'%'; avgEl.style.color=ivCol;}}
    var regEl=document.getElementById('greeksIvRegime');
    if(regEl) {{regEl.textContent=ivReg; regEl.style.color=ivCol;}}
  }};
  if (document.readyState === 'loading') {{
    document.addEventListener('DOMContentLoaded', _initGreeks);
  }} else {{
    setTimeout(_initGreeks, 80);
  }}
}})();
</script>"""


# =================================================================
#  SECTION 10 -- HTML ASSEMBLER
# =================================================================

def generate_html(tech, oc, md, ts, vix_data=None, multi_expiry_analyzed=None,
                  expiry_list=None, true_pop_map=None):
    oi_html        = build_oi_html(oc)               if oc   else ""
    kl_html        = build_key_levels_html(tech, oc) if tech else ""
    strat_html     = build_strategies_html(
                         oc, tech, md,
                         multi_expiry_analyzed=multi_expiry_analyzed,
                         expiry_list=expiry_list,
                         true_pop_map=true_pop_map)
    strikes_html   = build_strikes_html(oc)
    ticker_html    = build_ticker_bar(tech, oc, vix_data)
    gauge_html     = build_dual_gauge_hero(oc, tech, md, ts)
    greeks_sidebar = build_greeks_sidebar_html(oc)
    greeks_script  = build_greeks_script_html(oc)
    greeks_table   = build_greeks_table_html(oc)

    C = 2 * 3.14159 * 7
    cp    = tech["price"] if tech else 0
    bias  = md["bias"]; conf = md["confidence"]
    bull  = md["bull"]; bear  = md["bear"]; diff = md["diff"]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Nifty 50 Options Dashboard v20.0</title>
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
    <span style="color:rgba(255,255,255,.75);">Last report generated:&nbsp;<span style="color:#00c896;font-weight:600;">{ts}</span></span>
    <span style="color:rgba(255,255,255,.15);">|</span>
    <span style="color:rgba(255,255,255,.75);">IST&nbsp;<span id="liveClock" style="font-family:'DM Mono',monospace;color:#ffd166;font-weight:700;letter-spacing:1px;">--:--:--</span></span>
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
      <button class="sb-btn" onclick="go('strat',this);filterStrat('bullish',null)">&#9650; Bullish <span class="sb-badge" style="color:var(--bull);">9</span></button>
      <button class="sb-btn" onclick="go('strat',this);filterStrat('bearish',null)">&#9660; Bearish <span class="sb-badge" style="color:var(--bear);">9</span></button>
      <button class="sb-btn" onclick="go('strat',this);filterStrat('nondirectional',null)">&#8596; Non-Directional <span class="sb-badge" style="color:var(--neut);">20</span></button>
    </div>
    <div class="sb-sec">
      <div class="sb-lbl">OPTION CHAIN</div>
      <button class="sb-btn" onclick="go('strikes',this)">Top 5 Strikes</button>
    </div>
    </div>
  </aside>
  <main class="content">
    <div class="main-tabs">
      <button class="main-tab active" id="mainTabOI" onclick="switchMainTab('oi')">&#128202; OI Dashboard</button>
      <button class="main-tab" id="mainTabStrat" onclick="switchMainTab('strat')">&#128203; Option Strategies Reference</button>
    </div>
    <div id="mainPanelOI">
      <div id="oi">{oi_html}</div>
      <div id="kl">{kl_html}</div>
      {greeks_table}
      <div id="strikes">{strikes_html}</div>
      <div class="section">
        <div style="background:rgba(100,128,255,.06);border:1px solid rgba(100,128,255,.18);
                    border-left:3px solid #6480ff;border-radius:12px;padding:16px 18px;
                    font-size:18.8px;color:rgba(255,255,255,.75);line-height:1.8;">
          <strong style="color:rgba(255,255,255,.85);">DISCLAIMER:</strong>&nbsp;Educational only &mdash; NOT financial advice &middot; Smart PoP uses S/R, OI walls, bias &amp; PCR &middot; Use stop losses &middot; Consult a SEBI-registered advisor.
        </div>
      </div>
    </div>
    <div id="mainPanelStrat" style="display:none;">
      {strat_html}
    </div>
  </main>
</div>
<footer>
  <span>NiftyCraft &middot; v19.1 &middot; Holiday-Aware Expiry + Intraday P&amp;L Simulator</span>
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

function updateISTClock() {{
  const el = document.getElementById('liveClock');
  if (!el) return;
  const now = new Date();
  const istOffset = 5.5 * 60 * 60 * 1000;
  const ist = new Date(now.getTime() + (now.getTimezoneOffset() * 60 * 1000) + istOffset);
  const hh = String(ist.getHours()).padStart(2,'0');
  const mm = String(ist.getMinutes()).padStart(2,'0');
  const ss = String(ist.getSeconds()).padStart(2,'0');
  el.textContent = hh + ':' + mm + ':' + ss;
}}
updateISTClock();
setInterval(updateISTClock, 1000);

function switchMainTab(tab) {{
  document.getElementById('mainPanelOI').style.display    = tab === 'oi'    ? '' : 'none';
  document.getElementById('mainPanelStrat').style.display = tab === 'strat' ? '' : 'none';
  document.getElementById('mainTabOI').classList.toggle('active',    tab === 'oi');
  document.getElementById('mainTabStrat').classList.toggle('active', tab === 'strat');
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

// ── Payoff Chart ─────────────────────────────────────────────────────────────
function drawPayoffChart(card, m) {{
  const container = card.querySelector('.sc-payoff-inner');
  if (!container) return;

  const spot  = OC.spot;
  const mp    = m.mpRaw    === 999999 ? null : m.mpRaw;
  const ml    = m.mlRawVal === 999999 ? null : m.mlRawVal;
  const nd    = m.netDelta;
  const ng    = m.netGamma;
  const breakevens = m.beStr.replace(/[₹,]/g,'').split(' / ')
                      .map(v=>parseFloat(v)).filter(v=>!isNaN(v));

  // X range: spot ± 600 pts, step 25
  const xMin = Math.round((spot-600)/50)*50;
  const xMax = Math.round((spot+600)/50)*50;
  const steps=[];
  for(let x=xMin;x<=xMax;x+=25) steps.push(x);

  function expiryPnl(s){{
    const mv=s-spot;
    let p=nd*mv+0.5*ng*mv*mv;
    if(ml!==null) p=Math.max(-ml,p);
    if(mp!==null) p=Math.min(mp,p);
    return p;
  }}

  const pnlArr=steps.map(expiryPnl);
  const minPnl=Math.min(...pnlArr,0);
  const maxPnl=Math.max(...pnlArr,0);
  const pnlRange=maxPnl-minPnl||1;

  const W=520,H=310,padL=68,padR=18,padT=46,padB=52;
  const cW=W-padL-padR, cH=H-padT-padB;

  function xPx(s)   {{ return padL+(s-xMin)/(xMax-xMin)*cW; }}
  function yPx(p)   {{ return padT+cH-(p-minPnl)/pnlRange*cH; }}
  const y0=yPx(0);

  let pathStr='';
  steps.forEach((s,i)=>{{
    const px=xPx(s),py=yPx(pnlArr[i]);
    pathStr+=i===0?`M ${{px}} ${{py}}`:`L ${{px}} ${{py}}`;
  }});

  const profitClip=`M ${{padL}} ${{padT}} L ${{padL+cW}} ${{padT}} L ${{padL+cW}} ${{y0}} L ${{padL}} ${{y0}} Z`;
  const lossClip  =`M ${{padL}} ${{y0}} L ${{padL+cW}} ${{y0}} L ${{padL+cW}} ${{padT+cH}} L ${{padL}} ${{padT+cH}} Z`;

  const yTicks=[];
  for(let i=0;i<=5;i++) yTicks.push({{v:minPnl+pnlRange*i/5,y:yPx(minPnl+pnlRange*i/5)}});
  const xTicks=[];
  for(let x=xMin;x<=xMax;x+=200) xTicks.push(x);

  function fmtY(v){{
    const abs=Math.abs(Math.round(v));
    return (v>=0?'+':'-')+'₹'+(abs>=1000?(abs/1000).toFixed(1)+'k':abs);
  }}

  // Smart BE lines: stagger labels vertically to avoid overlap
  const beFiltered=breakevens.filter(b=>b>xMin&&b<xMax);
  const spx=xPx(spot);
  const beLines=beFiltered.map((b,bi)=>{{
    const bx=xPx(b);
    const lbl='BE \u20b9'+Math.round(b).toLocaleString('en-IN');
    const lblW=lbl.length*6+10;
    // Alternate label row: even=top row, odd=second row to avoid overlap
    const lblY=padT - 8 - (bi%2)*14;
    return `<line x1="${{bx}}" y1="${{padT}}" x2="${{bx}}" y2="${{padT+cH}}" stroke="#ffd166" stroke-width="1.2" stroke-dasharray="4,3" opacity=".8"/>
            <rect x="${{bx-lblW/2}}" y="${{lblY-11}}" width="${{lblW}}" height="13" rx="3" fill="rgba(30,20,0,.85)" stroke="rgba(255,209,102,.4)" stroke-width="1"/>
            <text x="${{bx}}" y="${{lblY}}" text-anchor="middle" font-family="DM Mono,monospace" font-size="9" font-weight="700" fill="#ffd166">${{lbl}}</text>`;
  }}).join('');

  // SPOT label: pill at bottom of chart, below x-axis labels — never overlaps
  const spotLbl='SPOT \u20b9'+Math.round(spot).toLocaleString('en-IN');
  const spotLblW=spotLbl.length*6+12;
  const spotPillX=Math.min(Math.max(spx,padL+spotLblW/2),padL+cW-spotLblW/2);
  const spotLblSvg=`
    <rect x="${{spotPillX-spotLblW/2}}" y="${{H-14}}" width="${{spotLblW}}" height="13" rx="3" fill="rgba(20,20,60,.9)" stroke="rgba(100,128,255,.5)" stroke-width="1"/>
    <text x="${{spotPillX}}" y="${{H-4}}" text-anchor="middle" font-family="DM Mono,monospace" font-size="9" font-weight="700" fill="#6480ff">${{spotLbl}}</text>`;

  const profitFillPts=pathStr+` L ${{xPx(steps[steps.length-1])}} ${{y0}} L ${{xPx(steps[0])}} ${{y0}} Z`;

  const svg=`<svg viewBox="0 0 ${{W}} ${{H}}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:100%;">
    <defs>
      <linearGradient id="pg_${{card.id}}" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#00c896" stop-opacity=".5"/>
        <stop offset="100%" stop-color="#00c896" stop-opacity=".02"/>
      </linearGradient>
      <linearGradient id="lg_${{card.id}}" x1="0" y1="1" x2="0" y2="0">
        <stop offset="0%" stop-color="#f04050" stop-opacity=".5"/>
        <stop offset="100%" stop-color="#f04050" stop-opacity=".02"/>
      </linearGradient>
      <clipPath id="cP_${{card.id}}"><path d="${{profitClip}}"/></clipPath>
      <clipPath id="cL_${{card.id}}"><path d="${{lossClip}}"/></clipPath>
    </defs>
    ${{yTicks.map(t=>`<line x1="${{padL}}" y1="${{t.y}}" x2="${{padL+cW}}" y2="${{t.y}}" stroke="rgba(255,255,255,.05)" stroke-width="1"/>`).join('')}}
    <line x1="${{padL}}" y1="${{y0}}" x2="${{padL+cW}}" y2="${{y0}}" stroke="rgba(255,255,255,.2)" stroke-width="1.5"/>
    <path d="${{profitFillPts}}" fill="url(#pg_${{card.id}})" clip-path="url(#cP_${{card.id}})"/>
    <path d="${{profitFillPts}}" fill="url(#lg_${{card.id}})" clip-path="url(#cL_${{card.id}})"/>
    ${{beLines}}
    <line x1="${{spx}}" y1="${{padT}}" x2="${{spx}}" y2="${{padT+cH}}" stroke="#6480ff" stroke-width="1.5" stroke-dasharray="5,3" opacity=".9"/>
    <path d="${{pathStr}}" fill="none" stroke="#00c896" stroke-width="2.5" stroke-linejoin="round" clip-path="url(#cP_${{card.id}})"/>
    <path d="${{pathStr}}" fill="none" stroke="#f04050" stroke-width="2.5" stroke-linejoin="round" clip-path="url(#cL_${{card.id}})"/>
    ${{yTicks.map(t=>`<text x="${{padL-5}}" y="${{t.y+4}}" text-anchor="end" font-family="DM Mono,monospace" font-size="10" fill="rgba(255,200,80,.75)">${{fmtY(t.v)}}</text>`).join('')}}
    ${{xTicks.map(x=>`<text x="${{xPx(x)}}" y="${{padT+cH+14}}" text-anchor="middle" font-family="DM Mono,monospace" font-size="9.5" fill="rgba(255,200,80,.65)">${{x.toLocaleString('en-IN')}}</text>`).join('')}}
    ${{spotLblSvg}}
    <text x="${{padL+cW/2}}" y="18" text-anchor="middle" font-family="DM Mono,monospace" font-size="11" font-weight="700" letter-spacing="1.5" fill="rgba(0,200,150,.85)">EXPIRY PAYOFF \u00b7 PER LOT</text>
  </svg>`;

  container.innerHTML=`
    <div style="padding:10px 14px 4px;font-family:DM Mono,monospace;font-size:11px;font-weight:700;
      letter-spacing:1.5px;color:rgba(0,200,150,.85);text-transform:uppercase;
      border-bottom:1px solid rgba(0,200,150,.12);">📈 PAYOFF AT EXPIRY</div>
    <div style="flex:1;min-height:270px;padding:6px 4px 2px;">${{svg}}</div>
    <div style="padding:6px 14px 8px;display:flex;gap:14px;border-top:1px solid rgba(255,255,255,.06);
      font-family:DM Mono,monospace;font-size:11px;flex-wrap:wrap;">
      <span style="color:#00c896;">● Profit</span>
      <span style="color:#f04050;">● Loss</span>
      <span style="color:#ffd166;">-- Breakeven</span>
      <span style="color:#6480ff;">-- Spot</span>
    </div>`;
}}


// ── Day Selector Setup (called after innerHTML injection) ────────────────────
function setupDaySelector(simId, nd, nt, nv, ng, maxL, maxP, maxDays) {{
  const _mv = [-500,-400,-300,-200,-150,-100,-50,0,50,100,150,200,300,400,500];
  function _pnl(mv, days) {{
    // Asymmetric IV: down moves spike IV 3x harder than up moves compress it
    const iv = mv < 0 ? -(mv / OC.spot) * 600 : -(mv / OC.spot) * 200;
    let p = nd*mv + 0.5*ng*mv*mv + nv*iv + (nt*days);
    if (maxL !== null) p = Math.max(-maxL, p);
    if (maxP !== null) p = Math.min(maxP*0.9, p);
    return Math.round(p);
  }}
  window['selDay_'+simId] = function(days) {{
    // Update button styles
    for (let d=1; d<=maxDays; d++) {{
      const b = document.getElementById('daybtn_'+simId+'_'+d);
      if (!b) continue;
      const a = (d===days);
      b.style.borderColor = a ? '#ffcc00' : 'rgba(255,185,0,.3)';
      b.style.color       = a ? '#ffcc00' : 'rgba(255,200,80,.5)';
      b.style.background  = a ? 'rgba(255,185,0,.15)' : 'transparent';
    }}
    // Update headers
    const isExp = (days === maxDays);
    const hdr = document.getElementById(simId+'_hdr');
    const col2 = document.getElementById(simId+'_col2');
    const dlbl = document.getElementById(simId+'_daylbl');
    if (hdr)  hdr.innerHTML  = isExp ? '📋 EXPIRY P&amp;L SCENARIOS' : '📋 '+days+' DAY'+(days>1?'S':'')+' P&amp;L SCENARIOS';
    if (col2) col2.innerHTML = isExp ? 'EXPIRY P&amp;L' : days+' DAY'+(days>1?'S':'')+' P&amp;L';
    if (dlbl) dlbl.textContent = days;
    // Rebuild rows
    const tbody = document.getElementById(simId+'_tbody');
    if (!tbody) return;
    tbody.innerHTML = _mv.map(mv => {{
      const pnl = _pnl(mv, days);
      const col = pnl>100?'#38d888':pnl>0?'#ffcc00':pnl>-200?'#ffaa00':'#f04050';
      const mc  = mv>0?'#38d888':mv<0?'#f04050':'#ffcc00';
      const mb  = mv>0?'rgba(56,216,136,.12)':mv<0?'rgba(240,64,80,.18)':'rgba(255,185,0,.18)';
      const ml  = mv>0?'+'+mv:mv===0?'Flat':String(mv);
      const pc  = maxP ? ((pnl/maxP)*100).toFixed(0)+'%' : '—';
      const rb  = mv===0 ? 'background:rgba(255,185,0,.05);' : '';
      return '<tr style="'+rb+'">'
        +'<td style="padding:6px 10px;white-space:nowrap;"><span style="font-family:DM Mono,monospace;font-size:13px;font-weight:700;padding:4px 8px;border-radius:4px;background:'+mb+';color:'+mc+';white-space:nowrap;display:inline-block;min-width:56px;text-align:center;">'+ml+(mv!==0?'p':'')+'</span></td>'
        +'<td style="padding:6px 8px;font-family:DM Mono,monospace;font-size:13px;color:rgba(255,200,80,.82);white-space:nowrap;text-align:left;">'+(OC.spot+mv).toLocaleString('en-IN')+'</td>'
        +'<td style="padding:6px 8px;font-family:DM Mono,monospace;font-weight:700;font-size:15px;color:'+col+';white-space:nowrap;text-align:right;">'+(pnl>=0?'+':'')+'₹'+Math.abs(pnl).toLocaleString('en-IN')+'</td>'
        +'<td style="padding:6px 6px;font-family:DM Mono,monospace;font-size:13px;font-weight:700;color:'+col+';text-align:right;white-space:nowrap;">'+pc+'</td>'
        +'</tr>';
    }}).join('');
  }};
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
        let _m = null;
        try{{
          const shape=card.dataset.shape, cat=card.dataset.cat;
          const scoreResult=smartPoP(shape,cat);
          _m=calcMetrics(shape,scoreResult.edgeScore);
          mel.innerHTML=renderMetrics(_m, scoreResult);
        }}catch(err){{mel.innerHTML='<div class="sc-loading">Could not calculate metrics</div>';}}
        // Payoff chart and day selector run separately so they never break metrics display
        if(_m){{
          try{{ drawPayoffChart(card, _m); }}catch(e){{}}
          try{{
            const _sid=mel.querySelector('[id$="_tbody"]');
            if(_sid){{
              const _simId=_sid.id.replace('_tbody','');
              const _daysLeft=(function(){{try{{const p=OC.expiry.split('-');const mo={{Jan:0,Feb:1,Mar:2,Apr:3,May:4,Jun:5,Jul:6,Aug:7,Sep:8,Oct:9,Nov:10,Dec:11}};const exp=new Date(Date.UTC(parseInt(p[2]),mo[p[1]],parseInt(p[0])));const nu=Date.now()+(new Date().getTimezoneOffset()*60000);const ni=new Date(nu+5.5*3600000);const td=new Date(Date.UTC(ni.getUTCFullYear(),ni.getUTCMonth(),ni.getUTCDate()));return Math.max(1,Math.round((exp-td)/86400000));}}catch(e){{return 4;}}}})();
              setupDaySelector(_simId,_m.netDelta,_m.netTheta,_m.netVega,_m.netGamma,
                _m.mlRawVal===999999?null:_m.mlRawVal,
                _m.mpRaw===999999?null:_m.mpRaw,_daysLeft);
            }}
          }}catch(e){{}}
        }}
      }}
    }}
  }}
}});
</script>
{greeks_script}
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
    live_vix = vix_data["value"] if vix_data else None

    # If VIX fetch fails, derive implied volatility from ATM option prices using
    # Newton-Raphson. Market-implied IV is always more accurate than a hardcoded
    # fallback for Black-Scholes Greek calculations.
    if live_vix is None:
        if oc_raw:
            live_vix = _derive_atm_iv_fallback(oc_raw)
            if live_vix:
                print(f"  VIX unavailable — NR-IV derived from ATM options: {live_vix:.2f}%")
                vix_data = {"value": live_vix, "prev_close": live_vix, "change": 0,
                            "change_pct": 0, "high": live_vix, "low": live_vix,
                            "status": "nr_derived"}
            else:
                live_vix = 18.0
                print(f"  WARNING: VIX + NR-IV both failed — using hardcoded 18.0 (Greeks unreliable)")
        else:
            live_vix = 18.0
            print(f"  WARNING: No option data for NR-IV — using hardcoded 18.0")
    # Fetch all 7 expiries for dropdown
    print("\n  Fetching next 7 expiries for dropdown...")
    multi_expiry_raw, expiry_list = nse.fetch_multiple_expiries(nse_session, nse_headers, n=7, vix=live_vix)
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
    md = compute_market_direction(tech, oc_analysis, live_vix=live_vix)
    print(f"  Bias={md['bias']}  Conf={md['confidence']}  Bull={md['bull']}  Bear={md['bear']}")
    if md.get("sma200_filter_active"):
        print(f"  SMA200 Filter ACTIVE — MACD/RSI bullish signals penalized (structural bear)")
    print(f"  VIX Regime={md['vix_regime'].upper()}  (live_vix={live_vix:.2f})")
    ivp_val = tech.get("ivp", "N/A") if tech else "N/A"
    ivp_lbl = ("CHEAP — avoid short premium" if isinstance(ivp_val, int) and ivp_val < 20
               else "EXPENSIVE — short premium favoured" if isinstance(ivp_val, int) and ivp_val > 70
               else "NORMAL" if isinstance(ivp_val, int) else "N/A")
    print(f"  IVP={ivp_val}%  ({ivp_lbl})")
    print(f"  Max Pain Weight={md['mp_weight']}  (days to expiry affects how much Max Pain scores)")
    if md.get("max_pain_shift"):
        mps = md["max_pain_shift"]
        print(f"  *** MAX PAIN SHIFT: {mps['signal']}")
    if md.get("exhaustion_flag"):
        ef = md["exhaustion_flag"]
        print(f"  *** EXHAUSTION OVERRIDE ({ef['type']}): {ef['signal']}")
        print(f"      → {ef['warning']}")
    if oc_analysis and oc_analysis.get("gex_flip_strike"):
        gfs = oc_analysis["gex_flip_strike"]
        regime = oc_analysis["gex_regime"]
        spot_v = oc_analysis["underlying"]
        print(f"  GEX Flip Strike=₹{gfs:,}  Regime={regime.upper()}  "
              f"(Spot {'above' if spot_v > gfs else 'below'} flip → "
              f"{'dampen' if regime == 'positive' else 'amplify'} moves)")

    print("\nGenerating Holiday-Aware Dashboard...")
    true_pop_map = _compute_true_pop_map(oc_analysis, vix=live_vix) if oc_analysis else {}
    print(f"  True PoP map: {len(true_pop_map)} strategies computed via N(d2)")
    html = generate_html(tech, oc_analysis, md, ts, vix_data=vix_data,
                     multi_expiry_analyzed=multi_expiry_analyzed,
                     expiry_list=expiry_list,
                     true_pop_map=true_pop_map)

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
        "max_pain":        oc_analysis["max_pain"]         if oc_analysis else None,
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
    print(f"  DONE  |  v22.0 · IVP + Theta/Vega Ratio + Theoretical EV")
    print(f"  Bias: {md['bias']}  |  Confidence: {md['confidence']}")
    print("  Holiday list: 2026 NSE official holidays pre-loaded")
    print("  Logic: Tuesday holiday → Monday → Friday (fallback)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    # Runs ONCE and exits — scheduling is handled by crontab / GitHub Actions.
    try:
        main()
    except Exception as e:
        print(f"\n  ERROR during run: {e}")
        raise
