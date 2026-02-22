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
from curl_requests import requests as curl_requests # Note: Fixed import if using curl_cffi
try:
    from curl_cffi import requests as curl_requests
except ImportError:
    import requests as curl_requests
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
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "referer": "https://www.nseindia.com/option-chain",
        }
        session = curl_requests.Session()
        try:
            session.get("https://www.nseindia.com/", headers=headers, impersonate="chrome", timeout=15)
            time.sleep(1.0)
            return session
        except Exception as e:
            print(f"  !! Session Error: {e}")
            return None

    def fetch_data(self):
        url = f"https://www.nseindia.com/api/option-chain-indices?symbol={self.symbol}"
        session = self._make_session()
        if not session: return None
        try:
            r = session.get(url, impersonate="chrome", timeout=20)
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            print(f"  !! Fetch Error: {e}")
        return None

# =================================================================
#  SECTION 2 -- GREEKS CALCULATOR (Black-Scholes)
# =================================================================

def _days_to_expiry(expiry_str):
    try:
        exp_dt = datetime.strptime(expiry_str, "%d-%b-%Y")
        now = datetime.now()
        diff = (exp_dt - now).days + 1
        return max(0.001, float(diff))
    except:
        return 0.5

def _bs_greeks(S, K, T, r, sigma, option_type="CE"):
    if T <= 0 or sigma <= 0:
        return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0}
    try:
        d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        gamma = _norm.pdf(d1) / (S * sigma * sqrt(T))
        vega = (S * _norm.pdf(d1) * sqrt(T)) / 100
        if option_type == "CE":
            delta = _norm.cdf(d1)
            theta = -(S * _norm.pdf(d1) * sigma / (2 * sqrt(T))) - r * K * exp(-r * T) * _norm.cdf(d2)
        else:
            delta = _norm.cdf(d1) - 1
            theta = -(S * _norm.pdf(d1) * sigma / (2 * sqrt(T))) + r * K * exp(-r * T) * _norm.cdf(-d2)
        return {
            "delta": round(delta, 4),
            "gamma": round(gamma, 6),
            "theta": round(theta / 365.0, 4),
            "vega": round(vega, 4)
        }
    except:
        return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0}

def _compute_greeks_for_row(r, spot, expiry_str, risk_free=0.065, vix=18.0):
    T = _days_to_expiry(expiry_str) / 365.0
    K = float(r["Strike"])
    
    def process_side(side):
        delta_nse = float(r.get(f"{side}_Delta", 0) or 0)
        theta_nse = float(r.get(f"{side}_Theta", 0) or 0)
        iv_raw = float(r.get(f"{side}_IV", 0) or 0)
        
        other_side = "PE" if side == "CE" else "CE"
        other_iv = float(r.get(f"{other_side}_IV", 0) or 0)
        iv_to_use = iv_raw if iv_raw > 0.1 else other_iv if other_iv > 0.1 else vix

        if abs(delta_nse) > 0.001 and abs(theta_nse) > 0.0001:
            return {
                "delta": round(delta_nse, 4),
                "theta": round(theta_nse, 4),
                "vega":  round(float(r.get(f"{side}_Vega", 0) or 0), 4),
                "gamma": round(float(r.get(f"{side}_Gamma", 0) or 0), 6),
            }
        else:
            return _bs_greeks(spot, K, T, risk_free, iv_to_use / 100, side)

    return process_side("CE"), process_side("PE")

# =================================================================
#  SECTION 3 -- DATA PROCESSING
# =================================================================

def analyze_option_chain(data, vix_val=18.0):
    if not data or "filtered" not in data: return None
    records = data["filtered"]["data"]
    spot = float(data["records"]["underlyingValue"])
    expiry = data["filtered"]["data"][0]["expiryDate"]
    
    rows = []
    for item in records:
        strike = item["strikePrice"]
        ce = item.get("CE", {})
        pe = item.get("PE", {})
        rows.append({
            "Strike": strike,
            "CE_LTP": ce.get("lastPrice", 0),
            "CE_Chg": ce.get("change", 0),
            "CE_IV": ce.get("impliedVolatility", 0),
            "CE_Delta": ce.get("pdelta", 0),
            "CE_Theta": ce.get("ptheta", 0),
            "CE_Gamma": ce.get("pgamma", 0),
            "CE_Vega": ce.get("pvega", 0),
            "PE_LTP": pe.get("lastPrice", 0),
            "PE_Chg": pe.get("change", 0),
            "PE_IV": pe.get("impliedVolatility", 0),
            "PE_Delta": pe.get("pdelta", 0),
            "PE_Theta": pe.get("ptheta", 0),
            "PE_Gamma": pe.get("pgamma", 0),
            "PE_Vega": pe.get("pvega", 0),
        })
    
    df = pd.DataFrame(rows)
    df["Dist"] = (df["Strike"] - spot).abs()
    atm_row = df.loc[df["Dist"].idxmin()]
    atm_strike = atm_row["Strike"]
    
    # Pre-calculate Greeks for EVERY strike so the UI can update
    chain_map = {}
    for _, row in df.iterrows():
        ce_g, pe_g = _compute_greeks_for_row(row, spot, expiry, 0.065, vix_val)
        chain_map[str(int(row["Strike"]))] = {
            "CE_LTP": row["CE_LTP"],
            "CE_Chg": row["CE_Chg"],
            "PE_LTP": row["PE_LTP"],
            "PE_Chg": row["PE_Chg"],
            "CE_Greeks": ce_g,
            "PE_Greeks": pe_g
        }

    return {
        "spot": spot,
        "expiry": expiry,
        "atm_strike": atm_strike,
        "chain_map": chain_map,
        "strikes": sorted(df["Strike"].unique().tolist())
    }

# =================================================================
#  SECTION 4 -- HTML GENERATOR
# =================================================================

def generate_html(oc, vix_data):
    ts = datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
    spot = oc["spot"]
    atm_strike = str(int(oc["atm_strike"]))
    chain_json = json.dumps(oc["chain_map"])
    
    # Get initial values for the ATM strike to show on page load
    initial = oc["chain_map"][atm_strike]

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Nifty Strategy</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{ background: #0b0f19; color: #e0e6ed; font-family: 'Segoe UI', sans-serif; margin: 20px; }}
            .card {{ background: #161b28; border: 1px solid #2d3343; padding: 20px; border-radius: 12px; max-width: 800px; margin: auto; }}
            .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }}
            .side {{ background: #1f2637; padding: 15px; border-radius: 8px; border-left: 4px solid #00ffcc; }}
            .side.put {{ border-left: 4px solid #ff4d4d; }}
            .label {{ color: #8892b0; font-size: 0.85em; margin-bottom: 4px; }}
            .val {{ font-size: 1.2em; font-weight: bold; }}
            select {{ background: #1f2637; color: white; border: 1px solid #444; padding: 8px; border-radius: 4px; width: 100%; margin-top: 10px; }}
            h2 {{ color: #00ffcc; margin-top: 0; }}
            .footer {{ font-size: 0.8em; color: #555; text-align: center; margin-top: 20px; }}
        </style>
    </head>
    <body>
        <div class="card">
            <h2>Nifty Options Greeks</h2>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>Spot: <b>{spot:.2f}</b> | VIX: <b>{vix_data}</b></div>
                <div style="text-align: right;">
                    <span class="label">Select Strike:</span><br>
                    <select id="strikeSelect" onchange="updateStrikeData()">
                        {" ".join([f'<option value="{int(s)}" {"selected" if int(s)==int(atm_strike) else ""}>{int(s)}</option>' for s in oc["strikes"]])}
                    </select>
                </div>
            </div>

            <h3 style="text-align:center; border-bottom: 1px solid #333; padding-bottom:10px;">
                Selected Strike: <span id="selectedStrikeDisplay">{atm_strike}</span>
            </h3>

            <div class="grid">
                <div class="side">
                    <div style="color: #00ffcc; font-weight: bold; margin-bottom: 10px;">CALL OPTION</div>
                    <div class="label">LTP</div>
                    <div class="val" id="CE_LTP">{initial['CE_LTP']:.2f}</div>
                    <div id="CE_Chg" style="font-size: 0.9em;">{initial['CE_Chg']:.2f}</div>
                    <hr style="border:0; border-top:1px solid #333; margin:15px 0;">
                    <div class="label">Delta: <span class="val" id="ce_delta">{initial['CE_Greeks']['delta']}</span></div>
                    <div class="label">Theta: <span class="val" id="ce_theta">{initial['CE_Greeks']['theta']}</span></div>
                    <div class="label">Gamma: <span class="val" id="ce_gamma">{initial['CE_Greeks']['gamma']}</span></div>
                    <div class="label">Vega:  <span class="val" id="ce_vega">{initial['CE_Greeks']['vega']}</span></div>
                </div>

                <div class="side put">
                    <div style="color: #ff4d4d; font-weight: bold; margin-bottom: 10px;">PUT OPTION</div>
                    <div class="label">LTP</div>
                    <div class="val" id="PE_LTP">{initial['PE_LTP']:.2f}</div>
                    <div id="PE_Chg" style="font-size: 0.9em;">{initial['PE_Chg']:.2f}</div>
                    <hr style="border:0; border-top:1px solid #333; margin:15px 0;">
                    <div class="label">Delta: <span class="val" id="pe_delta">{initial['PE_Greeks']['delta']}</span></div>
                    <div class="label">Theta: <span class="val" id="pe_theta">{initial['PE_Greeks']['theta']}</span></div>
                    <div class="label">Gamma: <span class="val" id="pe_gamma">{initial['PE_Greeks']['gamma']}</span></div>
                    <div class="label">Vega:  <span class="val" id="pe_vega">{initial['PE_Greeks']['vega']}</span></div>
                </div>
            </div>
            <div class="footer">Last Updated: {ts}</div>
        </div>

        <script>
            window.optionChainData = {chain_json};

            function updateStrikeData() {{
                const strike = document.getElementById("strikeSelect").value;
                const data = window.optionChainData[strike];
                if (!data) return;

                document.getElementById("selectedStrikeDisplay").innerText = strike;

                // Update LTP and Change
                const updatePrice = (side) => {{
                    const ltp = data[side + "_LTP"] || 0;
                    const chg = data[side + "_Chg"] || 0;
                    const color = chg >= 0 ? "#00ffcc" : "#ff4d4d";
                    document.getElementById(side + "_LTP").innerText = ltp.toFixed(2);
                    const chgEl = document.getElementById(side + "_Chg");
                    chgEl.innerText = (chg >= 0 ? "+" : "") + chg.toFixed(2);
                    chgEl.style.color = color;
                }};
                updatePrice("CE");
                updatePrice("PE");

                // Update Greeks
                const greeks = ["delta", "gamma", "theta", "vega"];
                greeks.forEach(g => {{
                    const ceVal = data.CE_Greeks ? data.CE_Greeks[g] : 0;
                    document.getElementById("ce_" + g).innerText = ceVal.toFixed(g === "gamma" ? 6 : 4);
                    
                    const peVal = data.PE_Greeks ? data.PE_Greeks[g] : 0;
                    document.getElementById("pe_" + g).innerText = peVal.toFixed(g === "gamma" ? 6 : 4);
                }});
            }}
        </script>
    </body>
    </html>
    """
    return html

# =================================================================
#  SECTION 5 -- MAIN EXECUTION
# =================================================================

if __name__ == "__main__":
    print("Fetching Nifty Option Chain...")
    fetcher = NSEOptionChain()
    raw_data = fetcher.fetch_data()
    
    # Try to get VIX for better fallback Greeks
    try:
        vix_ticker = yf.Ticker("^INDIAVIX")
        vix_val = vix_ticker.history(period="1d")["Close"].iloc[-1]
    except:
        vix_val = 15.0

    if raw_data:
        analysis = analyze_option_chain(raw_data, vix_val)
        if analysis:
            html_content = generate_html(analysis, round(vix_val, 2))
            with open("index.html", "w") as f:
                f.write(html_content)
            print("Dashboard Generated: index.html")
    else:
        print("Failed to fetch data from NSE.")
