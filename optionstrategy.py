#!/usr/bin/env python3
"""
Nifty 50 Options Strategy Dashboard — GitHub Pages Generator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Fetches live data from NSE → runs analysis → writes docs/index.html
Triggered by GitHub Actions on every push / scheduled run.

pip install curl_cffi pandas numpy yfinance pytz beautifulsoup4 requests
"""

import os, json, time, warnings, pytz, calendar
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from curl_cffi import requests as curl_requests

try:
    import requests as _req
except ImportError:
    _req = None

try:
    from bs4 import BeautifulSoup
    BS4_OK = True
except ImportError:
    BS4_OK = False

import yfinance as yf
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
#  SECTION 1 ── FII / DII DATA HELPERS  (from your script)
# ═══════════════════════════════════════════════════════════════

def _last_5_trading_days():
    ist_off = timedelta(hours=5, minutes=30)
    today   = (datetime.utcnow() + ist_off).date()
    days, d = [], today - timedelta(days=1)
    while len(days) < 5:
        if d.weekday() < 5:
            days.append(d)
        d -= timedelta(days=1)
    days.reverse()
    return days

def _parse_nse_fiidii(raw):
    if not isinstance(raw, list) or not raw:
        return []
    days = []
    for row in raw[:10]:
        try:
            dt_obj  = datetime.strptime(row.get("date", ""), "%d-%b-%Y")
            fii_net = float(row.get("fiiBuyValue", 0) or 0) - float(row.get("fiiSellValue", 0) or 0)
            dii_net = float(row.get("diiBuyValue", 0) or 0) - float(row.get("diiSellValue", 0) or 0)
            days.append({'date': dt_obj.strftime("%b %d"), 'day': dt_obj.strftime("%a"),
                         'fii': round(fii_net, 2), 'dii': round(dii_net, 2)})
        except Exception:
            continue
    if len(days) < 3:
        return []
    days = days[:5]
    days.reverse()
    return days

def _fetch_from_groww():
    if not BS4_OK or _req is None:
        return []
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://groww.in/",
        }
        resp = _req.get("https://groww.in/fii-dii-data", headers=headers, timeout=15)
        if resp.status_code != 200:
            return []
        soup  = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table")
        if not table:
            return []
        rows = table.find_all("tr")
        days = []
        for row in rows[1:]:
            cols = [td.get_text(strip=True) for td in row.find_all("td")]
            if len(cols) < 7:
                continue
            try:
                dt_obj  = datetime.strptime(cols[0], "%d %b %Y")
                fii_net = float(cols[3].replace(",", "").replace("+", ""))
                dii_net = float(cols[6].replace(",", "").replace("+", ""))
                days.append({'date': dt_obj.strftime("%b %d"), 'day': dt_obj.strftime("%a"),
                             'fii': round(fii_net, 2), 'dii': round(dii_net, 2)})
            except Exception:
                continue
            if len(days) == 5:
                break
        if len(days) >= 3:
            days.reverse()
            print(f"  ✅ FII/DII from Groww: {days[0]['date']} → {days[-1]['date']}")
            return days
        return []
    except Exception as e:
        print(f"  ⚠️  Groww scrape failed: {e}")
        return []

def _fetch_from_nse_curl():
    try:
        headers = {
            "authority": "www.nseindia.com",
            "accept": "application/json, text/plain, */*",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
            "referer": "https://www.nseindia.com/reports/fii-dii",
            "accept-language": "en-US,en;q=0.9",
        }
        s = curl_requests.Session()
        s.get("https://www.nseindia.com/", headers=headers, impersonate="chrome", timeout=12)
        time.sleep(1.2)
        s.get("https://www.nseindia.com/reports/fii-dii", headers=headers, impersonate="chrome", timeout=12)
        time.sleep(0.8)
        resp = s.get("https://www.nseindia.com/api/fiidiiTradeReact", headers=headers, impersonate="chrome", timeout=20)
        if resp.status_code == 200:
            days = _parse_nse_fiidii(resp.json())
            if days:
                print(f"  ✅ FII/DII from NSE (curl): {days[0]['date']} → {days[-1]['date']}")
                return days
    except Exception as e:
        print(f"  ⚠️  NSE curl failed: {e}")
    return []

def fetch_fii_dii_data():
    days = _fetch_from_groww()
    if days:
        return days
    days = _fetch_from_nse_curl()
    if days:
        return days
    print("  📌 FII/DII: using date-corrected fallback")
    tdays = _last_5_trading_days()
    placeholder = [(-1540.20, 2103.50), (823.60, 891.40), (-411.80, 1478.30), (69.45, 1174.21), (-972.13, 1666.98)]
    return [{'date': d.strftime('%b %d'), 'day': d.strftime('%a'),
             'fii': placeholder[i][0], 'dii': placeholder[i][1], 'fallback': True}
            for i, d in enumerate(tdays)]

def compute_fii_dii_summary(data):
    fii_vals = [d['fii'] for d in data]
    dii_vals = [d['dii'] for d in data]
    fii_avg  = sum(fii_vals) / len(fii_vals)
    dii_avg  = sum(dii_vals) / len(dii_vals)
    net_avg  = fii_avg + dii_avg
    fii_span = f'<span style="color:#ff5252;font-weight:700;">₹{fii_avg:.0f} Cr/day</span>'
    dii_span = f'<span style="color:#40c4ff;font-weight:700;">₹{dii_avg:+.0f} Cr/day</span>'
    net_span = f'<span style="color:#b388ff;font-weight:700;">₹{net_avg:+.0f} Cr/day</span>'

    if fii_avg > 0 and dii_avg > 0:
        label, emoji, color, badge_cls = 'STRONGLY BULLISH', '🚀', '#00e676', 'fii-bull'
        fii_span = f'<span style="color:#00e676;font-weight:700;">₹{fii_avg:+.0f} Cr/day</span>'
        insight  = (f"Both FIIs (avg {fii_span}) and DIIs (avg {dii_span}) are net buyers — "
                    f"strong dual institutional confirmation. Net combined flow: {net_span}.")
    elif fii_avg < 0 and dii_avg > 0 and dii_avg > abs(fii_avg):
        label, emoji, color, badge_cls = 'CAUTIOUSLY BULLISH', '📈', '#69f0ae', 'fii-cbull'
        insight  = (f"FIIs are net sellers (avg {fii_span}) but DIIs are absorbing strongly (avg {dii_span}). "
                    f"DII support is cushioning downside. Net combined flow: {net_span}.")
    elif fii_avg < 0 and dii_avg > 0:
        label, emoji, color, badge_cls = 'MIXED / NEUTRAL', '⚖️', '#ffd740', 'fii-neu'
        insight  = (f"FII selling (avg {fii_span}) is partly offset by DII buying (avg {dii_span}). "
                    f"Watch for 3+ consecutive days of FII buying. Net combined flow: {net_span}.")
    elif fii_avg < 0 and dii_avg < 0:
        label, emoji, color, badge_cls = 'BEARISH', '📉', '#ff5252', 'fii-bear'
        dii_span = f'<span style="color:#ff5252;font-weight:700;">₹{dii_avg:.0f} Cr/day</span>'
        insight  = (f"Both FIIs (avg {fii_span}) and DIIs (avg {dii_span}) are net sellers — "
                    f"clear bearish institutional pressure. Net combined flow: {net_span}.")
    else:
        label, emoji, color, badge_cls = 'NEUTRAL', '🔄', '#b0bec5', 'fii-neu'
        insight = "Mixed signals from institutional participants. Wait for a clearer trend."

    max_abs = max(abs(v) for row in data for v in (row['fii'], row['dii'])) or 1
    return {'fii_avg': fii_avg, 'dii_avg': dii_avg, 'net_avg': net_avg,
            'label': label, 'emoji': emoji, 'color': color,
            'badge_cls': badge_cls, 'insight': insight, 'max_abs': max_abs}


# ═══════════════════════════════════════════════════════════════
#  SECTION 2 ── NSE OPTION CHAIN FETCHER  (from your script)
# ═══════════════════════════════════════════════════════════════

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
            print(f"  ⚠️  Session warm-up: {e}")
        return session, headers

    def _upcoming_tuesday(self):
        ist_tz    = pytz.timezone('Asia/Kolkata')
        today_ist = datetime.now(ist_tz).date()
        weekday   = today_ist.weekday()
        days_ahead = 7 if weekday == 1 else (1 - weekday) % 7 or 7
        return (today_ist + timedelta(days=days_ahead)).strftime('%d-%b-%Y')

    def _fetch_available_expiries(self, session, headers):
        try:
            url  = f"https://www.nseindia.com/api/option-chain-v3?type=Indices&symbol={self.symbol}"
            resp = session.get(url, headers=headers, impersonate="chrome", timeout=20)
            if resp.status_code == 200:
                expiries = resp.json().get('records', {}).get('expiryDates', [])
                if expiries:
                    print(f"  📅 Available expiries: {expiries[:5]}")
                    return expiries[0]
        except Exception as e:
            print(f"  ⚠️  Expiry fetch: {e}")
        return None

    def _fetch_for_expiry(self, session, headers, expiry):
        api_url = (f"https://www.nseindia.com/api/option-chain-v3"
                   f"?type=Indices&symbol={self.symbol}&expiry={expiry}")
        for attempt in range(1, 3):
            try:
                print(f"    Attempt {attempt}: expiry={expiry}")
                resp = session.get(api_url, headers=headers, impersonate="chrome", timeout=30)
                print(f"    HTTP {resp.status_code}")
                if resp.status_code != 200:
                    time.sleep(2)
                    continue
                json_data  = resp.json()
                data       = json_data.get('records', {}).get('data', [])
                if not data:
                    return None
                rows = []
                for item in data:
                    strike = item.get('strikePrice')
                    ce     = item.get('CE', {})
                    pe     = item.get('PE', {})
                    rows.append({
                        'Strike': strike,
                        'CE_LTP': ce.get('lastPrice', 0), 'CE_OI': ce.get('openInterest', 0),
                        'CE_Vol': ce.get('totalTradedVolume', 0),
                        'PE_LTP': pe.get('lastPrice', 0), 'PE_OI': pe.get('openInterest', 0),
                        'PE_Vol': pe.get('totalTradedVolume', 0),
                        'CE_OI_Change': ce.get('changeinOpenInterest', 0),
                        'PE_OI_Change': pe.get('changeinOpenInterest', 0),
                    })
                df_full     = pd.DataFrame(rows).sort_values('Strike').reset_index(drop=True)
                underlying  = json_data.get('records', {}).get('underlyingValue', 0)
                atm_strike  = round(underlying / 50) * 50
                all_strikes = sorted(df_full['Strike'].unique())
                if atm_strike in all_strikes:
                    atm_idx = all_strikes.index(atm_strike)
                else:
                    atm_idx    = min(range(len(all_strikes)), key=lambda i: abs(all_strikes[i] - underlying))
                    atm_strike = all_strikes[atm_idx]
                lo = max(0, atm_idx - 10)
                hi = min(len(all_strikes) - 1, atm_idx + 10)
                df  = df_full[df_full['Strike'].isin(all_strikes[lo:hi+1])].reset_index(drop=True)
                print(f"    ✅ {len(df_full)} strikes → ATM±10 → {len(df)} rows")
                return {'expiry': expiry, 'df': df, 'underlying': underlying, 'atm_strike': atm_strike}
            except Exception as e:
                print(f"    ❌ Attempt {attempt}: {e}")
                time.sleep(2)
        return None

    def fetch(self):
        session, headers = self._make_session()
        expiry = self._upcoming_tuesday()
        print(f"  🗓️  Fetching option chain for: {expiry}")
        result = self._fetch_for_expiry(session, headers, expiry)
        if result is None:
            real_expiry = self._fetch_available_expiries(session, headers)
            if real_expiry and real_expiry != expiry:
                print(f"  🔄 Retrying with NSE expiry: {real_expiry}")
                result = self._fetch_for_expiry(session, headers, real_expiry)
        if result is None:
            print("  ❌ Option chain fetch failed.")
        return result


# ═══════════════════════════════════════════════════════════════
#  SECTION 3 ── OPTION CHAIN ANALYSIS  (from your script)
# ═══════════════════════════════════════════════════════════════

def analyze_option_chain(oc_data):
    if not oc_data:
        return None
    df = oc_data['df']
    total_ce_oi  = df['CE_OI'].sum()
    total_pe_oi  = df['PE_OI'].sum()
    total_ce_vol = df['CE_Vol'].sum()
    total_pe_vol = df['PE_Vol'].sum()
    pcr_oi   = total_pe_oi  / total_ce_oi  if total_ce_oi  > 0 else 0
    pcr_vol  = total_pe_vol / total_ce_vol if total_ce_vol > 0 else 0
    ce_chg   = int(df['CE_OI_Change'].sum())
    pe_chg   = int(df['PE_OI_Change'].sum())
    net_chg  = pe_chg + ce_chg

    # OI direction logic (from your script)
    if   ce_chg > 0 and pe_chg < 0:
        oi_dir, oi_sig, oi_icon, oi_cls = "Strong Bearish", "Call Build-up + Put Unwinding", "🔴", "bearish"
    elif ce_chg < 0 and pe_chg > 0:
        oi_dir, oi_sig, oi_icon, oi_cls = "Strong Bullish", "Put Build-up + Call Unwinding", "🟢", "bullish"
    elif ce_chg > 0 and pe_chg > 0:
        if   pe_chg > ce_chg * 1.5:
            oi_dir, oi_sig, oi_icon, oi_cls = "Bullish", "Put Build-up Dominant", "🟢", "bullish"
        elif ce_chg > pe_chg * 1.5:
            oi_dir, oi_sig, oi_icon, oi_cls = "Bearish", "Call Build-up Dominant", "🔴", "bearish"
        else:
            oi_dir, oi_sig, oi_icon, oi_cls = "Neutral (High Vol)", "Both Calls & Puts Building", "🟡", "neutral"
    elif ce_chg < 0 and pe_chg < 0:
        oi_dir, oi_sig, oi_icon, oi_cls = "Neutral (Unwinding)", "Both Calls & Puts Unwinding", "🟡", "neutral"
    else:
        if   net_chg > 0: oi_dir, oi_sig, oi_icon, oi_cls = "Moderately Bullish", "Net Put Accumulation", "🟢", "bullish"
        elif net_chg < 0: oi_dir, oi_sig, oi_icon, oi_cls = "Moderately Bearish", "Net Call Accumulation", "🔴", "bearish"
        else:              oi_dir, oi_sig, oi_icon, oi_cls = "Neutral", "Balanced OI Changes", "🟡", "neutral"

    max_ce_row  = df.loc[df['CE_OI'].idxmax()]
    max_pe_row  = df.loc[df['PE_OI'].idxmax()]
    df['pain']  = abs(df['CE_OI'] - df['PE_OI'])
    max_pain_row = df.loc[df['pain'].idxmin()]
    top_ce = df.nlargest(5, 'CE_OI')[['Strike', 'CE_OI', 'CE_LTP']].to_dict('records')
    top_pe = df.nlargest(5, 'PE_OI')[['Strike', 'PE_OI', 'PE_LTP']].to_dict('records')

    return {
        'expiry': oc_data['expiry'], 'underlying': oc_data['underlying'], 'atm_strike': oc_data['atm_strike'],
        'pcr_oi': round(pcr_oi, 3), 'pcr_vol': round(pcr_vol, 3),
        'total_ce_oi': int(total_ce_oi), 'total_pe_oi': int(total_pe_oi),
        'max_ce_strike': int(max_ce_row['Strike']), 'max_ce_oi': int(max_ce_row['CE_OI']),
        'max_pe_strike': int(max_pe_row['Strike']), 'max_pe_oi': int(max_pe_row['PE_OI']),
        'max_pain': int(max_pain_row['Strike']),
        'ce_chg': ce_chg, 'pe_chg': pe_chg, 'net_chg': net_chg,
        'oi_dir': oi_dir, 'oi_sig': oi_sig, 'oi_icon': oi_icon, 'oi_cls': oi_cls,
        'top_ce': top_ce, 'top_pe': top_pe, 'df': df,
    }


# ═══════════════════════════════════════════════════════════════
#  SECTION 4 ── TECHNICAL ANALYSIS  (from your script)
# ═══════════════════════════════════════════════════════════════

def get_technical_data():
    try:
        print("  Fetching technical data from yfinance...")
        nifty = yf.Ticker("^NSEI")
        df    = nifty.history(period="1y")
        if df.empty:
            return None
        df['SMA_20']  = df['Close'].rolling(20).mean()
        df['SMA_50']  = df['Close'].rolling(50).mean()
        df['SMA_200'] = df['Close'].rolling(200).mean()
        delta = df['Close'].diff()
        gain  = delta.where(delta > 0, 0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI']    = 100 - (100 / (1 + gain / loss))
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD']   = df['EMA_12'] - df['EMA_26']
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        latest = df.iloc[-1]
        cp     = latest['Close']

        # 1H key levels (from your script)
        s1 = s2 = r1 = r2 = None
        try:
            df_1h = nifty.history(interval="1h", period="60d")
            if not df_1h.empty:
                recent_1h = df_1h.tail(120)
                highs = sorted(recent_1h['High'].values)
                lows  = sorted(recent_1h['Low'].values)
                res_c = [h for h in highs if cp < h <= cp + 200]
                sup_c = [l for l in lows  if cp - 200 <= l < cp]
                if len(res_c) >= 4:
                    r1 = round(float(np.percentile(res_c, 40)) / 25) * 25
                    r2 = round(float(np.percentile(res_c, 80)) / 25) * 25
                if len(sup_c) >= 4:
                    s1 = round(float(np.percentile(sup_c, 70)) / 25) * 25
                    s2 = round(float(np.percentile(sup_c, 20)) / 25) * 25
                if r1 and r1 <= cp:   r1 = round((cp + 50)  / 25) * 25
                if r2 and r1 and r2 <= r1: r2 = r1 + 75
                if s1 and s1 >= cp:   s1 = round((cp - 50)  / 25) * 25
                if s2 and s1 and s2 >= s1: s2 = s1 - 75
                print(f"  ✓ 1H Levels: S2={s2} S1={s1} CMP={cp:.0f} R1={r1} R2={r2}")
        except Exception as e:
            print(f"  ⚠️  1H data: {e}")

        recent_d       = df.tail(60)
        resistance     = r1 if r1 else recent_d['High'].quantile(0.90)
        support        = s1 if s1 else recent_d['Low'].quantile(0.10)
        strong_res     = r2 if r2 else resistance + 100
        strong_sup     = s2 if s2 else support - 100

        print(f"  ✓ Technical | CMP={cp:.2f} RSI={latest['RSI']:.1f} MACD={latest['MACD']:.2f}")
        return {
            'price': cp,
            'sma20': latest['SMA_20'], 'sma50': latest['SMA_50'], 'sma200': latest['SMA_200'],
            'rsi': latest['RSI'], 'macd': latest['MACD'], 'signal_line': latest['Signal'],
            'support': support, 'resistance': resistance,
            'strong_sup': strong_sup, 'strong_res': strong_res,
        }
    except Exception as e:
        print(f"  ❌ Technical error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
#  SECTION 5 ── MARKET DIRECTION SCORING  (from your script)
# ═══════════════════════════════════════════════════════════════

def compute_market_direction(tech, oc_analysis):
    """Exact scoring logic from your NiftyHTMLAnalyzer.generate_analysis_data()"""
    if not tech:
        return {'bias': 'UNKNOWN', 'confidence': 'LOW', 'bull': 0, 'bear': 0}
    cp   = tech['price']
    bull = bear = 0

    for sma in ['sma20', 'sma50', 'sma200']:
        if cp > tech[sma]: bull += 1
        else:               bear += 1

    rsi = tech['rsi']
    if   rsi > 70: bear += 1
    elif rsi < 30: bull += 2

    if tech['macd'] > tech['signal_line']: bull += 1
    else:                                   bear += 1

    if oc_analysis:
        pcr = oc_analysis['pcr_oi']
        if   pcr > 1.2: bull += 2
        elif pcr < 0.7: bear += 2
        mp = oc_analysis['max_pain']
        if   cp > mp + 100: bear += 1
        elif cp < mp - 100: bull += 1

    diff = bull - bear
    print(f"  📊 Score → Bullish:{bull}  Bearish:{bear}  Diff:{diff}")

    if   diff >= 3:  bias, bias_cls = "BULLISH",  "bullish"; confidence = "HIGH" if diff >= 4 else "MEDIUM"
    elif diff <= -3: bias, bias_cls = "BEARISH",  "bearish"; confidence = "HIGH" if diff <= -4 else "MEDIUM"
    else:            bias, bias_cls = "SIDEWAYS", "neutral"; confidence = "MEDIUM"

    return {'bias': bias, 'bias_cls': bias_cls, 'confidence': confidence, 'bull': bull, 'bear': bear, 'diff': diff}


def recommend_strategies(bias, atm_strike, oi_dir):
    """Option strategies recommended by your script logic"""
    atm = atm_strike
    tech_map = {
        "BULLISH": [
            {'name': 'Bull Call Spread', 'legs': f'Buy {atm} CE · Sell {atm+200} CE', 'type': 'Debit', 'risk': 'Moderate'},
            {'name': 'Long Call',        'legs': f'Buy {atm} CE',                     'type': 'Debit', 'risk': 'High'},
            {'name': 'Bull Put Spread',  'legs': f'Sell {atm-100} PE · Buy {atm-200} PE', 'type': 'Credit', 'risk': 'Moderate'},
        ],
        "BEARISH": [
            {'name': 'Bear Put Spread',  'legs': f'Buy {atm} PE · Sell {atm-200} PE', 'type': 'Debit', 'risk': 'Moderate'},
            {'name': 'Long Put',         'legs': f'Buy {atm} PE',                     'type': 'Debit', 'risk': 'High'},
            {'name': 'Bear Call Spread', 'legs': f'Sell {atm+100} CE · Buy {atm+200} CE', 'type': 'Credit', 'risk': 'Moderate'},
        ],
        "SIDEWAYS": [
            {'name': 'Iron Condor',   'legs': f'Sell {atm+100} CE · Buy {atm+200} CE · Sell {atm-100} PE · Buy {atm-200} PE', 'type': 'Credit', 'risk': 'Low'},
            {'name': 'Iron Butterfly','legs': f'Sell {atm} CE · Sell {atm} PE · Buy {atm+100} CE · Buy {atm-100} PE',       'type': 'Credit', 'risk': 'Low'},
            {'name': 'Short Straddle','legs': f'Sell {atm} CE · Sell {atm} PE',                                              'type': 'Credit', 'risk': 'Very High'},
        ],
    }
    oi_map = {
        "Strong Bullish": {'name': 'Long Call',       'legs': f'Buy {atm} CE', 'signal': '🟢 Put build-up — bullish momentum'},
        "Bullish":        {'name': 'Long Call',       'legs': f'Buy {atm} CE', 'signal': '🟢 Put build-up dominant'},
        "Strong Bearish": {'name': 'Long Put',        'legs': f'Buy {atm} PE', 'signal': '🔴 Call build-up — bearish momentum'},
        "Bearish":        {'name': 'Long Put',        'legs': f'Buy {atm} PE', 'signal': '🔴 Call build-up dominant'},
        "Neutral (High Vol)": {'name': 'Long Straddle', 'legs': f'Buy {atm} CE + {atm} PE', 'signal': '🟡 Both building — big move expected'},
        "Neutral (Unwinding)":{'name': 'Iron Butterfly','legs': f'Sell {atm} CE+PE, Buy {atm+100} CE+{atm-100} PE', 'signal': '🟡 Unwinding — range bound'},
    }
    tech_strats = tech_map.get(bias, tech_map["SIDEWAYS"])
    oi_strat    = oi_map.get(oi_dir, {'name': 'Vertical Spread', 'legs': f'Near {atm}', 'signal': '🟡 Mixed signals'})
    return tech_strats, oi_strat


# ═══════════════════════════════════════════════════════════════
#  SECTION 6 ── STRATEGY DEFINITIONS  (educational sidebar)
# ═══════════════════════════════════════════════════════════════

ALL_STRATEGIES = {
    "bullish": {
        "label": "Bullish", "color": "#00d4a0",
        "items": [
            {"name": "Long Call",       "risk": "Limited",  "reward": "Unlimited", "legs": "BUY CALL (ATM)",
             "desc": "Buy a call option. Profits as stock rises. Risk limited to premium paid.",
             "mp": "Unlimited", "ml": "Premium Paid", "be": "Strike + Premium"},
            {"name": "Covered Call",    "risk": "Moderate", "reward": "Limited",   "legs": "OWN STOCK · SELL CALL (OTM)",
             "desc": "Own shares and sell a call against them. Generates income; caps upside.",
             "mp": "Strike − Cost + Premium", "ml": "Cost − Premium", "be": "Stock Cost − Premium"},
            {"name": "Bull Call Spread","risk": "Limited",  "reward": "Limited",   "legs": "BUY CALL (Low) · SELL CALL (High)",
             "desc": "Buy lower call, sell higher call. Reduces cost; caps profit at upper strike.",
             "mp": "Spread Width − Debit", "ml": "Net Debit", "be": "Lower Strike + Debit"},
            {"name": "Cash-Secured Put","risk": "Moderate", "reward": "Limited",   "legs": "SELL PUT (OTM/ATM)",
             "desc": "Sell a put holding enough cash. Collect premium; buy shares at discount if assigned.",
             "mp": "Premium Received", "ml": "Strike − Premium", "be": "Strike − Premium"},
        ]
    },
    "bearish": {
        "label": "Bearish", "color": "#ff4560",
        "items": [
            {"name": "Long Put",        "risk": "Limited",  "reward": "High",      "legs": "BUY PUT (ATM)",
             "desc": "Buy a put option. Profits as stock falls. Risk limited to premium paid.",
             "mp": "Strike − Premium", "ml": "Premium Paid", "be": "Strike − Premium"},
            {"name": "Bear Put Spread", "risk": "Limited",  "reward": "Limited",   "legs": "BUY PUT (High) · SELL PUT (Low)",
             "desc": "Buy higher put, sell lower put. Cheaper bearish bet with capped profit.",
             "mp": "Spread − Debit", "ml": "Net Debit", "be": "Higher Strike − Debit"},
            {"name": "Bear Call Spread","risk": "Limited",  "reward": "Limited",   "legs": "SELL CALL (Low) · BUY CALL (High)",
             "desc": "Sell lower call, buy higher call. Credit received; profit if stock stays below lower strike.",
             "mp": "Net Credit", "ml": "Spread − Credit", "be": "Lower Strike + Credit"},
        ]
    },
    "neutral": {
        "label": "Neutral / Volatility", "color": "#f0b429",
        "items": [
            {"name": "Iron Condor",     "risk": "Limited",  "reward": "Limited",   "legs": "SELL OTM PUT+CALL SPREADS",
             "desc": "Sell OTM put spread + OTM call spread. Profit if stock stays in a defined range.",
             "mp": "Net Credit", "ml": "Spread − Credit", "be": "Short strikes ± Credit"},
            {"name": "Straddle",        "risk": "Limited",  "reward": "Unlimited", "legs": "BUY CALL + PUT (ATM)",
             "desc": "Buy ATM call and put. Profit from a large move in either direction.",
             "mp": "Unlimited (both sides)", "ml": "Total Premium", "be": "Strike ± Total Premium"},
            {"name": "Strangle",        "risk": "Limited",  "reward": "Unlimited", "legs": "BUY OTM CALL + OTM PUT",
             "desc": "Buy OTM call and OTM put. Cheaper than straddle; needs a bigger move to profit.",
             "mp": "Unlimited (both sides)", "ml": "Total Premium", "be": "Strikes ± Total Premium"},
            {"name": "Butterfly Spread","risk": "Limited",  "reward": "Limited",   "legs": "BUY Low · SELL 2×Mid · BUY High",
             "desc": "Three strike combo. Maximum profit when stock lands exactly at middle strike.",
             "mp": "Mid − Low − Debit", "ml": "Net Debit", "be": "Low+Debit and High−Debit"},
        ]
    }
}


# ═══════════════════════════════════════════════════════════════
#  SECTION 7 ── HTML GENERATOR
# ═══════════════════════════════════════════════════════════════

def _badge(text, color, bg):
    return (f'<span style="display:inline-block;padding:3px 10px;border-radius:20px;'
            f'font-size:11px;font-weight:700;letter-spacing:.5px;'
            f'color:{color};background:{bg};border:1px solid {color}40;">{text}</span>')

def _leg_pill(action, opt_type, strike):
    col = {'BUY': '#00d4a0', 'SELL': '#ff4560', 'OWN': '#3d9eff'}.get(action, '#aaa')
    bg  = {'BUY': 'rgba(0,212,160,.12)', 'SELL': 'rgba(255,69,96,.12)', 'OWN': 'rgba(61,158,255,.12)'}.get(action, 'rgba(170,170,170,.1)')
    return (f'<span style="display:inline-flex;align-items:center;gap:5px;padding:3px 10px;'
            f'border-radius:6px;font-size:11px;font-weight:600;color:{col};background:{bg};'
            f'border:1px solid {col}30;">{action} {opt_type} <small style="opacity:.7">{strike}</small></span>')

def generate_fii_dii_html(data, summ):
    badge_map = {
        'fii-bull':  ('#00e676', 'rgba(0,230,118,.12)', 'rgba(0,230,118,.3)'),
        'fii-cbull': ('#69f0ae', 'rgba(105,240,174,.10)', 'rgba(105,240,174,.28)'),
        'fii-neu':   ('#ffd740', 'rgba(255,215,64,.10)', 'rgba(255,215,64,.28)'),
        'fii-bear':  ('#ff5252', 'rgba(255,82,82,.10)', 'rgba(255,82,82,.28)'),
    }
    sc, sbg, sbdr = badge_map.get(summ['badge_cls'], badge_map['fii-neu'])
    is_fallback   = any(r.get('fallback') for r in data)
    date_range    = f"{data[0]['date']} – {data[-1]['date']}" if data else ''
    live_badge    = ('<span class="live-est">⚠ ESTIMATED</span>' if is_fallback
                     else '<span class="live-tag">◉ LIVE</span>')
    max_abs = summ['max_abs'] or 1

    cards = []
    for row in data:
        fii_v, dii_v = row['fii'], row['dii']
        net_v  = fii_v + dii_v
        fw     = round(min(100, abs(fii_v) / max_abs * 100), 1)
        dw     = round(min(100, abs(dii_v) / max_abs * 100), 1)
        fc     = '#00d4ff' if fii_v >= 0 else '#ff4444'
        dc     = '#ffb300' if dii_v >= 0 else '#ff4444'
        nc     = '#34d399' if net_v >= 0 else '#f87171'
        fb_g   = 'linear-gradient(90deg,#00d4ff,#0090ff)' if fii_v >= 0 else 'linear-gradient(90deg,#ff4444,#ff0055)'
        db_g   = 'linear-gradient(90deg,#ffb300,#ff8f00)' if dii_v >= 0 else 'linear-gradient(90deg,#ff4444,#ff0055)'
        bdr    = 'rgba(0,212,255,.18)' if net_v >= 0 else 'rgba(255,68,68,.18)'
        cards.append(f"""
        <div class="fd-card" style="border-color:{bdr};">
          <div class="fd-card-head"><span class="fd-date">{row['date']}</span><span class="fd-day">{row['day']}</span></div>
          <div class="fd-blk">
            <div class="fd-blk-hd"><span class="fd-lbl fii">FII</span><span class="fd-val" style="color:{fc};">{'+' if fii_v>=0 else ''}{fii_v:,.0f}</span></div>
            <div class="fd-track"><div class="fd-fill" style="width:{fw}%;background:{fb_g};"></div></div>
          </div>
          <div class="fd-div"></div>
          <div class="fd-blk">
            <div class="fd-blk-hd"><span class="fd-lbl dii">DII</span><span class="fd-val" style="color:{dc};">{'+' if dii_v>=0 else ''}{dii_v:,.0f}</span></div>
            <div class="fd-track"><div class="fd-fill" style="width:{dw}%;background:{db_g};"></div></div>
          </div>
          <div class="fd-net"><span class="fd-net-lbl">NET</span><span class="fd-net-val" style="color:{nc};">{'+' if net_v>=0 else ''}{net_v:,.0f}</span></div>
        </div>""")

    fa, da, na = summ['fii_avg'], summ['dii_avg'], summ['net_avg']
    fc2 = '#00d4ff' if fa >= 0 else '#ff4444'
    dc2 = '#ffb300' if da >= 0 else '#ff4444'
    nc2 = '#c084fc' if na >= 0 else '#f87171'

    return f"""
<div class="section">
  <div class="sec-title">🏦 FII / DII Institutional Flow {live_badge}
    <span class="sec-sub">Last 5 Trading Days · {date_range}</span>
  </div>
  <div class="fd-grid">{''.join(cards)}</div>
  <div class="fd-avg">
    <div class="fd-avg-cell"><div class="fd-avg-ey">FII 5D Avg</div><div class="fd-avg-val" style="color:{fc2};">{'+' if fa>=0 else ''}{fa:,.0f}</div><div class="fd-avg-u">₹ Cr / day</div></div>
    <div class="fd-avg-sep"></div>
    <div class="fd-avg-cell"><div class="fd-avg-ey">DII 5D Avg</div><div class="fd-avg-val" style="color:{dc2};">{'+' if da>=0 else ''}{da:,.0f}</div><div class="fd-avg-u">₹ Cr / day</div></div>
    <div class="fd-avg-sep"></div>
    <div class="fd-avg-cell"><div class="fd-avg-ey">Net Combined</div><div class="fd-avg-val" style="color:{nc2};">{'+' if na>=0 else ''}{na:,.0f}</div><div class="fd-avg-u">₹ Cr / day</div></div>
  </div>
  <div class="fd-insight" style="background:{sbg};border:1px solid {sbdr};">
    <div class="fd-ins-hd"><span style="color:{sc};font-size:11px;font-weight:700;letter-spacing:1px;">📊 5-DAY INSIGHT</span>
      <span style="color:{sc};background:{sbg};border:1px solid {sbdr};padding:3px 12px;border-radius:20px;font-size:12px;font-weight:800;">{summ['emoji']} {summ['label']}</span>
    </div>
    <div class="fd-ins-txt">{summ['insight']}</div>
  </div>
</div>"""


def generate_oi_section_html(oc):
    ce = oc['ce_chg']; pe = oc['pe_chg']; net = oc['net_chg']
    bull_force = (abs(pe) if pe > 0 else 0) + (abs(ce) if ce < 0 else 0)
    bear_force = (abs(ce) if ce > 0 else 0) + (abs(pe) if pe < 0 else 0)
    total_f    = bull_force + bear_force or 1
    bull_pct   = round(bull_force / total_f * 100)
    bear_pct   = 100 - bull_pct

    def card(lbl, val, is_bull_signal, sub):
        col   = '#34d399' if is_bull_signal else '#fb7185'
        sig   = 'Bullish Signal' if is_bull_signal else 'Bearish Signal'
        bg    = 'rgba(16,185,129,.1)' if is_bull_signal else 'rgba(239,68,68,.1)'
        bdr   = 'rgba(16,185,129,.4)' if is_bull_signal else 'rgba(239,68,68,.4)'
        return f"""
        <div class="oi-card">
          <div class="oi-card-lbl">{lbl}</div>
          <div class="oi-card-val" style="color:{col};">{val:+,}</div>
          <div class="oi-card-sub">{sub}</div>
          <div class="oi-card-sig" style="color:{col};background:{bg};border:1px solid {bdr};">{sig}</div>
        </div>"""

    ce_bull = ce < 0   # Call unwinding = bullish
    pe_bull = pe > 0   # Put build-up   = bullish
    net_bull = net > 0

    dir_col = '#34d399' if oc['oi_cls'] == 'bullish' else ('#fb7185' if oc['oi_cls'] == 'bearish' else '#fbbf24')
    dir_bg  = ('rgba(10,30,20,.9)' if oc['oi_cls'] == 'bullish' else
               'rgba(30,10,14,.9)' if oc['oi_cls'] == 'bearish' else 'rgba(20,20,10,.9)')
    dir_bdr = ('rgba(16,185,129,.35)' if oc['oi_cls'] == 'bullish' else
               'rgba(239,68,68,.35)' if oc['oi_cls'] == 'bearish' else 'rgba(251,191,36,.3)')

    return f"""
<div class="section">
  <div class="sec-title">📊 Change in Open Interest
    <span class="sec-sub">ATM ±10 strikes only · Expiry: {oc['expiry']}</span>
  </div>
  <div class="oi-dir-box" style="background:{dir_bg};border:1px solid {dir_bdr};">
    <div style="font-size:10px;letter-spacing:2px;color:rgba(148,163,184,.5);margin-bottom:6px;">MARKET DIRECTION</div>
    <div style="font-size:26px;font-weight:700;color:{dir_col};margin-bottom:4px;">{oc['oi_dir']}</div>
    <div style="font-size:13px;color:{dir_col}80;">{oc['oi_sig']}</div>
    <div style="margin-top:16px;">
      <div style="font-size:10px;color:rgba(148,163,184,.5);letter-spacing:2px;margin-bottom:5px;">🟢 BULL STRENGTH</div>
      <div style="height:8px;background:rgba(0,0,0,.4);border-radius:4px;overflow:hidden;width:100%;max-width:320px;">
        <div style="width:{bull_pct}%;height:100%;background:linear-gradient(90deg,#10b981,#34d399);border-radius:4px;"></div>
      </div>
      <div style="font-size:12px;color:#34d399;margin-top:3px;">{bull_pct}% Bull · {bear_pct}% Bear</div>
    </div>
  </div>
  <div class="oi-cards">
    {card('CALL OI CHANGE', ce, ce_bull, 'CE open interest Δ')}
    {card('PUT OI CHANGE',  pe, pe_bull, 'PE open interest Δ')}
    {card('NET OI CHANGE',  net, net_bull, 'PE Δ + CE Δ')}
  </div>
  <div class="oi-legend">
    <span>📖 <b>Call OI +</b> = Writers selling calls → <span style="color:#fb7185;">Bearish</span></span>
    <span><b>Call OI −</b> = Unwinding → <span style="color:#34d399;">Bullish</span></span>
    <span><b>Put OI +</b> = Writers selling puts → <span style="color:#34d399;">Bullish</span></span>
    <span><b>Put OI −</b> = Unwinding → <span style="color:#fb7185;">Bearish</span></span>
  </div>
</div>"""


def generate_key_levels_html(tech, oc):
    cp  = tech['price']
    ss  = tech['strong_sup']
    s1  = tech['support']
    r1  = tech['resistance']
    sr  = tech['strong_res']
    rng = sr - ss or 1
    def pct(v): return round(max(3, min(97, (v - ss) / rng * 100)), 1)
    cp_pct = pct(cp)
    pts_r  = int(r1 - cp)
    pts_s  = int(cp - s1)
    mp_html = ''
    if oc:
        mp_pct = pct(oc['max_pain'])
        mp_html = f"""
      <div class="kl-node" style="left:{mp_pct}%;top:0;transform:translateX(-50%);">
        <div class="kl-dot" style="background:#ffb74d;box-shadow:0 0 8px #ffb74d;margin:0 auto 4px;"></div>
        <div class="kl-lbl" style="color:#ffb74d;">Max Pain</div>
        <div class="kl-val" style="color:#ffb74d;">₹{oc['max_pain']:,}</div>
      </div>"""

    return f"""
<div class="section">
  <div class="sec-title">📍 Key Levels (1H Candles · ATM ±200pts)</div>
  <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
    <span style="font-size:11px;color:#26c6da;font-weight:700;">◄ SUPPORT ZONE</span>
    <span style="font-size:11px;color:#f44336;font-weight:700;">RESISTANCE ZONE ►</span>
  </div>
  <div style="position:relative;height:58px;">
    <div class="kl-node" style="left:3%;bottom:0;transform:translateX(-50%);">
      <div class="kl-lbl" style="color:#26c6da;">Strong Sup</div>
      <div class="kl-val" style="color:#26c6da;">₹{ss:,.0f}</div>
      <div class="kl-dot" style="background:#26c6da;margin:5px auto 0;"></div>
    </div>
    <div class="kl-node" style="left:22%;bottom:0;transform:translateX(-50%);">
      <div class="kl-lbl" style="color:#00bcd4;">Support</div>
      <div class="kl-val" style="color:#00bcd4;">₹{s1:,.0f}</div>
      <div class="kl-dot" style="background:#00bcd4;box-shadow:0 0 8px #00bcd4;margin:5px auto 0;"></div>
    </div>
    <div style="position:absolute;left:{cp_pct}%;bottom:6px;transform:translateX(-50%);background:#4fc3f7;color:#000;font-size:11px;font-weight:700;padding:3px 12px;border-radius:6px;white-space:nowrap;box-shadow:0 0 14px rgba(79,195,247,.7);z-index:10;">▼ NOW ₹{cp:,.0f}</div>
    <div class="kl-node" style="left:75%;bottom:0;transform:translateX(-50%);">
      <div class="kl-lbl" style="color:#ff7043;">Resistance</div>
      <div class="kl-val" style="color:#ff7043;">₹{r1:,.0f}</div>
      <div class="kl-dot" style="background:#ff7043;box-shadow:0 0 8px #ff7043;margin:5px auto 0;"></div>
    </div>
    <div class="kl-node" style="left:95%;bottom:0;transform:translateX(-50%);">
      <div class="kl-lbl" style="color:#f44336;">Strong Res</div>
      <div class="kl-val" style="color:#f44336;">₹{sr:,.0f}</div>
      <div class="kl-dot" style="background:#f44336;margin:5px auto 0;"></div>
    </div>
  </div>
  <div style="position:relative;height:8px;border-radius:4px;background:linear-gradient(90deg,#26c6da 0%,#00bcd4 20%,#4fc3f7 40%,#ffb74d 58%,#ff7043 76%,#f44336 100%);">
    <div style="position:absolute;left:{cp_pct}%;top:50%;transform:translate(-50%,-50%);width:4px;height:20px;background:#fff;border-radius:2px;box-shadow:0 0 14px #fff;z-index:10;"></div>
  </div>
  <div style="position:relative;height:54px;">{mp_html}</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:4px;">
    <div class="kl-dist-box" style="border-color:rgba(244,67,54,.25);">
      <span style="color:#b0bec5;">📍 To Resistance</span><span style="color:#f44336;font-weight:700;">+{pts_r:,} pts</span>
    </div>
    <div class="kl-dist-box" style="border-color:rgba(0,188,212,.25);">
      <span style="color:#b0bec5;">📍 To Support</span><span style="color:#00bcd4;font-weight:700;">−{pts_s:,} pts</span>
    </div>
  </div>
</div>"""


def generate_market_direction_html(md, tech, oc):
    bias = md['bias']
    if   bias == 'BULLISH':  grad = 'linear-gradient(135deg,#00d4a0,#2ecc8a)'; arrow = '▲'; tagcol = '#00d4a0'
    elif bias == 'BEARISH':  grad = 'linear-gradient(135deg,#ff4560,#cc1133)'; arrow = '▼'; tagcol = '#ff4560'
    else:                     grad = 'linear-gradient(135deg,#f0b429,#f7931e)'; arrow = '↔'; tagcol = '#f0b429'

    conf_col = '#00d4a0' if md['confidence'] == 'HIGH' else '#f0b429'
    bull_w   = round(md['bull'] / (md['bull'] + md['bear']) * 100) if (md['bull'] + md['bear']) > 0 else 50
    bear_w   = 100 - bull_w

    rsi      = tech['rsi'] if tech else 0
    macd_bull = tech['macd'] > tech['signal_line'] if tech else False
    pcr       = oc['pcr_oi'] if oc else 0
    pcr_lbl   = '🟢 Bullish' if pcr > 1.2 else ('🔴 Bearish' if pcr < 0.7 else '🟡 Neutral')
    rsi_lbl   = '🔴 Overbought' if rsi > 70 else ('🟢 Oversold' if rsi < 30 else '🟡 Neutral')
    macd_lbl  = '🟢 Bullish' if macd_bull else '🔴 Bearish'

    sma_rows = ''
    if tech:
        cp = tech['price']
        for k, label in [('sma20', 'SMA 20'), ('sma50', 'SMA 50'), ('sma200', 'SMA 200')]:
            above = cp > tech[k]
            c   = '#34d399' if above else '#fb7185'
            sma_rows += (f'<div class="md-row"><span class="md-lbl">{label}</span>'
                         f'<span style="color:{c};font-weight:700;">{"▲ Above" if above else "▼ Below"} '
                         f'₹{tech[k]:,.0f}</span></div>')

    return f"""
<div class="section md-section">
  <div class="sec-title">🎯 Market Direction (Algorithmic)</div>
  <div class="md-widget">
    <div class="md-glow"></div>
    <div class="md-top-row">
      <div class="md-live-dot"></div>
      <span class="md-label">NSE NIFTY · LIVE ANALYSIS</span>
      <span class="md-pill" style="color:{conf_col};border-color:{conf_col}40;background:{conf_col}15;">{md['confidence']} CONFIDENCE</span>
    </div>
    <div class="md-main">
      <div class="md-arrow" style="background:{grad};-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">{arrow}</div>
      <div class="md-bias-text" style="background:{grad};-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">{bias}</div>
    </div>
    <div class="md-score-row">
      <span class="md-score-pill bull">🟢 Bull Score: {md['bull']}</span>
      <span class="md-score-pill bear">🔴 Bear Score: {md['bear']}</span>
      <span class="md-score-pill" style="color:#aaa;border-color:#aaa30;background:#aaa10;">Diff: {md['diff']:+d}</span>
    </div>
    <div class="md-meter-wrap">
      <div class="md-meter-row">
        <span class="md-meter-lbl" style="color:#34d399;">🟢 Bull</span>
        <div class="md-meter-track"><div style="width:{bull_w}%;height:100%;background:linear-gradient(90deg,#10b981,#34d399);border-radius:4px;"></div></div>
        <span style="color:#34d399;font-weight:700;">{bull_w}%</span>
      </div>
      <div class="md-meter-row">
        <span class="md-meter-lbl" style="color:#fb7185;">🔴 Bear</span>
        <div class="md-meter-track"><div style="width:{bear_w}%;height:100%;background:linear-gradient(90deg,#ef4444,#fb7185);border-radius:4px;"></div></div>
        <span style="color:#fb7185;font-weight:700;">{bear_w}%</span>
      </div>
    </div>
  </div>

  <div class="md-signals">
    <div class="md-sig-title">Signal Breakdown</div>
    {sma_rows}
    <div class="md-row"><span class="md-lbl">RSI (14)</span><span>{rsi_lbl} ({rsi:.1f})</span></div>
    <div class="md-row"><span class="md-lbl">MACD</span><span>{macd_lbl}</span></div>
    <div class="md-row"><span class="md-lbl">PCR (OI)</span><span>{pcr_lbl} ({pcr:.3f})</span></div>
  </div>

  <div class="md-logic">
    <div class="md-logic-title">📖 Scoring Logic</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:5px 16px;font-size:12px;color:rgba(176,190,197,.7);">
      <div><span style="color:#34d399;">BULLISH</span> when Diff ≥ +3</div>
      <div><span style="color:#fb7185;">BEARISH</span> when Diff ≤ −3</div>
      <div><span style="color:#fbbf24;">SIDEWAYS</span> Diff −2 to +2</div>
      <div><span style="color:#60a5fa;">HIGH</span> confidence when gap ≥ 4</div>
    </div>
  </div>
</div>"""


def generate_strategies_html(md, tech, oc):
    bias = md['bias']
    atm  = oc['atm_strike'] if oc else (round(tech['price'] / 50) * 50 if tech else 25000)
    oi_dir = oc['oi_dir'] if oc else 'Neutral'

    tech_strats, oi_strat = recommend_strategies(bias, atm, oi_dir)

    # Live recommended strategies banner
    bias_col   = '#00d4a0' if bias == 'BULLISH' else ('#ff4560' if bias == 'BEARISH' else '#f0b429')
    rec_html = f"""
<div class="rec-banner" style="border-color:{bias_col}40;background:{bias_col}08;">
  <div class="rec-banner-title" style="color:{bias_col};">
    {'🟢' if bias=='BULLISH' else '🔴' if bias=='BEARISH' else '⚪'} TODAY'S RECOMMENDED STRATEGIES — {bias}
  </div>
  <div class="rec-grid">"""

    for s in tech_strats:
        rec_html += f"""
    <div class="rec-card" style="border-color:{bias_col}30;">
      <div class="rec-name">{s['name']}</div>
      <div class="rec-legs">{s['legs']}</div>
      <div style="display:flex;gap:6px;margin-top:8px;">
        <span class="rec-tag">{s['type']}</span>
        <span class="rec-tag">{s['risk']} Risk</span>
      </div>
    </div>"""

    rec_html += f"""
  </div>
  <div class="rec-oi-box">
    <span class="rec-oi-lbl">🎯 OI Signal Strategy:</span>
    <span class="rec-oi-name">{oi_strat['name']}</span>
    <span class="rec-oi-legs">{oi_strat['legs']}</span>
    <span class="rec-oi-sig">{oi_strat['signal']}</span>
  </div>
</div>"""

    # Strategy directory
    dir_html = '<div class="strat-dir">'
    for direction, info in ALL_STRATEGIES.items():
        col = info['color']
        dir_html += f"""
  <div class="strat-group">
    <div class="strat-group-title" style="color:{col};">
      {'🟢' if direction=='bullish' else '🔴' if direction=='bearish' else '⚪'} {info['label']} Strategies
    </div>"""
        for s in info['items']:
            r_col = '#34d399' if s['risk'] == 'Limited' else '#fb7185' if s['risk'] == 'High' else '#fbbf24'
            rw_col = '#34d399' if s['reward'] == 'Unlimited' else '#fbbf24'
            dir_html += f"""
    <div class="strat-card">
      <div class="strat-card-top">
        <div class="strat-name">{s['name']}</div>
        <div style="display:flex;gap:6px;flex-wrap:wrap;">
          <span style="font-size:10px;padding:2px 8px;border-radius:4px;color:{r_col};background:{r_col}15;border:1px solid {r_col}30;">Risk: {s['risk']}</span>
          <span style="font-size:10px;padding:2px 8px;border-radius:4px;color:{rw_col};background:{rw_col}15;border:1px solid {rw_col}30;">Reward: {s['reward']}</span>
        </div>
      </div>
      <div class="strat-desc">{s['desc']}</div>
      <div class="strat-legs-row">{s['legs']}</div>
      <div class="strat-metrics">
        <div><span class="sm-lbl">Max Profit</span><span style="color:#34d399;">{s['mp']}</span></div>
        <div><span class="sm-lbl">Max Loss</span><span style="color:#fb7185;">{s['ml']}</span></div>
        <div><span class="sm-lbl">Breakeven</span><span style="color:#fbbf24;">{s['be']}</span></div>
      </div>
    </div>"""
        dir_html += '</div>'
    dir_html += '</div>'

    return f"""
<div class="section">
  <div class="sec-title">💡 Strategy Recommendations & Builder</div>
  {rec_html}
  <div style="margin-top:24px;">
    <div class="sec-title" style="border:none;padding:0;margin-bottom:14px;font-size:13px;">📚 All Strategies Reference</div>
    {dir_html}
  </div>
</div>"""


def generate_html(tech, oc, md, fii_data, fii_summ, ts):
    """Master HTML assembler"""
    cp      = tech['price'] if tech else 0
    expiry  = oc['expiry'] if oc else 'N/A'
    atm     = oc['atm_strike'] if oc else (round(cp / 50) * 50 if cp else 0)
    pcr     = oc['pcr_oi'] if oc else 0
    mp      = oc['max_pain'] if oc else 0

    bias = md['bias']
    bias_col = '#00d4a0' if bias == 'BULLISH' else ('#ff4560' if bias == 'BEARISH' else '#f0b429')
    bias_emoji = '🟢' if bias == 'BULLISH' else ('🔴' if bias == 'BEARISH' else '⚪')

    oi_html     = generate_oi_section_html(oc) if oc else ''
    kl_html     = generate_key_levels_html(tech, oc) if tech else ''
    fii_html    = generate_fii_dii_html(fii_data, fii_summ)
    md_html     = generate_market_direction_html(md, tech, oc)
    strat_html  = generate_strategies_html(md, tech, oc)

    # Top 5 strikes table
    strikes_html = ''
    if oc and (oc['top_ce'] or oc['top_pe']):
        def strike_rows_ce(rows):
            out = ''
            for i, r in enumerate(rows, 1):
                out += f"<tr><td>{i}</td><td><b>₹{int(r['Strike']):,}</b></td><td>{int(r['CE_OI']):,}</td><td style='color:#00bcd4;font-weight:700;'>₹{r['CE_LTP']:.2f}</td></tr>"
            return out
        def strike_rows_pe(rows):
            out = ''
            for i, r in enumerate(rows, 1):
                out += f"<tr><td>{i}</td><td><b>₹{int(r['Strike']):,}</b></td><td>{int(r['PE_OI']):,}</td><td style='color:#f44336;font-weight:700;'>₹{r['PE_LTP']:.2f}</td></tr>"
            return out
        strikes_html = f"""
<div class="section">
  <div class="sec-title">📋 Top 5 Strikes by Open Interest <span class="sec-sub">ATM ±10 only</span></div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;">
    <div>
      <div style="color:#00bcd4;font-weight:700;margin-bottom:10px;font-size:14px;">📞 CALL Options (CE)</div>
      <table class="s-table"><thead><tr><th>#</th><th>Strike</th><th>OI</th><th>LTP</th></tr></thead>
      <tbody>{strike_rows_ce(oc['top_ce'])}</tbody></table>
    </div>
    <div>
      <div style="color:#f44336;font-weight:700;margin-bottom:10px;font-size:14px;">📉 PUT Options (PE)</div>
      <table class="s-table"><thead><tr><th>#</th><th>Strike</th><th>OI</th><th>LTP</th></tr></thead>
      <tbody>{strike_rows_pe(oc['top_pe'])}</tbody></table>
    </div>
  </div>
</div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Nifty 50 Options Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500;600&family=Rajdhani:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0;}}
:root{{
  --bg:#080c10;--surf:#0e1318;--card:#111820;--bdr:#1a2530;--bdr2:#243040;
  --text:#c8d8e0;--muted:#5a7080;
  --bull:#00d4a0;--bear:#ff4560;--neut:#f0b429;--accent:#3d9eff;
  --ff-head:'Syne',sans-serif;--ff-mono:'JetBrains Mono',monospace;--ff-body:'Rajdhani',sans-serif;
}}
html{{scroll-behavior:smooth;}}
body{{background:var(--bg);color:var(--text);font-family:var(--ff-body);font-size:13px;line-height:1.6;min-height:100vh;}}
body::before{{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(255,255,255,.012) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.012) 1px,transparent 1px);background-size:40px 40px;pointer-events:none;z-index:0;}}

/* ── Layout ── */
.app{{position:relative;z-index:1;display:grid;grid-template-rows:auto 1fr auto;min-height:100vh;}}

/* ── Header ── */
header{{
  display:flex;align-items:center;justify-content:space-between;
  padding:14px 32px;border-bottom:1px solid var(--bdr);
  background:rgba(8,12,16,.97);backdrop-filter:blur(16px);
  position:sticky;top:0;z-index:200;
}}
.logo{{font-family:var(--ff-head);font-size:18px;font-weight:800;color:#fff;letter-spacing:-0.5px;}}
.logo span{{color:var(--bull);}}
.hdr-meta{{display:flex;align-items:center;gap:14px;font-size:11px;color:var(--muted);font-family:var(--ff-mono);}}
.live-dot{{width:7px;height:7px;border-radius:50%;background:var(--bull);box-shadow:0 0 8px var(--bull);animation:pulse 2s infinite;}}
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.3}}}}

/* ── Hero Market Direction Banner ── */
.hero-banner{{
  padding:24px 32px;
  background:linear-gradient(135deg,rgba(14,19,24,.9),rgba(8,12,16,.95));
  border-bottom:1px solid var(--bdr);
  display:flex;align-items:center;justify-content:space-between;gap:20px;flex-wrap:wrap;
}}
.hero-left{{display:flex;align-items:center;gap:20px;}}
.hero-dir{{font-family:var(--ff-head);font-size:42px;font-weight:800;line-height:1;letter-spacing:-1px;}}
.hero-sub{{font-size:13px;color:var(--muted);margin-top:4px;}}
.hero-price{{font-family:var(--ff-mono);font-size:28px;font-weight:600;color:#fff;}}
.hero-right{{display:flex;gap:20px;align-items:center;flex-wrap:wrap;}}
.hero-stat{{text-align:center;}}
.hero-stat-lbl{{font-size:10px;color:var(--muted);letter-spacing:1.5px;text-transform:uppercase;margin-bottom:3px;}}
.hero-stat-val{{font-family:var(--ff-mono);font-size:16px;font-weight:600;}}

/* ── Main Grid ── */
.main{{display:grid;grid-template-columns:300px 1fr;min-height:0;}}

/* ── Sidebar ── */
.sidebar{{
  border-right:1px solid var(--bdr);background:var(--surf);
  position:sticky;top:57px;height:calc(100vh - 57px);overflow-y:auto;
}}
.sidebar::-webkit-scrollbar{{width:3px;}}
.sidebar::-webkit-scrollbar-thumb{{background:var(--bdr2);border-radius:2px;}}

.sb-sec{{padding:16px 12px 8px;}}
.sb-label{{font-size:10px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-bottom:8px;padding:0 4px;}}
.sb-nav-btn{{
  display:flex;align-items:center;gap:8px;width:100%;padding:9px 12px;
  border-radius:8px;border:1px solid transparent;
  cursor:pointer;background:transparent;color:var(--muted);
  font-family:var(--ff-mono);font-size:12px;text-align:left;transition:all .15s;
}}
.sb-nav-btn:hover{{background:var(--card);color:var(--text);}}
.sb-nav-btn.active{{background:var(--card);border-color:var(--bdr2);color:var(--text);}}
.sb-badge-bull{{color:var(--bull);font-size:10px;margin-left:auto;font-weight:700;}}
.sb-badge-bear{{color:var(--bear);font-size:10px;margin-left:auto;font-weight:700;}}
.sb-badge-neut{{color:var(--neut);font-size:10px;margin-left:auto;font-weight:700;}}

/* ── Content ── */
.content{{overflow-y:auto;padding:0;}}

/* ── Section ── */
.section{{padding:24px 28px;border-bottom:1px solid var(--bdr);}}
.sec-title{{
  font-family:var(--ff-head);font-size:13px;font-weight:700;letter-spacing:2px;
  color:var(--accent);text-transform:uppercase;
  display:flex;align-items:center;gap:10px;flex-wrap:wrap;
  margin-bottom:18px;padding-bottom:12px;border-bottom:1px solid var(--bdr);
}}
.sec-sub{{font-size:11px;color:var(--muted);font-weight:400;letter-spacing:.5px;text-transform:none;margin-left:auto;}}

/* ── Live / Estimated badge ── */
.live-tag{{font-size:10px;font-weight:700;padding:2px 9px;border-radius:10px;background:rgba(0,230,118,.1);color:#00e676;border:1px solid rgba(0,230,118,.3);}}
.live-est{{font-size:10px;font-weight:700;padding:2px 9px;border-radius:10px;background:rgba(255,138,101,.1);color:#ff8a65;border:1px solid rgba(255,138,101,.3);}}

/* ── FII/DII ── */
.fd-grid{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:16px;}}
.fd-card{{background:rgba(255,255,255,.03);border:1px solid rgba(0,212,255,.15);border-radius:14px;padding:14px 12px 12px;display:flex;flex-direction:column;gap:10px;transition:all .2s;}}
.fd-card:hover{{transform:translateY(-2px);box-shadow:0 8px 24px rgba(0,0,0,.3);}}
.fd-card-head{{display:flex;justify-content:space-between;align-items:baseline;}}
.fd-date{{font-family:var(--ff-head);font-size:12px;font-weight:700;color:#e0f7fa;}}
.fd-day{{font-size:9px;letter-spacing:1.5px;color:rgba(128,222,234,.3);text-transform:uppercase;}}
.fd-blk{{display:flex;flex-direction:column;gap:4px;}}
.fd-blk-hd{{display:flex;justify-content:space-between;align-items:baseline;}}
.fd-lbl{{font-size:8px;font-weight:700;letter-spacing:2px;text-transform:uppercase;}}
.fd-lbl.fii{{color:rgba(0,212,255,.5);}}
.fd-lbl.dii{{color:rgba(255,179,0,.5);}}
.fd-val{{font-family:var(--ff-mono);font-size:14px;font-weight:700;}}
.fd-track{{height:4px;background:rgba(0,0,0,.35);border-radius:2px;overflow:hidden;}}
.fd-fill{{height:100%;border-radius:2px;}}
.fd-div{{height:1px;background:rgba(255,255,255,.04);}}
.fd-net{{display:flex;justify-content:space-between;align-items:baseline;padding-top:7px;border-top:1px solid rgba(255,255,255,.05);margin-top:auto;}}
.fd-net-lbl{{font-size:8px;letter-spacing:2px;color:rgba(255,255,255,.2);text-transform:uppercase;font-weight:700;}}
.fd-net-val{{font-family:var(--ff-mono);font-size:12px;font-weight:700;}}
.fd-avg{{display:grid;grid-template-columns:1fr auto 1fr auto 1fr;align-items:center;background:rgba(6,13,20,.8);border:1px solid rgba(79,195,247,.1);border-radius:12px;padding:16px 20px;margin-bottom:14px;}}
.fd-avg-cell{{text-align:center;}}
.fd-avg-ey{{font-size:8px;letter-spacing:2px;color:rgba(0,229,255,.4);text-transform:uppercase;margin-bottom:5px;font-weight:700;}}
.fd-avg-val{{font-family:var(--ff-head);font-size:22px;font-weight:800;line-height:1;}}
.fd-avg-u{{font-size:9px;color:#37474f;margin-top:3px;}}
.fd-avg-sep{{width:1px;height:44px;background:linear-gradient(180deg,transparent,rgba(79,195,247,.2),transparent);margin:0 12px;}}
.fd-insight{{border-radius:10px;padding:14px 16px;}}
.fd-ins-hd{{display:flex;align-items:center;gap:10px;margin-bottom:8px;flex-wrap:wrap;}}
.fd-ins-txt{{font-size:13px;color:#cfd8dc;line-height:1.85;font-weight:500;}}

/* ── OI Section ── */
.oi-dir-box{{border-radius:12px;padding:18px 20px;margin-bottom:16px;}}
.oi-cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:12px;}}
.oi-card{{background:rgba(20,28,45,.9);border:1px solid rgba(79,195,247,.12);border-radius:12px;padding:16px;}}
.oi-card-lbl{{font-size:9px;letter-spacing:2px;color:rgba(148,163,184,.5);text-transform:uppercase;margin-bottom:8px;}}
.oi-card-val{{font-family:var(--ff-head);font-size:26px;font-weight:700;margin-bottom:4px;}}
.oi-card-sub{{font-size:10px;color:rgba(100,116,139,.7);margin-bottom:12px;}}
.oi-card-sig{{display:block;padding:7px 12px;border-radius:6px;text-align:center;font-size:12px;font-weight:700;}}
.oi-legend{{display:flex;flex-wrap:wrap;gap:10px 20px;font-size:11px;color:rgba(176,190,197,.6);padding:10px 0;}}

/* ── Key Levels ── */
.kl-node{{position:absolute;text-align:center;}}
.kl-lbl{{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;line-height:1.3;white-space:nowrap;}}
.kl-val{{font-size:12px;font-weight:700;color:#fff;white-space:nowrap;margin-top:2px;}}
.kl-dot{{width:11px;height:11px;border-radius:50%;border:2px solid rgba(8,12,16,.9);}}
.kl-dist-box{{background:rgba(255,255,255,.03);border:1px solid;border-radius:8px;padding:9px 14px;display:flex;justify-content:space-between;align-items:center;}}

/* ── Market Direction Widget ── */
.md-section{{background:linear-gradient(135deg,rgba(14,19,24,.95),rgba(8,12,16,1));}}
.md-widget{{
  position:relative;overflow:hidden;background:rgba(255,255,255,.04);
  border:1px solid rgba(255,255,255,.08);border-radius:16px;
  padding:20px 24px;backdrop-filter:blur(20px);margin-bottom:16px;
}}
.md-glow{{position:absolute;top:-80%;left:-80%;width:260%;height:260%;background:conic-gradient(from 180deg,#ff6b35 0deg,#ffcd3c 120deg,#4ecdc4 240deg,#ff6b35 360deg);opacity:.05;animation:md-rot 10s linear infinite;border-radius:50%;pointer-events:none;}}
@keyframes md-rot{{to{{transform:rotate(360deg)}}}}
.md-top-row{{display:flex;align-items:center;gap:10px;position:relative;z-index:1;margin-bottom:14px;flex-wrap:wrap;}}
.md-live-dot{{width:6px;height:6px;border-radius:50%;background:#4ecdc4;box-shadow:0 0 8px #4ecdc4;animation:pulse 2s infinite;flex-shrink:0;}}
.md-label{{font-family:var(--ff-mono);font-size:9px;letter-spacing:3px;color:rgba(255,255,255,.3);text-transform:uppercase;}}
.md-pill{{font-family:var(--ff-mono);font-size:10px;font-weight:700;padding:3px 12px;border-radius:20px;letter-spacing:1px;border:1px solid;margin-left:auto;}}
.md-main{{display:flex;align-items:center;gap:16px;position:relative;z-index:1;margin-bottom:14px;}}
.md-arrow{{font-size:48px;font-weight:900;line-height:1;}}
.md-bias-text{{font-family:var(--ff-head);font-size:40px;font-weight:800;letter-spacing:-1px;line-height:1;}}
.md-score-row{{display:flex;gap:8px;flex-wrap:wrap;position:relative;z-index:1;margin-bottom:14px;}}
.md-score-pill{{font-family:var(--ff-mono);font-size:11px;font-weight:700;padding:4px 12px;border-radius:20px;border:1px solid;}}
.md-score-pill.bull{{color:#34d399;border-color:#34d39940;background:#34d39910;}}
.md-score-pill.bear{{color:#fb7185;border-color:#fb718540;background:#fb718510;}}
.md-meter-wrap{{display:flex;flex-direction:column;gap:10px;position:relative;z-index:1;}}
.md-meter-row{{display:flex;align-items:center;gap:10px;}}
.md-meter-lbl{{font-size:11px;font-weight:600;min-width:70px;}}
.md-meter-track{{flex:1;height:8px;background:rgba(0,0,0,.4);border-radius:4px;overflow:hidden;}}
.md-signals{{background:rgba(255,255,255,.03);border:1px solid var(--bdr);border-radius:12px;padding:14px 16px;margin-bottom:12px;}}
.md-sig-title{{font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:var(--muted);margin-bottom:10px;font-weight:600;}}
.md-row{{display:flex;justify-content:space-between;align-items:center;padding:5px 0;border-bottom:1px solid rgba(255,255,255,.04);font-size:12px;}}
.md-row:last-child{{border-bottom:none;}}
.md-lbl{{color:var(--muted);}}
.md-logic{{background:rgba(79,195,247,.04);border:1px solid rgba(79,195,247,.12);border-left:3px solid var(--accent);border-radius:8px;padding:10px 14px;}}
.md-logic-title{{font-size:10px;font-weight:700;color:var(--accent);letter-spacing:1.5px;margin-bottom:7px;}}

/* ── Strategy Recommendations ── */
.rec-banner{{border:1px solid;border-radius:14px;padding:18px 20px;margin-bottom:20px;}}
.rec-banner-title{{font-family:var(--ff-head);font-size:15px;font-weight:700;margin-bottom:14px;}}
.rec-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:14px;}}
.rec-card{{background:rgba(255,255,255,.04);border:1px solid;border-radius:10px;padding:14px;}}
.rec-name{{font-family:var(--ff-head);font-size:14px;font-weight:700;color:#fff;margin-bottom:5px;}}
.rec-legs{{font-family:var(--ff-mono);font-size:11px;color:var(--muted);}}
.rec-tag{{font-size:10px;padding:2px 8px;border-radius:4px;background:rgba(255,255,255,.06);color:var(--muted);border:1px solid var(--bdr2);}}
.rec-oi-box{{background:rgba(255,255,255,.03);border:1px solid var(--bdr);border-radius:8px;padding:12px 14px;display:flex;align-items:center;gap:10px;flex-wrap:wrap;font-size:12px;}}
.rec-oi-lbl{{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:1px;}}
.rec-oi-name{{font-weight:700;color:#fff;}}
.rec-oi-legs{{font-family:var(--ff-mono);font-size:11px;color:var(--muted);}}
.rec-oi-sig{{margin-left:auto;font-style:italic;color:var(--muted);}}

/* ── Strategy Directory ── */
.strat-dir{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;}}
.strat-group{{display:flex;flex-direction:column;gap:10px;}}
.strat-group-title{{font-family:var(--ff-head);font-size:13px;font-weight:700;margin-bottom:4px;}}
.strat-card{{background:var(--card);border:1px solid var(--bdr);border-radius:10px;padding:14px;transition:all .2s;}}
.strat-card:hover{{border-color:var(--bdr2);transform:translateY(-2px);box-shadow:0 6px 24px rgba(0,0,0,.3);}}
.strat-card-top{{display:flex;justify-content:space-between;align-items:flex-start;gap:8px;margin-bottom:8px;}}
.strat-name{{font-family:var(--ff-head);font-size:14px;font-weight:700;color:#fff;}}
.strat-desc{{font-size:11px;color:var(--muted);line-height:1.7;margin-bottom:8px;}}
.strat-legs-row{{font-family:var(--ff-mono);font-size:10px;color:rgba(61,158,255,.8);margin-bottom:10px;letter-spacing:.5px;}}
.strat-metrics{{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;}}
.strat-metrics>div{{background:rgba(255,255,255,.03);border:1px solid var(--bdr);border-radius:6px;padding:6px 8px;}}
.sm-lbl{{display:block;font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:2px;}}

/* ── Strikes table ── */
.s-table{{width:100%;border-collapse:collapse;border-radius:8px;overflow:hidden;}}
.s-table th{{background:linear-gradient(135deg,#4fc3f7,#26c6da);color:#000;padding:10px;font-size:11px;font-weight:700;text-align:left;}}
.s-table td{{padding:10px;border-bottom:1px solid rgba(79,195,247,.06);font-size:12px;color:var(--muted);}}
.s-table tr:hover{{background:rgba(79,195,247,.04);}}

/* ── Footer ── */
footer{{padding:16px 32px;border-top:1px solid var(--bdr);background:var(--surf);display:flex;justify-content:space-between;font-size:11px;color:var(--muted);font-family:var(--ff-mono);}}

/* ── Responsive ── */
@media(max-width:1024px){{
  .main{{grid-template-columns:1fr;}}
  .sidebar{{position:static;height:auto;border-right:none;border-bottom:1px solid var(--bdr);}}
  .hero-dir{{font-size:30px;}}
  .fd-grid{{grid-template-columns:repeat(3,1fr);}}
  .oi-cards{{grid-template-columns:1fr;}}
  .strat-dir{{grid-template-columns:1fr;}}
  .rec-grid{{grid-template-columns:1fr;}}
}}
@media(max-width:640px){{
  header{{padding:12px 16px;}}
  .hero-banner{{padding:16px;flex-direction:column;}}
  .section{{padding:16px;}}
  .fd-grid{{grid-template-columns:repeat(2,1fr);}}
  .fd-avg{{grid-template-columns:1fr;gap:10px;padding:14px;}}
  .fd-avg-sep{{display:none;}}
  .md-bias-text{{font-size:28px;}}
  div[style*="grid-template-columns:1fr 1fr"]{{grid-template-columns:1fr!important;}}
  .strat-metrics{{grid-template-columns:1fr;}}
  footer{{flex-direction:column;gap:6px;}}
}}
</style>
</head>
<body>
<div class="app">

<!-- ── Header ── -->
<header>
  <div class="logo">NIFTY<span>CRAFT</span></div>
  <div class="hdr-meta">
    <div class="live-dot"></div>
    <span>NSE Options Dashboard</span>
    <span style="color:var(--bdr2)">|</span>
    <span>{ts}</span>
  </div>
</header>

<!-- ── Hero Banner ── -->
<div class="hero-banner">
  <div class="hero-left">
    <div>
      <div class="hero-dir" style="background:{'linear-gradient(135deg,#00d4a0,#2ecc8a)' if bias=='BULLISH' else 'linear-gradient(135deg,#ff4560,#cc1133)' if bias=='BEARISH' else 'linear-gradient(135deg,#f0b429,#f7931e)'};-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
        {bias_emoji} {bias}
      </div>
      <div class="hero-sub">Market Direction · {md['confidence']} Confidence · Score {md['bull']}B vs {md['bear']}R</div>
    </div>
  </div>
  <div class="hero-right">
    <div class="hero-stat">
      <div class="hero-stat-lbl">NIFTY Spot</div>
      <div class="hero-stat-val" style="color:#fff;">₹{cp:,.2f}</div>
    </div>
    <div class="hero-stat">
      <div class="hero-stat-lbl">ATM Strike</div>
      <div class="hero-stat-val" style="color:var(--accent);">₹{atm:,}</div>
    </div>
    <div class="hero-stat">
      <div class="hero-stat-lbl">Expiry</div>
      <div class="hero-stat-val" style="color:var(--neut);">{expiry}</div>
    </div>
    <div class="hero-stat">
      <div class="hero-stat-lbl">PCR (OI)</div>
      <div class="hero-stat-val" style="color:{'var(--bull)' if pcr>1.2 else 'var(--bear)' if pcr<0.7 else 'var(--neut)'};">{pcr:.3f}</div>
    </div>
    <div class="hero-stat">
      <div class="hero-stat-lbl">Max Pain</div>
      <div class="hero-stat-val" style="color:var(--neut);">₹{mp:,}</div>
    </div>
  </div>
</div>

<!-- ── Main Grid ── -->
<div class="main">

  <!-- Sidebar -->
  <aside class="sidebar">
    <div class="sb-sec">
      <div class="sb-label">Live Analysis</div>
      <button class="sb-nav-btn active" onclick="scrollTo('md')">🎯 Market Direction</button>
      <button class="sb-nav-btn" onclick="scrollTo('oi')">📊 OI Change</button>
      <button class="sb-nav-btn" onclick="scrollTo('kl')">📍 Key Levels</button>
      <button class="sb-nav-btn" onclick="scrollTo('fd')">🏦 FII / DII Flow</button>
    </div>
    <div class="sb-sec">
      <div class="sb-label">Strategies</div>
      <button class="sb-nav-btn" onclick="scrollTo('strat')">💡 Recommendations</button>
      <button class="sb-nav-btn" onclick="scrollTo('ref-bull')">🟢 Bullish <span class="sb-badge-bull">4</span></button>
      <button class="sb-nav-btn" onclick="scrollTo('ref-bear')">🔴 Bearish <span class="sb-badge-bear">3</span></button>
      <button class="sb-nav-btn" onclick="scrollTo('ref-neut')">⚪ Neutral <span class="sb-badge-neut">4</span></button>
    </div>
    <div class="sb-sec">
      <div class="sb-label">Today's Signal</div>
      <div style="padding:10px 12px;background:var(--card);border:1px solid {bias_col}40;border-radius:10px;text-align:center;">
        <div style="font-size:24px;margin-bottom:4px;">{bias_emoji}</div>
        <div style="font-family:var(--ff-head);font-size:15px;font-weight:700;color:{bias_col};">{bias}</div>
        <div style="font-size:10px;color:var(--muted);margin-top:3px;">{md['confidence']} CONFIDENCE</div>
        <div style="margin-top:8px;font-size:11px;color:var(--muted);">Bull {md['bull']} vs Bear {md['bear']}</div>
      </div>
    </div>
  </aside>

  <!-- Content -->
  <main class="content">
    <div id="md">{md_html}</div>
    <div id="oi">{oi_html}</div>
    <div id="kl">{kl_html}</div>
    <div id="fd">{fii_html}</div>
    <div id="strat">{strat_html}</div>
    {strikes_html}
    <div class="section">
      <div style="background:rgba(255,183,77,.06);border:1px solid rgba(255,183,77,.2);border-left:3px solid #ffb74d;border-radius:10px;padding:16px 18px;font-size:13px;color:#ffb74d;line-height:1.8;">
        <strong>⚠ DISCLAIMER</strong><br>
        This dashboard is for <strong>EDUCATIONAL purposes only</strong> — NOT financial advice.<br>
        Always use stop losses. Consult a SEBI-registered investment advisor before trading.<br>
        Past performance does not guarantee future results.
      </div>
    </div>
  </main>
</div>

<footer>
  <span>NiftyCraft · NSE Options Dashboard · Deep Ocean Theme</span>
  <span>For Educational Purposes Only · © 2025</span>
</footer>
</div>

<script>
function scrollTo(id) {{
  const el = document.getElementById(id);
  if (el) el.scrollIntoView({{behavior:'smooth', block:'start'}});
  document.querySelectorAll('.sb-nav-btn').forEach(b => b.classList.remove('active'));
  event.currentTarget.classList.add('active');
}}
</script>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════════
#  SECTION 8 ── MAIN RUNNER
# ═══════════════════════════════════════════════════════════════

def main():
    ist_tz = pytz.timezone('Asia/Kolkata')
    ts     = datetime.now(ist_tz).strftime('%d-%b-%Y %H:%M IST')
    print("=" * 65)
    print("  NIFTY 50 OPTIONS DASHBOARD — GitHub Pages Generator")
    print(f"  {ts}")
    print("=" * 65)

    # 1. Option chain
    print("\n📡 [1/4] Fetching NSE Option Chain...")
    oc_raw      = NSEOptionChain().fetch()
    oc_analysis = analyze_option_chain(oc_raw) if oc_raw else None
    if oc_analysis:
        print(f"  ✅ Option chain ready | Spot={oc_analysis['underlying']:.2f} ATM={oc_analysis['atm_strike']}")
    else:
        print("  ⚠️  Option chain unavailable — continuing with technical data only")

    # 2. Technical data
    print("\n📈 [2/4] Fetching Technical Indicators...")
    tech = get_technical_data()

    # 3. FII/DII
    print("\n🏦 [3/4] Fetching FII/DII Institutional Flow...")
    fii_data = fetch_fii_dii_data()
    fii_summ = compute_fii_dii_summary(fii_data)
    print(f"  ✅ FII/DII | Direction: {fii_summ['label']}")

    # 4. Market direction scoring
    print("\n🎯 [4/4] Computing Market Direction...")
    md = compute_market_direction(tech, oc_analysis)
    print(f"  ✅ Direction: {md['bias']} ({md['confidence']} confidence)")

    # 5. Generate HTML
    print("\n🖥️  Generating HTML dashboard...")
    html = generate_html(tech, oc_analysis, md, fii_data, fii_summ, ts)

    # 6. Save
    os.makedirs("docs", exist_ok=True)
    out_path = os.path.join("docs", "index.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    size_kb = len(html) / 1024
    print(f"  ✅ Saved: {out_path}  ({size_kb:.1f} KB)")

    # 7. Save JSON metadata for badge/status
    meta = {
        'timestamp':   ts,
        'bias':        md['bias'],
        'confidence':  md['confidence'],
        'bull_score':  md['bull'],
        'bear_score':  md['bear'],
        'price':       round(tech['price'], 2) if tech else None,
        'expiry':      oc_analysis['expiry'] if oc_analysis else None,
        'pcr':         oc_analysis['pcr_oi'] if oc_analysis else None,
        'fii_signal':  fii_summ['label'],
        'oi_dir':      oc_analysis['oi_dir'] if oc_analysis else None,
    }
    with open(os.path.join("docs", "latest.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("  ✅ Saved: docs/latest.json")

    print("\n" + "=" * 65)
    print("  ✅ DONE — Push to GitHub to deploy to GitHub Pages")
    print(f"  📊 Market Bias: {md['bias']} | Confidence: {md['confidence']}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
