"""
S&P 500 COMPLETE STOCK ANALYZER v11
Technical + Fundamental Analysis with Email Delivery
Theme: Dark Slate (Redesigned)
VERSION 7:
  - New dark blue/charcoal theme from redesign sample
  - Grouped column headers: Identity · Trade Setup · Risk · Momentum · Valuation · External
  - Quick View / Detail View toggle (hides Momentum + Valuation in Quick mode)
  - RSI mini progress bar
  - Earnings row warning highlight + pulsing badge when within 14 days
  - Tooltips on all column headers
  - Full width + mobile responsive layout
  - NEW columns: Sector, Vol/Avg, ADX, Analyst Consensus,
                 Support Distance %, Earnings Date
  - 12-Month S/R lookback + round numbers + 52W levels
  - ATR-Based Stop Loss Near Real S/R Zones
  - Dynamic Target Promotion
  - Live Clock + Report Timestamp
  - DJI / NDX / SPX live index strip in header
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import pytz
import warnings
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os

warnings.filterwarnings('ignore')


class SP500CompleteAnalyzer:
    def __init__(self):
        self.sp500_stocks = {
            'NVDA': 'NVIDIA',
            'AAPL': 'Apple Inc.',
            'MSFT': 'Microsoft',
            'AMZN': 'Amazon',
            'GOOGL': 'Alphabet (Class A)',
            'GOOG': 'Alphabet (Class C)',
            'META': 'Meta Platforms',
            'TSLA': 'Tesla',
            'AVGO': 'Broadcom',
            'BRK-B': 'Berkshire Hathaway',
            'WMT': 'Walmart',
            'LLY': 'Eli Lilly',
            'JPM': 'JPMorgan Chase',
            'XOM': 'ExxonMobil',
            'V': 'Visa Inc.',
            'JNJ': 'Johnson & Johnson',
            'MA': 'Mastercard',
            'MU': 'Micron Technology',
            'ORCL': 'Oracle Corporation',
            'COST': 'Costco',
            'ABBV': 'AbbVie',
            'HD': 'Home Depot',
            'BAC': 'Bank of America',
            'PG': 'Procter & Gamble',
            'CVX': 'Chevron Corporation',
            'CAT': 'Caterpillar Inc.',
            'KO': 'Coca-Cola Company',
            'GE': 'GE Aerospace',
            'AMD': 'Advanced Micro Devices',
            'NFLX': 'Netflix',
            'PLTR': 'Palantir Technologies',
            'MRK': 'Merck & Co.',
            'CSCO': 'Cisco Systems',
            'PM': 'Philip Morris International',
            'LRCX': 'Lam Research',
            'AMAT': 'Applied Materials',
            'MS': 'Morgan Stanley',
            'WFC': 'Wells Fargo',
            'GS': 'Goldman Sachs',
            'RTX': 'RTX Corporation',
            'UNH': 'UnitedHealth Group',
            'TMUS': 'T-Mobile US',
            'IBM': 'IBM',
            'MCD': "McDonald's",
            'AXP': 'American Express',
            'INTC': 'Intel',
            'PEP': 'PepsiCo',
            'LIN': 'Linde plc',
            'GEV': 'GE Vernova',
            'VZ': 'Verizon',
        }
        self.results = []

    # =========================================================================
    #  UTILITY
    # =========================================================================
    def get_est_time(self):
        return datetime.now(pytz.timezone('US/Eastern'))

    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain  = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs    = gain / loss
        return (100 - (100 / (1 + rs))).iloc[-1]

    def calculate_macd(self, prices):
        ema12  = prices.ewm(span=12, adjust=False).mean()
        ema26  = prices.ewm(span=26, adjust=False).mean()
        macd   = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd.iloc[-1], signal.iloc[-1]

    # ------------------------------------------------------------------
    #  NEW v8: momentum-slope helpers
    # ------------------------------------------------------------------
    def calculate_rsi_slope(self, prices, period=14, lookback=5):
        """Return RSI(today) - RSI(lookback bars ago).  Negative = fading."""
        delta = prices.diff()
        gain  = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs    = gain / loss
        rsi_series = 100 - (100 / (1 + rs))
        if len(rsi_series) < lookback + 1:
            return 0.0
        return round(rsi_series.iloc[-1] - rsi_series.iloc[-(lookback + 1)], 2)

    def calculate_macd_hist_slope(self, prices):
        """Return histogram(today) - histogram(yesterday).  Negative = shrinking."""
        ema12  = prices.ewm(span=12, adjust=False).mean()
        ema26  = prices.ewm(span=26, adjust=False).mean()
        macd   = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist   = macd - signal
        if len(hist) < 2:
            return 0.0
        return round(hist.iloc[-1] - hist.iloc[-2], 4)

    def calculate_atr(self, df, period=14):
        high  = df['High']
        low   = df['Low']
        close = df['Close']
        tr    = pd.concat([high - low,
                           abs(high - close.shift(1)),
                           abs(low  - close.shift(1))], axis=1).max(axis=1)
        return round(tr.ewm(alpha=1 / period, adjust=False).mean().iloc[-1], 2)

    def calculate_adx(self, df, period=14):
        high  = df['High']
        low   = df['Low']
        close = df['Close']
        plus_dm  = high.diff()
        minus_dm = low.diff().abs()
        plus_dm[plus_dm < 0]   = 0
        minus_dm[minus_dm < 0] = 0
        plus_dm[plus_dm < minus_dm]  = 0
        minus_dm[minus_dm < plus_dm] = 0
        tr = pd.concat([high - low,
                        abs(high - close.shift(1)),
                        abs(low  - close.shift(1))], axis=1).max(axis=1)
        atr14    = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di  = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr14)
        minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr14)
        dx       = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx      = dx.ewm(alpha=1/period, adjust=False).mean()
        return round(adx.iloc[-1], 1)

    def calculate_volume_ratio(self, df):
        avg_vol = df['Volume'].tail(20).mean()
        if avg_vol == 0:
            return 1.0
        return round(df['Volume'].iloc[-1] / avg_vol, 2)

    def get_earnings_date(self, info):
        try:
            ts = info.get('earningsTimestamp') or \
                 info.get('earningsTimestampStart') or \
                 info.get('earningsDate')
            if ts:
                if isinstance(ts, (list, tuple)):
                    ts = ts[0]
                if isinstance(ts, (int, float)):
                    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                    return dt.strftime('%d %b %Y')
                if hasattr(ts, 'strftime'):
                    return ts.strftime('%d %b %Y')
        except Exception:
            pass
        return "N/A"

    def is_earnings_soon(self, earnings_date_str, days=14):
        """Returns True if earnings are within `days` days from now."""
        if earnings_date_str == "N/A":
            return False
        try:
            ed = datetime.strptime(earnings_date_str, '%d %b %Y')
            now = datetime.now()
            delta = (ed - now).days
            return 0 <= delta <= days
        except Exception:
            return False

    def fetch_index_data(self):
        """Fetch DJI, NDX, SPX prices via yfinance at report generation time."""
        indices = {
            'DJI': '^DJI',
            'NDX': '^NDX',
            'SPX': '^GSPC',
        }
        result = {}
        for label, sym in indices.items():
            try:
                d     = yf.Ticker(sym).history(period='2d')
                price = d['Close'].iloc[-1]
                prev  = d['Close'].iloc[-2]
                chg   = price - prev
                pct   = chg / prev * 100
                arrow = '▲' if chg >= 0 else '▼'
                cls   = 'up' if chg >= 0 else 'dn'
                sign  = '+' if chg >= 0 else ''
                result[label] = {
                    'price': f"{price:,.2f}",
                    'chg':   f"{arrow} {sign}{pct:.2f}%",
                    'cls':   cls,
                }
            except Exception:
                result[label] = {'price': 'N/A', 'chg': '—', 'cls': ''}
        return result

    # =========================================================================
    #  RESISTANCE & SUPPORT
    # =========================================================================
    def find_resistance_levels(self, df, current_price, num_levels=5):
        window      = 5
        swing_highs = []
        for src_days in [180, 252]:
            highs = df.tail(src_days)['High'].values
            for i in range(window, len(highs) - window):
                if highs[i] > max(highs[i-window:i]) and \
                   highs[i] > max(highs[i+1:i+window+1]):
                    swing_highs.append(highs[i])
        high_52w = df['High'].tail(252).max()
        if high_52w > current_price * 1.005:
            swing_highs.append(high_52w)
        magnitude = 10 ** (len(str(int(current_price))) - 2)
        step      = magnitude * 5
        level     = current_price
        for _ in range(20):
            level += step
            if level <= current_price * 1.30:
                swing_highs.append(level)
        if not swing_highs:
            return []
        swing_highs = sorted(set([round(h, 2) for h in swing_highs]))
        clusters, cluster = [], [swing_highs[0]]
        for lv in swing_highs[1:]:
            if (lv - cluster[-1]) / cluster[-1] < 0.015:
                cluster.append(lv)
            else:
                clusters.append(cluster); cluster = [lv]
        clusters.append(cluster)
        res = [{'level': round(sum(c)/len(c), 2), 'strength': len(c)}
               for c in clusters if sum(c)/len(c) > current_price * 1.005]
        return sorted(res, key=lambda x: x['level'])[:num_levels]

    def find_support_levels(self, df, current_price, num_levels=5):
        window     = 5
        swing_lows = []
        for src_days in [180, 252]:
            lows = df.tail(src_days)['Low'].values
            for i in range(window, len(lows) - window):
                if lows[i] < min(lows[i-window:i]) and \
                   lows[i] < min(lows[i+1:i+window+1]):
                    swing_lows.append(lows[i])
        low_52w = df['Low'].tail(252).min()
        if low_52w < current_price * 0.995:
            swing_lows.append(low_52w)
        magnitude = 10 ** (len(str(int(current_price))) - 2)
        step      = magnitude * 5
        level     = current_price
        for _ in range(20):
            level -= step
            if level >= current_price * 0.70 and level > 0:
                swing_lows.append(level)
        if not swing_lows:
            return []
        swing_lows = sorted(set([round(l, 2) for l in swing_lows]))
        clusters, cluster = [], [swing_lows[0]]
        for lv in swing_lows[1:]:
            if (lv - cluster[-1]) / cluster[-1] < 0.015:
                cluster.append(lv)
            else:
                clusters.append(cluster); cluster = [lv]
        clusters.append(cluster)
        sup = [{'level': round(sum(c)/len(c), 2), 'strength': len(c)}
               for c in clusters if sum(c)/len(c) < current_price * 0.995]
        return sorted(sup, key=lambda x: x['level'], reverse=True)[:num_levels]

    # =========================================================================
    #  DYNAMIC TARGETS
    # =========================================================================
    def calculate_dynamic_targets(self, current_price, resistance_levels,
                                   support_levels, target_price, atr):
        valid       = [r['level'] for r in resistance_levels
                       if r['level'] > current_price * 1.005]
        min_target  = current_price + (atr * 2)
        targets_hit = 0
        if len(valid) >= 2:
            t1, t2        = valid[0], valid[1]
            target_status = "Real S/R Levels"
        elif len(valid) == 1:
            t1 = valid[0]
            t2 = round(target_price, 2) if target_price and target_price > t1 * 1.01 \
                 else round(t1 * 1.04, 2)
            target_status = "Partial Real Levels"
        else:
            t1 = round(target_price, 2) if target_price and target_price > current_price * 1.005 \
                 else round(current_price * 1.03, 2)
            t2            = round(t1 * 1.04, 2)
            target_status = "ATH Zone — Projected"
        if t1 < min_target:
            t1            = round(min_target, 2)
            t2            = round(t1 * 1.04, 2)
            target_status += " (ATR Adj)"
        return round(t1, 2), round(t2, 2), targets_hit, target_status

    # =========================================================================
    #  FUNDAMENTAL SCORE
    # =========================================================================
    def get_fundamental_score(self, info):
        score = 0
        pe    = info.get('trailingPE', info.get('forwardPE', 0))
        pb    = info.get('priceToBook', 0)
        peg   = info.get('pegRatio', 0)
        if pe  and 0 < pe < 25:      score += 10
        elif pe  and 25 <= pe < 35:  score += 5
        if pb  and 0 < pb < 3:       score += 5
        elif pb  and 3 <= pb < 5:    score += 3
        if peg and 0 < peg < 1:      score += 10
        elif peg and 1 <= peg < 2:   score += 5
        roe = info.get('returnOnEquity', 0)
        roa = info.get('returnOnAssets', 0)
        pm  = info.get('profitMargins', 0)
        if roe and roe > 0.15:   score += 10
        elif roe and roe > 0.10: score += 5
        if roa and roa > 0.05:   score += 5
        elif roa and roa > 0.02: score += 3
        if pm  and pm  > 0.10:   score += 10
        elif pm  and pm  > 0.05: score += 5
        rg = info.get('revenueGrowth', 0)
        eg = info.get('earningsGrowth', 0)
        if rg and rg > 0.15:   score += 10
        elif rg and rg > 0.10: score += 7
        elif rg and rg > 0.05: score += 5
        if eg and eg > 0.15:   score += 10
        elif eg and eg > 0.10: score += 7
        elif eg and eg > 0.05: score += 5
        de = info.get('debtToEquity', 0)
        cr = info.get('currentRatio', 0)
        fc = info.get('freeCashflow', 0)
        if de is not None:
            if de < 50:    score += 10
            elif de < 100: score += 5
        else:
            score += 5
        if cr and cr > 1.5:  score += 10
        elif cr and cr > 1.0: score += 5
        if fc and fc > 0:    score += 5
        return min(score, 100)

    # =========================================================================
    #  MAIN ANALYSIS
    # =========================================================================
    def analyze_stock(self, symbol, name):
        try:
            stock = yf.Ticker(symbol)
            df    = stock.history(period='1y')
            info  = stock.info
            if df.empty or len(df) < 200:
                return None
            current_price = df['Close'].iloc[-1]
            sma_20  = df['Close'].rolling(20).mean().iloc[-1]
            sma_50  = df['Close'].rolling(50).mean().iloc[-1]
            sma_200 = df['Close'].rolling(200).mean().iloc[-1]
            rsi          = self.calculate_rsi(df['Close'])
            macd, signal = self.calculate_macd(df['Close'])
            atr          = self.calculate_atr(df)
            atr_pct      = round((atr / current_price) * 100, 2)
            adx          = self.calculate_adx(df)
            vol_ratio    = self.calculate_volume_ratio(df)
            # v8: momentum-slope metrics
            rsi_slope       = self.calculate_rsi_slope(df['Close'])
            macd_hist_slope = self.calculate_macd_hist_slope(df['Close'])
            high_52w = df['High'].tail(252).max()
            low_52w  = df['Low'].tail(252).min()
            resistance_levels = self.find_resistance_levels(df, current_price)
            support_levels    = self.find_support_levels(df, current_price)
            nearest_resistance = resistance_levels[0]['level'] if resistance_levels \
                                 else df.tail(60)['High'].quantile(0.90)
            nearest_support    = support_levels[0]['level'] if support_levels \
                                 else df.tail(60)['Low'].quantile(0.10)
            support_dist_pct = round(((current_price - nearest_support) / current_price) * 100, 2)
            tech_score = 0
            tech_score += 1 if current_price > sma_20  else -1
            tech_score += 1 if current_price > sma_50  else -1
            tech_score += 2 if current_price > sma_200 else -2
            if rsi < 30:
                tech_score += 2;  rsi_signal = "Oversold"
            elif rsi > 70:
                tech_score -= 2;  rsi_signal = "Overbought"
            else:
                rsi_signal = "Neutral"
            if macd > signal:
                tech_score += 1;  macd_signal = "Bullish"
            else:
                tech_score -= 1;  macd_signal = "Bearish"
            if adx > 25:
                tech_score = min(tech_score + 1, 6)

            # ── v8: momentum-reversal penalties ──────────────────────────
            # 1. RSI fading: high RSI that is now rolling over = danger
            if rsi > 55 and rsi_slope < -2:
                tech_score -= 2   # RSI peaked and falling fast
            elif rsi > 50 and rsi_slope < -1:
                tech_score -= 1   # mild RSI fade

            # 2. MACD histogram shrinking while positive = momentum loss
            macd_hist_now = macd - signal
            if macd_hist_now > 0 and macd_hist_slope < -0.05:
                tech_score -= 1   # bullish but weakening
            if macd_hist_now < 0 and macd_hist_slope < 0:
                tech_score -= 1   # histogram negative AND still falling

            # 3. Price extended above SMA-20 AND RSI falling = stretched entry
            price_ext_pct = ((current_price - sma_20) / sma_20) * 100
            if price_ext_pct > 7 and rsi_slope < -1:
                tech_score -= 1   # extended AND fading
            # ── end v8 penalties ─────────────────────────────────────────

            pe_ratio         = info.get('trailingPE', info.get('forwardPE', 0))
            pb_ratio         = info.get('priceToBook', 0)
            peg_ratio        = info.get('pegRatio', 0)
            market_cap       = info.get('marketCap', 0)
            dividend_yield   = info.get('dividendYield', 0)
            roe              = info.get('returnOnEquity', 0)
            roa              = info.get('returnOnAssets', 0)
            profit_margin    = info.get('profitMargins', 0)
            operating_margin = info.get('operatingMargins', 0)
            eps              = info.get('trailingEps', 0)
            revenue_growth   = info.get('revenueGrowth', 0)
            earnings_growth  = info.get('earningsGrowth', 0)
            debt_to_equity   = info.get('debtToEquity', 0)
            current_ratio    = info.get('currentRatio', 0)
            beta             = info.get('beta', 1.0)
            target_price     = info.get('targetMeanPrice', None)
            sector           = info.get('sector', 'N/A')
            analyst_key      = info.get('recommendationKey', 'N/A')
            analyst_map      = {
                'strongBuy': 'Strong Buy', 'buy': 'Buy',
                'hold': 'Hold', 'sell': 'Sell', 'strongSell': 'Strong Sell'
            }
            analyst_label    = analyst_map.get(analyst_key, analyst_key.title() if analyst_key else 'N/A')
            earnings_date    = self.get_earnings_date(info)
            earn_soon        = self.is_earnings_soon(earnings_date)
            fund_score = self.get_fundamental_score(info)
            tech_score_normalized = ((tech_score + 6) / 12) * 100
            combined_score        = (tech_score_normalized * 0.5) + (fund_score * 0.5)
            if combined_score >= 75:
                rating = "⭐⭐⭐⭐⭐ STRONG BUY";  recommendation = "STRONG BUY"
            elif combined_score >= 55:
                rating = "⭐⭐⭐⭐ BUY";           recommendation = "BUY"
            elif combined_score >= 45:
                rating = "⭐⭐⭐ HOLD";            recommendation = "HOLD"
            elif combined_score >= 30:
                rating = "⭐⭐ SELL";              recommendation = "SELL"
            else:
                rating = "⭐ STRONG SELL";         recommendation = "STRONG SELL"
            stock_beta = beta if beta else 1.0
            if stock_beta < 0.8:
                atr_multiplier = 1.0;  max_sl_pct = 5.0
            elif stock_beta < 1.2:
                atr_multiplier = 1.2;  max_sl_pct = 7.0
            elif stock_beta < 1.8:
                atr_multiplier = 1.5;  max_sl_pct = 10.0
            else:
                atr_multiplier = 2.0;  max_sl_pct = 12.0
            if recommendation in ["STRONG BUY", "BUY"]:
                atr_stop       = nearest_support - (atr * atr_multiplier)
                min_allowed_sl = current_price * (1 - max_sl_pct / 100)
                stop_loss      = max(atr_stop, min_allowed_sl)
                sl_percentage  = ((current_price - stop_loss) / current_price) * 100
                stop_type      = "ATR Stop" if atr_stop >= min_allowed_sl else "Beta Cap"
                target_1, target_2, targets_hit, target_status = \
                    self.calculate_dynamic_targets(
                        current_price, resistance_levels,
                        support_levels, target_price, atr)
                if target_1 <= current_price * 1.005:
                    recommendation = "HOLD"; rating = "⭐⭐⭐ HOLD"
                upside = ((target_1 - current_price) / current_price) * 100
            else:
                atr_stop       = nearest_resistance + (atr * atr_multiplier)
                max_allowed_sl = current_price * (1 + max_sl_pct / 100)
                stop_loss      = min(atr_stop, max_allowed_sl)
                sl_percentage  = ((stop_loss - current_price) / current_price) * 100
                stop_type      = "ATR Stop" if atr_stop <= max_allowed_sl else "Beta Cap"
                valid_sups = [s['level'] for s in support_levels
                              if s['level'] < current_price * 0.995]
                if len(valid_sups) >= 2:
                    target_1, target_2 = valid_sups[0], valid_sups[1]
                    target_status = "Real S/R Levels"
                elif len(valid_sups) == 1:
                    target_1 = valid_sups[0]; target_2 = round(target_1 * 0.96, 2)
                    target_status = "Partial Real Levels"
                else:
                    target_1 = round(current_price * 0.96, 2)
                    target_2 = round(current_price * 0.92, 2)
                    target_status = "Projected"
                targets_hit = 0
                upside      = ((current_price - target_1) / current_price) * 100
            risk        = abs(current_price - stop_loss)
            reward      = abs(target_1 - current_price)
            risk_reward = round(reward / risk, 2) if risk > 0 else 0
            if fund_score >= 80:   quality = "Excellent"
            elif fund_score >= 60: quality = "Good"
            elif fund_score >= 40: quality = "Average"
            else:                  quality = "Poor"
            return {
                'Symbol': symbol, 'Name': name, 'Price': round(current_price, 2),
                'Sector': sector,
                'RSI': round(rsi, 2), 'RSI_Signal': rsi_signal,
                'RSI_Slope': rsi_slope,
                'MACD': macd_signal,
                'MACD_Hist_Slope': round(macd_hist_slope, 4),
                'ADX': adx,
                'Vol_Ratio': vol_ratio,
                'SMA_20': round(sma_20, 2), 'SMA_50': round(sma_50, 2), 'SMA_200': round(sma_200, 2),
                'Support': round(nearest_support, 2), 'Resistance': round(nearest_resistance, 2),
                'Support_Dist_Pct': support_dist_pct,
                '52W_High': round(high_52w, 2), '52W_Low': round(low_52w, 2),
                'Tech_Score': tech_score, 'Tech_Score_Norm': round(tech_score_normalized, 1),
                'ATR': atr, 'ATR_Pct': atr_pct, 'ATR_Multiplier': atr_multiplier, 'Stop_Type': stop_type,
                'PE_Ratio':         round(pe_ratio, 2)           if pe_ratio else 0,
                'PB_Ratio':         round(pb_ratio, 2)           if pb_ratio else 0,
                'PEG_Ratio':        round(peg_ratio, 2)          if peg_ratio else 0,
                'ROE':              round(roe * 100, 2)          if roe else 0,
                'ROA':              round(roa * 100, 2)          if roa else 0,
                'Profit_Margin':    round(profit_margin * 100, 2)    if profit_margin else 0,
                'Operating_Margin': round(operating_margin * 100, 2) if operating_margin else 0,
                'EPS':              round(eps, 2)                if eps else 0,
                'Dividend_Yield':   round(dividend_yield * 100, 2)   if dividend_yield else 0,
                'Revenue_Growth':   round(revenue_growth * 100, 2)   if revenue_growth else 0,
                'Earnings_Growth':  round(earnings_growth * 100, 2)  if earnings_growth else 0,
                'Debt_to_Equity':   round(debt_to_equity, 2)    if debt_to_equity else 0,
                'Current_Ratio':    round(current_ratio, 2)     if current_ratio else 0,
                'Market_Cap':       round(market_cap / 1e9, 2)  if market_cap else 0,
                'Beta':             round(beta, 2)               if beta else 1.0,
                'Fund_Score':       round(fund_score, 1), 'Quality': quality,
                'Combined_Score':   round(combined_score, 1),
                'Rating': rating, 'Recommendation': recommendation,
                'Stop_Loss': round(stop_loss, 2), 'SL_Percentage': round(sl_percentage, 2),
                'Target_1': round(target_1, 2), 'Target_2': round(target_2, 2),
                'Target_Price': round(target_price, 2) if target_price else 0,
                'Upside': round(upside, 2), 'Risk_Reward': risk_reward,
                'Targets_Hit': targets_hit, 'Target_Status': target_status,
                'Analyst': analyst_label,
                'Earnings_Date': earnings_date,
                'Earn_Soon': earn_soon,
            }
        except Exception:
            return None

    # =========================================================================
    #  ANALYZE ALL
    # =========================================================================
    def analyze_all_stocks(self):
        print(f"🔍 Analyzing {len(self.sp500_stocks)} stocks...")
        print("⏳ ~2-3 minutes...\n")
        for idx, (symbol, name) in enumerate(self.sp500_stocks.items(), 1):
            result = self.analyze_stock(symbol, name)
            if result:
                self.results.append(result)
            if idx % 10 == 0:
                print(f"  [{idx}/{len(self.sp500_stocks)}] {name}")
        print(f"\n✅ {len(self.results)} stocks analyzed\n")

    # =========================================================================
    #  TOP RECOMMENDATIONS
    # =========================================================================
    def get_top_recommendations(self):
        df = pd.DataFrame(self.results)
        all_buys = df[df['Recommendation'].isin(['STRONG BUY', 'BUY'])]
        print(f"\n📊 BUY Filter Debug:")
        f1 = all_buys[all_buys['Upside'] > 0.5]
        f2 = f1[f1['Risk_Reward'] >= 0.5]
        f3 = f2[f2['Target_1'] > f2['Price']]
        # v8: drop stocks where RSI is high AND falling (momentum reversal)
        f4 = f3[~((f3['RSI'] > 60) & (f3['RSI_Slope'] < -2))]
        # v8: drop stocks where MACD histogram is shrinking while RSI is elevated
        f5 = f4[~((f4['MACD_Hist_Slope'] < -0.05) & (f4['RSI'] > 58))]
        # v9: drop stocks with high sell-off volume + falling RSI = institution selling day
        #     Vol_Ratio > 2.0 means today's volume is 2x the 20-day average
        #     Combined with RSI_Slope < 0 = momentum falling on big volume = dangerous entry
        f6 = f5[~((f5['Vol_Ratio'] > 2.0) & (f5['RSI_Slope'] < 0))]
        print(f"   {len(all_buys)} → {len(f1)} → {len(f2)} → {len(f3)} → {len(f4)} → {len(f5)} → {len(f6)} final (v9 vol+momentum filter)")
        # Log what was filtered by v9 so you can review next session
        v9_filtered = f5[((f5['Vol_Ratio'] > 2.0) & (f5['RSI_Slope'] < 0))]
        if not v9_filtered.empty:
            print(f"   ⚠️  v9 removed (high sell-off volume — check tomorrow):")
            for _, r in v9_filtered.iterrows():
                print(f"      {r['Symbol']:6s}  Vol×{r['Vol_Ratio']:.1f}  RSI slope {r['RSI_Slope']:+.1f}  RSI {r['RSI']:.0f}")
        top_buys = f6.nlargest(20, 'Combined_Score')
        all_sells = df[df['Recommendation'].isin(['STRONG SELL', 'SELL'])]
        s1 = all_sells[all_sells['Upside'] > 0.5]
        s2 = s1[s1['Risk_Reward'] >= 0.5]
        s3 = s2[s2['Target_1'] < s2['Price']]
        print(f"   SELL: {len(all_sells)} → {len(s3)} final\n")
        top_sells = s3.nsmallest(20, 'Combined_Score')
        return top_buys, top_sells

    # =========================================================================
    #  HTML — v7 Redesign: Dark Slate, Grouped Headers, Quick/Detail Toggle
    # =========================================================================
    def generate_email_html(self):
        df = pd.DataFrame(self.results)
        top_buys, top_sells = self.get_top_recommendations()

        now         = self.get_est_time()
        idx_data    = self.fetch_index_data()
        time_of_day = "Morning" if now.hour < 12 else "Evening"
        next_update = "4:30 PM" if now.hour < 12 else "9:30 AM (Next Day)"

        strong_buy_count  = len(df[df['Recommendation'] == 'STRONG BUY'])
        buy_count         = len(df[df['Recommendation'] == 'BUY'])
        hold_count        = len(df[df['Recommendation'] == 'HOLD'])
        sell_count        = len(df[df['Recommendation'] == 'SELL'])
        strong_sell_count = len(df[df['Recommendation'] == 'STRONG SELL'])

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
<title>Top US Market Influencers — {time_of_day} Report</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Syne:wght@600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
:root {{
  --bg:      #07080c;
  --surf:    #0d1117;
  --card:    #101620;
  --card2:   #141c28;
  --border:  rgba(255,255,255,0.06);
  --text:    #eaf4ff;
  --muted:   #9abccc;
  --accent:  #f59e0b;
  --green:   #22c55e;
  --red:     #ef4444;
  --blue:    #60a5fa;
  --teal:    #2dd4bf;
  --purple:  #a78bfa;
  --orange:  #f97316;

  --g-identity: rgba(245,158,11,0.08);
  --g-trade:    rgba(34,197,94,0.07);
  --g-risk:     rgba(239,68,68,0.06);
  --g-momentum: rgba(96,165,250,0.07);
  --g-value:    rgba(167,139,250,0.06);
  --g-external: rgba(45,212,191,0.06);
}}
*, *::before, *::after {{ margin:0; padding:0; box-sizing:border-box; }}

body {{
  font-family:'DM Sans',sans-serif;
  background:var(--bg); color:var(--text);
  font-size:14px; line-height:1.5;
  min-height:100vh;
}}
body::before {{
  content:''; position:fixed; inset:0;
  background-image:
    linear-gradient(rgba(245,158,11,0.02) 1px, transparent 1px),
    linear-gradient(90deg, rgba(245,158,11,0.02) 1px, transparent 1px);
  background-size:48px 48px;
  pointer-events:none; z-index:0;
}}

.page {{ position:relative; z-index:1; padding:20px 16px 60px; max-width:1800px; margin:0 auto; }}

/* ── HEADER ── */
.hdr {{
  display:flex; align-items:center; justify-content:space-between;
  margin-bottom:20px; flex-wrap:wrap; gap:12px;
  background:linear-gradient(135deg,var(--card),var(--card2));
  border:1px solid var(--border); border-radius:14px;
  padding:16px 20px;
}}
.hdr-left {{ display:flex; align-items:center; gap:14px; flex-wrap:wrap; gap:12px; }}
.hdr-icon {{
  width:42px; height:42px; flex-shrink:0;
  background:linear-gradient(135deg,var(--accent),#ef4444);
  border-radius:10px; display:flex; align-items:center;
  justify-content:center; font-size:20px;
}}
.hdr-title {{
  font-family:'Syne',sans-serif;
  font-size:clamp(13px,2vw,19px); font-weight:800; color:#fff;
}}
.hdr-sub {{ font-size:0.72em; color:#a8cce0; margin-top:2px; letter-spacing:0.3px; font-weight:600; }}

/* ── INDEX STRIP ── */
.idx-strip {{
  display:flex; align-items:center; gap:0;
  background:rgba(0,0,0,0.3); border:1px solid var(--border);
  border-radius:10px; overflow:hidden;
}}
.idx-item {{ display:flex; align-items:center; gap:10px; padding:8px 18px; }}
.idx-name {{ font-size:9px; font-weight:800; letter-spacing:2px; color:var(--muted); text-transform:uppercase; }}
.idx-price {{ font-family:'JetBrains Mono',monospace; font-size:14px; font-weight:700; color:#fff; }}
.idx-chg {{ font-family:'JetBrains Mono',monospace; font-size:11px; font-weight:700; }}
.idx-chg.up {{ color:var(--green); }}
.idx-chg.dn {{ color:var(--red); }}
.idx-sep {{ width:1px; height:28px; background:var(--border); }}

/* ── LIVE CLOCK ── */
.live-clock {{
  display:flex; flex-direction:column; align-items:center;
  padding:8px 18px; border-left:1px solid var(--border);
}}
.lc-label {{ font-size:8px; color:var(--muted); letter-spacing:2px; text-transform:uppercase; }}
.lc-time  {{ font-family:'JetBrains Mono',monospace; font-size:15px; font-weight:700; color:var(--green); letter-spacing:2px; margin-top:2px; }}
.lc-date  {{ font-family:'JetBrains Mono',monospace; font-size:9px; color:var(--muted); margin-top:1px; }}
.lc-last  {{ font-size:8px; color:var(--accent); margin-top:2px; white-space:nowrap; }}

/* ── VIEW TOGGLE ── */
.view-toggle {{
  display:flex; gap:4px; background:var(--card);
  border:1px solid var(--border); border-radius:8px; padding:3px;
}}
.vt-btn {{
  padding:6px 16px; border-radius:6px; border:none; cursor:pointer;
  font-family:'JetBrains Mono',monospace; font-size:0.68em; font-weight:700;
  letter-spacing:0.5px; text-transform:uppercase;
  background:transparent; color:var(--muted); transition:all 0.2s;
}}
.vt-btn.active {{ background:var(--accent); color:#000; }}
.vt-btn:hover:not(.active) {{ color:var(--text); }}

/* ── TICKER STRIP ── */
.ticker {{
  background:rgba(0,0,0,0.4); border:1px solid var(--border);
  border-radius:8px; margin-bottom:14px; overflow:hidden;
}}
.ticker-inner {{ display:flex; overflow-x:auto; scrollbar-width:none; }}
.ticker-inner::-webkit-scrollbar {{ display:none; }}
.ti {{
  display:flex; gap:6px; align-items:center;
  padding:6px 14px; border-right:1px solid var(--border);
  font-family:'JetBrains Mono',monospace; font-size:10px; white-space:nowrap;
}}
.ti-s {{ color:var(--accent); font-weight:700; }}
.ti-p {{ color:#fff; }}
.ti-u {{ color:var(--green); }}
.ti-d {{ color:var(--red); }}

/* ── KPI BAND ── */
.kpi-band {{
  display:grid; grid-template-columns:repeat(5,1fr);
  background:var(--card); border:1px solid var(--border);
  border-radius:12px; margin-bottom:20px; overflow:hidden;
}}
.kc {{ padding:14px 10px; border-right:1px solid var(--border); text-align:center; }}
.kc:last-child {{ border-right:none; }}
.kn {{ font-family:'Syne',sans-serif; font-size:clamp(22px,4vw,32px); font-weight:800; line-height:1; }}
.kl {{ font-size:8px; letter-spacing:1.5px; text-transform:uppercase; color:var(--muted); margin-top:4px; }}
.kbar {{ height:2px; border-radius:1px; margin:5px auto 0; width:30px; }}

/* ── LEGEND ── */
.legend {{
  display:flex; align-items:center; flex-wrap:wrap; gap:10px;
  background:var(--card); border:1px solid var(--border);
  border-radius:8px; padding:8px 16px; font-size:0.7em;
  color:var(--muted); margin-bottom:14px;
}}
.leg-item {{ display:flex; align-items:center; gap:6px; }}
.leg-dot {{ width:8px; height:8px; border-radius:50%; flex-shrink:0; }}

/* ── SECTION TITLE ── */
.sec-title {{ display:flex; align-items:center; gap:10px; margin-bottom:10px; }}
.sec-title-icon {{
  width:28px; height:28px; border-radius:6px;
  display:flex; align-items:center; justify-content:center; font-size:13px;
}}
.sec-title-text {{ font-family:'Syne',sans-serif; font-size:1em; font-weight:800; color:#fff; }}
.sec-title-line {{ flex:1; height:1px; background:var(--border); }}
.sec-title-note {{ font-size:0.68em; color:var(--muted); }}

/* ── TABLE WRAPPER ── */
.tbl-wrap {{
  width:100%; overflow-x:auto;
  border:1px solid var(--border); border-radius:12px;
  background:var(--card);
  box-shadow:0 8px 40px rgba(0,0,0,0.4);
  -webkit-overflow-scrolling:touch;
  margin-bottom:28px;
}}
table {{ width:100%; border-collapse:collapse; min-width:1200px; }}

/* ── GROUP HEADER ROW ── */
.grp-row th {{
  padding:5px 10px;
  font-family:'JetBrains Mono',monospace;
  font-size:0.58em; font-weight:700; letter-spacing:2px;
  text-transform:uppercase; border-bottom:none; text-align:center;
}}
.grp-identity {{ background:var(--g-identity); color:var(--accent);  border-right:2px solid rgba(245,158,11,0.2); }}
.grp-trade    {{ background:var(--g-trade);    color:var(--green);   border-right:2px solid rgba(34,197,94,0.2); }}
.grp-risk     {{ background:var(--g-risk);     color:#f87171;        border-right:2px solid rgba(239,68,68,0.2); }}
.grp-momentum {{ background:var(--g-momentum); color:var(--blue);    border-right:2px solid rgba(96,165,250,0.2); }}
.grp-value    {{ background:var(--g-value);    color:var(--purple);  border-right:2px solid rgba(167,139,250,0.2); }}
.grp-external {{ background:var(--g-external); color:var(--teal); }}

/* ── COLUMN HEADERS ── */
thead tr.col-hdr th {{
  font-size:0.74em; font-weight:900; letter-spacing:1px;
  text-transform:uppercase; color:#cce8ff;
  padding:10px 10px;
  border-bottom:2px solid rgba(255,255,255,0.18);
  white-space:nowrap; text-align:left;
  background:var(--card2);
}}
.gh-identity {{ border-right:2px solid rgba(245,158,11,0.15)!important; }}
.gh-trade    {{ border-right:2px solid rgba(34,197,94,0.12)!important; }}
.gh-risk     {{ border-right:2px solid rgba(239,68,68,0.12)!important; }}
.gh-momentum {{ border-right:2px solid rgba(96,165,250,0.12)!important; }}
.gh-value    {{ border-right:2px solid rgba(167,139,250,0.12)!important; }}

/* ── TABLE ROWS ── */
tbody tr {{ transition:background 0.15s; }}
tbody tr:hover td {{ background:rgba(245,158,11,0.04); }}
tbody tr:nth-child(even) td {{ background:rgba(255,255,255,0.04); }}
tbody tr:nth-child(even):hover td {{ background:rgba(245,158,11,0.04); }}

td {{
  padding:12px 10px; border-bottom:1px solid rgba(255,255,255,0.08);
  vertical-align:middle; white-space:nowrap;
}}
tbody tr:last-child td {{ border-bottom:none; }}

/* group separators on data cells */
td.sep-identity {{ border-right:2px solid rgba(245,158,11,0.1); }}
td.sep-trade    {{ border-right:2px solid rgba(34,197,94,0.08); }}
td.sep-risk     {{ border-right:2px solid rgba(239,68,68,0.08); }}
td.sep-momentum {{ border-right:2px solid rgba(96,165,250,0.08); }}
td.sep-value    {{ border-right:2px solid rgba(167,139,250,0.08); }}

/* ── CELL COMPONENTS ── */
.c-num {{ font-family:'JetBrains Mono',monospace; font-size:0.72em; color:#6a8090; text-align:center; }}
.c-name {{ font-size:0.90em; font-weight:700; color:#ffffff; line-height:1.2; }}
.c-sym  {{ font-family:'JetBrains Mono',monospace; font-size:0.68em; font-weight:800; color:#f59e0b; letter-spacing:1px; margin-top:1px; }}
.c-sector {{ font-size:0.72em; color:#a0c8e0; margin-top:2px; max-width:130px; overflow:hidden; text-overflow:ellipsis; font-weight:700; }}

/* Earnings warning */
tr.earn-warning td {{ background:rgba(239,68,68,0.04)!important; }}
tr.earn-warning td.earn-cell {{ background:rgba(239,68,68,0.1)!important; }}
/* v9: High sell-off volume warning */
tr.vol-warning td {{ background:rgba(249,115,22,0.04)!important; }}
.earn-badge {{
  font-family:'JetBrains Mono',monospace; font-size:0.65em; font-weight:700;
  padding:2px 7px; border-radius:4px; display:inline-block; white-space:nowrap;
}}
.earn-soon {{ background:rgba(239,68,68,0.18); color:#f87171; border:1px solid rgba(239,68,68,0.3); animation:pulse 2s infinite; }}
.earn-ok   {{ background:rgba(45,212,191,0.18); color:#60fff0; border:1px solid rgba(45,212,191,0.40); }}
.earn-na   {{ color:var(--muted); font-size:0.65em; }}
@keyframes pulse {{ 0%,100%{{opacity:1;}} 50%{{opacity:0.6;}} }}

.c-price {{ font-family:'JetBrains Mono',monospace; font-size:1.05em; font-weight:900; color:#ffc040; }}

.rating {{
  display:inline-flex; align-items:center; gap:4px;
  font-family:'JetBrains Mono',monospace;
  font-size:0.62em; font-weight:800;
  padding:4px 9px; border-radius:5px; white-space:nowrap; letter-spacing:0.5px;
}}
.r-sb {{ background:rgba(34,197,94,0.22);  color:#6dffaa; border:1px solid rgba(34,197,94,0.45); }}
.r-b  {{ background:rgba(45,212,191,0.18); color:#5fffee; border:1px solid rgba(45,212,191,0.40); }}
.r-s  {{ background:rgba(239,68,68,0.22);  color:#ff8080; border:1px solid rgba(239,68,68,0.45); }}
.r-ss {{ background:rgba(239,68,68,0.28);  color:#ffb0b0; border:1px solid rgba(239,68,68,0.55); }}

.c-score-n   {{ font-family:'JetBrains Mono',monospace; font-size:1.35em; font-weight:900; line-height:1; }}
.c-score-bar {{ height:7px; border-radius:4px; margin-top:7px; width:46px; opacity:1.0; box-shadow:0 0 8px currentColor; }}

.up {{ font-family:'JetBrains Mono',monospace; font-size:0.98em; font-weight:900; color:#90ffb8; }}
.dn {{ font-family:'JetBrains Mono',monospace; font-size:0.98em; font-weight:900; color:#ffaaaa; }}

.ts-badge {{
  display:inline-block; font-family:'JetBrains Mono',monospace;
  font-size:0.60em; font-weight:800; padding:2px 6px; border-radius:3px;
  margin-bottom:4px;
}}
.ts-real    {{ background:rgba(34,197,94,0.20);  color:#6dffaa; border:1px solid rgba(34,197,94,0.35); }}
.ts-partial {{ background:rgba(245,158,11,0.20); color:#ffd060; border:1px solid rgba(245,158,11,0.35); }}
.ts-ath     {{ background:rgba(96,165,250,0.20); color:#b0d8ff; border:1px solid rgba(96,165,250,0.35); }}
.ts-hit1    {{ background:rgba(34,197,94,0.28);  color:#6dffaa; border:1px solid rgba(34,197,94,0.50); }}
.ts-hit2    {{ background:rgba(45,212,191,0.28); color:#5fffee; border:1px solid rgba(45,212,191,0.50); }}

.c-t1 {{ font-family:'JetBrains Mono',monospace; font-size:0.95em; font-weight:900; color:#ffffff; }}
.c-t2 {{ font-size:0.78em; color:#d8f0ff; margin-top:3px; font-weight:800; }}
.c-sl {{ font-family:'JetBrains Mono',monospace; font-size:0.85em; font-weight:700; color:#ff8080; }}
.c-sl-sell {{ font-family:'JetBrains Mono',monospace; font-size:0.85em; font-weight:700; color:#ffd060; }}
.c-slpct {{ font-size:0.78em; color:#ffd080; margin-top:3px; font-weight:800; }}
.sl-badge {{
  display:inline-block; font-size:0.60em; font-weight:800;
  padding:2px 6px; border-radius:3px; margin-top:3px;
  font-family:'JetBrains Mono',monospace;
}}
.sl-atr  {{ background:rgba(34,197,94,0.18);  color:#6dffaa; border:1px solid rgba(34,197,94,0.3); }}
.sl-beta {{ background:rgba(245,158,11,0.18); color:#ffd060; border:1px solid rgba(245,158,11,0.3); }}

.c-rr {{ font-family:'JetBrains Mono',monospace; font-size:1.05em; font-weight:900; }}
.c-atr {{ font-family:'JetBrains Mono',monospace; font-size:0.82em; font-weight:700; color:#5fffee; }}
.c-atr-sub {{ font-size:0.76em; color:#b0d8e8; margin-top:3px; font-weight:800; }}
.c-beta {{ font-family:'JetBrains Mono',monospace; font-size:0.85em; font-weight:700; }}
.c-sd   {{ font-family:'JetBrains Mono',monospace; font-size:0.85em; font-weight:700; }}

/* RSI progress bar */
.c-rsi {{ font-family:'JetBrains Mono',monospace; font-size:0.90em; font-weight:800; }}
.c-rsi-lbl {{ font-size:0.65em; color:#8ab8c8; margin-top:2px; font-weight:600; }}
.rsi-track {{ width:40px; height:5px; background:rgba(255,255,255,0.08); border-radius:3px; margin-top:4px; overflow:hidden; }}
.rsi-fill  {{ height:100%; border-radius:3px; }}

.c-macd {{ font-size:0.76em; font-weight:800; padding:3px 8px; border-radius:4px; font-family:'JetBrains Mono',monospace; display:inline-block; }}
.macd-bull {{ background:rgba(34,197,94,0.18);  color:#6dffaa; border:1px solid rgba(34,197,94,0.3); }}
.macd-bear {{ background:rgba(239,68,68,0.18);  color:#ff8080; border:1px solid rgba(239,68,68,0.3); }}

.c-adx {{ font-family:'JetBrains Mono',monospace; font-size:0.88em; font-weight:800; }}
.c-adx-lbl {{ font-size:0.65em; margin-top:2px; font-weight:600; }}
.c-vol {{ font-family:'JetBrains Mono',monospace; font-size:0.85em; font-weight:700; }}
.c-vol-lbl {{ font-size:0.65em; color:#8ab8c8; margin-top:2px; font-weight:600; }}
.c-pe  {{ font-family:'JetBrains Mono',monospace; font-size:0.85em; font-weight:700; }}
.c-div {{ font-family:'JetBrains Mono',monospace; font-size:0.82em; font-weight:700; }}
.c-52w {{ font-family:'JetBrains Mono',monospace; font-size:0.82em; font-weight:700; }}

.analyst {{ font-family:'JetBrains Mono',monospace; font-size:0.67em; font-weight:800; padding:3px 8px; border-radius:4px; display:inline-block; }}
.an-sb {{ background:rgba(34,197,94,0.20);  color:#6dffaa; border:1px solid rgba(34,197,94,0.35); }}
.an-b  {{ background:rgba(96,165,250,0.18); color:#b0d8ff; border:1px solid rgba(96,165,250,0.35); }}
.an-h  {{ background:rgba(160,120,80,0.22); color:#e8c080; border:1px solid rgba(160,120,80,0.40); }}
.an-s  {{ background:rgba(239,68,68,0.18);  color:#ff9090; border:1px solid rgba(239,68,68,0.35); }}

.qual {{ font-size:0.67em; font-weight:800; padding:3px 8px; border-radius:4px; }}
.q-ex {{ background:rgba(34,197,94,0.20);  color:#6dffaa; border:1px solid rgba(34,197,94,0.35); }}
.q-gd {{ background:rgba(96,165,250,0.18); color:#b0d8ff; border:1px solid rgba(96,165,250,0.35); }}
.q-av {{ background:rgba(245,158,11,0.18); color:#ffd060; border:1px solid rgba(245,158,11,0.35); }}
.q-po {{ background:rgba(239,68,68,0.18);  color:#ff9090; border:1px solid rgba(239,68,68,0.35); }}

/* QUICK VIEW hidden cols */
.detail-col {{ transition:opacity 0.2s; }}
body.quick-view .detail-col {{ display:none; }}

/* TOOLTIP */
[data-tip] {{ position:relative; cursor:help; }}
[data-tip]::after {{
  content:attr(data-tip);
  position:absolute; bottom:calc(100% + 6px); left:50%; transform:translateX(-50%);
  background:#1a2030; border:1px solid var(--border);
  color:var(--text); font-size:0.68em; padding:5px 9px; border-radius:6px;
  white-space:nowrap; pointer-events:none; opacity:0;
  transition:opacity 0.15s; z-index:100;
  font-family:'DM Sans',sans-serif; font-weight:400;
}}
[data-tip]:hover::after {{ opacity:1; }}

/* ── DISCLAIMER ── */
.disc {{
  background:var(--card); border:1px solid var(--border);
  border-left:3px solid #f59e0b; padding:12px 16px;
  margin:16px 0; font-size:11px; color:var(--muted); line-height:1.7;
  border-radius:8px;
}}
.disc strong {{ color:#f87171; }}

/* ── FOOTER ── */
footer {{
  background:var(--card); border-top:1px solid var(--border);
  text-align:center; padding:14px; font-size:10px;
  color:var(--muted); letter-spacing:1px; border-radius:0 0 12px 12px;
}}
footer strong {{ color:var(--accent); }}

/* ── MOBILE ── */
@media(max-width:900px) {{
  .kpi-band {{ grid-template-columns:repeat(3,1fr); }}
  .idx-strip {{ display:none; }}
}}
@media(max-width:600px) {{
  .kpi-band {{ grid-template-columns:repeat(2,1fr); }}
  .live-clock {{ display:none; }}
  .hdr {{ padding:12px; }}
  .page {{ padding:10px 8px 40px; }}
}}
</style>
</head>
<body class="quick-view">

<div class="page">

<!-- HEADER -->
<div class="hdr">
  <div class="hdr-left">
    <div class="hdr-icon">🌅</div>
    <div>
      <div class="hdr-title">Top US Market Influencers · NASDAQ &amp; S&amp;P 500</div>
      <div class="hdr-sub">12M S/R · ATR Stops · Tech &amp; Fundamental Analysis v11 · Report: {now.strftime('%d %b %Y %I:%M %p')} EST</div>
    </div>
  </div>

  <!-- Index Strip -->
  <div class="idx-strip">
    <div class="idx-item">
      <span class="idx-name">DJI</span>
      <span class="idx-price">{idx_data['DJI']['price']}</span>
      <span class="idx-chg {idx_data['DJI']['cls']}">{idx_data['DJI']['chg']}</span>
    </div>
    <div class="idx-sep"></div>
    <div class="idx-item">
      <span class="idx-name">NDX</span>
      <span class="idx-price">{idx_data['NDX']['price']}</span>
      <span class="idx-chg {idx_data['NDX']['cls']}">{idx_data['NDX']['chg']}</span>
    </div>
    <div class="idx-sep"></div>
    <div class="idx-item">
      <span class="idx-name">SPX</span>
      <span class="idx-price">{idx_data['SPX']['price']}</span>
      <span class="idx-chg {idx_data['SPX']['cls']}">{idx_data['SPX']['chg']}</span>
    </div>
  </div>

  <!-- Live Clock + View Toggle -->
  <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
    <div class="live-clock" style="background:rgba(0,0,0,0.3);border:1px solid var(--border);border-radius:10px;">
      <div class="lc-label">EST TIME</div>
      <div class="lc-time" id="liveClock">--:-- --</div>
      <div class="lc-date" id="liveDate">{now.strftime('%d %b %Y')}</div>
      <div class="lc-last">Next: {next_update}</div>
    </div>
    <div class="view-toggle">
      <button class="vt-btn active" onclick="setView('quick',this)">⚡ Quick</button>
      <button class="vt-btn" onclick="setView('detail',this)">🔍 Detail</button>
    </div>
  </div>
</div>

<!-- TICKER -->
<div class="ticker"><div class="ticker-inner">
"""
        for t in self.results[:10]:
            pct  = ((t['Price'] - t['SMA_20']) / t['SMA_20']) * 100
            cls  = "ti-u" if pct >= 0 else "ti-d"
            sign = "+" if pct >= 0 else ""
            html += f'<div class="ti"><span class="ti-s">{t["Symbol"]}</span><span class="ti-p">${t["Price"]:,.2f}</span><span class="{cls}">{sign}{pct:.1f}%</span></div>'

        html += f"""</div></div>

<!-- KPI BAND -->
<div class="kpi-band">
  <div class="kc"><div class="kn" style="color:#f59e0b">{len(self.results)}</div><div class="kl">Analyzed</div><div class="kbar" style="background:#f59e0b"></div></div>
  <div class="kc"><div class="kn" style="color:#22c55e">{strong_buy_count}</div><div class="kl">Strong Buy</div><div class="kbar" style="background:#22c55e"></div></div>
  <div class="kc"><div class="kn" style="color:#2dd4bf">{buy_count}</div><div class="kl">Buy</div><div class="kbar" style="background:#2dd4bf"></div></div>
  <div class="kc"><div class="kn" style="color:#ef4444">{sell_count + strong_sell_count}</div><div class="kl">Sell</div><div class="kbar" style="background:#ef4444"></div></div>
  <div class="kc"><div class="kn" style="color:#60a5fa">{hold_count}</div><div class="kl">Hold</div><div class="kbar" style="background:#60a5fa"></div></div>
</div>

<!-- LEGEND -->
<div class="legend">
  <span style="font-weight:600;color:var(--text);">Legend:</span>
  <div class="leg-item"><div class="leg-dot" style="background:#f59e0b;"></div> Identity</div>
  <div class="leg-item"><div class="leg-dot" style="background:#22c55e;"></div> Trade Setup</div>
  <div class="leg-item"><div class="leg-dot" style="background:#ef4444;"></div> Risk</div>
  <div class="leg-item"><div class="leg-dot" style="background:#60a5fa;"></div> Momentum (Detail)</div>
  <div class="leg-item"><div class="leg-dot" style="background:#a78bfa;"></div> Valuation (Detail)</div>
  <div class="leg-item"><div class="leg-dot" style="background:#2dd4bf;"></div> External</div>
  <span style="margin-left:8px;border-left:1px solid var(--border);padding-left:8px;">
    <span style="color:#f87171;font-weight:600;">🔴 Pulsing row</span> = Earnings within 14 days — trade carefully
  </span>
  <span style="margin-left:8px;border-left:1px solid var(--border);padding-left:8px;">
    <span style="color:#fb923c;font-weight:600;">🟠 ⚠ Vol×</span> badge = High sell-off volume today — wait 1 session before entering
  </span>
</div>
"""

        # ── helpers ──────────────────────────────────────────────────────────
        def analyst_badge(label):
            m = {'Strong Buy': 'an-sb', 'Buy': 'an-b', 'Hold': 'an-h',
                 'Sell': 'an-s', 'Strong Sell': 'an-s'}
            cls = m.get(label, 'an-h')
            return f'<span class="analyst {cls}">{label}</span>'

        def adx_html(v):
            c = "#4ade80" if v >= 30 else ("#fbbf24" if v >= 20 else "#a07850")
            l = "Strong" if v >= 30 else ("Moderate" if v >= 20 else "Weak")
            return f'<div class="c-adx" style="color:{c}">{v:.0f}</div><div class="c-adx-lbl" style="color:{c}">{l}</div>'

        def vol_html(v):
            c = "#4ade80" if v >= 1.5 else ("#a07850" if v < 0.7 else "#d8e8f5")
            l = "High Vol" if v >= 1.5 else ("Low Vol" if v < 0.7 else "Avg Vol")
            return f'<div class="c-vol" style="color:{c}">{v:.1f}×</div><div class="c-vol-lbl">{l}</div>'

        def rsi_html(v, sig):
            c = "#f87171" if v > 70 else ("#4ade80" if v < 30 else "#93c5fd")
            return (f'<div class="c-rsi" style="color:{c}">{v:.0f}</div>'
                    f'<div class="rsi-track"><div class="rsi-fill" style="width:{min(v,100):.0f}%;background:{c}"></div></div>'
                    f'<div class="c-rsi-lbl">{sig}</div>')

        def ts_badge(ts, th):
            if th == 2:           return 'ts-hit2',  '✅ T1+T2 Hit'
            elif th == 1:         return 'ts-hit1',  '✅ T1 Hit'
            elif 'ATH' in ts:     return 'ts-ath',   '🚀 ATH Zone'
            elif 'Partial' in ts: return 'ts-partial','⚡ Partial S/R'
            else:                 return 'ts-real',  '📍 Real S/R'

        def sc_color(v):
            return "#4ade80" if v >= 75 else ("#2dd4bf" if v >= 55 else "#fbbf24")

        def earn_html(date_str, soon):
            if date_str == "N/A":
                return '<span class="earn-na">N/A</span>'
            cls = "earn-soon" if soon else "earn-ok"
            prefix = "⚠ " if soon else ""
            return f'<span class="earn-badge {cls}">{prefix}{date_str}</span>'

        # ── BUY TABLE ─────────────────────────────────────────────────────────
        if not top_buys.empty:
            html += """
<div class="sec-title">
  <div class="sec-title-icon" style="background:rgba(34,197,94,0.12);color:#4ade80;">▲</div>
  <span class="sec-title-text">Top 20 Buy Recommendations</span>
  <div class="sec-title-line"></div>
  <span class="sec-title-note">Quick = 12 cols · Detail = all 21 · Hover headers for tips</span>
</div>

<div class="tbl-wrap"><table>
  <thead>
    <tr class="grp-row">
      <th colspan="5" class="grp-identity">⬡ Identity</th>
      <th colspan="4" class="grp-trade">▲ Trade Setup</th>
      <th colspan="3" class="grp-risk">⚠ Risk</th>
      <th colspan="4" class="grp-momentum detail-col">◎ Momentum</th>
      <th colspan="3" class="grp-value detail-col">◈ Valuation</th>
      <th colspan="3" class="grp-external">⊕ External</th>
    </tr>
    <tr class="col-hdr">
      <th data-tip="Row number">#</th>
      <th data-tip="Company name, ticker, and sector">Stock / Sector</th>
      <th data-tip="Last closing price">Price</th>
      <th data-tip="Algorithm rating from combined tech + fundamental score">Rating</th>
      <th class="gh-identity" data-tip="Combined score 0–100 (50% tech + 50% fundamental)">Score</th>
      <th data-tip="% gain to Target 1">Upside</th>
      <th data-tip="T1 = nearest resistance · T2 = next resistance">Target T1/T2</th>
      <th data-tip="ATR-based stop below support zone">Stop Loss</th>
      <th class="gh-trade" data-tip="Reward ÷ Risk · ≥2× is ideal">R:R</th>
      <th data-tip="Average True Range — daily volatility in $">ATR</th>
      <th data-tip="Market sensitivity · &lt;1 = stable · &gt;1.5 = volatile">Beta</th>
      <th class="gh-risk" data-tip="How far price is above nearest support">Sup Dist</th>
      <th class="detail-col" data-tip="Relative Strength Index · &lt;30 oversold · &gt;70 overbought">RSI</th>
      <th class="detail-col" data-tip="MACD vs signal line direction">MACD</th>
      <th class="detail-col" data-tip="Average Directional Index · &gt;25 = strong trend">ADX</th>
      <th class="detail-col gh-momentum" data-tip="Today's volume ÷ 20-day avg · &gt;1.5× = high conviction">Vol/Avg</th>
      <th class="detail-col" data-tip="Price-to-Earnings · &lt;25 = reasonable">P/E</th>
      <th class="detail-col" data-tip="Annual dividend yield">Div%</th>
      <th class="detail-col gh-value" data-tip="Fundamental quality rating">Quality</th>
      <th data-tip="% below 52-week high">52W Hi%</th>
      <th data-tip="Wall Street analyst consensus">Analyst</th>
      <th class="earn-cell" data-tip="⚠ Row pulses red if earnings within 14 days">Earnings</th>
    </tr>
  </thead>
  <tbody>
"""
            for i, (_, row) in enumerate(top_buys.iterrows(), 1):
                rcls   = "r-sb" if row['Recommendation'] == "STRONG BUY" else "r-b"
                sc     = sc_color(row['Combined_Score'])
                upcls  = "up" if row['Upside'] >= 0 else "dn"
                w52    = ((row['Price'] - row['52W_High']) / row['52W_High']) * 100
                w52c   = "#f87171" if w52 >= -5 else ("#fbbf24" if w52 >= -20 else "#4ade80")
                betac  = "#f87171" if row['Beta'] > 1.5 else ("#fbbf24" if row['Beta'] > 1.0 else "#4ade80")
                rr     = row['Risk_Reward']
                rrc    = "#4ade80" if rr >= 2 else ("#2dd4bf" if rr >= 1 else "#f87171")
                pe_str = f"{row['PE_Ratio']:.1f}" if row['PE_Ratio'] > 0 else "N/A"
                pec    = "#a07850" if row['PE_Ratio'] <= 0 else ("#4ade80" if row['PE_Ratio'] < 25 else ("#fbbf24" if row['PE_Ratio'] < 40 else "#f87171"))
                div_str= f"{row['Dividend_Yield']:.2f}%" if row['Dividend_Yield'] > 0 else "—"
                divc   = "#4ade80" if row['Dividend_Yield'] > 0 else "#a07850"
                sdc    = "#4ade80" if row.get('Support_Dist_Pct', 0) <= 3 else ("#fbbf24" if row.get('Support_Dist_Pct', 0) <= 8 else "#f87171")
                qcls   = {"Excellent": "q-ex", "Good": "q-gd", "Average": "q-av", "Poor": "q-po"}.get(row['Quality'], "q-av")
                tbc, tbt = ts_badge(row.get('Target_Status', ''), row.get('Targets_Hit', 0))
                sltype = row.get('Stop_Type', 'ATR Stop')
                slcls  = "sl-atr" if sltype == "ATR Stop" else "sl-beta"
                sllbl  = f"{'📐' if sltype == 'ATR Stop' else '🔒'} {sltype}"
                soon      = row.get('Earn_Soon', False)
                vol_warn  = row.get('Vol_Ratio', 1.0) > 2.0 and row.get('RSI_Slope', 0) < 0
                rowcls    = "earn-warning" if soon else ("vol-warning" if vol_warn else "")
                vol_badge = ('<span style="font-size:0.6em;font-weight:700;padding:1px 5px;'
                             'border-radius:3px;background:rgba(249,115,22,0.15);'
                             'color:#fb923c;border:1px solid rgba(249,115,22,0.3);'
                             'margin-left:4px;" title="High sell-off volume today — wait 1 session">'
                             f'⚠ Vol×{row.get("Vol_Ratio",1.0):.1f}</span>') if vol_warn else ""

                html += f"""    <tr class="{rowcls}">
      <td><div class="c-num">{i}</div></td>
      <td>
        <div class="c-name">{row['Name']}{vol_badge}</div>
        <div class="c-sym">{row['Symbol']}</div>
        <div class="c-sector">{row.get('Sector','N/A')}</div>
      </td>
      <td><div class="c-price">${row['Price']:,.2f}</div></td>
      <td><span class="rating {rcls}">{row['Rating']}</span></td>
      <td class="sep-identity">
        <div class="c-score-n" style="color:{sc}">{row['Combined_Score']:.0f}</div>
        <div class="c-score-bar" style="background:{sc}"></div>
      </td>
      <td><span class="{upcls}">{row['Upside']:+.1f}%</span></td>
      <td>
        <span class="ts-badge {tbc}">{tbt}</span>
        <div class="c-t1">${row['Target_1']:,.2f}</div>
        <div class="c-t2">T2: ${row['Target_2']:,.2f}</div>
      </td>
      <td>
        <div class="c-sl">${row['Stop_Loss']:,.2f}</div>
        <div class="c-slpct">-{row['SL_Percentage']:.1f}%</div>
        <span class="sl-badge {slcls}">{sllbl}</span>
      </td>
      <td class="sep-trade"><div class="c-rr" style="color:{rrc}">{rr:.1f}×</div></td>
      <td>
        <div class="c-atr">${row['ATR']:,.2f}</div>
        <div class="c-atr-sub">{row['ATR_Pct']:.1f}% · {row['ATR_Multiplier']}×</div>
      </td>
      <td><div class="c-beta" style="color:{betac}">{row['Beta']:.2f}</div></td>
      <td class="sep-risk"><div class="c-sd" style="color:{sdc}">{row.get('Support_Dist_Pct',0):.1f}%</div></td>
      <td class="detail-col">{rsi_html(row['RSI'], row['RSI_Signal'])}</td>
      <td class="detail-col"><span class="c-macd {'macd-bull' if row['MACD']=='Bullish' else 'macd-bear'}">{row['MACD']}</span></td>
      <td class="detail-col">{adx_html(row.get('ADX',0))}</td>
      <td class="detail-col sep-momentum">{vol_html(row.get('Vol_Ratio',1.0))}</td>
      <td class="detail-col"><div class="c-pe" style="color:{pec}">{pe_str}</div></td>
      <td class="detail-col"><div class="c-div" style="color:{divc}">{div_str}</div></td>
      <td class="detail-col sep-value"><span class="qual {qcls}">{row['Quality']}</span></td>
      <td><div class="c-52w" style="color:{w52c}">{w52:+.1f}%</div></td>
      <td>{analyst_badge(row.get('Analyst','N/A'))}</td>
      <td class="earn-cell">{earn_html(row.get('Earnings_Date','N/A'), soon)}</td>
    </tr>
"""
            html += "  </tbody>\n</table></div>\n"

        # ── SELL TABLE ────────────────────────────────────────────────────────
        if not top_sells.empty:
            html += """
<div class="sec-title">
  <div class="sec-title-icon" style="background:rgba(239,68,68,0.12);color:#f87171;">▼</div>
  <span class="sec-title-text">Top 20 Sell Recommendations</span>
  <div class="sec-title-line"></div>
  <span class="sec-title-note">Quick = 12 cols · Detail = all 19 · Hover headers for tips</span>
</div>

<div class="tbl-wrap"><table>
  <thead>
    <tr class="grp-row">
      <th colspan="5" class="grp-identity">⬡ Identity</th>
      <th colspan="4" class="grp-trade">▼ Trade Setup</th>
      <th colspan="3" class="grp-risk">⚠ Risk</th>
      <th colspan="4" class="grp-momentum detail-col">◎ Momentum</th>
      <th colspan="3" class="grp-value detail-col">◈ Valuation</th>
      <th colspan="3" class="grp-external">⊕ External</th>
    </tr>
    <tr class="col-hdr">
      <th>#</th>
      <th>Stock / Sector</th>
      <th data-tip="Last closing price">Price</th>
      <th data-tip="Algorithm sell rating">Rating</th>
      <th class="gh-identity" data-tip="Combined score 0–100">Score</th>
      <th data-tip="% downside to Target 1">Downside</th>
      <th data-tip="T1 = nearest support · T2 = next support">Target T1/T2</th>
      <th data-tip="ATR-based stop above resistance">Stop Loss</th>
      <th class="gh-trade" data-tip="Reward ÷ Risk">R:R</th>
      <th data-tip="Average True Range in $">ATR</th>
      <th data-tip="Market beta sensitivity">Beta</th>
      <th class="gh-risk" data-tip="Distance to nearest support">Sup Dist</th>
      <th class="detail-col" data-tip="RSI · &lt;30 oversold · &gt;70 overbought">RSI</th>
      <th class="detail-col" data-tip="MACD signal direction">MACD</th>
      <th class="detail-col" data-tip="ADX trend strength">ADX</th>
      <th class="detail-col gh-momentum" data-tip="Volume vs 20-day average">Vol/Avg</th>
      <th class="detail-col" data-tip="P/E ratio">P/E</th>
      <th class="detail-col" data-tip="Dividend yield">Div%</th>
      <th class="detail-col gh-value" data-tip="Fundamental quality">Quality</th>
      <th data-tip="% below 52-week high">52W Hi%</th>
      <th data-tip="Analyst consensus">Analyst</th>
      <th class="earn-cell" data-tip="⚠ Pulses red if earnings within 14 days">Earnings</th>
    </tr>
  </thead>
  <tbody>
"""
            for i, (_, row) in enumerate(top_sells.iterrows(), 1):
                rcls   = "r-ss" if row['Recommendation'] == "STRONG SELL" else "r-s"
                sc     = "#f87171"
                dncls  = "dn" if row['Upside'] >= 0 else "up"
                w52    = ((row['Price'] - row['52W_High']) / row['52W_High']) * 100
                w52c   = "#f87171" if w52 >= -5 else ("#fbbf24" if w52 >= -20 else "#4ade80")
                betac  = "#f87171" if row['Beta'] > 1.5 else ("#fbbf24" if row['Beta'] > 1.0 else "#4ade80")
                rr     = row['Risk_Reward']
                rrc    = "#4ade80" if rr >= 2 else ("#fbbf24" if rr >= 1 else "#f87171")
                pe_str = f"{row['PE_Ratio']:.1f}" if row['PE_Ratio'] > 0 else "N/A"
                pec    = "#a07850" if row['PE_Ratio'] <= 0 else ("#f87171" if row['PE_Ratio'] > 40 else ("#fbbf24" if row['PE_Ratio'] > 25 else "#4ade80"))
                div_str= f"{row['Dividend_Yield']:.2f}%" if row['Dividend_Yield'] > 0 else "—"
                divc   = "#4ade80" if row['Dividend_Yield'] > 0 else "#a07850"
                sdc    = "#4ade80" if row.get('Support_Dist_Pct', 0) <= 3 else ("#fbbf24" if row.get('Support_Dist_Pct', 0) <= 8 else "#f87171")
                qcls   = {"Excellent": "q-ex", "Good": "q-gd", "Average": "q-av", "Poor": "q-po"}.get(row['Quality'], "q-av")
                tbc, tbt = ts_badge(row.get('Target_Status', ''), 0)
                sltype = row.get('Stop_Type', 'ATR Stop')
                slcls  = "sl-atr" if sltype == "ATR Stop" else "sl-beta"
                sllbl  = f"{'📐' if sltype == 'ATR Stop' else '🔒'} {sltype}"
                soon   = row.get('Earn_Soon', False)
                rowcls = "earn-warning" if soon else ""

                html += f"""    <tr class="{rowcls}">
      <td><div class="c-num">{i}</div></td>
      <td>
        <div class="c-name">{row['Name']}</div>
        <div class="c-sym">{row['Symbol']}</div>
        <div class="c-sector">{row.get('Sector','N/A')}</div>
      </td>
      <td><div class="c-price">${row['Price']:,.2f}</div></td>
      <td><span class="rating {rcls}">{row['Rating']}</span></td>
      <td class="sep-identity">
        <div class="c-score-n" style="color:{sc}">{row['Combined_Score']:.0f}</div>
        <div class="c-score-bar" style="background:{sc}"></div>
      </td>
      <td><span class="{dncls}">{row['Upside']:+.1f}%</span></td>
      <td>
        <span class="ts-badge {tbc}">{tbt}</span>
        <div class="c-t1">${row['Target_1']:,.2f}</div>
        <div class="c-t2">T2: ${row['Target_2']:,.2f}</div>
      </td>
      <td>
        <div class="c-sl-sell">${row['Stop_Loss']:,.2f}</div>
        <div class="c-slpct">+{row['SL_Percentage']:.1f}%</div>
        <span class="sl-badge {slcls}">{sllbl}</span>
      </td>
      <td class="sep-trade"><div class="c-rr" style="color:{rrc}">{rr:.1f}×</div></td>
      <td>
        <div class="c-atr">${row['ATR']:,.2f}</div>
        <div class="c-atr-sub">{row['ATR_Pct']:.1f}% · {row['ATR_Multiplier']}×</div>
      </td>
      <td><div class="c-beta" style="color:{betac}">{row['Beta']:.2f}</div></td>
      <td class="sep-risk"><div class="c-sd" style="color:{sdc}">{row.get('Support_Dist_Pct',0):.1f}%</div></td>
      <td class="detail-col">{rsi_html(row['RSI'], row['RSI_Signal'])}</td>
      <td class="detail-col"><span class="c-macd {'macd-bull' if row['MACD']=='Bullish' else 'macd-bear'}">{row['MACD']}</span></td>
      <td class="detail-col">{adx_html(row.get('ADX',0))}</td>
      <td class="detail-col sep-momentum">{vol_html(row.get('Vol_Ratio',1.0))}</td>
      <td class="detail-col"><div class="c-pe" style="color:{pec}">{pe_str}</div></td>
      <td class="detail-col"><div class="c-div" style="color:{divc}">{div_str}</div></td>
      <td class="detail-col sep-value"><span class="qual {qcls}">{row['Quality']}</span></td>
      <td><div class="c-52w" style="color:{w52c}">{w52:+.1f}%</div></td>
      <td>{analyst_badge(row.get('Analyst','N/A'))}</td>
      <td class="earn-cell">{earn_html(row.get('Earnings_Date','N/A'), soon)}</td>
    </tr>
"""
            html += "  </tbody>\n</table></div>\n"

        html += f"""
  <div class="disc">
    <strong>⚠ DISCLAIMER:</strong> For <strong>EDUCATIONAL PURPOSES ONLY</strong>. Not financial advice.
    Stop losses are ATR-based near real 12-month S/R zones. Targets derived from swing highs/lows,
    52-week extremes and round number levels. Earnings dates are estimates.
    Always conduct your own research, consult a registered financial advisor,
    and never invest more than you can afford to lose.
  </div>

  <footer>
    <strong>Top US Market Influencers: NASDAQ &amp; S&amp;P 500</strong>
    · 12M S/R · ATR Stops · Grouped Columns · Quick/Detail Toggle v10
    · Next Update: <strong>{next_update} EST</strong> · {now.strftime('%d %b %Y')}
  </footer>

</div><!-- /page -->

<script>
/* ── LIVE CLOCK ── */
function updateClock() {{
  var now = new Date();
  var est = new Date(now.toLocaleString('en-US', {{timeZone:'America/New_York'}}));
  var h = est.getHours(), m = est.getMinutes(), s = est.getSeconds();
  var ampm = h >= 12 ? 'PM' : 'AM';
  h = h % 12 || 12;
  var pad = n => String(n).padStart(2,'0');
  document.getElementById('liveClock').textContent = pad(h)+':'+pad(m)+' '+ampm+' EST';
  var months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  document.getElementById('liveDate').textContent = pad(est.getDate())+' '+months[est.getMonth()]+' '+est.getFullYear();
}}
updateClock();
setInterval(updateClock, 1000);

/* ── VIEW TOGGLE ── */
function setView(v, btn) {{
  document.querySelectorAll('.vt-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  if (v === 'quick') {{
    document.body.classList.add('quick-view');
  }} else {{
    document.body.classList.remove('quick-view');
  }}
}}
</script>

</body></html>"""
        return html

    # =========================================================================
    #  EMAIL
    # =========================================================================
    def send_email(self, to_email):
        try:
            from_email = os.environ.get('GMAIL_USER')
            password   = os.environ.get('GMAIL_APP_PASSWORD')
            if not from_email or not password:
                print("❌ Set GMAIL_USER and GMAIL_APP_PASSWORD"); return False
            now = self.get_est_time()
            tod = "Morning" if now.hour < 12 else "Evening"
            msg = MIMEMultipart('alternative')
            msg['From']    = from_email
            msg['To']      = to_email
            msg['Subject'] = f"🌅 US Market Report v7 — {tod} {now.strftime('%d %b %Y')}"
            msg.attach(MIMEText(self.generate_email_html(), 'html'))
            srv = smtplib.SMTP('smtp.gmail.com', 587)
            srv.starttls(); srv.login(from_email, password)
            srv.send_message(msg); srv.quit()
            print(f"✅ Email sent to {to_email}"); return True
        except Exception as e:
            print(f"❌ Email error: {e}"); return False

    # =========================================================================
    #  ENTRY
    # =========================================================================
    def generate_complete_report(self, send_email_flag=True, recipient_email=None):
        now = self.get_est_time()
        print("=" * 70)
        print("📊 S&P 500 ANALYZER v10 — Visibility Fixes")
        print(f"   {now.strftime('%d %b %Y, %I:%M %p EST')}")
        print("=" * 70)
        self.analyze_all_stocks()
        if send_email_flag and recipient_email:
            self.send_email(recipient_email)
        print("=" * 70); print("✅ DONE"); print("=" * 70)


# =============================================================================
#  RUN
# =============================================================================
def main():
    analyzer  = SP500CompleteAnalyzer()
    recipient = os.environ.get('RECIPIENT_EMAIL')
    analyzer.generate_complete_report(
        send_email_flag=bool(recipient), recipient_email=recipient)

if __name__ == "__main__":
    analyzer = SP500CompleteAnalyzer()
    analyzer.analyze_all_stocks()
    html = analyzer.generate_email_html()
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("✅ Report saved to index.html")
