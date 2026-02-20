#!/usr/bin/env python3
"""
Options Strategy Builder - HTML Generator
Generates a professional options strategy dashboard for GitHub Pages.
Run: python generate.py
Output: docs/index.html (served via GitHub Pages)
"""

import os
import json
from datetime import datetime

# ─── Strategy Definitions ─────────────────────────────────────────────────────

STRATEGIES = {
    "bullish": {
        "label": "Bullish",
        "icon": "↑",
        "color_var": "--bull",
        "strategies": [
            {
                "id": "long_call",
                "name": "Long Call",
                "risk": "Limited",
                "reward": "Unlimited",
                "complexity": 1,
                "iv_impact": "High IV hurts",
                "best_market": "Strong uptrend",
                "description": "Buy a call option giving you the right to purchase shares at the strike price. Maximum loss is the premium paid; upside is theoretically unlimited as the stock rises.",
                "max_profit": "Unlimited",
                "max_loss": "Premium Paid",
                "breakeven": "Strike + Premium",
                "legs": [
                    {"action": "BUY", "type": "CALL", "strike": "ATM / Slightly OTM", "qty": 1}
                ],
                "payoff_pts": [
                    [-50,-5],[-30,-5],[-20,-5],[-10,-5],[0,-5],[10,5],[20,15],[30,25],[50,45]
                ]
            },
            {
                "id": "covered_call",
                "name": "Covered Call",
                "risk": "Moderate",
                "reward": "Limited",
                "complexity": 2,
                "iv_impact": "High IV helps",
                "best_market": "Neutral to mildly bullish",
                "description": "Own 100 shares and sell an OTM call against them. Premium received lowers your cost basis and provides downside cushion. Profit is capped at the strike price.",
                "max_profit": "Strike − Cost + Premium",
                "max_loss": "Cost Basis − Premium",
                "breakeven": "Stock Cost − Premium",
                "legs": [
                    {"action": "OWN", "type": "STOCK", "strike": "Current Price", "qty": 100},
                    {"action": "SELL", "type": "CALL", "strike": "OTM", "qty": 1}
                ],
                "payoff_pts": [
                    [-50,-45],[-30,-25],[-10,-5],[0,5],[10,10],[20,10],[30,10],[50,10]
                ]
            },
            {
                "id": "bull_call_spread",
                "name": "Bull Call Spread",
                "risk": "Limited",
                "reward": "Limited",
                "complexity": 2,
                "iv_impact": "Neutral",
                "best_market": "Moderately bullish",
                "description": "Buy a lower-strike call and sell a higher-strike call. The short call reduces cost and risk but caps your maximum profit. Best when you have a specific price target.",
                "max_profit": "Spread Width − Net Debit",
                "max_loss": "Net Debit Paid",
                "breakeven": "Lower Strike + Net Debit",
                "legs": [
                    {"action": "BUY", "type": "CALL", "strike": "ATM (Lower)", "qty": 1},
                    {"action": "SELL", "type": "CALL", "strike": "OTM (Higher)", "qty": 1}
                ],
                "payoff_pts": [
                    [-50,-4],[-30,-4],[-10,-4],[0,-4],[5,-4],[10,1],[15,6],[20,6],[50,6]
                ]
            },
            {
                "id": "cash_secured_put",
                "name": "Cash-Secured Put",
                "risk": "Moderate",
                "reward": "Limited",
                "complexity": 2,
                "iv_impact": "High IV helps",
                "best_market": "Bullish, want to buy cheaper",
                "description": "Sell a put while holding cash equal to the obligation. If assigned, you buy shares at an effective discount (strike minus premium). If not assigned, you keep the premium.",
                "max_profit": "Premium Received",
                "max_loss": "Strike − Premium",
                "breakeven": "Strike − Premium",
                "legs": [
                    {"action": "SELL", "type": "PUT", "strike": "OTM / ATM", "qty": 1}
                ],
                "payoff_pts": [
                    [-50,-45],[-30,-25],[-10,-5],[0,5],[5,5],[10,5],[20,5],[50,5]
                ]
            }
        ]
    },
    "bearish": {
        "label": "Bearish",
        "icon": "↓",
        "color_var": "--bear",
        "strategies": [
            {
                "id": "long_put",
                "name": "Long Put",
                "risk": "Limited",
                "reward": "High",
                "complexity": 1,
                "iv_impact": "High IV hurts",
                "best_market": "Strong downtrend",
                "description": "Buy a put option giving you the right to sell shares at the strike price. Profits as the stock falls. Risk is limited to the premium paid.",
                "max_profit": "Strike − Premium (stock → 0)",
                "max_loss": "Premium Paid",
                "breakeven": "Strike − Premium",
                "legs": [
                    {"action": "BUY", "type": "PUT", "strike": "ATM / Slightly OTM", "qty": 1}
                ],
                "payoff_pts": [
                    [-50,45],[-30,25],[-10,5],[0,-5],[10,-5],[20,-5],[30,-5],[50,-5]
                ]
            },
            {
                "id": "bear_put_spread",
                "name": "Bear Put Spread",
                "risk": "Limited",
                "reward": "Limited",
                "complexity": 2,
                "iv_impact": "Neutral",
                "best_market": "Moderately bearish",
                "description": "Buy a higher-strike put and sell a lower-strike put. Reduces the cost of your bearish bet while capping maximum profit. Best with a specific downside target.",
                "max_profit": "Spread Width − Net Debit",
                "max_loss": "Net Debit Paid",
                "breakeven": "Higher Strike − Net Debit",
                "legs": [
                    {"action": "BUY", "type": "PUT", "strike": "ATM (Higher)", "qty": 1},
                    {"action": "SELL", "type": "PUT", "strike": "OTM (Lower)", "qty": 1}
                ],
                "payoff_pts": [
                    [-50,6],[-20,6],[-15,6],[-10,1],[-5,-4],[0,-4],[10,-4],[30,-4],[50,-4]
                ]
            },
            {
                "id": "bear_call_spread",
                "name": "Bear Call Spread",
                "risk": "Limited",
                "reward": "Limited",
                "complexity": 2,
                "iv_impact": "High IV helps",
                "best_market": "Mildly bearish / sideways",
                "description": "Sell a lower-strike call and buy a higher-strike call. This credit spread profits if the stock stays below the short call strike at expiry.",
                "max_profit": "Net Credit Received",
                "max_loss": "Spread Width − Net Credit",
                "breakeven": "Lower Strike + Net Credit",
                "legs": [
                    {"action": "SELL", "type": "CALL", "strike": "OTM (Lower)", "qty": 1},
                    {"action": "BUY", "type": "CALL", "strike": "OTM (Higher)", "qty": 1}
                ],
                "payoff_pts": [
                    [-50,5],[-20,5],[-10,5],[0,5],[5,5],[10,0],[15,-5],[20,-5],[50,-5]
                ]
            }
        ]
    },
    "neutral": {
        "label": "Neutral / Volatility",
        "icon": "↔",
        "color_var": "--neutral",
        "strategies": [
            {
                "id": "iron_condor",
                "name": "Iron Condor",
                "risk": "Limited",
                "reward": "Limited",
                "complexity": 4,
                "iv_impact": "High IV helps entry",
                "best_market": "Range-bound / low volatility",
                "description": "Sell an OTM put spread + sell an OTM call spread simultaneously. Collect premium on both sides and profit as long as the stock stays within your defined range until expiry.",
                "max_profit": "Total Net Credit",
                "max_loss": "Spread Width − Net Credit",
                "breakeven": "Lower: Short Put − Credit | Upper: Short Call + Credit",
                "legs": [
                    {"action": "BUY", "type": "PUT", "strike": "Far OTM (Lowest)", "qty": 1},
                    {"action": "SELL", "type": "PUT", "strike": "OTM (Lower-Mid)", "qty": 1},
                    {"action": "SELL", "type": "CALL", "strike": "OTM (Upper-Mid)", "qty": 1},
                    {"action": "BUY", "type": "CALL", "strike": "Far OTM (Highest)", "qty": 1}
                ],
                "payoff_pts": [
                    [-50,-8],[-30,-8],[-20,2],[-10,5],[0,5],[10,5],[20,2],[30,-8],[50,-8]
                ]
            },
            {
                "id": "straddle",
                "name": "Straddle",
                "risk": "Limited",
                "reward": "Unlimited",
                "complexity": 2,
                "iv_impact": "High IV hurts (buy low IV)",
                "best_market": "Expecting large move (earnings)",
                "description": "Buy both a call AND a put at the same strike price and expiry. Profit from a large move in either direction. Maximum loss occurs if the stock stays exactly at the strike.",
                "max_profit": "Unlimited (either direction)",
                "max_loss": "Total Premium Paid",
                "breakeven": "Strike ± Total Premium",
                "legs": [
                    {"action": "BUY", "type": "CALL", "strike": "ATM", "qty": 1},
                    {"action": "BUY", "type": "PUT", "strike": "ATM", "qty": 1}
                ],
                "payoff_pts": [
                    [-50,40],[-30,20],[-15,5],[-10,-2],[0,-10],[10,-2],[15,5],[30,20],[50,40]
                ]
            },
            {
                "id": "strangle",
                "name": "Strangle",
                "risk": "Limited",
                "reward": "Unlimited",
                "complexity": 2,
                "iv_impact": "High IV hurts (buy low IV)",
                "best_market": "Expecting big move, cost-conscious",
                "description": "Buy an OTM call and an OTM put at different strikes. Cheaper than a straddle but requires a larger price move to become profitable. Great before major events.",
                "max_profit": "Unlimited (either direction)",
                "max_loss": "Total Premium Paid",
                "breakeven": "Lower: Put Strike − Premium | Upper: Call Strike + Premium",
                "legs": [
                    {"action": "BUY", "type": "PUT", "strike": "OTM (Lower)", "qty": 1},
                    {"action": "BUY", "type": "CALL", "strike": "OTM (Higher)", "qty": 1}
                ],
                "payoff_pts": [
                    [-50,42],[-30,22],[-20,12],[-15,5],[-10,-3],[0,-6],[10,-3],[15,5],[20,12],[30,22],[50,42]
                ]
            },
            {
                "id": "butterfly_spread",
                "name": "Butterfly Spread",
                "risk": "Limited",
                "reward": "Limited",
                "complexity": 3,
                "iv_impact": "High IV hurts",
                "best_market": "Very low volatility expected",
                "description": "Buy one lower-strike call, sell two middle-strike calls, buy one higher-strike call. Maximum profit when the stock lands exactly on the middle strike at expiry.",
                "max_profit": "Middle Strike − Lower Strike − Net Debit",
                "max_loss": "Net Debit Paid",
                "breakeven": "Lower: Lower Strike + Net Debit | Upper: Higher Strike − Net Debit",
                "legs": [
                    {"action": "BUY", "type": "CALL", "strike": "Lower Strike", "qty": 1},
                    {"action": "SELL", "type": "CALL", "strike": "Middle Strike", "qty": 2},
                    {"action": "BUY", "type": "CALL", "strike": "Upper Strike", "qty": 1}
                ],
                "payoff_pts": [
                    [-50,-3],[-20,-3],[-10,-3],[0,-3],[5,3],[10,12],[15,3],[20,-3],[30,-3],[50,-3]
                ]
            }
        ]
    }
}

# ─── HTML Generator ───────────────────────────────────────────────────────────

def generate_html():
    strategies_json = json.dumps(STRATEGIES)
    now = datetime.now().strftime("%d %b %Y, %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Options Strategy Builder</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  /* ── Reset & Variables ── */
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  :root {{
    --bg:       #080c10;
    --surface:  #0e1318;
    --card:     #131920;
    --border:   #1e2830;
    --border2:  #263040;
    --text:     #cdd8e0;
    --muted:    #5a7080;
    --bull:     #00e5a0;
    --bull-dim: #00e5a022;
    --bear:     #ff4560;
    --bear-dim: #ff456022;
    --neutral:  #f0b429;
    --neutral-dim: #f0b42922;
    --accent:   #3d9eff;
    --font-head: 'Syne', sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
  }}

  html {{ scroll-behavior: smooth; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-mono);
    font-size: 13px;
    line-height: 1.6;
    min-height: 100vh;
    overflow-x: hidden;
  }}

  /* ── Grid Background ── */
  body::before {{
    content: '';
    position: fixed; inset: 0;
    background-image:
      linear-gradient(rgba(255,255,255,0.012) 1px, transparent 1px),
      linear-gradient(90deg, rgba(255,255,255,0.012) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none; z-index: 0;
  }}

  /* ── Layout ── */
  .app {{ position: relative; z-index: 1; display: flex; flex-direction: column; min-height: 100vh; }}

  /* ── Header ── */
  header {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 18px 32px;
    border-bottom: 1px solid var(--border);
    background: rgba(8,12,16,0.95);
    backdrop-filter: blur(12px);
    position: sticky; top: 0; z-index: 100;
  }}
  .logo {{
    font-family: var(--font-head);
    font-size: 20px; font-weight: 800;
    letter-spacing: -0.5px;
    color: #fff;
  }}
  .logo span {{ color: var(--bull); }}
  .header-meta {{
    font-size: 11px; color: var(--muted);
    display: flex; align-items: center; gap: 16px;
  }}
  .live-dot {{
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--bull);
    box-shadow: 0 0 8px var(--bull);
    animation: pulse 2s infinite;
  }}
  @keyframes pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:0.4}} }}

  /* ── Main Layout ── */
  .main {{ display: flex; flex: 1; }}

  /* ── Sidebar ── */
  .sidebar {{
    width: 280px; min-width: 280px;
    border-right: 1px solid var(--border);
    display: flex; flex-direction: column;
    background: var(--surface);
    position: sticky; top: 57px; height: calc(100vh - 57px);
    overflow-y: auto;
  }}
  .sidebar::-webkit-scrollbar {{ width: 4px; }}
  .sidebar::-webkit-scrollbar-track {{ background: transparent; }}
  .sidebar::-webkit-scrollbar-thumb {{ background: var(--border2); border-radius: 4px; }}

  .sidebar-section {{ padding: 20px 16px 8px; }}
  .sidebar-label {{
    font-size: 10px; font-weight: 500; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--muted);
    margin-bottom: 8px; padding: 0 4px;
  }}

  /* Direction Tabs */
  .direction-tabs {{
    display: flex; flex-direction: column; gap: 4px;
  }}
  .dir-tab {{
    display: flex; align-items: center; gap: 10px;
    padding: 10px 12px; border-radius: 8px;
    border: 1px solid transparent;
    cursor: pointer; transition: all 0.15s;
    background: transparent; color: var(--muted);
    font-family: var(--font-mono); font-size: 12px;
    text-align: left; width: 100%;
  }}
  .dir-tab:hover {{ background: var(--card); color: var(--text); }}
  .dir-tab.active.bull {{ background: var(--bull-dim); border-color: var(--bull); color: var(--bull); }}
  .dir-tab.active.bear {{ background: var(--bear-dim); border-color: var(--bear); color: var(--bear); }}
  .dir-tab.active.neutral {{ background: var(--neutral-dim); border-color: var(--neutral); color: var(--neutral); }}
  .dir-icon {{
    width: 28px; height: 28px; border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-size: 15px; font-weight: 700;
    background: rgba(255,255,255,0.05);
  }}
  .dir-tab.active.bull .dir-icon {{ background: var(--bull); color: #000; }}
  .dir-tab.active.bear .dir-icon {{ background: var(--bear); color: #fff; }}
  .dir-tab.active.neutral .dir-icon {{ background: var(--neutral); color: #000; }}
  .dir-count {{
    margin-left: auto;
    font-size: 10px; background: rgba(255,255,255,0.06);
    padding: 2px 7px; border-radius: 99px;
  }}

  /* Strategy List */
  .strategy-list {{ display: flex; flex-direction: column; gap: 3px; padding: 0 8px; }}
  .strat-item {{
    padding: 9px 12px; border-radius: 7px;
    cursor: pointer; transition: all 0.15s;
    border: 1px solid transparent;
    color: var(--muted); font-size: 12px;
    display: flex; align-items: center; gap: 8px;
  }}
  .strat-item:hover {{ background: var(--card); color: var(--text); }}
  .strat-item.active {{
    background: var(--card);
    border-color: var(--border2);
    color: var(--text);
  }}
  .strat-complexity {{ display: flex; gap: 2px; margin-left: auto; }}
  .dot {{
    width: 5px; height: 5px; border-radius: 50%;
    background: var(--border2);
  }}
  .dot.filled {{ background: var(--accent); }}

  /* ── Content ── */
  .content {{ flex: 1; overflow: auto; padding: 28px 32px; max-width: 1200px; }}

  /* ── Welcome State ── */
  .welcome {{
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    min-height: 60vh; text-align: center; gap: 16px;
  }}
  .welcome-icon {{ font-size: 64px; margin-bottom: 8px; }}
  .welcome h2 {{ font-family: var(--font-head); font-size: 32px; font-weight: 800; color: #fff; }}
  .welcome p {{ color: var(--muted); max-width: 380px; line-height: 1.8; }}
  .direction-cards {{
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;
    margin-top: 24px; width: 100%; max-width: 600px;
  }}
  .dir-card {{
    padding: 20px 16px; border-radius: 12px;
    border: 1px solid var(--border);
    background: var(--card); cursor: pointer;
    transition: all 0.2s; text-align: center;
  }}
  .dir-card:hover {{ transform: translateY(-2px); }}
  .dir-card.bull:hover {{ border-color: var(--bull); box-shadow: 0 0 24px var(--bull-dim); }}
  .dir-card.bear:hover {{ border-color: var(--bear); box-shadow: 0 0 24px var(--bear-dim); }}
  .dir-card.neutral:hover {{ border-color: var(--neutral); box-shadow: 0 0 24px var(--neutral-dim); }}
  .dir-card-icon {{ font-size: 28px; margin-bottom: 8px; }}
  .dir-card-label {{ font-family: var(--font-head); font-size: 14px; font-weight: 700; color: #fff; }}
  .dir-card-count {{ font-size: 11px; color: var(--muted); margin-top: 4px; }}

  /* ── Strategy Detail ── */
  .strategy-detail {{ display: none; animation: fadeIn 0.25s ease; }}
  .strategy-detail.visible {{ display: block; }}
  @keyframes fadeIn {{ from{{opacity:0;transform:translateY(8px)}} to{{opacity:1;transform:translateY(0)}} }}

  /* Header section */
  .detail-header {{ margin-bottom: 24px; }}
  .breadcrumb {{
    font-size: 11px; color: var(--muted); margin-bottom: 12px;
    display: flex; align-items: center; gap: 6px;
  }}
  .breadcrumb span {{ cursor: pointer; }}
  .breadcrumb span:hover {{ color: var(--text); }}
  .detail-title-row {{
    display: flex; align-items: flex-start; justify-content: space-between; gap: 16px;
  }}
  .detail-title {{
    font-family: var(--font-head); font-size: 30px; font-weight: 800; color: #fff;
  }}
  .detail-badges {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }}
  .badge {{
    padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 500;
    border: 1px solid;
  }}
  .badge.bull {{ color: var(--bull); border-color: var(--bull); background: var(--bull-dim); }}
  .badge.bear {{ color: var(--bear); border-color: var(--bear); background: var(--bear-dim); }}
  .badge.neutral {{ color: var(--neutral); border-color: var(--neutral); background: var(--neutral-dim); }}
  .badge.info {{ color: var(--accent); border-color: var(--accent); background: rgba(61,158,255,0.1); }}
  .badge.muted {{ color: var(--muted); border-color: var(--border2); background: transparent; }}

  /* Grid layout */
  .detail-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: auto auto;
    gap: 16px;
    margin-bottom: 20px;
  }}

  /* Cards */
  .card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
  }}
  .card-title {{
    font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--muted); margin-bottom: 14px; font-weight: 500;
  }}

  /* Description card spans full width */
  .card.full {{ grid-column: 1 / -1; }}

  .desc-text {{ color: var(--text); line-height: 1.9; font-size: 13px; }}

  /* P&L metrics */
  .metrics-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
  .metric {{ padding: 12px 14px; background: var(--surface); border-radius: 8px; border: 1px solid var(--border); }}
  .metric-label {{ font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }}
  .metric-value {{ font-size: 14px; font-weight: 500; color: #fff; font-family: var(--font-head); }}
  .metric-value.bull {{ color: var(--bull); }}
  .metric-value.bear {{ color: var(--bear); }}
  .metric-value.neutral {{ color: var(--neutral); }}
  .metric.breakeven {{ grid-column: 1 / -1; }}

  /* Legs */
  .legs-list {{ display: flex; flex-direction: column; gap: 8px; }}
  .leg {{
    display: flex; align-items: center; gap: 12px;
    padding: 10px 14px; border-radius: 8px;
    background: var(--surface); border: 1px solid var(--border);
  }}
  .leg-action {{
    font-size: 10px; font-weight: 600; padding: 3px 8px; border-radius: 4px;
    letter-spacing: 0.08em;
  }}
  .leg-action.BUY  {{ background: rgba(0,229,160,0.15); color: var(--bull); }}
  .leg-action.SELL {{ background: rgba(255,69,96,0.15);  color: var(--bear); }}
  .leg-action.OWN  {{ background: rgba(61,158,255,0.15); color: var(--accent); }}
  .leg-type {{ font-size: 12px; color: #fff; font-weight: 500; }}
  .leg-strike {{ font-size: 11px; color: var(--muted); margin-left: auto; }}
  .leg-qty {{
    font-size: 10px; padding: 2px 6px;
    border-radius: 4px; background: rgba(255,255,255,0.06);
    color: var(--muted);
  }}

  /* Chart */
  .chart-wrap {{
    position: relative;
    height: 220px;
  }}

  /* Strategy selector (multiple) */
  .strat-grid {{
    display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 14px;
  }}
  .strat-card {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; padding: 18px; cursor: pointer;
    transition: all 0.2s;
  }}
  .strat-card:hover {{
    border-color: var(--border2);
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
  }}
  .strat-card-name {{
    font-family: var(--font-head); font-size: 16px; font-weight: 700; color: #fff; margin-bottom: 6px;
  }}
  .strat-card-desc {{ font-size: 11px; color: var(--muted); line-height: 1.7; margin-bottom: 14px; }}
  .strat-card-meta {{ display: flex; gap: 8px; flex-wrap: wrap; }}

  /* NSE Builder Panel */
  .builder-panel {{
    margin-top: 20px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px; overflow: hidden;
  }}
  .builder-header {{
    padding: 14px 20px;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between;
  }}
  .builder-title {{
    font-family: var(--font-head); font-size: 14px; font-weight: 700; color: #fff;
  }}
  .builder-body {{ padding: 20px; }}
  .form-row {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }}
  .form-group {{ display: flex; flex-direction: column; gap: 6px; }}
  .form-group label {{ font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }}
  .form-control {{
    background: var(--surface); border: 1px solid var(--border2);
    color: var(--text); font-family: var(--font-mono); font-size: 12px;
    padding: 8px 12px; border-radius: 8px; outline: none;
    transition: border-color 0.15s;
  }}
  .form-control:focus {{ border-color: var(--accent); }}
  select.form-control {{ cursor: pointer; min-width: 140px; }}
  input.form-control {{ min-width: 120px; }}

  .btn {{
    padding: 8px 18px; border-radius: 8px; border: none;
    font-family: var(--font-mono); font-size: 12px; font-weight: 500;
    cursor: pointer; transition: all 0.15s;
  }}
  .btn-primary {{ background: var(--accent); color: #fff; }}
  .btn-primary:hover {{ filter: brightness(1.15); }}
  .btn-ghost {{
    background: transparent; color: var(--muted);
    border: 1px solid var(--border2);
  }}
  .btn-ghost:hover {{ border-color: var(--text); color: var(--text); }}

  .coming-soon {{
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 10px; color: var(--accent);
    background: rgba(61,158,255,0.1); border: 1px solid rgba(61,158,255,0.2);
    padding: 3px 10px; border-radius: 99px; letter-spacing: 0.06em;
  }}

  /* Footer */
  footer {{
    border-top: 1px solid var(--border);
    padding: 14px 32px;
    display: flex; align-items: center; justify-content: space-between;
    font-size: 11px; color: var(--muted);
    background: var(--surface);
  }}

  /* Scrollbar */
  .content::-webkit-scrollbar {{ width: 4px; }}
  .content::-webkit-scrollbar-track {{ background: transparent; }}
  .content::-webkit-scrollbar-thumb {{ background: var(--border2); border-radius: 4px; }}

  @media (max-width: 768px) {{
    .sidebar {{ display: none; }}
    .detail-grid {{ grid-template-columns: 1fr; }}
    .direction-cards {{ grid-template-columns: 1fr; }}
    .content {{ padding: 16px; }}
  }}
</style>
</head>
<body>
<div class="app">

  <!-- Header -->
  <header>
    <div class="logo">OPTIONS<span>CRAFT</span></div>
    <div class="header-meta">
      <div class="live-dot"></div>
      <span>NSE Options Strategy Builder</span>
      <span style="color:var(--border2)">|</span>
      <span>Generated: {now}</span>
    </div>
  </header>

  <div class="main">
    <!-- Sidebar -->
    <aside class="sidebar">
      <div class="sidebar-section">
        <div class="sidebar-label">Market Direction</div>
        <div class="direction-tabs">
          <button class="dir-tab bull" onclick="selectDirection('bullish')" id="tab-bullish">
            <div class="dir-icon">↑</div>
            <div>
              <div style="font-weight:600;color:inherit">Bullish</div>
              <div style="font-size:10px;opacity:0.7">Expecting rise</div>
            </div>
            <div class="dir-count">4</div>
          </button>
          <button class="dir-tab bear" onclick="selectDirection('bearish')" id="tab-bearish">
            <div class="dir-icon">↓</div>
            <div>
              <div style="font-weight:600;color:inherit">Bearish</div>
              <div style="font-size:10px;opacity:0.7">Expecting fall</div>
            </div>
            <div class="dir-count">3</div>
          </button>
          <button class="dir-tab neutral" onclick="selectDirection('neutral')" id="tab-neutral">
            <div class="dir-icon">↔</div>
            <div>
              <div style="font-weight:600;color:inherit">Neutral</div>
              <div style="font-size:10px;opacity:0.7">Range / volatility</div>
            </div>
            <div class="dir-count">4</div>
          </button>
        </div>
      </div>

      <div class="sidebar-section" id="strat-list-section" style="display:none">
        <div class="sidebar-label" id="strat-list-label">Strategies</div>
        <div class="strategy-list" id="strat-list"></div>
      </div>
    </aside>

    <!-- Main Content -->
    <main class="content">

      <!-- Welcome -->
      <div class="welcome" id="welcome-view">
        <div class="welcome-icon">📊</div>
        <h2>Options Strategy Builder</h2>
        <p>Select your market direction to explore strategies, visualise payoff diagrams, and plan your trades.</p>
        <div class="direction-cards">
          <div class="dir-card bull" onclick="selectDirection('bullish')">
            <div class="dir-card-icon">🟢</div>
            <div class="dir-card-label">Bullish</div>
            <div class="dir-card-count">4 strategies</div>
          </div>
          <div class="dir-card bear" onclick="selectDirection('bearish')">
            <div class="dir-card-icon">🔴</div>
            <div class="dir-card-label">Bearish</div>
            <div class="dir-card-count">3 strategies</div>
          </div>
          <div class="dir-card neutral" onclick="selectDirection('neutral')">
            <div class="dir-card-icon">⚪</div>
            <div class="dir-card-label">Neutral</div>
            <div class="dir-card-count">4 strategies</div>
          </div>
        </div>
      </div>

      <!-- Direction Overview (strategy cards) -->
      <div id="direction-view" style="display:none; animation: fadeIn 0.25s ease;">
        <div class="detail-header">
          <div class="breadcrumb">
            <span onclick="showWelcome()">Home</span>
            <span style="color:var(--border2)">›</span>
            <span id="dir-breadcrumb"></span>
          </div>
          <div class="detail-title" id="dir-title"></div>
          <div style="color:var(--muted);font-size:12px;margin-top:6px" id="dir-subtitle"></div>
        </div>
        <div class="strat-grid" id="dir-strat-grid"></div>
      </div>

      <!-- Strategy Detail -->
      <div id="detail-view" style="display:none;">
        <!-- populated dynamically -->
      </div>

    </main>
  </div>

  <footer>
    <span>OptionsCraft &copy; 2025 — For educational purposes only. Not financial advice.</span>
    <span style="color:var(--border2)">NSE API integration coming soon</span>
  </footer>
</div>

<script>
const DATA = {strategies_json};

let currentDir = null;
let currentStrat = null;
let payoffChart = null;

// ── Navigation ──────────────────────────────────────────────────────────────

function showWelcome() {{
  hide('welcome-view', false); show('welcome-view');
  hide('direction-view'); hide('detail-view');
  document.getElementById('strat-list-section').style.display = 'none';
  document.querySelectorAll('.dir-tab').forEach(t => t.classList.remove('active'));
  currentDir = null; currentStrat = null;
}}

function selectDirection(dir) {{
  currentDir = dir;
  currentStrat = null;
  const d = DATA[dir];

  // Update tabs
  document.querySelectorAll('.dir-tab').forEach(t => t.classList.remove('active'));
  const tab = document.getElementById('tab-' + dir);
  tab.classList.add('active');

  // Populate sidebar strategy list
  const list = document.getElementById('strat-list');
  list.innerHTML = '';
  d.strategies.forEach(s => {{
    const item = document.createElement('button');
    item.className = 'strat-item';
    item.id = 'si-' + s.id;
    item.onclick = () => selectStrategy(dir, s.id);
    let dots = '';
    for (let i = 1; i <= 4; i++) {{
      dots += `<div class="dot ${{i <= s.complexity ? 'filled' : ''}}"></div>`;
    }}
    item.innerHTML = `<span>${{s.name}}</span><div class="strat-complexity">${{dots}}</div>`;
    list.appendChild(item);
  }});
  document.getElementById('strat-list-label').textContent = d.label + ' Strategies';
  document.getElementById('strat-list-section').style.display = 'block';

  // Show direction overview
  hide('welcome-view'); hide('detail-view');
  show('direction-view');

  const dirMeta = {{
    bullish: {{ title: '🟢 Bullish Strategies', sub: 'Use when you expect the underlying asset\'s price to increase.' }},
    bearish: {{ title: '🔴 Bearish Strategies', sub: 'Use when you expect the underlying asset\'s price to decrease.' }},
    neutral: {{ title: '⚪ Neutral / Volatility Strategies', sub: 'Use when you expect the stock to stay in a range or make a large move in either direction.' }},
  }};
  document.getElementById('dir-breadcrumb').textContent = d.label;
  document.getElementById('dir-title').textContent = dirMeta[dir].title;
  document.getElementById('dir-subtitle').textContent = dirMeta[dir].sub;

  // Build strategy cards
  const grid = document.getElementById('dir-strat-grid');
  grid.innerHTML = '';
  d.strategies.forEach(s => {{
    const card = document.createElement('div');
    card.className = 'strat-card';
    card.onclick = () => selectStrategy(dir, s.id);
    let dots = '';
    for (let i = 1; i <= 4; i++) {{
      dots += `<div class="dot ${{i <= s.complexity ? 'filled' : ''}}" style="width:6px;height:6px;"></div>`;
    }}
    const riskClass = s.risk === 'Limited' ? 'bull' : 'bear';
    const rewClass = s.reward === 'Unlimited' ? 'bull' : (s.reward === 'Limited' ? 'muted' : 'neutral');
    card.innerHTML = `
      <div class="strat-card-name">${{s.name}}</div>
      <div class="strat-card-desc">${{s.description.substring(0,100)}}…</div>
      <div class="strat-card-meta">
        <span class="badge ${{riskClass}}">Risk: ${{s.risk}}</span>
        <span class="badge ${{rewClass}}">Reward: ${{s.reward}}</span>
        <span class="badge muted">IV: ${{s.iv_impact}}</span>
      </div>
      <div style="display:flex;gap:3px;margin-top:12px">${{dots}}</div>
    `;
    grid.appendChild(card);
  }});
}}

function selectStrategy(dir, stratId) {{
  currentDir = dir;
  currentStrat = stratId;

  const d = DATA[dir];
  const s = d.strategies.find(x => x.id === stratId);
  if (!s) return;

  // Update sidebar active
  document.querySelectorAll('.strat-item').forEach(el => el.classList.remove('active'));
  const si = document.getElementById('si-' + stratId);
  if (si) si.classList.add('active');

  hide('welcome-view'); hide('direction-view'); show('detail-view');

  renderDetail(dir, s);
}}

// ── Render Detail ───────────────────────────────────────────────────────────

function renderDetail(dir, s) {{
  const dirColor = {{ bullish: 'bull', bearish: 'bear', neutral: 'neutral' }}[dir];
  const dirLabel = {{ bullish: '🟢 Bullish', bearish: '🔴 Bearish', neutral: '⚪ Neutral' }}[dir];

  // Build legs HTML
  const legsHtml = s.legs.map(l => `
    <div class="leg">
      <span class="leg-action ${{l.action}}">${{l.action}}</span>
      <span class="leg-type">${{l.type}}</span>
      <span class="leg-strike">${{l.strike}}</span>
      <span class="leg-qty">x${{l.qty}}</span>
    </div>
  `).join('');

  const riskClass = s.risk === 'Limited' ? 'bull' : 'bear';
  const rewClass = s.reward === 'Unlimited' ? 'bull' : (s.reward === 'Limited' ? 'muted' : 'neutral');

  const dv = document.getElementById('detail-view');
  dv.style.display = 'block';
  dv.innerHTML = `
    <div class="detail-header">
      <div class="breadcrumb">
        <span onclick="showWelcome()">Home</span>
        <span style="color:var(--border2)">›</span>
        <span onclick="selectDirection('${{dir}}')">${{dirLabel}}</span>
        <span style="color:var(--border2)">›</span>
        <span style="color:var(--text)">${{s.name}}</span>
      </div>
      <div class="detail-title-row">
        <div>
          <div class="detail-title">${{s.name}}</div>
          <div class="detail-badges">
            <span class="badge ${{dirColor}}">${{dirLabel}}</span>
            <span class="badge ${{riskClass}}">Risk: ${{s.risk}}</span>
            <span class="badge ${{rewClass}}">Reward: ${{s.reward}}</span>
            <span class="badge info">IV: ${{s.iv_impact}}</span>
            <span class="badge muted">Best: ${{s.best_market}}</span>
          </div>
        </div>
      </div>
    </div>

    <div class="detail-grid">

      <!-- Description -->
      <div class="card full">
        <div class="card-title">Strategy Overview</div>
        <div class="desc-text">${{s.description}}</div>
      </div>

      <!-- Payoff Chart -->
      <div class="card">
        <div class="card-title">Payoff Diagram</div>
        <div class="chart-wrap">
          <canvas id="payoff-canvas"></canvas>
        </div>
      </div>

      <!-- P&L Summary -->
      <div class="card">
        <div class="card-title">P&amp;L Summary</div>
        <div class="metrics-grid">
          <div class="metric">
            <div class="metric-label">Max Profit</div>
            <div class="metric-value bull">${{s.max_profit}}</div>
          </div>
          <div class="metric">
            <div class="metric-label">Max Loss</div>
            <div class="metric-value bear">${{s.max_loss}}</div>
          </div>
          <div class="metric breakeven">
            <div class="metric-label">Breakeven Point(s)</div>
            <div class="metric-value neutral">${{s.breakeven}}</div>
          </div>
        </div>

        <!-- Strategy Legs -->
        <div style="margin-top:18px">
          <div class="card-title">Position Legs</div>
          <div class="legs-list">${{legsHtml}}</div>
        </div>
      </div>

    </div>

    <!-- Builder Panel -->
    <div class="builder-panel">
      <div class="builder-header">
        <div class="builder-title">Build This Strategy — NSE</div>
        <div class="coming-soon">⚡ NSE API Integration Coming Soon</div>
      </div>
      <div class="builder-body">
        <div class="form-row">
          <div class="form-group">
            <label>Underlying</label>
            <select class="form-control">
              <option>NIFTY</option>
              <option>BANKNIFTY</option>
              <option>FINNIFTY</option>
              <option>MIDCPNIFTY</option>
            </select>
          </div>
          <div class="form-group">
            <label>Expiry</label>
            <select class="form-control">
              <option>Weekly (nearest)</option>
              <option>Monthly</option>
              <option>Next Week</option>
            </select>
          </div>
          <div class="form-group">
            <label>Spot Price (LTP)</label>
            <input class="form-control" type="number" placeholder="e.g. 24500" readonly>
          </div>
          <div class="form-group">
            <label>Lots</label>
            <input class="form-control" type="number" value="1" min="1">
          </div>
          <div class="form-group" style="justify-content:flex-end;padding-top:16px">
            <button class="btn btn-ghost" onclick="alert('NSE API integration coming soon! You can add your NSE API endpoint to this project.')">
              🔗 Connect NSE API
            </button>
          </div>
        </div>
        <div style="font-size:11px;color:var(--muted);padding:12px;background:var(--surface);border-radius:8px;border:1px solid var(--border)">
          💡 <strong style="color:var(--text)">Next step:</strong> Provide your NSE option chain API endpoint and this panel will auto-populate strikes, LTP prices, and Greeks for <strong style="color:var(--text)">${{s.name}}</strong> in real-time.
        </div>
      </div>
    </div>
  `;

  // Render chart
  setTimeout(() => renderChart(dir, s), 50);
}}

// ── Payoff Chart ─────────────────────────────────────────────────────────────

function renderChart(dir, s) {{
  const canvas = document.getElementById('payoff-canvas');
  if (!canvas) return;
  if (payoffChart) {{ payoffChart.destroy(); payoffChart = null; }}

  const colorMap = {{ bullish: '#00e5a0', bearish: '#ff4560', neutral: '#f0b429' }};
  const col = colorMap[dir];
  const pts = s.payoff_pts;
  const labels = pts.map(p => (p[0] >= 0 ? '+' : '') + p[0] + '%');
  const values = pts.map(p => p[1]);

  // Build gradient
  const ctx = canvas.getContext('2d');
  const grad = ctx.createLinearGradient(0, 0, 0, 200);
  grad.addColorStop(0, col + '44');
  grad.addColorStop(1, col + '00');

  payoffChart = new Chart(canvas, {{
    type: 'line',
    data: {{
      labels,
      datasets: [{{
        label: 'P&L',
        data: values,
        borderColor: col,
        backgroundColor: grad,
        borderWidth: 2,
        pointRadius: 3,
        pointBackgroundColor: col,
        fill: true,
        tension: 0.35,
      }}]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      interaction: {{ intersect: false, mode: 'index' }},
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          backgroundColor: '#131920',
          borderColor: '#1e2830',
          borderWidth: 1,
          titleColor: '#cdd8e0',
          bodyColor: col,
          titleFont: {{ family: 'JetBrains Mono', size: 11 }},
          bodyFont: {{ family: 'JetBrains Mono', size: 12 }},
        }}
      }},
      scales: {{
        x: {{
          grid: {{ color: '#1e2830', drawBorder: false }},
          ticks: {{ color: '#5a7080', font: {{ family: 'JetBrains Mono', size: 10 }} }}
        }},
        y: {{
          grid: {{ color: '#1e2830', drawBorder: false }},
          ticks: {{
            color: '#5a7080',
            font: {{ family: 'JetBrains Mono', size: 10 }},
            callback: v => (v >= 0 ? '+' : '') + v
          }},
          // Zero line
          afterDataLimits: scale => {{
            scale.min = Math.min(scale.min, -2);
            scale.max = Math.max(scale.max, 2);
          }}
        }}
      }}
    }}
  }});
}}

// ── Helpers ──────────────────────────────────────────────────────────────────

function show(id) {{ document.getElementById(id).style.display = 'block'; }}
function hide(id)  {{ document.getElementById(id).style.display = 'none';  }}
</script>
</body>
</html>"""
    return html


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("docs", exist_ok=True)
    html = generate_html()
    out = os.path.join("docs", "index.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅  Generated: {out}")
    print(f"    Size: {len(html):,} bytes")
    print()
    print("Next steps:")
    print("  1. git add . && git commit -m 'feat: options strategy dashboard'")
    print("  2. git push origin main")
    print("  3. GitHub Actions will auto-deploy to GitHub Pages")
    print("  4. Access at: https://<your-username>.github.io/<repo-name>/")


if __name__ == "__main__":
    main()
