import os
import csv
from datetime import date, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from dotenv import load_dotenv
import time
import random

# ========= 加载 .env =========
load_dotenv()

# ========= 设置 Alpha Vantage API Key =========
ALPHA_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# ========= 设置代理 OpenAI Key 和 Backend URL =========
PROXY_KEY = os.getenv("OPENAI_API_KEY")
PROXY_BASE = os.getenv("CUSTOM_API_BASE")

# ========= 股票池和账户 =========
TICKERS = ["AAPL", "NVDA", "MSFT", "META", "GOOGL"]
INITIAL_CAPITAL = 10000.0
RECORD_FILE = "multi_stock_performance.csv"
SUMMARY_FILE = "daily_summary.csv"

# ========= 初始化 TradingAgents =========
config = DEFAULT_CONFIG.copy()
# config["deep_think_llm"] = "Qwen3-Coder-480B-A35B-Instruct-FP8"
# config["quick_think_llm"] = "Qwen3-Coder-480B-A35B-Instruct-FP8"
config["deep_think_llm"] = "Qwen3-Coder-30B-A3B-Instruct-FP8"
config["quick_think_llm"] = "Qwen3-Coder-30B-A3B-Instruct-FP8"
config["backend_url"] = PROXY_BASE
config["max_debate_rounds"] = 1
config["data_vendors"] = {
    "core_stock_apis": "yfinance",
    "technical_indicators": "yfinance",
    "fundamental_data": "alpha_vantage",
    "news_data": "alpha_vantage",
}

ta = TradingAgentsGraph(debug=False, config=config)

# ========= 初始化账户 =========
accounts = {}
if os.path.exists(RECORD_FILE):
    df_prev = pd.read_csv(RECORD_FILE)
    for t in TICKERS:
        df_t = df_prev[df_prev["ticker"] == t]
        if not df_t.empty:
            last_row = df_t.iloc[-1]
            accounts[t] = {
                "cash": float(last_row["cash"]),
                "position": float(last_row["position"]),
                "in_position": last_row["in_position"] == True,
                "total_value": float(last_row["total_value"]),
            }
        else:
            accounts[t] = {"cash": INITIAL_CAPITAL, "position": 0, "in_position": False, "total_value": INITIAL_CAPITAL}
else:
    for t in TICKERS:
        accounts[t] = {"cash": INITIAL_CAPITAL, "position": 0, "in_position": False, "total_value": INITIAL_CAPITAL}

# ========= 获取今日日期 =========
# today = str(date.today() - timedelta(days=1))
today = date(2025, 11, 3).strftime("%Y-%m-%d")
records = []

# ========= 遍历每只股票 =========
for ticker in TICKERS:
    print(f"\n=== {ticker} ===")

    # 获取当前价格
    data = yf.download(ticker, period="5d", interval="1d")
    if today not in data.index.strftime("%Y-%m-%d"):
        print(f"⚠️ 今天 {today} {ticker} 无数据，跳过。")
        continue

    price = float(data.loc[data.index.strftime("%Y-%m-%d") == today]["Close"].values[0])
    acc = accounts[ticker]

    # 调用 TA 内部 propagate（不要传额外参数）
    try:
        _, decision = ta.propagate(ticker, today)
    except Exception as e:
        print(f"⚠️ 股票 {ticker} 决策调用失败: {e}")
        decision = "HOLD"

    # 执行交易逻辑
    if decision == "BUY" and not acc["in_position"]:
        acc["position"] = acc["cash"] / price
        acc["cash"] = 0
        acc["in_position"] = True
    elif decision == "SELL" and acc["in_position"]:
        acc["cash"] = acc["position"] * price
        acc["position"] = 0
        acc["in_position"] = False

    # 更新账户
    acc["total_value"] = acc["cash"] + acc["position"] * price
    pnl_pct = (acc["total_value"] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    records.append({
        "date": today,
        "ticker": ticker,
        "decision": decision,
        "price": price,
        "cash": acc["cash"],
        "position": acc["position"],
        "in_position": acc["in_position"],
        "total_value": acc["total_value"],
        "pnl_pct": pnl_pct,
    })

    sleep_time = random.uniform(8, 10)
    print(f"💤 等待 {sleep_time:.1f} 秒再处理下一支股票…")
    time.sleep(sleep_time)

# ========= 保存每只股票记录 =========
file_exists = os.path.exists(RECORD_FILE)
with open(RECORD_FILE, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=records[0].keys())
    if not file_exists:
        writer.writeheader()
    writer.writerows(records)

# ========= 计算并保存每日汇总指标 =========
df = pd.read_csv(RECORD_FILE)
summary_records = []

print("\n📊 === 多股票策略绩效 ===")
for t in TICKERS:
    df_t = df[df["ticker"] == t]
    if len(df_t) < 2:
        continue
    returns = df_t["total_value"].pct_change().fillna(0)
    annual_return = (1 + returns.mean())**252 - 1
    annual_vol = returns.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol != 0 else 0
    cum_max = df_t["total_value"].cummax()
    drawdown = (df_t["total_value"] - cum_max) / cum_max
    max_dd = drawdown.min() * 100
    total_pnl = df_t["pnl_pct"].iloc[-1]

    print(f"{t}: AnnRet={annual_return*100:.2f}%, Sharpe={sharpe:.2f}, MaxDD={max_dd:.2f}%, TotalPnL={total_pnl:.2f}%")

    summary_records.append({
        "date": today,
        "ticker": t,
        "annual_return": annual_return * 100,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "total_pnl": total_pnl
    })

# ========= 计算组合平均绩效 =========
if summary_records:
    ann_ret_mean = np.mean([r["annual_return"] for r in summary_records])
    sharpe_mean = np.mean([r["sharpe"] for r in summary_records])
    max_dd_mean = np.mean([r["max_drawdown"] for r in summary_records])
    pnl_mean = np.mean([r["total_pnl"] for r in summary_records])
    summary_records.append({
        "date": today,
        "ticker": "PORTFOLIO_AVG",
        "annual_return": ann_ret_mean,
        "sharpe": sharpe_mean,
        "max_drawdown": max_dd_mean,
        "total_pnl": pnl_mean
    })
    print(f"\n📈 Portfolio Avg: AnnRet={ann_ret_mean:.2f}%, Sharpe={sharpe_mean:.2f}, MaxDD={max_dd_mean:.2f}%, TotalPnL={pnl_mean:.2f}%")

# ========= 保存汇总文件 =========
file_exists_summary = os.path.exists(SUMMARY_FILE)
with open(SUMMARY_FILE, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary_records[0].keys())
    if not file_exists_summary:
        writer.writeheader()
    writer.writerows(summary_records)

print("==============================\n")
print(f"📁 已保存每日结果到 {RECORD_FILE}")
print(f"📊 已保存每日绩效汇总到 {SUMMARY_FILE}\n")
