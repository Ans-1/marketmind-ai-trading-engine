# MarketMind 🧠📈

MarketMind is an autonomous, multi-asset algorithmic trading engine. It merges deterministic quantitative mathematics with Large Language Models (LLMs) to scan global markets, evaluate complex macroeconomic narratives, manage dynamic risk, and execute simulated trades.

Unlike standard "AI trading bots" that rely on LLMs to guess support/resistance levels, MarketMind utilizes **"The Great Divide"** architecture: purely mathematical engines handle price action, volatility, and position sizing, while AI agents are strictly reserved for reading unstructured data (news sentiment and macroeconomic narratives).

## 🏗️ Core Architecture

The system operates as a multi-stage quantitative funnel:

1. **The Omni-Screener:** Scans a global universe of Equities, Crypto, Forex, and Commodities. Uses Z-Score volatility normalization and volume ratios to find the highest momentum setups.
2. **The Pre-Flight Check (Early-Kill):** Hard mathematical checks prevent the expensive AI swarm from deploying on sideways or low-volume assets, saving API tokens and compute time.
3. **The Great Divide (Agent Swarm):**
   * 🧮 *Quantitative Pillar:* Python-native Price, Risk, and On-Chain engines calculate deterministic technical scores.
   * 🤖 *Qualitative Pillar:* LLM-powered Sentiment and Macro agents read news headlines and economic data to gauge narrative.
4. **The Scoring Engine:** Blends mathematical technicals with AI sentiment to output a final `Conviction Score` (-1.0 to 1.0).
5. **The Portfolio Manager (Sniper Sizing):** Uses Average True Range (ATR) to dynamically calculate Stop-Loss and Take-Profit distances, sizing positions strictly based on fractional risk parameters (e.g., risking exactly 1% of account equity).
6. **Execution Engine:** Translates the approved trade into a structured JSON Trade Ticket for broker routing.

## ⚙️ Tech Stack

* **Language:** Python 3
* **AI Orchestration:** LangGraph / LangChain
* **LLM Provider:** Groq API (Llama-3-70b-versatile)
* **Data Pipelines:** `yfinance` (Equities/Forex/Commodities), `pycoingecko` (Crypto)
* **Dependency Management:** `uv`

## 🚀 Usage & CLI Commands

MarketMind features a dynamic routing layer, allowing you to operate as a surgical sniper or a global radar.

### Omni Mode (Global Radar)
Scans the entire asset universe, finds the top 3 momentum setups globally, and deploys the AI swarm to allocate capital to the absolute winner.

    uv run integration_testing.py

### Category Mode (Sector Scan)
Restricts the Omni-Screener to a specific asset class.

    uv run integration_testing.py --category crypto

### Explicit Mode (Sniper)
Bypasses the global screener and targets a specific ticker. (Includes the Pre-Flight momentum check to ensure the asset is actually moving before waking up the AI).

    uv run integration_test.py --ticker AAPL --type equities

## 📂 Project Structure

    marketmind/
    ├── .env                    # API Keys (GIT IGNORED)
    ├── pyproject.toml          # Dependencies
    ├── unit_test.py            # Individual agent tests
    ├── integration_test.py     # CLI entry point and orchestrator
    └── core/
        ├── state.py            # LangGraph State definition
        ├── graph.py            # Node and Edge wiring
        ├── portfolio_manager.py# ATR math and position sizing
        ├── risk_firewall.py    # VIX overrides and drawdown limits
        ├── execution_engine.py # Simulated broker routing
        ├── regime_detector.py  # Volatility environment classification
        ├── scoring_engine.py   # Math/AI conviction blender
        ├── agents/             # The Divided Swarm (Math + LLMs)
        └── screener/           # The Omni-Screener Pipeline

## 🔒 Security & Environment Variables

Create a `.env` file in the root directory. **Never commit this file to version control.**

    GROQ_API_KEY=your_groq_api_key_here
    ALPACA_API_KEY=your_alpaca_key_here
    ALPACA_SECRET_KEY=your_alpaca_secret_here

## 🗺️ Engineering Roadmap

- [x] Multi-Agent Orchestration (LangGraph)
- [x] Omni-Screener & Z-Score Normalization
- [x] Risk Firewall & Dynamic Position Sizing (ATR)
- [x] The Great Divide Architecture (Conceptualized)
- [ ] **Refactor Agents:** Rewrite Price, Risk, and On-chain agents from LLM calls to pure Python math (pandas/numpy).
- [ ] **Live Execution:** Integrate `alpaca-py` SDK for live paper trading of generated Trade Tickets.
- [ ] **Backtesting Environment:** Build the Point-in-Time Data Router to simulate historical trades without lookahead bias.
- [ ] **Episodic Memory:** Add persistent vector storage (e.g., ChromaDB) so the AI remembers past trade successes/failures.

---
*Disclaimer: MarketMind is currently a simulated engine. Algorithmic trading carries significant financial risk. This software is for educational and research purposes.*