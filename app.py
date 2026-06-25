# app.py
"""
MarketMind Streamlit GUI.

Single-page interface that calls the FastAPI backend and renders results.

DESIGN PRINCIPLES:
- Clean, functional — not over-designed. The point is the data, not the styling.
- Every section maps to an architecture layer so users understand what they're
  seeing (agent swarm → synthesis → scoring → execution).
- Conviction score is displayed as a gauge with a clear color scale.
- Agent reports are expandable so the page isn't overwhelming by default.

RUNNING:
    # Start the FastAPI backend first:
    uvicorn api:app --reload --port 8000

    # Then in a separate terminal:
    streamlit run app.py
"""

import streamlit as st
import requests
import time

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="MarketMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# STYLING — minimal, purposeful
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #4361ee;
    }
    .signal-bullish { color: #2d6a4f; font-weight: 600; }
    .signal-bearish { color: #c1121f; font-weight: 600; }
    .signal-neutral { color: #6c757d; font-weight: 600; }
    .verdict-buy  { color: #2d6a4f; font-size: 1.4rem; font-weight: 700; }
    .verdict-sell { color: #c1121f; font-size: 1.4rem; font-weight: 700; }
    .verdict-hold { color: #e9a825; font-size: 1.4rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------

st.markdown('<p class="main-header">🧠 MarketMind</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Multi-agent AI trading analysis — '
    'quantitative math + LLM reasoning</p>',
    unsafe_allow_html=True
)

API_BASE = "http://localhost:8000"

# ---------------------------------------------------------------------------
# INPUT PANEL
# ---------------------------------------------------------------------------

col_input, col_spacer = st.columns([2, 3])

with col_input:
    with st.container(border=True):
        st.subheader("Analysis Parameters")

        ticker = st.text_input(
            "Ticker Symbol",
            value="AAPL",
            placeholder="AAPL / BTC-USD / EURUSD=X",
            help="Equities: AAPL, NVDA | Crypto: BTC-USD, ETH-USD | Forex: EURUSD=X"
        ).upper()

        asset_type = st.selectbox(
            "Asset Class",
            options=["equities", "crypto", "forex", "commodities"],
            help="Determines which data sources and agents are activated"
        )

        account_size = st.number_input(
            "Simulated Account Size (USD)",
            min_value=1_000.0,
            max_value=10_000_000.0,
            value=100_000.0,
            step=10_000.0,
            help="Used to calculate position size. This is paper trading only."
        )

        run_button = st.button(
            "🚀 Run Analysis",
            type="primary",
            use_container_width=True
        )

# ---------------------------------------------------------------------------
# ANALYSIS EXECUTION
# ---------------------------------------------------------------------------

if run_button:
    if not ticker:
        st.error("Please enter a ticker symbol.")
        st.stop()

    # Check API health first
    try:
        health = requests.get(f"{API_BASE}/health", timeout=3).json()
    except Exception:
        st.error(
            "⚠️ Cannot reach the MarketMind API. "
            "Start it with: `uvicorn api:app --reload --port 8000`"
        )
        st.stop()

    # Run the analysis
    with st.spinner(f"🔍 Deploying 5-agent swarm on {ticker}... (10–30 seconds)"):
        start = time.time()
        try:
            response = requests.post(
                f"{API_BASE}/analyze",
                json={
                    "ticker":     ticker,
                    "asset_type": asset_type,
                    "account_size": account_size,
                },
                timeout=120,
            )

            if response.status_code != 200:
                st.error(f"API error {response.status_code}: {response.text}")
                st.stop()

            data = response.json()
            elapsed = round(time.time() - start, 1)

        except requests.Timeout:
            st.error("Analysis timed out after 120 seconds. The LLM API may be slow.")
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.stop()

    st.success(f"✅ Analysis complete in {elapsed}s")

    # -----------------------------------------------------------------------
    # RESULTS PANEL
    # -----------------------------------------------------------------------

    st.divider()

    # --- Top-line verdict ---
    verdict       = data.get("final_verdict", "HOLD")
    conviction    = data.get("conviction_score", 0.0)
    regime        = data.get("market_regime", "UNKNOWN")
    confidence    = data.get("final_confidence", 0.0)

    verdict_color_class = {
        "BUY": "verdict-buy",
        "SELL": "verdict-sell",
        "HOLD": "verdict-hold"
    }.get(verdict, "verdict-hold")

    col_v, col_c, col_r, col_t = st.columns(4)

    with col_v:
        st.metric("Final Verdict", verdict)
    with col_c:
        score_display = f"{conviction:+.2f}" if conviction is not None else "N/A"
        st.metric("Conviction Score", score_display, help="-1.0 = max bearish, +1.0 = max bullish")
    with col_r:
        st.metric("Market Regime", regime.replace("_", " ").title() if regime else "Unknown")
    with col_t:
        st.metric("Run Time", f"{data.get('run_duration_seconds', 0):.1f}s")

    # Conviction score bar
    if conviction is not None:
        normalized = (conviction + 1.0) / 2.0   # -1→0 to +1→1 scale
        bar_color  = (
            "#2d6a4f" if conviction > 0.2
            else "#c1121f" if conviction < -0.2
            else "#6c757d"
        )
        st.markdown(f"""
        <div style="margin: 0.8rem 0 1.2rem 0;">
            <div style="
                background: #e9ecef;
                border-radius: 6px;
                height: 16px;
                position: relative;
            ">
                <div style="
                    background: {bar_color};
                    width: {normalized * 100:.1f}%;
                    height: 100%;
                    border-radius: 6px;
                "></div>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#888; margin-top:3px;">
                <span>−1.0 Bearish</span>
                <span>0 Neutral</span>
                <span>+1.0 Bullish</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Final reasoning
    reasoning = data.get("final_reasoning")
    if reasoning:
        st.info(f"**Synthesis Reasoning:** {reasoning}")

    st.divider()

    # --- Agent Reports ---
    st.subheader("Agent Swarm Reports")

    signal_emoji = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}
    agent_signals = data.get("agent_signals", [])

    if agent_signals:
        cols = st.columns(len(agent_signals))
        for i, sig in enumerate(agent_signals):
            with cols[i]:
                emoji  = signal_emoji.get(sig["signal"], "⚪")
                conf_pct = f"{sig['confidence'] * 100:.0f}%"
                with st.expander(
                    f"{emoji} **{sig['agent'].upper()}**  \n{sig['signal']} · {conf_pct}",
                    expanded=False
                ):
                    st.write(sig["summary"])
    else:
        st.warning("No agent signals returned.")

    st.divider()

    # --- Trade Ticket ---
    st.subheader("Trade Ticket")

    trade = data.get("proposed_trade")
    firewall_passed = data.get("firewall_passed")
    exec_status = data.get("execution_status", {})

    col_t1, col_t2, col_t3 = st.columns(3)

    with col_t1:
        if trade:
            action = trade.get("action", "HOLD")
            action_color = {
                "BUY": "green", "SELL_SHORT": "red", "HOLD": "orange"
            }.get(action, "gray")
            st.markdown(f"**Action:** :{action_color}[{action}]")
        else:
            st.write("**Action:** N/A")

    with col_t2:
        if trade:
            size = trade.get("position_size_usd", 0)
            st.metric("Position Size", f"${size:,.0f}")
        else:
            st.write("**Size:** N/A")

    with col_t3:
        if firewall_passed is True:
            st.success("🛡️ Firewall: PASSED")
        elif firewall_passed is False:
            st.error("🛡️ Firewall: BLOCKED")
        else:
            st.info("🛡️ Firewall: N/A")

    if trade and trade.get("reason"):
        st.caption(f"Reason: {trade['reason']}")

    exec_status_val = exec_status.get("status", "unknown") if exec_status else "unknown"
    st.caption(f"Execution status: `{exec_status_val}`")

    st.divider()

    # --- MLflow Link ---
    run_id = data.get("mlflow_run_id")
    if run_id:
        st.caption(
            f"📊 This run was logged to MLflow. "
            f"View at: [localhost:5000](http://localhost:5000) | Run ID: `{run_id}`"
        )

    # --- Raw JSON (for debugging / demo) ---
    with st.expander("🔍 Raw API Response (JSON)"):
        st.json(data)

# ---------------------------------------------------------------------------
# SIDEBAR — system info
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("System Status")

    try:
        health = requests.get(f"{API_BASE}/health", timeout=2).json()
        st.success("✅ API Online")
        memory = health.get("memory", {})
        st.write(f"Memory records: **{memory.get('total_records', 0)}**")
        st.write(f"Memory status: **{memory.get('status', 'unknown')}**")
        st.link_button("MLflow Dashboard", "http://localhost:5000")
        st.link_button("API Docs", "http://localhost:8000/docs")
    except Exception:
        st.error("❌ API Offline")
        st.write("Start: `uvicorn api:app --reload`")

    st.divider()
    st.caption(
        "**Architecture:** 5 parallel LLM agents → synthesis → "
        "quantitative scoring → risk firewall → trade ticket. "
        "Powered by LangGraph + Groq (Llama 3.3 70B)."
    )
