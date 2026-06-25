# core/tracking.py
"""
MLflow experiment tracking for MarketMind.

WHY MLFLOW:
Every run of the analysis pipeline produces a conviction score and trade
decision. Without logging, you have no way to answer:
  - Did the system consistently call Bitcoin bearish before it dropped?
  - What is the distribution of conviction scores over the last 90 days?
  - Did changing the regime weights improve or hurt signal quality?
  - How long does a full run take, and which agent is the bottleneck?

MLflow answers all of these by storing every run with its inputs, outputs,
parameters, and metrics in a queryable UI (localhost:5000 after `mlflow ui`).

DESIGN DECISION — why not just print to console:
Console output disappears. MLflow creates a permanent, searchable record.
When you're pitching this to an interviewer or investor, you can open the
MLflow UI and show 3 months of run history with conviction scores plotted
over time. That's real evidence. Print statements aren't.
"""

import time
import mlflow
from typing import Optional


# The experiment name groups all MarketMind runs together in the MLflow UI.
EXPERIMENT_NAME = "MarketMind-Analysis"


def start_run(ticker: str, asset_type: str) -> str:
    """
    Opens a new MLflow run and logs the input parameters.
    Returns the run_id so it can be stored in state and referenced later.
    
    Call this BEFORE invoking the LangGraph pipeline.
    """
    
    # End any stale run from a previous failed request
    if mlflow.active_run():
        mlflow.end_run()

    mlflow.set_experiment(EXPERIMENT_NAME)

    run = mlflow.start_run(run_name=f"{ticker}-{asset_type}-{int(time.time())}")

    # Log inputs as parameters (strings — MLflow params are always strings)
    mlflow.log_params({
        "ticker":     ticker,
        "asset_type": asset_type,
        "timestamp":  time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    })

    return run.info.run_id


def log_agent_signals(agent_signals: list) -> None:
    """
    Logs each agent's signal and confidence as MLflow metrics.
    
    We convert BULLISH/NEUTRAL/BEARISH to +1/0/-1 so they can be
    plotted as a numeric time series in the MLflow UI.
    """
    signal_to_int = {"BULLISH": 1, "NEUTRAL": 0, "BEARISH": -1}

    for sig in agent_signals:
        agent = sig.get("agent", "unknown")
        signal_val = signal_to_int.get(sig.get("signal", "NEUTRAL"), 0)
        confidence = sig.get("confidence", 0.0)

        mlflow.log_metrics({
            f"agent_{agent}_signal":     signal_val,
            f"agent_{agent}_confidence": confidence,
        })

        # Log any numeric raw_data fields (RSI, VIX, etc.)
        for key, value in sig.get("raw_data", {}).items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"agent_{agent}_{key}", float(value))


def log_final_result(state: dict) -> None:
    """
    Logs the full pipeline output at the end of a run.
    Call this AFTER the LangGraph pipeline completes.
    """
    # Conviction score and regime
    if state.get("conviction_score") is not None:
        mlflow.log_metric("conviction_score", state["conviction_score"])

    if state.get("market_regime"):
        mlflow.log_param("market_regime", state["market_regime"])

    # Synthesis verdict
    if state.get("final_verdict"):
        verdict_map = {"BUY": 1, "HOLD": 0, "SELL": -1}
        mlflow.log_params({
            "final_verdict":     state["final_verdict"],
            "final_reasoning":   (state.get("final_reasoning") or "")[:250],
        })
        mlflow.log_metric(
            "final_verdict_numeric",
            verdict_map.get(state.get("final_verdict", "HOLD"), 0)
        )

    # Trade details
    if state.get("proposed_trade"):
        trade = state["proposed_trade"]
        mlflow.log_params({
            "trade_action": trade.get("action", "UNKNOWN"),
        })
        mlflow.log_metric(
            "position_size_usd",
            trade.get("position_size_usd", 0.0)
        )

    # Firewall result
    if state.get("firewall_passed") is not None:
        mlflow.log_metric("firewall_passed", int(state["firewall_passed"]))

    # Latency: if run_start_time was recorded, log total duration
    if state.get("run_start_time"):
        duration = time.time() - state["run_start_time"]
        mlflow.log_metric("total_run_seconds", round(duration, 2))


def end_run() -> None:
    """Closes the active MLflow run. Always call this in a finally block."""
    mlflow.end_run()


def get_experiment_url() -> str:
    """Returns the local MLflow UI URL for this experiment."""
    return f"http://localhost:5000/#/experiments/{EXPERIMENT_NAME}"
