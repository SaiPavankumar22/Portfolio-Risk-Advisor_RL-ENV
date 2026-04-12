"""
app.py
======
FastAPI server exposing the OpenEnv API endpoints AND a Gradio interactive
UI mounted at /ui. The root / redirects to /ui so HF Spaces shows the
beautiful Gradio interface by default.

OpenEnv API endpoints (used by validators and agents):
    POST /reset?task=<name>
    POST /step
    GET  /state
    GET  /health
    GET  /tasks
    GET  /schema
    GET  /docs  (Swagger)
"""

import io
import base64
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import uvicorn
import gradio as gr
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, RedirectResponse

from models import PortfolioRiskAction, PortfolioRiskObservation
from env import PortfolioRiskEnv, PortfolioAction

app = FastAPI(
    title="Portfolio Risk Advisor",
    description=(
        "OpenEnv RL environment for AI-driven portfolio risk management. "
        "Four tasks: allocation_check (easy), risk_rebalancing (medium), "
        "stress_test_optimization (hard), regime_shift_adaptation (hard+). "
        "Interactive UI available at /ui."
    ),
    version="1.0.0",
)

_envs: Dict[str, PortfolioRiskEnv] = {}

VALID_TASKS = [
    "allocation_check",
    "risk_rebalancing",
    "stress_test_optimization",
    "regime_shift_adaptation",
]

@app.get("/health")
def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/reset")
def reset(
    task: str = Query("allocation_check", description="Task name")
) -> Dict[str, Any]:
    """Start new episode."""
    if task not in VALID_TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{task}'. Valid tasks: {VALID_TASKS}",
        )
    env = PortfolioRiskEnv(task=task)
    obs = env.reset()
    _envs["current"] = env
    return obs.model_dump()


@app.post("/step")
def step(action: PortfolioRiskAction) -> Dict[str, Any]:
    """Execute action step."""
    env = _envs.get("current")
    if env is None:
        raise HTTPException(
            status_code=400,
            detail="No active session — call POST /reset first.",
        )
    standalone = PortfolioAction(
        action_type=action.action_type,
        ticker=action.ticker,
        target_weight=action.target_weight,
        reasoning=action.reasoning,
    )
    try:
        obs, reward, done, info = env.step(standalone)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "observation": obs.model_dump(),
        "reward":      reward,
        "done":        done,
        "info":        info,
    }


@app.get("/state")
def state() -> Dict[str, Any]:
    """Get internal environment state."""
    env = _envs.get("current")
    return env.state() if env else {}


@app.get("/tasks")
def tasks() -> Dict[str, Any]:
    """List all available tasks with metadata."""
    return {
        "tasks": [
            {
                "name":        "allocation_check",
                "difficulty":  "easy",
                "max_steps":   5,
                "description": "Ensure no single asset exceeds 35% portfolio weight.",
            },
            {
                "name":        "risk_rebalancing",
                "difficulty":  "medium",
                "max_steps":   10,
                "description": "Reduce full-covariance portfolio volatility below 25%.",
            },
            {
                "name":        "stress_test_optimization",
                "difficulty":  "hard",
                "max_steps":   15,
                "description": "Minimize correlated market-shock losses while satisfying constraints.",
            },
            {
                "name":        "regime_shift_adaptation",
                "difficulty":  "hard+",
                "max_steps":   20,
                "description": "Adapt portfolio when market shifts from normal to crisis mid-episode.",
            },
        ]
    }


@app.get("/schema")
def schema() -> Dict[str, Any]:
    """JSON schemas for action and observation spaces."""
    return {
        "action":      PortfolioRiskAction.model_json_schema(),
        "observation": PortfolioRiskObservation.model_json_schema(),
    }


@app.get("/portfolio_plot")
def portfolio_plot() -> JSONResponse:
    """Get base64-encoded portfolio allocation pie chart."""
    env = _envs.get("current")
    if env is None:
        raise HTTPException(status_code=400, detail="No active session — call /reset first.")

    s       = env.state()
    weights = s.get("weights", {})
    if not weights:
        raise HTTPException(status_code=400, detail="Empty portfolio state.")

    tickers = list(weights.keys())
    values  = [weights[t] * 100 for t in tickers]
    colors  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
               "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(values, labels=tickers, autopct="%1.1f%%",
           colors=colors[:len(tickers)], startangle=90,
           wedgeprops={"linewidth": 1.5, "edgecolor": "white"})
    ax.set_title("Portfolio Allocation")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")

    return JSONResponse(content={"image_base64": img_b64, "format": "png"})


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/ui")


try:
    from gradio_ui import create_demo
    _demo = create_demo()
    gr.mount_gradio_app(app, _demo, path="/ui")
except Exception as _e:
    import sys
    print(f"[WARNING] Gradio UI could not be loaded: {_e}", file=sys.stderr)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
