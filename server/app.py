from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

try:
    from ..models import PortfolioRiskAction, PortfolioRiskObservation
    from ..env import PortfolioRiskEnv, PortfolioAction
except (ImportError, ValueError):
    from models import PortfolioRiskAction, PortfolioRiskObservation
    from env import PortfolioRiskEnv, PortfolioAction


app = FastAPI(
    title="Portfolio Risk Advisor",
    description="RL environment for AI-driven portfolio risk management. Three tasks: allocation_check (easy), risk_rebalancing (medium), stress_test_optimization (hard).",
    version="1.0.0",
)

_envs: Dict[str, PortfolioRiskEnv] = {}
VALID_TASKS = ["allocation_check", "risk_rebalancing", "stress_test_optimization"]


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(task: str = "allocation_check") -> Dict[str, Any]:
    if task not in VALID_TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task '{task}'. Valid: {VALID_TASKS}")
    env = PortfolioRiskEnv(task=task)
    obs = env.reset()
    _envs["current"] = env
    return obs.model_dump()


@app.post("/step")
def step(action: PortfolioRiskAction) -> Dict[str, Any]:
    env = _envs.get("current")
    if env is None:
        raise HTTPException(status_code=400, detail="No active session — call /reset first.")

    standalone_action = PortfolioAction(
        action_type=action.action_type,
        ticker=action.ticker,
        target_weight=action.target_weight,
        reasoning=action.reasoning,
    )

    try:
        obs, reward, done, info = env.step(standalone_action)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"observation": obs.model_dump(), "reward": reward, "done": done, "info": info}


@app.get("/state")
def state() -> Dict[str, Any]:
    env = _envs.get("current")
    return env.state() if env else {}


@app.get("/schema")
def schema() -> Dict[str, Any]:
    return {
        "action": PortfolioRiskAction.model_json_schema(),
        "observation": PortfolioRiskObservation.model_json_schema(),
        "tasks": [
            {"name": "allocation_check", "difficulty": "easy", "max_steps": 5,
             "description": "Ensure no single asset exceeds 35% portfolio weight."},
            {"name": "risk_rebalancing", "difficulty": "medium", "max_steps": 10,
             "description": "Reduce portfolio volatility below 25% while fixing weight violations."},
            {"name": "stress_test_optimization", "difficulty": "hard", "max_steps": 15,
             "description": "Minimize simulated loss under a market shock while satisfying all constraints."},
        ],
    }


def main(host: str = "0.0.0.0", port: int = 7860):
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(host=args.host, port=args.port)
