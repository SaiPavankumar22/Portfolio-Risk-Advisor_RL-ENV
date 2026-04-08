---
title: Portfolio Risk Advisor
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
  - finance
  - portfolio
  - risk-management
---

# Portfolio Risk Advisor

An OpenEnv reinforcement-learning environment where an AI agent acts as a quantitative
portfolio manager. Given a portfolio of real-world stock tickers with weights, prices,
volatilities and correlations, the agent must rebalance the portfolio to satisfy
risk constraints — simulating a task financial analysts do every day.

## Motivation

Portfolio risk management is a genuine, high-stakes real-world task. Human portfolio
managers must continuously rebalance multi-asset portfolios against:
- Single-stock concentration limits
- Volatility targets
- Stress-test survival requirements

This environment lets RL/LLM agents practice exactly these decisions under realistic,
randomly generated market scenarios — making it immediately useful for evaluating
agent judgment in financial domains.

---

## Tasks

| Task | Difficulty | Max Steps | Objective |
|---|---|---|---|
| `allocation_check` | Easy | 5 | Ensure no single asset exceeds 35% portfolio weight |
| `risk_rebalancing` | Medium | 10 | Reduce portfolio volatility below 25% while fixing weight violations |
| `stress_test_optimization` | Hard | 15 | Minimize simulated market-shock loss while satisfying all constraints |

### Grading

All graders return a score in **[0.0, 1.0]** — higher is better.

**allocation_check**
```
score = max(0, 1.0 - violations×0.3 - max(0, top_weight - 0.35)×2)
```

**risk_rebalancing**
```
vol_score       = max(0, 1 - max(0, portfolio_vol - 0.25) / 0.25)
score           = 0.6 × vol_score + 0.4 × allocation_score
```

**stress_test_optimization**
```
simulated_loss  = Σ weight[t] × vol[t] × 1.5  for t where vol[t] > 0.30
survival_score  = max(0, 1 - simulated_loss / 0.15)
score           = 0.5 × survival_score + 0.5 × rebalancing_score
```

An episode ends when `reward >= 0.95` (success) or `max_steps` is reached.

---

## Action Space

**Type:** `text` (structured JSON)

| Field | Type | Required | Description |
|---|---|---|---|
| `action_type` | str | ✅ | `"rebalance"` \| `"hold"` \| `"reduce"` \| `"increase"` |
| `ticker` | str | for rebalance/reduce/increase | Ticker symbol, e.g. `"AAPL"` |
| `target_weight` | float [0,1] | for rebalance | Desired portfolio weight |
| `reasoning` | str | optional | Agent's explanation |

**Example — rebalance:**
```json
{
  "action_type": "rebalance",
  "ticker": "TSLA",
  "target_weight": 0.15,
  "reasoning": "TSLA exceeds 35% cap; reducing to 15% to comply"
}
```

**Example — hold:**
```json
{
  "action_type": "hold",
  "reasoning": "All constraints satisfied, portfolio is compliant"
}
```

## Observation Space

**Type:** `structured` (JSON)

| Field | Type | Description |
|---|---|---|
| `task` | str | Current task name |
| `holdings` | Dict[str, float] | Portfolio weights, sum ≈ 1.0 |
| `prices` | Dict[str, float] | Current price per ticker |
| `volatilities` | Dict[str, float] | Annualized volatility per ticker |
| `correlations` | Dict[str, Dict[str, float]] | Pairwise correlation matrix |
| `constraints` | Dict[str, float] | `max_single_weight`, `max_portfolio_vol`, `min_assets` |
| `step_number` | int | Current step in the episode |
| `last_action_error` | str \| null | Error from the previous action, if any |

---

## HTTP API

| Endpoint | Method | Description |
|---|---|---|
| `/reset?task=<name>` | POST | Start a new episode |
| `/step` | POST | Execute an action, get obs/reward/done |
| `/state` | GET | Raw internal state |
| `/health` | GET | Liveness probe |
| `/schema` | GET | JSON schemas for action & observation |
| `/docs` | GET | Swagger UI |

### Quick test

```bash
# Reset (easy task)
curl -X POST "http://localhost:7860/reset?task=allocation_check"

# Take an action
curl -X POST "http://localhost:7860/step" \
  -H "Content-Type: application/json" \
  -d '{"action_type":"rebalance","ticker":"AAPL","target_weight":0.20,"reasoning":"reducing overweight"}'
```

---

## Setup & Usage

### Run with Docker

```bash
# Build
docker build -t portfolio-risk-env .

# Run
docker run -p 7860:7860 portfolio-risk-env
```

### Run locally

```bash
pip install -r server/requirements.txt

# Start server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Or via pyproject.toml entry point (requires uv)
uv run --project . server
```

### Run inference (requires HF token)

```bash
export HF_TOKEN=your_hf_token_here
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct   # optional, has default
python inference.py
```

Expected output:
```
[START] task=allocation_check env=portfolio_risk_env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=rebalance(TSLA,0.15) reward=0.72 done=false error=null
[STEP] step=2 action=hold reward=0.95 done=true error=null
[END] success=true steps=2 score=0.95 rewards=0.72,0.95

[START] task=risk_rebalancing ...
...
```

---

## Baseline Scores

Scores obtained with `Qwen/Qwen2.5-72B-Instruct` via HuggingFace router:

| Task | Avg Score | Success Rate |
|---|---|---|
| `allocation_check` | ~0.75 | ~70% |
| `risk_rebalancing` | ~0.55 | ~40% |
| `stress_test_optimization` | ~0.35 | ~15% |

*(Baseline will vary with random scenario seeds)*

---

## Project Structure

```
portfolio_risk_env/
├── env.py                 # Standalone environment (used by inference.py)
├── inference.py           # Baseline inference script (MANDATORY, at root)
├── models.py              # OpenEnv framework Pydantic models
├── client.py              # HTTP/WebSocket client
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata + dependencies
├── README.md              # This file
└── server/
    ├── app.py             # FastAPI HTTP server
    ├── portfolio_risk_env_environment.py  # OpenEnv Environment class
    ├── requirements.txt   # Server dependencies
    └── Dockerfile         # Container image definition
```

---

## Reward Signal

The reward is **dense** (non-sparse) — the grader runs after every step and
returns a score reflecting the current portfolio state, not just binary end-of-episode.
This means:

- The agent gets incremental feedback as it reduces violations
- Sub-optimal actions (e.g. wrong ticker) still receive partial credit if other constraints improve
- Invalid actions incur a −0.05 penalty

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | ✅ | — | Hugging Face / API key |
| `API_BASE_URL` | ❌ | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | ❌ | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |

