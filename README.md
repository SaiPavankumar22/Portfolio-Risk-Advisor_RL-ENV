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

# 📈 Portfolio Risk Advisor

## 🚀 Why This Environment Matters

- **Real-world task** — Portfolio rebalancing is performed daily by hedge funds, pension funds, and robo-advisors. This is not a toy problem.
- **Tests LLM reasoning under constraints** — The agent must read market data, identify violations, and explain its reasoning — exactly what a quant analyst does.
- **Combines optimization + decision-making under uncertainty** — Especially in the regime-shift task, the agent must anticipate and react to sudden market dislocations.
- **Novel 4th task** — Market regime adaptation is not found in other OpenEnv environments. It directly addresses the real challenge of managing portfolios through crisis periods.

---

## Overview

An OpenEnv RL environment where an AI agent acts as a quantitative portfolio manager.
Given a portfolio of real-world stock tickers with weights, prices, volatilities, and
a full correlation matrix, the agent rebalances under risk constraints — simulating
what financial analysts do every day.

### Key technical features

- **Full covariance matrix** for portfolio volatility (not the diagonal approximation)
- **Transaction costs** (10 bps per unit turnover) penalize over-trading
- **Correlated crash simulation** in stress test (market factor + idiosyncratic components)
- **Dynamic regime shift** (task 4): market transitions from normal → crisis mid-episode
- **Reasoning bonus**: agents that explain their actions with relevant financial keywords receive a small reward bonus
- **Agent memory**: `previous_rewards` field lets the model reason about its own trajectory

---

## Tasks

| Task | Difficulty | Max Steps | Objective |
|---|---|---|---|
| `allocation_check` | 🟢 Easy | 5 | Fix concentration violations (weight > 35%) |
| `risk_rebalancing` | 🟡 Medium | 10 | Reduce full-covariance vol below 25% + fix weights |
| `stress_test_optimization` | 🔴 Hard | 15 | Survive correlated market crash while staying compliant |
| `regime_shift_adaptation` | ⚡ Hard+ | 20 | Adapt portfolio when market shifts from normal to crisis |

### Grading formulas

All scores are clamped to **(0.01, 0.99)**.

**allocation_check**
```
violations = count(weight > 0.35)
excess     = max(0, top_weight - 0.35)
score      = clamp(1.0 - violations×0.25 - excess×2.0)
```

**risk_rebalancing** *(uses full covariance matrix)*
```
port_var   = Σᵢ Σⱼ wᵢ wⱼ σᵢ σⱼ ρᵢⱼ     ← full covariance
vol_score  = max(0, 1 - max(0, port_vol - 0.25) / 0.25)
score      = clamp(0.6 × vol_score + 0.4 × allocation_score)
```

**stress_test_optimization** *(correlated crash)*
```
market_shock   = -20% drawdown
portfolio_loss = Σ wᵢ × σᵢ × (avg_corr × |shock| + (1-avg_corr) × 0.5)
survival_score = max(0, 1 - loss / 0.15)
score          = clamp(0.5 × survival + 0.4 × rebalancing - concentration_penalty)
```

**regime_shift_adaptation**
```
Normal phase:  score = 0.4×vol_score + 0.3×alloc_score + 0.3×diversification
Crisis phase:  score = 0.4×crisis_vol_survival + 0.3×diversification + 0.3×alloc_score
```

An episode ends early (success) when `reward >= 0.90`, or when `max_steps` is reached.

### Reward shaping details

| Signal | Value | Trigger |
|---|---|---|
| Task grader score | 0.01–0.99 | Every step |
| Transaction cost | −0.1% × turnover | When weights change |
| Reasoning bonus | +0.01 per keyword (max +0.04) | Relevant financial terms in reasoning |
| Invalid action penalty | −0.02 to −0.05 | Bad ticker, out-of-range weight |

---

## Action Space

**Type:** `text` (structured JSON)

| Field | Type | Required | Description |
|---|---|---|---|
| `action_type` | str | ✅ | `"rebalance"` \| `"hold"` \| `"reduce"` \| `"increase"` |
| `ticker` | str | for rebalance/reduce/increase | Ticker symbol, e.g. `"AAPL"` |
| `target_weight` | float [0,1] | for rebalance | Desired portfolio weight |
| `reasoning` | str | optional | Agent explanation — earns small bonus |

---

## Observation Space

**Type:** `structured` (JSON)

| Field | Type | Description |
|---|---|---|
| `task` | str | Current task name |
| `holdings` | Dict[str, float] | Portfolio weights, sum ≈ 1.0 |
| `prices` | Dict[str, float] | Current price per ticker (USD) |
| `volatilities` | Dict[str, float] | Annualized volatility per ticker |
| `correlations` | Dict[str, Dict[str, float]] | Full pairwise correlation matrix |
| `constraints` | Dict[str, float] | `max_single_weight`, `max_portfolio_vol`, `min_assets` |
| `step_number` | int | Current step in the episode |
| `last_action_error` | str \| null | Error from the previous action, if any |
| `previous_rewards` | List[float] | Last 5 step rewards (agent memory) |
| `regime` | str | Market regime: `"normal"` or `"crisis"` |

---

## HTTP API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Redirects to interactive Gradio UI |
| `/ui` | GET | Interactive Gradio interface |
| `/reset?task=<n>` | POST | Start a new episode |
| `/step` | POST | Execute one action |
| `/state` | GET | Raw internal environment state |
| `/tasks` | GET | List all tasks |
| `/schema` | GET | JSON schemas for action + observation |
| `/portfolio_plot` | GET | Base64 portfolio pie chart |
| `/health` | GET | Liveness probe |
| `/docs` | GET | Swagger UI |

---

## Setup & Usage

### Run with Docker

```bash
docker build -t portfolio-risk-env .
docker run -p 7860:7860 portfolio-risk-env

# Test
curl http://localhost:7860/health
curl -X POST "http://localhost:7860/reset?task=allocation_check"
```

### Run locally

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Run inference

```bash
export HF_TOKEN=your_token_here
python inference.py
```

Expected output (all 4 tasks):
```
[START] task=allocation_check env=portfolio_risk_env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=rebalance(TSLA,0.35) reward=0.76 done=false error=null
[STEP] step=2 action=hold reward=0.94 done=true error=null
[END] success=true steps=2 score=0.94 rewards=0.76,0.94

[START] task=risk_rebalancing env=portfolio_risk_env model=Qwen/Qwen2.5-72B-Instruct
...
[START] task=regime_shift_adaptation env=portfolio_risk_env model=Qwen/Qwen2.5-72B-Instruct
...
```

---

## Baseline Scores

Measured with `Qwen/Qwen2.5-72B-Instruct` via HuggingFace router (seed=42):

| Task | Avg Score | Success Rate |
|---|---|---|
| `allocation_check` | ~0.84 | ~85% |
| `risk_rebalancing` | ~0.65 | ~55% |
| `stress_test_optimization` | ~0.48 | ~30% |
| `regime_shift_adaptation` | ~0.42 | ~20% |

---

## Project Structure

```
.
├── Dockerfile                          # Container definition
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── openenv.yaml                        # OpenEnv manifest (4 tasks, entry_point)
├── app.py                              # FastAPI server + Gradio mount
├── gradio_ui.py                        # Interactive Gradio interface
├── env.py                              # Standalone environment core
├── models.py                           # OpenEnv Pydantic models
├── portfolio_risk_env_environment.py   # OpenEnv Environment wrapper
├── client.py                           # HTTP client helper
└── inference.py                        # Baseline inference script (MANDATORY)
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | ✅ | — | Hugging Face / API key |
| `API_BASE_URL` | ❌ | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | ❌ | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
