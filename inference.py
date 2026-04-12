"""
Inference Script — Portfolio Risk Advisor
=========================================
Runs a model against all four tasks and emits structured stdout logs.

STDOUT FORMAT (mandatory):
    [START] task=<n> env=<benchmark> model=<model>
    [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>

Environment variables:
    HF_TOKEN      (required) Hugging Face / API key
    API_BASE_URL  (optional) LLM endpoint  [default: HuggingFace router]
    MODEL_NAME    (optional) Model id       [default: Qwen/Qwen2.5-72B-Instruct]
"""

import os
import json
import random

import numpy as np
from openai import OpenAI

from env import PortfolioRiskEnv, PortfolioAction, PortfolioObservation

random.seed(42)
np.random.seed(42)

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

BENCHMARK = "portfolio_risk_env"
TASKS     = [
    "allocation_check",
    "risk_rebalancing",
    "stress_test_optimization",
    "regime_shift_adaptation",
]

# LLM System prompt
SYSTEM_PROMPT = """You are a strict quantitative portfolio risk manager.

Return ONLY a single valid JSON object — no markdown, no prose, no extra text.

Hard rules:
- target_weight MUST be between 0.01 and the max_single_weight constraint (usually 0.30–0.35)
- ticker MUST be one of the tickers listed in holdings
- Fix the single largest weight violation first
- For risk_rebalancing: also reduce the asset with the highest volatility × weight contribution
- For stress_test_optimization: reduce assets with vol > 0.30 to minimise correlated crash losses
- For regime_shift_adaptation:
    * In normal regime: diversify and keep vol low, anticipating a possible crisis
    * In crisis regime: aggressively reduce high-vol, high-correlation positions; spread weights evenly
- Always mention the specific constraint or risk driver in your reasoning
- Never output invalid JSON

Valid output formats:
{"action_type": "rebalance", "ticker": "AAPL", "target_weight": 0.20, "reasoning": "AAPL exceeds 35% weight cap; reducing to manage concentration risk"}
{"action_type": "hold", "reasoning": "All weight and volatility constraints are satisfied; portfolio is compliant"}"""


def _is_compliant(obs: PortfolioObservation) -> bool:
    """Check if observation meets all constraints."""
    w       = obs.holdings
    v       = obs.volatilities
    c       = obs.correlations
    max_w   = obs.constraints.get("max_single_weight", 0.35)
    max_vol = obs.constraints.get("max_portfolio_vol",  0.25)

    if any(wt > max_w for wt in w.values()):
        return False

    # Full covariance vol
    tickers  = list(w.keys())
    port_var = sum(w[i] * w[j] * v[i] * v[j] * c[i][j] for i in tickers for j in tickers)
    port_vol = max(0.0, port_var) ** 0.5
    if port_vol > max_vol:
        return False

    if obs.task == "stress_test_optimization":
        if sum(w[t] for t in w if v[t] > 0.30) > 0.40:
            return False

    if obs.task == "regime_shift_adaptation" and obs.regime == "crisis":
        crisis_target = obs.constraints.get("crisis_vol_target", 0.40)
        if port_vol > crisis_target:
            return False

    return True


# ── Rule-based fallback ────────────────────────────────────────────────────────

def _fallback_policy(obs: PortfolioObservation) -> PortfolioAction:
    """Deterministic fix for constraint violations."""
    w       = obs.holdings
    v       = obs.volatilities
    c       = obs.correlations
    tickers = list(w.keys())
    max_w   = obs.constraints.get("max_single_weight", 0.35)
    max_vol = obs.constraints.get("max_portfolio_vol",  0.25)

    # Priority 1: Weight cap violation
    violators = {t: wt for t, wt in w.items() if wt > max_w}
    if violators:
        worst = max(violators, key=lambda t: violators[t])
        return PortfolioAction(
            action_type="rebalance",
            ticker=worst,
            target_weight=max_w,
            reasoning=(
                f"fallback: {worst} weight {violators[worst]:.3f} exceeds "
                f"max_single_weight cap {max_w}; rebalancing to reduce concentration risk"
            ),
        )

    # 2. Crisis regime: cut highest correlation × weight exposure
    if obs.task == "regime_shift_adaptation" and obs.regime == "crisis":
        avg_c = {t: sum(c[t][t2] * w[t2] for t2 in tickers) for t in tickers}
        riskiest = max(avg_c, key=lambda t: avg_c[t] * w[t])
        new_w = round(max(0.01, min(max_w, w[riskiest] * 0.5)), 4)
        return PortfolioAction(
            action_type="rebalance",
            ticker=riskiest,
            target_weight=new_w,
            reasoning=(
                f"fallback crisis: cut {riskiest} (high correlation exposure "
                f"{avg_c[riskiest]:.2f}); reducing exposure to correlated crash risk"
            ),
        )

    # 3. Stress task: halve the worst high-vol position
    if obs.task == "stress_test_optimization":
        high_vol = {t: v[t] for t in tickers if v[t] > 0.30}
        if high_vol:
            riskiest = max(high_vol, key=lambda t: w[t] * high_vol[t])
            new_w = round(max(0.01, min(max_w, w[riskiest] * 0.5)), 4)
            return PortfolioAction(
                action_type="rebalance",
                ticker=riskiest,
                target_weight=new_w,
                reasoning=(
                    f"fallback: halve {riskiest} (vol={high_vol[riskiest]:.2f}) "
                    "to reduce stress-scenario exposure"
                ),
            )

    # 4. Portfolio vol too high: reduce largest risk contributor
    port_var = sum(w[i] * w[j] * v[i] * v[j] * c[i][j] for i in tickers for j in tickers)
    port_vol = max(0.0, port_var) ** 0.5
    if port_vol > max_vol:
        rc = {t: w[t] * sum(w[t2] * v[t] * v[t2] * c[t][t2] for t2 in tickers) for t in tickers}
        worst = max(rc, key=lambda t: rc[t])
        new_w = round(max(0.01, min(max_w, w[worst] * 0.7)), 4)
        return PortfolioAction(
            action_type="rebalance",
            ticker=worst,
            target_weight=new_w,
            reasoning=(
                f"fallback: reduce {worst} to lower portfolio volatility "
                f"from {port_vol:.2f} toward target {max_vol:.2f}"
            ),
        )

    return PortfolioAction(
        action_type="hold",
        reasoning="fallback: all constraints satisfied; portfolio is compliant",
    )


def _safe_action(action: PortfolioAction, obs: PortfolioObservation) -> PortfolioAction:
    """Validate and clamp LLM output."""
    try:
        if action.action_type == "rebalance":
            if not action.ticker or action.ticker not in obs.holdings:
                return _fallback_policy(obs)
            if action.target_weight is None:
                return _fallback_policy(obs)
            max_w = obs.constraints.get("max_single_weight", 0.35)
            action.target_weight = round(max(0.01, min(max_w, action.target_weight)), 4)

        elif action.action_type in ("reduce", "increase"):
            if not action.ticker or action.ticker not in obs.holdings:
                return _fallback_policy(obs)

        return action
    except Exception:
        return _fallback_policy(obs)


def _parse_action(raw: str, obs: PortfolioObservation) -> PortfolioAction:
    """Extract PortfolioAction from LLM text."""
    text = raw.strip()

    # Strip markdown fences if present
    if "```" in text:
        for part in text.split("```"):
            part = part.strip().lstrip("json").strip()
            if part.startswith("{"):
                text = part
                break

    # Extract first JSON object
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        text = text[start : end + 1]

    try:
        data   = json.loads(text)
        action = PortfolioAction(**data)
        return _safe_action(action, obs)
    except Exception:
        return _fallback_policy(obs)


def run_task(task_name: str) -> None:
    env         = PortfolioRiskEnv(task=task_name)
    obs         = env.reset()
    rewards     = []
    step_num    = 0
    final_score = 0.0

    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    try:
        while True:

            if _is_compliant(obs):
                action = PortfolioAction(
                    action_type="hold",
                    reasoning="pre-check: all constraints satisfied; portfolio is compliant",
                )
            else:
                prompt = (
                    f"Portfolio state:\n"
                    f"{json.dumps(obs.model_dump(), indent=2)}\n\n"
                    f"Previous rewards (last steps): {obs.previous_rewards}\n"
                    f"Current market regime: {obs.regime}\n\n"
                    f"What single action do you take? Respond with JSON only."
                )
                try:
                    resp = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user",   "content": prompt},
                        ],
                        max_tokens=350,
                        temperature=0.0,
                        timeout=10,
                    )
                    raw    = resp.choices[0].message.content or ""
                    action = _parse_action(raw, obs)
                except Exception:
                    action = _fallback_policy(obs)

            obs, reward, done, info = env.step(action)
            step_num    += 1
            rewards.append(reward)
            final_score  = reward

            error_str  = obs.last_action_error or "null"
            action_str = action.action_type
            if action.ticker:
                tw = f"{action.target_weight:.2f}" if action.target_weight is not None else "?"
                action_str += f"({action.ticker},{tw})"

            print(
                f"[STEP] step={step_num} action={action_str} "
                f"reward={reward:.2f} done={str(done).lower()} error={error_str}",
                flush=True,
            )

            if done:
                break

    except Exception as exc:
        step_num += 1
        print(
            f"[STEP] step={step_num} action=error "
            f"reward=0.00 done=true error={str(exc)[:100]}",
            flush=True,
        )
        final_score = 0.0

    finally:
        env.close()
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        success     = final_score >= 0.7
        print(
            f"[END] success={str(success).lower()} steps={step_num} "
            f"score={final_score:.2f} rewards={rewards_str}",
            flush=True,
        )


if __name__ == "__main__":
    for task in TASKS:
        run_task(task)
        print()
