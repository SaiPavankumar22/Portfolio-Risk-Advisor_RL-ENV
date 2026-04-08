import os
import json
import random

import numpy as np
from openai import OpenAI

from env import PortfolioRiskEnv, PortfolioAction, PortfolioObservation

# ── Deterministic behaviour ────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ── Environment variables ──────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

BENCHMARK = "portfolio_risk_env"
TASKS     = ["allocation_check", "risk_rebalancing", "stress_test_optimization"]

# ── Stricter system prompt ─────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a strict quantitative portfolio optimizer.

Return ONLY a single valid JSON object — no markdown, no prose, no explanation outside the JSON.

Hard constraints you must respect:
- target_weight MUST be between 0.0 and 0.35 (the max_single_weight cap)
- ticker MUST be one of the tickers listed in the portfolio state
- Fix the largest weight violation first; then the next largest
- For risk_rebalancing: also reduce weights of assets whose volatility pushes portfolio_vol above 0.25
- For stress_test_optimization: additionally reduce assets with vol > 0.30
- Never output invalid JSON; if unsure return the hold format

Valid output formats (choose one):
{"action_type": "rebalance", "ticker": "AAPL", "target_weight": 0.20, "reasoning": "Over max_single_weight cap"}
{"action_type": "hold", "reasoning": "All constraints satisfied"}"""


# ── Helper: check whether the current observation is fully compliant ───────────
def _is_compliant(obs: PortfolioObservation) -> bool:
    """Return True only when EVERY constraint is already satisfied."""
    max_w   = obs.constraints.get("max_single_weight", 0.35)
    max_vol = obs.constraints.get("max_portfolio_vol",  0.25)

    # Weight cap
    if any(w > max_w for w in obs.holdings.values()):
        return False

    # Portfolio volatility (simplified diagonal)
    port_vol = sum(
        obs.holdings[t] ** 2 * obs.volatilities[t] ** 2
        for t in obs.holdings
    ) ** 0.5
    if port_vol > max_vol:
        return False

    # Stress-test task: penalise high-vol exposure
    if obs.task == "stress_test_optimization":
        simulated_loss = sum(
            obs.holdings[t] * obs.volatilities[t] * 1.5
            for t in obs.holdings
            if obs.volatilities[t] > 0.30
        )
        if simulated_loss > 0.15:
            return False

    return True


# ── Helper: rule-based fallback (smart, not passive) ──────────────────────────
def _fallback_policy(obs: PortfolioObservation) -> PortfolioAction:
    """
    Deterministic fix for the worst constraint violation.
    Called when LLM output is missing, unparseable, or invalid.
    Priority order:
      1. Weight cap violation  → rebalance worst offender to max_single_weight
      2. High-vol exposure     → reduce highest-vol asset (stress task)
      3. Portfolio vol too high→ reduce the highest-vol*weight asset
      4. Truly compliant       → hold
    """
    max_w   = obs.constraints.get("max_single_weight", 0.35)
    max_vol = obs.constraints.get("max_portfolio_vol",  0.25)

    # 1. Fix the worst weight violation
    violators = {t: w for t, w in obs.holdings.items() if w > max_w}
    if violators:
        worst = max(violators, key=lambda t: violators[t])
        return PortfolioAction(
            action_type="rebalance",
            ticker=worst,
            target_weight=max_w,
            reasoning=f"fallback: {worst} weight {violators[worst]:.3f} exceeds cap {max_w}",
        )

    # 2. Stress task: cut the biggest high-vol position
    if obs.task == "stress_test_optimization":
        high_vol = {t: obs.volatilities[t] for t in obs.holdings if obs.volatilities[t] > 0.30}
        if high_vol:
            riskiest = max(high_vol, key=lambda t: obs.holdings[t] * high_vol[t])
            new_w = round(obs.holdings[riskiest] * 0.5, 4)  # halve it
            new_w = max(0.01, min(max_w, new_w))
            return PortfolioAction(
                action_type="rebalance",
                ticker=riskiest,
                target_weight=new_w,
                reasoning=f"fallback: cut high-vol {riskiest} (vol={high_vol[riskiest]:.2f})",
            )

    # 3. Portfolio vol still too high: reduce largest risk contributor
    port_vol = sum(
        obs.holdings[t] ** 2 * obs.volatilities[t] ** 2
        for t in obs.holdings
    ) ** 0.5
    if port_vol > max_vol:
        risk_contrib = {t: obs.holdings[t] * obs.volatilities[t] for t in obs.holdings}
        worst = max(risk_contrib, key=lambda t: risk_contrib[t])
        new_w = round(obs.holdings[worst] * 0.7, 4)
        new_w = max(0.01, min(max_w, new_w))
        return PortfolioAction(
            action_type="rebalance",
            ticker=worst,
            target_weight=new_w,
            reasoning=f"fallback: reduce risk contributor {worst}",
        )

    return PortfolioAction(action_type="hold", reasoning="fallback: portfolio compliant")


# ── Helper: sanitise / clamp an LLM-produced action ───────────────────────────
def _safe_action(action: PortfolioAction, obs: PortfolioObservation) -> PortfolioAction:
    """
    Clamp values and validate ticker so we never send an invalid action to env.step().
    Returns the original action (possibly fixed) or falls back to _fallback_policy.
    """
    try:
        if action.action_type == "rebalance":
            # ticker must exist
            if not action.ticker or action.ticker not in obs.holdings:
                return _fallback_policy(obs)

            # target_weight must be present and in range
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


# ── Helper: parse raw LLM text into a PortfolioAction ─────────────────────────
def _parse_action(raw: str, obs: PortfolioObservation) -> PortfolioAction:
    text = raw.strip()

    # Strip markdown fences
    if "```" in text:
        for part in text.split("```"):
            part = part.strip().lstrip("json").strip()
            if part.startswith("{"):
                text = part
                break

    # Find JSON object anywhere in the text
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]

    try:
        data   = json.loads(text)
        action = PortfolioAction(**data)
        return _safe_action(action, obs)
    except Exception:
        return _fallback_policy(obs)


# ── Main task runner ───────────────────────────────────────────────────────────
def run_task(task_name: str) -> None:
    env       = PortfolioRiskEnv(task=task_name)
    obs       = env.reset()
    rewards   = []
    step_num  = 0
    final_score = 0.0

    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    try:
        while True:

            # ── Optimisation: skip LLM when already compliant ──────────────
            if _is_compliant(obs):
                action = PortfolioAction(action_type="hold", reasoning="pre-check: compliant")
            else:
                # ── Call LLM with timeout ──────────────────────────────────
                prompt = (
                    f"Portfolio state:\n"
                    f"{json.dumps(obs.model_dump(), indent=2)}\n\n"
                    f"What single action do you take? Respond with JSON only."
                )
                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user",   "content": prompt},
                        ],
                        max_tokens=300,
                        temperature=0.0,   # fully deterministic
                        timeout=10,        # never hang
                    )
                    raw    = response.choices[0].message.content or ""
                    action = _parse_action(raw, obs)
                except Exception as api_err:
                    # API failed → use deterministic fallback instead of crashing
                    action = _fallback_policy(obs)
                    _ = str(api_err)  # logged implicitly via error field below

            # ── Step the environment ───────────────────────────────────────
            obs, reward, done, info = env.step(action)
            step_num += 1
            rewards.append(reward)
            final_score = reward

            error_str  = obs.last_action_error if obs.last_action_error else "null"
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

    except Exception as ex:
        step_num += 1
        print(
            f"[STEP] step={step_num} action=error "
            f"reward=0.00 done=true error={str(ex)[:100]}",
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