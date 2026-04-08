import os
import json

from openai import OpenAI

from env import PortfolioRiskEnv, PortfolioAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

BENCHMARK = "portfolio_risk_env"
TASKS = ["allocation_check", "risk_rebalancing", "stress_test_optimization"]

SYSTEM_PROMPT = """You are a quantitative portfolio risk manager.
You receive a portfolio state as JSON and must respond with EXACTLY ONE JSON object — no markdown, no prose.

Valid response formats:
{"action_type": "rebalance", "ticker": "AAPL", "target_weight": 0.20, "reasoning": "Over-weight, reducing to comply with 35% cap"}
{"action_type": "hold", "reasoning": "Portfolio already compliant with all constraints"}

Rules:
- Use "rebalance" to set a specific ticker's weight to target_weight (0.0–1.0).
- Use "hold" only when ALL constraints are satisfied.
- Constraints: no single asset weight > max_single_weight (0.35), portfolio volatility < max_portfolio_vol (0.25).
- For stress_test_optimization: also minimize exposure to high-volatility assets (vol > 0.30).
- Respond with ONLY the JSON object. No extra text."""


def _parse_action(raw: str) -> PortfolioAction:
    text = raw.strip()
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                text = part
                break
    try:
        data = json.loads(text)
        return PortfolioAction(**data)
    except Exception as e:
        return PortfolioAction(action_type="hold", reasoning=f"parse_error: {e}")


def run_task(task_name: str) -> None:
    env = PortfolioRiskEnv(task=task_name)
    obs = env.reset()
    rewards = []
    step_num = 0
    final_score = 0.0

    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    try:
        while True:
            prompt = (
                f"Portfolio state:\n"
                f"{json.dumps(obs.model_dump(), indent=2)}\n\n"
                f"What action do you take? Respond with JSON only."
            )

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300,
                temperature=0.2,
            )

            raw = response.choices[0].message.content or ""
            action = _parse_action(raw)

            obs, reward, done, info = env.step(action)
            step_num += 1
            rewards.append(reward)
            final_score = reward

            error_str = obs.last_action_error if obs.last_action_error else "null"
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
        success = final_score >= 0.7
        print(
            f"[END] success={str(success).lower()} steps={step_num} "
            f"score={final_score:.2f} rewards={rewards_str}",
            flush=True,
        )


if __name__ == "__main__":
    for task in TASKS:
        run_task(task)
        print()
