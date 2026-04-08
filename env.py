import random
from pydantic import BaseModel
from typing import Optional, Dict


class PortfolioObservation(BaseModel):
    task: str
    holdings: Dict[str, float]
    prices: Dict[str, float]
    volatilities: Dict[str, float]
    correlations: Dict[str, Dict[str, float]]
    constraints: Dict[str, float]
    step_number: int
    last_action_error: Optional[str] = None


class PortfolioAction(BaseModel):
    action_type: str  # "rebalance" | "hold" | "reduce" | "increase"
    ticker: Optional[str] = None
    target_weight: Optional[float] = None
    reasoning: Optional[str] = None


class PortfolioReward(BaseModel):
    value: float
    breakdown: Dict[str, float]


TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "BRK", "JNJ", "JPM"]

TASK_MAX_STEPS = {
    "allocation_check": 5,
    "risk_rebalancing": 10,
    "stress_test_optimization": 15,
}


class PortfolioRiskEnv:

    def __init__(self, task: str = "allocation_check"):
        if task not in TASK_MAX_STEPS:
            raise ValueError(f"Unknown task '{task}'. Choose from: {list(TASK_MAX_STEPS)}")
        self.task = task
        self.max_steps = TASK_MAX_STEPS[task]
        self._state: Optional[dict] = None
        self.step_count = 0
        self.reward_history = []
        self.done = False
        self.last_error: Optional[str] = None

    def reset(self) -> PortfolioObservation:
        self.step_count = 0
        self.reward_history = []
        self.done = False
        self.last_error = None
        self._state = self._generate_scenario()
        return self._make_observation()

    def step(self, action: PortfolioAction):
        if self.done:
            raise ValueError("Episode is done — call reset() to start a new episode.")

        self.step_count += 1
        self.last_error = None
        reward = 0.0
        breakdown: Dict = {}

        if action.action_type == "rebalance" and action.ticker and action.target_weight is not None:
            if action.ticker not in self._state["weights"]:
                self.last_error = f"Unknown ticker: {action.ticker}"
            elif not (0.0 <= action.target_weight <= 1.0):
                self.last_error = "target_weight must be between 0.0 and 1.0"
                reward -= 0.05
            else:
                self._state["weights"][action.ticker] = round(action.target_weight, 4)
                total = sum(self._state["weights"].values())
                self._state["weights"] = {
                    k: round(v / total, 4) for k, v in self._state["weights"].items()
                }

        elif action.action_type in ("reduce", "increase") and action.ticker:
            if action.ticker not in self._state["weights"]:
                self.last_error = f"Unknown ticker: {action.ticker}"
            else:
                delta = 0.05 if action.action_type == "increase" else -0.05
                new_w = round(self._state["weights"][action.ticker] + delta, 4)
                new_w = max(0.01, min(1.0, new_w))
                self._state["weights"][action.ticker] = new_w
                total = sum(self._state["weights"].values())
                self._state["weights"] = {
                    k: round(v / total, 4) for k, v in self._state["weights"].items()
                }

        if self.task == "allocation_check":
            reward, breakdown = self._grade_allocation()
        elif self.task == "risk_rebalancing":
            reward, breakdown = self._grade_rebalancing()
        elif self.task == "stress_test_optimization":
            reward, breakdown = self._grade_stress_test()

        self.reward_history.append(reward)
        self.done = (self.step_count >= self.max_steps) or (reward >= 0.90)

        info = {"breakdown": breakdown, "last_error": self.last_error}
        return self._make_observation(), reward, self.done, info

    def state(self) -> dict:
        return self._state or {}

    def close(self):
        pass

    def _generate_scenario(self) -> dict:
        n = random.randint(4, 6)
        tickers = random.sample(TICKERS, n)

        raw = [random.random() for _ in tickers]
        total = sum(raw)
        weights = {t: round(r / total, 4) for t, r in zip(tickers, raw)}
        prices = {t: round(random.uniform(50, 500), 2) for t in tickers}
        vols = {t: round(random.uniform(0.12, 0.55), 3) for t in tickers}

        corr: Dict[str, Dict[str, float]] = {}
        for t1 in tickers:
            corr[t1] = {}
            for t2 in tickers:
                corr[t1][t2] = 1.0 if t1 == t2 else round(random.uniform(0.1, 0.7), 2)

        constraints = {
            "max_single_weight": 0.35,
            "max_portfolio_vol": 0.25,
            "min_assets": 4.0,
        }

        return {
            "tickers": tickers,
            "weights": weights,
            "prices": prices,
            "vols": vols,
            "corr": corr,
            "constraints": constraints,
        }

    def _make_observation(self) -> PortfolioObservation:
        s = self._state
        return PortfolioObservation(
            task=self.task,
            holdings=s["weights"],
            prices=s["prices"],
            volatilities=s["vols"],
            correlations=s["corr"],
            constraints=s["constraints"],
            step_number=self.step_count,
            last_action_error=self.last_error,
        )

    @staticmethod
    def _clamp(score: float) -> float:
        """Clamp score to open interval (0.01, 0.99) — validator requires strictly between 0 and 1."""
        return round(min(0.99, max(0.01, score)), 4)

    def _grade_allocation(self) -> tuple:
        constraints = self._state["constraints"]
        weights = self._state["weights"]
        max_w = constraints["max_single_weight"]

        violations = sum(1 for w in weights.values() if w > max_w)
        top_w = max(weights.values())

        raw = 1.0 - (violations * 0.3) - max(0.0, top_w - max_w) * 2
        score = self._clamp(raw)
        return score, {"violations": violations, "max_weight": round(top_w, 4)}

    def _grade_rebalancing(self) -> tuple:
        weights = self._state["weights"]
        vols = self._state["vols"]
        tickers = list(weights.keys())

        # simplified diagonal portfolio variance
        port_vol = sum(weights[t] ** 2 * vols[t] ** 2 for t in tickers) ** 0.5
        target_vol = self._state["constraints"]["max_portfolio_vol"]

        vol_score = max(0.0, 1.0 - max(0.0, port_vol - target_vol) / target_vol)
        alloc_score, _ = self._grade_allocation()
        raw = 0.6 * vol_score + 0.4 * alloc_score
        score = self._clamp(raw)
        return score, {
            "portfolio_vol": round(port_vol, 4),
            "target_vol": target_vol,
            "vol_score": round(vol_score, 4),
        }

    def _grade_stress_test(self) -> tuple:
        weights = self._state["weights"]
        vols = self._state["vols"]

        # stress multiplier on high-vol positions (vol > 0.30)
        simulated_loss = sum(
            weights[t] * vols[t] * 1.5
            for t in weights
            if vols[t] > 0.30
        )
        max_acceptable_loss = 0.15

        survival_score = max(0.0, 1.0 - simulated_loss / max_acceptable_loss)
        rebalance_score, _ = self._grade_rebalancing()
        raw = 0.5 * survival_score + 0.5 * rebalance_score
        score = self._clamp(raw)
        return score, {
            "simulated_loss": round(simulated_loss, 4),
            "survival_score": round(survival_score, 4),
        }
