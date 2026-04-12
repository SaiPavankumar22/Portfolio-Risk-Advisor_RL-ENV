"""
Portfolio Risk Advisor — Core Environment
==========================================
Standalone environment (no openenv dependency).
Used directly by inference.py and wrapped by portfolio_risk_env_environment.py.
"""

import random
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class PortfolioObservation(BaseModel):
    task: str
    holdings:     Dict[str, float]
    prices:       Dict[str, float]
    volatilities: Dict[str, float]
    correlations: Dict[str, Dict[str, float]]
    constraints:  Dict[str, float]
    step_number:  int
    last_action_error: Optional[str] = None
    # Agent memory: last 5 rewards so the model can reason about trajectory
    previous_rewards:  List[float] = Field(default_factory=list)
    # Market regime (used by regime_shift_adaptation task)
    regime: str = "normal"


class PortfolioAction(BaseModel):
    action_type:   str
    ticker:        Optional[str]  = None
    target_weight: Optional[float] = None
    reasoning:     Optional[str]  = None


class PortfolioReward(BaseModel):
    value:     float
    breakdown: Dict[str, float]


TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "BRK", "JNJ", "JPM"]

TASK_MAX_STEPS: Dict[str, int] = {
    "allocation_check":        5,
    "risk_rebalancing":       10,
    "stress_test_optimization": 15,
    "regime_shift_adaptation": 20,
}

SUCCESS_THRESHOLD = 0.90
TRANSACTION_COST_RATE = 0.001


class PortfolioRiskEnv:

    def __init__(self, task: str = "allocation_check"):
        if task not in TASK_MAX_STEPS:
            raise ValueError(
                f"Unknown task '{task}'. Choose from: {list(TASK_MAX_STEPS)}"
            )
        self.task      = task
        self.max_steps = TASK_MAX_STEPS[task]

        self._state:        Optional[dict] = None
        self._prev_weights: Dict[str, float] = {}

        self.step_count:    int   = 0
        self.reward_history: List[float] = []
        self.done:          bool  = False
        self.last_error:    Optional[str] = None

    def reset(self) -> PortfolioObservation:
        self.step_count    = 0
        self.reward_history = []
        self.done          = False
        self.last_error    = None
        self._state        = self._generate_scenario()
        self._prev_weights = dict(self._state["weights"])
        return self._make_observation()

    def step(
        self, action: PortfolioAction
    ) -> Tuple[PortfolioObservation, float, bool, dict]:
        if self.done:
            raise ValueError("Episode is done — call reset() to start a new episode.")

        self.step_count += 1
        self.last_error  = None
        penalty          = 0.0

        self._prev_weights = dict(self._state["weights"])

        if action.action_type == "rebalance":
            if not action.ticker:
                self.last_error = "rebalance requires a ticker"
                penalty = 0.02
            elif action.ticker not in self._state["weights"]:
                self.last_error = f"Unknown ticker: {action.ticker}"
                penalty = 0.05
            elif action.target_weight is None:
                self.last_error = "rebalance requires target_weight"
                penalty = 0.02
            elif not (0.0 <= action.target_weight <= 1.0):
                self.last_error = "target_weight must be in [0.0, 1.0]"
                penalty = 0.05
            else:
                self._state["weights"][action.ticker] = round(action.target_weight, 4)
                self._renormalize()

        elif action.action_type in ("reduce", "increase"):
            if not action.ticker:
                self.last_error = f"{action.action_type} requires a ticker"
                penalty = 0.02
            elif action.ticker not in self._state["weights"]:
                self.last_error = f"Unknown ticker: {action.ticker}"
                penalty = 0.05
            else:
                delta = 0.05 if action.action_type == "increase" else -0.05
                new_w = round(self._state["weights"][action.ticker] + delta, 4)
                self._state["weights"][action.ticker] = max(0.01, min(1.0, new_w))
                self._renormalize()

        elif action.action_type == "hold":
            pass  # deliberate no-op

        else:
            self.last_error = f"Unknown action_type: '{action.action_type}'"
            penalty = 0.05

        if (
            self.task == "regime_shift_adaptation"
            and self._state.get("regime") == "normal"
            and self.step_count >= self._state.get("regime_shift_step", 999)
        ):
            self._state["regime"] = "crisis"
            self._state["vols"]   = self._state["crisis_vols"]
            self._state["corr"]   = self._state["crisis_corr"]

        new_weights = self._state["weights"]
        turnover = sum(
            abs(new_weights.get(t, 0.0) - self._prev_weights.get(t, 0.0))
            for t in set(new_weights) | set(self._prev_weights)
        )
        transaction_cost = turnover * TRANSACTION_COST_RATE

        reasoning_bonus = self._reasoning_bonus(action.reasoning)

        if self.task == "allocation_check":
            raw_score, breakdown = self._grade_allocation()
        elif self.task == "risk_rebalancing":
            raw_score, breakdown = self._grade_rebalancing()
        elif self.task == "stress_test_optimization":
            raw_score, breakdown = self._grade_stress_test()
        else:
            raw_score, breakdown = self._grade_regime_shift()

        reward = self._clamp(raw_score + reasoning_bonus - penalty - transaction_cost)
        breakdown["transaction_cost"]  = round(transaction_cost, 4)
        breakdown["reasoning_bonus"]   = round(reasoning_bonus, 4)

        self.reward_history.append(reward)
        self.done = (self.step_count >= self.max_steps) or (reward >= SUCCESS_THRESHOLD)

        info = {"breakdown": breakdown, "last_error": self.last_error}
        return self._make_observation(), reward, self.done, info

    def state(self) -> dict:
        return self._state or {}

    def close(self) -> None:
        pass

    def _renormalize(self) -> None:
        total = sum(self._state["weights"].values())
        if total > 0:
            self._state["weights"] = {
                k: round(v / total, 4)
                for k, v in self._state["weights"].items()
            }

    def _portfolio_vol(
        self,
        weights: Optional[Dict[str, float]] = None,
        vols: Optional[Dict[str, float]] = None,
        corr: Optional[Dict] = None,
    ) -> float:
        """Full covariance portfolio volatility."""
        w = weights or self._state["weights"]
        v = vols    or self._state["vols"]
        c = corr    or self._state["corr"]
        tickers = list(w.keys())

        port_var = sum(
            w[i] * w[j] * v[i] * v[j] * c[i][j]
            for i in tickers
            for j in tickers
        )
        return max(0.0, port_var) ** 0.5

    @staticmethod
    def _clamp(score: float) -> float:
        """Keep score strictly inside (0.01, 0.99) — validator requirement."""
        return round(min(0.99, max(0.01, score)), 4)

    @staticmethod
    def _reasoning_bonus(reasoning: Optional[str]) -> float:
        """Bonus for agent reasoning with financial keywords."""
        if not reasoning:
            return 0.0
        text = reasoning.lower()
        keywords = [
            "volatility", "correlation", "weight", "risk", "diversif",
            "exceed", "cap", "rebalance", "constraint", "exposure",
            "regime", "crisis", "shock",
        ]
        matches = sum(1 for kw in keywords if kw in text)
        return round(min(0.04, matches * 0.01), 4)

    def _generate_scenario(self) -> dict:
        if self.task == "regime_shift_adaptation":
            return self._generate_regime_scenario()
        return self._generate_standard_scenario()

    def _generate_standard_scenario(self) -> dict:
        n       = random.randint(4, 6)
        tickers = random.sample(TICKERS, n)

        raw    = [random.random() for _ in tickers]
        total  = sum(raw)
        weights = {t: round(r / total, 4) for t, r in zip(tickers, raw)}
        prices  = {t: round(random.uniform(50, 500), 2) for t in tickers}
        vols    = {t: round(random.uniform(0.12, 0.55), 3) for t in tickers}

        corr: Dict[str, Dict[str, float]] = {}
        for t1 in tickers:
            corr[t1] = {}
            for t2 in tickers:
                corr[t1][t2] = 1.0 if t1 == t2 else round(random.uniform(0.05, 0.65), 2)

        constraints = {
            "max_single_weight": 0.35,
            "max_portfolio_vol":  0.25,
            "min_assets":         4.0,
        }

        return {
            "tickers":     tickers,
            "weights":     weights,
            "prices":      prices,
            "vols":        vols,
            "corr":        corr,
            "constraints": constraints,
            "regime":      "normal",
        }

    def _generate_regime_scenario(self) -> dict:
        """Generate scenario with regime shift from normal to crisis."""
        n       = random.randint(4, 6)
        tickers = random.sample(TICKERS, n)

        raw    = [random.random() for _ in tickers]
        total  = sum(raw)
        weights = {t: round(r / total, 4) for t, r in zip(tickers, raw)}
        prices  = {t: round(random.uniform(50, 500), 2) for t in tickers}

        # Normal regime: moderate vols and correlations
        vols    = {t: round(random.uniform(0.12, 0.35), 3) for t in tickers}
        corr: Dict[str, Dict[str, float]] = {}
        for t1 in tickers:
            corr[t1] = {}
            for t2 in tickers:
                corr[t1][t2] = 1.0 if t1 == t2 else round(random.uniform(0.05, 0.45), 2)

        crisis_vols = {t: round(vols[t] * random.uniform(1.8, 2.5), 3) for t in tickers}
        crisis_corr: Dict[str, Dict[str, float]] = {}
        for t1 in tickers:
            crisis_corr[t1] = {}
            for t2 in tickers:
                if t1 == t2:
                    crisis_corr[t1][t2] = 1.0
                else:
                    base = corr[t1][t2]
                    spike = round(min(0.95, base + random.uniform(0.35, 0.55)), 2)
                    crisis_corr[t1][t2] = spike

        regime_shift_step = random.randint(4, 10)

        constraints = {
            "max_single_weight":  0.30,    # tighter in this task
            "max_portfolio_vol":  0.20,    # tighter in normal regime
            "crisis_vol_target":  0.40,    # allowed vol in crisis
            "min_assets":         4.0,
        }

        return {
            "tickers":           tickers,
            "weights":           weights,
            "prices":            prices,
            "vols":              vols,
            "corr":              corr,
            "constraints":       constraints,
            "regime":            "normal",
            "regime_shift_step": regime_shift_step,
            "crisis_vols":       crisis_vols,
            "crisis_corr":       crisis_corr,
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
            previous_rewards=list(self.reward_history[-5:]),
            regime=s.get("regime", "normal"),
        )

    def _grade_allocation(self) -> Tuple[float, dict]:
        weights = self._state["weights"]
        max_w   = self._state["constraints"]["max_single_weight"]

        violations = sum(1 for w in weights.values() if w > max_w)
        top_w      = max(weights.values())
        excess     = max(0.0, top_w - max_w)

        score = 1.0 - (violations * 0.25) - (excess * 2.0)
        return self._clamp(score), {
            "violations": violations,
            "max_weight": round(top_w, 4),
            "excess":     round(excess, 4),
        }

    def _grade_rebalancing(self) -> Tuple[float, dict]:
        weights  = self._state["weights"]
        tickers  = list(weights.keys())
        target_v = self._state["constraints"]["max_portfolio_vol"]

        port_vol = self._portfolio_vol()

        vol_score   = max(0.0, 1.0 - max(0.0, port_vol - target_v) / target_v)
        alloc_score, _ = self._grade_allocation()
        score = 0.6 * vol_score + 0.4 * alloc_score

        return self._clamp(score), {
            "portfolio_vol":    round(port_vol, 4),
            "target_vol":       target_v,
            "vol_score":        round(vol_score, 4),
            "allocation_score": round(alloc_score, 4),
        }

    def _grade_stress_test(self) -> Tuple[float, dict]:
        """Stress test with correlated crash scenario."""
        weights = self._state["weights"]
        vols    = self._state["vols"]
        corr    = self._state["corr"]
        tickers = list(weights.keys())

        avg_corr = sum(
            corr[t1][t2]
            for t1 in tickers for t2 in tickers if t1 != t2
        ) / max(1, len(tickers) * (len(tickers) - 1))

        market_shock    = -0.20
        portfolio_loss  = 0.0
        for t in tickers:
            systematic   = weights[t] * vols[t] * avg_corr * abs(market_shock)
            idiosyncratic = weights[t] * vols[t] * (1 - avg_corr) * 0.5
            portfolio_loss += systematic + idiosyncratic

        max_acceptable = 0.15
        survival_score = max(0.0, 1.0 - portfolio_loss / max_acceptable)

        high_vol_exposure = sum(
            weights[t] for t in tickers if vols[t] > 0.30
        )
        concentration_penalty = max(0.0, high_vol_exposure - 0.30) * 0.5

        rebalance_score, rebalance_bd = self._grade_rebalancing()
        score = 0.5 * survival_score + 0.4 * rebalance_score - concentration_penalty

        return self._clamp(score), {
            "portfolio_loss":       round(portfolio_loss, 4),
            "survival_score":       round(survival_score, 4),
            "high_vol_exposure":    round(high_vol_exposure, 4),
            "avg_correlation":      round(avg_corr, 4),
            "rebalance_score":      round(rebalance_score, 4),
        }

    def _grade_regime_shift(self) -> Tuple[float, dict]:
        """Grade regime shift task: normal → crisis scenario."""
        weights = self._state["weights"]
        regime  = self._state.get("regime", "normal")
        tickers = list(weights.keys())

        if regime == "normal":
            target_v = self._state["constraints"]["max_portfolio_vol"]
            port_vol = self._portfolio_vol()
            vol_score   = max(0.0, 1.0 - max(0.0, port_vol - target_v) / target_v)
            alloc_score, _ = self._grade_allocation()

            # Diversification reward: lower Herfindahl index = more diversified
            herfindahl  = sum(w ** 2 for w in weights.values())
            n           = len(tickers)
            div_score   = 1.0 - (herfindahl - 1 / n) / (1 - 1 / n) if n > 1 else 0.5

            score = 0.4 * vol_score + 0.3 * alloc_score + 0.3 * div_score
            return self._clamp(score), {
                "regime":          "normal",
                "portfolio_vol":   round(port_vol, 4),
                "vol_score":       round(vol_score, 4),
                "diversification": round(div_score, 4),
            }

        else:
            # Crisis: use crisis vols/corrs (already applied to _state)
            crisis_target = self._state["constraints"].get("crisis_vol_target", 0.40)
            crisis_vol    = self._portfolio_vol()

            vol_survival = max(0.0, 1.0 - max(0.0, crisis_vol - crisis_target) / crisis_target)

            # Diversification is critical in a crisis (all correlations are high)
            herfindahl = sum(w ** 2 for w in weights.values())
            n          = len(tickers)
            div_score  = 1.0 - (herfindahl - 1 / n) / (1 - 1 / n) if n > 1 else 0.0

            # Weight constraint still applies
            max_w       = self._state["constraints"]["max_single_weight"]
            violations  = sum(1 for w in weights.values() if w > max_w)
            alloc_score = max(0.0, 1.0 - violations * 0.25)

            score = 0.4 * vol_survival + 0.3 * div_score + 0.3 * alloc_score
            return self._clamp(score), {
                "regime":          "crisis",
                "crisis_vol":      round(crisis_vol, 4),
                "crisis_target":   crisis_target,
                "vol_survival":    round(vol_survival, 4),
                "diversification": round(div_score, 4),
            }
