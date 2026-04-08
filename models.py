from typing import Dict, Optional

from pydantic import Field

from openenv.core.env_server.types import Action, Observation


class PortfolioRiskAction(Action):
    action_type: str = Field(..., description="rebalance | hold | reduce | increase")
    ticker: Optional[str] = Field(None, description="Ticker symbol, e.g. 'AAPL'")
    target_weight: Optional[float] = Field(None, ge=0.0, le=1.0, description="Target weight (0.0–1.0)")
    reasoning: Optional[str] = Field(None, description="Agent's reasoning")


class PortfolioRiskObservation(Observation):
    task: str = Field(..., description="Current task name")
    holdings: Dict[str, float] = Field(..., description="Portfolio weights per ticker")
    prices: Dict[str, float] = Field(..., description="Current price per ticker")
    volatilities: Dict[str, float] = Field(..., description="Annualized volatility per ticker")
    correlations: Dict[str, Dict[str, float]] = Field(..., description="Pairwise correlation matrix")
    constraints: Dict[str, float] = Field(..., description="Active portfolio constraints")
    step_number: int = Field(0, description="Current episode step")
    last_action_error: Optional[str] = Field(None, description="Error from the previous action, or None")
