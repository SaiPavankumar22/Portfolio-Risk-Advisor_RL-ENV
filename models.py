from typing import Any, Dict, List, Optional

from pydantic import Field

try:
    from openenv.core.env_server.types import Action, Observation
except ImportError:
    from pydantic import BaseModel as Action   # type: ignore
    from pydantic import BaseModel as Observation  # type: ignore


class PortfolioRiskAction(Action):
    action_type:   str            = Field(..., description="rebalance | hold | reduce | increase")
    ticker:        Optional[str]  = Field(None, description="Ticker symbol, e.g. 'AAPL'")
    target_weight: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Target portfolio weight (0.0–1.0)"
    )
    reasoning: Optional[str] = Field(None, description="Agent's reasoning for this action")


class PortfolioRiskObservation(Observation):
    task:         str               = Field(...,  description="Current task name")
    holdings:     Dict[str, float]  = Field(...,  description="Portfolio weights per ticker")
    prices:       Dict[str, float]  = Field(...,  description="Current price per ticker (USD)")
    volatilities: Dict[str, float]  = Field(...,  description="Annualized volatility per ticker")
    correlations: Dict[str, Dict[str, float]] = Field(
        ..., description="Pairwise correlation matrix"
    )
    constraints:  Dict[str, float]  = Field(
        ..., description="Active constraints: max_single_weight, max_portfolio_vol, min_assets"
    )
    step_number:  int               = Field(0,    description="Current step within the episode")
    last_action_error: Optional[str] = Field(
        None, description="Error message from previous action, or None"
    )
    previous_rewards: List[float]   = Field(
        default_factory=list, description="Last 5 step rewards (agent memory)"
    )
    regime: str = Field("normal", description="Market regime: 'normal' or 'crisis'")

    # Fields populated by the Environment wrapper
    reward:   float            = Field(0.0,  description="Reward received after last action")
    done:     bool             = Field(False, description="Whether the episode has ended")
    metadata: Dict[str, Any]  = Field(
        default_factory=dict, description="Extra info: breakdown, last_error"
    )
