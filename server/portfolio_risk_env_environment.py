from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import PortfolioRiskAction, PortfolioRiskObservation
    from ..env import PortfolioRiskEnv as _StandaloneEnv, PortfolioAction as _StandaloneAction
except ImportError:
    from models import PortfolioRiskAction, PortfolioRiskObservation
    from env import PortfolioRiskEnv as _StandaloneEnv, PortfolioAction as _StandaloneAction


class PortfolioRiskEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task: str = "allocation_check"):
        self._task = task
        self._env = _StandaloneEnv(task=task)
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(self) -> PortfolioRiskObservation:
        obs = self._env.reset()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        return self._wrap_obs(obs, reward=0.0, done=False)

    def step(self, action: PortfolioRiskAction) -> PortfolioRiskObservation:  # type: ignore[override]
        standalone_action = _StandaloneAction(
            action_type=action.action_type,
            ticker=action.ticker,
            target_weight=action.target_weight,
            reasoning=action.reasoning,
        )
        obs, reward, done, info = self._env.step(standalone_action)
        self._state.step_count += 1
        return self._wrap_obs(obs, reward=reward, done=done, metadata=info)

    @property
    def state(self) -> State:
        return self._state

    def _wrap_obs(self, obs, reward: float, done: bool, metadata: dict = None) -> PortfolioRiskObservation:
        return PortfolioRiskObservation(
            task=obs.task,
            holdings=obs.holdings,
            prices=obs.prices,
            volatilities=obs.volatilities,
            correlations=obs.correlations,
            constraints=obs.constraints,
            step_number=obs.step_number,
            last_action_error=obs.last_action_error,
            reward=reward,
            done=done,
            metadata=metadata or {},
        )
