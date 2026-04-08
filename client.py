from typing import Any, Dict

try:
    from openenv.core import EnvClient
    from openenv.core.client_types import StepResult
    from openenv.core.env_server.types import State

    from .models import PortfolioRiskAction, PortfolioRiskObservation

    class PortfolioRiskClient(EnvClient[PortfolioRiskAction, PortfolioRiskObservation, State]):

        def _step_payload(self, action: PortfolioRiskAction) -> Dict:
            return {
                "action_type": action.action_type,
                "ticker": action.ticker,
                "target_weight": action.target_weight,
                "reasoning": action.reasoning,
            }

        def _parse_result(self, payload: Dict) -> StepResult[PortfolioRiskObservation]:
            obs_data = payload.get("observation", payload)
            obs = PortfolioRiskObservation(
                task=obs_data.get("task", ""),
                holdings=obs_data.get("holdings", {}),
                prices=obs_data.get("prices", {}),
                volatilities=obs_data.get("volatilities", {}),
                correlations=obs_data.get("correlations", {}),
                constraints=obs_data.get("constraints", {}),
                step_number=obs_data.get("step_number", 0),
                last_action_error=obs_data.get("last_action_error"),
                done=payload.get("done", False),
                reward=payload.get("reward", 0.0),
                metadata=payload.get("info", {}),
            )
            return StepResult(
                observation=obs,
                reward=payload.get("reward", 0.0),
                done=payload.get("done", False),
            )

        def _parse_state(self, payload: Dict) -> State:
            return State(
                episode_id=payload.get("episode_id"),
                step_count=payload.get("step_count", 0),
            )

except ImportError:
    import requests

    class PortfolioRiskClient:  # type: ignore[no-redef]

        def __init__(self, base_url: str = "http://localhost:7860"):
            self.base_url = base_url.rstrip("/")

        def reset(self, task: str = "allocation_check") -> Dict[str, Any]:
            r = requests.post(f"{self.base_url}/reset", params={"task": task})
            r.raise_for_status()
            return r.json()

        def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
            r = requests.post(f"{self.base_url}/step", json=action)
            r.raise_for_status()
            return r.json()

        def state(self) -> Dict[str, Any]:
            r = requests.get(f"{self.base_url}/state")
            r.raise_for_status()
            return r.json()

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()
