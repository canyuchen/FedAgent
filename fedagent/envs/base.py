"""Async multi-turn text-environment contract for FedAgent on verl 0.8.

Every environment a FedAgent agent-loop drives implements this interface. It mirrors
the per-instance async contract verl 0.8's agent-loop expects (ONE env instance per
dataset row), generalised from the Phase 0(b) spike and aligned with VAGEN's
``GymBaseEnv``. WebShop (Phase 2) and ALFWorld (Phase 3) subclass this.

Observation convention: a dict with at least ``obs_str`` (the text shown to the model).
Image/multimodal envs may later add ``multi_modal_data`` without changing this contract.

The old verl-0.3.1 code drove envs in a *batched, synchronous* ``EnvironmentManager``
(one obs dict for the whole batch). verl 0.8's agent-loop is per-row async, so the
env becomes a single-instance object with ``await``-able reset/step.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

Obs = Dict[str, Any]


class BaseTextEnv(ABC):
    """One episode of one environment instance, driven by an AgentLoop."""

    def __init__(self, env_config: Optional[Dict[str, Any]] = None):
        self.env_config: Dict[str, Any] = dict(env_config or {})

    @abstractmethod
    async def system_prompt(self) -> Obs:
        """Return the system message (``{"obs_str": ...}``) shown once at episode start."""
        raise NotImplementedError

    @abstractmethod
    async def reset(self, seed: int = 0) -> Tuple[Obs, Dict[str, Any]]:
        """Reset to a fresh episode, deterministically in ``seed``. Returns ``(obs, info)``."""
        raise NotImplementedError

    @abstractmethod
    async def step(self, action_str: str) -> Tuple[Obs, float, bool, Dict[str, Any]]:
        """Apply the model's decoded text action. Returns ``(obs, reward, done, info)``.

        ``info`` should carry ``success`` (bool) so the agent-loop can record the
        episode outcome (FedAgent's headline metric is ``val/success_rate``).
        """
        raise NotImplementedError

    async def close(self) -> None:
        """Release any resources held by this instance (override if needed)."""
        return None
