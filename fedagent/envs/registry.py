"""Environment registry: ``env_name`` (carried on each dataset row) -> env class.

The agent-loop looks up the env class by the ``env_name`` of the dataset row it is
driving. New envs register here:
  - Phase 2: ``WebShop`` (remote env service)
  - Phase 3: ``ALFWorld``
"""
from typing import Any, Dict, Optional, Type

from fedagent.envs.base import BaseTextEnv
from fedagent.envs.tiny_guess import TinyGuessEnv
from fedagent.envs.webshop import WebShopEnv

ENV_REGISTRY: Dict[str, Type[BaseTextEnv]] = {
    "TinyGuess": TinyGuessEnv,
    "WebShop": WebShopEnv,  # HTTP client -> webshop_service (verl-agent-webshop env)
}


def make_env(env_name: str, env_config: Optional[Dict[str, Any]] = None) -> BaseTextEnv:
    if env_name not in ENV_REGISTRY:
        raise KeyError(f"Unknown env '{env_name}'. Registered: {sorted(ENV_REGISTRY)}")
    return ENV_REGISTRY[env_name](env_config or {})
