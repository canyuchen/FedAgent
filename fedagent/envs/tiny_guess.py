"""TinyGuess — a tiny, dependency-free async multi-turn text env.

Game = guess-the-number with higher/lower feedback. It is NOT part of the research
suite; it exists to validate the verl-0.8 wiring end-to-end (the Phase 0(b) spike
env, now a first-class ``BaseTextEnv`` so the package can prove itself before the
real WebShop/ALFWorld ports land).
"""
import re
from typing import Any, Dict, Optional, Tuple

from fedagent.envs.base import BaseTextEnv, Obs

_ANS = re.compile(r"<answer>\s*(-?\d+)\s*</answer>", re.IGNORECASE)
_INT = re.compile(r"-?\d+")


def parse_guess(text: str) -> Optional[int]:
    m = _ANS.search(text or "")
    if m:
        return int(m.group(1))
    nums = _INT.findall(text or "")
    return int(nums[-1]) if nums else None


class TinyGuessEnv(BaseTextEnv):
    """Guess a secret integer in [lo, hi]; env replies higher/lower; reward 1.0 on hit."""

    def __init__(self, env_config: Optional[Dict[str, Any]] = None):
        super().__init__(env_config)
        cfg = self.env_config
        self.lo = int(cfg.get("lo", 1))
        self.hi = int(cfg.get("hi", 50))
        self.max_turns = int(cfg.get("max_turns", 6))
        self.target = int(cfg.get("target", (self.lo + self.hi) // 2))
        self.turn = 0
        self.solved = False

    async def system_prompt(self) -> Obs:
        return {
            "obs_str": (
                f"You are playing guess-the-number. A secret integer is in "
                f"[{self.lo}, {self.hi}]. Each turn reply with EXACTLY one guess as "
                f"<answer>N</answer>. I will respond 'higher' (secret is larger) or "
                f"'lower' (secret is smaller). You have {self.max_turns} guesses."
            )
        }

    async def reset(self, seed: int = 0) -> Tuple[Obs, Dict[str, Any]]:
        self.turn = 0
        self.solved = False
        # derive a per-instance target from the seed for variety across the dataset
        span = self.hi - self.lo + 1
        self.target = self.lo + (int(seed) % span)
        return {"obs_str": "Make your first guess as <answer>N</answer>."}, {}

    async def step(self, action_str: str) -> Tuple[Obs, float, bool, Dict[str, Any]]:
        self.turn += 1
        g = parse_guess(action_str)
        if g is None:
            obs, reward = "Invalid response. Reply as <answer>N</answer>.", 0.0
        elif g == self.target:
            self.solved, obs, reward = True, "Correct!", 1.0
        elif g < self.target:
            obs, reward = "higher", 0.0
        else:
            obs, reward = "lower", 0.0
        done = self.solved or self.turn >= self.max_turns
        info = {"success": self.solved, "turns": self.turn}
        return {"obs_str": obs}, reward, done, info
