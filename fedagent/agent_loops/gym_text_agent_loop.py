"""GymTextAgentLoop — FedAgent's multi-turn text agent-loop for verl 0.8.

A verl ``AgentLoopBase`` subclass that drives ONE ``BaseTextEnv`` instance per dataset
row on verl's *native* async agent-loop seam:

    reset env -> (build prompt -> server.generate -> decode -> env.step -> append obs)* -> AgentLoopOutput

This is the verl-0.8 replacement for verl-agent's ``TrajectoryCollector.multi_turn_loop``.
It uses verl's STOCK trainer (no fork): the trajectory is returned as one
``AgentLoopOutput`` (concat multi-turn) with a response_mask that is 1 on
model-generated tokens and 0 on environment-observation tokens, so PPO/GRPO trains
only on the agent's actions. ``reward_score`` is the episode reward (GRPO groups are
formed by verl's ``rollout.n`` repeats of each row).

Fidelity note: per the agreed bar (SCIENTIFIC EQUIVALENCE), this uses verl 0.8's
native concat multi-turn rather than bit-reproducing the old per-turn/no-concat row
layout; outcome credit assignment over the action tokens is preserved.
"""
import logging
import os
from typing import Any, Dict, List
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

from fedagent.envs.registry import make_env

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("gym_text")
class GymTextAgentLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    @rollout_trace_op
    async def run(self, sampling_params: Dict[str, Any], **kwargs) -> AgentLoopOutput:
        env_name = kwargs.get("env_name", "TinyGuess")
        env = make_env(env_name, kwargs.get("config", {}) or {})
        seed = int(kwargs.get("seed", 0))
        max_turns = int(kwargs.get("max_turns", 6))

        sys_obs = await env.system_prompt()
        init_obs, _ = await env.reset(seed=seed)
        messages = [
            {"role": "system", "content": sys_obs["obs_str"]},
            {"role": "user", "content": init_obs["obs_str"]},
        ]

        metrics: Dict[str, Any] = {}
        prompt_ids = await self.apply_chat_template(messages)
        cur_ids = list(prompt_ids)
        response_ids: List[int] = []
        response_mask: List[int] = []
        env_rewards: List[float] = []
        success = False
        turns = 0

        try:
            for _ in range(max_turns):
                with simple_timer("generate_sequences", metrics):
                    out: TokenOutput = await self.server_manager.generate(
                        request_id=uuid4().hex, prompt_ids=cur_ids, sampling_params=sampling_params
                    )
                gen = out.token_ids
                response_ids += gen
                response_mask += [1] * len(gen)  # 1 = model-generated -> trained on
                cur_ids = cur_ids + gen
                turns += 1

                text = self.tokenizer.decode(gen, skip_special_tokens=True)
                messages.append({"role": "assistant", "content": text})
                obs, reward, done, info = await env.step(text)
                env_rewards.append(float(reward))
                success = bool(info.get("success", False))

                if done or len(response_ids) >= self.response_length:
                    break

                # append the env observation as a user turn; its tokens are NOT generated
                messages.append({"role": "user", "content": obs["obs_str"]})
                new_ids = await self.apply_chat_template(messages)
                obs_tokens = new_ids[len(cur_ids):] if len(new_ids) > len(cur_ids) else []
                response_ids += obs_tokens
                response_mask += [0] * len(obs_tokens)  # 0 = observation -> masked out of loss
                cur_ids = new_ids
        finally:
            # always release the env (e.g. return a pooled remote WebShop session),
            # even if generate/step raises mid-episode.
            await env.close()

        return AgentLoopOutput(
            prompt_ids=prompt_ids[-self.prompt_length:],
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            num_turns=turns,
            reward_score=float(sum(env_rewards)),
            metrics=metrics,
            extra_fields={
                "turn_scores": [],
                "tool_rewards": [],
                "reward_extra_info": {"traj_success": float(success)},
            },
        )
