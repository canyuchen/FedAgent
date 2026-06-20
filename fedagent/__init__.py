"""FedAgent — federated RL for LLM agents, on verl 0.8.

This package is the **verl-0.8 overlay** for FedAgent: a thin layer that imports
verl 0.8 as a library and adds only what FedAgent needs on top of verl's *stock*
PPO/GRPO trainer and *native* async agent-loop rollout:

  - ``envs/``        async multi-turn environments (BaseTextEnv contract)
  - ``agent_loops/`` verl ``AgentLoopBase`` subclasses that drive those envs
  - ``data/``        the env-spec dataset (verl ``custom_cls``) + (later) per-client partitioning
  - ``config/``      Hydra configs layered on verl's ``ppo_trainer`` base
  - ``main_ppo_fed`` the federated entry point (reads fed env-vars, calls stock ``run_ppo``)

Design (see docs / project memory): FedAgent's federation is **subprocess-per-round**
orchestration that lives in the verl-agnostic ``core/`` package (unchanged). verl 0.8's
native ``AgentLoopManager`` removes the only reason the old verl-0.3.1 code forked the
trainer, so this overlay uses verl's **stock** ``RayPPOTrainer`` — no trainer fork.
"""

__version__ = "0.1.0.dev0"
