"""FedAgent — federated RL for LLM agents, on verl 0.8.

This package is the **verl-0.8 overlay** for FedAgent: a thin layer that imports
verl 0.8 as a library and adds only what FedAgent needs on top of verl's *stock*
PPO/GRPO trainer and *native* async agent-loop rollout:

  - ``fed/``         the federated round loop (``run_fed.py``) + JSON metrics logger
  - ``envs/``        async multi-turn environments (BaseTextEnv contract) + per-env services
  - ``agent_loops/`` verl ``AgentLoopBase`` subclasses that drive those envs
  - ``hetero/``      the two-level (task + environment) heterogeneity constructions
  - ``data/``        the env-spec dataset (verl ``custom_cls``) + per-client partitioning
  - ``config/``      Hydra configs layered on verl's ``ppo_trainer`` base + the paper matrix
  - ``main_ppo_fed`` the per-client entry point (reads fed env-vars, calls stock ``run_ppo``)

Design (see ``docs/``): FedAgent's federation is **subprocess-per-(client,round)**
orchestration in ``fed/run_fed.py`` (verl-agnostic — it never imports verl). verl 0.8's
native ``AgentLoopManager`` removes the only reason the old verl-0.3.1 code forked the
trainer, so this overlay uses verl's **stock** ``RayPPOTrainer`` — no trainer fork.
"""

__version__ = "0.1.0.dev0"
