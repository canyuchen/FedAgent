"""FedAgent federated entry point on verl 0.8 (thin overlay).

    python -m fedagent.main_ppo_fed <hydra overrides...>

Loads the FedAgent Hydra config (``config/fedagent_ppo.yaml``, layered on verl's stock
``ppo_trainer``) and runs verl's STOCK ``run_ppo``. No trainer fork: multi-turn rollout
is handled by verl's native ``AgentLoopManager`` via the AgentLoop registered in
``config/agent.yaml``.

This is the verl-0.8 replacement for verl-agent's ``verl/trainer/main_ppo_fed.py``.
In verl 0.3.1 that file existed to thread a ``TrajectoryCollector`` + envs into a FORKED
trainer; verl 0.8's native agent-loop removes that need, so this entry just configures
and launches the stock trainer.

PHASE 4-5 (planned): read the federated bridge from the environment
(CLIENT_ID / CLIENT_NUM / ROUND_NUM / PARTITION_STRATEGY / FEDPROX_MU /
INITIAL_MODEL_PATH and the heterogeneity vars), which flow into per-client data
partitioning (``AgenticDataset._partition_specs``), the FedProx proximal term (a small
actor-engine hook), and the JSON metrics logger -- all WITHOUT forking verl's trainer.
The ``run_ppo(config, task_runner_class=...)`` hook is the clean place to inject any
fed-specific worker/env setup if needed.
"""
import hydra

# Import the agent-loop module so its @register("gym_text") runs in this process too.
# (verl also imports it by _target_ from config/agent.yaml on the rollout workers.)
from fedagent.agent_loops import gym_text_agent_loop  # noqa: F401
from verl.trainer.main_ppo import run_ppo


@hydra.main(config_path="config", config_name="fedagent_ppo", version_base=None)
def main(config):
    run_ppo(config)


if __name__ == "__main__":
    main()
