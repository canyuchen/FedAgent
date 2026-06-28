"""Persistent-trainer entry for lever #4 (docs/acceleration.md).

    FEDAGENT_PERSISTENT=1 FEDAGENT_PERSISTENT_PLAN=plan.json \
        python -m fedagent.fed.persistent_main <hydra overrides...>

Loads the same FedAgent Hydra config as ``main_ppo_fed`` but drives a PersistentFedTaskRunner:
ONE init_workers() then fit()-per-client over the plan (avoiding the per-client cold-start).
The worker-class reload patch is injected per-process by ``sitecustomize.py`` (gated on
FEDAGENT_PERSISTENT=1), so it lands on every Ray FSDP worker. Counterpart to the
subprocess-per-client driver ``fedagent.main_ppo_fed`` -- same config surface, different
TaskRunner, so an A/B (persistent vs subprocess) is apples-to-apples.
"""
import hydra
import ray

# register the gym_text agent loop in this process (and via agent.yaml on workers)
from fedagent.agent_loops import gym_text_agent_loop  # noqa: F401
from fedagent.fed.persistent_task_runner import PersistentFedTaskRunner
from verl.trainer.main_ppo import run_ppo


@hydra.main(config_path="../config", config_name="fedagent_ppo", version_base=None)
def main(config):
    # run_ppo calls task_runner_class.remote(), so hand it a ray.remote-wrapped class
    # (matches main_ppo.py:83's default ray.remote(num_cpus=1)(TaskRunner)).
    runner_cls = ray.remote(num_cpus=1)(PersistentFedTaskRunner)
    run_ppo(config, task_runner_class=runner_cls)


if __name__ == "__main__":
    main()
