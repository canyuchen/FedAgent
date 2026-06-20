# `fedagent/` — FedAgent verl-0.8 overlay

The verl-0.8 home for FedAgent's framework-facing code. A **thin overlay**: it imports
verl 0.8 as a library and adds only what FedAgent needs on top of verl's **stock**
PPO/GRPO trainer and **native** async agent-loop rollout — **no trainer fork**.

Why this exists / how it relates to the rest of the repo:

| Layer | Where | Migration |
|---|---|---|
| Federated control plane (round loop, client subprocesses, aggregation scheduling) | `core/`, `utils/model_aggregation.py` | unchanged (verl-agnostic); only `script_builder` retargets the entry (Phase 6) |
| **In-framework hooks (envs, agent-loops, dataset, FedProx, logger, entry)** | **`fedagent/` (this package)** | **re-created on verl 0.8 here** |
| verl 0.3.1 reference + baselines | `third_party/verl-agent/` | kept as reference + Phase-8 baseline oracle, retired after validation |

## Layout
```
fedagent/
├── main_ppo_fed.py     # entry: `python -m fedagent.main_ppo_fed` -> verl stock run_ppo
├── config/
│   ├── fedagent_ppo.yaml   # Hydra config layered on verl's ppo_trainer (hydra.searchpath)
│   ├── agent.yaml          # agent-loop registry (name -> AgentLoopBase _target_)
│   └── envs/tiny_guess.yaml
├── envs/               # BaseTextEnv async contract; TinyGuess now, WebShop/ALFWorld next
├── agent_loops/        # GymTextAgentLoop (multi-turn) — verl AgentLoopBase subclass
├── data/agentic_dataset.py  # verl custom_cls; emits env-spec rows; Phase-4 partition seam
└── scripts/run_smoke.sh
```

## Status
- **Phase 1 (current):** package skeleton + Hydra wiring; validated end-to-end with the
  `TinyGuess` env (rollout → GRPO → actor update → checkpoint) on stock verl 0.8.
- **Next:** Phase 2 WebShop (remote env service), Phase 3 ALFWorld, Phase 4 heterogeneity
  (`partition_strategy` into `AgenticDataset._partition_specs`), Phase 5 FedProx + JSON logger.

## Run the smoke test
```bash
srun --jobid=<JID> --overlap bash fedagent/scripts/run_smoke.sh
```
Requires the `fedagent-verl08` conda env and a local Qwen2.5-0.5B-Instruct snapshot.
