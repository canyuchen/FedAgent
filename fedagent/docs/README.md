# FedAgent documentation

User-facing documentation for the **FedAgent verl-0.8 overlay** — federated reinforcement
learning for LLM agents, built as a thin overlay on stock verl 0.8. Start with the package
overview in [`../README.md`](../README.md), then:

| Doc | Read it for |
|---|---|
| [architecture.md](./architecture.md) | How the overlay works: the federated round loop, the in-framework hooks, the remote env services, FedProx, eval. |
| [installation.md](./installation.md) | The three conda envs (trainer + WebShop + ALFWorld services), data, and models. |
| [running.md](./running.md) | Running `run_fed.py`: run modes, GPUs, baselines, FedProx, validation, worked examples. |
| [configuration.md](./configuration.md) | The config-file decoder and the full federated-runner key reference. |
| [heterogeneity.md](./heterogeneity.md) | The two-level (task vs environment) heterogeneity suite and how to select each arm. |
| [reproducing.md](./reproducing.md) | The paper's 176-config matrix mapped to run commands; 3-seed replication; baselines. |
| [migration.md](./migration.md) | What changed from the verl-agent-0.3.1 fork, the environment-engine reuse, and the fidelity record. |

## Per-component references

Each `fedagent/` subpackage has its own README with code-level detail:

- [`../fed/`](../fed/README.md) — federated round loop + metrics logger
- [`../agent_loops/`](../agent_loops/README.md) — multi-turn agent rollout (`GymTextAgentLoop`)
- [`../envs/`](../envs/README.md) — `BaseTextEnv` contract + registry; TinyGuess / WebShop / ALFWorld clients
- [`../hetero/`](../hetero/README.md) — the heterogeneity constructions
- [`../envs/webshop/service/`](../envs/webshop/service/README.md) · [`../envs/alfworld/service/`](../envs/alfworld/service/README.md) — remote env services
- [`../data/`](../data/README.md) — `AgenticDataset` (verl `custom_cls`)
- [`../config/`](../config/README.md) — configs + the paper matrix
- [`../EXPERIMENTS.md`](../EXPERIMENTS.md) — the running experiment log

## Scope

These docs describe the **verl-0.8 overlay** (the live system, under `fedagent/`). The repo's
top-level `docs/` and `README.md` describe the *original* verl-agent-0.3.1 artifact and are
retained as historical reference; see [migration.md](./migration.md) for the relationship.
