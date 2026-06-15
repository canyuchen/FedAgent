# Architecture & code map

FedAgent **extends** the vendored
[verl-agent](https://github.com/langfengQ/verl-agent) training framework rather
than wrapping it, so first-party code lives in **two layers**:

1. **Control plane** — FedAgent's own federated orchestration, at the repository
   top level. It drives training: shards data/environments per client, spawns
   per-client verl-agent runs, aggregates the resulting models, and advances
   rounds.
2. **In-framework hooks** — FedAgent's algorithm woven into verl-agent's
   extension points. These files live *inside* `third_party/verl-agent/` because
   they are imported by (and run as part of) the vendored package: the partition
   strategies are imported by the environment package, and the federated trainer
   must be runnable as `python -m verl.trainer.main_ppo_fed`. They sit beside the
   upstream originals and use a `*_fed` suffix where a counterpart exists.

> Everything under `third_party/verl-agent/` that is **not** listed in Layer 2
> below is unmodified upstream (Apache-2.0). The exhaustive list of FedAgent
> additions and edits is in
> [`third_party/verl-agent/CHANGES.md`](../third_party/verl-agent/CHANGES.md).

## Code tree (first-party)

The files listed under `third_party/verl-agent/` are FedAgent's hooks; everything
else there is unmodified upstream and elided.

```text
fedagent/
│
├── core/                          ── Layer 1: control plane ──────────────────
│   ├── custom_fed_server.py          federated server entry; drives the round loop
│   ├── fed/
│   │   ├── round_orchestrator.py     per round: select, launch, collect, aggregate
│   │   ├── script_builder.py         render each client's verl-agent launch script
│   │   ├── client_runner.py          launch + supervise one client subprocess
│   │   ├── aggregator.py             FedAvg / FedProx driver
│   │   └── checkpoint_manager.py, session_manager.py, config_helpers.py
│   └── fed_ray_ppo_trainer.py, ppo_model_wrapper.py, extra_metrics.py
│
├── utils/model_aggregation.py        aggregation math (FedAvg/FedProx, incl. FSDP)
│
├── tools/
│   ├── run_federated.py              CLI front-end; resolve paths; run the server
│   ├── resolve_paths.py              output-dir / meta naming (single source of truth)
│   ├── generate_uniform_configs.py, verify_train_val_disjoint.py
│   ├── aggregation/                  aggregation verification + diagnostics
│   ├── env_heterogeneity/            gen_holdout_{webshop,alfworld}.py
│   └── monitor/                      checkpoint_monitor.py
│
├── eval/                             eval_{webshop,alfworld}.sh, merge_trajectories.py, view_results.py
├── scripts/                          setup_env.sh, runners, verl-agent launch scripts, plotting/
├── tests/heterogenous/               partition simulations + test_alfworld_fed.py
├── config/, docs/                    experiment configs (W&B stripped) + docs
│
└── third_party/verl-agent/          ── vendored upstream (Apache-2.0) ──────────
    ├── agent_system/environments/
    │   ├── partition_strategy.py     core heterogeneity constructions
    │   └── fed_env_manager.py        federated env managers
    ├── verl/trainer/
    │   ├── main_ppo_fed.py           federated PPO/GRPO entry (python -m verl.trainer.main_ppo_fed)
    │   └── ppo/ray_trainer_fed.py    Ray federated trainer
    ├── verl/utils/
    │   ├── checkpoint/fsdp_checkpoint_manager_fed.py   federated checkpoint manager
    │   └── tracking_fed.py           per-round / per-client tracking
    └── ...                           unmodified upstream (veRL, verl-agent, WebShop, ALFWorld)
```

## Layer 1 — Control plane (first-party, top level)

| Path | Role |
|---|---|
| `core/custom_fed_server.py` | Federated server entry point — drives the whole round loop. |
| `core/fed/round_orchestrator.py` | Per-round scheduling: select clients, launch, collect, aggregate. |
| `core/fed/script_builder.py` | Renders each client's verl-agent launch script (env vars, partition kwargs, resume paths). |
| `core/fed/client_runner.py` | Launches and supervises a single client's training subprocess. |
| `core/fed/aggregator.py`, `utils/model_aggregation.py` | Model aggregation (FedAvg / FedProx), including the FSDP-sharded path. |
| `core/fed/checkpoint_manager.py`, `session_manager.py`, `config_helpers.py` | Checkpoint bookkeeping, resume/session state, config helpers. |
| `core/fed_ray_ppo_trainer.py`, `core/ppo_model_wrapper.py`, `core/extra_metrics.py` | Ray/PPO glue and metric hooks on the control side. |
| `tools/run_federated.py` | CLI front-end (`--smart/restart/direct-resume`) → resolves paths → invokes the server. |
| `tools/resolve_paths.py` | Single source of truth for output-dir / meta naming from a config. |
| `tools/generate_uniform_configs.py`, `verify_train_val_disjoint.py` | Config-matrix generation and a train/val split sanity check. |
| `tools/aggregation/` | Aggregation verification and diagnostic toolbox (standalone). |
| `tools/env_heterogeneity/gen_holdout_{webshop,alfworld}.py` | Generate the held-out distractor sets for env-level heterogeneity. |
| `tools/monitor/checkpoint_monitor.py` | Live run / checkpoint health monitor. |
| `eval/` | Checkpoint evaluation and trajectory collection (`eval_*.sh`, `merge_trajectories.py`, `view_results.py`). |
| `scripts/` | `setup_env.sh`, the federated runners, the verl-agent base launch scripts, and `plotting/`. |
| `tests/heterogenous/` | Partition-strategy simulations and a federated-sharding smoke test. |
| `config/`, `docs/` | Curated experiment configs (W&B stripped) and user documentation. |

## Layer 2 — In-framework hooks (first-party, inside `third_party/verl-agent/`)

| Path (under `third_party/verl-agent/`) | Role | Why it lives here |
|---|---|---|
| `agent_system/environments/partition_strategy.py` | **The core contribution** — all client data-partition strategies (task-level: preference / coverage / hardness) and the environment-level heterogeneity constructions. | Imported by the env package (`webshop/envs.py`, `alfworld/alfred_tw_env.py`) and by `fed_env_manager.py`. |
| `agent_system/environments/fed_env_manager.py` | Federated environment managers — wire per-client task partitions and env variants into the rollout loop. | Part of the env-manager dispatch. |
| `verl/trainer/main_ppo_fed.py` | Federated PPO/GRPO entry point (federated counterpart of upstream `main_ppo.py`). | Must run as `python -m verl.trainer.main_ppo_fed`. |
| `verl/trainer/ppo/ray_trainer_fed.py` | Ray federated trainer — client-local updates plus server-side aggregation. | Imports verl trainer internals. |
| `verl/utils/checkpoint/fsdp_checkpoint_manager_fed.py`, `verl/utils/tracking_fed.py` | Federated checkpoint manager and per-round / per-client tracking. | Plug into verl's checkpoint / tracking. |
| *edits to upstream env files* (`webshop/envs.py`, `alfworld/alfred_tw_env.py`, `webshop/.../engine.py`, `env_manager.py`, …) | Additive FedAgent hooks where the data/training plane needs them. | See CHANGES.md for the precise list. |

## Why the hooks are not a top-level package

`partition_strategy.py` is imported by **upstream** env files
(`webshop/envs.py`, `alfworld/alfred_tw_env.py`), and the federated trainer must
be importable as `verl.trainer.main_ppo_fed`. Relocating these to a top-level
package would require editing more upstream files (increasing divergence from the
vendored source) and risks the tight env/trainer integration. Keeping them in
place — beside the upstream originals, with the `*_fed` convention and this map —
preserves a clean, minimally-divergent vendor copy while still making *what is
ours* explicit.
