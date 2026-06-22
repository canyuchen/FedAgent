# Architecture

FedAgent is **federated reinforcement learning for LLM agents**. This document explains how
the `fedagent/` package implements it as a **thin overlay on stock verl 0.8** — what runs
where, and how a federated round actually executes.

## Design principle: overlay, not fork

The original FedAgent forked verl-agent 0.3.1 and wove federated logic *into* the trainer.
This version imports **stock verl 0.8 as a library** and adds everything through verl's public
extension points — **no patched verl tree**:

| Extension point | What FedAgent plugs in |
|---|---|
| `data.custom_cls` | [`data/agentic_dataset.py`](../data/README.md) — emits env-spec rows instead of static text |
| agent-loop registry (`agent.yaml`) | [`agent_loops/`](../agent_loops/README.md) — `GymTextAgentLoop`, multi-turn rollout |
| Hydra `searchpath` | [`config/fedagent_ppo.yaml`](../config/README.md) — layered on verl's stock `ppo_trainer` |
| interpreter startup (`sitecustomize.py`) | FedProx proximal term, gated on `FEDPROX_MU` |
| process boundary (HTTP) | [`envs/webshop/service/`](../envs/webshop/service/README.md), [`envs/alfworld/service/`](../envs/alfworld/service/README.md) — remote envs |

The benefit: verl 0.8's trainer, FSDP engine, async agent-loop rollout, and model merger are
used **as-is**, so the framework tracks upstream without fork maintenance.

## Two planes

**Control plane** — [`fed/run_fed.py`](../fed/README.md). The federated round loop. It is
verl-agnostic: it never imports verl; a client is just a subprocess
(`python -m fedagent.main_ppo_fed`). It orchestrates subprocesses, FedAvg, and merging.

**In-framework hooks** — `envs/`, `agent_loops/`, `data/`, `fedprox.py`. These run *inside*
the verl client process, reached through the extension points above.

## The federated round loop

`run_fed.py` runs `T` rounds. Each round trains the selected clients as **separate
subprocesses**, FedAvgs their FSDP checkpoints, merges to a HuggingFace model, and the next
round starts from that merged model:

```
base model ─┐
            ▼
   ROUND r:                          (select_clients: seeded per round)
   ┌─────────────────────────────────────────────────────────────────┐
   │  for each selected client c (SEQUENTIAL):                         │
   │     python -m fedagent.main_ppo_fed                               │
   │         actor_rollout_ref.model.path = model_r                    │
   │         trainer.default_local_dir   = round_r/client_c/ckpt       │
   │         env FEDAGENT_BASE_SEED = base_seed + r*100 + c            │
   │         env WEBSHOP_SERVICE_URL = client c's service             │
   │     → round_r/client_c/.../actor   (FSDP shards, ws = n_gpus)     │
   └─────────────────────────────────────────────────────────────────┘
            │  client actor dirs
            ▼
   FedAvg:  torchrun --nproc_per_node=ws aggregate_fedavg_fsdp.py
            --client-actor-dirs c0,c1  --output-actor-dir round_r/aggregated/.../actor
            ▼
   merge:   python -m verl.model_merger merge --backend fsdp
            → round_r/aggregated/hf            (complete HF model)
            │
            └──> model_{r+1} = round_r/aggregated/hf   ← the loop closes here
```

`model_1 = base model`; `model_r = round_{r-1}/aggregated/hf` for `r > 1`. PPO
(`adv_estimator=gae`) federates the **critic** the same way, in parallel with the actor.

The relevant functions in `run_fed.py`: `run` (driver), `select_clients`, `run_client`,
`fedavg`, `merge_to_hf`, `cleanup_round_checkpoints`, `eval_global`.

## Anatomy of one client subprocess

```
python -m fedagent.main_ppo_fed                       (verl stock run_ppo + FedAgent config)
  └─ verl PPO/GRPO trainer
       ├─ AgenticDataset (data.custom_cls)            → N env-spec rows, seeded by FEDAGENT_BASE_SEED
       ├─ GymTextAgentLoop (agent-loop registry)      → multi-turn rollout per row
       │     reset → generate → parse action → env.step → repeat (until done / max_turns)
       │     └─ BaseTextEnv: WebShopEnv / AlfworldEnv  → HTTP → remote env service
       ├─ advantage (GRPO group of G=8, or GAE w/ critic)
       └─ actor update → FSDP checkpoint shards
```

The env client (`envs/webshop.py`, `envs/alfworld.py`) is a **thin HTTP client**; the heavy
WebShop/ALFWorld engine runs in the remote service. See [envs/](../envs/README.md).

## Remote env services (and why)

WebShop, ALFWorld, and the trainer have **mutually conflicting dependencies** (WebShop's
Java/pyserini/gym 0.24; ALFWorld's TextWorld/Fast-Downward/torchvision; verl 0.8). So each
environment runs as its **own HTTP service in its own conda env**, one service per client:

```
trainer (fedagent-verl08)  ──HTTP──>  client 0 service (verl-agent-webshop, :8080)
                           ──HTTP──>  client 1 service (verl-agent-webshop, :8081)
                                      ...
                           ──HTTP──>  shared unperturbed VAL service (:8090)
```

`run_fed.py` launches one service per participating client (`start_webshop_services` /
`start_alfworld_services`), waits for each `/health`, and tears them down at the end. The
services `sys.path`-inject the vendored engine from `third_party/verl-agent/` — the **same
code the original FedAgent used**, so the environment MDP is unchanged (see
[migration.md](./migration.md)). This isolation is also why the service packages live at the
top level of `fedagent/`, not under `envs/`.

## Heterogeneity injection

`run_fed.py` passes the `partition_strategy` + its knobs to each client's service as env vars
(`PARTITION_STRATEGY`, `OMEGA`, `SIZE_STD`, `SUCCESS_STD`, `ENV_DIV`, `KEEP_RATIO`,
`VARIANT_N`, `CLIENT_ID`, `CLIENT_NUM`, …). The service calls [`hetero/`](../hetero/README.md)
to build *that client's* data shard from the real shuffled `server.goals`. Two levels:
environment (catalog) and task (goal distribution). See [heterogeneity.md](./heterogeneity.md).

## FedProx

When `fedprox_mu > 0`, `run_fed.py` sets `FEDPROX_MU` in the client env. The repo-root
`sitecustomize.py` runs at interpreter startup in **every** process (client + its Ray
workers) and, gated on that var, patches the FSDP optimizer step to add the proximal term.
It is deliberately **not** a Ray `runtime_env` hook (that clobbered verl's per-worker
`CUDA_VISIBLE_DEVICES`). `mu = 0` → plain FedAvg.

## Evaluation

A single **unperturbed** validation service (full env, held-out val split, no heterogeneity)
scores the **aggregated global model** every `test_freq` rounds — plus the base model at
round 0 (`val_before_train`) — at sampling temperature `val_temperature`. `eval_global` runs
a verl `val_only` pass and parses the round→success/reward curve into
`federated_summary.json`. A failed eval never aborts the run (it is measurement, not the loop).

## Outputs

Per run, under `output_dir/`: `round_*/client_*/training.log` + `json_logs/metrics.json`
(FedAgent plot format), `round_*/aggregated/hf` (the round's global model), the per-service
logs, and `federated_summary.json` (the round history + the unperturbed val curve). Consumed
FSDP shards are deleted after each merge (`cleanup_checkpoints`) to bound disk to ~one round.

## See also

- [running.md](./running.md) — how to run it (modes, GPUs, baselines, FedProx, eval)
- [configuration.md](./configuration.md) — every config key
- [reproducing.md](./reproducing.md) — the paper config matrix
- [migration.md](./migration.md) — what changed from verl-agent 0.3.1, and the fidelity record
