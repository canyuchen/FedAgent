# Running FedAgent

How to run the federated loop with [`fed/run_fed.py`](../fed/README.md). For the config-key
reference see [configuration.md](./configuration.md); for the paper experiments see
[reproducing.md](./reproducing.md).

## Basics

Run inside the **`fedagent-verl08`** conda env, on a GPU node, from the repo root:

```bash
python -m fedagent.fed.run_fed --config fedagent/config/<name>.yaml
```

The config sets the environment, federation protocol, heterogeneity, and the 4-GPU recipe.
For WebShop/ALFWorld, `run_fed.py` **launches the per-client env services itself** (one per
client, in their own conda env) and tears them down at the end — you do not start them
manually. TinyGuess runs in-process (no service).

CLI flags override the YAML:

| Flag | Overrides | Use |
|---|---|---|
| `--config <yaml>` | — | the federated config (required) |
| `--model-path <dir>` | `model_path` | base model (offline: a local HF snapshot) |
| `--output-dir <dir>` | `output_dir` | where rounds/logs/checkpoints land |
| `--rounds <T>` | `total_rounds` | shorten for a quick run |
| `--clients <N>` | `total_clients` | (caps `clients_per_round` to ≤ N) |
| `--n-gpus <k>` | `n_gpus_per_node` | e.g. `4` for a 4-GPU node |
| `--base-seed <s>` | `base_seed` | 3-seed replication |
| `--port-base <p>` | `webshop_base_port` | run two jobs on one node without port clashes |
| `--fedprox-mu <mu>` | `fedprox_mu` | `>0` enables FedProx (else FedAvg) |
| `--local-client-id <k>` | `local_client_id` | Local baseline: pin client k |

## Run modes

The mode is implied by the config (no separate flag):

| Mode | How | Meaning |
|---|---|---|
| **Federated** (default) | `total_clients > 1`, `local_client_id < 0` | FedAvg across the selected clients each round |
| **Centralized** | `total_clients: 1` | one model on the pooled data (FedAvg of 1 client = identity) |
| **Local** | `local_client_id: k ≥ 0` | pin client k, train alone, no federation (paper "Local Agent Training") |

## Algorithm: GRPO vs PPO

- **GRPO** (default) — `adv_estimator: grpo`, group size **G = 8** (`rollout.n=8`). No critic.
- **PPO** — `adv_estimator: gae`. The value model (critic) is **federated alongside the actor**
  each round (round-1 critic = base model; thereafter the aggregated critic). The PPO configs
  carry the critic block in `client_overrides`.

## GPUs

`n_gpus_per_node` (or `--n-gpus`) sets the FSDP world size; the FedAvg aggregator runs
`torchrun --nproc_per_node=<world_size>` to match the saved shards. The paper recipe is 4 GPUs.
For a single-GPU debug run, set `--n-gpus 1` and use a small config (see smokes below).

## FedProx

```bash
python -m fedagent.fed.run_fed --config <...> --fedprox-mu 0.1
```

`fedprox_mu > 0` sets `FEDPROX_MU` in the client env; `sitecustomize.py` (repo root, on the
client + Ray workers' `PYTHONPATH`) adds the proximal term at the FSDP optimizer step. `mu=0`
→ plain FedAvg. Eval passes never enable it.

## Validation

If `val_env_spec` is set, a shared **unperturbed** val service scores the aggregated global
model every `test_freq` rounds (+ the base model at round 0 when `val_before_train: true`), at
`val_temperature`. The round→success/reward curve goes to `federated_summary.json`. With
`val_env_spec` unset, eval is off. A failed eval never aborts the run.

## Smoke tests (fast, for wiring checks)

The hand-written `fedagent/config/fed_*.yaml` are small smokes (e.g. 2 clients × a few rounds):

```bash
# In-process, no service — fastest end-to-end check of the federated loop
python -m fedagent.fed.run_fed --config fedagent/config/fed_tinyguess_2cl_2rd.yaml

# WebShop smoke (launches 2 services)
python -m fedagent.fed.run_fed --config fedagent/config/fed_webshop_homog_long.yaml --rounds 2

# Lower the GRPO group size for a cheaper smoke
python -m fedagent.fed.run_fed --config <...> \
  client_overrides='[actor_rollout_ref.rollout.n=2]'   # (or edit the config)
```

## Worked examples (paper configs)

```bash
# WebShop main, GRPO, Qwen2.5-1.5B, 4 GPUs, 70 rounds
python -m fedagent.fed.run_fed \
  --config fedagent/config/paper/uniform/Qwen2.5-1.5B-Instruct/main/grpo/fed_webshop_grpo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-uniform.yaml

# Same, second seed
python -m fedagent.fed.run_fed --config <...same...> --base-seed 21

# Environment-level heterogeneity (Catalog Split, div 0.7)
python -m fedagent.fed.run_fed \
  --config fedagent/config/paper/env_heterogeneity/catalog_split/fed_webshop_grpo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-catalog_split_div-0.7_keep-0.7.yaml

# Centralized baseline
python -m fedagent.fed.run_fed \
  --config fedagent/config/paper/uniform/Qwen2.5-1.5B-Instruct/centralized/grpo/fed_webshop_grpo_total-1_cl-per-rd-1_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-uniform.yaml
```

## Resume

The federation owns resume at the **round level**: re-running the same `--output-dir` continues
from the last completed round's aggregated model (each client's per-run auto-resume is disabled
so a crashed in-flight round never FedAvgs partial weights). Consumed FSDP shards are deleted
after each merge to keep peak disk to ~one round (toggle with `cleanup_checkpoints`).

## Outputs

Under `output_dir/`: `round_*/client_*/training.log` + `json_logs/metrics.json`,
`round_*/aggregated/hf` (the round's global model), per-service logs, and
`federated_summary.json` (round history + the unperturbed val curve). See
[architecture.md](./architecture.md#outputs).
