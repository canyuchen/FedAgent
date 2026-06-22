# Reproducing the paper

This is the per-experiment reproduction guide for the FedAgent **verl-0.8
overlay** — the thin layer that re-runs the paper's config matrix on
**unmodified verl 0.8**. Every cell of the matrix is a single YAML under
[`../config/paper/`](../config/paper/), and every cell runs with one command:

```bash
python -m fedagent.fed.run_fed --config fedagent/config/paper/<...>.yaml
```

Read this together with [`../config/README.md`](../config/README.md) (the four
config types and the `paper/` naming convention), [`../fed/README.md`](../fed/README.md)
(the runner internals — the round loop, FedAvg, baselines, eval), and
[`./heterogeneity.md`](./heterogeneity.md) (the two-level heterogeneity suite the
het arms instantiate). This is **scientific-equivalence** reproduction, not
bit-identical — see [the fidelity note](#scientific-equivalence-not-bit-identical).

---

## Prerequisites

- **Conda env `fedagent-verl08`** (py3.12, stock verl 0.8). Activate it first;
  `run_fed` sets `PYTHONPATH` to the repo root so `fedagent` and the root
  `sitecustomize.py` (FedProx) are importable in every subprocess it spawns.
- **A 4-GPU node.** The `paper/` configs pin `n_gpus_per_node: 4` (FSDP world
  size 4); `--n-gpus` overrides it.
- **The env service env.** WebShop and ALFWorld arms talk to one remote HTTP
  service per client; `run_fed` **launches the services itself**, but their conda
  env / data must be installed and on PATH. `tinyguess` runs in-process.
- **Models.** Each config sets `model_path` to an HF id
  (e.g. `Qwen/Qwen2.5-1.5B-Instruct`) which auto-downloads. On an offline cluster
  pass `--model-path <local snapshot>` to point at a pre-fetched directory.
- **Hardness arms only** need a `trajectories_file` (a `task_id`→success-label
  map). It is **not** shipped — generate it first:

  ```bash
  python tools/verl08_migration/gen_hardness_trajectories.py   # writes data/hardness/<model>_<env>_trajectories.json
  ```

  The hardness configs reference exactly this path (e.g.
  `data/hardness/qwen2.5-1.5b_webshop_trajectories.json`); the run aborts at
  service start if it is missing.
- **ALFWorld arms** drive episodes at `max_turns: 50` (the original
  `max_steps=50`) paired with a **widened context window**
  (`rollout.max_model_len=16384`, `response_length=8192`). This is flagged
  **GPU-VERIFY** in `config/envs/alfworld.yaml`: confirm no OOM / prompt
  truncation at 50 turns on your GPUs, and raise `max_model_len` if verbose rooms
  truncate before `done`.

---

## The one-command run pattern

Every command below is the full invocation; the only thing that changes is the
config path. CLI flags (`--rounds --clients --n-gpus --base-seed --fedprox-mu
--local-client-id --model-path`) override the YAML.

```bash
conda activate fedagent-verl08

# Uniform main table, GRPO, WebShop (Qwen2.5-1.5B):
python -m fedagent.fed.run_fed --config \
  fedagent/config/paper/uniform/Qwen2.5-1.5B-Instruct/main/grpo/fed_webshop_grpo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-uniform.yaml

# Environment-level heterogeneity: catalog split (div 0.7):
python -m fedagent.fed.run_fed --config \
  fedagent/config/paper/env_heterogeneity/catalog_split/fed_webshop_grpo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-catalog_split_div-0.7_keep-0.7.yaml

# Task-level heterogeneity: preference skew (omega 0.99):
python -m fedagent.fed.run_fed --config \
  fedagent/config/paper/task_heterogeneity/grpo/webshop/fed_webshop_grpo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-preference_omega-0.99.yaml

# PPO arm (federates the critic too — adv_estimator: gae):
python -m fedagent.fed.run_fed --config \
  fedagent/config/paper/uniform/Qwen2.5-1.5B-Instruct/main/ppo/fed_webshop_ppo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-uniform.yaml

# ALFWorld (uniform main, GRPO):
python -m fedagent.fed.run_fed --config \
  fedagent/config/paper/uniform/Qwen2.5-1.5B-Instruct/main/grpo/fed_alfworld_grpo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-uniform.yaml

# Baselines (same family, different mode):
python -m fedagent.fed.run_fed --config \
  fedagent/config/paper/uniform/Qwen2.5-1.5B-Instruct/centralized/grpo/fed_webshop_grpo_total-1_cl-per-rd-1_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-uniform.yaml
python -m fedagent.fed.run_fed --config \
  fedagent/config/paper/uniform/Qwen2.5-1.5B-Instruct/local_client1/grpo/fed_webshop_grpo_total-100_cl-per-rd-1_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-uniform.yaml
```

The federation protocol is baked into the `paper/` configs and matches the paper:
**N = 100** clients (`total_clients`), **M = 2** sampled per round
(`clients_per_round`), **T = 70** rounds (`total_rounds`), **E = 3** local epochs
(`epochs_per_round`). Each round trains the selected clients from the previous
round's merged FedAvg model, re-aggregates, and (every `test_freq` rounds) scores
the global model on the shared unperturbed val set.

---

## The experiment matrix

176 configs total under `config/paper/`, mirroring the original paper structure.
The **main table is the 4-backbone uniform sweep across WebShop + ALFWorld**; the
heterogeneity and decentralized families are run on a single backbone
(Qwen2.5-1.5B-Instruct).

| Family | Config dir | What it studies | Backbones | Count |
|---|---|---|---|---|
| **Uniform (main)** | `uniform/<Model>/{main,main_seed1,main_seed2}/{grpo,ppo}/` | Headline federated result, 3 seeds, GRPO + PPO, on WebShop + ALFWorld | 4 | (in 112) |
| **Uniform (baselines)** | `uniform/<Model>/{centralized,local_client1,local_client2,local_client3}/{grpo,ppo}/` | Centralized & Local-Agent references for the same cells | 4 | (in 112) |
| **Env heterogeneity** | `env_heterogeneity/<strategy>[_ppo]/` | Robustness to a hidden transition kernel (catalog split, BM25 field/reweight, lookalike, rank) — WebShop only | Qwen2.5-1.5B | 16 |
| **Task heterogeneity** | `task_heterogeneity/{grpo,ppo}/{webshop,alfworld}/` | Robustness to an observable goal-distribution skew (preference, coverage, hardness) | Qwen2.5-1.5B | 24 |
| **Decentralized** | `decentralized/{ep_per_round_change,samples_change,selected_cl_change}/{grpo,ppo}/` | Protocol-knob ablations on the uniform baseline | Qwen2.5-1.5B | 24 |

The `uniform/` family (112) is the four backbones — `Qwen2.5-1.5B-Instruct`,
`Qwen2.5-3B-Instruct`, `Qwen2.5-7B-Instruct`, `Llama-3.2-3B-Instruct` — each with
7 run kinds (`main`, `main_seed1`, `main_seed2`, `centralized`, `local_client1-3`)
× `{grpo, ppo}` × `{webshop, alfworld}`. The het / decentralized families are
Qwen2.5-1.5B only. `config/paper/` mirrors the original tree's structure and
naming; contents are verl-0.8 `run_fed` configs, regenerable with
`tools/verl08_migration/gen_paper_configs.py` (see
[`../config/README.md`](../config/README.md)).

---

## Three-seed replication

The main table reports three seeds. They are already **separate configs** —
`main`, `main_seed1`, `main_seed2` — differing only in `base_seed`:

| Run kind | `base_seed` |
|---|---|
| `main` | 42 |
| `main_seed1` | 21 |
| `main_seed2` | 84 |

`base_seed` drives both the per-round client selection and the round-threaded
data seed (`FEDAGENT_BASE_SEED = base_seed + round*100 + client_id`), so the three
runs explore distinct client schedules and goal draws. You can also reproduce a
seed by overriding any base config on the CLI:

```bash
python -m fedagent.fed.run_fed --config <main config> --base-seed 21   # == main_seed1
```

---

## Baselines (centralized & local) vs federated

The runner derives the mode from the config — there is no separate flag (see
[`../fed/README.md`](../fed/README.md#baseline-modes)):

| Mode | Selected by | Behavior |
|---|---|---|
| **federated** | `total_clients: 100` (default) | FedAvg across the 2 sampled clients each round. |
| **centralized** | `total_clients: 1` | One model on the pooled data; FedAvg of a single client is the identity, so the run is continued central training. |
| **local** | `local_client_id >= 0` (`clients_per_round: 1`) | The paper's *Local Agent Training*: pin one client's data shard every round, no federation. |

The three local configs pin distinct clients of the 100-way partition:

| Config dir | `local_client_id` |
|---|---|
| `local_client1/` | 21 |
| `local_client2/` | 42 |
| `local_client3/` | 84 |

`--local-client-id` overrides it for any base config.

**Epoch budget.** Both baselines run **T = 70 × E = 3 = 210 epochs**, matching the
federated arms' total. The original paper ran the baselines as 1 round × 210
epochs; in this overlay the per-round FedAvg of a single client/shard is a no-op,
but **goal variety is drawn per round** (the round-threaded data seed re-draws
each client's goals every round), so the runner keeps **70 rounds** to reproduce
that variety — same total epochs, same goal coverage.

---

## Outputs

Each run writes everything under the config's `output_dir`:

- **`federated_summary.json`** — per-round provenance (clients selected, the model
  each round started from, aggregated actor + HF paths, the critic chain for PPO)
  plus the `mode`, `partition_strategy`, final model, and the **`val_curve`**.
- **Per-round logs** — `round_<r>/client_<c>/training.log`,
  `round_<r>/aggregated/{aggregate,merge}_*.log`, and the per-service logs
  (`webshop_service_client<c>.log` / `alfworld_service_client<c>.log`).
- **`round_<r>/client_<c>/json_logs/metrics.json`** — each client's `training.log`
  re-parsed into the FedAgent plot schema (`[{"step", "metrics"}, ...]`).
- **The unperturbed val success curve** — `eval_global` scores the aggregated
  global model on the shared unperturbed val service every `test_freq: 5` rounds
  (plus the final round), with `val_before_train: true` adding the base model as
  the round-0 point and `val_temperature: 0.4`. The curve lands in
  `federated_summary.json` (`val_curve`) and the round-`r` eval dumps live in
  `round_<r>/eval/`.

`tools/verl08_migration/summarize_fed_run.py` post-processes a run directory.

> **Disk.** Consumed FSDP shards are deleted after each merge
> (`cleanup_checkpoints`, on by default), keeping every `training.log` and the
> merged HF; peak disk stays roughly one round's worth.

---

## Scientific-equivalence, not bit-identical

This overlay reproduces the paper's **science** — the same federation protocol
(N/M/T/E), the same algorithms (GRPO G = 8, PPO/GAE with a federated critic), the
same heterogeneity construction, and the same unperturbed-val measurement — on
**stock verl 0.8** with no trainer fork. It is **not** bit-for-bit identical to
the original verl-agent 0.3.1 stack (different rollout engine, FSDP checkpoint
layout, and RNG threading). For the full fidelity record — what is preserved,
what changed, and why — see [`./migration.md`](./migration.md).
