# Configuration

FedAgent is a **thin overlay on unmodified verl 0.8** — there is no trainer fork. Every
run is driven by configuration: a flat YAML the federated runner reads, a Hydra base that
composes verl's stock `ppo_trainer`, an agent-loop registry, and per-episode env specs.
This page is the **config-file decoder** and the **federated-runner key reference**.

See the package overview in [`../README.md`](../README.md), the federated driver in
[`../fed/README.md`](../fed/README.md), and the already-written
[`../config/README.md`](../config/README.md) (the folder map) which this page expands. For
the heterogeneity arms see [`./heterogeneity.md`](./heterogeneity.md); for running the
loop, [`./running.md`](./running.md); for the figure-by-figure matrix,
[`./reproducing.md`](./reproducing.md).

---

## The four config types

| Type | File(s) | Consumed by | Role |
|---|---|---|---|
| **Hydra base config** | [`config/fedagent_ppo.yaml`](../config/fedagent_ppo.yaml) | `fedagent.main_ppo_fed` (`@hydra.main(config_name="fedagent_ppo")`) | Training config for **one client**: composes verl's stock `ppo_trainer` via `hydra.searchpath` and overrides only the leaves FedAgent needs. |
| **Agent registry** | [`config/agent.yaml`](../config/agent.yaml) | verl's `AgentLoopManager` (via `actor_rollout_ref.rollout.agent.agent_loop_config_path`) | Maps each `agent_name` on a dataset row to its `AgentLoopBase` class. |
| **Env spec** | [`config/envs/*.yaml`](../config/envs/) | `fedagent.data.agentic_dataset.AgenticDataset` (via `data.train_files` / `data.val_files`) | Declares the env pool: one dataset row per episode (`n_envs` rows, distinct seeds). |
| **Federated-runner config** | `config/fed_*.yaml`, `config/paper/**/*.yaml` | `python -m fedagent.fed.run_fed --config <file>` | The **outer** layer: top-level federation knobs; keys == `run_fed.py`'s `DEFAULTS` dict. Drives the round loop, FedAvg, env services, and validation. |

The runner is outermost: `run_fed` reads the flat config, launches per-client env
services, then shells out to `main_ppo_fed` (which loads `fedagent_ppo.yaml`) **once per
client per round**, injecting `data.train_files=<env_spec>`, the model path, and the
`client_overrides` as Hydra CLI overrides.

### `fedagent_ppo.yaml` — the Hydra base

Composes verl's **stock `ppo_trainer`**, resolved through `hydra.searchpath` -> verl's
`trainer/config` dir (exported as `$VERL_CFG`; `run_fed` falls back to
`verl.__file__/trainer/config`):

```yaml
defaults:
  - ppo_trainer
  - _self_
hydra:
  searchpath:
    - file://${oc.env:VERL_CFG}
```

It overrides only FedAgent leaves: **GRPO** (`algorithm.adv_estimator: grpo`,
`use_kl_in_reward: false`), **group size G=8** (`rollout.n: 8`; smokes re-pin to 2 via
`client_overrides`), **async multi-turn rollout** (`rollout.name: vllm`, `mode: async`,
`multi_turn.enable: true`, `agent.default_agent_loop: gym_text`), the **paper actor
objective on every arm** (`use_kl_loss: true`, `kl_loss_coef: 0.01`,
`kl_loss_type: low_var_kl`, `entropy_coeff: 0.001` — verl 0.8 defaults differ), the
**custom dataset** (`data.custom_cls.name: AgenticDataset`), `reward_model.enable: false`,
and `trainer.logger: [console]`. Machine/run-specific leaves (`model.path`,
`data.{train,val}_files`, `custom_cls.path`, `agent_loop_config_path`,
`default_local_dir`) and the struct-additive
`+actor_rollout_ref.model.override_config.attn_implementation` are supplied on the CLI.

### `agent.yaml` and `envs/*.yaml`

`agent.yaml` is a list mapping `agent_name` -> `AgentLoopBase` `_target_`; the only entry
is `gym_text` -> `fedagent.agent_loops.gym_text_agent_loop.GymTextAgentLoop`, the concat
multi-turn loop every env uses. Each env spec lists pools; `AgenticDataset` emits `n_envs`
rows per pool (distinct seeds == distinct episodes). WebShop/ALFWorld are HTTP clients;
the service URL comes from `WEBSHOP_SERVICE_URL` / `ALFWORLD_SERVICE_URL` (set per client
by `run_fed`), so it is **not** pinned in the spec.

| Spec | `n_envs` | `max_turns` | Used for |
|---|---|---|---|
| `tiny_guess.yaml` | 64 | 6 | `TinyGuess`, in-process wiring proof (runner default `env_kind=tinyguess`). |
| `webshop.yaml` | 16 | 6 | WebShop smoke (small budget). |
| `webshop_15.yaml` | 8 | 15 | WebShop **GRPO** train (`n_envs=8` == original GRPO train_data_size). |
| `webshop_15_ppo.yaml` | 64 | 15 | WebShop **PPO** train (`n_envs=64` == original PPO train_data_size). |
| `webshop_15_val.yaml` | 500 | 15 | WebShop validation: held-out `goals[0:500]` on the full catalog. |
| `alfworld.yaml` | 8 | 50 | ALFWorld train (game shards; `max_turns=50` == original `max_steps`). |
| `alfworld_val.yaml` | — | 50 | ALFWorld validation (`eval_in_distribution` games). |

---

## Filename decoder — the `paper/` tree

`config/paper/` holds the full paper-scale runs in a family tree that **mirrors the
original FedAgent** `config/`. Every leaf is a flat runner config whose name encodes its
protocol:

```
fed_<env>_<algo>_total-<N>_cl-per-rd-<M>_rd-<T>_ep-per-cl-<E>_min-goals-per-cl-<G>_p-<strategy>_<knobs>.yaml
```

| Token | Runner key | Meaning |
|---|---|---|
| `<env>` | `env_kind` | `webshop` or `alfworld`. |
| `<algo>` | `adv_estimator` | `grpo` (no critic) or `ppo` (= `gae`, federates the critic). |
| `total-<N>` | `total_clients` | Client population N (`100`; `1` for centralized). |
| `cl-per-rd-<M>` | `clients_per_round` | Clients selected per round M (`2`; `1` for local/centralized). |
| `rd-<T>` | `total_rounds` | Communication rounds T (`70`). |
| `ep-per-cl-<E>` | `epochs_per_round` | Local epochs per client per round E (`3`). |
| `min-goals-per-cl-<G>` | `min_goals_per_client` | Minimum goals per client's shard (`100`). |
| `p-<strategy>` | `partition_strategy` | `uniform` (== IID, runner key `""`) or a heterogeneity strategy. |
| `<knobs>` | strategy knobs | Strategy params, e.g. `div-0.7_keep-0.7`, `omega-0.99`, `std-256`, `N-4`. |

The constant cell across the matrix is `total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100`
(N=100, M=2, T=70, E=3, G=100). Baselines and the decentralized ablations vary exactly one
of these tokens.

### Directory families

| Family | Layout | What varies |
|---|---|---|
| `uniform/<Model>/<setting>/<algo>/` | per-backbone IID + baselines | the **setting** (see below). |
| `env_heterogeneity/<strategy>[_ppo]/` | webshop only | the env-level perturbation strategy (`_ppo` = `adv_estimator: gae`). |
| `task_heterogeneity/<algo>/<env>/` | grpo+ppo × webshop+alfworld | the task-level partition (preference / coverage / hardness). |
| `decentralized/<change>/<algo>/` | webshop+alfworld | one protocol knob (`selected_cl_change`, `ep_per_round_change`, `samples_change`). |

**Backbones** (one `uniform/<Model>/` subdir each): `Qwen2.5-1.5B-Instruct`,
`Qwen2.5-3B-Instruct`, `Qwen2.5-7B-Instruct`, `Llama-3.2-3B-Instruct`. The
`env_heterogeneity`, `task_heterogeneity`, and `decentralized` trees are generated for the
1.5B backbone. `env_heterogeneity` is **webshop-only** (the catalog/BM25/lookalike/rank
arms perturb the WebShop catalog + search engine and have no ALFWorld analogue).

### Uniform settings

| Setting | Differs by | Runner keys |
|---|---|---|
| `main` | the IID anchor (seed 42) | `total_clients: 100`, `clients_per_round: 2`, `base_seed: 42`. |
| `main_seed1` / `main_seed2` | 3-seed replication | `base_seed: 21` / `84` (the original varied the shuffle seed 42/21/84). |
| `centralized` | one model on pooled data | `total_clients: 1`, `clients_per_round: 1` (FedAvg of one client == identity). |
| `local_client1` / `2` / `3` | "Local Agent Training" | `local_client_id: 21` / `42` / `84` (pin one client of 100, `clients_per_round: 1`, no federation). |

So **3-seed replication** = `base_seed` 42 / 21 / 84 across `main`, `main_seed1`,
`main_seed2`; the **Local** baselines pin clients `21`, `42`, `84`. Regenerate the whole
tree with `tools/verl08_migration/gen_paper_configs.py` (one invocation per backbone ×
env-kind × algo).

---

## Federated-runner key reference

Every key below is an entry in `run_fed.py`'s `DEFAULTS` dict; anything omitted from a
config falls back to the default. The CLI flags `--model-path --output-dir --rounds
--clients --n-gpus --base-seed --port-base --fedprox-mu --local-client-id` override the
YAML. Package-relative paths (`env_spec`, `val_env_spec`, `custom_cls_path`,
`agent_config_path`, `webshop_run_service`, `alfworld_run_service`) resolve against
`fedagent/`.

### Core loop

| Key | Default | Meaning |
|---|---|---|
| `model_path` | `""` | Base HF model dir for round 1; `""` => auto-discover a local Qwen2.5-0.5B-Instruct snapshot. |
| `output_dir` | `/tmp/xbb9020_fedagent_fed_tinyguess` | Run root: per-round client/aggregated checkpoints, logs, `federated_summary.json`. |
| `env_spec` | `config/envs/tiny_guess.yaml` | Env spec -> `data.{train,val}_files` for every client. |
| `custom_cls_path` | `data/agentic_dataset.py` | Path to `AgenticDataset` (-> `data.custom_cls.path`). |
| `agent_config_path` | `config/agent.yaml` | Agent-loop registry (-> `rollout.agent.agent_loop_config_path`). |
| `total_clients` | `2` | Client population N. |
| `clients_per_round` | `2` | Clients selected per round M (deterministic seeded sampling when `M < N`). |
| `total_rounds` | `2` | Communication rounds T. |
| `epochs_per_round` | `1` | Local epochs E per client per round (-> `trainer.total_epochs`). |
| `base_seed` | `42` | Master seed; per-(round,client) env seed = `base_seed + round*100 + client`. |
| `n_gpus_per_node` | `2` | FSDP world size per client run (== aggregator `nproc`). |
| `total_training_steps` | `1` | Per-client-round step cap (smokes); `<=0` => emit `null` so verl runs full E epochs. |
| `save_freq` | `1` | verl `trainer.save_freq` (paper configs use a huge value to save only the round's last step). |
| `weights` | `""` | FedAvg weights (e.g. by client data size); `""` => uniform average. |
| `wait_between_clients` | `5` | Seconds between sequential client runs (let Ray/GPU release). |
| `client_overrides` | `[]` | Extra `key=value` Hydra overrides applied to every client (see below). |
| `cleanup_checkpoints` | `True` | Delete consumed FSDP shards after each merge (keep HF + logs); disk hygiene. |
| `adv_estimator` | `grpo` | `grpo` (no critic) or `gae` (PPO: FedAvg actor **and** critic). |

### Env services

| Key | Default | Meaning |
|---|---|---|
| `env_kind` | `tinyguess` | `tinyguess` (in-process), `webshop`, or `alfworld` (remote services). |
| `webshop_run_service` | `envs/webshop/service/run_service.sh` | Launcher for a WebShop service. |
| `webshop_base_port` | `8080` | Client `c`'s service -> `webshop_base_port + c`. |
| `webshop_pool_size` | `8` | Env pool per WebShop service (must be `>= gen_batch`). |
| `search_return_n` | `200` | `WEBSHOP_SEARCH_RETURN_N`: BM25 top-K (paper=200; engine default 50 drops targets under env-het filtering). |
| `alfworld_run_service` | `envs/alfworld/service/run_service.sh` | Launcher for an ALFWorld service. |
| `alfworld_base_port` | `8200` | Client `c`'s service -> `alfworld_base_port + c`. |
| `alfworld_pool_size` | `4` | TextWorld env pool per ALFWorld service (must be `>= gen_batch`). |
| `alfworld_train_eval` | `train` | ALFWorld game split: `train` / `eval_in_distribution` / `eval_out_of_distribution`. |
| `alfworld_task_types` | `""` | `""` => all 6 types; else comma-sep IDs (1=Pick..6=Pick2) for the eval breakdown. |
| `service_health_timeout` | `900` | Seconds to wait for each service `/health` (pool warmup takes minutes). |

### Heterogeneity

| Key | Default | Meaning |
|---|---|---|
| `partition_strategy` | `""` | `""` (IID) \| `catalog_split`/`task_disjoint` (env) \| `preference`/`coverage`/`hardness` (task) \| `bm25_field_subset`/`bm25_reweight`/`lookalike`/`rank_wrapper` (env variants). |
| `env_div` | `0.7` | catalog-split heterogeneity strength. |
| `keep_ratio` | `0.7` | catalog-split distractor density. |
| `omega` | `0.5` | preference (task-het) Dirichlet spread. |
| `size_std` | `1.0` | coverage (task-het) Beta dispersion (xi). |
| `success_std` | `1.0` | hardness (task-het) Beta dispersion (xi'). |
| `variant_n` | `0` | env-variant arms (bm25/lookalike/rank): # variants in pool (`0` => fn default 2/4). |
| `trajectories_file` | `""` | hardness: **required** `task_id`->success-labels file. |
| `min_goals_per_client` | `100` | Minimum goals per client's shard. |

See [`./heterogeneity.md`](./heterogeneity.md) for the full taxonomy and how each knob maps
to an arm.

### Baselines

| Key | Default | Meaning |
|---|---|---|
| `local_client_id` | `-1` | `>=0` => **Local** baseline: train only this client of `total_clients`, no federation. |

**Mode selection** (all via the same schema): **Federated** = default (`total_clients=N>1`,
`local_client_id<0`); **Centralized** = `total_clients=1` (per-round FedAvg of one client is
the identity, so the loop is `T*E` epochs of centralized training); **Local** =
`local_client_id=k>=0`; **FedProx** = `fedprox_mu>0`; **PPO** = `adv_estimator=gae`.

### Eval (unperturbed global-model validation)

| Key | Default | Meaning |
|---|---|---|
| `val_env_spec` | `""` | `""` => **no eval**; else the UNPERTURBED val env-spec. |
| `test_freq` | `5` | Eval the aggregated global model every K rounds (+ the final round). |
| `val_before_train` | `True` | Also eval the base model before round 1 (the round-0 point). |
| `val_temperature` | `0.4` | Val sampling temperature (paper `val_kwargs.temperature=0.4`). |
| `webshop_val_port` | `8090` | Shared unperturbed WebShop val service port. |
| `alfworld_val_port` | `8290` | Shared unperturbed ALFWorld val service port. |
| `alfworld_val_split` | `eval_in_distribution` | ALFWorld val games (the 274-game in-distribution eval set). |

Eval scores the **global** model (base on round 0, else the round's aggregated HF) on one
shared unperturbed val service via a verl `val_only` pass (`adv_estimator=grpo`, no critic,
FedProx off), so every arm is measured on the same fixed set. A failed eval never aborts the
run — it is measurement, not the loop.

### FedProx

| Key | Default | Meaning |
|---|---|---|
| `fedprox_mu` | `0.0` | `>0` => client-side FedProx proximal term (else FedAvg). |

`fedprox_mu>0` is bridged to each client (and its Ray workers) via the env var `FEDPROX_MU`,
which `sitecustomize.py` reads at interpreter startup to patch `FSDPEngine.optimizer_step`
with the proximal term — chosen over a Ray `runtime_env` hook so verl's per-worker
`CUDA_VISIBLE_DEVICES` isolation is preserved.

---

## `client_overrides` and `adv_estimator`

`client_overrides` is a list of extra `key=value` **Hydra overrides** appended verbatim to
every client's `main_ppo_fed` command (and reused for eval, so the rollout shape matches).
It is where each arm pins the rollout/batch/context shape that the base `fedagent_ppo.yaml`
leaves at smoke defaults. The key ones:

| Override | Role |
|---|---|
| `actor_rollout_ref.rollout.n=8` | **GRPO group size G** (8 in the paper). |
| `data.train_batch_size=8` (PPO: `64`) | Prompts per optimizer step; pair with `actor_rollout_ref.actor.ppo_mini_batch_size`. |
| `data.max_prompt_length` / `max_response_length` (`2048` / `6144`) | Token budgets; mirror on `rollout.prompt_length` / `response_length`. |
| `actor_rollout_ref.rollout.max_model_len=8192` | vLLM context window (ALFWorld widens to `16384`). |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | vLLM KV-cache fraction (`0.5`–`0.6`). |

For **PPO** (`adv_estimator: gae`) the overrides also enable and shape the critic, and
`save_contents=[model]` makes the value-model checkpoint FedAvg-able:

```yaml
adv_estimator: gae
client_overrides:
  - actor_rollout_ref.actor.checkpoint.save_contents=[model]
  - critic.optim.lr=1e-5
  - critic.model.use_remove_padding=true
  - critic.model.enable_gradient_checkpointing=true
  - critic.fsdp.optimizer_offload=true
  - critic.ppo_micro_batch_size_per_gpu=4
  - critic.checkpoint.save_contents=[model]
  - trainer.critic_warmup=0
```

**GRPO vs PPO:** GRPO (the default, `rollout.n=G=8`, no critic) leaves the client command
byte-identical to the verified path. PPO (`adv_estimator=gae`) flips `need_critic` on; the
runner federates the value model **alongside the actor every round** (round-1 critic = the
base model, thereafter the aggregated critic), reusing the same FedAvg + merge machinery —
the merger auto-detects `...ForTokenClassification` vs `...ForCausalLM` from the shard's
`huggingface/config.json`.

---

## A worked config

A real `uniform/main/grpo` WebShop config (`fed_webshop_grpo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-uniform.yaml`):

```yaml
env_kind: webshop
env_spec: config/envs/webshop_15.yaml
val_env_spec: config/envs/webshop_15_val.yaml
output_dir: /tmp/xbb9020_fedpaper/uniform/webshop_grpo_uniform
model_path: Qwen/Qwen2.5-1.5B-Instruct

total_clients: 100
clients_per_round: 2
total_rounds: 70
epochs_per_round: 3
base_seed: 42

n_gpus_per_node: 4
total_training_steps: 0        # 0 => full E epochs/round (no per-round step cap)
save_freq: 100000              # save only the round's last step
test_freq: 5
val_before_train: true
val_temperature: 0.4
partition_strategy: ""         # IID

client_overrides:
  - data.train_batch_size=8
  - actor_rollout_ref.rollout.n=8
  - actor_rollout_ref.rollout.max_model_len=8192
  # ... prompt/response/mini-batch/gpu_mem leaves
```

Run it directly:

```bash
python -m fedagent.fed.run_fed \
    --config fedagent/config/paper/uniform/Qwen2.5-1.5B-Instruct/main/grpo/fed_webshop_grpo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-uniform.yaml \
    --model-path /path/to/Qwen2.5-1.5B-Instruct      # offline: a local snapshot
```

See [`./running.md`](./running.md) for modes, GPUs, and worked examples, and
[`./reproducing.md`](./reproducing.md) for the full matrix mapped to commands.
