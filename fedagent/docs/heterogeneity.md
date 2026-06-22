# Heterogeneity

FedAgent's scientific core is a **two-level heterogeneity suite**: a set of
partition strategies that inject *controlled* statistical difference across the
federated clients, separated into two structurally different channels so the
paper's headline claim can be **measured** rather than assumed — federated agent
RL is **robust** to **task-level** heterogeneity (the task descriptor is in the
prompt, so the policy can condition on it) but **worst-case non-robust** to
**environment-level** heterogeneity (the transition kernel is hidden, so the
policy only senses it through successor states). This is the *Input-Dynamics
Asymmetry*.

This guide is the operator's view: which arms exist, the exact `run_fed` knob
that selects each, the paper values, and how to wire one up. For the
construction internals (the verbatim partition functions, the seed-42 science
red line, the Beta-sizing helpers) read [`../hetero/README.md`](../hetero/README.md);
for where the suite sits on stock verl 0.8 read [`../README.md`](../README.md).

## The two levels

- **Environment-level** strategies perturb the **transition kernel / catalog** —
  the agent never observes the change directly, only through the successor states
  the environment returns. On WebShop these are `catalog_split` (disjoint
  per-client product catalog) plus the four env-variant arms `bm25_field_subset`,
  `bm25_reweight`, `lookalike`, and `rank_wrapper`, which perturb the four stages
  of WebShop's retrieval pipeline (encoding / matching / content+matching /
  rendering).
- **Task-level** strategies perturb the **goal distribution over a shared,
  unperturbed environment** (the full catalog), and the goal is observable in the
  prompt: `preference` (skewed goal-category mix), `coverage` (unequal goal-count
  coverage), and `hardness` (easy/hard goal mix). `task_disjoint` is the clean
  ablation against `catalog_split` — it serves the **same** disjoint goal slice
  but over the **full** catalog, so any divergence between the two isolates the
  environment effect.

Throughout the task-level arms the transition kernel is held fixed; throughout
the environment-level arms the task split is held **uniform** — so divergence in
each sweep is attributable to that level alone. Every arm is scored on the same
shared **unperturbed** validation service (`partition_strategy=""` /
`"uniform"`), which is what makes the cross-arm curves comparable.

## The arms

Every arm is selected by the single `run_fed` key `partition_strategy` plus that
strategy's knob; `run_fed.py` exports them as env vars to each per-client remote
env service, which calls the matching `*_for_client` constructor in
[`../hetero/`](../hetero/). Paper values are the endpoints actually present in
`config/paper/{env,task}_heterogeneity/`.

| `partition_strategy` | Level | Env(s) | Knob (`run_fed` key) | Paper values | Controls |
|---|---|---|---|---|---|
| `catalog_split` | environment | WebShop | `env_div`, `keep_ratio` | `env_div` ∈ {0.0, 0.3, 0.7, 1.0}, `keep_ratio` 0.7 | Cross-client catalog divergence (+ distractor density); disjoint goal slice |
| `task_disjoint` | task | WebShop | `env_div`, `keep_ratio` | same as `catalog_split` | The **same** disjoint goal slice but the **full** catalog (env-effect ablation) |
| `preference` | task | WebShop, ALFWorld | `omega` | `omega` ∈ {0.01, 0.99} | Dirichlet spread of the per-client goal-**category** marginal |
| `coverage` | task | WebShop, ALFWorld | `size_std` | `size_std` ∈ {256, 1} | Beta dispersion of per-client goal **count** (larger = more uniform) |
| `hardness` | task | WebShop, ALFWorld | `success_std` (+ `trajectories_file`) | `success_std` ∈ {256, 1} | Beta dispersion of the per-client easy/hard goal mix |
| `bm25_field_subset` | environment | WebShop | `variant_n` | `variant_n` ∈ {4, 8} | Search-index **field subset** (encoding stage) |
| `bm25_reweight` | environment | WebShop | `variant_n` | `variant_n` ∈ {4, 8} | BM25 **scoring** (`k1`/`b`) reweighting (matching stage) |
| `lookalike` | environment | WebShop | `variant_n` | `variant_n` ∈ {2, 4} | Adversarial **lookalike products** appended to the catalog (content + matching) |
| `rank_wrapper` | environment | WebShop | `variant_n` | `variant_n` 4 | Search-engine **ranking** wrapper (rendering stage) |

Notes on the knobs:

- `env_div` ∈ [0, 1] sets how far the per-client catalogs diverge; `keep_ratio`
  sets the distractor density kept around each client's protected target ASINs.
  Both feed `catalog_split_for_client`. The `env_div=0.0` config is the
  near-homogeneous floor of the Catalog Split sweep.
- For `coverage`/`hardness`, `size_std`/`success_std` are the Beta
  **concentration** (forwarded as the dispersion parameter), so they set the
  spread *inversely*: the large endpoint (256) is near-uniform, the small
  endpoint (1) is the extreme imbalance.
- `variant_n` is the number of variants in the per-client pool (passed as `N`).
  A value of `0` in the runner defaults means "use the constructor's own default"
  (4 for bm25/rank, 2 for lookalike); the paper configs set it explicitly. One
  constructor (`bm25_variant_for_client`) serves **both** bm25 arms — the service
  passes `variant_pool="fields_only"` for `bm25_field_subset` and the default
  pool for `bm25_reweight`.

## Selecting an arm

Set `partition_strategy` and the strategy's knob(s) in the `run_fed` YAML, then
launch with `python -m fedagent.fed.run_fed --config <yaml>`. Examples copied
from the paper config tree.

Catalog Split (environment level), the four-point `env_div` sweep at
`keep_ratio: 0.7`
(`config/paper/env_heterogeneity/catalog_split/...div-0.7_keep-0.7.yaml`):

```yaml
env_kind: webshop
search_return_n: 200          # env-het perturbs the catalog -> paper top-K (>= 100 required)
partition_strategy: catalog_split
env_div: 0.7
keep_ratio: 0.7
```

Preference (task level), the extreme endpoint
(`config/paper/task_heterogeneity/grpo/webshop/...preference_omega-0.99.yaml`):

```yaml
env_kind: webshop
partition_strategy: preference
omega: 0.99                    # 0.01 = near-uniform, 0.99 = extreme
```

Hardness (task level) needs a per-backbone success-labels file
(`config/paper/task_heterogeneity/grpo/webshop/...hardness_success_std-1.yaml`):

```yaml
env_kind: webshop
partition_strategy: hardness
success_std: 1
trajectories_file: data/hardness/qwen2.5-1.5b_webshop_trajectories.json
```

An env-variant arm (environment level) is just a strategy name plus `variant_n`
(`config/paper/env_heterogeneity/bm25_reweighting/...bm25_reweighting_N-4.yaml`):

```yaml
env_kind: webshop
search_return_n: 200
partition_strategy: bm25_reweight   # or bm25_field_subset / lookalike / rank_wrapper
variant_n: 4
```

The env-variant arms set `search_return_n: 200` (the runner aborts under 100):
raising the BM25 top-K keeps the rendered result page full after aggressive
per-client filtering so a target is never silently dropped. The task-level arms
leave it at the engine default (50), matching the non-het baselines.

## Which arms apply to which env

- **WebShop** gets the **whole** suite: both environment-level families (catalog
  filter + the four retrieval-pipeline variants) and all three task-level arms.
  These live under `config/paper/env_heterogeneity/` (WebShop-only) and
  `config/paper/task_heterogeneity/{grpo,ppo}/webshop/`.
- **ALFWorld** gets only the **env-agnostic** subset — the three task-level arms
  (`preference` / `coverage` / `hardness`) plus uniform/homogeneous, under
  `config/paper/task_heterogeneity/{grpo,ppo}/alfworld/`. There is **no**
  `env_heterogeneity/` ALFWorld arm: the WebShop env variants perturb WebShop's
  retrieval pipeline specifically and do not transfer. For ALFWorld the runner
  forwards only the knobs the strategy needs (`omega` / `size_std` /
  `success_std` / `trajectories_file`) into its remote service.

## The hardness labels file

`hardness` is the one arm with a required external input: a `trajectories_file`
mapping `task_id -> success`, used to bucket goals into high/low success before
the Beta allocation assigns each client its easy-goal count. There is **no
usable default** — the constructor raises if the file is missing.

Generate it **once per backbone** (the labels depend on the reference policy)
with `tools/verl08_migration/gen_hardness_trajectories.py`: it rolls the
reference checkpoint over the whole training catalog and writes per-goal success
labels keyed on the **same** `task_id` formula the partitioner uses
(`f"{asin}_{md5(goal_options)}"`), so the lookup resolves only against the real
goal dicts. Then point each Hardness config's `trajectories_file` at the output.

## Load-bearing invariant

The three task-level partitioners (`preference` / `coverage` / `hardness`) are
**content-dependent**: which goals a client gets depends on each goal's
category / size / hardness. So the WebShop service **defers** them to runtime and
partitions the env's **real, seed-42-shuffled `server.goals`** once the env pool
is warmed — the served goal at index *i* carries exactly the property the
partition selected. (The environment-level arms are safe to compute at import:
`catalog_split` / `task_disjoint` use an order-independent contiguous index
range, and the bm25/lookalike/rank variants keep the goal split uniform and only
return an `env_kwargs` fragment merged into `gym.make`.) Do not swap the
deferred task-level partitions onto a reconstructed-goal list — the offline
fallback is not order-faithful and will not match a real labels file.

---

See [`./reproducing.md`](./reproducing.md) for the full config-to-figure mapping
and [`./configuration.md`](./configuration.md) for the complete `run_fed` field
reference. The constructor internals and the env-var bridge are documented in
[`../hetero/README.md`](../hetero/README.md).
