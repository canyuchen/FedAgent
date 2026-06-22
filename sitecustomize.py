"""Repo-root sitecustomize: auto-imported by CPython at interpreter startup for every
process that has this repo root on PYTHONPATH (the federated driver AND its Ray workers,
since run_fed sets PYTHONPATH=REPO_ROOT and Ray workers inherit it).

Purpose: inject FedAgent's FedProx optimizer patch into every actor worker WITHOUT a Ray
`runtime_env.worker_process_setup_hook`. That hook works (the patch fires) but the
cluster-level `runtime_env` clobbers verl's per-worker `CUDA_VISIBLE_DEVICES` assignment,
so all FSDP ranks land on GPU 0 ("Duplicate GPU detected: rank N and rank 0 both on CUDA
device ..."). sitecustomize runs at plain interpreter startup, touches no runtime_env, and
so leaves verl's GPU isolation intact.

Safety: this runs in EVERY python process on PYTHONPATH, including the WebShop / ALFWorld
env-service conda envs (which do NOT have verl installed). It is therefore:
  - gated on FEDPROX_MU being set (normal FedAvg runs / services never set it -> no-op,
    and `fedagent` is never even imported), and
  - wrapped in a broad try/except (a missing verl/fedagent, or any import error in a
    non-trainer env, degrades to a silent no-op rather than breaking the process).
The patch itself (monkeypatching FSDPEngine.optimizer_step) is CUDA-free -- importing
FSDPEngine does not initialize CUDA -- so it is safe to run before verl assigns devices.
"""
import os

if os.environ.get("FEDPROX_MU"):
    try:
        from fedagent.fedprox import maybe_enable_from_env

        maybe_enable_from_env()
    except Exception:
        # non-trainer env (e.g. env-service conda env without verl), or any startup-time
        # import issue -> no-op. FedProx simply will not be active in that process.
        pass
