"""WebShop remote env service — wraps the in-process WebAgentTextEnv behind HTTP.

Runs in the ``verl-agent-webshop`` conda env (has gym 0.24 / pyserini / Java / the
Lucene index). The verl-0.8 trainer (incompatible env) drives WebShop through the thin
``fedagent.envs.webshop.WebShopEnv`` HTTP client. We:

  - pre-warm a POOL of ``WebAgentTextEnv`` instances (``gym.make`` ~26s each) so episodes
    don't pay JVM+index startup;
  - serve episodes via borrow(``/create``) -> ``/reset(goal)`` -> ``/step(text)``* -> return(``/close``);
  - parse the model's action text SERVER-SIDE with the original ``webshop_projection``
    (loaded in isolation), then call the gym env -- mirroring verl-agent's WebshopWorker.

Launch via ``webshop_service/run_service.sh``. Phase 4: heterogeneity env_kwargs
(catalog_filter_asins / bm25_in_memory_config / ...) get read from the environment here
so the whole pool reflects one client's variant.
"""
import asyncio
import importlib.util
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

_HERE = os.path.dirname(os.path.abspath(__file__))
_VERL_AGENT = os.path.abspath(os.path.join(_HERE, "..", "..", "third_party", "verl-agent"))
_WEBSHOP = os.path.join(
    _VERL_AGENT, "agent_system", "environments", "env_package", "webshop", "webshop"
)
if _WEBSHOP not in sys.path:
    sys.path.append(_WEBSHOP)

# Load the original action parser in isolation (it only imports re/typing), avoiding
# the agent_system package __init__ (which would pull verl-0.3.1/torch).
_PROJ = os.path.join(
    _VERL_AGENT, "agent_system", "environments", "env_package", "webshop", "projection.py"
)
_spec = importlib.util.spec_from_file_location("webshop_projection_mod", _PROJ)
_proj = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_proj)
webshop_projection = _proj.webshop_projection

POOL_SIZE = int(os.environ.get("WEBSHOP_POOL_SIZE", "4"))
NUM_GOALS = int(os.environ.get("WEBSHOP_NUM_GOALS", "6910"))  # confirmed pool size in standalone smoke
ENV_KWARGS = {"observation_mode": "text", "num_products": None}

_pool: asyncio.Queue = None
_sessions: dict = {}


def _make_env(seed: int):
    import gym
    from web_agent_site.envs import WebAgentTextEnv  # noqa: F401  (registers the gym id)

    return gym.make("WebAgentTextEnv-v0", **dict(ENV_KWARGS, seed=seed))


def _avail(env) -> dict:
    try:
        return env.get_available_actions()
    except Exception:
        return {"has_search_bar": False, "clickables": []}


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _pool
    _pool = asyncio.Queue()
    envs = await asyncio.gather(*[asyncio.to_thread(_make_env, i) for i in range(POOL_SIZE)])
    for e in envs:
        _pool.put_nowait(e)
    print(f"[webshop-service] warmed {POOL_SIZE} envs (NUM_GOALS={NUM_GOALS})", flush=True)
    yield
    while _pool is not None and not _pool.empty():
        try:
            _pool.get_nowait().close()
        except Exception:
            pass


app = FastAPI(lifespan=_lifespan)


class Sid(BaseModel):
    session_id: str


class ResetReq(BaseModel):
    session_id: str
    seed: int = 0


class StepReq(BaseModel):
    session_id: str
    text: str


@app.get("/health")
async def health():
    return {"ok": True, "free": _pool.qsize() if _pool else 0, "sessions": len(_sessions)}


@app.post("/create")
async def create(r: Sid):
    env = await _pool.get()  # borrow (waits if the pool is exhausted)
    _sessions[r.session_id] = env
    return {"ok": True}


@app.post("/reset")
async def reset(r: ResetReq):
    env = _sessions.get(r.session_id)
    if env is None:
        raise HTTPException(404, "unknown session")

    def _do():
        res = env.reset(session=int(r.seed) % NUM_GOALS)
        obs = res[0] if isinstance(res, tuple) else res
        return obs, _avail(env)

    obs, avail = await asyncio.to_thread(_do)
    return {"obs": obs, "available_actions": avail}


@app.post("/step")
async def step(r: StepReq):
    env = _sessions.get(r.session_id)
    if env is None:
        raise HTTPException(404, "unknown session")

    def _do():
        acts, valids = webshop_projection([r.text])  # parse <action>..</action> server-side
        obs, reward, done, info = env.step(acts[0])
        info = info or {}
        won = bool(info.get("won", float(reward) == 1.0))
        return obs, float(reward), bool(done), _avail(env), won, int(valids[0])

    obs, reward, done, avail, won, valid = await asyncio.to_thread(_do)
    return {
        "obs": obs,
        "reward": reward,
        "done": done,
        "available_actions": avail,
        "success": won,
        "is_action_valid": valid,
    }


@app.post("/close")
async def close(r: Sid):
    env = _sessions.pop(r.session_id, None)
    if env is not None:
        _pool.put_nowait(env)  # return to the pool for the next episode
    return {"ok": True}
