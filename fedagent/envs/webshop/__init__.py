"""WebShop environment — two halves of one env.

- ``webshop_env.py``  the trainer-side ``WebShopEnv`` (BaseTextEnv): a thin async HTTP
                      client, imported in the trainer env (``fedagent-verl08``).
- ``service/``        the out-of-process backend (FastAPI) that holds the real WebShop
                      gym env + Lucene/Java; runs in the ``verl-agent-webshop`` conda env
                      and is **never** imported trainer-side.

Only the client is re-exported here, so ``import fedagent.envs.webshop`` pulls just
``httpx`` + the ``BaseTextEnv`` contract — never WebShop's conflicting deps, which live
behind the HTTP boundary in ``service/``.
"""
from fedagent.envs.webshop.webshop_env import WebShopEnv

__all__ = ["WebShopEnv"]
