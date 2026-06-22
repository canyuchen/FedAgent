"""WebShop remote env service backend (runs in the verl-agent-webshop conda env, NOT the trainer env).

Lives at ``fedagent.envs.webshop.service`` but is **never imported trainer-side** — only the
sibling client ``fedagent.envs.webshop.WebShopEnv`` (``../webshop_env.py``) is. Importing this
subpackage pulls WebShop's conflicting deps (gym 0.24 / pyserini / torch 2.6), so it is loaded
only by ``run_service.sh`` (``uvicorn fedagent.envs.webshop.service.server:app``) inside that env.
The empty ``fedagent.envs.__init__`` + client-only ``fedagent.envs.webshop.__init__`` keep the
import chain inert, so the package can live under ``envs/`` without leaking deps trainer-side.
"""
