"""WebShop remote env service (runs in the verl-agent-webshop conda env, NOT the trainer env).

Kept separate from ``fedagent.envs`` so importing the package in the trainer env never
pulls WebShop's conflicting deps (gym 0.24 / pyserini / torch 2.6). Only the HTTP
client ``fedagent.envs.webshop.WebShopEnv`` is imported trainer-side.
"""
