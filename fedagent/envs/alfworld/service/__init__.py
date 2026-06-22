"""ALFWorld remote env service backend (runs in the verl-agent-alfworld conda env, NOT the trainer env).

Lives at ``fedagent.envs.alfworld.service`` but is **never imported trainer-side** — only the
sibling client ``fedagent.envs.alfworld.AlfworldEnv`` (``../alfworld_env.py``) is. Importing this
subpackage pulls ALFWorld's heavy/conflicting deps (alfworld / textworld / gymnasium / torch +
torchvision pinned for the env), so it is loaded only by ``run_service.sh``
(``uvicorn fedagent.envs.alfworld.service.server:app``) inside that env. Mirrors
``fedagent.envs.webshop.service`` exactly.
"""
