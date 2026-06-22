"""ALFWorld environment — two halves of one env.

- ``alfworld_env.py``  the trainer-side ``AlfworldEnv`` (BaseTextEnv): a thin async HTTP
                       client, imported in the trainer env (``fedagent-verl08``).
- ``service/``         the out-of-process backend (FastAPI) that holds the real
                       ALFWorld/TextWorld env; runs in the ``verl-agent-alfworld`` conda
                       env and is **never** imported trainer-side.

Only the client is re-exported here, so ``import fedagent.envs.alfworld`` pulls just
``httpx`` + the ``BaseTextEnv`` contract — never ALFWorld's conflicting deps, which live
behind the HTTP boundary in ``service/``.
"""
from fedagent.envs.alfworld.alfworld_env import AlfworldEnv

__all__ = ["AlfworldEnv"]
