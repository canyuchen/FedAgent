"""AgenticDataset — verl ``custom_cls`` dataset that emits one row per env instance.

verl 0.8 loads this via ``data.custom_cls.{path,name}``; its non-tensor columns are
passed as ``**kwargs`` to ``AgentLoop.run()`` (so ``env_name``/``seed``/``config``/
``max_turns`` reach the agent-loop). This is the verl-0.8 equivalent of verl-agent's
``AgenticDataset`` / ``fed_make_envs`` env enumeration.

Input: an env-spec YAML (``data.train_files``) of the form::

    envs:
      - name: TinyGuess
        n_envs: 64
        max_turns: 6
        agent_name: gym_text     # optional (default: gym_text)
        config: {lo: 1, hi: 50}

Output: ``n_envs`` rows per spec, each a distinct env instance (distinct ``seed``).
GRPO grouping is handled downstream by verl's ``rollout.n`` (each row is repeated n
times -> one GRPO group per env instance).

Stock-trainer contract (validated in Phase 0(b)): stock verl ``_get_gen_batch`` does
NOT pop tensor keys before unioning the agent-loop output back onto the batch, so the
dataset must NOT emit ``input_ids``/``attention_mask``/``position_ids`` (the agent-loop
generates them). We emit a single non-colliding ``ds_dummy`` tensor purely for batch
sizing. (VAGEN can emit dummy input_ids only because it forks ``_get_gen_batch``; we
don't fork the trainer.)

PHASE 4 SEAM: ``_partition_specs`` is where FedAgent heterogeneity plugs in --
``partition_strategy.py`` will turn the global env pool into this client's slice,
driven by env vars (PARTITION_STRATEGY, CLIENT_ID, CLIENT_NUM, OMEGA, SIZE_STD,
SUCCESS_STD, ENV_DIV, HOLDOUT_FILE, ...), preserving deterministic-by-client-id
assignment. For now (non-federated) it is identity.
"""
import os
from typing import Any, Dict, List

import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

DEFAULT_AGENT_LOOP = "gym_text"


class AgenticDataset(Dataset):
    def __init__(self, data_files, tokenizer=None, processor=None, config=None, **kwargs):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        specs = self._load_specs(data_files)
        specs = self._partition_specs(specs)

        # base seed for per-instance variety; the federated bridge sets this per
        # client-round (Phase 4/6) so each client sees a distinct, reproducible draw.
        base_seed = int(os.environ.get("FEDAGENT_BASE_SEED", 0))

        self.items: List[Dict[str, Any]] = []
        for si, s in enumerate(specs):
            n = int(s.get("n_envs", 64))
            max_turns = int(s.get("max_turns", 6))
            env_cfg = s.get("config", {}) or {}
            name = s.get("name", "TinyGuess")
            agent_name = s.get("agent_name", DEFAULT_AGENT_LOOP)
            for i in range(n):
                self.items.append(
                    {
                        "env_name": name,
                        "seed": base_seed * 100_000 + si * 1_000 + i,
                        "config": env_cfg,
                        "max_turns": max_turns,
                        "agent_name": agent_name,
                        "data_source": name.lower(),
                        # verl's agent-loop postprocess stores kwargs["raw_prompt"]; our
                        # loop builds its own prompt from the env, so this is a placeholder.
                        "raw_prompt": [{"role": "user", "content": f"<{name} episode>"}],
                        # single non-colliding dummy tensor (see module docstring).
                        "ds_dummy": torch.tensor([0]),
                    }
                )

    @staticmethod
    def _load_specs(data_files) -> List[Dict[str, Any]]:
        path = data_files[0] if isinstance(data_files, (list, tuple)) else data_files
        try:
            cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
            specs = cfg.get("envs", []) if isinstance(cfg, dict) else []
        except Exception:
            specs = []
        if not specs:
            specs = [{"name": "TinyGuess", "n_envs": 64, "max_turns": 6, "config": {"lo": 1, "hi": 50}}]
        return specs

    def _partition_specs(self, specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """PHASE 4 hook: apply FedAgent heterogeneity partitioning. Identity for now."""
        return specs

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]
