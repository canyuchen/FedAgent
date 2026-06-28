#!/usr/bin/env python
"""Compare two FSDP actor/critic checkpoint dirs tensor-by-tensor (equivalence harness for
lever #4, docs/acceleration.md). Used to prove that the PERSISTENT-trainer path produces
per-client checkpoints numerically identical (within fp noise) to the SUBPROCESS path.

Both dirs must hold matching ``model_world_size_<W>_rank_<R>.pt`` shards (same world_size +
FSDP config -> identical sharding, so we can compare rank-by-rank). Reports per-rank and
overall max/mean abs diff + the worst-offending parameter, and exits non-zero if any tensor
exceeds --atol (default 0: bit-identical; bump for bf16/optimizer-noise tolerance).

    python tools/verl08_migration/compare_fsdp_checkpoints.py \
        --a round_1/client_0/checkpoints/global_step_1/actor \
        --b round_1/client_0_persistent/checkpoints/global_step_1/actor \
        --atol 1e-6
"""
import argparse
import sys
from pathlib import Path

import torch


def shard_files(d: Path):
    """{(world_size, rank): path} for every model shard in dir d."""
    out = {}
    for p in sorted(d.glob("model_world_size_*_rank_*.pt")):
        try:
            ws = int(p.name.split("model_world_size_")[1].split("_rank_")[0])
            rk = int(p.name.split("_rank_")[1].split(".pt")[0])
        except (ValueError, IndexError):
            continue
        out[(ws, rk)] = p
    return out


def load_state(path: Path) -> dict:
    """Load a shard to CPU as a flat {name: tensor} dict (handles raw state_dict or wrappers)."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and not any(torch.is_tensor(v) for v in obj.values()):
        for key in ("model", "state_dict", "module"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    return {k: v for k, v in obj.items() if torch.is_tensor(v)}


def compare_dir(a: Path, b: Path, atol: float) -> int:
    sa, sb = shard_files(a), shard_files(b)
    if not sa:
        print(f"ERROR: no shards in {a}", file=sys.stderr); return 2
    if set(sa) != set(sb):
        print(f"ERROR: shard sets differ\n  A: {sorted(sa)}\n  B: {sorted(sb)}", file=sys.stderr); return 2

    g_max = 0.0; g_sum = 0.0; g_n = 0; worst = (None, 0.0); mism = 0
    for key in sorted(sa):
        ta, tb = load_state(sa[key]), load_state(sb[key])
        if set(ta) != set(tb):
            only_a = set(ta) - set(tb); only_b = set(tb) - set(ta)
            print(f"  shard {key}: KEY MISMATCH (+{len(only_b)} -{len(only_a)}) e.g. {list((only_a|only_b))[:3]}")
            mism += 1; continue
        s_max = 0.0
        for name, va in ta.items():
            vb = tb[name]
            if va.shape != vb.shape:
                print(f"  shard {key} {name}: SHAPE {tuple(va.shape)} vs {tuple(vb.shape)}"); mism += 1; continue
            d = (va.float() - vb.float()).abs()
            m = float(d.max()) if d.numel() else 0.0
            g_sum += float(d.sum()); g_n += d.numel()
            if m > worst[1]: worst = (f"{key}:{name}", m)
            s_max = max(s_max, m)
        g_max = max(g_max, s_max)
        print(f"  shard ws{key[0]}_rank{key[1]}: max|Δ|={s_max:.3e}")

    mean = g_sum / g_n if g_n else 0.0
    print(f"\n  OVERALL max|Δ|={g_max:.3e}  mean|Δ|={mean:.3e}  worst={worst[0]} ({worst[1]:.3e})")
    ok = (g_max <= atol) and mism == 0
    print(f"  VERDICT: {'EQUIVALENT' if ok else 'DIFFERENT'} (atol={atol:g}, key/shape mismatches={mism})")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="Compare two FSDP checkpoint dirs tensor-by-tensor")
    ap.add_argument("--a", required=True, help="checkpoint dir A (e.g. subprocess path actor/)")
    ap.add_argument("--b", required=True, help="checkpoint dir B (e.g. persistent path actor/)")
    ap.add_argument("--atol", type=float, default=0.0, help="max abs diff allowed (0 = bit-identical)")
    args = ap.parse_args()
    print(f"compare A={args.a}\n        B={args.b}")
    sys.exit(compare_dir(Path(args.a), Path(args.b), args.atol))


if __name__ == "__main__":
    main()
