"""FedAgent heterogeneity partitioning for verl 0.8 (the science 'crown jewel').

Ported from verl-agent's ``agent_system/environments/partition_strategy.py`` (which
imports matplotlib/seaborn at module top, so the specific functions are copied
VERBATIM into clean numpy-only modules rather than imported). Each module preserves
the exact per-client assignment so catalogs/slices are bit-identical to the 0.3.1
baseline (science red line: deterministic per-client-id assignment, base_seed=42).
"""
