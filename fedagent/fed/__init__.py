"""FedAgent federated loop for verl 0.8 (thin overlay).

``run_fed`` is the verl-0.8 replacement for ``core/custom_fed_server.py``'s round
loop: one training subprocess per (client, round) -> FedAvg the FSDP checkpoints
-> re-enter the next round from the aggregated model. Verl-agnostic: each client
is just ``python -m fedagent.main_ppo_fed``; aggregation shells out to the
validated matched-PG aggregator under torchrun.
"""
