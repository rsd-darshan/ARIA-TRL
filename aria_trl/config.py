"""Configuration for ARIA continual learning in LLM fine-tuning."""

from dataclasses import dataclass


@dataclass
class ARIAConfig:
    """
    ARIA configuration for LLM continual learning.

    Attributes:
        plasticity_lambda: Weight for bimodal gate specialization loss.
            Higher → stronger push toward 0/1 gating (faster decision).
            Recommended: 0.01-0.05

        spc_lambda: Weight for Slow-Pathway Consolidation (Fisher penalty).
            Higher → stronger protection of old task knowledge.
            Recommended: 50-200

        adapter_dim: Bottleneck dimension of task-specific adapters.
            Recommended: 32-128 (trade-off between efficiency and capacity)

        slow_lr_ratio: Learning rate multiplier for slow pathway, on top of
            the gate-driven gradient dampening by (1-mean_gate) the trainer
            already applies every step. Default is 1.0 — one throttle, not
            two multiplied together. Lower this only if you've confirmed the
            gate-driven dampening alone isn't stabilizing the slow pathway
            enough; stacking both by default just makes the slow pathway
            learn at a fraction of a fraction of base_lr for no added benefit.

        warmup_steps: Steps before the plasticity (gate-specialization) loss
            activates. None (default) auto-computes one epoch's worth of
            steps from the trainer's actual train_dataset size and batch
            size, so the gate reliably specializes within the task's real
            step budget instead of a fixed guess that may never be reached
            at small scale or fire too late at large scale. Pass an explicit
            int only if you specifically want a different warmup length.

        consolidation_steps_per_task: Number of eval steps used for Fisher estimation.
            If None, use entire eval dataset.
            Recommended: None (use entire eval set for stable Fisher)
    """

    plasticity_lambda: float = 0.01
    spc_lambda: float = 100.0
    adapter_dim: int = 64
    slow_lr_ratio: float = 1.0
    warmup_steps: int | None = None
    consolidation_steps_per_task: int | None = None
