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

        slow_lr_ratio: Learning rate multiplier for slow pathway.
            slow_lr = base_lr * slow_lr_ratio
            Recommended: 0.1-0.5 (slower updates to stable knowledge)

        warmup_steps: Steps before plasticity loss activates.
            Prevents early gate collapse during initialization.
            Recommended: 500-1000

        consolidation_steps_per_task: Number of eval steps used for Fisher estimation.
            If None, use entire eval dataset.
            Recommended: None (use entire eval set for stable Fisher)
    """

    plasticity_lambda: float = 0.01
    spc_lambda: float = 100.0
    adapter_dim: int = 64
    slow_lr_ratio: float = 0.5
    warmup_steps: int = 500
    consolidation_steps_per_task: int | None = None
