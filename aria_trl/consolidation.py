"""Slow-Pathway Consolidation (SPC) via Fisher Information estimation."""

from typing import Dict, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class FisherConsolidator:
    """
    Estimates and stores Fisher Information for slow-pathway weights.

    After each task, estimates diagonal Fisher on slow-pathway weights
    using the validation dataset. Stores means (θ*) and Fisher (F) for
    use as regularization penalty in subsequent tasks.

    Fisher estimation:
        F_s = E[(∇_θ_s log p(y|x))²]  computed on validation data
    """

    def __init__(self, model: nn.Module, device: torch.device):
        """
        Args:
            model: The LLM model being fine-tuned
            device: torch.device (cuda, cpu, etc.)
        """
        self.model = model
        self.device = device
        self.task_means: List[Dict[str, torch.Tensor]] = []
        self.task_fishers: List[Dict[str, torch.Tensor]] = []

    def _get_slow_named_parameters(self):
        """Iterate over slow-pathway parameters (from PlasticityGatedMLP layers)."""
        for name, module in self.model.named_modules():
            if hasattr(module, "slow_parameters"):
                # This is a PlasticityGatedMLP
                for i, param in enumerate(module.slow_parameters()):
                    # Create unique name for this slow param
                    param_name = f"{name}.slow_param_{i}"
                    yield param_name, param

    def consolidate(
        self,
        task_id: int,
        eval_loader: DataLoader,
        max_steps: int | None = None,
    ) -> None:
        """
        Estimate Fisher Information on slow-pathway weights using eval data.

        Args:
            task_id: Which task just finished
            eval_loader: Validation DataLoader
            max_steps: Limit to this many batches (None = all batches)
        """
        self.model.eval()

        # Initialize storage
        means = {}
        fishers = {}
        for name, param in self._get_slow_named_parameters():
            means[name] = param.detach().cpu().clone()
            fishers[name] = torch.zeros_like(param, device="cpu")

        # Accumulate Fisher over eval set
        num_batches = 0
        for batch_idx, batch in enumerate(eval_loader):
            if max_steps is not None and batch_idx >= max_steps:
                break

            # Handle different input formats
            if isinstance(batch, dict):
                batch = {k: v.to(self.device) for k, v in batch.items()}
            else:
                batch = batch[0].to(self.device), batch[1].to(self.device)

            self.model.zero_grad()

            # Forward pass
            if isinstance(batch, dict):
                outputs = self.model(**batch)
            else:
                input_ids, labels = batch
                outputs = self.model(input_ids=input_ids, labels=labels)

            # Compute loss for backward
            if hasattr(outputs, "loss"):
                loss = outputs.loss
            else:
                # Fallback if loss not in output
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    batch[1].view(-1),
                )

            loss.backward()

            # Accumulate squared gradients (diagonal Fisher)
            for name, param in self._get_slow_named_parameters():
                if param.grad is not None:
                    fishers[name] += param.grad.data.cpu() ** 2

            num_batches += 1

        # Average Fisher
        if num_batches > 0:
            for name in fishers:
                fishers[name] /= num_batches

        # Store for this task
        self.task_means.append(means)
        self.task_fishers.append(fishers)

    def compute_spc_loss(self, current_step: int) -> torch.Tensor:
        """
        Compute SPC regularization loss (Fisher-weighted penalty).

        Penalizes deviation of current slow weights from means stored
        after previous tasks, weighted by Fisher Information.

        Args:
            current_step: Current training step (for optional decay)

        Returns:
            Scalar tensor: SPC regularization loss
        """
        if not self.task_fishers:
            return torch.tensor(0.0, device=self.device)

        spc_loss = torch.tensor(0.0, device=self.device)

        for means, fishers in zip(self.task_means, self.task_fishers):
            for name, param in self._get_slow_named_parameters():
                if name in means and name in fishers:
                    mean = means[name].to(self.device)
                    fisher = fishers[name].to(self.device)

                    # Fisher-weighted squared deviation
                    spc_loss = spc_loss + (fisher * (param - mean) ** 2).sum()

        return spc_loss

    def reset(self):
        """Clear all stored Fisher information and means (start fresh)."""
        self.task_means = []
        self.task_fishers = []
