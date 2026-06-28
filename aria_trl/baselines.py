"""Baseline continual learning methods for comparison."""

import torch
import torch.nn as nn
from typing import Dict, List
from transformers import PreTrainedModel, TrainingArguments


class EWCTrainer:
    """Elastic Weight Consolidation baseline."""

    def __init__(
        self,
        model: PreTrainedModel,
        device: str,
        ewc_lambda: float = 0.4,
    ):
        self.model = model
        self.device = device
        self.ewc_lambda = ewc_lambda
        self.fisher_matrices: Dict[int, Dict[str, torch.Tensor]] = {}
        self.task_means: Dict[int, Dict[str, torch.Tensor]] = {}

    def consolidate(self, task_id: int, eval_loader):
        """Estimate Fisher Information Matrix on eval set."""
        self.model.eval()
        fisher = {}
        task_mean = {}

        # Save current weights
        for name, param in self.model.named_parameters():
            task_mean[name] = param.data.clone()

        # Estimate Fisher on eval set
        fisher_sum = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}

        for batch in eval_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Fix label key naming
            if 'label' in batch:
                batch['labels'] = batch.pop('label')

            self.model.zero_grad()
            outputs = self.model(**batch)
            loss = outputs.loss

            if loss is not None:
                loss.backward()

                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        fisher_sum[name] += param.grad.data ** 2

        # Average Fisher
        num_batches = len(eval_loader)
        for name in fisher_sum:
            fisher[name] = fisher_sum[name] / num_batches

        self.fisher_matrices[task_id] = fisher
        self.task_means[task_id] = task_mean

    def compute_ewc_loss(self) -> torch.Tensor:
        """Compute EWC regularization penalty."""
        if not self.fisher_matrices:
            return torch.tensor(0.0, device=self.device)

        ewc_loss = torch.tensor(0.0, device=self.device)

        for task_id, fisher in self.fisher_matrices.items():
            for name, param in self.model.named_parameters():
                if name in fisher:
                    ewc_loss += (
                        fisher[name] *
                        (param - self.task_means[task_id][name]) ** 2
                    ).sum()

        return self.ewc_lambda * ewc_loss
