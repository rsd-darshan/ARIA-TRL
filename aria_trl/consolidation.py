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

    def _protected_params(self):
        """
        Yield (name, param) for every backbone parameter worth protecting.

        This is deliberately *not* slow-pathway-only. Fast/slow separation
        only prevents forgetting if the parts of the model outside the FFN
        (attention, LayerNorm, embeddings) are also protected — leaving them
        fully trainable lets task N+1 shift task N's hidden representations
        even with every slow-pathway weight provably unchanged. The one
        thing excluded is the fast pathway itself: it must stay free to
        adapt each task, or fast/slow degenerates into "slow pathway
        trained at half speed," which is worse than plain EWC at learning
        new tasks.
        """
        fast_param_ids = set()
        for module in self.model.modules():
            if hasattr(module, "fast_in") and hasattr(module, "fast_out"):
                fast_param_ids.update(id(p) for p in module.fast_in.parameters())
                fast_param_ids.update(id(p) for p in module.fast_out.parameters())
        for name, param in self.model.named_parameters():
            if id(param) in fast_param_ids:
                continue
            yield name, param

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
        for name, param in self._protected_params():
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
                # HF datasets commonly use "label" (singular); the model's
                # forward expects "labels" to compute loss internally. Without
                # this rename, loss silently comes back None and Fisher stays
                # all-zero for the entire consolidation call.
                if "label" in batch and "labels" not in batch:
                    batch["labels"] = batch.pop("label")
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
            if hasattr(outputs, "loss") and outputs.loss is not None:
                loss = outputs.loss
            else:
                # Fallback if loss not in output
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                target = batch["labels"] if isinstance(batch, dict) else batch[1]
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target.view(-1),
                )

            loss.backward()

            # Accumulate squared gradients (diagonal Fisher)
            for name, param in self._protected_params():
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
            for name, param in self._protected_params():
                if name in means and name in fishers and param.requires_grad:
                    mean = means[name].to(self.device)
                    fisher = fishers[name].to(self.device)

                    # Fisher-weighted squared deviation
                    spc_loss = spc_loss + (fisher * (param - mean) ** 2).sum()

        return spc_loss

    def reset(self):
        """Clear all stored Fisher information and means (start fresh)."""
        self.task_means = []
        self.task_fishers = []
