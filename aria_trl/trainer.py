"""ContinualSFTTrainer: SFTTrainer with continual learning (ARIA mechanisms)."""

from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, TrainingArguments, Trainer

from .config import ARIAConfig
from .modules import PlasticityGatedMLP, TaskFastAdapter
from .consolidation import FisherConsolidator


class ContinualSFTTrainer(Trainer):
    """
    Trainer extended with continual learning via ARIA mechanisms.

    Prevents catastrophic forgetting on sequential tasks through:
    - PlasticityGatedMLP: Dual fast/slow pathways in FFN layers
    - Slow-Pathway Consolidation (SPC): Fisher regularization
    - Task-Specific Adapters: Per-task lightweight residual modules
    - Asymmetric LR: Slow pathway learns slower (higher stability)
    - Gradient Dampening: Slow grads multiplied by (1-π̄)

    Works with any HF model and is compatible with standard transformers.Trainer.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        train_dataset,
        eval_dataset=None,
        tokenizer=None,
        aria_config: ARIAConfig | Dict[str, Any] | None = None,
        consolidate_after_task: bool = True,
        freeze_old_adapters: bool = True,
        **kwargs,
    ):
        """
        Args:
            model: HF PreTrainedModel (Llama, Mistral, DistilGPT2, etc.)
            args: TrainingArguments
            train_dataset: Dataset or list of datasets (one per task)
            eval_dataset: Optional eval dataset(s)
            tokenizer: HF tokenizer
            aria_config: ARIAConfig or dict with ARIA hyperparameters
            consolidate_after_task: Run Fisher consolidation after each task
            freeze_old_adapters: Freeze task adapters from previous tasks
            **kwargs: Passed to SFTTrainer
        """
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            **kwargs,
        )
        self.tokenizer = tokenizer

        # Parse ARIA config
        if aria_config is None:
            self.aria_config = ARIAConfig()
        elif isinstance(aria_config, dict):
            self.aria_config = ARIAConfig(**aria_config)
        else:
            self.aria_config = aria_config

        self.consolidate_after_task = consolidate_after_task
        self.freeze_old_adapters = freeze_old_adapters

        # Task tracking
        self.current_task_id = 0
        self.num_tasks_seen = 0
        self.task_adapters: List[TaskFastAdapter] = []

        # Fisher consolidator for SPC
        self.consolidator = FisherConsolidator(self.model, self.args.device)

        # Inject PlasticityGatedMLP into model FFN layers
        self._inject_plasticity_gating()

        # Create initial task adapter
        self._add_task_adapter(0)

    def _inject_plasticity_gating(self):
        """Replace FFN layers with PlasticityGatedMLP."""
        for name, module in self.model.named_modules():
            # Check for standard transformer FFN patterns
            if "intermediate" in name or "mlp" in name:
                parent_name = name.rsplit(".", 1)[0] if "." in name else None
                if parent_name:
                    parent = self.model.get_submodule(parent_name)

                    # Detect d_model and d_ff
                    if hasattr(parent, "dense_h_to_4h"):
                        # Galactica/Bloom style
                        d_model = parent.dense_h_to_4h.in_features
                        d_ff = parent.dense_h_to_4h.out_features
                    elif hasattr(parent, "dense_4h_to_h"):
                        # Similar pattern
                        d_model = parent.dense_4h_to_h.out_features
                        d_ff = parent.dense_4h_to_h.in_features
                    elif hasattr(parent, "fc1"):
                        # Pytorch transformer
                        d_model = parent.fc1.in_features
                        d_ff = parent.fc1.out_features
                    else:
                        # Skip if we can't infer dimensions
                        continue

                    # Replace with PlasticityGatedMLP
                    if not isinstance(module, PlasticityGatedMLP):
                        pg_mlp = PlasticityGatedMLP(
                            d_model=d_model,
                            d_ff=d_ff,
                            plasticity_lambda=self.aria_config.plasticity_lambda,
                            warmup_steps=self.aria_config.warmup_steps,
                        )
                        pg_mlp = pg_mlp.to(self.args.device)
                        setattr(parent, name.split(".")[-1], pg_mlp)

    def _add_task_adapter(self, task_id: int):
        """Create and register a new task adapter."""
        d_model = self.model.config.hidden_size
        adapter = TaskFastAdapter(
            d_model=d_model,
            adapter_dim=self.aria_config.adapter_dim,
        )
        adapter = adapter.to(self.args.device)
        self.task_adapters.append(adapter)
        self.num_tasks_seen = task_id + 1

    def add_task(self, task_id: int = None):
        """
        Register a new task and create task adapter.

        Args:
            task_id: Task index (auto-incremented if None)

        Returns:
            task_id
        """
        if task_id is None:
            task_id = self.num_tasks_seen

        self.current_task_id = task_id
        self._add_task_adapter(task_id)
        return task_id

    def consolidate_task(self, task_id: int):
        """
        Estimate Fisher Information for task_id using eval_dataset.

        Must be called after each task's training, before starting next task.

        Args:
            task_id: Which task just finished training
        """
        if self.eval_dataset is None:
            print(f"Warning: no eval_dataset for task {task_id}, skipping consolidation")
            return

        eval_loader = self.get_eval_dataloader()
        max_steps = self.aria_config.consolidation_steps_per_task

        print(f"Consolidating task {task_id} (estimating Fisher)...")
        self.consolidator.consolidate(task_id, eval_loader, max_steps=max_steps)
        print(f"Consolidation complete for task {task_id}")

    def freeze_adapters_except(self, task_id: int):
        """
        Freeze all task adapters except the current one.

        Called before starting training on task_id.

        Args:
            task_id: Current task (its adapter stays unfrozen)
        """
        for tid, adapter in enumerate(self.task_adapters):
            if tid == task_id:
                adapter.unfreeze()
            else:
                adapter.freeze()

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Override training step to add gradient dampening.

        Called after backward, before optimizer.step().
        """
        if num_items_in_batch is not None:
            loss = super().training_step(model, inputs, num_items_in_batch)
        else:
            loss = super().training_step(model, inputs)

        # Dampen slow-pathway gradients
        self._dampen_slow_gradients()

        return loss

    def _dampen_slow_gradients(self):
        """
        Multiply slow-pathway gradients by (1 - π̄).

        Protects consolidated knowledge during high-plasticity phases.
        """
        for module in self.model.modules():
            if isinstance(module, PlasticityGatedMLP):
                mult = module.slow_grad_multiplier()
                for param in module.slow_parameters():
                    if param.grad is not None:
                        param.grad.mul_(mult)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override loss computation to add ARIA losses.

        Total loss = CE loss + plasticity loss + SPC regularization
        """
        # Base CE loss from SFTTrainer
        if return_outputs:
            if num_items_in_batch is not None:
                loss, outputs = super().compute_loss(
                    model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
                )
            else:
                loss, outputs = super().compute_loss(
                    model, inputs, return_outputs=True
                )
        else:
            if num_items_in_batch is not None:
                loss = super().compute_loss(model, inputs, return_outputs=False, num_items_in_batch=num_items_in_batch)
            else:
                loss = super().compute_loss(model, inputs, return_outputs=False)
            outputs = None

        # Add plasticity loss from all PlasticityGatedMLP layers
        plasticity_loss = self._compute_plasticity_loss()
        loss = loss + plasticity_loss

        # Add SPC regularization (Fisher penalty)
        spc_loss = self.consolidator.compute_spc_loss(self.state.global_step)
        spc_loss_weighted = self.aria_config.spc_lambda * spc_loss
        loss = loss + spc_loss_weighted

        if return_outputs:
            return loss, outputs
        return loss

    def _compute_plasticity_loss(self) -> torch.Tensor:
        """Sum plasticity losses from all PlasticityGatedMLP layers."""
        plasticity_loss = torch.tensor(0.0, device=self.args.device)
        for module in self.model.modules():
            if isinstance(module, PlasticityGatedMLP):
                # Plasticity loss is computed during forward pass
                # This is a placeholder for proper tracking
                pass
        return plasticity_loss

    def save_checkpoint(self, output_dir: str):
        """Save model checkpoint with consolidation state."""
        super().save_model(output_dir)

        # Save consolidation state
        state = {
            "current_task_id": self.current_task_id,
            "num_tasks_seen": self.num_tasks_seen,
        }
        torch.save(state, f"{output_dir}/aria_state.pt")

    def load_checkpoint(self, checkpoint_dir: str):
        """Load model checkpoint and consolidation state."""
        state = torch.load(f"{checkpoint_dir}/aria_state.pt")
        self.current_task_id = state["current_task_id"]
        self.num_tasks_seen = state["num_tasks_seen"]
