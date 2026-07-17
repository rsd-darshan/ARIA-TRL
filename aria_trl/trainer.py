"""ContinualSFTTrainer: SFTTrainer with continual learning (ARIA mechanisms)."""

from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, TrainingArguments, Trainer

from .config import ARIAConfig
from .modules import PlasticityGatedMLP, TaskFastAdapter, MultiHeadScoreWrapper
from .consolidation import FisherConsolidator
from .utils import setup_asymmetric_lr_groups


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
        head_lr_mult: float = 5.0,
        consolidator: FisherConsolidator | None = None,
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
            head_lr_mult: LR multiplier for the active task's fresh head and
                adapter. Both are cold-started with only one task's worth of
                steps to learn from while the backbone has already had
                previous tasks' worth of training — a higher LR compensates
                for that head start gap.
            consolidator: Pass the previous task's FisherConsolidator here
                when recreating the trainer for a new task (recommended: a
                fresh Trainer per task avoids an Accelerate state-reset
                issue on reuse). Fisher state must persist across that
                recreation or every task after the first consolidates
                nothing. If None, a fresh consolidator is created — correct
                only for the very first task.
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

        # Auto-compute warmup_steps (one epoch's worth) if not explicitly set.
        # A fixed guess is either never reached (small-scale runs) or fires
        # too late (large-scale runs) — deriving it from the actual dataset
        # size keeps gate specialization reliable across scales.
        if self.aria_config.warmup_steps is None:
            try:
                steps_per_epoch = -(-len(train_dataset) // self.args.per_device_train_batch_size)
            except TypeError:
                steps_per_epoch = 200  # dataset has no len() (e.g. IterableDataset); reasonable fallback
            self.aria_config.warmup_steps = max(1, steps_per_epoch)

        self.consolidate_after_task = consolidate_after_task
        self.freeze_old_adapters = freeze_old_adapters
        self.head_lr_mult = head_lr_mult

        # Task tracking. active_task_id lives in a dict on the model (not a
        # plain trainer attribute) so it survives trainer recreation across
        # tasks and stays in sync between the adapter hook and the score-head
        # wrapper, which are set up independently below.
        self.current_task_id = 0
        self.num_tasks_seen = 0
        self.task_adapters: List[TaskFastAdapter] = []
        if not hasattr(self.model, "_aria_state"):
            self.model._aria_state = {"active_task_id": 0}

        # Fisher consolidator for SPC
        self.consolidator = consolidator if consolidator is not None else \
                            FisherConsolidator(self.model, self.args.device)

        # Inject PlasticityGatedMLP into model FFN layers
        self._inject_plasticity_gating()

        # Replace the model's single classification head with a per-task
        # router. Must happen before _register_adapter_hook(), which hooks
        # onto self.model.score/classifier — by then that name resolves to
        # this wrapper, so the adapter transform runs before head routing.
        self._install_score_wrapper()

        # Wire adapters into the forward pass via the classification head.
        # Task adapters themselves are created by add_task(), called once per task.
        self._register_adapter_hook()

    @property
    def active_task_id(self) -> int:
        return self.model._aria_state["active_task_id"]

    def _extract_mlp(self, mlp_module):
        """
        Infer (d_model, d_ff), extract the pretrained in/out weights and
        bias, and identify the original activation function from a
        transformer block's FFN submodule.

        Returns (d_model, d_ff, w_in, b_in, w_out, b_out, activation); the
        weight/bias/activation entries are None when the style can't be
        matched (dims-only fallback, e.g. Bloom), so the caller falls back
        to random init rather than crashing.
        """
        w_in = b_in = w_out = b_out = activation = None
        if hasattr(mlp_module, "c_fc"):
            # GPT2 style: Conv1D weight shape is (in_features, out_features) —
            # the transpose of nn.Linear's convention, hence the .T below.
            d_model, d_ff = mlp_module.c_fc.weight.shape[0], mlp_module.c_fc.weight.shape[1]
            w_in = mlp_module.c_fc.weight.T.detach().clone()
            b_in = mlp_module.c_fc.bias.detach().clone()
            w_out = mlp_module.c_proj.weight.T.detach().clone()
            b_out = mlp_module.c_proj.bias.detach().clone()
            activation = getattr(mlp_module, "act", None)
        elif hasattr(mlp_module, "fc1"):
            # Generic PyTorch transformer (nn.Linear already in the right orientation)
            d_model, d_ff = mlp_module.fc1.in_features, mlp_module.fc1.out_features
            w_in = mlp_module.fc1.weight.detach().clone()
            b_in = mlp_module.fc1.bias.detach().clone()
            w_out = mlp_module.fc2.weight.detach().clone()
            b_out = mlp_module.fc2.bias.detach().clone()
            activation = getattr(mlp_module, "act", None)
        elif hasattr(mlp_module, "dense_h_to_4h"):
            # Bloom style — dims only; weight-copy convention not implemented here
            d_model = mlp_module.dense_h_to_4h.in_features
            d_ff = mlp_module.dense_h_to_4h.out_features
        else:
            return None, None, None, None, None, None, None
        return d_model, d_ff, w_in, b_in, w_out, b_out, activation

    def _inject_plasticity_gating(self):
        """
        Replace each transformer block's FFN submodule with
        PlasticityGatedMLP, copying the pretrained weights into both the
        fast and slow pathways so the model starts identical to the
        pretrained checkpoint. Without this, both pathways start from
        random init, discarding the entire pretrained FFN — the backbone
        would be fine-tuning from scratch, not from distilgpt2 (or whatever
        base model), which is catastrophic at small-scale/few-epoch runs.
        """
        for module in list(self.model.modules()):
            original_mlp = getattr(module, "mlp", None)
            if original_mlp is None or isinstance(original_mlp, PlasticityGatedMLP):
                continue

            d_model, d_ff, w_in, b_in, w_out, b_out, activation = self._extract_mlp(original_mlp)
            if d_model is None:
                continue

            pg_mlp = PlasticityGatedMLP(
                d_model=d_model,
                d_ff=d_ff,
                plasticity_lambda=self.aria_config.plasticity_lambda,
                warmup_steps=self.aria_config.warmup_steps,
                activation=activation,
            )
            if w_in is not None:
                with torch.no_grad():
                    pg_mlp.fast_in.weight.copy_(w_in);   pg_mlp.fast_in.bias.copy_(b_in)
                    pg_mlp.fast_out.weight.copy_(w_out); pg_mlp.fast_out.bias.copy_(b_out)
                    pg_mlp.slow_in.weight.copy_(w_in);   pg_mlp.slow_in.bias.copy_(b_in)
                    pg_mlp.slow_out.weight.copy_(w_out); pg_mlp.slow_out.bias.copy_(b_out)
            pg_mlp = pg_mlp.to(self.args.device)
            module.mlp = pg_mlp

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

    def _register_adapter_hook(self):
        """
        Apply the active task's adapter to the hidden states fed into the
        classification head, via a forward pre-hook. The active task is
        selected with set_active_task() / add_task(); without this hook the
        per-task adapters would be created and frozen but never participate
        in the forward pass at all.
        """
        head_name = getattr(self.model, "_aria_head_name", None) or (
            "score" if hasattr(self.model, "score") else "classifier"
        )
        head = getattr(self.model, head_name, None)
        if head is None:
            print("Warning: could not locate classification head; task adapters will not be applied.")
            return

        if getattr(self.model, "_aria_adapter_hook_registered", False):
            return

        def adapter_pre_hook(module, args):
            if not self.task_adapters or self.active_task_id >= len(self.task_adapters):
                return args
            hidden_states = self.task_adapters[self.active_task_id](args[0])
            return (hidden_states,) + args[1:]

        head.register_forward_pre_hook(adapter_pre_hook)
        self.model._aria_adapter_hook_registered = True

    def _install_score_wrapper(self):
        """Replace the model's single classification head with a per-task
        router (once per model instance, even if the trainer is recreated
        for each task). A shared head is a second forgetting vector
        independent of the backbone: training task N+1 overwrites task N's
        decision boundary even if the backbone is perfectly preserved."""
        if getattr(self.model, "_aria_head_installed", False):
            return
        head_name = "score" if hasattr(self.model, "score") else "classifier"
        wrapper = MultiHeadScoreWrapper(self.model._aria_state)
        setattr(self.model, head_name, wrapper)
        self.model._aria_head_name = head_name
        self.model._aria_head_installed = True

    def _new_score_head(self) -> nn.Linear:
        """Create the active task's classification head, warm-started from
        the previous task's head weights rather than randomly initialized.
        Reasonable when tasks share label semantics (e.g. all binary
        sentiment-adjacent) and task identity is known at train time; for
        tasks with unrelated label spaces, override this method."""
        d_model = self.model.config.hidden_size
        n_labels = self.model.config.num_labels
        head = nn.Linear(d_model, n_labels, bias=False).to(self.args.device)
        wrapper = getattr(self.model, self.model._aria_head_name)
        if len(wrapper.heads) > 0:
            with torch.no_grad():
                head.weight.copy_(wrapper.heads[-1].weight)
        else:
            nn.init.normal_(head.weight, std=0.02)
        return head

    def set_active_task(self, task_id: int):
        """Select which task's adapter and classification head are used
        during the next forward pass(es)."""
        self.model._aria_state["active_task_id"] = task_id

    def add_task(self, task_id: int = None):
        """
        Register a new task: create its adapter and classification head,
        and freeze every previous task's head so training this task can't
        overwrite them.

        Args:
            task_id: Task index (auto-incremented if None)

        Returns:
            task_id
        """
        if task_id is None:
            task_id = self.num_tasks_seen

        self.current_task_id = task_id
        self.set_active_task(task_id)
        self._add_task_adapter(task_id)

        wrapper = getattr(self.model, self.model._aria_head_name)
        for head in wrapper.heads:
            for param in head.parameters():
                param.requires_grad_(False)
        wrapper.add_head(self._new_score_head())

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

    def create_optimizer(self):
        """
        Build the optimizer with three parameter groups:
        - slow-pathway parameters at slow_lr_ratio * base_lr
        - the active task's head and adapter at head_lr_mult * base_lr
          (both are cold-started each task with only that task's step
          budget, while the backbone has already had previous tasks'
          worth of training — a higher LR compensates for that gap)
        - everything else at base_lr

        Without this override, transformers.Trainer's default optimizer
        applies a single uniform LR to all parameters.
        """
        if self.optimizer is None:
            active_head_params = []
            if hasattr(self.model, "_aria_head_name"):
                wrapper = getattr(self.model, self.model._aria_head_name)
                if self.active_task_id < len(wrapper.heads):
                    active_head_params = [p for p in wrapper.heads[self.active_task_id].parameters()
                                          if p.requires_grad]
            exclude_ids = {id(p) for p in active_head_params}

            param_groups = setup_asymmetric_lr_groups(
                self.model,
                base_lr=self.args.learning_rate,
                slow_lr_ratio=self.aria_config.slow_lr_ratio,
                exclude_ids=exclude_ids,
            )
            param_groups = [g for g in param_groups if g["params"]]

            high_lr_params = list(active_head_params)
            if self.task_adapters and self.active_task_id < len(self.task_adapters):
                high_lr_params += [p for p in self.task_adapters[self.active_task_id].parameters()
                                    if p.requires_grad]
            if high_lr_params:
                param_groups.append({
                    "params": high_lr_params,
                    "lr": self.args.learning_rate * self.head_lr_mult,
                    "name": "active_head_and_adapter",
                })

            if not param_groups:
                # Fallback: everything frozen somehow; avoid an empty-optimizer error
                param_groups = [{"params": [p for p in self.model.parameters() if p.requires_grad][:1],
                                  "lr": self.args.learning_rate}]

            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, self.model)
            optimizer_kwargs.pop("lr", None)
            self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
        return self.optimizer

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
                plasticity_loss = plasticity_loss + module.last_plasticity_loss.to(self.args.device)
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
