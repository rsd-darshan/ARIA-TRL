"""ARIA modules: PlasticityGatedMLP and TaskFastAdapter."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PlasticityGatedMLP(nn.Module):
    """
    Dual fast/slow pathway MLP for continual learning.

    Routes computation per token via learned gate π ∈ (0,1):
        output = π * fast(x) + (1-π) * slow(x)

    Fast pathway: volatile, task-specific learning
    Slow pathway: stable, task-generic consolidated knowledge

    Bimodal specialization loss pushes π toward extremes (0 or 1),
    ensuring clear fast/slow separation.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        plasticity_lambda: float = 0.01,
        warmup_steps: int = 500,
        dropout: float = 0.1,
        activation=None,
    ):
        """
        Args:
            d_model: Hidden dimension (typically 768 for DistilBERT-like)
            d_ff: Feed-forward dimension (typically 4*d_model)
            plasticity_lambda: Weight for specialization loss
            warmup_steps: Steps before plasticity loss activates
            dropout: Dropout rate
            activation: Activation function to use in both pathways. Pass
                the original FFN's activation (e.g. GPT-2's gelu_new) when
                replacing a pretrained block — defaulting to plain F.gelu
                here would silently mismatch what the pretrained weights
                being copied in were actually trained with.
        """
        super().__init__()

        # Fast pathway: volatile, learns new task-specific patterns
        self.fast_in = nn.Linear(d_model, d_ff)
        self.fast_out = nn.Linear(d_ff, d_model)

        # Slow pathway: stable, consolidates general knowledge
        self.slow_in = nn.Linear(d_model, d_ff)
        self.slow_out = nn.Linear(d_ff, d_model)

        # Learned gate: per-token routing
        self.gate_net = nn.Sequential(
            nn.Linear(d_model, d_ff // 4),
            nn.ReLU(),
            nn.Linear(d_ff // 4, 1),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout)
        self.plasticity_lambda = plasticity_lambda
        self.warmup_steps = warmup_steps
        self.mean_gate = 0.5
        self.step = 0
        self.last_plasticity_loss = torch.tensor(0.0)
        self.act = activation if activation is not None else F.gelu

    def forward(self, x: torch.Tensor, force_slow: bool = False) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)
            force_slow: If True, route entirely through slow pathway (for old tasks at eval)

        Returns:
            output: (batch, seq_len, d_model)

        The bimodal specialization loss for this forward pass is stored in
        self.last_plasticity_loss, since drop-in MLP replacement requires a
        single-tensor return matching the original module's call site.
        """
        if force_slow:
            # Old task at eval: use only consolidated slow pathway
            π = torch.zeros(*x.shape[:-1], 1, device=x.device, dtype=x.dtype)
        else:
            π = self.gate_net(x)  # (batch, seq_len, 1)

        self.mean_gate = float(π.detach().mean().item())

        # Dual pathways
        h_fast = self.act(self.fast_in(x))
        h_slow = self.act(self.slow_in(x))

        # Route by gate
        out = π * self.fast_out(h_fast) + (1 - π) * self.slow_out(h_slow)
        out = self.dropout(out)

        # Plasticity loss: bimodal specialization (push π toward 0 or 1)
        if self.step >= self.warmup_steps:
            self.last_plasticity_loss = self.plasticity_lambda / (π * (1 - π) + 1e-4).mean()
        else:
            self.last_plasticity_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        self.step += 1
        return out

    def slow_parameters(self):
        """Return all slow-pathway parameters."""
        params = list(self.slow_in.parameters()) + list(self.slow_out.parameters())
        return params

    def slow_grad_multiplier(self) -> float:
        """Gradient dampening multiplier for slow pathway: (1 - π̄)."""
        return 1.0 - self.mean_gate

    def reset_step(self):
        """Reset step counter (call at start of each epoch)."""
        self.step = 0


class TaskFastAdapter(nn.Module):
    """
    Task-specific residual adapter (LoRA-like).

    Lightweight bottleneck residual module:
        h_new = h + adapter(h)

    Adapter is frozen after task training to prevent overwriting
    previous task-specific representations.

    Bottleneck design (compress → relu → expand):
        h (d_model) → compress (adapter_dim) → relu → expand (d_model)
        Zero-initialized expansion means adapter starts as identity.
    """

    def __init__(self, d_model: int, adapter_dim: int = 64):
        """
        Args:
            d_model: Hidden dimension
            adapter_dim: Bottleneck dimension (compression factor)
        """
        super().__init__()
        self.d_model = d_model
        self.adapter_dim = adapter_dim

        # Compress to bottleneck
        self.down = nn.Linear(d_model, adapter_dim)

        # Expand back
        self.up = nn.Linear(adapter_dim, d_model)

        # Zero-init expansion so adapter is identity at init
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) or (batch, d_model)

        Returns:
            x + adapter(x) — residual connection
        """
        return x + self.up(F.relu(self.down(x)))

    def freeze(self):
        """Freeze adapter parameters (call after task training)."""
        for param in self.parameters():
            param.requires_grad_(False)

    def unfreeze(self):
        """Unfreeze adapter parameters."""
        for param in self.parameters():
            param.requires_grad_(True)


class MultiHeadScoreWrapper(nn.Module):
    """
    Routes the forward pass to the active task's classification head.

    A single shared head is a second forgetting vector independent of the
    backbone: training task 2 overwrites task 1's decision boundary even if
    the backbone itself is perfectly preserved. This wrapper gives each task
    its own head, selected via a shared mutable state dict (so it stays in
    sync with whatever else reads/writes active_task_id, e.g. the adapter
    hook) rather than a plain instance attribute.
    """

    def __init__(self, state: dict):
        super().__init__()
        self.heads = nn.ModuleList()
        self.state = state  # shared with the owning trainer, e.g. {"active_task_id": 0}

    def add_head(self, head: nn.Module):
        self.heads.append(head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        task_id = min(self.state["active_task_id"], len(self.heads) - 1)
        return self.heads[task_id](x)
