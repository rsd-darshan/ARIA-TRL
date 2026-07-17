"""Utility functions for aria-trl."""

from typing import Dict, List
import torch
import numpy as np


def compute_continual_metrics(
    task_accuracies: Dict[int, Dict[int, float]],
) -> Dict[str, float]:
    """
    Compute continual learning metrics from task accuracy matrices.

    Args:
        task_accuracies: Dict mapping task_id -> {eval_task_id -> accuracy}
        Example: {0: {0: 0.95}, 1: {0: 0.92, 1: 0.90}, ...}

    Returns:
        Dict with metrics:
        - avg_accuracy: Average accuracy on all tasks at end
        - forgetting: Average backward transfer (how much old tasks forget)
        - forward_transfer: Average forward transfer (learning from past helps new tasks)
    """
    num_tasks = len(task_accuracies)

    # Flatten into matrix: acc[t, j] = accuracy on task j after training task t
    max_task_id = max(task_accuracies.keys())
    acc_matrix = np.zeros((max_task_id + 1, max_task_id + 1))

    for task_id, accuracies in task_accuracies.items():
        for eval_id, acc in accuracies.items():
            acc_matrix[task_id, eval_id] = acc

    T = num_tasks

    # Average accuracy (diagonal and upper triangle)
    avg_acc = np.mean([acc_matrix[t, j] for t in range(T) for j in range(t + 1)])

    # Forgetting: BWT = 1/(T-1) * sum of (a_{T,j} - a_{j,j}) for j < T
    forgetting = 0.0
    if T > 1:
        for j in range(T - 1):
            forgetting += max(0, acc_matrix[j, j] - acc_matrix[T - 1, j])
        forgetting /= (T - 1)

    # Forward transfer: FWT = 1/(T-1) * sum of (a_{t,t+1} - baseline)
    # Baseline is random chance (0.5 for binary)
    forward_transfer = 0.0
    if T > 1:
        for t in range(T - 1):
            # Accuracy on task t+1 before training on it
            # We use 0.5 as random baseline for binary classification
            forward_transfer += acc_matrix[t, t + 1] - 0.5
        forward_transfer /= (T - 1)

    return {
        "avg_accuracy": float(avg_acc),
        "forgetting": float(forgetting),
        "forward_transfer": float(forward_transfer),
    }


def get_slow_parameters(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    """
    Collect all slow-pathway parameters from a model.

    Returns all parameters from PlasticityGatedMLP slow pathways.
    """
    from .modules import PlasticityGatedMLP

    params = []
    for module in model.modules():
        if isinstance(module, PlasticityGatedMLP):
            params.extend(module.slow_parameters())
    return params


def get_fast_parameters(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    """
    Collect all fast-pathway parameters from a model.

    Returns all parameters from PlasticityGatedMLP fast pathways.
    """
    from .modules import PlasticityGatedMLP

    params = []
    for module in model.modules():
        if isinstance(module, PlasticityGatedMLP):
            fast_params = [
                p for n, p in module.named_parameters()
                if "slow" not in n and "gate" not in n
            ]
            params.extend(fast_params)
    return params


def setup_asymmetric_lr_groups(
    model: torch.nn.Module,
    base_lr: float,
    slow_lr_ratio: float = 1.0,
    exclude_ids: set | None = None,
) -> List[Dict]:
    """
    Create optimizer param groups with asymmetric learning rates.

    Args:
        model: The model
        base_lr: Base learning rate for fast pathway
        slow_lr_ratio: Multiplier for slow pathway LR
        exclude_ids: Parameter ids to leave out of both groups entirely
            (the caller assigns them to their own group instead — e.g. an
            active task head/adapter that should train at a different LR)

    Returns:
        List of param groups ready for optimizer
    """
    exclude_ids = exclude_ids or set()
    slow_params = [p for p in get_slow_parameters(model) if id(p) not in exclude_ids]
    slow_param_ids = {id(p) for p in slow_params}

    param_groups = [
        {
            "params": [p for p in model.parameters()
                      if p.requires_grad and id(p) not in slow_param_ids and id(p) not in exclude_ids],
            "lr": base_lr,
            "name": "fast_pathway",
        },
        {
            "params": slow_params,
            "lr": base_lr * slow_lr_ratio,
            "name": "slow_pathway",
        },
    ]

    return param_groups
