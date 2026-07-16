"""
ARIA-TRL vs Standard Fine-Tuning: Sequential Domain Sentiment Benchmark
Tasks: Movies (SST-2) -> Restaurants (Yelp) -> Social Media (Emotion)
Metrics: Average Accuracy (ACC) + Backward Transfer (BWT), 3 seeds
"""

import os, random, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
warnings.filterwarnings("ignore")

# Maximize CPU thread usage
torch.set_num_threads(min(8, os.cpu_count() or 4))
torch.set_num_interop_threads(2)

# ─────────────────────────────────────────────
# aria-trl: inlined (all fixes applied)
# ─────────────────────────────────────────────

class PlasticityGatedMLP(nn.Module):
    def __init__(self, d_model, d_ff, plasticity_lambda=0.001, warmup_steps=200, dropout=0.1, activation=None):
        super().__init__()
        self.fast_in  = nn.Linear(d_model, d_ff)
        self.fast_out = nn.Linear(d_ff, d_model)
        self.slow_in  = nn.Linear(d_model, d_ff)
        self.slow_out = nn.Linear(d_ff, d_model)
        self.gate_net = nn.Sequential(
            nn.Linear(d_model, d_ff // 4), nn.ReLU(),
            nn.Linear(d_ff // 4, 1), nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)
        self.plasticity_lambda = plasticity_lambda
        self.warmup_steps = warmup_steps
        self.mean_gate = 0.5
        self.step = 0
        self.last_plasticity_loss = torch.tensor(0.0)
        # Use same activation as the original FFN (critical for pretrained representations)
        self.act = activation if activation is not None else F.gelu

    def forward(self, x):
        pi = self.gate_net(x)
        self.mean_gate = float(pi.detach().mean().item())
        h_fast = self.act(self.fast_in(x))
        h_slow = self.act(self.slow_in(x))
        out = pi * self.fast_out(h_fast) + (1 - pi) * self.slow_out(h_slow)
        out = self.dropout(out)
        if self.step >= self.warmup_steps:
            self.last_plasticity_loss = self.plasticity_lambda / (pi * (1 - pi) + 1e-4).mean()
        else:
            self.last_plasticity_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        self.step += 1
        return out

    def slow_parameters(self):
        return list(self.slow_in.parameters()) + list(self.slow_out.parameters())

    def slow_grad_multiplier(self):
        return 1.0 - self.mean_gate


class TaskFastAdapter(nn.Module):
    def __init__(self, d_model, adapter_dim=64):
        super().__init__()
        self.down = nn.Linear(d_model, adapter_dim)
        self.up   = nn.Linear(adapter_dim, d_model)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(F.relu(self.down(x)))

    def freeze(self):
        for p in self.parameters(): p.requires_grad_(False)

    def unfreeze(self):
        for p in self.parameters(): p.requires_grad_(True)


class MultiHeadScoreWrapper(nn.Module):
    """Routes forward to the active task's score head; prevents head forgetting across tasks."""
    def __init__(self, state):
        super().__init__()
        self.heads = nn.ModuleList()
        self.state = state  # shared reference to model._aria_state

    def add_head(self, head):
        self.heads.append(head)

    def forward(self, x):
        tid = min(self.state["active_task_id"], len(self.heads) - 1)
        return self.heads[tid](x)


class FisherConsolidator:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.task_means   = []
        self.task_fishers = []

    def _protected_params(self):
        """Yield (name, param) for backbone parameters worth protecting with EWC.

        Excludes the fast pathway of any PlasticityGatedMLP: fast/slow only
        works if the fast lane stays genuinely free to adapt each task. Only
        the slow lane, attention, LayerNorm, and embeddings are protected.
        """
        fast_ids = set()
        for m in self.model.modules():
            if isinstance(m, PlasticityGatedMLP):
                fast_ids.update(id(p) for p in m.fast_in.parameters())
                fast_ids.update(id(p) for p in m.fast_out.parameters())
        for name, param in self.model.named_parameters():
            if id(param) in fast_ids:
                continue
            yield name, param

    def consolidate(self, task_id, eval_loader, max_steps=None):
        self.model.eval()
        means   = {n: p.detach().cpu().clone() for n, p in self._protected_params()}
        fishers = {n: torch.zeros_like(p, device="cpu") for n, p in self._protected_params()}
        n_batches = 0
        for i, batch in enumerate(eval_loader):
            if max_steps and i >= max_steps:
                break
            batch = {k: v.to(self.device) for k, v in batch.items()}
            # Rename 'label' → 'labels' so GPT2 computes loss internally
            if "label" in batch and "labels" not in batch:
                batch["labels"] = batch.pop("label")
            self.model.zero_grad()
            out = self.model(**batch)
            loss = out.loss
            if loss is not None:
                loss.backward()
                for name, param in self._protected_params():
                    if param.grad is not None:
                        fishers[name] += param.grad.data.cpu() ** 2
            n_batches += 1
        if n_batches:
            for k in fishers: fishers[k] /= n_batches
        self.task_means.append(means)
        self.task_fishers.append(fishers)

    def compute_spc_loss(self):
        if not self.task_fishers:
            return torch.tensor(0.0, device=self.device)
        loss = torch.tensor(0.0, device=self.device)
        for means, fishers in zip(self.task_means, self.task_fishers):
            for name, param in self._protected_params():
                if name in means and param.requires_grad:
                    mean   = means[name].to(self.device)
                    fisher = fishers[name].to(self.device)
                    loss = loss + (fisher * (param - mean) ** 2).sum()
        return loss


from transformers import Trainer, TrainingArguments, PreTrainedModel
from typing import List, Dict, Any, Optional


def _get_slow_params(model):
    params = []
    for m in model.modules():
        if isinstance(m, PlasticityGatedMLP):
            params.extend(m.slow_parameters())
    return params


def _setup_lr_groups(model, base_lr, slow_ratio=0.5, exclude_ids=None):
    exclude_ids = exclude_ids or set()
    slow_ids = {id(p) for p in _get_slow_params(model)} - exclude_ids
    return [
        {"params": [p for p in model.parameters()
                    if p.requires_grad and id(p) not in slow_ids and id(p) not in exclude_ids], "lr": base_lr},
        {"params": [p for p in model.parameters() if p.requires_grad and id(p) in slow_ids],    "lr": base_lr * slow_ratio},
    ]


class ContinualSFTTrainer(Trainer):
    def __init__(self, model, args, train_dataset, eval_dataset=None,
                 tokenizer=None, plasticity_lambda=0.001, spc_lambda=1.0,
                 adapter_dim=64, slow_lr_ratio=0.5, warmup_steps=200,
                 consolidator=None, **kwargs):
        super().__init__(model=model, args=args, train_dataset=train_dataset,
                         eval_dataset=eval_dataset, **kwargs)
        self.tokenizer         = tokenizer
        self.plasticity_lambda = plasticity_lambda
        self.spc_lambda        = spc_lambda
        self.adapter_dim       = adapter_dim
        self.slow_lr_ratio     = slow_lr_ratio
        self.warmup_steps      = warmup_steps
        self.num_tasks_seen    = 0
        # Consolidator is passed in so Fisher state survives trainer recreation per task
        self.consolidator = consolidator if consolidator is not None else \
                            FisherConsolidator(self.model, self.args.device)
        # Adapter + head state lives on the model so it survives trainer recreation
        if not hasattr(self.model, '_aria_state'):
            self.model._aria_state = {"task_adapters": [], "active_task_id": 0}
            self._install_score_wrapper()
        self._inject_gating()
        self._register_adapter_hook()

    def _inject_gating(self):
        for module in list(self.model.modules()):
            orig = getattr(module, "mlp", None)
            if orig is None or isinstance(orig, PlasticityGatedMLP):
                continue
            d_model = d_ff = None
            w_in = b_in = w_out = b_out = activation = None
            if hasattr(orig, "c_fc"):        # GPT2 Conv1D: weight shape (in, out)
                d_model, d_ff = orig.c_fc.weight.shape[0], orig.c_fc.weight.shape[1]
                w_in  = orig.c_fc.weight.T.detach().clone()   # (d_ff, d_model)
                b_in  = orig.c_fc.bias.detach().clone()
                w_out = orig.c_proj.weight.T.detach().clone() # (d_model, d_ff)
                b_out = orig.c_proj.bias.detach().clone()
                # Preserve the original activation (GPT2 uses gelu_new, not F.gelu)
                activation = getattr(orig, "act", None)
            elif hasattr(orig, "fc1"):       # generic transformer nn.Linear
                d_model, d_ff = orig.fc1.in_features, orig.fc1.out_features
                w_in  = orig.fc1.weight.detach().clone()
                b_in  = orig.fc1.bias.detach().clone()
                w_out = orig.fc2.weight.detach().clone()
                b_out = orig.fc2.bias.detach().clone()
                activation = getattr(orig, "act", None)
            elif hasattr(orig, "dense_h_to_4h"):  # Bloom
                d_model, d_ff = orig.dense_h_to_4h.in_features, orig.dense_h_to_4h.out_features
            if d_model is None:
                continue
            pg = PlasticityGatedMLP(d_model, d_ff, self.plasticity_lambda, self.warmup_steps,
                                    activation=activation)
            # Copy pretrained FFN weights into both pathways so model starts at pretrained state
            if w_in is not None:
                with torch.no_grad():
                    pg.fast_in.weight.copy_(w_in);   pg.fast_in.bias.copy_(b_in)
                    pg.fast_out.weight.copy_(w_out);  pg.fast_out.bias.copy_(b_out)
                    pg.slow_in.weight.copy_(w_in);   pg.slow_in.bias.copy_(b_in)
                    pg.slow_out.weight.copy_(w_out);  pg.slow_out.bias.copy_(b_out)
            module.mlp = pg.to(self.args.device)

    @property
    def task_adapters(self):
        return self.model._aria_state["task_adapters"]

    @property
    def active_task_id(self):
        return self.model._aria_state["active_task_id"]

    @active_task_id.setter
    def active_task_id(self, v):
        self.model._aria_state["active_task_id"] = v

    def _install_score_wrapper(self):
        """Replace model.score with a MultiHeadScoreWrapper (done once per model)."""
        head_name = "score" if hasattr(self.model, "score") else "classifier"
        wrapper = MultiHeadScoreWrapper(self.model._aria_state)
        setattr(self.model, head_name, wrapper)
        self.model._aria_head_name = head_name

    def _new_score_head(self):
        d_model = self.model.config.hidden_size
        n_labels = self.model.config.num_labels
        h = nn.Linear(d_model, n_labels, bias=False).to(self.args.device)
        # Warm-start from the previous task's head instead of random init —
        # all tasks here are binary sentiment-adjacent, so transfer is real
        # and head identity is already known at train time.
        head_name = getattr(self.model, '_aria_head_name', 'score')
        wrapper = getattr(self.model, head_name, None)
        if wrapper is not None and len(wrapper.heads) > 0:
            with torch.no_grad():
                h.weight.copy_(wrapper.heads[-1].weight)
        else:
            nn.init.normal_(h.weight, std=0.02)
        return h

    def _register_adapter_hook(self):
        # Register once per model instance; reads state from model._aria_state
        if getattr(self.model, '_aria_hook_registered', False):
            return
        head = getattr(self.model, "score", None) or getattr(self.model, "classifier", None)
        if head is None:
            return
        state = self.model._aria_state
        def hook(module, args):
            adapters = state["task_adapters"]
            tid = state["active_task_id"]
            if not adapters or tid >= len(adapters):
                return args
            new_h = adapters[tid](args[0])
            return (new_h,) + args[1:]
        head.register_forward_pre_hook(hook)
        self.model._aria_hook_registered = True

    def add_task(self, task_id):
        self.model._aria_state["active_task_id"] = task_id
        d_model = self.model.config.hidden_size
        adapter = TaskFastAdapter(d_model, self.adapter_dim).to(self.args.device)
        self.model._aria_state["task_adapters"].append(adapter)
        # Freeze all existing score heads before adding the new one
        head_name = getattr(self.model, '_aria_head_name', 'score')
        wrapper = getattr(self.model, head_name)
        for h in wrapper.heads:
            for p in h.parameters(): p.requires_grad_(False)
        # Fresh score head for this task
        wrapper.add_head(self._new_score_head())
        self.num_tasks_seen = task_id + 1
        return task_id

    def set_active_task(self, task_id):
        # wrapper reads active_task_id from _aria_state directly; no extra work needed
        self.model._aria_state["active_task_id"] = task_id

    def freeze_adapters_except(self, task_id):
        for i, a in enumerate(self.model._aria_state["task_adapters"]):
            a.freeze() if i != task_id else a.unfreeze()

    def consolidate_task(self, task_id):
        if self.eval_dataset is None:
            return
        print(f"  Consolidating task {task_id}...")
        self.consolidator.consolidate(task_id, self.get_eval_dataloader())
        print(f"  Done.")

    def create_optimizer(self):
        if self.optimizer is None:
            head_lr_mult = 5.0
            # Active head gets its own high-LR group; it's cold-started each task
            # and only has ~one epoch worth of steps to learn from.
            active_head_params = []
            head_name = getattr(self.model, '_aria_head_name', 'score')
            wrapper = getattr(self.model, head_name, None)
            if wrapper is not None and self.active_task_id < len(wrapper.heads):
                active_head_params = [p for p in wrapper.heads[self.active_task_id].parameters()
                                      if p.requires_grad]
            exclude_ids = {id(p) for p in active_head_params}

            groups = [g for g in _setup_lr_groups(self.model, self.args.learning_rate,
                                                   self.slow_lr_ratio, exclude_ids=exclude_ids)
                      if g["params"]]  # drop empty groups (backbone may be frozen)

            if active_head_params:
                groups.append({"params": active_head_params, "lr": self.args.learning_rate * head_lr_mult})

            # Include active task adapter parameters so the adapter actually trains
            if self.task_adapters and self.active_task_id < len(self.task_adapters):
                adapter_params = [p for p in self.task_adapters[self.active_task_id].parameters()
                                  if p.requires_grad]
                if adapter_params:
                    groups.append({"params": adapter_params, "lr": self.args.learning_rate * head_lr_mult})
            if not groups:
                # Fallback: if somehow all params are frozen, add a dummy to avoid optimizer error
                groups = [{"params": [p for p in self.model.parameters() if p.requires_grad][:1],
                           "lr": self.args.learning_rate}]
            opt_cls, opt_kwargs = self.get_optimizer_cls_and_kwargs(self.args, self.model)
            opt_kwargs.pop("lr", None)
            self.optimizer = opt_cls(groups, **opt_kwargs)
        return self.optimizer

    def training_step(self, model, inputs, num_items_in_batch=None):
        if num_items_in_batch is not None:
            loss = super().training_step(model, inputs, num_items_in_batch)
        else:
            loss = super().training_step(model, inputs)
        for m in self.model.modules():
            if isinstance(m, PlasticityGatedMLP):
                mult = m.slow_grad_multiplier()
                for p in m.slow_parameters():
                    if p.grad is not None: p.grad.mul_(mult)
        return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        kw = {"return_outputs": return_outputs}
        if num_items_in_batch is not None:
            kw["num_items_in_batch"] = num_items_in_batch
        result = super().compute_loss(model, inputs, **kw)
        loss, outputs = (result if return_outputs else (result, None))

        p_loss = sum(
            m.last_plasticity_loss.to(self.args.device)
            for m in self.model.modules() if isinstance(m, PlasticityGatedMLP)
        )
        spc_loss = self.spc_lambda * self.consolidator.compute_spc_loss()
        loss = loss + p_loss + spc_loss

        return (loss, outputs) if return_outputs else loss


class EWCTrainer(Trainer):
    """Textbook EWC: plain model (no fast/slow pathway, no adapters, shared head),
    Fisher penalty only. Isolates what ARIA-TRL's extra mechanisms add beyond
    standard EWC by holding the Fisher penalty strength identical to aria-trl's
    spc_lambda."""
    def __init__(self, model, args, train_dataset, eval_dataset=None,
                 ewc_lambda=15.0, consolidator=None, **kwargs):
        super().__init__(model=model, args=args, train_dataset=train_dataset,
                         eval_dataset=eval_dataset, **kwargs)
        self.ewc_lambda = ewc_lambda
        self.consolidator = consolidator if consolidator is not None else \
                            FisherConsolidator(self.model, self.args.device)

    def consolidate_task(self, task_id):
        if self.eval_dataset is None:
            return
        print(f"  Consolidating task {task_id} (EWC)...")
        self.consolidator.consolidate(task_id, self.get_eval_dataloader())
        print(f"  Done.")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        kw = {"return_outputs": return_outputs}
        if num_items_in_batch is not None:
            kw["num_items_in_batch"] = num_items_in_batch
        result = super().compute_loss(model, inputs, **kw)
        loss, outputs = (result if return_outputs else (result, None))
        loss = loss + self.ewc_lambda * self.consolidator.compute_spc_loss()
        return (loss, outputs) if return_outputs else loss


# ─────────────────────────────────────────────
# Benchmark: Sequential Domain Sentiment Learning
# Uses HuggingFace Datasets Server parquet API directly —
# avoids load_dataset loading-script issues on Python 3.12.
# ─────────────────────────────────────────────

import requests
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

TASKS      = ["sst2", "yelp", "emotion"]
TASK_NAMES = ["Movies (SST-2)", "Restaurants (Yelp)", "Social (Emotion)"]
SEEDS      = [42, 123, 7]
MODEL_NAME = "distilgpt2"


def _probe_cuda():
    """Return 'cuda' only if a basic tensor op actually works on GPU."""
    if not torch.cuda.is_available():
        return "cpu"
    try:
        t = torch.zeros(4, dtype=torch.long).cuda()
        _ = (t != -100).sum()
        return "cuda"
    except Exception as e:
        print(f"CUDA probe failed ({e}), falling back to CPU")
        return "cpu"


DEVICE = _probe_cuda()
# Scale dataset size to device: GPU gets 2000/500, CPU gets 600/150
if DEVICE == "cuda":
    TRAIN_SIZE, TEST_SIZE = 2000, 500
else:
    TRAIN_SIZE, TEST_SIZE = 200, 80

print(f"Device: {DEVICE}  |  Train size: {TRAIN_SIZE}  |  Test size: {TEST_SIZE}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _hf_parquet_urls(dataset_id, split, config=None):
    """Query HF Datasets Server for parquet file URLs."""
    resp = requests.get(
        "https://datasets-server.huggingface.co/parquet",
        params={"dataset": dataset_id},
        timeout=60,
    )
    resp.raise_for_status()
    files = resp.json().get("parquet_files", [])
    return [f["url"] for f in files
            if f["split"] == split and (config is None or f.get("config") == config)]


def _load_parquet_df(dataset_id, split, config=None, max_files=2):
    """Load parquet files from HuggingFace Hub directly with pandas."""
    urls = _hf_parquet_urls(dataset_id, split, config)
    if not urls:
        raise ValueError(f"No parquet files found for {dataset_id}/{split}")
    dfs = [pd.read_parquet(url) for url in urls[:max_files]]
    return pd.concat(dfs, ignore_index=True)


def _balanced_sample(texts, labels, n, rng):
    arr = np.array(labels)
    pos_idx = np.where(arr == 1)[0]
    neg_idx = np.where(arr == 0)[0]
    half = n // 2
    chosen = np.concatenate([
        rng.choice(pos_idx, min(half, len(pos_idx)), replace=False),
        rng.choice(neg_idx, min(half, len(neg_idx)), replace=False),
    ])
    rng.shuffle(chosen)
    return [texts[i] for i in chosen], [int(labels[i]) for i in chosen]


def load_task_dataset(task, train_n, test_n, seed=42):
    """Load and subsample one task from HuggingFace Hub via parquet."""
    rng = np.random.default_rng(seed)

    if task == "sst2":
        # stanfordnlp/sst2: columns sentence/label (0=neg,1=pos)
        tr_df = _load_parquet_df("stanfordnlp/sst2", "train")
        te_df = _load_parquet_df("stanfordnlp/sst2", "validation")
        tr_t = tr_df["sentence"].astype(str).tolist()
        tr_l = tr_df["label"].tolist()
        te_t = te_df["sentence"].astype(str).tolist()
        te_l = te_df["label"].tolist()

    elif task == "yelp":
        # Yelp/yelp_review_full: columns label(0-4)/text, binarize 0-1→0, 3-4→1
        tr_df = _load_parquet_df("Yelp/yelp_review_full", "train", max_files=1)
        te_df = _load_parquet_df("Yelp/yelp_review_full", "test",  max_files=1)
        def _binarize_yelp(df):
            texts, labels = [], []
            for lbl, txt in zip(df["label"].tolist(), df["text"].astype(str).tolist()):
                if int(lbl) in (0, 1):
                    texts.append(txt[:256]); labels.append(0)
                elif int(lbl) in (3, 4):
                    texts.append(txt[:256]); labels.append(1)
            return texts, labels
        tr_t, tr_l = _binarize_yelp(tr_df)
        te_t, te_l = _binarize_yelp(te_df)

    elif task == "emotion":
        # dair-ai/emotion: columns text/label (0=sad,1=joy,2=love,3=anger,4=fear,5=surprise)
        tr_df = _load_parquet_df("dair-ai/emotion", "train")
        te_df = _load_parquet_df("dair-ai/emotion", "test")
        def _binarize_emotion(df):
            texts, labels = [], []
            for lbl, txt in zip(df["label"].tolist(), df["text"].astype(str).tolist()):
                if int(lbl) in (1, 2):    # joy, love → positive
                    texts.append(txt); labels.append(1)
                elif int(lbl) in (0, 3, 4):  # sadness, anger, fear → negative
                    texts.append(txt); labels.append(0)
            return texts, labels
        tr_t, tr_l = _binarize_emotion(tr_df)
        te_t, te_l = _binarize_emotion(te_df)

    else:
        raise ValueError(f"Unknown task: {task}")

    tr_t, tr_l = _balanced_sample(tr_t, tr_l, train_n, rng)
    te_t, te_l = _balanced_sample(te_t, te_l, test_n, rng)

    return {
        "train": Dataset.from_dict({"text": tr_t, "label": tr_l}),
        "test":  Dataset.from_dict({"text": te_t, "label": te_l}),
    }


MAX_LEN = 128 if DEVICE == "cuda" else 48

def tokenize_dataset(dataset, tokenizer):
    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN, padding="max_length")
    return dataset.map(tok, batched=True, remove_columns=["text"])


def compute_accuracy(preds, labels):
    return float((np.argmax(preds, axis=1) == labels).mean())


def direct_eval(model, dataset, device, batch_size=64):
    """Evaluate model directly, bypassing HF Trainer to avoid mode/dropout side-effects."""
    from torch.utils.data import DataLoader
    ds = dataset.with_format("torch")
    model.eval()
    all_logits, all_labels = [], []
    loader = DataLoader(ds, batch_size=batch_size)
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            lkey           = "labels" if "labels" in batch else "label"
            labels         = batch[lkey].cpu().numpy()
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(out.logits.cpu().numpy())
            all_labels.append(labels)
    return compute_accuracy(np.concatenate(all_logits), np.concatenate(all_labels))


def bwt(acc_matrix, T):
    """Backward Transfer: mean drop on old tasks after full training."""
    if T < 2:
        return 0.0
    total = 0.0
    for j in range(T - 1):
        total += acc_matrix[T-1][j] - acc_matrix[j][j]
    return total / (T - 1)


def avg_accuracy(acc_matrix, T):
    """Average accuracy on all tasks at end of training."""
    return float(np.mean([acc_matrix[T-1][j] for j in range(T)]))


def load_fresh_model():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.config.pad_token_id = tok.pad_token_id
    return model.to(DEVICE), tok


def make_training_args(task_id, task, method, seed=42):
    epochs  = 5  # equal across all three methods and both devices
    batch   = 16 if DEVICE == "cuda" else 32
    return TrainingArguments(
        output_dir=f"{'/kaggle/working' if os.path.exists('/kaggle') else '/tmp/aria_bm'}/ckpt_{method}_{task_id}_{task}",
        learning_rate=2e-4,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=64,
        save_strategy="no",
        logging_steps=9999,
        eval_strategy="no",
        report_to=[],
        disable_tqdm=True,
        fp16=False,
        use_cpu=(DEVICE == "cpu"),
        seed=seed,  # propagate outer seed so Trainer doesn't reset to 42 for all runs
    )


def run_standard_ft(task_datasets, tok, seed):
    """Train Standard FT sequentially, return accuracy matrix."""
    set_seed(seed)
    model, _ = load_fresh_model()
    acc_matrix = {}

    for task_id, task in enumerate(TASKS):
        train_ds = tokenize_dataset(task_datasets[task]["train"], tok)
        args     = make_training_args(task_id, task, "stdft", seed=seed)

        trainer = Trainer(
            model=model, args=args,
            train_dataset=train_ds,
        )
        trainer.train()

        acc_matrix[task_id] = {}
        for prev_id, prev_task in enumerate(TASKS):
            if prev_id > task_id:
                break
            test_ds = tokenize_dataset(task_datasets[prev_task]["test"], tok)
            acc_matrix[task_id][prev_id] = direct_eval(model, test_ds, DEVICE)

    return acc_matrix


def run_aria(task_datasets, tok, seed, spc_lambda=15.0):
    """Train ARIA-TRL sequentially, return accuracy matrix."""
    set_seed(seed)
    model, _ = load_fresh_model()
    acc_matrix = {}
    consolidator = None  # passed forward so Fisher state survives trainer recreation

    for task_id, task in enumerate(TASKS):
        train_ds = tokenize_dataset(task_datasets[task]["train"], tok)
        test_ds  = tokenize_dataset(task_datasets[task]["test"],  tok)
        args     = make_training_args(task_id, task, "aria", seed=seed)
        # Warmup = one epoch's worth of steps, so gate specialization actually
        # activates within the training run instead of never triggering.
        steps_per_epoch = -(-len(train_ds) // args.per_device_train_batch_size)  # ceil div

        # New Trainer per task (avoids Accelerate state reset bug on re-use),
        # but consolidator is passed in so SPC Fisher data persists.
        # Adapter state lives on the model via model._aria_state so it also persists.
        trainer = ContinualSFTTrainer(
            model=model, args=args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            tokenizer=tok,
            plasticity_lambda=0.001,
            spc_lambda=spc_lambda,
            adapter_dim=64,
            slow_lr_ratio=1.0,
            warmup_steps=steps_per_epoch,
            consolidator=consolidator,
        )
        trainer.add_task(task_id)
        if task_id > 0:
            trainer.freeze_adapters_except(task_id)

        trainer.train()

        if task_id < len(TASKS) - 1:
            trainer.consolidate_task(task_id)

        consolidator = trainer.consolidator  # carry Fisher state to next task

        acc_matrix[task_id] = {}
        for prev_id, prev_task in enumerate(TASKS):
            if prev_id > task_id:
                break
            prev_test = tokenize_dataset(task_datasets[prev_task]["test"], tok)
            trainer.set_active_task(prev_id)
            acc_matrix[task_id][prev_id] = direct_eval(model, prev_test, DEVICE)

        trainer.set_active_task(task_id)

    return acc_matrix


def run_ewc(task_datasets, tok, seed):
    """Train with standard (textbook) EWC sequentially, return accuracy matrix."""
    set_seed(seed)
    model, _ = load_fresh_model()
    acc_matrix = {}
    consolidator = None  # passed forward so Fisher state persists across tasks

    for task_id, task in enumerate(TASKS):
        train_ds = tokenize_dataset(task_datasets[task]["train"], tok)
        test_ds  = tokenize_dataset(task_datasets[task]["test"],  tok)
        args     = make_training_args(task_id, task, "ewc", seed=seed)

        trainer = EWCTrainer(
            model=model, args=args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            ewc_lambda=15.0,  # matches aria-trl's spc_lambda for a fair comparison
            consolidator=consolidator,
        )
        trainer.train()

        if task_id < len(TASKS) - 1:
            trainer.consolidate_task(task_id)
        consolidator = trainer.consolidator

        acc_matrix[task_id] = {}
        for prev_id, prev_task in enumerate(TASKS):
            if prev_id > task_id:
                break
            test_ds_prev = tokenize_dataset(task_datasets[prev_task]["test"], tok)
            acc_matrix[task_id][prev_id] = direct_eval(model, test_ds_prev, DEVICE)

    return acc_matrix


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("\n" + "="*70)
    print("Loading datasets...")
    _, tok = load_fresh_model()
    task_datasets = {task: load_task_dataset(task, TRAIN_SIZE, TEST_SIZE) for task in TASKS}
    for task, d in task_datasets.items():
        print(f"  {task}: {len(d['train'])} train / {len(d['test'])} test")

    T = len(TASKS)

    methods = {
        "standard_ft": run_standard_ft,
        "ewc":         run_ewc,
        "aria_trl":    run_aria,
    }
    method_labels = {"standard_ft": "Standard FT", "ewc": "EWC", "aria_trl": "ARIA-TRL"}
    accs = {m: [] for m in methods}
    bwts = {m: [] for m in methods}
    per_seed_matrices = {m: {} for m in methods}

    for seed in SEEDS:
        print(f"\n{'='*70}")
        print(f"SEED {seed}")
        print(f"{'='*70}")

        for m, run_fn in methods.items():
            print(f"\n--- {method_labels[m]} ---")
            matrix = run_fn(task_datasets, tok, seed)
            acc = avg_accuracy(matrix, T)
            bw  = bwt(matrix, T)
            accs[m].append(acc)
            bwts[m].append(bw)
            per_seed_matrices[m][seed] = matrix
            print(f"  {method_labels[m]:<12} ACC={acc:.4f}  BWT={bw:.4f}")
            for t in range(T):
                row = " | ".join(f"{matrix[t].get(j, float('nan')):.3f}" for j in range(T))
                print(f"  After task {t} ({TASK_NAMES[t]}): {row}")

    print("\n" + "="*70)
    print("FINAL RESULTS (mean ± std over 3 seeds)")
    print("="*70)
    print(f"{'Method':<15} {'Avg Accuracy':>14} {'Backward Transfer':>18}")
    print("-"*50)
    for m in methods:
        print(f"{method_labels[m]:<15} {np.mean(accs[m]):>8.4f} ± {np.std(accs[m]):.4f}"
              f"   {np.mean(bwts[m]):>8.4f} ± {np.std(bwts[m]):.4f}")
    print()

    acc_delta_ewc  = np.mean(accs["aria_trl"]) - np.mean(accs["ewc"])
    bwt_delta_ewc  = np.mean(bwts["aria_trl"]) - np.mean(bwts["ewc"])
    acc_delta_stdft = np.mean(accs["aria_trl"]) - np.mean(accs["standard_ft"])
    bwt_delta_stdft = np.mean(bwts["aria_trl"]) - np.mean(bwts["standard_ft"])

    print(f"ARIA-TRL vs Standard FT — ACC delta: {acc_delta_stdft:+.4f}  BWT delta: {bwt_delta_stdft:+.4f}")
    print(f"ARIA-TRL vs EWC         — ACC delta: {acc_delta_ewc:+.4f}  BWT delta: {bwt_delta_ewc:+.4f}")
    print("="*70)

    # ── Save results to JSON ──────────────────────────────────────────────────
    import json
    results = {
        "benchmark": "Sequential Domain Sentiment Benchmark",
        "description": "3-task continual learning: SST-2 -> Yelp Reviews -> Emotion",
        "model": MODEL_NAME,
        "datasets": ["stanfordnlp/sst2", "Yelp/yelp_review_full", "dair-ai/emotion"],
        "task_names": TASK_NAMES,
        "seeds": SEEDS,
        "train_size": TRAIN_SIZE,
        "test_size": TEST_SIZE,
        "methods_compared": list(methods.keys()),
        "per_seed_results": {},
        "final_summary": {},
    }
    for seed in SEEDS:
        results["per_seed_results"][f"seed_{seed}"] = {}
        for m in methods:
            matrix = per_seed_matrices[m][seed]
            results["per_seed_results"][f"seed_{seed}"][m] = {
                "acc_matrix": {
                    f"after_task_{t}": [matrix[t].get(j) for j in range(t + 1)]
                    for t in range(T)
                },
                "ACC": round(avg_accuracy(matrix, T), 4),
                "BWT": round(bwt(matrix, T), 4),
            }
    for m in methods:
        results["final_summary"][m] = {
            "ACC_mean": round(float(np.mean(accs[m])), 4),
            "ACC_std":  round(float(np.std(accs[m])), 4),
            "BWT_mean": round(float(np.mean(bwts[m])), 4),
            "BWT_std":  round(float(np.std(bwts[m])), 4),
        }
    results["final_summary"]["deltas"] = {
        "aria_vs_standard_ft": {"ACC_delta": round(float(acc_delta_stdft), 4), "BWT_delta": round(float(bwt_delta_stdft), 4)},
        "aria_vs_ewc":         {"ACC_delta": round(float(acc_delta_ewc), 4),   "BWT_delta": round(float(bwt_delta_ewc), 4)},
    }

    out_dir = "results" if os.path.isdir("results") else "."
    out_path = os.path.join(out_dir, "kaggle_benchmark_results_3way.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
