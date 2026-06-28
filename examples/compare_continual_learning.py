"""
Compare continual learning methods on sequential text classification tasks.

Compares:
1. Standard fine-tuning (baseline - no forgetting prevention)
2. EWC (Elastic Weight Consolidation)
3. ARIA-TRL (dual pathways + Fisher consolidation + adapters)

Measures: Classification accuracy, forgetting, forward transfer
"""

import torch
import random
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from aria_trl import ContinualSFTTrainer, ARIAConfig
from aria_trl.baselines import EWCTrainer
from torch.utils.data import DataLoader


def set_seed(seed: int):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_task_datasets(task_name: str, num_samples: int = 200) -> DatasetDict:
    """Create synthetic task dataset with proper train/test split."""
    random.seed(42)

    if task_name == "sentiment":
        texts = [
            "This movie was absolutely fantastic! I loved every second.",
            "Terrible waste of time. Completely disappointed.",
            "The acting was phenomenal and the story gripping.",
            "Not worth watching. Poorly made.",
        ]
        labels = [1, 0, 1, 0]

    elif task_name == "toxicity":
        texts = [
            "You are a wonderful person!",
            "That is a rude comment.",
            "Great job on your work!",
            "You are completely useless.",
        ]
        labels = [0, 1, 0, 1]

    elif task_name == "spam":
        texts = [
            "Check out this amazing deal!",
            "Hi, how are you?",
            "Click here to win $1000!",
            "Let me know if you need anything.",
        ]
        labels = [1, 0, 1, 0]

    else:
        raise ValueError(f"Unknown task: {task_name}")

    # Expand to num_samples
    all_texts = texts * (num_samples // len(texts) + 1)
    all_labels = labels * (num_samples // len(labels) + 1)
    all_texts = all_texts[:num_samples]
    all_labels = all_labels[:num_samples]

    # Shuffle
    indices = list(range(len(all_texts)))
    random.shuffle(indices)
    all_texts = [all_texts[i] for i in indices]
    all_labels = [all_labels[i] for i in indices]

    # Split: 160 train, 40 test (NOT eval from train)
    train_split = int(0.8 * len(all_texts))
    train_data = Dataset.from_dict({
        "text": all_texts[:train_split],
        "label": all_labels[:train_split],
    })
    test_data = Dataset.from_dict({
        "text": all_texts[train_split:],
        "label": all_labels[train_split:],
    })

    return {"train": train_data, "test": test_data}


def compute_accuracy(predictions, labels):
    """Compute classification accuracy."""
    return (np.argmax(predictions, axis=1) == labels).mean()


class StandardFTTrainer(Trainer):
    """Standard fine-tuning (no forgetting prevention)."""
    pass


def train_standard_ft(model, tokenizer, task_datasets, device):
    """Train with standard fine-tuning (baseline)."""
    print("\n" + "="*80)
    print("STANDARD FINE-TUNING (Baseline)")
    print("="*80)

    task_accuracies = {}

    for task_id, (task_name, datasets) in enumerate(task_datasets.items()):
        print(f"\nTask {task_id + 1}: {task_name.upper()}")

        train_dataset = datasets["train"]
        test_dataset = datasets["test"]

        # Tokenize
        def tokenize_fn(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=128,
                padding="max_length",
            )

        train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
        test_dataset = test_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

        # Training
        training_args = TrainingArguments(
            output_dir=f"./checkpoints/ft_task_{task_id}_{task_name}",
            learning_rate=2e-4,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            save_strategy="no",
            logging_steps=20,
            eval_strategy="no",
            report_to=[],
        )

        trainer = StandardFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )

        trainer.train()

        # Evaluate on all test sets seen so far
        if task_id not in task_accuracies:
            task_accuracies[task_id] = {}

        for prev_task_id, (prev_task_name, prev_datasets) in enumerate(task_datasets.items()):
            if prev_task_id > task_id:
                break

            prev_test = prev_datasets["test"]
            prev_test = prev_test.map(tokenize_fn, batched=True, remove_columns=["text"])

            outputs = trainer.predict(prev_test)
            acc = compute_accuracy(outputs.predictions, outputs.label_ids)
            task_accuracies[task_id][prev_task_id] = acc
            print(f"  Accuracy on {prev_task_name}: {acc:.4f}")

    return task_accuracies


def train_ewc(model, tokenizer, task_datasets, device):
    """Train with EWC baseline."""
    print("\n" + "="*80)
    print("EWC (Elastic Weight Consolidation)")
    print("="*80)

    consolidator = EWCTrainer(model, device, ewc_lambda=0.4)
    task_accuracies = {}

    for task_id, (task_name, datasets) in enumerate(task_datasets.items()):
        print(f"\nTask {task_id + 1}: {task_name.upper()}")

        train_dataset = datasets["train"]
        test_dataset = datasets["test"]

        # Tokenize
        def tokenize_fn(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=128,
                padding="max_length",
            )

        train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
        test_dataset = test_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

        # Training with EWC loss
        training_args = TrainingArguments(
            output_dir=f"./checkpoints/ewc_task_{task_id}_{task_name}",
            learning_rate=2e-4,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            save_strategy="no",
            logging_steps=20,
            eval_strategy="no",
            report_to=[],
        )

        class EWCTrainerWithLoss(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                loss = super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
                ewc_loss = consolidator.compute_ewc_loss()
                total_loss = loss + ewc_loss if isinstance(loss, torch.Tensor) else loss
                return (total_loss, outputs) if return_outputs else total_loss

        trainer = EWCTrainerWithLoss(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )

        trainer.train()

        # Consolidate after task
        test_loader = DataLoader(test_dataset, batch_size=16)
        consolidator.consolidate(task_id, test_loader)

        # Evaluate on all test sets
        if task_id not in task_accuracies:
            task_accuracies[task_id] = {}

        for prev_task_id, (prev_task_name, prev_datasets) in enumerate(task_datasets.items()):
            if prev_task_id > task_id:
                break

            prev_test = prev_datasets["test"]
            prev_test = prev_test.map(tokenize_fn, batched=True, remove_columns=["text"])

            outputs = trainer.predict(prev_test)
            acc = compute_accuracy(outputs.predictions, outputs.label_ids)
            task_accuracies[task_id][prev_task_id] = acc
            print(f"  Accuracy on {prev_task_name}: {acc:.4f}")

    return task_accuracies


def train_aria(model, tokenizer, task_datasets, device):
    """Train with ARIA-TRL."""
    print("\n" + "="*80)
    print("ARIA-TRL (Dual Pathways + Fisher Consolidation)")
    print("="*80)

    aria_config = ARIAConfig(
        plasticity_lambda=0.01,
        spc_lambda=100.0,
        adapter_dim=64,
        slow_lr_ratio=0.5,
    )

    task_accuracies = {}

    for task_id, (task_name, datasets) in enumerate(task_datasets.items()):
        print(f"\nTask {task_id + 1}: {task_name.upper()}")

        train_dataset = datasets["train"]
        test_dataset = datasets["test"]

        # Tokenize
        def tokenize_fn(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=128,
                padding="max_length",
            )

        train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
        test_dataset = test_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

        # Training
        training_args = TrainingArguments(
            output_dir=f"./checkpoints/aria_task_{task_id}_{task_name}",
            learning_rate=2e-4,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            save_strategy="no",
            logging_steps=20,
            eval_strategy="no",
            report_to=[],
        )

        trainer = ContinualSFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            aria_config=aria_config,
            consolidate_after_task=True,
            freeze_old_adapters=True,
        )

        trainer.add_task(task_id)
        if task_id > 0:
            trainer.freeze_adapters_except(task_id)

        trainer.train()

        # Consolidate
        if task_id < len(task_datasets) - 1:
            trainer.consolidate_task(task_id)

        # Evaluate
        if task_id not in task_accuracies:
            task_accuracies[task_id] = {}

        for prev_task_id, (prev_task_name, prev_datasets) in enumerate(task_datasets.items()):
            if prev_task_id > task_id:
                break

            prev_test = prev_datasets["test"]
            prev_test = prev_test.map(tokenize_fn, batched=True, remove_columns=["text"])

            outputs = trainer.predict(prev_test)
            acc = compute_accuracy(outputs.predictions, outputs.label_ids)
            task_accuracies[task_id][prev_task_id] = acc
            print(f"  Accuracy on {prev_task_name}: {acc:.4f}")

    return task_accuracies


def print_results(results_dict):
    """Print comparison results."""
    print("\n" + "="*80)
    print("COMPARISON: FORGETTING & FORWARD TRANSFER")
    print("="*80)

    for method_name, task_accs in results_dict.items():
        print(f"\n{method_name}:")
        print("-" * 60)

        # Build accuracy matrix
        num_tasks = max([max(accs.keys()) for accs in task_accs.values()]) + 1

        # Print header
        print("After training task N, accuracy on each task:")
        print("Task trained | Task 0 | Task 1 | Task 2")
        print("-" * 50)

        # Print rows
        for train_task_id in range(num_tasks):
            if train_task_id in task_accs:
                accs = task_accs[train_task_id]
                row = f"     {train_task_id}      |"
                for test_task_id in range(num_tasks):
                    if test_task_id in accs:
                        row += f" {accs[test_task_id]:.4f} |"
                    else:
                        row += "   -   |"
                print(row)

        # Calculate forgetting
        if 0 in task_accs and 1 in task_accs:
            task0_acc_after_task1 = task_accs[1][0]
            task0_acc_immediate = task_accs[0][0]
            forgetting_at_task2 = max(0, task0_acc_immediate - task0_acc_after_task1)
            print(f"\nForgetting (Task 0 after Task 2): {forgetting_at_task2:.4f}")

        # Calculate average accuracy
        final_task_id = max(task_accs.keys())
        avg_acc = np.mean([task_accs[final_task_id][i] for i in range(final_task_id + 1)])
        print(f"Average accuracy (final): {avg_acc:.4f}")

        # Forward transfer
        if 1 in task_accs and 0 in task_accs:
            task1_with_pretrain = task_accs[1][1]
            # Estimate task1 without pretraining (would be random)
            ft_from_scratch = 0.5  # random for binary classification
            forward_transfer = task1_with_pretrain - ft_from_scratch
            print(f"Forward transfer (Task 1): +{forward_transfer:.4f}")


def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("="*80)
    print("CONTINUAL LEARNING COMPARISON: Standard FT vs EWC vs ARIA-TRL")
    print("="*80)

    # Load model and tokenizer
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    def load_model():
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        model.config.pad_token_id = tokenizer.pad_token_id
        return model.to(device)

    # Create task datasets
    tasks = ["sentiment", "toxicity", "spam"]
    task_datasets = {task: create_task_datasets(task, num_samples=200) for task in tasks}

    results = {}

    # Standard FT
    model_ft = load_model()
    results["Standard FT"] = train_standard_ft(model_ft, tokenizer, task_datasets, device)

    # EWC
    model_ewc = load_model()
    results["EWC"] = train_ewc(model_ewc, tokenizer, task_datasets, device)

    # ARIA-TRL
    model_aria = load_model()
    results["ARIA-TRL"] = train_aria(model_aria, tokenizer, task_datasets, device)

    # Print results
    print_results(results)

    print("\n" + "="*80)
    print("Experiment complete!")
    print("="*80)


if __name__ == "__main__":
    main()
