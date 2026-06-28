"""
Example: Continual learning on DistilGPT2 with sequential text classification tasks.

Three tasks:
1. Sentiment analysis (positive/negative)
2. Toxicity detection (toxic/non-toxic)
3. Spam detection (spam/non-spam)

Each task is trained sequentially. After each task, Fisher consolidation
protects the learned knowledge from being overwritten by the next task.

This demonstrates how ARIA prevents catastrophic forgetting in LLM fine-tuning.
"""

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from aria_trl import ContinualSFTTrainer, ARIAConfig
from aria_trl.utils import compute_continual_metrics


def create_task_datasets(task_name: str, num_samples: int = 200) -> DatasetDict:
    """Create synthetic task dataset."""
    import random

    random.seed(42)

    if task_name == "sentiment":
        texts = [
            "This movie was absolutely fantastic! I loved every second.",
            "Terrible waste of time. Completely disappointed.",
            "The acting was phenomenal and the story gripping.",
            "Not worth watching. Poorly made.",
        ]
        labels = [1, 0, 1, 0]  # positive=1, negative=0

    elif task_name == "toxicity":
        texts = [
            "You are a wonderful person!",
            "That is a rude comment.",
            "Great job on your work!",
            "You are completely useless.",
        ]
        labels = [0, 1, 0, 1]  # toxic=1, non-toxic=0

    elif task_name == "spam":
        texts = [
            "Check out this amazing deal!",
            "Hi, how are you?",
            "Click here to win $1000!",
            "Let me know if you need anything.",
        ]
        labels = [1, 0, 1, 0]  # spam=1, non-spam=0

    else:
        raise ValueError(f"Unknown task: {task_name}")

    # Expand to num_samples
    all_texts = texts * (num_samples // len(texts) + 1)
    all_labels = labels * (num_samples // len(labels) + 1)
    all_texts = all_texts[:num_samples]
    all_labels = all_labels[:num_samples]

    # Random shuffle
    indices = list(range(len(all_texts)))
    random.shuffle(indices)
    all_texts = [all_texts[i] for i in indices]
    all_labels = [all_labels[i] for i in indices]

    # Split train/eval
    split = int(0.8 * len(all_texts))
    train_data = Dataset.from_dict({
        "text": all_texts[:split],
        "label": all_labels[:split],
    })
    eval_data = Dataset.from_dict({
        "text": all_texts[split:],
        "label": all_labels[split:],
    })

    return DatasetDict({"train": train_data, "validation": eval_data})


def main():
    """Run continual learning experiment on DistilGPT2."""
    print("\n" + "=" * 80)
    print("ARIA-TRL: Continual Learning on DistilGPT2")
    print("=" * 80 + "\n")

    # Load model and tokenizer
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # ARIA configuration
    aria_config = ARIAConfig(
        plasticity_lambda=0.01,      # bimodal gate specialization
        spc_lambda=100.0,            # Fisher consolidation strength
        adapter_dim=32,              # task adapter bottleneck
        slow_lr_ratio=0.5,           # asymmetric LR: slow=0.5x fast
        warmup_steps=100,            # before plasticity loss activates
    )

    # Sequential tasks
    tasks = ["sentiment", "toxicity", "spam"]
    task_datasets = {}
    for task in tasks:
        task_datasets[task] = create_task_datasets(task, num_samples=200)

    # Track accuracies for metrics
    all_task_accuracies = {}

    # Training
    trainer = None
    for task_id, task_name in enumerate(tasks):
        print(f"\n{'='*80}")
        print(f"Task {task_id + 1}/{len(tasks)}: {task_name.upper()}")
        print(f"{'='*80}\n")

        # Get datasets
        train_dataset = task_datasets[task_name]["train"]
        eval_dataset = task_datasets[task_name]["validation"]

        # Tokenize
        def tokenize_fn(examples):
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                max_length=128,
                padding="max_length",
            )
            # For causal LM, labels = input_ids (model will compute language modeling loss)
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        train_dataset = train_dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=["text"],
        )
        eval_dataset = eval_dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=["text"],
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"./checkpoints/task_{task_id}_{task_name}",
            learning_rate=2e-4,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            save_strategy="no",
            logging_steps=10,
            eval_strategy="epoch",
            report_to=[],
        )

        # Create new trainer for each task
        trainer = ContinualSFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            aria_config=aria_config,
            consolidate_after_task=True,
            freeze_old_adapters=True,
        )

        # Add task and freeze previous adapters
        trainer_task_id = trainer.add_task(task_id)
        if task_id > 0:
            trainer.freeze_adapters_except(task_id)

        # Train on this task
        print(f"Training on {task_name}...")
        trainer.train()

        # Consolidate (Fisher estimation) before next task
        if task_id < len(tasks) - 1:
            trainer.consolidate_task(task_id)

        # Evaluate on all tasks seen so far
        print(f"\nEvaluating on all tasks...")
        task_accuracies = {}
        for eval_task_id, eval_task_name in enumerate(tasks[:task_id + 1]):
            eval_data = task_datasets[eval_task_name]["validation"].map(
                tokenize_fn,
                batched=True,
                remove_columns=["text"],
            )
            metrics = trainer.evaluate(eval_dataset=eval_data)
            acc = metrics.get("eval_accuracy", 0.0)
            task_accuracies[eval_task_id] = acc
            print(f"  {eval_task_name:12s}: {acc:.4f}")

        all_task_accuracies[task_id] = task_accuracies

    # Compute continual learning metrics
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}\n")

    metrics = compute_continual_metrics(all_task_accuracies)
    print(f"Average Accuracy:  {metrics['avg_accuracy']:.4f}")
    print(f"Forgetting:        {metrics['forgetting']:.4f}")
    print(f"Forward Transfer:  {metrics['forward_transfer']:.4f}")

    print("\nAccuracy Matrix (rows=training task, cols=eval task):")
    print("Task\tSentiment\tToxicity\tSpam")
    for task_id in range(len(tasks)):
        row = f"{tasks[task_id]}\t"
        for eval_id in range(task_id + 1):
            acc = all_task_accuracies[task_id][eval_id]
            row += f"{acc:.4f}\t\t"
        print(row)

    print(f"\n{'='*80}")
    print("Experiment complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
