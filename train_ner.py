import argparse
import json
from typing import Dict, List, Tuple

import numpy as np
from datasets import load_dataset
import evaluate
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

Span = Tuple[int, int, str]


def collect_labels(dataset) -> List[str]:
    labels = set()
    for row in dataset:
        for ent in row["entities"]:
            labels.add(ent["label"])
    ordered = sorted(labels)
    bio = ["O"]
    for lab in ordered:
        bio.append(f"B-{lab}")
        bio.append(f"I-{lab}")
    return bio


def build_span_map(spans: List[Span]) -> List[Span]:
    # Already a list of (start, end, label); no change, kept for clarity.
    return spans


def encode_examples(examples, tokenizer, label2id):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        return_offsets_mapping=True,
    )
    all_labels = []
    o_id = label2id["O"]
    for offsets, ents, text in zip(tokenized["offset_mapping"], examples["entities"], examples["text"]):
        seq_labels = []
        spans = build_span_map([(e["start"], e["end"], e["label"]) for e in ents])
        for i, (tok_start, tok_end) in enumerate(offsets):
            if tok_start == tok_end == 0:
                seq_labels.append(-100)
                continue
            seq_labels.append(o_id)
        for start, end, label in spans:
            # Assign B/I tags where token overlaps the entity span.
            first = True
            for i, (tok_start, tok_end) in enumerate(offsets):
                if tok_start is None or tok_end is None:
                    continue
                if tok_start == tok_end == 0:
                    continue
                if tok_start >= end or tok_end <= start:
                    continue
                tag = f"B-{label}" if first else f"I-{label}"
                seq_labels[i] = label2id[tag]
                first = False
        all_labels.append(seq_labels)
    tokenized["labels"] = all_labels
    tokenized.pop("offset_mapping", None)
    return tokenized


def decode_predictions(predictions, labels, id2label):
    preds = np.argmax(predictions, axis=2)
    true_predictions = []
    true_labels = []
    for pred_seq, label_seq in zip(preds, labels):
        pred_labels = []
        gold_labels = []
        for p, l in zip(pred_seq, label_seq):
            if l == -100:
                continue
            pred_labels.append(id2label.get(int(p), "O"))
            gold_labels.append(id2label.get(int(l), "O"))
        true_predictions.append(pred_labels)
        true_labels.append(gold_labels)
    return true_predictions, true_labels


def main():
    parser = argparse.ArgumentParser(description="Finetune token classification model on synthetic data")
    parser.add_argument("--train", required=True, help="train jsonl path")
    parser.add_argument("--val", required=True, help="validation jsonl path")
    parser.add_argument("--model", default="allegro/herbert-base-cased", help="HF model checkpoint")
    parser.add_argument("--output", default="outputs/ner", help="where to save model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_files = {"train": args.train, "validation": args.val}
    raw = load_dataset("json", data_files=data_files)

    label_list = collect_labels(raw["train"])  # assume val labels subset
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def preprocess(examples):
        return encode_examples(examples, tokenizer, label2id)

    tokenized = raw.map(preprocess, batched=True, remove_columns=["text", "placeholders", "values", "entities", "meta"])

    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)
    metric = evaluate.load("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        preds, refs = decode_predictions(predictions, labels, id2label)
        results = metric.compute(predictions=preds, references=refs)
        # Use micro F1 over entities
        f1 = results["overall_f1"]
        precision = results["overall_precision"]
        recall = results["overall_recall"]
        return {"f1": f1, "precision": precision, "recall": recall}

    training_args = TrainingArguments(
        output_dir=args.output,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        seed=args.seed,
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()
