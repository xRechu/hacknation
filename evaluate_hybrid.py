import argparse
import json
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer

Span = Tuple[int, int, str]

# Reuse conservative patterns to guarantee high-precision matches.
import re
PATTERNS: Dict[str, List[re.Pattern]] = {
    # Keep only high-precision classes to avoid FP storm.
    "email": [re.compile(r"[A-Za-z0-9_.+%-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")],
    "phone": [re.compile(r"\+?\d{1,2}[\s-]?\d{2,3}[\s-]?\d{3}[\s-]?\d{3}")],
    "pesel": [re.compile(r"(?<!\d)(\d{11})(?!\d)")],
    "bank-account": [re.compile(r"PL\s?\d{2}(?:\s?\d{4}){6}")],
    "document-number": [re.compile(r"[A-Z]{2}\d{6}")],
}


def dedup(spans: List[Span]) -> List[Span]:
    seen = set()
    out = []
    for s, e, l in spans:
        key = (s, e, l)
        if key in seen:
            continue
        seen.add(key)
        out.append((s, e, l))
    return out


def regex_spans(text: str) -> List[Span]:
    spans: List[Span] = []
    for label, regs in PATTERNS.items():
        for rgx in regs:
            for m in rgx.finditer(text):
                spans.append((m.start(), m.end(), label))
    return dedup(spans)


def spans_from_bio(offsets, labels, id2label) -> List[Span]:
    spans: List[Span] = []
    current = None  # (start, end, label)
    for (start, end), lab_id in zip(offsets, labels):
        if start == end == 0:  # likely special token
            continue
        tag = id2label.get(int(lab_id), "O")
        if tag == "O":
            if current:
                spans.append(current)
                current = None
            continue
        prefix, label = tag.split("-", 1)
        if prefix == "B":
            if current:
                spans.append(current)
            current = (start, end, label)
        elif prefix == "I":
            if current and current[2] == label:
                current = (current[0], end, label)
            else:
                current = (start, end, label)
    if current:
        spans.append(current)
    return spans


def evaluate(gold: List[List[Span]], pred: List[List[Span]]):
    tp = fp = fn = 0
    for g, p in zip(gold, pred):
        g_set = set(g)
        p_set = set(p)
        tp += len(g_set & p_set)
        fp += len(p_set - g_set)
        fn += len(g_set - p_set)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def main():
    parser = argparse.ArgumentParser(description="Evaluate model + regex hybrid on jsonl dataset")
    parser.add_argument("--model", required=True, help="Path to fine-tuned model directory")
    parser.add_argument("--data", required=True, help="jsonl file to evaluate")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--regex_weight", type=float, default=1.0, help="If >0, regex spans are added")
    parser.add_argument("--prob_threshold", type=float, default=0.0, help="Min prob to keep non-O label; else set to O")
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model).to(device)
    id2label = model.config.id2label
    label2id = {v: k for k, v in id2label.items()}
    o_id = label2id.get("O")
    if o_id is None:
        raise ValueError("Model config lacks 'O' label; re-train with O class.")

    ds = load_dataset("json", data_files={"data": args.data})["data"]

    gold_spans = []
    pred_spans = []

    model.eval()
    for row in ds:
        text = row["text"]
        encoded = tokenizer(text, return_offsets_mapping=True, return_tensors="pt", truncation=True)
        offsets = encoded.pop("offset_mapping")[0].tolist()
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits[0]
        probs = F.softmax(logits, dim=-1)
        max_probs, pred_ids = probs.max(dim=-1)
        pred_ids = pred_ids.cpu().tolist()
        max_probs = max_probs.cpu().tolist()

        # Apply probability threshold: low-confidence -> O
        pred_ids = [pid if p >= args.prob_threshold else o_id for pid, p in zip(pred_ids, max_probs)]
        model_spans = spans_from_bio(offsets, pred_ids, id2label)
        combined = model_spans
        if args.regex_weight > 0:
            combined = dedup(model_spans + regex_spans(text))
        pred_spans.append(combined)
        gold_spans.append([(e["start"], e["end"], e["label"]) for e in row["entities"]])

    metrics = evaluate(gold_spans, pred_spans)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
