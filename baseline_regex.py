import argparse
import json
import re
from collections import defaultdict
from typing import Dict, List, Tuple

Span = Tuple[int, int, str]

# High-precision patterns for contact/ID-like entities; conservative to avoid FP storms.
PATTERNS: Dict[str, List[re.Pattern]] = {
    "email": [
        re.compile(r"[A-Za-z0-9_.+%-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    ],
    "phone": [
        re.compile(r"\+?\d{1,2}[\s-]?\d{2,3}[\s-]?\d{3}[\s-]?\d{3}")
    ],
    "pesel": [
        re.compile(r"(?<!\d)(\d{11})(?!\d)")
    ],
    "bank-account": [
        re.compile(r"PL\s?\d{2}(?:\s?\d{4}){6}")
    ],
    "document-number": [
        re.compile(r"[A-Z]{2}\d{6}")
    ],
    "address": [
        re.compile(r"\b(?:ul\.|al\.|aleja|plac|pl\.|os\.|ulica)\s+[^,;]+?\d[^,;]*")
    ],
    "city": [
        re.compile(r"\b(?:miasto\s+)?([A-ZŁŚŻŹĆÓŃ][\w\-ąćęłńóśżź]+)")
    ],
}


def load_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def dedup_spans(spans: List[Span]) -> List[Span]:
    seen = set()
    uniq = []
    for s, e, label in spans:
        key = (s, e, label)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((s, e, label))
    return uniq


def predict_spans(text: str) -> List[Span]:
    spans: List[Span] = []
    for label, regexes in PATTERNS.items():
        for rgx in regexes:
            for m in rgx.finditer(text):
                spans.append((m.start(), m.end(), label))
    return dedup_spans(spans)


def evaluate(gold: List[List[Span]], pred: List[List[Span]]) -> Dict[str, float]:
    tp = fp = fn = 0
    for g_spans, p_spans in zip(gold, pred):
        g_set = set(g_spans)
        p_set = set(p_spans)
        tp += len(g_set & p_set)
        fp += len(p_set - g_set)
        fn += len(g_set - p_set)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def main():
    parser = argparse.ArgumentParser(description="Regex baseline evaluator")
    parser.add_argument("--data", required=True, help="Path to jsonl with text and entities")
    args = parser.parse_args()

    data = load_jsonl(args.data)
    gold: List[List[Span]] = []
    pred: List[List[Span]] = []
    for row in data:
        gold.append([(ent["start"], ent["end"], ent["label"]) for ent in row.get("entities", [])])
        pred.append(predict_spans(row["text"]))

    metrics = evaluate(gold, pred)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
