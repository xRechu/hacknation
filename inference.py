import argparse
import re
import json
from typing import List, Tuple, Dict, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForTokenClassification

# (Start, End, Label)
Span = Tuple[int, int, str]

class Anonymizer:
    def __init__(self, model_path: str, device: str = None, prob_threshold: float = 0.5):
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            
        print(f"Loading model from {model_path} to {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()
        
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
        self.o_id = self.label2id.get("O", 0)
        self.prob_threshold = prob_threshold

        # Patterns from baseline/evaluate
        self.patterns = {
             "email": [re.compile(r"[A-Za-z0-9_.+%-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")],
             "phone": [re.compile(r"\+?\d{1,2}[\s-]?\d{2,3}[\s-]?\d{3}[\s-]?\d{3}")],
             "pesel": [re.compile(r"(?<!\d)(\d{11})(?!\d)")],
             "bank-account": [re.compile(r"PL\s?\d{2}(?:\s?\d{4}){6}")],
             "document-number": [re.compile(r"[A-Z]{2}\d{6}")],
             "credit-card-number": [re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b")],
        }

    def _get_regex_spans(self, text: str) -> List[Span]:
        spans = []
        for label, patterns in self.patterns.items():
            for pat in patterns:
                for match in pat.finditer(text):
                    spans.append((match.start(), match.end(), label))
        return spans

    def _spans_from_bio(self, offsets, labels) -> List[Span]:
        spans = []
        current = None
        for (start, end), lab_id in zip(offsets, labels):
            if start == end == 0: continue
            
            tag = self.id2label.get(int(lab_id), "O")
            if tag == "O":
                if current:
                    spans.append(current)
                    current = None
                continue
            
            try:
                prefix, label = tag.split("-", 1)
            except ValueError:
                # Handle cases where tag might be malformed or just label
                prefix, label = "I", tag
            
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

    def _get_model_spans_internal(self, text: str) -> List[Span]:
        # Pre-check length
        if not text.strip():
            return []
            
        encoded = self.tokenizer(text, return_offsets_mapping=True, return_tensors="pt", truncation=True, max_length=512)
        offsets = encoded.pop("offset_mapping")[0].tolist()
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        with torch.no_grad():
            logits = self.model(**encoded).logits[0]
            
        probs = F.softmax(logits, dim=-1)
        max_probs, pred_ids = probs.max(dim=-1)
        
        pred_ids = pred_ids.cpu().tolist()
        max_probs = max_probs.cpu().tolist()
        
        final_ids = [pid if p >= self.prob_threshold else self.o_id for pid, p in zip(pred_ids, max_probs)]
        return self._spans_from_bio(offsets, final_ids)

    def predict(self, text: str) -> List[Span]:
        # Regex
        r_spans = self._get_regex_spans(text)
        # Model
        m_spans = self._get_model_spans_internal(text)
        
        # Combine
        combined = list(set(r_spans + m_spans))
        combined.sort(key=lambda x: x[0])
        return combined

    def anonymize(self, text: str) -> str:
        spans = self.predict(text)
        
        # Sort by start, then by length desc (to prioritize longer matches)
        spans.sort(key=lambda x: (x[0], -(x[1]-x[0])))
        
        chars = list(text)
        kept_spans = []
        mask = [False] * len(text)
        
        for s, e, l in spans:
            # Check overlap
            collision = False
            for i in range(s, e):
                if mask[i]:
                    collision = True
                    break
            if not collision:
                kept_spans.append((s, e, l))
                for i in range(s, e):
                    mask[i] = True
        
        # Apply replacement in reverse order
        kept_spans.sort(key=lambda x: x[0], reverse=True)
        
        for s, e, l in kept_spans:
            replacement = f"{{{l}}}"
            chars[s:e] = list(replacement)
            
        return "".join(chars)

def main():
    parser = argparse.ArgumentParser(description="Inference for PII Anonymizer")
    parser.add_argument("--model", required=True, help="Path to trained model directory")
    parser.add_argument("--text", help="Text to anonymize")
    parser.add_argument("--file", help="File with lines to anonymize")
    parser.add_argument("--output", help="Optional output file; if not set, prints to stdout")
    parser.add_argument("--device", help="Device (cpu, cuda, mps)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold")
    
    args = parser.parse_args()
    
    anon = Anonymizer(args.model, device=args.device, prob_threshold=args.threshold)
    
    if args.text:
        print(f"Original: {args.text}")
        print(f"Anonymized: {anon.anonymize(args.text)}")
    elif args.file:
        out_lines = []
        with open(args.file, 'r') as f:
            for line in f:
                # Preserve line breaks; do not drop empty lines
                raw = line.rstrip("\n")
                if raw:
                    out_lines.append(anon.anonymize(raw))
                else:
                    out_lines.append("")

        if args.output:
            with open(args.output, 'w') as fout:
                for i, l in enumerate(out_lines):
                    fout.write(l)
                    if i != len(out_lines) - 1:
                        fout.write("\n")
        else:
            for l in out_lines:
                print(l)
    else:
        print("Please provide --text or --file")

if __name__ == "__main__":
    main()
