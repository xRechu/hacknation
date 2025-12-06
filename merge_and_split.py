#!/usr/bin/env python3
"""Merge synthetic.jsonl with auto-labeled data (if available) and split into train/val/test.
- Expects synthetic.jsonl in the current directory.
- If auto_labels.jsonl exists, it will be included; otherwise falls back to synthetic only.
Output: train.jsonl, val.jsonl, test.jsonl (80/10/10 split).
"""
import json
import random
from pathlib import Path

random.seed(42)

SYN_PATH = Path("synthetic.jsonl")
AUTO_PATH = Path("auto_labels.jsonl")  # adjust if your auto labels are under a different name


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save_jsonl(path: Path, data):
    path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in data))


def main():
    if not SYN_PATH.exists():
        raise FileNotFoundError(f"Brak pliku {SYN_PATH}")

    synthetic = load_jsonl(SYN_PATH)
    sources = [synthetic]
    if AUTO_PATH.exists():
        auto = load_jsonl(AUTO_PATH)
        sources.append(auto)
        print(f"Wczytano auto-labeled: {len(auto)} przykładów")
    else:
        print("Brak auto_labels.jsonl - używam tylko synthetic.jsonl")

    all_data = [item for src in sources for item in src]
    random.shuffle(all_data)

    n = len(all_data)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    train = all_data[:n_train]
    val = all_data[n_train:n_train + n_val]
    test = all_data[n_train + n_val:]

    save_jsonl(Path("train.jsonl"), train)
    save_jsonl(Path("val.jsonl"), val)
    save_jsonl(Path("test.jsonl"), test)

    print(f"train={len(train)}, val={len(val)}, test={len(test)}, total={n}")


if __name__ == "__main__":
    main()
