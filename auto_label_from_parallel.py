#!/usr/bin/env python3
"""
Heurystyczne auto-etykietowanie par (orig.txt, anonymized.txt).
Zakłada, że w anonymized.txt realne dane zastąpiono placeholderami typu {name}, {pesel} itp.
Dla każdej linii:
- jeśli linie są identyczne lub brak placeholderów, pomijamy
- dopasowujemy placeholdery do fragmentów, które różnią się między anonymized i orig
- wynik: auto_labels.jsonl z polami: text (oryginał), entities (start,end,label), meta
"""
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Dict

SRC_ORIG = Path("info/orig.txt")
SRC_ANON = Path("info/anonymized.txt")
OUT = Path("auto_labels.jsonl")
PLACEHOLDER_RE = re.compile(r"\{([^}]+)\}|\[([^\]]+)\]")


def find_entities(orig: str, anon: str) -> List[Dict]:
    placeholders = list(PLACEHOLDER_RE.finditer(anon))
    if not placeholders:
        return []
    sm = SequenceMatcher(None, anon, orig)
    matches = sm.get_matching_blocks()
    entities = []
    prev_end_a = 0
    prev_end_o = 0
    m_idx = 0
    for ph in placeholders:
        ph_start, ph_end = ph.span()
        # znajdź najbliższy matching block zaczynający się po placeholderze
        while m_idx < len(matches) and matches[m_idx].a < ph_end:
            m_idx += 1
        # matching block po placeholderze (anchor)
        if m_idx < len(matches):
            anchor = matches[m_idx]
            anchor_a_start = anchor.a
            anchor_o_start = anchor.b
        else:
            anchor_a_start = len(anon)
            anchor_o_start = len(orig)
        # fragment w orig odpowiadający placeholderowi: między prev_end_o a anchor_o_start - (anchor_a_start - ph_end)
        # oszacuj długość różnicy po stronie anon
        delta = anchor_a_start - ph_end
        cand_end_o = max(prev_end_o, anchor_o_start - max(delta, 0))
        # weź substring od prev_end_o do cand_end_o
        if cand_end_o > prev_end_o:
            span_start = prev_end_o
            span_end = cand_end_o
            label = ph.group(1) or ph.group(2)
            entities.append({"start": span_start, "end": span_end, "label": label})
            prev_end_o = span_end
        prev_end_a = ph_end
    return entities


def main():
    if not SRC_ORIG.exists() or not SRC_ANON.exists():
        raise FileNotFoundError("orig.txt lub anonymized.txt nie istnieje")
    orig_lines = SRC_ORIG.read_text().splitlines()
    anon_lines = SRC_ANON.read_text().splitlines()
    if len(orig_lines) != len(anon_lines):
        print(f"Ostrzeżenie: różne długości plików ({len(orig_lines)} vs {len(anon_lines)})")
    total = min(len(orig_lines), len(anon_lines))
    out = []
    for i in range(total):
        o = orig_lines[i]
        a = anon_lines[i]
        if o == a:
            continue
        ents = find_entities(o, a)
        if not ents:
            continue
        out.append({"text": o, "entities": ents, "meta": {"source": "auto"}})
    OUT.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out))
    print(f"Zapisano {len(out)} przykładów do {OUT}")


if __name__ == "__main__":
    main()
