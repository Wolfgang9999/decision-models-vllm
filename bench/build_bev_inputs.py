"""Benchmark inputs from the public BEV Decision Mix dataset (huggingface.co/datasets/avbiswas/bev-decision).

python build_bev_inputs.py OUT_DIR [--cases 64] [--seed 0]

The dataset has short states with 1-9 labelled questions each. A case concatenates dataset states (as numbered
documents) until it has about 13,400 tokens, and asks 11 of their questions (labels removed; kept in labels.jsonl).
Each case is cut to the four state sizes of the benchmark (about 500, 2,000, 4,000 and 13,200 tokens, at whitespace),
counted with joey-1's tokenizer. Writes states.jsonl, payloads.jsonl and labels.jsonl in the formats grid_bench.py
reads. The data is not redistributed here (the dataset's license is not stated); run this script to rebuild it.
"""
import argparse
import json
import random
from pathlib import Path

import pandas as pd
from tokenizers import Tokenizer

URL = "https://huggingface.co/datasets/avbiswas/bev-decision/resolve/main/data/test.parquet"
BUCKETS = {"500": 500, "2000": 2000, "4000": 4000, "ref": 13200}
PROMPT = "USER: Read the documents below and answer the questions about them.\n"
QUESTIONS = 11


def cut(tok, text, budget):
    words = text.split(" ")
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(tok.encode(" ".join(words[:mid])).ids) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return " ".join(words[:lo])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--cases", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_pretrained("baselabs/joey-1")
    prompt_tokens = len(tok.encode(PROMPT).ids)
    rows = pd.read_parquet(URL).sample(frac=1.0, random_state=a.seed).to_dict("records")
    states, payloads, labels, i = [], [], [], 0
    for c in range(a.cases):
        docs, questions, golds, n_tok = [], {}, {}, 0
        while n_tok < BUCKETS["ref"] + 200 and i < len(rows):
            row = rows[i]
            i += 1
            docs.append(f"Document {len(docs) + 1}:\n{row['state'].strip()}")
            n_tok += len(tok.encode(docs[-1]).ids) + 2
            for q in json.loads(row["questions_json"]).values():
                if len(questions) < QUESTIONS:
                    key = f"q{len(questions)}"
                    golds[key] = q.pop("label", None)
                    questions[key] = q
        if len(questions) < QUESTIONS:
            break
        text = "\n\n".join(docs)
        case = f"bev-{c:02d}"
        for bucket, total in BUCKETS.items():
            d = cut(tok, text, total - prompt_tokens)
            states.append({"bucket": bucket, "case_id": case, "conversation": PROMPT, "documents": d,
                           "tokens": prompt_tokens + len(tok.encode(d).ids)})
        payloads.append({"case_id": case, "question_count": QUESTIONS, "body": {"questions": questions}})
        labels.append({"case_id": case, "labels": golds})
        print(case, len(docs), "documents", flush=True)
    for name, data in (("states.jsonl", states), ("payloads.jsonl", payloads), ("labels.jsonl", labels)):
        (a.out / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data))
    print(len(payloads), "cases")


if __name__ == "__main__":
    main()
