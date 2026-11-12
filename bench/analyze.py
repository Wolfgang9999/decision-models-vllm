"""Summarise grid_bench results: throughput/latency tables and agreement with the OneJev reference arm.

python analyze.py RESULTS_DIR [RESULTS_DIR ...] [--ref A] [--out report.md]

Each RESULTS_DIR holds summary.jsonl and raw_<ARM>.jsonl as written by grid_bench.py. Several dirs can be passed
(e.g. the notebook results and results_deployed/joey-1); arms are merged by name.

Throughput: for every (arm, bucket, questions) the peak decisions/s over the concurrency ladder, the concurrency where
it happened, p50/p90 there, and c1 p50 (service time). Only valid windows count: invalid rows (e.g. arm A round 0 ref,
whose server was killed) are dropped with --drop "A:0:ref".
Agreement: requests are matched on their key with the round stripped (same bucket, questions, concurrency, index), so
inputs are byte-identical when both arms ran the same round, and differ only in the nonce's round digit otherwise.
A decision flips when the top option changes (choice/score argmax, noul at 0.5).
"""
import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


def load(dirs, drop):
    summ, raw = [], defaultdict(dict)
    for d in map(Path, dirs):
        for line in (d / "summary.jsonl").read_text().splitlines():
            s = json.loads(line)
            if f"{s['arm']}:{s['round']}:{s['bucket']}" in drop:
                continue
            summ.append(s)
        for f in d.glob("raw_*.jsonl"):
            arm = f.stem[4:]
            for line in f.read_text().splitlines():
                r = json.loads(line)
                if not r.get("valid") or not r.get("answers"):
                    continue
                key = re.sub(r"-r\d+-", "-", r["key"]) + (f"#p{r['part']}" if r.get("part") is not None else "")
                bucket = key.split("-")[1] if key.startswith("split-") else key.split("-")[0]
                rnd = re.search(r"-r(\d+)-", r["key"]).group(1)
                if f"{arm}:{rnd}:{bucket}" in drop:
                    continue
                raw[arm][key] = r["answers"]
    return summ, raw


def top(v):
    return (v >= 0.5) if isinstance(v, (int, float)) else max(v, key=v.get)


def delta(a, b):
    if isinstance(a, (int, float)):
        return abs(a - b)
    return max(abs(a[o] - b.get(o, 0.0)) for o in a)


def agreement(ref, other):
    flips = n = 0
    ds = []
    for key, qa in other.items():
        qr = ref.get(key)
        if not qr:
            continue
        for q, a in qa.items():
            if q not in qr or a is None or qr[q] is None:
                continue
            n += 1
            flips += top(a) != top(qr[q])
            ds.append(delta(a, qr[q]))
    if not n:
        return None
    ds.sort()
    return {"decisions": n, "flips": flips, "flip_pct": 100 * flips / n, "mean": statistics.mean(ds),
            "p99": ds[int(0.99 * (len(ds) - 1))], "max": ds[-1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--ref", default="A")
    ap.add_argument("--drop", nargs="*", default=["A:0:ref"])
    ap.add_argument("--out")
    a = ap.parse_args()
    summ, raw = load(a.dirs, set(a.drop))
    out = []
    grid = [s for s in summ if s.get("scenario") is None]
    cells = defaultdict(list)
    for s in grid:
        cells[(s["bucket"], s["questions"], s["arm"])].append(s)
    arms = sorted({s["arm"] for s in grid})
    out.append("## Peak decisions/s (concurrency, p50 / p90 s at the peak; c1 p50 s)\n")
    out.append("| state | q | " + " | ".join(arms) + " |")
    out.append("|---|---|" + "---|" * len(arms))
    for bucket in ["500", "1000", "2000", "4000", "ref"]:
        for q in sorted({k[1] for k in cells}):
            row = []
            for arm in arms:
                rows = cells.get((bucket, q, arm))
                if not rows:
                    row.append("")
                    continue
                best = max(rows, key=lambda s: s["decisions_per_s"])
                c1 = next((s for s in rows if s["concurrency"] == 1), None)
                row.append(f"**{best['decisions_per_s']:.1f}** (c{best['concurrency']}, {best['p50_s']} / {best['p90_s']}; "
                           f"c1 {c1['p50_s'] if c1 else '-'})")
            if any(row):
                out.append(f"| {bucket} | {q} | " + " | ".join(row) + " |")
    split = [s for s in summ if s.get("scenario")]
    if split:
        out.append("\n## Same state across 3 requests (4, 4, 3 questions): peak decisions/s (c; p50 of request 1 -> 3)\n")
        sp = defaultdict(list)
        for s in split:
            sp[(s["bucket"], s["arm"])].append(s)
        sarms = sorted({s["arm"] for s in split})
        out.append("| state | " + " | ".join(sarms) + " |")
        out.append("|---|" + "---|" * len(sarms))
        for bucket in ["500", "1000", "2000", "4000", "ref"]:
            row = []
            for arm in sarms:
                rows = sp.get((bucket, arm))
                if not rows:
                    row.append("")
                    continue
                b = max(rows, key=lambda s: s["decisions_per_s"])
                row.append(f"**{b['decisions_per_s']:.1f}** (c{b['concurrency']}; {b.get('p50_req1_s')} -> {b.get('p50_req3_s')})")
            if any(row):
                out.append(f"| {bucket} | " + " | ".join(row) + " |")
    if a.ref in raw:
        out.append(f"\n## Agreement with arm {a.ref} (matched requests)\n")
        out.append("| arm | decisions compared | flips | flip % | mean abs prob diff | p99 | max |")
        out.append("|---|---|---|---|---|---|---|")
        for arm in sorted(raw):
            if arm == a.ref:
                continue
            g = agreement(raw[a.ref], raw[arm])
            if g:
                out.append(f"| {arm} | {g['decisions']} | {g['flips']} | {g['flip_pct']:.2f} | {g['mean']:.4f} | "
                           f"{g['p99']:.4f} | {g['max']:.4f} |")
    text = "\n".join(out)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
