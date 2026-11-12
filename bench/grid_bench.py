"""State size x questions per state x concurrency grid against one local decision server.

python -I grid_bench.py URL OUT_DIR ARM [--round N] [--buckets 500 2000 4000 ref] [--qs 1 11 44]

Method (Confluence 6439698741 / 6432064407): closed loop; per window, c requests in flight for a 20 s launch window,
then up to 60 s to drain, unfinished requests count as timeouts. A ladder (1 4 8 16 32 64) stops after two levels
without an RPS gain, or when errors + timeouts exceed 20% of launched. No retries.
Every request carries its own nonce, derived from (bucket, questions, concurrency, round, index), so it is cold, and
every server answers byte-identical requests: agreement against OneJev is computed per request key afterwards.
After the grid, the split scenario sends each state as 3 sequential requests of 4, 4 and 3 questions with the state
byte-identical, which only a cross-request prefix cache can speed up (p50 per request position is reported).
"""
import argparse, asyncio, json, math, os, time
from pathlib import Path

import httpx

# Inputs (not shipped; our measurements used internal data):
#   BENCH_STATES:   JSON lines {"bucket": "500"|"2000"|"4000"|"ref", "case_id", "conversation", "documents"}
#   BENCH_PAYLOADS: JSON lines {"case_id", "body": {"questions": {...}}} giving each case's own questions
STATES = Path(os.environ.get("BENCH_STATES", "states.jsonl"))
PAYLOADS = Path(os.environ.get("BENCH_PAYLOADS", "payloads.jsonl"))
LEVELS = [1, 4, 8, 16, 32, 64]
LAUNCH, DRAIN = 20.0, 60.0
MODEL = os.environ.get("BENCH_MODEL", "joey-1")


def load():
    qdefs = {p["case_id"]: p["body"]["questions"] for p in map(json.loads, PAYLOADS.read_text().splitlines())}
    states = {}
    for r in map(json.loads, STATES.read_text().splitlines()):
        if r["case_id"] in qdefs:
            states.setdefault(r["bucket"], []).append(r)
    return states, qdefs


def make_body(cases, qdefs, k, Q, key):
    c = cases[k % len(cases)]
    own = qdefs[c["case_id"]]
    keys = list(own)
    if Q <= len(keys):
        rot = keys[k % len(keys):] + keys[:k % len(keys)]
        qs = {q: own[q] for q in rot[:Q]}
    else:  # 44 = own 11 + 33 copies (same construction as the size sweep)
        qs = dict(own)
        for j in (1, 2, 3):
            for q in keys:
                qs[f"x{j}_{q}"] = own[q]
    return {"model": MODEL, "state": {"run_nonce": key, "conversation": c["conversation"], "documents": c["documents"]},
            "questions": qs}


def valid(body, resp):
    answers = (resp or {}).get("answers") or {}
    if set(answers) != set(body["questions"]):
        return False
    return all(answers[q].get("type") == body["questions"][q]["type"] for q in answers)


def compact(answers):
    """Per question: option probabilities (choice/score) or P(yes) (noul)."""
    out = {}
    for q, a in answers.items():
        out[q] = a.get("probabilities") if a.get("type") in ("choice", "score") else a.get("noul")
    return out


async def window(client, url, cases, qdefs, conc, Q, cell, raw):
    sem = asyncio.Semaphore(conc)
    recs, tasks = [], set()
    t0 = time.perf_counter()

    async def one(k):
        key = f"{cell}-{k}"
        body = make_body(cases, qdefs, k, Q, key)
        t = time.perf_counter()
        rec = {"key": key, "k": k}
        try:
            r = await client.post(url + "/v1/systemone", json=body)
            rec["status"] = r.status_code
            resp = r.json() if r.status_code == 200 else None
            rec["valid"] = r.status_code == 200 and valid(body, resp)
            rec["kind"] = "ok" if rec["valid"] else (f"http_{r.status_code}" if r.status_code != 200 else "invalid")
            if rec["valid"]:
                rec["answers"] = compact(resp["answers"])
            elif r.status_code != 200:
                rec["text"] = r.text[:300]
        except asyncio.CancelledError:
            rec.update(valid=False, kind="window_cut")
        except httpx.TimeoutException:
            rec.update(valid=False, kind="timeout")
        except Exception as e:
            rec.update(valid=False, kind=f"exc:{type(e).__name__}")
        finally:
            sem.release()
        rec["elapsed"] = time.perf_counter() - t
        rec["t_end"] = time.perf_counter() - t0
        rec["decisions"] = len(body["questions"]) if rec["valid"] else 0
        recs.append(rec)

    k = 0
    while time.perf_counter() - t0 < LAUNCH:
        await sem.acquire()
        if time.perf_counter() - t0 >= LAUNCH or sum(r["kind"] == "exc:ConnectError" for r in recs) >= 20:
            sem.release()
            break
        tasks.add(asyncio.create_task(one(k)))
        k += 1
    done, pending = await asyncio.wait(tasks, timeout=max(0.0, LAUNCH + DRAIN - (time.perf_counter() - t0)))
    for t in pending:
        t.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    makespan = (time.perf_counter() - t0) if pending else max([r["t_end"] for r in recs] + [1e-9])
    ok = [r for r in recs if r["valid"]]
    lat = sorted(r["elapsed"] for r in ok)
    pct = lambda p: round(lat[min(len(lat) - 1, math.ceil(len(lat) * p) - 1)], 3) if lat else None
    kinds = {}
    for r in recs:
        if not r["valid"]:
            kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    for r in recs:
        raw.write(json.dumps({"cell": cell, **{x: r.get(x) for x in ("key", "valid", "kind", "elapsed", "answers", "text")}},
                             ensure_ascii=False) + "\n")
    raw.flush()
    timeouts = kinds.get("timeout", 0) + kinds.get("window_cut", 0)
    return {"launched": k, "valid_requests": len(ok), "decisions": sum(r["decisions"] for r in ok),
            "errors": sum(kinds.values()) - timeouts, "timeouts": timeouts, "error_kinds": kinds,
            "makespan_s": round(makespan, 2), "rps": round(len(ok) / makespan, 3),
            "decisions_per_s": round(sum(r["decisions"] for r in ok) / makespan, 3),
            "p50_s": pct(.5), "p90_s": pct(.9), "p95_s": pct(.95)}


SPLIT = (4, 4, 3)  # one state asked in 3 sequential requests, state byte-identical


async def split_window(client, url, cases, qdefs, conc, cell, raw):
    """c sessions in flight; a session sends one state as 3 sequential requests of 4, 4 and 3 of its questions."""
    sem = asyncio.Semaphore(conc)
    recs, tasks = [], set()
    t0 = time.perf_counter()

    async def session(k):
        key = f"{cell}-{k}"
        c = cases[k % len(cases)]
        own = qdefs[c["case_id"]]
        keys, start = list(own), 0
        try:
            for part, n in enumerate(SPLIT):
                qs = {q: own[q] for q in keys[start:start + n]}
                start += n
                body = {"model": MODEL, "questions": qs,
                        "state": {"run_nonce": key, "conversation": c["conversation"], "documents": c["documents"]}}
                t = time.perf_counter()
                rec = {"key": key, "part": part}
                try:
                    r = await client.post(url + "/v1/systemone", json=body)
                    resp = r.json() if r.status_code == 200 else None
                    rec["valid"] = r.status_code == 200 and valid(body, resp)
                    rec["kind"] = "ok" if rec["valid"] else (f"http_{r.status_code}" if r.status_code != 200 else "invalid")
                    if rec["valid"]:
                        rec["answers"] = compact(resp["answers"])
                except httpx.TimeoutException:
                    rec.update(valid=False, kind="timeout")
                except Exception as e:
                    rec.update(valid=False, kind=f"exc:{type(e).__name__}")
                rec["elapsed"] = time.perf_counter() - t
                rec["t_end"] = time.perf_counter() - t0
                rec["decisions"] = len(qs) if rec["valid"] else 0
                recs.append(rec)
        except asyncio.CancelledError:
            recs.append({"key": key, "part": -1, "valid": False, "kind": "window_cut", "elapsed": 0.0,
                         "t_end": time.perf_counter() - t0, "decisions": 0})
        finally:
            sem.release()

    k = 0
    while time.perf_counter() - t0 < LAUNCH:
        await sem.acquire()
        if time.perf_counter() - t0 >= LAUNCH:
            sem.release()
            break
        tasks.add(asyncio.create_task(session(k)))
        k += 1
    done, pending = await asyncio.wait(tasks, timeout=max(0.0, LAUNCH + DRAIN - (time.perf_counter() - t0)))
    for t in pending:
        t.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    makespan = (time.perf_counter() - t0) if pending else max([r["t_end"] for r in recs] + [1e-9])
    for r in recs:
        raw.write(json.dumps({"cell": cell, **{x: r.get(x) for x in ("key", "part", "valid", "kind", "elapsed", "answers")}},
                             ensure_ascii=False) + "\n")
    raw.flush()
    ok = [r for r in recs if r["valid"]]
    by_part = {}
    for part in range(len(SPLIT)):
        lat = sorted(r["elapsed"] for r in ok if r["part"] == part)
        by_part[f"p50_req{part + 1}_s"] = round(lat[len(lat) // 2], 3) if lat else None
    bad = [r for r in recs if not r["valid"]]
    timeouts = sum(1 for r in bad if r["kind"] in ("timeout", "window_cut"))
    return {"sessions_launched": k, "valid_requests": len(ok), "decisions": sum(r["decisions"] for r in ok),
            "errors": len(bad) - timeouts, "timeouts": timeouts, "makespan_s": round(makespan, 2),
            "decisions_per_s": round(sum(r["decisions"] for r in ok) / makespan, 3), **by_part}


async def main(a):
    states, qdefs = load()
    a.out.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=httpx.Timeout(LAUNCH + DRAIN + 30, connect=20),
                                 limits=httpx.Limits(max_connections=200, max_keepalive_connections=200)) as client:
        warm = make_body(states["500"], qdefs, 0, 11, f"warm-{a.arm}-{time.time()}")
        r = await client.post(a.url + "/v1/systemone", json=warm)
        if r.status_code != 200 or not valid(warm, r.json()):
            raise SystemExit(f"warm-up failed {r.status_code}: {r.text[:400]}")
        with (a.out / f"raw_{a.arm}.jsonl").open("a") as raw, (a.out / "summary.jsonl").open("a") as summ:
            for bucket in (a.buckets if a.grid else []):
                for Q in a.qs:
                    best, flat = 0.0, 0
                    for conc in LEVELS:
                        cell = f"{bucket}-q{Q}-c{conc}-r{a.round}"
                        s = await window(client, a.url, states[bucket], qdefs, conc, Q, cell, raw)
                        s = {"arm": a.arm, "round": a.round, "bucket": bucket, "questions": Q, "concurrency": conc, **s,
                             "end_unix": time.time()}
                        summ.write(json.dumps(s) + "\n"); summ.flush()
                        print(json.dumps(s), flush=True)
                        if s["error_kinds"].get("exc:ConnectError", 0) and not s["valid_requests"]:
                            raise SystemExit(f"server unreachable in {cell}; aborting this arm")
                        if (s["errors"] + s["timeouts"]) > 0.2 * max(1, s["launched"]):
                            break
                        flat = flat + 1 if s["rps"] <= best else 0
                        best = max(best, s["rps"])
                        if flat >= 2:
                            break
            if a.split:
                for bucket in a.buckets:
                    best, flat = 0.0, 0
                    for conc in LEVELS:
                        cell = f"split-{bucket}-c{conc}-r{a.round}"
                        s = await split_window(client, a.url, states[bucket], qdefs, conc, cell, raw)
                        s = {"arm": a.arm, "round": a.round, "scenario": "split_4_4_3", "bucket": bucket,
                             "questions": sum(SPLIT), "concurrency": conc, **s, "end_unix": time.time()}
                        summ.write(json.dumps(s) + "\n"); summ.flush()
                        print(json.dumps(s), flush=True)
                        if (s["errors"] + s["timeouts"]) > 0.2 * max(1, s["valid_requests"] + s["errors"] + s["timeouts"]):
                            break
                        flat = flat + 1 if s["decisions_per_s"] <= best else 0
                        best = max(best, s["decisions_per_s"])
                        if flat >= 2:
                            break


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("url"); ap.add_argument("out", type=Path); ap.add_argument("arm")
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--buckets", nargs="+", default=["500", "2000", "4000", "ref"])
    ap.add_argument("--qs", type=int, nargs="+", default=[1, 11, 44])
    ap.add_argument("--levels", type=int, nargs="+", help="override the concurrency ladder (e.g. 4 for a correctness pass)")
    ap.add_argument("--no-grid", dest="grid", action="store_false", help="skip the size x questions grid")
    ap.add_argument("--no-split", dest="split", action="store_false", help="skip the 4/4/3 same-state scenario")
    a = ap.parse_args()
    a.url = a.url.rstrip("/")
    if a.levels:
        LEVELS[:] = a.levels
    asyncio.run(main(a))
