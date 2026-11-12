"""Run bench/gpu_bench.sh on Modal GPUs inside the published image and download the results.

BENCH_IMAGE_TAG=hopper-fa3 modal run bench/modal_bench.py --gpu H100 --arms "T T3 FI F4 BARE" --branch hopper-fa3
BENCH_IMAGE_TAG=hopper-fa3 modal run bench/modal_bench.py --gpu B200 --arms "T FI F4 BARE" --branch hopper-fa3

BENCH_IMAGE_TAG picks the image (default latest); --branch picks where the bench script and inputs are fetched from.

Results land in bench/results/<gpu>-<model>/ (summary.jsonl, raw_<arm>.jsonl, serve logs, report.md).
"""
import io
import os
import subprocess
import tarfile
from pathlib import Path

import modal

REPO = "https://raw.githubusercontent.com/Wolfgang9999/decision-models-vllm"
app = modal.App("decision-models-vllm-bench")


def image(tag):
    return modal.Image.from_registry(f"ghcr.io/wolfgang9999/decision-models-vllm:{tag}").entrypoint([])


def run(model, arms, branch, grid):
    env = dict(os.environ, MODEL=model, ARMS=arms, BRANCH=branch, OUT="/tmp/results/out", RESULTS_PORT="0")
    if grid:
        env["GRID"] = grid
    script = subprocess.run(["curl", "-fsSL", f"{REPO}/{branch}/bench/gpu_bench.sh"], capture_output=True, check=True)
    subprocess.run(["bash", "-c", script.stdout.decode()], env=env, check=False)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add("/tmp/results/out", arcname=".")
    return buf.getvalue()


TAG = os.environ.get("BENCH_IMAGE_TAG", "latest")
FN = {gpu: app.function(image=image(TAG), gpu=gpu, timeout=4 * 3600, serialized=True, name=f"bench_{gpu.lower()}")(run)
      for gpu in ("H100", "B200")}


@app.local_entrypoint()
def main(gpu: str = "H100", arms: str = "T FI F4 BARE", model: str = "baselabs/joey-1", branch: str = "main",
         grid: str = ""):
    data = FN[gpu].remote(model, arms, branch, grid)
    out = Path(__file__).parent / "results" / f"{gpu.lower()}-{model.split('/')[-1]}"
    out.mkdir(parents=True, exist_ok=True)
    tarfile.open(fileobj=io.BytesIO(data), mode="r:gz").extractall(out, filter="data")
    print(f"results in {out}")
    print((out / "report.md").read_text() if (out / "report.md").exists() else "(no report)")
