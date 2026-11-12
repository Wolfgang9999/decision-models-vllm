"""Run bench/gpu_bench.sh on Modal GPUs inside the published image and download the results.

BENCH_IMAGE_TAG=hopper-fa3 modal run bench/modal_bench.py --gpu H100 --arms "T T3 FI F4 BARE" --branch hopper-fa3
BENCH_IMAGE_TAG=hopper-fa3 modal run bench/modal_bench.py --gpu B200 --arms "T FI F4 BARE" --branch hopper-fa3

BENCH_IMAGE_TAG picks the image (default latest); --branch picks where the bench script and inputs are fetched from;
--inputs-dir DIR sends DIR/states.jsonl and DIR/payloads.jsonl instead of the repo's public inputs.

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
    # The vLLM image ships python3 only; Modal looks for `python` to detect the version.
    return (modal.Image.from_registry(f"ghcr.io/wolfgang9999/decision-models-vllm:{tag}")
            .dockerfile_commands(["RUN ln -sf $(command -v python3) /usr/local/bin/python"])
            .entrypoint([]))


def run(model, arms, branch, grid, inputs=None):
    env = dict(os.environ, MODEL=model, ARMS=arms, BRANCH=branch, OUT="/tmp/results/out", RESULTS_PORT="0")
    if inputs:  # private inputs sent from the caller instead of the repo's public ones
        Path("/tmp/inputs").mkdir(parents=True, exist_ok=True)
        for name, data in inputs.items():
            Path("/tmp/inputs", name).write_bytes(data)
        env["INPUTS_DIR"] = "/tmp/inputs"
    if grid:
        env["GRID"] = grid
    script = subprocess.run(["curl", "-fsSL", f"{REPO}/{branch}/bench/gpu_bench.sh"], capture_output=True, check=True)
    subprocess.run(["bash", "-c", script.stdout.decode()], env=env, check=False)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add("/tmp/results/out", arcname=".")
    return buf.getvalue()


TAG = os.environ.get("BENCH_IMAGE_TAG", "latest")
IMAGE = image(TAG)


@app.function(image=IMAGE, gpu="H100", timeout=4 * 3600)
def bench_h100(model, arms, branch, grid, inputs=None):
    return run(model, arms, branch, grid, inputs)


@app.function(image=IMAGE, gpu="B200", timeout=4 * 3600)
def bench_b200(model, arms, branch, grid, inputs=None):
    return run(model, arms, branch, grid, inputs)


FN = {"H100": bench_h100, "B200": bench_b200}


@app.local_entrypoint()
def main(gpu: str = "H100", arms: str = "T FI F4 BARE", model: str = "baselabs/joey-1", branch: str = "main",
         grid: str = "", inputs_dir: str = ""):
    inputs = None
    if inputs_dir:
        inputs = {n: (Path(inputs_dir) / n).read_bytes() for n in ("states.jsonl", "payloads.jsonl")}
    data = FN[gpu].remote(model, arms, branch, grid, inputs)
    out = Path(__file__).parent / "results" / f"{gpu.lower()}-{model.split('/')[-1]}-{arms.replace(' ', '')}"
    out.mkdir(parents=True, exist_ok=True)
    tarfile.open(fileobj=io.BytesIO(data), mode="r:gz").extractall(out, filter="data")
    print(f"results in {out}")
    print((out / "report.md").read_text() if (out / "report.md").exists() else "(no report)")
