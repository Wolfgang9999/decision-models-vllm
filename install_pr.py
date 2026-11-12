"""Install the pinned PR's Python package over its matching native base wheel,
then overlay the Triton GDN prefill-checkpoint patch (A100 prefix caching).

The PR does not change csrc/CMake or Python native bindings. Its Rust frontend
changes are irrelevant to the Python `vllm serve` entrypoint used by this image.
"""
import compileall
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request

PR_SHA = "af2ee49269522fc99a2ad4bba1dd9b6066ba4c2a"
NATIVE_SHA = "c9af9acc6ac22f6351e214eb0d7d9a7a26dfa870"
ARCHIVE_SHA256 = "cb316fe9c41e04313f12a60b4ecc39ed755edb303377566bde2734ee26c6f295"


def install():
    distribution = importlib.metadata.distribution("vllm")
    if "gc9af9acc6" not in distribution.version:
        raise RuntimeError(f"Wrong native wheel: {distribution.version}")
    destination = Path(distribution.locate_file("vllm"))
    native_before = {str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in destination.rglob("*.so")}
    if not native_before:
        raise RuntimeError("Native CUDA extensions missing from the base wheel")
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        archive = temporary / "source.tar.gz"
        with urllib.request.urlopen(
            f"https://codeload.github.com/vllm-project/vllm/tar.gz/{PR_SHA}", timeout=180
        ) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
            raise RuntimeError("Pinned source archive checksum mismatch")
        with tarfile.open(archive) as source:
            source.extractall(temporary, filter="data")
        package = temporary / f"vllm-{PR_SHA}" / "vllm"
        # Keep compiled extensions and package data from the matching native wheel.
        shutil.copytree(package, destination, dirs_exist_ok=True)
        for path in package.rglob("*.py"):
            installed = destination / path.relative_to(package)
            if path.read_bytes() != installed.read_bytes():
                raise RuntimeError(f"Source installation mismatch: {path}")
    overlay = Path("/opt/decision-build/overlay")
    overlay_hashes = {}
    for relative, base_sha in json.loads((overlay / "manifest.json").read_text()).items():
        installed = destination / relative
        # The overlay was cut from this exact PR file; refuse to stack it on anything else.
        if hashlib.sha256(installed.read_bytes()).hexdigest() != base_sha:
            raise RuntimeError(f"Overlay base mismatch: {relative}")
        shutil.copyfile(overlay / "vllm" / relative, installed)
        overlay_hashes[relative] = hashlib.sha256(installed.read_bytes()).hexdigest()
    native_after = {str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in destination.rglob("*.so")}
    if native_before != native_after:
        raise RuntimeError("Python overlay changed native extensions")
    for relative in ["model_executor/models/qwen3_5_slot.py",
                     "v1/core/sched/prefix_lm_scheduler.py",
                     "entrypoints/openai/decisions/slot_prompt.py"]:
        if not (destination / relative).is_file():
            raise RuntimeError(f"Missing PR implementation: {relative}")
    if not compileall.compile_dir(destination, quiet=1):
        raise RuntimeError("PR Python compilation failed")
    provenance = {"pr": 63114, "python_source_sha": PR_SHA,
                  "native_wheel_sha": NATIVE_SHA, "native_version": distribution.version,
                  "archive_sha256": ARCHIVE_SHA256, "frontend": "python",
                  "overlay": "triton GDN prefill checkpoints", "overlay_files": overlay_hashes,
                  "native_extensions": native_after}
    Path("/opt/decision-build/provenance.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == "__main__":
    install()
