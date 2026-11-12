"""Apply the overlay (open vLLM PRs, see README) to an installed vLLM built from the pinned commit.

python install_overlay.py [--force]

Every file the overlay replaces must match the pinned commit's copy (SHA256 in overlay/manifest.json), so the overlay
never lands on a different vLLM silently. --force skips that check (use only with a vLLM close to the pin).
"""
import argparse
import compileall
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OVERLAY = HERE / "overlay"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        sys.exit("vllm is not installed in this Python")
    target = Path(list(spec.submodule_search_locations)[0]).parent
    manifest = json.loads((OVERLAY / "manifest.json").read_text())
    bad = []
    for rel, sha in manifest["base_sha256"].items():
        dst = target / rel
        if sha is None:
            continue
        if not dst.is_file() or hashlib.sha256(dst.read_bytes()).hexdigest() != sha:
            bad.append(rel)
    if bad and not args.force:
        sys.exit(f"installed vllm does not match commit {manifest['vllm_commit'][:12]} in: {', '.join(bad)} "
                 "(install the pinned build, or pass --force)")
    for rel in manifest["base_sha256"]:
        shutil.copyfile(OVERLAY / rel, target / rel)
    if not compileall.compile_dir(target / "vllm", quiet=1):
        sys.exit("overlay did not compile")
    print(f"overlay applied to {target / 'vllm'} ({len(manifest['base_sha256'])} files, "
          f"vLLM {manifest['vllm_commit'][:12]}{', forced' if bad else ''})")


if __name__ == "__main__":
    main()
