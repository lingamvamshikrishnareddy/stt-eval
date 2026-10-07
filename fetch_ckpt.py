#!/usr/bin/env python3
"""Download the model file of training checkpoints from a HF repo as <out>/step_<N>.pt.

Finds the file by name, so the repo layout does not need to be known: any path containing 'step_<N>/' and ending in
'model/pp_00/tp_00/sdp_00.pt' (or a single 'step_<N>*.pt').

  python fetch_ckpt.py --repo guruawe/octopus-asr-omniASR-1B-run3 --steps 12000 20000 --out /workspace/octo/ckpts
"""
from __future__ import annotations
import argparse, os, shutil, sys
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download


def find(files, step):
    hits = [f for f in files if f"step_{step}/" in f and f.endswith("model/pp_00/tp_00/sdp_00.pt")]
    if not hits:
        hits = [f for f in files if Path(f).name.startswith(f"step_{step}") and f.endswith(".pt")]
    if len(hits) != 1:
        listing = "\n  ".join(f for f in files if f"step_{step}" in f) or "(nothing with that step)"
        sys.exit(f"step_{step}: expected one model file, found {len(hits)}:\n  {listing}")
    return hits[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--steps", nargs="+", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--repo_type", default="model")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    files = HfApi().list_repo_files(a.repo, repo_type=a.repo_type)
    for step in a.steps:
        dst = out / f"step_{step}.pt"
        if dst.exists():
            print(f"step_{step}: present ({dst.stat().st_size / 1e9:.2f} GB)"); continue
        src = find(files, step)
        print(f"step_{step}: downloading {src}", flush=True)
        dl = out / "_dl"
        p = hf_hub_download(a.repo, src, repo_type=a.repo_type, local_dir=dl)
        os.replace(p, dst)
        shutil.rmtree(dl, ignore_errors=True)
        print(f"step_{step}: {dst} ({dst.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
