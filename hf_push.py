#!/usr/bin/env python3
"""Upload the final model: <model> as model.pt, plus everything in <final> (README.md model card, card yaml, results/).

  python hf_push.py --repo guruawe/octopus-asr-omniASR-1B-v1 --model blends/s12000_a0.7.pt --final final --private 1
"""
from __future__ import annotations
import argparse, os
from huggingface_hub import HfApi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--model", required=True); ap.add_argument("--final", required=True)
    ap.add_argument("--private", type=int, default=1)
    a = ap.parse_args()
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(a.repo, repo_type="model", private=bool(a.private), exist_ok=True)
    print(f"uploading {a.model} ({os.path.getsize(a.model) / 1e9:.2f} GB) -> {a.repo}/model.pt", flush=True)
    api.upload_file(path_or_fileobj=a.model, path_in_repo="model.pt", repo_id=a.repo,
                    commit_message="Add blended model weights")
    api.upload_folder(folder_path=a.final, repo_id=a.repo, commit_message="Add model card and FLEURS results")
    print(f"done: https://huggingface.co/{a.repo}" + (" (private)" if a.private else ""))


if __name__ == "__main__":
    main()
