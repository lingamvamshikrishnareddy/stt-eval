#!/usr/bin/env python3
"""Save a base model card's weights as a plain state-dict .pt, the --base-ckpt input of 'stt_blend.py --memory-efficient'.
Same tensors stt_blend.py uses in its in-RAM path (fairseq2 load_model(...).state_dict(), float32).

  python export_base.py --card omniASR_LLM_1B_v2 --out ckpts/base.pt
"""
from __future__ import annotations
import argparse, os
from pathlib import Path
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", default="omniASR_LLM_1B_v2"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import omnilingual_asr.models.inference.pipeline  # noqa: F401  (registers the omnilingual model family)
    from fairseq2.models.hub import load_model
    sd = load_model(a.card, device=torch.device("cpu"), dtype=torch.float32).state_dict()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.pt")
    torch.save({k: v.contiguous() for k, v in sd.items()}, tmp)
    os.replace(tmp, out)
    print(f"wrote {out}: {len(sd)} tensors, {out.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
