#!/usr/bin/env python3
"""Average fine-tuned checkpoints and blend them with the base model weights (weight-space interpolation, WiSE-FT).

  python stt_blend.py --ckpts run_r2/ws_1.x/checkpoints/step_6000 .../step_8000 .../step_10000 --alpha 0.7 --out blends/avg3_a07.pt

alpha = weight of the (averaged) fine-tuned weights: 1.0 = averaged checkpoints only, 0.0 = the base model.
The output keeps the structure of the fine-tuned model file (model/pp_00/tp_00/sdp_00.pt), so
'stt_fleurs_eval.py score --ckpt <out.pt>' loads it like any checkpoint. Floating-point tensors are blended, others copied.
"""
from __future__ import annotations
import argparse
import mmap
import os
import shutil
import tempfile
from pathlib import Path
import torch


def model_file(p) -> Path:
    p = Path(p)
    return p if p.suffix == ".pt" else p / "model/pp_00/tp_00/sdp_00.pt"


def tensors(obj) -> dict:
    """The dict that holds the model tensors inside a checkpoint object (returned by reference)."""
    if isinstance(obj, dict) and obj and all(torch.is_tensor(v) for v in obj.values()):
        return obj
    if isinstance(obj, dict):
        for k in ("model", "model_state_dict", "state_dict"):
            if isinstance(obj.get(k), dict):
                return tensors(obj[k])
    top = list(obj)[:10] if isinstance(obj, dict) else type(obj).__name__
    raise SystemExit(f"can't find the model tensors in this checkpoint (top level: {top})")


def blend_mapped(files, alpha, base_file, out):
    """Blend one tensor at a time using reclaimable file-backed storage.

    The shared mapping is a disposable copy, never an input checkpoint. Saving
    it again produces a normal torch archive with valid CRCs and metadata.
    Separate read-only input mappings also keep tied weights from being blended
    twice when multiple state-dict keys refer to the same storage.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.resolve() in {Path(f).resolve() for f in files + ([base_file] if base_file else [])}:
        raise SystemExit("output must differ from all input checkpoints")
    states = [tensors(torch.load(f, map_location="cpu", weights_only=False, mmap=True)) for f in files]
    keys = set(states[0])
    if any(set(s) != keys for s in states[1:]):
        raise SystemExit("fine-tuned checkpoints have different tensor names")
    base = tensors(torch.load(base_file, map_location="cpu", weights_only=False, mmap=True)) if alpha < 1 else None
    if base is not None and keys - set(base):
        raise SystemExit(f"base checkpoint is missing tensors: {sorted(keys - set(base))[:3]}")
    for k in keys:
        refs = states + ([base] if base is not None else [])
        if any(s[k].shape != states[0][k].shape for s in refs):
            raise SystemExit(f"shape mismatch for {k}")
        if any(s[k].is_floating_point() != states[0][k].is_floating_point() for s in refs):
            raise SystemExit(f"incompatible tensor types for {k}")
    fd, work = tempfile.mkstemp(prefix="blend-work-", suffix=".pt", dir=out.parent)
    os.close(fd)
    fd, final_tmp = tempfile.mkstemp(prefix="blend-save-", suffix=".pt", dir=out.parent)
    os.close(fd)
    try:
        shutil.copyfile(files[0], work)
        with torch.serialization.set_default_mmap_options(mmap.MAP_SHARED):
            raw = torch.load(work, map_location="cpu", weights_only=False, mmap=True)
        dest = tensors(raw)
        with torch.inference_mode():
            for i, k in enumerate(dest, 1):
                if dest[k].is_floating_point():
                    acc = torch.zeros_like(states[0][k], dtype=torch.float64)
                    for s in states:
                        acc.add_(s[k].double(), alpha=1 / len(states))
                    acc.mul_(alpha)
                    if base is not None:
                        acc.add_(base[k].double(), alpha=1 - alpha)
                    dest[k].copy_(acc)
                    del acc
                if i % 100 == 0:
                    print(f"blended {i}/{len(dest)} tensors", flush=True)
        torch.save(raw, final_tmp)
        os.replace(final_tmp, out)
        print(f"wrote {out}: average of {len(files)} checkpoint(s), alpha {alpha} ({len(dest)} tensors)")
    finally:
        Path(work).unlink(missing_ok=True)
        Path(final_tmp).unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpts", nargs="+", required=True, help="checkpoint step dirs or model .pt files (averaged)")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--base", default="omniASR_LLM_1B_v2", help="base model card for blending")
    ap.add_argument("--out", required=True)
    ap.add_argument("--memory-efficient", action="store_true", help="mmap inputs and blend one tensor at a time")
    ap.add_argument("--base-ckpt", help="base .pt checkpoint in matching state-dict format; required for low-RAM blending")
    a = ap.parse_args()
    if not 0 <= a.alpha <= 1:
        ap.error("--alpha must be between 0 and 1")

    files = [model_file(c) for c in a.ckpts]
    if a.memory_efficient:
        if a.alpha < 1 and not a.base_ckpt:
            ap.error("--memory-efficient with alpha < 1 requires --base-ckpt")
        blend_mapped(files, a.alpha, model_file(a.base_ckpt) if a.base_ckpt else None, a.out)
        return
    if a.base_ckpt:
        ap.error("--base-ckpt is only supported with --memory-efficient")
    raw = torch.load(files[0], map_location="cpu", weights_only=False)
    sd = tensors(raw)
    keys = list(sd)
    acc = {k: sd[k].double() if sd[k].is_floating_point() else sd[k] for k in keys}
    for f in files[1:]:
        other = tensors(torch.load(f, map_location="cpu", weights_only=False))
        if list(other) != keys:
            raise SystemExit(f"{f} has different tensor names than {files[0]}")
        for k in keys:
            if acc[k].is_floating_point():
                acc[k] += other[k].double()
    n = len(files)
    for k in keys:
        if acc[k].is_floating_point():
            acc[k] /= n

    if a.alpha < 1.0:
        import omnilingual_asr.models.inference.pipeline  # noqa: F401  (registers the omnilingual model family)
        from fairseq2.models.hub import load_model
        base = load_model(a.base, device=torch.device("cpu"), dtype=torch.float32).state_dict()
        missing = [k for k in keys if k not in base]
        if missing:
            raise SystemExit(f"{len(missing)} checkpoint tensors not in the base model, e.g. {missing[:3]}; "
                             f"base has e.g. {list(base)[:3]}")
        bad = [k for k in keys if acc[k].is_floating_point() and acc[k].shape != base[k].shape]
        if bad:
            raise SystemExit(f"shape mismatch for {bad[:3]}")
        for k in keys:
            if acc[k].is_floating_point():
                acc[k] = a.alpha * acc[k] + (1 - a.alpha) * base[k].double()

    for k in keys:
        sd[k] = acc[k].to(sd[k].dtype) if acc[k].is_floating_point() else acc[k]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(raw, out)
    print(f"wrote {out}: average of {n} checkpoint(s), alpha {a.alpha} ({len(keys)} tensors)")


if __name__ == "__main__":
    main()
