#!/usr/bin/env python3
"""Pick the best candidate on FLEURS dev, then compare it with the base model on FLEURS test and write a model card.

Result dirs are <res>/<name>_<split>/summary.tsv, as written by 'stt_fleurs_eval.py score'.
Macro error = mean over languages of the per-language corpus error (WER; CER for zh/ja/th/my).

  python final_report.py pick   --res res --split dev                 # prints the winner's name on stdout
  python final_report.py report --res res --winner s12000_a0.7 --target 19.5 --out final ...
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from stt_common import OMNI2SHORT   # noqa: E402

WATCH = ["eng_Latn", "jpn_Jpan"]     # forgetting checks from run 3


def load(res: Path, name: str, split: str) -> pd.DataFrame:
    f = res / f"{name}_{split}" / "summary.tsv"
    if not f.exists():
        sys.exit(f"missing {f}")
    return pd.read_csv(f, sep="\t").set_index("language")


def pick(a):
    res = Path(a.res)
    names = sorted(d.name[: -len(a.split) - 1] for d in res.glob(f"*_{a.split}") if (d / "summary.tsv").exists())
    if "base" not in names:
        sys.exit("base result missing")
    base = load(res, "base", a.split)
    rows = []
    for n in names:
        s = load(res, n, a.split)
        if set(s.index) != set(base.index):
            sys.exit(f"{n}: {len(s)} languages scored, base has {len(base)} (incomplete run?)")
        rows.append(dict(name=n, macro=s.error.mean(), improved=int((s.error < base.error).sum()),
                         loops=int(s.loops.sum()), **{w: s.error.get(w) for w in WATCH}))
    t = pd.DataFrame(rows).sort_values("macro")
    print(f"FLEURS {a.split} macro error ({len(base)} languages):\n" + t.round(2).to_string(index=False), file=sys.stderr)
    print(t.iloc[0]["name"])


def report(a):
    res, out = Path(a.res), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    b, m = load(res, "base", "test"), load(res, a.winner, "test")
    if set(b.index) != set(m.index):
        sys.exit("base and winner were scored on different languages")
    cmp_ = pd.DataFrame({"metric": b.metric, "n": b.n, "base": b.error, "model": m.error})
    cmp_["delta"] = cmp_.model - cmp_.base
    cmp_ = cmp_.sort_values("delta")
    (out / "results").mkdir(exist_ok=True)
    cmp_.to_csv(out / "results/fleurs_test_compare.tsv", sep="\t", float_format="%.2f")
    b.to_csv(out / "results/fleurs_test_base_summary.tsv", sep="\t", float_format="%.4f")
    m.to_csv(out / "results/fleurs_test_model_summary.tsv", sep="\t", float_format="%.4f")

    dev = []
    for d in sorted(res.glob("*_dev")):
        if (d / "summary.tsv").exists():
            s = pd.read_csv(d / "summary.tsv", sep="\t")
            dev.append((d.name[:-4], s.error.mean()))
    dev.sort(key=lambda x: x[1])

    bm, mm = b.error.mean(), m.error.mean()
    better = int((cmp_.delta < 0).sum())
    worse5 = cmp_[cmp_.delta > 5]
    status = "MET" if mm < a.target else "NOT MET"
    lines = [
        f"## FLEURS test, {len(cmp_)} languages, {int(b.n.sum())} utterances",
        "",
        "| | macro error | loops |",
        "|---|---|---|",
        f"| base `{a.base_card}` | {bm:.2f} | {int(b.loops.sum())} |",
        f"| this model (`{a.winner}`) | **{mm:.2f}** | {int(m.loops.sum())} |",
        "",
        f"Change: {mm - bm:+.2f} points; {better}/{len(cmp_)} languages improved. Target < {a.target}: **{status}**.",
        "",
        "Forgetting check: " + ", ".join(f"{w} {b.error[w]:.2f} → {m.error[w]:.2f}" for w in WATCH if w in cmp_.index) + ".",
        "",
        "Languages worse by more than 5 points: " + (", ".join(f"{l} ({d:+.1f})" for l, d in worse5.delta.items()) or "none") + ".",
        "",
        "### Candidate selection (FLEURS dev, lower is better)",
        "",
        "| candidate | dev macro error |",
        "|---|---|",
        *[f"| `{n}`{' ← chosen' if n == a.winner else ''} | {v:.2f} |" for n, v in dev],
        "",
        "### Per language (FLEURS test)",
        "",
        "| language | metric | base | this model | Δ |",
        "|---|---|---|---|---|",
        *[f"| {l} | {r.metric} | {r.base:.2f} | {r.model:.2f} | {r.delta:+.2f} |" for l, r in cmp_.sort_index().iterrows()],
    ]
    results_md = "\n".join(lines) + "\n"
    (out / "results.md").write_text(results_md)

    # inference card: same family/arch/tokenizer as the base card, checkpoint filled in by the user
    base_card = next((d for y in Path(a.cards).glob("*.yaml") for d in yaml.safe_load_all(y.read_text())
                      if d and d.get("name") == a.base_card), None)
    if base_card is None:
        sys.exit(f"card {a.base_card} not found in {a.cards}")
    card_name = "octopus_asr_1b_v1"
    card = dict(name=card_name, model_family=base_card["model_family"], model_arch=base_card["model_arch"],
                checkpoint="file:///ABSOLUTE/PATH/TO/model.pt", tokenizer_ref=base_card["tokenizer_ref"])
    (out / f"{card_name}.yaml").write_text(yaml.safe_dump(card, sort_keys=False))

    step, alpha = a.winner.split("_a") if "_a" in a.winner else (a.winner, "1.0")
    langs = sorted({OMNI2SHORT.get(l, l.split("_")[0]) for l in cmp_.index})
    readme = f"""---
license: {a.license}
library_name: omnilingual-asr
pipeline_tag: automatic-speech-recognition
tags: [asr, speech, multilingual, fleurs, omnilingual-asr, wise-ft]
language:
{chr(10).join(f"- {l}" for l in langs)}
---

# Octopus ASR — omniASR 1B v1

Fine-tune of Meta's Omnilingual ASR `{a.base_card}` on 64 languages, released as a WiSE-FT weight blend:
`{alpha} × fine-tuned ({step.replace('s', 'step_', 1)} of run 3) + {1 - float(alpha):.2f} × base`.
Blending recovers part of what fine-tuning forgot (English, Japanese) while keeping most of the gain.
Training checkpoints and logs: [{a.src_repo}](https://huggingface.co/{a.src_repo}).

## Evaluation

Scoring: NFC, casefold, punctuation removed, native digits → ASCII, script-specific fixes (Malayalam chillu,
Bengali khanda ta, Arabic diacritics, Hebrew points). WER per language (CER for Chinese, Japanese, Thai, Burmese),
summed over the language's utterances; macro = plain mean over languages. Konkani is trained but has no FLEURS data.
The blend weight was chosen on FLEURS dev; FLEURS test was scored once, for the base model and the chosen model.

{results_md}
## Usage

The file is a fairseq2 checkpoint for the [omnilingual-asr](https://github.com/facebookresearch/omnilingual-asr)
library (editable install). Download `model.pt`, copy `{card_name}.yaml` into
`src/omnilingual_asr/cards/models/` with `checkpoint:` set to the absolute path of `model.pt`, then:

```python
import torch
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

pipe = ASRInferencePipeline(model_card="{card_name}", device="cuda", dtype=torch.bfloat16)
print(pipe.transcribe(["audio.wav"], lang=["hin_Deva"], batch_size=1))
```

Language codes are ISO 639-3 + script (`eng_Latn`, `hin_Deva`, `cmn_Hans`, ...). Audio up to ~40 s per segment.
"""
    (out / "README.md").write_text(readme)
    print(f"macro error: base {bm:.2f} -> model {mm:.2f} ({mm - bm:+.2f}); target < {a.target}: {status}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pick"); p.add_argument("--res", required=True); p.add_argument("--split", default="dev")
    r = sub.add_parser("report")
    r.add_argument("--res", required=True); r.add_argument("--winner", required=True); r.add_argument("--out", required=True)
    r.add_argument("--target", type=float, default=19.5); r.add_argument("--base-card", default="omniASR_LLM_1B_v2")
    r.add_argument("--cards", required=True); r.add_argument("--repo", required=True); r.add_argument("--src-repo", required=True)
    r.add_argument("--license", default="apache-2.0")
    a = ap.parse_args()
    {"pick": pick, "report": report}[a.cmd](a)
