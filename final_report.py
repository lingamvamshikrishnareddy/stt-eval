#!/usr/bin/env python3
"""Pick the best candidate on FLEURS dev, then compare it with the base model on FLEURS test and write a model card.

Result dirs are <res>/<name>_<split>/ as written by 'stt_fleurs_eval.py score': summary.tsv plus one <lang>.tsv of
per-utterance errors per language. Macro WER/CER = mean over languages of the per-language corpus error
(WER; CER for zh/ja/th/my).

Selection rule (declared before the step_20000 candidates were scored; dev results for the step_12000 blends were
already known):
  1. lowest dev macro WER/CER;
  2. candidates within --tol (0.10) points of it count as tied;
  3. among tied: fewest languages worse than base by > 2 points, then the smallest worst change on the protected
     languages (eng, jpn, cmn), then lower alpha of the same checkpoint, then lowest macro.

  python final_report.py pick   --res res --split dev                 # prints the winner's name on stdout
  python final_report.py report --res res --winner s12000_a0.5 --out final ...
"""
from __future__ import annotations
import argparse, math, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from stt_common import OMNI2SHORT   # noqa: E402

PROTECTED = ["eng_Latn", "jpn_Jpan", "cmn_Hans"]     # forgetting checks (run 3: English, Japanese; dev: Chinese)
N_BOOT = 10_000


def load(res: Path, name: str, split: str) -> pd.DataFrame:
    f = res / f"{name}_{split}" / "summary.tsv"
    if not f.exists():
        sys.exit(f"missing {f}")
    return pd.read_csv(f, sep="\t").set_index("language")


def alpha_of(name: str) -> float:
    m = re.search(r"_a([0-9.]+)$", name)
    return float(m.group(1)) if m else 1.0


def stats(s: pd.DataFrame, base: pd.DataFrame) -> dict:
    d = (s.error - base.error).reindex(base.index)
    wer, cer = s.metric == "WER", s.metric == "CER"
    return dict(macro=s.error.mean(), macro_wer=s.error[wer].mean(), macro_cer=s.error[cer].mean(),
                improved=int((d < 0).sum()), worse=int((d > 0).sum()), worse2=int((d > 2).sum()), worse5=int((d > 5).sum()),
                worst=d.max(), median_delta=d.median(), loops=int(s.loops.sum()),
                protected_worst=max(d.get(p, -np.inf) for p in PROTECTED),
                **{p: s.error.get(p, np.nan) for p in PROTECTED})


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
        rows.append(dict(name=n, **stats(s, base)))
    t = pd.DataFrame(rows).sort_values("macro")
    cand = t[t.name != "base"]
    best = cand.macro.min()
    tied = cand[cand.macro <= best + a.tol].copy()
    tied["ckpt"] = tied.name.str.replace(r"_a[0-9.]+$", "", regex=True)
    tied["alpha"] = tied.name.map(alpha_of)
    winner = tied.sort_values(["worse2", "protected_worst", "alpha", "macro"]).iloc[0]["name"]
    if cand.macro.min() >= t[t.name == "base"].macro.iloc[0]:
        winner = "base"
    cols = ["name", "macro", "macro_wer", "macro_cer", "improved", "worse", "worse2", "worse5", "worst", "median_delta",
            "protected_worst", *PROTECTED, "loops"]
    print(f"FLEURS {a.split} ({len(base)} languages). Rule: lowest macro WER/CER; within {a.tol} points -> fewest "
          f"languages >2 pts worse than base, then smallest protected-language change, then lower alpha.\n"
          f"Tied with the best ({best:.2f}): {', '.join(tied.name)}\n"
          + t[cols].round(2).to_string(index=False) + f"\nwinner: {winner}", file=sys.stderr)
    print(winner)


def per_utt(res: Path, name: str, lang: str) -> pd.DataFrame:
    f = res / f"{name}_test" / f"{lang}.tsv"
    if not f.exists():
        sys.exit(f"missing {f}")
    return pd.read_csv(f, sep="\t", keep_default_na=False)


def bootstrap(res: Path, winner: str, langs, n_boot: int = N_BOOT, seed: int = 0):
    """Paired, language-stratified, sentence-clustered bootstrap of the macro error difference (winner - base).
    Within each language, sentence clusters (rows sharing a reference text: FLEURS has several recordings per
    sentence) are resampled with replacement; both models use the same draw. Per-language error is a ratio of sums."""
    rng = np.random.default_rng(seed)
    deltas = np.zeros(n_boot)
    for lang in langs:
        b, m = per_utt(res, "base", lang), per_utt(res, winner, lang)
        if len(b) != len(m) or not (b.ref.values == m.ref.values).all():
            sys.exit(f"{lang}: base and winner test rows are not the same utterances in the same order")
        g = b.groupby("ref", sort=False)
        cl = pd.DataFrame({"be": g.err.sum(), "me": m.groupby(b.ref.values, sort=False).err.sum(), "n": g.len.sum()})
        be, me, n = cl.be.values, cl.me.values, cl.n.values
        idx = rng.integers(0, len(cl), size=(n_boot, len(cl)))
        deltas += 100 * (me[idx].sum(1) - be[idx].sum(1)) / np.maximum(n[idx].sum(1), 1)
    deltas /= len(langs)
    return deltas


def sign_test(k_better: int, n: int) -> float:
    """Two-sided exact binomial test of k improvements among n non-tied languages (p = 0.5)."""
    if n == 0:
        return 1.0
    k = min(k_better, n - k_better)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


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

    st_b, st_m = stats(b, b), stats(m, b)
    bm, mm = st_b["macro"], st_m["macro"]
    rel = 100 * (mm - bm) / bm
    boot = bootstrap(res, a.winner, sorted(cmp_.index))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p_better = float((boot < 0).mean())
    nontied = int((cmp_.delta != 0).sum())
    p_sign = sign_test(st_m["improved"], nontied)
    if hi < 0:
        status = "BEATS BASE (95% CI below 0)"
    elif lo > 0:
        status = "WORSE THAN BASE (95% CI above 0)"
    else:
        status = "NO SIGNIFICANT DIFFERENCE FROM BASE (95% CI includes 0)"
    (out / "results/bootstrap.tsv").write_text(
        f"delta\tci_low\tci_high\tp_delta_below_0\tn_boot\n{mm - bm:.4f}\t{lo:.4f}\t{hi:.4f}\t{p_better:.4f}\t{N_BOOT}\n")

    dev = []
    for d in sorted(res.glob("*_dev")):
        if (d / "summary.tsv").exists():
            s = pd.read_csv(d / "summary.tsv", sep="\t")
            dev.append((d.name[:-4], s.error.mean()))
    dev.sort(key=lambda x: x[1])
    n_wer, n_cer = int((cmp_.metric == "WER").sum()), int((cmp_.metric == "CER").sum())
    worse2 = cmp_[cmp_.delta > 2]

    lines = [
        f"## FLEURS test: {len(cmp_)} languages, {int(b.n.sum())} utterances",
        "",
        "| | macro WER/CER | macro WER ({} langs) | macro CER ({} langs) | loops |".format(n_wer, n_cer),
        "|---|---|---|---|---|",
        f"| base `{a.base_card}` | {bm:.2f} | {st_b['macro_wer']:.2f} | {st_b['macro_cer']:.2f} | {st_b['loops']} |",
        f"| this model (`{a.winner}`) | **{mm:.2f}** | {st_m['macro_wer']:.2f} | {st_m['macro_cer']:.2f} | {st_m['loops']} |",
        "",
        f"Change in macro WER/CER: **{mm - bm:+.2f} points** ({rel:+.1f}% relative), "
        f"95% CI [{lo:+.2f}, {hi:+.2f}] (paired bootstrap, {N_BOOT:,} resamples of sentence clusters within each "
        f"language). Verdict: **{status}**.",
        "",
        f"Languages: {st_m['improved']} better, {st_m['worse']} worse, {len(cmp_) - nontied} unchanged "
        f"(sign test p = {p_sign:.2g}; secondary). Median change {st_m['median_delta']:+.2f}, worst {st_m['worst']:+.2f}; "
        f"{st_m['worse2']} languages worse by > 2 points, {st_m['worse5']} by > 5.",
        "",
        "Protected languages: " + ", ".join(f"{p} {b.error[p]:.2f} → {m.error[p]:.2f} ({m.error[p] - b.error[p]:+.2f})"
                                            for p in PROTECTED if p in cmp_.index) + ".",
        "",
        "Languages worse by more than 2 points: "
        + (", ".join(f"{l} ({d:+.1f})" for l, d in worse2.delta.items()) or "none") + ".",
        "",
        "### Candidate selection (FLEURS dev, 100 utterances per language, lower is better)",
        "",
        "| candidate | dev macro WER/CER |",
        "|---|---|",
        *[f"| `{n}`{' ← chosen' if n == a.winner else ''} | {v:.2f} |" for n, v in dev],
        "",
        "### Per language (FLEURS test)",
        "",
        "| language | metric | n | base | this model | Δ |",
        "|---|---|---|---|---|---|",
        *[f"| {l} | {r.metric} | {int(r.n)} | {r.base:.2f} | {r.model:.2f} | {r.delta:+.2f} |"
          for l, r in cmp_.sort_index().iterrows()],
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

    step = re.sub(r"_a[0-9.]+$", "", a.winner)
    alpha = alpha_of(a.winner)
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
`{alpha:g} × fine-tuned ({step.replace('s', 'step_', 1)} of run 3) + {1 - alpha:.2f} × base`.
Training checkpoints and logs: [{a.src_repo}](https://huggingface.co/{a.src_repo}).

## Evaluation

- **Metric:** macro WER/CER = plain mean over languages of each language's corpus error rate (errors summed over
  its utterances / reference length). WER for {n_wer} languages, CER for Chinese, Japanese, Thai and Burmese.
  Both are reported separately below. Konkani is trained but has no FLEURS data.
- **Normalisation (both models):** NFC, casefold, punctuation removed, native digits → ASCII, script-specific
  fixes (Malayalam chillu, Bengali khanda ta, Arabic diacritics, Hebrew points). Empty outputs count as errors.
- **Protocol:** candidates (checkpoints and blend weights) were compared on FLEURS dev, 100 utterances per language
  with a fixed seed; the in-training checkpoint check also used FLEURS dev. The selection rule was fixed before the
  step_20000 candidates were scored. FLEURS test was scored once, for the base model and the chosen model only.
- **Uncertainty:** paired bootstrap of the macro difference; within each language, sentence clusters are resampled
  and both models use the same draw. One training run, so seed-to-seed variation is not measured.
- **Disclosures:** `azj_Latn` and `est_Latn` are not in the model's language-hint list, so neither model gets a
  language hint for them. Overlap between the fine-tuning data and FLEURS dev/test was not audited here; the base
  model's pretraining exposure is unknown. Retention is only measured on the languages below, all of which were in
  the fine-tuning set.

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
    print(f"macro WER/CER: base {bm:.2f} -> model {mm:.2f} ({mm - bm:+.2f}, {rel:+.1f}%), 95% CI [{lo:+.2f}, {hi:+.2f}]; "
          f"{st_m['improved']} better / {st_m['worse']} worse: {status}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pick"); p.add_argument("--res", required=True); p.add_argument("--split", default="dev")
    p.add_argument("--tol", type=float, default=0.10, help="macro points within which candidates count as tied")
    r = sub.add_parser("report")
    r.add_argument("--res", required=True); r.add_argument("--winner", required=True); r.add_argument("--out", required=True)
    r.add_argument("--target", type=float, default=19.5, help="unused; kept for older run_all.sh")
    r.add_argument("--base-card", default="omniASR_LLM_1B_v2")
    r.add_argument("--cards", required=True); r.add_argument("--repo", required=True); r.add_argument("--src-repo", required=True)
    r.add_argument("--license", default="apache-2.0")
    a = ap.parse_args()
    {"pick": pick, "report": report}[a.cmd](a)
