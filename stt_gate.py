#!/usr/bin/env python3
"""Per-language validation report and the run-2 abort gate.

Reads the recipe's validation transcripts (<ws>/transcriptions/rank_0.{ref,hyp}.txt, one block of N lines per
validation round: step 0 when validate_at_start, then every --every steps), maps each reference to its language by
text (validation output order is not file order), scores every language with stt_score and compares it to step 0.

Gate (latest round vs step 0):
  FAIL  macro error worse by more than --macro_tol points, or loop count above max(2 x step-0 loops, step-0 + 10)
  WARN  any language worse by more than --lang_tol points (20 utterances per language are noisy: review, don't abort)
On FAIL it writes <ws>/GATE_FAIL with the reasons; with --stop_pid it also stops training (the newest checkpoint stays).

  python stt_gate.py --ws run_r2/ws_1.xxxx --dev /home/jovyan/work/ft_data_r2 [--stop_pid $(cat run_r2/train.pid)]
"""
from __future__ import annotations
import argparse, collections, os, signal, sys
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from stt_common import is_degenerate, list_partitions   # noqa: E402
from stt_score import errors, normalize, uses_cer       # noqa: E402


def key(t):
    return normalize(t).replace(" ", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", required=True); ap.add_argument("--dev", required=True)
    ap.add_argument("--every", type=int, default=2000)
    ap.add_argument("--macro_tol", type=float, default=3.0); ap.add_argument("--lang_tol", type=float, default=10.0)
    ap.add_argument("--stop_pid", type=int, default=None, help="training process (its own session): whole group is stopped")
    ap.add_argument("--stop_wrapper_pid", type=int, default=None,
                    help="finetune_omnilingual.py train --foreground wrapper: stopped FIRST so it can't auto-restart training")
    a = ap.parse_args()

    text2lang = collections.defaultdict(set)
    n_dev = 0                                                           # rows, not unique texts: rounds are row blocks
    for (corpus, split, lang), fl in list_partitions(a.dev).items():
        if split == "dev":
            for f in fl:
                for t in pq.ParquetFile(f).read(columns=["text"]).column(0).to_pylist():
                    text2lang[key(t)].add(lang)
                    n_dev += 1
    tr = Path(a.ws) / "transcriptions"
    refs = (tr / "rank_0.ref.txt").read_text().splitlines()
    hyps = (tr / "rank_0.hyp.txt").read_text().splitlines()
    n_rounds = len(refs) // n_dev
    if n_rounds == 0:
        print(f"no complete validation round yet ({len(refs)} of {n_dev} lines)"); return 0

    rows = []
    for r in range(n_rounds):
        step = r * a.every
        acc = collections.defaultdict(lambda: [0, 0, 0, 0])            # err, len, n, loops
        unmatched = 0
        for ref, hyp in zip(refs[r * n_dev:(r + 1) * n_dev], hyps[r * n_dev:(r + 1) * n_dev]):
            langs = text2lang.get(key(ref))
            if not langs or len(langs) > 1:
                unmatched += 1; continue
            lang = next(iter(langs))
            e, n = errors(ref, hyp, lang)
            s = acc[lang]
            rk, hk = key(ref), key(hyp)                                 # characters, so scripts without spaces work
            s[0] += e; s[1] += n; s[2] += 1; s[3] += int(is_degenerate(hyp) or (len(hk) > 1.5 * len(rk) and len(hk) > 20))
        for lang, (e, n, k, loops) in acc.items():
            rows.append(dict(step=step, language=lang, metric="CER" if uses_cer(lang) else "WER",
                             error=100 * e / max(n, 1), n=k, loops=loops))
        if unmatched:
            print(f"step {step}: {unmatched} reference lines could not be matched to one language")
    df = pd.DataFrame(rows)
    df.to_csv(Path(a.ws) / "per_lang_gate.tsv", sep="\t", index=False, float_format="%.2f")

    piv = df.pivot(index="language", columns="step", values="error")
    first, last = piv.columns.min(), piv.columns.max()
    piv["delta"] = piv[last] - piv[first]
    print(piv.round(1).sort_values("delta").to_string())
    loops = df.groupby("step").loops.sum()
    macro = df.groupby("step").error.mean()
    print("\nmacro error by step: " + ", ".join(f"{s}: {v:.2f}" for s, v in macro.items()))
    print("loops by step:       " + ", ".join(f"{s}: {v}" for s, v in loops.items()))
    if last == first:
        print("only the step-0 round so far"); return 0

    fail, warn = [], []
    if macro[last] > macro[first] + a.macro_tol:
        fail.append(f"macro error {macro[first]:.2f} -> {macro[last]:.2f} (tolerance +{a.macro_tol})")
    if loops[last] > max(2 * loops[first], loops[first] + 10):
        fail.append(f"loops {loops[first]} -> {loops[last]}")
    worse = piv[piv.delta > a.lang_tol]
    if len(worse):
        warn.append("languages worse by more than %.0f points: %s" % (a.lang_tol, ", ".join(f"{l} ({d:+.1f})" for l, d in worse.delta.items())))
    better = (piv.delta < 0).sum()
    print(f"\nstep {last} vs step {first}: {better}/{len(piv)} languages improved")
    for w in warn:
        print("WARN", w)
    if fail:
        (Path(a.ws) / "GATE_FAIL").write_text("\n".join(fail) + "\n")
        print("GATE FAIL:", "; ".join(fail))
        for pid, group in ((a.stop_wrapper_pid, False), (a.stop_pid, True)):
            if not pid:
                continue
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM) if group else os.kill(pid, signal.SIGTERM)
                print(f"sent SIGTERM to {'process group' if group else 'process'} {pid}")
            except ProcessLookupError:
                print(f"process {pid} already gone")
        return 2
    print("GATE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
