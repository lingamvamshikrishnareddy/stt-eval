#!/usr/bin/env python3
"""FLEURS evaluation for the 64 training languages (63 have FLEURS; Konkani does not).

  build   download FLEURS validation + test (parquet, no loading script) into the training layout, plus a small
          per-language sentinel set used as the in-training dev split
  score   transcribe with a base model card or a fine-tuned checkpoint, write per-utterance hyps and a per-language
          error table (WER, or CER for zh/ja/th/my; Yoruba/Igbo also tone-insensitive)

  python stt_fleurs_eval.py build --out /home/jovyan/work/fleurs_eval --sentinel_root /home/jovyan/work/ft_data_r2 --sentinel_n 20
  python stt_fleurs_eval.py score --eval /home/jovyan/work/fleurs_eval --split dev --n 100 --model omniASR_LLM_1B_v2 --out res/base_dev
  python stt_fleurs_eval.py score ... --ckpt run_r2/ws_1.xxx/checkpoints/step_4000 --out res/s4000_dev
"""
from __future__ import annotations
import argparse, hashlib, io, json, os, random, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from stt_common import SCHEMA, SR, is_degenerate, iter_rows, list_partitions, log, to_flac_int8   # noqa: E402
from stt_score import TONE_LANGS, errors, uses_cer                                                # noqa: E402

FLEURS = {
    "afr_Latn": "af_za", "arb_Arab": "ar_eg", "asm_Beng": "as_in", "azj_Latn": "az_az", "ben_Beng": "bn_in",
    "bul_Cyrl": "bg_bg", "cat_Latn": "ca_es", "ces_Latn": "cs_cz", "cmn_Hans": "cmn_hans_cn", "cym_Latn": "cy_gb",
    "dan_Latn": "da_dk", "deu_Latn": "de_de", "ell_Grek": "el_gr", "eng_Latn": "en_us", "est_Latn": "et_ee",
    "fil_Latn": "fil_ph", "fin_Latn": "fi_fi", "fra_Latn": "fr_fr", "guj_Gujr": "gu_in", "hau_Latn": "ha_ng",
    "heb_Hebr": "he_il", "hin_Deva": "hi_in", "hrv_Latn": "hr_hr", "hun_Latn": "hu_hu", "ibo_Latn": "ig_ng",
    "ind_Latn": "id_id", "ita_Latn": "it_it", "jpn_Jpan": "ja_jp", "kan_Knda": "kn_in", "kat_Geor": "ka_ge",
    "kaz_Cyrl": "kk_kz", "khk_Cyrl": "mn_mn", "kor_Hang": "ko_kr", "lit_Latn": "lt_lt", "lvs_Latn": "lv_lv",
    "mal_Mlym": "ml_in", "mar_Deva": "mr_in", "mya_Mymr": "my_mm", "nld_Latn": "nl_nl", "nob_Latn": "nb_no",
    "npi_Deva": "ne_np", "ory_Orya": "or_in", "pan_Guru": "pa_in", "pes_Arab": "fa_ir", "pol_Latn": "pl_pl",
    "por_Latn": "pt_br", "ron_Latn": "ro_ro", "rus_Cyrl": "ru_ru", "slk_Latn": "sk_sk", "slv_Latn": "sl_si",
    "spa_Latn": "es_419", "swe_Latn": "sv_se", "swh_Latn": "sw_ke", "tam_Taml": "ta_in", "tel_Telu": "te_in",
    "tha_Thai": "th_th", "tur_Latn": "tr_tr", "ukr_Cyrl": "uk_ua", "urd_Arab": "ur_pk", "vie_Latn": "vi_vn",
    "yor_Latn": "yo_ng", "zsm_Latn": "ms_my", "zul_Latn": "zu_za",
}
MAX_SEC = 30.0          # = training max_audio_len


def write_part(root, corpus, split, lang, recs):
    d = Path(root) / "version=0" / f"corpus={corpus}" / f"split={split}" / f"language={lang}"
    d.mkdir(parents=True, exist_ok=True)
    tbl = pa.table({"text": [r[0] for r in recs], "audio_bytes": pa.array([r[1] for r in recs], pa.list_(pa.int8())),
                    "audio_size": [r[2] for r in recs]}, schema=SCHEMA)
    tmp = d / "part-00000.parquet.tmp"
    pq.write_table(tbl, tmp, compression="snappy")
    os.replace(tmp, d / "part-00000.parquet")


def build(a):
    from huggingface_hub import HfApi, hf_hub_download
    files = HfApi().list_repo_files("google/fleurs", repo_type="dataset")
    langs = [l for l in (a.langs.split(",") if a.langs else FLEURS) if l in FLEURS]
    rng = random.Random(a.seed)
    for lang in langs:
        code = FLEURS[lang]
        for hf_split, split in (("validation", "dev"), ("test", "test")):
            done = Path(a.out) / "version=0" / "corpus=fleurs" / f"split={split}" / f"language={lang}" / "part-00000.parquet"
            need_sentinel = split == "dev" and bool(a.sentinel_root)
            if done.exists() and not need_sentinel:
                log(f"{lang} {split}: exists"); continue
            recs, ids = [], []
            for sh in sorted(f for f in files if f.startswith(f"parquet-data/{code}/{hf_split}-")):
                p = hf_hub_download("google/fleurs", sh, repo_type="dataset", cache_dir=a.cache)
                for r in pq.read_table(p, columns=["id", "audio", "transcription"]).to_pylist():
                    wave, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32", always_2d=False)
                    if wave.ndim > 1:
                        wave = wave.mean(axis=1)
                    if sr != SR:
                        import librosa
                        wave = librosa.resample(wave, orig_sr=sr, target_sr=SR)
                    if not r["transcription"] or len(wave) > MAX_SEC * SR:
                        continue
                    recs.append((r["transcription"], to_flac_int8(wave), len(wave)))
                    ids.append(int(r["id"]))
            if not recs:
                log(f"{lang} ({code}) {split}: NO DATA"); continue
            if not done.exists():
                write_part(a.out, "fleurs", split, lang, recs)
            if need_sentinel:
                by_id = {}
                for rec, i in zip(recs, ids):
                    by_id.setdefault(i, rec)                      # one recording per sentence
                pick = rng.sample(sorted(by_id), min(a.sentinel_n, len(by_id)))
                write_part(a.sentinel_root, "fleurs_sentinel", "dev", lang, [by_id[i] for i in pick])
            log(f"{lang} ({code}) {split}: {len(recs)} utterances")
    texts = [(lang, split, t) for (c, split, lang), fl in list_partitions(a.out).items() if c == "fleurs"
             for f in fl for t in pq.read_table(f, columns=["text"]).column(0).to_pylist()]
    pd.DataFrame(texts, columns=["language", "split", "text"]).to_csv(Path(a.out) / "texts.tsv", sep="\t", index=False)
    log(f"build done; {len(texts)} reference texts -> texts.tsv (for the leakage check)")


def _train_one(job):
    """FLEURS train split of one language -> <root>/version=0/corpus=fleurs_train/split=train/language=<lang>/ (1000-row parts)."""
    lang, code, root, cache = job
    from huggingface_hub import HfApi, hf_hub_download
    final = Path(root) / "version=0" / "corpus=fleurs_train" / "split=train" / f"language={lang}"
    if final.exists() and any(final.glob("*.parquet")):
        return lang, -1, 0.0
    # build outside the dataset tree and move in when complete: a killed worker must not leave a half language behind
    d = Path(root) / "_tmp_fleurs_train" / lang
    import shutil
    shutil.rmtree(d, ignore_errors=True)
    shards = sorted(f for f in HfApi().list_repo_files("google/fleurs", repo_type="dataset")
                    if f.startswith(f"parquet-data/{code}/train-"))
    recs, part, n, sec = [], 0, 0, 0.0

    def flush():
        nonlocal recs, part
        if recs:
            tmp = d / f"part-fleurs-{part:05d}.parquet.tmp"
            pq.write_table(pa.table({"text": [r[0] for r in recs], "audio_bytes": pa.array([r[1] for r in recs], pa.list_(pa.int8())),
                                     "audio_size": [r[2] for r in recs]}, schema=SCHEMA), tmp, compression="snappy")
            os.replace(tmp, d / f"part-fleurs-{part:05d}.parquet")
            part += 1
            recs = []
    d.mkdir(parents=True, exist_ok=True)
    for sh in shards:
        p = hf_hub_download("google/fleurs", sh, repo_type="dataset", cache_dir=cache)
        for b in pq.ParquetFile(p).iter_batches(batch_size=200, columns=["audio", "transcription"]):
            for r in b.to_pylist():
                wave, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32", always_2d=False)
                if wave.ndim > 1:
                    wave = wave.mean(axis=1)
                if sr != SR:
                    import librosa
                    wave = librosa.resample(wave, orig_sr=sr, target_sr=SR)
                if not r["transcription"] or len(wave) > MAX_SEC * SR:
                    continue
                recs.append((r["transcription"], to_flac_int8(wave), len(wave)))
                n += 1; sec += len(wave) / SR
                if len(recs) >= 1000:
                    flush()
        os.remove(os.path.realpath(p))                     # free the ~2 GB WAV parquet right away
    flush()
    final.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(final, ignore_errors=True)
    os.replace(d, final)
    return lang, n, sec / 3600


def train(a):
    from multiprocessing import Pool
    langs = [l for l in (a.langs.split(",") if a.langs else FLEURS) if l in FLEURS]
    jobs = [(l, FLEURS[l], a.root, a.cache) for l in langs]
    tot = 0.0
    with Pool(a.workers) as pool:
        for lang, n, h in pool.imap_unordered(_train_one, jobs):
            tot += max(h, 0)
            log(f"{lang}: " + ("already present, kept" if n < 0 else f"{n} utterances, {h:.1f} h"))
    log(f"FLEURS train done: {tot:.1f} h added")


def ckpt_card(ckpt: Path, base: str) -> str:
    """Register a fine-tuned checkpoint (a step dir or a .pt file) as a model card, like finetune_omnilingual eval-ckpt."""
    import yaml
    repo = Path(os.environ.get("OMNI_ASR_REPO", "/home/jovyan/work/omnilingual-asr"))
    pt = ckpt if ckpt.suffix == ".pt" else ckpt / "model/pp_00/tp_00/sdp_00.pt"
    if not pt.exists():
        raise SystemExit(f"{pt} missing")
    cards = repo / "src/omnilingual_asr/cards/models"
    b = next(d for y in cards.glob("*.yaml") for d in yaml.safe_load_all(y.read_text()) if d and d.get("name") == base)
    import hashlib
    name = "ft_" + hashlib.md5(str(pt.resolve()).encode()).hexdigest()[:12]
    (cards / f"{name}.yaml").write_text(yaml.safe_dump(dict(name=name, model_family=b["model_family"], model_arch=b["model_arch"],
                                                          checkpoint=f"file://{pt.resolve()}", tokenizer_ref=b["tokenizer_ref"]),
                                                     sort_keys=False))
    return name


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_tsv(frame, path, **kwargs):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, sep="\t", index=False, **kwargs)
    os.replace(tmp, path)


def score(a):
    import torch
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
    if a.batch < 1 or a.resume_every < 0:
        raise SystemExit("batch must be positive; resume_every must be non-negative")
    dtype_name = a.dtype
    if dtype_name == "auto":
        dtype_name = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    if dtype_name == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise SystemExit("this GPU does not support BF16; use --dtype float16 (T4) or float32")
    card = ckpt_card(Path(a.ckpt), a.model) if a.ckpt else a.model
    is_llm = "_LLM_" in a.model
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    parts = {k: v for k, v in list_partitions(a.eval).items() if k[0] == a.corpus and k[1] == a.split}
    want = set(a.langs.split(",")) if a.langs else None
    parts = {k: v for k, v in parts.items() if not want or k[2] in want}
    if not parts:
        raise SystemExit("no evaluation languages found")
    if want and want != {k[2] for k in parts}:
        raise SystemExit(f"missing requested languages: {sorted(want - {k[2] for k in parts})}")
    if a.resume_every:
        pt = None if not a.ckpt else (Path(a.ckpt) if Path(a.ckpt).suffix == ".pt" else Path(a.ckpt) / "model/pp_00/tp_00/sdp_00.pt")
        revision = Path(a.eval) / ".hf_revision"
        manifest = dict(model=a.model, checkpoint_sha256=file_sha256(pt) if pt else None,
                        corpus=a.corpus, split=a.split, n=a.n, seed=a.seed, batch=a.batch, dtype=dtype_name,
                        languages=sorted(k[2] for k in parts),
                        data_revision=revision.read_text().strip() if revision.exists() else None,
                        scoring_sha256=file_sha256(Path(__file__).with_name("stt_score.py")),
                        evaluator_sha256=file_sha256(__file__))
        dst_manifest = out / "eval_config.json"
        if dst_manifest.exists():
            if json.loads(dst_manifest.read_text()) != manifest:
                raise SystemExit("resume configuration differs from saved results; choose a new output directory")
        else:
            if any(out.glob("*.tsv")):
                raise SystemExit("existing results have no resume manifest; choose a new output directory")
            dst_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    pipe = ASRInferencePipeline(model_card=card, device="cuda", dtype=getattr(torch, dtype_name))
    log(f"inference dtype={dtype_name}, batch={a.batch}; resume every {a.resume_every} utterances")
    summary = []
    for (corpus, split, lang), fl in sorted(parts.items()):
        dst = out / f"{lang}.tsv"
        if not dst.exists():
            rows = [r for f in fl for r in iter_rows(f)]
            if a.n:
                rows = random.Random(a.seed).sample(rows, min(a.n, len(rows)))
            order = sorted(range(len(rows)), key=lambda i: int(rows[i]["audio_size"]))
            hyps = [None] * len(rows)
            partial = out / f"{lang}.partial.tsv"
            if a.resume_every and partial.exists():
                saved = pd.read_csv(partial, sep="\t", keep_default_na=False)
                seen = set()
                for r in saved.itertuples():
                    i = int(r.row_index)
                    if i in seen or not 0 <= i < len(rows) or rows[i]["text"] != r.ref:
                        raise SystemExit(f"invalid resume rows for {lang}")
                    seen.add(i)
                    hyps[i] = r.hyp
                log(f"{lang}: resumed {len(seen)}/{len(rows)} utterances")
            order = [i for i in order if hyps[i] is None]
            pending, started = 0, time.monotonic()
            for s in range(0, len(order), a.batch):
                idx = order[s:s + a.batch]
                res = pipe.transcribe([np.asarray(rows[i]["audio_bytes"], dtype=np.int8) for i in idx],
                                      lang=[lang] * len(idx) if is_llm else None, batch_size=len(idx))
                if len(res) != len(idx):
                    raise RuntimeError(f"{lang}: expected {len(idx)} hypotheses, received {len(res)}")
                for i, h in zip(idx, res):
                    hyps[i] = h
                pending += len(idx)
                if a.resume_every and (pending >= a.resume_every or s + a.batch >= len(order)):
                    done = [(i, rows[i]["text"], h) for i, h in enumerate(hyps) if h is not None]
                    atomic_tsv(pd.DataFrame(done, columns=["row_index", "ref", "hyp"]), partial)
                    log(f"{lang}: saved {len(done)}/{len(rows)} utterances; {time.monotonic() - started:.0f}s this session")
                    pending = 0
            recs = []
            for r, h in zip(rows, hyps):
                e, n = errors(r["text"], h, lang)
                ce, cn = errors(r["text"], h, lang, unit="char")
                te, tn = errors(r["text"], h, lang, tone_insensitive=True) if lang in TONE_LANGS else (e, n)
                recs.append((r["text"], h, e, n, ce, cn, te, tn, is_degenerate(h)))
            atomic_tsv(pd.DataFrame(recs, columns=["ref", "hyp", "err", "len", "cerr", "clen", "terr", "tlen", "loop"]), dst)
            partial.unlink(missing_ok=True)
        d = pd.read_csv(dst, sep="\t", keep_default_na=False)
        summary.append(dict(language=lang, n=len(d), metric="CER" if uses_cer(lang) else "WER",
                            error=100 * d.err.sum() / max(d.len.sum(), 1), cer=100 * d.cerr.sum() / max(d.clen.sum(), 1),
                            tone_insensitive=100 * d.terr.sum() / max(d.tlen.sum(), 1), loops=int(d.loop.sum())))
        log(f"{lang}: {summary[-1]['metric']} {summary[-1]['error']:.1f} (n={len(d)}, loops={summary[-1]['loops']})")
        atomic_tsv(pd.DataFrame(summary), out / "summary.tsv", float_format="%.6f")
    s = pd.DataFrame(summary)
    log(f"macro error over {len(s)} languages: {s.error.mean():.2f}  -> {out / 'summary.tsv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True); b.add_argument("--langs", default="")
    b.add_argument("--sentinel_root", default=None); b.add_argument("--sentinel_n", type=int, default=20)
    b.add_argument("--cache", default=None); b.add_argument("--seed", type=int, default=7)
    s = sub.add_parser("score")
    s.add_argument("--eval", required=True); s.add_argument("--out", required=True)
    s.add_argument("--corpus", default="fleurs", help="fleurs, or fleurs_sentinel with --eval pointing at ft_data_r2")
    s.add_argument("--split", default="dev"); s.add_argument("--langs", default="")
    s.add_argument("--model", default="omniASR_LLM_1B_v2", help="base model card (also the base of --ckpt)")
    s.add_argument("--ckpt", default=None, help="checkpoint step dir or .pt (fine-tuned / averaged / interpolated weights)")
    s.add_argument("--n", type=int, default=0, help="utterances per language (0 = all)")
    s.add_argument("--batch", type=int, default=16); s.add_argument("--seed", type=int, default=7)
    s.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    s.add_argument("--resume-every", type=int, default=0, help="atomically save partial hypotheses every N utterances (0 = each complete language only)")
    t = sub.add_parser("train", help="add FLEURS train (corpus fleurs_train) to a training root; languages already there are kept")
    t.add_argument("--root", required=True); t.add_argument("--langs", default="")
    t.add_argument("--workers", type=int, default=4); t.add_argument("--cache", default=None)
    a = ap.parse_args()
    {"build": build, "score": score, "train": train}[a.cmd](a)
