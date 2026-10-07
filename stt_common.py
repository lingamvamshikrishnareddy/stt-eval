"""Shared helpers for the STT fine-tune pipeline:
finetune_omnilingual.py, distill_teacher_labels.py, noise_augment.py, bakeoff.py.

Dataset layout (Omnilingual ASR "mixture parquet" v0):
  <root>/version=0/corpus=<c>/split=<train|dev>/language=<iso639-3_Script>/part-*.parquet
  file columns: text, audio_bytes (list<int8> = FLAC bytes), audio_size (int64 samples @16 kHz)
  corpus / split / language exist ONLY as folder names (the reader exposes them as dictionary columns)
"""
from __future__ import annotations
import collections, hashlib, io, json, os, re, shutil, time, unicodedata, uuid, zlib
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

SR = 16_000
MAX_LLM_SECONDS = 39.5          # the non-"Unlimited" LLM variants cap around 40 s of audio
# PHYSICAL schema of a part file. corpus / split / language are NOT stored in the file: the recipe opens the dataset with
# hive partitioning and gets them (as dictionary columns) from the folder names -- a physical string column of the same
# name makes pyarrow fail with "Unable to merge: Field corpus has incompatible types" (found in the CTC smoke test).
SCHEMA = pa.schema([("text", pa.string()), ("audio_bytes", pa.list_(pa.int8())), ("audio_size", pa.int64())])
_PART_RE = re.compile(r"corpus=([^/]+)/split=([^/]+)/language=([^/]+)/[^/]+\.parquet$")

# short code -> Omnilingual ISO639-3_Script code (same table the merge used, plus the extras it added)
SHORT2OMNI = {
 "af": "afr_Latn", "ar": "arb_Arab", "as": "asm_Beng", "az": "azj_Latn", "bg": "bul_Cyrl", "bn": "ben_Beng",
 "cs": "ces_Latn", "da": "dan_Latn", "de": "deu_Latn", "el": "ell_Grek", "en": "eng_Latn", "es": "spa_Latn",
 "fa": "pes_Arab", "fi": "fin_Latn", "fr": "fra_Latn", "gu": "guj_Gujr", "ha": "hau_Latn", "he": "heb_Hebr",
 "hi": "hin_Deva", "hr": "hrv_Latn", "hu": "hun_Latn", "id": "ind_Latn", "ig": "ibo_Latn", "it": "ita_Latn",
 "ja": "jpn_Jpan", "ka": "kat_Geor", "kk": "kaz_Cyrl", "kn": "kan_Knda", "ko": "kor_Hang", "ml": "mal_Mlym",
 "mr": "mar_Deva", "ms": "zsm_Latn", "my": "mya_Mymr", "nb": "nob_Latn", "ne": "npi_Deva", "nl": "nld_Latn",
 "or": "ory_Orya", "pa": "pan_Guru", "pl": "pol_Latn", "pt": "por_Latn", "ru": "rus_Cyrl", "sv": "swe_Latn",
 "sw": "swh_Latn", "ta": "tam_Taml", "te": "tel_Telu", "th": "tha_Thai", "tl": "fil_Latn", "uk": "ukr_Cyrl",
 "ur": "urd_Arab", "vi": "vie_Latn", "yo": "yor_Latn", "zh": "cmn_Hans", "zu": "zul_Latn",
 "ro": "ron_Latn", "ca": "cat_Latn", "cy": "cym_Latn", "lt": "lit_Latn", "mn": "khk_Cyrl", "et": "est_Latn",
 "sl": "slv_Latn", "tr": "tur_Latn", "sd": "snd_Arab", "lv": "lvs_Latn", "yue": "yue_Hant", "sk": "slk_Latn",
 "sa": "san_Deva", "kok": "gom_Deva", "mai": "mai_Deva", "no": "nob_Latn", "wuu": "wuu_Hans",
}
OMNI2SHORT = {v: k for k, v in SHORT2OMNI.items()}


def to_omni(code: str) -> str:
    return code if "_" in code else SHORT2OMNI.get(code, code)


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def free_gb(path: str | Path) -> float:
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


def atomic_json(path: str | Path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    os.replace(tmp, path)


# ------------------------------------------------------------------ text
_ZW = dict.fromkeys(map(ord, "​‌‍⁠﻿"), None)


def clean_text(s: str | None) -> str:
    """NFC, drop zero-width chars, collapse whitespace (what the merge stored)."""
    return " ".join(unicodedata.normalize("NFC", s or "").translate(_ZW).split())


def spoken_text(s: str | None) -> str:
    """Casefold, drop punctuation/symbols, KEEP letters, combining marks (Indic vowel signs), digits and apostrophes."""
    s = clean_text(s).casefold()
    out = []
    for ch in s:
        cat = unicodedata.category(ch)
        out.append(ch if (cat[0] in "LMN" or ch == "'") else " ")
    return " ".join("".join(out).split())


_DROP_TAG = re.compile(r"[<\[{]\s*(?:unintelligible|inaudible|unclear|unk)\s*[>\]}]", re.I)
_TAG = re.compile(r"<(?!\s)[^<>]{1,25}(?<!\s)>|\[(?!\s)[^\[\]]{1,25}(?<!\s)\]|\{(?!\s)[^{}]{1,25}(?<!\s)\}")   # no space just inside the brackets

# Verbatim special tokens (kept in 'verbatim' style, stripped in 'spoken')
# CosyVoice3-compatible names: [breath] [short_pause] [laughter] [cough]
# STT-side tokens for annotating speaker disfluencies: [UMM] [UH] [PAUSE]
_VERBATIM_TOKENS = re.compile(
    r"\[(breath|short_pause|laughter|cough|umm|uhh?|pause(?:_short|_long)?|sigh|yawn)\]",
    re.I
)
# Tokens that are meaningful in verbatim mode but must be normalized for the spoken form
_VERBATIM_TO_SPOKEN = {
    "umm": "um", "uh": "uh", "uhh": "uh",         # keep as words in spoken
    "breath": "", "short_pause": "", "laughter": "", # strip in spoken (not speech content)
    "cough": "", "pause": "", "pause_short": "",
    "pause_long": "", "sigh": "", "yawn": "",
}


def strip_markup(s: str | None, keep_verbatim: bool = False) -> str | None:
    """Annotation tags are not speech: <noise>, <pause>, [laughter], {balloon} ... are removed.
    With keep_verbatim=True the known verbatim tokens ([breath], [umm] etc.) are preserved.
    An utterance that contains an <unintelligible>-style marker returns None (caller drops the row)."""
    s = s or ""
    if _DROP_TAG.search(s):
        return None
    if keep_verbatim:
        # Preserve verbatim tokens, remove everything else
        def _sub(m):
            name = m.group(0)[1:-1].lower().replace("-", "_")
            return m.group(0) if _VERBATIM_TOKENS.fullmatch(m.group(0)) else " "
        return re.sub(r"<(?!\s)[^<>]{1,25}(?<!\s)>|\[(?!\s)[^\[\]]{1,25}(?<!\s)\]|\{(?!\s)[^{}]{1,25}(?<!\s)\}", _sub, s)
    return _TAG.sub(" ", s)


def normalize(s: str | None, style: str = "spoken") -> str:
    """Training-text normaliser. Returns '' when the text is unusable (callers must drop empty results).

    Styles:
      spoken   -- casefold + strip punctuation/markup; [breath] etc. stripped (default, training)
      verbatim -- NFC + keep [breath]/[umm]/[uh] tokens; strip only unknowns (for call-center QA)
      as_is    -- NFC only, no case/punct changes (for reference text comparison)
    """
    if style == "verbatim":
        t = strip_markup(s, keep_verbatim=True)
        if t is None:
            return ""
        # Normalize verbatim tokens to lowercase canonical form
        t = _VERBATIM_TOKENS.sub(lambda m: "[" + m.group(1).lower().rstrip("h") + "]" if m.group(1).lower() in ("uhh",) else m.group(0).lower(), t)
        return clean_text(t)
    t = strip_markup(s, keep_verbatim=False)
    if t is None:
        return ""
    if style == "as_is":
        return clean_text(t)
    if style == "spoken":
        return spoken_text(t)
    raise ValueError(f"unknown text style {style!r}")


def text_key(text: str, lang_code: str) -> str:
    """Stable id of (language, spoken-form text) -- used for hold-out / suspect lists across every script."""
    return hashlib.md5(f"{to_omni(lang_code)}||{spoken_text(text)}".encode("utf-8")).hexdigest()


def load_keys(path: str | None) -> set[str]:
    if not path:
        return set()
    return {ln.strip() for ln in open(path) if ln.strip()}


# ------------------------------------------------------------------ script sanity
SCRIPT_RANGES = {
    "Deva": [(0x0900, 0x097F)], "Beng": [(0x0980, 0x09FF)], "Guru": [(0x0A00, 0x0A7F)], "Gujr": [(0x0A80, 0x0AFF)],
    "Orya": [(0x0B00, 0x0B7F)], "Taml": [(0x0B80, 0x0BFF)], "Telu": [(0x0C00, 0x0C7F)], "Knda": [(0x0C80, 0x0CFF)],
    "Mlym": [(0x0D00, 0x0D7F)], "Arab": [(0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
    "Cyrl": [(0x0400, 0x04FF)], "Hebr": [(0x0590, 0x05FF)], "Thai": [(0x0E00, 0x0E7F)],
    "Hang": [(0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F)],
    "Jpan": [(0x3040, 0x30FF), (0x4E00, 0x9FFF)], "Hans": [(0x4E00, 0x9FFF)], "Hant": [(0x4E00, 0x9FFF)],
    "Geor": [(0x10A0, 0x10FF), (0x2D00, 0x2D2F)], "Grek": [(0x0370, 0x03FF), (0x1F00, 0x1FFF)],
    "Mymr": [(0x1000, 0x109F)], "Latn": [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F), (0x1E00, 0x1EFF)],
}


def script_ratio(text: str, lang_code: str) -> float:
    """Share of letters written in the language's expected script (1.0 if the script is unknown or there are no letters)."""
    script = lang_code.split("_")[-1] if "_" in lang_code else None
    ranges = SCRIPT_RANGES.get(script)
    letters = [c for c in text if c.isalpha()]
    if not ranges or not letters:
        return 1.0
    return sum(any(a <= ord(c) <= b for a, b in ranges) for c in letters) / len(letters)


def is_degenerate(text: str) -> bool:
    """Empty output or a repetition loop ('ha ha ha ha ...'), the usual failure of autoregressive ASR."""
    t = (text or "").strip()
    if not t:
        return True
    b = t.encode("utf-8")
    if len(b) >= 40 and len(zlib.compress(b)) / len(b) < 0.18:
        return True
    toks = t.split()
    if len(toks) >= 8 and collections.Counter(toks).most_common(1)[0][1] / len(toks) > 0.5:
        return True
    return False


def chars_per_second(text: str, n_samples: int) -> float:
    return len(text.replace(" ", "")) / max(n_samples / SR, 1e-6)


def cer(ref: str, hyp: str) -> float:
    """Character error rate in [0, inf); both inputs must already be normalized. Empty ref -> nan."""
    if not ref:
        return float("nan")
    import jiwer
    return float(jiwer.cer(ref, hyp)) if hyp else 1.0


# ------------------------------------------------------------------ audio
def to_flac_int8(wave: np.ndarray, sr: int = SR) -> np.ndarray:
    buf = io.BytesIO()
    sf.write(buf, np.clip(wave, -1.0, 1.0), sr, format="FLAC", subtype="PCM_16")
    return np.frombuffer(buf.getvalue(), dtype=np.uint8).astype(np.int8)


def from_int8(arr) -> tuple[np.ndarray, int]:
    wave, sr = sf.read(io.BytesIO(np.asarray(arr, dtype=np.int8).tobytes()), dtype="float32", always_2d=False)
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    return wave, sr


# ------------------------------------------------------------------ dataset io
def parse_partition(path) -> tuple[str, str, str] | None:
    m = _PART_RE.search(str(path).replace("\\", "/"))
    return m.groups() if m else None


def list_partitions(root) -> dict[tuple[str, str, str], list[Path]]:
    out = collections.defaultdict(list)
    for f in sorted(Path(root).rglob("*.parquet")):
        p = parse_partition(f)
        if p:
            out[p].append(f)
    return dict(out)


def iter_rows(path, columns=("text", "audio_bytes", "audio_size")):
    """Yield dict rows from one part file without turning audio into Python int lists (28x memory blow-up).
    Only columns that exist in the file are read (source repos also store language; our own output files do not)."""
    pf = pq.ParquetFile(path)
    tbl = pf.read(columns=[c for c in columns if c in pf.schema_arrow.names])
    cols = {c: tbl.column(c).combine_chunks() for c in tbl.column_names}
    ab = cols.get("audio_bytes")
    if ab is not None:
        vals = ab.values.to_numpy(zero_copy_only=False)
        offs = ab.offsets.to_numpy()
    others = {c: v.to_pylist() for c, v in cols.items() if c != "audio_bytes"}
    for i in range(tbl.num_rows):
        row = {c: v[i] for c, v in others.items()}
        if ab is not None:
            row["audio_bytes"] = vals[offs[i]:offs[i + 1]]
        yield row


class PartWriter:
    """Buffered, atomic, collision-proof writer for the v0 layout (unique run id in every filename)."""

    def __init__(self, root, rows_per_part: int = 1000):
        self.root = Path(root) / "version=0"
        self.rows_per_part = rows_per_part
        self.run = uuid.uuid4().hex[:8]
        self.buf = collections.defaultdict(list)
        self.n = collections.Counter()
        self.rows = collections.Counter()      # (corpus, split, lang) -> rows written
        self.samples = collections.Counter()   # (corpus, split, lang) -> audio samples written
        self.files: list[Path] = []

    def add(self, corpus, split, lang, text, audio_int8, audio_size):
        k = (corpus, split, lang)
        self.buf[k].append((text, audio_int8, int(audio_size)))
        if len(self.buf[k]) >= self.rows_per_part:
            self._flush(k)

    def _flush(self, k):
        rows = self.buf.pop(k, [])
        if not rows:
            return
        corpus, split, lang = k
        d = self.root / f"corpus={corpus}" / f"split={split}" / f"language={lang}"
        d.mkdir(parents=True, exist_ok=True)
        fp = d / f"part-{self.run}-{self.n[k]:05d}.parquet"
        tmp = fp.with_suffix(".parquet.tmp")
        tbl = pa.table({
            "text": pa.array([r[0] for r in rows], pa.string()),
            "audio_bytes": pa.array([r[1] for r in rows], pa.list_(pa.int8())),
            "audio_size": pa.array([r[2] for r in rows], pa.int64()),
        }, schema=SCHEMA)
        pq.write_table(tbl, tmp, compression="snappy")
        os.replace(tmp, fp)
        self.n[k] += 1
        self.rows[k] += len(rows)
        self.samples[k] += sum(r[2] for r in rows)
        self.files.append(fp)

    def flush_all(self):
        for k in list(self.buf):
            self._flush(k)

    close = flush_all

    def hours(self) -> float:
        return sum(self.samples.values()) / SR / 3600


def compute_stats(root, out_tsv) -> Path:
    """TSV (corpus, language, hours) the mixture sampler weights by. Reads only `audio_size`; partition values come
    from the path, so hive-vs-column schema clashes cannot happen. Train split only: dev-only corpora must not take
    sampling weight, and train partitions missing from this file are silently skipped by the recipe."""
    hours = collections.defaultdict(float)
    for (corpus, split, lang), files in list_partitions(root).items():
        if split != "train":
            continue
        for f in files:
            col = pq.ParquetFile(f).read(columns=["audio_size"]).column(0).to_numpy()
            hours[(corpus, lang)] += float(col.sum()) / SR / 3600
    out_tsv = Path(out_tsv); out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_tsv, "w") as fh:
        fh.write("corpus\tlanguage\thours\n")
        for (corpus, lang), h in sorted(hours.items()):
            fh.write(f"{corpus}\t{lang}\t{h}\n")
    return out_tsv


def retry(fn, what: str, n: int = 5):
    for a in range(n):
        try:
            return fn()
        except Exception as e:
            if a == n - 1:
                raise
            w = min(60, 3 * 2 ** a)
            log(f"  retry {a + 1}/{n} {what}: {str(e)[:80]} ({w}s)")
            time.sleep(w)
