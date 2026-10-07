"""Scoring normalisation and WER/CER for STT evaluation (run 2 onward).

Only used to SCORE (ref vs hyp); training text is untouched. Both sides go through the same function, so any rule here
only removes differences that are conventions, never real recognition errors.

    from stt_score import normalize, errors
    e, n = errors(ref, hyp, "mal_Mlym")       # edit distance, reference length (words, or chars for CER languages)
"""
from __future__ import annotations
import re
import unicodedata

CER_LANGS = {"cmn_Hans", "yue_Hant", "jpn_Jpan", "tha_Thai", "mya_Mymr"}
TONE_LANGS = {"yor_Latn", "ibo_Latn"}

# Malayalam: consonant + virama + ZWJ is the legacy spelling of an atomic chillu. Must run BEFORE ZWJ is dropped.
_CHILLU = {"ണ": "ൺ", "ന": "ൻ", "ര": "ർ", "ല": "ൽ", "ള": "ൾ", "ക": "ൿ"}
_CHILLU_RE = re.compile("([ണനരലളക])്‍")
# Bengali/Assamese: TA + virama + ZWJ is the legacy spelling of KHANDA TA.
_KHANDA_RE = re.compile("ত্‍")
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿"), None)

_AR_DIACRITICS = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭـ]")   # harakat, Quranic marks, tatweel
_AR_ALEF = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي"})
_FA_UR = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک"})        # Arabic yeh/kaf -> Persian forms
_HE_POINTS = re.compile("[֑-ׇ]")
_TONE_MARKS = {"̀", "́", "̂", "̄", "̌"}                             # underdot U+0323 is kept
_DIGIT_GAP = re.compile(r"(?<=\d) (?=\d)")


def normalize(text: str | None, lang: str = "", tone_insensitive: bool = False) -> str:
    s = unicodedata.normalize("NFC", text or "")
    if lang.startswith("mal_"):
        s = _CHILLU_RE.sub(lambda m: _CHILLU[m.group(1)], s)
    if lang.endswith("_Beng"):
        s = _KHANDA_RE.sub("ৎ", s)
    s = s.translate(_ZERO_WIDTH)
    if lang.endswith("_Arab"):
        s = _AR_DIACRITICS.sub("", s)
        s = s.translate(_FA_UR if lang in ("pes_Arab", "urd_Arab") else _AR_ALEF)
    if lang.endswith("_Hebr"):
        s = _HE_POINTS.sub("", s)
    if tone_insensitive and lang in TONE_LANGS:
        s = unicodedata.normalize("NFC", "".join(c for c in unicodedata.normalize("NFD", s) if c not in _TONE_MARKS))
    s = s.casefold()
    out = []
    for ch in s:
        cat = unicodedata.category(ch)
        if cat == "Nd":
            out.append(str(unicodedata.digit(ch)))          # native digits -> ASCII
        elif cat[0] in "LM" or cat == "No":
            out.append(ch)
        else:
            out.append(" ")                                 # punctuation, symbols, apostrophes
    s = " ".join("".join(out).split())
    return _DIGIT_GAP.sub("", s)


try:
    from rapidfuzz.distance import Levenshtein as _L

    def _edit(a, b) -> int:
        return _L.distance(a, b)
except ImportError:
    def _edit(a, b) -> int:
        prev = list(range(len(b) + 1))
        for i in range(1, len(a) + 1):
            cur = [i] + [0] * len(b)
            for j in range(1, len(b) + 1):
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
            prev = cur
        return prev[-1]


def uses_cer(lang: str) -> bool:
    return lang in CER_LANGS


def errors(ref: str, hyp: str, lang: str, unit: str = "auto", tone_insensitive: bool = False) -> tuple[int, int]:
    """(edit distance, reference length) after normalisation. unit: auto | word | char.
    Sum both over a language and divide for a corpus-level rate (don't average per-utterance rates)."""
    r, h = normalize(ref, lang, tone_insensitive), normalize(hyp, lang, tone_insensitive)
    if unit == "auto":
        unit = "char" if uses_cer(lang) else "word"
    if unit == "char":
        r, h = r.replace(" ", ""), h.replace(" ", "")
        return _edit(r, h), len(r)
    rw, hw = r.split(), h.split()
    return _edit(rw, hw), len(rw)


def char_error_rate(ref: str, hyp: str, lang: str) -> float:
    e, n = errors(ref, hyp, lang, unit="char")
    return e / n if n else float("nan")


# ------------------------------------------------------------------ script-agnostic comparison (label audit only)
# A CTC model without a language hint may write correct speech in another script (Bengali/Assamese speech in
# Devanagari, Telugu in Kannada). Comparing romanised "sound skeletons" makes that free while an English lecture with a
# translated label stays far apart. Chinese/Japanese keep native-character CER (romanising kanji is unreliable).
NATIVE_CER_LANGS = {"cmn_Hans", "yue_Hant", "jpn_Jpan"}
_UROMAN = None
_NON_ALNUM = re.compile(r"[^a-z0-9]")
_REPEAT = re.compile(r"(.)\1+")


def roman_key(text: str | None, lang: str = "") -> str:
    """Romanise (uroman), keep a-z0-9, drop 'h' (aspiration marks differ between romanisations), collapse repeats."""
    global _UROMAN
    if _UROMAN is None:
        import uroman
        _UROMAN = uroman.Uroman()
    s = _UROMAN.romanize_string(normalize(text, lang)).lower()
    s = _NON_ALNUM.sub("", unicodedata.normalize("NFKD", s))
    return _REPEAT.sub(r"\1", s.replace("h", ""))


def roman_cer(ref: str, hyp: str, lang: str) -> float:
    if lang in NATIVE_CER_LANGS:
        return char_error_rate(ref, hyp, lang)
    r, h = roman_key(ref, lang), roman_key(hyp, lang)
    return _edit(r, h) / len(r) if r else float("nan")


_LATIN = re.compile(r"[A-Za-zÀ-ɏ]")


def latin_share(text: str | None) -> float:
    """Share of letters that are Latin -- for a non-Latin language, a CTC transcript that is mostly Latin is English speech."""
    letters = [c for c in (text or "") if c.isalpha()]
    return sum(bool(_LATIN.match(c)) for c in letters) / len(letters) if letters else 0.0


if __name__ == "__main__":
    ok = True

    def check(name, got, want):
        global ok
        ok &= got == want
        print(f"{'PASS' if got == want else 'FAIL'}  {name}: {got!r}" + ("" if got == want else f" (want {want!r})"))

    check("malayalam chillu", normalize("അവന്‍", "mal_Mlym"), normalize("അവൻ", "mal_Mlym"))
    check("bengali khanda ta", normalize("হঠাত্‍", "ben_Beng"), normalize("হঠাৎ", "ben_Beng"))
    check("yoruba strict keeps tones", normalize("Ọmọ́", "yor_Latn"), "ọmọ́")
    check("yoruba tone-insensitive keeps underdot", normalize("Ọmọ́", "yor_Latn", tone_insensitive=True), "ọmọ")
    check("vietnamese keeps diacritics", normalize("Việt Nam!", "vie_Latn"), "việt nam")
    check("native digits + digit groups", normalize("१२ ३४५ रुपये", "hin_Deva"), "12345 रुपये")
    check("hindi matras kept", normalize("नमस्ते, दुनिया।", "hin_Deva"), "नमस्ते दुनिया")
    check("arabic harakat/alef", normalize("أَهْلاً", "arb_Arab"), "اهلا")
    check("persian yeh/kaf", normalize("كتابي", "pes_Arab"), "کتابی")
    check("hebrew points", normalize("שָׁלוֹם", "heb_Hebr"), "שלום")
    check("chinese uses CER", errors("你好世界", "你好世节", "cmn_Hans"), (1, 4))
    check("hindi uses WER", errors("मैं घर जा रहा हूँ", "मैं घर जा रही हूँ", "hin_Deva"), (1, 5))
    raise SystemExit(0 if ok else 1)
