# stt-eval

Final evaluation and release pipeline for octopus-asr (Omnilingual ASR `omniASR_LLM_1B_v2` fine-tune, run 3).

```bash
export HF_TOKEN=hf_...            # never commit it
bash run_all.sh 2>&1 | tee run_all.log
```

`run_all.sh` downloads checkpoints, builds FLEURS dev+test, makes WiSE-FT blends, picks the best on FLEURS **dev**,
scores base + winner on the full FLEURS **test**, writes a model card and pushes to `guruawe/octopus-asr-omniASR-1B-v1`
(private by default). Each stage resumes after a crash. Settings come from environment variables at the top of the script
(`WORK`, `ALPHAS`, `DEV_N`, `BATCH`, `PUSH=0`, `PRIVATE=0`, ...).

Needs: CUDA GPU (24 GB is enough), ~100 GB disk, 48 GB+ RAM for blending, `fairseq2` + an **editable** install of
`omnilingual-asr` at `$OMNI_ASR_REPO`.

| file | role |
|---|---|
| `run_all.sh` | the whole pipeline |
| `fetch_ckpt.py` | download `step_N` model files from the run-3 HF repo |
| `stt_blend.py` | checkpoint averaging + WiSE-FT blend with the base model |
| `stt_fleurs_eval.py` | FLEURS build + scoring |
| `stt_score.py`, `stt_common.py` | normalisation, WER/CER, dataset helpers |
| `stt_gate.py` | in-training per-language gate (reference) |
| `final_report.py` | dev pick, test comparison, model card |
| `hf_push.py` | upload to Hugging Face |
