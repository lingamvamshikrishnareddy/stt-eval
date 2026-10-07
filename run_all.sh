#!/usr/bin/env bash
# Final pipeline for octopus-asr run 3, on one GPU box. Every stage is resumable: re-run the same command after a crash.
#
#   1. fetch    step_12000 / step_20000 model files from the run-3 HF repo
#   2. fleurs   build FLEURS dev + test (63 of the 64 languages; Konkani has no FLEURS)
#   3. blend    WiSE-FT: step_12000 at several alphas, other steps at 0.7
#   4. dev      score base + every candidate on FLEURS dev (DEV_N utterances per language)
#   5. pick     lowest dev macro error wins -- the test split is never used for choosing
#   6. test     score base + the winner on the full FLEURS test split
#   7. report   per-language comparison + model card
#   8. push     winner + card + results -> DST_REPO
#
#   export HF_TOKEN=hf_...
#   bash run_all.sh 2>&1 | tee run_all.log
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
: "${HF_TOKEN:?export HF_TOKEN=hf_... first}"
export HF_TOKEN

W=${WORK:-/workspace/octo}
PY=${PY:-python3}
export OMNI_ASR_REPO=${OMNI_ASR_REPO:-$W/omnilingual-asr}
BASE=${BASE_CARD:-omniASR_LLM_1B_v2}
SRC_REPO=${SRC_REPO:-guruawe/octopus-asr-omniASR-1B-run3}
DST_REPO=${DST_REPO:-guruawe/octopus-asr-omniASR-1B-v1}
MAIN_STEP=${MAIN_STEP:-12000}          # best gate checkpoint (gate macro 19.96)
OTHER_STEPS=${OTHER_STEPS:-20000}      # compared at alpha 0.7 only
ALPHAS=${ALPHAS:-0.5 0.7 0.85}         # alpha = weight of the fine-tuned model
DEV_N=${DEV_N:-100}
BATCH=${BATCH:-16}
TARGET=${TARGET:-19.5}
PUSH=${PUSH:-1}
PRIVATE=${PRIVATE:-1}
LICENSE=${LICENSE:-apache-2.0}

mkdir -p "$W"/{ckpts,blends,res,final}
log() { echo "[$(date +%H:%M:%S)] $*"; }

# ---------------------------------------------------------------- 0. preflight
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY - <<'EOF'
import os, sys, pathlib
import torch, fairseq2, omnilingual_asr
repo = pathlib.Path(os.environ["OMNI_ASR_REPO"]).resolve()
pkg = pathlib.Path(omnilingual_asr.__file__).resolve()
print(f"torch {torch.__version__} cuda={torch.cuda.is_available()} bf16={torch.cuda.is_bf16_supported()}")
print(f"fairseq2 {getattr(fairseq2, '__version__', '?')}  omnilingual_asr {pkg}")
if repo not in pkg.parents:
    sys.exit(f"omnilingual_asr must be an editable install from {repo} (fine-tuned checkpoints are registered as cards there)")
EOF
free=$(df -BG --output=avail "$W" | tail -1 | tr -dc 0-9)
[ "$free" -lt 80 ] && log "WARNING: only ${free} GB free in $W; ~100 GB recommended"

# ---------------------------------------------------------------- 1. fetch checkpoints
$PY "$HERE/fetch_ckpt.py" --repo "$SRC_REPO" --steps $MAIN_STEP $OTHER_STEPS --out "$W/ckpts"

# ---------------------------------------------------------------- 2. FLEURS dev + test
if [ ! -f "$W/fleurs_eval/texts.tsv" ]; then
    log "building FLEURS dev + test"
    $PY "$HERE/stt_fleurs_eval.py" build --out "$W/fleurs_eval" --cache "$W/hf_cache"
    rm -rf "$W/hf_cache"                                  # raw WAV parquet, ~tens of GB
fi

# ---------------------------------------------------------------- 3. blends (candidate name -> model file)
declare -A CK
CK[base]=""
for s in $MAIN_STEP $OTHER_STEPS; do CK[s${s}_a1.0]="$W/ckpts/step_${s}.pt"; done
blend() {   # step alpha
    local out="$W/blends/s$1_a$2.pt"
    if [ ! -f "$out" ]; then
        log "blend step_$1 alpha=$2"
        $PY "$HERE/stt_blend.py" --ckpts "$W/ckpts/step_$1.pt" --alpha "$2" --base "$BASE" --out "$out.tmp.pt"
        mv "$out.tmp.pt" "$out"
    fi
    CK[s$1_a$2]=$out
}
for a in $ALPHAS; do blend $MAIN_STEP $a; done
for s in $OTHER_STEPS; do blend $s 0.7; done

# ---------------------------------------------------------------- 4. dev sweep
score() {   # name split n
    local args=(--eval "$W/fleurs_eval" --split "$2" --n "$3" --model "$BASE" --batch "$BATCH" --out "$W/res/$1_$2")
    [ -n "${CK[$1]}" ] && args+=(--ckpt "${CK[$1]}")
    [ "$2" = test ] && args+=(--resume-every 256)
    log "score $1 on $2 (n=$3)"
    $PY "$HERE/stt_fleurs_eval.py" score "${args[@]}"
}
for name in base $(printf '%s\n' "${!CK[@]}" | grep -v '^base$' | sort); do score "$name" dev "$DEV_N"; done

# ---------------------------------------------------------------- 5. pick on dev
WIN=$($PY "$HERE/final_report.py" pick --res "$W/res" --split dev)
log "winner on dev: $WIN (${CK[$WIN]})"
[ "$WIN" = base ] && { log "no candidate beats the base model on dev; stopping before test/push"; exit 3; }

# ---------------------------------------------------------------- 6. full test (base + winner only)
score base test 0
score "$WIN" test 0

# ---------------------------------------------------------------- 7. report + model card
$PY "$HERE/final_report.py" report --res "$W/res" --winner "$WIN" --target "$TARGET" \
    --base-card "$BASE" --cards "$OMNI_ASR_REPO/src/omnilingual_asr/cards/models" \
    --repo "$DST_REPO" --src-repo "$SRC_REPO" --license "$LICENSE" --out "$W/final"
cat "$W/final/results.md"

# ---------------------------------------------------------------- 8. push
if [ "$PUSH" = 1 ]; then
    $PY "$HERE/hf_push.py" --repo "$DST_REPO" --model "${CK[$WIN]}" --final "$W/final" --private "$PRIVATE"
else
    log "PUSH=0: not uploading. Files are in $W/final; model file: ${CK[$WIN]}"
fi
log "done"
