#!/usr/bin/env bash
# Полный прогон: обучение -> оценка (LoRA и zero-shot) -> графики. Логи в logs/.
# Предполагается, что данные уже скачаны и собраны:
#   python src/download.py model gqa-train llava && python src/prepare_data.py
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/Scripts/python.exe}
export PYTHONIOENCODING=utf-8
mkdir -p logs

step() { echo "[$(date +%H:%M:%S)] $*"; }

step "train";            $PY -u src/train_lora.py "$@"                                > logs/train.log 2>&1
step "eval lora gqa";    $PY -u src/eval_gqa.py --tag lora --adapter outputs/lora-adapter > logs/eval_lora_gqa.log 2>&1
step "eval lora mmb";    $PY -u src/eval_mmbench.py --tag lora --adapter outputs/lora-adapter > logs/eval_lora_mmb.log 2>&1
[ -f results/zeroshot/gqa_testdev_metrics.json ] || \
  { step "eval zs gqa";  $PY -u src/eval_gqa.py --tag zeroshot                        > logs/eval_zeroshot_gqa.log 2>&1; }
step "eval zs mmb";      $PY -u src/eval_mmbench.py --tag zeroshot                    > logs/eval_zeroshot_mmb.log 2>&1
step "eval val";         $PY -u src/eval_gqa.py --tag zeroshot --split val            > logs/eval_val.log 2>&1
                         $PY -u src/eval_gqa.py --tag lora --adapter outputs/lora-adapter --split val >> logs/eval_val.log 2>&1
step "figures";          $PY src/make_figures.py > logs/figures.log 2>&1; $PY src/analyze.py > logs/analyze.log 2>&1
step "PIPELINE DONE"
