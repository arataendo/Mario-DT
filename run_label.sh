#!/bin/bash
# 配置から難易度を予測するモデルの学習用に、多数のステージに DT で難易度を付け、予測の当たり方を確かめる（研究室PC用）。
#
#   1. label_stages.py: 生成器のステージと、それに編集をかけたステージを COUNT 個作り、DT で難易度を付ける
#   2. difficulty_model.py --extra: それで学習し、既存の 261 個（DT を 10 本 × 2 回測った精密なラベル）で評価
#      → label_out/<NAME>/report.txt
#
# 使い方:
#   WORKERS=8 ./run_label.sh
# 1ステージずつではなく CHUNK 個ずつ保存するので、止まっても再実行で続きから走る。
set -euo pipefail
cd "$(dirname "$0")"
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f .venv/bin/activate ]; then
  set +u; source .venv/bin/activate; set -u
fi
command -v python >/dev/null || { echo "❌ python が見つかりません。source .venv/bin/activate を実行してください"; exit 1; }

WORKERS=${WORKERS:-8}
COUNT=${COUNT:-3000}
EPISODES=${EPISODES:-5}
NAME=${NAME:-v1}
DT_MODEL=${DT_MODEL:-models/mario_dt_20260924_111016_epoch20.pth}
export PYTHONIOENCODING=utf-8 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy

echo "======== 1/2 DT で難易度を付ける（$COUNT 個 × 5 target × $EPISODES 本） ========"
python -u label_stages.py --model "$DT_MODEL" --count "$COUNT" --episodes "$EPISODES" \
  --workers "$WORKERS" --name "$NAME"

echo "======== 2/2 学習して、既存のステージで評価 ========"
python difficulty_model.py --extra "label_out/$NAME/labels.jsonl" | tee "label_out/$NAME/report.txt"
