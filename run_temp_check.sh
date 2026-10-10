#!/bin/bash
# 高い target（180, 235）だけサンプリングの温度を下げると、DT の腕前の上限が上がり、
# かつ target による条件付けが保たれるかを確かめる（研究室PC用。再学習はしない）。
#
#   設定: T1.0（今の DT 難易度）/ hi0.7 / hi0.5（180 と 235 だけ温度 0.7 / 0.5）
#   [A] 学習用・テスト用の 13 ステージ × 5 target × 20 本
#   [B] 評価コーパス 60 ステージ × 5 target × 10 本（T1.0 は既存の validity_out/v1/dt.json と同じ条件）
#   → temp_out/report.txt
#
# 使い方:
#   WORKERS=8 ./run_temp_check.sh
# 各段の結果を保存するので、途中で止まっても再実行で続きから走る。
set -euo pipefail
cd "$(dirname "$0")"
# 仮想環境を有効にし忘れても動くように（tmux の新しいシェルでは毎回必要になる）
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f .venv/bin/activate ]; then
  set +u; source .venv/bin/activate; set -u
fi
command -v python >/dev/null || { echo "❌ python が見つかりません。source .venv/bin/activate を実行してください"; exit 1; }

WORKERS=${WORKERS:-8}
DT_MODEL=${DT_MODEL:-models/mario_dt_20260924_111016_epoch20.pth}
OUT=temp_out
LEVELS=Level_easy_01,Level_medium_01,Level_hard_01,Level_hard_02
LEVELS=$LEVELS,Level_test_easy,Level_test_easy_02,Level_test_easy_03
LEVELS=$LEVELS,Level_test_medium,Level_test_medium_02,Level_test_medium_03
LEVELS=$LEVELS,Level_test_hard,Level_test_hard_02,Level_test_hard_03

export PYTHONIOENCODING=utf-8 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
for f in "$DT_MODEL" corpus/v1/manifest.json validity_out/v1/dt.json validity_out/v1/panel.json \
         validity_out/v1/rule_panel.json validity_out/v1/planner_panel.json; do
  [ -f "$f" ] || { echo "❌ $f がありません"; exit 1; }
done
mkdir -p "$OUT"

declare -A MAP=([t10]="" [hi07]="180:0.7,235:0.7" [hi05]="180:0.5,235:0.5")

echo "======== [A] 学習用・テスト用 13 ステージ ========"
for k in t10 hi07 hi05; do
  [ -f "$OUT/levels_$k.json" ] && { echo "$k: 既存"; continue; }
  echo "$k: 温度 ${MAP[$k]:-全 target 1.0}"
  python difficulty.py --model "$DT_MODEL" --levels "$LEVELS" --episodes 20 --workers "$WORKERS" \
    --seed 0 --temp-map "${MAP[$k]}" --output "$OUT/levels_$k.json" | sed -n 1p
done

echo "======== [B] 評価コーパス 60 ステージ ========"
for k in hi07 hi05; do
  [ -f "$OUT/corpus_$k.json" ] && { echo "$k: 既存"; continue; }
  echo "$k: 温度 ${MAP[$k]}"
  python difficulty.py --model "$DT_MODEL" --levels-from corpus/v1/manifest.json --episodes 10 \
    --workers "$WORKERS" --seed 0 --temp-map "${MAP[$k]}" --output "$OUT/corpus_$k.json" | sed -n 1p
done

echo "======== 分析 ========"
python analyze_temperature.py \
  --levels T1.0="$OUT/levels_t10.json" hi0.7="$OUT/levels_hi07.json" hi0.5="$OUT/levels_hi05.json" \
  --corpus T1.0=validity_out/v1/dt.json hi0.7="$OUT/corpus_hi07.json" hi0.5="$OUT/corpus_hi05.json" \
  | tee "$OUT/report.txt"
