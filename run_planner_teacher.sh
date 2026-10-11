#!/bin/bash
# 教師を先読みプランナーに替え、生成器で作った多数のステージで集めたデータで DT を学習し直す一式（研究室PC用）。
#
#   今の DT: 教師 = PPO、学習ステージ 11 個。温度を下げても未知のステージでは上限が上がらなかった
#            （temp_out/report.txt）→ 原因は汎化と考えられる
#   新しい DT: 教師 = plan_h8（評価コーパスの難しい帯でも 82% クリア）、学習ステージ 200 個（生成器、評価とは別シード）
#
#   1. 学習用ステージを生成（シード 800000〜。評価コーパス 300000〜・生成実験 600000〜 と重ならない）
#   2. プランナーでデータ収集（ステージごとに6本、ε = 0,0,0,0.3,0.6,0.9）  ← いちばん時間がかかる
#   3. 報酬の付け替え（今の DT と同じ設定）
#   4. DT の学習（今の DT と同じ: frame stack 2、20 エポック × 1000 ステップ）
#   5. 評価: 学習用・テスト用 13 ステージ × 20 本、評価コーパス 60 ステージ × 10 本 × 2 シード
#   6. 分析 → planner_out/report.txt（今の DT との比較）、planner_out/validity_report.txt（信頼性・妥当性）
#
# 使い方:
#   WORKERS=8 ./run_planner_teacher.sh
# 各段の結果を保存するので、途中で止まっても再実行で続きから走る（学習も最後に保存したエポックから再開）。
set -euo pipefail
cd "$(dirname "$0")"
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f .venv/bin/activate ]; then
  set +u; source .venv/bin/activate; set -u
fi
command -v python >/dev/null || { echo "❌ python が見つかりません。source .venv/bin/activate を実行してください"; exit 1; }

WORKERS=${WORKERS:-8}
N_LEVELS=${N_LEVELS:-200}
EPOCHS=${EPOCHS:-20}
RUN=${RUN:-mario_dt_planner}
DATA=dt_dataset_planner
OUT=planner_out
LEVELS=Level_easy_01,Level_medium_01,Level_hard_01,Level_hard_02
LEVELS=$LEVELS,Level_test_easy,Level_test_easy_02,Level_test_easy_03
LEVELS=$LEVELS,Level_test_medium,Level_test_medium_02,Level_test_medium_03
LEVELS=$LEVELS,Level_test_hard,Level_test_hard_02,Level_test_hard_03

export PYTHONIOENCODING=utf-8 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
step() { echo; echo "======== $* ========  $(date '+%m/%d %H:%M')"; }
for f in corpus/v1/manifest.json validity_out/v1/dt.json validity_out/v1/dt_retest.json validity_out/v1/panel.json \
         validity_out/v1/rule_panel.json validity_out/v1/planner_panel.json temp_out/levels_t10.json; do
  [ -f "$f" ] || { echo "❌ $f がありません"; exit 1; }
done
mkdir -p "$OUT"

step "1/6 学習用ステージを生成"
[ -f corpus/train_gen/manifest.json ] && echo "既存" || python make_corpus.py --name train_gen \
  --count "$N_LEVELS" --seed-base 800000

step "2/6 プランナーでデータ収集（${N_LEVELS} ステージ × 6 本）"
[ -f "$DATA/metadata.pkl" ] && echo "既存" || python -u collect_planner_dataset.py \
  --levels-from corpus/train_gen/manifest.json --episodes-per-level 6 --epsilons 0,0,0,0.3,0.6,0.9 \
  --workers "$WORKERS" --out "$DATA" 2>&1 | tee -a "$OUT/collect.log"

step "3/6 報酬の付け替え"
[ -f "$DATA/metadata_rebalanced.pkl" ] && echo "既存" || python rebalance_dt_dataset.py \
  --input "$DATA/metadata.pkl" --output "$DATA/metadata_rebalanced.pkl" | tee "$OUT/rebalance.log"

step "4/6 DT の学習（$EPOCHS エポック）"
last=$(ls models/${RUN}_epoch*.pth 2>/dev/null | sed -E 's/.*_epoch([0-9]+)\.pth/\1/' | sort -n | tail -1 || true)
last=${last:-0}
if [ "$last" -ge "$EPOCHS" ]; then
  echo "既存（epoch $last）"
elif [ "$last" -gt 0 ]; then
  python -u train_dt.py --dataset "$DATA/metadata_rebalanced.pkl" --epochs $((EPOCHS - last)) \
    --steps-per-epoch 1000 --batch-size 64 --num-workers 8 \
    --resume-from "models/${RUN}_epoch${last}.pth" 2>&1 | tee -a "$OUT/train.log"
else
  python -u train_dt.py --dataset "$DATA/metadata_rebalanced.pkl" --epochs "$EPOCHS" \
    --steps-per-epoch 1000 --batch-size 64 --num-workers 8 --frame-stack 2 \
    --run-name "$RUN" 2>&1 | tee -a "$OUT/train.log"
fi
MODEL=models/${RUN}_epoch${EPOCHS}.pth

step "5/6 評価"
[ -f "$OUT/levels.json" ] || python difficulty.py --model "$MODEL" --levels "$LEVELS" --episodes 20 \
  --workers "$WORKERS" --seed 0 --output "$OUT/levels.json" | sed -n 1p
[ -f "$OUT/dt.json" ] || python difficulty.py --model "$MODEL" --levels-from corpus/v1/manifest.json \
  --episodes 10 --workers "$WORKERS" --seed 0 --output "$OUT/dt.json" | sed -n 1p
[ -f "$OUT/dt_retest.json" ] || python difficulty.py --model "$MODEL" --levels-from corpus/v1/manifest.json \
  --episodes 10 --workers "$WORKERS" --seed 1000 --output "$OUT/dt_retest.json" | sed -n 1p

step "6/6 分析"
python analyze_temperature.py \
  --levels PPO教師=temp_out/levels_t10.json プランナー教師="$OUT/levels.json" \
  --corpus PPO教師=validity_out/v1/dt.json プランナー教師="$OUT/dt.json" | tee "$OUT/report.txt"
python analyze_validity.py --manifest corpus/v1/manifest.json --dt "$OUT/dt.json" --dt-retest "$OUT/dt_retest.json" \
  --panel validity_out/v1/panel.json --rule-panel validity_out/v1/rule_panel.json \
  --planner-panel validity_out/v1/planner_panel.json > "$OUT/validity_report.txt"
grep -A3 "^\[信頼性\]" "$OUT/validity_report.txt"
echo
echo "※ 新しい DT はプランナーを模倣しているので、プランナー群との一致は独立な検証にならない"
echo "※ 評価コーパスは学習用ステージと同じ生成器で作っている。汎化の公平な確認はテスト用9ステージで見る"
