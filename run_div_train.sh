#!/bin/bash
# 多数のステージで集めたデータ（run_div_collect.sh）で DT を学習し直し、今の DT と比べる（研究室PC用）。
#
#   1. CPU サーバーから運んだ tar を確かめて展開
#   2. 今のデータ（dt_dataset_v10、11 ステージ）と合わせ、報酬を今の DT と同じ設定で付け替える
#   3. DT を学習（今の DT と同じ: frame stack 2、20 エポック × 1000 ステップ）
#   4. 評価: 学習用・テスト用 13 ステージ × 20 本、評価コーパス 60 ステージ × 10 本 × 2 シード
#   5. 分析 → div_out/report.txt（今の DT との比較）、div_out/validity_report.txt
#
# 使い方:
#   WORKERS=8 ./run_div_train.sh
# 各段の結果を保存するので、途中で止まっても再実行で続きから走る（学習も最後に保存したエポックから再開）。
set -euo pipefail
cd "$(dirname "$0")"
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f .venv/bin/activate ]; then
  set +u; source .venv/bin/activate; set -u
fi
command -v python >/dev/null || { echo "❌ python が見つかりません。source .venv/bin/activate を実行してください"; exit 1; }

WORKERS=${WORKERS:-8}
EPOCHS=${EPOCHS:-20}
HIDDEN=${HIDDEN:-128}   # モデルの大きさ（今の DT は 128 / 3 層）。新しく学習するときだけ効く
NLAYER=${NLAYER:-3}
RUN=${RUN:-mario_dt_div}
DATA=${DATA:-dt_dataset_div}
OUT=${OUT:-div_out}
LEVELS=Level_easy_01,Level_medium_01,Level_hard_01,Level_hard_02
LEVELS=$LEVELS,Level_test_easy,Level_test_easy_02,Level_test_easy_03
LEVELS=$LEVELS,Level_test_medium,Level_test_medium_02,Level_test_medium_03
LEVELS=$LEVELS,Level_test_hard,Level_test_hard_02,Level_test_hard_03
export PYTHONIOENCODING=utf-8 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
step() { echo; echo "======== $* ========  $(date '+%m/%d %H:%M')"; }
for f in dt_dataset_v10/metadata.pkl corpus/v1/manifest.json validity_out/v1/dt.json validity_out/v1/panel.json \
         validity_out/v1/rule_panel.json validity_out/v1/planner_panel.json temp_out/levels_t10.json; do
  [ -f "$f" ] || { echo "❌ $f がありません"; exit 1; }
done
mkdir -p "$OUT"

step "1/5 データの展開"
if [ -f "$DATA/metadata.pkl" ]; then
  echo "既存"
else
  [ -f "$DATA.tar" ] || { echo "❌ $DATA.tar がありません。CPU サーバーから運んでください"; exit 1; }
  [ -f "$DATA.tar.sha256" ] && sha256sum -c "$DATA.tar.sha256"
  tar -xf "$DATA.tar"
fi

step "2/5 今のデータと合わせて、報酬を付け替える"
if [ -f "$DATA/metadata_rebalanced.pkl" ]; then
  echo "既存"
else
  python - "$DATA" <<'EOF'
import pickle, sys
data = sys.argv[1]
eps = []
for p in ("dt_dataset_v10/metadata.pkl", f"{data}/metadata.pkl"):
    with open(p, "rb") as f:
        e = pickle.load(f)
    print(f"  {p}: {len(e)} エピソード、{sum(len(x['actions']) for x in e)} ステップ")
    eps += e
with open(f"{data}/metadata_merged.pkl", "wb") as f:
    pickle.dump(eps, f)
EOF
  python rebalance_dt_dataset.py --input "$DATA/metadata_merged.pkl" --output "$DATA/metadata_rebalanced.pkl" \
    | tee "$OUT/rebalance.log"
fi

step "3/5 DT の学習（$EPOCHS エポック）"
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
    --hidden-size "$HIDDEN" --n-layer "$NLAYER" --run-name "$RUN" 2>&1 | tee -a "$OUT/train.log"
fi
MODEL=models/${RUN}_epoch${EPOCHS}.pth

step "4/5 評価"
[ -f "$OUT/levels.json" ] || python difficulty.py --model "$MODEL" --levels "$LEVELS" --episodes 20 \
  --workers "$WORKERS" --seed 0 --output "$OUT/levels.json" | sed -n 1p
[ -f "$OUT/dt.json" ] || python difficulty.py --model "$MODEL" --levels-from corpus/v1/manifest.json \
  --episodes 10 --workers "$WORKERS" --seed 0 --output "$OUT/dt.json" | sed -n 1p
[ -f "$OUT/dt_retest.json" ] || python difficulty.py --model "$MODEL" --levels-from corpus/v1/manifest.json \
  --episodes 10 --workers "$WORKERS" --seed 1000 --output "$OUT/dt_retest.json" | sed -n 1p

step "5/5 分析"
python analyze_temperature.py \
  --levels 11ステージ=temp_out/levels_t10.json 多様なステージ="$OUT/levels.json" \
  --corpus 11ステージ=validity_out/v1/dt.json 多様なステージ="$OUT/dt.json" | tee "$OUT/report.txt"
python analyze_validity.py --manifest corpus/v1/manifest.json --dt "$OUT/dt.json" --dt-retest "$OUT/dt_retest.json" \
  --panel validity_out/v1/panel.json --rule-panel validity_out/v1/rule_panel.json \
  --planner-panel validity_out/v1/planner_panel.json > "$OUT/validity_report.txt"
grep -A3 "^\[信頼性\]" "$OUT/validity_report.txt"
echo
echo "※ 評価コーパスは学習用ステージと同じ生成器で作っている。汎化の公平な確認はテスト用9ステージで見る"
echo "持ち帰り: git add -f $OUT/ && git commit -m 多様なステージ && git push"
