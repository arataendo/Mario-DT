#!/bin/bash
# DT の教師（PPO）自体を、生成した多数のステージで追加学習させる（CPU サーバー用）。
#
#   今の PPO（level11 系統）は 11 ステージでしか学習しておらず、生成したステージでは弱い
#   （評価コーパスの難しい帯でクリア率 48%）。DT は教師を真似るので、教師が知らないステージで
#   弱いままだと、ステージを増やしても DT の上限は上がらない。そこで先に PPO を鍛える。
#
#   1. 学習用ステージを生成（1000 個、シード 1000000〜。run_div_collect.sh と同じもの）
#   2. 800万ステップの PPO から、11 ステージ + 生成した 1000 ステージで追加学習（毎エピソードくじ引き）
#   3. 追加学習前後の PPO を、評価コーパス 60 ステージと未知のテスト用 9 ステージで比べる
#      → ppo_div_out/report.txt
#
# 使い方:
#   WORKERS=48 TOTAL=10000000 ./run_ppo_div.sh
# 100万ステップごとにチェックポイントを保存し、止まっても再実行で最後のチェックポイントから続きを学習する。
set -euo pipefail
cd "$(dirname "$0")"
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f .venv/bin/activate ]; then
  set +u; source .venv/bin/activate; set -u
fi
command -v python >/dev/null || { echo "❌ python が見つかりません。source .venv/bin/activate を実行してください"; exit 1; }

WORKERS=${WORKERS:-48}
TOTAL=${TOTAL:-10000000}
LR=${LR:-1e-4}
BASE_STEPS=8000000
BASE=models/mario_ppo_level11_checkpoint_${BASE_STEPS}_steps.zip
DIR=models/ppo_div
OUT=ppo_div_out
ORIG=Level1-1,Level_custom,Level_easy_01,Level_easy_02,Level_easy_03,Level_medium_01,Level_medium_02,Level_medium_03
ORIG=$ORIG,Level_hard_01,Level_hard_02,Level_hard_03
TEST=Level_test_easy,Level_test_easy_02,Level_test_easy_03,Level_test_medium,Level_test_medium_02
TEST=$TEST,Level_test_medium_03,Level_test_hard,Level_test_hard_02,Level_test_hard_03
export PYTHONIOENCODING=utf-8 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
step() { echo; echo "======== $* ========  $(date '+%m/%d %H:%M')"; }
[ -f "$BASE" ] || { echo "❌ $BASE がありません（元の PC から models/ に送ってください）"; exit 1; }
[ -f corpus/v1/manifest.json ] || python make_corpus.py --name v1 --count 60 | tail -1   # 評価コーパス（シードで再現）
mkdir -p "$DIR" "$OUT"

step "1/3 学習用ステージを生成（1000 個）"
[ -f corpus/train_div/manifest.json ] && echo "既存" || python make_corpus.py --name train_div \
  --count 1000 --seed-base 1000000

step "2/3 PPO の追加学習（${TOTAL} ステップ、並列 ${WORKERS}、学習率 ${LR}）"
latest=$(ls "$DIR"/mario_ppo_level11_checkpoint_*_steps.zip 2>/dev/null \
  | sed -E 's/.*_checkpoint_([0-9]+)_steps\.zip/\1/' | sort -n | tail -1 || true)
if [ -n "$latest" ]; then
  load="$DIR/mario_ppo_level11_checkpoint_${latest}_steps.zip"
  remain=$((BASE_STEPS + TOTAL - latest))
else
  load="$BASE"
  remain=$TOTAL
fi
if [ "$remain" -le 0 ]; then
  echo "既存（${latest} ステップまで学習済み）"
else
  echo "$load から、残り $remain ステップ"
  python -u train_ppo.py --random-level --levels "$ORIG" --levels-from corpus/train_div/manifest.json \
    --load-model "$load" --total-steps "$remain" --num-envs "$WORKERS" --device cpu --lr "$LR" \
    --model-dir "$DIR" --log-dir logs/ppo_div 2>&1 | grep -v "^\s*|" | tee -a "$OUT/train.log"
fi

step "3/3 追加学習の前後を比べる"
# 比べるモデル: 追加学習前（800万）と、追加学習の途中・最後のチェックポイント（最大5個）
cks=$(ls "$DIR"/mario_ppo_level11_checkpoint_*_steps.zip | sed -E 's/.*_checkpoint_([0-9]+)_steps\.zip/\1/' | sort -n)
n=$(echo "$cks" | wc -l)
pick=$(echo "$cks" | awk -v n="$n" 'NR == n || NR % int((n + 4) / 5) == 0')
MODELS="$BASE"
for s in $pick; do MODELS="$MODELS,$DIR/mario_ppo_level11_checkpoint_${s}_steps.zip"; done
echo "$MODELS" | tr ',' '\n'
[ -f "$OUT/eval_corpus.json" ] || python panel_difficulty.py --levels-from corpus/v1/manifest.json --models "$MODELS" \
  --episodes 8 --workers "$WORKERS" --device cpu --seed 0 --output "$OUT/eval_corpus.json" | sed -n 1p
[ -f "$OUT/eval_test.json" ] || python panel_difficulty.py --levels "$TEST" --models "$MODELS" \
  --episodes 20 --workers "$WORKERS" --device cpu --seed 0 --output "$OUT/eval_test.json" | sed -n 1p
python - "$OUT" <<'EOF' | tee "$OUT/report.txt"
import json, sys
import numpy as np
out = sys.argv[1]
corpus = json.load(open(f"{out}/eval_corpus.json", encoding="utf-8"))
test = json.load(open(f"{out}/eval_test.json", encoding="utf-8"))
man = sorted(json.load(open("corpus/v1/manifest.json", encoding="utf-8")), key=lambda m: m["difficulty_param"])
bands = [("コーパス 易しい1/3", [m["path"] for m in man[:20]]), ("コーパス 中くらい1/3", [m["path"] for m in man[20:40]]),
         ("コーパス 難しい1/3", [m["path"] for m in man[40:]])]
names = [a["model"] for a in next(iter(corpus.values()))["agents"]]
for d in ("easy", "medium", "hard"):
    bands.append((f"テスト {d}", [k for k in test if f"Level_test_{d}" in k]))
src = lambda k: corpus if k in corpus else test
print("クリア率（各モデル。最初が追加学習前、残りは追加学習の途中・最後。テスト用は学習で一度も見ていないステージ）")
print(f"{'':22s} " + " ".join(f"{n.split('_checkpoint_')[-1].replace('_steps.zip', ''):>10s}" for n in names))
for label, ks in bands:
    vals = [np.mean([src(k)[k]["agents"][i]["clear_rate"] for k in ks]) for i in range(len(names))]
    print(f"{label:22s} " + " ".join(f"{v * 100:9.0f}%" for v in vals))
EOF
echo
echo "持ち帰り: git add -f $OUT/ && git commit -m 'PPO の追加学習' && git push"
