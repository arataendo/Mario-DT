#!/bin/bash
# DT の教師（PPO）自体を、生成した多数のステージで追加学習させる（CPU サーバー用）。
#
#   今の PPO（level11 系統）は 11 ステージでしか学習しておらず、生成したステージでは弱い
#   （評価コーパスの難しい帯でクリア率 48%）。DT は教師を真似るので、教師が知らないステージで
#   弱いままだと、ステージを増やしても DT の上限は上がらない。そこで先に PPO を鍛える。
#
#   1. 学習用ステージを生成（STAGES 個）
#   2. 800万ステップの PPO から、11 ステージ + 生成したステージで追加学習（毎エピソードくじ引き）
#   3. 追加学習前後の PPO を、評価コーパス 60 ステージと未知のテスト用 9 ステージで比べる
#   4. 高難度の評価専用ステージ 60 個（つまみ 0.7〜0.95、どの学習にも使わない）で、
#      これまでの追加学習（models/ppo_div*）をすべて並べて比べる
#      → ppo_<NAME>_out/report.txt
#
# 使い方:
#   WORKERS=48 ./run_ppo_div.sh                                   # 1000 ステージ（ppo_div、実行済み）
#   WORKERS=48 NAME=div10k STAGES=10000 ./run_ppo_div.sh          # 1万ステージ。開始点・ステップ数・学習率は同じ
#   WORKERS=48 NAME=div10k STAGES=10000 TOTAL=30000000 ./run_ppo_div.sh
#                                         # 1万ステージを続きから 3000 万ステップまで延長（評価もやり直す）
#   WORKERS=48 NAME=div10k_cur STAGES=10000 CURRICULUM=1 ./run_ppo_div.sh
#                                         # 苦手なステージほど多く出す（同じ 1万ステージ。MIX=0.5 で半分は均等）
# 学習の 1/10 ごとにチェックポイントを保存し、止まっても再実行で最後のチェックポイントから続きを学習する。
set -euo pipefail
cd "$(dirname "$0")"
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f .venv/bin/activate ]; then
  set +u; source .venv/bin/activate; set -u
fi
command -v python >/dev/null || { echo "❌ python が見つかりません。source .venv/bin/activate を実行してください"; exit 1; }

WORKERS=${WORKERS:-48}
TOTAL=${TOTAL:-10000000}
LR=${LR:-1e-4}
NAME=${NAME:-div}
STAGES=${STAGES:-1000}
CURRICULUM=${CURRICULUM:-0}
MIX=${MIX:-0.5}
CUR_ARGS=""
[ "$CURRICULUM" = 1 ] && CUR_ARGS="--curriculum --curriculum-mix $MIX"
# 学習用ステージのシード。評価コーパス 300000〜、生成実験 600000〜、DT 用 800000〜/900000〜、
# 1000 ステージ 1000000〜、1万ステージ 2000000〜、高難度の評価用 3000000〜 で重ならない
if [ "$NAME" = div ]; then SEED_BASE=${SEED_BASE:-1000000}; else SEED_BASE=${SEED_BASE:-2000000}; fi
BASE_STEPS=8000000
BASE=models/mario_ppo_level11_checkpoint_${BASE_STEPS}_steps.zip
TRAIN=corpus/train_${NAME}
DIR=models/ppo_${NAME}
OUT=ppo_${NAME}_out
ORIG=Level1-1,Level_custom,Level_easy_01,Level_easy_02,Level_easy_03,Level_medium_01,Level_medium_02,Level_medium_03
ORIG=$ORIG,Level_hard_01,Level_hard_02,Level_hard_03
TEST=Level_test_easy,Level_test_easy_02,Level_test_easy_03,Level_test_medium,Level_test_medium_02
TEST=$TEST,Level_test_medium_03,Level_test_hard,Level_test_hard_02,Level_test_hard_03
export PYTHONIOENCODING=utf-8 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
step() { echo; echo "======== $* ========  $(date '+%m/%d %H:%M')"; }
# 比べるモデル: 追加学習前（800万）と、追加学習の途中・最後のチェックポイント（最大5個）
models_of() {
  local cks n pick m="$BASE"
  cks=$(ls "$1"/mario_ppo_level11_checkpoint_*_steps.zip 2>/dev/null \
    | sed -E 's/.*_checkpoint_([0-9]+)_steps\.zip/\1/' | sort -n)
  n=$(echo "$cks" | wc -l)
  pick=$(echo "$cks" | awk -v n="$n" 'NR == n || NR % int((n + 4) / 5) == 0')
  for s in $pick; do m="$m,$1/mario_ppo_level11_checkpoint_${s}_steps.zip"; done
  echo "$m"
}
# 評価: 同じモデルの組み合わせで評価済みなら飛ばす（学習を延長してチェックポイントが増えたらやり直す）
evaluate() {
  local f=$1 models=$2; shift 2
  if [ -f "$f" ] && [ "$(cat "$f.models" 2>/dev/null)" = "$models" ]; then echo "既存: $f"; return; fi
  python panel_difficulty.py --models "$models" --workers "$WORKERS" --device cpu --seed 0 --output "$f" "$@" | sed -n 1p
  echo "$models" > "$f.models"
}
[ -f "$BASE" ] || { echo "❌ $BASE がありません（元の PC から models/ に送ってください）"; exit 1; }
[ -f corpus/v1/manifest.json ] || python make_corpus.py --name v1 --count 60 | tail -1   # 評価コーパス（シードで再現）
mkdir -p "$DIR" "$OUT"

step "1/4 学習用ステージを生成（$STAGES 個、シード $SEED_BASE〜）"
[ -f "$TRAIN/manifest.json" ] && echo "既存" || python make_corpus.py --name "train_${NAME}" \
  --count "$STAGES" --seed-base "$SEED_BASE" | tail -4

step "2/4 PPO の追加学習（${TOTAL} ステップ、並列 ${WORKERS}、学習率 ${LR}${CUR_ARGS:+、苦手なステージを多く出す mix=$MIX}）"
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
  python -u train_ppo.py --random-level --levels "$ORIG" --levels-from "$TRAIN/manifest.json" \
    --load-model "$load" --total-steps "$remain" --num-envs "$WORKERS" --device cpu --lr "$LR" \
    --model-dir "$DIR" --log-dir "logs/ppo_${NAME}" $CUR_ARGS 2>&1 | grep -v "^\s*|" | tee -a "$OUT/train.log"
fi

step "3/4 追加学習の前後を比べる（評価コーパスとテスト用ステージ）"
MODELS=$(models_of "$DIR")
echo "$MODELS" | tr ',' '\n'
evaluate "$OUT/eval_corpus.json" "$MODELS" --levels-from corpus/v1/manifest.json --episodes 8
evaluate "$OUT/eval_test.json" "$MODELS" --levels "$TEST" --episodes 20

step "4/4 高難度の評価専用ステージで、これまでの追加学習を比べる"
[ -f corpus/hard_eval/manifest.json ] || python make_corpus.py --name hard_eval --count 60 \
  --dmin 0.7 --dmax 0.95 --seed-base 3000000 | tail -4
for d in models/ppo_div*/; do
  d=${d%/}
  ls "$d"/mario_ppo_level11_checkpoint_*_steps.zip >/dev/null 2>&1 || continue
  evaluate "$OUT/eval_hard_$(basename "$d").json" "$(models_of "$d")" \
    --levels-from corpus/hard_eval/manifest.json --episodes 8
done

python - "$OUT" <<'EOF' | tee "$OUT/report.txt"
import glob, json, os, sys
import numpy as np
out = sys.argv[1]
steps = lambda n: n.split("_checkpoint_")[-1].replace("_steps.zip", "")
corpus = json.load(open(f"{out}/eval_corpus.json", encoding="utf-8"))
test = json.load(open(f"{out}/eval_test.json", encoding="utf-8"))
man = sorted(json.load(open("corpus/v1/manifest.json", encoding="utf-8")), key=lambda m: m["difficulty_param"])
bands = [("コーパス 易しい1/3", [m["path"] for m in man[:20]]), ("コーパス 中くらい1/3", [m["path"] for m in man[20:40]]),
         ("コーパス 難しい1/3", [m["path"] for m in man[40:]])]
names = [a["model"] for a in next(iter(corpus.values()))["agents"]]
for d in ("easy", "medium", "hard"):
    bands.append((f"テスト {d}", [k for k in test if f"Level_test_{d}" in k]))
src = lambda k: corpus if k in corpus else test
print(f"[1] クリア率（{out}。最初が追加学習前、残りは追加学習の途中・最後。テスト用は学習で一度も見ていないステージ）")
print(f"{'':22s} " + " ".join(f"{steps(n):>10s}" for n in names))
for label, ks in bands:
    vals = [np.mean([src(k)[k]["agents"][i]["clear_rate"] for k in ks]) for i in range(len(names))]
    print(f"{label:22s} " + " ".join(f"{v * 100:9.0f}%" for v in vals))

hm = sorted(json.load(open("corpus/hard_eval/manifest.json", encoding="utf-8")), key=lambda m: m["difficulty_param"])
halves = [("つまみ 0.70〜0.82", [m["path"] for m in hm[:30]]), ("つまみ 0.83〜0.95", [m["path"] for m in hm[30:]])]
print("\n[2] 高難度の評価専用ステージ 60 個（どの学習にも使っていない）でのクリア率 / 到達率")
print(f"    {'学習':12s} {'ステップ':>10s}   {'全体':>14s}   " + "   ".join(f"{h:>16s}" for h, _ in halves))
base_done = False
for f in sorted(glob.glob(f"{out}/eval_hard_*.json")):
    r = json.load(open(f, encoding="utf-8"))
    run = os.path.basename(f)[len("eval_hard_"):-len(".json")]
    agents = [a["model"] for a in next(iter(r.values()))["agents"]]
    for i, n in enumerate(agents):
        if i == 0 and base_done:
            continue   # 追加学習前は同じシードで同じ値になるので1回だけ
        base_done = True
        label = "（追加学習前）" if i == 0 else run
        cell = lambda ks: (f"{np.mean([r[k]['agents'][i]['clear_rate'] for k in ks]) * 100:4.0f}% / "
                           f"{np.mean([r[k]['agents'][i]['progress'] for k in ks]) * 100:3.0f}%")
        print(f"    {label:12s} {steps(n):>10s}   {cell(list(r)):>14s}   "
              + "   ".join(f"{cell(ks):>16s}" for _, ks in halves))
EOF
echo
echo "持ち帰り: git add -f $OUT/ && git commit -m 'PPO の追加学習（$NAME）' && git push"
echo "学習した PPO は git に載らないので: cp -r $DIR ~/Mario-DT/models/"
