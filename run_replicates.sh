#!/bin/bash
# 段階3 v3: 独立な探索を複数本回し、3つの生成方法（つまみ1個 / 生成して選ぶ / 編集探索）を
# 計算量をそろえて比べる一式（研究室PC用）。
#   1. 候補の評価と編集探索（generate_replicates.py）
#   2. DT で別シード (seed=1000) で測り直す
#   3. PPO パネル（どの方法にも使っていない審判）
#   4. ルールベースのパネル（combo の目的関数に使ったので参考）
#   5. 分析 → gen_out/<NAME>/report.txt
#
# 使い方:
#   ./run_replicates.sh
#   WORKERS=16 ./run_replicates.sh      # コア数・メモリに余裕があれば増やすと速い（nproc / free -g で確認）
#   REPLICATES=4 ./run_replicates.sh    # 時間を減らしたいとき（独立な標本が減る）
#
# 各段の結果を保存するので、途中で止まっても再実行で続きから走る。
set -euo pipefail
cd "$(dirname "$0")"

NAME=${NAME:-v3}
TARGETS=${TARGETS:-0.25,0.35,0.45,0.55,0.65}
REPLICATES=${REPLICATES:-6}
MU=${MU:-4}
LAM=${LAM:-6}
GENERATIONS=${GENERATIONS:-5}
EPISODES=${EPISODES:-6}
RULE_EPISODES=${RULE_EPISODES:-4}
WORKERS=${WORKERS:-8}
DT_MODEL=${DT_MODEL:-models/mario_dt_20260924_111016_epoch20.pth}
OUT=${OUT:-gen_out/$NAME}
CORPUS=${CORPUS:-v1}

export PYTHONIOENCODING=utf-8 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
step() { echo; echo "======== $* ========"; }

need="$DT_MODEL corpus/$CORPUS/manifest.json validity_out/$CORPUS/dt.json validity_out/$CORPUS/panel.json validity_out/$CORPUS/rule_panel.json"
need="$need $(python -c "from panel_difficulty import DEFAULT_PANEL; print(' '.join(DEFAULT_PANEL))")"
missing=""
for f in $need; do [ -f "$f" ] || missing="$missing $f"; done
if [ -n "$missing" ]; then
  echo "❌ 次のファイルがありません:"
  for f in $missing; do echo "   $f"; done
  exit 1
fi

step "1/5 候補の評価と編集探索（目標 $TARGETS × $REPLICATES 本, μ=$MU λ=$LAM G=$GENERATIONS, 並列 $WORKERS）"
python generate_replicates.py --model "$DT_MODEL" --targets "$TARGETS" --replicates "$REPLICATES" \
  --mu "$MU" --lam "$LAM" --generations "$GENERATIONS" --episodes "$EPISODES" \
  --rule-episodes "$RULE_EPISODES" --workers "$WORKERS" --out "$OUT" \
  --corpus-manifest "corpus/$CORPUS/manifest.json" --corpus-dt "validity_out/$CORPUS/dt.json" \
  --corpus-rule "validity_out/$CORPUS/rule_panel.json"

M="$OUT/validate_manifest.json"
step "2/5 DT で別シードで測り直す (seed=1000)"
[ -f "$OUT/val_dt.json" ] && echo "既存" || python difficulty.py --model "$DT_MODEL" --levels-from "$M" \
  --episodes 10 --workers "$WORKERS" --seed 1000 --output "$OUT/val_dt.json" | tail -2

step "3/5 PPO パネル（審判）"
[ -f "$OUT/val_ppo.json" ] && echo "既存" || python panel_difficulty.py --levels-from "$M" \
  --episodes 8 --workers "$WORKERS" --seed 1000 --output "$OUT/val_ppo.json" | tail -2

step "4/5 ルールベースのパネル"
[ -f "$OUT/val_rule.json" ] && echo "既存" || python rule_panel.py --levels-from "$M" \
  --episodes 8 --workers "$WORKERS" --seed 1000 --output "$OUT/val_rule.json" | tail -2

step "5/5 分析"
python analyze_replicates.py --gen-dir "$OUT" --corpus-manifest "corpus/$CORPUS/manifest.json" \
  --corpus-dt "validity_out/$CORPUS/dt.json" --corpus-ppo "validity_out/$CORPUS/panel.json" \
  --corpus-rule "validity_out/$CORPUS/rule_panel.json" | tee "$OUT/report.txt"
