#!/bin/bash
# 段階3 v5: 難易度関数を大きい DT（多様なステージ・幅256・6層・20エポック）に差し替えて、
# 指定難易度の生成をやり直し、v4（今の DT）と同じ審判で比べる一式（研究室PC用）。
#
#   v4: 目的関数 = 今の DT + ルール + PPO
#   v5: 目的関数 = 大きい DT + ルール + PPO （DT 以外は v4 と同じ）
#   審判 = 先読みプランナー（どの目的関数にも含まれない）
#
#   目標 D* の尺度は v4 と同じ（今の DT でのコーパスの D_progress）にそろえる（--ref-dt）。
#   大きい DT の値はコーパス上の分布の対応でこの尺度に写す。これで目標の意味とつまみが v4 と同じになり、
#   「つまみ1個」は v4 と同一のステージになる（分析で審判の値が一致することを確かめる）。
#
#   1. v5 の探索（候補のシードは v4 と同じ）
#   2. v5 の検証（大きい DT・PPO・ルール・プランナー。別シード 1000）
#   3. 分析 → gen_out/v5/report.txt（v5 単体 + 同じ審判で v4 と比較）
#
# 使い方:
#   WORKERS=8 ./run_v5.sh
#   OBJECTIVE=dt NAME=v5dt WORKERS=8 ./run_v5.sh   # 目的関数を大きい DT だけにする（パネルを混ぜない）
# 各段の結果を保存するので、途中で止まっても再実行で続きから走る。
set -euo pipefail
cd "$(dirname "$0")"
# このフォルダの .venv を使う。別の仮想環境が有効になっていても切り替える
# （別の環境のまま動いて「No module named 'stable_baselines3'」で止まったことがある）
if [ -f .venv/bin/activate ] && [ "${VIRTUAL_ENV:-}" != "$(pwd)/.venv" ]; then
  set +u; source .venv/bin/activate; set -u
fi
command -v python >/dev/null || { echo "❌ python が見つかりません。source .venv/bin/activate を実行してください"; exit 1; }
python -c "import torch, stable_baselines3, pygame" 2>/dev/null || {
  echo "❌ $(command -v python) に必要なライブラリがありません。"
  echo "   このフォルダで python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt を済ませてください"
  exit 1; }

WORKERS=${WORKERS:-8}
OBJECTIVE=${OBJECTIVE:-combo3}
NAME=${NAME:-v5}
DT_MODEL=${DT_MODEL:-models/mario_dt_div_big_epoch20.pth}
DT_CORPUS=${DT_CORPUS:-div_big_out/dt.json}   # 大きい DT でのコーパスの結果（seed 0、run_div_train.sh で測ったもの）
PLANNER_AGENTS=${PLANNER_AGENTS:-plan_h8,plan_h4,plan_h2}
PLANNER_EPISODES=${PLANNER_EPISODES:-4}
V4=gen_out/v4
OUT=gen_out/$NAME
CORPUS=v1

export PYTHONIOENCODING=utf-8 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
step() { echo; echo "======== $* ========  $(date '+%m/%d %H:%M')"; }

need="$DT_MODEL $DT_CORPUS corpus/$CORPUS/manifest.json validity_out/$CORPUS/dt.json validity_out/$CORPUS/panel.json"
need="$need validity_out/$CORPUS/rule_panel.json validity_out/$CORPUS/planner_panel.json"
need="$need $V4/validate_manifest.json $V4/val_dt.json $V4/val_ppo.json $V4/val_rule.json $V4/val_planner.json"
need="$need $(python -c "from panel_difficulty import DEFAULT_PANEL; print(' '.join(DEFAULT_PANEL))")"
missing=""
for f in $need; do [ -f "$f" ] || missing="$missing $f"; done
if [ -n "$missing" ]; then
  echo "❌ 次のファイルがありません:"; for f in $missing; do echo "   $f"; done; exit 1
fi
# 分析用。--corpus-dt は D* の尺度（今の DT）。v5 の DT の値は meta.json に書いた対応で写す
CORPUS_ARGS="--corpus-manifest corpus/$CORPUS/manifest.json --corpus-dt validity_out/$CORPUS/dt.json \
  --corpus-ppo validity_out/$CORPUS/panel.json --corpus-rule validity_out/$CORPUS/rule_panel.json"

step "1/3 $NAME の探索（目的関数 $OBJECTIVE、DT $DT_MODEL）"
python generate_replicates.py --model "$DT_MODEL" --objective "$OBJECTIVE" --workers "$WORKERS" --out "$OUT" \
  --corpus-manifest "corpus/$CORPUS/manifest.json" --corpus-dt "$DT_CORPUS" --ref-dt "validity_out/$CORPUS/dt.json" \
  --corpus-ppo "validity_out/$CORPUS/panel.json" --corpus-rule "validity_out/$CORPUS/rule_panel.json"

M="$OUT/validate_manifest.json"
step "2/3 $NAME の検証"
[ -f "$OUT/val_dt.json" ] || python difficulty.py --model "$DT_MODEL" --levels-from "$M" \
  --episodes 10 --workers "$WORKERS" --seed 1000 --output "$OUT/val_dt.json" | tail -1
[ -f "$OUT/val_ppo.json" ] || python panel_difficulty.py --levels-from "$M" \
  --episodes 8 --workers "$WORKERS" --seed 1000 --output "$OUT/val_ppo.json" | tail -1
[ -f "$OUT/val_rule.json" ] || python rule_panel.py --levels-from "$M" \
  --episodes 8 --workers "$WORKERS" --seed 1000 --output "$OUT/val_rule.json"
[ -f "$OUT/val_planner.json" ] || python planner_panel.py --levels-from "$M" --agents "$PLANNER_AGENTS" \
  --episodes "$PLANNER_EPISODES" --workers "$WORKERS" --seed 1000 --output "$OUT/val_planner.json"

step "3/3 分析"
python analyze_replicates.py --gen-dir "$OUT" --compare "$V4" $CORPUS_ARGS \
  --corpus-planner "validity_out/$CORPUS/planner_panel.json" | tee "$OUT/report.txt"
echo
echo "持ち帰り: git add -f $OUT/ && git commit -m '段階3 $NAME' && git pull --no-rebase --no-edit && git push"
