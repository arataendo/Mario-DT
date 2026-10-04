#!/bin/bash
# 段階3 v4: 目的関数の評価器の種類を増やすと、独立な審判への伝わり方が上がるかを確かめる一式（研究室PC用）。
#
#   v3: 目的関数 = DT + ルール         （PPO は審判だったが、v4 では目的関数に入る）
#   v4: 目的関数 = DT + ルール + PPO
#   審判 = 先読みプランナー（学習もルールの手書きもしない。どの目的関数にも含まれない）
#
#   1. プランナーでコーパスを測る（審判としての妥当性: 腕前の幅、他パネルとの一致）
#   2. プランナーで v3 の結果を測る
#   3. v4 の探索（候補のシードは v3 と同じなので「つまみ1個」は v3 と同一のステージ）
#   4. v4 の検証（DT・PPO・ルール・プランナー）
#   5. 分析 → gen_out/v4/report.txt（v4 単体 + 同じ審判で v3 と比較）
#
# 使い方:
#   WORKERS=8 ./run_v4.sh
# 各段の結果を保存するので、途中で止まっても再実行で続きから走る。
# プランナー・ルールの評価は1エピソードごとに保存し、子プロセスが死んだり固まったりしても
# 作り直して続ける（以前は Pool.map が子プロセスの OOM で永久に止まることがあった）。
set -euo pipefail
cd "$(dirname "$0")"

WORKERS=${WORKERS:-8}
DT_MODEL=${DT_MODEL:-models/mario_dt_20260924_111016_epoch20.pth}
PLANNER_AGENTS=${PLANNER_AGENTS:-plan_h8,plan_h4,plan_h2}
PLANNER_EPISODES=${PLANNER_EPISODES:-4}
V3=gen_out/v3
OUT=gen_out/v4
CORPUS=v1

export PYTHONIOENCODING=utf-8 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
step() { echo; echo "======== $* ========"; }

need="$DT_MODEL corpus/$CORPUS/manifest.json validity_out/$CORPUS/dt.json validity_out/$CORPUS/panel.json"
need="$need validity_out/$CORPUS/rule_panel.json $V3/validate_manifest.json $V3/val_dt.json $V3/val_ppo.json $V3/val_rule.json"
need="$need $(python -c "from panel_difficulty import DEFAULT_PANEL; print(' '.join(DEFAULT_PANEL))")"
missing=""
for f in $need; do [ -f "$f" ] || missing="$missing $f"; done
if [ -n "$missing" ]; then
  echo "❌ 次のファイルがありません:"; for f in $missing; do echo "   $f"; done; exit 1
fi
CORPUS_ARGS="--corpus-manifest corpus/$CORPUS/manifest.json --corpus-dt validity_out/$CORPUS/dt.json \
  --corpus-ppo validity_out/$CORPUS/panel.json --corpus-rule validity_out/$CORPUS/rule_panel.json"

step "1/5 プランナーでコーパスを測る（審判としての妥当性）"
[ -f "validity_out/$CORPUS/planner_panel.json" ] && echo "既存" || python planner_panel.py \
  --levels-from "corpus/$CORPUS/manifest.json" --agents "$PLANNER_AGENTS" --episodes "$PLANNER_EPISODES" \
  --workers "$WORKERS" --seed 0 --output "validity_out/$CORPUS/planner_panel.json"
python analyze_validity.py --manifest "corpus/$CORPUS/manifest.json" --dt "validity_out/$CORPUS/dt.json" \
  --dt-retest "validity_out/$CORPUS/dt_retest.json" --panel "validity_out/$CORPUS/panel.json" \
  --rule-panel "validity_out/$CORPUS/rule_panel.json" --planner-panel "validity_out/$CORPUS/planner_panel.json" \
  > "validity_out/$CORPUS/report_with_planner.txt"
sed -n '/パネル\] エージェント別/,/量では分からない/p' "validity_out/$CORPUS/report_with_planner.txt"

step "2/5 プランナーで v3 の結果を測る"
[ -f "$V3/val_planner.json" ] && echo "既存" || python planner_panel.py --levels-from "$V3/validate_manifest.json" \
  --agents "$PLANNER_AGENTS" --episodes "$PLANNER_EPISODES" --workers "$WORKERS" --seed 1000 \
  --output "$V3/val_planner.json"

step "3/5 v4 の探索（目的関数 = DT + ルール + PPO）"
python generate_replicates.py --model "$DT_MODEL" --objective combo3 --workers "$WORKERS" --out "$OUT" $CORPUS_ARGS

M="$OUT/validate_manifest.json"
step "4/5 v4 の検証"
[ -f "$OUT/val_dt.json" ] || python difficulty.py --model "$DT_MODEL" --levels-from "$M" \
  --episodes 10 --workers "$WORKERS" --seed 1000 --output "$OUT/val_dt.json" | tail -1
[ -f "$OUT/val_ppo.json" ] || python panel_difficulty.py --levels-from "$M" \
  --episodes 8 --workers "$WORKERS" --seed 1000 --output "$OUT/val_ppo.json" | tail -1
[ -f "$OUT/val_rule.json" ] || python rule_panel.py --levels-from "$M" \
  --episodes 8 --workers "$WORKERS" --seed 1000 --output "$OUT/val_rule.json"
[ -f "$OUT/val_planner.json" ] || python planner_panel.py --levels-from "$M" --agents "$PLANNER_AGENTS" \
  --episodes "$PLANNER_EPISODES" --workers "$WORKERS" --seed 1000 --output "$OUT/val_planner.json"

step "5/5 分析"
python analyze_replicates.py --gen-dir "$OUT" --compare "$V3" $CORPUS_ARGS \
  --corpus-planner "validity_out/$CORPUS/planner_panel.json" | tee "$OUT/report.txt"
