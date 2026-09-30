#!/bin/bash
# 段階3: 指定難易度のステージを編集探索で作り、独立な基準で検証する一式（研究室PC用）。
#   1. 探索（generate_by_difficulty.py）。初期集団 = 難易度つまみで狙ったステージ（ベースライン）
#      目的関数: OBJECTIVE=combo（既定。DT の高い target + ルールパネル）/ OBJECTIVE=dt（v1 と同じ DT だけ）
#      combo では PPO パネルを探索に使わず、最終検証の独立な審判として取っておく
#   2. DT で別シード (seed=1000) で測り直す（探索時のシードへの過適合を見る）
#   3. PPO パネルで測る
#   4. ルールベースのパネルで測る
#   5. 分析 → gen_out/<NAME>/report.txt（gen_out/v1 があれば比較表も出す）
#
# 使い方:
#   ./run_generation.sh
#   TARGETS=0.3,0.5,0.7 GENERATIONS=20 WORKERS=12 ./run_generation.sh
#
# 探索は世代ごと、検証は段ごとに結果を保存するので、途中で止まっても再実行で続きから走る。
set -euo pipefail
cd "$(dirname "$0")"

OBJECTIVE=${OBJECTIVE:-combo}
if [ "$OBJECTIVE" = "combo" ]; then NAME=${NAME:-v2}; else NAME=${NAME:-v1}; fi
TARGETS=${TARGETS:-0.25,0.45,0.65}
MU=${MU:-6}
LAM=${LAM:-8}
GENERATIONS=${GENERATIONS:-15}
EPISODES=${EPISODES:-6}
RULE_EPISODES=${RULE_EPISODES:-4}
COMPARE=${COMPARE:-gen_out/v1}
WORKERS=${WORKERS:-8}
DT_MODEL=${DT_MODEL:-models/mario_dt_20260924_111016_epoch20.pth}
OUT=${OUT:-gen_out/$NAME}
CORPUS=${CORPUS:-v1}

export PYTHONIOENCODING=utf-8 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
step() { echo; echo "======== $* ========"; }

# 段階2の結果（つまみ→難易度の関係、DT→パネルの関係）を使うので、揃っているか先に確かめる
need="$DT_MODEL corpus/$CORPUS/manifest.json validity_out/$CORPUS/dt.json validity_out/$CORPUS/panel.json validity_out/$CORPUS/rule_panel.json"
need="$need $(python -c "from panel_difficulty import DEFAULT_PANEL; print(' '.join(DEFAULT_PANEL))")"
missing=""
for f in $need; do [ -f "$f" ] || missing="$missing $f"; done
if [ -n "$missing" ]; then
  echo "❌ 次のファイルがありません:"
  for f in $missing; do echo "   $f"; done
  echo "   （corpus/ が無ければ: python make_corpus.py --name $CORPUS --count 60 で同じものが再現されます）"
  exit 1
fi

step "1/5 探索（目的関数 $OBJECTIVE, 目標 $TARGETS, μ=$MU λ=$LAM, $GENERATIONS 世代）"
python generate_by_difficulty.py --model "$DT_MODEL" --targets "$TARGETS" --out "$OUT" \
  --mu "$MU" --lam "$LAM" --generations "$GENERATIONS" --episodes "$EPISODES" --workers "$WORKERS" \
  --objective "$OBJECTIVE" --rule-episodes "$RULE_EPISODES" \
  --corpus-manifest "corpus/$CORPUS/manifest.json" --corpus-dt "validity_out/$CORPUS/dt.json" \
  --corpus-rule "validity_out/$CORPUS/rule_panel.json"

M="$OUT/validate_manifest.json"
step "2/5 DT で別シードで測り直す (seed=1000)"
[ -f "$OUT/val_dt.json" ] && echo "既存" || python difficulty.py --model "$DT_MODEL" --levels-from "$M" \
  --episodes 10 --workers "$WORKERS" --seed 1000 --output "$OUT/val_dt.json" | tail -2

step "3/5 PPO パネル"
[ -f "$OUT/val_ppo.json" ] && echo "既存" || python panel_difficulty.py --levels-from "$M" \
  --episodes 8 --workers "$WORKERS" --seed 1000 --output "$OUT/val_ppo.json" | tail -2

step "4/5 ルールベースのパネル"
[ -f "$OUT/val_rule.json" ] && echo "既存" || python rule_panel.py --levels-from "$M" \
  --episodes 8 --workers "$WORKERS" --seed 1000 --output "$OUT/val_rule.json" | tail -2

step "5/5 分析"
CMP=""
# 別の探索結果（v1 = DT だけ）の検証が揃っていれば、並べて比べる
if [ "$COMPARE" != "$OUT" ] && [ -f "$COMPARE/val_rule.json" ]; then CMP="--compare $COMPARE"; fi
python analyze_generation.py --gen-dir "$OUT" $CMP --corpus-manifest "corpus/$CORPUS/manifest.json" \
  --corpus-dt "validity_out/$CORPUS/dt.json" --corpus-ppo "validity_out/$CORPUS/panel.json" \
  --corpus-rule "validity_out/$CORPUS/rule_panel.json" | tee "$OUT/report.txt"
