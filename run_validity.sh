#!/bin/bash
# 段階2: DT による難易度の信頼性・妥当性を検証する一式を回す（研究室PC用）。
#   1. コーパス生成（難易度つまみを 0.05〜0.95 で均等に振ったステージ群）
#   2. DT で難易度を測る（シード0）
#   3. DT で別シードでもう一度測る（再測定の一致 = 信頼性）
#   4. PPO パネルで難易度を測る（妥当性の基準）
#   5. ルールベースのパネルで難易度を測る（PPO と系統が独立で、腕前の幅も広い基準）
#   6. 分析
#
# 使い方:
#   ./run_validity.sh                     # 既定: 60 ステージ
#   COUNT=100 WORKERS=12 ./run_validity.sh
#
# 各段の結果ファイルが既にあれば飛ばすので、途中で止まっても再実行で続きから走る。
# pipefail: "python ... | tail" で python が失敗したら止まる（無いと tail の成功で握りつぶされる）
set -euo pipefail
cd "$(dirname "$0")"

NAME=${NAME:-v1}
COUNT=${COUNT:-60}
WORKERS=${WORKERS:-8}
DT_MODEL=${DT_MODEL:-models/mario_dt_20260924_111016_epoch20.pth}
DT_EPISODES=${DT_EPISODES:-10}
PANEL_EPISODES=${PANEL_EPISODES:-8}
OUT=${OUT:-validity_out/$NAME}

export PYTHONIOENCODING=utf-8 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
mkdir -p "$OUT"
MAN=corpus/$NAME/manifest.json

step() { echo; echo "======== $* ========"; }

# 何時間も回してから「モデルが無い」で落ちないよう、使うモデルが揃っているか最初に確かめる
PANEL=$(python -c "from panel_difficulty import DEFAULT_PANEL; print(' '.join(DEFAULT_PANEL))")
missing=""
for f in "$DT_MODEL" $PANEL; do [ -f "$f" ] || missing="$missing $f"; done
if [ -n "$missing" ]; then
  echo "❌ 次のモデルがありません（元のPCから転送してください）:"
  for f in $missing; do echo "   $f"; done
  exit 1
fi

step "1/6 コーパス生成"
[ -f "$MAN" ] && echo "既存: $MAN" || python make_corpus.py --name "$NAME" --count "$COUNT"

step "2/6 DT で難易度を測る (seed=0)"
[ -f "$OUT/dt.json" ] && echo "既存" || python difficulty.py --model "$DT_MODEL" --levels-from "$MAN" \
  --episodes "$DT_EPISODES" --workers "$WORKERS" --seed 0 --output "$OUT/dt.json" | tail -3

step "3/6 DT で別シードでもう一度 (seed=1000)"
[ -f "$OUT/dt_retest.json" ] && echo "既存" || python difficulty.py --model "$DT_MODEL" --levels-from "$MAN" \
  --episodes "$DT_EPISODES" --workers "$WORKERS" --seed 1000 --output "$OUT/dt_retest.json" | tail -3

step "4/6 PPO パネルで難易度を測る"
[ -f "$OUT/panel.json" ] && echo "既存" || python panel_difficulty.py --levels-from "$MAN" \
  --episodes "$PANEL_EPISODES" --workers "$WORKERS" --seed 0 --output "$OUT/panel.json" | tail -3

step "5/6 ルールベースのパネルで難易度を測る"
[ -f "$OUT/rule_panel.json" ] && echo "既存" || python rule_panel.py --levels-from "$MAN" \
  --episodes "$PANEL_EPISODES" --workers "$WORKERS" --seed 0 --output "$OUT/rule_panel.json" | tail -3

step "6/6 分析"
python analyze_validity.py --manifest "$MAN" --dt "$OUT/dt.json" --dt-retest "$OUT/dt_retest.json" \
  --panel "$OUT/panel.json" --rule-panel "$OUT/rule_panel.json" \
  --csv "$OUT/per_level.csv" | tee "$OUT/report.txt"
