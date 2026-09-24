#!/bin/bash
# DT の評価マトリクス（test 9 ステージ + 学習済み 4 ステージ × target × エピソード）を
# ステージ単位で並列に回す。研究室PCの多コアを使うための汎用版。
#
# 使い方:
#   ./run_eval.sh <model.pth> <出力ディレクトリ> [eval_dt_matrix.py への追加引数...]
# 例:
#   ./run_eval.sh models/mario_dt_20260921_175028_epoch20.pth eval_out/v10s --sample
#
# 調整できる環境変数:
#   JOBS=6          同時に走らせるステージ数（既定: CPU コア数の半分、最大 13）
#   TARGETS=0,60,120,180,235
#   EPISODES=10
#
# ステージごとに <出力>/<ステージ>.json を確定させるので、途中で止まっても
# 同じコマンドを再実行すれば終わったステージは飛ばして続きから走る。
set -u
if [ $# -lt 2 ]; then
  sed -n '5,10p' "$0"; exit 1
fi
MODEL=$1; OUT=$2; shift 2
EXTRA="$*"
TARGETS=${TARGETS:-0,60,120,180,235}
EPISODES=${EPISODES:-10}
NPROC=$(nproc 2>/dev/null || echo 4)
JOBS=${JOBS:-$(( NPROC / 2 > 0 ? NPROC / 2 : 1 ))}
[ "$JOBS" -gt 13 ] && JOBS=13

LEVELS="Level_test_easy Level_test_easy_02 Level_test_easy_03
Level_test_medium Level_test_medium_02 Level_test_medium_03
Level_test_hard Level_test_hard_02 Level_test_hard_03
Level_easy_01 Level_medium_01 Level_hard_01 Level_hard_02"

mkdir -p "$OUT"
# 1 プロセスあたりの torch スレッドを絞る。絞らないと JOBS 個のプロセスが
# それぞれ全コアを取り合って、並列にしたのにかえって遅くなる。
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export PYTHONIOENCODING=utf-8 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
export MODEL OUT EXTRA TARGETS EPISODES

run_one() {
  L=$1
  if [ -f "$OUT/$L.json" ]; then echo "⏭️  $L (済)"; return 0; fi
  if python -u eval_dt_matrix.py --model "$MODEL" --levels "$L" \
       --targets "$TARGETS" --episodes "$EPISODES" --max-steps 500 \
       --output "$OUT/$L.json.partial" $EXTRA > "$OUT/$L.log" 2>&1; then
    mv "$OUT/$L.json.partial" "$OUT/$L.json"; echo "✅ $L"
  else
    echo "❌ $L 失敗（$OUT/$L.log を参照）"
  fi
}
export -f run_one

echo "モデル: $MODEL  追加引数: ${EXTRA:-なし}  並列数: $JOBS"
printf "%s\n" $LEVELS | xargs -P "$JOBS" -I{} bash -c 'run_one {}'
python summarize_eval.py "$OUT"
