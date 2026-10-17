#!/bin/bash
# DT の学習ステージを増やす（11 個 → 生成した N_LEVELS 個）ためのデータ収集（CPU サーバー用）。
#
#   今の DT は学習データが 11 ステージ分しかなく、知らないステージで上限が低い（temp_out/report.txt）。
#   教師は今の DT と同じ PPO（level11 系統 650万〜800万ステップの6時点）のまま、ステージだけを増やす。
#   こうすると、変わるのは「ステージの多様さ」だけになる。
#
#   1. 学習用ステージを生成（シード 1000000〜。評価コーパス・生成実験・他の学習用と重ならない）
#   2. PPO で収集（ステージごとに 7 本、ε = 0,0,0,0,0.3,0.6,0.9。今のデータの ε=0 の割合 64% に近い 57%）
#   3. 研究室 PC に運ぶための tar を作る
#
# 使い方:
#   WORKERS=48 ./run_div_collect.sh
# 1エピソードずつ保存するので、止まっても再実行で続きから走る。
set -euo pipefail
cd "$(dirname "$0")"
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f .venv/bin/activate ]; then
  set +u; source .venv/bin/activate; set -u
fi
command -v python >/dev/null || { echo "❌ python が見つかりません。source .venv/bin/activate を実行してください"; exit 1; }

WORKERS=${WORKERS:-48}
N_LEVELS=${N_LEVELS:-1000}
DATA=${DATA:-dt_dataset_div}
TEACHERS=""
for s in 6500000 6800000 7100000 7400000 7700000 8000000; do
  f=models/mario_ppo_level11_checkpoint_${s}_steps.zip
  [ -f "$f" ] || { echo "❌ 教師のモデル $f がありません（元の PC から models/ に送ってください）"; exit 1; }
  TEACHERS="$TEACHERS${TEACHERS:+,}$f"
done
export PYTHONIOENCODING=utf-8 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # 多数の子プロセスでスレッドを奪い合わない
step() { echo; echo "======== $* ========  $(date '+%m/%d %H:%M')"; }

step "1/3 学習用ステージを生成（$N_LEVELS 個）"
[ -f corpus/train_div/manifest.json ] && echo "既存" || python make_corpus.py --name train_div \
  --count "$N_LEVELS" --seed-base 1000000

step "2/3 PPO で収集（$N_LEVELS ステージ × 7 本）"
[ -f "$DATA/metadata.pkl" ] && echo "既存" || python -u collect_planner_dataset.py \
  --levels-from corpus/train_div/manifest.json --teacher "$TEACHERS" \
  --episodes-per-level 7 --epsilons 0,0,0,0,0.3,0.6,0.9 --workers "$WORKERS" --out "$DATA" \
  2>&1 | tee -a "$DATA.collect.log"

step "3/3 研究室 PC に運ぶ tar を作る"
[ -f "$DATA.tar" ] && echo "既存" || { tar -cf "$DATA.tar" "$DATA/metadata.pkl" "$DATA/frames" "$DATA.collect.log" \
  && sha256sum "$DATA.tar" > "$DATA.tar.sha256"; }
ls -lh "$DATA.tar"
echo
echo "研究室 PC で:  scp <このサーバー>:$(pwd)/$DATA.tar{,.sha256} ~/Mario-DT/  →  ./run_div_train.sh"
