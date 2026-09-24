#!/bin/bash
# v8 と同じ単一系統 (6.5M〜8.0M) の方策に ε ランダム行動を混ぜ、低〜中収益の手本を作る。
# v8 は 74% が最高収益帯で低収益データが無く、target を下げても手抜きしなかった（条件付けが弱い）。
# 別系統を混ぜず同じ方策を劣化させるので、v5/v7 で起きた模倣対象の二峰化は起きない。
# 中断しても --resume で続きから。PC ごと落ちたらこのスクリプトを再実行。
export PYTHONIOENCODING=utf-8
for i in $(seq 1 50); do
  python -u collect_dt_dataset.py     --models models/mario_ppo_level11_checkpoint_6500000_steps.zip,models/mario_ppo_level11_checkpoint_6800000_steps.zip,models/mario_ppo_level11_checkpoint_7100000_steps.zip,models/mario_ppo_level11_checkpoint_7400000_steps.zip,models/mario_ppo_level11_checkpoint_7700000_steps.zip,models/mario_ppo_level11_checkpoint_8000000_steps.zip     --episodes-per-level 36 --epsilons 0.3,0.6,0.9 --max-frames 4000     --checkpoint-every 10 --resume     --output-dir dt_dataset_v9/frames --output-meta dt_dataset_v9/metadata.pkl && break
  echo "=== 異常終了。10秒後に再開します (試行 $i) ==="
  sleep 10
done
touch COLLECT_V9_DONE
