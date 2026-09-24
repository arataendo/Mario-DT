#!/bin/bash
# (G1) 単一系統・スタイル収束後のチェックポイント群 (6.5M〜8.0M) だけから DT 用データを収集する。
# 新旧2系統を混ぜた v5/v7 で easy が崩れた原因（模倣対象の二峰化）を取り除くのが目的。
# 中断しても --resume で続きから再開できる。PC ごと落ちた場合はこのスクリプトを再実行する。
export PYTHONIOENCODING=utf-8
for i in $(seq 1 50); do
  python -u collect_dt_dataset.py     --models models/mario_ppo_level11_checkpoint_6500000_steps.zip,models/mario_ppo_level11_checkpoint_6800000_steps.zip,models/mario_ppo_level11_checkpoint_7100000_steps.zip,models/mario_ppo_level11_checkpoint_7400000_steps.zip,models/mario_ppo_level11_checkpoint_7700000_steps.zip,models/mario_ppo_level11_checkpoint_8000000_steps.zip     --episodes-per-level 64 --max-frames 8000 --checkpoint-every 10 --resume     --output-dir dt_dataset_v8/frames --output-meta dt_dataset_v8/metadata.pkl && break
  echo "=== 異常終了。10秒後に再開します (試行 $i) ==="
  sleep 10
done
touch COLLECT_V8_DONE
