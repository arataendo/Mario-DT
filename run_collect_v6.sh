#!/bin/bash
# (E) easy/medium を hard と同じ 128本/レベル に揃えるための追加収集。
# 途中でプロセスが落ちても --resume で収集済み分から再開する。
# PC ごと落ちた場合は、このスクリプトをもう一度実行すれば続きから再開できる。
export PYTHONIOENCODING=utf-8
for i in $(seq 1 50); do
  python -u collect_dt_dataset.py     --models models/mario_ppo_level11_checkpoint_5000000_steps_plus_3000k_20260909_004949.zip,models/mario_ppo_level11_checkpoint_5000000_steps.zip     --levels Level_easy_01,Level_easy_02,Level_easy_03,Level_medium_01,Level_medium_02,Level_medium_03     --episodes-per-level 80 --max-frames 8000 --checkpoint-every 10 --resume     --output-dir dt_dataset_v6/frames --output-meta dt_dataset_v6/metadata.pkl && break
  echo "=== 異常終了。10秒後に再開します (試行 $i) ==="
  sleep 10
done
touch COLLECT_V6_DONE
