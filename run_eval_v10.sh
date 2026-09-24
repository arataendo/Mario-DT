#!/bin/bash
export PYTHONIOENCODING=utf-8
for L in Level_test_easy Level_test_easy_02 Level_test_easy_03          Level_test_medium Level_test_medium_02 Level_test_medium_03          Level_test_hard Level_test_hard_02 Level_test_hard_03          Level_easy_01 Level_medium_01 Level_hard_01 Level_hard_02; do
  [ -f dt_eval_v10/$L.json ] && continue
  python -u eval_dt_matrix.py --model models/mario_dt_20260921_175028_epoch20.pth --levels $L     --targets 0,60,120,180,235 --episodes 10 --max-steps 500 > dt_eval_v10/$L.log 2>&1
  mv dt_eval_matrix_result.json dt_eval_v10/$L.json
done
touch EVAL_V10_DONE
