"""
先読みプランナーのエージェント群（planner_agent.py）でステージの難易度を測る。

DT・ルール・PPO のどれとも仕組みが違う「審判」。出力形式・環境シードの規則は
rule_panel.py / panel_difficulty.py と同じで、analyze_* にそのまま渡せる。

使用例:
    python planner_panel.py --levels-from corpus/v1/manifest.json --episodes 4 --workers 8         --output validity_out/v1/planner_panel.json
    python planner_panel.py --calibrate --episodes 2   # 学習用ステージで腕前の幅を確かめる
"""

import planner_agent
from rule_panel import main

if __name__ == "__main__":
    main(presets=planner_agent.PRESETS, run_fn=planner_agent.run_episode, seed_offset=900,
         desc="先読みプランナーのエージェント群でステージの難易度を測る")
