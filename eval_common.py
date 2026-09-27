"""
評価スクリプト（difficulty.py / panel_difficulty.py / rule_panel.py）で共通に使う小さな関数。

torch や transformers に依存させないため独立したモジュールにしている。
rule_panel.py は子プロセスでこれを読み込むが、Windows（spawn）では子プロセスごとに
import をやり直すので、重いライブラリを引き込むとメモリが足りず固まる。
"""

import json
import os

import numpy as np


def read_levels(levels_arg, manifest_arg):
    """--levels（カンマ区切り）か --levels-from（make_corpus.py の manifest.json）からステージ一覧を得る"""
    if manifest_arg:
        with open(manifest_arg, encoding="utf-8") as f:
            return [m["path"] for m in json.load(f)]
    return [s.strip() for s in levels_arg.split(",") if s.strip()]


def summarize_panel(results, paths):
    by = {}
    for r in results:
        by.setdefault(r["level"], {}).setdefault(r["agent"], []).append(r)
    out = {}
    for lv, per_a in by.items():
        agents = []
        for a, p in enumerate(paths):
            rs = per_a.get(a, [])
            agents.append(dict(model=os.path.basename(p), n=len(rs),
                               clear_rate=float(np.mean([r["cleared"] for r in rs])),
                               progress=float(np.mean([r["progress"] for r in rs]))))
        out[lv] = dict(
            D_panel_clear=float(1 - np.mean([x["clear_rate"] for x in agents])),
            D_panel_progress=float(1 - np.mean([x["progress"] for x in agents])),
            agents=agents,
        )
    return out
