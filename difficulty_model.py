"""
ステージの配置（穴・パイプ・敵・ブロック）だけから、エージェントにとっての難易度を予測する（代理モデル）。

DT や各エージェント群に実際に遊ばせると、1ステージあたり数十〜数百エピソードかかる。
配置から難易度を直接予測できれば、生成や分析で何千ものステージを一瞬で評価できる。

ステージの表し方:
  - 特徴量: 量（敵・穴・パイプの数）に加えて、配置（障害物の間の助走、敵と障害物の近さ、
    区間ごとの密度、最初の障害物の位置など）を数値にしたもの
  - 列ごとの系列: 各列に「穴か」「パイプの高さ」「クリボー」「ノコノコ」「ブロック」を並べた 5×194 の配列（CNN 用）

ラベル（既存の測定結果を集める。新しくエージェントを動かさない）:
  - 評価コーパス 60 ステージ（validity_out/v1/）
  - 生成実験 v1〜v4 の検証ステージ（gen_out/v*/）。v3 と v4 の「つまみ1個」は同じステージなので1つにまとめる
  ラベルは各エージェント（群）の D_progress（1 − 平均到達率）

評価は、同じ探索から生まれたステージ（似ている）が学習と評価に分かれないよう、グループ単位の交差検証で行う。

使用例:
    python difficulty_model.py            # 交差検証で、どれだけ当たるかを表示
"""

import argparse
import hashlib
import json
import os

import numpy as np
from scipy.stats import spearmanr

from level_editor import EditableLevel

CHANNELS = ("穴", "パイプ", "クリボー", "ノコノコ", "ブロック")
TARGETS = {"dt": "DT", "ppo": "PPO 群", "rule": "ルール群", "planner": "プランナー群"}


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# ステージの表し方
# ---------------------------------------------------------------------------

def _block_columns(lv):
    cols = []
    for name, items in lv.extras.items():
        if isinstance(items, list):
            cols += [it[0] for it in items if isinstance(it, list) and it]
    return cols


def encode(data, length=194):
    """列ごとの系列 (len(CHANNELS), length)"""
    lv = EditableLevel.from_json(data)
    a = np.zeros((len(CHANNELS), length), dtype=np.float32)
    n = min(lv.length, length)
    a[0, :n] = [0.0 if g else 1.0 for g in lv.ground[:n]]
    for x, h in lv.pipes.items():
        a[1, x:min(x + 2, length)] = h / 3.0
    for ch, xs in ((2, lv.goombas), (3, lv.koopas), (4, _block_columns(lv))):
        for x in xs:
            if 0 <= x < length:
                a[ch, x] += 1.0
    return a


def features(data):
    """配置を数値にした特徴量（名前 → 値）"""
    lv = EditableLevel.from_json(data)
    L = lv.length
    s = lv.summary()
    runs = lv.gap_runs()
    hz = lv.hazards()
    enemies = sorted(lv.goombas + lv.koopas)
    blocks = _block_columns(lv)
    f = dict(s)
    f["max_gap"] = max((w for _, w in runs), default=0)
    f["pipe_height_sum"] = sum(lv.pipes.values())
    f["blocks"] = len(blocks)
    # 障害物の間の助走（短いほど連続して跳ぶ必要がある）
    gaps_between = [b[0] - a[1] for a, b in zip(hz, hz[1:])]
    f["min_runway"] = min(gaps_between, default=L)
    f["tight_runways"] = sum(g <= 3 for g in gaps_between)
    # 最初の障害物・敵の位置（序盤ほど弱いプレイヤーが止まる）
    f["first_hazard"] = (hz[0][0] / L) if hz else 1.0
    f["first_enemy"] = (enemies[0] / L) if enemies else 1.0
    # 敵と障害物の組み合わせ
    f["enemy_near_hazard"] = sum(any(a - 3 <= x < b + 3 for a, b in hz) for x in enemies)
    f["enemy_pairs_close"] = sum(b - a <= 4 for a, b in zip(enemies, enemies[1:]))
    f["koopa_near_pipe"] = sum(any(abs(x - px) <= 3 for px in lv.pipes) for x in lv.koopas)
    f["block_over_hazard"] = sum(any(a - 2 <= x < b + 2 for a, b in hz) for x in blocks)
    # 区間ごとの密度（4分割）と、20 列の窓での最大の混み具合
    load_col = np.zeros(L)
    for a, b in hz:
        load_col[a:b] += 1
    for x, h in lv.pipes.items():
        load_col[x] += h
    for x in enemies:
        load_col[min(x, L - 1)] += 1
    for q in range(4):
        f[f"load_q{q + 1}"] = float(load_col[q * L // 4:(q + 1) * L // 4].sum())
    win = np.convolve(load_col, np.ones(20), mode="valid")
    f["max_window_load"] = float(win.max()) if len(win) else 0.0
    return f


# ---------------------------------------------------------------------------
# ラベル付きステージを集める
# ---------------------------------------------------------------------------

def _panel_D(res, key):
    return res[key]["D_panel_progress"] if res and key in res else None


def collect_labeled():
    """[{path, data, labels{target: [測定値, ...]}, group}]（同じ内容のステージはまとめる）"""
    items = {}

    def add(path, group, labels):
        data = load(path)
        h = hashlib.sha1(json.dumps(data["level"], sort_keys=True).encode()).hexdigest()
        it = items.setdefault(h, dict(path=path, data=data, labels={}, group=group))
        for k, v in labels.items():
            if v is None:
                continue
            vals = it["labels"].setdefault(k, [])
            # 同じステージを同じシードで測った値（v3 と v4 の「つまみ1個」など）は完全に一致するので、
            # 2回目として数えない（数えると「測り直しの一致」が見かけ上 1 になる）
            if not any(abs(float(v) - u) < 1e-12 for u in vals):
                vals.append(float(v))

    def opt(path):
        return load(path) if os.path.exists(path) else None

    # 評価コーパス
    c = "validity_out/v1"
    dt, dtr = load(f"{c}/dt.json"), opt(f"{c}/dt_retest.json")
    pp, ru, pl = load(f"{c}/panel.json"), opt(f"{c}/rule_panel.json"), opt(f"{c}/planner_panel.json")
    for m in load("corpus/v1/manifest.json"):
        p = m["path"]
        add(p, f"corpus:{p}", dict(dt=dt[p]["D_progress"], ppo=_panel_D(pp, p),
                                   rule=_panel_D(ru, p), planner=_panel_D(pl, p)))
        if dtr:
            add(p, f"corpus:{p}", dict(dt=dtr[p]["D_progress"]))

    # 生成実験 v1〜v4
    for v in ("v1", "v2", "v3", "v4"):
        g = f"gen_out/{v}"
        if not os.path.exists(f"{g}/validate_manifest.json"):
            continue
        vd, vp, vr, vpl = (opt(f"{g}/{n}.json") for n in ("val_dt", "val_ppo", "val_rule", "val_planner"))
        retests = [opt(f"{g}/{n}.json") for n in ("val_ppo_retest", "val_planner_retest")]
        for m in load(f"{g}/validate_manifest.json"):
            p = m["path"]
            if not os.path.exists(p):
                continue
            # 同じ探索（v3/v4 は run、v1/v2 は目標ごとの1本の探索）から生まれたものは同じグループ
            if "run" in m:
                group = f"run:{m['run']}"
            elif m.get("method") == "search":
                group = f"{v}:search:{m['target']}"
            else:
                group = f"{v}:{p}"
            add(p, group, dict(dt=vd[p]["D_progress"] if vd and p in vd else None,
                               ppo=_panel_D(vp, p), rule=_panel_D(vr, p), planner=_panel_D(vpl, p)))
            for name, r in zip(("ppo", "planner"), retests):
                if r and p in r:
                    add(p, group, {name: _panel_D(r, p)})
    return list(items.values())


def load_extra(path):
    """label_stages.py が付けた大量のラベル（DT のみ、1ステージ1回の測定）"""
    from label_stages import from_compact
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                items.append(dict(path=r["path"], data=from_compact(r["level"]).to_json(),
                                  labels={"dt": [r["D_progress"]]}, group=f"extra:{r['id']}"))
    return items


# ---------------------------------------------------------------------------
# モデル
# ---------------------------------------------------------------------------

class Ridge:
    """標準化した特徴量でのリッジ回帰。正則化の強さは学習データの中の一個抜きで選ぶ"""

    def fit(self, X, y, alphas=(0.1, 0.3, 1, 3, 10, 30, 100)):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = np.column_stack([np.ones(len(X)), (X - self.mu) / self.sd])
        best = None
        for a in alphas:
            P = np.eye(Z.shape[1]) * a
            P[0, 0] = 0
            H = Z @ np.linalg.solve(Z.T @ Z + P, Z.T)
            loo = ((y - H @ y) / (1 - np.diag(H) + 1e-9)) ** 2
            if best is None or loo.mean() < best[0]:
                best = (loo.mean(), a)
        P = np.eye(Z.shape[1]) * best[1]
        P[0, 0] = 0
        self.w = np.linalg.solve(Z.T @ Z + P, Z.T @ y)
        self.alpha = best[1]
        return self

    def predict(self, X):
        return np.column_stack([np.ones(len(X)), (X - self.mu) / self.sd]) @ self.w


class CNN:
    """列ごとの系列を読む小さな 1 次元 CNN（何本か学習して平均する）"""

    def __init__(self, n_models=5, epochs=400, lr=3e-3, wd=1e-3):
        self.n_models, self.epochs, self.lr, self.wd = n_models, epochs, lr, wd

    @staticmethod
    def _net():
        import torch
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv1d(len(CHANNELS), 32, 5, padding=2), nn.ReLU(),
                    nn.Conv1d(32, 32, 5, padding=4, dilation=2), nn.ReLU(),
                    nn.Conv1d(32, 32, 5, padding=8, dilation=4), nn.ReLU())
                self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

            def forward(self, x):
                h = self.conv(x)
                # 列方向に平均と最大をとる（どこにあっても同じように効く特徴 + いちばん厳しい箇所）
                return self.head(torch.cat([h.mean(-1), h.amax(-1)], dim=-1)).squeeze(-1)

        return Net()

    def fit(self, X, y):
        import torch
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Xt, yt = torch.tensor(X, device=self.dev), torch.tensor(y, dtype=torch.float32, device=self.dev)
        self.models = []
        for k in range(self.n_models):
            torch.manual_seed(k)
            net = self._net().to(self.dev)
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.wd)
            for _ in range(self.epochs):
                opt.zero_grad()
                loss = ((net(Xt) - yt) ** 2).mean()
                loss.backward()
                opt.step()
            net.eval()
            self.models.append(net)
        return self

    def predict(self, X):
        import torch
        with torch.no_grad():
            Xt = torch.tensor(X, device=self.dev)
            return np.mean([m(Xt).cpu().numpy() for m in self.models], axis=0)


# ---------------------------------------------------------------------------
# 評価
# ---------------------------------------------------------------------------

def group_folds(groups, k=5, seed=0):
    uniq = sorted(set(groups))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    fold_of = {g: i % k for i, g in enumerate(uniq)}
    return np.array([fold_of[g] for g in groups])


def cv_predict(make, X, y, folds):
    pred = np.zeros(len(y))
    for f in sorted(set(folds)):
        tr, te = folds != f, folds == f
        pred[te] = make().fit(X[tr], y[tr]).predict(X[te])
    return pred


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main():
    ap = argparse.ArgumentParser(description="配置から難易度を予測するモデルを交差検証で評価する")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--no-cnn", action="store_true")
    ap.add_argument("--extra", default=None,
                    help="label_stages.py の labels.jsonl。指定すると、それで学習して既存のステージで評価する")
    a = ap.parse_args()
    if a.extra:
        return evaluate_extra(a)

    items = collect_labeled()
    names = list(features(items[0]["data"]))
    F = np.array([[features(it["data"])[n] for n in names] for it in items], dtype=float)
    S = np.stack([encode(it["data"]) for it in items])
    base_cols = [names.index(n) for n in ("enemies", "gap_tiles", "pipes")]
    groups = [it["group"] for it in items]
    print(f"ラベル付きステージ {len(items)} 個（グループ {len(set(groups))} 個）、特徴量 {len(names)} 個\n")

    print(f"{'予測する難易度':14s} {'n':>4s}  {'量だけ（線形）':>16s}  {'配置の特徴量（線形）':>20s}"
          + ("" if a.no_cnn else f"  {'CNN（列の系列）':>18s}") + "   測り直しの一致（上限の目安）")
    for key, label in TARGETS.items():
        idx = [i for i, it in enumerate(items) if key in it["labels"]]
        if len(idx) < 20:
            continue
        y = np.array([np.mean(items[i]["labels"][key]) for i in idx])
        folds = group_folds([groups[i] for i in idx], a.folds)
        cols = []
        for X, make in ((F[idx][:, base_cols], Ridge), (F[idx], Ridge)) + (() if a.no_cnn else ((S[idx], CNN),)):
            p = cv_predict(make, X, y, folds)
            cols.append(f"ρ={spearmanr(y, p).statistic:+.2f} R²={r2(y, p):+.2f}")
        # 2回測ったステージでの、1回目と2回目の一致（ラベル自体の揺れ。予測はこれを超えにくい）
        rep = [items[i]["labels"][key][:2] for i in idx if len(items[i]["labels"][key]) >= 2]
        ceil = f"ρ={spearmanr([r[0] for r in rep], [r[1] for r in rep]).statistic:+.2f}（{len(rep)} 個）" if len(rep) >= 10 else "—"
        print(f"{label:14s} {len(idx):4d}  " + "  ".join(f"{c:>18s}" for c in cols) + f"   {ceil}")

    # 線形モデルで効いている特徴量（DT）
    idx = [i for i, it in enumerate(items) if "dt" in it["labels"]]
    y = np.array([np.mean(items[i]["labels"]["dt"]) for i in idx])
    m = Ridge().fit(F[idx], y)
    order = np.argsort(-np.abs(m.w[1:]))
    print(f"\nDT の難易度に効く特徴量（標準化した係数、大きい順。正則化 {m.alpha}）")
    for j in order[:12]:
        print(f"  {names[j]:20s} {m.w[1 + j]:+.3f}")


def evaluate_extra(a):
    """大量のラベル（1回測定、粗い）で学習し、既存のステージ（DT を2回・10本ずつ測った精密なラベル）で評価する。
    学習に使う数を変えて、データを増やすと当たり方がどう変わるかも見る"""
    extra = load_extra(a.extra)
    test = [it for it in collect_labeled() if "dt" in it["labels"]]
    names = list(features(test[0]["data"]))
    feat = lambda its: np.array([[features(it["data"])[n] for n in names] for it in its], dtype=float)
    Ftr, Fte = feat(extra), feat(test)
    Str, Ste = np.stack([encode(it["data"]) for it in extra]), np.stack([encode(it["data"]) for it in test])
    ytr = np.array([it["labels"]["dt"][0] for it in extra])
    yte = np.array([np.mean(it["labels"]["dt"]) for it in test])
    base = [names.index(n) for n in ("enemies", "gap_tiles", "pipes")]
    print(f"学習: {a.extra} の {len(extra)} 個 → 評価: 既存の {len(test)} 個（DT の難易度）\n")
    print(f"{'学習に使う数':>10s}  {'量だけ（線形）':>16s}  {'配置の特徴量（線形）':>20s}"
          + ("" if a.no_cnn else f"  {'CNN（列の系列）':>18s}"))
    rng = np.random.default_rng(0)
    order = rng.permutation(len(extra))
    sizes = [n for n in (250, 500, 1000, 2000) if n < len(extra)] + [len(extra)]
    for n in sizes:
        idx = order[:n]
        cols = []
        models = ((Ftr[:, base], Fte[:, base], Ridge), (Ftr, Fte, Ridge))
        for Xtr, Xte, make in models + (() if a.no_cnn else ((Str, Ste, CNN),)):
            p = make().fit(Xtr[idx], ytr[idx]).predict(Xte)
            cols.append(f"ρ={spearmanr(yte, p).statistic:+.2f} R²={r2(yte, p):+.2f}")
        print(f"{n:10d}  " + "  ".join(f"{c:>18s}" for c in cols))


if __name__ == "__main__":
    main()
