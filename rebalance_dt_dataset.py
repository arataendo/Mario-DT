"""
DT 学習用データセットの収益分布をリバランスする。

背景 (diagnose_rtg.py / diagnose_rtg_drift.py の診断結果):
  元の報酬設計はクリア時 +100 / 死亡時 -25 という大きな終端報酬を持つため、
  エピソード総収益が
      非クリア: 〜156.7   クリア: 269.2 〜
  の二峰分布になり、間の 112.5 幅が「構造的に到達不可能」だった。
  DT は target_return を連続的なスキル指標として使うため、この空白があると
  中間の target が外挿になり、条件付けが機能しない。

  この空白は追加のデータ収集では埋まらない（到達不可能なので）。
  終端報酬を付け替えて収益を連続化するのが正しい対処。

  フレーム画像はそのまま再利用できるので、再収集は不要。
  終端ステップの報酬を差し替えて returns_to_go を計算し直すだけでよい。

既定値 (--clear-bonus 25 --death-penalty 0) は幅25のビンで空白が消え、
かつクリア群 (>=194.2) が非クリア群 (<=181.7) より必ず高くなる値。

使用方法:
    python rebalance_dt_dataset.py --input dt_dataset_v3/metadata.pkl \
        --output dt_dataset_v3/metadata_rebalanced.pkl
"""

import argparse
import pickle

import numpy as np

# 収集時に MarioGymEnv._calculate_reward が終端ステップへ加算していた値
ORIG_CLEAR_BONUS = 100.0
ORIG_DEATH_PENALTY = 25.0
ORIG_TIME_PENALTY = 0.01   # 収集時に毎ステップ引かれていた時間ペナルティ


def drop_inconsistent(episodes):
    """クリアしていないのにクリア群より高収益なエピソードを除く。

    報酬にはコイン・スコア由来の項があり、これは進行度と違って
    「同じ場所に留まったまま敵を倒し続ける」だけで際限なく積み上がる。
    実際、上限 2000 ステップまで走った timeout エピソードが進行度 87% のまま
    return 304〜310 を記録し、クリア群 (194〜237) を上回っていた。

    こうしたエピソードを高収益サンプルとして残すと、DT は高い target_return に対して
    「先へ進まずスコアを稼ぐ」挙動を模倣してしまい、収益条件付けの意味が壊れる。
    件数は少ないので単純に除外する。
    （恒久対策としては報酬側でスコア項に上限を設けるべきだが、
      それには再収集が必要なのでここでは扱わない）
    """
    clears = [ep for ep in episodes if ep.get("final_reason") == "level_complete"]
    if not clears:
        return episodes, []
    min_clear = min(float(ep["rewards"].sum()) for ep in clears)
    kept, dropped = [], []
    for ep in episodes:
        if ep.get("final_reason") != "level_complete" and float(ep["rewards"].sum()) >= min_clear:
            dropped.append(ep)
        else:
            kept.append(ep)
    return kept, dropped


def rebalance(episodes, clear_bonus, death_penalty, time_penalty=ORIG_TIME_PENALTY):
    """終端報酬と時間ペナルティを付け替え、returns_to_go を計算し直す。

    time_penalty を上げる理由:
      元の 0.01/step では、311歩のクリアと783歩のクリアの収益差がわずか 4.7 しかなく、
      全クリアが幅40の狭い帯に潰れていた（σ=6.4）。
      その結果 target を 180 から 235 へ上げても指す内容がほとんど変わらず、
      最高 target 付近はデータのない外挿になっていた。
      ペナルティを上げるとクリアの「速さ・無駄のなさ」が収益に反映され、
      高収益帯が広がって target による細かい制御が効くようになる。
    """
    out = []
    for ep in episodes:
        if len(ep["actions"]) == 0:
            continue
        rewards = ep["rewards"].astype(np.float32).copy()
        # 時間ペナルティの差分を全ステップに適用（元の 0.01 を剥がして新しい値を課す）
        if time_penalty != ORIG_TIME_PENALTY:
            rewards -= (time_penalty - ORIG_TIME_PENALTY)
        cleared = ep.get("final_reason") == "level_complete"

        # 終端ステップに乗っている元のボーナス/ペナルティを剥がして新しい値を乗せる。
        # タイムアウト打ち切りの場合はどちらも乗っていないので触らない。
        if cleared:
            rewards[-1] += -ORIG_CLEAR_BONUS + clear_bonus
        elif ep.get("final_reason") == "game_over":
            rewards[-1] += ORIG_DEATH_PENALTY - death_penalty

        new_ep = dict(ep)
        new_ep["rewards"] = rewards
        new_ep["returns_to_go"] = np.cumsum(rewards[::-1])[::-1].astype(np.float32)
        out.append(new_ep)
    return out


def report(episodes, title):
    ret = np.array([ep["rewards"].sum() for ep in episodes])
    reason = np.array([ep.get("final_reason", "?") for ep in episodes])
    clear = reason == "level_complete"
    print("=" * 70)
    print(title)
    print("=" * 70)
    print(f"  return: min={ret.min():.1f} 中央={np.median(ret):.1f} max={ret.max():.1f} 平均={ret.mean():.1f}")
    if clear.any() and (~clear).any():
        gap = ret[clear].min() - ret[~clear].max()
        print(f"  非クリア最大={ret[~clear].max():.1f} / クリア最小={ret[clear].min():.1f} → ギャップ={gap:.1f}")
    lo = int(np.floor(ret.min() / 25) * 25)
    hi = int(np.ceil(ret.max() / 25) * 25)
    for b in range(lo, hi, 25):
        n = int(((ret >= b) & (ret < b + 25)).sum())
        flag = "  ← 空白" if n < max(1, len(ret) * 0.01) else ""
        print(f"    [{b:4d},{b + 25:4d}): {n:4d} ({n / len(ret) * 100:5.1f}%) "
              f"{'#' * int(60 * n / len(ret))}{flag}")
    print()


def main():
    parser = argparse.ArgumentParser(description="DTデータセットの収益分布を連続化する")
    parser.add_argument("--input", type=str, default="dt_dataset_v3/metadata.pkl")
    parser.add_argument("--output", type=str, default="dt_dataset_v3/metadata_rebalanced.pkl")
    parser.add_argument("--clear-bonus", type=float, default=25.0,
                        help="クリア時に終端ステップへ与える報酬（元は100）")
    parser.add_argument("--death-penalty", type=float, default=0.0,
                        help="死亡時に終端ステップから引く報酬（元は25）")
    parser.add_argument("--time-penalty", type=float, default=ORIG_TIME_PENALTY,
                        help="1ステップあたりの時間ペナルティ（元は0.01）。"
                             "上げるとクリアの速さが収益に反映され、高収益帯が広がる")
    parser.add_argument("--keep-inconsistent", action="store_true",
                        help="クリア群より高収益な非クリアエピソード（報酬ファーミング）を除外しない")
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        episodes = [ep for ep in pickle.load(f) if len(ep["actions"]) > 0]
    report(episodes, f"変更前: {args.input} ({len(episodes)} エピソード)")

    new_eps = rebalance(episodes, args.clear_bonus, args.death_penalty, args.time_penalty)

    if not args.keep_inconsistent:
        new_eps, dropped = drop_inconsistent(new_eps)
        if dropped:
            print(f"⚠️  クリア群より高収益な非クリアエピソードを {len(dropped)} 本除外しました")
            for ep in dropped:
                print(f"     level={ep['level']:16s} reason={ep.get('final_reason'):10s} "
                      f"steps={len(ep['actions']):4d} return={ep['rewards'].sum():7.1f} "
                      f"progress={ep.get('progress', 0) * 100:5.1f}%")
            print()

    report(new_eps, f"変更後: clear_bonus={args.clear_bonus} death_penalty={args.death_penalty}")

    with open(args.output, "wb") as f:
        pickle.dump(new_eps, f)
    print(f"💾 保存しました: {args.output} ({len(new_eps)} エピソード)")
    print("   ※ フレーム画像は元のまま共有しているので再収集は不要です")


if __name__ == "__main__":
    main()
