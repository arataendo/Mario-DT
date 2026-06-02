# Mario Game - Gymnasium 機械学習環境

マリオゲームを OpenAI Gymnasium 環境として実装し、機械学習エージェントで訓練可能にしました。

## 実装内容

### ✅ 完了した変更

#### 1. **Gymnasium環境の実装** (`classes/MarioGymEnv.py`)
- Gymnasium v1.3+ 互換の `MarioEnv` クラス
- Dict 型の観測空間: `{'image': RGB画像, 'state': 状態ベクトル}`
- Discrete(8) アクション空間
- 進行度ベースの報酬関数
- `render_mode` サポート: `'human'`, `'rgb_array'`, `None`

#### 2. **エージェント入力マッピング** (`classes/AgentInput.py`)
- 8つのアクションから Mario の入力トレイトへの変換
- 行動マッピング:
  - 0: NOP
  - 1: Left, 2: Right, 3: Jump
  - 4: Left+Jump, 5: Right+Jump
  - 6: Dash, 7: Right+Dash

#### 3. **Mario.py の改造**
- `input_source` パラメータ追加 (`'human'` / `'agent'`)
- 入力ソースに応じた処理の切り替え

#### 4. **Headless (画面非表示) 対応**
- `Level.py`: `screen=None` サポート
- `Dashboard.py`: 画面非表示時のスキップ処理
- `Pause.py`: pygame リソース初期化の条件分岐

#### 5. **依存パッケージ更新** (`requirements.txt`)
```
pygame>=2.0.0
scipy>=1.4.1
gymnasium>=0.26.0
numpy>=1.19.0
```

### 📋 観測空間

```python
{
    'image': Box(0, 255, (3, 480, 640), uint8),  # RGB画像
    'state': Box(-inf, inf, (9,), float32),      # 状態ベクトル
}
```

**状態ベクトルの要素:**
```python
[
    mario_x,           # X座標 (正規化)
    mario_y,           # Y座標
    velocity_x,        # 速度X
    velocity_y,        # 速度Y
    camera_x,          # カメラ位置
    progress,          # 進行度 (0-1)
    coins_collected,   # コイン数
    entities_count,    # エンティティ数
    powerup_state,     # パワーアップ状態 (0=小, 1=大)
]
```

### 🎮 アクション空間

```
Discrete(8)
0: NOP (何もしない)
1: Left (左移動)
2: Right (右移動)
3: Jump (ジャンプ)
4: Left + Jump
5: Right + Jump
6: Dash (ダッシュ/スプリント)
7: Right + Dash
```

### 🏆 報酬関数

```python
基本報酬 = 進行度_変化 * 0.1     # 右への移動距離に基づく
時間ペナルティ = -0.01 per step
コイン報酬 = +1.0 (1コイン)
敵撃破ボーナス = ポイント/100
ゲームオーバー = -10.0
ステージクリア = +100.0
```

## セットアップ

### 前提条件
- Python 3.11 または 3.12 (pygame 互換性のため Python 3.14+ は避ける)
- 仮想環境推奨

### インストール

```bash
# 1. 仮想環境作成
python -m venv .venv
.venv/Scripts/activate  # Windows
source .venv/bin/activate  # macOS/Linux

# 2. 依存パッケージをインストール
pip install -r requirements.txt

# または個別にインストール
pip install gymnasium numpy scipy
pip install pygame  # Python 3.11/3.12 推奨
```

## 使用方法

### 基本的な例

```python
from classes.MarioGymEnv import MarioEnv

# 環境作成（headless モード）
env = MarioEnv(level='Level1-1', render_mode=None)

# リセット
obs, info = env.reset()

# ステップ実行
action = env.action_space.sample()  # ランダムアクション
obs, reward, terminated, truncated, info = env.step(action)

env.close()
```

### 強化学習での使用 (Stable Baselines3 例)

```python
from classes.MarioGymEnv import MarioEnv
from stable_baselines3 import PPO

env = MarioEnv(random_level=True, render_mode=None)

# モデル作成
model = PPO("MultiInputPolicy", env, verbose=1)

# 訓練
model.learn(total_timesteps=100000)

# 推論
obs, info = env.reset()
while True:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

## テスト

### 構造テスト（pygame 不要）
```bash
python test_structure.py
```

このテストは以下を確認します:
- Gymnasium API の互換性
- AgentInput アクション変換の正確性
- 観測/アクション空間の定義
- モジュール間の依存関係

**予想結果:** すべてのテストが ✓ PASS

### 統合テスト（pygame 必須）
```bash
python test_gym_env.py
```

このテストは以下を確認します:
- 環境の初期化とリセット
- ステップ実行ループ
- ランダムエージェント実行
- 複数エピソード実行
- 異なる render_mode での動作

## 既存ゲーム互換性

既存の `main.py` は変更なしで動作します（人間プレイモード）。

```bash
python main.py
```

## ファイル構成

```
classes/
  ├── MarioGymEnv.py          # ★ 新規: Gymnasium 環境
  ├── AgentInput.py           # ★ 新規: Gym アクション→入力変換
  ├── Input.py                # 既存: 人間入力（変更なし）
  ├── Level.py                # 修正: headless対応
  ├── Dashboard.py            # 修正: headless対応
  └── Pause.py                # 修正: headless対応
entities/
  └── Mario.py                # 修正: input_source パラメータ追加

test_structure.py             # ★ 新規: 構造テスト
test_gym_env.py               # ★ 新規: 統合テスト
GYM_QUICK_START.py            # ★ 新規: クイックスタートガイド
requirements.txt              # 修正: gymnasium追加
```

## 既知の制限事項

### pygame のビルド問題
- Python 3.14 では pygame をソースからビルドできません（distutils 削除）
- **解決策:** Python 3.11 または 3.12 を使用
- 代替案: より新しいバージョンの pygame-ce (Community Edition) を試す

### パフォーマンス
- `render_mode=None` (headless) で最速訓練
- `render_mode='human'` では画面描画による遅延あり
- 大規模訓練は GPU 対応マシンで実行を推奨

## 今後の拡張案

1. **Wrapper実装**
   - `FrameStackWrapper`: 複数フレームをスタック
   - `NormalizeObservationWrapper`: 状態正規化
   - `NormalizeRewardWrapper`: 報酬正規化

2. **報酬エンジニアリング**
   - カスタマイズ可能な報酬関数
   - 進行度レベルの調整

3. **マルチレベル対応**
   - すべてのレベルのランダム選択
   - レベル難度の段階的増加

4. **Video録画**
   - `gymnasium.wrappers.RecordVideo` 対応

5. **並列訓練**
   - `stable-baselines3` の `SubprocVecEnv`

## 参考リンク

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- [pygame Documentation](https://www.pygame.org/docs/)

## トラブルシューティング

### pygame インストールエラー
```
ModuleNotFoundError: No module named 'distutils.msvccompiler'
```
**原因:** Python 3.14 で distutils が削除
**解決策:** Python 3.11 または 3.12 へダウングレード

### gym/gymnasium インポートエラー
```
ModuleNotFoundError: No module named 'gym'
```
**原因:** パッケージがインストールされていない
**解決策:** `pip install gymnasium`

### 観測値が NaN
**原因:** ゲームが初期化されていない
**解決策:** `env.reset()` を呼び出す前に `step()` を呼ばない

## ライセンス

このゲームは教育目的のプロジェクトです。元のマリオゲーム資産は任天堂に帰属します。

---

**実装日:** 2026年6月2日  
**対応Gymnasium:** v1.3.0+  
**対応Python:** 3.11, 3.12
