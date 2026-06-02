# PPO で Mario ゲームを学習する

このガイドでは、Stable Baselines3 の PPO (Proximal Policy Optimization) を使用して、カスタム Mario ゲーム環境で強化学習を実行する方法を説明します。

## 概要

- **モデル**: PPO (Proximal Policy Optimization)
- **フレームワーク**: Stable Baselines3
- **学習時間**: 100,000 ステップ（4 並列環境で約 1-2 時間）
- **ハードウェア**: CPU で実行可能、CUDA 対応 GPU で高速化可能

## セットアップ

### 1. 必要なパッケージをインストール

```bash
pip install -r requirements.txt
```

**必須パッケージ**:
- `stable-baselines3`: 強化学習フレームワーク
- `torch`: ニューラルネットワークバックエンド
- `gymnasium`: ゲーム環境のインターフェース
- `pygame-ce`: ゲーム実行エンジン
- `numpy`, `scipy`: 数値計算ライブラリ

### 2. 環境の確認

```bash
python test_structure.py  # 基本的な構造テスト
python test_gym_env.py    # 統合テスト
```

## 学習を実行する

### 基本的な使い方

```bash
# デフォルト設定で 100,000 ステップ学習
python train_ppo.py
```

出力例:
```
============================================================
🎮 Mario PPO 学習を開始します
============================================================
📊 設定:
   - レベル: Level1-1
   - 総ステップ数: 100,000
   - 並列環境数: 4
   - ログディレクトリ: ./logs/mario_ppo_level1-1_100k_20240601_123456
   - モデル保存先: ./models/mario_ppo_level1-1_100k.zip

🚀 学習を開始します...
```

### カスタマイズ

```bash
# Level1-2 で 200,000 ステップ学習
python train_ppo.py --level Level1-2 --total-steps 200000

# 8 並列環境を使用（高速化）
python train_ppo.py --num-envs 8

# GPU を使用
python train_ppo.py --device cuda

# カスタムログディレクトリ
python train_ppo.py --log-dir ./my_logs --model-dir ./my_models
```

### パラメータ説明

```
--level LEVEL               : 学習するレベル (デフォルト: Level1-1)
--total-steps N             : 総学習ステップ数 (デフォルト: 100000)
--num-envs N                : 並列環境数 (デフォルト: 4)
--device [auto|cuda|cpu]    : デバイス (デフォルト: auto)
--log-dir DIR               : TensorBoard ログディレクトリ
--model-dir DIR             : モデル保存ディレクトリ
```

## 学習進捗を監視する

### TensorBoard で可視化

```bash
# ログディレクトリを指定
tensorboard --logdir ./logs

# ブラウザで http://localhost:6006 を開く
```

TensorBoard で以下の情報を確認できます:
- **Episode Reward**: エピソードごとの累積報酬
- **Loss**: ポリシーと価値関数の損失
- **Learning Rate**: 学習率の推移

## モデルで推論する

### 基本的な推論

```bash
# 学習済みモデルで 5 エピソード実行
python infer_ppo.py --model models/mario_ppo_level1-1_100k.zip
```

出力例:
```
エピソード 1/5: 報酬=  145.23, ステップ= 842, X=145, コイン=1, 🔄 タイムアップ
エピソード 2/5: 報酬=  138.56, ステップ= 920, X=142, コイン=2, 🔄 タイムアップ
エピソード 3/5: 報酬=  152.34, ステップ=1000, X=156, コイン=0, ✅ クリア

📊 推論結果
✅ 平均報酬: 145.38 (+/- 6.89)
✅ 平均ステップ: 920.7
✅ 最大報酬: 152.34
✅ 最小報酬: 138.56
```

### カスタマイズ

```bash
# 画面に表示して実行
python infer_ppo.py --model models/mario_ppo_level1-1_100k.zip --render

# 異なるレベルでテスト（転移学習）
python infer_ppo.py --model models/mario_ppo_level1-1_100k.zip --level Level1-2

# 10 エピソード実行
python infer_ppo.py --model models/mario_ppo_level1-1_100k.zip --episodes 10

# 確率的な行動を取る（多様性）
python infer_ppo.py --model models/mario_ppo_level1-1_100k.zip --no-deterministic
```

### パラメータ説明

```
--model MODEL               : モデルファイルパス (必須)
--level LEVEL               : プレイするレベル (デフォルト: Level1-1)
--episodes N                : 実行するエピソード数 (デフォルト: 5)
--max-steps N               : エピソードあたりの最大ステップ (デフォルト: 1000)
--render                    : 画面に表示する
--no-deterministic          : 確率的な行動を取る
```

## ハイパーパラメータ

学習可能なハイパーパラメータは `train_ppo.py` で調整できます：

```python
model = PPO(
    learning_rate=3e-4,      # 学習率
    n_steps=2048,            # バッチサイズ = n_steps × num_envs
    batch_size=64,           # SGD バッチサイズ
    n_epochs=10,             # 各バッチの学習エポック数
    gamma=0.99,              # 割引率（遠い報酬の重み）
    gae_lambda=0.95,         # GAE パラメータ（バイアス-分散トレードオフ）
    clip_range=0.2,          # PPO クリップ範囲（更新幅の制限）
    ent_coef=0.01,           # エントロピー係数（探索促進）
    vf_coef=0.5,             # 価値関数損失係数
    max_grad_norm=0.5,       # 勾配クリッピング
)
```

## トラブルシューティング

### メモリ不足エラー

並列環境数を減らす：
```bash
python train_ppo.py --num-envs 2
```

### 学習が遅い

GPU を使用：
```bash
python train_ppo.py --device cuda --num-envs 8
```

### 推論時にモデルが見つからない

モデルが正しく保存されているか確認：
```bash
ls models/mario_ppo*.zip
```

### `pygame.error: (2, 'No such file or directory')`

背景画像ファイルが見つからない場合があります。headless モード（`render_mode=None`）を使用しているため、このエラーは無視して問題ありません。

## 結果の理解

### 報酬スコア

- **進行度報酬**: マリオが右に進むほど増加（0.1 × ピクセル移動距離）
- **ステップペナルティ**: -0.01 × ステップ数（時間を効率よく使うことを促進）
- **コイン報酬**: コイン取得時に +1.0
- **敵撃破報酬**: 敵撃破時に +2.5

平均報酬が 100 以上であれば、良好な学習進捗です。

### エピソードの終了状態

- **✅ クリア**: マリオが十分に進んで level 完了
- **🔄 タイムアップ**: 最大ステップに達した
- **💀 ゲームオーバー**: マリオが敵に接触

## 高度な使用方法

### 複数レベルで転移学習

```bash
# Level1-1 で 100k ステップ
python train_ppo.py --level Level1-1 --total-steps 100000

# Level1-2 で 50k ステップ さらに学習
python train_ppo.py --level Level1-2 --total-steps 50000
# （ただし、同じモデル保存先にすると上書きされるので注意）
```

### モデルを統合してファインチューニング

```python
from stable_baselines3 import PPO

# 既存モデルを読み込んで続ける
model = PPO.load("models/mario_ppo_level1-1_100k", env=env)
model.learn(total_timesteps=50000)
model.save("models/mario_ppo_level1-1_150k")
```

## 参考資料

- [Stable Baselines3 ドキュメント](https://stable-baselines3.readthedocs.io/)
- [Gymnasium（旧 OpenAI Gym）](https://gymnasium.farama.org/)
- [PPO アルゴリズムの論文](https://arxiv.org/abs/1707.06347)

## ライセンス

このコードは参考用です。Mario は Nintendo の商標です。

---

**質問や問題がある場合は、GitHub Issues を作成してください。**
