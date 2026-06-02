# PPO で Mario ゲームを学習 - 実装完成

## 概要

Stable Baselines3 の PPO (Proximal Policy Optimization) を使用して、カスタム Mario ゲーム環境で強化学習を実行するための完全なシステムを実装しました。

## 実装内容

### 作成したファイル

#### 1. **train_ppo.py** - メイン学習スクリプト
- PPO エージェントで 100,000 ステップ（またはカスタム値）学習
- 複数 CPU コアで並列化（デフォルト: 4 環境）
- GPU サポート（CUDA/CPU 自動選択）
- TensorBoard でのログ記録
- モデルチェックポイント自動保存
- 学習済みモデルの評価

**主要パラメータ**:
```
--level          : 学習レベル (デフォルト: Level1-1)
--total-steps    : 総学習ステップ (デフォルト: 100000)
--num-envs       : 並列環境数 (デフォルト: 4)
--device         : GPU/CPU選択 (デフォルト: auto)
```

#### 2. **infer_ppo.py** - 推論スクリプト
- 学習済みモデルでゲームをプレイ
- 複数エピソード実行と統計集計
- 画面表示機能
- 決定論的/確率的行動のサポート
- 他レベルでの転移学習テスト

**主要パラメータ**:
```
--model          : モデルファイルパス (必須)
--level          : プレイレベル (デフォルト: Level1-1)
--episodes       : 実行エピソード数 (デフォルト: 5)
--render         : 画面表示フラグ
```

#### 3. **test_ppo_setup.py** - 統合テストスクリプト
- 環境初期化テスト
- モデル作成テスト
- 短い学習ループテスト
- 推論機能テスト
- モデル保存テスト

#### 4. **TRAINING_GUIDE.md** - 詳細ドキュメント
- セットアップ手順
- 使用方法
- ハイパーパラメータ説明
- TensorBoard 監視方法
- トラブルシューティング
- 転移学習の例

### 実装の主要な設計判断

#### 観測処理
```python
Dict観測（環境） → DictToImageWrapper → Box(0-1, float32)
    ↓
MarioEnv返す
    ├─ image: (3, 480, 640) uint8
    └─ state: (9,) float32
    
画像を正規化
    0-255 → 0-1 (float32)
```

#### ポリシー構成
- **ポリシータイプ**: CnnPolicy（CNN ベース）
- **特性抽出器**: NatureCNN（4層 Conv2D）
- **正規化**: normalize_images=False（既に正規化済みのため）

#### ハイパーパラメータ
- Learning Rate: 3e-4
- Batch Size: 64
- N Epochs: 10
- Gamma (割引率): 0.99
- GAE Lambda: 0.95
- PPO Clip Range: 0.2
- Entropy Coefficient: 0.01

### 動作確認

```
✅ ステップ 1: パッケージをインポート... 成功
✅ ステップ 2: DictToImageWrapper をテスト... 成功
✅ ステップ 3: 環境を初期化... 成功
✅ ステップ 4: DictToImageWrapper を適用... 成功
✅ ステップ 5: 環境の reset... 成功
✅ ステップ 6: PPO モデルを初期化... 成功
✅ ステップ 7: 短い学習ループ... 成功
✅ ステップ 8: 推論... 成功 (10 ステップ)
✅ ステップ 9: モデル保存... 成功
```

## 使用方法

### 基本的な学習

```bash
# デフォルト設定で 100,000 ステップ学習
python train_ppo.py

# 出力:
# - models/mario_ppo_level1-1_100k.zip
# - logs/mario_ppo_level1-1_100k_*/
```

### 学習の監視

```bash
# TensorBoard を起動
tensorboard --logdir ./logs

# ブラウザで http://localhost:6006 を開く
```

### 推論実行

```bash
# 学習済みモデルで 5 エピソード実行
python infer_ppo.py --model models/mario_ppo_level1-1_100k.zip

# 画面に表示して実行
python infer_ppo.py --model models/mario_ppo_level1-1_100k.zip --render

# 10 エピソード実行
python infer_ppo.py --model models/mario_ppo_level1-1_100k.zip --episodes 10
```

### カスタマイズ例

```bash
# Level1-2 で 200,000 ステップ学習
python train_ppo.py --level Level1-2 --total-steps 200000

# 8 並列環境で GPU 使用
python train_ppo.py --num-envs 8 --device cuda

# CPU のみ使用
python train_ppo.py --device cpu
```

## パッケージ要件

```
pygame-ce>=2.0.0
scipy>=1.4.1
gymnasium>=0.26.0  (Note: 1.2.3 - stable-baselines3 との互換性)
numpy>=1.19.0
stable-baselines3>=2.0.0
torch>=2.0.0
```

## 参考資料

- [Stable Baselines3 ドキュメント](https://stable-baselines3.readthedocs.io/)
- [Gymnasium](https://gymnasium.farama.org/)
- [PPO 論文 (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)

## トラブルシューティング

### メモリ不足
```bash
python train_ppo.py --num-envs 2  # 並列環境数を減らす
```

### 学習が遅い
```bash
python train_ppo.py --device cuda --num-envs 8  # GPU + 高並列化
```

### モデルが見つからない
```bash
ls models/  # モデルが保存されているか確認
```

## 次のステップ

1. ✅ 環境設定完了
2. ✅ スクリプト実装完了
3. ✅ テスト完了

実行準備完了！以下のコマンドで学習を開始できます：

```bash
python train_ppo.py --total-steps 100000
```

---

**作成日**: 2026年6月2日  
**バージョン**: 1.0  
**ステータス**: 本番利用可能 ✅
