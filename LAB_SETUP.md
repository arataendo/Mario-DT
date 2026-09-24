# 研究室PC（Linux + NVIDIA GPU）で学習・評価を回す手順

手元の Windows 機はメモリ不調（認識 7.7GB）で学習が極端に遅くなったため、学習と評価を研究室PCへ移す。

| 何を | どうやって運ぶ |
|---|---|
| コード・レベル・評価結果 | GitHub（`dt-pipeline` ブランチ） |
| 学習データ（PNG 約37万枚）・モデル | `mario_dt_data.tar` 1本にまとめて scp |

PNG を1枚ずつ送ると1ファイルごとの往復で非常に遅いので、tar 1本にまとめている。
tar の中の metadata は、Windows 形式のパス（`frames\ep00000.png`）を `/` 区切りに直してある。

## 1. 研究室PCでリポジトリを clone

```bash
git clone -b dt-pipeline https://github.com/arataendo/Mario-DT.git
cd Mario-DT
```

## 2. 手元のPCから学習データを送る

手元の Windows 機（リポジトリのルート）で:

```bash
scp transfer/mario_dt_data.tar transfer/mario_dt_data.tar.sha256 <ユーザー名>@<研究室PC>:~/Mario-DT/
```

tar は約1.4GB（sha256: `5a748d13…`）。作り直すときは `python pack_for_transfer.py`（数十分）。

## 3. 研究室PCでセットアップ

```bash
bash setup_lab.sh
```

venv 作成 → `pip install -r requirements.txt` → tar の検証と展開 → `check_lab_env.py` での動作確認、まで自動で行う。
最後に全項目 ✅ になれば準備完了。よくあるつまずきは画面に対処法が出る:

- **GPU が見えない** → `nvidia-smi` の CUDA バージョンに合う torch を入れ直す（例: `pip install torch --index-url https://download.pytorch.org/whl/cu121`、古いドライバなら `cu118`）
- **`import cv2` で libGL エラー** → `sudo apt install -y libgl1 libglib2.0-0`（sudo が無ければ `pip uninstall -y opencv-python && pip install opencv-python-headless`）
- **Python が 3.10 未満** → `PYTHON=python3.12 bash setup_lab.sh`

## 4. 学習を回す

SSH が切れても止まらないよう tmux の中で実行する（`Ctrl-b d` で抜ける、`tmux attach -t mario` で戻る）。

```bash
tmux new -s mario
source .venv/bin/activate
```

Linux では DataLoader の並列読み込みが使える（Windows 機では使えなかった）。PNG のデコードが律速なので `--num-workers 8` にする。

**A. frame_stack=2（1マス穴の取りこぼし対策）を epoch 4 から再開**

```bash
python -u train_dt.py --dataset dt_dataset_v10/metadata_rebalanced.pkl \
  --epochs 16 --steps-per-epoch 1000 --batch-size 64 --num-workers 8 \
  --resume-from models/mario_dt_20260924_111016_epoch4.pth 2>&1 | tee train_fs2.log
```

`frame_stack` はチェックポイントから自動で読むので指定不要。保存先は `models/mario_dt_20260924_111016_epoch{5..20}.pth`。

**B. 時間ペナルティ 0.10（最高 target 付近の分解能を上げる）**

```bash
python -u train_dt.py --dataset dt_dataset_v10/metadata_tp010.pkl \
  --epochs 20 --steps-per-epoch 1000 --batch-size 64 --num-workers 8 \
  --frame-stack 2 2>&1 | tee train_tp010.log
```

A と B は別々のランなので、研究室PCのコア数・GPU メモリに余裕があれば別の tmux ウィンドウで同時に回してよい（1ランの GPU 使用量は 2GB 未満）。
B の `--frame-stack` は A の結果を見て 1 か 2 を決める。

## 5. 評価する

```bash
./run_eval.sh models/mario_dt_20260924_111016_epoch20.pth eval_out/v11_fs2 --sample
python summarize_eval.py eval_out/v10_sample eval_out/v11_fs2
```

- test 9 ステージ + 学習済み 4 ステージ × target {0,60,120,180,235} × 10 エピソードを、ステージ単位で並列に回す
- 並列数は `JOBS=8 ./run_eval.sh ...` で変更（既定はコア数の半分）
- 途中で止まっても同じコマンドで続きから走る
- **推論は必ず `--sample` を付ける**。argmax だと低 target で学習した「乱れた行動分布」が捨てられ、条件付けが効かなくなる
- `eval_out/v10_sample/` が現行最良モデル（`mario_dt_20260921_175028_epoch20.pth` + `--sample`）の結果。比較の基準にする

## 6. 結果を持ち帰る

評価結果（`eval_out/`）は小さいので Git で運ぶ:

```bash
git add eval_out/ && git commit -m "評価結果: v11_fs2" && git push
```

モデルが必要なら scp で取ってくる（`models/` は .gitignore 済み）。

## 主なファイル

| ファイル | 役割 |
|---|---|
| `train_dt.py` | DT の学習（`--frame-stack`、`--resume-from`） |
| `eval_dt_matrix.py` / `run_eval.sh` | 評価（1プロセス / ステージ並列） |
| `summarize_eval.py` | 評価結果の集計・比較 |
| `infer_dt.py` | 1ステージを遊ばせる（`--sample`、`--render`） |
| `collect_dt_dataset.py` | PPO で DT 用データを集める（`--epsilons`、`--resume`） |
| `rebalance_dt_dataset.py` | 終端報酬・時間ペナルティの付け替え |
| `generate_level.py` | 難易度指定でステージ生成 |
| `check_lab_env.py` | 研究室PCの動作確認 |
