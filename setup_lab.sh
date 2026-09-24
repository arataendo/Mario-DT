#!/bin/bash
# 研究室PC（Linux + NVIDIA GPU）の初回セットアップ。
#   1. Python 仮想環境 (.venv) を作り、requirements.txt を入れる
#   2. 転送した mario_dt_data.tar を検証して展開する
#   3. check_lab_env.py で GPU・画面なし実行・データ・学習1ステップを確認する
#
# 使い方（clone したリポジトリのルートで。tar はルートに置いておく）:
#   bash setup_lab.sh
#
# 何度実行しても安全（venv と展開済みデータはそのまま使う）。
set -eu
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
ver=$($PY -c 'import sys; print("%d.%d" % sys.version_info[:2])')
if ! $PY -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "❌ Python $ver は古すぎます（transformers 5 系に 3.10 以上が必要）。"
  echo "   PYTHON=python3.12 bash setup_lab.sh のように新しい Python を指定してください。"
  exit 1
fi
echo "🐍 Python $ver"

# ---- 1. 仮想環境 ----
if [ ! -d .venv ]; then
  if ! $PY -m venv .venv; then
    echo "❌ venv を作れませんでした。Ubuntu なら: sudo apt install -y python3-venv"
    exit 1
  fi
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt

# ---- 2. 学習データの展開 ----
if [ -f dt_dataset_v10/metadata_rebalanced.pkl ]; then
  echo "📦 学習データは展開済み"
elif [ -f mario_dt_data.tar ]; then
  if [ -f mario_dt_data.tar.sha256 ]; then
    echo "🔍 転送で壊れていないか確認中..."
    sha256sum -c mario_dt_data.tar.sha256
  else
    echo "⚠️  mario_dt_data.tar.sha256 が無いので整合性チェックを省略します"
  fi
  echo "📦 展開中（37万ファイル、数分かかります）..."
  tar -xf mario_dt_data.tar
  echo "   展開完了。tar は消してかまいません: rm mario_dt_data.tar"
else
  echo "⚠️  mario_dt_data.tar がありません。先に元のPCから転送してください（LAB_SETUP.md 参照）"
fi

chmod +x run_eval.sh run_collect_*.sh run_eval_v10*.sh 2>/dev/null || true

# ---- 3. 動作確認 ----
echo
python check_lab_env.py
