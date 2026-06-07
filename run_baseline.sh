#!/bin/bash

# =========================================================
# run_baselines.sh
# 批量运行 baseline.py
# 模型：
#   LSTM
#   TCN
#   Transformer encoder-decoder
#   Informer encoder-decoder
#   Autoformer encoder-decoder
#   FEDformer encoder-decoder
#   PatchTST
#   DLinear
#   iTransformer
#   TimesNet
#   TimeMixer
# =========================================================

set -e

# -----------------------------
# 1. 数据配置
# -----------------------------

DATA_PATH="./GEF.csv"
DATA_NAME="GEF"
DATE_COL="date"
COLS="load,solar,wind"
TARGET="all"

# -----------------------------
# 2. 任务长度配置
# -----------------------------

SEQ_LEN=168
LABEL_LEN=48
PRED_LEN=24

# -----------------------------
# 3. 训练配置
# -----------------------------

BATCH_SIZE=1024
TRAIN_EPOCHS=200
PATIENCE=10
LR=0.0005
WEIGHT_DECAY=0.00001
LOSS="mse"
SEED=2026

# -----------------------------
# 4. 模型公共配置
# -----------------------------

D_MODEL=256
N_HEADS=8
E_LAYERS=2
D_LAYERS=1
D_FF=512
DROPOUT=0.1

# -----------------------------
# 5. 输出目录
# -----------------------------

CHECKPOINTS="./checkpoints_baselines"
RESULTS="./results_baselines"

# -----------------------------
# 6. GPU 开关
# 如果不用 GPU，把 USE_GPU="" 即可
# -----------------------------

USE_GPU="--use_gpu"

# -----------------------------
# 7. 创建日志目录
# -----------------------------

mkdir -p logs_baselines_OPSD

# -----------------------------
# 8. 模型列表
# -----------------------------

MODELS=(
  "LSTM"
  "TCN"
  "Transformer"
  "Informer"
  "DLinear"
  "iTransformer"
  "PatchTST"
  "TimeMixer"
  "FEDformer"
  "TimesNet"
  "Autoformer"


)

# -----------------------------
# 9. 逐个运行
# -----------------------------

for MODEL in "${MODELS[@]}"
do
  echo "============================================================"
  echo "Running model: ${MODEL}"
  echo "============================================================"

  MODEL_ID="${DATA_NAME}_${TARGET}_${MODEL}_seed${SEED}"

  EXTRA_ARGS=""

  if [ "$MODEL" = "PatchTST" ]; then
    EXTRA_ARGS="--patch_len 16 --patch_stride 8"
  fi

  if [ "$MODEL" = "Informer" ]; then
    EXTRA_ARGS="--informer_factor 5 --informer_distil"
  fi

  if [ "$MODEL" = "Autoformer" ]; then
    EXTRA_ARGS="--moving_avg 25 --autoformer_factor 1.0"
  fi

  if [ "$MODEL" = "FEDformer" ]; then
    EXTRA_ARGS="--moving_avg 25 --fedformer_modes 32 --mode_select low"
  fi

  if [ "$MODEL" = "TimesNet" ]; then
    EXTRA_ARGS="--top_k_period 5"
  fi

  if [ "$MODEL" = "TimeMixer" ]; then
    EXTRA_ARGS="--timemixer_scales 3 --timemixer_kernel 25"
  fi

  python baseline.py \
    --data_path "${DATA_PATH}" \
    --data_name "${DATA_NAME}" \
    --date_col "${DATE_COL}" \
    --cols "${COLS}" \
    --target "${TARGET}" \
    --seq_len ${SEQ_LEN} \
    --label_len ${LABEL_LEN} \
    --pred_len ${PRED_LEN} \
    --model "${MODEL}" \
    --d_model ${D_MODEL} \
    --n_heads ${N_HEADS} \
    --e_layers ${E_LAYERS} \
    --d_layers ${D_LAYERS} \
    --d_ff ${D_FF} \
    --dropout ${DROPOUT} \
    --batch_size ${BATCH_SIZE} \
    --train_epochs ${TRAIN_EPOCHS} \
    --patience ${PATIENCE} \
    --learning_rate ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --loss ${LOSS} \
    --seed ${SEED} \
    --model_id "${MODEL_ID}" \
    --checkpoints "${CHECKPOINTS}" \
    --results "${RESULTS}" \
    ${USE_GPU} \
    ${EXTRA_ARGS} \
    2>&1 | tee "logs_baselines/${MODEL_ID}.log"

  echo ""
  echo "Finished model: ${MODEL}"
  echo ""

done

echo "============================================================"
echo "All baseline models finished."
echo "Summary file:"
echo "${RESULTS}/summary_baselines.csv"
echo "============================================================"