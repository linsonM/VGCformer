#!/bin/bash

# =========================================================
# run_loss_ablation_GEF_OPSD.sh
# 一键运行 GEF + OPSD 两个数据集的 BASE_LOSS 消融实验
# loss ablation:
#   1) adaptive_ramp
#   2) mae
#   3) huber
# =========================================================

set -e

PYTHON_BIN="python"
SCRIPT="main.py"

# =========================================================
# 1. 公共训练配置
# =========================================================

DATE_COL="date"
COLS="load,solar,wind"
TARGET="all"

TRAIN_RATIO=0.7
VAL_RATIO=0.1

SEQ_LEN=168
PRED_LEN=24

D_MODEL=256
N_HEADS=8
E_LAYERS=1
D_FF=512
DROPOUT=0.1
GATE_HIDDEN=128

# -----------------------------
# baseline 参数
# -----------------------------
BASE_HILL_D_MAX=2.0
BASE_RESIDUAL_WEIGHT=0.1

BASE_ATTN_STRIDE=4
BASE_ATTN_MIN_TOKENS=8

BASE_LOCAL_WINDOW=48
BASE_RAMP_TOPK=0
BASE_PERIODS="24,48,168"

BASE_SEASONAL_LAMBDA=0.1
BASE_RECENT_LAMBDA=0.05
BASE_PERIOD_SIGMA=6.0

BASE_LEARNING_RATE=0.005
BASE_WEIGHT_DECAY=0.001

# -----------------------------
# loss 相关参数
# -----------------------------
BASE_LOSS="adaptive_ramp"
BASE_HUBER_DELTA=0.2
BASE_RAMP_WEIGHT=0.1

# =========================================================
# 2. BASE_LOSS 消融列表
# =========================================================
# adaptive_ramp: 原始 baseline
# mae          : MAE loss 消融
# huber        : Huber loss 消融
# =========================================================

LOSS_ABLATIONS=(
  "adaptive_ramp"
  "mae"
  "huber"
)

BATCH_SIZE=1024
TRAIN_EPOCHS=200
PATIENCE=10
LR_PATIENCE=3
CLIP_GRAD=1.0

SEED=2024
NUM_WORKERS=0

USE_GPU="--use_gpu"
export CUDA_VISIBLE_DEVICES=0

# 如果不用 GPU，把上一行注释掉，并改成：
# USE_GPU=""

# =========================================================
# 3. 模型组件设置
# =========================================================
# 这里保持 full，只做 BASE_LOSS 消融。
# 如果你想在每个组件消融下都做 loss 消融，
# 可以取消下面其他 ablation 的注释。
# =========================================================

ABLATIONS=(
  "full"
#  "std_attn"
#  "fixed_hill"
#  "learn_hill_no_gate"
#  "no_ramp_kv"
#  "no_local_gate"
#  "no_seasonal_bias"
#  "no_asym_hill"
#  "no_channel"
#  "no_time"
#  "no_residual"
)

# =========================================================
# 4. 通用运行函数：单个实验
# =========================================================

run_one_exp () {
  DATA_PATH=$1
  DATA_NAME=$2
  ABLATION=$3
  EXP_LOSS=$4
  CHECKPOINTS=$5
  RESULTS=$6
  LOGS=$7

  MODEL_ID="${DATA_NAME}_${TARGET}_${ABLATION}_loss_${EXP_LOSS}_seed${SEED}"
  LOG_FILE="${LOGS}/${MODEL_ID}.log"

  echo ""
  echo "======================================================================"
  echo "Running Dataset   : ${DATA_NAME}"
  echo "Running Ablation  : ${ABLATION}"
  echo "Running Loss      : ${EXP_LOSS}"
  echo "MODEL_ID          : ${MODEL_ID}"
  echo "DATA_PATH         : ${DATA_PATH}"
  echo "======================================================================"

  ${PYTHON_BIN} ${SCRIPT} \
    --data_path "${DATA_PATH}" \
    --data_name "${DATA_NAME}" \
    --date_col "${DATE_COL}" \
    --cols "${COLS}" \
    --target "${TARGET}" \
    --train_ratio ${TRAIN_RATIO} \
    --val_ratio ${VAL_RATIO} \
    --seq_len ${SEQ_LEN} \
    --pred_len ${PRED_LEN} \
    --d_model ${D_MODEL} \
    --n_heads ${N_HEADS} \
    --e_layers ${E_LAYERS} \
    --d_ff ${D_FF} \
    --dropout ${DROPOUT} \
    --gate_hidden ${GATE_HIDDEN} \
    --hill_d_max ${BASE_HILL_D_MAX} \
    --residual_weight ${BASE_RESIDUAL_WEIGHT} \
    --attn_stride ${BASE_ATTN_STRIDE} \
    --attn_min_tokens ${BASE_ATTN_MIN_TOKENS} \
    --local_window ${BASE_LOCAL_WINDOW} \
    --ramp_topk ${BASE_RAMP_TOPK} \
    --periods "${BASE_PERIODS}" \
    --seasonal_lambda ${BASE_SEASONAL_LAMBDA} \
    --recent_lambda ${BASE_RECENT_LAMBDA} \
    --period_sigma ${BASE_PERIOD_SIGMA} \
    --ablation "${ABLATION}" \
    --batch_size ${BATCH_SIZE} \
    --train_epochs ${TRAIN_EPOCHS} \
    --patience ${PATIENCE} \
    --learning_rate ${BASE_LEARNING_RATE} \
    --weight_decay ${BASE_WEIGHT_DECAY} \
    --lr_patience ${LR_PATIENCE} \
    --loss "${EXP_LOSS}" \
    --huber_delta ${BASE_HUBER_DELTA} \
    --ramp_weight ${BASE_RAMP_WEIGHT} \
    --clip_grad ${CLIP_GRAD} \
    --seed ${SEED} \
    --num_workers ${NUM_WORKERS} \
    --model_id "${MODEL_ID}" \
    --checkpoints "${CHECKPOINTS}" \
    --results "${RESULTS}" \
    ${USE_GPU} \
    2>&1 | tee "${LOG_FILE}"

  echo ""
  echo "Finished: ${MODEL_ID}"
  echo ""
}

# =========================================================
# 5. 跑单个数据集全部 loss 消融
# =========================================================

run_dataset_all_loss_ablations () {
  DATA_PATH=$1
  DATA_NAME=$2

  CHECKPOINTS="./checkpoints_loss_ablation_${DATA_NAME}"
  RESULTS="./results_loss_ablation_${DATA_NAME}"
  LOGS="./logs_loss_ablation_${DATA_NAME}"

  mkdir -p "${CHECKPOINTS}"
  mkdir -p "${RESULTS}"
  mkdir -p "${LOGS}"

  echo ""
  echo "######################################################################"
  echo "Start BASE_LOSS ablations for dataset: ${DATA_NAME}"
  echo "Data path  : ${DATA_PATH}"
  echo "Checkpoints: ${CHECKPOINTS}"
  echo "Results    : ${RESULTS}"
  echo "Logs       : ${LOGS}"
  echo "######################################################################"
  echo ""

  for ABLATION in "${ABLATIONS[@]}"
  do
    for EXP_LOSS in "${LOSS_ABLATIONS[@]}"
    do
      run_one_exp \
        "${DATA_PATH}" \
        "${DATA_NAME}" \
        "${ABLATION}" \
        "${EXP_LOSS}" \
        "${CHECKPOINTS}" \
        "${RESULTS}" \
        "${LOGS}"
    done
  done

  echo ""
  echo "######################################################################"
  echo "Finished all BASE_LOSS ablations for dataset: ${DATA_NAME}"
  echo "Summary file: ${RESULTS}/summary_all.csv"
  echo "######################################################################"
  echo ""
}

# =========================================================
# 6. 依次运行 GEF 与 OPSD
# =========================================================

run_dataset_all_loss_ablations "./GEF.csv" "GEF"
run_dataset_all_loss_ablations "./OPSD.csv" "OPSD"

echo ""
echo "======================================================================"
echo "All BASE_LOSS ablation experiments for GEF and OPSD are finished."
echo ""
echo "GEF summary : ./results_loss_ablation_GEF/summary_all.csv"
echo "OPSD summary: ./results_loss_ablation_OPSD/summary_all.csv"
echo "======================================================================"