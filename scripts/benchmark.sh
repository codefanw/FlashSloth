#!/usr/bin/env bash
if [ -z "$1" ]; then
  MODEL_PATH="/mnt/82_store/tb/github_upload/checkpoints/FlashSloth_HD-fft-3.7M"
  echo "No MODEL_PATH provided, using default: $MODEL_PATH"
else
  MODEL_PATH="$1"
fi

WORK_DIR=/mnt/82_store/tb/github_upload/FlashSloth
export PYTHONPATH=${WORK_DIR}
MODEL_CKPT=$(basename "$MODEL_PATH")
OUTPUT_DIR_EVAL=${WORK_DIR}/eval_output/$MODEL_CKPT
mkdir -p ${OUTPUT_DIR_EVAL}



# cd ${WORK_DIR}
# DATASET_NAME=mme
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/eval/${DATASET_NAME}.sh \
#     ${MODEL_PATH} $OUTPUT_DIR_EVAL


# DATASET_NAME=gqa
# cd ${WORK_DIR}
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/eval/${DATASET_NAME}.sh \
#     ${MODEL_PATH} $OUTPUT_DIR_EVAL
    

# DATASET_NAME=textvqa
# cd ${WORK_DIR}
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/eval/${DATASET_NAME}.sh \
#     ${MODEL_PATH} $OUTPUT_DIR_EVAL

# DATASET_NAME=pope
# cd ${WORK_DIR}
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/eval/${DATASET_NAME}.sh \
#     ${MODEL_PATH} $OUTPUT_DIR_EVAL

DATASET_NAME=mmbench
cd ${WORK_DIR}
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/eval/${DATASET_NAME}.sh \
    ${MODEL_PATH} $OUTPUT_DIR_EVAL
