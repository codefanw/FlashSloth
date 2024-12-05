#!/bin/bash
# uncomment the following lines to shutoff the internet access
# export HF_HUB_OFFLINE=True
# export HF_DATASETS_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
export FlashSloth_SILIENT_OTHERS=true
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

SPLIT="mmbench_dev"

MODEL_PATH=$1
OUTPUT_MMB_DIR=$2/MMB/

# # merge eval
# MODEL_CKPT="flashsloth-v1-3b-fft-phi2"
#MODEL_CKPT="flashsloth-v1-3b" # eval your own checkpoint
# EVAL_CKPT="${MODEL_CKPT//\//_}_1"
EVAL_CKPT=$(basename "$MODEL_PATH")
# MODEL_PATH="/data/tb/flashsloth-main/checkpoints/flashsloth-v1-3b-fft-attention-complete"
# MODEL_PATH="./checkpoints/$MODEL_CKPT" # eval your own checkpoint



for IDX in $(seq 0 $((CHUNKS-1))); do
   LOCAL_RANK=$IDX CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m flashsloth.eval.model_vqa_mmbench \
       --model-path $MODEL_PATH \
       --question-file /mnt/82_store/luogen/tb/mywork/benchmark_data/mmbench/mmbench_dev_20230712.tsv \
       --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/$EVAL_CKPT/${CHUNKS}_${IDX}.jsonl \
       --num-chunks $CHUNKS \
       --chunk-idx $IDX \
       --temperature 0 \
       --conv-mode phi2 \
       --single-pred-prompt &
done

wait

# lora eval
#MODEL_CKPT="flashsloth-v1-3b-stage2-lora"
#EVAL_CKPT="${MODEL_CKPT//\//_}_1"
#MODEL_BASE=checkpoints/base/phi-2
#
#for IDX in $(seq 0 $((CHUNKS-1))); do
#    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m flashsloth.eval.model_vqa_mmbench \
#        --model-path ./checkpoints/$MODEL_CKPT \
#        --model-base $MODEL_BASE  \
#        --question-file ./playground/data/eval/mmbench/mmbench_dev_20230712.tsv \
#        --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/$EVAL_CKPT/${CHUNKS}_${IDX}.jsonl \
#        --num-chunks $CHUNKS \
#        --chunk-idx $IDX \
#        --temperature 0 \
#        --conv-mode phi2 &
#done
#
#wait


output_file=./playground/data/eval/mmbench/answers/$SPLIT/$EVAL_CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/mmbench/answers/$SPLIT/$EVAL_CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


mkdir -p ./playground/data/eval/mmbench/answers_upload/$SPLIT/$EVAL_CKPT

mkdir -p $OUTPUT_MMB_DIR
> $OUTPUT_MMB_DIR/output.txt


python scripts/convert_mmbench_for_submission.py \
    --annotation-file /mnt/82_store/luogen/tb/mywork/benchmark_data/mmbench/mmbench_dev_20230712.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT/$EVAL_CKPT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT/$EVAL_CKPT \
    --experiment merge

python /mnt/82_store/luogen/tb/mywork/benchmark_data/mmbench/eval.py --result ./playground/data/eval/mmbench/answers_upload/$SPLIT/$EVAL_CKPT/merge.xlsx --meta /mnt/82_store/luogen/tb/mywork/benchmark_data/mmbench/mmbench_dev_20230712.tsv >> $OUTPUT_MMB_DIR/output.txt 2>&1

