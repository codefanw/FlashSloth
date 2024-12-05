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

SPLIT="llava_MME"

MODEL_PATH=$1
OUTPUT_MME_DIR=$2/MME/

EVAL_CKPT=$(basename "$MODEL_PATH")
# # merge eval
# MODEL_CKPT="flashsloth-v1-3b-fft-phi2"
# MODEL_CKPT="flashsloth-v1-3b" # eval your own checkpoint
# EVAL_CKPT="${MODEL_CKPT//\//_}_1"
# MODEL_PATH="/data/tb/flashsloth-main/checkpoints/flashsloth-v1-3b-fft-attention-complete"
# MODEL_PATH="./checkpoints/$MODEL_CKPT" # eval your own checkpoint

for IDX in $(seq 0 $((CHUNKS-1))); do
   LOCAL_RANK=$IDX CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m flashsloth.eval.model_vqa_loader \
       --model-path $MODEL_PATH \
       --question-file ./playground/data/eval/MME/llava_mme.jsonl \
       --image-folder /mnt/82_store/luogen/tb/mywork/benchmark_data/mme/images/ \
       --answers-file ./playground/data/eval/MME/answers/$SPLIT/$EVAL_CKPT/${CHUNKS}_${IDX}.jsonl \
       --num-chunks $CHUNKS \
       --chunk-idx $IDX \
       --temperature 0 \
       --conv-mode phi2 &
done

wait


# lora eval
#MODEL_CKPT="flashsloth-v1-3b-stage2-lora"
#EVAL_CKPT="${MODEL_CKPT//\//_}_1"
#MODEL_BASE=checkpoints/base/phi-2
#
#for IDX in $(seq 0 $((CHUNKS-1))); do
#    LOCAL_RANK=$IDX CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m flashsloth.eval.model_vqa_loader \
#        --model-path ./checkpoints/$MODEL_CKPT \
#        --model-base $MODEL_BASE  \
#        --question-file ./playground/data/eval/MME/llava_mme.jsonl \
#        --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
#        --answers-file ./playground/data/eval/MME/answers/$SPLIT/$EVAL_CKPT/${CHUNKS}_${IDX}.jsonl \
#        --num-chunks $CHUNKS \
#        --chunk-idx $IDX \
#        --temperature 0 \
#        --conv-mode phi2 &
#done
#
#wait

output_file=./playground/data/eval/MME/answers/$SPLIT.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/MME/answers/$SPLIT/$EVAL_CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

mkdir -p $OUTPUT_MME_DIR
> $OUTPUT_MME_DIR/output.txt 

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment $SPLIT >> $OUTPUT_MME_DIR/output.txt 2>&1

cd eval_tool

python calculation.py --results_dir answers/$SPLIT >> $OUTPUT_MME_DIR/output.txt 2>&1
