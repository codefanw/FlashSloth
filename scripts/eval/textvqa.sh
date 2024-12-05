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

SPLIT="llava_textvqa_val"

MODEL_PATH=$1
OUTPUT_TextVQA_DIR=$2/TextVQA/

EVAL_CKPT=$(basename "$MODEL_PATH")
# merge eval
# MODEL_CKPT="milvlg/flashsloth-v1-3b"
# MODEL_CKPT="flashsloth-v1-3b-fft-phi2" # eval your own checkpoint
# EVAL_CKPT="${MODEL_CKPT//\//_}_1"
# MODEL_PATH=$MODEL_CKPT
# MODEL_PATH="/data/tb/flashsloth-main/checkpoints/flashsloth-v1-3b-fft-attention-complete" # eval your own checkpoint


for IDX in $(seq 0 $((CHUNKS-1))); do
   LOCAL_RANK=$IDX CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m flashsloth.eval.model_vqa_loader \
       --model-path $MODEL_PATH \
       --question-file /mnt/82_store/luogen/tb/mywork/benchmark_data/textvqa/llava_textvqa_val_v051_ocr.jsonl \
       --image-folder /mnt/82_store/luogen/tb/mywork/benchmark_data/textvqa/train_images \
       --answers-file ./playground/data/eval/textvqa/answers/$SPLIT/$EVAL_CKPT/${CHUNKS}_${IDX}.jsonl \
       --num-chunks $CHUNKS \
       --chunk-idx $IDX \
       --temperature 0 \
       --conv-mode phi2 &
done

wait

# # lora eval
# MODEL_CKPT="flashsloth-v1-3b-stage2-lora"
# EVAL_CKPT="${MODEL_CKPT//\//_}_1"
# MODEL_BASE=checkpoints/base/phi-2

# for IDX in $(seq 0 $((CHUNKS-1))); do
#     LOCAL_RANK=$IDX CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m flashsloth.eval.model_vqa_loader \
#         --model-path ./checkpoints/$MODEL_CKPT \
#         --model-base $MODEL_BASE  \
#         --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
#         --image-folder ./playground/data/eval/textvqa/train_images \
#         --answers-file ./playground/data/eval/textvqa/answers/$SPLIT/$EVAL_CKPT/${CHUNKS}_${IDX}.jsonl \
#         --num-chunks $CHUNKS \
#         --chunk-idx $IDX \
#         --temperature 0 \
#         --conv-mode phi2 &
# done

# wait


output_file=./playground/data/eval/textvqa/answers/$SPLIT/$EVAL_CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/textvqa/answers/$SPLIT/$EVAL_CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

mkdir -p $OUTPUT_TextVQA_DIR
> $OUTPUT_TextVQA_DIR/output.txt

python -m flashsloth.eval.eval_textvqa \
    --annotation-file /mnt/82_store/luogen/tb/mywork/benchmark_data/textvqa/TextVQA_0.5.1_val.json \
    --result-file $output_file  >> $OUTPUT_TextVQA_DIR/output.txt 2>&1
