#!/bin/bash

# uncomment the following lines to shutoff the internet access
# export HF_HUB_OFFLINE=True
# export HF_DATASETS_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
export FlashSloth_SILIENT_OTHERS=true

# if not use all GPUs 
# deepspeed --include localhost:0,1,2,3 --master_port 29600

deepspeed  --master_port 29600 flashsloth/train/train_pretrain.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ./checkpoints/base/phi-2_lqformer\
    --version plain \
    --data_path /mnt/82_store/luogen/tb/mywork/blip_laion_cc_sbu_558k.json \
    --image_folder /mnt/82_store/luogen/tb/mywork/pretrain_images \
    --vision_tower checkpoints/base/siglip-so400m-patch14-384 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter False \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 False \
    --fp16 True \
    --output_dir ./checkpoints/flashsloth-stage1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 3000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 3072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none
