MODEL_PATH="/mnt/82_store/tb/github_upload/checkpoints/FlashSloth_HD-fft-3.7M"
python3 -m accelerate.commands.launch \
    --num_processes=8 \
    -m lmms_eval \
    --model FlashSloth \
    --model_args pretrained="$MODEL_PATH" \
    --tasks mme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.5_mme  \
    --output_path ./logs/ 
# python3 -m accelerate.commands.launch \
#     --num_processes=8 \
#     -m lmms_eval \
#     --model FlashSloth \
#     --model_args pretrained="$MODEL_PATH" \
#     --tasks mmbench_en \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_v1.5_mme  \
#     --output_path ./logs/ 

# python3 -m accelerate.commands.launch \
#     --num_processes=8 \
#     -m lmms_eval \
#     --model FlashSloth \
#     --model_args pretrained="$MODEL_PATH" \
#     --tasks gqa,scienceqa_img,pope \
#     --batch_size 1 \
#     --output_path ./logs/ 

# python3 -m accelerate.commands.launch \
#     --num_processes=8 \
#     -m lmms_eval \
#     --model FlashSloth \
#     --model_args pretrained="$MODEL_PATH" \
#     --tasks ai2d,chartqa \
#     --batch_size 1 \
#     --output_path ./logs/

# python3 -m accelerate.commands.launch \
#     --num_processes=8 \
#     -m lmms_eval \
#     --model FlashSloth \
#     --model_args pretrained="$MODEL_PATH" \
#     --tasks ocrbench,infovqa,mmmu_val \
#     --batch_size 1 \
#     --output_path ./logs/

# python3 -m accelerate.commands.launch \
#     --num_processes=8 \
#     -m lmms_eval \
#     --model FlashSloth \
#     --model_args pretrained="$MODEL_PATH" \
#     --tasks mmvet \
#     --batch_size 1 \
#     --output_path ./logs/ \
#     --predict_only
# python3 -m accelerate.commands.launch \
#     --num_processes=8 \
#     -m lmms_eval \
#     --model FlashSloth \
#     --model_args pretrained="$MODEL_PATH" \
#     --tasks seedbench \
#     --batch_size 1 \
#     --output_path ./logs/ 
# python3 -m accelerate.commands.launch \
#     --num_processes=8 \
#     -m lmms_eval \
#     --model FlashSloth \
#     --model_args pretrained="$MODEL_PATH" \
#     --tasks docvqa,realworldqa \
#     --batch_size 1 \
#     --output_path ./logs/ 
# python3 -m accelerate.commands.launch \
#     --num_processes=8 \
#     -m lmms_eval \
#     --model FlashSloth \
#     --model_args pretrained="$MODEL_PATH" \
#     --tasks mathvista_testmini \
#     --batch_size 1 \
#     --output_path ./logs/ 

