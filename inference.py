import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from flashsloth.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, LEARNABLE_TOKEN, LEARNABLE_TOKEN_INDEX
from flashsloth.conversation import conv_templates, SeparatorStyle
from flashsloth.model.builder import load_pretrained_model
from flashsloth.utils import disable_torch_init
from flashsloth.mm_utils import tokenizer_image_token, process_images, process_images_hd_inference, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image


def main():
    image_path = "/mnt/82_store/luogen/tb/speed_test/b4ec9749e3488959fe0752712aa6437.png"
    text = "Describe this photo in detail."
    model_path = "/mnt/82_store/tb/github_upload/checkpoints/FlashSloth_HD-fft-3.7M"
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)
    torch.set_printoptions(threshold=torch.inf)
    keywords = ['</s>']
    text = DEFAULT_IMAGE_TOKEN + '\n' + text
    text = text + LEARNABLE_TOKEN
    image = Image.open(image_path).convert('RGB')
    if model.config.image_hd:
        image_tensor = process_images_hd_inference([image], image_processor, model.config)[0]
    else:
        image_tensor = process_images([image], image_processor, model.config)[0]
    image_tensor = image_tensor.unsqueeze(0)
    conv = conv_templates["phi2"].copy()
    conv.append_message(conv.roles[0], text)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Tokenize text
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(device='cuda', non_blocking=True)

    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
            max_new_tokens=1024,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=[stopping_criteria]
        )
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    print(outputs)

if __name__ == "__main__":
    main()
