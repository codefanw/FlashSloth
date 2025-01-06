import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from flashsloth.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN, LEARNABLE_TOKEN, LEARNABLE_TOKEN_INDEX
)
from flashsloth.conversation import conv_templates, SeparatorStyle
from flashsloth.model.builder import load_pretrained_model
from flashsloth.utils import disable_torch_init
from flashsloth.mm_utils import (
    tokenizer_image_token, process_images, process_images_hd_inference,
    get_model_name_from_path, KeywordsStoppingCriteria
)
from PIL import Image
import gradio as gr

from transformers import TextIteratorStreamer
from threading import Thread

disable_torch_init()

MODEL_PATH_HD = "/data/tb/rebuttal/FlashSloth/model/flashsloth_hd"
MODEL_PATH_NEW = "Tongbo/FlashSloth-3.2B"

model_name_hd = get_model_name_from_path(MODEL_PATH_HD)
model_name_new = get_model_name_from_path(MODEL_PATH_NEW)

models = {
    "FlashSloth HD": load_pretrained_model(MODEL_PATH_HD, None, model_name_hd),
    "FlashSloth": load_pretrained_model(MODEL_PATH_NEW, None, model_name_new)
}

for key in models:
    tokenizer, model, image_processor, context_len = models[key]
    model.to('cuda')
    model.eval()

def generate_description(image, prompt_text, temperature, top_p, max_tokens, selected_model):
    """
    生成图片描述的函数，支持流式输出，并根据选择的模型进行处理。
    新增参数:
      - selected_model: 用户选择的模型名称
    """
    keywords = ['</s>']

    tokenizer, model, image_processor, context_len = models[selected_model]

    text = DEFAULT_IMAGE_TOKEN + '\n' + prompt_text
    text = text + LEARNABLE_TOKEN

    image = image.convert('RGB')
    if model.config.image_hd:
        image_tensor = process_images_hd_inference([image], image_processor, model.config)[0]
    else:
        image_tensor = process_images([image], image_processor, model.config)[0]
    image_tensor = image_tensor.unsqueeze(0).to(dtype=torch.float16, device='cuda', non_blocking=True)

    conv = conv_templates["phi2"].copy()
    conv.append_message(conv.roles[0], text)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    input_ids = input_ids.unsqueeze(0).to(device='cuda', non_blocking=True)

    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    streamer = TextIteratorStreamer(
        tokenizer=tokenizer, 
        skip_prompt=True, 
        skip_special_tokens=True
    )

    generation_kwargs = dict(
        inputs=input_ids,
        images=image_tensor,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=int(max_tokens),  
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        stopping_criteria=[stopping_criteria],
        streamer=streamer            
    )

    def _generate():
        with torch.inference_mode():
            model.generate(**generation_kwargs)

    generation_thread = Thread(target=_generate)
    generation_thread.start()

    partial_text = ""
    for new_text in streamer:
        partial_text += new_text
        yield partial_text

    generation_thread.join()

custom_css = """
<style>
/* 增大标题字体 */
#title {
    font-size: 80px !important;
    text-align: center;
    margin-bottom: 20px;
}

/* 增大描述文字字体 */
#description {
    font-size: 24px !important;
    text-align: center;
    margin-bottom: 40px;
}

/* 增大标签和输入框的字体 */
.gradio-container * {
    font-size: 18px !important;
}

/* 增大按钮字体 */
button {
    font-size: 20px !important;
    padding: 10px 20px;
}

/* 增大输出文本的字体 */
.output_text {
    font-size: 20px !important;
}
</style>
"""

with gr.Blocks(css=custom_css) as demo:
    gr.HTML(custom_css)
    gr.HTML("<h1 style='font-size:70px; text-align:center;'>FlashSloth 多模态大模型 Demo</h1>")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="上传图片")

            temperature_slider = gr.Slider(
                minimum=0.01, 
                maximum=1.0, 
                step=0.05, 
                value=0.7, 
                label="Temperature"
            )
            topp_slider = gr.Slider(
                minimum=0.01, 
                maximum=1.0, 
                step=0.05, 
                value=0.9, 
                label="Top-p"
            )
            maxtoken_slider = gr.Slider(
                minimum=64, 
                maximum=3072, 
                step=1, 
                value=3072, 
                label="Max Tokens"
            )
            
            model_dropdown = gr.Dropdown(
                choices=list(models.keys()),
                value=list(models.keys())[0],
                label="选择模型",
                type="value"
            )
            
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(
                lines=3, 
                placeholder="Describe this photo in detail.", 
                label="问题提示"
            )
            submit_button = gr.Button("生成答案", variant="primary")

            output_text = gr.Textbox(
                label="生成的答案", 
                interactive=False, 
                lines=15, 
                elem_classes=["output_text"]
            )

    submit_button.click(
        fn=generate_description, 
        inputs=[image_input, prompt_input, temperature_slider, topp_slider, maxtoken_slider, model_dropdown], 
        outputs=output_text, 
        show_progress=True
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=8888)
