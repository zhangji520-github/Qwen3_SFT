from unsloth import FastLanguageModel
from transformers import TextStreamer
import torch

model_name = "/root/autodl-tmp/based_Unsloth/saved_models/qwen3_16bit"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,  # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length=1024,
    dtype=torch.bfloat16,
    load_in_4bit=False,
    load_in_8bit=True,
    device_map="auto",
)
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

messages = [
    {
        "role": "user",
        "content": "一个8岁的男孩跌倒时右手掌撑地，导致右腕剧痛、肿胀和活动障碍，并出现‘餐叉’畸形。请描述此情况下可能发生的具体骨折类型。",
    }
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
)

text_streamer = TextStreamer(tokenizer)
_ = model.generate(input_ids, streamer=text_streamer, max_new_tokens=1024)
