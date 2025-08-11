from unsloth import FastLanguageModel
from transformers import TextStreamer
import torch

# 模型加载
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


class CaptureStreamer(TextStreamer):
    def __init__(self, tokenizer, skip_prompt: bool = False, **kwargs):
        super().__init__(tokenizer, skip_prompt=skip_prompt, **kwargs)
        self.generated_text = ""  # 用于存储完整输出

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """重写方法捕获最终文本"""
        self.generated_text += text  # 累积输出
        super().on_finalized_text(text, stream_end=stream_end)  # 保持原样输出到终端

    def get_output(self) -> str:
        """获取完整生成内容"""
        return self.generated_text.strip()


def ask(question, is_thinking=True, save_to_file=None):
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=is_thinking,
    )

    # 使用自定义的 CaptureStreamer
    streamer = CaptureStreamer(tokenizer, skip_prompt=True)

    # 生成响应
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        _ = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.6,
            streamer=streamer,  # 关键：使用自定义的 streamer
            pad_token_id=tokenizer.eos_token_id,
        )

    # 获取完整输出
    full_output = streamer.get_output()

    # 保存到文件
    if save_to_file:
        try:
            with open(save_to_file, "w", encoding="utf-8") as f:
                f.write(full_output)
            print(f"✅ 成功写入文件: {save_to_file}")
        except Exception as e:
            print(f"❌ 写入文件失败: {e}")

    return full_output


# 测试集中的数据
ask("人生的意义是什么")

print("\n\n\n")
print("####################################################")
print("####################################################")
print("####################################################")
print("####################################################")
print("\n\n\n")

ask("人生的意义是什么型", is_thinking=False)
