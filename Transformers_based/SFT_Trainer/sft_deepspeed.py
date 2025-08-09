"""
修复说明：
1. 移除了DeepSpeed配置，避免与Unsloth的兼容性冲突
2. 现在使用普通的python命令启动而不是deepspeed命令：
   python sft_train_muxue.py
3. Unsloth本身提供了高效的内存优化，无需DeepSpeed
"""
import unsloth
from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from datasets import load_dataset,Dataset
from swanlab.integration.transformers import SwanLabCallback
import pandas as pd
from trl import SFTTrainer, SFTConfig
from transformers import TextStreamer,BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model
# import deepspeed  # 注释掉DeepSpeed，避免与Unsloth冲突
# DS_CONFIG = "/root/autodl-tmp/ds_z2_offload_config.json"
from typing import Optional, List, Union
import sys
from unsloth.chat_templates import get_chat_template


model_name = "/root/autodl-tmp/Qwen/Qwen3-8B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)

device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} 
# device_map = {"": "cuda:0"} 

# 在执行 8-bit 量化矩阵乘法（MatMul8bitLt）时，输入张量的数据类型是 bfloat16 或 float32，
# 但 bitsandbytes 的 8-bit 量化内核只支持 float16 作为输入精度，
# 所以系统会自动把输入从 bfloat16 / float32 转换成 float16。
# 设置量化配置
# 8bit量化配置
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
)
# 4bit配置
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.bfloat16,  # 计算时使用 bfloat16
#     bnb_4bit_use_double_quant=True,         # 双层量化，进一步压缩
#     bnb_4bit_quant_type="nf4",              # 量化类型：nf4 或 fp4
# )
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # torch_dtype=torch.bfloat16,
    device_map=device_map,
    torch_dtype=torch.float16,   # ✅ 改为 float16，匹配 bitsandbytes 要求
    quantization_config=bnb_config,          # 8bit量化关键参数 权重就以 int8 格式加载，torch_dtype 被自动忽略
    # load_in_8bit=True
)

model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=4,  # Lora 秩
    lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1,  # Dropout 比例
)

model = get_peft_model(model, config)


#**************数据集加载与预处理
# import json
# def load_json_to_data(file_path):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         data = json.load(file)
#     # 将数据转换为DataFrame
#     return Dataset.from_list(data)
# file_path = r'F:\workspace\Fine_tuning\based_transformers\SFT_Trainer\muice-dataset-train.catgirl\muice-dataset-train.catgirl.json'
# 加载数据 json加载
# dataset = load_json_to_data(file_path)


# modelscope数据集下载
from modelscope.msdatasets import MsDataset
ds =  MsDataset.load('himzhzx/muice-dataset-train.catgirl',cache_dir='/root/autodl-tmp/muxue_data/',split=['train'])
ds = ds[0] 
# map需要的映射函数，应用模板转化为ChatML格式
def format_chat_example(example):
    """
    将原始数据格式转换为聊天格式
    """
    # 创建消息列表
    messages = []
    
    # 添加用户消息
    user_content = example["instruction"]
    if example["input"]:  # 如果有input，将其添加到instruction后面
        user_content += "\n" + example["input"]
    
    messages.append({"role": "user", "content": user_content})
    
    # 添加助手消息
    messages.append({"role": "assistant", "content": example["output"]})
    
    return {"texts": messages}

# 应用映射函数
formatted_dataset = ds.map(format_chat_example,remove_columns=ds.column_names)

# 应用模型的apply_chat_template转化为模型可以识别的文本形式
def apply_chat_template_to_qwen(example, tokenizer):
    """
    将聊天格式的数据应用tokenizer的chat template转换为模型输入文本
    """
    # 获取messages
    messages = example["texts"]
    
    # 应用chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False           # 弄成False避免生格式化的时候有提示
    )
    
    return {"text": text}

# 应用函数到数据集
formatted_dataset_with_text = formatted_dataset.map(
    lambda x: apply_chat_template_to_qwen(x, tokenizer),
    remove_columns=["texts"]
)
ds = formatted_dataset_with_text.shuffle(seed=52)

train_ds = ds.shuffle(seed=42)

# ********配置跟踪Swanlab********
swanlab_callback = SwanLabCallback(
    project="Qwen3-8B-fintune",
    experiment_name="Qwen3-8B-no_reasoning_catgirl",
    description="使用通义千问Qwen3-8B模型在猫娘数据集上微调。",
    config={
        "model": "Qwen/Qwen3-8B",
        "dataset": "himzhzx/muice-dataset-train.catgirl",
        "train_data_number": len(train_ds),
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
    }
)

TrainerArgs = SFTConfig(
        # ========== 输出和保存配置 ==========
        output_dir="./lora_model",  # 模型输出目录
        save_steps=50,  # 每50步保存一次检查点
        logging_steps=5,  # 每5步记录一次日志
        logging_first_step=5,  # 第5步开始记录日志
        
        # ========== 训练超参数 ==========
        per_device_train_batch_size=1,  # 每设备批次大小
        gradient_accumulation_steps=16,  # 梯度累积步数（模拟batch_size=16）
        num_train_epochs=3,  # 训练轮数
        learning_rate=2e-4,  # 学习率（初期可用较大值）
        warmup_steps=5,  # 学习率预热步数
        
        # ========== 优化器和调度器 ==========
        optim="adamw_8bit",  # 8位AdamW优化器，节省内存，与Unsloth配合使用
        weight_decay=0.01,  # 权重衰减，防止过拟合
        lr_scheduler_type="linear",  # 线性学习率衰减
        max_grad_norm=1.0,  # 梯度裁剪阈值
        
        # ========== 分布式和精度配置 ==========
        bf16=False,           # ❌ 如果你没用量化，你可以用bf16
        fp16=True,            # ✅ 开启 fp16
        # deepspeed=DS_CONFIG,  # 注释掉DeepSpeed配置，避免与Unsloth冲突
        
        # ========== 其他配置 ==========
        seed=3407,  # 随机种子，确保可重现性
        report_to="none",  # 不使用额外的实验追踪工具
)

trainer = SFTTrainer(
    model = model,
    tokenizer=tokenizer,
    train_dataset = train_ds,
    eval_dataset = None,  # 可以设置评估数据集
    callbacks=[swanlab_callback],
    args = TrainerArgs,
    dataset_text_field = "text", # 可选，但建议加上以保证代码清晰
)

# 显示当前内存统计
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

# 显示最终内存和时间统计
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


# 保存模型 以及lora的配置信息
model.save_pretrained("./lora_model")  # Local saving 保存了模型的核心部分（LoRA权重），用于后续的推理、部署或分享
tokenizer.save_pretrained("./lora_model") # Local saving
