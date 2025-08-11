import unsloth
import torch
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from swanlab.integration.transformers import SwanLabCallback

# ========== QLoRA 实现：导入基本模型与分词器 with unsloth ==========
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# 【QLoRA 修改点 1】：启用 4-bit 量化
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=2048,
    full_finetuning=False,
    # ✅ QLoRA 关键配置：启用 4-bit 量化
    load_in_4bit=True,  # 这是 QLoRA 的核心：4-bit 量化
    load_in_8bit=False,  # 确保不使用 8-bit（QLoRA 通常用 4-bit）
)

# load the tokenizer and the model with transformers
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.bfloat16,
#     device_map=device_map
# )

# ========== QLoRA 实现：基于Unsloth的带 LoRA 的 PEFT 模型 ==========
# 【QLoRA 修改点 2】：LoRA 配置优化
model = FastLanguageModel.get_peft_model(
    model,
    # ✅ QLoRA 优化：适中的 rank 值（4-bit 量化下推荐 16-64）
    r=16,  # 对于 4-bit 量化，可以尝试更高的 rank（如 32, 64）
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj",
        "gate_proj",
    ],  # attention层和MLP层最好都进行
    # ✅ QLoRA 优化：lora_alpha 通常设置为 rank 的 1-2 倍
    lora_alpha=16,  # 可以设置为 32 (2*rank) 来增强 LoRA 的影响
    lora_dropout=0,  # QLoRA 中通常设为 0
    bias="none",
    # ✅ QLoRA 关键：使用 unsloth 的梯度检查点以节省内存
    use_gradient_checkpointing="unsloth",  # 对 QLoRA 非常重要
    random_state=3407,
    use_rslora=False,  # 可以尝试启用 RS-LoRA
    loftq_config=None,  # 对于 QLoRA 可以考虑启用 LoftQ
)

# ========== 预处理成Qwen3适配的ChatML数据集 ==========
# 这里省略数据处理部分，按用户要求


# swanlab 检测配置
swanlab_callback = SwanLabCallback(
    project="Qwen3-8B-fintune",
    experiment_name="Qwen3-8B-combind-8-11-QLoRA",  # 标注为 QLoRA
    description="8月11使用通义千问Qwen3-8B模型在FreedomIntelligence/medical-o1-reasoning-SFT和BAAI/IndustryInstruction_Health-Medicine数据集上进行QLoRA微调。",
    config={
        "model": "Qwen/Qwen3-8B",
        "dataset": "https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT",
        "train_data_number": len(combined_dataset),  # 这个变量需要从数据处理部分获取
        "lora_rank": 16,
        "lora_alpha": 16,
        "lora_dropout": 0,
        # ✅ QLoRA 标识
        "quantization": "4-bit",
        "method": "QLoRA",
    },
)

# ========== QLoRA 训练器配置 ==========
trainer = SFTTrainer(
    model=model,  # PEFT 包装后的模型
    # processing_class=tokenizer,  # 分词器
    tokenizer=tokenizer,
    dataset_text_field="text",
    train_dataset=train_ds,  # 训练数据集，需要从数据处理部分获取
    eval_dataset=None,  # 评估数据集（此处未设置）
    callbacks=[swanlab_callback],  # 实验追踪回调
    args=SFTConfig(
        # ========== 输出和保存配置 ==========
        output_dir="./qlora_model—811",  # 改为 qlora_model
        save_steps=100,  # 每100步保存一次检查点
        logging_steps=5,  # 每5步记录一次日志
        logging_first_step=5,  # 第5步开始记录日志
        # ========== QLoRA 训练超参数 ==========
        # ✅ QLoRA 修改点 3：调整批次大小和学习率
        per_device_train_batch_size=2,  # QLoRA 可以支持稍大的批次
        gradient_accumulation_steps=8,  # 相应减少累积步数
        num_train_epochs=4,  # 训练轮数
        # ✅ QLoRA 学习率调整：4-bit 量化可能需要稍高的学习率
        learning_rate=3e-4,  # QLoRA 通常比普通 LoRA 需要稍高的学习率
        warmup_steps=10,  # 稍微增加预热步数
        # ========== 优化器和调度器 ==========
        # ✅ QLoRA 优化器：8-bit AdamW 与量化模型配合良好
        optim="adamw_8bit",  # 8位AdamW优化器，节省内存
        weight_decay=0.01,  # 权重衰减，防止过拟合
        lr_scheduler_type="linear",  # 线性学习率衰减
        max_grad_norm=1.0,  # 梯度裁剪阈值
        # ========== 分布式和精度配置 ==========
        # ✅ QLoRA 精度设置
        bf16=True,  # 使用 bf16 精度
        fp16=False,  # 不使用 fp16
        # deepspeed=DS_CONFIG,  # DeepSpeed 配置文件
        # ========== 其他配置 ==========
        seed=3407,  # 随机种子，确保可重现性
        report_to="none",  # 不使用额外的实验追踪工具
    ),
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
    f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# ========== QLoRA 模型保存 ==========
# ✅ QLoRA 修改点 4：保存方式
model.save_pretrained("qlora_model—811")  # Local saving
tokenizer.save_pretrained("qlora_model—811")

# 可选：保存为不同格式
# model.save_pretrained_merged("qlora_model—811", tokenizer, save_method="merged_4bit")  # 4-bit 合并保存
# model.save_pretrained_merged("qlora_model—811", tokenizer, save_method="lora")  # 只保存 LoRA 权重
