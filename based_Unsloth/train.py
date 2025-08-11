import unsloth
import torch
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from swanlab.integration.transformers import SwanLabCallback

# 导入基本模型与分词器 with unsloth
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name, max_seq_length=2048, full_finetuning=False
)

# load the tokenizer and the model with transformers
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.bfloat16,
#     device_map=device_map
# )

# ========== 基于Unsloth的带 lora 的peft模型 ==========
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # recommand 8,16,32
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj",
        "gate_proj",
    ],  # attention层和MLP层最好都进行
    lora_alpha=16,  # recommand rank or 2 * rank
    lora_dropout=0,
    bias="none",
    # [新特性] "unsloth"模式减少30%显存，可适应2倍大的批次大小
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,  # 不使用RS-LoRA 除非你的rank给的比较大
    loftq_config=None,  # 会使用预训练权重中前'r'个奇异向量来初始化 LoRA 矩阵。这种方法能提升准确率，但可能在训练初期引发显著的内存峰值
)

# ========== 预处理成Qwen3适配的ChatML数据集 ==========


# swanlab 检测配置
swanlab_callback = SwanLabCallback(
    project="Qwen3-8B-fintune",
    experiment_name="Qwen3-8B-combind-8-11",
    description="8月11使用通义千问Qwen3-8B模型在FreedomIntelligence/medical-o1-reasoning-SFT和BAAI/IndustryInstruction_Health-Medicine数据集上微调。",
    config={
        "model": "Qwen/Qwen3-8B",
        "dataset": "https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT",
        "train_data_number": len(combined_dataset),
        "lora_rank": 16,
        "lora_alpha": 16,
        "lora_dropout": 0,
    },
)

# 训练器配置
trainer = SFTTrainer(
    model=model,  # PEFT 包装后的模型
    # processing_class=tokenizer,  # 分词器
    tokenizer=tokenizer,
    dataset_text_field="text",
    train_dataset=train_ds,  # 训练数据集
    eval_dataset=None,  # 评估数据集（此处未设置）
    callbacks=[swanlab_callback],  # 实验追踪回调
    args=SFTConfig(
        # ========== 输出和保存配置 ==========
        output_dir="./lora_model—811",  # 模型输出目录
        save_steps=100,  # 每100步保存一次检查点
        logging_steps=5,  # 每5步记录一次日志
        logging_first_step=5,  # 第5步开始记录日志
        # ========== 训练超参数 ==========
        per_device_train_batch_size=1,  # 每设备批次大小
        gradient_accumulation_steps=16,  # 梯度累积步数（模拟batch_size=16）
        num_train_epochs=4,  # 训练轮数
        learning_rate=2e-4,  #  # 学习率（长期训练可降至2e-5）对于常规 LoRA/QLoRA 微调，我们建议从 2e-4 开始作为初始学习率。
        warmup_steps=5,  # 学习率预热步数
        # ========== 优化器和调度器 ==========
        optim="adamw_8bit",  # 8位AdamW优化器，节省内存
        weight_decay=0.01,  # 权重衰减，防止过拟合
        lr_scheduler_type="linear",  # 线性学习率衰减
        max_grad_norm=1.0,  # 梯度裁剪阈值 计算所有梯度的L2范数，如果大于1，则按比例缩放整个梯度向量，防止梯度爆炸
        # ========== 分布式和精度配置 ==========
        bf16=True,
        fp16=False,
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


model.save_pretrained("lora_model")  # Local saving
tokenizer.save_pretrained("lora_model")
