import torch
from typing import Dict, Any
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from configs.config import config
from utils.logger import logger

def get_bnb_config() -> BitsAndBytesConfig:
    """
    配置BitsAndBytes量化参数
    
    Returns:
        BitsAndBytesConfig对象
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",  # 使用NF4量化类型
        bnb_4bit_compute_dtype=torch.float16,  # 计算使用float16
        bnb_4bit_quant_storage=torch.float16  # 存储量化参数使用float16
    )

def get_lora_config() -> LoraConfig:
    """
    配置LoRA参数
    
    Returns:
        LoraConfig对象
    """
    return LoraConfig(
        r=8,  # LoRA注意力维度
        lora_alpha=32,  # LoRA缩放参数
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # 注意力投影层
            "gate_proj", "up_proj", "down_proj"  # FFN层
        ],
        lora_dropout=0.05,  # Dropout概率
        bias="none",  # 不训练偏置参数
        task_type="CAUSAL_LM",  # 任务类型
        inference_mode=False  # 训练模式
    )

def load_model_and_tokenizer() -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    加载量化模型和分词器
    
    Returns:
        模型和分词器的元组
    """
    try:
        logger.info(f"Loading model from {config.MODEL_PATH}")
        
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            config.MODEL_PATH,
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token  # 设置pad token
        
        # 加载量化模型
        model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_PATH,
            quantization_config=get_bnb_config(),
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # 准备模型进行kbit训练
        model = prepare_model_for_kbit_training(model)
        
        # 应用LoRA适配器
        lora_config = get_lora_config()
        model = get_peft_model(model, lora_config)
        
        # 打印可训练参数信息
        model.print_trainable_parameters()
        
        logger.info("Model and tokenizer loaded successfully")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading model and tokenizer: {str(e)}")
        raise

def get_training_args(output_dir: str = "./results") -> TrainingArguments:
    """
    获取训练参数配置
    
    Args:
        output_dir: 训练结果输出目录
        
    Returns:
        TrainingArguments对象
    """
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,  # 梯度累积，等效增大batch size
        learning_rate=5e-5,  # 学习率
        num_train_epochs=10,  # 训练轮数
        lr_scheduler_type="cosine_with_restarts",  # 学习率调度器
        warmup_steps=500,  # 预热步数
        weight_decay=0.01,  # 权重衰减
        fp16=True,  # 使用混合精度训练
        logging_steps=10,  # 日志记录步数
        evaluation_strategy="steps",  # 评估策略
        eval_steps=100,  # 评估步数
        save_strategy="steps",  # 保存策略
        save_steps=100,  # 保存步数
        load_best_model_at_end=True,  # 训练结束加载最佳模型
        metric_for_best_model="loss",  # 最佳模型评估指标
        greater_is_better=False,  # 指标越小越好
        report_to="tensorboard",  # 报告到tensorboard
        seed=42,  # 随机种子
        data_seed=42,  # 数据随机种子
        optim="paged_adamw_8bit",  # 使用8bit优化器
        max_grad_norm=0.3,  # 梯度裁剪
        max_steps=-1,  # 最大步数，-1表示由epochs决定
        remove_unused_columns=False  # 不移除未使用的列
    )

def get_data_collator(tokenizer: AutoTokenizer) -> DataCollatorForLanguageModeling:
    """
    获取数据整理器
    
    Args:
        tokenizer: 分词器
        
    Returns:
        DataCollatorForLanguageModeling对象
    """
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # 不使用掩码语言模型
        pad_to_multiple_of=8  # 填充到8的倍数
    )

def save_lora_model(model: AutoModelForCausalLM, output_dir: str = None) -> None:
    """
    保存LoRA模型权重
    
    Args:
        model: 训练好的模型
        output_dir: 输出目录，默认为配置中的路径
    """
    try:
        output_dir = output_dir or config.LORA_MODEL_PATH
        
        # 保存LoRA适配器权重
        model.save_pretrained(output_dir)
        logger.info(f"LoRA model saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error saving LoRA model: {str(e)}")
        raise
