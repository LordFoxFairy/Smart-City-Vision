# -*- coding: utf-8 -*-
"""
模型微调训练脚本

该脚本用于对多模态模型（如Qwen-VL）进行端到端的微调。
它集成了QLoRA进行高效参数微调，并使用一个自定义的、包含多任务损失的训练器，
以同时优化分类、分割和模态平衡任务。

核心功能:
1.  使用QLoRA加载4-bit量化模型以节省显存。
2.  通过模型包装器 (MultimodalModelWrapper) 为基础模型附加一个分割头。
3.  使用自定义数据集类 (CityEventDataset) 加载图像和文本数据。
4.  使用自定义训练器 (CustomTrainer) 并调用 `MultimodalLoss` 计算复合损失。
5.  通过命令行参数进行灵活配置。

运行示例:
python model_finetuning/train.py \
    --model_path ./models/qwen-vl-72b \
    --dataset_csv ./data/annotations.csv \
    --image_dir ./data/images \
    --output_dir ./results/qwen-vl-finetuned
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, PreTrainedModel, AutoTokenizer
from datasets import Dataset as HFDataset
from typing import Dict, Any
from PIL import Image
import numpy as np
import pandas as pd
import os
import argparse
import math

from model_finetuning.qlora_config import (
    load_model_and_tokenizer,
    get_training_args,
    get_data_collator
)
from model_finetuning.loss_functions import MultimodalLoss
from utils.logger import logger


class MultimodalModelWrapper(PreTrainedModel):
    """
    多模态模型包装器。

    该包装器封装了Hugging Face的基础模型，并为其附加了一个自定义的分割头。
    这么做的目的是为了让模型的输出能够匹配 `MultimodalLoss` 函数的输入要求，
    即同时返回分类logits、视觉/文本特征以及分割掩码。
    """

    def __init__(self, base_model, vision_feature_dim=4096, num_vision_patches=256):
        super().__init__(base_model.config)
        self.base_model = base_model
        self.vision_feature_dim = vision_feature_dim
        self.num_vision_patches = num_vision_patches

        # 分割头：将展平的视觉特征序列转换为2D特征图，然后进行分割
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(self.vision_feature_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, kernel_size=1)  # 分割为2类: 背景 vs 目标
        )

    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, **kwargs):
        # 1. 通过基础模型进行前向传播，获取包含隐藏层状态的输出
        outputs = self.base_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            output_hidden_states=True,
            **kwargs
        )

        logits = outputs.logits
        last_hidden_state = outputs.hidden_states[-1]

        # 2. 从隐藏层状态中提取视觉和文本特征
        # 注意：这里的切片索引是基于对模型架构的假设，需要根据实际模型进行调整。
        # 假设视觉patches在前，文本tokens在后。
        vision_patches = last_hidden_state[:, :self.num_vision_patches, :]
        text_tokens = last_hidden_state[:, self.num_vision_patches:, :]

        vision_feat = vision_patches.mean(dim=1)
        text_feat = text_tokens.mean(dim=1)

        # 3. 将视觉patches重塑为2D特征图，并通过分割头
        # (batch_size, num_patches, dim) -> (batch_size, dim, sqrt(num_patches), sqrt(num_patches))
        grid_size = int(math.sqrt(self.num_vision_patches))
        vision_grid = vision_patches.permute(0, 2, 1).reshape(
            -1, self.vision_feature_dim, grid_size, grid_size
        )
        seg_mask = self.segmentation_head(vision_grid)

        # 将分割掩码上采样至标准输入尺寸 (e.g., 224x224)
        seg_mask = nn.functional.interpolate(
            seg_mask, size=(224, 224), mode='bilinear', align_corners=False
        )

        return {
            "logits": logits,
            "vision_feat": vision_feat,
            "text_feat": text_feat,
            "seg_mask": seg_mask
        }


class CityEventDataset(Dataset):
    """
    用于加载城市事件数据的自定义数据集。
    从一个CSV文件和对应的图像文件夹中读取数据。
    """

    def __init__(self, csv_file: str, image_dir: str, tokenizer: AutoTokenizer):
        self.annotations = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.tokenizer = tokenizer

        # 定义图像预处理流程
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx) -> Dict[str, Any]:
        row = self.annotations.iloc[idx]

        # 加载和处理图像
        image_path = os.path.join(self.image_dir, row['image_file'])
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.transform(image)

        # 加载和处理文本
        text = row['description']
        tokenized = self.tokenizer(
            text, padding="max_length", truncation=True, max_length=512, return_tensors="pt"
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": tokenized["input_ids"].squeeze(),
            "labels": torch.tensor(row['label'], dtype=torch.long),
            "seg_gt": torch.randint(0, 2, (224, 224), dtype=torch.long),  # 占位符：应加载真实的分割掩码
            "modal_mask": torch.tensor([1, 1], dtype=torch.float32),  # 假设图像和文本都存在
        }


class CustomTrainer(Trainer):
    """
    自定义训练器，用于集成 `MultimodalLoss`。
    通过重写 `compute_loss` 方法，我们可以替换掉Hugging Face默认的损失计算逻辑。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multimodal_loss_fct = MultimodalLoss()
        logger.info("CustomTrainer initialized with MultimodalLoss.")

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        seg_gt = inputs.pop("seg_gt")
        modal_mask = inputs.pop("modal_mask")

        outputs = model(**inputs)

        targets = {"label": labels, "seg_gt": seg_gt, "modal_mask": modal_mask}
        epoch = self.state.epoch if self.state.epoch is not None else 0

        loss, loss_components = self.multimodal_loss_fct(
            pred=outputs, target=targets, epoch=int(epoch)
        )

        # 在训练过程中记录各个损失分量，便于监控
        if self.is_world_process_zero() and self.state.global_step > 0 and self.state.global_step % self.args.logging_steps == 0:
            self.log(loss_components)

        return (loss, outputs) if return_outputs else loss


def create_dummy_dataset(path: str = "./data", num_samples: int = 100):
    """创建一个虚拟数据集用于测试"""
    logger.info("Creating a dummy dataset for demonstration...")
    image_dir = os.path.join(path, "images")
    os.makedirs(image_dir, exist_ok=True)

    annotations = []
    for i in range(num_samples):
        img_file = f"dummy_{i}.png"
        img_path = os.path.join(image_dir, img_file)
        Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255)).save(img_path)

        annotations.append({
            "image_file": img_file,
            "description": f"这是第{i}号城市事件的描述，内容为道路积水。",
            "label": i % 5  # 假设有5个类别
        })

    csv_path = os.path.join(path, "annotations.csv")
    pd.DataFrame(annotations).to_csv(csv_path, index=False)
    logger.info(f"Dummy dataset created at {path}")
    return csv_path, image_dir


def main(args):
    logger.info("Starting model fine-tuning process...")

    # 1. 加载QLoRA量化模型和分词器
    base_model, tokenizer = load_model_and_tokenizer(args.model_path)

    # 2. 将基础模型包装起来，以集成自定义的分割头
    model = MultimodalModelWrapper(base_model)

    # 3. 准备数据集
    if args.use_dummy_data:
        csv_path, image_dir = create_dummy_dataset()
    else:
        csv_path, image_dir = args.dataset_csv, args.image_dir

    logger.info(f"Loading dataset from: {csv_path}")
    train_dataset = CityEventDataset(csv_file=csv_path, image_dir=image_dir, tokenizer=tokenizer)

    # 4. 获取训练参数
    training_args = get_training_args(output_dir=args.output_dir)

    # 5. 初始化自定义训练器
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=get_data_collator(tokenizer)
    )

    # 6. 开始训练
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training finished.")

    # 7. 保存最终的模型
    logger.info("Saving fine-tuned model...")
    trainer.save_model()
    logger.info(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多模态模型微调训练脚本")
    parser.add_argument("--model_path", type=str, required=True, help="预训练模型路径")
    parser.add_argument("--dataset_csv", type=str, help="指向数据集CSV文件的路径")
    parser.add_argument("--image_dir", type=str, help="图像文件夹路径")
    parser.add_argument("--output_dir", type=str, default="./results/fine-tuned-model", help="模型输出和检查点目录")
    parser.add_argument("--use_dummy_data", action='store_true', help="如果未提供数据集，则创建并使用虚拟数据集")

    args = parser.parse_args()

    if not args.use_dummy_data and (not args.dataset_csv or not args.image_dir):
        raise ValueError("必须提供 --dataset_csv 和 --image_dir，或者使用 --use_dummy_data")

    main(args)
