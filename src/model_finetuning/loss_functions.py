import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class DiceLoss(nn.Module):
    """Dice损失函数，用于语义分割任务"""

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算Dice损失

        Args:
            pred: 预测张量，形状为(batch_size, num_classes, height, width)
            target: 目标张量，形状为(batch_size, height, width)

        Returns:
            Dice损失值
        """
        # 计算每个类别的Dice系数
        batch_size, num_classes = pred.shape[0], pred.shape[1]
        total_loss = 0.0

        # 对每个样本计算损失
        for i in range(batch_size):
            # 对每个类别计算损失
            for c in range(num_classes):
                # 获取当前样本和类别的预测和目标
                pred_flat = pred[i, c].view(-1)
                target_flat = (target[i] == c).float().view(-1)

                # 计算交并集
                intersection = (pred_flat * target_flat).sum()
                union = pred_flat.sum() + target_flat.sum()

                # 计算Dice系数和损失
                dice = (2. * intersection + self.smooth) / (union + self.smooth)
                total_loss += 1 - dice

        # 平均损失
        return total_loss / (batch_size * num_classes)

class FocalLoss(nn.Module):
    """Focal损失函数，用于解决类别不平衡问题"""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算Focal损失

        Args:
            pred: 预测张量，形状为(batch_size, num_classes, height, width)
            target: 目标张量，形状为(batch_size, height, width)

        Returns:
            Focal损失值
        """
        # 将预测转换为概率
        pred_softmax = F.softmax(pred, dim=1)

        # 获取目标类别的概率
        target_onehot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        p_t = (pred_softmax * target_onehot).sum(dim=1)

        # 计算交叉熵损失
        ce_loss = F.cross_entropy(pred, target, reduction='none')

        # 计算Focal损失
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss

        return focal_loss.mean()

class ModalBalanceLoss(nn.Module):
    """模态平衡损失，用于平衡视觉和文本模态的贡献"""

    def __init__(self, initial_beta: float = 0.5, beta_decay: float = 0.99):
        super().__init__()
        self.initial_beta = initial_beta  # 初始文本权重
        self.beta_decay = beta_decay  # 权重衰减率

    def forward(
        self,
        vision_feat: torch.Tensor,
        text_feat: torch.Tensor,
        modal_mask: torch.Tensor,
        epoch: int
    ) -> torch.Tensor:
        """
        计算模态平衡损失

        Args:
            vision_feat: 视觉特征，形状为(batch_size, feat_dim)
            text_feat: 文本特征，形状为(batch_size, feat_dim)
            modal_mask: 模态掩码，指示有效模态，形状为(batch_size, 2)
            epoch: 当前训练轮数，用于动态调整权重

        Returns:
            模态平衡损失值
        """
        # 动态调整文本模态权重 (随训练进行逐渐增加)
        beta_t = self.initial_beta * (self.beta_decay ** epoch)

        # 计算特征归一化
        vision_norm = F.normalize(vision_feat, p=2, dim=1)
        text_norm = F.normalize(text_feat, p=2, dim=1)

        # 计算模态内一致性损失
        vision_consistency = F.mse_loss(
            vision_norm,
            vision_norm.mean(dim=0, keepdim=True).expand_as(vision_norm)
        )

        text_consistency = F.mse_loss(
            text_norm,
            text_norm.mean(dim=0, keepdim=True).expand_as(text_norm)
        )

        # 计算模态间相似性损失
        cross_similarity = 1 - F.cosine_similarity(vision_norm, text_norm, dim=1).mean()

        # 应用模态掩码 (处理缺失模态)
        vision_mask = modal_mask[:, 0].mean()
        text_mask = modal_mask[:, 1].mean()

        # 组合损失
        balance_loss = (
            (1 - beta_t) * vision_consistency * vision_mask +
            beta_t * text_consistency * text_mask +
            cross_similarity
        )

        return balance_loss

class MultimodalLoss(nn.Module):
    """多模态损失函数，组合多种损失"""

    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()  # 分类损失
        self.modal_balance_loss = ModalBalanceLoss()  # 模态平衡损失
        self.dice_loss = DiceLoss()  # 积水区域分割损失
        self.focal_loss = FocalLoss()  # Focal损失

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        epoch: int
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算多模态总损失

        Args:
            pred: 预测结果字典
            target: 目标标签字典
            epoch: 当前训练轮数

        Returns:
            总损失和各损失分量的字典
        """
        # 1. 交叉熵损失（主分类损失）
        ce = self.ce_loss(pred["logits"], target["label"])

        # 2. 模态平衡损失（动态调整视觉/文本权重）
        mb_loss = self.modal_balance_loss(
            pred["vision_feat"],
            pred["text_feat"],
            target["modal_mask"],
            epoch  # 用于动态调整β_t
        )

        # 3. 积水分割损失（Dice+Focal）
        seg_loss = self.dice_loss(pred["seg_mask"], target["seg_gt"]) + \
                   self.focal_loss(pred["seg_mask"], target["seg_gt"])

        # 总损失（带权重）
        total_loss = ce + 0.3 * mb_loss + 0.5 * seg_loss

        # 返回总损失和各分量
        loss_components = {
            "total_loss": total_loss.item(),
            "ce_loss": ce.item(),
            "modal_balance_loss": mb_loss.item(),
            "segmentation_loss": seg_loss.item()
        }

        return total_loss, loss_components
