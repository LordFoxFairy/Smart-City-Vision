import cv2
import torch
import numpy as np
from typing import Tuple, Union, Optional
from torchvision import transforms
from transformers import BertTokenizer, BertModel
from ultralytics import YOLO
from configs.config import config
from utils.logger import logger


class TSNStabilizer:
    """基于时序分段网络(TSN)的视频稳流去抖器"""

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.frame_buffer = []
        self.prev_gray = None
        self.transform = None

    def stabilize(self, frame: cv2.Mat) -> cv2.Mat:
        """
        稳定视频帧
        
        Args:
            frame: 输入视频帧
            
        Returns:
            稳定后的视频帧
        """
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 初始化
        if self.prev_gray is None:
            self.prev_gray = gray
            self.frame_buffer.append(frame)
            return frame

        # 计算光流
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        # 计算平均位移
        dx = np.mean(flow[..., 0])
        dy = np.mean(flow[..., 1])

        # 构建变换矩阵
        M = np.float32([[1, 0, -dx], [0, 1, -dy]])

        # 应用变换
        h, w = frame.shape[:2]
        stabilized = cv2.warpAffine(frame, M, (w, h))

        # 更新缓冲区和前一帧
        self.frame_buffer.append(stabilized)
        if len(self.frame_buffer) > self.window_size:
            self.frame_buffer.pop(0)
        self.prev_gray = gray

        return stabilized


class ImageEnhancer:
    """图像增强器，处理低光照和噪声问题"""

    def __init__(self, retinex_strength: float = 0.8, denoise_strength: int = 10):
        self.retinex_strength = retinex_strength
        self.denoise_strength = denoise_strength

    def enhance(self, frame: cv2.Mat) -> cv2.Mat:
        """
        增强图像质量
        
        Args:
            frame: 输入图像
            
        Returns:
            增强后的图像
        """
        # 转换为RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Retinex增强（处理低光照）
        retinex = self._multi_scale_retinex(rgb)

        # 非局部均值降噪
        denoised = cv2.fastNlMeansDenoisingColored(
            retinex,
            None,
            self.denoise_strength,
            self.denoise_strength,
            7,
            21
        )

        # 转换回BGR
        return cv2.cvtColor(denoised, cv2.COLOR_RGB2BGR)

    def _multi_scale_retinex(self, img: cv2.Mat) -> cv2.Mat:
        """多尺度Retinex算法实现"""
        # 将图像转换为浮点型
        img_float = img.astype(np.float32) / 255.0

        # 高斯模糊核
        scales = [15, 80, 250]
        retinex = np.zeros_like(img_float)

        for sigma in scales:
            # 高斯模糊
            blur = cv2.GaussianBlur(img_float, (0, 0), sigma)
            # 避免除以零
            blur = np.maximum(blur, 1e-10)
            # 计算Retinex
            retinex += np.log1p(img_float) - np.log1p(blur)

        # 平均并转换回图像格式
        retinex = retinex / len(scales)
        # 归一化
        retinex = (retinex - np.min(retinex)) / (np.max(retinex) - np.min(retinex))
        # 应用强度并转换回8位
        return np.uint8(retinex * 255 * self.retinex_strength)


class FeatureExtractor:
    """特征提取器，处理图像和文本特征"""

    def __init__(self):
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device for feature extraction: {self.device}")

        # 初始化视频稳流器
        self.stabilizer = TSNStabilizer()

        # 初始化图像增强器
        self.enhancer = ImageEnhancer()

        # 文本编码器
        self.bert_tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_PATH)
        self.bert_model = BertModel.from_pretrained(config.BERT_MODEL_PATH).to(self.device)
        self.bert_model.eval()

        # 图像特征提取器 (ResNet50)
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        # 移除最后一层分类层，只保留特征提取部分
        self.resnet = torch.nn.Sequential(*list(self.resnet.children())[:-1]).to(self.device)
        self.resnet.eval()

        # 图像预处理变换
        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # 目标检测模型 (YOLOv8n-slim)
        self.yolo = YOLO(config.YOLO_MODEL_PATH)
        self.yolo.to(self.device)

    def process_video_frame(self, frame: cv2.Mat) -> Tuple[torch.Tensor, Dict]:
        """
        处理视频帧并提取特征
        
        Args:
            frame: 输入视频帧
            
        Returns:
            特征张量和处理元数据
        """
        metadata = {
            "original_shape": frame.shape,
            "processing_steps": []
        }

        try:
            # 1. 视频稳流去抖
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()

            stabilized_frame = self.stabilizer.stabilize(frame)

            end_time.record()
            torch.cuda.synchronize()
            metadata["processing_steps"].append({
                "step": "stabilization",
                "time_ms": start_time.elapsed_time(end_time)
            })

            # 2. 图像增强
            start_time.record()

            enhanced_frame = self.enhancer.enhance(stabilized_frame)

            end_time.record()
            torch.cuda.synchronize()
            metadata["processing_steps"].append({
                "step": "enhancement",
                "time_ms": start_time.elapsed_time(end_time)
            })

            # 3. 目标检测 (YOLOv8)
            start_time.record()

            results = self.yolo(enhanced_frame, conf=0.7)
            detections = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": box.conf[0].item(),
                        "class": self.yolo.names[int(box.cls[0])]
                    })
            metadata["detections"] = detections

            end_time.record()
            torch.cuda.synchronize()
            metadata["processing_steps"].append({
                "step": "detection",
                "time_ms": start_time.elapsed_time(end_time),
                "detections_count": len(detections)
            })

            # 4. 图像特征提取 (ResNet50)
            start_time.record()

            # 转换为RGB并应用预处理
            rgb_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
            img_tensor = self.image_transform(rgb_frame).unsqueeze(0).to(self.device)

            # 提取特征
            with torch.no_grad():
                features = self.resnet(img_tensor).squeeze()  # 输出2048维特征

            end_time.record()
            torch.cuda.synchronize()
            metadata["processing_steps"].append({
                "step": "feature_extraction",
                "time_ms": start_time.elapsed_time(end_time),
                "feature_dim": features.shape[0]
            })

            # 计算总处理时间
            total_time = sum(step["time_ms"] for step in metadata["processing_steps"])
            metadata["total_processing_time_ms"] = total_time

            logger.debug(f"Frame processing completed in {total_time:.2f}ms")

            return features, metadata

        except Exception as e:
            logger.error(f"Error processing video frame: {str(e)}")
            raise

    def process_text(self, text: str) -> Tuple[torch.Tensor, Dict]:
        """
        处理文本并提取特征
        
        Args:
            text: 输入文本
            
        Returns:
            特征张量和处理元数据
        """
        metadata = {
            "original_length": len(text),
            "processing_time_ms": 0
        }

        try:
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()

            # 文本编码
            inputs = self.bert_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding="max_length"
            )

            # 将输入移至设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 提取特征
            with torch.no_grad():
                outputs = self.bert_model(**inputs)

            # 使用CLS token或平均池化作为特征
            features = outputs.last_hidden_state.mean(dim=1).squeeze()  # 768维向量

            end_time.record()
            torch.cuda.synchronize()
            metadata["processing_time_ms"] = start_time.elapsed_time(end_time)
            metadata["feature_dim"] = features.shape[0]

            logger.debug(f"Text processing completed in {metadata['processing_time_ms']:.2f}ms")

            return features, metadata

        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            raise


# 创建特征提取器实例
feature_extractor = FeatureExtractor()
