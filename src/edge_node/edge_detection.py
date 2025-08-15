import cv2
import torch
import numpy as np
import time
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
import base64
import requests
from configs.config import config
from utils.logger import logger

class FeatureExtractor:
    """特征提取器，提取图像的局部和全局特征"""
    
    def __init__(self):
        # 初始化SURF特征提取器
        self.surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)
        
        # 确保OpenCV支持SURF
        if not hasattr(cv2.xfeatures2d, 'SURF_create'):
            logger.warning("OpenCV does not support SURF. Using ORB instead.")
            self.surf = cv2.ORB_create()
    
    def extract_surf_features(self, roi: np.ndarray, max_features: int = 128) -> np.ndarray:
        """
        提取SURF特征
        
        Args:
            roi: 感兴趣区域图像
            max_features: 最大特征数量
            
        Returns:
            提取的特征向量
        """
        # 转换为灰度图
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        
        # 检测关键点和计算描述符
        keypoints, descriptors = self.surf.detectAndCompute(gray, None)
        
        if descriptors is None:
            return np.zeros((max_features, 64), dtype=np.float32)  # SURF特征维度为64
        
        # 限制特征数量
        if len(descriptors) > max_features:
            descriptors = descriptors[:max_features]
        # 如果特征不足，用零填充
        elif len(descriptors) < max_features:
            descriptors = np.pad(
                descriptors, 
                ((0, max_features - len(descriptors)), (0, 0)), 
                mode='constant'
            )
        
        return descriptors

class EdgeDetector:
    """边缘节点目标检测器，用于前置过滤低置信度目标"""
    
    def __init__(self, server_url: str = f"http://{config.SERVICE_HOST}:{config.SERVICE_PORT}"):
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Edge detector using device: {self.device}")
        
        # 加载轻量化模型（YOLOv8n-slim）
        self.model = self._load_model()
        
        # 特征提取器
        self.feature_extractor = FeatureExtractor()
        
        # 中心服务器URL
        self.server_url = server_url
        
        # 性能指标
        self.performance_metrics = {
            "detection_time": [],
            "feature_extraction_time": [],
            "network_transfer_time": [],
            "frames_processed": 0,
            "objects_detected": 0,
            "objects_transferred": 0
        }
    
    def _load_model(self) -> YOLO:
        """加载YOLO模型并优化"""
        try:
            logger.info(f"Loading YOLO model from {config.YOLO_MODEL_PATH}")
            
            # 加载模型
            model = YOLO(config.YOLO_MODEL_PATH)
            
            # 移动到设备
            model = model.to(self.device)
            
            # 模型优化
            model.fuse()  # 层融合
            model.eval()  # 评估模式
            
            logger.info("YOLO model loaded and optimized for edge device")
            return model
            
        except Exception as e:
            logger.error(f"Error loading YOLO model: {str(e)}")
            raise
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        预处理帧图像
        
        Args:
            frame: 输入帧
            
        Returns:
            预处理后的帧
        """
        # 调整大小（保持纵横比）
        h, w = frame.shape[:2]
        max_dim = 640  # 边缘设备处理的最大尺寸
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        
        # 转换为RGB（YOLO默认输入）
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
        return frame
    
    def detect_and_filter(self, frame: np.ndarray, confidence_threshold: float = 0.7) -> Tuple[List[Dict], np.ndarray]:
        """
        检测并过滤低置信度目标
        
        Args:
            frame: 输入帧
            confidence_threshold: 置信度阈值
            
        Returns:
            高置信度目标特征列表和处理后的帧
        """
        start_time = time.time()
        self.performance_metrics["frames_processed"] += 1
        
        try:
            # 预处理帧
            processed_frame = self.preprocess_frame(frame)
            
            # 目标检测
            det_start = time.time()
            results = self.model(
                processed_frame, 
                conf=confidence_threshold,
                device=self.device,
                verbose=False
            )
            det_time = time.time() - det_start
            self.performance_metrics["detection_time"].append(det_time)
            
            high_conf_features = []
            
            # 处理检测结果
            for result in results:
                for box in result.boxes:
                    self.performance_metrics["objects_detected"] += 1
                    
                    # 只处理高置信度目标
                    if box.conf[0] >= confidence_threshold:
                        # 提取ROI
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        roi = processed_frame[y1:y2, x1:x2]
                        
                        if roi.size == 0:  # 跳过空ROI
                            continue
                            
                        # 提取特征
                        feat_start = time.time()
                        surf_feat = self.feature_extractor.extract_surf_features(roi)
                        
                        # 记录特征提取时间
                        feat_time = time.time() - feat_start
                        self.performance_metrics["feature_extraction_time"].append(feat_time)
                        
                        # 存储特征和相关信息
                        high_conf_features.append({
                            "feature": {
                                "surf": surf_feat.tolist(),
                            },
                            "bbox": [x1, y1, x2, y2],
                            "confidence": box.conf[0].item(),
                            "class": self.model.names[int(box.cls[0])],
                            "timestamp": time.time()
                        })
                        
                        self.performance_metrics["objects_transferred"] += 1
        
            # 计算总处理时间
            total_time = time.time() - start_time
            logger.debug(
                f"Edge detection completed in {total_time:.2f}s. "
                f"Detected {len(high_conf_features)} high-confidence objects."
            )
            
            return high_conf_features, processed_frame
            
        except Exception as e:
            logger.error(f"Error in detect_and_filter: {str(e)}")
            return [], frame
    
    def encode_frame(self, frame: np.ndarray) -> str:
        """
        将帧编码为base64字符串
        
        Args:
            frame: 输入帧
            
        Returns:
            base64编码字符串
        """
        # 转换为JPEG
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        # 编码为base64
        return base64.b64encode(buffer).decode('utf-8')
    
    def send_to_server(self, features: List[Dict], frame: np.ndarray, cam_id: str) -> Dict:
        """
        将特征和帧发送到中心服务器
        
        Args:
            features: 特征列表
            frame: 处理后的帧
            cam_id: 摄像头ID
            
        Returns:
            服务器响应
        """
        if not features:
            logger.debug("No high-confidence features to send")
            return {"status": "no_data"}
        
        try:
            start_time = time.time()
            
            # 编码帧
            encoded_frame = self.encode_frame(frame)
            
            # 准备数据
            payload = {
                "cam_id": cam_id,
                "timestamp": time.time(),
                "frame": encoded_frame,
                "features": features,
                "metrics": {
                    "frame_size": len(encoded_frame),
                    "num_features": len(features)
                }
            }
            
            # 发送到服务器
            response = requests.post(
                f"{self.server_url}/api/edge_upload",
                json=payload,
                timeout=5.0
            )
            response.raise_for_status()
            
            # 记录传输时间
            transfer_time = time.time() - start_time
            self.performance_metrics["network_transfer_time"].append(transfer_time)
            
            logger.debug(
                f"Data sent to server. Response: {response.status_code}, "
                f"Transfer time: {transfer_time:.2f}s"
            )
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error sending data to server: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_performance_report(self) -> Dict:
        """生成性能报告"""
        report = {
            "frames_processed": self.performance_metrics["frames_processed"],
            "objects_detected": self.performance_metrics["objects_detected"],
            "objects_transferred": self.performance_metrics["objects_transferred"],
            "transfer_rate": (
                self.performance_metrics["objects_transferred"] / 
                self.performance_metrics["objects_detected"] 
                if self.performance_metrics["objects_detected"] > 0 
                else 0
            ),
        }
        
        # 添加平均时间
        for metric in ["detection_time", "feature_extraction_time", "network_transfer_time"]:
            if self.performance_metrics[metric]:
                report[f"avg_{metric}"] = sum(self.performance_metrics[metric]) / len(self.performance_metrics[metric])
            else:
                report[f"avg_{metric}"] = 0
        
        return report

# 创建边缘检测器实例
edge_detector = EdgeDetector()
