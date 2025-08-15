import os
from dotenv import load_dotenv
from datetime import timedelta

# 加载环境变量
load_dotenv()

class Config:
    """系统全局配置"""
    
    # 服务配置
    SERVICE_HOST = os.getenv("SERVICE_HOST", "0.0.0.0")
    SERVICE_PORT = int(os.getenv("SERVICE_PORT", 8000))
    WORKERS = int(os.getenv("WORKERS", 4))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "info")
    
    # 数据来源配置
    RTSP_URLS = os.getenv("RTSP_URLS", "rtsp://camera1:554/stream,rtsp://camera2:554/stream").split(",")
    HOTLINE_API = os.getenv("HOTLINE_API", "http://api.12345.gov.cn/events")
    WEATHER_API = os.getenv("WEATHER_API", "http://api.weather.gov.cn/current")
    
    # 模型配置
    MODEL_PATH = os.getenv("MODEL_PATH", "./models/qwen-vl-72b")
    LORA_MODEL_PATH = os.getenv("LORA_MODEL_PATH", "./models/qwen-vl-72b-lora")
    YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "./models/yolov8n-slim.pt")
    BERT_MODEL_PATH = os.getenv("BERT_MODEL_PATH", "bert-base-chinese")
    
    # 数据库配置
    MILVUS_URI = os.getenv("MILVUS_URI", "./milvus.db")
    MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "city_events")
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    
    # 性能配置
    INFERENCE_TIMEOUT = float(os.getenv("INFERENCE_TIMEOUT", 0.2))  # 推理超时时间(秒)
    WORKORDER_TIMEOUT = float(os.getenv("WORKORDER_TIMEOUT", 1.5))  # 工单生成超时时间(秒)
    FRAME_PROCESS_FPS = int(os.getenv("FRAME_PROCESS_FPS", 25))  # 帧处理帧率
    
    # 缓存配置
    CACHE_TTL = timedelta(minutes=int(os.getenv("CACHE_TTL", 10)))  # 缓存过期时间
    
    # 日志配置
    LOG_DIR = os.getenv("LOG_DIR", "./logs")
    LOG_ROTATION = os.getenv("LOG_ROTATION", "100MB")
    LOG_RETENTION = os.getenv("LOG_RETENTION", "7 days")

# 创建配置实例
config = Config()

# 确保日志目录存在
os.makedirs(config.LOG_DIR, exist_ok=True)
