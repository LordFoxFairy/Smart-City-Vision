import cv2
import json
import requests
import time
from typing import Dict, List, Tuple, Optional
from rtsp import RTSPReader  # 假设这是一个优化的RTSP读取库
from datetime import datetime
from configs.config import config
from utils.logger import logger

class RTSPConnectionManager:
    """RTSP连接管理器，处理连接池和重连逻辑"""
    
    def __init__(self):
        self.connections = {}
        self.retry_count = 3
        self.retry_delay = 2  # 秒
        
    def get_reader(self, rtsp_url: str) -> RTSPReader:
        """获取RTSP读取器，自动处理重连"""
        if rtsp_url not in self.connections or not self.connections[rtsp_url].is_connected():
            logger.info(f"Connecting to RTSP stream: {rtsp_url}")
            for attempt in range(self.retry_count):
                try:
                    reader = RTSPReader(
                        rtsp_url, 
                        fps=config.FRAME_PROCESS_FPS, 
                        codec='H.264'
                    )
                    self.connections[rtsp_url] = reader
                    logger.info(f"Successfully connected to {rtsp_url}")
                    return reader
                except Exception as e:
                    logger.error(f"Failed to connect to {rtsp_url} (attempt {attempt + 1}/{self.retry_count}): {str(e)}")
                    if attempt < self.retry_count - 1:
                        time.sleep(self.retry_delay)
            
            raise ConnectionError(f"Could not connect to RTSP stream {rtsp_url} after {self.retry_count} attempts")
        
        return self.connections[rtsp_url]
    
    def close_all(self):
        """关闭所有RTSP连接"""
        for url, reader in self.connections.items():
            try:
                reader.close()
                logger.info(f"Closed RTSP connection: {url}")
            except Exception as e:
                logger.error(f"Error closing RTSP connection {url}: {str(e)}")
        self.connections.clear()

class APIClient:
    """API客户端，处理HTTP请求和错误"""
    
    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout
        
    def get(self, url: str, params: Optional[Dict] = None) -> Dict:
        """发送GET请求并返回解析后的JSON"""
        try:
            start_time = time.time()
            response = requests.get(
                url, 
                params=params or {}, 
                timeout=self.timeout
            )
            response.raise_for_status()  # 抛出HTTP错误状态码
            
            # 记录请求耗时
            duration = (time.time() - start_time) * 1000
            logger.debug(f"API request to {url} completed in {duration:.2f}ms")
            
            return response.json()
        except requests.exceptions.Timeout:
            logger.error(f"API request to {url} timed out after {self.timeout}s")
            raise
        except requests.exceptions.HTTPError as e:
            logger.error(f"API request to {url} failed with HTTP error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error in API request to {url}: {str(e)}")
            raise

class MultimodalIngestor:
    """多模态数据摄入器，处理视频流、文本和气象数据"""
    
    def __init__(self):
        self.rtsp_manager = RTSPConnectionManager()
        self.api_client = APIClient()
        self.last_weather_fetch = {}  # 缓存最后一次气象数据获取时间
        self.weather_cache = {}  # 气象数据缓存
        
    def get_video_frames(self, cam_id: str) -> Tuple[cv2.Mat, float]:
        """
        读取RTSP流并返回处理后帧
        
        Args:
            cam_id: 摄像头ID，对应配置中的RTSP URL索引
            
        Returns:
            处理后的帧和时间戳
        """
        try:
            # 从配置中获取对应的RTSP URL
            rtsp_url = config.RTSP_URLS[int(cam_id)]
            
            # 获取RTSP读取器并读取帧
            reader = self.rtsp_manager.get_reader(rtsp_url)
            frame, timestamp = reader.read()
            
            if frame is None:
                raise ValueError(f"Failed to read frame from camera {cam_id}")
                
            # 检查延迟
            current_time = time.time()
            delay = (current_time - timestamp) * 1000  # 转换为毫秒
            if delay > 200:  # 延迟控制≤200ms
                logger.warning(f"High latency for camera {cam_id}: {delay:.2f}ms")
            
            return frame, timestamp
            
        except Exception as e:
            logger.error(f"Error getting video frame from camera {cam_id}: {str(e)}")
            raise
    
    def get_12345_text(self, latest_n: int = 10) -> List[Dict]:
        """
        拉取12345热线文本数据
        
        Args:
            latest_n: 拉取最新的N条数据
            
        Returns:
            热线文本数据列表
        """
        try:
            data = self.api_client.get(
                config.HOTLINE_API,
                params={"limit": latest_n}
            )
            
            # 验证数据格式
            if not isinstance(data, list):
                raise ValueError("12345 API returned unexpected data format")
                
            logger.info(f"Fetched {len(data)} records from 12345 hotline API")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching 12345 text data: {str(e)}")
            raise
    
    def get_weather_data(self, area_id: str, force_refresh: bool = False) -> Dict:
        """
        获取实时气象数据
        
        Args:
            area_id: 区域ID
            force_refresh: 是否强制刷新缓存
            
        Returns:
            气象数据字典
        """
        try:
            # 检查缓存是否有效（5分钟内）
            current_time = time.time()
            cache_valid = (
                area_id in self.weather_cache and 
                area_id in self.last_weather_fetch and
                (current_time - self.last_weather_fetch[area_id]) < 300  # 5分钟
            )
            
            if cache_valid and not force_refresh:
                logger.debug(f"Using cached weather data for area {area_id}")
                return self.weather_cache[area_id]
                
            # 缓存无效，从API获取
            data = self.api_client.get(
                config.WEATHER_API,
                params={"area": area_id}
            )
            
            # 记录获取时间和缓存数据
            self.last_weather_fetch[area_id] = current_time
            self.weather_cache[area_id] = data
            
            logger.info(f"Fetched weather data for area {area_id}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching weather data for area {area_id}: {str(e)}")
            # 如果有缓存，返回缓存数据
            if area_id in self.weather_cache:
                logger.warning(f"Returning cached weather data for area {area_id}")
                return self.weather_cache[area_id]
            raise
    
    def close(self):
        """关闭所有资源"""
        self.rtsp_manager.close_all()
        logger.info("MultimodalIngestor resources closed")

# 创建单例实例
ingestor = MultimodalIngestor()
