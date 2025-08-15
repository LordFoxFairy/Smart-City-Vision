import signal
import sys
import time
from typing import Any, Dict
from configs.config import config
from utils.logger import logger
from service.api_server import app
from data_processing.multimodal_ingestion import ingestor
from inference_engine.realtime_inference import inference_engine
from langgraph_agent.agent_core import city_agent

# 全局状态
system_state: Dict[str, Any] = {
    "running": False,
    "start_time": 0,
    "modules": {
        "ingestor": False,
        "inference": False,
        "agent": False,
        "api": False
    }
}

def signal_handler(sig, frame):
    """信号处理器，用于优雅关闭系统"""
    logger.info(f"接收到信号 {sig}，正在关闭系统...")
    shutdown_system()
    sys.exit(0)

def initialize_system():
    """初始化系统组件"""
    logger.info("开始初始化城市事件检测与处理系统...")
    
    try:
        # 记录启动时间
        system_state["start_time"] = time.time()
        
        # 初始化数据摄入器
        logger.info("初始化数据摄入器...")
        # 摄入器在导入时已初始化
        system_state["modules"]["ingestor"] = True
        
        # 初始化推理引擎
        logger.info("初始化推理引擎...")
        # 推理引擎在导入时已初始化
        system_state["modules"]["inference"] = True
        
        # 初始化智能体
        logger.info("初始化AutoGen智能体...")
        # 智能体在导入时已初始化
        system_state["modules"]["agent"] = True
        
        # 标记系统为运行状态
        system_state["running"] = True
        
        logger.info("系统初始化完成")
        return True
        
    except Exception as e:
        logger.error(f"系统初始化失败: {str(e)}")
        shutdown_system()
        return False

def shutdown_system():
    """关闭系统组件"""
    logger.info("开始关闭系统...")
    
    # 停止标记
    system_state["running"] = False
    
    # 关闭数据摄入器
    if system_state["modules"]["ingestor"]:
        try:
            logger.info("关闭数据摄入器...")
            ingestor.close()
        except Exception as e:
            logger.error(f"关闭数据摄入器出错: {str(e)}")
    
    # 关闭推理引擎
    if system_state["modules"]["inference"]:
        try:
            logger.info("关闭推理引擎...")
            inference_engine.close()
        except Exception as e:
            logger.error(f"关闭推理引擎出错: {str(e)}")
    
    # 计算运行时间
    uptime = time.time() - system_state["start_time"]
    logger.info(f"系统已关闭，运行时间: {uptime:.2f}秒")

def run_api_server():
    """运行API服务器"""
    try:
        logger.info("启动API服务器...")
        import uvicorn
        uvicorn.run(
            "service.api_server:app",
            host=config.SERVICE_HOST,
            port=config.SERVICE_PORT,
            workers=config.WORKERS,
            log_level=config.LOG_LEVEL.lower()
        )
        system_state["modules"]["api"] = True
    except Exception as e:
        logger.error(f"API服务器启动失败: {str(e)}")
        raise

def main():
    """系统主函数"""
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 初始化系统
    if not initialize_system():
        sys.exit(1)
    
    # 运行API服务器
    try:
        run_api_server()
    except Exception as e:
        logger.error(f"系统运行出错: {str(e)}")
        shutdown_system()
        sys.exit(1)

if __name__ == "__main__":
    main()
