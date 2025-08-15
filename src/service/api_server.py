from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import base64
import cv2
import numpy as np
import time
from typing import Dict, Any, List, Optional
from configs.config import config
from utils.logger import logger
from inference_engine.realtime_inference import inference_engine
# 替换为新的LangGraph Agent
from langgraph_agent.agent_core import city_agent_graph
from data_processing.multimodal_ingestion import ingestor

# 初始化FastAPI应用
app = FastAPI(
    title="城市事件检测与处理系统API",
    description="提供城市道路积水等事件的实时检测、分析和工单生成服务",
    version="1.1.0 (LangGraph Refactored)"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 数据模型
class EventRequest(BaseModel):
    frame: str
    text_context: str
    cam_id: Optional[str] = "0"
    event_type: Optional[str] = "flooding"


class EdgeUploadRequest(BaseModel):
    cam_id: str
    timestamp: float
    frame: str
    features: List[Dict[str, Any]]
    metrics: Dict[str, Any]


class HealthCheckResponse(BaseModel):
    status: str
    timestamp: float
    services: Dict[str, str]
    version: str


# 工具函数
def base64_to_image(base64_str: str) -> np.ndarray:
    try:
        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("无法解码图像数据")
        return img
    except Exception as e:
        logger.error(f"base64转图像失败: {str(e)}")
        raise HTTPException(status_code=400, detail=f"图像解码错误: {str(e)}")


# 后台任务
def background_analysis(frame: np.ndarray, context: Dict[str, Any]):
    try:
        logger.info(f"开始后台分析任务: {context.get('request_id')}")
        time.sleep(5)
        logger.info(f"后台分析任务完成: {context.get('request_id')}")
    except Exception as e:
        logger.error(f"后台分析任务出错: {str(e)}")


# API端点
@app.get("/health", response_model=HealthCheckResponse)
def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "services": {
            "inference_engine": "active",
            "city_agent_graph": "active",
            "database": "active",
            "data_ingestor": "active"
        },
        "version": "1.1.0"
    }


@app.post("/api/detect_flooding")
def detect_flooding(
        request: EventRequest,
        background_tasks: BackgroundTasks
):
    request_id = f"req-{int(time.time())}-{hash(request.frame) % 1000:03d}"
    logger.info(f"处理积水检测请求: {request_id}")

    try:
        start_time = time.time()

        frame = base64_to_image(request.frame)
        result = inference_engine.infer_road_flooding(frame, request.text_context)

        event_data = {
            "type": request.event_type,
            "location": result.get("location", ""),
            "flood_prob": result.get("flood_prob", 0),
            "description": result.get("description", ""),
            "similar_cases": result.get("similar_cases", []),
            "weather": ingestor.get_weather_data(result.get("location", "")) if result.get("location") else {}
        }

        agent_result = city_agent_graph.process_event(event_data)
        workorder = agent_result.get("workorder", {})

        if not workorder:
            raise Exception("工单生成失败")

        #  使用后台任务异步执行事件归档
        if result.get("frame_feat") is not None:
            background_tasks.add_task(
                inference_engine.archive_event,
                frame_feat=result["frame_feat"],
                workorder=workorder
            )

        total_time = (time.time() - start_time) * 1000
        logger.info(
            f"积水检测请求处理完成: {request_id}, "
            f"耗时: {total_time:.2f}ms, "
            f"工单ID: {workorder.get('workorder_id')}"
        )

        # 从返回结果中移除原始特征，减小响应体大小
        result.pop("frame_feat", None)

        return {
            "request_id": request_id,
            "inference_result": result,
            "workorder": workorder,
            "processing_time_ms": total_time
        }

    except Exception as e:
        logger.error(f"处理积水检测请求出错 {request_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理请求时出错: {str(e)}")


@app.post("/api/edge_upload")
def edge_upload(request: EdgeUploadRequest):
    request_id = f"edge-{int(time.time())}-{hash(request.frame) % 1000:03d}"
    logger.info(f"接收边缘节点上传数据: {request_id}, 摄像头ID: {request.cam_id}")

    try:
        start_time = time.time()

        frame = base64_to_image(request.frame) if request.frame else None
        text_context = f"摄像头ID: {request.cam_id}, 时间戳: {request.timestamp}"

        if frame is not None:
            result = inference_engine.infer_road_flooding(frame, text_context)

            event_data = {
                "type": "flooding",
                "location": f"摄像头 {request.cam_id} 监控区域",
                "flood_prob": result.get("flood_prob", 0),
                "description": result.get("description", ""),
                "similar_cases": result.get("similar_cases", [])
            }

            # *** 修改点：调用新的LangGraph Agent ***
            agent_result = city_agent_graph.process_event(event_data)
            workorder = agent_result.get("workorder", {})
        else:
            result = {"status": "no_frame_received"}
            workorder = {}

        total_time = (time.time() - start_time) * 1000
        logger.info(
            f"边缘节点数据处理完成: {request_id}, "
            f"耗时: {total_time:.2f}ms"
        )

        return {
            "request_id": request_id,
            "status": "success",
            "inference_result": result,
            "workorder": workorder,
            "processing_time_ms": total_time
        }

    except Exception as e:
        logger.error(f"处理边缘节点上传数据出错 {request_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理上传数据时出错: {str(e)}")


@app.get("/api/weather/{area_id}")
def get_weather(area_id: str):
    """获取指定区域的气象数据"""
    try:
        weather_data = ingestor.get_weather_data(area_id)
        return {
            "area_id": area_id,
            "weather_data": weather_data,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"获取气象数据出错 (area_id: {area_id}): {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取气象数据失败: {str(e)}")

@app.get("/api/hotline")
def get_hotline_data(latest_n: int = 10):
    """获取最新的12345热线数据"""
    try:
        hotline_data = ingestor.get_12345_text(latest_n)
        return {
            "count": len(hotline_data),
            "data": hotline_data,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"获取热线数据出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取热线数据失败: {str(e)}")

# 启动服务
if __name__ == "__main__":
    logger.info(f"启动API服务: {config.SERVICE_HOST}:{config.SERVICE_PORT}")
    uvicorn.run(
        "api_server:app",
        host=config.SERVICE_HOST,
        port=config.SERVICE_PORT,
        workers=config.WORKERS,
        log_level=config.LOG_LEVEL.lower()
    )
