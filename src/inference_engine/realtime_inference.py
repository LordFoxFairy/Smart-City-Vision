import time
import torch
import numpy as np
from typing import Dict, List, Any, Tuple
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from milvus import default_server
from milvus.client import MilvusClient
from neo4j import GraphDatabase
from configs.config import config
from utils.logger import logger
from data_processing.preprocessing import feature_extractor


class MilvusVectorDB:
    """Milvus向量数据库客户端，用于相似案例检索与归档"""

    def __init__(self):
        self.client = MilvusClient(uri=config.MILVUS_URI)
        self.collection_name = config.MILVUS_COLLECTION
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """确保集合存在，不存在则创建"""
        if not self.client.has_collection(collection_name=self.collection_name):
            logger.info(f"Creating Milvus collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=2048,
                metric_type="IP",
                index_type="IVF_FLAT",
                index_param={"nlist": 1024}
            )
        else:
            logger.info(f"Using existing Milvus collection: {self.collection_name}")
            self.client.load_collection(collection_name=self.collection_name)

    def insert(self, features: np.ndarray, metadata: Dict[str, Any]) -> int:
        """插入特征向量和元数据，用于事件归档"""
        try:
            if len(features.shape) > 1:
                features = features.flatten()
            data = [{"vector": features.tolist(), "metadata": metadata}]
            result = self.client.insert(collection_name=self.collection_name, data=data)
            logger.info(f"Successfully inserted {result['insert_count']} vector(s) into Milvus.")
            return result["insert_count"]
        except Exception as e:
            logger.error(f"Error inserting into Milvus: {str(e)}")
            raise

    def search(self, features: np.ndarray, limit: int = 3) -> List[Dict[str, Any]]:
        """搜索相似向量"""
        try:
            if len(features.shape) > 1:
                features = features.flatten()
            results = self.client.search(
                collection_name=self.collection_name,
                data=[features.tolist()],
                limit=limit,
                output_fields=["metadata"]
            )
            return [{"id": hit["id"], "distance": hit["distance"], "metadata": hit["entity"]["metadata"]} for hit in
                    results[0]]
        except Exception as e:
            logger.error(f"Error searching Milvus: {str(e)}")
            return []


class Neo4jKnowledgeGraph:
    """Neo4j知识图谱客户端，用于关联查询与事件记录"""

    def __init__(self):
        self.driver = None
        try:
            self.driver = GraphDatabase.driver(
                config.NEO4J_URI,
                auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
            )
            self._test_connection()
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j, Knowledge Graph features will be disabled: {str(e)}")

    def _test_connection(self):
        if self.driver:
            with self.driver.session() as session:
                session.run("MATCH (n) RETURN count(n) AS count")
            logger.info("Successfully connected to Neo4j")

    def query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        if not self.driver: return []
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Error executing Neo4j query: {str(e)}")
            return []

    def create_event_node(self, workorder: Dict[str, Any]):
        """
        在知识图谱中创建并关联事件节点，用于事件归档。
        """
        if not self.driver:
            logger.warning("Neo4j connection not available, skipping event node creation.")
            return

        query = """
        // 创建或更新事件节点
        MERGE (e:FloodEvent {workorder_id: $workorder_id})
        ON CREATE SET e.created_at = $created_at
        SET e.location = $location,
            e.description = $description,
            e.priority = $priority,
            e.status = $status

        // 关联到地理位置节点 (假设已存在)
        WITH e
        MERGE (l:Location {name: $location})
        MERGE (e)-[:OCCURRED_AT]->(l)
        """
        try:
            with self.driver.session() as session:
                session.run(query, parameters=workorder)
            logger.info(f"Successfully created/updated event node {workorder['workorder_id']} in Neo4j.")
        except Exception as e:
            logger.error(f"Failed to create event node in Neo4j: {str(e)}")

    def get_pipe_info_by_location(self, location: str) -> List[Dict[str, Any]]:
        """根据位置获取相关排水管道信息"""
        query = """
        MATCH (p:Pipe)-[:LOCATED_IN]->(l:Location {name: $location})
        RETURN p.id AS pipe_id, p.capacity AS capacity, p.material AS material
        LIMIT 5
        """
        return self.query(query, parameters={"location": location})

    def close(self):
        if self.driver: self.driver.close()


class InferenceEngine:
    """推理引擎，处理多模态输入并生成推理结果"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device for inference: {self.device}")

        self.model, self.tokenizer = self._load_model()
        self.milvus_db = MilvusVectorDB()
        self.neo4j_kg = Neo4jKnowledgeGraph()

        self.vqa_pipeline = pipeline(
            "visual-question-answering",
            model=self.model, tokenizer=self.tokenizer, device=0 if self.device.type == "cuda" else -1
        )

    def _load_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """加载基础模型和LoRA适配器"""
        try:
            logger.info(f"Loading base model from {config.MODEL_PATH}")
            model = AutoModelForCausalLM.from_pretrained(
                config.MODEL_PATH, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
            )
            if config.LORA_MODEL_PATH:
                logger.info(f"Loading LoRA adapter from {config.LORA_MODEL_PATH}")
                model = PeftModel.from_pretrained(model, config.LORA_MODEL_PATH, torch_dtype=torch.float16)
                model = model.merge_and_unload()

            tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH, trust_remote_code=True)
            tokenizer.pad_token = tokenizer.eos_token
            model.eval()
            logger.info("Model loaded successfully for inference")
            return model, tokenizer
        except Exception as e:
            logger.error(f"Error loading inference model: {str(e)}")
            raise

    def infer_road_flooding(self, frame: np.ndarray, text_context: str) -> Dict[str, Any]:
        """道路积水识别推理"""
        start_time = time.time()
        inference_result = {
            "timestamp": start_time, "processing_metrics": {}, "flood_prob": 0.0,
            "similar_cases": [], "pipe_info": [], "frame_feat": None,
            "location": text_context.split("地点:")[-1].strip() if "地点:" in text_context else ""
        }

        try:
            # 1. 提取图像特征
            feature_start = time.time()
            frame_feat, _ = feature_extractor.process_video_frame(frame)
            inference_result["frame_feat"] = frame_feat  # 保存特征以供后续归档
            inference_result["processing_metrics"]["feature_extraction_ms"] = (time.time() - feature_start) * 1000

            # 2. 多模态推理 (Qwen-VL)
            vqa_start = time.time()
            vqa_result = self.vqa_pipeline(
                image=frame,
                question=f"检测图像中的积水区域，分析积水严重程度（0-100%），结合上下文信息：{text_context}。请返回积水概率（0-1）和简要描述。",
                top_k=1
            )
            inference_result["processing_metrics"]["vqa_inference_ms"] = (time.time() - vqa_start) * 1000
            inference_result["flood_prob"] = vqa_result[0]["score"]
            inference_result["description"] = vqa_result[0]["answer"]

            # 3. 检索相似案例 (Milvus)
            milvus_start = time.time()
            inference_result["similar_cases"] = self.milvus_db.search(frame_feat.cpu().numpy(), limit=3)
            inference_result["processing_metrics"]["milvus_search_ms"] = (time.time() - milvus_start) * 1000

            # 4. 知识图谱关联排水管网
            kg_start = time.time()
            if inference_result["location"]:
                inference_result["pipe_info"] = self.neo4j_kg.get_pipe_info_by_location(inference_result["location"])
            inference_result["processing_metrics"]["kg_query_ms"] = (time.time() - kg_start) * 1000

            total_time = (time.time() - start_time) * 1000
            inference_result["processing_metrics"]["total_inference_ms"] = total_time

            logger.debug(f"Road flooding inference completed in {total_time:.2f}ms")
            return inference_result

        except Exception as e:
            logger.error(f"Error in road flooding inference: {str(e)}")
            inference_result["error"] = str(e)
            return inference_result

    def archive_event(self, frame_feat: torch.Tensor, workorder: Dict[str, Any]):
        """
        归档已处理的事件，用于系统持续学习。
        """
        try:
            logger.info(f"Archiving event for workorder: {workorder['workorder_id']}")

            # 1. 归档到 Milvus
            milvus_metadata = {
                "workorder_id": workorder["workorder_id"],
                "timestamp": workorder["created_at"],
                "location": workorder["location"],
                "description": workorder["description"],
                "priority": workorder["priority"]
            }
            self.milvus_db.insert(frame_feat.cpu().numpy(), milvus_metadata)

            # 2. 归档到 Neo4j
            self.neo4j_kg.create_event_node(workorder)

            logger.info(f"Successfully archived event {workorder['workorder_id']}.")

        except Exception as e:
            logger.error(f"Failed to archive event {workorder.get('workorder_id', 'N/A')}: {str(e)}")

    def close(self):
        """关闭推理引擎资源"""
        self.neo4j_kg.close()
        logger.info("Inference engine resources closed")

# 创建推理引擎实例
inference_engine = InferenceEngine()
