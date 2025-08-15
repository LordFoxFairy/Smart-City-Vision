import time
from datetime import datetime
from typing import Dict, Any, TypedDict
from langgraph.graph import StateGraph, END
from configs.config import config
from utils.logger import logger
from tools import RuleBasedReasoner, WorkOrderGenerator


# --- LangGraph 状态定义 ---

class AgentState(TypedDict):
    """
    定义了图（Graph）中传递的状态。
    """
    original_event_data: Dict[str, Any]
    facts_for_rules: Dict[str, Any]
    rule_engine_output: Dict[str, Any]
    work_order: Dict[str, Any]
    start_time: float


# --- LangGraph 节点定义 ---

class AgentNodes:
    """
    定义了图中所有可执行的节点。
    """

    def __init__(self):
        # 使用保留原始接口的工具类
        self.rule_engine = RuleBasedReasoner()
        self.workorder_generator = WorkOrderGenerator()

    def analyze_event(self, state: AgentState) -> Dict[str, Any]:
        """
        节点1：分析事件，准备事实数据。
        """
        logger.info("Node: analyze_event")
        event_data = state['original_event_data']
        facts = {
            "event_type": event_data.get("type", "unknown"),
            "location": event_data.get("location", ""),
            "timestamp": datetime.now().isoformat(),
            "severity": min(100, max(0, int(event_data.get("flood_prob", 0) * 100))),
            "weather": event_data.get("weather", {}),
            "related_cases_count": len(event_data.get("similar_cases", []))
        }
        return {"facts_for_rules": facts}

    def execute_rule_engine(self, state: AgentState) -> Dict[str, Any]:
        """
        节点2：执行规则引擎。
        """
        logger.info("Node: execute_rule_engine")
        facts = state['facts_for_rules']
        rule_result = self.rule_engine.match(facts)
        return {"rule_engine_output": rule_result}

    def generate_work_order(self, state: AgentState) -> Dict[str, Any]:
        """
        节点3：生成最终的工单。
        """
        logger.info("Node: generate_work_order")
        event_data = state['original_event_data']
        rule_result = state['rule_engine_output']
        workorder = self.workorder_generator.generate(event_data, rule_result)

        total_time = time.time() - state['start_time']
        if total_time > config.WORKORDER_TIMEOUT:
            logger.warning(
                f"Workorder generation timeout: {total_time:.2f}s > {config.WORKORDER_TIMEOUT}s"
            )

        workorder['processing_time_ms'] = total_time * 1000
        return {"work_order": workorder}


# --- 构建并编译图 ---

class CityAgentGraph:
    def __init__(self):
        self.nodes = AgentNodes()
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("analyze_event", self.nodes.analyze_event)
        workflow.add_node("execute_rule_engine", self.nodes.execute_rule_engine)
        workflow.add_node("generate_work_order", self.nodes.generate_work_order)
        workflow.set_entry_point("analyze_event")
        workflow.add_edge("analyze_event", "execute_rule_engine")
        workflow.add_edge("execute_rule_engine", "generate_work_order")
        workflow.add_edge("generate_work_order", END)
        return workflow.compile()

    def process_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("CityAgentGraph processing event...")
        initial_state = {"original_event_data": event_data, "start_time": time.time()}
        try:
            final_state = self.graph.invoke(initial_state)
            return {"workorder": final_state.get('work_order', {}), "status": "success"}
        except Exception as e:
            logger.error(f"Error processing event in graph: {str(e)}")
            return {"error": str(e), "status": "failed"}


# 创建一个全局的Agent实例
city_agent_graph = CityAgentGraph()
