# -*- coding: utf-8 -*-
"""
深度研究Agent系统 (DeepResearch Agent)
基于LLM驱动的智能研究助手，具备多阶段推理、知识图谱构建、自适应学习等能力

主要特性:
1. 多阶段研究管道 - 问题分析、信息收集、深度分析、综合推理
2. 知识图谱构建 - 动态构建和更新领域知识图谱
3. 自适应推理机制 - 根据问题复杂度选择合适的推理策略
4. 持续学习能力 - 从研究过程中学习和改进
5. 多模态信息融合 - 整合文本、数据、结构化知识
"""

import json
import time
import random
import hashlib
import re
import math
from typing import Dict, List, Any, Optional, Union, Callable, Tuple, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import sqlite3


class ResearchPhase(Enum):
    """研究阶段枚举"""
    PROBLEM_ANALYSIS = "problem_analysis"
    INFORMATION_GATHERING = "information_gathering"
    DEEP_ANALYSIS = "deep_analysis"
    SYNTHESIS = "synthesis"
    VALIDATION = "validation"
    CONCLUSION = "conclusion"


class ReasoningMode(Enum):
    """推理模式枚举"""
    QUICK = "quick"           # 快速推理
    THOROUGH = "thorough"     # 彻底推理
    CREATIVE = "creative"     # 创意推理
    ANALYTICAL = "analytical" # 分析推理
    SYSTEMATIC = "systematic" # 系统推理


@dataclass
class ResearchQuery:
    """研究查询结构"""
    query: str
    domain: str = "general"
    complexity: int = 1  # 1-5复杂度等级
    urgency: int = 1     # 1-5紧急程度
    depth_required: int = 3  # 1-5深度要求
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchStep:
    """研究步骤记录"""
    phase: ResearchPhase
    step_type: str  # "thought", "action", "observation", "insight"
    content: str
    confidence: float = 0.0
    sources: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeNode:
    """知识图谱节点"""
    id: str
    content: str
    node_type: str  # "concept", "fact", "relation", "hypothesis"
    domain: str
    confidence: float = 0.0
    connections: Set[str] = field(default_factory=set)
    evidence: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ResearchInsight:
    """研究洞察"""
    content: str
    insight_type: str  # "pattern", "contradiction", "gap", "connection"
    confidence: float
    supporting_evidence: List[str] = field(default_factory=list)
    implications: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class IntelligentReasoner:
    """智能推理器 - 自适应选择推理策略"""
    
    def __init__(self):
        self.reasoning_patterns = {
            ReasoningMode.QUICK: self._quick_reasoning,
            ReasoningMode.THOROUGH: self._thorough_reasoning, 
            ReasoningMode.CREATIVE: self._creative_reasoning,
            ReasoningMode.ANALYTICAL: self._analytical_reasoning,
            ReasoningMode.SYSTEMATIC: self._systematic_reasoning
        }
        self.performance_history = defaultdict(list)
    
    def select_reasoning_mode(self, query: ResearchQuery, context: Dict[str, Any]) -> ReasoningMode:
        """智能选择推理模式"""
        # 基于查询特征选择推理模式
        if query.complexity <= 2 and query.urgency >= 4:
            return ReasoningMode.QUICK
        elif query.depth_required >= 4:
            return ReasoningMode.THOROUGH
        elif "创新" in query.query or "新" in query.query:
            return ReasoningMode.CREATIVE
        elif "分析" in query.query or "比较" in query.query:
            return ReasoningMode.ANALYTICAL
        else:
            return ReasoningMode.SYSTEMATIC
    
    def reason(self, query: ResearchQuery, context: Dict[str, Any], mode: Optional[ReasoningMode] = None) -> List[str]:
        """执行推理"""
        if mode is None:
            mode = self.select_reasoning_mode(query, context)
        
        return self.reasoning_patterns[mode](query, context)
    
    def _quick_reasoning(self, query: ResearchQuery, context: Dict[str, Any]) -> List[str]:
        """快速推理模式"""
        reasoning_steps = [
            f"快速分析问题: {query.query}",
            "识别核心关键词和概念",
            "调用已有知识进行直接匹配",
            "生成初步答案"
        ]
        return reasoning_steps
    
    def _thorough_reasoning(self, query: ResearchQuery, context: Dict[str, Any]) -> List[str]:
        """彻底推理模式"""
        reasoning_steps = [
            f"深入分析问题的多个维度: {query.query}",
            "分解问题为子问题",
            "系统性收集相关信息",
            "多角度分析每个子问题",
            "综合分析结果",
            "验证推理逻辑",
            "形成全面结论"
        ]
        return reasoning_steps
    
    def _creative_reasoning(self, query: ResearchQuery, context: Dict[str, Any]) -> List[str]:
        """创意推理模式"""
        reasoning_steps = [
            f"从创新角度重新审视问题: {query.query}",
            "寻找非传统的思考角度",
            "联想相关但不直接的领域知识",
            "生成多个假设性方案",
            "评估创新方案的可行性",
            "整合最有潜力的创新想法"
        ]
        return reasoning_steps
    
    def _analytical_reasoning(self, query: ResearchQuery, context: Dict[str, Any]) -> List[str]:
        """分析推理模式"""
        reasoning_steps = [
            f"系统分析问题结构: {query.query}",
            "识别变量和影响因素",
            "建立因果关系模型",
            "量化分析各因素权重",
            "对比不同方案的优劣",
            "得出基于数据的结论"
        ]
        return reasoning_steps
    
    def _systematic_reasoning(self, query: ResearchQuery, context: Dict[str, Any]) -> List[str]:
        """系统推理模式"""
        reasoning_steps = [
            f"系统性地梳理问题: {query.query}",
            "构建问题的概念框架",
            "按逻辑顺序收集信息",
            "建立知识结构图",
            "进行结构化分析",
            "形成系统性结论"
        ]
        return reasoning_steps


class KnowledgeGraph:
    """动态知识图谱"""
    
    def __init__(self):
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.connections: Dict[str, Dict[str, float]] = defaultdict(dict)  # 连接权重
        self.domains: Dict[str, Set[str]] = defaultdict(set)
        self.update_history = []
    
    def add_node(self, node: KnowledgeNode) -> str:
        """添加知识节点"""
        self.nodes[node.id] = node
        self.domains[node.domain].add(node.id)
        
        # 自动发现连接
        self._discover_connections(node)
        
        self.update_history.append({
            "action": "add_node",
            "node_id": node.id,
            "timestamp": datetime.now()
        })
        
        return node.id
    
    def add_connection(self, node1_id: str, node2_id: str, weight: float = 1.0, relation_type: str = "related"):
        """添加节点连接"""
        if node1_id in self.nodes and node2_id in self.nodes:
            self.connections[node1_id][node2_id] = weight
            self.connections[node2_id][node1_id] = weight
            
            # 更新节点连接集合
            self.nodes[node1_id].connections.add(node2_id)
            self.nodes[node2_id].connections.add(node1_id)
    
    def _discover_connections(self, new_node: KnowledgeNode):
        """自动发现节点连接"""
        for existing_id, existing_node in self.nodes.items():
            if existing_id != new_node.id:
                # 计算相似度
                similarity = self._calculate_similarity(new_node.content, existing_node.content)
                
                if similarity > 0.3:  # 相似度阈值
                    self.add_connection(new_node.id, existing_id, similarity)
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def get_related_nodes(self, node_id: str, max_distance: int = 2) -> List[KnowledgeNode]:
        """获取相关节点"""
        if node_id not in self.nodes:
            return []
        
        related = []
        visited = set()
        queue = [(node_id, 0)]
        
        while queue:
            current_id, distance = queue.pop(0)
            
            if current_id in visited or distance > max_distance:
                continue
            
            visited.add(current_id)
            
            if distance > 0:  # 不包括自己
                related.append(self.nodes[current_id])
            
            # 添加邻居节点
            for neighbor_id in self.nodes[current_id].connections:
                if neighbor_id not in visited:
                    queue.append((neighbor_id, distance + 1))
        
        # 按连接权重排序
        related.sort(key=lambda node: self.connections.get(node_id, {}).get(node.id, 0), reverse=True)
        
        return related
    
    def search_nodes(self, query: str, domain: Optional[str] = None) -> List[KnowledgeNode]:
        """搜索相关节点"""
        query_words = set(re.findall(r'\w+', query.lower()))
        results = []
        
        for node in self.nodes.values():
            if domain and node.domain != domain:
                continue
            
            node_words = set(re.findall(r'\w+', node.content.lower()))
            similarity = len(query_words & node_words) / len(query_words | node_words) if query_words | node_words else 0
            
            if similarity > 0.1:  # 最低相似度阈值
                results.append((node, similarity))
        
        # 按相似度排序
        results.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in results[:10]]  # 返回前10个结果
    
    def get_domain_overview(self, domain: str) -> Dict[str, Any]:
        """获取领域概览"""
        if domain not in self.domains:
            return {}
        
        node_ids = self.domains[domain]
        nodes = [self.nodes[nid] for nid in node_ids]
        
        # 统计信息
        node_types = defaultdict(int)
        avg_confidence = 0
        
        for node in nodes:
            node_types[node.node_type] += 1
            avg_confidence += node.confidence
        
        avg_confidence /= len(nodes) if nodes else 1
        
        # 找到中心节点（连接最多的节点）
        central_nodes = sorted(nodes, key=lambda n: len(n.connections), reverse=True)[:3]
        
        return {
            "domain": domain,
            "total_nodes": len(nodes),
            "node_types": dict(node_types),
            "average_confidence": avg_confidence,
            "central_nodes": [{"id": n.id, "content": n.content[:100]} for n in central_nodes],
            "last_updated": max([n.timestamp for n in nodes]) if nodes else None
        }


class ResearchPipeline:
    """多阶段研究管道"""
    
    def __init__(self, knowledge_graph: KnowledgeGraph, reasoner: IntelligentReasoner):
        self.knowledge_graph = knowledge_graph
        self.reasoner = reasoner
        self.research_tools = self._initialize_tools()
        self.phase_handlers = {
            ResearchPhase.PROBLEM_ANALYSIS: self._analyze_problem,
            ResearchPhase.INFORMATION_GATHERING: self._gather_information,
            ResearchPhase.DEEP_ANALYSIS: self._deep_analysis,
            ResearchPhase.SYNTHESIS: self._synthesize,
            ResearchPhase.VALIDATION: self._validate,
            ResearchPhase.CONCLUSION: self._conclude
        }
    
    def _initialize_tools(self) -> Dict[str, Callable]:
        """初始化研究工具"""
        return {
            "web_search": self._web_search_tool,
            "literature_search": self._literature_search_tool,
            "data_analysis": self._data_analysis_tool,
            "expert_knowledge": self._expert_knowledge_tool,
            "trend_analysis": self._trend_analysis_tool
        }
    
    def execute_research(self, query: ResearchQuery) -> Dict[str, Any]:
        """执行完整研究流程"""
        research_context = {
            "query": query,
            "steps": [],
            "insights": [],
            "knowledge_used": [],
            "tools_called": [],
            "confidence_scores": []
        }
        
        # 按阶段执行研究
        for phase in ResearchPhase:
            phase_result = self.phase_handlers[phase](query, research_context)
            research_context["steps"].extend(phase_result["steps"])
            research_context["insights"].extend(phase_result.get("insights", []))
            research_context["tools_called"].extend(phase_result.get("tools_called", []))
            
            # 更新知识图谱
            if "new_knowledge" in phase_result:
                for knowledge in phase_result["new_knowledge"]:
                    self._add_to_knowledge_graph(knowledge, query.domain)
        
        return research_context
    
    def _analyze_problem(self, query: ResearchQuery, context: Dict[str, Any]) -> Dict[str, Any]:
        """问题分析阶段"""
        steps = []
        insights = []
        
        # 分析问题复杂度
        complexity_factors = self._assess_complexity(query.query)
        steps.append(ResearchStep(
            phase=ResearchPhase.PROBLEM_ANALYSIS,
            step_type="analysis",
            content=f"问题复杂度评估: {complexity_factors}",
            confidence=0.8
        ))
        
        # 识别关键概念
        key_concepts = self._extract_key_concepts(query.query)
        steps.append(ResearchStep(
            phase=ResearchPhase.PROBLEM_ANALYSIS,
            step_type="insight",
            content=f"识别关键概念: {key_concepts}",
            confidence=0.9
        ))
        
        # 确定研究范围
        scope = self._determine_research_scope(query, key_concepts)
        steps.append(ResearchStep(
            phase=ResearchPhase.PROBLEM_ANALYSIS,
            step_type="planning",
            content=f"确定研究范围: {scope}",
            confidence=0.7
        ))
        
        return {
            "steps": steps,
            "insights": insights,
            "key_concepts": key_concepts,
            "scope": scope
        }
    
    def _gather_information(self, query: ResearchQuery, context: Dict[str, Any]) -> Dict[str, Any]:
        """信息收集阶段"""
        steps = []
        tools_called = []
        new_knowledge = []
        
        # 从知识图谱搜索相关信息
        related_nodes = self.knowledge_graph.search_nodes(query.query, query.domain)
        steps.append(ResearchStep(
            phase=ResearchPhase.INFORMATION_GATHERING,
            step_type="action",
            content=f"从知识图谱检索到 {len(related_nodes)} 个相关节点",
            confidence=0.8
        ))
        
        # 模拟使用各种工具收集信息
        information_sources = ["web_search", "literature_search", "expert_knowledge"]
        
        for tool_name in information_sources:
            if tool_name in self.research_tools:
                tool_result = self.research_tools[tool_name](query.query)
                steps.append(ResearchStep(
                    phase=ResearchPhase.INFORMATION_GATHERING,
                    step_type="action",
                    content=f"使用{tool_name}工具: {tool_result}",
                    confidence=0.7
                ))
                tools_called.append(tool_name)
                
                # 将工具结果转换为知识
                new_knowledge.append({
                    "content": tool_result,
                    "source": tool_name,
                    "confidence": 0.7
                })
        
        return {
            "steps": steps,
            "tools_called": tools_called,
            "new_knowledge": new_knowledge
        }
    
    def _deep_analysis(self, query: ResearchQuery, context: Dict[str, Any]) -> Dict[str, Any]:
        """深度分析阶段"""
        steps = []
        insights = []
        
        # 使用智能推理器进行深度分析
        reasoning_steps = self.reasoner.reason(query, context)
        
        for i, reasoning_step in enumerate(reasoning_steps):
            steps.append(ResearchStep(
                phase=ResearchPhase.DEEP_ANALYSIS,
                step_type="thought",
                content=reasoning_step,
                confidence=0.8 - i * 0.05  # 置信度随推理深度递减
            ))
        
        # 识别模式和关联
        patterns = self._identify_patterns(context)
        if patterns:
            insights.append(ResearchInsight(
                content=f"发现关键模式: {patterns}",
                insight_type="pattern",
                confidence=0.7
            ))
        
        # 发现知识缺口
        gaps = self._identify_knowledge_gaps(query, context)
        if gaps:
            insights.append(ResearchInsight(
                content=f"识别知识缺口: {gaps}",
                insight_type="gap",
                confidence=0.6
            ))
        
        return {
            "steps": steps,
            "insights": insights
        }
    
    def _synthesize(self, query: ResearchQuery, context: Dict[str, Any]) -> Dict[str, Any]:
        """综合阶段"""
        steps = []
        insights = []
        
        # 整合所有信息
        synthesis_content = self._integrate_information(context)
        steps.append(ResearchStep(
            phase=ResearchPhase.SYNTHESIS,
            step_type="synthesis",
            content=synthesis_content,
            confidence=0.8
        ))
        
        # 生成新的洞察
        new_insights = self._generate_insights(context)
        insights.extend(new_insights)
        
        return {
            "steps": steps,
            "insights": insights
        }
    
    def _validate(self, query: ResearchQuery, context: Dict[str, Any]) -> Dict[str, Any]:
        """验证阶段"""
        steps = []
        
        # 逻辑一致性检查
        consistency_score = self._check_logical_consistency(context)
        steps.append(ResearchStep(
            phase=ResearchPhase.VALIDATION,
            step_type="validation",
            content=f"逻辑一致性评分: {consistency_score:.2f}",
            confidence=consistency_score
        ))
        
        # 证据支持度检查
        evidence_score = self._assess_evidence_support(context)
        steps.append(ResearchStep(
            phase=ResearchPhase.VALIDATION,
            step_type="validation", 
            content=f"证据支持度评分: {evidence_score:.2f}",
            confidence=evidence_score
        ))
        
        return {
            "steps": steps
        }
    
    def _conclude(self, query: ResearchQuery, context: Dict[str, Any]) -> Dict[str, Any]:
        """结论阶段"""
        steps = []
        
        # 生成最终结论
        conclusion = self._generate_conclusion(query, context)
        steps.append(ResearchStep(
            phase=ResearchPhase.CONCLUSION,
            step_type="conclusion",
            content=conclusion,
            confidence=0.9
        ))
        
        # 提出后续研究方向
        future_directions = self._suggest_future_research(query, context)
        steps.append(ResearchStep(
            phase=ResearchPhase.CONCLUSION,
            step_type="suggestion",
            content=f"后续研究建议: {future_directions}",
            confidence=0.7
        ))
        
        return {
            "steps": steps
        }
    
    # 工具实现
    def _web_search_tool(self, query: str) -> str:
        """模拟网络搜索工具"""
        search_results = {
            "人工智能": "AI技术正在快速发展，在各个领域都有广泛应用",
            "机器学习": "机器学习是AI的核心技术，包括监督学习、无监督学习等",
            "深度学习": "深度学习基于神经网络，在图像识别和自然语言处理等领域表现出色",
            "python": "Python是最受欢迎的编程语言之一，特别适合数据科学和AI开发"
        }
        
        for key in search_results:
            if key in query.lower():
                return f"网络搜索结果: {search_results[key]}"
        
        return f"网络搜索结果: 关于'{query}'的最新信息和观点"
    
    def _literature_search_tool(self, query: str) -> str:
        """模拟文献搜索工具"""
        return f"文献搜索结果: 找到与'{query}'相关的学术论文和研究报告"
    
    def _data_analysis_tool(self, query: str) -> str:
        """模拟数据分析工具"""
        return f"数据分析结果: 对'{query}'相关数据进行统计分析，发现重要趋势"
    
    def _expert_knowledge_tool(self, query: str) -> str:
        """模拟专家知识工具"""
        return f"专家知识: 领域专家对'{query}'的专业见解和经验分享"
    
    def _trend_analysis_tool(self, query: str) -> str:
        """模拟趋势分析工具"""
        return f"趋势分析: '{query}'领域的发展趋势和未来预测"
    
    # 分析方法
    def _assess_complexity(self, query: str) -> Dict[str, Any]:
        """评估问题复杂度"""
        factors = {
            "query_length": len(query.split()),
            "question_types": len(re.findall(r'[?？]', query)),
            "technical_terms": len(re.findall(r'[A-Z][a-z]+|技术|算法|模型', query)),
            "complexity_score": min(5, (len(query.split()) + len(re.findall(r'[?？]', query)) * 2) / 10)
        }
        return factors
    
    def _extract_key_concepts(self, query: str) -> List[str]:
        """提取关键概念"""
        # 简单的关键词提取
        words = re.findall(r'\w+', query)
        # 过滤常见词汇，保留重要概念
        important_words = [w for w in words if len(w) > 3 and w not in ['什么', '如何', '为什么', '怎么']]
        return important_words[:5]  # 返回前5个重要词汇
    
    def _determine_research_scope(self, query: ResearchQuery, key_concepts: List[str]) -> Dict[str, Any]:
        """确定研究范围"""
        return {
            "primary_domain": query.domain,
            "key_concepts": key_concepts,
            "depth_level": query.depth_required,
            "estimated_time": query.complexity * 10,  # 分钟
            "required_tools": ["web_search", "literature_search"]
        }
    
    def _identify_patterns(self, context: Dict[str, Any]) -> List[str]:
        """识别模式"""
        # 模拟模式识别
        patterns = [
            "技术发展呈指数增长趋势",
            "跨领域应用越来越普遍",
            "开源社区推动创新"
        ]
        return random.sample(patterns, random.randint(1, 2))
    
    def _identify_knowledge_gaps(self, query: ResearchQuery, context: Dict[str, Any]) -> List[str]:
        """识别知识缺口"""
        gaps = [
            "缺乏最新的实证研究数据",
            "需要更多的案例研究",
            "理论与实践之间的桥梁待建立"
        ]
        return random.sample(gaps, random.randint(1, 2))
    
    def _integrate_information(self, context: Dict[str, Any]) -> str:
        """整合信息"""
        step_count = len(context["steps"])
        insight_count = len(context["insights"])
        tool_count = len(set(context["tools_called"]))
        
        return f"整合了 {step_count} 个研究步骤，发现 {insight_count} 个关键洞察，使用了 {tool_count} 种工具"
    
    def _generate_insights(self, context: Dict[str, Any]) -> List[ResearchInsight]:
        """生成洞察"""
        insights = []
        
        # 基于研究步骤生成洞察
        if len(context["steps"]) > 10:
            insights.append(ResearchInsight(
                content="研究过程揭示了问题的多层次结构",
                insight_type="connection",
                confidence=0.8
            ))
        
        if len(context["tools_called"]) > 2:
            insights.append(ResearchInsight(
                content="多工具融合提供了更全面的视角",
                insight_type="pattern",
                confidence=0.7
            ))
        
        return insights
    
    def _check_logical_consistency(self, context: Dict[str, Any]) -> float:
        """检查逻辑一致性"""
        # 模拟逻辑一致性检查
        return random.uniform(0.7, 0.95)
    
    def _assess_evidence_support(self, context: Dict[str, Any]) -> float:
        """评估证据支持度"""
        # 基于工具调用数量和洞察数量评估
        tool_score = min(1.0, len(set(context["tools_called"])) / 3)
        insight_score = min(1.0, len(context["insights"]) / 5)
        return (tool_score + insight_score) / 2
    
    def _generate_conclusion(self, query: ResearchQuery, context: Dict[str, Any]) -> str:
        """生成结论"""
        key_findings = len(context["insights"])
        confidence = sum(step.confidence for step in context["steps"]) / len(context["steps"])
        
        return f"基于深度研究分析，对问题'{query.query}'的研究发现了 {key_findings} 个关键洞察，整体置信度为 {confidence:.2f}。研究表明该问题具有多维度特征，需要综合考虑多个因素。"
    
    def _suggest_future_research(self, query: ResearchQuery, context: Dict[str, Any]) -> str:
        """建议后续研究方向"""
        suggestions = [
            "深入研究具体应用场景",
            "扩大样本规模进行验证",
            "探索跨领域的关联性",
            "开发更精确的评估方法"
        ]
        return "; ".join(random.sample(suggestions, 2))
    
    def _add_to_knowledge_graph(self, knowledge: Dict[str, Any], domain: str):
        """添加知识到图谱"""
        node_id = hashlib.md5(knowledge["content"].encode()).hexdigest()[:16]
        
        node = KnowledgeNode(
            id=node_id,
            content=knowledge["content"],
            node_type="fact",
            domain=domain,
            confidence=knowledge.get("confidence", 0.5),
            evidence=[knowledge.get("source", "unknown")]
        )
        
        self.knowledge_graph.add_node(node)


class DeepResearchAgent:
    """深度研究Agent - 主类"""
    
    def __init__(self, name: str = "DeepResearch Agent", domain: str = "general"):
        self.name = name
        self.domain = domain
        
        # 初始化核心组件
        self.knowledge_graph = KnowledgeGraph()
        self.reasoner = IntelligentReasoner()
        self.pipeline = ResearchPipeline(self.knowledge_graph, self.reasoner)
        
        # 研究历史和状态
        self.research_history = []
        self.performance_metrics = {
            "total_research_count": 0,
            "avg_confidence": 0.0,
            "domain_expertise": defaultdict(float),
            "reasoning_mode_usage": defaultdict(int)
        }
        
        # 学习和适应参数
        self.learning_rate = 0.1
        self.confidence_threshold = 0.7
        self.max_research_depth = 5
    
    def research(self, query: str, domain: Optional[str] = None, 
                complexity: int = 3, depth_required: int = 3, urgency: int = 2) -> Dict[str, Any]:
        """执行深度研究"""
        
        # 构建研究查询
        research_query = ResearchQuery(
            query=query,
            domain=domain or self.domain,
            complexity=complexity,
            depth_required=depth_required,
            urgency=urgency
        )
        
        print(f"🔍 开始深度研究: {query}")
        print(f"🎨 研究配置: 复杂度={complexity}, 深度={depth_required}, 紧急度={urgency}")
        
        # 选择推理模式
        reasoning_mode = self.reasoner.select_reasoning_mode(research_query, {})
        print(f"🧠 选择推理模式: {reasoning_mode.value}")
        
        # 执行研究管道
        research_result = self.pipeline.execute_research(research_query)
        
        # 处理结果
        final_result = self._process_research_result(research_query, research_result, reasoning_mode)
        
        # 更新学习指标
        self._update_learning_metrics(research_query, research_result, reasoning_mode)
        
        # 保存研究历史
        self.research_history.append({
            "query": research_query,
            "result": final_result,
            "timestamp": datetime.now()
        })
        
        return final_result
    
    def _process_research_result(self, query: ResearchQuery, result: Dict[str, Any], 
                                reasoning_mode: ReasoningMode) -> Dict[str, Any]:
        """处理研究结果"""
        
        # 计算总体置信度
        total_confidence = sum(step.confidence for step in result["steps"]) / len(result["steps"])
        
        # 生成最终答案
        final_answer = self._generate_comprehensive_answer(query, result)
        
        # 提取关键洞察
        key_insights = [insight.content for insight in result["insights"]]
        
        # 评估研究质量
        quality_score = self._assess_research_quality(result)
        
        return {
            "query": query.query,
            "domain": query.domain,
            "reasoning_mode": reasoning_mode.value,
            "final_answer": final_answer,
            "key_insights": key_insights,
            "total_confidence": total_confidence,
            "quality_score": quality_score,
            "research_steps": len(result["steps"]),
            "tools_used": len(set(result["tools_called"])),
            "knowledge_nodes_consulted": len(result["knowledge_used"]),
            "phases_completed": len(ResearchPhase),
            "detailed_steps": [{
                "phase": step.phase.value,
                "type": step.step_type,
                "content": step.content,
                "confidence": step.confidence
            } for step in result["steps"]],
            "timestamp": datetime.now().isoformat()
        }
    
    def _generate_comprehensive_answer(self, query: ResearchQuery, result: Dict[str, Any]) -> str:
        """生成综合答案"""
        
        # 获取结论步骤
        conclusion_steps = [step for step in result["steps"] if step.phase == ResearchPhase.CONCLUSION]
        
        if conclusion_steps:
            main_conclusion = conclusion_steps[0].content
        else:
            main_conclusion = f"对于问题'{query.query}'，经过深入研究分析，我们得出了综合性的研究结果。"
        
        # 整合关键洞察
        insights_summary = ""
        if result["insights"]:
            insights_list = [f"- {insight.content}" for insight in result["insights"][:3]]
            insights_summary = f"\n\n💡 关键洞察:\n" + "\n".join(insights_list)
        
        # 整合工具使用结果
        tools_summary = ""
        if result["tools_called"]:
            unique_tools = list(set(result["tools_called"]))
            tools_summary = f"\n\n🔧 使用工具: {', '.join(unique_tools)}"
        
        # 综合答案
        comprehensive_answer = f"{main_conclusion}{insights_summary}{tools_summary}"
        
        return comprehensive_answer
    
    def _assess_research_quality(self, result: Dict[str, Any]) -> float:
        """评估研究质量"""
        
        # 多维度评估
        factors = {
            "completeness": min(1.0, len(result["steps"]) / 15),  # 完整性
            "depth": min(1.0, len([s for s in result["steps"] if s.step_type == "thought"]) / 8),  # 思考深度
            "diversity": min(1.0, len(set(result["tools_called"])) / 3),  # 工具多样性
            "insights": min(1.0, len(result["insights"]) / 3),  # 洞察数量
            "confidence": sum(step.confidence for step in result["steps"]) / len(result["steps"])  # 平均置信度
        }
        
        # 加权平均
        weights = {
            "completeness": 0.2,
            "depth": 0.25,
            "diversity": 0.2,
            "insights": 0.2,
            "confidence": 0.15
        }
        
        quality_score = sum(factors[key] * weights[key] for key in factors)
        return round(quality_score, 3)
    
    def _update_learning_metrics(self, query: ResearchQuery, result: Dict[str, Any], 
                                reasoning_mode: ReasoningMode):
        """更新学习指标"""
        
        # 更新总体指标
        self.performance_metrics["total_research_count"] += 1
        
        # 更新平均置信度
        current_confidence = sum(step.confidence for step in result["steps"]) / len(result["steps"])
        old_avg = self.performance_metrics["avg_confidence"]
        count = self.performance_metrics["total_research_count"]
        self.performance_metrics["avg_confidence"] = (old_avg * (count - 1) + current_confidence) / count
        
        # 更新领域专业度
        domain = query.domain
        old_expertise = self.performance_metrics["domain_expertise"][domain]
        self.performance_metrics["domain_expertise"][domain] = (
            old_expertise * (1 - self.learning_rate) + current_confidence * self.learning_rate
        )
        
        # 更新推理模式使用统计
        self.performance_metrics["reasoning_mode_usage"][reasoning_mode.value] += 1
    
    def get_knowledge_overview(self, domain: Optional[str] = None) -> Dict[str, Any]:
        """获取知识概览"""
        target_domain = domain or self.domain
        return self.knowledge_graph.get_domain_overview(target_domain)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        return {
            "agent_name": self.name,
            "primary_domain": self.domain,
            "performance_metrics": dict(self.performance_metrics),
            "total_knowledge_nodes": len(self.knowledge_graph.nodes),
            "research_history_count": len(self.research_history),
            "knowledge_domains": list(self.knowledge_graph.domains.keys()),
            "last_research": self.research_history[-1]["timestamp"].isoformat() if self.research_history else None
        }
    
    def add_domain_knowledge(self, content: str, domain: str, node_type: str = "concept", 
                           confidence: float = 0.8) -> str:
        """添加领域知识"""
        
        node_id = hashlib.md5(f"{content}{domain}{datetime.now()}".encode()).hexdigest()[:16]
        
        node = KnowledgeNode(
            id=node_id,
            content=content,
            node_type=node_type,
            domain=domain,
            confidence=confidence
        )
        
        self.knowledge_graph.add_node(node)
        print(f"✅ 已添加知识节点: {content[:50]}...")
        
        return node_id
    
    def explore_research_topic(self, topic: str, max_depth: int = 3) -> Dict[str, Any]:
        """探索性研究主题"""
        
        exploration_results = []
        
        # 生成多个相关问题
        related_questions = self._generate_related_questions(topic)
        
        for i, question in enumerate(related_questions[:max_depth]):
            print(f"\n🔎 探索问题 {i+1}: {question}")
            
            result = self.research(
                query=question,
                complexity=2,  # 中等复杂度
                depth_required=2,  # 中等深度
                urgency=1  # 低紧急度
            )
            
            exploration_results.append({
                "question": question,
                "result": result
            })
        
        return {
            "topic": topic,
            "exploration_results": exploration_results,
            "total_questions_explored": len(exploration_results),
            "timestamp": datetime.now().isoformat()
        }
    
    def _generate_related_questions(self, topic: str) -> List[str]:
        """生成相关问题"""
        
        question_templates = [
            f"{topic}的核心原理是什么？",
            f"{topic}在实际中有哪些应用？",
            f"{topic}的发展趋势如何？",
            f"{topic}面临的主要挑战是什么？",
            f"{topic}与其他领域有什么关联？"
        ]
        
        return question_templates
    
    def collaborative_research(self, main_query: str, perspectives: List[str]) -> Dict[str, Any]:
        """协作式研究 - 从多个视角研究同一问题"""
        
        print(f"🤝 开始协作式研究: {main_query}")
        print(f"👁️ 研究视角: {', '.join(perspectives)}")
        
        perspective_results = []
        
        for perspective in perspectives:
            perspective_query = f"从{perspective}视角分析: {main_query}"
            
            print(f"\n🔍 研究视角: {perspective}")
            
            result = self.research(
                query=perspective_query,
                complexity=3,
                depth_required=3
            )
            
            perspective_results.append({
                "perspective": perspective,
                "query": perspective_query,
                "result": result
            })
        
        # 综合分析
        synthesis = self._synthesize_perspectives(main_query, perspective_results)
        
        return {
            "main_query": main_query,
            "perspectives": perspectives,
            "perspective_results": perspective_results,
            "synthesis": synthesis,
            "timestamp": datetime.now().isoformat()
        }
    
    def _synthesize_perspectives(self, main_query: str, perspective_results: List[Dict]) -> Dict[str, Any]:
        """综合多视角研究结果"""
        
        # 收集所有洞察
        all_insights = []
        for result in perspective_results:
            all_insights.extend(result["result"]["key_insights"])
        
        # 计算平均置信度
        avg_confidence = sum(result["result"]["total_confidence"] for result in perspective_results) / len(perspective_results)
        
        # 找到共同主题
        common_themes = self._identify_common_themes([result["result"]["final_answer"] for result in perspective_results])
        
        # 生成综合结论
        synthesis_conclusion = f"通过多视角分析'{main_query}'，我们发现了{len(set(all_insights))}个独特洞察。共同主题包括：{', '.join(common_themes)}。多视角分析提供了更全面和深入的理解。"
        
        return {
            "synthesis_conclusion": synthesis_conclusion,
            "all_insights": list(set(all_insights)),
            "common_themes": common_themes,
            "average_confidence": avg_confidence,
            "perspectives_count": len(perspective_results)
        }
    
    def _identify_common_themes(self, texts: List[str]) -> List[str]:
        """识别共同主题"""
        
        # 简单的关键词数据
        word_counts = defaultdict(int)
        
        for text in texts:
            words = re.findall(r'\w+', text.lower())
            for word in words:
                if len(word) > 3:  # 只考虑较长的词
                    word_counts[word] += 1
        
        # 找到出现频率高的词汇
        common_words = [word for word, count in word_counts.items() if count >= len(texts) // 2]
        
        return common_words[:5]  # 返回前5个共同主题


def demo_basic_research():
    """演示基础研究功能"""
    print("\n" + "=" * 80)
    print("🔍 DeepResearch Agent - 基础研究演示")
    print("=" * 80)
    
    # 创建研究Agent
    agent = DeepResearchAgent(name="深度研究助手", domain="人工智能")
    
    # 添加一些基础知识
    knowledge_base = [
        ("人工智能是使机器能够执行通常需要人类智能的任务的技术", "人工智能", "concept"),
        ("机器学习是人工智能的一个分支，使系统能够从数据中学习", "人工智能", "concept"),
        ("深度学习使用神经网络模拟人脑的学习过程", "人工智能", "concept")
    ]
    
    print("\n📚 正在添加领域知识...")
    for content, domain, node_type in knowledge_base:
        agent.add_domain_knowledge(content, domain, node_type)
    
    # 演示基础研究
    result = agent.research(
        query="什么是深度学习？",
        complexity=3,
        depth_required=3,
        urgency=2
    )
    
    print(f"\n🎆 研究结果:")
    print(f"  ✅ 置信度: {result['total_confidence']:.2f}")
    print(f"  📋 研究步骤数: {result['research_steps']}")
    print(f"  🔧 使用工具数: {result['tools_used']}")
    print(f"  💡 关键洞察数: {len(result['key_insights'])}")
    print(f"  🏆 质量评分: {result['quality_score']}")
    
    print(f"\n📜 最终答案:")
    print(result['final_answer'])
    
    return agent


def main():
    """主函数 - 运行演示"""
    print("🎆 欢迎使用 DeepResearch Agent 系统")
    print("🔬 这是一个基于 LLM 驱动的深度研究智能体")
    
    print("\n选择演示模式:")
    print("1. 基础研究功能演示")
    print("2. 高级研究功能演示")
    print("3. 推理模式测试")
    print("4. 交互式研穦模式")
    print("0. 退出")
    
    while True:
        try:
            choice = input("\n请选择 (0-4): ").strip()
            
            if choice == '0':
                print("\n👋 感谢使用 DeepResearch Agent!")
                break
            elif choice == '1':
                demo_basic_research()
            elif choice == '2':
                print("🚧 高级功能待实现...")
            elif choice == '3':
                print("🚧 推理模式测试待实现...")
            elif choice == '4':
                print("🚧 交互式模式待实现...")
            else:
                print("❌ 无效选择，请输入 0-4")
                
        except KeyboardInterrupt:
            print("\n\n👋 程序被中断，再见！")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")


if __name__ == "__main__":
    main()