"""
自进化自学习Agent实现
作者：山泽

这个Agent具有以下自进化能力：
1. 经验记忆和学习
2. 策略自动优化
3. 反思和改进机制
4. 知识图谱构建
5. 动态工具学习
"""

import json
import time
import random
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
# import numpy as np  # 如果需要numpy，请先安装: pip install numpy
try:
    import numpy as np
except ImportError:
    print("警告: numpy未安装，使用内置替代方案")
    import random
    import math
    
    class MockLinalg:
        def norm(self, vector):
            return math.sqrt(sum(x*x for x in vector))
    
    class MockRandom:
        def rand(self, size):
            return [random.random() for _ in range(size)]
    
    class MockNumpy:
        def __init__(self):
            self.random = MockRandom()
            self.linalg = MockLinalg()
            
        def mean(self, data):
            return sum(data) / len(data) if data else 0
            
        def dot(self, a, b):
            return sum(x*y for x,y in zip(a,b))
    
    np = MockNumpy()
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Experience:
    """经验记录"""
    task: str
    context: Dict[str, Any]
    action: str
    result: Any
    success: bool
    reward: float
    timestamp: float
    reflection: Optional[str] = None
    
    def to_dict(self):
        return asdict(self)

@dataclass
class Strategy:
    """策略记录"""
    name: str
    description: str
    conditions: Dict[str, Any]
    actions: List[str]
    success_rate: float
    usage_count: int
    last_updated: float
    
    def to_dict(self):
        return asdict(self)

class KnowledgeGraph:
    """知识图谱管理"""
    
    def __init__(self):
        self.nodes = {}  # 概念节点
        self.edges = {}  # 关系边
        self.node_embeddings = {}  # 节点嵌入向量
        
    def add_concept(self, concept: str, properties: Dict[str, Any]):
        """添加概念节点"""
        self.nodes[concept] = {
            'properties': properties,
            'created_at': time.time(),
            'access_count': 0
        }
        # 生成简单的嵌入向量（实际应用中应使用更复杂的方法）
        self.node_embeddings[concept] = np.random.rand(128)
        
    def add_relation(self, from_concept: str, to_concept: str, relation: str, weight: float = 1.0):
        """添加关系边"""
        if from_concept not in self.edges:
            self.edges[from_concept] = {}
        if to_concept not in self.edges[from_concept]:
            self.edges[from_concept][to_concept] = {}
        
        self.edges[from_concept][to_concept][relation] = {
            'weight': weight,
            'created_at': time.time()
        }
        
    def find_related_concepts(self, concept: str, max_distance: int = 2) -> List[str]:
        """找到相关概念"""
        if concept not in self.nodes:
            return []
            
        visited = set()
        queue = deque([(concept, 0)])
        related = []
        
        while queue:
            current, distance = queue.popleft()
            if current in visited or distance > max_distance:
                continue
                
            visited.add(current)
            if distance > 0:
                related.append(current)
                
            # 添加邻接节点
            if current in self.edges:
                for neighbor in self.edges[current]:
                    if neighbor not in visited:
                        queue.append((neighbor, distance + 1))
                        
        return related
        
    def get_concept_similarity(self, concept1: str, concept2: str) -> float:
        """计算概念相似度"""
        if concept1 not in self.node_embeddings or concept2 not in self.node_embeddings:
            return 0.0
            
        vec1 = self.node_embeddings[concept1]
        vec2 = self.node_embeddings[concept2]
        
        # 余弦相似度
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)

class ReflectionModule:
    """反思模块"""
    
    def __init__(self):
        self.reflection_history = []
        
    def reflect_on_experience(self, experience: Experience) -> str:
        """对经验进行反思"""
        reflection_prompts = [
            f"为什么任务 '{experience.task}' 的结果是 {'成功' if experience.success else '失败'}？",
            f"在处理 '{experience.task}' 时，我可以如何改进？",
            f"这次经验教会了我什么关于 '{experience.task}' 的新知识？"
        ]
        
        # 简化版反思（实际应用中会调用LLM）
        if experience.success:
            reflection = f"成功完成任务的关键因素：{experience.action}在上下文{experience.context}中表现良好"
        else:
            reflection = f"失败原因分析：{experience.action}在处理{experience.task}时不够有效，需要调整策略"
            
        self.reflection_history.append({
            'experience_id': id(experience),
            'reflection': reflection,
            'timestamp': time.time()
        })
        
        return reflection
        
    def identify_patterns(self, recent_experiences: List[Experience]) -> List[str]:
        """识别经验模式"""
        patterns = []
        
        # 成功模式识别
        successful_actions = [exp.action for exp in recent_experiences if exp.success]
        if successful_actions:
            action_counts = {}
            for action in successful_actions:
                action_counts[action] = action_counts.get(action, 0) + 1
            
            most_successful = max(action_counts.items(), key=lambda x: x[1])
            patterns.append(f"高成功率动作模式：{most_successful[0]}（成功{most_successful[1]}次）")
            
        # 失败模式识别
        failed_experiences = [exp for exp in recent_experiences if not exp.success]
        if failed_experiences:
            common_failures = {}
            for exp in failed_experiences:
                key = f"{exp.task}:{exp.action}"
                common_failures[key] = common_failures.get(key, 0) + 1
                
            if common_failures:
                most_common_failure = max(common_failures.items(), key=lambda x: x[1])
                patterns.append(f"常见失败模式：{most_common_failure[0]}（失败{most_common_failure[1]}次）")
                
        return patterns

class SelfEvolvingAgent:
    """自进化Agent"""
    
    def __init__(self, name: str = "SelfEvolvingAgent"):
        self.name = name
        self.experiences = []  # 经验库
        self.strategies = {}   # 策略库
        self.knowledge_graph = KnowledgeGraph()
        self.reflection_module = ReflectionModule()
        
        # 学习参数
        self.learning_rate = 0.1
        self.exploration_rate = 0.2
        self.memory_size = 1000
        
        # 性能指标
        self.performance_history = []
        self.total_tasks = 0
        self.successful_tasks = 0
        
        # 工具库
        self.available_tools = {
            'search': self._search_tool,
            'calculate': self._calculate_tool,
            'analyze': self._analyze_tool,
            'plan': self._plan_tool
        }
        
        # 初始化基础策略
        self._initialize_base_strategies()
        
    def _initialize_base_strategies(self):
        """初始化基础策略"""
        base_strategies = [
            Strategy(
                name="探索策略",
                description="在不确定情况下进行探索",
                conditions={"uncertainty": "high"},
                actions=["search", "analyze"],
                success_rate=0.5,
                usage_count=0,
                last_updated=time.time()
            ),
            Strategy(
                name="利用策略", 
                description="使用已知有效的方法",
                conditions={"confidence": "high"},
                actions=["plan", "execute"],
                success_rate=0.8,
                usage_count=0,
                last_updated=time.time()
            )
        ]
        
        for strategy in base_strategies:
            self.strategies[strategy.name] = strategy
            
    def _search_tool(self, query: str) -> Dict[str, Any]:
        """搜索工具"""
        # 模拟搜索结果
        return {
            'results': [f"搜索结果 {i} for {query}" for i in range(3)],
            'confidence': random.uniform(0.5, 1.0)
        }
        
    def _calculate_tool(self, expression: str) -> Dict[str, Any]:
        """计算工具"""
        try:
            # 简单的数学表达式计算
            result = eval(expression)
            return {'result': result, 'success': True}
        except:
            return {'result': None, 'success': False}
            
    def _analyze_tool(self, data: Any) -> Dict[str, Any]:
        """分析工具"""
        # 模拟分析过程
        return {
            'analysis': f"分析结果：{data}包含{len(str(data))}个字符",
            'insights': ["洞察1", "洞察2"],
            'confidence': random.uniform(0.6, 0.9)
        }
        
    def _plan_tool(self, goal: str) -> Dict[str, Any]:
        """规划工具"""
        # 模拟规划过程
        steps = [f"步骤{i+1}: 处理{goal}的第{i+1}部分" for i in range(3)]
        return {
            'plan': steps,
            'estimated_effort': random.randint(1, 10),
            'success_probability': random.uniform(0.7, 0.95)
        }
        
    def perceive_environment(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """环境感知"""
        # 分析当前环境状态
        perception = {
            'current_context': context,
            'relevant_experiences': self._find_relevant_experiences(context),
            'applicable_strategies': self._find_applicable_strategies(context),
            'uncertainty_level': self._assess_uncertainty(context)
        }
        
        return perception
        
    def _find_relevant_experiences(self, context: Dict[str, Any]) -> List[Experience]:
        """找到相关经验"""
        relevant_experiences = []
        
        for exp in self.experiences[-50:]:  # 检查最近的50个经验
            similarity = self._calculate_context_similarity(exp.context, context)
            if similarity > 0.5:  # 相似度阈值
                relevant_experiences.append(exp)
                
        return sorted(relevant_experiences, key=lambda x: x.reward, reverse=True)[:10]
        
    def _calculate_context_similarity(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """计算上下文相似度"""
        common_keys = set(context1.keys()) & set(context2.keys())
        if not common_keys:
            return 0.0
            
        similarity_scores = []
        for key in common_keys:
            if isinstance(context1[key], str) and isinstance(context2[key], str):
                # 简单的字符串相似度
                similarity = len(set(context1[key].split()) & set(context2[key].split())) / \
                           len(set(context1[key].split()) | set(context2[key].split()))
                similarity_scores.append(similarity)
            elif context1[key] == context2[key]:
                similarity_scores.append(1.0)
            else:
                similarity_scores.append(0.0)
                
        return np.mean(similarity_scores) if similarity_scores else 0.0
        
    def _find_applicable_strategies(self, context: Dict[str, Any]) -> List[Strategy]:
        """找到适用的策略"""
        applicable_strategies = []
        
        for strategy in self.strategies.values():
            if self._strategy_matches_context(strategy, context):
                applicable_strategies.append(strategy)
                
        return sorted(applicable_strategies, key=lambda x: x.success_rate, reverse=True)
        
    def _strategy_matches_context(self, strategy: Strategy, context: Dict[str, Any]) -> bool:
        """检查策略是否匹配当前上下文"""
        for condition_key, condition_value in strategy.conditions.items():
            if condition_key in context:
                if context[condition_key] != condition_value:
                    return False
            else:
                # 如果上下文中没有这个条件，根据策略类型进行推断
                if condition_key == "uncertainty":
                    uncertainty = self._assess_uncertainty(context)
                    if (condition_value == "high" and uncertainty < 0.5) or \
                       (condition_value == "low" and uncertainty > 0.5):
                        return False
                        
        return True
        
    def _assess_uncertainty(self, context: Dict[str, Any]) -> float:
        """评估不确定性"""
        # 基于上下文信息评估不确定性
        uncertainty_factors = []
        
        # 任务复杂度
        task_complexity = len(str(context).split()) / 100.0
        uncertainty_factors.append(min(task_complexity, 1.0))
        
        # 相关经验数量
        relevant_exp_count = len(self._find_relevant_experiences(context))
        experience_factor = max(0.0, 1.0 - relevant_exp_count / 10.0)
        uncertainty_factors.append(experience_factor)
        
        return np.mean(uncertainty_factors)
        
    def decide_action(self, perception: Dict[str, Any]) -> str:
        """决策行动"""
        context = perception['current_context']
        applicable_strategies = perception['applicable_strategies']
        relevant_experiences = perception['relevant_experiences']
        
        # 策略选择
        if applicable_strategies and random.random() > self.exploration_rate:
            # 利用：选择最佳策略
            best_strategy = applicable_strategies[0]
            action = random.choice(best_strategy.actions)
        else:
            # 探索：尝试新的行动
            if relevant_experiences:
                # 基于相关经验选择
                successful_actions = [exp.action for exp in relevant_experiences if exp.success]
                if successful_actions:
                    action = random.choice(successful_actions)
                else:
                    action = random.choice(list(self.available_tools.keys()))
            else:
                # 随机选择
                action = random.choice(list(self.available_tools.keys()))
                
        return action
        
    def execute_action(self, action: str, context: Dict[str, Any]) -> Any:
        """执行动作"""
        if action in self.available_tools:
            tool = self.available_tools[action]
            
            # 根据上下文准备工具参数
            if action == 'search':
                query = context.get('query', '默认查询')
                return tool(query)
            elif action == 'calculate':
                expression = context.get('expression', '1+1')
                return tool(expression)
            elif action == 'analyze':
                data = context.get('data', context)
                return tool(data)
            elif action == 'plan':
                goal = context.get('goal', '默认目标')
                return tool(goal)
            else:
                return tool(str(context))
        else:
            return {'error': f'未知动作: {action}'}
            
    def evaluate_result(self, result: Any, expected_outcome: Any = None) -> Tuple[bool, float]:
        """评估结果"""
        # 简单的结果评估逻辑
        if isinstance(result, dict):
            if 'error' in result:
                return False, -1.0
            elif 'success' in result:
                success = result['success']
                reward = 1.0 if success else -0.5
                return success, reward
            elif 'confidence' in result:
                confidence = result['confidence']
                success = confidence > 0.7
                reward = confidence if success else -0.3
                return success, reward
            else:
                # 默认认为有结果就是成功
                return True, 0.5
        else:
            return True, 0.3
            
    def learn_from_experience(self, experience: Experience):
        """从经验中学习"""
        # 添加经验到记忆库
        self.experiences.append(experience)
        
        # 限制记忆库大小
        if len(self.experiences) > self.memory_size:
            self.experiences.pop(0)
            
        # 更新知识图谱
        self._update_knowledge_graph(experience)
        
        # 反思经验
        reflection = self.reflection_module.reflect_on_experience(experience)
        experience.reflection = reflection
        
        # 更新或创建策略
        self._update_strategies(experience)
        
        # 更新性能指标
        self.total_tasks += 1
        if experience.success:
            self.successful_tasks += 1
            
        current_success_rate = self.successful_tasks / self.total_tasks
        self.performance_history.append({
            'timestamp': time.time(),
            'success_rate': current_success_rate,
            'total_tasks': self.total_tasks
        })
        
        # 自适应学习率调整
        self._adjust_learning_parameters()
        
    def _update_knowledge_graph(self, experience: Experience):
        """更新知识图谱"""
        # 提取关键概念
        task_concept = f"task:{experience.task}"
        action_concept = f"action:{experience.action}"
        
        # 添加或更新概念节点
        self.knowledge_graph.add_concept(task_concept, {
            'type': 'task',
            'description': experience.task,
            'success_count': 0,
            'failure_count': 0
        })
        
        self.knowledge_graph.add_concept(action_concept, {
            'type': 'action', 
            'description': experience.action
        })
        
        # 添加关系
        relation_type = 'succeeds_with' if experience.success else 'fails_with'
        weight = experience.reward
        
        self.knowledge_graph.add_relation(
            task_concept, action_concept, relation_type, weight
        )
        
    def _update_strategies(self, experience: Experience):
        """更新策略"""
        task_type = experience.task.split(':')[0] if ':' in experience.task else experience.task
        strategy_name = f"策略_{task_type}_{experience.action}"
        
        if strategy_name in self.strategies:
            # 更新现有策略
            strategy = self.strategies[strategy_name]
            strategy.usage_count += 1
            
            # 更新成功率（使用指数移动平均）
            alpha = self.learning_rate
            if experience.success:
                strategy.success_rate = (1 - alpha) * strategy.success_rate + alpha * 1.0
            else:
                strategy.success_rate = (1 - alpha) * strategy.success_rate + alpha * 0.0
                
            strategy.last_updated = time.time()
        else:
            # 创建新策略
            initial_success_rate = 1.0 if experience.success else 0.0
            new_strategy = Strategy(
                name=strategy_name,
                description=f"针对{task_type}任务的{experience.action}策略",
                conditions={'task_type': task_type},
                actions=[experience.action],
                success_rate=initial_success_rate,
                usage_count=1,
                last_updated=time.time()
            )
            self.strategies[strategy_name] = new_strategy
            
    def _adjust_learning_parameters(self):
        """自适应调整学习参数"""
        if len(self.performance_history) >= 10:
            recent_performance = self.performance_history[-10:]
            avg_success_rate = np.mean([p['success_rate'] for p in recent_performance])
            
            # 如果性能下降，增加探索率
            if avg_success_rate < 0.6:
                self.exploration_rate = min(0.5, self.exploration_rate + 0.05)
            elif avg_success_rate > 0.8:
                self.exploration_rate = max(0.1, self.exploration_rate - 0.02)
                
    def self_evolve(self):
        """自我进化过程"""
        logger.info("开始自我进化过程...")
        
        # 1. 分析最近的经验模式
        recent_experiences = self.experiences[-100:] if len(self.experiences) >= 100 else self.experiences
        patterns = self.reflection_module.identify_patterns(recent_experiences)
        
        logger.info(f"识别到的模式: {patterns}")
        
        # 2. 策略优化
        self._optimize_strategies()
        
        # 3. 知识整合
        self._integrate_knowledge()
        
        # 4. 能力扩展
        self._expand_capabilities()
        
        logger.info("自我进化完成")
        
    def _optimize_strategies(self):
        """优化策略"""
        # 移除低效策略
        strategies_to_remove = []
        for name, strategy in self.strategies.items():
            if strategy.usage_count > 10 and strategy.success_rate < 0.3:
                strategies_to_remove.append(name)
                
        for name in strategies_to_remove:
            del self.strategies[name]
            logger.info(f"移除低效策略: {name}")
            
        # 合并相似策略
        self._merge_similar_strategies()
        
    def _merge_similar_strategies(self):
        """合并相似策略"""
        strategy_list = list(self.strategies.values())
        
        for i, strategy1 in enumerate(strategy_list):
            for j, strategy2 in enumerate(strategy_list[i+1:], i+1):
                if self._strategies_similar(strategy1, strategy2):
                    # 合并策略
                    merged_name = f"合并_{strategy1.name}_{strategy2.name}"
                    merged_strategy = Strategy(
                        name=merged_name,
                        description=f"合并策略: {strategy1.description} + {strategy2.description}",
                        conditions={**strategy1.conditions, **strategy2.conditions},
                        actions=list(set(strategy1.actions + strategy2.actions)),
                        success_rate=(strategy1.success_rate * strategy1.usage_count + 
                                    strategy2.success_rate * strategy2.usage_count) / 
                                   (strategy1.usage_count + strategy2.usage_count),
                        usage_count=strategy1.usage_count + strategy2.usage_count,
                        last_updated=time.time()
                    )
                    
                    # 更新策略库
                    del self.strategies[strategy1.name]
                    del self.strategies[strategy2.name]
                    self.strategies[merged_name] = merged_strategy
                    
                    logger.info(f"合并策略: {strategy1.name} + {strategy2.name} -> {merged_name}")
                    break
                    
    def _strategies_similar(self, strategy1: Strategy, strategy2: Strategy) -> bool:
        """判断策略是否相似"""
        # 检查条件相似性
        common_conditions = set(strategy1.conditions.keys()) & set(strategy2.conditions.keys())
        if len(common_conditions) / max(len(strategy1.conditions), len(strategy2.conditions)) > 0.5:
            # 检查动作相似性
            common_actions = set(strategy1.actions) & set(strategy2.actions)
            if len(common_actions) / max(len(strategy1.actions), len(strategy2.actions)) > 0.5:
                return True
        return False
        
    def _integrate_knowledge(self):
        """整合知识"""
        # 发现知识图谱中的新关联
        concepts = list(self.knowledge_graph.nodes.keys())
        
        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i+1:]:
                similarity = self.knowledge_graph.get_concept_similarity(concept1, concept2)
                if similarity > 0.8:  # 高相似度阈值
                    # 添加相似性关系
                    self.knowledge_graph.add_relation(concept1, concept2, 'similar_to', similarity)
                    
    def _expand_capabilities(self):
        """扩展能力"""
        # 基于成功经验尝试发现新的工具组合
        successful_experiences = [exp for exp in self.experiences[-50:] if exp.success]
        
        # 分析成功的动作序列
        action_sequences = []
        for i in range(len(successful_experiences) - 1):
            if successful_experiences[i].timestamp < successful_experiences[i+1].timestamp:
                sequence = (successful_experiences[i].action, successful_experiences[i+1].action)
                action_sequences.append(sequence)
        
        # 发现常见的成功序列
        sequence_counts = {}
        for seq in action_sequences:
            sequence_counts[seq] = sequence_counts.get(seq, 0) + 1
            
        # 创建新的组合工具
        for sequence, count in sequence_counts.items():
            if count >= 3:  # 出现3次以上的序列
                combo_name = f"combo_{sequence[0]}_{sequence[1]}"
                if combo_name not in self.available_tools:
                    self.available_tools[combo_name] = self._create_combo_tool(sequence)
                    logger.info(f"发现新的工具组合: {combo_name}")
                    
    def _create_combo_tool(self, sequence: Tuple[str, str]) -> Callable:
        """创建组合工具"""
        def combo_tool(context):
            # 依次执行序列中的工具
            result1 = self.available_tools[sequence[0]](context)
            # 将第一个工具的结果作为第二个工具的输入
            enhanced_context = {**context, 'previous_result': result1}
            result2 = self.available_tools[sequence[1]](enhanced_context)
            
            return {
                'sequence_results': [result1, result2],
                'final_result': result2,
                'combo_success': True
            }
        return combo_tool
        
    def process_task(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """处理任务的主要接口"""
        if context is None:
            context = {'task': task}
        else:
            context['task'] = task
            
        logger.info(f"处理任务: {task}")
        
        # 1. 环境感知
        perception = self.perceive_environment(context)
        
        # 2. 决策行动
        action = self.decide_action(perception)
        
        # 3. 执行动作
        result = self.execute_action(action, context)
        
        # 4. 评估结果
        success, reward = self.evaluate_result(result)
        
        # 5. 创建经验
        experience = Experience(
            task=task,
            context=context,
            action=action,
            result=result,
            success=success,
            reward=reward,
            timestamp=time.time()
        )
        
        # 6. 学习
        self.learn_from_experience(experience)
        
        # 7. 定期自我进化
        if len(self.experiences) % 50 == 0:  # 每50个经验进化一次
            self.self_evolve()
            
        return {
            'task': task,
            'action': action,
            'result': result,
            'success': success,
            'reward': reward,
            'learning_insights': experience.reflection
        }
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.performance_history:
            return {'message': '暂无性能数据'}
            
        latest_performance = self.performance_history[-1]
        
        # 计算趋势
        if len(self.performance_history) >= 10:
            recent_avg = np.mean([p['success_rate'] for p in self.performance_history[-10:]])
            overall_avg = np.mean([p['success_rate'] for p in self.performance_history])
            trend = 'improving' if recent_avg > overall_avg else 'declining'
        else:
            trend = 'insufficient_data'
            
        return {
            'total_tasks': self.total_tasks,
            'successful_tasks': self.successful_tasks,
            'current_success_rate': latest_performance['success_rate'],
            'trend': trend,
            'strategies_count': len(self.strategies),
            'experiences_count': len(self.experiences),
            'exploration_rate': self.exploration_rate,
            'knowledge_concepts': len(self.knowledge_graph.nodes)
        }
        
    def save_state(self, filepath: str):
        """保存Agent状态"""
        state = {
            'name': self.name,
            'experiences': [exp.to_dict() for exp in self.experiences],
            'strategies': {name: strategy.to_dict() for name, strategy in self.strategies.items()},
            'performance_history': self.performance_history,
            'total_tasks': self.total_tasks,
            'successful_tasks': self.successful_tasks,
            'learning_rate': self.learning_rate,
            'exploration_rate': self.exploration_rate,
            'knowledge_graph_nodes': self.knowledge_graph.nodes,
            'knowledge_graph_edges': self.knowledge_graph.edges,
            'timestamp': time.time()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Agent状态已保存到: {filepath}")
        
    def load_state(self, filepath: str):
        """加载Agent状态"""
        with open(filepath, 'r', encoding='utf-8') as f:
            state = json.load(f)
            
        self.name = state['name']
        self.total_tasks = state['total_tasks']
        self.successful_tasks = state['successful_tasks']
        self.learning_rate = state['learning_rate']
        self.exploration_rate = state['exploration_rate']
        self.performance_history = state['performance_history']
        
        # 重建经验
        self.experiences = []
        for exp_dict in state['experiences']:
            exp = Experience(**exp_dict)
            self.experiences.append(exp)
            
        # 重建策略
        self.strategies = {}
        for name, strategy_dict in state['strategies'].items():
            strategy = Strategy(**strategy_dict)
            self.strategies[name] = strategy
            
        # 重建知识图谱
        self.knowledge_graph.nodes = state['knowledge_graph_nodes']
        self.knowledge_graph.edges = state['knowledge_graph_edges']
        
        logger.info(f"Agent状态已从 {filepath} 加载")


def demo_self_evolving_agent():
    """演示自进化Agent"""
    print("=== 自进化自学习Agent演示 ===")
    
    # 创建Agent
    agent = SelfEvolvingAgent("学习型AI助手")
    
    # 模拟任务序列
    tasks = [
        ("搜索Python教程", {'query': 'Python基础教程', 'difficulty': 'beginner'}),
        ("计算复合利率", {'expression': '1000 * (1.05 ** 10)', 'context': 'finance'}),
        ("分析用户数据", {'data': {'users': 100, 'active': 80, 'retention': 0.75}}),
        ("制定学习计划", {'goal': '掌握机器学习', 'timeframe': '3个月'}),
        ("搜索机器学习资源", {'query': '机器学习入门', 'difficulty': 'intermediate'}),
        ("分析学习进度", {'data': {'completed': 5, 'total': 20, 'avg_score': 85}}),
        ("计算学习效率", {'expression': '85 / 100 * 0.8', 'context': 'learning'}),
        ("优化学习策略", {'goal': '提高学习效率', 'current_rate': 0.68})
    ]
    
    print(f"\n准备处理 {len(tasks)} 个任务...\n")
    
    # 处理任务并观察学习过程
    for i, (task, context) in enumerate(tasks, 1):
        print(f"--- 任务 {i}: {task} ---")
        
        result = agent.process_task(task, context)
        
        print(f"选择的行动: {result['action']}")
        print(f"执行结果: {result['success']}")
        print(f"奖励值: {result['reward']:.2f}")
        print(f"学习洞察: {result['learning_insights']}")
        
        # 显示当前性能
        if i % 3 == 0:  # 每3个任务显示一次性能摘要
            performance = agent.get_performance_summary()
            print(f"\n📊 当前性能摘要:")
            print(f"  总任务数: {performance['total_tasks']}")
            print(f"  成功率: {performance['current_success_rate']:.2%}")
            print(f"  策略数量: {performance['strategies_count']}")
            print(f"  知识概念: {performance['knowledge_concepts']}")
            print(f"  探索率: {performance['exploration_rate']:.2f}")
            
        print()
        
    # 最终性能报告
    print("\n=== 最终学习报告 ===")
    final_performance = agent.get_performance_summary()
    
    print(f"总处理任务: {final_performance['total_tasks']}")
    print(f"最终成功率: {final_performance['current_success_rate']:.2%}")
    print(f"性能趋势: {final_performance['trend']}")
    print(f"学习策略数: {final_performance['strategies_count']}")
    print(f"知识概念数: {final_performance['knowledge_concepts']}")
    
    # 显示学到的策略
    print("\n🧠 学到的策略:")
    for name, strategy in agent.strategies.items():
        if strategy.usage_count > 0:
            print(f"  {name}: 成功率 {strategy.success_rate:.2%}, 使用 {strategy.usage_count} 次")
            
    # 保存Agent状态
    agent.save_state("agent_state.json")
    print("\n💾 Agent状态已保存到 agent_state.json")
    
    return agent


if __name__ == "__main__":
    # 运行演示
    agent = demo_self_evolving_agent()