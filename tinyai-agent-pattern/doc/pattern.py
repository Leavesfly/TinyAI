# -*- coding: utf-8 -*-
"""
常用Agent模式实现
包含ReAct、Reflect、Planning、Multi-Agent等模式
"""

import json
import time
import random
import hashlib
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
import re


class AgentState(Enum):
    """Agent状态枚举"""
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    REFLECTING = "reflecting"
    PLANNING = "planning"
    DONE = "done"


@dataclass
class Step:
    """步骤记录"""
    step_type: str  # "thought", "action", "observation", "reflection", "plan"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Action:
    """动作结构"""
    name: str
    arguments: Dict[str, Any]
    result: Optional[Any] = None
    error: Optional[str] = None


class BaseAgent(ABC):
    """Agent基类"""

    def __init__(self, name: str, max_steps: int = 10):
        self.name = name
        self.max_steps = max_steps
        self.steps = []
        self.state = AgentState.THINKING
        self.tools = {}
        self.memory = []

    def add_tool(self, name: str, func: Callable, description: str):
        """添加工具"""
        self.tools[name] = {
            "function": func,
            "description": description
        }

    def call_tool(self, action: Action) -> Any:
        """调用工具"""
        if action.name not in self.tools:
            action.error = f"工具 {action.name} 不存在"
            return None

        try:
            result = self.tools[action.name]["function"](**action.arguments)
            action.result = result
            return result
        except Exception as e:
            action.error = str(e)
            return None

    def add_step(self, step_type: str, content: str, metadata: Optional[Dict] = None):
        """添加步骤记录"""
        step = Step(step_type, content, metadata=metadata or {})
        self.steps.append(step)
        return step

    @abstractmethod
    def process(self, query: str) -> str:
        """处理查询"""
        pass

    def get_steps_summary(self) -> str:
        """获取步骤摘要"""
        summary = []
        for i, step in enumerate(self.steps, 1):
            summary.append(f"{i}. {step.step_type.upper()}: {step.content}")
        return "\n".join(summary)


class ReActAgent(BaseAgent):
    """ReAct模式Agent: Reasoning + Acting

    交替进行推理(Reasoning)和行动(Acting)，通过观察结果来指导下一步行动
    格式: Thought -> Action -> Observation -> Thought -> Action -> ...
    """

    def __init__(self, name: str = "ReAct Agent", max_steps: int = 10):
        super().__init__(name, max_steps)
        self._register_default_tools()

    def _register_default_tools(self):
        """注册默认工具"""
        self.add_tool("calculator", self._calculator, "数学计算工具")
        self.add_tool("search", self._search, "搜索工具")
        self.add_tool("memory", self._memory_lookup, "记忆查找工具")

    def _calculator(self, expression: str) -> str:
        """计算器工具"""
        try:
            # 简单的数学表达式计算
            allowed_chars = set('0123456789+-*/()., ')
            if not all(c in allowed_chars for c in expression):
                return "错误：包含非法字符"

            result = eval(expression)
            return f"计算结果: {result}"
        except Exception as e:
            return f"计算错误: {str(e)}"

    def _search(self, query: str) -> str:
        """模拟搜索工具"""
        # 模拟搜索结果
        search_results = {
            "天气": "今天天气晴朗，温度25度",
            "新闻": "今日科技新闻：AI技术取得新突破",
            "python": "Python是一种高级编程语言，简单易学",
            "机器学习": "机器学习是人工智能的核心技术",
        }

        for key in search_results:
            if key in query.lower():
                return f"搜索结果: {search_results[key]}"

        return "搜索结果: 未找到相关信息"

    def _memory_lookup(self, keyword: str) -> str:
        """记忆查找工具"""
        relevant_memories = [mem for mem in self.memory if keyword.lower() in mem.lower()]
        if relevant_memories:
            return f"相关记忆: {'; '.join(relevant_memories)}"
        return "未找到相关记忆"

    def _think(self, query: str, context: str = "") -> str:
        """思考步骤"""
        prompt = f"问题: {query}\n"
        if context:
            prompt += f"上下文: {context}\n"

        # 模拟思考过程
        thoughts = [
            f"我需要分析这个问题：{query}",
            "让我思考一下需要什么信息来回答这个问题",
            "我应该使用什么工具来获取所需信息？",
            "基于当前信息，我的下一步行动是什么？"
        ]

        # 根据问题类型选择思考
        if any(op in query for op in ['+', '-', '*', '/', '计算', '算']):
            return "这是一个数学问题，我需要使用计算器工具"
        elif any(word in query for word in ['搜索', '查找', '什么是', '天气', '新闻']):
            return "这需要搜索信息，我应该使用搜索工具"
        elif "记忆" in query or "之前" in query:
            return "这需要查找记忆，我应该使用记忆查找工具"
        else:
            return random.choice(thoughts)

    def _parse_action(self, thought: str) -> Optional[Action]:
        """从思考中解析出行动"""
        if "计算器" in thought:
            # 尝试从原始查询中提取数学表达式
            if hasattr(self, '_current_query'):
                query = self._current_query
                # 提取数字和运算符
                math_pattern = r'[\d+\-*/().\s]+'
                match = re.search(math_pattern, query)
                if match:
                    expression = match.group().strip()
                    return Action("calculator", {"expression": expression})

        elif "搜索" in thought:
            if hasattr(self, '_current_query'):
                return Action("search", {"query": self._current_query})

        elif "记忆" in thought:
            if hasattr(self, '_current_query'):
                # 提取关键词
                keywords = self._current_query.split()
                if keywords:
                    return Action("memory", {"keyword": keywords[0]})

        return None

    def process(self, query: str) -> str:
        """ReAct主处理流程"""
        self._current_query = query
        self.steps = []

        # 添加到记忆
        self.memory.append(f"用户询问: {query}")

        context = ""

        for step_num in range(self.max_steps):
            # Step 1: Think (思考)
            thought = self._think(query, context)
            self.add_step("thought", thought)

            # Step 2: Act (行动)
            action = self._parse_action(thought)
            if action:
                self.add_step("action", f"{action.name}({action.arguments})")

                # 执行动作
                result = self.call_tool(action)

                # Step 3: Observe (观察)
                if action.result:
                    observation = str(action.result)
                    self.add_step("observation", observation)
                    context += f"\n{observation}"

                    # 判断是否完成
                    if "计算结果" in observation or "搜索结果" in observation:
                        # 生成最终答案
                        final_thought = f"基于观察结果，我可以回答用户的问题了"
                        self.add_step("thought", final_thought)

                        answer = f"根据我的分析和工具使用，{observation}"
                        self.add_step("answer", answer)
                        return answer

                elif action.error:
                    error_obs = f"工具执行失败: {action.error}"
                    self.add_step("observation", error_obs)
                    context += f"\n{error_obs}"
            else:
                # 如果没有解析出行动，直接给出答案
                answer = f"基于我的思考：{thought}"
                self.add_step("answer", answer)
                return answer

        return "很抱歉，我无法在限定步骤内完成这个任务"


class ReflectAgent(BaseAgent):
    """Reflect模式Agent: 具有自我反思能力

    在执行任务后进行反思，评估执行效果并改进策略
    """

    def __init__(self, name: str = "Reflect Agent", max_steps: int = 10):
        super().__init__(name, max_steps)
        self.reflections = []
        self._register_default_tools()

    def _register_default_tools(self):
        """注册工具"""
        self.add_tool("analyze", self._analyze, "分析工具")
        self.add_tool("evaluate", self._evaluate, "评估工具")

    def _analyze(self, text: str) -> str:
        """分析工具"""
        analysis = [
            f"文本长度: {len(text)} 字符",
            f"词汇数量: {len(text.split())} 个",
            f"包含问号: {'是' if '?' in text or '？' in text else '否'}",
            f"情感倾向: {'积极' if any(word in text for word in ['好', '棒', '优秀']) else '中性'}"
        ]
        return "分析结果: " + "; ".join(analysis)

    def _evaluate(self, criteria: str) -> str:
        """评估工具"""
        scores = {
            "准确性": random.randint(7, 10),
            "完整性": random.randint(6, 9),
            "清晰度": random.randint(8, 10)
        }
        return f"评估结果({criteria}): " + "; ".join([f"{k}:{v}/10" for k, v in scores.items()])

    def _initial_attempt(self, query: str) -> str:
        """初始尝试"""
        self.add_step("initial_attempt", f"开始处理查询: {query}")

        # 简单的初始响应
        if "分析" in query:
            action = Action("analyze", {"text": query})
            result = self.call_tool(action)
            self.add_step("action", f"使用分析工具: {action.arguments}")
            self.add_step("observation", str(result))
            return str(result)
        else:
            response = f"对于问题'{query}'，我的初始回答是：这是一个需要仔细思考的问题。"
            self.add_step("initial_response", response)
            return response

    def _reflect(self, initial_response: str, query: str) -> str:
        """反思过程"""
        reflection_prompt = f"""
        原始问题: {query}
        初始回答: {initial_response}
        
        反思要点:
        1. 我的回答是否完整地解决了问题？
        2. 是否有遗漏的重要信息？
        3. 回答的质量如何？
        4. 如何改进？
        """

        # 模拟反思过程
        reflections = []

        # 完整性反思
        if len(initial_response) < 50:
            reflections.append("回答过于简短，可能不够完整")
        else:
            reflections.append("回答长度适中")

        # 相关性反思
        query_keywords = set(query.lower().split())
        response_keywords = set(initial_response.lower().split())
        overlap = len(query_keywords & response_keywords)

        if overlap < 2:
            reflections.append("回答与问题的相关性可能不足")
        else:
            reflections.append("回答与问题相关性良好")

        # 工具使用反思
        if not any(step.step_type == "action" for step in self.steps):
            reflections.append("可能需要使用工具来提供更准确的信息")
        else:
            reflections.append("适当使用了工具")

        reflection = "反思结果: " + "; ".join(reflections)
        self.add_step("reflection", reflection)
        self.reflections.append(reflection)

        return reflection

    def _improve(self, initial_response: str, reflection: str, query: str) -> str:
        """基于反思改进回答"""
        improvements = []

        if "简短" in reflection:
            improvements.append("提供更详细的解释")

        if "相关性不足" in reflection:
            improvements.append("更直接地回答问题")

        if "需要使用工具" in reflection:
            # 使用评估工具
            action = Action("evaluate", {"criteria": "回答质量"})
            result = self.call_tool(action)
            self.add_step("action", f"使用评估工具: {action.arguments}")
            self.add_step("observation", str(result))
            improvements.append(f"工具评估: {result}")

        if not improvements:
            improvements.append("回答已经比较完善")

        improved_response = f"{initial_response}\n\n改进补充: {'; '.join(improvements)}"
        self.add_step("improvement", improved_response)

        return improved_response

    def process(self, query: str) -> str:
        """Reflect主处理流程"""
        self.steps = []

        # 第一步：初始尝试
        initial_response = self._initial_attempt(query)

        # 第二步：反思
        reflection = self._reflect(initial_response, query)

        # 第三步：改进
        improved_response = self._improve(initial_response, reflection, query)

        # 第四步：最终反思（可选）
        final_reflection = f"最终反思：通过反思和改进，我提供了更好的回答"
        self.add_step("final_reflection", final_reflection)

        return improved_response


class PlanningAgent(BaseAgent):
    """Planning模式Agent: 先制定计划再执行

    将复杂任务分解为子任务，制定执行计划，然后按计划执行
    """

    def __init__(self, name: str = "Planning Agent", max_steps: int = 15):
        super().__init__(name, max_steps)
        self.plan = []
        self.current_task_index = 0
        self._register_default_tools()

    def _register_default_tools(self):
        """注册工具"""
        self.add_tool("research", self._research, "研究工具")
        self.add_tool("analyze", self._analyze, "分析工具")
        self.add_tool("synthesize", self._synthesize, "综合工具")
        self.add_tool("validate", self._validate, "验证工具")

    def _research(self, topic: str) -> str:
        """研究工具"""
        research_db = {
            "python": "Python是一种解释型、面向对象的编程语言，具有简洁的语法",
            "机器学习": "机器学习是让计算机从数据中学习模式的技术",
            "深度学习": "深度学习使用神经网络来模拟人脑的学习过程",
            "ai": "人工智能是使机器能够模拟人类智能的技术"
        }

        for key in research_db:
            if key in topic.lower():
                return f"研究结果: {research_db[key]}"

        return f"研究结果: 关于'{topic}'的基础信息已收集"

    def _analyze(self, data: str) -> str:
        """分析工具"""
        return f"分析结果: 对'{data[:50]}...'进行了深入分析，发现了关键模式和趋势"

    def _synthesize(self, components: str) -> str:
        """综合工具"""
        return f"综合结果: 将多个组件整合形成完整的解决方案"

    def _validate(self, solution: str) -> str:
        """验证工具"""
        return f"验证结果: 解决方案经过验证，质量评分: {random.randint(8, 10)}/10"

    def _create_plan(self, query: str) -> List[Dict[str, Any]]:
        """制定计划"""
        self.add_step("planning", f"开始为查询制定计划: {query}")

        # 根据查询类型制定不同的计划
        if "分析" in query or "研究" in query:
            plan = [
                {"task": "research", "description": "收集相关信息", "args": {"topic": query}},
                {"task": "analyze", "description": "分析收集的信息", "args": {"data": "收集的信息"}},
                {"task": "synthesize", "description": "综合分析结果", "args": {"components": "分析结果"}},
                {"task": "validate", "description": "验证最终结果", "args": {"solution": "最终结果"}}
            ]
        elif "学习" in query or "教" in query:
            plan = [
                {"task": "research", "description": "研究学习主题", "args": {"topic": query}},
                {"task": "analyze", "description": "分析学习要点", "args": {"data": "学习材料"}},
                {"task": "synthesize", "description": "整理学习大纲", "args": {"components": "学习要点"}}
            ]
        else:
            # 通用计划
            plan = [
                {"task": "research", "description": "收集基础信息", "args": {"topic": query}},
                {"task": "analyze", "description": "分析问题", "args": {"data": query}},
                {"task": "synthesize", "description": "形成解决方案", "args": {"components": "问题分析"}}
            ]

        plan_description = "\n".join([f"{i+1}. {task['description']}" for i, task in enumerate(plan)])
        self.add_step("plan", f"制定的执行计划:\n{plan_description}")

        return plan

    def _execute_plan(self, plan: List[Dict[str, Any]]) -> str:
        """执行计划"""
        results = []

        for i, task in enumerate(plan):
            self.add_step("executing", f"执行步骤 {i+1}: {task['description']}")

            # 执行任务
            action = Action(task["task"], task["args"])
            result = self.call_tool(action)

            self.add_step("action", f"{action.name}({action.arguments})")
            self.add_step("observation", str(result))

            results.append(result)

            # 更新下一个任务的参数（简单实现）
            if i + 1 < len(plan):
                next_task = plan[i + 1]
                if "data" in next_task["args"]:
                    next_task["args"]["data"] = str(result)
                elif "components" in next_task["args"]:
                    next_task["args"]["components"] = str(result)
                elif "solution" in next_task["args"]:
                    next_task["args"]["solution"] = str(result)

        return "\n".join(str(r) for r in results)

    def process(self, query: str) -> str:
        """Planning主处理流程"""
        self.steps = []

        # 第一阶段：制定计划
        plan = self._create_plan(query)
        self.plan = plan

        # 第二阶段：执行计划
        execution_results = self._execute_plan(plan)

        # 第三阶段：总结结果
        summary = f"计划执行完成。基于{len(plan)}个步骤的执行，我为您的问题'{query}'提供了全面的解决方案。"
        self.add_step("summary", summary)

        final_answer = f"{summary}\n\n执行结果:\n{execution_results}"
        return final_answer


class CollaborativeAgent(BaseAgent):
    """协作式Agent：多个Agent协同工作"""

    def __init__(self, name: str = "Collaborative Agent"):
        super().__init__(name)
        self.specialists = {}
        self.coordination_history = []

    def add_specialist(self, name: str, agent: BaseAgent):
        """添加专家Agent"""
        self.specialists[name] = agent

    def _route_query(self, query: str) -> str:
        """路由查询到合适的专家"""
        if any(word in query.lower() for word in ['计算', '数学', '算']):
            return "calculator_expert"
        elif any(word in query.lower() for word in ['分析', '研究', '深入']):
            return "analysis_expert"
        elif any(word in query.lower() for word in ['计划', '规划', '步骤']):
            return "planning_expert"
        else:
            return "general_expert"

    def process(self, query: str) -> str:
        """协作处理流程"""
        self.steps = []

        # 第一步：分析查询并路由
        expert_type = self._route_query(query)
        self.add_step("routing", f"将查询路由到: {expert_type}")

        # 第二步：调用专家
        if expert_type in self.specialists:
            expert = self.specialists[expert_type]
            expert_result = expert.process(query)
            self.add_step("expert_consultation", f"{expert_type}处理结果: {expert_result}")

            # 第三步：验证和整合（可选择另一个专家验证）
            if len(self.specialists) > 1:
                # 选择另一个专家进行验证
                validators = [name for name in self.specialists.keys() if name != expert_type]
                if validators:
                    validator_name = validators[0]
                    validator = self.specialists[validator_name]
                    validation_query = f"请验证这个回答的质量: {expert_result}"
                    validation_result = validator.process(validation_query)
                    self.add_step("validation", f"{validator_name}验证结果: {validation_result}")

            return expert_result
        else:
            return f"抱歉，我没有找到处理'{query}'的专家"


def create_sample_tools():
    """创建示例工具"""
    tools = {}

    def weather_tool(city: str) -> str:
        """天气查询工具"""
        weather_data = {
            "北京": "晴天，25°C",
            "上海": "多云，22°C",
            "广州": "雨天，28°C",
            "深圳": "晴天，30°C"
        }
        return weather_data.get(city, f"{city}天气信息暂不可用")

    def news_tool(category: str) -> str:
        """新闻查询工具"""
        news_data = {
            "科技": "AI技术取得新突破，GPT-5即将发布",
            "财经": "股市今日上涨2%，科技股表现优异",
            "体育": "世界杯预选赛结果公布",
            "娱乐": "新电影获得票房冠军"
        }
        return news_data.get(category, f"{category}新闻暂无更新")

    tools["weather"] = weather_tool
    tools["news"] = news_tool

    return tools


def demo_react_agent():
    """演示ReAct Agent"""
    print("\n" + "=" * 60)
    print("🤖 ReAct Agent 演示")
    print("=" * 60)

    agent = ReActAgent()

    test_queries = [
        "计算 25 * 4 + 10",
        "搜索 Python 编程",
        "查找我的记忆中关于学习的内容"
    ]

    for query in test_queries:
        print(f"\n📝 查询: {query}")
        print("-" * 40)

        result = agent.process(query)
        print(f"\n🎯 结果: {result}")

        print(f"\n📋 执行步骤:")
        print(agent.get_steps_summary())
        print("\n" + "="*40)


def demo_reflect_agent():
    """演示Reflect Agent"""
    print("\n" + "=" * 60)
    print("🪞 Reflect Agent 演示")
    print("=" * 60)

    agent = ReflectAgent()

    test_queries = [
        "分析这段文本的特点",
        "如何提高工作效率？"
    ]

    for query in test_queries:
        print(f"\n📝 查询: {query}")
        print("-" * 40)

        result = agent.process(query)
        print(f"\n🎯 结果: {result}")

        print(f"\n🪞 反思记录:")
        for i, reflection in enumerate(agent.reflections, 1):
            print(f"  {i}. {reflection}")

        print(f"\n📋 执行步骤:")
        print(agent.get_steps_summary())
        print("\n" + "="*40)


def demo_planning_agent():
    """演示Planning Agent"""
    print("\n" + "=" * 60)
    print("📋 Planning Agent 演示")
    print("=" * 60)

    agent = PlanningAgent()

    test_queries = [
        "研究机器学习的应用领域",
        "学习Python编程的完整计划"
    ]

    for query in test_queries:
        print(f"\n📝 查询: {query}")
        print("-" * 40)

        result = agent.process(query)
        print(f"\n🎯 结果: {result}")

        print(f"\n📋 执行计划:")
        for i, task in enumerate(agent.plan, 1):
            print(f"  {i}. {task['description']}")

        print(f"\n📋 详细步骤:")
        print(agent.get_steps_summary())
        print("\n" + "="*40)


def demo_collaborative_agent():
    """演示协作Agent"""
    print("\n" + "=" * 60)
    print("🤝 Collaborative Agent 演示")
    print("=" * 60)

    # 创建协作Agent和专家
    coordinator = CollaborativeAgent("协调者")

    # 添加专家
    coordinator.add_specialist("calculator_expert", ReActAgent("计算专家"))
    coordinator.add_specialist("analysis_expert", ReflectAgent("分析专家"))
    coordinator.add_specialist("planning_expert", PlanningAgent("规划专家"))

    test_queries = [
        "计算 15 * 8 + 25",
        "分析当前AI发展趋势",
        "制定学习深度学习的计划"
    ]

    for query in test_queries:
        print(f"\n📝 查询: {query}")
        print("-" * 40)

        result = coordinator.process(query)
        print(f"\n🎯 结果: {result}")

        print(f"\n📋 协调步骤:")
        print(coordinator.get_steps_summary())
        print("\n" + "="*40)


def compare_agent_patterns():
    """比较不同Agent模式"""
    print("\n" + "=" * 80)
    print("🔍 Agent模式对比分析")
    print("=" * 80)

    patterns = {
        "ReAct Agent": {
            "描述": "推理与行动交替进行，通过观察结果指导下一步",
            "优势": ["逻辑清晰", "可解释性强", "适合需要工具调用的任务"],
            "劣势": ["可能陷入局部循环", "对复杂任务分解能力有限"],
            "适用场景": ["数学计算", "信息查询", "简单推理任务"]
        },
        "Reflect Agent": {
            "描述": "具有自我反思能力，能够评估和改进自己的回答",
            "优势": ["自我改进", "质量控制", "持续学习"],
            "劣势": ["计算开销较大", "可能过度反思"],
            "适用场景": ["内容生成", "质量要求高的任务", "创意写作"]
        },
        "Planning Agent": {
            "描述": "先制定详细计划再执行，适合复杂任务分解",
            "优势": ["任务分解能力强", "执行有条理", "适合复杂项目"],
            "劣势": ["规划开销大", "不够灵活", "可能过度规划"],
            "适用场景": ["项目管理", "研究任务", "学习规划"]
        },
        "Collaborative Agent": {
            "描述": "多个专家Agent协同工作，发挥各自优势",
            "优势": ["专业化分工", "质量验证", "互补优势"],
            "劣势": ["协调复杂", "资源消耗大", "通信开销"],
            "适用场景": ["复杂问题解决", "多领域任务", "高质量要求"]
        }
    }

    for pattern_name, info in patterns.items():
        print(f"\n📊 {pattern_name}")
        print("-" * 50)
        print(f"📝 描述: {info['描述']}")
        print(f"✅ 优势: {', '.join(info['优势'])}")
        print(f"❌ 劣势: {', '.join(info['劣势'])}")
        print(f"🎯 适用场景: {', '.join(info['适用场景'])}")


def advanced_agent_patterns():
    """介绍高级Agent模式"""
    print("\n" + "=" * 80)
    print("🚀 高级Agent模式介绍")
    print("=" * 80)

    advanced_patterns = {
        "Tree of Thoughts (ToT)": {
            "描述": "以树状结构探索多个思考路径，选择最优解",
            "特点": ["多路径探索", "回溯机制", "最优解选择"],
            "实现要点": ["状态表示", "搜索策略", "评估函数"]
        },
        "Chain of Thought (CoT)": {
            "描述": "逐步推理，通过中间步骤得出最终答案",
            "特点": ["步骤推理", "逻辑链条", "透明过程"],
            "实现要点": ["提示工程", "步骤分解", "逻辑验证"]
        },
        "Multi-Agent Debate": {
            "描述": "多个Agent辩论讨论，通过不同观点得出更好的结论",
            "特点": ["观点对抗", "论据交换", "共识达成"],
            "实现要点": ["角色设定", "辩论规则", "结论总结"]
        },
        "Self-Consistency": {
            "描述": "生成多个推理路径，选择最一致的答案",
            "特点": ["多次采样", "一致性检查", "投票机制"],
            "实现要点": ["多样性生成", "一致性度量", "结果聚合"]
        },
        "AutoGPT Pattern": {
            "描述": "自主设定目标、制定计划、执行任务的循环模式",
            "特点": ["目标导向", "自主规划", "持续执行"],
            "实现要点": ["目标分解", "进度跟踪", "自主调整"]
        }
    }

    for pattern_name, info in advanced_patterns.items():
        print(f"\n🎯 {pattern_name}")
        print("-" * 50)
        print(f"📝 描述: {info['描述']}")
        print(f"⭐ 特点: {', '.join(info['特点'])}")
        print(f"🔧 实现要点: {', '.join(info['实现要点'])}")


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("🤖 Agent模式完全指南")
    print("=" * 80)
    print("""
    这个演示包含了常用的Agent模式实现：
    
    1. ReAct Agent - 推理与行动结合
    2. Reflect Agent - 自我反思改进
    3. Planning Agent - 计划导向执行
    4. Collaborative Agent - 多Agent协作
    
    选择要演示的模式：
    1 - ReAct Agent 演示
    2 - Reflect Agent 演示  
    3 - Planning Agent 演示
    4 - Collaborative Agent 演示
    5 - Agent模式对比
    6 - 高级Agent模式介绍
    7 - 全部演示
    0 - 退出
    """)

    while True:
        try:
            choice = input("\n请选择 (0-7): ").strip()

            if choice == '0':
                print("\n👋 感谢使用Agent模式演示系统！")
                break
            elif choice == '1':
                demo_react_agent()
            elif choice == '2':
                demo_reflect_agent()
            elif choice == '3':
                demo_planning_agent()
            elif choice == '4':
                demo_collaborative_agent()
            elif choice == '5':
                compare_agent_patterns()
            elif choice == '6':
                advanced_agent_patterns()
            elif choice == '7':
                demo_react_agent()
                demo_reflect_agent()
                demo_planning_agent()
                demo_collaborative_agent()
                compare_agent_patterns()
                advanced_agent_patterns()
            else:
                print("❌ 无效选择，请输入 0-7")

        except KeyboardInterrupt:
            print("\n\n👋 程序被中断，再见！")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")


if __name__ == "__main__":
    main()