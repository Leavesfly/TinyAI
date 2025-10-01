# -*- coding: utf-8 -*-
"""
从零构建的基于LLM的Multi-Agent系统
包含完整的Agent架构、通信机制、协作框架和实际应用示例
"""

import asyncio
import json
import uuid
import time
from typing import Dict, List, Any, Optional, Callable, Union, AsyncGenerator
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
import logging
from contextlib import asynccontextmanager


# ================== 基础数据结构 ==================

class MessageType(Enum):
    """消息类型枚举"""
    TEXT = "text"
    TASK = "task"
    RESULT = "result"
    ERROR = "error"
    SYSTEM = "system"
    BROADCAST = "broadcast"


class AgentState(Enum):
    """Agent状态枚举"""
    IDLE = "idle"
    BUSY = "busy"
    THINKING = "thinking"
    COMMUNICATING = "communicating"
    ERROR = "error"
    OFFLINE = "offline"


@dataclass
class Message:
    """消息类"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    receiver_id: str = ""
    message_type: MessageType = MessageType.TEXT
    content: Any = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 1  # 1-10, 10最高优先级

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "message_type": self.message_type.value,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority
        }


@dataclass
class Task:
    """任务类"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    assigned_to: Optional[str] = None
    created_by: str = ""
    status: str = "pending"  # pending, in_progress, completed, failed
    priority: int = 1
    deadline: Optional[datetime] = None
    result: Any = None
    subtasks: List['Task'] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


# ================== LLM模拟器 ==================

class LLMSimulator:
    """LLM模拟器 - 模拟真实的LLM API调用"""

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        self.temperature = 0.7
        self.max_tokens = 2048

        # 预定义的回复模板，用于模拟不同类型的Agent回复
        self.response_templates = {
            "analyst": [
                "根据数据分析，我发现{}的关键指标显示{}趋势。建议重点关注{}方面的改进。",
                "从分析结果来看，{}表现出{}的特征。我建议采取{}策略来优化。",
                "数据显示{}，这表明{}。为了改善情况，建议实施{}措施。"
            ],
            "researcher": [
                "通过深入研究，我发现{}领域存在{}的现象。相关文献表明{}。",
                "我的研究表明{}具有{}的特性。基于现有研究，我认为{}。",
                "研究发现{}与{}之间存在关联。建议进一步探索{}方向。"
            ],
            "coordinator": [
                "根据团队情况，我建议{}负责{}任务。预计完成时间为{}。",
                "为了提高效率，我重新分配任务：{}。请各位按照新的安排执行。",
                "项目进度更新：{}已完成，{}正在进行中，{}需要加快速度。"
            ],
            "executor": [
                "我已经完成了{}任务，结果是{}。下一步建议执行{}。",
                "正在执行{}操作，当前进度{}%。预计还需要{}时间完成。",
                "任务执行遇到{}问题，已采取{}措施，现在状态是{}。"
            ],
            "critic": [
                "从质量角度看，{}存在{}问题。建议在{}方面进行改进。",
                "评估结果表明{}达到了{}标准，但在{}方面仍有提升空间。",
                "这个方案的优点是{}，但需要注意{}风险。建议调整{}。"
            ]
        }

    async def chat_completion(self, messages: List[Dict[str, str]],
                            agent_type: str = "general") -> str:
        """模拟聊天完成API"""
        # 模拟API调用延迟
        await asyncio.sleep(0.5 + len(messages) * 0.1)

        # 获取最后一条用户消息
        last_message = messages[-1]["content"] if messages else ""

        # 根据Agent类型选择回复模板
        templates = self.response_templates.get(agent_type, ["我理解了{}，将会{}。"])
        template = templates[hash(last_message) % len(templates)]

        # 简单的内容提取和替换
        keywords = self._extract_keywords(last_message)
        response = self._fill_template(template, keywords)

        return response

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 简化的关键词提取
        keywords = []
        if "分析" in text:
            keywords.extend(["市场数据", "上升", "用户体验"])
        elif "研究" in text:
            keywords.extend(["人工智能", "显著", "深度学习"])
        elif "任务" in text:
            keywords.extend(["Alice", "文档编写", "2小时"])
        elif "执行" in text:
            keywords.extend(["数据处理", "85", "30分钟"])
        elif "评估" in text:
            keywords.extend(["产品质量", "优秀", "用户界面"])
        else:
            keywords.extend(["项目", "进展顺利", "继续推进"])

        return keywords

    def _fill_template(self, template: str, keywords: List[str]) -> str:
        """填充模板"""
        try:
            return template.format(*keywords)
        except:
            return template.replace("{}", "相关内容")


# ================== 通信系统 ==================

class MessageBus:
    """消息总线 - 负责Agent间的通信"""

    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.message_history: List[Message] = []
        self.max_history = 1000

    def subscribe(self, agent_id: str, callback: Callable) -> None:
        """订阅消息"""
        if agent_id not in self.subscribers:
            self.subscribers[agent_id] = []
        self.subscribers[agent_id].append(callback)

    def unsubscribe(self, agent_id: str) -> None:
        """取消订阅"""
        if agent_id in self.subscribers:
            del self.subscribers[agent_id]

    async def publish(self, message: Message) -> None:
        """发布消息"""
        # 记录消息历史
        self.message_history.append(message)
        if len(self.message_history) > self.max_history:
            self.message_history = self.message_history[-self.max_history:]

        # 广播消息
        if message.receiver_id == "broadcast":
            for agent_id, callbacks in self.subscribers.items():
                if agent_id != message.sender_id:  # 不发送给自己
                    for callback in callbacks:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(message)
                            else:
                                callback(message)
                        except Exception as e:
                            logging.error(f"Error delivering message to {agent_id}: {e}")
        else:
            # 点对点消息
            if message.receiver_id in self.subscribers:
                for callback in self.subscribers[message.receiver_id]:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(message)
                        else:
                            callback(message)
                    except Exception as e:
                        logging.error(f"Error delivering message to {message.receiver_id}: {e}")

    def get_conversation_history(self, agent1_id: str, agent2_id: str,
                               limit: int = 10) -> List[Message]:
        """获取两个Agent之间的对话历史"""
        messages = []
        for msg in reversed(self.message_history):
            if ((msg.sender_id == agent1_id and msg.receiver_id == agent2_id) or
                (msg.sender_id == agent2_id and msg.receiver_id == agent1_id)):
                messages.append(msg)
                if len(messages) >= limit:
                    break
        return list(reversed(messages))


# ================== Agent基类 ==================

class BaseAgent(ABC):
    """Agent基类"""

    def __init__(self, agent_id: str, name: str, role: str,
                 message_bus: MessageBus, llm: LLMSimulator):
        self.agent_id = agent_id
        self.name = name
        self.role = role
        self.state = AgentState.IDLE
        self.message_bus = message_bus
        self.llm = llm

        # Agent配置
        self.capabilities: List[str] = []
        self.tools: Dict[str, Callable] = {}
        self.memory: List[Dict[str, Any]] = []
        self.max_memory = 100

        # 工作相关
        self.current_task: Optional[Task] = None
        self.task_queue: List[Task] = []
        self.completed_tasks: List[Task] = []

        # 性能指标
        self.metrics = {
            "tasks_completed": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "average_response_time": 0.0,
            "error_count": 0
        }

        # 订阅消息
        self.message_bus.subscribe(self.agent_id, self._handle_message)

        # 启动Agent
        self._running = True
        self._task_loop = None

    async def start(self) -> None:
        """启动Agent"""
        self.state = AgentState.IDLE
        self._task_loop = asyncio.create_task(self._run_loop())
        logging.info(f"Agent {self.name} started")

    async def stop(self) -> None:
        """停止Agent"""
        self._running = False
        self.state = AgentState.OFFLINE
        if self._task_loop:
            self._task_loop.cancel()
        self.message_bus.unsubscribe(self.agent_id)
        logging.info(f"Agent {self.name} stopped")

    async def _run_loop(self) -> None:
        """Agent主循环"""
        while self._running:
            try:
                if self.state == AgentState.IDLE and self.task_queue:
                    # 执行队列中的任务
                    task = self.task_queue.pop(0)
                    await self._execute_task(task)
                else:
                    # 执行周期性工作
                    await self._periodic_work()

                await asyncio.sleep(0.1)  # 避免过度占用CPU

            except Exception as e:
                logging.error(f"Error in {self.name} run loop: {e}")
                self.state = AgentState.ERROR
                self.metrics["error_count"] += 1
                await asyncio.sleep(1)

    async def _handle_message(self, message: Message) -> None:
        """处理接收到的消息"""
        self.metrics["messages_received"] += 1

        # 记忆消息
        self._remember({
            "type": "message_received",
            "message": message.to_dict(),
            "timestamp": datetime.now().isoformat()
        })

        # 根据消息类型处理
        if message.message_type == MessageType.TASK:
            task = Task(**message.content)
            self.task_queue.append(task)
        elif message.message_type == MessageType.TEXT:
            await self._handle_text_message(message)
        elif message.message_type == MessageType.SYSTEM:
            await self._handle_system_message(message)

    async def _handle_text_message(self, message: Message) -> None:
        """处理文本消息"""
        # 构建对话上下文
        context = self._build_conversation_context(message.sender_id)

        # 生成回复
        response = await self._generate_response(message.content, context)

        # 发送回复
        await self.send_message(
            receiver_id=message.sender_id,
            content=response,
            message_type=MessageType.TEXT
        )

    async def _handle_system_message(self, message: Message) -> None:
        """处理系统消息"""
        if message.content.get("command") == "status":
            status = self._get_status()
            await self.send_message(
                receiver_id=message.sender_id,
                content=status,
                message_type=MessageType.RESULT
            )

    def _build_conversation_context(self, other_agent_id: str) -> List[Dict[str, str]]:
        """构建对话上下文"""
        messages = self.message_bus.get_conversation_history(
            self.agent_id, other_agent_id, limit=5
        )

        context = []
        for msg in messages:
            role = "assistant" if msg.sender_id == self.agent_id else "user"
            context.append({
                "role": role,
                "content": str(msg.content)
            })

        return context

    async def _generate_response(self, input_text: str,
                               context: List[Dict[str, str]]) -> str:
        """生成回复"""
        self.state = AgentState.THINKING

        # 添加系统提示
        system_prompt = f"你是{self.name}，角色是{self.role}。{self._get_role_prompt()}"

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(context)
        messages.append({"role": "user", "content": input_text})

        try:
            response = await self.llm.chat_completion(
                messages, agent_type=self._get_agent_type()
            )
            self.state = AgentState.IDLE
            return response
        except Exception as e:
            self.state = AgentState.ERROR
            return f"抱歉，我遇到了一些问题：{e}"

    async def _execute_task(self, task: Task) -> None:
        """执行任务"""
        self.state = AgentState.BUSY
        self.current_task = task
        task.status = "in_progress"
        task.assigned_to = self.agent_id

        start_time = time.time()

        try:
            # 执行具体任务
            result = await self._perform_task(task)

            # 更新任务状态
            task.status = "completed"
            task.result = result
            task.updated_at = datetime.now()

            self.completed_tasks.append(task)
            self.metrics["tasks_completed"] += 1

            # 通知任务创建者
            if task.created_by and task.created_by != self.agent_id:
                await self.send_message(
                    receiver_id=task.created_by,
                    content={
                        "task_id": task.id,
                        "status": "completed",
                        "result": result
                    },
                    message_type=MessageType.RESULT
                )

        except Exception as e:
            task.status = "failed"
            task.result = {"error": str(e)}
            task.updated_at = datetime.now()
            self.metrics["error_count"] += 1

            logging.error(f"Task {task.id} failed in {self.name}: {e}")

        finally:
            # 更新性能指标
            execution_time = time.time() - start_time
            self._update_response_time(execution_time)

            self.current_task = None
            self.state = AgentState.IDLE

    async def send_message(self, receiver_id: str, content: Any,
                         message_type: MessageType = MessageType.TEXT,
                         priority: int = 1) -> None:
        """发送消息"""
        message = Message(
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content,
            priority=priority
        )

        await self.message_bus.publish(message)
        self.metrics["messages_sent"] += 1

        # 记忆发送的消息
        self._remember({
            "type": "message_sent",
            "message": message.to_dict(),
            "timestamp": datetime.now().isoformat()
        })

    def _remember(self, memory_item: Dict[str, Any]) -> None:
        """添加记忆"""
        self.memory.append(memory_item)
        if len(self.memory) > self.max_memory:
            self.memory = self.memory[-self.max_memory:]

    def _update_response_time(self, execution_time: float) -> None:
        """更新平均响应时间"""
        total_tasks = self.metrics["tasks_completed"] + 1
        current_avg = self.metrics["average_response_time"]
        new_avg = (current_avg * (total_tasks - 1) + execution_time) / total_tasks
        self.metrics["average_response_time"] = new_avg

    def _get_status(self) -> Dict[str, Any]:
        """获取Agent状态"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "role": self.role,
            "state": self.state.value,
            "current_task": self.current_task.id if self.current_task else None,
            "task_queue_length": len(self.task_queue),
            "metrics": self.metrics
        }

    @abstractmethod
    async def _perform_task(self, task: Task) -> Any:
        """执行具体任务 - 子类需要实现"""
        pass

    @abstractmethod
    def _get_role_prompt(self) -> str:
        """获取角色提示 - 子类需要实现"""
        pass

    @abstractmethod
    def _get_agent_type(self) -> str:
        """获取Agent类型 - 子类需要实现"""
        pass

    async def _periodic_work(self) -> None:
        """周期性工作 - 子类可以重写"""
        pass


# ================== 具体Agent实现 ==================

class AnalystAgent(BaseAgent):
    """分析师Agent"""

    def __init__(self, agent_id: str, message_bus: MessageBus, llm: LLMSimulator):
        super().__init__(agent_id, f"分析师-{agent_id[-4:]}", "数据分析师", message_bus, llm)
        self.capabilities = ["数据分析", "趋势预测", "报告生成"]

    def _get_role_prompt(self) -> str:
        return ("你擅长数据分析和趋势预测。你会仔细分析数据，"
                "发现关键模式，并提供有价值的洞察和建议。")

    def _get_agent_type(self) -> str:
        return "analyst"

    async def _perform_task(self, task: Task) -> Dict[str, Any]:
        """执行分析任务"""
        if "分析" in task.description:
            # 模拟数据分析过程
            await asyncio.sleep(2)  # 模拟分析时间

            return {
                "analysis_type": "数据分析",
                "findings": [
                    "数据显示上升趋势",
                    "关键指标超出预期",
                    "需要关注潜在风险"
                ],
                "recommendations": [
                    "继续监控关键指标",
                    "加强风险控制措施",
                    "优化数据收集流程"
                ],
                "confidence": 0.85
            }
        else:
            return {"message": "任务不在我的专业范围内"}


class ResearcherAgent(BaseAgent):
    """研究员Agent"""

    def __init__(self, agent_id: str, message_bus: MessageBus, llm: LLMSimulator):
        super().__init__(agent_id, f"研究员-{agent_id[-4:]}", "研究员", message_bus, llm)
        self.capabilities = ["文献调研", "实验设计", "理论分析"]

    def _get_role_prompt(self) -> str:
        return ("你是一位严谨的研究员，擅长文献调研、实验设计和理论分析。"
                "你会深入研究问题，寻找科学依据，提供基于证据的结论。")

    def _get_agent_type(self) -> str:
        return "researcher"

    async def _perform_task(self, task: Task) -> Dict[str, Any]:
        """执行研究任务"""
        if "研究" in task.description:
            await asyncio.sleep(3)  # 模拟研究时间

            return {
                "research_type": "理论研究",
                "methodology": "文献分析法",
                "key_findings": [
                    "相关理论支持假设",
                    "实验数据验证理论",
                    "发现新的研究方向"
                ],
                "literature_review": "基于20篇相关论文的分析",
                "next_steps": ["深入实验验证", "扩大样本规模", "跨领域合作"]
            }
        else:
            return {"message": "需要更具体的研究方向"}


class CoordinatorAgent(BaseAgent):
    """协调员Agent"""

    def __init__(self, agent_id: str, message_bus: MessageBus, llm: LLMSimulator):
        super().__init__(agent_id, f"协调员-{agent_id[-4:]}", "项目协调员", message_bus, llm)
        self.capabilities = ["任务分配", "进度跟踪", "团队协调"]
        self.team_agents: List[str] = []

    def _get_role_prompt(self) -> str:
        return ("你是团队协调员，负责任务分配、进度跟踪和团队协调。"
                "你会根据team成员的能力合理分配任务，确保项目顺利进行。")

    def _get_agent_type(self) -> str:
        return "coordinator"

    def add_team_member(self, agent_id: str) -> None:
        """添加团队成员"""
        if agent_id not in self.team_agents:
            self.team_agents.append(agent_id)

    async def _perform_task(self, task: Task) -> Dict[str, Any]:
        """执行协调任务"""
        if "协调" in task.description or "分配" in task.description:
            # 模拟任务分配
            assignments = []
            for i, agent_id in enumerate(self.team_agents):
                subtask = Task(
                    title=f"子任务-{i+1}",
                    description=f"执行{task.title}的第{i+1}部分",
                    created_by=self.agent_id,
                    assigned_to=agent_id
                )
                assignments.append(subtask.id)

                # 发送子任务
                await self.send_message(
                    receiver_id=agent_id,
                    content=subtask.__dict__,
                    message_type=MessageType.TASK
                )

            return {
                "coordination_type": "任务分配",
                "assigned_tasks": assignments,
                "team_size": len(self.team_agents),
                "estimated_completion": "2-3小时"
            }
        else:
            return {"message": "协调任务处理完成"}

    async def _periodic_work(self) -> None:
        """定期检查团队状态"""
        if len(self.team_agents) > 0 and self.metrics["messages_sent"] % 10 == 0:
            # 定期向团队成员发送状态查询
            for agent_id in self.team_agents:
                await self.send_message(
                    receiver_id=agent_id,
                    content={"command": "status"},
                    message_type=MessageType.SYSTEM
                )


class ExecutorAgent(BaseAgent):
    """执行员Agent"""

    def __init__(self, agent_id: str, message_bus: MessageBus, llm: LLMSimulator):
        super().__init__(agent_id, f"执行员-{agent_id[-4:]}", "任务执行员", message_bus, llm)
        self.capabilities = ["任务执行", "工具使用", "结果报告"]

    def _get_role_prompt(self) -> str:
        return ("你是高效的任务执行员，专注于完成分配的具体任务。"
                "你会认真执行每个步骤，使用合适的工具，并及时报告进度。")

    def _get_agent_type(self) -> str:
        return "executor"

    async def _perform_task(self, task: Task) -> Dict[str, Any]:
        """执行具体任务"""
        # 模拟不同类型的执行任务
        execution_steps = []

        if "数据" in task.description:
            execution_steps = [
                "连接数据源",
                "提取数据",
                "处理数据",
                "生成报告"
            ]
        elif "文档" in task.description:
            execution_steps = [
                "收集信息",
                "编写草稿",
                "审核内容",
                "最终确认"
            ]
        else:
            execution_steps = [
                "分析任务需求",
                "制定执行计划",
                "逐步执行",
                "完成验收"
            ]

        # 模拟执行过程
        for i, step in enumerate(execution_steps):
            await asyncio.sleep(0.5)  # 模拟执行时间
            logging.info(f"{self.name} 执行步骤 {i+1}: {step}")

        return {
            "execution_type": "任务执行",
            "steps_completed": execution_steps,
            "total_steps": len(execution_steps),
            "execution_time": len(execution_steps) * 0.5,
            "status": "成功完成"
        }


class CriticAgent(BaseAgent):
    """评审员Agent"""

    def __init__(self, agent_id: str, message_bus: MessageBus, llm: LLMSimulator):
        super().__init__(agent_id, f"评审员-{agent_id[-4:]}", "质量评审员", message_bus, llm)
        self.capabilities = ["质量评估", "代码审查", "改进建议"]

    def _get_role_prompt(self) -> str:
        return ("你是严格的质量评审员，负责评估工作质量和提供改进建议。"
                "你会从多个角度审查结果，指出问题并提供建设性的改进方案。")

    def _get_agent_type(self) -> str:
        return "critic"

    async def _perform_task(self, task: Task) -> Dict[str, Any]:
        """执行评审任务"""
        if "评审" in task.description or "评估" in task.description:
            await asyncio.sleep(2)  # 模拟评审时间

            # 生成评审报告
            strengths = [
                "总体结构清晰",
                "关键功能完整",
                "文档比较规范"
            ]

            weaknesses = [
                "部分细节需要完善",
                "错误处理机制可以改进",
                "性能优化空间较大"
            ]

            recommendations = [
                "增加单元测试覆盖率",
                "优化算法效率",
                "改进用户界面体验"
            ]

            return {
                "review_type": "质量评审",
                "overall_score": 7.5,
                "strengths": strengths,
                "weaknesses": weaknesses,
                "recommendations": recommendations,
                "approval_status": "有条件通过"
            }
        else:
            return {"message": "评审任务已完成"}


# ================== 多Agent协作系统 ==================

class MultiAgentSystem:
    """多Agent系统管理器"""

    def __init__(self):
        self.message_bus = MessageBus()
        self.llm = LLMSimulator()
        self.agents: Dict[str, BaseAgent] = {}
        self.teams: Dict[str, List[str]] = {}
        self.system_metrics = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "active_agents": 0,
            "total_messages": 0
        }
        self._running = False

    async def add_agent(self, agent_class, agent_id: Optional[str] = None) -> str:
        """添加Agent"""
        if not agent_id:
            agent_id = f"{agent_class.__name__.lower()}_{len(self.agents)+1}"

        agent = agent_class(agent_id, self.message_bus, self.llm)
        self.agents[agent_id] = agent

        if self._running:
            await agent.start()

        self.system_metrics["active_agents"] = len(self.agents)
        logging.info(f"Added agent: {agent.name} (ID: {agent_id})")
        return agent_id

    async def remove_agent(self, agent_id: str) -> bool:
        """移除Agent"""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            await agent.stop()
            del self.agents[agent_id]

            # 从团队中移除
            for team_agents in self.teams.values():
                if agent_id in team_agents:
                    team_agents.remove(agent_id)

            self.system_metrics["active_agents"] = len(self.agents)
            logging.info(f"Removed agent: {agent.name}")
            return True
        return False

    def create_team(self, team_name: str, agent_ids: List[str]) -> bool:
        """创建团队"""
        # 验证所有Agent都存在
        for agent_id in agent_ids:
            if agent_id not in self.agents:
                logging.error(f"Agent {agent_id} not found")
                return False

        self.teams[team_name] = agent_ids.copy()

        # 如果有协调员，告知团队成员
        coordinators = [aid for aid in agent_ids
                       if isinstance(self.agents[aid], CoordinatorAgent)]

        for coord_id in coordinators:
            coord_agent = self.agents[coord_id]
            if isinstance(coord_agent, CoordinatorAgent):
                for agent_id in agent_ids:
                    if agent_id != coord_id:
                        coord_agent.add_team_member(agent_id)

        logging.info(f"Created team '{team_name}' with {len(agent_ids)} members")
        return True

    async def assign_task(self, task: Task, agent_id: Optional[str] = None,
                         team_name: Optional[str] = None) -> bool:
        """分配任务"""
        self.system_metrics["total_tasks"] += 1

        if agent_id:
            # 分配给特定Agent
            if agent_id in self.agents:
                await self.agents[agent_id].send_message(
                    receiver_id=agent_id,
                    content=task.__dict__,
                    message_type=MessageType.TASK
                )
                return True
        elif team_name:
            # 分配给团队（通过协调员）
            if team_name in self.teams:
                team_agents = self.teams[team_name]
                coordinators = [aid for aid in team_agents
                               if isinstance(self.agents[aid], CoordinatorAgent)]

                if coordinators:
                    # 有协调员，分配给协调员
                    coord_id = coordinators[0]
                    await self.message_bus.publish(Message(
                        sender_id="system",
                        receiver_id=coord_id,
                        message_type=MessageType.TASK,
                        content=task.__dict__
                    ))
                else:
                    # 没有协调员，分配给第一个Agent
                    first_agent = team_agents[0]
                    await self.message_bus.publish(Message(
                        sender_id="system",
                        receiver_id=first_agent,
                        message_type=MessageType.TASK,
                        content=task.__dict__
                    ))
                return True

        logging.error("Task assignment failed")
        return False

    async def start_system(self) -> None:
        """启动系统"""
        self._running = True

        # 启动所有Agent
        start_tasks = []
        for agent in self.agents.values():
            start_tasks.append(agent.start())

        if start_tasks:
            await asyncio.gather(*start_tasks)

        logging.info(f"Multi-agent system started with {len(self.agents)} agents")

    async def stop_system(self) -> None:
        """停止系统"""
        self._running = False

        # 停止所有Agent
        stop_tasks = []
        for agent in self.agents.values():
            stop_tasks.append(agent.stop())

        if stop_tasks:
            await asyncio.gather(*stop_tasks)

        logging.info("Multi-agent system stopped")

    async def broadcast_message(self, content: str, sender_id: str = "system") -> None:
        """广播消息"""
        message = Message(
            sender_id=sender_id,
            receiver_id="broadcast",
            message_type=MessageType.BROADCAST,
            content=content
        )

        await self.message_bus.publish(message)
        self.system_metrics["total_messages"] += 1

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        agent_statuses = {}
        for agent_id, agent in self.agents.items():
            agent_statuses[agent_id] = agent._get_status()

        return {
            "system_metrics": self.system_metrics,
            "agents": agent_statuses,
            "teams": self.teams,
            "message_history_length": len(self.message_bus.message_history)
        }

    async def simulate_conversation(self, agent1_id: str, agent2_id: str,
                                  initial_message: str, rounds: int = 3) -> List[Message]:
        """模拟两个Agent之间的对话"""
        conversation = []

        # 发送初始消息
        message = Message(
            sender_id=agent1_id,
            receiver_id=agent2_id,
            message_type=MessageType.TEXT,
            content=initial_message
        )

        await self.message_bus.publish(message)
        conversation.append(message)

        # 等待对话轮次
        for _ in range(rounds - 1):
            await asyncio.sleep(2)  # 等待回复

            # 获取最新的对话历史
            recent_messages = self.message_bus.get_conversation_history(
                agent1_id, agent2_id, limit=2
            )

            if len(recent_messages) > len(conversation):
                conversation.extend(recent_messages[len(conversation):])

        return conversation


# ================== 演示和测试 ==================

async def demo_basic_agent_communication():
    """演示基本的Agent通信"""
    print("\n" + "=" * 50)
    print("🤖 基本Agent通信演示")
    print("=" * 50)

    system = MultiAgentSystem()

    # 添加不同类型的Agent
    analyst_id = await system.add_agent(AnalystAgent)
    researcher_id = await system.add_agent(ResearcherAgent)

    # 启动系统
    await system.start_system()

    print(f"\n创建了两个Agent：")
    print(f"- 分析师 (ID: {analyst_id})")
    print(f"- 研究员 (ID: {researcher_id})")

    # 模拟对话
    print("\n🔄 开始Agent间对话...")
    conversation = await system.simulate_conversation(
        analyst_id, researcher_id,
        "你好，我想了解一下你的研究领域",
        rounds=3
    )

    print("\n💬 对话记录：")
    for i, msg in enumerate(conversation, 1):
        sender_name = system.agents[msg.sender_id].name
        receiver_name = system.agents[msg.receiver_id].name if msg.receiver_id in system.agents else "未知"
        print(f"{i}. {sender_name} -> {receiver_name}: {msg.content}")

    await system.stop_system()
    print("\n✅ 基本通信演示完成")


async def demo_task_assignment_and_execution():
    """演示任务分配和执行"""
    print("\n" + "=" * 50)
    print("📋 任务分配和执行演示")
    print("=" * 50)

    system = MultiAgentSystem()

    # 添加各种类型的Agent
    coord_id = await system.add_agent(CoordinatorAgent)
    analyst_id = await system.add_agent(AnalystAgent)
    executor_id = await system.add_agent(ExecutorAgent)
    critic_id = await system.add_agent(CriticAgent)

    # 创建团队
    system.create_team("项目团队", [coord_id, analyst_id, executor_id, critic_id])

    # 启动系统
    await system.start_system()

    print(f"\n创建了项目团队：")
    for agent_id in [coord_id, analyst_id, executor_id, critic_id]:
        agent = system.agents[agent_id]
        print(f"- {agent.name} (角色: {agent.role})")

    # 创建和分配任务
    tasks = [
        Task(
            title="市场数据分析",
            description="分析最新的市场趋势数据",
            created_by="user"
        ),
        Task(
            title="执行数据处理任务",
            description="处理收集到的数据并生成报告",
            created_by="user"
        ),
        Task(
            title="质量评审任务",
            description="评审生成的分析报告质量",
            created_by="user"
        )
    ]

    print("\n📝 分配任务：")
    await system.assign_task(tasks[0], analyst_id)
    print(f"- 任务1分配给：{system.agents[analyst_id].name}")

    await system.assign_task(tasks[1], executor_id)
    print(f"- 任务2分配给：{system.agents[executor_id].name}")

    await system.assign_task(tasks[2], critic_id)
    print(f"- 任务3分配给：{system.agents[critic_id].name}")

    # 等待任务执行
    print("\n⏳ 等待任务执行...")
    await asyncio.sleep(5)

    # 显示系统状态
    status = system.get_system_status()
    print("\n📊 系统状态：")
    print(f"- 总任务数：{status['system_metrics']['total_tasks']}")
    print(f"- 活跃Agent数：{status['system_metrics']['active_agents']}")

    print("\n🎯 Agent任务完成情况：")
    for agent_id, agent_status in status['agents'].items():
        agent_name = system.agents[agent_id].name
        completed = agent_status['metrics']['tasks_completed']
        print(f"- {agent_name}: 已完成 {completed} 个任务")

    await system.stop_system()
    print("\n✅ 任务执行演示完成")


async def demo_team_collaboration():
    """演示团队协作"""
    print("\n" + "=" * 50)
    print("👥 团队协作演示")
    print("=" * 50)

    system = MultiAgentSystem()

    # 创建完整的团队
    coord_id = await system.add_agent(CoordinatorAgent)
    analyst_id = await system.add_agent(AnalystAgent)
    researcher_id = await system.add_agent(ResearcherAgent)
    executor_id = await system.add_agent(ExecutorAgent)
    critic_id = await system.add_agent(CriticAgent)

    # 创建团队
    system.create_team("AI研发团队", [coord_id, analyst_id, researcher_id, executor_id, critic_id])

    # 启动系统
    await system.start_system()

    print("\n🏢 AI研发团队成员：")
    for agent_id in [coord_id, analyst_id, researcher_id, executor_id, critic_id]:
        agent = system.agents[agent_id]
        print(f"- {agent.name}: {agent.role}")
        print(f"  能力: {', '.join(agent.capabilities)}")

    # 创建复杂项目任务
    project_task = Task(
        title="AI产品开发项目",
        description="协调开发一个新的AI产品，包括需求分析、技术研究、实施和质量评估",
        created_by="product_manager",
        priority=5
    )

    print(f"\n🚀 启动项目：{project_task.title}")

    # 分配给团队（通过协调员）
    await system.assign_task(project_task, team_name="AI研发团队")

    # 广播项目启动消息
    await system.broadcast_message(
        "🎉 新项目正式启动！请各位team成员积极配合，确保项目成功！",
        sender_id=coord_id
    )

    print("\n📢 已发送项目启动广播")

    # 模拟项目执行过程
    print("\n⚙️ 模拟项目执行过程...")
    await asyncio.sleep(3)

    # 显示team协作情况
    print("\n📈 团队协作情况：")
    for agent_id in [coord_id, analyst_id, researcher_id, executor_id, critic_id]:
        agent = system.agents[agent_id]
        print(f"\n{agent.name} ({agent.role}):")
        print(f"  当前状态: {agent.state.value}")
        print(f"  任务队列: {len(agent.task_queue)} 个待处理任务")
        print(f"  已完成: {agent.metrics['tasks_completed']} 个任务")
        print(f"  消息统计: 发送 {agent.metrics['messages_sent']}, 接收 {agent.metrics['messages_received']}")

    # 显示消息交互历史
    print("\n💬 最近的消息交互：")
    recent_messages = system.message_bus.message_history[-5:]
    for msg in recent_messages:
        sender_name = msg.sender_id
        if msg.sender_id in system.agents:
            sender_name = system.agents[msg.sender_id].name

        print(f"  {sender_name}: {str(msg.content)[:100]}...")

    await system.stop_system()
    print("\n✅ 团队协作演示完成")


async def main():
    """主函数 - 运行所有演示"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("🌟 从零构建的基于LLM的Multi-Agent系统")
    print("=" * 60)
    print("\n这个系统展示了完整的多智能体架构：")
    print("• 🧠 智能Agent：分析师、研究员、协调员、执行员、评审员")
    print("• 📡 通信系统：消息总线、点对点通信、广播机制")
    print("• 🏗️  架构设计：模块化、可扩展、异步处理")
    print("• 🤝 协作机制：任务分配、团队协调、状态同步")
    print("• 📊 监控系统：性能指标、状态跟踪、历史记录")

    try:
        # 运行所有演示
        await demo_basic_agent_communication()
        await asyncio.sleep(1)

        await demo_task_assignment_and_execution()
        await asyncio.sleep(1)

        await demo_team_collaboration()

        print("\n" + "=" * 60)
        print("🎉 所有演示完成！")
        print("\n这个Multi-Agent系统具备以下特点：")
        print("✨ 完全从零构建，无外部依赖")
        print("✨ 基于LLM的智能对话能力")
        print("✨ 灵活的消息通信机制")
        print("✨ 支持复杂的团队协作")
        print("✨ 实时状态监控和指标统计")
        print("✨ 异步执行，高性能处理")
        print("\n💡 可以基于这个框架继续扩展：")
        print("• 添加更多专业的Agent类型")
        print("• 集成真实的LLM API")
        print("• 增加工具调用能力")
        print("• 添加持久化存储")
        print("• 构建Web界面进行可视化管理")

    except KeyboardInterrupt:
        print("\n👋 演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {e}")
        logging.error(f"Demo error: {e}", exc_info=True)


if __name__ == "__main__":
    # 运行演示
    asyncio.run(main())