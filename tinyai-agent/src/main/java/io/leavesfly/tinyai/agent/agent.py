# -*- coding: utf-8 -*-
"""
高级LLM Agent系统
包含完整的上下文工程：长短期记忆、RAG、工具调用等
"""

import json
import time
import hashlib
# import numpy as np  # 注释掉numpy导入，使用原生Python实现
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from collections import defaultdict, deque
import sqlite3
import re
import math


@dataclass
class Memory:
    """记忆单元"""
    id: str
    content: str
    memory_type: str  # 'working', 'episodic', 'semantic'
    timestamp: datetime
    importance: float = 0.0
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    """消息结构"""
    role: str  # 'user', 'assistant', 'system', 'tool'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCall:
    """工具调用结构"""
    id: str
    name: str
    arguments: Dict[str, Any]
    result: Optional[Any] = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class MemoryManager:
    """记忆管理系统"""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.working_memory = deque(maxlen=10)  # 工作记忆容量限制
        self.episodic_memory = []  # 情节记忆
        self.semantic_memory = {}  # 语义记忆
        self.memory_index = {}  # 记忆索引

        # 初始化数据库
        self._init_database()

    def _init_database(self):
        """初始化记忆数据库"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                timestamp REAL NOT NULL,
                importance REAL DEFAULT 0.0,
                access_count INTEGER DEFAULT 0,
                last_accessed REAL,
                embedding TEXT,
                metadata TEXT
            )
        """)
        self.conn.commit()

    def add_memory(self, content: str, memory_type: str,
                   importance: float = 0.0, metadata: Optional[Dict[str, Any]] = None) -> str:
        """添加记忆"""
        memory_id = hashlib.md5(f"{content}{datetime.now()}".encode()).hexdigest()[:16]

        memory = Memory(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            timestamp=datetime.now(),
            importance=importance,
            metadata=metadata or {}
        )

        # 根据记忆类型存储
        if memory_type == "working":
            self.working_memory.append(memory)
        elif memory_type == "episodic":
            self.episodic_memory.append(memory)
        elif memory_type == "semantic":
            key = self._extract_semantic_key(content)
            self.semantic_memory[key] = memory

        # 存储到数据库
        self._save_memory_to_db(memory)

        return memory_id

    def _extract_semantic_key(self, content: str) -> str:
        """提取语义记忆的键"""
        # 简单的关键词提取
        words = re.findall(r'\w+', content.lower())
        return ' '.join(sorted(set(words))[:5])  # 取前5个唯一词作为键

    def _save_memory_to_db(self, memory: Memory):
        """保存记忆到数据库"""
        self.conn.execute("""
            INSERT OR REPLACE INTO memories
            (id, content, memory_type, timestamp, importance, access_count,
             last_accessed, embedding, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            memory.id, memory.content, memory.memory_type,
            memory.timestamp.timestamp(), memory.importance, memory.access_count,
            memory.last_accessed.timestamp(),
            json.dumps(memory.embedding) if memory.embedding else None,
            json.dumps(memory.metadata)
        ))
        self.conn.commit()

    def retrieve_memories(self, query: str, memory_type: Optional[str] = None,
                         limit: int = 5) -> List[Memory]:
        """检索相关记忆"""
        memories = []

        # 从工作记忆检索
        if not memory_type or memory_type == "working":
            for memory in self.working_memory:
                if self._is_relevant(query, memory.content):
                    memories.append(memory)

        # 从情节记忆检索
        if not memory_type or memory_type == "episodic":
            for memory in self.episodic_memory:
                if self._is_relevant(query, memory.content):
                    memories.append(memory)

        # 从语义记忆检索
        if not memory_type or memory_type == "semantic":
            for key, memory in self.semantic_memory.items():
                if self._is_relevant(query, memory.content):
                    memories.append(memory)

        # 按重要性和相关性排序
        memories.sort(key=lambda m: (m.importance, m.access_count), reverse=True)

        # 更新访问统计
        for memory in memories[:limit]:
            memory.access_count += 1
            memory.last_accessed = datetime.now()
            self._save_memory_to_db(memory)

        return memories[:limit]

    def _is_relevant(self, query: str, content: str) -> bool:
        """判断记忆是否相关（简单的关键词匹配）"""
        query_words = set(re.findall(r'\w+', query.lower()))
        content_words = set(re.findall(r'\w+', content.lower()))

        # 计算词汇重叠度
        overlap = len(query_words & content_words)
        return overlap > 0

    def consolidate_memories(self):
        """记忆整合（将重要的工作记忆转移到长期记忆）"""
        current_time = datetime.now()

        for memory in list(self.working_memory):
            # 根据访问频率和重要性决定是否转移到长期记忆
            if memory.access_count > 2 or memory.importance > 0.7:
                if memory.memory_type == "working":
                    # 转移到情节记忆
                    memory.memory_type = "episodic"
                    self.episodic_memory.append(memory)
                    self._save_memory_to_db(memory)

    def get_memory_stats(self) -> Dict[str, Any]:
        """获取记忆统计"""
        return {
            "working_memory_count": len(self.working_memory),
            "episodic_memory_count": len(self.episodic_memory),
            "semantic_memory_count": len(self.semantic_memory),
            "total_memories": len(self.working_memory) + len(self.episodic_memory) + len(self.semantic_memory)
        }


class SimpleEmbedding:
    """简单的文本嵌入（基于TF-IDF的模拟）"""

    def __init__(self, dim: int = 128):
        self.dim = dim
        self.vocab = {}
        self.idf = {}

    def fit(self, texts: List[str]):
        """训练嵌入模型"""
        # 构建词汇表
        word_counts = defaultdict(int)
        doc_word_counts = defaultdict(set)

        for i, text in enumerate(texts):
            words = re.findall(r'\w+', text.lower())
            for word in words:
                word_counts[word] += 1
                doc_word_counts[word].add(i)

        # 构建词汇表索引
        self.vocab = {word: i for i, word in enumerate(word_counts.keys())}

        # 计算IDF
        num_docs = len(texts)
        for word, docs in doc_word_counts.items():
            self.idf[word] = math.log(num_docs / len(docs))

    def encode(self, text: str) -> List[float]:
        """编码文本为向量"""
        if not self.vocab:
            return [0.0] * self.dim

        words = re.findall(r'\w+', text.lower())
        word_counts = defaultdict(int)
        for word in words:
            word_counts[word] += 1

        # 计算TF-IDF向量
        vector = [0.0] * min(self.dim, len(self.vocab))

        for word, count in word_counts.items():
            if word in self.vocab:
                idx = self.vocab[word] % len(vector)
                tf = count / len(words)
                idf = self.idf.get(word, 1.0)
                vector[idx] += tf * idf

        # 归一化
        norm = math.sqrt(sum(x**2 for x in vector))
        if norm > 0:
            vector = [x / norm for x in vector]

        return vector + [0.0] * (self.dim - len(vector))

    def similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(x**2 for x in vec1))
        norm2 = math.sqrt(sum(x**2 for x in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)


class RAGSystem:
    """检索增强生成系统"""

    def __init__(self, embedding_dim: int = 128):
        self.documents = []  # 文档库
        self.document_embeddings = []  # 文档嵌入
        self.embedding_model = SimpleEmbedding(embedding_dim)
        self.index = {}  # 文档索引

    def add_document(self, doc_id: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """添加文档"""
        document = {
            "id": doc_id,
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now()
        }

        self.documents.append(document)

        # 更新嵌入模型
        all_texts = [doc["content"] for doc in self.documents]
        self.embedding_model.fit(all_texts)

        # 重新计算所有文档的嵌入
        self.document_embeddings = []
        for doc in self.documents:
            embedding = self.embedding_model.encode(doc["content"])
            self.document_embeddings.append(embedding)

        # 更新索引
        self.index[doc_id] = len(self.documents) - 1

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """检索相关文档"""
        if not self.documents:
            return []

        query_embedding = self.embedding_model.encode(query)

        # 计算相似度
        similarities = []
        for i, doc_embedding in enumerate(self.document_embeddings):
            similarity = self.embedding_model.similarity(query_embedding, doc_embedding)
            similarities.append((similarity, i))

        # 排序并返回top-k
        similarities.sort(reverse=True)

        results = []
        for similarity, idx in similarities[:top_k]:
            doc = self.documents[idx].copy()
            doc["similarity"] = similarity
            results.append(doc)

        return results

    def get_context(self, query: str, max_length: int = 1000) -> str:
        """获取查询相关的上下文"""
        relevant_docs = self.retrieve(query)

        context_parts = []
        current_length = 0

        for doc in relevant_docs:
            content = doc["content"]
            if current_length + len(content) <= max_length:
                context_parts.append(f"文档 {doc['id']}: {content}")
                current_length += len(content)
            else:
                # 截断最后一个文档
                remaining = max_length - current_length
                if remaining > 50:  # 至少保留50个字符
                    truncated = content[:remaining-3] + "..."
                    context_parts.append(f"文档 {doc['id']}: {truncated}")
                break

        return "\n\n".join(context_parts)


class ToolRegistry:
    """工具注册表"""

    def __init__(self):
        self.tools = {}

    def register(self, name: str, func: Callable, description: str,
                 parameters: Optional[Dict[str, Any]] = None):
        """注册工具"""
        self.tools[name] = {
            "function": func,
            "description": description,
            "parameters": parameters or {}
        }

    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """获取工具"""
        return self.tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """列出所有工具"""
        return [
            {
                "name": name,
                "description": info["description"],
                "parameters": info["parameters"]
            }
            for name, info in self.tools.items()
        ]

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> ToolCall:
        """调用工具"""
        tool_call = ToolCall(
            id=hashlib.md5(f"{name}{time.time()}".encode()).hexdigest()[:16],
            name=name,
            arguments=arguments
        )

        if name not in self.tools:
            tool_call.error = f"工具 '{name}' 不存在"
            return tool_call

        try:
            tool = self.tools[name]
            result = tool["function"](**arguments)
            tool_call.result = result
        except Exception as e:
            tool_call.error = str(e)

        return tool_call


class ContextEngine:
    """上下文工程引擎"""

    def __init__(self, max_context_length: int = 4000):
        self.max_context_length = max_context_length
        self.conversation_history = []
        self.system_prompts = []

    def add_system_prompt(self, prompt: str):
        """添加系统提示"""
        self.system_prompts.append(prompt)

    def add_message(self, message: Message):
        """添加消息到对话历史"""
        self.conversation_history.append(message)

    def build_context(self, current_query: str, relevant_memories: Optional[List[Memory]] = None,
                     rag_context: Optional[str] = None, tools_info: Optional[str] = None) -> str:
        """构建完整上下文"""
        context_parts = []

        # 1. 系统提示
        if self.system_prompts:
            context_parts.append("系统指令：\n" + "\n".join(self.system_prompts))

        # 2. 工具信息
        if tools_info:
            context_parts.append(f"可用工具：\n{tools_info}")

        # 3. 相关记忆
        if relevant_memories:
            memory_text = []
            for memory in relevant_memories:
                memory_text.append(f"[{memory.memory_type}记忆] {memory.content}")
            context_parts.append("相关记忆：\n" + "\n".join(memory_text))

        # 4. RAG上下文
        if rag_context:
            context_parts.append(f"相关文档：\n{rag_context}")

        # 5. 对话历史（压缩）
        compressed_history = self._compress_conversation_history()
        if compressed_history:
            context_parts.append(f"对话历史：\n{compressed_history}")

        # 6. 当前查询
        context_parts.append(f"当前问题：{current_query}")

        # 组合并截断
        full_context = "\n\n".join(context_parts)

        if len(full_context) > self.max_context_length:
            full_context = self._truncate_context(full_context)

        return full_context

    def _compress_conversation_history(self) -> str:
        """压缩对话历史"""
        if not self.conversation_history:
            return ""

        # 保留最近的几轮对话
        recent_messages = self.conversation_history[-6:]  # 最近3轮对话

        compressed = []
        for msg in recent_messages:
            role_map = {"user": "用户", "assistant": "助手", "system": "系统"}
            role = role_map.get(msg.role, msg.role)

            content = msg.content
            if len(content) > 200:  # 截断过长内容
                content = content[:197] + "..."

            compressed.append(f"{role}: {content}")

        return "\n".join(compressed)

    def _truncate_context(self, context: str) -> str:
        """截断上下文以适应长度限制"""
        if len(context) <= self.max_context_length:
            return context

        # 优先保留系统指令和当前查询
        lines = context.split('\n')

        # 找到重要部分的索引
        system_end = 0
        current_query_start = len(lines)

        for i, line in enumerate(lines):
            if line.startswith("系统指令："):
                system_end = i
            elif line.startswith("当前问题："):
                current_query_start = i
                break

        # 保留系统指令和当前查询
        important_parts = lines[:system_end+1] + lines[current_query_start:]
        important_text = '\n'.join(important_parts)

        # 计算剩余空间
        remaining_space = self.max_context_length - len(important_text)

        if remaining_space > 100:  # 如果还有空间，添加其他内容
            middle_parts = lines[system_end+1:current_query_start]
            middle_text = '\n'.join(middle_parts)

            if len(middle_text) <= remaining_space:
                return context
            else:
                # 截断中间部分
                truncated_middle = middle_text[:remaining_space-50] + "\n\n[内容被截断...]\n"
                return '\n'.join(lines[:system_end+1]) + '\n' + truncated_middle + '\n'.join(lines[current_query_start:])

        return important_text


# 基础工具函数
def calculator_tool(operation: str, a: float, b: float) -> Dict[str, Any]:
    """计算器工具"""
    try:
        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        elif operation == "multiply":
            result = a * b
        elif operation == "divide":
            if b == 0:
                return {"error": "除零错误"}
            result = a / b
        else:
            return {"error": f"不支持的操作: {operation}"}

        return {"result": result, "expression": f"{a} {operation} {b} = {result}"}
    except Exception as e:
        return {"error": str(e)}


def time_tool() -> Dict[str, Any]:
    """时间工具"""
    now = datetime.now()
    return {
        "current_time": now.isoformat(),
        "formatted_time": now.strftime("%Y年%m月%d日 %H:%M:%S"),
        "timestamp": now.timestamp()
    }


# 全局笔记存储
_note_storage = {"notes": {}, "next_id": 1}


def note_tool(action: str, content: str = "", note_id: Optional[str] = None) -> Dict[str, Any]:
    """笔记工具"""
    global _note_storage

    if action == "create":
        if not content:
            return {"error": "笔记内容不能为空"}

        note_id = str(_note_storage["next_id"])
        _note_storage["notes"][note_id] = {
            "id": note_id,
            "content": content,
            "created_at": datetime.now().isoformat()
        }
        _note_storage["next_id"] += 1

        return {"message": f"已创建笔记 {note_id}", "note": _note_storage["notes"][note_id]}

    elif action == "list":
        return {"notes": list(_note_storage["notes"].values())}

    elif action == "get":
        if note_id and note_id in _note_storage["notes"]:
            return {"note": _note_storage["notes"][note_id]}
        else:
            return {"error": f"笔记 {note_id} 不存在"}

    elif action == "delete":
        if note_id and note_id in _note_storage["notes"]:
            deleted_note = _note_storage["notes"].pop(note_id)
            return {"message": f"已删除笔记 {note_id}", "deleted_note": deleted_note}
        else:
            return {"error": f"笔记 {note_id} 不存在"}

    else:
        return {"error": f"不支持的操作: {action}"}


class AdvancedAgent:
    """高级LLM Agent"""

    def __init__(self, name: str, system_prompt: str = "", max_context_length: int = 4000):
        self.name = name
        self.system_prompt = system_prompt

        # 核心组件
        self.memory_manager = MemoryManager()
        self.rag_system = RAGSystem()
        self.tool_registry = ToolRegistry()
        self.context_engine = ContextEngine(max_context_length)

        # 对话状态
        self.conversation_history = []
        self.current_session_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:16]

        # 初始化系统提示
        if system_prompt:
            self.context_engine.add_system_prompt(system_prompt)

        # 注册默认工具
        self._register_default_tools()

    def _register_default_tools(self):
        """注册默认工具"""
        self.tool_registry.register(
            "calculator", calculator_tool,
            "执行数学计算：加法、减法、乘法、除法",
            {
                "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
                "a": {"type": "number"},
                "b": {"type": "number"}
            }
        )

        self.tool_registry.register(
            "time", time_tool,
            "获取当前时间信息",
            {}
        )

        self.tool_registry.register(
            "note", note_tool,
            "管理笔记：创建、查看、列出、删除笔记",
            {
                "action": {"type": "string", "enum": ["create", "list", "get", "delete"]},
                "content": {"type": "string", "description": "笔记内容（创建时必需）"},
                "note_id": {"type": "string", "description": "笔记ID（获取/删除时必需）"}
            }
        )

    def add_knowledge(self, content: str, doc_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """添加知识到RAG系统"""
        if not doc_id:
            doc_id = hashlib.md5(content.encode()).hexdigest()[:16]

        self.rag_system.add_document(doc_id, content, metadata)

        # 同时添加到语义记忆
        self.memory_manager.add_memory(
            content, "semantic",
            importance=0.8,  # 知识文档重要性较高
            metadata={"type": "knowledge", "doc_id": doc_id}
        )

    def register_tool(self, name: str, func: Callable, description: str, parameters: Optional[Dict[str, Any]] = None):
        """注册新工具"""
        self.tool_registry.register(name, func, description, parameters)

    def _parse_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """解析文本中的工具调用（简单实现）"""
        tool_calls = []

        # 寻找工具调用模式，如：[tool:calculator(operation="add", a=10, b=5)]
        pattern = r'\[tool:(\w+)\(([^\]]*)\)\]'
        matches = re.findall(pattern, text)

        for tool_name, args_str in matches:
            if tool_name in self.tool_registry.tools:
                try:
                    # 简单的参数解析
                    args = {}
                    if args_str.strip():
                        # 处理简单的key=value格式
                        for arg_pair in args_str.split(','):
                            if '=' in arg_pair:
                                key, value = arg_pair.split('=', 1)
                                key = key.strip().strip('"\'')
                                value = value.strip().strip('"\'')

                                # 尝试转换数值类型
                                try:
                                    if '.' in value:
                                        value = float(value)
                                    elif value.isdigit():
                                        value = int(value)
                                except:
                                    pass  # 保持字符串类型

                                args[key] = value

                    tool_calls.append({
                        "name": tool_name,
                        "arguments": args
                    })
                except Exception as e:
                    print(f"解析工具调用失败: {e}")

        return tool_calls

    def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[ToolCall]:
        """执行工具调用"""
        results = []

        for call_spec in tool_calls:
            tool_call = self.tool_registry.call_tool(
                call_spec["name"],
                call_spec["arguments"]
            )
            results.append(tool_call)

            # 记录工具调用到工作记忆
            self.memory_manager.add_memory(
                f"调用工具 {tool_call.name}: {tool_call.arguments} -> {tool_call.result or tool_call.error}",
                "working",
                importance=0.6
            )

        return results

    def _simulate_llm_response(self, context: str) -> str:
        """模拟LLM响应（实际使用时应调用真实的LLM API）"""
        # 这里是一个简化的响应生成逻辑
        # 在实际应用中，这里应该调用GPT、Claude等LLM API

        context_lower = context.lower()

        # 检查是否需要使用工具
        if "计算" in context or "算" in context:
            # 提取数字和操作符
            numbers = re.findall(r'\d+(?:\.\d+)?', context)
            if "+" in context and len(numbers) >= 2:
                return f"我来帮你计算。[tool:calculator(operation=\"add\", a={numbers[0]}, b={numbers[1]})]"
            elif "*" in context and len(numbers) >= 2:
                return f"我来帮你计算。[tool:calculator(operation=\"multiply\", a={numbers[0]}, b={numbers[1]})]"

        elif "时间" in context or "现在几点" in context:
            return "让我查看当前时间。[tool:time()]"

        elif "创建笔记" in context or "记录" in context:
            # 提取笔记内容
            content_match = re.search(r'[""](.*?)[""]', context)
            if content_match:
                content = content_match.group(1)
                return f"我来为你创建笔记。[tool:note(action=\"create\", content=\"{content}\")]"

        elif "查看笔记" in context or "笔记列表" in context:
            return "我来查看你的笔记列表。[tool:note(action=\"list\")]"

        # 默认对话响应
        responses = [
            "我理解了你的问题，让我来帮助你。",
            "这是一个很好的问题。基于我的知识和记忆...",
            "根据我们之前的对话和相关信息...",
            "我可以为你提供以下帮助..."
        ]

        import random
        return random.choice(responses)

    def process_message(self, user_input: str) -> str:
        """处理用户消息的主要方法"""
        # 1. 记录用户输入
        user_message = Message(role="user", content=user_input)
        self.conversation_history.append(user_message)
        self.context_engine.add_message(user_message)

        # 2. 记录到情节记忆
        self.memory_manager.add_memory(
            f"用户说: {user_input}",
            "episodic",
            importance=0.5
        )

        # 3. 检索相关记忆
        relevant_memories = self.memory_manager.retrieve_memories(user_input, limit=3)

        # 4. 检索相关知识（RAG）
        rag_context = self.rag_system.get_context(user_input, max_length=800)

        # 5. 获取工具信息
        tools_info = "\n".join([
            f"- {tool['name']}: {tool['description']}"
            for tool in self.tool_registry.list_tools()
        ])

        # 6. 构建完整上下文
        full_context = self.context_engine.build_context(
            user_input,
            relevant_memories=relevant_memories,
            rag_context=rag_context if rag_context.strip() else None,
            tools_info=tools_info
        )

        # 7. 生成响应（模拟LLM调用）
        response = self._simulate_llm_response(full_context)

        # 8. 检查并执行工具调用
        tool_calls = self._parse_tool_calls(response)
        tool_results = []

        if tool_calls:
            tool_results = self._execute_tool_calls(tool_calls)

            # 更新响应，包含工具执行结果
            tool_outputs = []
            for tool_call in tool_results:
                if tool_call.result:
                    tool_outputs.append(f"工具 {tool_call.name} 执行结果: {tool_call.result}")
                elif tool_call.error:
                    tool_outputs.append(f"工具 {tool_call.name} 执行错误: {tool_call.error}")

            if tool_outputs:
                # 移除原始的工具调用标记
                cleaned_response = re.sub(r'\[tool:[^\]]+\]', '', response).strip()
                response = cleaned_response + "\n\n" + "\n".join(tool_outputs)

        # 9. 记录助手响应
        assistant_message = Message(role="assistant", content=response)
        self.conversation_history.append(assistant_message)
        self.context_engine.add_message(assistant_message)

        # 10. 记录到工作记忆
        self.memory_manager.add_memory(
            f"我回复: {response}",
            "working",
            importance=0.4
        )

        # 11. 定期整合记忆
        if len(self.conversation_history) % 10 == 0:
            self.memory_manager.consolidate_memories()

        return response

    def get_stats(self) -> Dict[str, Any]:
        """获取Agent统计信息"""
        return {
            "name": self.name,
            "session_id": self.current_session_id,
            "conversation_length": len(self.conversation_history),
            "memory_stats": self.memory_manager.get_memory_stats(),
            "rag_documents": len(self.rag_system.documents),
            "available_tools": len(self.tool_registry.tools)
        }

    def export_conversation(self) -> List[Dict[str, Any]]:
        """导出对话历史"""
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "metadata": msg.metadata
            }
            for msg in self.conversation_history
        ]


def demo_advanced_agent():
    """演示高级Agent系统"""
    print("=" * 60)
    print("🤖 高级LLM Agent系统演示")
    print("=" * 60)

    # 创建高级Agent
    agent = AdvancedAgent(
        name="高级助手",
        system_prompt="你是一个智能助手，拥有记忆、知识库和工具使用能力。你可以帮助用户解决各种问题。"
    )

    # 添加一些知识到RAG系统
    knowledge_base = [
        "人工智能（AI）是由人类开发的智能系统，能够执行通常需要人类智能的任务。",
        "机器学习是人工智能的一个分支，使用统计技术使计算机能够在没有明确编程的情况下学习。",
        "深度学习是机器学习的一个子集，它模仿人脑的神经网络结构。",
        "Python是一种高级编程语言，广泛用于数据科学、机器学习和人工智能开发。",
        "大语言模型（LLM）是一种基于深度学习的人工智能模型，能够理解和生成人类语言。"
    ]

    for i, knowledge in enumerate(knowledge_base):
        agent.add_knowledge(knowledge, f"kb_{i+1}", {"topic": "AI知识"})

    print(f"\n📊 Agent统计信息: {agent.get_stats()}")
    print("\n📝 输入 'help' 查看帮助，输入 'quit' 退出")
    print("-" * 60)

    # 交互循环
    while True:
        try:
            user_input = input("\n👤 你: ").strip()

            if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                print("\n👋 再见！感谢使用高级Agent系统！")
                break

            if not user_input:
                continue

            if user_input.lower() == 'help':
                print("""
💡 帮助信息:
- 与 Agent 进行自然对话
- '统计' - 查看 Agent 统计信息
- '记忆' - 查看记忆统计
- '对话历史' - 导出对话历史
- 'quit' - 退出程序

🔧 可用功能:
- 数学计算: '计算 10 + 5'
- 时间查询: '现在几点?'
- 笔记管理: '创建笔记 "学习Python"', '查看笔记'
- 知识问答: '什么是人工智能?'
                """)
                continue

            if user_input == '统计':
                stats = agent.get_stats()
                print(f"\n📊 Agent统计信息:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                continue

            if user_input == '记忆':
                memory_stats = agent.memory_manager.get_memory_stats()
                print(f"\n🧠 记忆统计:")
                for key, value in memory_stats.items():
                    print(f"  {key}: {value}")
                continue

            if user_input == '对话历史':
                history = agent.export_conversation()
                print(f"\n📋 对话历史 ({len(history)} 条消息):")
                for msg in history[-5:]:  # 显示最近5条
                    role_icon = "👤" if msg['role'] == 'user' else "🤖"
                    print(f"  {role_icon} {msg['role']}: {msg['content'][:100]}...")
                continue

            # 处理正常消息
            response = agent.process_message(user_input)
            print(f"\n🤖 {agent.name}: {response}")

        except KeyboardInterrupt:
            print("\n\n👋 程序被中断，再见！")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")


def demo_components():
    """演示各个组件"""
    print("\n" + "=" * 50)
    print("🧠 记忆系统演示")
    print("=" * 50)

    memory_manager = MemoryManager()

    # 添加不同类型的记忆
    print("\n正在添加记忆...")

    # 工作记忆
    memory_manager.add_memory("用户问了关于Python的问题", "working", importance=0.6)
    memory_manager.add_memory("正在学习机器学习", "working", importance=0.8)

    # 情节记忆
    memory_manager.add_memory("今天用户第一次使用系统", "episodic", importance=0.7)
    memory_manager.add_memory("用户对AI非常感兴趣", "episodic", importance=0.9)

    # 语义记忆
    memory_manager.add_memory("Python是一种编程语言", "semantic", importance=0.9)
    memory_manager.add_memory("机器学习是AI的一个分支", "semantic", importance=0.8)

    # 显示记忆统计
    stats = memory_manager.get_memory_stats()
    print(f"记忆统计: {stats}")

    # 测试记忆检索
    print("\n正在检索相关记忆...")

    queries = ["Python编程", "机器学习", "用户兴趣"]

    for query in queries:
        print(f"\n查询: '{query}'")
        memories = memory_manager.retrieve_memories(query, limit=2)
        for memory in memories:
            print(f"  - [{memory.memory_type}] {memory.content} (重要性: {memory.importance})")

    # RAG系统演示
    print("\n" + "=" * 50)
    print("📚 RAG检索增强生成系统演示")
    print("=" * 50)

    rag = RAGSystem()

    # 添加文档
    documents = [
        ("python_basics", "Python是一种解释型、面向对象的高级编程语言。"),
        ("ml_intro", "机器学习是人工智能的一个分支，使用统计技术使计算机能够从数据中学习。"),
        ("llm_overview", "大语言模型（LLM）是一种基于深度学习的AI模型。")
    ]

    print("\n正在添加文档到RAG系统...")
    for doc_id, content in documents:
        rag.add_document(doc_id, content)
        print(f"  已添加: {doc_id}")

    # 测试检索
    print("\n正在测试检索功能...")
    query = "Python编程语言"
    results = rag.retrieve(query, top_k=2)

    for i, result in enumerate(results, 1):
        print(f"  {i}. 文档: {result['id']} (相似度: {result['similarity']:.3f})")
        print(f"     内容: {result['content']}")


if __name__ == "__main__":
    print("🚀 启动高级LLM Agent系统...")

    # 选择运行模式
    print("\n请选择运行模式:")
    print("1. 交互式高级Agent演示")
    print("2. 系统组件演示")

    try:
        choice = input("\n请输入选项 (1 或 2): ").strip()

        if choice == "1":
            demo_advanced_agent()
        elif choice == "2":
            demo_components()
        else:
            print("无效选项，默认运行交互式演示")
            demo_advanced_agent()

    except KeyboardInterrupt:
        print("\n\n👋 程序被中断，再见！")
