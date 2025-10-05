# TinyAI Agent Context 模块

## 概述

TinyAI Agent Context 模块是一个完整的Java版本高级LLM Agent系统，参考Python实现`agent_mem.py`创建。该模块实现了记忆管理、RAG（检索增强生成）、工具调用和上下文工程等核心功能。

## 功能特性

### 🧠 记忆管理系统 (MemoryManager)
- **工作记忆**: 容量受限的短期记忆（默认10条）
- **情节记忆**: 对话和事件的长期存储
- **语义记忆**: 知识和概念的结构化存储
- **记忆整合**: 重要记忆自动从工作记忆转移到长期记忆
- **SQLite数据库**: 持久化存储记忆数据

### 📚 RAG检索增强生成系统 (RAGSystem)
- **文档管理**: 添加、更新、删除知识文档
- **语义检索**: 基于TF-IDF的文本相似度计算
- **上下文生成**: 根据查询自动构建相关上下文
- **向量化**: 文本嵌入和相似度匹配

### 🔧 工具注册表 (ToolRegistry)
- **工具注册**: 动态注册和管理工具函数
- **工具调用**: 安全的工具执行和结果管理
- **默认工具**: 内置计算器、时间、笔记等基础工具
- **错误处理**: 完善的异常捕获和错误报告

### 🎯 上下文工程引擎 (ContextEngine)
- **上下文构建**: 整合系统指令、记忆、RAG和工具信息
- **对话历史管理**: 智能压缩和管理对话记录
- **长度控制**: 自动截断超长上下文
- **优先级排序**: 重要信息优先保留

### 🤖 高级Agent (AdvancedAgent)
- **统一接口**: 整合所有子系统的主入口
- **对话处理**: 完整的消息处理流程
- **状态管理**: 会话状态和统计信息
- **配置灵活**: 可自定义系统提示和参数

## 核心类结构

```
io.leavesfly.tinyai.agent/
├── Memory.java              # 记忆单元模型
├── Message.java             # 消息结构
├── ToolCall.java            # 工具调用结构
├── Document.java            # 文档模型
├── RetrievalResult.java     # 检索结果
├── MemoryManager.java       # 记忆管理器
├── SimpleEmbedding.java     # 文本嵌入系统
├── RAGSystem.java           # RAG检索系统
├── ToolRegistry.java        # 工具注册表
├── ContextEngine.java       # 上下文引擎
├── AdvancedAgent.java       # 主Agent类
├── AgentHelper.java         # 辅助工具类
└── AgentDemo.java           # 演示示例
```

## 依赖要求

### 必需依赖
- **Java 8+**: 基础运行环境
- **SQLite JDBC**: 数据库连接驱动
  ```xml
  <dependency>
      <groupId>org.xerial</groupId>
      <artifactId>sqlite-jdbc</artifactId>
      <version>3.41.2.2</version>
  </dependency>
  ```

### 可选依赖
- **JUnit**: 单元测试框架
- **JFreeChart**: 图表绘制（如需要）

## 快速开始

### 1. 基本使用

```java
// 创建Agent
AdvancedAgent agent = new AdvancedAgent(
    "我的助手",
    "你是一个智能助手，能够帮助用户解决各种问题。"
);

// 添加知识
agent.addKnowledge("Java是一种面向对象的编程语言", "java_info");

// 处理对话
String response = agent.processMessage("什么是Java？");
System.out.println(response);
```

### 2. 工具使用

```java
// 使用内置计算器
Map<String, Object> args = new HashMap<>();
args.put("operation", "add");
args.put("a", 10);
args.put("b", 5);

ToolCall result = agent.getToolRegistry().callTool("calculator", args);
System.out.println("计算结果: " + result.getResult());
```

### 3. 记忆检索

```java
// 检索相关记忆
MemoryManager memoryManager = agent.getMemoryManager();
List<Memory> memories = memoryManager.retrieveMemories("编程", 3);

for (Memory memory : memories) {
    System.out.println("记忆: " + memory.getContent());
}
```

### 4. RAG检索

```java
// RAG文档检索
RAGSystem ragSystem = agent.getRagSystem();
List<RetrievalResult> results = ragSystem.retrieve("人工智能", 2);

for (RetrievalResult result : results) {
    System.out.println("文档: " + result.getDocument().getContent());
    System.out.println("相似度: " + result.getSimilarity());
}
```

## 运行演示

### 命令行演示
```bash
export JAVA_HOME=/Library/Java/JavaVirtualMachines/jdk-17.jdk/Contents/Home
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.agent.AgentDemo" -pl tinyai-agent
```

### 单元测试
```bash
export JAVA_HOME=/Library/Java/JavaVirtualMachines/jdk-17.jdk/Contents/Home
mvn test -pl tinyai-agent
```

## 设计特点

### 🔄 最小依赖
- 减少第三方库依赖，提高兼容性
- 使用Java标准库实现核心功能
- 仅SQLite作为必需外部依赖

### 🧩 模块化设计
- 每个组件职责单一，可独立使用
- 清晰的接口定义和依赖关系
- 易于扩展和定制

### 💾 数据持久化
- SQLite数据库存储记忆数据
- 支持内存模式和文件模式
- 自动数据库初始化和管理

### 🔧 工具生态
- 灵活的工具注册机制
- 内置常用工具（计算器、时间、笔记）
- 支持自定义工具扩展

### 🎛️ 可配置性
- 可调节的记忆容量和上下文长度
- 灵活的系统提示配置
- 自定义嵌入维度和相似度阈值

## 性能特征

- **内存使用**: 轻量级设计，内存占用较小
- **响应速度**: 本地计算，无网络延迟
- **扩展性**: 支持大量文档和记忆存储
- **稳定性**: 异常处理完善，系统稳定可靠

## 已知限制

1. **文本嵌入**: 使用简化的TF-IDF实现，精度有限
2. **记忆检索**: 基于关键词匹配，语义理解有限
3. **LLM模拟**: 当前为简单模拟，需要集成真实LLM
4. **多语言支持**: 主要针对中文优化

## 扩展方向

### 🚀 功能增强
- 集成真实LLM API（OpenAI、Claude等）
- 增强文本嵌入算法（Word2Vec、BERT）
- 添加更多内置工具和功能
- 支持多模态输入（图像、音频）

### 🔧 技术改进
- 优化记忆检索算法
- 提升RAG检索精度
- 增加并发处理能力
- 完善错误处理机制

### 🌐 生态集成
- Spring Boot集成
- Web API接口
- 分布式部署支持
- 云原生架构适配

## 贡献指南

欢迎提交Issue和Pull Request来改进这个项目：

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

## 许可证

本项目采用开源许可证，具体请参考项目根目录的LICENSE文件。

## 作者

山泽 - TinyAI Agent模块实现

---

*这个模块是TinyAI项目的一部分，旨在提供一个完整的Java版本LLM Agent解决方案。*