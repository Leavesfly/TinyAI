# TinyAI自进化Agent模块

## 简介

这是一个基于Java实现的自进化自学习Agent系统，参考Python版本的设计理念，具备经验记忆、策略优化、反思改进、知识图谱构建等核心能力。

## 快速开始

### 运行演示程序
```bash
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.agent.evol.EvolDemo"
```

### 运行测试
```bash
mvn test
```

### 快速演示
```java
SelfEvolvingAgent agent = new SelfEvolvingAgent("我的Agent");
TaskResult result = agent.processTask("搜索信息", null);
System.out.println("执行结果: " + result.isSuccess());
```

## 核心特性

- 🧠 **自学习能力**：从每次任务执行中学习和改进
- 🔄 **自适应能力**：根据环境动态调整策略  
- 🚀 **自进化能力**：持续优化策略和扩展能力
- 📊 **知识图谱**：构建概念关系网络
- 🤔 **反思机制**：深度分析成功失败原因

## 技术亮点

- ✅ 纯Java实现，减少第三方依赖
- ✅ 模块化设计，易于扩展维护
- ✅ 全面测试覆盖
- ✅ 详细文档说明

## 模块结构

```
src/main/java/io/leavesfly/tinyai/agent/evol/
├── Experience.java         # 经验记录
├── Strategy.java          # 策略管理
├── KnowledgeGraph.java    # 知识图谱
├── ReflectionModule.java  # 反思模块
├── SelfEvolvingAgent.java # 核心引擎
└── EvolDemo.java         # 演示程序
```

## 详细文档

请参阅 [详细实现总结](doc/README.md) 了解完整的设计思路和技术细节。

---
作者：山泽