# TinyAI Agent Pattern 模块

## 概述

TinyAI Agent Pattern 模块是基于Python版本pattern.py实现的Java版本，提供了多种常用的Agent模式实现。该模块尽量减少第三方库依赖，仅使用Java标准库和必要的测试框架。

## 主要特性

### 🤖 支持的Agent模式

1. **ReAct Agent (推理-行动模式)**
   - 交替进行推理(Reasoning)和行动(Acting)
   - 通过观察结果指导下一步行动
   - 适合需要工具调用的任务

2. **Reflect Agent (反思模式)**
   - 具有自我反思能力
   - 能够评估并改进自己的回答
   - 适合质量要求高的任务

3. **Planning Agent (规划模式)**
   - 先制定详细计划再执行
   - 将复杂任务分解为子任务
   - 适合复杂项目管理

4. **Collaborative Agent (协作模式)**
   - 多个专家Agent协同工作
   - 根据问题自动选择合适的专家
   - 支持结果验证和整合

### 🛠️ 核心组件

- **BaseAgent**: 所有Agent的抽象基类
- **AgentState**: Agent状态枚举
- **Step**: 步骤记录类
- **Action**: 动作结构类
- **SampleTools**: 示例工具集合

## 快速开始

### 1. 运行演示程序

```bash
cd /Users/yefei.yf/Qoder/TinyAI
export JAVA_HOME="/Library/Java/JavaVirtualMachines/jdk-17.jdk/Contents/Home"
mvn clean compile -pl tinyai-agent-pattern
mvn exec:java -pl tinyai-agent-pattern -Dexec.mainClass="io.leavesfly.tinyai.agent.pattern.AgentPatternMain"
```

### 2. 基本使用示例

#### ReAct Agent
```java
ReActAgent agent = new ReActAgent();
String result = agent.process("计算 25 * 4 + 10");
System.out.println(result);
System.out.println(agent.getStepsSummary());
```

#### Reflect Agent
```java
ReflectAgent agent = new ReflectAgent();
String result = agent.process("分析这段代码的优缺点");
System.out.println(result);
// 查看反思记录
agent.getReflections().forEach(System.out::println);
```

#### Planning Agent
```java
PlanningAgent agent = new PlanningAgent();
String result = agent.process("制定学习Java的计划");
System.out.println(result);
// 查看执行计划
agent.getPlan().forEach(task -> System.out.println(task.getDescription()));
```

#### Collaborative Agent
```java
CollaborativeAgent coordinator = new CollaborativeAgent();
coordinator.addSpecialist("calculator", new ReActAgent("计算专家"));
coordinator.addSpecialist("analyst", new ReflectAgent("分析专家"));

String result = coordinator.process("分析并计算项目成本");
System.out.println(result);
```

### 3. 自定义工具

```java
ReActAgent agent = new ReActAgent();

// 添加自定义工具
agent.addTool("weather", SampleTools.createWeatherTool(), "天气查询");
agent.addTool("translate", SampleTools.createTranslateTool(), "文本翻译");

String result = agent.process("查询北京天气");
```

## 项目结构

```
tinyai-agent-pattern/
├── src/main/java/io/leavesfly/tinyai/agent/pattern/
│   ├── AgentState.java          # Agent状态枚举
│   ├── Step.java                # 步骤记录类
│   ├── Action.java              # 动作结构类
│   ├── BaseAgent.java           # Agent抽象基类
│   ├── ReActAgent.java          # ReAct模式实现
│   ├── ReflectAgent.java        # Reflect模式实现
│   ├── PlanningAgent.java       # Planning模式实现
│   ├── CollaborativeAgent.java  # 协作模式实现
│   ├── SampleTools.java         # 示例工具集合
│   ├── AgentPatternDemo.java    # 演示类
│   └── AgentPatternMain.java    # 主程序
├── src/test/java/io/leavesfly/tinyai/agent/pattern/
│   └── AgentPatternTest.java    # 单元测试
├── doc/
│   └── pattern.py               # Python原版实现
└── pom.xml                      # Maven配置
```

## 运行测试

```bash
mvn test -pl tinyai-agent-pattern
```

## 特性对比

| Agent模式 | 适用场景 | 优势 | 劣势 |
|-----------|----------|------|------|
| ReAct | 数学计算、信息查询 | 逻辑清晰、可解释性强 | 可能陷入局部循环 |
| Reflect | 内容生成、质量要求高的任务 | 自我改进、质量控制 | 计算开销较大 |
| Planning | 项目管理、研究任务 | 任务分解能力强 | 规划开销大、不够灵活 |
| Collaborative | 复杂问题、多领域任务 | 专业化分工、互补优势 | 协调复杂、资源消耗大 |

## 高级模式介绍

该模块还介绍了以下高级Agent模式的概念：

- **Tree of Thoughts (ToT)**: 以树状结构探索多个思考路径
- **Chain of Thought (CoT)**: 逐步推理，通过中间步骤得出答案
- **Multi-Agent Debate**: 多Agent辩论讨论
- **Self-Consistency**: 生成多个推理路径，选择最一致的答案
- **AutoGPT Pattern**: 自主设定目标、制定计划、执行任务

## 依赖说明

该模块尽量减少第三方依赖：

- **核心功能**: 仅使用Java标准库
- **测试**: JUnit 4.x
- **构建**: Maven

## 注意事项

1. 所有Agent都是线程安全的，但不建议并发使用同一个实例
2. 工具函数应该是无副作用的纯函数
3. Agent的memory会随着使用逐渐增长，必要时需要清理
4. 复杂任务可能需要调整maxSteps参数

## 扩展开发

### 创建自定义Agent

```java
public class CustomAgent extends BaseAgent {
    
    public CustomAgent() {
        super("Custom Agent", 10);
        registerTools();
    }
    
    private void registerTools() {
        addTool("custom_tool", this::customTool, "自定义工具");
    }
    
    private Object customTool(Map<String, Object> args) {
        // 自定义工具实现
        return "自定义结果";
    }
    
    @Override
    public String process(String query) {
        // 自定义处理逻辑
        return "处理结果";
    }
}
```

### 创建自定义工具

```java
public static Function<Map<String, Object>, Object> createCustomTool() {
    return args -> {
        // 工具实现逻辑
        return "工具执行结果";
    };
}
```

## 作者

山泽

## 许可证

本项目遵循TinyAI项目的许可证。