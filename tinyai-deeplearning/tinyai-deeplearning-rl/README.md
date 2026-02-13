# TinyAI Reinforcement Learning 强化学习模块 (tinyai-deeplearning-rl)

## 模块概述

`tinyai-deeplearning-rl` 是 TinyAI 深度学习框架的强化学习核心模块，提供了完整的强化学习算法实现和环境管理功能。本模块实现了从经典的多臂老虎机问题到现代深度强化学习算法的全套解决方案，是构建智能决策系统的核心组件。

## 核心架构

### 设计理念

本模块采用标准的强化学习架构设计，遵循 OpenAI Gym 接口规范，通过智能体-环境交互框架构建完整的强化学习系统：

- **Agent（智能体）**：决策制定者，负责选择动作和学习策略
- **Environment（环境）**：交互对象，提供状态转移和奖励信号
- **Policy（策略）**：动作选择机制，平衡探索与利用
- **Experience（经验）**：交互记录，支持经验回放学习
- **ReplayBuffer（经验缓冲区）**：经验存储和采样管理

```mermaid
graph TB
    subgraph "强化学习核心架构"
        Agent[Agent 智能体]
        Environment[Environment 环境]
        Policy[Policy 策略]
        Experience[Experience 经验]
        ReplayBuffer[ReplayBuffer 经验缓冲区]
    end
    
    subgraph "智能体实现"
        DQNAgent[DQNAgent 深度Q网络]
        REINFORCEAgent[REINFORCEAgent 策略梯度]
        BanditAgent[BanditAgent 多臂老虎机]
        EpsilonGreedyBandit[EpsilonGreedyBanditAgent ε-贪心老虎机]
        UCBBandit[UCBBanditAgent UCB老虎机]
        ThompsonBandit[ThompsonSamplingBanditAgent 汤普森采样]
    end
    
    subgraph "环境实现"
        CartPole[CartPoleEnvironment 倒立摆]
        GridWorld[GridWorldEnvironment 网格世界]
        MultiArmedBandit[MultiArmedBanditEnvironment 多臂老虎机]
    end
    
    subgraph "策略实现"
        EpsilonGreedy[EpsilonGreedyPolicy ε-贪心策略]
    end
    
    Agent --> Policy
    Agent --> ReplayBuffer
    Agent --> Experience
    Environment --> Experience
    
    Agent <--> Environment
    
    DQNAgent --> Agent
    REINFORCEAgent --> Agent
    BanditAgent --> Agent
    EpsilonGreedyBandit --> BanditAgent
    UCBBandit --> BanditAgent
    ThompsonBandit --> BanditAgent
    
    CartPole --> Environment
    GridWorld --> Environment
    MultiArmedBandit --> Environment
    
    EpsilonGreedy --> Policy
```

### 核心组件

#### 1. 基础抽象类
- [`Agent`](src/main/java/io/leavesfly/tinyai/rl/Agent.java) - 智能体抽象基类
- [`Environment`](src/main/java/io/leavesfly/tinyai/rl/Environment.java) - 环境抽象基类
- [`Policy`](src/main/java/io/leavesfly/tinyai/rl/Policy.java) - 策略抽象基类
- [`Experience`](src/main/java/io/leavesfly/tinyai/rl/Experience.java) - 经验数据结构
- [`ReplayBuffer`](src/main/java/io/leavesfly/tinyai/rl/ReplayBuffer.java) - 经验回放缓冲区

## 功能特性

### 🤖 多样化智能体算法

#### 深度强化学习智能体 (agent)

##### DQN 深度Q网络
[`DQNAgent`](src/main/java/io/leavesfly/tinyai/rl/agent/DQNAgent.java) - 深度Q网络算法实现

**核心特性：**
- 使用神经网络逼近Q函数
- 经验回放机制提高数据利用率  
- 目标网络稳定训练过程
- ε-贪婪策略平衡探索与利用

```java
// 创建DQN智能体
DQNAgent dqnAgent = new DQNAgent(
    "CartPole-DQN",     // 智能体名称
    4,                  // 状态空间维度
    2,                  // 动作空间维度  
    new int[]{128, 128}, // 隐藏层尺寸
    0.001f,             // 学习率
    1.0f,               // 初始探索率
    0.99f,              // 折扣因子
    32,                 // 批次大小
    10000,              // 经验缓冲区大小
    100                 // 目标网络更新频率
);
```

##### REINFORCE 策略梯度
[`REINFORCEAgent`](src/main/java/io/leavesfly/tinyai/rl/agent/REINFORCEAgent.java) - 策略梯度算法实现

**核心特性：**
- 直接优化策略函数
- 支持连续和离散动作空间
- 基线函数减少方差
- 蒙特卡洛采样估计梯度

#### 多臂老虎机智能体

##### 基础老虎机智能体
[`BanditAgent`](src/main/java/io/leavesfly/tinyai/rl/agent/BanditAgent.java) - 多臂老虎机基类

##### ε-贪心策略
[`EpsilonGreedyBanditAgent`](src/main/java/io/leavesfly/tinyai/rl/agent/EpsilonGreedyBanditAgent.java) - ε-贪心老虎机

**算法原理：**
- 以概率 ε 随机探索
- 以概率 (1-ε) 选择当前最优臂
- 简单有效的探索-利用平衡策略

##### UCB 上置信区间
[`UCBBanditAgent`](src/main/java/io/leavesfly/tinyai/rl/agent/UCBBanditAgent.java) - UCB算法实现

**算法原理：**
- 基于上置信区间的选择策略
- 考虑均值估计和不确定性
- 理论上有最优的遗憾界限

##### 汤普森采样
[`ThompsonSamplingBanditAgent`](src/main/java/io/leavesfly/tinyai/rl/agent/ThompsonSamplingBanditAgent.java) - 贝叶斯采样算法

**算法原理：**
- 基于贝叶斯推理的采样策略
- 维护每个臂的后验分布
- 根据后验分布采样进行决策

### 🌍 多样化环境实现

#### 经典控制环境

##### CartPole 倒立摆环境
[`CartPoleEnvironment`](src/main/java/io/leavesfly/tinyai/rl/environment/CartPoleEnvironment.java) - 经典控制问题

**环境特性：**
- 4维连续状态空间（位置、速度、角度、角速度）
- 2维离散动作空间（左推、右推）
- 目标：保持杆子平衡尽可能长时间
- 适合测试深度强化学习算法

```java
// 创建CartPole环境
CartPoleEnvironment env = new CartPoleEnvironment();
Variable state = env.reset();

// 环境交互循环
while (!env.isDone()) {
    Variable action = agent.selectAction(state);
    Environment.StepResult result = env.step(action);
    
    agent.learn(new Experience(state, action, result.getNextState(), 
                              result.getReward(), result.isDone()));
    state = result.getNextState();
}
```

##### GridWorld 网格世界环境
[`GridWorldEnvironment`](src/main/java/io/leavesfly/tinyai/rl/environment/GridWorldEnvironment.java) - 离散状态空间环境

**环境特性：**
- 离散网格状态空间
- 4方向移动动作（上、下、左、右）
- 可配置奖励和障碍物
- 适合测试基础强化学习算法

##### MultiArmedBandit 多臂老虎机环境
[`MultiArmedBanditEnvironment`](src/main/java/io/leavesfly/tinyai/rl/environment/MultiArmedBanditEnvironment.java) - 经典决策问题

**环境特性：**
- 多个老虎机臂（动作选择）
- 每个臂有不同的奖励分布
- 探索-利用权衡的典型场景
- 适合测试老虎机算法

### 🎯 策略机制 (policy)

#### ε-贪心策略
[`EpsilonGreedyPolicy`](src/main/java/io/leavesfly/tinyai/rl/policy/EpsilonGreedyPolicy.java) - 经典探索策略

**策略特性：**
- 可配置的探索率 ε
- 自动探索率衰减
- 支持不同的衰减策略
- 简单高效的实现

```java
// 创建ε-贪心策略
EpsilonGreedyPolicy policy = new EpsilonGreedyPolicy(
    stateDim, actionDim, 0.1f,  // 状态维度、动作维度、探索率
    state -> model.forward(state) // Q值函数
);

// 选择动作
Variable action = policy.selectAction(currentState);
```

### 💾 经验管理系统

#### 经验回放缓冲区
[`ReplayBuffer`](src/main/java/io/leavesfly/tinyai/rl/ReplayBuffer.java) - 高效的经验存储和采样

**核心功能：**
- 固定大小的循环缓冲区
- 随机采样防止数据相关性
- 高效的内存管理
- 灵活的采样策略

```java
// 创建经验缓冲区
ReplayBuffer buffer = new ReplayBuffer(10000);

// 存储经验
buffer.push(experience);

// 批量采样学习
if (buffer.canSample(batchSize)) {
    Experience[] batch = buffer.sample(batchSize);
    agent.learnBatch(batch);
}
```

#### 经验数据结构
[`Experience`](src/main/java/io/leavesfly/tinyai/rl/Experience.java) - 标准化的经验表示

**数据字段：**
- 状态 (State)
- 动作 (Action) 
- 下一状态 (Next State)
- 奖励 (Reward)
- 是否结束 (Done)

## 智能体-环境交互模式

### 标准交互流程

```mermaid
sequenceDiagram
    participant Env as 环境
    participant Agent as 智能体
    participant Buffer as 经验缓冲区
    participant Network as 神经网络
    
    loop 训练循环
        Env->>Agent : 当前状态
        Agent->>Agent : 选择动作 (ε-贪心)
        Agent->>Env : 执行动作
        Env->>Agent : 下一状态, 奖励, 是否结束
        Agent->>Buffer : 存储经验
        
        alt 经验足够
            Buffer->>Agent : 采样批次
            Agent->>Network : 计算目标Q值
            Agent->>Network : 计算当前Q值
            Network->>Agent : 计算损失
            Agent->>Network : 反向传播更新
            
            alt 达到更新频率
                Network->>Network : 更新目标网络
            end
        end
    end
```

## 技术依赖

本模块依赖以下 TinyAI 核心模块：

- `tinyai-dl-ml` - 机器学习核心模块，提供模型训练和优化支持

外部依赖：
- `jfreechart` - 图表可视化库，用于训练过程监控
- `junit` - 单元测试框架

## 📚 演示代码 (Demo)

本模块提供了一系列精心设计的演示程序,帮助您快速上手强化学习:

### 演示代码清单

#### 基础入门演示

| 演示代码 | 难度 | 时长 | 说明 |
|---------|------|------|------|
| **QuickStartDemo** | ⭐ | 3分钟 | 最简单的入门示例,3分钟了解强化学习 |
| **BasicConceptsDemo** | ⭐⭐ | 10分钟 | 详解状态、动作、奖励、策略、价值函数 |
| **BanditAlgorithmsDemo** | ⭐⭐ | 15分钟 | ε-贪心、UCB、汤普森采样算法对比 |
| **DQNCartPoleDemo** | ⭐⭐⭐ | 20分钟 | DQN完整训练流程,解决CartPole问题 |
| **CustomDevelopmentDemo** | ⭐⭐⭐⭐ | 30分钟 | 自定义环境和智能体,扩展框架 |

#### 🎓 教育性增强演示 (新增)

| 演示代码 | 难度 | 时长 | 说明 |
|---------|------|------|------|
| **AlgorithmVisualizationDemo** | ⭐⭐ | 15分钟 | 可视化展示DQN内部机制,Q值变化、损失曲线 |
| **InteractiveLearningDemo** | ⭐⭐ | 20分钟 | 交互式实验,调整参数观察学习效果 |
| **AlgorithmComparisonDemo** | ⭐⭐⭐ | 15分钟 | 并排对比多种算法,累积遗憾分析 |
| **StepByStepDebugDemo** | ⭐⭐⭐ | 25分钟 | 逐步调试DQN算法,理解每一步计算 |
| **LearningAnimationDemo** | ⭐⭐ | 10分钟 | 动画展示智能体学习过程 |

### 快速开始

```bash
# 1. 快速入门 (推荐第一个运行)
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.rl.demo.QuickStartDemo" \
  -pl tinyai-deeplearning-rl

# 2. 核心概念
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.rl.demo.BasicConceptsDemo" \
  -pl tinyai-deeplearning-rl

# 3. 算法对比
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.rl.demo.BanditAlgorithmsDemo" \
  -pl tinyai-deeplearning-rl

# 4. DQN实战
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.rl.demo.DQNCartPoleDemo" \
  -pl tinyai-deeplearning-rl

# 5. 自定义开发
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.rl.demo.CustomDevelopmentDemo" \
  -pl tinyai-deeplearning-rl

# 🎓 教育性增强演示
# 6. 算法可视化 (展示DQN内部机制)
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.rl.demo.AlgorithmVisualizationDemo" \
  -pl tinyai-deeplearning-rl

# 7. 交互式学习 (调整参数观察效果)
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.rl.demo.InteractiveLearningDemo" \
  -pl tinyai-deeplearning-rl

# 8. 算法对比分析 (并排对比多种算法)
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.rl.demo.AlgorithmComparisonDemo" \
  -pl tinyai-deeplearning-rl

# 9. 逐步调试 (理解每一步计算)
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.rl.demo.StepByStepDebugDemo" \
  -pl tinyai-deeplearning-rl

# 10. 学习动画 (动画展示学习过程)
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.rl.demo.LearningAnimationDemo" \
  -pl tinyai-deeplearning-rl
```

### 学习路径

**零基础用户 (约30分钟):**
1. QuickStartDemo → 理解强化学习基本流程
2. BasicConceptsDemo → 掌握核心概念
3. BanditAlgorithmsDemo → 学习简单算法
4. DQNCartPoleDemo → 尝试深度强化学习

**有基础用户 (约1小时):**
1. BanditAlgorithmsDemo → 算法对比分析
2. DQNCartPoleDemo → DQN深入学习
3. CustomDevelopmentDemo → 自定义开发
4. 参考 `tinyai-deeplearning-case` 中的完整案例

**深入学习用户 (约2小时):**
1. AlgorithmVisualizationDemo → 可视化理解DQN内部机制
2. InteractiveLearningDemo → 交互式实验探索参数影响
3. AlgorithmComparisonDemo → 系统对比多种算法
4. StepByStepDebugDemo → 深入理解算法每一步计算
5. LearningAnimationDemo → 直观感受学习过程

**开发者:**
- 直接查看 CustomDevelopmentDemo 了解扩展方法
- 参考 `src/main/java/io/leavesfly/tinyai/rl/demo/README.java` 获取完整文档

---

## 使用示例

### DQN算法完整示例

```java
import io.leavesfly.tinyai.rl.agent.DQNAgent;
import io.leavesfly.tinyai.rl.environment.CartPoleEnvironment;
import io.leavesfly.tinyai.rl.Experience;

// 1. 创建环境
CartPoleEnvironment env = new CartPoleEnvironment();

// 2. 创建DQN智能体
DQNAgent agent = new DQNAgent(
    "CartPole-DQN",           // 名称
    env.getStateDim(),        // 状态维度 (4)
    env.getActionDim(),       // 动作维度 (2)
    new int[]{128, 128},      // 隐藏层
    0.001f,                   // 学习率
    1.0f,                     // 初始探索率
    0.99f,                    // 折扣因子
    32,                       // 批次大小
    10000,                    // 缓冲区大小
    100                       // 目标网络更新频率
);

// 3. 训练循环
int episodes = 1000;
for (int episode = 0; episode < episodes; episode++) {
    Variable state = env.reset();
    float episodeReward = 0f;
    
    while (!env.isDone()) {
        // 选择动作
        Variable action = agent.selectAction(state);
        
        // 执行动作
        Environment.StepResult result = env.step(action);
        
        // 创建经验
        Experience experience = new Experience(
            state, action, result.getNextState(),
            result.getReward(), result.isDone()
        );
        
        // 学习
        agent.learn(experience);
        
        // 更新状态
        state = result.getNextState();
        episodeReward += result.getReward();
    }
    
    // 探索率衰减
    agent.decayEpsilon(0.995f);
    
    // 打印训练信息
    if (episode % 100 == 0) {
        System.out.printf("Episode %d: Reward = %.2f, Epsilon = %.3f%n",
                         episode, episodeReward, agent.getEpsilon());
    }
}
```

### 多臂老虎机算法示例

```java
import io.leavesfly.tinyai.rl.agent.UCBBanditAgent;
import io.leavesfly.tinyai.rl.environment.MultiArmedBanditEnvironment;

// 1. 创建多臂老虎机环境
MultiArmedBanditEnvironment env = new MultiArmedBanditEnvironment(
    new float[]{0.1f, 0.4f, 0.8f, 0.3f}  // 每个臂的奖励期望
);

// 2. 创建UCB智能体
UCBBanditAgent agent = new UCBBanditAgent(
    "UCB-Bandit",            // 名称
    env.getActionDim(),      // 臂的数量
    2.0f                     // UCB参数
);

// 3. 学习循环
int steps = 10000;
for (int step = 0; step < steps; step++) {
    Variable state = env.getCurrentState();
    Variable action = agent.selectAction(state);
    
    Environment.StepResult result = env.step(action);
    
    Experience experience = new Experience(
        state, action, result.getNextState(),
        result.getReward(), false
    );
    
    agent.learn(experience);
}

// 4. 输出结果
System.out.println("最终策略分布：");
for (int i = 0; i < env.getActionDim(); i++) {
    System.out.printf("臂 %d: 平均奖励 = %.3f%n", i, agent.getAverageReward(i));
}
```

### 自定义环境示例

```java
public class CustomEnvironment extends Environment {
    
    public CustomEnvironment() {
        super(customStateDim, customActionDim, maxSteps);
    }
    
    @Override
    public Variable reset() {
        // 重置环境到初始状态
        currentState = generateInitialState();
        done = false;
        currentStep = 0;
        return currentState;
    }
    
    @Override
    public StepResult step(Variable action) {
        // 状态转移逻辑
        Variable nextState = computeNextState(currentState, action);
        float reward = computeReward(currentState, action, nextState);
        boolean done = isTerminal(nextState) || currentStep >= maxSteps;
        
        currentState = nextState;
        currentStep++;
        this.done = done;
        
        return new StepResult(nextState, reward, done, getInfo());
    }
    
    @Override
    public Variable sampleAction() {
        // 随机动作采样
        return new Variable(NdArray.of(random.nextInt(actionDim)));
    }
    
    @Override
    public boolean isValidAction(Variable action) {
        // 动作有效性检查
        int actionValue = (int) action.getValue().getNumber().floatValue();
        return actionValue >= 0 && actionValue < actionDim;
    }
}
```

## 算法对比分析

### 多臂老虎机算法对比

| 算法 | 探索策略 | 理论保证 | 计算复杂度 | 适用场景 |
|------|----------|----------|------------|----------|
| **ε-贪心** | 固定概率探索 | 简单遗憾界 | O(1) | 在线学习、快速决策 |
| **UCB** | 置信区间探索 | 最优遗憾界 | O(1) | 理论最优、稳定环境 |
| **汤普森采样** | 贝叶斯采样 | 最优遗憾界 | O(k) | 贝叶斯优化、不确定环境 |

### 深度强化学习算法对比

| 算法 | 类型 | 状态空间 | 动作空间 | 样本效率 | 稳定性 |
|------|------|----------|----------|----------|---------|
| **DQN** | 值函数 | 连续 | 离散 | 中等 | 较好 |
| **REINFORCE** | 策略梯度 | 连续 | 连续/离散 | 较低 | 一般 |

## 测试覆盖

模块包含完整的单元测试，覆盖：
- 各种智能体算法的正确性测试
- 环境交互逻辑验证
- 经验缓冲区功能测试
- 策略机制有效性验证
- 端到端集成测试

运行测试：
```bash
cd /Users/yefei.yf/Qoder/TinyAI
export JAVA_HOME=/Library/Java/JavaVirtualMachines/jdk-17.jdk/Contents/Home
mvn test -pl tinyai-deeplearning-rl
```

## 模块特色

### 🏗️ 标准化设计
- 遵循 OpenAI Gym 接口规范
- 统一的智能体-环境交互模式
- 可扩展的算法实现框架

### 🧠 算法丰富性
- 从经典老虎机到现代深度强化学习
- 多种探索策略和学习算法
- 理论与实践相结合的实现

### ⚡ 高性能实现
- 高效的经验回放机制
- 优化的内存管理
- 支持批量学习和并行训练

### 🔧 易用性设计
- 简洁的API接口
- 丰富的预置环境和算法
- 详细的文档和示例

## 开发指南

### 添加新的智能体算法

```java
public class CustomAgent extends Agent {
    
    public CustomAgent(String name, int stateDim, int actionDim, 
                      float learningRate, float epsilon, float gamma) {
        super(name, stateDim, actionDim, learningRate, epsilon, gamma);
        // 初始化自定义参数
    }
    
    @Override
    public Variable selectAction(Variable state) {
        // 实现动作选择逻辑
        return customActionSelection(state);
    }
    
    @Override
    public void learn(Experience experience) {
        // 实现学习更新逻辑
        customLearningUpdate(experience);
    }
    
    @Override
    public void learnBatch(Experience[] experiences) {
        // 实现批量学习逻辑
        customBatchLearning(experiences);
    }
    
    @Override
    public void storeExperience(Experience experience) {
        // 实现经验存储逻辑
        customExperienceStorage(experience);
    }
}
```

### 添加新的环境

```java
public class NewEnvironment extends Environment {
    
    public NewEnvironment() {
        super(stateDim, actionDim, maxSteps);
        // 环境特定的初始化
    }
    
    @Override
    public Variable reset() {
        // 实现环境重置逻辑
        return initialState;
    }
    
    @Override
    public StepResult step(Variable action) {
        // 实现状态转移逻辑
        return new StepResult(nextState, reward, done, info);
    }
    
    @Override
    public Variable sampleAction() {
        // 实现随机动作采样
        return randomAction;
    }
    
    @Override
    public boolean isValidAction(Variable action) {
        // 实现动作有效性检查
        return isValid;
    }
}
```

## 性能优化建议

### 训练稳定性优化
- **经验回放**：使用足够大的缓冲区打破数据相关性
- **目标网络**：定期更新目标网络稳定训练
- **探索策略**：合理设置探索率和衰减策略
- **奖励设计**：设计合理的奖励函数引导学习

### 超参数调优
- **学习率**：从 0.001 开始，根据收敛情况调整
- **批次大小**：32-128，平衡计算效率和梯度稳定性
- **缓冲区大小**：10000-100000，根据内存限制选择
- **探索率**：从 1.0 衰减到 0.01，保持适度探索

## 版本信息

- **当前版本**: 1.0-SNAPSHOT
- **Java 版本**: 17+
- **构建工具**: Maven 3.6+
- **算法支持**: DQN、REINFORCE、多臂老虎机系列

## 相关模块

- [`tinyai-dl-ml`](../tinyai-dl-ml/README.md) - 机器学习核心模块
- [`tinyai-dl-case`](../tinyai-dl-case/README.md) - 应用示例模块
- [`tinyai-dl-nnet`](../tinyai-dl-nnet/README.md) - 神经网络层模块

---

**TinyAI Reinforcement Learning 模块** - 让智能决策变得简单、高效、可靠 🎯