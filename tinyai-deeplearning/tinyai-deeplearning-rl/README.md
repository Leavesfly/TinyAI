# TinyAI Reinforcement Learning 强化学习模块

> **教学导向的现代强化学习框架** - 清晰的架构设计 + 完整的算法实现 + 丰富的教学文档

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]() [![Java](https://img.shields.io/badge/Java-17%2B-orange)]() [![License](https://img.shields.io/badge/license-Apache%202.0-blue)]()

## 📖 模块简介

`tinyai-deeplearning-rl` 是 TinyAI 深度学习框架的强化学习核心模块,实现了从经典多臂老虎机到现代深度强化学习的完整算法体系。

### 🎯 设计目标

1. **教学导向**: 清晰的代码结构和详尽的注释,帮助学习者深入理解强化学习
2. **架构整洁**: 消除代码重复,通过继承和组合建立统一框架
3. **易于扩展**: 新增算法只需继承基类并实现2-3个核心方法
4. **工业实践**: 遵循OpenAI Gym规范,代码质量达到生产级别

### ✨ 核心特性

- ✅ **统一框架**: ValueBasedAgent和PolicyBasedAgent两大基类统一训练流程
- ✅ **代码精简**: 通过架构重构平均精简代码52%
- ✅ **组件复用**: 3个工具类消除重复代码(TrainingStatistics, QValueComputer, ModelUtil)
- ✅ **算法丰富**: 10个算法实现,覆盖Bandit、DQN、策略梯度三大类别
- ✅ **完整生态**: 3个经典环境 + 10个教学Demo + 完善的文档体系

---

## 🏗️ 核心架构

### 算法分类体系

本模块采用清晰的三层继承设计,通过**模板方法模式**统一训练流程:

```
Agent (抽象基类) ← MDP框架定义
 │
 ├── ValueBasedAgent (基于值函数) ← 统一经验回放+TD学习
 │    ├── DQNAgent              深度Q网络
 │    └── DoubleDQNAgent         双Q网络(解决过估计)
 │
 ├── PolicyBasedAgent (基于策略) ← 统一回合采样+策略梯度
 │    └── REINFORCEAgent         蒙特卡罗策略梯度
 │
 └── BanditAgent (多臂老虎机) ← 探索算法
      ├── EpsilonGreedyBanditAgent  ε-贪心
      ├── UCBBanditAgent            上置信区间
      └── ThompsonSamplingBanditAgent  汤普森采样
```

### 架构设计亮点

#### 1. 统一训练流程 - ValueBasedAgent

```java
public abstract class ValueBasedAgent extends Agent {
    @Override
    public void learn(Experience experience) {
        // 1. 存储经验到缓冲区
        storeExperience(experience);
        
        // 2. 检查是否有足够经验
        if (shouldLearn()) {
            // 3. 采样批次
            Experience[] batch = sampleBatch();
            
            // 4. 计算目标Q值 (子类实现差异)
            Variable targetQ = computeTargetQValues(batch);
            Variable currentQ = computeCurrentQValues(batch);
            
            // 5. 反向传播更新
            Variable loss = lossFunction.loss(targetQ, currentQ);
            model.clearGrads();
            loss.backward();
            optimizer.update();
            
            // 6. 定期同步目标网络
            if (shouldUpdateTarget()) updateTargetNetwork();
        }
    }
    
    // 子类只需实现这两个方法即可完成新算法
    protected abstract Variable computeTargetQValues(Experience[] experiences);
    protected abstract Variable computeCurrentQValues(Experience[] experiences);
}
```

**教学价值**: 
- 学习者清晰看到Value-Based算法的**统一框架**
- DQN vs DoubleDQN的差异仅在**目标Q值计算**上(约20行代码)
- 复用经验回放、目标网络同步、训练统计等通用逻辑

#### 2. 公共工具组件

| 工具类 | 职责 | 消除重复 |
|--------|------|----------|
| **TrainingStatistics** | 统一训练指标管理 | 消除各Agent重复的统计代码 |
| **QValueComputer** | Q值相关计算 | 贪婪选择、最大值提取、堆叠 |
| **ModelUtil** | 模型操作 | 网络创建、权重复制、软更新 |

**使用示例**:

```java
// 创建Q网络 - 一行代码搞定
Model qNetwork = ModelUtil.createQNetwork("DQN_Q", 4, 2, new int[]{128, 128});

// 权重复制 - 目标网络同步
ModelUtil.copyWeights(onlineNetwork, targetNetwork);

// Q值计算 - 自动处理Variable堆叠
Variable batchQValues = QValueComputer.stackVariables(qArray, batchSize);
```

#### 3. 代码精简效果

通过架构重构,显著提升代码质量:

| Agent类 | 重构前 | 重构后 | 精简率 | 关键改进 |
|---------|--------|--------|--------|----------|
| **DQNAgent** | 459行 | 158行 | **65.6%** | 继承ValueBasedAgent,使用工具类 |
| **DoubleDQNAgent** | 363行 | 177行 | **51.2%** | 仅重写目标Q值计算(20行差异) |
| **REINFORCEAgent** | 563行 | 340行 | **39.6%** | 继承PolicyBasedAgent,自动管理回合 |
| **平均精简** | - | - | **52.1%** | - |

**精简来源**:
- ❌ 删除重复的训练统计代码 → TrainingStatistics
- ❌ 删除重复的Q值计算方法 → QValueComputer  
- ❌ 删除重复的网络创建代码 → ModelUtil
- ❌ 删除重复的训练循环逻辑 → 继承基类统一流程

---

## 🚀 快速开始

### 环境要求

- Java 17+
- Maven 3.6+
- TinyAI DeepLearning Framework

### 安装

```bash
git clone https://github.com/your-repo/TinyAI.git
cd TinyAI
mvn clean install -DskipTests
```

### 5分钟上手 - DQN训练

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
for (int episode = 0; episode < 1000; episode++) {
    Variable state = env.reset();
    float episodeReward = 0f;
    
    while (!env.isDone()) {
        // 选择动作 → 执行 → 学习
        Variable action = agent.selectAction(state);
        Environment.StepResult result = env.step(action);
        
        agent.learn(new Experience(state, action, result.getNextState(),
                                  result.getReward(), result.isDone()));
        
        state = result.getNextState();
        episodeReward += result.getReward();
    }
    
    agent.decayEpsilon(0.995f);
    
    if (episode % 100 == 0) {
        System.out.printf("Episode %d: Reward = %.2f, Epsilon = %.3f%n",
                         episode, episodeReward, agent.getEpsilon());
    }
}
```

### DoubleDQN vs DQN - 只需改一个类名!

```java
// DQN → DoubleDQN: 完全相同的训练代码,更稳定的性能
DoubleDQNAgent agent = new DoubleDQNAgent(
    "CartPole-DoubleDQN", 4, 2, new int[]{128, 128},
    0.001f, 1.0f, 0.99f, 32, 10000, 100
);

// 后续训练代码完全一致!
// 提示: DoubleDQN通过解耦动作选择和评估,减少Q值过估计
```

---

## 📚 核心组件详解

### 1. Agent - 智能体基类

#### Agent抽象基类

定义强化学习智能体的**标准MDP接口**:

```java
public abstract class Agent {
    // 核心属性
    protected String name;          // 智能体名称
    protected int stateDim;         // 状态空间维度
    protected int actionDim;        // 动作空间维度
    protected Model model;          // 主网络
    protected float learningRate;   // 学习率
    protected float epsilon;        // 探索率
    protected float gamma;          // 折扣因子
    
    // 核心接口
    public abstract Variable selectAction(Variable state);    // 动作选择
    public abstract void learn(Experience experience);        // 单步学习
    public abstract void learnBatch(Experience[] experiences); // 批量学习
}
```

#### ValueBasedAgent - 基于值函数

**统一流程**: 经验回放 + TD学习 + 目标网络同步

**子类实现**:
- `DQNAgent`: 使用目标网络计算max Q值
- `DoubleDQNAgent`: 在线网络选择动作,目标网络评估Q值

**关键方法**:
```java
protected abstract Variable computeTargetQValues(Experience[] experiences);
protected abstract Variable computeCurrentQValues(Experience[] experiences);
```

#### PolicyBasedAgent - 基于策略

**统一流程**: 回合采样 + 蒙特卡罗回报 + 策略梯度

**子类实现**:
- `REINFORCEAgent`: 策略梯度 + 可选基线

**关键方法**:
```java
protected abstract Variable computePolicyLoss(List<Float> returns);
```

### 2. 算法实现详解

#### DQNAgent - 深度Q网络

**算法原理**:
```
目标Q值: y = r + γ * max_a' Q_target(s', a')
损失函数: L = (y - Q(s,a))^2
```

**关键创新**:
- 经验回放: 打破数据相关性
- 目标网络: 稳定训练过程
- ε-贪心: 平衡探索与利用

**代码示例** (完整实现仅158行):
```java
public class DQNAgent extends ValueBasedAgent {
    @Override
    protected Variable computeTargetQValues(Experience[] experiences) {
        for (Experience exp : experiences) {
            if (exp.isDone()) {
                targetArray[i] = new Variable(NdArray.of(exp.getReward()));
            } else {
                Variable nextQValues = targetModel.forward(exp.getNextState());
                Variable maxNextQ = QValueComputer.findMaxQValue(nextQValues);
                targetArray[i] = reward.add(maxNextQ.mul(gamma));
            }
        }
        return QValueComputer.stackVariables(targetArray, batchSize);
    }
}
```

#### DoubleDQNAgent - 双Q网络

**解决问题**: DQN存在Q值**过估计**问题

**算法创新**:
```
DQN问题: max操作同时负责选择和评估,倾向高估
  y = r + γ * max_a' Q_target(s', a')

DoubleDQN解决: 解耦选择和评估
  a_best = argmax_a' Q_online(s', a')  ← 在线网络选择
  y = r + γ * Q_target(s', a_best)      ← 目标网络评估
```

**与DQN对比**:
| 特性 | DQN | DoubleDQN |
|------|-----|-----------|
| 动作选择 | 目标网络 | **在线网络** |
| Q值评估 | 目标网络 | 目标网络 |
| 过估计问题 | 存在 | **缓解** |
| 代码差异 | - | **仅20行** |

#### REINFORCEAgent - 策略梯度

**算法原理**:
```
策略梯度定理: ∇J(θ) = E[∇log π(a|s) * G_t]
损失函数: L = -Σ log π(a_t|s_t) * G_t
蒙特卡罗回报: G_t = Σ_{k=0}^∞ γ^k * r_{t+k}
基线减方差: L = -Σ log π(a_t|s_t) * (G_t - V(s_t))
```

**关键特性**:
- 直接优化策略函数 π(a|s)
- 支持连续和离散动作空间
- 可选基线网络减少方差
- 适合回合式任务

**代码示例**:
```java
REINFORCEAgent agent = new REINFORCEAgent(
    "CartPole-REINFORCE",
    4, 2, new int[]{128, 64},
    0.001f,    // 学习率
    0.99f,     // 折扣因子
    true       // 使用基线
);
```

### 3. 多臂老虎机算法

#### EpsilonGreedyBanditAgent - ε-贪心

**算法原理**:
- 以概率 ε 随机探索
- 以概率 (1-ε) 选择当前最优臂
- 支持探索率动态衰减

**代码示例**:
```java
EpsilonGreedyBanditAgent agent = new EpsilonGreedyBanditAgent("Bandit", 10, 0.1f);

// 训练
for (int step = 0; step < 10000; step++) {
    int arm = agent.selectArm();
    float reward = environment.pull(arm);
    agent.learn(new Experience(null, new Variable(NdArray.of(arm)), reward, null, false));
}

// 动态调整探索率
agent.decayEpsilon(0.99f);
```

#### UCBBanditAgent - 上置信区间

**UCB公式**:
```
UCB(i) = Q(i) + c * sqrt(ln(t) / N(i))
       ↑       ↑
    期望奖励  不确定性
```

**优势**: 理论最优的遗憾界

#### ThompsonSamplingBanditAgent - 汤普森采样

**算法原理**: 基于贝叶斯推理的采样策略

---

## 🌍 环境实现

### CartPoleEnvironment - 倒立摆

**经典控制问题**:
- **状态空间**: 4维 [位置, 速度, 角度, 角速度]
- **动作空间**: 2维 [向左推, 向右推]
- **目标**: 保持杆子平衡尽可能长时间
- **物理模拟**: 基于经典力学方程

```java
CartPoleEnvironment env = new CartPoleEnvironment();
Variable state = env.reset();

while (!env.isDone()) {
    Variable action = agent.selectAction(state);
    Environment.StepResult result = env.step(action);
    // ...
}
```

### GridWorldEnvironment - 网格世界

**离散状态空间**:
- 网格移动环境
- 4方向动作(上下左右)
- 可配置奖励和障碍物

### MultiArmedBanditEnvironment - 多臂老虎机

**探索-利用权衡**:
- 多个臂,不同奖励分布
- 支持高斯分布奖励
- 记录累积遗憾值

---

## 🎓 教学Demo

本模块提供10个精心设计的教学Demo,帮助快速上手:

### 基础入门 (3个Demo)

| Demo | 难度 | 时长 | 说明 |
|------|------|------|------|
| **QuickStartDemo** | ⭐ | 3分钟 | 最简入门,3分钟了解RL |
| **BasicConceptsDemo** | ⭐⭐ | 10分钟 | 详解状态、动作、奖励、策略 |
| **BanditAlgorithmsDemo** | ⭐⭐ | 15分钟 | ε-贪心、UCB、汤普森采样对比 |

### 深度强化学习 (2个Demo)

| Demo | 难度 | 时长 | 说明 |
|------|------|------|------|
| **DQNCartPoleDemo** | ⭐⭐⭐ | 20分钟 | DQN完整训练流程 |
| **CustomDevelopmentDemo** | ⭐⭐⭐⭐ | 30分钟 | 自定义环境和智能体 |

### 教育性增强 (5个Demo)

| Demo | 难度 | 时长 | 说明 |
|------|------|------|------|
| **AlgorithmVisualizationDemo** | ⭐⭐ | 15分钟 | 可视化DQN内部机制 |
| **InteractiveLearningDemo** | ⭐⭐ | 20分钟 | 交互式调参实验 |
| **AlgorithmComparisonDemo** | ⭐⭐⭐ | 15分钟 | 并排对比多种算法 |
| **StepByStepDebugDemo** | ⭐⭐⭐ | 25分钟 | 逐步调试DQN算法 |
| **LearningAnimationDemo** | ⭐⭐ | 10分钟 | 动画展示学习过程 |

### 运行Demo

```bash
# 快速入门
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.rl.demo.QuickStartDemo" \
  -pl tinyai-deeplearning-rl

# DQN实战
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.rl.demo.DQNCartPoleDemo" \
  -pl tinyai-deeplearning-rl

# 算法对比
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.rl.demo.AlgorithmComparisonDemo" \
  -pl tinyai-deeplearning-rl
```

---

## 🔧 扩展开发

### 添加新算法 - 只需3步!

**示例**: 实现Dueling DQN

```java
// 步骤1: 继承ValueBasedAgent
public class DuelingDQNAgent extends ValueBasedAgent {
    
    public DuelingDQNAgent(...) {
        super(...);
        // 创建Dueling网络结构 (分离价值流和优势流)
        this.model = ModelUtil.createDuelingQNetwork(...);
        this.targetModel = ModelUtil.createDuelingQNetwork(...);
        ModelUtil.copyWeights(model, targetModel);
    }
    
    // 步骤2: 实现目标Q值计算 (可复用DQN或DoubleDQN的逻辑)
    @Override
    protected Variable computeTargetQValues(Experience[] experiences) {
        // 与DoubleDQN相同的实现
        return super.computeTargetQValues(experiences);
    }
    
    // 步骤3: 实现当前Q值计算 (Dueling架构自动合并V和A)
    @Override
    protected Variable computeCurrentQValues(Experience[] experiences) {
        // Dueling网络forward自动处理 Q(s,a) = V(s) + A(s,a) - mean(A)
        // ...
    }
}

// 完成! 自动继承经验回放、目标网络同步等所有功能
```

### 添加新环境

```java
public class MyEnvironment extends Environment {
    
    public MyEnvironment() {
        super(stateDim, actionDim, maxSteps);
    }
    
    @Override
    public Variable reset() {
        // 重置环境到初始状态
        return initialState;
    }
    
    @Override
    public StepResult step(Variable action) {
        // 状态转移逻辑
        return new StepResult(nextState, reward, done, info);
    }
    
    @Override
    public Variable sampleAction() {
        return new Variable(NdArray.of(random.nextInt(actionDim)));
    }
    
    @Override
    public boolean isValidAction(Variable action) {
        int a = action.getValue().getNumber().intValue();
        return a >= 0 && a < actionDim;
    }
}
```

---

## 📊 算法性能对比

### 多臂老虎机算法

| 算法 | 探索策略 | 理论保证 | 计算复杂度 | 适用场景 |
|------|----------|----------|------------|----------|
| **ε-贪心** | 固定概率探索 | 简单遗憾界 | O(1) | 在线学习、快速决策 |
| **UCB** | 置信区间探索 | 最优遗憾界 | O(1) | 理论最优、稳定环境 |
| **汤普森采样** | 贝叶斯采样 | 最优遗憾界 | O(k) | 贝叶斯优化、不确定环境 |

### 深度强化学习算法

| 算法 | 类型 | 状态空间 | 动作空间 | 样本效率 | 稳定性 | 代码行数 |
|------|------|----------|----------|----------|---------|----------|
| **DQN** | 值函数 | 连续 | 离散 | 中等 | 较好 | 158行 |
| **DoubleDQN** | 值函数 | 连续 | 离散 | 中等 | 更好 | 177行 |
| **REINFORCE** | 策略梯度 | 连续 | 连续/离散 | 较低 | 一般 | 340行 |

---

## 🧪 测试体系

### 测试覆盖

```bash
# 运行所有测试
mvn test -pl tinyai-deeplearning-rl

# 测试覆盖范围
# - 组件级: Experience, ReplayBuffer, Policy
# - 算法级: DQN, DoubleDQN, REINFORCE, Bandits
# - 集成级: Agent-Environment交互, 完整训练流程
```

### 测试示例

```java
@Test
public void testDQNTraining() {
    CartPoleEnvironment env = new CartPoleEnvironment();
    DQNAgent agent = new DQNAgent(...);
    
    // 训练100回合
    for (int episode = 0; episode < 100; episode++) {
        Variable state = env.reset();
        while (!env.isDone()) {
            Variable action = agent.selectAction(state);
            Environment.StepResult result = env.step(action);
            agent.learn(new Experience(...));
            state = result.getNextState();
        }
    }
    
    // 验证学习效果
    assertTrue(agent.getAverageReward() > initialReward);
}
```

---

## 🎯 超参数调优指南

| 参数 | 推荐范围 | 典型值 | 说明 |
|------|----------|--------|------|
| **学习率** | 0.0001-0.01 | 0.001 | 从小开始,根据收敛情况调整 |
| **批次大小** | 32-128 | 64 | 平衡计算效率和梯度稳定性 |
| **缓冲区大小** | 10000-100000 | 10000 | 根据内存限制和任务复杂度 |
| **初始探索率** | 0.9-1.0 | 1.0 | 充分探索 |
| **探索率衰减** | 0.995-0.999 | 0.995 | 保持适度探索 |
| **最小探索率** | 0.01-0.05 | 0.01 | 避免完全停止探索 |
| **折扣因子** | 0.9-0.99 | 0.99 | 根据任务时间跨度调整 |
| **目标网络更新频率** | 100-1000 | 100 | 平衡稳定性和更新速度 |

---

## 📖 依赖关系

### 内部依赖

- **tinyai-deeplearning-ml**: 模型训练、优化器、损失函数
- **tinyai-deeplearning-func**: 自动微分、Variable系统
- **tinyai-deeplearning-ndarr**: 多维数组、张量运算
- **tinyai-deeplearning-nnet**: 神经网络层、激活函数

### 外部依赖

- **JUnit 4**: 单元测试框架

---

## 📚 学习路径

### 零基础用户 (约30分钟)

1. QuickStartDemo → 理解RL基本流程
2. BasicConceptsDemo → 掌握核心概念
3. BanditAlgorithmsDemo → 学习简单算法
4. DQNCartPoleDemo → 尝试深度RL

### 有基础用户 (约1小时)

1. BanditAlgorithmsDemo → 算法对比分析
2. DQNCartPoleDemo → DQN深入学习
3. CustomDevelopmentDemo → 自定义开发
4. AlgorithmComparisonDemo → 系统对比

### 深入学习用户 (约2小时)

1. AlgorithmVisualizationDemo → 可视化理解机制
2. InteractiveLearningDemo → 交互式实验
3. StepByStepDebugDemo → 深入算法细节
4. 阅读源码 → 理解架构设计

---

## 🔍 常见问题

### Q1: 为什么DQN和DoubleDQN的训练代码完全一样?

**A**: 这正是架构设计的优势! 两个算法继承了相同的`ValueBasedAgent`基类,训练流程完全统一。算法差异仅体现在`computeTargetQValues()`方法的实现上(约20行代码)。

### Q2: 如何选择合适的算法?

**A**: 
- **离散动作空间** → DQN或DoubleDQN
- **连续动作空间** → REINFORCE或PPO
- **简单探索问题** → Bandit算法
- **稳定性要求高** → DoubleDQN > DQN
- **样本效率要求高** → DQN系列 > 策略梯度

### Q3: 训练不收敛怎么办?

**A**: 
1. 降低学习率(0.001 → 0.0001)
2. 增大批次大小(32 → 64)
3. 增大缓冲区(10000 → 50000)
4. 调整探索率衰减(0.995 → 0.999)
5. 检查奖励函数设计

### Q4: 如何保存和加载模型?

```java
// 保存
agent.saveModel("checkpoints/my_agent.model");

// 加载
agent.loadModel("checkpoints/my_agent.model");
```

---

## 🚧 版本信息

- **当前版本**: 1.0-SNAPSHOT
- **Java版本**: 17+
- **构建工具**: Maven 3.6+
- **最后更新**: 2026-02-19

---

## 🌟 未来规划

### 算法扩展

- [ ] Dueling DQN (分离价值和优势函数)
- [ ] Prioritized Experience Replay (优先经验回放)
- [ ] A2C/A3C (Advantage Actor-Critic)
- [ ] PPO (Proximal Policy Optimization)
- [ ] Rainbow DQN (集成多种改进)

### 性能优化

- [ ] 多线程并行训练
- [ ] GPU加速支持
- [ ] 分布式训练框架

### 工具增强

- [ ] TensorBoard可视化
- [ ] 超参数自动调优
- [ ] 模型压缩和部署

---

## 📄 相关文档

- [技术架构文档](doc/技术架构文档.md) - 深入技术细节
- [API文档](doc/API文档.md) - 完整API参考
- [算法详解](doc/算法详解.md) - 算法原理和实现

---

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出建议!

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

---

## 📜 许可证

Apache License 2.0

---

<p align="center">
  <b>TinyAI Reinforcement Learning</b><br>
  让强化学习变得简单、清晰、可靠 🎯
</p>
