# TinyAI Reinforcement Learning 强化学习模块

> **教学导向的现代强化学习框架** - 清晰的架构设计 + 完整的算法实现 + 丰富的教学文档

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]() [![Java](https://img.shields.io/badge/Java-17%2B-orange)]() [![License](https://img.shields.io/badge/license-Apache%202.0-blue)]()

## 模块简介

`tinyai-deeplearning-rl` 是 TinyAI 深度学习框架的强化学习核心模块，实现了从经典多臂老虎机到现代深度强化学习的完整算法体系。

### 设计目标

1. **教学导向**: 清晰的代码结构和详尽的注释，帮助学习者深入理解强化学习
2. **架构整洁**: 接口与抽象类分离，消除代码重复，通过继承和组合建立统一框架
3. **易于扩展**: 新增算法只需继承基类并实现少量核心方法
4. **工业实践**: 遵循 OpenAI Gym 规范，代码质量达到生产级别

### 核心特性

- **接口驱动**: `Agent` 设计为接口，`AbstractAgent` 作为神经网络 Agent 的基类，BanditAgent 直接实现接口
- **统一流程**: `ValueBasedAgent` 和 `PolicyBasedAgent` 两大基类通过模板方法模式统一训练流程
- **算法丰富**: 涵盖 Bandit、DQN 系列、策略梯度（REINFORCE、PPO）四大类别
- **组件复用**: `TrainingStatistics`、`QValueComputer`、`ModelUtil` 三个工具类消除重复代码
- **动作空间**: `ActionSpace` 接口体系，区分离散与连续两种动作空间
- **优先回放**: 新增 `PrioritizedReplayBuffer`，支持基于 TD 误差的优先级采样

---

## 核心架构

### 算法分类体系

```
Agent (接口) ← MDP 标准交互协议
 │
 ├── AbstractAgent (基于神经网络的通用基类)
 │    ├── ValueBasedAgent (基于值函数) ← 统一经验回放 + TD 学习
 │    │    ├── DQNAgent              深度 Q 网络
 │    │    └── DoubleDQNAgent        双 Q 网络（解决过估计）
 │    │
 │    ├── PolicyBasedAgent (基于策略) ← 统一回合采样 + 策略梯度
 │    │    └── REINFORCEAgent        蒙特卡罗策略梯度
 │    │
 │    └── PPOAgent                  近端策略优化（独立实现）
 │
 └── BanditAgent (多臂老虎机) ← 直接实现 Agent 接口
      ├── EpsilonGreedyBanditAgent  ε-贪心
      ├── UCBBanditAgent            上置信区间
      └── ThompsonSamplingBanditAgent  汤普森采样
```

### 为什么 Agent 是接口

```java
/**
 * Agent 被设计为接口而非抽象类，因为不同类型的 Agent 差异很大：
 * - 基于神经网络的 Agent（DQN/PPO）需要 model、optimizer 等字段
 * - 多臂老虎机 Agent（Bandit）不需要神经网络，只需统计信息
 * - 使用接口可以让各分支自由选择继承体系，避免被迫继承无用字段
 */
public interface Agent {
    Variable selectAction(Variable state);
    void learn(Experience experience);
    void learnBatch(Experience[] experiences);
    void storeExperience(Experience experience);
    // ...
    
    // 默认方法：维度兼容性校验
    default void validateCompatibility(Environment environment) { ... }
    
    // 默认方法：便捷模式切换
    default void train() { setTraining(true); }
    default void eval()  { setTraining(false); }
}
```

### AbstractAgent - 神经网络 Agent 公共基类

为使用神经网络的 RL 算法提供通用基础设施（BanditAgent 不继承此类）：

```java
public abstract class AbstractAgent implements Agent {
    protected String name;
    protected int stateDim;
    protected int actionDim;
    protected Model model;        // 主神经网络
    protected float learningRate;
    protected float epsilon;      // 探索率
    protected float gamma;        // 折扣因子
    protected int trainingStep;
    protected boolean training;
    
    // 统一的 epsilon 衰减
    public void decayEpsilon(float decayRate) {
        this.epsilon = Math.max(0.01f, this.epsilon * decayRate);
    }
}
```

### 统一训练流程 - ValueBasedAgent

```java
public abstract class ValueBasedAgent extends AbstractAgent {
    @Override
    public void learn(Experience experience) {
        // 1. 存储经验到缓冲区
        storeExperience(experience);
        
        // 2. 检查是否有足够经验
        if (shouldLearn()) {
            // 3. 采样批次
            Experience[] batch = sampleBatch();
            
            // 4. 计算目标 Q 值（子类实现差异）
            Variable targetQ  = computeTargetQValues(batch);
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

**教学价值**：
- DQN 与 DoubleDQN 的差异仅在**目标 Q 值计算**上（约 20 行代码）
- 复用经验回放、目标网络同步、训练统计等通用逻辑

---

## 公共组件

### 工具类

| 工具类 | 职责 | 消除重复 |
|--------|------|----------|
| **TrainingStatistics** | 统一训练指标管理（奖励、损失等统计） | 消除各 Agent 重复的统计代码 |
| **QValueComputer** | Q 值相关计算（贪婪选择、最大值提取、堆叠） | 消除各 Agent 重复的 Q 值代码 |
| **ModelUtil** | 模型操作（网络创建、权重复制、软更新） | 消除各 Agent 重复的网络构建代码 |

```java
// 创建 Q 网络 - 一行代码
Model qNetwork = ModelUtil.createQNetwork("DQN_Q", 4, 2, new int[]{128, 128});

// 权重复制 - 目标网络同步
ModelUtil.copyWeights(onlineNetwork, targetNetwork);

// Q 值计算 - 自动处理 Variable 堆叠
Variable batchQValues = QValueComputer.stackVariables(qArray, batchSize);
```

### ActionSpace 动作空间体系

```
ActionSpace (接口)
 ├── DiscreteActionSpace   离散动作空间（有限个动作，适合 DQN 等）
 └── ContinuousActionSpace 连续动作空间（连续值，适合 PPO/DDPG 等）
```

| 动作空间类型 | 适合算法 | 示例 |
|---|---|---|
| **DiscreteActionSpace** | DQN、DoubleDQN、REINFORCE | CartPole（左/右 2 个动作） |
| **ContinuousActionSpace** | PPO、DDPG | 机器人关节角度控制 |

### ReplayBuffer vs PrioritizedReplayBuffer

| 缓冲区 | 采样方式 | 适用场景 |
|--------|----------|----------|
| **ReplayBuffer** | 均匀随机采样 | 标准 DQN、快速实验 |
| **PrioritizedReplayBuffer** | 按 TD 误差优先级采样（Sum Tree 实现） | 提升样本利用率，加快收敛 |

### Policy 策略组件

```java
// 独立的 ε-贪婪策略组件，可插拔使用
EpsilonGreedyPolicy policy = new EpsilonGreedyPolicy(
    stateDim, actionDim, 1.0f, qFunction
);
Variable action = policy.selectAction(state);
```

---

## 算法实现详解

### DQNAgent - 深度 Q 网络

**算法原理**：
```
目标 Q 值: y = r + γ * max_a' Q_target(s', a')
损失函数:   L = (y - Q(s, a))²
```

**关键创新**：经验回放（打破数据相关性）+ 目标网络（稳定训练）+ ε-贪心（探索利用平衡）

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

### DoubleDQNAgent - 双 Q 网络

**解决问题**：DQN 的 Q 值**过估计**问题

```
DQN:       y = r + γ * max_a' Q_target(s', a')        ← 选择和评估都用目标网络
DoubleDQN: a* = argmax_a' Q_online(s', a')             ← 在线网络选择动作
           y  = r + γ * Q_target(s', a*)               ← 目标网络评估 Q 值
```

**与 DQN 的代码差异仅约 20 行**，集中在 `computeTargetQValues()` 方法。

| 特性 | DQN | DoubleDQN |
|------|-----|-----------|
| 动作选择 | 目标网络 | **在线网络** |
| Q 值评估 | 目标网络 | 目标网络 |
| 过估计问题 | 存在 | **缓解** |

### REINFORCEAgent - 策略梯度

**算法原理**：
```
策略梯度定理: ∇J(θ) = E[∇log π(a|s) * G_t]
损失函数:     L = -Σ log π(a_t|s_t) * G_t
蒙特卡罗回报: G_t = Σ_{k=0}^∞ γ^k * r_{t+k}
基线减方差:   L = -Σ log π(a_t|s_t) * (G_t - V(s_t))
```

```java
REINFORCEAgent agent = new REINFORCEAgent(
    "CartPole-REINFORCE",
    4, 2, new int[]{128, 64},
    0.001f,   // 学习率
    0.99f,    // 折扣因子
    true      // 使用基线
);
```

### PPOAgent - 近端策略优化

**算法特点**：目前最流行的策略梯度算法之一

```
PPO 裁剪目标: L = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
              其中 r_t = π_θ(a|s) / π_θ_old(a|s)
```

**关键超参数**：

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `clipEpsilon` | PPO 裁剪参数，限制策略更新幅度 | 0.2 |
| `ppoEpochs` | 每批数据的优化轮数 | 4-10 |
| `batchSize` | mini-batch 大小 | 64 |

```java
PPOAgent agent = new PPOAgent(
    "CartPole-PPO",
    4, 2, new int[]{128, 64},
    0.0003f,  // 学习率
    0.99f,    // 折扣因子
    0.2f,     // clipEpsilon
    10,       // ppoEpochs
    64        // batchSize
);
```

### 多臂老虎机算法

| 算法 | 探索策略 | 理论保证 | 适用场景 |
|------|----------|----------|----------|
| **EpsilonGreedyBanditAgent** | 固定概率随机探索 | 简单遗憾界 | 在线学习、快速决策 |
| **UCBBanditAgent** | 置信区间 `Q(i) + c*√(ln(t)/N(i))` | 最优遗憾界 | 理论最优、稳定环境 |
| **ThompsonSamplingBanditAgent** | 贝叶斯后验采样 | 最优遗憾界 | 贝叶斯优化、不确定环境 |

---

## 环境实现

### CartPoleEnvironment - 倒立摆

- **状态空间**: 4 维 [位置, 速度, 角度, 角速度]
- **动作空间**: 离散 2 维 [向左推, 向右推]
- **目标**: 保持杆子平衡，基于经典力学方程模拟

### GridWorldEnvironment - 网格世界

- **状态空间**: 离散网格位置
- **动作空间**: 4 方向（上下左右）
- **特性**: 可配置奖励和障碍物

### MultiArmedBanditEnvironment - 多臂老虎机

- 多个臂，支持高斯分布奖励
- 记录累积遗憾值，方便算法对比

---

## 快速开始

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

### DQN 训练示例

```java
import io.leavesfly.tinyai.rl.agent.DQNAgent;
import io.leavesfly.tinyai.rl.environment.CartPoleEnvironment;
import io.leavesfly.tinyai.rl.Experience;

// 1. 创建环境
CartPoleEnvironment env = new CartPoleEnvironment();

// 2. 创建 DQN 智能体
DQNAgent agent = new DQNAgent(
    "CartPole-DQN",
    env.getStateDim(),   // 状态维度 (4)
    env.getActionDim(),  // 动作维度 (2)
    new int[]{128, 128}, // 隐藏层
    0.001f,              // 学习率
    1.0f,                // 初始探索率
    0.99f,               // 折扣因子
    32,                  // 批次大小
    10000,               // 缓冲区大小
    100                  // 目标网络更新频率
);

// 3. 可选：校验维度兼容性
agent.validateCompatibility(env);

// 4. 训练循环
for (int episode = 0; episode < 1000; episode++) {
    Variable state = env.reset();
    
    while (!env.isDone()) {
        Variable action = agent.selectAction(state);
        Environment.StepResult result = env.step(action);
        
        agent.learn(new Experience(state, action, result.getNextState(),
                                  result.getReward(), result.isDone()));
        
        state = result.getNextState();
    }
    
    agent.decayEpsilon(0.995f);
    
    if (episode % 100 == 0) {
        System.out.printf("Episode %d: Epsilon = %.3f%n", episode, agent.getEpsilon());
    }
}
```

### DoubleDQN vs DQN - 只需改一个类名

```java
// 完全相同的训练代码，更稳定的性能
DoubleDQNAgent agent = new DoubleDQNAgent(
    "CartPole-DoubleDQN", 4, 2, new int[]{128, 128},
    0.001f, 1.0f, 0.99f, 32, 10000, 100
);
// 后续训练代码完全一致！
```

---

## 扩展开发

### 添加新的 Value-Based 算法（以 Dueling DQN 为例）

```java
// 步骤 1: 继承 ValueBasedAgent
public class DuelingDQNAgent extends ValueBasedAgent {
    
    public DuelingDQNAgent(...) {
        super(...);
        // 创建 Dueling 网络（分离价值流和优势流）
        this.model = ModelUtil.createDuelingQNetwork(...);
        this.targetModel = ModelUtil.createDuelingQNetwork(...);
        ModelUtil.copyWeights(model, targetModel);
    }
    
    // 步骤 2: 实现目标 Q 值计算（可直接复用 DoubleDQN 逻辑）
    @Override
    protected Variable computeTargetQValues(Experience[] experiences) { ... }
    
    // 步骤 3: 实现当前 Q 值计算（Dueling 网络自动合并 V 和 A）
    @Override
    protected Variable computeCurrentQValues(Experience[] experiences) { ... }
}
// 完成！自动继承经验回放、目标网络同步等所有功能
```

### 添加新环境

```java
public class MyEnvironment extends Environment {
    
    public MyEnvironment() {
        super(stateDim, actionDim, maxSteps);
    }
    
    @Override
    public Variable reset() {
        return initialState;
    }
    
    @Override
    public StepResult step(Variable action) {
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

## 教学 Demo

本模块提供 10 个精心设计的教学 Demo：

| 分类 | Demo | 难度 | 说明 |
|------|------|------|------|
| 基础入门 | **QuickStartDemo** | ⭐ | 3 分钟了解 RL 基本流程 |
| 基础入门 | **BasicConceptsDemo** | ⭐⭐ | 详解状态、动作、奖励、策略 |
| 基础入门 | **BanditAlgorithmsDemo** | ⭐⭐ | ε-贪心、UCB、汤普森采样对比 |
| 深度 RL | **DQNCartPoleDemo** | ⭐⭐⭐ | DQN 完整训练流程 |
| 深度 RL | **CustomDevelopmentDemo** | ⭐⭐⭐⭐ | 自定义环境和智能体 |
| 教育增强 | **AlgorithmVisualizationDemo** | ⭐⭐ | 可视化 DQN 内部机制 |
| 教育增强 | **InteractiveLearningDemo** | ⭐⭐ | 交互式调参实验 |
| 教育增强 | **AlgorithmComparisonDemo** | ⭐⭐⭐ | 并排对比多种算法 |
| 教育增强 | **StepByStepDebugDemo** | ⭐⭐⭐ | 逐步调试 DQN 算法 |
| 教育增强 | **LearningAnimationDemo** | ⭐⭐ | 动画展示学习过程 |

```bash
# 快速入门
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.rl.demo.QuickStartDemo" \
  -pl tinyai-deeplearning-rl

# DQN 实战
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.rl.demo.DQNCartPoleDemo" \
  -pl tinyai-deeplearning-rl

# 算法对比
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.rl.demo.AlgorithmComparisonDemo" \
  -pl tinyai-deeplearning-rl
```

---

## 超参数调优指南

| 参数 | 推荐范围 | 典型值 | 说明 |
|------|----------|--------|------|
| **学习率** | 0.0001-0.01 | 0.001 | 从小开始，根据收敛情况调整 |
| **批次大小** | 32-128 | 64 | 平衡计算效率和梯度稳定性 |
| **缓冲区大小** | 10000-100000 | 10000 | 根据内存和任务复杂度 |
| **初始探索率** | 0.9-1.0 | 1.0 | 充分探索 |
| **探索率衰减** | 0.995-0.999 | 0.995 | 保持适度探索 |
| **最小探索率** | 0.01-0.05 | 0.01 | 避免完全停止探索 |
| **折扣因子** | 0.9-0.99 | 0.99 | 根据任务时间跨度调整 |
| **目标网络更新频率** | 100-1000 | 100 | 平衡稳定性和更新速度 |

---

## 测试体系

```bash
# 运行所有测试
mvn test -pl tinyai-deeplearning-rl
```

测试覆盖范围：
- **组件级**: `Experience`、`ReplayBuffer`、`Policy`
- **算法级**: `DQNAgent`、`DoubleDQNAgent`、`REINFORCEAgent`、Bandit 算法
- **集成级**: Agent-Environment 交互、完整训练流程

---

## 依赖关系

### 内部依赖

- **tinyai-deeplearning-ml**: 模型训练、优化器、损失函数
- **tinyai-deeplearning-func**: 自动微分、Variable 系统
- **tinyai-deeplearning-ndarr**: 多维数组、张量运算
- **tinyai-deeplearning-nnet**: 神经网络层、激活函数

### 外部依赖

- **JUnit 4**: 单元测试框架

---

## 相关文档

- [技术架构文档](doc/技术架构文档.md) - 深入技术细节
- [API 文档](doc/API文档.md) - 完整 API 参考

---

## 版本信息

- **当前版本**: 1.0-SNAPSHOT
- **Java 版本**: 17+
- **构建工具**: Maven 3.6+

---

<p align="center">
  <b>TinyAI Reinforcement Learning</b><br>
  让强化学习变得简单、清晰、可靠
</p>
