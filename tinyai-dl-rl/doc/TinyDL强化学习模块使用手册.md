# TinyDL 强化学习模块

TinyDL强化学习模块是基于TinyDL深度学习框架构建的强化学习工具包，提供了完整的强化学习算法实现和训练环境。

## 📋 目录

- [模块概述](#模块概述)
- [核心特性](#核心特性)
- [架构设计](#架构设计)
- [快速开始](#快速开始)
- [算法实现](#算法实现)
- [环境支持](#环境支持)
- [示例教程](#示例教程)
- [API参考](#api参考)
- [最佳实践](#最佳实践)
- [常见问题](#常见问题)

## 🎯 模块概述

TinyDL强化学习模块实现了主流的强化学习算法，包括价值函数方法和策略梯度方法。模块设计遵循OpenAI Gym的标准接口，具有良好的扩展性和易用性。

### 主要功能

- **深度Q网络 (DQN)**: 实现了经典的DQN算法，包括经验回放和目标网络
- **策略梯度 (REINFORCE)**: 实现了基础的策略梯度算法，支持基线机制
- **环境接口**: 提供了标准化的环境接口，支持自定义环境
- **训练工具**: 包含完整的训练循环和性能评估工具
- **可视化**: 支持训练过程可视化和结果分析

## ✨ 核心特性

### 🔧 算法实现
- **DQN算法**: 深度Q网络，适用于离散动作空间
- **REINFORCE算法**: 策略梯度方法，支持连续和离散动作空间
- **经验回放**: 提高样本利用效率
- **目标网络**: 提升训练稳定性
- **ε-贪婪策略**: 平衡探索与利用

### 🏃 环境支持
- **CartPole**: 经典的倒立摆控制问题
- **GridWorld**: 网格世界导航问题
- **自定义环境**: 支持用户自定义环境

### 📊 训练与评估
- **自动训练循环**: 简化训练过程
- **性能监控**: 实时监控训练指标
- **模型评估**: 完整的评估工具
- **结果可视化**: 训练曲线和性能分析

## 🏗️ 架构设计

强化学习模块采用分层设计，主要包含以下组件：

```
io.leavesfly.tinydl.modality.rl/
├── Environment.java              # 环境抽象基类
├── Agent.java                   # 智能体抽象基类
├── Policy.java                  # 策略抽象基类
├── Experience.java              # 经验数据类
├── ReplayBuffer.java            # 经验回放缓冲区
├── agent/                       # 具体算法实现
│   ├── DQNAgent.java           # DQN智能体
│   └── REINFORCEAgent.java     # REINFORCE智能体
├── environment/                 # 环境实现
│   ├── CartPoleEnvironment.java # CartPole环境
│   └── GridWorldEnvironment.java# GridWorld环境
└── policy/                     # 策略实现
    └── EpsilonGreedyPolicy.java # ε-贪婪策略
```

### 设计原则

1. **模块化**: 各组件职责清晰，便于扩展
2. **标准化**: 遵循强化学习领域的标准接口
3. **可复用**: 算法和环境可以灵活组合
4. **可扩展**: 支持添加新的算法和环境

## 🚀 快速开始

### 1. CartPole环境 + DQN算法

```java
// 创建环境
Environment env = new CartPoleEnvironment();

// 创建DQN智能体
DQNAgent agent = new DQNAgent(
    "CartPole_DQN",
    env.getStateDim(),      // 状态维度：4
    env.getActionDim(),     // 动作维度：2
    new int[]{128, 128},    // 隐藏层尺寸
    0.001f,                 // 学习率
    1.0f,                   // 初始探索率
    0.99f,                  // 折扣因子
    32,                     // 批次大小
    10000,                  // 缓冲区大小
    100                     // 目标网络更新频率
);

// 训练循环
for (int episode = 0; episode < 1000; episode++) {
    Variable state = env.reset();
    
    while (!env.isDone()) {
        Variable action = agent.selectAction(state);
        Environment.StepResult result = env.step(action);
        
        Experience experience = new Experience(
            state, action, result.getReward(), 
            result.getNextState(), result.isDone()
        );
        
        agent.learn(experience);
        state = result.getNextState();
    }
}
```

### 2. GridWorld环境 + REINFORCE算法

```java
// 创建GridWorld环境
Environment env = GridWorldEnvironment.createSimpleMaze();

// 创建REINFORCE智能体
REINFORCEAgent agent = new REINFORCEAgent(
    "GridWorld_REINFORCE",
    env.getStateDim(),      // 状态维度：2
    env.getActionDim(),     // 动作维度：4
    new int[]{64, 64},      // 隐藏层尺寸
    0.01f,                  // 学习率
    0.99f,                  // 折扣因子
    true                    // 使用基线
);

// 训练循环
for (int episode = 0; episode < 2000; episode++) {
    Variable state = env.reset();
    
    while (!env.isDone()) {
        Variable action = agent.selectAction(state);
        Environment.StepResult result = env.step(action);
        
        Experience experience = new Experience(
            state, action, result.getReward(), 
            result.getNextState(), result.isDone()
        );
        
        agent.learn(experience);
        state = result.getNextState();
    }
    
    // REINFORCE在回合结束时学习
    agent.learnFromEpisode();
}
```

## 🤖 算法实现

### Deep Q-Network (DQN)

DQN是第一个成功将深度学习应用于强化学习的算法，主要特点：

**核心思想**:
- 使用神经网络逼近Q函数
- 通过Bellman方程更新Q值
- 使用ε-贪婪策略进行动作选择

**关键技术**:
- **经验回放**: 打破数据相关性，提高学习效率
- **目标网络**: 提供稳定的学习目标
- **ε-贪婪策略**: 平衡探索与利用

**适用场景**:
- 离散动作空间
- 样本效率要求较高
- 需要稳定训练的场景

### REINFORCE算法

REINFORCE是经典的策略梯度算法，基于蒙特卡罗方法：

**核心思想**:
- 直接优化策略网络
- 使用策略梯度定理
- 通过采样估计梯度

**关键技术**:
- **策略梯度**: ∇θ J(θ) = E[∇θ log π(a|s) * R]
- **基线机制**: 减少方差，提高学习稳定性
- **蒙特卡罗采样**: 使用完整回合估计回报

**适用场景**:
- 连续动作空间
- 随机策略学习
- 简单实现需求

## 🌍 环境支持

### CartPole环境

经典的倒立摆控制问题：

```java
CartPoleEnvironment env = new CartPoleEnvironment();

// 环境特性
- 状态空间: [位置, 速度, 角度, 角速度] (4维连续)
- 动作空间: [向左推, 向右推] (2维离散)  
- 奖励函数: 每步+1，倒下时回合结束
- 终止条件: 角度超过12°或位置超出边界
```

### GridWorld环境

网格世界导航问题：

```java
// 创建简单迷宫
GridWorldEnvironment env = GridWorldEnvironment.createSimpleMaze();

// 创建带随机障碍物的环境
GridWorldEnvironment env = GridWorldEnvironment.createWithRandomObstacles(8, 8, 0.2f);

// 环境特性
- 状态空间: [x坐标, y坐标] (2维离散)
- 动作空间: [上, 下, 左, 右] (4维离散)
- 奖励函数: 到达目标+10，碰撞-1，每步-0.01
- 终止条件: 到达目标位置
```

### 自定义环境

继承Environment类实现自定义环境：

```java
public class CustomEnvironment extends Environment {
    
    public CustomEnvironment() {
        super(stateDim, actionDim, maxSteps);
    }
    
    @Override
    public Variable reset() {
        // 实现环境重置逻辑
        return currentState;
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
        return true;
    }
}
```

## 📖 示例教程

### 1. 基础训练示例

运行CartPole环境的DQN训练：

```bash
java io.leavesfly.tinydl.example.rl.CartPoleDQNExample
```

### 2. 策略梯度示例

运行GridWorld环境的REINFORCE训练：

```bash
java io.leavesfly.tinydl.example.rl.GridWorldREINFORCEExample
```

### 3. 算法比较示例

比较不同算法的性能：

```bash
java io.leavesfly.tinydl.example.rl.RLAlgorithmComparison
```

### 示例输出解读

```
Episode 100: 奖励=89.00, 步数=89, Epsilon=0.905, 损失=0.003421, 缓冲区使用率=3.20%
Episode 200: 奖励=156.00, 步数=156, Epsilon=0.819, 损失=0.002891, 缓冲区使用率=6.40%
```

- **奖励**: 回合累积奖励，越高越好
- **步数**: 回合持续步数，CartPole中越多越好
- **Epsilon**: 当前探索率，会逐渐衰减
- **损失**: 神经网络训练损失
- **缓冲区使用率**: 经验回放缓冲区的使用情况

## 📚 API参考

### Environment类

环境抽象基类，定义了标准的强化学习环境接口。

```java
public abstract class Environment {
    // 重置环境到初始状态
    public abstract Variable reset();
    
    // 执行动作，返回下一状态、奖励、是否结束
    public abstract StepResult step(Variable action);
    
    // 随机采样动作
    public abstract Variable sampleAction();
    
    // 检查动作是否有效
    public abstract boolean isValidAction(Variable action);
}
```

### Agent类

智能体抽象基类，定义了学习智能体的标准接口。

```java
public abstract class Agent {
    // 根据状态选择动作
    public abstract Variable selectAction(Variable state);
    
    // 从单个经验学习
    public abstract void learn(Experience experience);
    
    // 从经验批次学习
    public abstract void learnBatch(Experience[] experiences);
    
    // 存储经验
    public abstract void storeExperience(Experience experience);
}
```

### DQNAgent类

DQN算法的具体实现。

```java
// 构造函数
public DQNAgent(String name, int stateDim, int actionDim, int[] hiddenSizes,
                float learningRate, float epsilon, float gamma,
                int batchSize, int bufferSize, int targetUpdateFreq)

// 关键方法
public Variable selectAction(Variable state)           // 动作选择
public void learn(Experience experience)              // 单步学习
public float getAverageLoss()                         // 获取平均损失
public float getCurrentEpsilon()                      // 获取当前探索率
```

### REINFORCEAgent类

REINFORCE算法的具体实现。

```java
// 构造函数
public REINFORCEAgent(String name, int stateDim, int actionDim, int[] hiddenSizes,
                     float learningRate, float gamma, boolean useBaseline)

// 关键方法
public Variable selectAction(Variable state)           // 动作选择
public void learn(Experience experience)              // 存储经验
public void learnFromEpisode()                        // 回合结束学习
public float getAverageReturn()                       // 获取平均回报
```

## 💡 最佳实践

### 1. 超参数调优

**DQN算法建议**:
- 学习率: 0.0001 - 0.001
- 探索率衰减: 从1.0衰减到0.01
- 批次大小: 32 - 128
- 缓冲区大小: 10,000 - 100,000
- 目标网络更新频率: 100 - 1000步

**REINFORCE算法建议**:
- 学习率: 0.001 - 0.01
- 使用基线减少方差
- 折扣因子: 0.95 - 0.99

### 2. 训练技巧

**数据预处理**:
```java
// 状态归一化
public Variable getNormalizedState() {
    float[] normalized = new float[stateDim];
    for (int i = 0; i < stateDim; i++) {
        normalized[i] = (state[i] - mean[i]) / std[i];
    }
    return new Variable(new NdArray(normalized, new Shape(1, stateDim)));
}
```

**奖励设计**:
- 奖励应该与目标相关
- 避免过于稀疏的奖励
- 考虑奖励的尺度

**训练监控**:
```java
// 定期评估性能
if (episode % 100 == 0) {
    evaluateAgent(agent, env, 10);
}

// 保存最佳模型
if (currentPerformance > bestPerformance) {
    agent.saveModel("best_model.pkl");
    bestPerformance = currentPerformance;
}
```

### 3. 调试指南

**常见问题及解决方案**:

1. **学习不收敛**:
   - 检查学习率是否过大
   - 确认奖励函数设计合理
   - 增加网络容量或训练时间

2. **过度探索**:
   - 调整ε衰减策略
   - 检查探索-利用平衡

3. **训练不稳定**:
   - 使用目标网络（DQN）
   - 使用基线（REINFORCE）
   - 调整批次大小

### 4. 性能优化

**内存优化**:
```java
// 定期清理经验缓冲区
if (replayBuffer.size() > maxBufferSize) {
    replayBuffer.clear();
}

// 使用适当的缓冲区大小
int bufferSize = Math.min(100000, maxEpisodes * averageEpisodeLength);
```

**计算优化**:
```java
// 批量处理经验
if (replayBuffer.canSample(batchSize)) {
    Experience[] batch = replayBuffer.sample(batchSize);
    agent.learnBatch(batch);
}
```

## ❓ 常见问题

### Q1: 如何选择合适的算法？

**A1**: 
- **离散动作空间**: DQN或REINFORCE都适用
- **连续动作空间**: 优先选择REINFORCE
- **样本效率要求高**: 选择DQN（经验回放）
- **简单实现**: 选择REINFORCE
- **稳定性要求高**: 选择DQN（目标网络）

### Q2: 训练不收敛怎么办？

**A2**:
1. 检查环境实现是否正确
2. 调整学习率（通常减小）
3. 增加网络容量
4. 检查奖励函数设计
5. 增加训练时间

### Q3: 如何提高训练效率？

**A3**:
1. 使用经验回放（DQN）
2. 调整批次大小
3. 使用预训练模型
4. 优化奖励函数设计
5. 使用适当的状态表示

### Q4: 如何实现自定义环境？

**A4**:
继承Environment类并实现必要方法：
```java
public class MyEnvironment extends Environment {
    public MyEnvironment() {
        super(stateDim, actionDim, maxSteps);
    }
    
    // 实现抽象方法
    public Variable reset() { /* ... */ }
    public StepResult step(Variable action) { /* ... */ }
    public Variable sampleAction() { /* ... */ }
    public boolean isValidAction(Variable action) { /* ... */ }
}
```

### Q5: 如何保存和加载模型？

**A5**:
```java
// 保存模型
agent.saveModel("model_checkpoint.pkl");

// 加载模型
agent.loadModel("model_checkpoint.pkl");
```

注意：当前版本的保存/加载功能需要根据TinyDL框架的序列化机制进一步完善。

---

## 📞 技术支持

如果您在使用过程中遇到问题，请：

1. 查看本文档的常见问题部分
2. 检查示例代码的实现
3. 参考TinyDL框架的核心文档
4. 提交issue或联系开发团队

**开发团队**: TinyDL强化学习模块开发团队  
**版本**: 0.01  
**更新时间**: 2024年

---

*该文档会随着模块的更新而持续完善，欢迎反馈和建议！*