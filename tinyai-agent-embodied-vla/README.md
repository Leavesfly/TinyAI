# TinyAI Agent Embodied VLA

> 视觉-语言-动作（Vision-Language-Action, VLA）具身智能模块

## 📖 模块简介

`tinyai-agent-embodied-vla` 是 TinyAI 智能体系统层的高级具身智能模块，专注于实现基于**视觉-语言-动作（VLA）架构**的端到端具身智能系统。该模块通过融合视觉感知、自然语言理解和动作生成三大核心能力，构建能够理解指令、感知环境并执行复杂操作任务的智能体。

## 🎯 核心特性

- 🎯 **统一架构**：VLA三模态统一建模，共享Transformer骨干网络
- 🧠 **语言引导**：自然语言指令引导视觉注意力和动作生成
- 👁️ **视觉理解**：深度图像特征提取与场景语义理解
- 🤖 **精准控制**：连续动作空间与离散动作空间统一建模
- 🔄 **闭环反馈**：执行结果反馈到感知层，形成完整闭环
- 📚 **零样本泛化**：支持通过语言指令完成未训练过的新任务

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                     VLA智能体核心层                           │
│  ┌──────────────┬──────────────┬──────────────┬───────────┐ │
│  │ VLA编码器     │ 跨模态融合    │ VLA解码器     │ 学习引擎   │ │
│  └──────────────┴──────────────┴──────────────┴───────────┘ │
└─────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────┐
│                     模态处理层                                │
│  ┌──────────────┬──────────────┬──────────────┐             │
│  │ 视觉处理      │ 语言处理      │ 动作处理      │             │
│  └──────────────┴──────────────┴──────────────┘             │
└─────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────┐
│                     环境仿真层                                │
│  ┌──────────────┬──────────────┬──────────────┐             │
│  │ 机器人环境    │ 操作任务      │ 场景管理      │             │
│  └──────────────┴──────────────┴──────────────┘             │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 快速开始

### 安装依赖

```bash
cd tinyai-agent-embodied-vla
mvn clean install
```

### 运行演示程序

```bash
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.agent.vla.VLADemo"
```

### 基本使用

```java
// 1. 创建VLA智能体
VLAAgent agent = new VLAAgent(
    768,      // 隐藏层维度
    8,        // 注意力头数
    6,        // Transformer层数
    7         // 动作维度
);

// 2. 准备输入
VisionInput visionInput = new VisionInput(rgbImage);
LanguageInput languageInput = new LanguageInput("Pick up the red cube");
ProprioceptionInput proprioInput = new ProprioceptionInput(jointPositions, jointVelocities);

VLAState state = new VLAState(visionInput, languageInput, proprioInput);

// 3. 预测动作
VLAAction action = agent.predict(state);

// 4. 执行动作
RobotEnvironment env = new SimpleRobotEnv();
VLAState nextState = env.step(action);
```

## 📦 核心组件

### 编码器模块

- **VisionEncoder**: 视觉编码器（CNN + Transformer）
- **LanguageEncoder**: 语言编码器（GPT Transformer）
- **ProprioceptionEncoder**: 本体感知编码器（MLP）

### 融合层

- **CrossModalAttention**: 跨模态注意力机制
- **VLATransformerCore**: VLA Transformer核心

### 解码器模块

- **ActionDecoder**: 动作解码器（连续+离散动作）
- **LanguageFeedbackGenerator**: 语言反馈生成器

### 环境仿真

- **RobotEnvironment**: 机器人环境接口
- **SimpleRobotEnv**: 简单机器人仿真环境
- **TaskScenario**: 任务场景定义

### 学习引擎

- **VLALearningEngine**: VLA学习引擎
- **BehaviorCloningLearner**: 行为克隆学习器
- **RLLearner**: 强化学习器

## 🎓 支持的任务场景

| 任务类型 | 难度 | 描述 |
|---------|------|------|
| PickAndPlace | ⭐⭐ | 拾取物体并放置到目标位置 |
| StackBlocks | ⭐⭐⭐ | 堆叠多个方块 |
| OpenDrawer | ⭐⭐⭐ | 打开抽屉 |
| PourWater | ⭐⭐⭐⭐ | 倒水任务 |
| AssembleParts | ⭐⭐⭐⭐⭐ | 组装零件 |

## 📊 性能指标

| 指标 | 目标值 |
|-----|--------|
| 推理延迟 | < 50ms |
| 训练速度 | > 100 steps/s |
| 简单任务成功率 | > 90% |
| 复杂任务成功率 | > 60% |

## 🔧 技术亮点

### 多模态融合

- 视觉、语言、本体感知三种模态在统一Transformer框架下建模
- 跨模态注意力机制实现深度信息交互
- 语言指令动态引导视觉注意力权重

### 端到端学习

- 从原始感知输入直接到动作输出的可微分学习
- 支持监督学习（行为克隆）和强化学习两种范式
- 预训练+微调的迁移学习能力

### 工程优势

- 纯Java实现，完全基于TinyAI生态
- 充分复用GPT、RL等已有模块
- 模块化设计，易于扩展和定制

## 📚 文档

- [技术架构文档](doc/技术架构文档.md)
- [使用指南](doc/使用指南.md)
- [API参考](doc/API参考.md)

## 🤝 与其他模块关系

| 模块 | 关系 |
|-----|------|
| tinyai-agent-embodied | 基础架构参考，环境接口复用 |
| tinyai-model-gpt | Transformer骨干网络复用 |
| tinyai-deeplearning-rl | 强化学习算法复用 |
| tinyai-deeplearning-nnet | 神经网络层复用 |

## 📄 许可证

本模块遵循 TinyAI 项目的开源许可证。

## 👥 贡献者

TinyAI 开发团队

---

**TinyAI** - 纯Java实现的轻量级深度学习与智能体框架
