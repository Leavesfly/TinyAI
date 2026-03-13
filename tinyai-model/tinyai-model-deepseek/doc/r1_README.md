<div align="center">

# 🧠 DeepSeek-R1 推理增强模型

**基于 MoE 架构 + 强化学习驱动推理能力涌现**

[![Java](https://img.shields.io/badge/Java-17+-orange.svg)](https://openjdk.java.net/)
[![TinyAI](https://img.shields.io/badge/TinyAI-Framework-blue.svg)](../../README.md)
[![DeepSeek](https://img.shields.io/badge/DeepSeek-R1-green.svg)](https://arxiv.org/abs/2501.12948)

</div>

## 📋 目录

- [概述](#-概述)
- [架构设计](#-架构设计)
- [核心组件](#-核心组件)
- [使用示例](#-使用示例)
- [训练流程](#-训练流程)
- [测试验证](#-测试验证)
- [与官方架构的差异](#-与官方架构的差异)
- [参考资料](#-参考资料)

## 🎯 概述

DeepSeek-R1 是 TinyAI 框架中的**推理增强语言模型**实现，基于 [DeepSeek-R1 论文 (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948)。

根据官方论文，R1 与 V3 共享相同的 MoE 基础架构，核心区别在于**训练方式**：V3 采用标准预训练 + SFT + RLHF，而 R1 在 V3-Base 之上通过**纯强化学习（GRPO/RLVR）**驱动推理能力的自然涌现。

本实现忠实复现了这一设计理念：**R1 Block 直接复用 V3 的 MoE TransformerBlock**，无需 config 转换，仅扩展 RL 训练配置。

### 核心特性

| 特性 | 说明 |
|------|------|
| **MoE 架构** | 复用 V3 的 MoE TransformerBlock，Pre-RMSNorm + RoPE |
| **RL 驱动推理** | 通过 RLHF + RLVR 训练器优化推理能力 |
| **GRPO 算法** | RLVR 使用 Group Relative Policy Optimization（对标论文） |
| **可验证奖励** | 数学/代码/逻辑验证器自动评估推理正确性 |
| **无位置嵌入** | Token 嵌入层不含位置编码，位置信息由 RoPE 在注意力层提供 |
| **无 MTP 头** | R1 不含 MTP（Multi-Token Prediction），这是 V3 特有的训练辅助机制 |
| **Variable 完整性** | 所有计算在 Variable 层面完成，梯度正确回传 |
| **多阶段训练** | 预训练 → 后训练(SFT) → RLHF → RLVR(GRPO) 完整流程 |

### 模型规格

| 配置 | 嵌入维度 | 层数 | 注意力头 | 专家数 | Top-K | 序列长度 |
|------|---------|------|---------|--------|-------|---------|
| **Tiny** | 64 | 6 | 8 | 4 | 2 | 128 |
| **Small** | 512 | 8 | 8 | 8 | 2 | 1024 |
| **Standard** | 768 | 12 | 12 | 8 | 2 | 2048 |

## 🏗️ 架构设计

### 整体架构

```
DeepSeek-R1 模型架构（MoE）
├── DeepSeekR1Model (extends DeepSeekModelBase)
│   └── DeepSeekR1Block (extends Module)
│       ├── DeepSeekR1TokenEmbedding
│       │   ├── Token 嵌入 [vocabSize, nEmbd]  （无位置嵌入，RoPE 在注意力层）
│       │   └── Dropout 正则化
│       ├── N × DeepSeekV3TransformerBlock（✅ 直接复用 V3 组件）
│       │   ├── Pre-RMSNorm
│       │   ├── MultiHeadAttention（含 RoPE）
│       │   ├── 残差连接
│       │   ├── Pre-RMSNorm
│       │   ├── DeepSeekV3MoELayer（共享专家 + Top-K 路由专家）
│       │   └── 残差连接
│       ├── Final RMSNorm
│       └── Linear 输出投影 [nEmbd → vocabSize]
└── DeepSeekR1Config (extends DeepSeekBaseConfig)
    ├── 基础 MoE 架构配置（继承自 DeepSeekBaseConfig）
    └── RL 训练配置（R1 特有）
```

### R1 与 V3 的关系

```
DeepSeekBaseConfig（共享 MoE 基础架构）
    ├── DeepSeekV3Config（V3 特有：MTP、代码生成、推理增强）
    └── DeepSeekR1Config（R1 特有：RL 训练参数）

关键设计：DeepSeekR1Config 继承 DeepSeekBaseConfig，
         DeepSeekV3TransformerBlock 接受 DeepSeekBaseConfig，
         因此 R1Block 可直接传入 R1Config，无需任何 config 转换。
```

**R1 与 V3 Block 的核心差异**：
- R1 不包含 MTP（Multi-Token Prediction）头 — MTP 是 V3 的训练辅助机制
- R1 的 Token 嵌入层无位置嵌入参数 — RoPE 在注意力层提供位置信息
- R1 推理能力通过 RL 训练自然涌现，无需显式推理模块

### 数据流

```
输入 Token IDs [batch, seq]
    ↓
DeepSeekR1TokenEmbedding
    Token 嵌入（仅）→ [batch, seq, embd]
    ↓
N × DeepSeekV3TransformerBlock（MoE + RoPE）
    MHA（RoPE 位置编码）→ MoE FFN → 残差连接
    ↓
Final RMSNorm
    ↓
Linear 输出投影 → Logits [batch, seq, vocab]
```

## 🔧 核心组件

### 1. 模型配置（`DeepSeekR1Config`）

继承 `DeepSeekBaseConfig`，与 V3 共享 MoE 基础架构配置，仅扩展 RL 训练参数。

**预设配置工厂方法**：

```java
// 微型配置（快速测试）：64维, 6层, 8头, 4专家, Top-2, 128序列
DeepSeekR1Config.createTinyConfig();

// 小型配置（学习实验）：512维, 8层, 8头, 8专家, Top-2, 1024序列
DeepSeekR1Config.createSmallConfig();

// 标准配置（标准应用）：768维, 12层, 12头, 8专家, Top-2, 2048序列
DeepSeekR1Config.createStandardConfig();
```

**R1 特有的 RL 训练配置**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `rlExplorationRate` | 0.1 | RL 探索率 |
| `rlDiscountFactor` | 0.99 | 奖励折扣因子 |
| `rlPolicyLearningRate` | 1e-5 | 策略学习率 |
| `rlValueLearningRate` | 1e-4 | 值函数学习率 |
| `rlBatchSize` | 32 | RL 训练批次大小 |
| `rlClipRange` | 0.2 | PPO clip 范围 |
| `rlEntropyCoefficient` | 0.01 | 熵系数（鼓励探索） |
| `rlMaxGradNorm` | 1.0 | 最大梯度范数（梯度裁剪） |
| `rlRewardScale` | 1.0 | 奖励缩放因子 |

### 2. 主体块（`DeepSeekR1Block`）

R1 的核心计算模块，**直接复用 V3 的 MoE TransformerBlock**。

由于 `DeepSeekV3TransformerBlock` 已重构为接受 `DeepSeekBaseConfig`，`DeepSeekR1Config` 继承自 `DeepSeekBaseConfig`，因此可以直接传入，无需 config 转换：

```java
// R1Block 初始化 Transformer 层（无需 config 转换）
for (int i = 0; i < config.getNLayer(); i++) {
    DeepSeekV3TransformerBlock block =
        new DeepSeekV3TransformerBlock(name + "_transformer_" + i, config);
    transformerBlocks.add(block);
}
```

**支持三种前向传播模式**：

| 方法 | 说明 | 返回值 |
|------|------|--------|
| `forward(Variable... inputs)` | 标准前向传播 | `Variable` logits |
| `forward(Variable tokenIds, TaskType taskType)` | 带任务类型的前向传播 | `Variable` logits |
| `forwardWithDetails(Variable tokenIds, TaskType taskType)` | 带 MoE 损失的详细输出 | `DetailedForwardResult` |

`DetailedForwardResult` 包含两个字段：
- **`logits`** — 最终 logits 输出
- **`avgMoELoss`** — 各 Transformer 层 MoE 负载均衡损失的均值

### 3. Token 嵌入层（`DeepSeekR1TokenEmbedding`）

**注意：R1 的 Token 嵌入层不含位置嵌入**，位置信息由 RoPE（Rotary Position Embedding）在注意力层内部提供。

完全在 Variable 层面实现，确保梯度正确回传：

```java
// Token 嵌入：使用 indexSelect 算子
Variable flatIds = tokenIds.reshape(Shape.of(batchSize * sequenceLength));
Variable flatEmbeds = tokenEmbedParam.indexSelect(0, flatIds);
Variable tokenEmbeds = flatEmbeds.reshape(Shape.of(batchSize, sequenceLength, embeddingDim));

// 直接应用 Dropout（无位置嵌入合并步骤）
return dropout.forward(tokenEmbeds);
```

**Variable 算子使用**：
- **`indexSelect`** — 索引选择（替代手动 NdArray 操作）
- **`reshape`** — 形状变换

### 4. 模型类（`DeepSeekR1Model`）

提供高层 API，支持多种推理模式：

| 方法 | 说明 | 返回值 |
|------|------|--------|
| `predict(Variable tokenIds)` | 标准预测 | `Variable` logits |
| `predict(Variable tokenIds, TaskType taskType)` | 带任务类型的预测 | `Variable` logits |
| `performReasoning(Variable tokenIds)` | 推理任务（REASONING 类型） | `ReasoningResult` |
| `solveMath(Variable tokenIds)` | 数学计算任务（MATH 类型） | `ReasoningResult` |

`ReasoningResult` 包含三个字段：
- **`logits`** — 最终 logits 输出
- **`moeLoss`** — MoE 负载均衡损失
- **`taskType`** — 任务类型（REASONING / MATH 等）

**工厂方法**：

```java
DeepSeekR1Model.createTinyModel("name");      // 微型模型（64维）
DeepSeekR1Model.createSmallModel("name");     // 小型模型（512维）
DeepSeekR1Model.createStandardModel("name"); // 标准模型（768维）
```

## 💻 使用示例

### 1. 创建模型并推理

```java
// 创建微型模型（用于快速测试）
DeepSeekR1Model model = DeepSeekR1Model.createTinyModel("DeepSeek-R1-Tiny");
model.printModelInfo();
System.out.println(model.getConfigSummary());

// 准备输入 [batch_size=2, seq_len=8]
DeepSeekR1Config config = model.getConfig();
float[][] inputData = new float[2][8];
for (int b = 0; b < 2; b++) {
    for (int s = 0; s < 8; s++) {
        inputData[b][s] = (float) (Math.random() * config.getVocabSize());
    }
}

// 执行标准推理
Variable logits = model.predict(new Variable(NdArray.of(inputData)));
System.out.println("输出形状: " + logits.getValue().getShape());
// 输出形状: [2, 8, 50257]
```

### 2. 带详细信息的推理（获取 MoE 损失）

```java
// 推理任务（利用 RL 训练涌现的推理能力）
NdArray tokenIds = NdArray.of(inputData);
DeepSeekR1Model.ReasoningResult result = model.performReasoning(new Variable(tokenIds));

System.out.println("任务类型: " + result.taskType.getDescription());
System.out.println("MoE 损失: " + result.moeLoss);
System.out.println("输出形状: " + result.logits.getValue().getShape());
```

### 3. 带任务类型的推理

```java
import io.leavesfly.tinyai.deepseek.base.TaskType;

// 数学任务
DeepSeekR1Model.ReasoningResult mathResult =
    model.solveMath(new Variable(tokenIds));

// 推理任务（指定任务类型）
Variable reasoningLogits = model.predict(
    new Variable(tokenIds), TaskType.REASONING);
```

### 4. 自定义配置

```java
// 创建自定义 MoE 配置
DeepSeekR1Config config = new DeepSeekR1Config();
config.setVocabSize(5000);
config.setNEmbd(128);
config.setNLayer(4);
config.setNHead(4);
config.setNInner(512);
config.setNPositions(256);

// MoE 配置（R1 使用与 V3 相同的 MoE 架构）
config.setNumExperts(4);
config.setTopK(2);
config.setExpertHiddenDim(512);

// RL 训练配置（R1 特有）
config.setRlExplorationRate(0.1);
config.setRlDiscountFactor(0.99);
config.setRlClipRange(0.2);

// 创建模型
DeepSeekR1Model model = new DeepSeekR1Model("DeepSeek-R1-Custom", config);
System.out.println(model.getConfigSummary());
```

### 5. 对比不同规模的模型

```java
DeepSeekR1Model tinyModel     = DeepSeekR1Model.createTinyModel("R1-Tiny");
DeepSeekR1Model smallModel    = DeepSeekR1Model.createSmallModel("R1-Small");
DeepSeekR1Model standardModel = DeepSeekR1Model.createStandardModel("R1-Standard");

System.out.println(tinyModel);
// DeepSeekR1Model{name='R1-Tiny', params=..., nLayer=6, nEmbd=64, experts=4}
System.out.println(smallModel);
// DeepSeekR1Model{name='R1-Small', params=..., nLayer=8, nEmbd=512, experts=8}
System.out.println(standardModel);
// DeepSeekR1Model{name='R1-Standard', params=..., nLayer=12, nEmbd=768, experts=8}
```

## 📈 训练流程

### 多阶段训练

```
阶段1: 预训练（因果语言建模）
  DeepSeekR1Pretrain — SGD 优化器（内存效率优于 Adam）
    ↓
阶段2: 后训练（监督微调 SFT）
  DeepSeekR1Posttrain — 推理质量监督优化
    ↓
阶段3: 人类反馈强化学习（RLHF）
  DeepSeekR1RLHFTrainer → 委托 DeepSeekV3RLHFTrainer 执行
    ↓
阶段4: 可验证奖励强化学习（RLVR）
  DeepSeekR1RLVRTrainer — GRPO 算法，R1 独有
```

### 预训练（`DeepSeekR1Pretrain`）

```java
DeepSeekR1Model model = DeepSeekR1Model.createTinyModel("R1-Pretrain");
DeepSeekR1Dataset dataset = new DeepSeekR1Dataset(...);

DeepSeekR1Pretrain pretrainer = new DeepSeekR1Pretrain(model, dataset);
pretrainer.train();
```

核心特性：
- **SGD 优化器** — 相比 Adam 减少临时 NdArray 创建，降低内存占用
- 支持 Warmup + Cosine 衰减学习率调度
- 支持可选的**并行训练模式**（多线程）

### RLHF 训练（`DeepSeekR1RLHFTrainer`）

R1 基于 V3 底座，RLHF 核心算法由 `DeepSeekV3RLHFTrainer` 实现，R1 侧作为适配层：

```java
DeepSeekR1RLHFTrainer trainer = new DeepSeekR1RLHFTrainer(model, dataset);
trainer.configure(
    3,       // maxEpochs
    1e-5f,   // learningRate
    1.0f,    // rewardWeight（人类反馈权重）
    0.5f     // qualityWeight（MoE 质量奖励权重）
);
trainer.train();
```

**RLHF 奖励计算**：

```
综合奖励 = rewardWeight × avgHumanReward + qualityWeight × (1 - moeLoss)
```

算法：**Reward-weighted Regression（奖励加权回归）**
```
L = -reward × sum(mask × CE_loss) / sum(mask)
```

### RLVR 训练（`DeepSeekR1RLVRTrainer`）

R1 独有的训练阶段，使用 **GRPO（Group Relative Policy Optimization）** 算法，通过自动化验证器评估推理结果，无需人类标注。

**GRPO 训练流程（对标论文第 4 节）**：

```
1. 对每个问题 q，从当前策略采样 G 个输出 {o1,...,oG}
2. 验证器为每个输出计算奖励 {r1,...,rG}
3. 组内相对优势：A_i = (r_i - mean(r)) / (std(r) + ε)
4. PPO clip 目标：L = clip(A_g, -ε, +ε) × CE(logits, o_g)
5. 反向传播更新参数
```

**GRPO vs 简单策略梯度对比**：

| 维度 | 简单策略梯度 | GRPO |
|------|------------|------|
| 采样策略 | 每问题 1 个输出 | 每问题 G 个输出（组采样） |
| 基线 | 固定基线或无 | 组内均值奖励（相对优势） |
| 优势函数 | r 或 r-b | (r_i-mean)/std（组内归一化） |
| 策略更新 | 无约束 | PPO-clip 防止策略剧变 |
| 值函数网络 | 需要 | **不需要**（GRPO 的核心优势） |

**支持的验证器类型**：

| 验证器 | 类 | 说明 |
|--------|-----|------|
| 数学验证器 | `MathVerifier` | 验证数学计算结果的正确性 |
| 代码验证器 | `CodeVerifier` | 验证代码语法和逻辑正确性 |
| 逻辑验证器 | `LogicVerifier` | 验证逻辑推理过程的有效性 |

```java
// RLVR 训练
DeepSeekR1RLVRDataset rlvrDataset = new DeepSeekR1RLVRDataset(...);
DeepSeekR1RLVRTrainer rlvrTrainer = new DeepSeekR1RLVRTrainer(model, rlvrDataset);
rlvrTrainer.train();
```

### 推理引擎（`DeepSeekR1Inference`）

支持文本生成和推理执行。

## 🧪 测试验证

### 编译验证

```bash
cd tinyai-model-deepseek
mvn clean compile
```

### 运行演示

```bash
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.deepseek.r1.DeepSeekR1Demo"
```

演示程序（`DeepSeekR1Demo.java`）包含 4 个示例：

| 示例 | 说明 |
|------|------|
| **示例1** | 创建 Tiny 模型，打印架构信息和配置摘要 |
| **示例2** | 基础推理 — 标准前向传播，验证输出形状 |
| **示例3** | 带详细信息的推理 — 包含 MoE 损失的 `ReasoningResult` |
| **示例4** | 序列生成 — 自回归贪婪解码生成新 token |

## 🔍 与官方架构的差异

### 官方 DeepSeek-R1 架构

| 特性 | 官方 DeepSeek-R1 |
|------|------------------|
| **基础模型** | 基于 DeepSeek-V3-Base |
| **总参数量** | 671B 参数 |
| **激活参数** | 每 token 激活 37B |
| **架构类型** | MoE（稀疏路由 + 共享专家） |
| **层数** | 61 层 Transformer |
| **注意力机制** | Multi-Head Latent Attention (MLA) |
| **上下文长度** | 128K tokens |
| **训练方式** | 纯 RL 驱动推理能力涌现（GRPO） |
| **关键创新** | RL 直接激励推理能力，无需大量 SFT 数据 |

**官方训练流程**：
- **DeepSeek-R1-Zero**：仅使用 RL 训练，无监督微调，展示推理能力自然涌现
- **DeepSeek-R1**：多阶段训练（冷启动数据 + RL + 拒绝采样 SFT + 二次 RL）

### 核心差异对比

| 维度 | 官方 R1 | 本项目实现 | 说明 |
|------|---------|-----------|------|
| **模型规模** | 671B | 可配置（Tiny~Standard） | 教育级缩放 |
| **MoE 架构** | ✅ 稀疏路由 + 共享专家 | ✅ 可配置专家数 + 共享专家 | **架构一致** |
| **R1 复用 V3** | ✅ 基于 V3-Base | ✅ 复用 V3 TransformerBlock | **设计一致** |
| **GRPO 算法** | ✅ | ✅ `DeepSeekR1RLVRTrainer` | **算法一致** |
| **注意力** | MLA | 标准 MHA | 简化实现 |
| **推理能力** | RL 自然涌现 | RLHF/RLVR 训练器支持 | 理念一致 |
| **上下文** | 128K | 128~2048 | 硬件限制 |
| **MTP** | ✅ | ❌（MTP 是 V3 特有） | 未实现 |

### 设计理念

本项目**忠实于官方 R1 的核心设计理念**：

- ✅ **R1 复用 V3 的 MoE 架构** — `DeepSeekR1Block` 直接使用 `DeepSeekV3TransformerBlock`
- ✅ **GRPO 驱动推理** — `DeepSeekR1RLVRTrainer` 实现 GRPO 算法
- ✅ **多阶段训练** — 预训练 → SFT → RLHF → RLVR 完整流程
- ✅ **配置继承** — `DeepSeekR1Config` 继承 `DeepSeekBaseConfig`，与 V3 共享基础配置
- ✅ **无 config 转换** — R1Config 可直接传给 V3 的 TransformerBlock

**未实现的官方特性**：
- ❌ MLA (Multi-Head Latent Attention) 注意力机制
- ❌ Multi Token Prediction (MTP)（V3 特有，R1 不含）
- ❌ 128K 超长上下文支持
- ❌ 671B 参数规模

### 教育价值

| 维度 | 说明 |
|------|------|
| **架构还原** | 忠实复现 R1 复用 V3 MoE 架构的设计，无 config 转换 |
| **算法还原** | GRPO 算法对标论文实现，包含组采样 + 相对优势 + PPO-clip |
| **可运行性** | 小规模参数可在普通硬件上运行和调试 |
| **训练完整** | 包含预训练、SFT、RLHF、RLVR 完整训练流程 |
| **代码可读** | Variable 层面完整计算图，利于学习自动微分 |

## 📚 参考资料

### 技术论文
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948)
- [DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437)

### 官方资源
- 官方 GitHub：[DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)

### 相关文档
- [DeepSeek 模块主 README](../README.md)
- [DeepSeek-V3 文档](V3_README.md)

### 源代码

| 文件 | 说明 |
|------|------|
| [`DeepSeekR1Model.java`](../src/main/java/io/leavesfly/tinyai/deepseek/r1/DeepSeekR1Model.java) | 模型主类，提供推理 API |
| [`DeepSeekR1Config.java`](../src/main/java/io/leavesfly/tinyai/deepseek/r1/DeepSeekR1Config.java) | 配置类，继承 BaseConfig + RL 参数 |
| [`DeepSeekR1Block.java`](../src/main/java/io/leavesfly/tinyai/deepseek/r1/DeepSeekR1Block.java) | 主体块，直接复用 V3 MoE TransformerBlock |
| [`DeepSeekR1TokenEmbedding.java`](../src/main/java/io/leavesfly/tinyai/deepseek/r1/DeepSeekR1TokenEmbedding.java) | Token 嵌入层（无位置嵌入） |
| [`DeepSeekR1Demo.java`](../src/main/java/io/leavesfly/tinyai/deepseek/r1/DeepSeekR1Demo.java) | 完整示例代码（4 个示例） |
| [`DeepSeekR1Pretrain.java`](../src/main/java/io/leavesfly/tinyai/deepseek/r1/training/DeepSeekR1Pretrain.java) | 预训练器（SGD，支持并行） |
| [`DeepSeekR1Posttrain.java`](../src/main/java/io/leavesfly/tinyai/deepseek/r1/training/DeepSeekR1Posttrain.java) | 后训练器（SFT） |
| [`DeepSeekR1RLHFTrainer.java`](../src/main/java/io/leavesfly/tinyai/deepseek/r1/training/DeepSeekR1RLHFTrainer.java) | RLHF 训练器（委托 V3 RLHF 实现） |
| [`DeepSeekR1RLVRTrainer.java`](../src/main/java/io/leavesfly/tinyai/deepseek/r1/training/DeepSeekR1RLVRTrainer.java) | RLVR 训练器（GRPO 算法） |
| [`DeepSeekR1Inference.java`](../src/main/java/io/leavesfly/tinyai/deepseek/r1/training/DeepSeekR1Inference.java) | 推理引擎 |
| [`DeepSeekR1Dataset.java`](../src/main/java/io/leavesfly/tinyai/deepseek/r1/training/dataset/DeepSeekR1Dataset.java) | 预训练/SFT/RLHF 数据集 |
| [`DeepSeekR1RLVRDataset.java`](../src/main/java/io/leavesfly/tinyai/deepseek/r1/training/dataset/DeepSeekR1RLVRDataset.java) | RLVR 可验证奖励数据集 |
| [`MathVerifier.java`](../src/main/java/io/leavesfly/tinyai/deepseek/r1/training/verifier/MathVerifier.java) | 数学验证器 |
| [`CodeVerifier.java`](../src/main/java/io/leavesfly/tinyai/deepseek/r1/training/verifier/CodeVerifier.java) | 代码验证器 |
| [`LogicVerifier.java`](../src/main/java/io/leavesfly/tinyai/deepseek/r1/training/verifier/LogicVerifier.java) | 逻辑验证器 |
| [`DeepSeekBaseConfig.java`](../src/main/java/io/leavesfly/tinyai/deepseek/base/DeepSeekBaseConfig.java) | 基础配置（V3 和 R1 共享） |

---

<div align="center">
  <p><strong>DeepSeek-R1</strong> — MoE 架构 + GRPO 驱动推理能力涌现</p>
  <p>架构复用 V3 · GRPO 强化学习 · 可验证奖励 · 教育友好</p>
</div>
