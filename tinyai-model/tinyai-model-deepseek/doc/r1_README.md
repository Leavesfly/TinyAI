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

根据官方论文，R1 与 V3 共享相同的 MoE 基础架构（671B 参数，8 专家，Top-2 路由），核心区别在于**训练方式**：V3 采用标准预训练 + SFT，而 R1 在 V3-Base 之上通过**纯强化学习（RL）**驱动推理能力的自然涌现。

本实现忠实复现了这一设计理念：**R1 复用 V3 的 MoE TransformerBlock**，仅扩展 RL 训练配置。

### 核心特性

| 特性 | 说明 |
|------|------|
| **MoE 架构** | 复用 V3 的 MoE TransformerBlock，支持多专家路由 |
| **RL 驱动推理** | 通过 RLHF/RLVR 训练器优化推理能力 |
| **任务感知** | 支持 REASONING、MATH 等多种任务类型 |
| **序列生成** | 贪婪解码的自回归文本生成 |
| **Variable 完整性** | 所有计算在 Variable 层面完成，梯度正确回传 |
| **多阶段训练** | 预训练 → 后训练(SFT) → 强化学习(RL) 完整流程 |

### 模型规格

| 配置 | 嵌入维度 | 层数 | 注意力头 | 专家数 | Top-K | 序列长度 |
|------|---------|------|---------|--------|-------|---------|
| **Tiny** | 256 | 6 | 8 | 4 | 2 | 512 |
| **Small** | 512 | 8 | 8 | 8 | 2 | 1024 |
| **Standard** | 768 | 12 | 12 | 8 | 2 | 2048 |

## 🏗️ 架构设计

### 整体架构

```
DeepSeek-R1 模型架构（MoE）
├── DeepSeekR1Model (extends Model)
│   └── DeepSeekR1Block (extends Module)
│       ├── DeepSeekR1TokenEmbedding
│       │   ├── Token 嵌入 [vocabSize, nEmbd]
│       │   ├── 位置嵌入 [nPositions, nEmbd]
│       │   └── Dropout 正则化
│       ├── DeepSeekV3TransformerBlock × N 层（✅ 复用 V3 的 MoE 组件）
│       │   ├── Pre-LayerNorm
│       │   ├── 多头自注意力 (MHA)
│       │   ├── 残差连接
│       │   ├── Pre-LayerNorm
│       │   ├── MoE 前馈网络（多专家 + Top-K 路由）
│       │   └── 残差连接
│       ├── Final LayerNorm
│       └── 输出投影层 Linear [nEmbd → vocabSize]
└── DeepSeekR1Config (extends DeepSeekBaseConfig)
    ├── 基础 MoE 架构配置（继承自 V3）
    └── RL 训练配置（R1 特有）
```

### R1 与 V3 的关系

```
DeepSeekBaseConfig（共享基础配置）
    ├── DeepSeekV3Config（V3 特有：代码生成、MTP 等）
    └── DeepSeekR1Config（R1 特有：RL 训练参数）

关键设计：R1Block 内部将 R1Config 转换为 V3Config，
         直接复用 DeepSeekV3TransformerBlock 组件
```

### 数据流

```
输入 Token IDs [batch, seq]
    ↓
DeepSeekR1TokenEmbedding
    Token 嵌入 + 位置嵌入 → [batch, seq, embd]
    ↓
DeepSeekV3TransformerBlock × N（MoE 架构）
    MHA → MoE FFN → 残差连接
    ↓
Final LayerNorm
    ↓
Linear 输出投影 → Logits [batch, seq, vocab]
```

## 🔧 核心组件

### 1. 模型配置 (`DeepSeekR1Config`)

继承 `DeepSeekBaseConfig`，与 V3 共享 MoE 基础架构配置，仅扩展 RL 训练参数。

**预设配置工厂方法**：

```java
// 微型配置（快速测试）：256维, 6层, 8头, 4专家, Top-2, 512序列
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
| `rlClipRange` | 0.2 | PPO clip 范围 |
| `rlEntropyCoefficient` | 0.01 | 熵系数（鼓励探索） |
| `rlMaxGradNorm` | 1.0 | 最大梯度范数 |
| `rlBatchSize` | 32 | RL 训练批次大小 |
| `rlRewardScale` | 1.0 | 奖励缩放因子 |

### 2. 主体块 (`DeepSeekR1Block`)

R1 的核心计算模块，**直接复用 V3 的 MoE TransformerBlock**。

初始化时将 R1Config 转换为 V3Config，复用 V3 组件：

```java
// R1Block 内部实现
DeepSeekV3Config v3Config = convertToV3Config(config);
DeepSeekV3TransformerBlock block = new DeepSeekV3TransformerBlock(name, v3Config);
```

**支持三种前向传播模式**：

| 方法 | 说明 | 返回值 |
|------|------|--------|
| `forward(Variable... inputs)` | 标准前向传播 | `Variable` logits |
| `forward(Variable tokenIds, TaskType taskType)` | 带任务类型的前向传播 | `Variable` logits |
| `forwardWithDetails(Variable tokenIds, TaskType taskType)` | 带 MoE 损失的详细输出 | `DetailedForwardResult` |

`DetailedForwardResult` 包含最终 logits 和平均 MoE 负载均衡损失。

### 3. Token 嵌入层 (`DeepSeekR1TokenEmbedding`)

完全在 Variable 层面实现，确保梯度正确回传：

```java
// Token 嵌入：使用 indexSelect 算子
Variable flatIds = tokenIds.reshape(Shape.of(batchSize * sequenceLength));
Variable flatEmbeds = tokenEmbedParam.indexSelect(0, flatIds);
Variable tokenEmbeds = flatEmbeds.reshape(Shape.of(batchSize, sequenceLength, embeddingDim));

// 位置嵌入：返回 [1, seq, embd]，依赖广播机制自动扩展
Variable posEmbeds = posEmbedParam.indexSelect(0, posIds);
Variable posEmbeds3D = posEmbeds.reshape(Shape.of(1, sequenceLength, embeddingDim));

// 合并嵌入并应用 Dropout
Variable combined = tokenEmbeds.add(positionEmbeds);
return dropout.forward(combined);
```

**Variable 算子使用**：
- **`indexSelect`** — 索引选择（替代手动 NdArray 操作）
- **`reshape`** — 形状变换
- **`add`** — 嵌入合并（利用广播机制）

### 4. 模型类 (`DeepSeekR1Model`)

提供高层 API，支持多种推理模式：

| 方法 | 说明 |
|------|------|
| `predict(Variable tokenIds)` | 标准预测，返回 logits |
| `predict(Variable tokenIds, TaskType taskType)` | 带任务类型的预测 |
| `performReasoning(Variable tokenIds)` | 推理任务，返回 `ReasoningResult` |
| `solveMath(Variable tokenIds)` | 数学计算任务 |
| `generateSequence(NdArray promptIds, int maxNewTokens)` | 贪婪解码序列生成 |

`ReasoningResult` 包含三个字段：
- **`logits`** — 最终 logits 输出
- **`moeLoss`** — MoE 负载均衡损失
- **`taskType`** — 任务类型（REASONING / MATH 等）

**工厂方法**：

```java
DeepSeekR1Model.createTinyModel("name");      // 微型模型
DeepSeekR1Model.createSmallModel("name");     // 小型模型
DeepSeekR1Model.createStandardModel("name");  // 标准模型
```

## 💻 使用示例

### 1. 创建模型并推理

```java
// 创建微型模型（用于快速测试）
DeepSeekR1Model model = DeepSeekR1Model.createTinyModel("DeepSeek-R1-Tiny");
model.printModelInfo();

// 准备输入 [batch_size=2, seq_len=8]
DeepSeekR1Config config = model.getConfig();
float[][] inputData = new float[2][8];
for (int b = 0; b < 2; b++) {
    for (int s = 0; s < 8; s++) {
        inputData[b][s] = (float) (Math.random() * config.getVocabSize());
    }
}

// 执行推理
Variable logits = model.predict(new Variable(NdArray.of(inputData)));
System.out.println("输出形状: " + logits.getValue().getShape());
// 输出形状: [2, 8, 50257]
```

### 2. 带详细信息的推理

```java
// 执行推理任务（返回 MoE 损失等详细信息）
NdArray tokenIds = NdArray.of(inputData);
DeepSeekR1Model.ReasoningResult result = model.performReasoning(new Variable(tokenIds));

System.out.println("任务类型: " + result.taskType.getDescription());
System.out.println("MoE 损失: " + result.moeLoss);
System.out.println("输出形状: " + result.logits.getValue().getShape());
```

### 3. 序列生成

```java
// 自回归文本生成（贪婪解码）
float[][] promptData = new float[][]{{1, 15, 23, 42, 100}};
NdArray promptIds = NdArray.of(promptData);
int maxNewTokens = 10;

NdArray generated = model.generateSequence(promptIds, maxNewTokens);
System.out.println("生成序列形状: " + generated.getShape());
// 生成序列形状: [1, 15]

// 打印生成的 token
for (int i = 0; i < 15; i++) {
    System.out.printf("%.0f ", generated.get(0, i));
}
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
config.setNumExperts(8);
config.setTopK(2);
config.setExpertHiddenDim(512);

// RL 训练配置（R1 特有）
config.setRlExplorationRate(0.1);
config.setRlDiscountFactor(0.99);

// 创建模型
DeepSeekR1Model model = new DeepSeekR1Model("DeepSeek-R1-Custom", config);
System.out.println(model.getConfigSummary());
```

### 5. 对比不同规模的模型

```java
DeepSeekR1Model tinyModel = DeepSeekR1Model.createTinyModel("R1-Tiny");
DeepSeekR1Model smallModel = DeepSeekR1Model.createSmallModel("R1-Small");
DeepSeekR1Model standardModel = DeepSeekR1Model.createStandardModel("R1-Standard");

// 打印对比信息
System.out.println(tinyModel);
// DeepSeekR1Model{name='R1-Tiny', params=..., nLayer=6, nEmbd=256, experts=4}
System.out.println(smallModel);
// DeepSeekR1Model{name='R1-Small', params=..., nLayer=8, nEmbd=512, experts=8}
System.out.println(standardModel);
// DeepSeekR1Model{name='R1-Standard', params=..., nLayer=12, nEmbd=768, experts=8}
```

## 📈 训练流程

### 多阶段训练

```
阶段1: 预训练（因果语言建模）
  DeepSeekR1PretrainV2 — 继承 DeepSeekBasePretrain，使用 SGD 优化器
    ↓
阶段2: 后训练（监督微调 SFT）
  DeepSeekR1Posttrain — 推理质量优化
    ↓
阶段3: 强化学习（RL）
  DeepSeekR1RLHFTrainer — 人类反馈强化学习
  DeepSeekR1RLVRTrainer — 可验证奖励强化学习
```

### 预训练 (`DeepSeekR1PretrainV2`)

继承 `DeepSeekBasePretrain`，实现因果语言建模。R1 的预训练使用 SGD 优化器以减少内存占用：

```java
DeepSeekR1Model model = DeepSeekR1Model.createTinyModel("R1-Pretrain");
DeepSeekR1Dataset dataset = new DeepSeekR1Dataset(...);

DeepSeekR1PretrainV2 pretrainer = new DeepSeekR1PretrainV2(model, dataset);
pretrainer.train();
```

预训练损失计算流程：
1. 前向传播获取 logits `[batch, seq, vocab]`
2. Reshape 为 2D `[batch*seq, vocab]`
3. 使用 SoftmaxCE 计算交叉熵损失
4. 反向传播 + 梯度裁剪 + SGD 更新

### RLHF 训练 (`DeepSeekR1RLHFTrainer`)

通过人类反馈的强化学习优化推理质量：

```java
DeepSeekR1RLHFTrainer trainer = new DeepSeekR1RLHFTrainer(model, dataset);

// 配置训练参数
trainer.configure(
    3,       // maxEpochs
    1e-5f,   // learningRate
    1.0f,    // rewardWeight（人类反馈权重）
    0.5f     // qualityWeight（MoE 质量奖励权重）
);

// 开始训练
trainer.train();
```

**RLHF 奖励计算**：

```
综合奖励 = rewardWeight × avgHumanReward + qualityWeight × (1 - moeLoss)
```

- **人类反馈奖励** — 来自标注数据的奖励信号
- **MoE 质量奖励** — MoE 负载均衡损失越小，奖励越大（`1 - moeLoss`）
- 训练目标：最大化综合奖励（损失 = 负奖励）

训练过程包含梯度裁剪（默认 `maxGradNorm=0.5`）和检查点保存。

### RLVR 训练 (`DeepSeekR1RLVRTrainer`)

基于可验证奖励的强化学习训练，通过自动化验证器评估推理结果质量，无需人类标注。

### 推理引擎 (`DeepSeekR1Inference`)

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

演示程序（`DeepSeekR1Demo.java`）包含 6 个示例：

| 示例 | 说明 |
|------|------|
| **示例1** | 创建模型并打印架构信息和配置摘要 |
| **示例2** | 基础推理 — 标准前向传播，验证输出形状 |
| **示例3** | 带详细信息的推理 — 包含 MoE 损失的 `ReasoningResult` |
| **示例4** | 序列生成 — 自回归贪婪解码生成新 token |
| **示例5** | 自定义配置模型 — 灵活的 MoE + RL 配置 |
| **示例6** | 对比不同规模的模型 — Tiny / Small / Standard 参数对比 |

## 🔍 与官方架构的差异

### 官方 DeepSeek-R1 架构

根据 [DeepSeek-R1 论文 (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948)：

| 特性 | 官方 DeepSeek-R1 |
|------|------------------|
| **基础模型** | 基于 DeepSeek-V3-Base |
| **总参数量** | 671B 参数 |
| **激活参数** | 每 token 激活 37B |
| **架构类型** | MoE（8 专家，Top-2 路由） |
| **层数** | 61 层 Transformer |
| **注意力机制** | Multi-Head Latent Attention (MLA) |
| **上下文长度** | 128K tokens |
| **训练方式** | 纯 RL 驱动推理能力涌现 |
| **关键创新** | RL 直接激励推理能力，无需大量 SFT 数据 |

**官方训练流程**：
- **DeepSeek-R1-Zero**：仅使用 RL 训练，无监督微调，展示推理能力自然涌现
- **DeepSeek-R1**：多阶段训练（冷启动数据 + RL + 拒绝采样 SFT + 二次 RL）

### 核心差异对比

| 维度 | 官方 R1 | 本项目实现 | 说明 |
|------|---------|-----------|------|
| **模型规模** | 671B | 可配置（Tiny~Standard） | 教育级缩放 |
| **MoE 架构** | ✅ 8 专家 | ✅ 可配置专家数 | **架构一致** |
| **R1 复用 V3** | ✅ 基于 V3-Base | ✅ 复用 V3 TransformerBlock | **设计一致** |
| **注意力** | MLA | 标准 MHA | 简化实现 |
| **推理能力** | RL 自然涌现 | RLHF/RLVR 训练器支持 | 理念一致，规模不同 |
| **上下文** | 128K | 512~2048 | 硬件限制 |
| **MTP** | ✅ | ❌ | 未实现 |

### 设计理念

本项目**忠实于官方 R1 的核心设计理念**：

- ✅ **R1 复用 V3 的 MoE 架构** — `DeepSeekR1Block` 内部直接使用 `DeepSeekV3TransformerBlock`
- ✅ **RL 驱动推理** — 提供 RLHF 和 RLVR 两种强化学习训练器
- ✅ **多阶段训练** — 预训练 → SFT → RL 完整流程
- ✅ **配置继承** — `DeepSeekR1Config` 继承 `DeepSeekBaseConfig`，与 V3 共享基础配置

**未实现的官方特性**：
- ❌ MLA (Multi-Head Latent Attention) 注意力机制
- ❌ Multi Token Prediction (MTP)
- ❌ 128K 超长上下文支持
- ❌ 671B 参数规模

### 教育价值

| 维度 | 说明 |
|------|------|
| **架构还原** | 忠实复现 R1 复用 V3 MoE 架构的设计 |
| **可运行性** | 小规模参数可在普通硬件上运行和调试 |
| **训练完整** | 包含预训练、SFT、RLHF、RLVR 完整训练流程 |
| **代码可读** | Variable 层面完整计算图，利于学习自动微分 |
| **实验友好** | 灵活的配置系统，易于修改和扩展 |

## 📚 参考资料

### 技术论文
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948)
- [DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437)

### 官方资源
- 官方 GitHub：[DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)
- 官方 API：[DeepSeek API Docs](https://api-docs.deepseek.com/)

### 相关文档
- [DeepSeek 模块主 README](../README.md)
- [DeepSeek-V3 文档](V3_README.md)

### 源代码

| 文件 | 说明 |
|------|------|
| [`DeepSeekR1Model.java`](../src/main/java/io/leavesfly/tinyai/deepseek/r1/DeepSeekR1Model.java) | 模型主类，提供推理和生成 API |
| [`DeepSeekR1Config.java`](../src/main/java/io/leavesfly/tinyai/deepseek/r1/DeepSeekR1Config.java) | 配置类，继承 BaseConfig + RL 参数 |
| [`DeepSeekR1Block.java`](../src/main/java/io/leavesfly/tinyai/deepseek/r1/DeepSeekR1Block.java) | 主体块，复用 V3 MoE TransformerBlock |
| [`DeepSeekR1TokenEmbedding.java`](../src/main/java/io/leavesfly/tinyai/deepseek/r1/DeepSeekR1TokenEmbedding.java) | Token + 位置嵌入层 |
| [`DeepSeekR1Demo.java`](../src/main/java/io/leavesfly/tinyai/deepseek/r1/DeepSeekR1Demo.java) | 完整示例代码（6 个示例） |
| [`DeepSeekR1PretrainV2.java`](../src/main/java/io/leavesfly/tinyai/deepseek/r1/training/DeepSeekR1PretrainV2.java) | 预训练器（因果语言建模） |
| [`DeepSeekR1Posttrain.java`](../src/main/java/io/leavesfly/tinyai/deepseek/r1/training/DeepSeekR1Posttrain.java) | 后训练器（SFT 微调） |
| [`DeepSeekR1RLHFTrainer.java`](../src/main/java/io/leavesfly/tinyai/deepseek/r1/training/DeepSeekR1RLHFTrainer.java) | RLHF 强化学习训练器 |
| [`DeepSeekR1RLVRTrainer.java`](../src/main/java/io/leavesfly/tinyai/deepseek/r1/training/DeepSeekR1RLVRTrainer.java) | RLVR 可验证奖励训练器 |
| [`DeepSeekR1Inference.java`](../src/main/java/io/leavesfly/tinyai/deepseek/r1/training/DeepSeekR1Inference.java) | 推理引擎 |

---

<div align="center">
  <p><strong>DeepSeek-R1</strong> — MoE 架构 + RL 驱动推理</p>
  <p>架构复用 V3 | 强化学习训练 | 教育友好</p>
</div>
