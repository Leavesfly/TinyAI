# DeepSeek-V3 技术文档

## 📋 模型概述

DeepSeek-V3 是一个基于**混合专家模型（MoE, Mixture of Experts）**的大语言模型。V3 采用 **Pre-RMSNorm + RoPE + 纯 MoE** 架构，推理和代码生成能力通过 MoE 专家网络自然涌现，而非显式的专用模块。

该模型完全基于 TinyAI 框架的 **V2 API** 实现，所有计算在 Variable 层面完成，支持完整的自动微分。

### 核心特性

| 特性 | 说明 |
|------|------|
| **MoE 架构** | 多专家网络 + Top-K 路由，每次仅激活约 25% 参数 |
| **Pre-RMSNorm** | 使用 RMSNorm（非 LayerNorm），提升训练稳定性 |
| **RoPE 位置编码** | 旋转位置编码，在注意力层提供位置信息 |
| **任务感知路由** | 支持 REASONING、CODING、MATH、GENERAL、MULTIMODAL 5 种任务类型 |
| **MTP 辅助训练** | Multi-Token Prediction 头，V3 特有的训练辅助机制 |
| **Variable 完整性** | 所有计算在 Variable 层面，梯度完整回传 |

### 与 R1 的关系

```
DeepSeekBaseConfig（共享 MoE 基础架构配置）
    ├── DeepSeekV3Config（V3 特有：MTP、代码生成、推理增强配置）
    └── DeepSeekR1Config（R1 特有：RL 训练参数）

DeepSeekV3TransformerBlock ← R1Block 直接复用此组件
```

R1 与 V3 使用完全相同的 MoE TransformerBlock，区别仅在于训练方式：
- **V3**：标准预训练 + 后训练（SFT）+ RLHF
- **R1**：V3-Base + 纯强化学习（GRPO/RLVR），推理能力自然涌现

## 🏗️ 架构设计

### 整体架构

```
DeepSeekV3Model (extends DeepSeekModelBase)
└── DeepSeekV3Block (Module)
    ├── DeepSeekV3TokenEmbedding
    │   ├── Token 嵌入 [vocabSize, nEmbd]  (Parameter)
    │   ├── 位置嵌入 [nPositions, nEmbd]  (Parameter)
    │   └── Dropout
    ├── N × DeepSeekV3TransformerBlock
    │   ├── Pre-RMSNorm
    │   ├── MultiHeadAttention（含 RoPE）
    │   ├── 残差连接
    │   ├── Pre-RMSNorm
    │   ├── DeepSeekV3MoELayer（共享专家 + 路由专家）
    │   └── 残差连接
    ├── Final RMSNorm
    ├── Linear 输出投影 [nEmbd → vocabSize]
    └── DeepSeekV3MTPHead（可选，当 mtpDepth > 0）
```

### 数据流

```
输入 Token IDs [batch, seq]
    ↓
DeepSeekV3TokenEmbedding
    Token 嵌入 + 位置嵌入 → [batch, seq, embd]
    ↓
N × DeepSeekV3TransformerBlock（MoE 架构）
    MHA（RoPE）→ MoE FFN → 残差连接
    ↓
Final RMSNorm
    ↓
Linear 输出投影 → Logits [batch, seq, vocab]
    ↓（训练时可选）
DeepSeekV3MTPHead → 额外 token 预测损失
```

## 🔧 核心组件

### 1. 模型配置（`DeepSeekV3Config`）

继承 `DeepSeekBaseConfig`，扩展 V3 特有的推理、MTP 和代码生成配置。

**预设配置工厂方法**：

```java
// 极小型配置（默认 JVM 堆内存下可运行）
// 64维, 2层, 2头, 2专家, Top-1路由, 序列长度32, 词表1000
DeepSeekV3Config.createMicroConfig();

// 微型配置（快速测试）
// 256维, 6层, 8头, 4专家, Top-2路由, 序列长度512
DeepSeekV3Config.createTinyConfig();

// 小型配置（学习实验）
// 512维, 8层, 8头, 6专家, Top-2路由, 序列长度1024
DeepSeekV3Config.createSmallConfig();

// 标准配置（标准应用）
// 768维, 12层, 12头, 8专家, Top-2路由, 序列长度2048
DeepSeekV3Config.createStandardConfig();
```

**V3 特有配置参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `mtpDepth` | 1 | MTP 预测深度（额外预测 token 数），0 表示禁用 |
| `mtpLossWeight` | 0.3 | MTP 损失权重（相对主 LM 损失） |
| `reasoningHiddenDim` | 1536 | 推理隐藏层维度 |
| `confidenceThreshold` | 0.75 | 推理置信度阈值 |
| `enableSelfCorrection` | true | 是否启用自我纠错机制 |
| `codeQualityDim` | 4 | 代码质量评估维度数 |
| `numProgrammingLanguages` | 10 | 支持的编程语言数量 |
| `codeAnalysisHiddenDim` | 512 | 代码分析隐藏层维度 |
| `syntaxValidatorHiddenDim` | 256 | 语法验证器隐藏层维度 |

**基础 MoE 配置（继承自 `DeepSeekBaseConfig`）**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `vocabSize` | 50257 | 词汇表大小 |
| `nPositions` | 2048 | 最大序列长度 |
| `nEmbd` | 768 | 嵌入维度 |
| `nLayer` | 12 | Transformer 层数 |
| `nHead` | 12 | 注意力头数 |
| `numExperts` | 8 | 路由专家数量 |
| `numSharedExperts` | 1 | 共享专家数量（每次必激活） |
| `topK` | 2 | Top-K 专家选择数 |
| `expertHiddenDim` | 3072 | 专家隐藏层维度 |
| `loadBalanceLossWeight` | 0.01 | 负载均衡损失权重 |

### 2. 主体块（`DeepSeekV3Block`）

负责 V3 的完整前向传播，支持普通前向和带详细信息的前向传播。

**前向传播模式**：

| 方法 | 说明 | 返回值 |
|------|------|--------|
| `forward(Variable... inputs)` | 标准前向传播 | `Variable` logits |
| `forwardWithDetails(Variable tokenIds)` | 带 MoE 损失和隐藏状态 | `DetailedForwardResult` |

`DetailedForwardResult` 包含三个字段：
- **`logits`** — 最终 logits 输出
- **`avgMoELoss`** — 各 Transformer 层 MoE 负载均衡损失的均值
- **`hiddenStates`** — Final RMSNorm 后的隐藏状态（供 MTP Head 使用）

### 3. Token 嵌入层（`DeepSeekV3TokenEmbedding`）

完全在 Variable 层面实现，确保梯度正确回传：

```java
// Token 嵌入：使用 indexSelect 算子
Variable flatIds = tokenIds.reshape(Shape.of(batchSize * seqLen));
Variable flatEmbeds = tokenEmbedParam.indexSelect(0, flatIds);
Variable tokenEmbeds = flatEmbeds.reshape(Shape.of(batchSize, seqLen, nEmbd));

// 位置嵌入：repeat 扩展 batch 维度
Variable posEmbeds = posEmbedParam.indexSelect(0, posIds);
Variable posEmbeds3D = posEmbeds.reshape(Shape.of(1, seqLen, nEmbd));
Variable posExpanded = posEmbeds3D.repeat(batchSize, 1, 1);

// 合并并 Dropout
Variable combined = tokenEmbeds.add(posExpanded);
return dropout.forward(combined);
```

**Variable 算子使用**：
- `indexSelect` — 索引选择嵌入向量
- `reshape` — 形状变换
- `repeat` — 维度重复扩展
- `add` — 嵌入合并

### 4. MTP Head（`DeepSeekV3MTPHead`）

DeepSeek-V3 论文的核心创新，在训练时预测额外的多个 token，提升模型的预测能力。

- 与主 LM 头**共享** Token 嵌入权重和输出投影层
- 仅在 `mtpDepth > 0` 时初始化
- 训练时计算 MTP 辅助损失（权重由 `mtpLossWeight` 控制）

### 5. 模型类（`DeepSeekV3Model`）

提供高层 API：

| 方法 | 说明 |
|------|------|
| `predict(Variable tokenIds)` | 标准预测，返回 logits |
| `predictWithDetails(Variable tokenIds)` | 带 MoE 损失和隐藏状态的详细预测 |
| `printModelInfo()` | 打印模型架构和配置详情 |
| `getConfigSummary()` | 获取配置摘要字符串 |

**工厂方法**：

```java
DeepSeekV3Model.createMicroModel("name");     // 极小型（默认堆内存可跑）
DeepSeekV3Model.createTinyModel("name");      // 微型
DeepSeekV3Model.createSmallModel("name");     // 小型
DeepSeekV3Model.createStandardModel("name"); // 标准
```

## 💻 使用示例

### 1. 创建模型并推理

```java
import io.leavesfly.tinyai.deepseek.v3.*;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;

// 创建小型模型
DeepSeekV3Model model = DeepSeekV3Model.createSmallModel("DeepSeek-V3-Small");
model.printModelInfo();
System.out.println(model.getConfigSummary());

// 标准推理
float[][] input = {{1, 15, 23, 42, 100, 200}};
Variable logits = model.predict(new Variable(NdArray.of(input)));
System.out.println("输出形状: " + logits.getValue().getShape());
// 输出形状: [1, 6, 50257]
```

### 2. 带详细信息的推理（获取 MoE 损失）

```java
// 获取详细推理结果（含 MoE 负载均衡损失和隐藏状态）
DeepSeekV3Block.DetailedForwardResult result =
    model.predictWithDetails(new Variable(NdArray.of(input)));

System.out.println("MoE 负载均衡损失: " + result.avgMoELoss);
System.out.println("Logits 形状: " + result.logits.getValue().getShape());
System.out.println("隐藏状态形状: " + result.hiddenStates.getValue().getShape());
```

### 3. 自定义配置

```java
// 自定义配置
DeepSeekV3Config config = new DeepSeekV3Config();

// 基础架构
config.setVocabSize(10000);
config.setNEmbd(256);
config.setNLayer(4);
config.setNHead(4);
config.setNInner(1024);
config.setNPositions(512);

// MoE 配置
config.setNumExperts(4);
config.setTopK(2);
config.setExpertHiddenDim(1024);

// V3 特有：MTP 配置
config.setMtpDepth(1);         // 启用 MTP
config.setMtpLossWeight(0.3);  // MTP 损失权重

// V3 特有：代码生成配置
config.setNumProgrammingLanguages(10);

// 创建模型
DeepSeekV3Model model = new DeepSeekV3Model("custom-v3", config);
System.out.println(model);
// DeepSeekV3Model{name='custom-v3', params=..., activeParams=..., nLayer=4, nEmbd=256, experts=4}
```

### 4. 通过推理引擎生成序列

```java
import io.leavesfly.tinyai.deepseek.v3.training.DeepSeekV3Inference;

DeepSeekV3Inference inference = new DeepSeekV3Inference(model);

// 贪婪解码
int[] promptIds = {1, 15, 23, 42};
DeepSeekV3Inference.GenerationResult result =
    inference.generateGreedy(promptIds, 20);

// Temperature 采样
DeepSeekV3Inference.GenerationResult sampledResult =
    inference.generateWithTemperature(promptIds, 20, 0.8f);
```

## 📊 模型规格

| 配置 | 嵌入维度 | 层数 | 注意力头 | 专家数 | Top-K | 序列长度 | 说明 |
|------|---------|------|---------|--------|-------|---------|------|
| **Micro** | 64 | 2 | 2 | 2 | 1 | 32 | 默认 JVM 堆内存可跑 |
| **Tiny** | 256 | 6 | 8 | 4 | 2 | 512 | 快速测试 |
| **Small** | 512 | 8 | 8 | 6 | 2 | 1024 | 学习实验 |
| **Standard** | 768 | 12 | 12 | 8 | 2 | 2048 | 标准应用 |

## 🔬 训练流程

### 多阶段训练

```
阶段1: 预训练（因果语言建模 + MTP 辅助损失）
  DeepSeekV3Pretrain — Adam 优化器，Warmup + Cosine 衰减
    ↓
阶段2: 后训练（监督微调 SFT）
  DeepSeekV3Posttrain — Answer-only Loss Mask，早停机制
    ↓
阶段3: 强化学习（RLHF）
  DeepSeekV3RLHFTrainer — Reward-weighted Regression
```

### 预训练（`DeepSeekV3Pretrain`）

```java
DeepSeekV3Model model = DeepSeekV3Model.createTinyModel("V3-Pretrain");
DeepSeekV3Dataset dataset = new DeepSeekV3Dataset(...);

DeepSeekV3Pretrain trainer = new DeepSeekV3Pretrain(model, dataset);
trainer.configure(
    10,        // maxEpochs
    2.5e-4f,   // learningRate（支持 Warmup + Cosine 衰减）
    2000,      // warmupSteps
    1.0f       // maxGradNorm
);
trainer.train();
```

核心特性：
- **Adam 优化器** — β1=0.9, β2=0.999
- **MoE 负载均衡损失** — 权重由 `loadBalanceLossWeight` 控制
- **Warmup + Cosine 衰减** — 学习率调度
- **梯度裁剪** — 防止梯度爆炸

### 后训练（`DeepSeekV3Posttrain`）

```java
DeepSeekV3Posttrain posttrain = new DeepSeekV3Posttrain(model, trainDataset, valDataset);
posttrain.train();
```

核心特性：
- **Answer-only Loss Mask** — 仅对 assistant 回复区域计算损失
- **早停机制** — `patience` 轮无改善则停止
- **低学习率** — 默认 2.5e-5（比预训练低 10 倍）

### RLHF 训练（`DeepSeekV3RLHFTrainer`）

算法：**Reward-weighted Regression（奖励加权回归）**

```
L = -reward × sum(mask × CE_loss) / sum(mask)
```

```java
DeepSeekV3RLHFTrainer rlhfTrainer = new DeepSeekV3RLHFTrainer(model, dataset);
rlhfTrainer.configure(
    3,       // maxEpochs
    1e-5f,   // learningRate
    1.0f,    // rewardWeight（人类反馈权重）
    0.5f     // moeQualityWeight（MoE 质量权重）
);
rlhfTrainer.train();
```

> R1 的 RLHF 训练通过委托本类实现（`DeepSeekR1RLHFTrainer` → `DeepSeekV3RLHFTrainer`）。

### 推理引擎（`DeepSeekV3Inference`）

继承 `DeepSeekBaseInference`，支持 4 种生成策略：

| 策略 | 方法 | 说明 |
|------|------|------|
| 贪婪解码 | `generateGreedy(promptIds, maxNewTokens)` | 每步选概率最高的 token |
| Temperature 采样 | `generateWithTemperature(promptIds, maxNewTokens, temp)` | 控制生成随机性 |
| Top-K 采样 | `generateTopK(promptIds, maxNewTokens, k)` | 从 Top-K 候选中采样 |
| Top-P 采样 | `generateTopP(promptIds, maxNewTokens, p)` | 核采样（累积概率） |

## 🧪 运行演示

### 编译项目

```bash
cd tinyai-model-deepseek
mvn clean compile
```

### 运行演示程序

```bash
# V3 模型演示（包含模型创建 + MoE 分析）
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.deepseek.v3.DeepSeekV3Demo"
```

演示程序（`DeepSeekV3Demo.java`）包含 2 个示例：

| 示例 | 说明 |
|------|------|
| **示例1** | 创建 Small 模型，打印架构信息和配置摘要 |
| **示例5** | MoE 分析 — 打印专家配置、参数激活率，执行推理 |

## 🔍 关键设计说明

### 纯 MoE 架构

V3 不包含显式的推理模块或代码模块，推理和代码生成能力**通过 MoE 专家网络自然涌现**。这与旧版本（有独立的 `ReasoningBlock`/`CodeBlock`）有本质区别。

### MTP 辅助训练机制

MTP Head 与主模型**共享权重**（Token 嵌入 + 输出投影），训练时预测额外的 token，提升模型的序列理解能力。推理时 MTP Head 不参与计算。

### 共享专家（`numSharedExperts`）

DeepSeekMoE 的核心创新：
- **路由专家** — Top-K 选择，每次激活 `topK` 个
- **共享专家** — 每次必激活，提供稳定的通用知识基础

## 📚 参考资料

### 技术论文
- [DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437)
- [DeepSeekMoE: Towards Ultimate Expert Specialization](https://arxiv.org/abs/2401.06066)

### 官方资源
- [DeepSeek-V3 GitHub](https://github.com/deepseek-ai/DeepSeek-V3)

### 相关文档
- [DeepSeek 模块主 README](../README.md)
- [DeepSeek-R1 文档](r1_README.md)

### 源代码

| 文件 | 说明 |
|------|------|
| [`DeepSeekV3Model.java`](../src/main/java/io/leavesfly/tinyai/deepseek/v3/DeepSeekV3Model.java) | 模型主类 |
| [`DeepSeekV3Config.java`](../src/main/java/io/leavesfly/tinyai/deepseek/v3/DeepSeekV3Config.java) | 配置类（V3 特有参数） |
| [`DeepSeekV3Block.java`](../src/main/java/io/leavesfly/tinyai/deepseek/v3/DeepSeekV3Block.java) | 主体块（含 MTP Head） |
| [`DeepSeekV3TransformerBlock.java`](../src/main/java/io/leavesfly/tinyai/deepseek/v3/DeepSeekV3TransformerBlock.java) | Transformer 块（R1 复用此组件） |
| [`DeepSeekV3MoELayer.java`](../src/main/java/io/leavesfly/tinyai/deepseek/v3/DeepSeekV3MoELayer.java) | MoE 专家层 |
| [`DeepSeekV3MTPHead.java`](../src/main/java/io/leavesfly/tinyai/deepseek/v3/DeepSeekV3MTPHead.java) | Multi-Token Prediction 头 |
| [`DeepSeekV3TokenEmbedding.java`](../src/main/java/io/leavesfly/tinyai/deepseek/v3/DeepSeekV3TokenEmbedding.java) | Token + 位置嵌入层 |
| [`DeepSeekV3Demo.java`](../src/main/java/io/leavesfly/tinyai/deepseek/v3/DeepSeekV3Demo.java) | 演示程序 |
| [`DeepSeekV3Pretrain.java`](../src/main/java/io/leavesfly/tinyai/deepseek/v3/training/DeepSeekV3Pretrain.java) | 预训练器 |
| [`DeepSeekV3Posttrain.java`](../src/main/java/io/leavesfly/tinyai/deepseek/v3/training/DeepSeekV3Posttrain.java) | 后训练器（SFT） |
| [`DeepSeekV3RLHFTrainer.java`](../src/main/java/io/leavesfly/tinyai/deepseek/v3/training/DeepSeekV3RLHFTrainer.java) | RLHF 训练器 |
| [`DeepSeekV3Inference.java`](../src/main/java/io/leavesfly/tinyai/deepseek/v3/training/DeepSeekV3Inference.java) | 推理引擎 |
| [`DeepSeekBaseConfig.java`](../src/main/java/io/leavesfly/tinyai/deepseek/base/DeepSeekBaseConfig.java) | 基础配置（V3 和 R1 共享） |

---

<div align="center">
  <p><strong>DeepSeek-V3</strong> — Pre-RMSNorm + RoPE + 纯 MoE 架构</p>
  <p>多专家路由 · MTP 辅助训练 · 完整训练流程 · 纯 Java 实现</p>
</div>
