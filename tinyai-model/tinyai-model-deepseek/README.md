# DeepSeek 模型实现

基于 TinyAI 框架实现的 DeepSeek 系列大语言模型，包含 **DeepSeek-V3** 和 **DeepSeek-R1** 两个模型。100% 基于 **nnet v2 API**，采用 **Pre-RMSNorm + RoPE + 纯 MoE** 架构，所有计算在 Variable 层面完成，支持完整的自动微分。

## ✨ 核心特点

- **共享 MoE 基础架构** — R1 和 V3 共享 `DeepSeekBaseConfig` 和 `DeepSeekV3TransformerBlock`，最大化代码复用
- **Pre-RMSNorm + RoPE** — 对标官方架构，使用 RMSNorm 和旋转位置编码
- **纯 MoE 架构** — 共享专家 + Top-K 路由专家，推理/代码能力通过专家网络自然涌现
- **GRPO 强化学习** — R1 使用 Group Relative Policy Optimization，对标论文算法
- **MTP 辅助训练** — V3 特有的 Multi-Token Prediction 训练机制
- **Variable 完整性** — 所有计算在 Variable 层面，梯度完整回传

## 📁 文件结构

```
tinyai-model-deepseek/
├── src/main/java/io/leavesfly/tinyai/deepseek/
│   ├── base/                                   # 共享基类层
│   │   ├── DeepSeekBaseConfig.java             # MoE 基础配置（V3/R1 共享）
│   │   ├── DeepSeekModelBase.java              # 模型基类
│   │   ├── DeepSeekTokenEmbeddingBase.java     # 嵌入层基类
│   │   ├── TaskType.java                       # 5 种任务类型枚举
│   │   ├── dataset/
│   │   │   └── DeepSeekBaseDataset.java        # 数据集基类
│   │   ├── inference/
│   │   │   └── DeepSeekBaseInference.java      # 推理引擎基类
│   │   └── utils/
│   │       ├── TrainingMonitor.java            # 训练监控工具
│   │       └── CheckpointManager.java          # 检查点管理器
│   ├── v3/                                     # DeepSeek-V3
│   │   ├── DeepSeekV3Config.java               # V3 配置（继承 BaseConfig）
│   │   ├── DeepSeekV3TokenEmbedding.java       # Token + 位置嵌入（Variable 层面）
│   │   ├── DeepSeekV3Attention.java            # 注意力模块（含 RoPE）
│   │   ├── DeepSeekV3TransformerBlock.java     # Transformer 块（R1 复用此组件）
│   │   ├── DeepSeekV3MoELayer.java             # 混合专家层（批量计算）
│   │   ├── DeepSeekV3MTPHead.java              # Multi-Token Prediction 头（V3 特有）
│   │   ├── DeepSeekV3Block.java                # V3 主体块（含 MTP Head）
│   │   ├── DeepSeekV3Model.java                # V3 模型类
│   │   ├── DeepSeekV3Demo.java                 # 演示程序
│   │   └── training/
│   │       ├── DeepSeekV3Dataset.java          # 数据集
│   │       ├── DeepSeekV3Pretrain.java         # 预训练器（Adam）
│   │       ├── DeepSeekV3Posttrain.java        # 后训练器（SFT）
│   │       ├── DeepSeekV3RLHFTrainer.java      # RLHF 训练器（奖励加权回归）
│   │       ├── DeepSeekV3Inference.java        # 推理引擎（4 种生成策略）
│   │       └── DeepSeekV3TrainDemo.java        # 训练演示
│   └── r1/                                     # DeepSeek-R1
│       ├── DeepSeekR1Config.java               # R1 配置（继承 BaseConfig）
│       ├── DeepSeekR1TokenEmbedding.java       # Token 嵌入（无位置嵌入）
│       ├── DeepSeekR1Block.java                # R1 主体块（复用 V3 TransformerBlock）
│       ├── DeepSeekR1Model.java                # R1 模型类
│       ├── DeepSeekR1Demo.java                 # 演示程序
│       └── training/
│           ├── dataset/
│           │   ├── DeepSeekR1Dataset.java      # 预训练/SFT/RLHF 数据集
│           │   └── DeepSeekR1RLVRDataset.java  # RLVR 可验证奖励数据集
│           ├── verifier/
│           │   ├── Verifier.java               # 验证器接口
│           │   ├── VerificationResult.java     # 验证结果
│           │   ├── MathVerifier.java           # 数学验证器
│           │   ├── CodeVerifier.java           # 代码验证器
│           │   └── LogicVerifier.java          # 逻辑验证器
│           ├── DeepSeekR1Pretrain.java         # 预训练器（SGD，支持并行）
│           ├── DeepSeekR1Posttrain.java        # 后训练器（SFT）
│           ├── DeepSeekR1RLHFTrainer.java      # RLHF 训练器（委托 V3）
│           ├── DeepSeekR1RLVRTrainer.java      # RLVR 训练器（GRPO 算法）
│           ├── DeepSeekR1Inference.java        # 推理引擎
│           └── demo/
│               ├── DeepSeekR1TrainDemo.java    # 训练演示
│               ├── DeepSeekR1DatasetGenerator.java
│               └── DeepSeekR1TokenizerUtil.java
└── doc/
    ├── V3_README.md                            # V3 详细技术文档
    └── r1_README.md                            # R1 详细技术文档
```

## 🎯 模型对比

| 特性 | DeepSeek-V3 | DeepSeek-R1 |
|------|------------|------------|
| **基础架构** | Pre-RMSNorm + RoPE + 纯 MoE | Pre-RMSNorm + RoPE + 纯 MoE（复用 V3） |
| **位置编码** | Token 嵌入层内（位置嵌入参数） | RoPE 在注意力层（Token 嵌入无位置嵌入） |
| **MTP Head** | ✅ 有（训练辅助机制） | ❌ 无 |
| **推理优化器** | Adam | SGD（内存效率更高） |
| **训练流程** | 预训练 → SFT → RLHF | 预训练 → SFT → RLHF → RLVR(GRPO) |
| **RL 算法** | Reward-weighted Regression | GRPO（Group Relative Policy Optimization） |
| **参数激活率** | ~25%（Top-2 / 8 专家） | ~25%（Top-2 / 8 专家） |

## 🚀 快速开始

### 环境要求

- **Java**: JDK 17+
- **Maven**: 3.6+
- **内存**: 推荐 4GB+（Tiny 配置可在默认堆内存下运行）

### DeepSeek-V3 基本使用

```java
import io.leavesfly.tinyai.deepseek.v3.*;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;

// 创建模型（4 种规模可选）
DeepSeekV3Model model = DeepSeekV3Model.createTinyModel("v3-tiny");  // 快速测试
// DeepSeekV3Model model = DeepSeekV3Model.createSmallModel("v3-small");
// DeepSeekV3Model model = DeepSeekV3Model.createStandardModel("v3-std");

// 打印架构信息
model.printModelInfo();
System.out.println(model.getConfigSummary());

// 标准推理
float[][] input = {{1, 15, 23, 42}};
Variable logits = model.predict(new Variable(NdArray.of(input)));
System.out.println("输出形状: " + logits.getValue().getShape());

// 带详细信息的推理（含 MoE 损失）
DeepSeekV3Block.DetailedForwardResult result =
    model.predictWithDetails(new Variable(NdArray.of(input)));
System.out.println("MoE 负载均衡损失: " + result.avgMoELoss);
```

### DeepSeek-R1 基本使用

```java
import io.leavesfly.tinyai.deepseek.r1.*;
import io.leavesfly.tinyai.deepseek.base.TaskType;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;

// 创建模型（3 种规模可选）
DeepSeekR1Model model = DeepSeekR1Model.createTinyModel("r1-tiny");
// DeepSeekR1Model model = DeepSeekR1Model.createSmallModel("r1-small");
// DeepSeekR1Model model = DeepSeekR1Model.createStandardModel("r1-std");

// 标准推理
float[][] input = {{1, 15, 23, 42}};
Variable logits = model.predict(new Variable(NdArray.of(input)));

// 推理任务（利用 RL 涌现的推理能力）
DeepSeekR1Model.ReasoningResult result =
    model.performReasoning(new Variable(NdArray.of(input)));
System.out.println("任务类型: " + result.taskType.getDescription());
System.out.println("MoE 损失: " + result.moeLoss);

// 带任务类型的推理
Variable mathLogits = model.predict(new Variable(NdArray.of(input)), TaskType.MATH);
```

### 自定义配置

```java
// V3 自定义配置
DeepSeekV3Config v3Config = new DeepSeekV3Config();
v3Config.setVocabSize(10000);
v3Config.setNEmbd(256);
v3Config.setNLayer(4);
v3Config.setNHead(4);
v3Config.setNumExperts(4);
v3Config.setTopK(2);
v3Config.setMtpDepth(1);           // 启用 MTP
v3Config.setMtpLossWeight(0.3);
DeepSeekV3Model customV3 = new DeepSeekV3Model("custom-v3", v3Config);

// R1 自定义配置
DeepSeekR1Config r1Config = new DeepSeekR1Config();
r1Config.setNEmbd(128);
r1Config.setNLayer(4);
r1Config.setNumExperts(4);
r1Config.setRlClipRange(0.2);      // GRPO clip 范围
r1Config.setRlDiscountFactor(0.99);
DeepSeekR1Model customR1 = new DeepSeekR1Model("custom-r1", r1Config);
```

## 📊 模型规格

### DeepSeek-V3

| 配置 | 嵌入维度 | 层数 | 注意力头 | 专家数 | Top-K | 序列长度 |
|------|---------|------|---------|--------|-------|---------|
| **Micro** | 64 | 2 | 2 | 2 | 1 | 32 |
| **Tiny** | 256 | 6 | 8 | 4 | 2 | 512 |
| **Small** | 512 | 8 | 8 | 6 | 2 | 1024 |
| **Standard** | 768 | 12 | 12 | 8 | 2 | 2048 |

### DeepSeek-R1

| 配置 | 嵌入维度 | 层数 | 注意力头 | 专家数 | Top-K | 序列长度 |
|------|---------|------|---------|--------|-------|---------|
| **Tiny** | 64 | 6 | 8 | 4 | 2 | 128 |
| **Small** | 512 | 8 | 8 | 8 | 2 | 1024 |
| **Standard** | 768 | 12 | 12 | 8 | 2 | 2048 |

## 🔬 训练流程

### DeepSeek-V3 训练

```
预训练 (DeepSeekV3Pretrain)
  Adam 优化器 + Warmup/Cosine 衰减 + MoE 负载均衡损失
    ↓
后训练 SFT (DeepSeekV3Posttrain)
  Answer-only Loss Mask + 早停机制
    ↓
RLHF (DeepSeekV3RLHFTrainer)
  奖励加权回归：L = -reward × CE_loss
```

### DeepSeek-R1 训练

```
预训练 (DeepSeekR1Pretrain)
  SGD 优化器（内存效率），支持多线程并行
    ↓
后训练 SFT (DeepSeekR1Posttrain)
    ↓
RLHF (DeepSeekR1RLHFTrainer)
  委托 DeepSeekV3RLHFTrainer 执行（R1 基于 V3 底座）
    ↓
RLVR / GRPO (DeepSeekR1RLVRTrainer)  ← R1 独有
  Group Relative Policy Optimization + 可验证奖励
  验证器：MathVerifier / CodeVerifier / LogicVerifier
```

**GRPO 核心流程**（对标论文第 4 节）：
1. 对每个问题采样 G 个输出，验证器计算奖励 `{r1,...,rG}`
2. 组内相对优势：`A_i = (r_i - mean(r)) / (std(r) + ε)`
3. PPO-clip 策略更新（无需值函数网络）

### 推理生成

`DeepSeekV3Inference` 支持 4 种生成策略：

| 策略 | 方法 |
|------|------|
| 贪婪解码 | `generateGreedy(promptIds, maxNewTokens)` |
| Temperature 采样 | `generateWithTemperature(promptIds, maxNewTokens, temp)` |
| Top-K 采样 | `generateTopK(promptIds, maxNewTokens, k)` |
| Top-P 采样 | `generateTopP(promptIds, maxNewTokens, p)` |

## 🧪 运行演示

```bash
# 编译
mvn clean compile

# V3 演示（模型创建 + MoE 分析）
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.deepseek.v3.DeepSeekV3Demo"

# R1 演示（模型创建 + 推理 + 序列生成）
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.deepseek.r1.DeepSeekR1Demo"
```

## 📚 详细文档

| 文档 | 内容 |
|------|------|
| [doc/V3_README.md](doc/V3_README.md) | V3 架构、配置、API、训练详解 |
| [doc/r1_README.md](doc/r1_README.md) | R1 架构、GRPO 算法、训练详解 |

## 🔗 相关说明

### 共享基础架构

`DeepSeekBaseConfig` 定义了 V3 和 R1 共享的 MoE 基础配置：

```
DeepSeekBaseConfig
    ├── MoE 配置：numExperts, topK, numSharedExperts, expertHiddenDim
    ├── 任务感知配置：enableTaskAwareRouting, numTaskTypes
    └── 基础架构：vocabSize, nEmbd, nLayer, nHead, nPositions
        ├── DeepSeekV3Config（扩展：mtpDepth, mtpLossWeight, 代码生成参数）
        └── DeepSeekR1Config（扩展：RL 训练参数）
```

### R1 直接复用 V3 TransformerBlock

`DeepSeekV3TransformerBlock` 接受 `DeepSeekBaseConfig`，`DeepSeekR1Config` 继承自 `DeepSeekBaseConfig`，因此 R1Block 可直接传入 R1Config **无需任何 config 转换**：

```java
// R1Block 中直接复用 V3 组件（无 config 转换）
DeepSeekV3TransformerBlock block =
    new DeepSeekV3TransformerBlock(name + "_transformer_" + i, config); // config 是 R1Config
```

### 参考资料

- [DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437)
- [DeepSeek-R1 Paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948)

---

<div align="center">
  <p><strong>DeepSeek-V3 + DeepSeek-R1</strong> — Pre-RMSNorm · RoPE · 纯 MoE · GRPO · 纯 Java 实现</p>
</div>
