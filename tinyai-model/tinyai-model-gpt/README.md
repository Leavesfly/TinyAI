# TinyAI GPT-3 模型模块

## 模块概述

`tinyai-model-gpt` 模块是 TinyAI 框架中 GPT-3（Generative Pre-trained Transformer 3）模型的完整实现。该模块基于 TinyAI V2 API，提供了从 125M 到 175B 参数规模的多配置模型支持，适用于学习研究、实用应用和大规模生成任务。

## 主要特性

- **完整 GPT-3 架构实现**：并行注意力计算、Pre-LayerNorm 结构
- **多规模模型配置**：Small(125M)、Medium(350M)、Large(1.3B)、XL(175B)
- **高级注意力机制**：RoPE 旋转位置编码、稀疏注意力（局部窗口+步长全局）
- **推理优化**：KV Cache 加速自回归生成、梯度检查点节省显存
- **训练支持**：预训练、微调、推理完整流程
- **模块化设计**：完全基于 TinyAI V2 Module API

## 模块结构

```
tinyai-model-gpt/
├── src/main/java/io/leavesfly/tinyai/gpt3/
│   ├── GPT3Model.java              # 模型主类
│   ├── GPT3Config.java             # 配置类
│   ├── GPT3MainBlock.java          # 主体块（Token嵌入 + Transformer堆叠 + 输出）
│   ├── GPT3TransformerBlock.java   # Transformer块（并行架构）
│   ├── GPT3Attention.java          # 增强注意力（RoPE + 稀疏注意力 + KV Cache）
│   ├── GPT3TokenEmbedding.java     # Token嵌入层
│   ├── GPT3KVCache.java            # KV缓存（推理加速）
│   ├── GPT3Demo.java               # 演示程序
│   └── training/                   # 训练相关
│       ├── GPT3Dataset.java        # 数据集
│       ├── GPT3Pretrain.java       # 预训练
│       ├── GPT3Finetune.java       # 微调
│       ├── GPT3Inference.java      # 推理
│       └── GPT3TrainDemo.java      # 训练演示
├── src/test/java/io/leavesfly/tinyai/gpt3/
│   ├── GPT3ModelTest.java          # 模型测试
│   ├── GPT3ConfigTest.java         # 配置测试
│   └── GPT3TokenEmbeddingTest.java # 嵌入层测试
├── doc/
│   └── gpt3_README.md              # 详细技术文档
└── pom.xml
```

## GPT-3 核心架构

### 1. 并行注意力架构

GPT-3 的核心创新是**并行计算注意力和前馈网络**，与 GPT-2 的串行架构不同：

```
输入 x
  ├─ LayerNorm1 → GPT3Attention → attn_output
  └─ LayerNorm2 → FeedForward   → mlp_output
合并: x + attn_output + mlp_output → 输出
```

**优势**：
- 提高计算效率，两个分支可并行执行
- 更好的梯度流动
- 训练更稳定

### 2. 增强注意力机制

GPT3Attention 支持三种高级特性，可通过配置独立开启：

| 特性 | 配置项 | 作用 |
|------|--------|------|
| **RoPE 旋转位置编码** | `useRotaryEmbedding` | 更好的长序列位置建模 |
| **稀疏注意力** | `sparseAttention` | 降低计算复杂度，支持更长序列 |
| **KV Cache** | `useCache` | 加速自回归推理 |

**稀疏注意力模式**：
```
每个 Query 可关注：
  - 局部窗口：最近 N 个 Token（sparseLocalWindow）
  - 步长全局：每隔 M 个 Token（sparseStrideSize）
```

### 3. Pre-LayerNorm 结构

所有 Transformer 块采用 Pre-LayerNorm，在子层之前应用归一化：

```java
// GPT3TransformerBlock 核心逻辑
Variable attnInput = layerNorm1.forward(x);
Variable attnOutput = attention.forward(attnInput);
Variable mlpInput = layerNorm2.forward(x);
Variable mlpOutput = feedForward(mlpInput);
return x.add(attnOutput).add(mlpOutput);
```

## 模型配置

### 预设配置对比

| 配置 | 参数量 | 层数 | 隐藏维度 | 注意力头 | FFN维度 | 特殊优化 |
|------|--------|------|----------|----------|---------|----------|
| Small | 125M | 12 | 768 | 12 | 3072 | 并行架构 |
| Medium | 350M | 24 | 1024 | 16 | 4096 | 并行架构 |
| Large | 1.3B | 24 | 2048 | 32 | 8192 | RoPE + 稀疏注意力 + 梯度检查点 |
| XL | 175B | 96 | 12288 | 96 | 49152 | 全部优化特性 |

### 配置参数详解

```java
GPT3Config config = new GPT3Config();
// 基础配置
config.setVocabSize(50257);        // 词汇表大小
config.setNPositions(2048);        // 最大序列长度
config.setNEmbd(768);              // 嵌入维度
config.setNLayer(12);              // Transformer层数
config.setNHead(12);               // 注意力头数
config.setNInner(3072);            // FFN中间维度

// Dropout 配置
config.setResidPdrop(0.1);         // 残差 Dropout
config.setEmbdPdrop(0.1);          // 嵌入 Dropout
config.setAttnPdrop(0.1);          // 注意力 Dropout

// GPT-3 特有配置
config.setParallelAttention(true); // 并行架构
config.setUseRotaryEmbedding(true);// RoPE 位置编码
config.setRotaryPct(0.25);         // RoPE 作用维度比例
config.setSparseAttention(true);   // 稀疏注意力
config.setSparseLocalWindow(256);  // 局部窗口大小
config.setSparseStrideSize(128);   // 步长全局间隔
config.setGradientCheckpointing(true); // 梯度检查点
config.setUseCache(true);          // KV Cache
```

## 快速开始

### 1. 依赖配置

```xml
<dependency>
    <groupId>io.leavesfly.tinyai</groupId>
    <artifactId>tinyai-model-gpt</artifactId>
    <version>1.0-SNAPSHOT</version>
</dependency>
```

### 2. 创建模型

```java
import io.leavesfly.tinyai.gpt3.*;

// 方式一：使用预设配置
GPT3Model smallModel = GPT3Model.createSmallModel("gpt3-small");   // 125M
GPT3Model mediumModel = GPT3Model.createMediumModel("gpt3-medium"); // 350M
GPT3Model largeModel = GPT3Model.createLargeModel("gpt3-large");   // 1.3B
GPT3Model xlModel = GPT3Model.createXLModel("gpt3-xl");           // 175B

// 方式二：自定义配置
GPT3Config config = GPT3Config.createSmallConfig();
config.setNLayer(6);
config.setNEmbd(512);
GPT3Model customModel = new GPT3Model("my-gpt3", config);
```

### 3. 前向推理

```java
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;

// 准备输入 (batchSize=2, seqLen=10)
NdArray tokenIds = NdArray.of(new float[][]{
    {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000},
    {110, 210, 310, 410, 510, 610, 710, 810, 910, 1010}
});

// 前向传播
Variable logits = model.predict(new Variable(tokenIds));
// 输出形状: (2, 10, 50257) -> (batch, seq, vocab)
```

### 4. 文本生成

#### 基础生成（无 KV Cache）

```java
NdArray promptIds = NdArray.of(new float[][]{{100, 200, 300}});
NdArray generated = model.generateSequence(promptIds, 50);
// 生成 50 个新 Token
```

#### 加速生成（带 KV Cache）

```java
// 使用 KV Cache，每步只计算 1 个 Token
// 计算量从 O(n²) 降至 O(n) 每步
NdArray generated = model.generateWithCache(promptIds, 50);
```

### 5. 模型信息查看

```java
// 打印详细模型信息
model.printModelInfo();

// 获取配置摘要
String summary = model.getConfigSummary();

// 获取参数数量
long params = model.getConfig().estimateParameterCount();
```

## 核心组件详解

### GPT3MainBlock

主体块包含完整的前向计算流程：

```
TokenIDs → TokenEmbedding → [TransformerBlock × N] → LayerNorm → OutputProjection → Logits
```

- `GPT3TokenEmbedding`: Token 嵌入 + 位置嵌入
- `GPT3TransformerBlock`: 并行注意力块（支持梯度检查点）
- `LayerNorm`: 最终归一化
- `Linear`: 输出投影到词汇表维度

### GPT3Attention

增强注意力层，支持：

1. **RoPE 旋转位置编码**
   - 对 Q 和 K 的前 `rotaryDim` 维度应用旋转
   - 公式：`x' = [x0*cos - x1*sin, x0*sin + x1*cos]`
   - 保持向量模长不变，是等距变换

2. **稀疏注意力掩码**
   - 局部窗口：关注最近 N 个 Token
   - 步长全局：每隔 M 个 Token 作为"锚点"
   - 大幅降低计算复杂度

3. **KV Cache**
   - 缓存历史 Token 的 K/V 向量
   - 增量推理时仅计算新 Token
   - 支持滑动窗口（超出 maxCacheLen 时丢弃最早 Token）

### GPT3KVCache

KV 缓存实现，用于加速自回归生成：

```java
GPT3KVCache cache = new GPT3KVCache(batchSize, numHeads, headDim, maxSeqLen);

// 更新缓存
NdArray[] kv = cache.update(newK, newV);
// kv[0] = 完整的 K, kv[1] = 完整的 V

// 清空缓存
cache.clear();
```

## 训练支持

### 预训练

```java
// 使用 GPT3Pretrain 进行预训练
GPT3Pretrain pretrain = new GPT3Pretrain(config, trainData);
pretrain.train(epochs, learningRate);
```

### 微调

```java
// 使用 GPT3Finetune 进行微调
GPT3Finetune finetune = new GPT3Finetune(pretrainedModel);
finetune.finetune(trainData, epochs, learningRate);
```

## 测试验证

```bash
# 运行所有测试
mvn test

# 运行特定测试
mvn test -Dtest=GPT3ModelTest
mvn test -Dtest=GPT3ConfigTest
mvn test -Dtest=GPT3TokenEmbeddingTest
```

## 性能优化建议

### 训练优化

- **梯度检查点**：大模型启用 `gradientCheckpointing=true` 节省显存
- **稀疏注意力**：长序列场景启用 `sparseAttention=true` 降低计算量
- **混合精度**：配合框架的混合精度训练支持

### 推理优化

- **KV Cache**：自回归生成时必须启用，提升 10-100 倍速度
- **批量推理**：充分利用批处理并行计算
- **模型量化**：生产环境可考虑 INT8 量化

## 依赖关系

```
tinyai-model-gpt
  └── tinyai-deeplearning-ml
        └── TinyAI 核心框架 (NdArray, Variable, Module, Layer 等)
```

## 参考资料

- [GPT-3 论文](https://arxiv.org/abs/2005.14165): Language Models are Few-Shot Learners
- [RoPE 论文](https://arxiv.org/abs/2104.09864): RoFormer: Enhanced Transformer with Rotary Position Embedding
- 详细技术文档: `doc/gpt3_README.md`

---

**作者**: leavesfly  
**版本**: 2.0  
**最后更新**: 2025年