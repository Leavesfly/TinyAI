# DeepSeek-Rx 技术文档

## ⚠️ 重要说明

**本实现是受 DeepSeek-R1 启发的教育性推理模型,而非官方架构的完整复现。**

本项目通过显式设计的推理和反思模块,帮助学习者直观理解推理机制的核心概念。与官方 DeepSeek-R1 的技术架构存在本质差异,详见[架构差异对比](#-与官方架构的差异)部分。

## 📋 模型概述

DeepSeek-R1 是一个具备**深度推理和自我反思能力**的大语言模型,通过多步推理和反思机制实现复杂任务的可解释性处理。该模型采用 Pre-LayerNorm 架构,完全基于 TinyAI 框架的 **V2 API** 实现。

### 核心特性

- 🧠 **多步推理** - 支持最多7步迭代推理过程，逐步逼近最优答案
- 🔄 **自我反思** - 从5个维度评估推理质量，提供改进建议
- 📊 **置信度评估** - 动态评估每步推理的可信度
- 💡 **思维链生成** - 输出完整的推理过程，增强可解释性
- ✅ **完整Variable层面** - 所有计算在Variable层面，梯度完整回传

### 技术亮点

1. **迭代推理机制**：支持7步渐进式推理，每步评估置信度
2. **多维度反思**：从准确性、逻辑性、完整性、创新性、可行性5个维度评估
3. **Variable层面计算**：Token嵌入使用`indexSelect`、`reshape`、`repeat`算子
4. **自适应阈值**：根据任务复杂度动态调整置信度阈值

## 🏗️ 架构设计

### 整体架构图

```
┌──────────────────────────────────────────────────────────────┐
│                    DeepSeekR1Model                           │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              DeepSeekR1Block (主体块)                   │ │
│  │  ┌───────────────────────────────────────────────────┐  │ │
│  │  │  DeepSeekR1TokenEmbedding (✅ Variable层面)       │  │ │
│  │  │  - indexSelect选择Token嵌入                       │  │ │
│  │  │  - reshape + repeat扩展Position嵌入                │  │ │
│  │  └───────────────────────────────────────────────────┘  │ │
│  │  ┌───────────────────────────────────────────────────┐  │ │
│  │  │  N × [DeepSeekR1TransformerBlock]                 │  │ │
│  │  │  - MultiHeadAttention (V2)                        │  │ │
│  │  │  - LayerNorm (V2)                                 │  │ │
│  │  │  - Linear (V2)                                    │  │ │
│  │  │  - GELU (V2)                                      │  │ │
│  │  │  - Dropout (V2)                                   │  │ │
│  │  └───────────────────────────────────────────────────┘  │ │
│  │  ┌───────────────────────────────────────────────────┐  │ │
│  │  │  DeepSeekR1ReasoningBlock (多步推理)              │  │ │
│  │  │  ┌──────────────────────────────────────────────┐ │  │ │
│  │  │  │  第1步推理 → 置信度评估                      │ │  │ │
│  │  │  │  第2步推理 → 置信度评估                      │ │  │ │
│  │  │  │  ...                                         │ │  │ │
│  │  │  │  第7步推理 → 置信度评估                      │ │  │ │
│  │  │  └──────────────────────────────────────────────┘ │  │ │
│  │  │  - 最多7步迭代推理                                │  │ │
│  │  │  - 每步动态置信度评估                             │  │ │
│  │  │  - 推理结果验证                                   │  │ │
│  │  └───────────────────────────────────────────────────┘  │ │
│  │  ┌───────────────────────────────────────────────────┐  │ │
│  │  │  DeepSeekR1ReflectionBlock (自我反思)             │  │ │
│  │  │  ┌──────────────────────────────────────────────┐ │  │ │
│  │  │  │  质量评估 (5个维度)                          │ │  │ │
│  │  │  │  1. 准确性评估                               │ │  │ │
│  │  │  │  2. 逻辑性评估                               │ │  │ │
│  │  │  │  3. 完整性评估                               │ │  │ │
│  │  │  │  4. 创新性评估                               │ │  │ │
│  │  │  │  5. 可行性评估                               │ │  │ │
│  │  │  └──────────────────────────────────────────────┘ │  │ │
│  │  │  - 综合质量评分                                   │  │ │
│  │  │  - 改进建议生成                                   │  │ │
│  │  │  - 是否需要重新推理                               │  │ │
│  │  └───────────────────────────────────────────────────┘  │ │
│  │  │  LayerNorm (V2) + Linear (V2)                        │  │ │
│  │  └───────────────────────────────────────────────────┘  │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

### 核心组件

#### 1. DeepSeekR1Config（完全独立配置类，481行）

**基础配置**：
- `vocabSize`: 词汇表大小（默认50257）
- `nPositions`: 最大序列长度（默认2048）
- `nEmbd`: 嵌入维度（默认512）
- `nLayer`: Transformer层数（默认12）
- `nHead`: 注意力头数（默认8）
- `nInner`: 前馈网络维度（默认2048）

**推理配置**：
- `maxReasoningSteps`: 最大推理步骤数（默认7）
- `confidenceThreshold`: 置信度阈值（默认0.7）
- `reasoningHiddenDim`: 推理模块隐藏层维度（默认1024）
- `enableIterativeReasoning`: 是否启用迭代推理（默认true）

**反思配置**：
- `reflectionHiddenDim`: 反思模块隐藏层维度（默认1024）
- `qualityThreshold`: 质量评估阈值（默认0.75）
- `numQualityDimensions`: 质量评估维度数量（默认5）
- `enableSelfImprovement`: 是否启用自我改进（默认true）

**Dropout配置**：
- `residPdrop`: 残差dropout（默认0.1）
- `embdPdrop`: 嵌入dropout（默认0.1）
- `attnPdrop`: 注意力dropout（默认0.1）

**预设配置工厂方法**：
```java
// 微型配置（快速测试）
DeepSeekR1Config.createTinyConfig()
// 256维, 6层, 8头, 7步推理, 512序列长度

// 标准配置（标准应用）
DeepSeekR1Config.createStandardConfig()
// 512维, 12层, 8头, 7步推理, 2048序列长度

// 小型配置（学习实验）
DeepSeekR1Config.createSmallConfig()
// 384维, 8层, 8头, 5步推理, 1024序列长度

// 大型配置（高级应用）
DeepSeekR1Config.createLargeConfig()
// 768维, 18层, 12头, 7步推理, 2048序列长度
```

#### 2. DeepSeekR1TokenEmbedding（V2 Module，完全Variable层面）

**核心实现**：
```java
// ✅ 完全在Variable层面实现
private Variable getTokenEmbeddingsV2(Variable tokenIds, Variable tokenEmbedParam,
                                      int batchSize, int sequenceLength) {
    // 1. 展平tokenIds: [batch, seq] -> [batch*seq]
    Variable flatIds = tokenIds.reshape(Shape.of(-1));
    
    // 2. 使用indexSelect选择嵌入: [batch*seq, embd]
    Variable flatEmbeds = tokenEmbedParam.indexSelect(0, flatIds);
    
    // 3. Reshape回3D: [batch, seq, embd]
    return flatEmbeds.reshape(Shape.of(batchSize, sequenceLength, embeddingDim));
}

private Variable getPositionEmbeddingsV2(Variable posEmbedParam, int batchSize, int sequenceLength) {
    // 1. 创建位置索引
    Variable posIds = new Variable(NdArray.of(posIndices));
    
    // 2. indexSelect选择位置嵌入
    Variable posEmbeds = posEmbedParam.indexSelect(0, posIds);
    
    // 3. Reshape + repeat扩展batch维度
    Variable posEmbeds3D = posEmbeds.reshape(Shape.of(1, sequenceLength, embeddingDim));
    return posEmbeds3D.repeat(batchSize, 1, 1);
}
```

**Variable算子使用**：
- ✅ `indexSelect` - 索引选择嵌入向量
- ✅ `reshape` - 形状变换
- ✅ `repeat` - 维度重复扩展
- ✅ `add` - 嵌入相加

#### 3. DeepSeekR1ReasoningBlock（多步推理模块）

**推理机制**：

```java
/**
 * 执行多步推理
 * 
 * 推理流程：
 * 1. 初始化推理状态
 * 2. 迭代推理（最多7步）
 *    - 执行单步推理
 *    - 评估置信度
 *    - 判断是否需要继续
 * 3. 返回推理结果
 */
public ReasoningResult performReasoning(Variable input, Variable context) {
    List<ReasoningStep> steps = new ArrayList<>();
    Variable currentState = input;
    
    for (int step = 0; step < config.getMaxReasoningSteps(); step++) {
        // 单步推理
        Variable stepOutput = reasoningLayer.forward(currentState);
        
        // 置信度评估
        float confidence = evaluateConfidence(stepOutput);
        
        // 记录推理步骤
        steps.add(new ReasoningStep(step + 1, stepOutput, confidence));
        
        // 判断是否达到置信度阈值
        if (confidence >= config.getConfidenceThreshold()) {
            break;
        }
        
        // 更新状态
        currentState = stepOutput;
    }
    
    return new ReasoningResult(steps);
}
```

**关键特性**：
- ✅ 最多7步迭代推理
- ✅ 每步动态评估置信度
- ✅ 置信度达标提前终止
- ✅ 完整记录推理轨迹

#### 4. DeepSeekR1ReflectionBlock（自我反思模块）

**反思机制**：

```java
/**
 * 执行自我反思
 * 
 * 反思维度（5个）：
 * 1. 准确性 (Accuracy) - 推理结果的正确性
 * 2. 逻辑性 (Logic) - 推理过程的逻辑连贯性
 * 3. 完整性 (Completeness) - 是否考虑所有相关因素
 * 4. 创新性 (Creativity) - 是否有新颖的见解
 * 5. 可行性 (Feasibility) - 结果的实际可行性
 */
public ReflectionResult reflect(Variable reasoningOutput, Variable originalInput) {
    // 1. 评估准确性
    float accuracyScore = evaluateAccuracy(reasoningOutput, originalInput);
    
    // 2. 评估逻辑性
    float logicScore = evaluateLogic(reasoningOutput);
    
    // 3. 评估完整性
    float completenessScore = evaluateCompleteness(reasoningOutput);
    
    // 4. 评估创新性
    float creativityScore = evaluateCreativity(reasoningOutput);
    
    // 5. 评估可行性
    float feasibilityScore = evaluateFeasibility(reasoningOutput);
    
    // 综合评分
    float qualityScore = (accuracyScore + logicScore + completenessScore + 
                         creativityScore + feasibilityScore) / 5.0f;
    
    // 生成改进建议
    List<String> suggestions = generateImprovementSuggestions(
        accuracyScore, logicScore, completenessScore, 
        creativityScore, feasibilityScore
    );
    
    // 判断是否需要重新推理
    boolean needsRefinement = qualityScore < config.getQualityThreshold();
    
    return new ReflectionResult(
        qualityScore,
        accuracyScore,
        logicScore,
        completenessScore,
        creativityScore,
        feasibilityScore,
        suggestions,
        needsRefinement
    );
}
```

**反思结果**：
```java
public static class ReflectionResult {
    float qualityScore;           // 综合质量评分 (0-1)
    float accuracyScore;          // 准确性评分
    float logicScore;             // 逻辑性评分
    float completenessScore;      // 完整性评分
    float creativityScore;        // 创新性评分
    float feasibilityScore;       // 可行性评分
    List<String> suggestions;     // 改进建议
    boolean needsRefinement;      // 是否需要重新推理
}
```

## 🚀 使用指南

### 1. 基本使用

```java


// 创建模型（使用工厂方法）
DeepSeekR1Model model=DeepSeekR1Model.createStandardModel("deepseek-r1");

// 打印模型信息
        model.printModelInfo();

// 基础推理
        NdArray tokenIds=NdArray.of(new int[][]{{1,15,23,42}});
        Variable output=model.predict(new Variable(tokenIds));
        System.out.println("输出形状: "+output.getValue().getShape());
```

### 2. 多步推理

```java
// 执行带详细信息的推理
DeepSeekR1Model.ReasoningOutput result = 
    model.performReasoning(new Variable(tokenIds));

System.out.println("推理步骤数: " + result.numReasoningSteps);
System.out.println("平均置信度: " + result.averageConfidence);
System.out.println("质量评分: " + result.qualityScore);

// 获取详细推理过程
DeepSeekR1Block.DetailedForwardResult detailedResult = 
    model.predictWithDetails(new Variable(tokenIds));

// 推理结果
DeepSeekR1ReasoningBlock.ReasoningResult reasoningResult = 
    detailedResult.reasoningResult;
System.out.println("推理步骤: " + reasoningResult.numSteps);
System.out.println("置信度: " + reasoningResult.averageConfidence);

// 反思结果
DeepSeekR1ReflectionBlock.ReflectionResult reflectionResult = 
    detailedResult.reflectionResult;
System.out.println("质量评分: " + reflectionResult.qualityScore);
System.out.println("准确性: " + reflectionResult.accuracyScore);
System.out.println("逻辑性: " + reflectionResult.logicScore);
System.out.println("完整性: " + reflectionResult.completenessScore);
System.out.println("创新性: " + reflectionResult.creativityScore);
System.out.println("可行性: " + reflectionResult.feasibilityScore);
System.out.println("需要改进: " + reflectionResult.needsRefinement);

// 改进建议
for (String suggestion : reflectionResult.suggestions) {
    System.out.println("- " + suggestion);
}
```

### 3. 自定义配置

```java
// 创建自定义配置
DeepSeekR1Config config = new DeepSeekR1Config();

// 基础配置
config.setVocabSize(50257);
config.setNEmbd(512);
config.setNLayer(12);
config.setNHead(8);

// 推理配置
config.setMaxReasoningSteps(7);          // 最多7步推理
config.setConfidenceThreshold(0.7f);     // 置信度阈值
config.setEnableIterativeReasoning(true); // 启用迭代推理

// 反思配置
config.setQualityThreshold(0.75f);       // 质量阈值
config.setNumQualityDimensions(5);       // 5个评估维度
config.setEnableSelfImprovement(true);   // 启用自我改进

// 创建模型
DeepSeekR1Model model = new DeepSeekR1Model("custom-r1", config);
```

### 4. 序列生成

```java
// 贪婪解码生成序列
NdArray promptIds = NdArray.of(new int[][]{{1, 2, 3}});
NdArray generated = model.generateSequence(
    promptIds, 
    50  // 最大生成50个token
);

System.out.println("生成序列长度: " + generated.getShape().getDimension(1));
```

## 📊 性能特点

### 模型规模

| 配置 | 参数量 | 层数 | 维度 | 注意力头 | 推理步骤 | 序列长度 |
|------|-------|------|------|---------|---------|---------|
| Tiny | ~20M | 6 | 256 | 8 | 7 | 512 |
| Small | ~60M | 8 | 384 | 8 | 5 | 1024 |
| Standard | ~100M | 12 | 512 | 8 | 7 | 2048 |
| Large | ~350M | 18 | 768 | 12 | 7 | 2048 |

### 推理特性

| 特性 | 描述 | 优势 |
|------|------|------|
| 多步推理 | 最多7步迭代 | 逐步逼近最优答案 |
| 置信度评估 | 每步动态评估 | 自适应终止条件 |
| 自我反思 | 5维度评估 | 全面质量保证 |
| 改进建议 | 自动生成 | 可解释性强 |

### V2组件覆盖

| 组件 | 使用位置 | Variable层面 |
|------|----------|------------|
| Module | 所有层基类 | ✅ |
| Parameter | Token/Position嵌入 | ✅ |
| LayerNorm | Transformer块、最终层 | ✅ |
| MultiHeadAttention | Transformer块 | ✅ |
| Linear | MLP、推理、反思、输出 | ✅ |
| GELU | MLP | ✅ |
| Dropout | 所有分支 | ✅ |

## 🔬 训练支持

### 训练器

R1提供完整的训练支持，位于`training/`目录：

1. **DeepSeekR1Pretrain** - 预训练
   - 从随机初始化开始训练
   - 大规模语料预训练

2. **DeepSeekR1Finetune** - 微调
   - 在预训练模型基础上微调
   - 任务特定数据适配

3. **DeepSeekR1RLTrainer** - 强化学习训练器
   - 基于奖励的模型优化
   - 支持PPO、DPO等RL算法
   - 奖励函数考虑准确性、推理质量、反思深度

4. **DeepSeekR1Inference** - 推理
   - 高效推理实现
   - 支持批量推理

5. **DeepSeekR1Evaluator** - 评估器
   - 模型性能评估
   - 多维度指标计算

6. **DeepSeekR1Generator** - 生成器
   - 文本生成实现
   - 支持多种解码策略

### 强化学习训练

```java
// 创建RL训练器
DeepSeekR1RLTrainer trainer = new DeepSeekR1RLTrainer(
    maxEpoch,
    trainingMonitor,
    evaluator
);

// 初始化
trainer.init(dataset, model, lossFunction, optimizer);

// 设置奖励权重
trainer.setRewardWeights(
    0.4f,  // 准确性权重
    0.3f,  // 推理质量权重
    0.2f,  // 反思深度权重
    0.1f   // 一致性权重
);

// 训练
trainer.trainRL();
```

## 🧪 测试验证

### 编译验证

```bash
# 编译模块
cd tinyai-model-deepseek
mvn clean compile

# 运行测试
mvn test -Dtest="DeepSeekR1Test"
```

### 功能验证

运行演示程序：
```bash
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.deepseek.rx.DeepSeekR1Demo"
```

### 验证清单

- ✅ 模型创建和初始化
- ✅ Token嵌入Variable层面计算
- ✅ 多步推理功能
- ✅ 置信度评估
- ✅ 自我反思机制
- ✅ 质量评分计算
- ✅ 改进建议生成
- ✅ 梯度完整回传
- ✅ 编译通过无错误

## 🔍 核心优势

### 1. Variable层面完整性

**TokenEmbedding**：
- ✅ 使用`indexSelect`索引选择，而非手动NdArray操作
- ✅ 使用`reshape`和`repeat`进行形状变换和扩展
- ✅ 完整计算图，梯度正确回传到嵌入参数

**与V3相同的实现**：
```java
// ✅ 完全在Variable层面
Variable flatIds = tokenIds.reshape(Shape.of(-1));
Variable flatEmbeds = tokenEmbedParam.indexSelect(0, flatIds);
return flatEmbeds.reshape(Shape.of(batchSize, seqLen, nEmbd));
```

### 2. 多步推理机制

**迭代推理流程**：
1. 初始化推理状态
2. 执行单步推理
3. 评估置信度
4. 判断是否继续（置信度 vs 阈值）
5. 更新状态
6. 重复步骤2-5，最多7次

**优势**：
- ✅ 逐步逼近最优答案
- ✅ 自适应终止条件
- ✅ 完整推理轨迹记录
- ✅ 可解释性强

### 3. 自我反思评估

**5个评估维度**：

| 维度 | 含义 | 作用 |
|------|------|------|
| 准确性 | 推理结果的正确性 | 确保答案可靠 |
| 逻辑性 | 推理过程的逻辑连贯性 | 确保推理合理 |
| 完整性 | 是否考虑所有相关因素 | 确保答案全面 |
| 创新性 | 是否有新颖的见解 | 鼓励创造性思维 |
| 可行性 | 结果的实际可行性 | 确保答案实用 |

**综合评分**：
```java
qualityScore = (accuracy + logic + completeness + creativity + feasibility) / 5.0
```

**改进建议生成**：
- 针对每个低分维度给出具体建议
- 帮助模型自我改进
- 增强可解释性

### 4. 与V3的对比

| 特性 | R1 | V3 |
|------|----|----|---|
| 架构 | 标准Transformer | MoE (8专家) |
| 主要能力 | 推理、反思 | 代码生成、多任务 |
| 推理机制 | 7步迭代推理 | 任务感知推理 |
| 反思机制 | 5维度完整反思 | 自我纠错 |
| 参数效率 | 全部激活 | 激活~25% |
| 适用场景 | 推理、问题求解 | 代码、数学、多模态 |

## 🔍 与官方架构的差异

### 官方 DeepSeek-R1 架构特征

根据 DeepSeek 官方技术报告([arXiv:2501.12948](https://arxiv.org/abs/2501.12948)),真实的 DeepSeek-R1 具有以下核心特征:

| 特性 | 官方 DeepSeek-R1 |
|------|------------------|
| **基础模型** | 基于 DeepSeek-V3-Base |
| **总参数量** | 671B(6710亿)参数 |
| **激活参数** | 每token激活 37B 参数 |
| **架构类型** | **Mixture-of-Experts (MoE)** |
| **层数** | 61层 Transformer |
| **注意力机制** | **Multi-Head Latent Attention (MLA)** |
| **上下文长度** | 128K tokens |
| **训练方式** | **纯强化学习(RL)** 驱动推理能力 |
| **推理能力** | 通过RL自然涌现的思维链推理 |
| **关键创新** | RL直接激励推理能力,无需大量SFT数据 |

**官方训练流程**:
- **DeepSeek-R1-Zero**: 仅使用RL训练,无监督微调,展示推理能力自然涌现
- **DeepSeek-R1**: 多阶段训练(冷启动数据 + RL + 拒绝采样SFT + 二次RL)

### 核心差异对比表

| 对比维度 | 官方 DeepSeek-R1 | 本项目实现 | 差异程度 |
|---------|-----------------|-----------|---------|
| **模型规模** | 671B参数 | 20M-350M | **~2000倍差距** |
| **架构类型** | MoE(8专家) | 标准Dense Transformer | **架构完全不同** |
| **注意力** | MLA(潜在注意力) | 标准多头注意力 | **机制不同** |
| **推理方式** | RL涌现的隐式推理 | 显式7步迭代模块 | **哲学不同** |
| **反思机制** | 通过RL奖励信号学习 | 显式5维度评估模块 | **实现方式不同** |
| **上下文** | 128K | 512-2048 | **~60倍差距** |
| **训练范式** | RL为主导,涌现推理 | 预训练+微调+RL | **方法不同** |

### 设计理念差异

#### 官方 DeepSeek-R1
- **核心理念**: 通过强化学习**自然涌现**推理能力
- **推理机制**: 模型在RL训练中自发学习思维链推理,无需人为设计
- **技术创新**: 证明了纯RL可以激发LLM的推理能力,无需大量监督数据
- **架构选择**: 继承V3的MoE架构,专注于通过RL优化推理质量

#### 本项目实现
- **核心理念**: 通过**显式模块设计**展示推理过程
- **推理机制**: 明确的7步迭代推理模块,每步评估置信度
- **教育价值**: 让学习者直观理解"推理-反思"的概念和实现
- **架构选择**: 标准Transformer,降低硬件门槛,便于学习和实验

### 适用场景说明

#### 官方 DeepSeek-R1 适用于:
- ✅ 工业级生产环境
- ✅ 复杂推理任务(数学、科学问题)
- ✅ 需要强大推理能力的实际应用
- ✅ 大规模部署场景

#### 本项目实现适用于:
- ✅ **AI教育和学习**
- ✅ **推理机制概念验证**
- ✅ **算法原型开发**
- ✅ **小规模实验和研究**
- ✅ **自动微分框架学习**

### 教育价值

尽管与官方架构存在显著差异,本项目作为教育性实现具有独特价值:

| 维度 | 教育价值 |
|------|----------|
| **概念展示** | ✅ 显式的7步推理、5维反思清晰展示了"推理-反思"核心思想 |
| **可理解性** | ✅ 模块化设计让学习者直观理解推理过程的每个环节 |
| **可运行性** | ✅ 小规模参数可在普通硬件上运行和调试 |
| **框架完整** | ✅ 包含预训练、微调、RL训练器的完整流程 |
| **代码可读** | ✅ Variable层面完整计算图,利于学习自动微分 |
| **实验友好** | ✅ 易于修改和扩展,适合算法实验 |

### 局限性说明

本项目**未实现**官方 DeepSeek-R1 的以下核心特性:

- ❌ MoE(Mixture-of-Experts)架构
- ❌ MLA(Multi-Head Latent Attention)注意力机制
- ❌ 纯RL涌现推理能力的训练范式
- ❌ Multi Token Prediction (MTP)
- ❌ 128K超长上下文支持
- ❌ 671B参数规模

### 总结

**本项目是一个优秀的教育性推理模型实现**,通过显式模块设计帮助学习者理解推理和反思的概念。虽然与官方 DeepSeek-R1 的技术架构存在本质差异,但作为教学工具和概念验证平台,这种简化是合理且必要的。

如果你需要官方 DeepSeek-R1 的完整功能,请访问:
- 官方论文: [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
- 官方GitHub: [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)
- 官方API: [DeepSeek API Docs](https://api-docs.deepseek.com/)


## 📚 参考资料

### 相关文档
- [DeepSeek-R1 主README](../README.md)
- [训练文档](training/)
- [推理机制详细说明](r1/README.md)

### 技术论文
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- Chain-of-Thought Prompting
- Self-Reflection in Language Models

### 源代码
- [DeepSeekR1Model.java](../src/main/java/io/leavesfly/tinyai/deepseek/r1/DeepSeekR1Model.java)
- [DeepSeekR1Config.java](../src/main/java/io/leavesfly/tinyai/deepseek/r1/DeepSeekR1Config.java)
- [DeepSeekR1ReasoningBlock.java](../src/main/java/io/leavesfly/tinyai/deepseek/r1/DeepSeekR1ReasoningBlock.java)
- [DeepSeekR1ReflectionBlock.java](../src/main/java/io/leavesfly/tinyai/deepseek/r1/DeepSeekR1ReflectionBlock.java)

---

<div align="center">
  <p><strong>DeepSeek-R1</strong> - 多步推理与自我反思</p>
  <p>可解释推理 | 质量评估 | 自我改进</p>
</div>
