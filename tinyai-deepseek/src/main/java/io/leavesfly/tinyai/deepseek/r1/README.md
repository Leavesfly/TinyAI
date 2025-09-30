# DeepSeek R1 实现

基于TinyAI框架的DeepSeek R1模型Java实现，参考了Python版本的设计思路，实现了思维链推理、多步推理和自我反思等核心功能。

## 🌟 核心特性

- **多步思维链推理**: 将复杂问题分解为多个推理步骤
- **自我反思机制**: 评估推理质量并提供改进建议
- **多种推理策略**: 支持逐步推理、分析式推理、比较式推理等
- **强化学习训练**: 基于REINFORCE算法优化推理能力
- **置信度评估**: 实时评估推理过程的可信度
- **可解释性**: 提供详细的推理过程和分析报告

## 📁 项目结构

```
r1/
├── ReasoningStep.java         # 推理步骤数据类
├── ReasoningChain.java        # 推理链数据类
├── MultiHeadAttention.java    # 多头注意力机制
├── TransformerBlock.java      # Transformer块
├── ReasoningModule.java       # 推理模块
├── ReflectionModule.java      # 自我反思模块
├── DeepSeekR1Model.java       # 主模型
├── DeepSeekR1Result.java      # 模型输出结果
├── RLTrainer.java             # 强化学习训练器
├── TrainingMetrics.java       # 训练指标和数据类
├── ChainOfThoughtPrompting.java # 思维链提示处理器
├── CoTProcessingResult.java   # 思维链处理结果
├── TestR1.java               # 演示和测试代码
├── r1.py                     # Python参考实现
└── README.md                 # 本文档
```

## 🏗️ 架构设计

### 模型架构
```
Token Embedding + Position Embedding
    ↓
N × TransformerBlock
    ↓
ReasoningModule (推理模块)
    ↓
ReflectionModule (反思模块)
    ↓
LayerNorm + Output Projection
```

### 核心组件

1. **MultiHeadAttention**: 实现缩放点积注意力机制
2. **TransformerBlock**: 包含自注意力和前馈网络
3. **ReasoningModule**: 执行多步推理，包含思维编码、行动预测、置信度评估
4. **ReflectionModule**: 进行质量评估、一致性检查、风险识别
5. **ChainOfThoughtPrompting**: 生成思维链提示并处理推理任务

## 🚀 快速开始

### 基础使用

```java
// 1. 创建模型
DeepSeekR1Model model = new DeepSeekR1Model(
    "my_r1_model",
    1000,  // 词汇表大小
    256,   // 模型维度
    6      // Transformer层数
);

// 2. 准备输入
NdArray inputIds = createTokenSequence("你的问题");

// 3. 执行推理
DeepSeekR1Result result = model.performReasoning(
    inputIds, 
    "解释深度学习的基本原理"
);

// 4. 查看结果
System.out.println(result.getSummary());
System.out.println(result.getDetailedReport());
```

### 思维链推理

```java
// 1. 创建思维链处理器
ChainOfThoughtPrompting cotProcessor = new ChainOfThoughtPrompting(model);

// 2. 处理问题
CoTProcessingResult result = cotProcessor.processWithCoT(
    "比较深度学习和机器学习的区别"
);

// 3. 查看推理过程
System.out.println(result.getDetailedReport());
```

### 强化学习训练

```java
// 1. 创建训练器
RLTrainer trainer = new RLTrainer(model, 0.001);

// 2. 准备训练数据
TrainingData trainData = new TrainingData();
trainData.addSample(inputIds, targetIds, "训练问题");

// 3. 训练模型
TrainingHistory history = trainer.train(trainData, 10);
```

## 🎯 推理策略

支持多种推理策略，自动根据问题类型选择：

- **STEP_BY_STEP**: 逐步推理，适用于一般问题
- **PROBLEM_SOLVING**: 问题解决，适用于解题类问题
- **ANALYTICAL**: 分析式推理，适用于分析类问题
- **COMPARATIVE**: 比较式推理，适用于对比分析
- **DEDUCTIVE**: 演绎推理，适用于逻辑推导
- **INDUCTIVE**: 归纳推理，适用于总结概括

## 📊 评估指标

### 推理质量评估
- **质量分数**: 推理过程的整体质量 (0-1)
- **一致性分数**: 推理步骤的逻辑一致性 (0-1)
- **风险分数**: 推理结果的风险评估 (0-1)
- **置信度**: 模型对结果的信心程度 (0-1)

### 训练指标
- **奖励值**: 强化学习的奖励信号
- **策略损失**: 策略梯度损失
- **价值损失**: 价值函数损失
- **熵损失**: 探索鼓励损失

## 🧪 运行演示

执行测试类查看完整演示：

```bash
java io.leavesfly.tinyai.deepseek.r1.TestR1
```

演示内容包括：
1. 基础推理能力展示
2. 思维链推理演示
3. 自我反思功能测试
4. 强化学习训练示例
5. 批量处理演示
6. 性能基准测试
7. 错误处理演示

## 💡 实现亮点

### 1. 模块化设计
每个组件都可以独立使用和测试，便于扩展和维护。

### 2. 多层推理
- 基础Transformer推理
- 思维链推理
- 自我反思评估

### 3. 自适应策略
根据问题类型自动选择最适合的推理策略。

### 4. 强化学习优化
使用REINFORCE算法持续改进推理能力。

### 5. 详细的可解释性
提供完整的推理过程记录和质量分析报告。

## 🔧 配置选项

### 模型配置
```java
DeepSeekR1Model model = new DeepSeekR1Model(
    "model_name",
    vocabSize,          // 词汇表大小
    dModel,             // 模型维度  
    numLayers,          // Transformer层数
    numHeads,           // 注意力头数
    dFF,                // 前馈网络维度
    maxSeqLength,       // 最大序列长度
    maxReasoningSteps,  // 最大推理步骤
    dropoutRate,        // Dropout比率
    reasoningThreshold  // 推理置信度阈值
);
```

### 训练配置
```java
RLTrainer trainer = new RLTrainer(
    model,
    learningRate,    // 学习率
    gamma,           // 折扣因子
    entropyCoeff,    // 熵系数
    valueCoeff       // 价值函数系数
);
```

## 📈 性能优化

### 推理优化
- 自适应推理步骤数量
- 早停机制（达到置信度阈值）
- 高效的注意力计算

### 内存优化  
- 渐进式状态更新
- 及时释放中间结果
- 批处理优化

## 🤝 扩展指南

### 添加新的推理策略
1. 在`ReasoningStrategy`枚举中添加新策略
2. 在`ChainOfThoughtPrompting`中添加对应模板
3. 更新策略选择逻辑

### 自定义奖励函数
1. 继承`RLTrainer`类
2. 重写`computeReward`方法
3. 实现特定领域的奖励计算

### 集成新的评估指标
1. 扩展`ReflectionModule`
2. 添加新的评估维度
3. 更新反思报告格式

## 🐛 已知限制

1. **简化的注意力机制**: 当前实现为演示版本，可以进一步优化
2. **内存使用**: 大规模推理时需要注意内存管理
3. **并行化**: 暂不支持多GPU训练
4. **分词器**: 使用简化的分词器，实际应用需要更强大的tokenizer

## 📄 参考资料

- [DeepSeek R1 论文](https://arxiv.org/abs/2501.12948)
- [TinyAI框架文档](../../../README.md)
- [Python参考实现](r1.py)

## 📮 反馈与贡献

欢迎提交Issue和Pull Request来改进这个实现！

---

*基于TinyAI框架构建 - 让AI推理更加透明和可控*