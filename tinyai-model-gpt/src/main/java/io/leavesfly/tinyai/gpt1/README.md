# GPT-1 实现说明

## 概述

本目录包含基于TinyAI框架实现的GPT-1 (Generative Pre-trained Transformer 1) 模型。这是OpenAI在2018年发布的第一个GPT模型的Java实现，完全遵循原论文的架构设计。

## 项目结构

```
tinyai-model-gpt/src/main/java/io/leavesfly/tinyai/gpt1/
├── GPT1Config.java          # GPT-1模型配置类
├── GPT1TokenEmbedding.java  # Token和位置嵌入层
├── GPT1TransformerBlock.java # Transformer解码器块
├── GPT1OutputHead.java      # 输出投影层
├── GPT1Block.java           # GPT-1核心模型块（继承Block）
├── GPT1Model.java           # GPT-1模型封装（继承Model）
├── GPT1Example.java         # 使用示例
├── GptDemo.java             # 综合演示程序
└── README.md                # 本文档
```

## 核心特性

### 🏗️ 架构特点
- **仅解码器架构**：使用Transformer解码器，适合自回归语言建模
- **因果掩码**：多头自注意力使用因果掩码，防止未来信息泄露
- **Post-LayerNorm**：与GPT-2不同，使用原始Transformer的Post-LayerNorm结构
- **学习位置嵌入**：使用可学习的位置嵌入而非固定的正弦位置编码

### 📏 模型规格
- **词汇表大小**：默认40,000（可配置）
- **最大序列长度**：默认512（可配置）
- **隐藏维度**：默认768（可配置）
- **Transformer层数**：默认12（可配置）
- **注意力头数**：默认12（可配置）
- **激活函数**：GELU

## 快速开始

### 1. 创建模型

```java
// 创建小型演示模型
GPT1Model model = GPT1Model.createTinyModel("my-gpt1");

// 创建中型模型
GPT1Model model = GPT1Model.createMediumModel("medium-gpt1");

// 创建完整GPT-1模型
GPT1Model model = GPT1Model.createFullModel("full-gpt1", 40000);

// 使用自定义配置
GPT1Config config = new GPT1Config(5000, 256, 512, 8, 8);
GPT1Model model = new GPT1Model("custom-gpt1", config);
```

### 2. 基础使用

```java
// 前向传播
int[] inputTokens = {1, 2, 3, 4, 5};
Variable logits = model.predictNextToken(inputTokens);

// 文本生成
List<Integer> prompt = Arrays.asList(1, 2, 3);
List<Integer> generated = model.generateText(prompt, 50, 1.0);

// 批量处理
Variable batchInput = createBatchInput(batchSize, seqLength);
Variable batchOutput = model.batchPredict(batchInput);
```

### 3. 运行演示

```java
// 运行完整演示
GptDemo.main(new String[0]);

// 运行特定演示模块
GptDemo.runDemo("quick");        // 快速开始
GptDemo.runDemo("detailed");     // 详细功能
GptDemo.runDemo("architecture"); // 架构展示
GptDemo.runDemo("performance");  // 性能测试
```

## 详细说明

### GPT1Config 配置类

管理模型的所有超参数和配置选项：

```java
GPT1Config config = new GPT1Config(
    vocabSize,           // 词汇表大小
    maxSequenceLength,   // 最大序列长度
    hiddenSize,          // 隐藏层维度
    numLayers,           // Transformer层数
    numAttentionHeads,   // 注意力头数
    intermediateSize,    // 前馈网络维度
    residualDropoutProb, // 残差dropout概率
    embeddingDropoutProb,// 嵌入dropout概率
    attentionDropoutProb,// 注意力dropout概率
    layerNormEpsilon,    // 层归一化epsilon
    initializerRange,    // 参数初始化范围
    activationFunction   // 激活函数类型
);

// 验证配置有效性
config.validate();
```

### GPT1Block 核心模型

继承自TinyAI的Block类，实现完整的GPT-1架构：

```java
// 创建GPT1Block
GPT1Block block = new GPT1Block("gpt1-block", config);

// 前向传播
Variable input = new Variable(inputTensorIds);
Variable output = block.layerForward(input);

// 获取模型信息
long paramCount = block.getParameterCount();
block.printModelInfo();
```

### GPT1Model 模型封装

继承自TinyAI的Model类，提供高级接口：

```java
// 创建模型
GPT1Model model = new GPT1Model("my-gpt1", config);

// 语言建模
Variable logits = model.predict(tokenIds);

// 文本生成
List<Integer> generated = model.generateText(prompt, maxLength, temperature);

// 模型保存和加载
model.saveModel("path/to/model.tinyai");
GPT1Model loaded = (GPT1Model) GPT1Model.loadModel("path/to/model.tinyai");
```

## 架构细节

### 1. Token嵌入层 (GPT1TokenEmbedding)
- 将离散token ID转换为连续向量
- 包含token嵌入和位置嵌入
- 支持dropout正则化

### 2. Transformer块 (GPT1TransformerBlock)
- 多头自注意力（带因果掩码）
- 前馈网络
- 残差连接和层归一化
- Post-LayerNorm结构

### 3. 输出头 (GPT1OutputHead)
- 线性投影到词汇表维度
- 支持权重共享（与token嵌入共享）
- 可选偏置项

## 与GPT-2的区别

| 特性 | GPT-1 | GPT-2 |
|------|-------|-------|
| 层归一化 | Post-LayerNorm | Pre-LayerNorm |
| 位置编码 | 学习位置嵌入 | 学习位置嵌入 |
| 激活函数 | GELU | GELU |
| 架构 | 仅解码器 | 仅解码器 |
| 掩码机制 | 因果掩码 | 因果掩码 |

## 性能优化建议

### 1. 模型大小选择
- **开发/测试**：使用Tiny配置（256维，6层）
- **实验**：使用Medium配置（512维，8层）
- **生产**：使用Full配置（768维，12层）

### 2. 内存优化
- 限制批次大小和序列长度
- 使用权重共享减少参数量
- 适当配置dropout概率

### 3. 训练建议
- 使用学习率预热和衰减
- 梯度裁剪防止梯度爆炸
- 定期保存检查点

## 示例代码

### 完整的使用示例

```java
public class GPT1Usage {
    public static void main(String[] args) {
        // 1. 创建模型
        GPT1Model model = GPT1Model.createTinyModel("demo");
        
        // 2. 显示模型信息
        model.printModelInfo();
        
        // 3. 准备输入数据
        int[] prompt = {1, 2, 3, 4};
        
        // 4. 预测下一个token
        Variable nextTokenLogits = model.predictNextToken(prompt);
        System.out.println("Next token logits shape: " + 
                          nextTokenLogits.getValue().getShape());
        
        // 5. 生成文本序列
        List<Integer> promptList = Arrays.asList(1, 2, 3);
        List<Integer> generated = model.generateText(promptList, 20, 1.0);
        System.out.println("Generated sequence: " + generated);
        
        // 6. 保存模型
        model.saveModel("my_gpt1_model.tinyai");
        
        // 7. 加载模型
        GPT1Model loadedModel = (GPT1Model) GPT1Model.loadModel("my_gpt1_model.tinyai");
        System.out.println("Model loaded successfully!");
    }
}
```

## 常见问题

### Q: 如何调整模型大小？
A: 通过GPT1Config类调整hiddenSize、numLayers、numAttentionHeads等参数。

### Q: 支持哪些激活函数？
A: 目前支持GELU和ReLU，推荐使用GELU以获得更好的性能。

### Q: 如何处理长序列？
A: 确保序列长度不超过maxSequenceLength，或者考虑序列分割和拼接。

### Q: 模型训练需要多长时间？
A: 这取决于数据集大小、模型规模和硬件配置。建议从小模型开始实验。

## 技术支持

如果您在使用过程中遇到问题，请：

1. 检查配置参数是否合理
2. 确认输入数据格式正确
3. 查看详细的错误信息和堆栈跟踪
4. 参考示例代码和演示程序

## 未来改进

- [ ] 实现真正的dropout功能
- [ ] 添加更多采样策略（top-k、top-p）
- [ ] 支持beam search解码
- [ ] 优化内存使用和计算效率
- [ ] 添加更多预训练检查点

---

**注意**：本实现基于TinyAI框架，主要用于教育和研究目的。在生产环境中使用前，请进行充分的测试和优化。