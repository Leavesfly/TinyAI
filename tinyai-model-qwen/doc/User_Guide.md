# Qwen3模型使用指南

## 概述

本指南介绍如何使用基于TinyAI框架实现的Qwen3大语言模型。该实现包含了完整的Transformer解码器架构，支持现代大语言模型的各种先进特性。

## 快速开始

### 基础使用

```java
// 1. 创建配置
Qwen3Config config = Qwen3Config.createDemoConfig();

// 2. 初始化模型
Qwen3Model model = new Qwen3Model("qwen3-demo", config);

// 3. 准备输入
NdArray inputIds = NdArray.of(Shape.of(1, 10)); // [batch_size, seq_len]
// 填充token ID...

// 4. 前向传播
Variable output = model.forward(new Variable(inputIds));
// 输出形状: [batch_size, seq_len, vocab_size]
```

### 使用预设配置

```java
// 小型测试配置
Qwen3Model smallModel = Qwen3Model.createSmallModel("qwen3-small");

// 演示配置
Qwen3Model demoModel = Qwen3Model.createDemoModel("qwen3-demo");

// 自定义配置
Qwen3Config customConfig = new Qwen3Config();
customConfig.setVocabSize(50000);
customConfig.setHiddenSize(1024);
customConfig.setNumHiddenLayers(12);
// ... 设置其他参数
Qwen3Model customModel = new Qwen3Model("custom", customConfig);
```

## 核心组件

### 1. Qwen3Config - 配置管理

```java
Qwen3Config config = new Qwen3Config();

// 基础配置
config.setVocabSize(32000);           // 词汇表大小
config.setHiddenSize(2048);           // 隐藏层维度
config.setIntermediateSize(5632);     // 前馈网络中间层维度
config.setNumHiddenLayers(24);        // 解码器层数
config.setNumAttentionHeads(16);      // 注意力头数
config.setNumKeyValueHeads(16);       // 键值头数（用于GQA）

// 位置编码配置
config.setMaxPositionEmbeddings(8192); // 最大序列长度
config.setRopeTheta(10000.0);          // RoPE基础频率

// 归一化配置
config.setRmsNormEps(1e-6);            // RMSNorm epsilon

// 特殊token配置
config.setPadTokenId(0);               // 填充token
config.setBosTokenId(1);               // 开始token
config.setEosTokenId(2);               // 结束token

// 验证配置
config.validate(); // 检查配置有效性
```

### 2. Qwen3Model - 完整模型

```java
// 创建模型
Qwen3Model model = new Qwen3Model("my-qwen3", config);

// 模型信息
long paramCount = model.countParameters();
double sizeMB = model.getModelSizeMB();
String summary = model.getModelSummary();
String detailedInfo = model.getModelDetailedInfo();

// 前向传播
Variable output = model.forward(new Variable(inputIds));

// 输入验证
int[][] validInput = {{1, 2, 3}, {4, 5, 6}};
model.validateInput(validInput);

// 模式设置
model.setInferenceMode();  // 推理模式
model.setTrainingMode();   // 训练模式

// 模型保存和加载（继承自TinyAI Model类）
model.saveModel("model.bin");
Qwen3Model loadedModel = (Qwen3Model) Model.loadModel("model.bin");
```

### 3. Qwen3Block - 核心网络块

```java
// 创建不带语言模型头的Block
Qwen3Block block = new Qwen3Block("qwen3-block", config, false);

// 创建带语言模型头的Block
Qwen3Block blockWithLM = new Qwen3Block("qwen3-block-lm", config, true);

// 前向传播
Variable hiddenStates = block.layerForward(new Variable(inputIds));

// 参数统计
long blockParams = block.countParameters();

// 获取组件
Embedding embedTokens = block.getEmbedTokens();
Qwen3DecoderLayer[] decoders = block.getDecoderLayers();
RMSNormLayer finalNorm = block.getFinalNorm();
```

## 演示程序

### 1. 完整演示 (Qwen3Demo)

```java
// 运行完整演示
Qwen3Demo.main(new String[]{});

// 或分别运行各个演示
Qwen3Demo.modelInfoDemo();      // 模型信息展示
Qwen3Demo.tokenizerDemo();      // 分词器功能演示
Qwen3Demo.textGenerationDemo(); // 文本生成演示
Qwen3Demo.chatDemo();           // 聊天对话演示
```

### 2. 快速演示 (Qwen3QuickDemo)

```java
// 运行快速演示
Qwen3QuickDemo.main(new String[]{});

// 或分别运行
Qwen3QuickDemo.quickStart();        // 基础使用演示
Qwen3QuickDemo.configDemo();        // 配置对比演示
Qwen3QuickDemo.performanceDemo();   // 性能测试演示
```

### 3. 使用简单分词器

```java
Qwen3Demo.SimpleTokenizer tokenizer = new Qwen3Demo.SimpleTokenizer(32000);

// 单文本编码
List<Integer> tokens = tokenizer.encode("你好，世界！");
String decoded = tokenizer.decode(tokens);

// 批量编码
List<String> texts = Arrays.asList("文本1", "文本2", "文本3");
Qwen3Demo.TokenizerResult result = tokenizer.batchEncode(texts, true, null);
```

### 4. 聊天机器人

```java
Qwen3Model model = Qwen3Model.createDemoModel("chatbot");
Qwen3Demo.SimpleTokenizer tokenizer = new Qwen3Demo.SimpleTokenizer();
Qwen3Demo.Qwen3ChatBot chatbot = new Qwen3Demo.Qwen3ChatBot(model, tokenizer);

// 设置生成参数
chatbot.setGenerationParams(100, 0.7, 0.9, 50);

// 开始对话
String response1 = chatbot.chat("你好");
String response2 = chatbot.chat("介绍一下自己");

// 获取对话历史
List<Map<String, String>> history = chatbot.getConversationHistory();

// 清除历史
chatbot.clearHistory();
```

## 配置参考

### 预设配置对比

| 配置类型 | 参数量 | 隐藏维度 | 层数 | 注意力头 | 用途 |
|---------|--------|----------|------|----------|------|
| 小型配置 | ~16M | 512 | 4 | 8 | 概念验证、测试 |
| 演示配置 | ~62M | 512 | 6 | 8 | 功能展示、学习 |
| 标准配置 | ~1.8B | 2048 | 24 | 16 | 实际应用 |

### 配置创建方法

```java
// 预设配置
Qwen3Config smallConfig = Qwen3Config.createSmallConfig();
Qwen3Config demoConfig = Qwen3Config.createDemoConfig();

// 自定义配置
Qwen3Config customConfig = new Qwen3Config();
// 设置各种参数...
customConfig.validate(); // 验证配置
```

## 架构特性

### 1. RMS归一化 (RMSNorm)
- 相比LayerNorm减少计算量
- 提升训练稳定性
- 现代LLM标准选择

```java
RMSNormLayer rmsNorm = new RMSNormLayer("norm", hiddenSize, eps);
Variable normalized = rmsNorm.layerForward(input);
```

### 2. 旋转位置编码 (RoPE)
- 支持长序列外推
- 相对位置编码
- 保持向量模长不变

```java
RotaryPositionalEmbeddingLayer rope = 
    new RotaryPositionalEmbeddingLayer("rope", headDim);
NdArray[] rotated = rope.applyRotaryPosEmb(query, key, seqLen);
```

### 3. 分组查询注意力 (GQA)
- 减少40-60%的KV缓存内存
- 保持模型表达能力
- 支持高效推理

```java
Qwen3AttentionLayer attention = new Qwen3AttentionLayer("attn", config);
Variable attentionOutput = attention.layerForward(hiddenStates);
```

### 4. SwiGLU激活函数
- 结合Swish激活和门控机制
- 提升非线性表达能力
- 现代LLM标准激活函数

```java
Qwen3MLPLayer mlp = new Qwen3MLPLayer("mlp", config);
Variable mlpOutput = mlp.layerForward(hiddenStates);
```

## 性能优化

### 1. 批量推理
```java
// 使用更大的批次大小提高吞吐量
int batchSize = 4; // 根据内存调整
NdArray batchInput = NdArray.of(Shape.of(batchSize, seqLen));
// 填充批次数据...
Variable batchOutput = model.forward(new Variable(batchInput));
```

### 2. 序列长度优化
```java
// 根据实际需要调整序列长度
int optimalSeqLen = 512; // 平衡性能和功能
NdArray input = NdArray.of(Shape.of(batchSize, optimalSeqLen));
```

### 3. 模型大小选择
```java
// 根据应用场景选择合适的模型配置
if (useCase.equals("testing")) {
    model = Qwen3Model.createSmallModel("test");
} else if (useCase.equals("demo")) {
    model = Qwen3Model.createDemoModel("demo");
} else {
    // 使用自定义大型配置
    model = new Qwen3Model("production", customLargeConfig);
}
```

## 错误处理

### 1. 配置验证
```java
try {
    config.validate();
} catch (IllegalArgumentException e) {
    System.err.println("配置错误: " + e.getMessage());
    // 修正配置...
}
```

### 2. 输入验证
```java
try {
    model.validateInput(inputData);
} catch (IllegalArgumentException e) {
    System.err.println("输入数据错误: " + e.getMessage());
    // 修正输入数据...
}
```

### 3. 形状检查
```java
// 确保输入形状正确
if (inputIds.getShape().getDimNum() != 2) {
    throw new IllegalArgumentException("输入必须是2D张量 [batch_size, seq_len]");
}
```

## 扩展开发

### 1. 自定义层
```java
// 继承现有层并自定义
public class CustomLayer extends Qwen3DecoderLayer {
    public CustomLayer(String name, Qwen3Config config) {
        super(name, config);
        // 添加自定义初始化...
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        // 自定义前向传播逻辑
        return super.layerForward(inputs);
    }
}
```

### 2. 自定义配置
```java
// 扩展配置类
public class ExtendedQwen3Config extends Qwen3Config {
    private boolean useCustomFeature = false;
    
    // 添加新的配置选项...
    public boolean isUseCustomFeature() { return useCustomFeature; }
    public void setUseCustomFeature(boolean use) { this.useCustomFeature = use; }
}
```

### 3. 模型微调
```java
// 获取模型参数进行微调
Map<String, Parameter> params = model.getAllParams();
// 实现参数更新逻辑...
```

## 常见问题

### Q1: 如何调整模型大小？
A: 通过修改配置参数来调整模型大小：
```java
config.setHiddenSize(1024);        // 调整隐藏维度
config.setNumHiddenLayers(12);     // 调整层数
config.setNumAttentionHeads(8);    // 调整注意力头数
```

### Q2: 如何处理长序列？
A: Qwen3支持通过RoPE处理长序列：
```java
config.setMaxPositionEmbeddings(4096); // 增加最大序列长度
```

### Q3: 如何减少内存使用？
A: 可以通过以下方式减少内存：
- 使用较小的批次大小
- 使用分组查询注意力（已内置）
- 选择较小的模型配置

### Q4: 如何添加新的特殊token？
A: 扩展SimpleTokenizer类：
```java
public class ExtendedTokenizer extends Qwen3Demo.SimpleTokenizer {
    // 添加新的特殊token处理逻辑
}
```

## 性能基准

基于演示配置的性能数据（仅供参考）：

| 批次大小 | 序列长度 | 推理时间 | 内存使用 |
|---------|----------|----------|----------|
| 1 | 10 | ~50ms | ~100MB |
| 2 | 10 | ~80ms | ~150MB |
| 4 | 10 | ~120ms | ~200MB |
| 1 | 20 | ~80ms | ~120MB |

*注：实际性能取决于硬件配置和Java虚拟机设置*

## 相关资源

- [TinyAI框架文档](../README.md)
- [Qwen3模型架构说明](./25_README_qwen3.md)
- [API参考文档](./API_Reference.md)
- [Python版本实现对比](./25_qwen3_demo.py)

## 联系支持

如有问题或建议，请通过以下方式联系：
- 项目Issues
- 技术讨论
- 功能建议

---

*文档版本：1.0*  
*最后更新：2025年10月4日*