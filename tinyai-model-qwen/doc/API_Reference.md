# Qwen3模型 API 参考文档

## 概述

本文档详细介绍了基于TinyAI架构实现的Qwen3大语言模型的所有API接口、类和方法。

## 核心类

### 1. Qwen3Config

配置类，管理Qwen3模型的所有超参数。

#### 构造方法

```java
public Qwen3Config()
```
使用默认参数创建配置实例。

```java
public static Qwen3Config createTinyConfig()
```
创建用于测试和演示的小型配置。

#### 主要属性

| 属性名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| vocabSize | int | 151936 | 词汇表大小 |
| hiddenSize | int | 4096 | 隐藏层维度 |
| numHiddenLayers | int | 32 | Transformer层数 |
| numAttentionHeads | int | 32 | 注意力头数 |
| numKeyValueHeads | int | 32 | KV头数（GQA） |
| intermediateSize | int | 11008 | 前馈网络中间层维度 |
| maxPositionEmbeddings | int | 32768 | 最大位置编码长度 |
| rmsNormEps | float | 1e-6f | RMSNorm的epsilon值 |
| ropeTheta | float | 1000000.0f | RoPE的theta参数 |

#### 主要方法

```java
public int getHeadDim()
```
返回每个注意力头的维度（hiddenSize / numAttentionHeads）。

```java
public int getNumKeyValueGroups()
```
返回KV头分组数（numAttentionHeads / numKeyValueHeads）。

### 2. Qwen3Model

主模型类，继承自TinyAI的Model类，提供完整的语言模型功能。

#### 构造方法

```java
public Qwen3Model(String name, Qwen3Config config)
```
使用指定名称和配置创建模型实例。

```java
public static Qwen3Model createTinyModel(String name)
```
创建用于测试的小型模型实例。

#### 核心方法

```java
public Variable forwardWithLogits(Variable inputIds)
```
- **功能**: 执行前向传播，返回logits
- **参数**: 
  - inputIds: 输入token序列，形状为(seq_len,)或(batch_size, seq_len)
- **返回**: 输出logits，形状为(batch_size, seq_len, vocab_size)

```java
public int predictNextToken(NdArray inputIds)
```
- **功能**: 预测序列的下一个token
- **参数**: inputIds - 输入token序列
- **返回**: 预测的token ID

```java
public NdArray generate(NdArray inputIds, int maxLength)
```
- **功能**: 自回归文本生成
- **参数**: 
  - inputIds: 初始输入序列
  - maxLength: 生成的最大长度
- **返回**: 完整的生成序列

```java
public void printModelInfo()
```
打印详细的模型信息，包括架构和参数统计。

### 3. Qwen3Block

核心网络块，继承自TinyAI的Block类，实现完整的Transformer架构。

#### 构造方法

```java
public Qwen3Block(String name, Qwen3Config config)
```
使用指定名称和配置创建网络块。

#### 主要方法

```java
public Variable layerForward(Variable... inputs)
```
- **功能**: 执行前向传播
- **参数**: inputs[0] - 输入token IDs
- **返回**: 隐藏状态表示

```java
public String getModelStats()
```
返回模型统计信息字符串。

```java
public void printArchitecture()
```
打印模型架构信息。

### 4. 组件类

#### RMSNorm

RMS归一化层实现。

```java
public RMSNorm(String name, int hiddenSize, float eps)
public RMSNorm(String name, int hiddenSize) // 使用默认eps=1e-6f
```

**特性**:
- 支持2D和3D输入
- 数值稳定的实现
- 可训练的缩放参数

#### SiLULayer

SiLU/Swish激活函数层。

```java
public SiLULayer(String name, Shape inputShape)
public SiLULayer(String name) // 任意形状
```

**特性**:
- 数值稳定的sigmoid计算
- 支持1D、2D、3D输入
- 静态方法支持直接调用

#### RotaryPositionalEmbedding

旋转位置编码实现。

```java
public RotaryPositionalEmbedding(String name, int dim, int maxPositionEmbeddings, float base)
public RotaryPositionalEmbedding(String name, int dim, int maxPositionEmbeddings) // 默认base=10000.0f
```

**特性**:
- 支持相对位置编码
- 任意序列长度处理
- 高效的旋转变换实现

#### Qwen3Attention

多头注意力机制，支持GQA（分组查询注意力）。

```java
public Qwen3Attention(String name, Qwen3Config config, int layerIdx, boolean useMask)
public Qwen3Attention(String name, Qwen3Config config, int layerIdx) // 默认使用掩码
```

**特性**:
- 分组查询注意力（GQA）
- 集成RoPE位置编码
- 因果掩码支持
- 缩放点积注意力

#### Qwen3MLP

前馈神经网络，使用SwiGLU激活。

```java
public Qwen3MLP(String name, Qwen3Config config)
```

**特性**:
- SwiGLU激活函数
- 门控线性单元
- 三层架构（gate_proj, up_proj, down_proj）

#### Qwen3DecoderLayer

Transformer解码器层。

```java
public Qwen3DecoderLayer(String name, Qwen3Config config, int layerIdx)
```

**特性**:
- Pre-LayerNorm架构
- 残差连接
- 组合自注意力和MLP

## 使用示例

### 基本使用

```java
// 创建模型
Qwen3Model model = Qwen3Model.createTinyModel("my_qwen3");

// 准备输入
NdArray inputIds = NdArray.of(new float[]{1, 15, 25, 35, 45});

// 前向传播
Variable logits = model.forwardWithLogits(new Variable(inputIds));

// 预测下一个token
int nextToken = model.predictNextToken(inputIds);

// 文本生成
NdArray generated = model.generate(inputIds, 20);
```

### 批次处理

```java
// 批次输入 (batch_size=2, seq_len=5)
NdArray batchInput = NdArray.of(Shape.of(2, 5));
// ... 填充数据

// 批次前向传播
Variable batchLogits = model.forwardWithLogits(new Variable(batchInput));

// 批次生成
NdArray batchGenerated = model.generate(batchInput, 15);
```

### 模型信息查看

```java
// 打印完整模型信息
model.printModelInfo();

// 获取配置
Qwen3Config config = model.getConfig();
System.out.println("模型层数: " + config.getNumHiddenLayers());
System.out.println("注意力头数: " + config.getNumAttentionHeads());
```

## 异常处理

### 常见异常

1. **IllegalArgumentException**: 输入维度不匹配
2. **IndexOutOfBoundsException**: Token ID超出词汇表范围
3. **RuntimeException**: 模型未正确初始化

### 最佳实践

1. 总是验证输入token ID范围 [0, vocab_size)
2. 确保输入形状符合预期
3. 处理长序列时注意内存使用
4. 使用小型配置进行开发和测试

## 性能优化

### 推荐配置

- **开发/测试**: 使用 `createTinyConfig()`
- **小型部署**: 4-8层，512-1024隐藏维度
- **生产环境**: 根据硬件资源调整层数和维度

### 内存优化

- 使用较小的批次大小
- 适当设置最大序列长度
- 考虑使用GQA减少KV缓存

## 兼容性

- **TinyAI版本**: 兼容当前框架版本
- **Java版本**: 需要Java 8+
- **依赖**: 依赖tinyai-nnet、tinyai-func、tinyai-ndarr模块