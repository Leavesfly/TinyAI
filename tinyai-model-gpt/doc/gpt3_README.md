# GPT-3 模型实现

基于TinyAI框架实现的GPT-3语言模型，支持多种规模配置和先进特性。

## 📁 文件结构

```
tinyai-model-gpt/src/main/java/io/leavesfly/tinyai/gpt3/
├── GPT3Config.java              # GPT-3配置类
├── GPT3Model.java               # GPT-3模型类（继承Model）
├── GPT3TransformerBlock.java    # GPT-3 Transformer块（继承Block）
├── GPT3RotaryEmbedding.java     # 旋转位置编码(RoPE)
├── GPT3SparseAttention.java     # 稀疏注意力机制
├── GPT3Demo.java                # 演示程序
└── GPT3Test.java                # 测试套件
```

## 🎯 核心特性

### 1. 多规模模型支持
- **小型模型 (125M参数)**: 768维, 12层, 12头
- **中型模型 (350M参数)**: 1024维, 24层, 16头  
- **大型模型 (1.3B参数)**: 2048维, 24层, 32头
- **超大型模型 (175B参数)**: 12288维, 96层, 96头

### 2. 架构设计
- **解码器-only Transformer**: 专为自回归语言建模设计
- **Pre-LayerNorm结构**: 提高训练稳定性
- **并行注意力和MLP**: GPT-3的关键优化
- **残差连接**: 支持深层网络训练

### 3. 先进特性
- **旋转位置编码(RoPE)**: 更好的位置感知能力
- **稀疏注意力机制**: 支持更长序列，降低计算复杂度
- **Few-shot学习**: 强大的上下文学习能力
- **梯度检查点**: 内存效率优化

## 🚀 快速开始

### 基本使用

```java
// 创建GPT-3模型
GPT3Model model = GPT3Model.createSmallModel("my-gpt3");

// 前向传播
NdArray tokenIds = NdArray.of(Shape.of(1, 10)); // 输入token序列
Variable output = model.forward(new Variable(tokenIds));

// 文本生成
NdArray generated = model.generateSequence(tokenIds, 20);

// Few-shot学习
NdArray context = createFewShotContext(); // 创建上下文
NdArray result = model.fewShotGenerate(context, 15);
```

### 自定义配置

```java
// 创建自定义配置
GPT3Config config = new GPT3Config(
    50000,  // vocab_size
    1024,   // n_positions  
    512,    // n_embd
    8,      // n_layer
    8       // n_head
);

// 启用高级特性
config.setSparseAttention(true);
config.setParallelAttention(true);
config.setRotaryPct(0.25);

// 创建模型
GPT3Model model = new GPT3Model("custom-gpt3", config);
```

## 🧪 运行演示

### 1. 基础演示
```bash
# 在TinyAI项目根目录下编译并运行
javac -cp "your-classpath" tinyai-model-gpt/src/main/java/io/leavesfly/tinyai/gpt3/GPT3Demo.java
java io.leavesfly.tinyai.gpt3.GPT3Demo
```

### 2. 测试套件
```bash
java io.leavesfly.tinyai.gpt3.GPT3Test
```

## 📊 模型对比

| 规模 | 参数量 | 嵌入维度 | 层数 | 注意力头 | 稀疏注意力 |
|------|--------|----------|------|----------|------------|
| 小型 | 125M | 768 | 12 | 12 | ❌ |
| 中型 | 350M | 1024 | 24 | 16 | ❌ |
| 大型 | 1.3B | 2048 | 24 | 32 | ✅ |
| 超大型 | 175B | 12288 | 96 | 96 | ✅ |

## 🏗️ 架构说明

### GPT3Model (继承Model)
- 封装完整的GPT-3模型
- 提供高级API接口
- 支持模型保存和加载
- 集成训练和推理功能

### GPT3TransformerBlock (继承Block) 
- 实现单个Transformer解码器块
- 支持并行和串行计算模式
- 集成层归一化、注意力和前馈网络
- 可配置的dropout和激活函数

### 核心组件
- **GPT3Config**: 统一的配置管理
- **GPT3RotaryEmbedding**: 旋转位置编码实现
- **GPT3SparseAttention**: 稀疏注意力机制
- **复用组件**: 使用tinyai-nnet的LayerNorm、FeedForward等

## 🔧 技术实现

### 1. 继承关系
```
GPT3Model extends Model
GPT3TransformerBlock extends Block
GPT3RotaryEmbedding extends Layer
GPT3SparseAttention extends Layer
```

### 2. 优先使用现有组件
- `LayerNorm`: 来自tinyai-nnet.layer.transformer
- `MultiHeadAttention`: 来自tinyai-nnet.layer.transformer  
- `FeedForward`: 来自tinyai-nnet.layer.transformer
- `GPT2TokenEmbedding`: 复用GPT-2的嵌入层
- `GPT2OutputHead`: 复用GPT-2的输出头

### 3. 创新特性
- **并行计算**: 注意力和MLP同时计算
- **旋转编码**: 改进的位置表示
- **稀疏注意力**: 高效处理长序列
- **Few-shot**: 上下文学习能力

## 📈 性能特点

### 优势
- ✅ 模块化设计，易于扩展
- ✅ 支持多种模型规模
- ✅ 兼容TinyAI训练框架
- ✅ 完整的测试覆盖
- ✅ 详细的文档和示例

### 适用场景
- 🎯 自然语言生成
- 🎯 文本补全和续写
- 🎯 Few-shot任务学习
- 🎯 对话系统
- 🎯 代码生成

## 🔍 代码示例

### 模型信息展示
```java
GPT3Model model = GPT3Model.createLargeModel("gpt3-large");
model.printModelInfo();
// 输出模型架构、参数数量、配置信息等
```

### 旋转位置编码
```java
GPT3RotaryEmbedding rope = new GPT3RotaryEmbedding("rope", 64, 2048);
Variable[] rotated = rope.applyRotaryPositionEmbedding(query, key, seqLen);
```

### 稀疏注意力
```java
GPT3Config config = GPT3Config.createLargeConfig();
config.setSparseAttention(true);

GPT3SparseAttention sparseAttn = new GPT3SparseAttention(
    "sparse_attn", config.getNEmbd(), config.getNHead(), 0, config
);
```

## 🎓 学习资源

### 相关论文
- "Language Models are Few-Shot Learners" (GPT-3)
- "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- "Sparse Transformers: Attention with Local and Strided Patterns"

### 技术博客
- GPT-3架构详解
- 旋转位置编码原理
- 稀疏注意力机制

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进GPT-3实现！

### 开发环境
- Java 8+
- TinyAI框架依赖
- 充足的内存（大型模型需要）

### 扩展建议
- [ ] 添加更多激活函数支持
- [ ] 实现更多稀疏注意力模式
- [ ] 支持模型并行训练
- [ ] 添加量化支持

---

*基于TinyAI框架实现，遵循架构设计原则，提供企业级的GPT-3模型实现。*