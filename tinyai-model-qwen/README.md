# TinyAI-Model-Qwen

[![Java](https://img.shields.io/badge/Java-17+-blue.svg)](https://www.oracle.com/java/)
[![TinyAI](https://img.shields.io/badge/TinyAI-1.0.0-green.svg)](https://github.com/leavesfly/TinyAI)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)
[![Maven](https://img.shields.io/badge/Maven-3.6+-red.svg)](https://maven.apache.org/)

## 📖 项目概述

`tinyai-model-qwen` 是 TinyAI 框架中基于 **Qwen3 大语言模型** 架构的完整实现模块。该模块提供了现代 Transformer 架构的所有核心特性，包括分组查询注意力(GQA)、旋转位置编码(RoPE)、SwiGLU激活函数、RMSNorm归一化等先进技术，为大语言模型的研究和应用提供了强大的基础设施。

## 🚀 核心特性

### ✨ 主要优势

- **🎯 现代架构**: 基于 Transformer 的 decoder-only 架构，遵循 Qwen3 官方设计
- **⚡ 分组查询注意力(GQA)**: 显著减少 KV 缓存内存占用，提升推理效率
- **🔄 旋转位置编码(RoPE)**: 支持任意长度序列的相对位置编码，具备外推能力
- **🎆 SwiGLU 激活**: 门控线性单元，提升模型表达能力和性能
- **📊 RMSNorm 归一化**: 简化高效的归一化方法，减少计算开销
- **🏢 Pre-LayerNorm**: 训练稳定的架构设计，避免梯度消失问题

### 🏗️ 架构遵循

- **🧩 Qwen3Block**: 继承 TinyAI 的 `Block` 类，实现核心神经网络计算图
- **🌍 Qwen3Model**: 继承 TinyAI 的 `Model` 类，提供完整模型封装和管理
- **🔧 组件复用**: 优先使用 tinyai-nnet 已有实现（LinearLayer、Embedding 等）
- **📝 中文文档**: 完整的中文注释和文档支持，符合用户偏好

### 🚀 功能支持

- ✅ 单序列和批次处理
- ✅ 自回归文本生成
- ✅ 灵活的配置系统
- ✅ 模型保存和加载
- ✅ 完整的测试覆盖
- ✅ 详细的性能统计

## 📦 模块结构

```
tinyai-model-qwen/
├── src/main/java/io/leavesfly/tinyai/qwen3/
│   ├── Qwen3Model.java               # 🌍 主模型类，继承 Model
│   ├── Qwen3Block.java               # 🧩 核心网络块，继承 Block
│   ├── Qwen3Config.java              # ⚙️ 模型配置类
│   ├── Qwen3DecoderLayer.java        # 🔄 Transformer 解码器层
│   ├── Qwen3Attention.java           # 👁️ 多头注意力机制（支持 GQA）
│   ├── Qwen3MLP.java                 # 🧠 SwiGLU 前馈网络
│   ├── RMSNorm.java                  # 📊 RMS 归一化层
│   ├── SiLULayer.java                # ⚡ SiLU 激活函数
│   ├── RotaryPositionalEmbedding.java # 🔄 旋转位置编码
│   └── Qwen3Demo.java                # 🎨 完整演示程序
├── src/test/java/                     # 🧪 单元测试
│   └── io/leavesfly/tinyai/qwen3/
│       └── Qwen3Test.java            # ✅ 全面测试用例
├── doc/                               # 📚 技术文档
│   ├── Architecture.md               # 架构设计文档
│   ├── API_Reference.md              # API 参考文档
│   ├── User_Guide.md                 # 用户使用指南
│   ├── Development_Guide.md          # 开发指南
│   └── Deployment_Guide.md           # 部署指南
└── pom.xml                            # 🛠️ Maven 配置

## 📍 依赖关系

```xml
<dependencies>
    <!-- TinyAI 深度学习核心模块 -->
    <dependency>
        <groupId>io.leavesfly.tinyai</groupId>
        <artifactId>tinyai-deeplearning-ml</artifactId>
    </dependency>
    
    <!-- 强化学习模块 -->
    <dependency>
        <groupId>io.leavesfly.tinyai</groupId>
        <artifactId>tinyai-deeplearning-rl</artifactId>
    </dependency>
    
    <!-- GPT 模型依赖 -->
    <dependency>
        <groupId>io.leavesfly.tinyai</groupId>
        <artifactId>tinyai-model-gpt</artifactId>
    </dependency>
</dependencies>
```

## 🚀 快速开始

### 环境要求

- **Java**: 17 或更高版本
- **Maven**: 3.6+
- **TinyAI**: 框架依赖
- **内存**: 推荐 8GB+ （大型模型）

### 💻 基础使用示例

```java
import io.leavesfly.tinyai.qwen3.*;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.func.Variable;

public class Qwen3QuickStart {
    public static void main(String[] args) {
        // 1. 创建小型 Qwen3 模型（适合测试和学习）
        Qwen3Model model = Qwen3Model.createTinyModel("demo_qwen3");
        
        // 2. 准备输入数据（token ID 序列）
        NdArray inputIds = NdArray.of(new float[]{1, 15, 25, 35, 45});
        
        // 3. 执行前向传播，获得 logits
        Variable logits = model.forwardWithLogits(new Variable(inputIds));
        System.out.println("输出形状: " + logits.getValue().getShape());
        
        // 4. 预测下一个 token
        int nextToken = model.predictNextToken(inputIds);
        System.out.println("预测的下一个 token: " + nextToken);
        
        // 5. 文本生成（自回归生成 20 个 token）
        NdArray generated = model.generate(inputIds, 20);
        System.out.println("生成序列长度: " + generated.getShape().getDimension(1));
        
        // 6. 查看模型信息
        model.printModelInfo();
    }
}
```

### 🔧 预设模型配置

```java
// 🐜 超小型模型（适合测试和学习）
Qwen3Model tinyModel = Qwen3Model.createTinyModel("tiny_qwen3");
// 配置: 1K 词汇表，256 维度，4 层，8 头

// 📈 中型模型（适合实验和原型验证）
Qwen3Config mediumConfig = new Qwen3Config();
mediumConfig.setVocabSize(32000);
mediumConfig.setHiddenSize(768);
mediumConfig.setNumHiddenLayers(12);
mediumConfig.setNumAttentionHeads(12);
Qwen3Model mediumModel = new Qwen3Model("medium_qwen3", mediumConfig);

// 🚀 大型模型（接近生产环境配置）
Qwen3Config largeConfig = new Qwen3Config();
largeConfig.setVocabSize(151936);  // Qwen3 官方词汇表大小
largeConfig.setHiddenSize(4096);
largeConfig.setNumHiddenLayers(32);
largeConfig.setNumAttentionHeads(32);
largeConfig.setNumKeyValueHeads(32); // GQA 支持
Qwen3Model largeModel = new Qwen3Model("large_qwen3", largeConfig);
```

## 🔍 技术架构

### 核心组件详解

#### 1. 🌍 模型封装层 (Qwen3Model)
```java
// 设计原则：继承 TinyAI 的 Model 类，提供统一的模型管理接口
public class Qwen3Model extends Model {
    private Qwen3Config config;
    private Qwen3Block qwen3Block;
    private LinearLayer lmHead;  // 语言模型头
}
```

#### 2. 🧩 核心网络层 (Qwen3Block)
```java
// 数据流：
Input(token_ids) 
    ↓
EmbedTokens(vocab_size → hidden_size)
    ↓
Layer0(hidden_size → hidden_size)
    ↓
Layer1(hidden_size → hidden_size)
    ↓
...
    ↓
LayerN(hidden_size → hidden_size)
    ↓
FinalNorm(hidden_size → hidden_size)
    ↓
Output(hidden_states)
```

#### 3. 🔄 解码器层 (Qwen3DecoderLayer)
```
输入 → LayerNorm → SelfAttention → 残差连接 → LayerNorm → MLP → 残差连接 → 输出
```

#### 4. 👁️ 注意力机制 (Qwen3Attention)

**分组查询注意力 (GQA)**:
```java
Q: [batch, num_heads, seq_len, head_dim]
K: [batch, num_kv_heads, seq_len, head_dim] 
V: [batch, num_kv_heads, seq_len, head_dim]

// K,V 重复扩展以匹配 Q 的头数
K_expanded: [batch, num_heads, seq_len, head_dim]
V_expanded: [batch, num_heads, seq_len, head_dim]
```

**旋转位置编码 (RoPE)**:
```java
// 公式：
q_m = q * cos(mθ) + rotate(q) * sin(mθ)
k_n = k * cos(nθ) + rotate(k) * sin(nθ)
```

#### 5. 🧠 前馈网络 (Qwen3MLP)

**SwiGLU 激活机制**:
```java
gate = SiLU(gate_proj(x))
up = up_proj(x)
output = down_proj(gate ⊙ up)
```

## 📖 文档

我们提供了完整的中文文档：

- **[API参考文档](doc/API_Reference.md)** - 详细的API接口说明
- **[架构设计文档](doc/Architecture.md)** - 深入的架构设计和技术细节
- **[使用指南](doc/User_Guide.md)** - 从入门到进阶的完整教程
- **[开发指南](doc/Development_Guide.md)** - 开发规范和最佳实践
- **[部署指南](doc/Deployment_Guide.md)** - 生产环境部署完整方案

## 🧪 测试

运行完整的测试套件：

```bash
# 编译项目
mvn clean compile

# 运行所有测试
mvn test

# 运行特定测试
mvn test -Dtest=Qwen3Test

# 查看测试报告
open target/surefire-reports/index.html
```

### 测试覆盖

- ✅ **基本功能测试** - 模型创建、配置验证
- ✅ **架构测试** - 组件初始化、接口兼容
- ✅ **前向传播测试** - 单序列、批次处理
- ✅ **文本生成测试** - 贪心生成、逐步生成
- ✅ **组件单元测试** - RMSNorm、SiLU、RoPE等
- ✅ **性能基准测试** - 推理速度、内存使用

## 🔧 配置参数

### 预设配置

| 配置类型 | 词汇表 | 隐藏维度 | 层数 | 注意力头 | 参数量 | 适用场景 |
|----------|--------|----------|------|----------|--------|----------|
| **Tiny** | 1,000 | 256 | 4 | 8 | ~1.2M | 开发测试 |
| **Small** | 32,000 | 768 | 12 | 12 | ~85M | 轻量部署 |
| **Medium** | 50,000 | 1024 | 24 | 16 | ~340M | 标准应用 |
| **Large** | 151,936 | 4096 | 32 | 32 | ~7B | 生产环境 |

### 性能对比

| 模型大小 | 推理速度 | 内存占用 | 生成质量 | 推荐用途 |
|----------|----------|----------|----------|----------|
| Tiny | 🚀🚀🚀🚀🚀 | 💾 | ⭐⭐ | 开发调试 |
| Small | 🚀🚀🚀🚀 | 💾💾 | ⭐⭐⭐ | 原型验证 |
| Medium | 🚀🚀🚀 | 💾💾💾 | ⭐⭐⭐⭐ | 产品部署 |
| Large | 🚀🚀 | 💾💾💾💾💾 | ⭐⭐⭐⭐⭐ | 高质量应用 |

## 🚀 示例项目

### 简单聊天机器人

```java
public class SimpleChatBot {
    private final Qwen3Model model;
    
    public SimpleChatBot() {
        this.model = Qwen3Model.createTinyModel("chatbot");
    }
    
    public String chat(String message) {
        // 简化的tokenization（实际需要proper tokenizer）
        NdArray inputIds = tokenize(message);
        NdArray response = model.generate(inputIds, 50);
        return detokenize(response);
    }
}
```

### 文本补全工具

```java
public class TextCompletion {
    private final Qwen3Model model;
    
    public TextCompletion() {
        this.model = Qwen3Model.createTinyModel("completion");
    }
    
    public String complete(String prompt, int maxLength) {
        NdArray promptIds = tokenize(prompt);
        NdArray completed = model.generate(promptIds, maxLength);
        return detokenize(completed);
    }
}
```

## 🎨 最佳实践

### 1. 模型选择

```java
// 开发阶段：使用Tiny配置
Qwen3Model devModel = Qwen3Model.createTinyModel("development");

// 生产环境：根据资源选择合适配置
Qwen3Config prodConfig = new Qwen3Config();
prodConfig.setHiddenSize(1024);  // 根据硬件调整
prodConfig.setNumHiddenLayers(16);
Qwen3Model prodModel = new Qwen3Model("production", prodConfig);
```

### 2. 内存优化

```java
// 使用GQA减少内存
config.setNumKeyValueHeads(config.getNumAttentionHeads() / 2);

// 限制序列长度
config.setMaxPositionEmbeddings(2048);

// 小批次处理
int batchSize = 1;  // 根据内存情况调整
```

### 3. 性能监控

```java
// 性能测试示例
public void benchmarkModel() {
    long startTime = System.currentTimeMillis();
    
    for (int i = 0; i < 100; i++) {
        model.forwardWithLogits(new Variable(testInput));
    }
    
    long endTime = System.currentTimeMillis();
    double avgTime = (endTime - startTime) / 100.0;
    System.out.println("平均推理时间: " + avgTime + "ms");
}
```
