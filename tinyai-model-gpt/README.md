# TinyAI GPT模型系列

## 模块概述

`tinyai-model-gpt` 模块是 TinyAI 框架中专门用于实现 GPT（Generative Pre-trained Transformer）系列模型的组件。该模块实现了从 GPT-1 到 GPT-3 的完整架构演进，为自然语言处理和生成任务提供了强大的基础模型支持。

## 主要特性

- **完整的 GPT 架构演进**：从 GPT-1 到 GPT-3 的完整实现
- **模块化设计**：每个 GPT 版本独立实现，便于对比和研究
- **灵活的配置系统**：支持从小型测试模型到大规模生产模型的多种配置
- **统一的编程接口**：继承自 TinyAI 的 Model 和 Block 基类，保持一致的 API
- **自回归文本生成**：支持多种生成策略，包括贪婪搜索和采样生成
- **现代化实现**：基于 Java 的面向对象设计，易于理解和扩展

## 模块结构

```
tinyai-model-gpt/
├── src/main/java/io/leavesfly/tinyai/
│   ├── gpt1/          # GPT-1 实现
│   │   ├── GPT1Model.java
│   │   ├── GPT1Block.java
│   │   ├── GPT1Config.java
│   │   ├── GPT1TokenEmbedding.java
│   │   ├── GPT1TransformerBlock.java
│   │   └── GPT1OutputHead.java
│   ├── gpt2/          # GPT-2 实现
│   │   ├── GPT2Model.java
│   │   ├── GPT2Block.java
│   │   ├── GPT2Config.java
│   │   ├── GPT2TokenEmbedding.java
│   │   ├── GPT2TransformerBlock.java
│   │   └── GPT2OutputHead.java
│   ├── gpt3/          # GPT-3 实现
│   │   ├── GPT3Model.java
│   │   ├── GPT3Config.java
│   │   ├── GPT3MainBlock.java
│   │   └── GPT3TransformerBlock.java
│   └── nlp/           # 通用 NLP 组件
├── doc/               # 文档和参考资料
└── README.md
```

## GPT 模型架构对比

### 整体演进概览

| 特性 | GPT-1 (2018) | GPT-2 (2019) | GPT-3 (2020) |
|------|--------------|--------------|--------------|
| **原始参数量** | 117M | 117M-1.5B | 125M-175B |
| **架构改进** | 基础 Transformer | Pre-LayerNorm | 并行注意力 |
| **序列长度** | 512 | 1024 | 2048 |
| **主要创新** | 无监督预训练 | 零样本学习 | Few-shot 学习 |

### 详细架构差异分析

#### 1. 层归一化位置（Layer Normalization Position）

**GPT-1：Post-LayerNorm 结构**
- 遵循原始 Transformer 设计
- 在子层（注意力/前馈网络）之后应用层归一化
- 可能存在梯度消失问题

```java
// GPT-1 伪代码结构
residual = input
output = layerNorm(residual + attention(input))
residual = output
output = layerNorm(residual + feedForward(output))
```

**GPT-2：Pre-LayerNorm 结构**
- 在子层之前应用层归一化
- 改善训练稳定性和收敛速度
- 更好的梯度流动

```java
// GPT-2 伪代码结构
output = input + attention(layerNorm(input))
output = output + feedForward(layerNorm(output))
```

**GPT-3：并行 Pre-LayerNorm**
- 保持 Pre-LayerNorm 结构
- 引入并行注意力和前馈网络计算
- 提高计算效率

```java
// GPT-3 并行计算伪代码
ln1_output = layerNorm1(input)
ln2_output = layerNorm2(input)
attn_output = attention(ln1_output)
mlp_output = feedForward(ln2_output)
output = input + attn_output + mlp_output
```

#### 2. 配置参数对比

| 配置项 | GPT-1 默认 | GPT-2 默认 | GPT-3 默认 |
|--------|------------|------------|------------|
| 词汇表大小 | 40,000 | 50,257 | 50,257 |
| 最大序列长度 | 512 | 1024 | 2048 |
| 隐藏层维度 | 768 | 768 | 12,288 |
| Transformer 层数 | 12 | 12 | 96 |
| 注意力头数 | 12 | 12 | 96 |
| 前馈网络维度 | 3072 | 3072 | 49,152 |
| 激活函数 | GELU | GELU | GELU_NEW |

#### 3. 核心技术特性

**GPT-1 特性：**
- 基础的 Transformer 解码器架构
- 无监督预训练 + 有监督微调范式
- 因果掩码的自注意力机制
- 标准的残差连接和 Dropout

**GPT-2 特性：**
- Pre-LayerNorm 改进架构
- 更大的模型规模（最大 1.5B 参数）
- 零样本任务迁移能力
- 改进的位置编码机制

**GPT-3 特性：**
- 并行注意力和前馈网络计算
- 稀疏注意力机制支持（可选）
- 梯度检查点优化
- Few-shot 上下文学习能力
- 旋转位置编码（RoPE）支持

#### 4. 实现架构差异

**组件命名规范：**
- GPT-1：`GPT1Block`、`GPT1Model`、`GPT1Config`
- GPT-2：`GPT2Block`、`GPT2Model`、`GPT2Config`  
- GPT-3：`GPT3MainBlock`、`GPT3Model`、`GPT3Config`

**核心块设计：**

```java
// GPT-1 架构
public class GPT1Block extends Block {
    private GPT1TokenEmbedding tokenEmbedding;
    private List<GPT1TransformerBlock> transformerBlocks;
    private LayerNorm finalLayerNorm;
    private GPT1OutputHead outputHead;
}

// GPT-2 架构  
public class GPT2Block extends Block {
    private GPT2TokenEmbedding tokenEmbedding;
    private List<GPT2TransformerBlock> transformerBlocks;
    private LayerNorm finalLayerNorm;
    private GPT2OutputHead outputHead;
}

// GPT-3 架构
public class GPT3MainBlock extends Block {
    private GPT2TokenEmbedding tokenEmbedding;    // 复用 GPT-2 组件
    private List<GPT3TransformerBlock> transformerBlocks;
    private LayerNorm finalLayerNorm;
    private GPT2OutputHead outputHead;           // 复用 GPT-2 组件
}
```

## 快速开始

### 1. 依赖配置

在 `pom.xml` 中添加依赖：

```xml
<dependency>
    <groupId>io.leavesfly.tinyai</groupId>
    <artifactId>tinyai-model-gpt</artifactId>
    <version>1.0-SNAPSHOT</version>
</dependency>
```

### 2. 创建和使用模型

#### GPT-1 模型示例

```java
import io.leavesfly.tinyai.gpt1.*;

// 创建小型 GPT-1 模型
GPT1Model model = GPT1Model.createTinyModel("gpt1-tiny");

// 或使用自定义配置
GPT1Config config = new GPT1Config(5000, 256, 512, 8, 8);
GPT1Model customModel = new GPT1Model("gpt1-custom", config);

// 文本生成
List<Integer> prompt = Arrays.asList(100, 200, 300);
List<Integer> generated = model.generateText(prompt, 50);

// 打印模型信息
model.printModelInfo();
```

#### GPT-2 模型示例

```java
import io.leavesfly.tinyai.gpt2.*;

// 创建不同规模的 GPT-2 模型
GPT2Model smallModel = GPT2Model.createSmallModel("gpt2-small");
GPT2Model mediumModel = GPT2Model.createMediumModel("gpt2-medium");

// 预测下一个 token
NdArray tokenIds = NdArray.of(new int[][]{{100, 200, 300}});
int nextToken = smallModel.predictNextToken(tokenIds);

// 生成序列
NdArray sequence = smallModel.generateSequence(tokenIds, 100);
```

#### GPT-3 模型示例

```java
import io.leavesfly.tinyai.gpt3.*;

// 创建 GPT-3 模型
GPT3Model model = GPT3Model.createLargeModel("gpt3-large");

// Few-shot 学习生成
NdArray contextTokens = NdArray.of(new int[][]{{/* 上下文示例 */}});
NdArray result = model.fewShotGenerate(contextTokens, 50);

// 配置并行计算
GPT3Config config = GPT3Config.createXLConfig();
config.setParallelAttention(true);
config.setSparseAttention(true);
GPT3Model xlModel = new GPT3Model("gpt3-xl", config);
```

### 3. 模型配置和训练

```java
// 自定义配置示例
GPT2Config config = new GPT2Config();
config.setVocabSize(30000);
config.setNPositions(512);
config.setNEmbd(1024);
config.setNLayer(24);
config.setNHead(16);

// 创建模型
GPT2Model model = new GPT2Model("my-gpt2", config);

// 模型信息
System.out.println("模型参数量：" + model.getGPT2Block().getParameterCount());
System.out.println("配置摘要：" + model.getConfigSummary());
```

## 性能和规模

### 参数数量估算

| 模型配置 | 层数 | 隐藏维度 | 注意力头数 | 估算参数量 |
|----------|------|----------|------------|------------|
| GPT-1 Tiny | 6 | 256 | 8 | ~2.3M |
| GPT-1 Base | 12 | 768 | 12 | ~117M |
| GPT-2 Small | 6 | 256 | 8 | ~2.3M |
| GPT-2 Medium | 24 | 1024 | 16 | ~350M |
| GPT-2 Large | 36 | 1280 | 20 | ~774M |
| GPT-3 Small | 12 | 768 | 12 | ~125M |
| GPT-3 Medium | 24 | 1024 | 16 | ~350M |
| GPT-3 Large | 24 | 2048 | 32 | ~1.3B |
| GPT-3 XL | 96 | 12288 | 96 | ~175B |

### 计算优化

- **GPT-1**：传统串行计算，稳定可靠
- **GPT-2**：Pre-LayerNorm 优化，训练更稳定
- **GPT-3**：并行计算优化，支持梯度检查点，内存效率更高

## 扩展和定制

### 1. 自定义 Transformer 块

```java
// 继承基础类实现自定义块
public class CustomGPTBlock extends GPT2TransformerBlock {
    public CustomGPTBlock(String name, GPT2Config config, int layerIndex) {
        super(name, config, layerIndex);
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        // 自定义前向传播逻辑
        return super.layerForward(inputs);
    }
}
```

### 2. 添加新的配置选项

```java
// 扩展配置类
public class ExtendedGPT3Config extends GPT3Config {
    private boolean useFlashAttention = false;
    private double attentionScale = 1.0;
    
    // getter/setter 方法
    public boolean isUseFlashAttention() { return useFlashAttention; }
    public void setUseFlashAttention(boolean useFlashAttention) { 
        this.useFlashAttention = useFlashAttention; 
    }
}
```

## 测试和验证

### 单元测试

```java
// 运行所有测试
mvn test

// 运行特定测试类
mvn test -Dtest=GPT2ModelTest
mvn test -Dtest=GPT3ConfigTest
```

### 模型验证

```java
// 验证模型输入输出
GPT2Model model = GPT2Model.createSmallModel("test");
NdArray testInput = NdArray.of(new int[][]{{1, 2, 3, 4, 5}});

try {
    model.validateInput(testInput);
    Variable output = model.predict(testInput);
    System.out.println("输出形状：" + output.getValue().getShape());
} catch (IllegalArgumentException e) {
    System.err.println("输入验证失败：" + e.getMessage());
}
```

## 文档和资源

- **API 文档**：详细的 JavaDoc 文档
- **架构图表**：`doc/` 目录下的 Mermaid 图表
- **Python 参考**：`doc/` 目录下的对应 Python 实现
- **实现报告**：`doc/GPT1_REPORT.md` 等详细实现说明

## 贡献指南

1. **代码规范**：遵循项目的 Java 编码规范
2. **注释要求**：使用中文注释，符合 JavaDoc 标准
3. **测试覆盖**：新功能需要完整的单元测试
4. **文档更新**：重要修改需要更新相应文档

## 许可证

本项目遵循 TinyAI 框架的开源许可证。

---

**作者**: 山泽  
**版本**: 1.0  
**最后更新**: 2025年10月