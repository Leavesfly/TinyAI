# GPT-3 模型实现

基于TinyAI框架实现的GPT-3语言模型，采用解码器-only Transformer架构，引入了并行注意力计算、旋转位置编码(RoPE)、稀疏注意力等先进技术，支持超大规模参数配置和Few-shot学习能力。

## 📁 文件结构

```
tinyai-model-gpt/src/main/java/io/leavesfly/tinyai/gpt3/
├── GPT3Config.java              # GPT-3配置类（支持175B参数）
├── GPT3Model.java               # GPT-3模型类（继承Model）
├── GPT3MainBlock.java           # GPT-3主体块（继承Block）
├── GPT3TransformerBlock.java    # GPT-3 Transformer解码器块
├── GPT3RotaryEmbedding.java     # 旋转位置编码(RoPE)实现
├── GPT3Demo.java                # 完整演示程序
└── test/                        # 测试套件
```

## 🎯 核心特性

### 1. 多规模超大模型支持
- **小型模型**: 768维, 12层, 12头 (125M参数，学习测试)
- **中型模型**: 1024维, 24层, 16头 (350M参数，实用应用)
- **大型模型**: 2048维, 24层, 32头 (1.3B参数，高质量生成)
- **超大型模型**: 12288维, 96层, 96头 (175B参数，顶级性能)

### 2. 先进架构设计
- **并行注意力与MLP**: 同时计算注意力和前馈网络，提升计算效率
- **旋转位置编码(RoPE)**: 相对位置编码，支持任意长度序列
- **稀疏注意力机制**: 大型模型采用稀疏注意力节省计算和内存
- **梯度检查点**: 训练时节省内存的梯度累积策略
- **Pre-LayerNorm结构**: 稳定的深层网络训练

### 3. Few-shot学习能力
- **零样本学习**: 无需微调即可执行新任务
- **上下文学习**: 基于示例快速理解任务模式  
- **任务泛化**: 支持分类、生成、翻译等多种任务
- **强大推理**: 展现类人的逻辑推理能力

## 🏗️ 网络架构图

### GPT-3整体架构
```mermaid
graph TB
    Input["Token IDs<br/>(batch_size, seq_len)"] --> TokenEmbed["Token嵌入层<br/>GPT2TokenEmbedding<br/>复用GPT-2实现"]
    TokenEmbed --> TransBlock1["GPT-3 Transformer块 1<br/>并行注意力+MLP"]
    TransBlock1 --> TransBlock2["GPT-3 Transformer块 2<br/>旋转位置编码"]
    TransBlock2 --> TransBlockN["...<br/>GPT-3 Transformer块 N<br/>(最多96层)"]
    TransBlockN --> FinalLN["最终层归一化<br/>LayerNorm"]
    FinalLN --> OutputHead["输出头<br/>GPT2OutputHead<br/>复用GPT-2实现"]
    OutputHead --> Output["Logits<br/>(batch_size, seq_len, vocab_size)"]
    
    RoPE["旋转位置编码<br/>GPT3RotaryEmbedding"] --> TransBlock1
    RoPE --> TransBlock2
    RoPE --> TransBlockN
    
    style TokenEmbed fill:#e1f5fe
    style TransBlock1 fill:#f3e5f5
    style TransBlock2 fill:#f3e5f5
    style TransBlockN fill:#f3e5f5
    style FinalLN fill:#fff3e0
    style OutputHead fill:#e8f5e8
    style RoPE fill:#ffe0e0
```

### GPT3TransformerBlock并行架构
```mermaid
graph TD
    BlockInput["输入<br/>(batch_size, seq_len, n_embd)"] --> Split{"并行分支"}
    
    Split --> LN1["LayerNorm 1<br/>注意力分支"]
    Split --> LN2["LayerNorm 2<br/>MLP分支"]
    
    LN1 --> RoPE["旋转位置编码<br/>GPT3RotaryEmbedding"]
    RoPE --> MHA["多头自注意力<br/>带因果掩码"]
    MHA --> AttnOut["注意力输出"]
    
    LN2 --> FFN["前馈网络<br/>Linear→GELU→Linear"]
    FFN --> MLPOut["MLP输出"]
    
    AttnOut --> Combine["合并输出<br/>input + attn + mlp"]
    MLPOut --> Combine
    BlockInput --> Combine
    
    Combine --> BlockOutput["输出<br/>(batch_size, seq_len, n_embd)"]
    
    style Split fill:#fff3e0
    style MHA fill:#e1f5fe
    style FFN fill:#f3e5f5
    style RoPE fill:#ffe0e0
    style Combine fill:#e8f5e8
```

### 旋转位置编码(RoPE)机制
```mermaid
graph TD
    SeqLen["序列长度"] --> GenPos["生成位置索引<br/>[0, 1, 2, ..., L-1]"]
    GenPos --> CalcFreq["计算频率<br/>pos * inv_freq"]
    
    CalcFreq --> CosSin["计算cos/sin值<br/>cos(θ), sin(θ)"]
    
    QueryKey["Query/Key向量<br/>(B, L, H, D)"] --> Split["分离前后半部分<br/>x1, x2"]
    
    CosSin --> Rotate["旋转变换<br/>x1*cos - x2*sin<br/>x1*sin + x2*cos"]
    Split --> Rotate
    
    Rotate --> RotatedQK["旋转后的Q/K<br/>(B, L, H, D)"]
    
    style GenPos fill:#e1f5fe
    style CalcFreq fill:#f3e5f5
    style CosSin fill:#fff3e0
    style Rotate fill:#ffe0e0
```

### GPT-3与前代模型架构对比
```mermaid
graph LR
    subgraph GPT1["GPT-1 (Post-LN)"]
        G1Input["输入"] --> G1Attn["注意力"]
        G1Attn --> G1Add1["残差连接"]
        G1Add1 --> G1LN1["LayerNorm"]
        G1LN1 --> G1FFN["前馈网络"]
        G1FFN --> G1Add2["残差连接"]
        G1Add2 --> G1LN2["LayerNorm"]
    end
    
    subgraph GPT2["GPT-2 (Pre-LN)"]
        G2Input["输入"] --> G2LN1["LayerNorm"]
        G2LN1 --> G2Attn["注意力"]
        G2Attn --> G2Add1["残差连接"]
        G2Add1 --> G2LN2["LayerNorm"]
        G2LN2 --> G2FFN["前馈网络"]
        G2FFN --> G2Add2["残差连接"]
    end
    
    subgraph GPT3["GPT-3 (并行计算)"]
        G3Input["输入"] --> G3Split{"并行分支"}
        G3Split --> G3LN1["LayerNorm 1"]
        G3Split --> G3LN2["LayerNorm 2"]
        G3LN1 --> G3Attn["注意力+RoPE"]
        G3LN2 --> G3FFN["前馈网络"]
        G3Attn --> G3Combine["合并"]
        G3FFN --> G3Combine
        G3Input --> G3Combine
    end
    
    style G1LN1 fill:#ffe0e0
    style G1LN2 fill:#ffe0e0
    style G2LN1 fill:#e1f5fe
    style G2LN2 fill:#e1f5fe
    style G3Split fill:#fff3e0
    style G3Combine fill:#e8f5e8
```

### 稀疏注意力模式(大型模型)
```mermaid
graph TD
    SeqInput["输入序列<br/>(长序列)"] --> SparsePattern["稀疏注意力模式"]
    
    SparsePattern --> Local["局部注意力<br/>Local Attention"]
    SparsePattern --> Strided["步长注意力<br/>Strided Attention"]
    SparsePattern --> Global["全局注意力<br/>Global Attention"]
    
    Local --> Combine["注意力组合"]
    Strided --> Combine
    Global --> Combine
    
    Combine --> SparseOut["稀疏注意力输出<br/>降低计算复杂度"]
    
    style Local fill:#e1f5fe
    style Strided fill:#f3e5f5
    style Global fill:#fff3e0
    style Combine fill:#e8f5e8
```

### 类图关系
```mermaid
classDiagram
    class GPT3Model {
        -GPT3Config config
        -GPT3MainBlock gpt3Block
        +GPT3Model(String, GPT3Config)
        +createSmallModel(String) GPT3Model
        +createXLModel(String) GPT3Model
        +fewShotGenerate(NdArray, int) NdArray
        +generateSequence(NdArray, int) NdArray
    }
    
    class GPT3MainBlock {
        -GPT3Config config
        -GPT2TokenEmbedding tokenEmbedding
        -List~GPT3TransformerBlock~ transformerBlocks
        -LayerNorm finalLayerNorm
        -GPT2OutputHead outputHead
        +layerForward(Variable...) Variable
        +generateWithContext(NdArray, int) NdArray
    }
    
    class GPT3TransformerBlock {
        -LayerNorm layerNorm1
        -LayerNorm layerNorm2
        -MultiHeadAttention attention
        -FeedForward feedForward
        +forwardParallel(Variable) Variable
        +forwardSequential(Variable) Variable
    }
    
    class GPT3RotaryEmbedding {
        -int rotaryDim
        -double base
        -NdArray invFreq
        +generateRotaryEmbedding(int) NdArray[]
        +applyRotaryPositionEmbedding(Variable, Variable, int) Variable[]
    }
    
    class GPT3Config {
        -boolean sparseAttention
        -boolean parallelAttention
        -double rotaryPct
        +createXLConfig() GPT3Config
        +estimateParameterCount() long
    }
    
    GPT3Model --> GPT3MainBlock : "包含"
    GPT3MainBlock --> GPT3TransformerBlock : "包含多个"
    GPT3TransformerBlock --> GPT3RotaryEmbedding : "使用"
    GPT3Model --> GPT3Config : "配置"
    GPT3TransformerBlock --> MultiHeadAttention : "使用"
    GPT3TransformerBlock --> FeedForward : "使用"
```

## 🚀 快速开始

### 基本使用

```java
// 创建不同规模的GPT-3模型
GPT3Model smallModel = GPT3Model.createSmallModel("gpt3-small");      // 125M参数
GPT3Model mediumModel = GPT3Model.createMediumModel("gpt3-medium");   // 350M参数
GPT3Model largeModel = GPT3Model.createLargeModel("gpt3-large");      // 1.3B参数
GPT3Model xlModel = GPT3Model.createXLModel("gpt3-xl");               // 175B参数

// 标准前向传播
NdArray tokenIds = NdArray.of(Shape.of(1, 20)); // 输入序列
Variable output = model.forward(new Variable(tokenIds));

// 文本生成
NdArray generated = model.generateSequence(tokenIds, 50);

// Few-shot学习生成
NdArray context = createFewShotContext(); // 创建包含示例的上下文
NdArray fewShotResult = model.fewShotGenerate(context, 30);
```

### Few-shot学习示例

```java
// 情感分析Few-shot示例
public class GPT3FewShotExample {
    public static void demonstrateSentimentAnalysis() {
        GPT3Model model = GPT3Model.createMediumModel("gpt3-sentiment");
        
        // 构建Few-shot上下文："句子 -> 情感标签"的示例
        String[] examples = {
            "这部电影真的很棒！ -> 正面",
            "我觉得这个产品很糟糕。 -> 负面", 
            "今天天气不错。 -> 中性"
        };
        
        // 编码上下文为token序列
        NdArray context = encodeExamples(examples);
        
        // 添加新的待分析句子
        NdArray newSentence = encodeText("这家餐厅的服务很差。 ->");
        NdArray fullContext = concatenate(context, newSentence);
        
        // 生成分类结果
        NdArray result = model.fewShotGenerate(fullContext, 5);
        
        System.out.println("Few-shot分类结果: " + decodeTokens(result));
    }
}
```

### 高级配置

```java
// 创建超大型GPT-3配置(175B参数)
GPT3Config xlConfig = new GPT3Config();
xlConfig.setNEmbd(12288);           // 嵌入维度
xlConfig.setNLayer(96);             // 96层Transformer
xlConfig.setNHead(96);              // 96个注意力头
xlConfig.setNInner(49152);          // 前馈网络维度
xlConfig.setSparseAttention(true);   // 启用稀疏注意力
xlConfig.setParallelAttention(true); // 启用并行计算
xlConfig.setGradientCheckpointing(true); // 启用梯度检查点

// 旋转位置编码配置
xlConfig.setRotaryPct(0.25);        // 25%维度使用RoPE
xlConfig.setUseCache(true);         // 启用KV缓存

// 验证并创建模型
xlConfig.validate();
GPT3Model xlModel = new GPT3Model("gpt3-175b", xlConfig);

// 打印模型信息
xlModel.printModelInfo();
System.out.println("估算参数数量: " + xlConfig.estimateParameterCount());
```

### 旋转位置编码使用

```java
// 创建并使用旋转位置编码
GPT3RotaryEmbedding rope = new GPT3RotaryEmbedding("rope", 64, 2048);

// 生成位置编码
NdArray[] cosAndSin = rope.generateRotaryEmbedding(128);
NdArray cos = cosAndSin[0];  // cos值
NdArray sin = cosAndSin[1];  // sin值

// 对Query和Key应用旋转编码
Variable query = new Variable(queryTensor);  // (B, L, H, D)
Variable key = new Variable(keyTensor);      // (B, L, H, D)

Variable[] rotated = rope.applyRotaryPositionEmbedding(query, key, 128);
Variable rotatedQuery = rotated[0];
Variable rotatedKey = rotated[1];
```

## 🔍 核心优势

### 1. 并行计算优化
- **同时计算**: 注意力和MLP并行执行，显著提升训练和推理速度
- **内存效率**: 梯度检查点技术减少大型模型的内存占用
- **硬件友好**: 充分利用现代GPU的并行计算能力

### 2. 位置编码创新
- **相对位置**: RoPE提供更好的位置理解能力
- **长序列支持**: 支持任意长度序列而不损失性能
- **旋转不变**: 保持向量模长不变的优雅数学性质

### 3. 稀疏注意力
- **计算复杂度**: 从O(n²)降低到O(n√n)
- **内存占用**: 大幅减少长序列的内存需求
- **性能保持**: 在减少计算的同时保持模型性能

### 4. Few-shot学习
- **快速适应**: 无需微调即可执行新任务
- **上下文理解**: 从少量示例中学习任务模式
- **任务泛化**: 支持分类、生成、推理等多种任务

## 📊 性能特点

### 模型规模对比
| 模型规模 | 参数量 | 层数 | 维度 | 头数 | 特殊特性 |
|---------|-------|------|------|------|----------|
| 小型    | 125M  | 12   | 768  | 12   | 基础学习 |
| 中型    | 350M  | 24   | 1024 | 16   | 实用应用 |
| 大型    | 1.3B  | 24   | 2048 | 32   | 稀疏注意力 |
| 超大型  | 175B  | 96   | 12288| 96   | 全部优化特性 |

### Few-shot学习能力
- **零样本**: 无示例直接执行任务
- **单样本**: 一个示例快速理解
- **少样本**: 2-10个示例达到良好性能
- **多样本**: 更多示例进一步提升效果

## 🧪 完整演示

运行`GPT3Demo.java`查看完整功能演示：

```java
public class GPT3Demo {
    public static void main(String[] args) {
        // 1. 模型创建演示
        demonstrateModelCreation();
        
        // 2. 架构分析
        demonstrateArchitectureAnalysis();
        
        // 3. 前向传播演示
        demonstrateForwardPass();
        
        // 4. 文本生成演示
        demonstrateTextGeneration();
        
        // 5. Few-shot学习演示
        demonstrateFewShotLearning();
        
        // 6. 旋转位置编码演示
        demonstrateRotaryEmbedding();
    }
}
```

## 🔧 扩展开发

### 自定义注意力机制
```java
// 扩展稀疏注意力模式
public class CustomSparseAttention extends MultiHeadAttention {
    @Override
    protected NdArray computeAttentionMask(int seqLen) {
        // 实现自定义的稀疏注意力模式
        return createCustomSparseMask(seqLen);
    }
}
```

### 自定义位置编码
```java
// 扩展位置编码机制
public class CustomPositionEmbedding extends GPT3RotaryEmbedding {
    @Override
    public Variable[] applyRotaryPositionEmbedding(Variable query, Variable key, int seqLen) {
        // 实现自定义的位置编码逻辑
        return customRotaryTransform(query, key, seqLen);
    }
}
```

## 📚 技术参考

- **论文**: "Language Models are Few-Shot Learners" (GPT-3)
- **架构**: Transformer解码器-only架构
- **位置编码**: Rotary Position Embedding (RoPE)
- **优化技术**: 并行注意力、稀疏注意力、梯度检查点
- **学习范式**: Few-shot学习、上下文学习

---

**注意**: GPT-3是大规模语言模型，完整的175B参数模型需要大量计算资源。本实现提供了完整的架构和多种规模配置，可根据实际资源情况选择合适的模型规模进行实验和应用。