# TinyAI MiniMind 模块设计文档

## 一、模块概述

### 1.1 模块定位

`tinyai-model-minimind` 是 TinyAI 项目中对轻量级语言模型 MiniMind 的 Java 实现模块,旨在完整还原开源项目 [MiniMind](https://github.com/jingyaogong/minimind) 的核心功能。该模块基于 TinyAI 深度学习框架,特别是 `tinyai-deeplearning-nnet` 的 V2 架构,实现一个仅 26M 参数的超小型 GPT 风格语言模型,涵盖预训练、后训练、推理和应用等全生命周期。

### 1.2 目标价值

| 价值维度 | 描述 |
|---------|------|
| **教育价值** | 提供完整的 LLM 训练流程示例,帮助学习者理解从零训练语言模型的全过程 |
| **研究价值** | 展示在资源受限环境下训练小参数量模型的最佳实践 |
| **工程价值** | 验证 TinyAI V2 架构在实际大语言模型场景的可行性和完整性 |
| **应用价值** | 提供可在普通 GPU 上快速训练和部署的轻量级语言模型方案 |

### 1.3 核心特性

- **极致轻量化**: 模型参数量仅 26M,是 GPT-3 的 1/7000
- **快速训练**: 单卡 GPU 2 小时内完成预训练
- **全流程覆盖**: 包含 Tokenizer、预训练、SFT、LoRA、DPO、RLAIF 等完整训练流程
- **多模态扩展**: 支持 MoE 架构和视觉多模态能力
- **纯 Java 实现**: 基于 TinyAI 框架,无第三方深度学习库依赖

## 二、技术架构设计

### 2.1 模块依赖关系

```mermaid
graph TB
    MiniMind[tinyai-model-minimind]
    ML[tinyai-deeplearning-ml]
    NNet[tinyai-deeplearning-nnet]
    Func[tinyai-deeplearning-func]
    NdArr[tinyai-deeplearning-ndarr]
    RL[tinyai-deeplearning-rl]
    
    MiniMind --> ML
    MiniMind -.可选依赖.-> RL
    ML --> NNet
    ML --> Func
    NNet --> Func
    Func --> NdArr
    
    style MiniMind fill:#e1f5ff
    style NNet fill:#ffe1e1
    style ML fill:#e1ffe1
```

**依赖说明**:

| 依赖模块 | 依赖类型 | 用途 |
|---------|---------|------|
| `tinyai-deeplearning-ml` | 必需 | 模型训练框架(Model、Trainer)、优化器、损失函数 |
| `tinyai-deeplearning-nnet` | 必需 | **V2 神经网络层和模块(强制使用 V2)** |
| `tinyai-deeplearning-rl` | 可选 | RLAIF 训练(PPO/GRPO/SPO 算法) |

**重要说明**:
- **移除 `tinyai-deeplearning-nl` 依赖**: 该模块是嵌入学习模块,非 NLP 工具库,MiniMind 自行实现 Tokenizer
- **强制使用 V2 API**: 所有神经网络组件必须来自 `tinyai-deeplearning-nnet` 的 V2 包,避免使用 V1 组件

### 2.2 整体架构设计

```mermaid
graph TB
    subgraph 应用层
        WebUI[Web 界面]
        CLI[命令行工具]
        API[API 服务]
    end
    
    subgraph 推理层
        Generator[文本生成器]
        Sampler[采样策略]
        Cache[KV-Cache]
    end
    
    subgraph 模型层
        MiniMindModel[MiniMind 模型]
        MiniMindBlock[MiniMind Block]
        TransformerLayer[Transformer 层]
        Attention[多头注意力]
    end
    
    subgraph 训练层
        Pretrain[预训练器]
        SFT[监督微调]
        LoRA[LoRA 微调]
        DPO[DPO 训练]
        RLAIF[RLAIF 训练]
    end
    
    subgraph 数据层
        Tokenizer[分词器]
        Dataset[数据集加载]
        Processor[数据预处理]
    end
    
    subgraph 基础设施层
        V2Module[V2 Module API]
        Optimizer[优化器]
        Trainer[训练框架]
    end
    
    WebUI --> Generator
    CLI --> Generator
    API --> Generator
    
    Generator --> MiniMindModel
    Sampler --> MiniMindModel
    Cache --> TransformerLayer
    
    MiniMindModel --> MiniMindBlock
    MiniMindBlock --> TransformerLayer
    TransformerLayer --> Attention
    
    Pretrain --> MiniMindModel
    SFT --> MiniMindModel
    LoRA --> MiniMindModel
    DPO --> MiniMindModel
    RLAIF --> MiniMindModel
    
    Pretrain --> Dataset
    SFT --> Dataset
    
    Dataset --> Tokenizer
    Processor --> Tokenizer
    
    MiniMindModel --> V2Module
    Pretrain --> Trainer
    SFT --> Trainer
    Trainer --> Optimizer
```

### 2.3 核心模块结构

#### 2.3.1 包结构设计

```
tinyai-model-minimind/
├── src/main/java/io/leavesfly/tinyai/minimind/
│   ├── model/                      # 模型定义
│   │   ├── MiniMindModel.java         # 模型主类(继承 ml.Model)
│   │   ├── MiniMindBlock.java         # 模型块(继承 nnet.v2.Module)
│   │   ├── MiniMindConfig.java        # 模型配置
│   │   ├── MiniMindTransformerLayer.java  # Transformer 层(继承 Module)
│   │   ├── attention/                 # 注意力机制(自行实现)
│   │   │   ├── MultiHeadAttention.java    # 多头注意力(继承 Module)
│   │   │   ├── GroupedQueryAttention.java # GQA(继承 Module)
│   │   │   └── KVCache.java               # KV-Cache 管理
│   │   ├── embedding/                 # 嵌入层(自行实现)
│   │   │   ├── TokenEmbedding.java        # Token 嵌入(继承 Module)
│   │   │   └── RotaryPositionEmbedding.java  # RoPE(继承 Module)
│   │   └── moe/                       # MoE 扩展(可选,自行实现)
│   │       ├── MoELayer.java              # MoE 层(继承 Module)
│   │       └── ExpertRouter.java          # 专家路由(继承 Module)
│   ├── tokenizer/                  # 分词器
│   │   ├── MiniMindTokenizer.java     # 自定义分词器
│   │   ├── Vocabulary.java            # 词汇表管理
│   │   └── BPETokenizer.java          # BPE 分词实现
│   ├── training/                   # 训练组件
│   │   ├── pretrain/                  # 预训练
│   │   │   ├── PretrainConfig.java
│   │   │   ├── PretrainTrainer.java
│   │   │   └── PretrainDataset.java
│   │   ├── sft/                       # 监督微调
│   │   │   ├── SFTConfig.java
│   │   │   ├── SFTTrainer.java
│   │   │   └── SFTDataset.java
│   │   ├── lora/                      # LoRA 微调
│   │   │   ├── LoRAConfig.java
│   │   │   ├── LoRAAdapter.java
│   │   │   └── LoRATrainer.java
│   │   ├── dpo/                       # DPO 训练
│   │   │   ├── DPOConfig.java
│   │   │   ├── DPOTrainer.java
│   │   │   └── DPODataset.java
│   │   └── rlaif/                     # RLAIF 训练(可选)
│   │       ├── PPOTrainer.java
│   │       ├── GRPOTrainer.java
│   │       └── SPOTrainer.java
│   ├── inference/                  # 推理组件
│   │   ├── TextGenerator.java         # 文本生成器
│   │   ├── GenerationConfig.java      # 生成配置
│   │   ├── sampler/                   # 采样策略
│   │   │   ├── GreedySampler.java
│   │   │   ├── TopKSampler.java
│   │   │   ├── TopPSampler.java
│   │   │   └── TemperatureSampler.java
│   │   └── cache/                     # KV-Cache
│   │       ├── KVCacheManager.java
│   │       └── IncrementalCache.java
│   ├── data/                       # 数据处理
│   │   ├── DataProcessor.java         # 数据处理器
│   │   ├── TextCleaner.java           # 文本清洗
│   │   ├── DataCollator.java          # 数据整理
│   │   └── loader/                    # 数据加载
│   │       ├── PretrainDataLoader.java
│   │       ├── SFTDataLoader.java
│   │       └── DPODataLoader.java
│   ├── util/                       # 工具类
│   │   ├── ModelRegistry.java         # 模型注册
│   │   ├── CheckpointManager.java     # 检查点管理
│   │   ├── MetricsTracker.java        # 指标跟踪
│   │   └── ConfigLoader.java          # 配置加载
│   └── application/                # 应用接口
│       ├── ChatInterface.java         # 对话接口
│       ├── APIServer.java             # API 服务
│       └── WebUI.java                 # Web 界面
├── src/main/resources/
│   ├── configs/                    # 配置文件
│   │   ├── minimind-small.json       # 小型模型配置(26M)
│   │   ├── minimind-medium.json      # 中型模型配置(108M)
│   │   └── minimind-moe.json         # MoE 模型配置(145M)
│   └── prompts/                    # 提示词模板
│       └── chat-template.txt
├── src/test/java/                  # 测试代码
│   ├── model/                         # 模型测试
│   ├── training/                      # 训练测试
│   └── inference/                     # 推理测试
└── doc/                            # 文档
    ├── 技术架构文档.md
    ├── MiniMind模型设计.md
    ├── 预训练指南.md
    ├── 微调指南.md
    └── API参考.md
```

## 三、核心组件设计

### 3.1 模型架构设计

#### 3.1.1 MiniMind 模型配置

```mermaid
classDiagram
    class MiniMindConfig {
        +int vocabSize
        +int maxSeqLen
        +int hiddenSize
        +int numLayers
        +int numHeads
        +int ffnHiddenSize
        +float dropout
        +String activationFunction
        +boolean useRoPE
        +boolean preLayerNorm
        +float epsilon
        
        +MiniMindConfig createSmallConfig()
        +MiniMindConfig createMediumConfig()
        +MiniMindConfig createMoEConfig()
        +int getHeadDim()
        +validate()
    }
    
    class ModelPreset {
        <<enumeration>>
        SMALL_26M
        MEDIUM_108M
        MOE_145M
    }
    
    MiniMindConfig --> ModelPreset
```

**配置参数表**:

| 配置项 | Small (26M) | Medium (108M) | MoE (145M) |
|-------|-------------|---------------|------------|
| 词汇表大小 | 6,400 | 6,400 | 6,400 |
| 最大序列长度 | 512 | 512 | 512 |
| 隐藏层维度 | 512 | 768 | 512 |
| Transformer 层数 | 8 | 16 | 8 |
| 注意力头数 | 16 | 16 | 16 |
| FFN 隐藏维度 | 1,024 | 2,048 | 1,024 |
| Dropout 比例 | 0.1 | 0.1 | 0.1 |
| 激活函数 | SiLU | SiLU | SiLU |
| 位置编码 | RoPE | RoPE | RoPE |
| 层归一化 | Pre-LN | Pre-LN | Pre-LN |
| 特殊机制 | - | - | 4 专家 MoE |

#### 3.1.2 模型继承体系

```mermaid
classDiagram
    class Model {
        <<TinyAI ML>>
        #Module module
        +forward()
        +getAllParams()
        +train()
        +eval()
    }
    
    class Module {
        <<TinyAI NNet V2>>
        #Map~String,Parameter~ parameters
        #Map~String,Module~ modules
        +forward()
        +namedParameters()
        +clearGrads()
    }
    
    class MiniMindModel {
        -MiniMindBlock miniMindBlock
        -MiniMindConfig config
        +MiniMindModel(name, config)
        +predict(tokenIds)
        +generate(prompt, maxLen)
        +getConfig()
        +printModelInfo()
    }
    
    class MiniMindBlock {
        -TokenEmbedding tokenEmbedding
        -List~MiniMindTransformerLayer~ layers
        -LayerNorm finalNorm
        -Linear lmHead
        +MiniMindBlock(config)
        +forward(tokenIds, kvCache)
    }
    
    class MiniMindTransformerLayer {
        -LayerNorm attentionNorm
        -MultiHeadAttention attention
        -LayerNorm ffnNorm
        -FeedForwardNetwork ffn
        +forward(x, kvCache)
    }
    
    Model <|-- MiniMindModel
    Module <|-- MiniMindBlock
    Module <|-- MiniMindTransformerLayer
    
    MiniMindModel --> MiniMindBlock
    MiniMindModel --> MiniMindConfig
    MiniMindBlock --> MiniMindTransformerLayer
```

**设计原则**:

1. **MiniMindModel** 继承 `io.leavesfly.tinyai.ml.Model`,作为对外统一接口
2. **MiniMindBlock** 继承 `core.io.leavesfly.tinyai.nnet.Module`,构建核心网络结构
3. **优先使用 V2 组件**,如 `Linear`、`LayerNorm`、`Dropout` 等来自 `tinyai-deeplearning-nnet/v2`
4. **避免使用 V1 组件**,除非 V2 中缺少必要功能,此时需要通过适配器桥接

#### 3.1.3 Transformer 层设计

**核心计算流程**:

```mermaid
graph TB
    Input[输入 x] --> Norm1[LayerNorm 1]
    Norm1 --> Attn[多头注意力]
    Attn --> Add1[残差连接 +]
    Input --> Add1
    Add1 --> Norm2[LayerNorm 2]
    Norm2 --> FFN[前馈网络]
    FFN --> Add2[残差连接 +]
    Add1 --> Add2
    Add2 --> Output[输出]
    
    KVCache[KV-Cache] -.增量推理.-> Attn
```

**层内组件映射**:

| 组件 | 实现来源 | 具体类 | 功能描述 |
|------|---------|--------|---------|
| Layer Normalization | **V2 必需** | `norm.layer.io.leavesfly.tinyai.nnet.LayerNorm` | Pre-LN 归一化 |
| Multi-Head Attention | **自行实现** | `io.leavesfly.tinyai.minimind.model.attention.MultiHeadAttention` | 支持 RoPE、KV-Cache、因果掩码 |
| Feed-Forward Network | **V2 必需** | `dnn.layer.io.leavesfly.tinyai.nnet.Linear` × 2 | SiLU 激活 + 两层线性变换 |
| SiLU Activation | **V2 必需** | `activation.layer.io.leavesfly.tinyai.nnet.SiLU` | Swish 激活函数 |
| Dropout | **V2 必需** | `io.leavesfly.tinyai.nnet.v2.layer.norm.Dropout` | 正则化 |
| Token Embedding | **自行实现** | `io.leavesfly.tinyai.minimind.model.embedding.TokenEmbedding` | 词汇表嵌入查找 |
| RoPE | **自行实现** | `io.leavesfly.tinyai.minimind.model.embedding.RotaryPositionEmbedding` | 旋转位置编码 |

### 3.2 注意力机制设计

#### 3.2.1 多头注意力实现

```mermaid
graph LR
    X[输入 X] --> WQ[WQ 投影]
    X --> WK[WK 投影]
    X --> WV[WV 投影]
    
    WQ --> RoPE_Q[RoPE Q]
    WK --> RoPE_K[RoPE K]
    
    RoPE_Q --> Split_Q[多头分割]
    RoPE_K --> Split_K[多头分割]
    WV --> Split_V[多头分割]
    
    Split_Q --> Attn[Scaled Dot-Product]
    Split_K --> Attn
    Split_V --> Attn
    
    Attn --> Concat[多头拼接]
    Concat --> WO[WO 投影]
    WO --> Out[输出]
    
    KVCache[(KV Cache)] -.缓存.-> Split_K
    KVCache -.缓存.-> Split_V
```

**关键特性**:

1. **RoPE 位置编码**: 旋转位置嵌入,支持长度外推
2. **因果掩码**: 自回归生成,禁止看到未来 token
3. **KV-Cache**: 增量推理加速,避免重复计算
4. **多头并行**: 将注意力分解为多个子空间

#### 3.2.2 RoPE 位置编码

**编码原理**:

- 通过旋转矩阵对 Q、K 向量进行位置编码
- 相对位置通过向量旋转角度差异体现
- 支持序列长度外推(如 YaRN 算法)

**实现要求**: 
- 继承 `core.io.leavesfly.tinyai.nnet.Module`
- 位置: `io.leavesfly.tinyai.minimind.model.embedding.RotaryPositionEmbedding`
- 参考原版 MiniMind 的 RoPE 实现,确保功能一致性

### 3.3 Tokenizer 设计

**实现策略**: 完全自行实现 BPE Tokenizer,不依赖 `tinyai-deeplearning-nl` 模块

**原因说明**: `tinyai-deeplearning-nl` 是嵌入学习模块而非 NLP 工具库,MiniMind 需自主实现分词功能以完整还原原版能力

#### 3.3.1 分词器架构

```mermaid
graph TB
    Raw[原始文本] --> Norm[文本归一化]
    Norm --> PreTok[预分词]
    PreTok --> BPE[BPE 合并]
    BPE --> PostProc[后处理]
    PostProc --> TokenIds[Token IDs]
    
    Vocab[(词汇表)] --> BPE
    SpecialToks[特殊 Token] --> PostProc
    
    TokenIds --> Decode[解码器]
    Decode --> Text[文本输出]
```

**核心功能**:

| 功能 | 描述 |
|------|------|
| **文本归一化** | Unicode 规范化、大小写处理 |
| **预分词** | 按空格、标点符号初步切分 |
| **BPE 编码** | 字节对编码,子词切分 |
| **特殊 Token** | `<|im_start|>`、`<|im_end|>`、`<pad>`、`<unk>` |
| **词汇表管理** | 词汇索引、逆向映射 |
| **批量编码** | 填充、截断、注意力掩码 |

#### 3.3.2 词汇表结构

| Token 类型 | Token 示例 | ID 范围 |
|-----------|-----------|---------|
| 特殊 Token | `<|im_start|>`, `<|im_end|>`, `<pad>`, `<unk>` | 0-3 |
| 常用字符 | 字母、数字、标点 | 4-255 |
| BPE 合并 | 高频子词组合 | 256-6399 |

**词汇表大小**: 6,400 (与原始 MiniMind 保持一致)

### 3.4 训练流程设计

#### 3.4.1 预训练阶段

```mermaid
graph TB
    subgraph 数据准备
        RawData[原始文本] --> Clean[数据清洗]
        Clean --> Tokenize[分词编码]
        Tokenize --> Chunk[序列切分]
    end
    
    subgraph 训练循环
        Chunk --> Batch[批次构建]
        Batch --> Forward[前向传播]
        Forward --> Loss[计算损失]
        Loss --> Backward[反向传播]
        Backward --> Update[参数更新]
        Update --> Batch
    end
    
    subgraph 监控与保存
        Loss --> Metrics[指标记录]
        Update --> Checkpoint[检查点保存]
        Metrics --> Monitor[训练监控]
    end
```

**预训练目标**:

- **损失函数**: 交叉熵损失(Cross-Entropy Loss)
- **优化器**: AdamW
- **学习率调度**: Cosine Annealing with Warmup
- **训练策略**: 
  - 序列长度 512
  - 批次大小 32-64
  - 梯度累积支持
  - 混合精度训练(可选)

**数据处理流程**:

1. **数据清洗**: 移除低质量文本、去重
2. **分词编码**: 使用 MiniMindTokenizer
3. **序列构建**: 固定长度切分,添加特殊 Token
4. **批次整理**: 填充、掩码构建

#### 3.4.2 监督微调(SFT)

```mermaid
graph TB
    subgraph 数据格式
        Instruction[指令] --> Format[格式化]
        Input[输入] --> Format
        Output[输出] --> Format
        Format --> Template[对话模板]
    end
    
    subgraph 训练过程
        Template --> Encode[编码]
        Encode --> Mask[标签掩码]
        Mask --> Train[微调训练]
        Train --> Evaluate[评估]
    end
    
    subgraph 优化策略
        Train --> GradClip[梯度裁剪]
        GradClip --> LRSchedule[学习率调整]
        LRSchedule --> EarlyStopping[早停]
    end
```

**SFT 数据格式**:

```
{
  "instruction": "用户指令或问题",
  "input": "可选的补充输入",
  "output": "期望的模型输出"
}
```

**训练特点**:

- **仅计算输出部分损失**: 输入部分不参与损失计算
- **对话模板**: 使用 `<|im_start|>` 和 `<|im_end|>` 标记
- **较小学习率**: 避免灾难性遗忘
- **轮次控制**: 通常 3-10 个 epoch

#### 3.4.3 LoRA 微调

```mermaid
graph TB
    subgraph LoRA 原理
        Original[原始权重 W] -.冻结.-> Output
        LowRank[低秩分解] --> A[矩阵 A]
        LowRank --> B[矩阵 B]
        A --> Mult[A × B]
        B --> Mult
        Mult --> Scale[缩放 α/r]
        Scale --> Add[+ W]
        Add --> Output[输出]
    end
    
    subgraph 应用位置
        QProj[Q 投影] --> LoRA1[LoRA]
        VProj[V 投影] --> LoRA2[LoRA]
        KProj[K 投影] -.可选.-> LoRA3[LoRA]
        OProj[O 投影] -.可选.-> LoRA4[LoRA]
    end
```

**LoRA 配置**:

| 参数 | 默认值 | 描述 |
|------|-------|------|
| Rank (r) | 8 | 低秩分解秩 |
| Alpha (α) | 16 | 缩放因子 |
| Dropout | 0.1 | LoRA Dropout |
| 目标模块 | Q, V 投影 | 应用 LoRA 的层 |

**优势**:

- 大幅减少可训练参数(减少 90%+)
- 保持原模型参数不变
- 支持多任务适配器切换
- 训练速度快、显存占用低

#### 3.4.4 DPO 训练

```mermaid
graph TB
    subgraph DPO 数据
        Prompt[提示词] --> Chosen[偏好回答]
        Prompt --> Rejected[拒绝回答]
    end
    
    subgraph DPO 损失计算
        Chosen --> LogProb_C[Log Prob 偏好]
        Rejected --> LogProb_R[Log Prob 拒绝]
        LogProb_C --> Diff[概率差]
        LogProb_R --> Diff
        Diff --> Sigmoid[Sigmoid]
        Sigmoid --> Loss[DPO Loss]
    end
    
    subgraph 参考模型
        RefModel[参考模型] -.KL 散度约束.-> Loss
    end
```

**DPO 原理**:

- **直接偏好优化**: 无需奖励模型,直接从偏好数据学习
- **损失函数**: 
  - 最大化偏好回答的概率
  - 最小化拒绝回答的概率
  - KL 散度约束防止偏离参考模型

**数据格式**:

```
{
  "prompt": "用户提示词",
  "chosen": "人类偏好的回答",
  "rejected": "质量较低的回答"
}
```

#### 3.4.5 RLAIF 训练(可选)

**支持算法**:

| 算法 | 全称 | 特点 |
|------|------|------|
| **PPO** | Proximal Policy Optimization | 策略梯度方法,稳定性好 |
| **GRPO** | Group Relative Policy Optimization | PPO 变体,相对奖励 |
| **SPO** | Simplified Policy Optimization | 简化版 PPO,计算高效 |

**训练流程**:

1. 使用模型生成多个候选回答
2. 计算奖励信号(可用规则或小型奖励模型)
3. 根据奖励优化策略
4. 迭代更新模型

### 3.5 推理组件设计

#### 3.5.1 文本生成器

```mermaid
graph LR
    Prompt[输入提示词] --> Tokenize[分词]
    Tokenize --> Init[初始化 KV-Cache]
    
    Init --> Loop{生成循环}
    Loop --> Forward[模型前向]
    Forward --> Logits[获取 Logits]
    Logits --> Sample[采样策略]
    Sample --> Token[生成 Token]
    Token --> Stop{停止条件?}
    
    Stop -->|否| Append[追加 Token]
    Append --> Loop
    
    Stop -->|是| Decode[解码输出]
    Decode --> Text[生成文本]
```

**停止条件**:

- 达到最大生成长度
- 生成 EOS Token(`<|im_end|>`)
- 重复检测触发(可选)

#### 3.5.2 采样策略

| 策略 | 描述 | 适用场景 |
|------|------|---------|
| **Greedy** | 选择概率最高的 Token | 确定性生成,重复性高 |
| **Top-K** | 从概率最高的 K 个 Token 中采样 | 平衡多样性和质量 |
| **Top-P (Nucleus)** | 从累积概率达到 P 的 Token 集合中采样 | 动态词汇表,灵活性高 |
| **Temperature** | 调整 Logits 的温度参数 | 控制生成随机性 |

**采样参数**:

```
GenerationConfig:
  - maxLength: 512
  - temperature: 0.8
  - topK: 50
  - topP: 0.9
  - repetitionPenalty: 1.1
  - doSample: true
```

#### 3.5.3 KV-Cache 机制

**缓存目的**: 避免在自回归生成时重复计算历史 Token 的 Key 和 Value

**实现策略**:

1. **初始化**: 第一次前向传播,计算所有位置的 K、V
2. **缓存存储**: 将每层的 K、V 按位置缓存
3. **增量计算**: 后续只计算新 Token 的 K、V,与缓存拼接
4. **内存管理**: 动态扩展、定期清理

**性能提升**: 推理速度提升 2-3 倍

## 四、数据处理设计

### 4.1 数据集类型

| 阶段 | 数据集类型 | 数据格式 | 数据量建议 |
|------|-----------|---------|-----------|
| **预训练** | 通用文本语料 | 纯文本 | 100M+ tokens |
| **SFT** | 指令-回答对 | JSONL(instruction, input, output) | 10K-100K 样本 |
| **LoRA** | 垂直领域数据 | 同 SFT | 1K-10K 样本 |
| **DPO** | 偏好对比数据 | JSONL(prompt, chosen, rejected) | 5K-50K 样本 |
| **RLAIF** | 提示词数据 | JSONL(prompt) | 10K+ 样本 |

### 4.2 数据预处理流程

```mermaid
graph TB
    RawData[原始数据] --> Validate[格式验证]
    Validate --> Clean[数据清洗]
    Clean --> Dedupe[去重]
    Dedupe --> Filter[质量过滤]
    Filter --> Tokenize[分词编码]
    Tokenize --> Pack[序列打包]
    Pack --> Cache[缓存存储]
    Cache --> DataLoader[数据加载器]
```

**清洗规则**:

1. 移除低质量文本(长度过短、乱码、重复字符)
2. 过滤敏感内容
3. 规范化标点符号和空格
4. 统一字符编码(UTF-8)

**序列打包策略**:

- **固定长度**: 每个样本固定 512 tokens
- **填充截断**: 不足填充 `<pad>`,超长截断
- **动态批次**: 按序列长度分组,减少填充浪费

### 4.3 数据加载器设计

**核心功能**:

1. **批次构建**: 自动填充、注意力掩码生成
2. **打乱策略**: 训练时随机打乱,评估时顺序读取
3. **缓存机制**: 预处理结果缓存到磁盘
4. **多线程加载**: 异步加载提升吞吐量

**数据整理器(Collator)职责**:

- 批次内序列对齐
- 构建 attention_mask
- 构建 labels(SFT 时掩码输入部分)
- 数据类型转换(List → NdArray)

## 五、模型规模设计

### 5.1 模型变体对比

```mermaid
graph LR
    subgraph Small 26M
        S_Emb[Token Emb<br/>6400×512] --> S_Layers[8 Layers<br/>512 dim]
        S_Layers --> S_Head[LM Head<br/>512×6400]
    end
    
    subgraph Medium 108M
        M_Emb[Token Emb<br/>6400×768] --> M_Layers[16 Layers<br/>768 dim]
        M_Layers --> M_Head[LM Head<br/>768×6400]
    end
    
    subgraph MoE 145M
        MoE_Emb[Token Emb<br/>6400×512] --> MoE_Layers[8 Layers<br/>4 Experts]
        MoE_Layers --> MoE_Head[LM Head<br/>512×6400]
    end
```

### 5.2 参数量计算

**Small 模型(26M)**:

| 组件 | 参数量计算 | 参数量 |
|------|----------|-------|
| Token Embedding | 6400 × 512 | 3.3M |
| 8 × Transformer Layer | - | - |
| - Attention (QKV) | 8 × (512 × 512 × 3) | 6.3M |
| - Attention Output | 8 × (512 × 512) | 2.1M |
| - FFN Layer 1 | 8 × (512 × 1024) | 4.2M |
| - FFN Layer 2 | 8 × (1024 × 512) | 4.2M |
| - LayerNorm | 8 × 2 × 512 | 0.008M |
| LM Head | 512 × 6400 | 3.3M |
| **总计** | - | **≈ 26M** |

**Medium 模型(108M)**:

- Embedding: 6400 × 768 = 4.9M
- 16 Layers: 每层 ≈ 5.4M → 86.4M
- LM Head: 768 × 6400 = 4.9M
- **总计**: ≈ 108M

**MoE 模型(145M)**:

- Embedding: 3.3M
- 8 Layers with 4 Experts: 每层 4 个 FFN 专家
  - 每个专家 FFN: 512 × 1024 + 1024 × 512 ≈ 1M
  - 8 层 × 4 专家 = 32M(FFN)
  - 其他组件: 8.4M
- LM Head: 3.3M
- **总计**: ≈ 145M

### 5.3 计算与内存需求

| 模型 | 参数量 | FP32 内存 | FP16 内存 | 训练显存(估算) | 推理显存(估算) |
|------|-------|----------|----------|---------------|---------------|
| Small | 26M | 104MB | 52MB | 2-4GB | 0.5-1GB |
| Medium | 108M | 432MB | 216MB | 8-12GB | 2-3GB |
| MoE | 145M | 580MB | 290MB | 10-16GB | 3-4GB |

**训练显存占用**:

- 模型参数
- 梯度(与参数同大小)
- 优化器状态(AdamW: 2 倍参数)
- 激活值(与批次大小、序列长度相关)
- **总计**: 约 8-12 倍参数量

**优化策略**:

- 梯度累积减少批次大小
- 梯度检查点减少激活值缓存
- 混合精度训练(FP16/BF16)

## 六、接口设计

### 6.1 模型 API

#### 6.1.1 模型创建

```
功能: 创建预定义配置的模型

方法签名:
  - MiniMindModel.createSmallModel(String name) → MiniMindModel
  - MiniMindModel.createMediumModel(String name) → MiniMindModel
  - MiniMindModel.createMoEModel(String name) → MiniMindModel
  
参数:
  - name: 模型名称

返回值:
  - 已初始化的 MiniMindModel 实例
```

#### 6.1.2 文本生成

```
功能: 根据提示词生成文本

方法签名:
  - MiniMindModel.generate(String prompt, GenerationConfig config) → String
  - MiniMindModel.generate(List<Integer> tokenIds, int maxLength) → List<Integer>
  
参数:
  - prompt: 输入提示词(字符串)
  - tokenIds: 输入 Token IDs(整数列表)
  - config: 生成配置(温度、采样策略等)
  - maxLength: 最大生成长度

返回值:
  - 生成的文本或 Token IDs
```

#### 6.1.3 模型推理

```
功能: 前向传播,获取 Logits

方法签名:
  - MiniMindModel.predict(NdArray tokenIds) → Variable
  
参数:
  - tokenIds: 输入 Token IDs,形状 [batch_size, seq_len]

返回值:
  - Logits 输出,形状 [batch_size, seq_len, vocab_size]
```

### 6.2 训练 API

#### 6.2.1 预训练

```
功能: 执行预训练

方法签名:
  - PretrainTrainer.train(MiniMindModel model, PretrainConfig config) → void
  
参数:
  - model: 待训练模型
  - config: 预训练配置
    - dataPath: 数据路径
    - batchSize: 批次大小
    - learningRate: 学习率
    - numEpochs: 训练轮数
    - saveSteps: 保存间隔

行为:
  - 加载并预处理数据
  - 执行训练循环
  - 定期保存检查点
  - 记录训练指标
```

#### 6.2.2 监督微调

```
功能: 执行 SFT 微调

方法签名:
  - SFTTrainer.train(MiniMindModel model, SFTConfig config) → void
  
参数:
  - model: 预训练模型
  - config: SFT 配置
    - dataPath: 指令数据路径
    - batchSize: 批次大小
    - learningRate: 学习率
    - numEpochs: 微调轮数

行为:
  - 加载指令数据
  - 应用对话模板
  - 仅计算输出部分损失
  - 保存微调后模型
```

#### 6.2.3 LoRA 微调

```
功能: 应用 LoRA 适配器进行微调

方法签名:
  - LoRATrainer.train(MiniMindModel model, LoRAConfig config) → LoRAAdapter
  
参数:
  - model: 基础模型
  - config: LoRA 配置
    - rank: LoRA 秩
    - alpha: 缩放因子
    - targetModules: 目标模块列表

返回值:
  - LoRA 适配器,可单独保存和加载
```

#### 6.2.4 DPO 训练

```
功能: 执行 DPO 偏好对齐训练

方法签名:
  - DPOTrainer.train(MiniMindModel model, DPOConfig config) → void
  
参数:
  - model: SFT 模型
  - config: DPO 配置
    - dataPath: 偏好数据路径
    - beta: KL 散度系数
    - refModelPath: 参考模型路径(可选)

行为:
  - 加载偏好对比数据
  - 计算 DPO 损失
  - 更新模型参数
```

### 6.3 Tokenizer API

```
功能: 分词与解码

方法签名:
  - MiniMindTokenizer.encode(String text) → List<Integer>
  - MiniMindTokenizer.decode(List<Integer> tokenIds) → String
  - MiniMindTokenizer.encodeBatch(List<String> texts, boolean padding, int maxLength) 
      → EncodedBatch
  
参数:
  - text: 待编码文本
  - tokenIds: 待解码 Token IDs
  - padding: 是否填充
  - maxLength: 最大长度

返回值:
  - Token IDs 或解码文本
  - EncodedBatch 包含 input_ids, attention_mask
```

### 6.4 配置管理 API

```
功能: 从文件加载和保存配置

方法签名:
  - MiniMindConfig.fromJson(String filePath) → MiniMindConfig
  - MiniMindConfig.toJson(String filePath) → void
  
参数:
  - filePath: JSON 配置文件路径

行为:
  - 解析 JSON 格式配置
  - 验证配置有效性
  - 序列化配置到文件
```

## 七、扩展功能设计

### 7.1 MoE 架构支持

```mermaid
graph TB
    Input[输入 X] --> Router[专家路由器]
    
    Router --> Expert1[专家 1 FFN]
    Router --> Expert2[专家 2 FFN]
    Router --> Expert3[专家 3 FFN]
    Router --> Expert4[专家 4 FFN]
    
    Expert1 --> Weight1[权重 w1]
    Expert2 --> Weight2[权重 w2]
    Expert3 --> Weight3[权重 w3]
    Expert4 --> Weight4[权重 w4]
    
    Weight1 --> Combine[加权组合]
    Weight2 --> Combine
    Weight3 --> Combine
    Weight4 --> Combine
    
    Combine --> Output[输出]
    
    Router -.负载均衡损失.-> Loss[辅助损失]
```

**设计要点**:

1. **专家路由**: Top-K 路由,每个 Token 选择 K 个(如 2 个)专家
2. **负载均衡**: 添加辅助损失确保专家使用均衡
3. **稀疏激活**: 仅激活部分专家,降低计算量
4. **参数共享**: Attention 层共享,仅 FFN 层使用 MoE

### 7.2 模型蒸馏支持

**蒸馏目标**: 从大模型(如 GPT-3)蒸馏到 MiniMind

**蒸馏方法**:

| 方法 | 损失函数 | 描述 |
|------|---------|------|
| **Logits 蒸馏** | KL 散度(学生 Logits, 教师 Logits) | 学习教师模型的输出分布 |
| **Hidden States 蒸馏** | MSE(学生 Hidden, 教师 Hidden) | 学习中间层表示 |
| **注意力蒸馏** | MSE(学生 Attention, 教师 Attention) | 学习注意力模式 |

### 7.3 模型压缩与量化

| 技术 | 描述 | 效果 |
|------|------|------|
| **权重量化** | FP32 → INT8/INT4 | 模型大小减少 75%-93% |
| **激活量化** | 动态量化激活值 | 推理速度提升 2-3 倍 |
| **剪枝** | 移除冗余参数 | 参数量减少 20-50% |
| **知识蒸馏** | 训练更小的模型 | 参数量减少 50-90% |

### 7.4 多模态扩展(可选)

**MiniMind-V 视觉扩展**:

```mermaid
graph LR
    Image[图像] --> ViT[视觉编码器]
    ViT --> Proj[投影层]
    
    Text[文本] --> TextEmb[文本嵌入]
    
    Proj --> Concat[拼接]
    TextEmb --> Concat
    
    Concat --> MiniMind[MiniMind Transformer]
    MiniMind --> Output[多模态输出]
```

**扩展组件**:

- 视觉编码器(如 ViT)
- 视觉-文本投影层
- 跨模态注意力机制

## 八、测试策略

### 8.1 单元测试

| 测试模块 | 测试内容 | 覆盖目标 |
|---------|---------|---------|
| **模型组件** | 各层前向传播输出形状正确性 | 100% |
| **注意力机制** | RoPE 编码、掩码、KV-Cache | 100% |
| **Tokenizer** | 编码解码可逆性、特殊 Token | 100% |
| **数据处理** | 批次构建、填充、掩码生成 | 100% |
| **训练组件** | 损失计算、梯度反向传播 | 90%+ |

### 8.2 集成测试

**测试场景**:

1. **端到端预训练**: 小规模数据集(1000 样本),训练 1 个 epoch,验证损失下降
2. **SFT 微调**: 加载预训练模型,微调 100 样本,验证生成质量
3. **LoRA 适配**: 应用 LoRA,训练 50 样本,验证适配器保存加载
4. **推理性能**: 批量生成,验证吞吐量和延迟
5. **检查点恢复**: 中断训练,从检查点恢复,验证状态一致性

### 8.3 性能测试

| 指标 | 测试方法 | 目标 |
|------|---------|------|
| **训练速度** | 测量每秒处理 Token 数 | Small: 2000+ tokens/s |
| **推理延迟** | 单个生成的时间 | Small: <100ms/token |
| **内存占用** | 训练和推理峰值显存 | 符合设计估算 |
| **模型收敛** | 损失曲线、评估指标 | 与 PyTorch 版本对齐 |

### 8.4 对比验证

**与原始 MiniMind(PyTorch)对比**:

1. **输出一致性**: 相同输入,Logits 差异 < 1e-3
2. **性能对比**: 训练速度、推理速度在合理范围(±20%)
3. **生成质量**: 人工评估生成文本的流畅性和相关性

## 九、部署与应用

### 9.1 命令行工具

**功能列表**:

| 命令 | 功能 | 示例 |
|------|------|------|
| `train-pretrain` | 预训练模型 | `java -jar minimind.jar train-pretrain --config pretrain.json` |
| `train-sft` | SFT 微调 | `java -jar minimind.jar train-sft --config sft.json` |
| `train-lora` | LoRA 微调 | `java -jar minimind.jar train-lora --config lora.json` |
| `generate` | 文本生成 | `java -jar minimind.jar generate --prompt "你好" --max-len 50` |
| `chat` | 对话模式 | `java -jar minimind.jar chat --model-path ./model` |
| `evaluate` | 模型评估 | `java -jar minimind.jar evaluate --test-data test.jsonl` |

### 9.2 API 服务

**服务接口**:

```
POST /v1/completions
请求体:
{
  "prompt": "你好,请介绍一下自己",
  "max_tokens": 100,
  "temperature": 0.8,
  "top_p": 0.9
}

响应:
{
  "id": "cmpl-xxx",
  "object": "text_completion",
  "created": 1234567890,
  "model": "minimind-small",
  "choices": [{
    "text": "生成的文本...",
    "index": 0,
    "finish_reason": "stop"
  }]
}
```

**兼容 OpenAI API**: 支持标准的 Chat Completions 格式

### 9.3 Web 界面

**功能模块**:

1. **对话界面**: 实时聊天,支持多轮对话
2. **参数调整**: 温度、Top-P、Top-K 等参数实时调整
3. **模型切换**: 在 Small、Medium、MoE 模型间切换
4. **训练监控**: 实时显示训练损失、学习率曲线
5. **模型管理**: 上传、下载、删除模型检查点

**技术栈建议**: 使用轻量级框架如 Javalin + 前端模板引擎

### 9.4 第三方集成

**支持集成场景**:

| 场景 | 集成方式 | 用途 |
|------|---------|------|
| **FastGPT** | OpenAI API 协议 | 知识库问答 |
| **Open-WebUI** | 标准 LLM 接口 | 通用聊天界面 |
| **Langchain4j** | 自定义 LLM 实现 | Java 应用集成 |

## 十、功能还原对照

### 10.1 与原版 MiniMind 功能对照表

**核心功能还原度检查**:

| 功能模块 | 原版 MiniMind (PyTorch) | TinyAI 实现 | 还原度 | 备注 |
|---------|------------------------|------------|-------|------|
| **Tokenizer** | BPE 分词器,6400 词汇表 | 自行实现 BPE | ✅ 100% | 完整实现 BPE 算法 |
| **模型架构** | Transformer Decoder | 基于 V2 Module | ✅ 100% | 使用 V2 API 构建 |
| **位置编码** | RoPE | 自行实现 | ✅ 100% | 旋转位置嵌入 |
| **注意力机制** | Multi-Head + Causal Mask | 自行实现 | ✅ 100% | 支持 KV-Cache |
| **激活函数** | SiLU (Swish) | V2 SiLU 层 | ✅ 100% | 使用 V2 组件 |
| **层归一化** | LayerNorm (Pre-LN) | V2 LayerNorm | ✅ 100% | 使用 V2 组件 |
| **预训练** | 自回归语言模型训练 | PretrainTrainer | ✅ 100% | 交叉熵损失 |
| **SFT 微调** | 指令微调,掩码损失 | SFTTrainer | ✅ 100% | 对话模板支持 |
| **LoRA** | 低秩适配,Q/V 投影 | LoRAAdapter | ✅ 100% | 参数高效微调 |
| **DPO** | 直接偏好优化 | DPOTrainer | ✅ 100% | 无需奖励模型 |
| **RLAIF** | PPO/GRPO/SPO | RLAIFTrainer | ✅ 100% | 可选依赖 RL 模块 |
| **MoE 架构** | 4 专家稀疏 MoE | MoELayer | ✅ 100% | 负载均衡损失 |
| **文本生成** | Greedy/Top-K/Top-P/Temperature | TextGenerator | ✅ 100% | 多种采样策略 |
| **KV-Cache** | 增量推理加速 | KVCacheManager | ✅ 100% | 内存优化 |
| **模型规模** | 26M/108M/145M | 三种配置 | ✅ 100% | 参数量对齐 |
| **数据处理** | 清洗、去重、分词 | DataProcessor | ✅ 100% | 完整数据流程 |
| **检查点** | 保存/加载/恢复训练 | CheckpointManager | ✅ 100% | 状态完整性 |
| **命令行工具** | train/generate/chat/evaluate | CLI | ✅ 100% | 一致的用户体验 |
| **API 服务** | OpenAI 兼容接口 | APIServer | ✅ 100% | 标准协议 |

**设计原则**:
1. **架构对齐**: 严格遵循原版 MiniMind 的 Transformer Decoder 架构
2. **参数一致**: 模型配置(层数、维度、头数)与原版完全一致
3. **功能完整**: 覆盖预训练、微调、推理、部署全流程
4. **输出验证**: 相同输入下,Logits 输出误差 < 1e-3

### 10.2 V2 组件使用规范

**必须使用的 V2 组件**:

```java
// 基础模块

// 容器

// 线性层

// 激活函数

// 归一化
import io.leavesfly.tinyai.nnet.v2.layer.norm.Dropout;

// 初始化
        import io.leavesfly.tinyai.nnet.v2.init.KaimingUniform;
        import io.leavesfly.tinyai.nnet.v2.init.Xavier;
```

**禁止使用的 V1 组件**:

```java
// ❌ 禁止使用

import io.leavesfly.tinyai.nnet.layer.dnn.AffineLayer;  // 使用 V2 Linear 代替

```

**自行实现的组件**:

| 组件 | 原因 | 继承关系 |
|------|------|----------|
| `MultiHeadAttention` | V2 中无 RoPE + KV-Cache 支持 | 继承 `Module` |
| `TokenEmbedding` | V2 中无嵌入查找层 | 继承 `Module` |
| `RotaryPositionEmbedding` | V2 中无 RoPE 实现 | 继承 `Module` |
| `MoELayer` | V2 中无 MoE 支持 | 继承 `Module` |
| `MiniMindTokenizer` | 独立工具类,无需继承 | 纯 Java 类 |

### 10.3 关键实现细节

#### 10.3.1 MultiHeadAttention 实现要点

```
设计要求:
  - 继承 core.io.leavesfly.tinyai.nnet.Module
  - Q、K、V 投影使用 V2 Linear 层
  - 输出投影使用 V2 Linear 层
  - 支持因果掩码(Causal Mask)
  - 集成 RoPE 位置编码
  - 支持 KV-Cache 增量计算
  - 支持 Flash Attention(可选优化)

参数注册:
  - registerModule("q_proj", new Linear(...))
  - registerModule("k_proj", new Linear(...))
  - registerModule("v_proj", new Linear(...))
  - registerModule("o_proj", new Linear(...))
  - registerModule("rope", new RotaryPositionEmbedding(...))

forward 方法:
  public Variable forward(Variable... inputs) {
      Variable x = inputs[0];
      Variable kvCache = inputs.length > 1 ? inputs[1] : null;
      
      // Q、K、V 投影
      Variable q = qProj.forward(x);
      Variable k = kProj.forward(x);
      Variable v = vProj.forward(x);
      
      // 应用 RoPE
      q = rope.forward(q);
      k = rope.forward(k);
      
      // KV-Cache 处理
      if (kvCache != null) {
          // 拼接历史 K、V
      }
      
      // 计算注意力
      Variable attn = scaledDotProductAttention(q, k, v, causalMask);
      
      // 输出投影
      return oProj.forward(attn);
  }
```

#### 10.3.2 TokenEmbedding 实现要点

```
设计要求:
  - 继承 core.io.leavesfly.tinyai.nnet.Module
  - 嵌入矩阵形状: [vocabSize, embeddingDim]
  - 支持权重共享(与 LM Head 共享)

参数注册:
  - registerParameter("weight", new Parameter(embeddingMatrix))

forward 方法:
  public Variable forward(Variable... inputs) {
      Variable tokenIds = inputs[0];  // [batch, seq_len]
      // 查找嵌入向量
      return embeddingLookup(weight, tokenIds);
  }
```

#### 10.3.3 MiniMindBlock 组合示例

```
设计结构:
  public class MiniMindBlock extends Module {
      private TokenEmbedding tokenEmbedding;
      private ModuleList transformerLayers;  // V2 ModuleList
      private LayerNorm finalNorm;           // V2 LayerNorm
      private Linear lmHead;                 // V2 Linear
      
      public MiniMindBlock(MiniMindConfig config) {
          super("MiniMindBlock");
          
          // 注册子模块
          tokenEmbedding = registerModule("token_embedding", 
              new TokenEmbedding(config.vocabSize, config.hiddenSize));
          
          transformerLayers = new ModuleList();
          for (int i = 0; i < config.numLayers; i++) {
              transformerLayers.add(
                  new MiniMindTransformerLayer(config, i));
          }
          registerModule("layers", transformerLayers);
          
          finalNorm = registerModule("final_norm", 
              new LayerNorm(config.hiddenSize));
          
          lmHead = registerModule("lm_head", 
              new Linear(config.hiddenSize, config.vocabSize, false));
      }
      
      @Override
      public Variable forward(Variable... inputs) {
          Variable tokenIds = inputs[0];
          
          // Token 嵌入
          Variable x = tokenEmbedding.forward(tokenIds);
          
          // Transformer 层
          for (Module layer : transformerLayers) {
              x = layer.forward(x);
          }
          
          // 最终归一化
          x = finalNorm.forward(x);
          
          // LM Head
          return lmHead.forward(x);
      }
  }
```

## 十一、文档与示例

### 11.1 文档结构

```
doc/
├── 技术架构文档.md              # 本文档
├── MiniMind模型设计.md          # 模型详细设计
├── 快速开始指南.md              # 快速上手教程
├── 预训练指南.md                # 预训练详细步骤
├── 微调指南.md                  # SFT/LoRA/DPO 指南
├── API参考.md                   # 完整 API 文档
├── 性能优化指南.md              # 性能调优建议
├── 常见问题FAQ.md               # 问题排查
└── 对比PyTorch版本.md           # 与原版对比说明
```

### 11.2 示例代码

**示例列表**:

1. **01-模型创建与推理.java**: 创建模型并生成文本
2. **02-预训练示例.java**: 完整预训练流程
3. **03-SFT微调示例.java**: 指令微调流程
4. **04-LoRA微调示例.java**: LoRA 适配器训练
5. **05-DPO训练示例.java**: 偏好对齐训练
6. **06-文本生成策略.java**: 不同采样策略对比
7. **07-模型评估.java**: 评估指标计算
8. **08-模型部署.java**: API 服务部署

### 11.3 README 内容要点

**核心内容**:

1. **项目介绍**: MiniMind 是什么,为什么选择它
2. **特性列表**: 26M 参数、快速训练、全流程支持
3. **快速开始**: 5 分钟运行第一个示例
4. **安装说明**: Maven 依赖配置
5. **使用示例**: 代码片段展示核心功能
6. **模型下载**: 预训练模型权重链接(如有)
7. **性能指标**: 训练时间、推理速度、生成质量
8. **贡献指南**: 如何参与项目

## 十二、实施计划

### 12.1 阶段划分

```mermaid
gantt
    title MiniMind 模块开发计划
    dateFormat YYYY-MM-DD
    section 第一阶段
    基础架构搭建          :a1, 2025-01-01, 7d
    模型组件实现          :a2, after a1, 10d
    Tokenizer实现         :a3, after a1, 5d
    
    section 第二阶段
    预训练功能开发        :b1, after a2, 7d
    数据处理组件          :b2, after a3, 5d
    单元测试              :b3, after b1, 3d
    
    section 第三阶段
    SFT微调实现           :c1, after b3, 5d
    LoRA功能开发          :c2, after c1, 5d
    推理组件开发          :c3, after a2, 7d
    
    section 第四阶段
    DPO训练实现           :d1, after c2, 5d
    性能优化              :d2, after c3, 5d
    集成测试              :d3, after d1, 3d
    
    section 第五阶段
    应用层开发            :e1, after d3, 7d
    文档编写              :e2, after d3, 5d
    发布准备              :e3, after e1, 3d
```

### 12.2 里程碑目标

| 里程碑 | 交付物 | 完成标准 |
|-------|-------|---------|
| **M1: 基础模型** | MiniMindModel、MiniMindBlock、基础层 | 可成功前向传播,单元测试通过 |
| **M2: 预训练能力** | PretrainTrainer、数据加载 | 小规模数据集训练,损失下降 |
| **M3: 微调能力** | SFT、LoRA 训练器 | 完成指令微调,生成质量可接受 |
| **M4: 推理优化** | 文本生成器、KV-Cache | 推理速度达标,多种采样策略 |
| **M5: 完整系统** | 命令行工具、API 服务、文档 | 端到端流程可运行,文档完善 |

### 12.3 优先级排序

**P0(必须实现)**:

- MiniMindModel 和 MiniMindBlock 核心架构
- MiniMindTokenizer 分词器
- 预训练功能(PretrainTrainer)
- SFT 微调功能
- 基础文本生成(Greedy、Top-K、Top-P)
- 命令行工具
- 核心文档(README、快速开始)

**P1(高优先级)**:

- LoRA 微调
- DPO 训练
- KV-Cache 优化
- 模型保存加载
- 性能测试
- API 服务
- 完整文档

**P2(可选)**:

- RLAIF 训练(PPO/GRPO/SPO)
- MoE 架构
- 模型蒸馏
- 多模态扩展
- Web 界面
- 第三方集成

### 12.4 风险与应对

| 风险 | 影响 | 应对策略 |
|------|------|---------|
| **V2 API 功能缺失** | 高 | 优先使用 V2,缺失部分通过适配器桥接 V1 或自行实现 |
| **性能不达标** | 中 | 引入性能分析工具,针对性优化热点代码 |
| **与原版对齐困难** | 中 | 建立详细的对比测试,逐层验证输出一致性 |
| **文档不完善** | 低 | 每个模块开发完成后立即编写文档 |
| **时间超期** | 中 | 优先保证 P0 功能,P1/P2 按实际情况调整 |

## 十三、质量保障

### 13.1 代码规范

**遵循 TinyAI 项目规范**:

1. **命名规范**: 类名 PascalCase,方法名 camelCase
2. **注释规范**: 中文注释,符合 JavaDoc 标准
3. **包结构**: 按功能模块清晰划分
4. **依赖管理**: 仅依赖 TinyAI 内部模块,避免引入第三方深度学习库

### 13.2 测试覆盖

**覆盖率目标**:

- 核心模型组件: 100%
- 训练组件: 90%+
- 工具类: 80%+
- 整体覆盖率: 85%+

### 13.3 性能基准

**建立性能基线**:

- 预训练速度: Tokens/秒
- 微调速度: Samples/秒
- 推理延迟: ms/token
- 内存占用: GB

**持续监控**: 每次代码变更后运行性能测试,防止性能退化

### 13.4 文档质量

**文档检查清单**:

- [ ] API 文档完整,包含所有公开接口
- [ ] 每个模块有对应的设计文档
- [ ] 示例代码可运行,结果符合预期
- [ ] README 清晰易懂,新手可快速上手
- [ ] 常见问题 FAQ 覆盖典型使用场景

## 十四、V2 API 使用最佳实践

### 14.1 模块继承规范

**核心原则**: 所有神经网络组件必须继承 `core.io.leavesfly.tinyai.nnet.Module`。

```java
import core.io.leavesfly.tinyai.nnet.Module;

public class MyCustomLayer extends Module {
    public MyCustomLayer() {
        super("MyCustomLayer");
        // 初始化参数
    }

    @Override
    public Variable forward(Variable... inputs) {
        // 实现前向传播
        return output;
    }
}
```

**子模块注册**:
```java
// ✅ 正确: 使用 registerModule 注册子模块
Linear linear = registerModule("linear", new Linear(512, 1024));

// ❌ 错误: 直接赋值,参数不会被跟踪
Linear linear = new Linear(512, 1024);
```

**参数注册**:
```java
// ✅ 正确: 使用 registerParameter 注册参数
Parameter weight = registerParameter("weight", 
    new Parameter(new NdArray(shape)));

// ❌ 错误: 直接赋值,优化器无法访问
NdArray weight = new NdArray(shape);
```

### 14.2 常见陷阱和解决方案

#### 陷阱1: Dropout 路径错误

**错误示例**:

```java
// ❌ 错误: 使用了 V1 路径
```

**正确做法**:
```java
// ✅ 正确: 使用 V2 路径
import io.leavesfly.tinyai.nnet.v2.layer.norm.Dropout;

Dropout dropout = registerModule("dropout", new Dropout(0.1f));
```

#### 陷阱2: NdArray 拷贝问题

**问题**: `NdArray` 缺少 `copy()` 方法。

**解决方案**:
```java
// 手动拷贝 buffer
public static NdArray copyNdArray(NdArray src) {
    NdArray dst = new NdArray(src.getShape());
    System.arraycopy(src.getBuffer(), 0, 
                     dst.getBuffer(), 0, 
                     src.getBuffer().length);
    return dst;
}
```

#### 陷阱3: Shape 维度获取

**错误示例**:
```java
// ❌ 错误: Shape 没有 getShapeDims() 方法
int[] dims = shape.getShapeDims();
```

**正确做法**:
```java
// ✅ 正确: 使用 getDims()
int[] dims = shape.getDims();
```

#### 陷阱4: 梯度清零

**问题**: 训练时必须手动清空梯度。

**正确做法**:
```java
// 每个训练步前清空梯度
model.clearGrads();

// 前向传播
Variable output = model.forward(input);

// 计算损失
Variable loss = lossFunction.forward(output, target);

// 反向传播
loss.backward();

// 参数更新
optimizer.step();
```

### 14.3 性能优化技巧

#### 技巧1: 及时关闭训练模式

```java
// 训练时
model.setTraining(true);
// ... 训练代码

// 推理时必须关闭训练模式
model.setTraining(false);
Variable output = model.predict(input);
```

**影响**: 训练模式下 Dropout 会随机丢弃,LayerNorm 会更新统计量。

#### 技巧2: 使用 KV-Cache

```java
// 首次生成
KVCacheManager cache = new KVCacheManager(numLayers);
int[] firstToken = model.generate(prompt, 1, cache);

// 后续生成,复用缓存
for (int i = 0; i < maxNewTokens; i++) {
    int[] nextToken = model.generate(firstToken, 1, cache);
    // ...
}
```

**加速比**: 5-10× (生成长序列时)

#### 技巧3: 批量推理

```java
// 单个样本 (慢)
for (String text : texts) {
    Variable output = model.predict(tokenize(text));
}

// 批量处理 (快 2-5×)
EncodedBatch batch = tokenizer.encodeBatch(texts, true, maxLen);
Variable outputs = model.predict(new Variable(batch.toNdArray()));
```

### 14.4 常见问题 FAQ

#### Q1: 如何确认使用了 V2 组件?

**A**: 检查 import 路径,必须包含 `v2`:

```java

import io.leavesfly.tinyai.nnet.layer.dnn.AffineLayer; // ❌ V1
```

#### Q2: 如何调试梯度消失/爆炸?

**A**: 
1. 添加梯度裁剪:
   ```java
   config.setMaxGradNorm(1.0f);
   ```

2. 检查梯度范数:
   ```java
   float gradNorm = computeGradNorm(model.getAllParams());
   System.out.println("Grad Norm: " + gradNorm);
   ```

3. 使用 Pre-LayerNorm 结构(更稳定)

#### Q3: 训练时内存不足?

**A**:
1. 减小批次大小: `config.setBatchSize(16)`
2. 使用梯度累积: `config.setGradientAccumulationSteps(4)`
3. 减小序列长度: `config.setMaxSeqLen(256)`

#### Q4: 如何验证模型输出正确性?

**A**: 与原版 MiniMind 对比:
```java
// 1. 使用相同的随机种子
Random.setSeed(42);

// 2. 使用相同的输入
int[] input = {1, 2, 3, 4, 5};

// 3. 比较 logits
Variable output = model.predict(new Variable(input));
// 对比 PyTorch 版本的输出,误差应 < 1e-3
```

### 14.5 实现细节补充

#### MoE 架构实现状态

**已实现组件** (P3-3 完成):
- ✅ `ExpertNetwork.java` (158行) - FFN专家网络
- ✅ `ExpertRouter.java` (262行) - Top-K路由,Softmax归一化
- ✅ `MoELayer.java` (319行) - 专家组合,加权输出
- ✅ `LoadBalanceLoss.java` (120行) - 负载均衡损失
- ✅ `MoEConfig.java` (297行) - 4/8/16专家配置
- ✅ `MoETransformerLayer.java` (226行) - 替换FFN层

**使用示例**:
```java
// 创建 MoE 配置
MiniMindConfig config = MiniMindConfig.createMoEConfig();
config.setUseMoE(true);
config.setNumExperts(4);
config.setNumExpertsPerToken(2);

// 创建模型
MiniMindModel moeModel = new MiniMindModel("moe-model", config);

// 训练时添加负载均衡损失
float balanceLoss = moeModel.getLoadBalanceLoss();
totalLoss = taskLoss + 0.01f * balanceLoss;
```

#### DPO 训练实现状态

**已实现组件** (P3-1 完成):
- ✅ `DPODataset.java` (260行) - 偏好对数据加载
- ✅ `DPOConfig.java` (178行) - Beta参数、KL散度配置
- ✅ `DPOLoss.java` (163行) - DPO损失函数
- ✅ `DPOTrainer.java` (380行) - 策略+参考模型训练
- ✅ `DPOE2ETest.java` (210行) - 端到端测试

**关键参数**:
```java
DPOConfig dpoConfig = new DPOConfig();
dpoConfig.setBeta(0.1f);          // KL惩罚系数,越大越保守
dpoConfig.setLearningRate(5e-6f); // DPO用极小学习率
```

#### RLAIF 训练实现状态

**已实现组件** (P3-2 部分完成):
- ✅ `SPOTrainer.java` (250行) - Simple Preference Optimization
- ✅ `PPOTrainer.java` (412行) - Proximal Policy Optimization
- ✅ `GRPOTrainer.java` (365行) - Group Relative Policy Optimization
- ✅ `ValueNetwork.java` (186行) - Critic价值网络

**训练建议**:
```java
// SPO: 最简单,适合快速实验
SPOConfig spoConfig = new SPOConfig();
spoConfig.setLearningRate(1e-5f);

// PPO: 经典算法,稳定性好
PPOConfig ppoConfig = new PPOConfig();
ppoConfig.setClipRange(0.2f);
ppoConfig.setValueLossCoef(0.5f);

// GRPO: 组相对优化,样本效率高
GRPOConfig grpoConfig = new GRPOConfig();
grpoConfig.setGroupSize(4);
```

#### API 服务实现状态

**已实现组件** (P3-5 完成):
- ✅ `SimpleJSON.java` (288行) - 零依赖JSON工具
- ✅ `MiniMindAPIServer.java` (207行) - HttpServer服务器
- ✅ `CompletionHandler.java` (214行) - /v1/completions
- ✅ `ChatCompletionHandler.java` (246行) - /v1/chat/completions
- ✅ `API-GUIDE.md` (361行) - 完整使用指南

**启动服务**:
```bash
# Linux/Mac
./bin/start-api.sh 8080

# Windows
.\bin\start-api.bat 8080

# 测试健康检查
curl http://localhost:8080/health
```

**OpenAI 兼容性**: 完全兼容 OpenAI API 规范,可直接替换 OpenAI 端点。

## 十五、总结

### 15.1 模块价值

`tinyai-model-minimind` 模块通过在 TinyAI 框架上完整复现 MiniMind 轻量级语言模型,实现了以下核心价值:

1. **教育价值**: 为学习者提供从零训练 LLM 的完整代码示例
2. **技术验证**: 验证 TinyAI V2 架构在真实 LLM 场景的可行性
3. **实用工具**: 提供可在普通硬件上快速训练的轻量级语言模型方案
4. **开放生态**: 丰富 TinyAI 模型库,推动 Java AI 社区发展

### 15.2 技术亮点

- **极致轻量**: 26M 参数,资源消耗极低
- **全流程覆盖**: 预训练、SFT、LoRA、DPO、RLAIF 完整支持
- **纯 Java 实现**: 基于 TinyAI,无第三方深度学习框架依赖
- **性能优化**: KV-Cache、混合精度、梯度累积等优化技术
- **易用性强**: 命令行、API、Web 多种使用方式
- **零第三方依赖**: API服务基于Java标准库HttpServer,JSON手动实现

### 15.3 实现进度总结

**已完成核心功能** (截至 2025-12-07):
- ✅ 模型架构 (MiniMindModel, MiniMindBlock)
- ✅ BPE Tokenizer (完整训练和使用)
- ✅ 预训练 (PretrainTrainer)
- ✅ SFT微调 (SFTTrainer)
- ✅ LoRA微调 (LoRAAdapter, LoRALinear)
- ✅ DPO训练 (DPOTrainer)
- ✅ RLAIF训练 (SPO/PPO/GRPO)
- ✅ MoE架构 (4专家MoE)
- ✅ 命令行工具 (train/generate/chat/evaluate)
- ✅ API服务 (OpenAI兼容)
- ✅ 完整文档 (API参考、快速开始、模型设计)

**文档资源**:
- ✅ [快速开始指南.md](./快速开始指南.md) (627行) - 5分钟上手
- ✅ [MiniMind模型设计.md](./MiniMind模型设计.md) (977行) - 架构深入解析
- ✅ [API参考.md](./API参考.md) (1053行) - 完整API文档
- ✅ [CLI-GUIDE.md](./CLI-GUIDE.md) - 命令行工具
- ✅ [API-GUIDE.md](./API-GUIDE.md) (361行) - API服务指南

### 15.4 后续演进

**潜在扩展方向**:

1. **模型规模扩展**: 支持更大的模型变体(如 500M、1B)
2. **多模态融合**: 接入视觉、音频等模态
3. **分布式训练**: 支持多机多卡训练
4. **推理加速**: 集成 ONNX、TensorRT 等推理引擎
5. **应用场景**: 垂直领域定制(医疗、法律、金融等)

---

**设计完成**: 本文档详细设计了 `tinyai-model-minimind` 模块的架构、组件、接口和实施计划,并补充了 V2 API 使用最佳实践和实现细节,为后续开发和使用提供全面指导。

**最后更新**: 2025-12-07  
**文档版本**: v2.0 (新增V2最佳实践、FAQ、实现状态)
