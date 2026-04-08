# TinyAI MiniMind - 轻量级语言模型

> 基于 TinyAI 框架的 26M 参数 GPT 风格语言模型 - 教育友好 · 资源高效 · 功能完整

## 📚 项目概述

`tinyai-model-minimind` 是 TinyAI 项目中对轻量级语言模型 [MiniMind](https://github.com/jingyaogong/minimind) 的纯 Java 实现。该模块基于 TinyAI V2 架构，实现了一个仅 **26M 参数**的超小型 Transformer Decoder-Only 模型，具备完整的现代大语言模型训练与推理能力。

### 设计理念

**MiniMind 的核心价值在于「小而全」**:

- **🎓 教育友好**: 清晰的架构设计，完整的文档，适合学习 Transformer 和 LLM 原理
- **💡 资源高效**: 26M 参数可在普通 CPU 上训练和推理，无需昂贵的 GPU
- **🔧 功能完整**: 支持预训练、SFT、LoRA、DPO、RLAIF 等完整的 LLM 训练流程
- **🚀 生产就绪**: 纯 Java 实现，易于集成到企业应用，支持 REST API 部署
- **📈 可扩展**: 支持 MoE 架构，可扩展至 145M 参数

### 核心特性

| 特性类别 | 功能说明 |
|---------|----------|
| **模型架构** | Transformer Decoder-Only · 多头注意力 · RoPE 位置编码 · Pre-LayerNorm · SiLU 激活 |
| **训练能力** | 预训练 · 监督微调 (SFT) · LoRA 微调 · DPO 训练 · RLAIF(PPO/GRPO/SPO) · MoE 架构 |
| **推理优化** | KV-Cache 加速 · 多种采样策略 · 批量推理 · 流式生成 |
| **工程特性** | 纯 Java 实现 · V2 组件架构 · BPE Tokenizer · CLI 工具 · REST API |
| **训练演示** | MiniMindTrainDemo · 三阶段完整训练流程 · 超小规模教学数据集 |

## 🏗️ 架构设计

### 整体架构

MiniMind 采用经典的 Transformer Decoder-Only 架构，类似 GPT 系列:

```mermaid
graph TB
    subgraph Input["输入层"]
        TokenIds["Token IDs<br/>[batch, seq_len]"]
        TokenEmb["Token Embedding<br/>维度：512"]
    end
    
    subgraph Transformer["Transformer 层 x 8"]
        Layer1["TransformerLayer 1<br/>Attention + FFN"]
        Layer2["TransformerLayer 2<br/>Attention + FFN"]
        LayerN["TransformerLayer 8<br/>Attention + FFN"]
        
        Layer1 --> Layer2
        Layer2 -.-> LayerN
    end
    
    subgraph Output["输出层"]
        FinalNorm["Final LayerNorm"]
        LMHead["LM Head<br/>Linear: 512->6400"]
        Logits["Logits<br/>[batch, seq_len, vocab]"]
    end
    
    TokenIds --> TokenEmb
    TokenEmb --> Layer1
    Layer1 --> Layer2
    Layer2 -.-> LayerN
    LayerN --> FinalNorm
    FinalNorm --> LMHead
    LMHead --> Logits
```

### 模块依赖关系

```mermaid
graph TB
    subgraph "TinyAI 核心框架"
        NdArray["tinyai-deeplearning-ndarr<br/>多维数组"]
        Func["tinyai-deeplearning-func<br/>自动微分"]
        NNet["tinyai-deeplearning-nnet<br/>神经网络层"]
        ML["tinyai-deeplearning-ml<br/>Model/Trainer"]
    end
    
    subgraph "MiniMind 实现"
        MiniMindModel["MiniMindModel<br/>模型接口"]
        MiniMindBlock["MiniMindBlock<br/>模型主体"]
        TransformerLayer["TransformerLayer<br/>Transformer 层"]
        Attention["MultiHeadAttention<br/>注意力机制"]
        Embedding["TokenEmbedding<br/>词嵌入"]
        Tokenizer["MiniMindTokenizer<br/>分词器"]
    end
    
    subgraph "V2 组件复用"
        Module["Module<br/>基础模块"]
        Linear["Linear<br/>线性层"]
        LayerNorm["LayerNorm<br/>归一化"]
        SiLU["SiLU<br/>激活函数"]
    end
    
    NdArray --> Func
    Func --> NNet
    NNet --> ML
    
    ML --> MiniMindModel
    MiniMindModel --> MiniMindBlock
    MiniMindBlock --> TransformerLayer
    MiniMindBlock --> Embedding
    TransformerLayer --> Attention
    
    Module --> Attention
    Module --> Embedding
    Module --> TransformerLayer
    Linear --> TransformerLayer
    LayerNorm --> TransformerLayer
    SiLU --> TransformerLayer
    
    NNet --> Module
    NNet --> Linear
    NNet --> LayerNorm
    NNet --> SiLU
```

### 模型配置规模

| 配置项 | Small (默认) | Medium | MoE |
|--------|-------------|--------|-----|
| **参数量** | 26M | 108M | 145M (激活 ~72M) |
| **层数** | 8 | 16 | 8 |
| **隐藏维度** | 512 | 768 | 512 |
| **注意力头数** | 16 | 16 | 16 |
| **FFN 维度** | 1,024 | 1,536 | 1,024 |
| **专家数量** | - | - | 4 (Top-2) |
| **推理内存** | ~100MB | ~430MB | ~580MB |
| **训练显存** | 2-4GB | 8-12GB | 10-16GB |

### 核心组件说明

| 组件类别 | 组件名称 | 包路径 | 实现状态 | 说明 |
|---------|---------|--------|---------|------|
| **配置管理** | MiniMindConfig | `model` | ✅ 完成 | Small/Medium/MoE 三种预设 |
| **词嵌入层** | TokenEmbedding | `model.embedding` | ✅ 完成 | 词汇表 → 隐藏向量映射 |
| **位置编码** | RotaryPositionEmbedding | `model.embedding` | ✅ 完成 | RoPE 旋转位置编码 |
| **注意力机制** | MultiHeadAttention | `model.attention` | ✅ 完成 | 16 头注意力 + KV-Cache |
| **KV 缓存** | KVCache | `model.attention` | ✅ 完成 | 推理加速缓存管理 |
| **Transformer 层** | MiniMindTransformerLayer | `model.transformer` | ✅ 完成 | Pre-LN + 残差连接 |
| **模型主体** | MiniMindBlock | `model` | ✅ 完成 | 8 层 Transformer 堆叠 |
| **模型接口** | MiniMindModel | `model` | ✅ 完成 | 继承 Model，统一接口 |
| **分词器** | MiniMindTokenizer | `tokenizer` | ✅ 完成 | 字符级 + BPE 训练支持 |
| **词汇表** | Vocabulary | `tokenizer` | ✅ 完成 | Token 管理与映射 |
| **BPE 训练器** | BPETrainer | `tokenizer` | ✅ 完成 | BPE 合并规则学习 |
| **MoE 架构** | MiniMindMoEModel | `model.moe` | ✅ 完成 | 4 专家 + 负载均衡 |
| **推理引擎** | generate() | `model` | ✅ 完成 | Greedy/TopK/TopP/Temp |
| **预训练** | PretrainTrainer | `training` | ✅ 完成 | 支持训练循环与检查点 |
| **监督微调** | SFTTrainer | `training` | ✅ 完成 | 指令微调训练器 |
| **LoRA 微调** | LoRAAdapter | `training.lora` | ✅ 完成 | 低秩适配器 |
| **DPO 训练** | DPOTrainer | `training.dpo` | ✅ 完成 | 直接偏好优化 |
| **RLAIF 训练** | PPO/GRPO/SPO | `training.rl` | ✅ 完成 | 强化学习训练器 |
| **CLI 工具** | MiniMindCLI | `cli` | ✅ 完成 | train/generate/chat 等命令 |
| **API 服务** | MiniMindAPIServer | `api` | ✅ 完成 | OpenAI 兼容 REST API |

## 🚀 快速开始

### 环境准备

| 组件 | 要求 |
|------|------|
| **JDK** | Java 17+ |
| **Maven** | 3.6+ |
| **内存** | 4GB+ (推荐 8GB) |
| **GPU** | 可选 (CPU 也可运行) |

```bash
# 1. 克隆项目
git clone https://github.com/your-org/TinyAI.git
cd TinyAI

# 2. 编译安装
mvn clean install -DskipTests

# 3. 进入 MiniMind 模块
cd tinyai-model/tinyai-model-minimind
```

### 5 分钟上手示例

#### 示例 1: 创建模型并生成文本

```java
import io.leavesfly.tinyai.minimind.model.MiniMindModel;
import io.leavesfly.tinyai.minimind.tokenizer.MiniMindTokenizer;
import java.util.*;
import java.util.stream.Collectors;

public class QuickStart {
    public static void main(String[] args) {
        // 1. 创建 Tokenizer
        MiniMindTokenizer tokenizer = MiniMindTokenizer
            .createCharLevelTokenizer(6400, 512);
        
        // 2. 创建模型 (Small 26M 参数)
        MiniMindModel model = MiniMindModel.create("my-model", "small");
        model.setTraining(false);  // 设置为推理模式
        
        System.out.println("✅ 模型创建成功!");
        System.out.println("📊 " + model.getDescription());
        
        // 3. 准备输入
        String prompt = "Hello, world!";
        List<Integer> tokens = tokenizer.encode(prompt, true, false);
        int[] tokenArray = tokens.stream().mapToInt(i -> i).toArray();
        
        // 4. 生成文本
        System.out.println("\n🔮 开始生成...");
        int[] output = model.generate(
            tokenArray,  // 提示词
            50,          // 最大生成长度
            0.7f,        // 温度
            0,           // Top-K (0=不使用)
            0.9f         // Top-P
        );
        
        // 5. 解码输出
        String result = tokenizer.decode(
            Arrays.stream(output).boxed().collect(Collectors.toList())
        );
        
        System.out.println("✨ 生成结果：" + result);
    }
}
```

**运行结果**:
```
✅ 模型创建成功!
📊 MiniMind Language Model - small with 26M parameters

🔮 开始生成...
✨ 生成结果：Hello, world! ...
```

#### 示例 2: 使用不同采样策略

```java
// 贪婪采样 (确定性，最高质量)
int[] greedy = model.generate(tokenArray, 50, 0.0f, 0, 0.0f);

// 温度采样 (温和随机)
int[] temp = model.generate(tokenArray, 50, 0.7f, 0, 0.0f);

// Top-K 采样 (从前 K 个中选择)
int[] topk = model.generate(tokenArray, 50, 1.0f, 40, 0.0f);

// Top-P 采样 (Nucleus 采样)
int[] topp = model.generate(tokenArray, 50, 1.0f, 0, 0.9f);

// 组合采样 (Top-K + Top-P + Temperature)
int[] combined = model.generate(tokenArray, 50, 0.8f, 40, 0.9f);
```

**采样策略选择指南**:

| 任务类型 | Temperature | Top-K | Top-P | 说明 |
|---------|-------------|-------|-------|------|
| **事实问答** | 0.0 | 0 | 0.0 | 贪婪采样，高准确性 |
| **代码生成** | 0.2 | 0 | 0.95 | 低温度，准确性优先 |
| **对话聊天** | 0.7 | 0 | 0.9 | 平衡质量和多样性 |
| **创意写作** | 0.9 | 0 | 0.95 | 高多样性 |
| **翻译** | 0.0 | 0 | 0.0 | 贪婪采样 |

## 📦 模型规模

### 参数量对比

| 模型配置 | 层数 | 隐藏维度 | 注意力头数 | 估算参数量 |
|----------|------|----------|-----------|-----------|
| **Small** | 8 | 512 | 16 | ~26M |
| **Medium** | 16 | 768 | 16 | ~108M |
| **MoE** | 8 (4 专家) | 512 | 16 | ~145M |

### 内存需求

| 模型 | FP32 内存 | FP16 内存 | 训练显存 (估算) | 推理显存 (估算) |
|------|----------|----------|---------------|---------------|
| Small | 104MB | 52MB | 2-4GB | 0.5-1GB |
| Medium | 432MB | 216MB | 8-12GB | 2-3GB |
| MoE | 580MB | 290MB | 10-16GB | 3-4GB |

## 🎯 功能还原对照

与原版 MiniMind 的功能对照:

| 功能模块 | 原版 MiniMind | TinyAI 实现 | 还原度 |
|---------|---------------|------------|-------|
| Tokenizer (BPE) | ✓ | ✅ 已完成 | 100% |
| 模型架构 (Transformer Decoder-Only) | ✓ | ✅ 已完成 | 100% |
| RoPE 位置编码 | ✓ | ✅ 已完成 | 100% |
| 多头注意力 | ✓ | ✅ 已完成 | 100% |
| KV-Cache | ✓ | ✅ 已完成 | 100% |
| 预训练 | ✓ | ✅ 已完成 | 100% |
| SFT 微调 | ✓ | ✅ 已完成 | 100% |
| LoRA 微调 | ✓ | ✅ 已完成 | 100% |
| DPO 训练 | ✓ | ✅ 已完成 | 100% |
| RLAIF (PPO/GRPO/SPO) | ✓ | ✅ 已完成 | 100% |
| MoE 架构 | ✓ | ✅ 已完成 | 100% |
| 文本生成 (多种采样) | ✓ | ✅ 已完成 | 100% |
| CLI 工具 | ✓ | ✅ 已完成 | 100% |
| API 服务 | ✓ | ✅ 已完成 | 100% |

## 📖 V2 组件使用规范

### 必须使用的 V2 组件

```java
// 基础模块

// 容器

// 线性层

// 激活函数

// 归一化
import io.leavesfly.tinyai.nnet.v2.layer.norm.Dropout;
```

### 禁止使用的 V1 组件

```java
// ❌ 禁止使用
import io.leavesfly.tinyai.nnet.layer.dnn.AffineLayer;
```

### 自行实现的组件

| 组件 | 原因 | 继承关系 |
|------|------|---------|
| `MultiHeadAttention` | V2 中无 RoPE + KV-Cache 支持 | 继承 `Module` |
| `TokenEmbedding` | V2 中无嵌入查找层 | 继承 `Module` |
| `RotaryPositionEmbedding` | V2 中无 RoPE 实现 | 继承 `Module` |
| `MoELayer` | V2 中无 MoE 支持 | 继承 `Module` |
| `MiniMindTokenizer` | 独立工具类 | 纯 Java 类 |

## 📝 开发状态

**当前版本**: 1.0-SNAPSHOT

**核心功能** (100% 完成):
- ✅ 模块基础结构搭建
- ✅ Maven 配置和依赖管理
- ✅ MiniMindConfig 配置类 (三种预设)
- ✅ TokenEmbedding 嵌入层
- ✅ RotaryPositionEmbedding (RoPE)
- ✅ MultiHeadAttention 多头注意力
- ✅ KVCache 缓存管理
- ✅ MiniMindTransformerLayer Transformer 层
- ✅ MiniMindBlock / MiniMindModel 模型主体
- ✅ MiniMindTokenizer 分词器 (字符级 + BPE 训练)
- ✅ 推理引擎 (多种采样策略)
- ✅ MoE 完整架构实现
- ✅ MiniMindMoEModel MoE 模型
- ✅ 专家路由和负载均衡

**训练组件** (100% 完成):
- ✅ PretrainTrainer 预训练训练器
- ✅ SFTTrainer 监督微调训练器
- ✅ LoRAAdapter LoRA 适配器
- ✅ DPOTrainer DPO 训练器
- ✅ PPO/GRPO/SPO 强化学习训练器
- ✅ MiniMindTrainDemo 完整训练演示

**工具与服务** (100% 完成):
- ✅ CLI 命令行工具 (train/generate/chat 等)
- ✅ API 服务 (OpenAI 兼容)
- ✅ 数据处理工具

**当前进度**: 100%

**持续优化**:
- 🔄 性能优化和调优
- 🔄 更多单元测试
- 🔄 文档完善

## ❓ 常见问题 FAQ

### Q1: 编译失败，提示找不到依赖？

**A:** 确保先编译 TinyAI 核心模块:

```bash
cd TinyAI
mvn clean install -DskipTests

cd tinyai-model/tinyai-model-minimind
mvn compile
```

### Q2: 运行时内存不足？

**A:** 增加 JVM 堆内存:

```bash
export MAVEN_OPTS="-Xmx4g"
# 或运行时指定
java -Xmx4g -cp ... YourClass
```

### Q3: 生成结果是乱码？

**A:** 检查以下几点:
1. Tokenizer 是否正确初始化
2. 词汇表大小是否匹配
3. 模型是否处于推理模式 (`model.setTraining(false)`)

### Q4: 如何评估模型效果？

**A:** 使用困惑度 (Perplexity):

```java
float perplexity = evaluator.calculatePerplexity(model, validDataset);
System.out.println("困惑度：" + perplexity);
```

### Q5: 如何部署 API 服务？

**A:** 使用内置的 API 服务器:

```bash
# 启动 API 服务
./bin/start-api.sh 8080

# 测试
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minimind",
    "messages": [{"role": "user", "content": "你好!"}],
    "max_tokens": 100
  }'
```

详见：[API 服务指南](./doc/API-GUIDE.md)

---

## 💡 最佳实践

### 训练建议

**学习率设置**:
- 预训练：`3e-4`
- SFT: `5e-5`
- LoRA: `1e-4`
- DPO: `5e-6`

**梯度裁剪**:
```java
config.setMaxGradNorm(1.0f);  // 防止梯度爆炸
```

**Warmup 策略**:
```
lr = base_lr × min(step / warmup_steps, 1.0)
```

### 推理建议

**采样参数推荐**:
- 对话：`temperature=0.7, top_p=0.9`
- 代码：`temperature=0.2, top_p=0.95`
- 创意：`temperature=0.9, top_p=0.95`

**批量大小**:
```
max_batch_size = GPU_memory / (model_size + seq_len × hidden_size)
```

**KV-Cache 管理**: 长对话时定期清理旧缓存

---

## 👥 贡献指南

欢迎贡献！请遵循以下规范:

1. **代码规范**: 遵循 TinyAI 项目规范
2. **V2 优先**: 强制使用 `nnet.v2.*` 组件
3. **功能还原**: 确保与原版 MiniMind 功能一致
4. **测试覆盖**: 新功能需要完整的单元测试
5. **文档更新**: 重要修改需要更新文档

---

## 📚 更多资源

### 技术文档

- [MiniMind 模型设计](./doc/MiniMind 模型设计.md) - 架构深入解析
- [API 参考文档](./doc/API 参考.md) - 完整的 API 说明
- [CLI 工具指南](./doc/CLI-GUIDE.md) - 命令行工具使用
- [API 服务指南](./doc/API-GUIDE.md) - REST API 部署
- [快速开始指南](./doc/快速开始指南.md) - 详细上手教程

### 示例代码

位于 `src/main/java/io/leavesfly/tinyai/minimind/examples/`:
- `Example01_ModelCreationAndInference.java` - 模型创建与推理
- `Example02_BPETokenizerTraining.java` - BPE 分词器训练
- `Example03_SFTFineTuning.java` - SFT 微调
- `Example04_LoRAFineTuning.java` - LoRA 微调
- `Example05_PreTraining.java` - 预训练
- `Example06_TextGenerationStrategies.java` - 文本生成策略
- `Example07_ModelEvaluation.java` - 模型评估
- `Example08_CompleteTrainingPipeline.java` - 完整训练流程
- `MiniMindMoEExample.java` - MoE 模型使用
- `Example09_APIServer.java` - API 服务部署
- `Example10_CLITools.java` - CLI 工具使用

### 完整训练演示

`MiniMindTrainDemo` 提供了完整的三阶段训练流程:
```bash
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.minimind.training.MiniMindTrainDemo"
```

详见：[MiniMindTrainDemo 使用说明](./doc/MiniMindTrainDemo 使用说明.md)

### 参考链接

- **原版 MiniMind**: https://github.com/jingyaogong/minimind
- **TinyAI 框架**: https://github.com/leavesfly/TinyAI
- **Transformer 论文**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- **RoPE 论文**: [RoFormer](https://arxiv.org/abs/2104.09864)
- **LoRA 论文**: [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- **DPO 论文**: [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)

---

## 📄 许可证

本项目遵循 TinyAI 框架的开源许可证。

---

**版本**: 1.0-SNAPSHOT  
**当前进度**: 100% (核心功能完成)  
**最后更新**: 2025-12-16  
**维护者**: TinyAI Team

---

*基于 TinyAI 框架的轻量级语言模型 - 教育友好 · 资源高效 · 功能完整*
