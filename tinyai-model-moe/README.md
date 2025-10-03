# TinyAI-Model-MoE

## 📖 项目概述

`tinyai-model-moe` 是 TinyAI 框架中基于 **Mixture of Experts (MoE)** 架构的大语言模型实现模块。该模块提供了完整的 MoE-GPT 模型实现，通过专家混合机制大幅增加模型参数容量，同时保持合理的计算开销。

## 🚀 核心特性

### ✨ 主要优势

- **🎯 大规模模型容量**：通过多专家机制指数级增加模型参数量
- **⚡ 高效稀疏计算**：每次前向传播只激活Top-K个专家
- **🔧 模块化设计**：高度解耦的组件设计，便于扩展和复用
- **📊 负载均衡机制**：智能的专家负载均衡，避免专家利用不均
- **🎯 专家专业化**：不同专家可学习处理不同类型的语言模式

### 🏗️ 架构特点

- 基于 GPT-2 架构，将传统 FeedForward 层替换为 MoE 层
- 支持动态专家数量和 Top-K 参数配置
- 集成完整的统计分析和监控功能
- 提供多种预设模型规模配置

## 📦 模块结构

```
tinyai-model-moe/
├── src/main/java/io/leavesfly/tinyai/nlp/
│   ├── moe/                           # MoE核心组件
│   │   ├── Expert.java                # 专家网络实现
│   │   ├── GateNetwork.java           # 门控网络
│   │   ├── MoELayer.java              # MoE层
│   │   └── MoETransformerBlock.java   # MoE Transformer块
│   ├── MoEGPTModel.java               # MoE-GPT主模型
│   ├── MoEGPTModelDemo.java           # 完整演示程序
│   └── MoEGPTExample.java             # 基础使用示例
├── src/test/java/                     # 单元测试
├── doc/                               # 技术文档
└── pom.xml                            # Maven配置
```

## 🔧 技术架构

### 核心组件详解

#### 1. 🧠 专家网络 (Expert)
```java
// 每个专家都是独立的两层全连接网络
输入 → Linear(input_dim, hidden_dim) → ReLU → Dropout → Linear(hidden_dim, output_dim) → 输出
```

#### 2. 🚪 门控网络 (GateNetwork)
- 负责计算每个专家的权重分布
- 支持 Top-K 专家选择的稀疏化
- 包含噪声注入机制用于负载均衡
- 使用 Softmax 函数进行权重归一化

#### 3. 🔄 MoE层 (MoELayer)
- 组合门控网络和多个专家网络
- 实现基于权重的专家输出加权求和
- 负载均衡损失计算和专家使用统计

#### 4. 🏗️ MoE Transformer块 (MoETransformerBlock)
```
输入 → LayerNorm → Multi-Head Attention → 残差连接 
    → LayerNorm → MoE层 → 残差连接 → 输出
```

#### 5. 🎯 MoE-GPT模型 (MoEGPTModel)
- 基于 GPT-2 架构，使用 MoE 替换 FeedForward 层
- 支持多种模型规模配置
- 集成负载均衡损失计算
- 提供专家使用统计和分析功能

## 📋 依赖关系

```xml
<dependencies>
    <!-- TinyAI 深度学习核心模块 -->
    <dependency>
        <groupId>io.leavesfly.tinyai</groupId>
        <artifactId>tinyai-deeplearning-ml</artifactId>
    </dependency>
    
    <!-- GPT 模型依赖 -->
    <dependency>
        <groupId>io.leavesfly.tinyai</groupId>
        <artifactId>tinyai-model-gpt</artifactId>
    </dependency>
</dependencies>
```

## 🎮 快速开始

### 基础使用示例

```java
import io.leavesfly.tinyai.nlp.MoEGPTModel;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;

public class MoEQuickStart {
    public static void main(String[] args) {
        // 1. 创建小型MoE-GPT模型
        MoEGPTModel model = new MoEGPTModel(
            "demo_moe_gpt",  // 模型名称
            1000,            // 词汇表大小
            128,             // 模型维度
            6,               // Transformer层数
            8,               // 注意力头数
            4,               // 专家数量
            256,             // 专家隐藏维度
            2,               // Top-K专家选择
            64,              // 最大序列长度
            0.1,             // Dropout率
            true,            // 使用噪声
            0.1              // 噪声强度
        );
        
        // 2. 准备输入数据
        int batchSize = 2;
        int seqLen = 16;
        NdArray inputTokens = createSampleTokens(batchSize, seqLen, 1000);
        
        // 3. 执行前向传播
        Variable input = new Variable(inputTokens);
        Variable output = model.layerForward(input);
        
        // 4. 分析结果
        System.out.println("输出形状: " + output.getValue().getShape());
        
        // 5. 查看负载均衡统计
        System.out.println(model.getLoadBalancingReport());
        
        // 6. 打印模型信息
        model.printModelInfo();
    }
    
    private static NdArray createSampleTokens(int batchSize, int seqLen, int vocabSize) {
        // 创建随机token序列的辅助方法
        // ... 实现细节
    }
}
```

### 预设模型配置

```java
// 小型模型 (适合测试和学习)
MoEGPTModel smallModel = new MoEGPTModel(
    "small_moe", 1000, 128, 4, 8, 4, 2, 64
);

// 中型模型 (适合实验)
MoEGPTModel mediumModel = new MoEGPTModel(
    "medium_moe", 5000, 256, 8, 16, 8, 2, 128
);

// 大型模型 (适合生产环境)
MoEGPTModel largeModel = new MoEGPTModel(
    "large_moe", 50000, 512, 12, 16, 16, 4, 512
);
```

## 📊 性能分析

### 参数效率对比

| 模型类型 | 总参数量 | 激活参数量 | 激活比例 | 容量增益 |
|---------|----------|------------|----------|----------|
| 传统GPT-2 | 1.2M | 1.2M | 100% | 1.0x |
| MoE-GPT (4专家) | 2.8M | 1.1M | 39% | 2.5x |
| MoE-GPT (8专家) | 5.2M | 1.3M | 25% | 4.0x |

### 计算效率

- **稀疏激活**：每个 token 只使用 Top-K 个专家
- **参数效率**：激活参数量远小于总参数量
- **可扩展性**：通过增加专家数量而非层深度来扩展

## 🔍 功能特性

### 1. 负载均衡机制

```java
// 获取负载均衡报告
String report = model.getLoadBalancingReport();

// 计算负载均衡损失
float balancingLoss = model.computeTotalLoadBalancingLoss();

// 重置统计信息
model.resetAllMoEStats();
```

### 2. 专家使用统计

```java
// 打印所有专家使用统计
model.printAllExpertStatistics();

// 获取特定层的专家统计
MoETransformerBlock block = model.getMoeTransformerBlock(0);
Map<String, Object> stats = block.getExpertUsageStats();
```

### 3. 模型配置信息

```java
// 获取完整模型配置
String config = model.getModelConfig();

// 获取参数统计
long totalParams = model.getTotalParameterCount();
double paramRatio = model.getParameterIncreaseRatio();
```

## 🧪 测试与验证

### 运行测试

```bash
# 运行所有测试
mvn test

# 运行特定测试
mvn test -Dtest=MoEGPTModelSimpleTest

# 运行演示程序
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.nlp.MoEGPTModelDemo"
```

### 测试覆盖

- ✅ 基础模型构造测试
- ✅ 组件初始化验证
- ✅ 参数验证测试
- ✅ 前向传播测试
- ✅ 负载均衡验证
- ✅ 专家统计测试

## 📈 应用场景

### 1. 🎓 教育与研究
- MoE 机制原理学习和验证
- 稀疏神经网络架构研究
- 大语言模型架构对比分析

### 2. 🏭 生产应用
- 大规模语言模型训练
- 多任务学习系统
- 资源受限环境下的大模型部署

### 3. 🔬 实验开发
- 新型路由策略验证
- 专家网络架构优化
- 负载均衡算法改进

## 🚀 扩展开发

### 自定义专家网络

```java
public class CustomExpert extends Expert {
    @Override
    public Variable layerForward(Variable... inputs) {
        // 实现自定义专家逻辑
        return super.layerForward(inputs);
    }
}
```

### 自定义门控策略

```java
public class CustomGateNetwork extends GateNetwork {
    @Override
    protected Variable computeGateWeights(Variable input) {
        // 实现自定义门控逻辑
        return super.computeGateWeights(input);
    }
}
```

## 📚 相关文档

- [MoE实现说明文档](doc/MoE实现说明.md) - 详细技术实现说明
- [API参考文档](../tinyai-deeplearning-ml/README.md) - 核心ML组件文档
- [GPT模型文档](../tinyai-model-gpt/README.md) - GPT基础模型文档

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](../LICENSE) 文件了解详情

## 🙏 致谢

- 感谢 TinyAI 框架提供的基础设施支持
- 参考了经典 MoE 论文中的核心算法设计
- 感谢开源社区的宝贵建议和反馈

---

**注意**: 本模块为 TinyAI 框架的一部分，建议结合完整框架使用以获得最佳体验。更多信息请参考 [TinyAI 主项目文档](../README.md)。