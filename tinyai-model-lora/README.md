# TinyAI LoRA 模块

> 高效参数微调技术的完整实现 - 基于TinyAI深度学习框架

![Java](https://img.shields.io/badge/Java-17+-brightgreen.svg)
![Maven](https://img.shields.io/badge/Maven-3.6+-green.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

## 📋 概述

TinyAI LoRA模块实现了完整的LoRA（Low-Rank Adaptation）微调技术，这是一种参数高效的微调方法，能够在保持模型性能的同时大幅减少可训练参数数量。本模块基于TinyAI深度学习框架构建，提供了从配置管理到模型构建的全栈实现。

### 🎯 核心优势

- **参数高效**: 可实现95%+的参数减少，显著降低内存占用
- **性能保持**: 在减少参数的同时保持原有模型性能
- **灵活控制**: 支持动态启用/禁用LoRA适配器
- **易于集成**: 与TinyAI框架无缝集成，API简洁易用
- **权重管理**: 支持权重冻结、合并等高级功能

## 🔬 技术原理

LoRA通过低秩矩阵分解技术将原始权重矩阵分解为：

```
W' = W + ΔW = W + A × B × scaling
```

其中：
- **W**: 冻结的预训练权重矩阵 (d × k)
- **A**: 可训练的下降矩阵 (d × r)  
- **B**: 可训练的上升矩阵 (r × k)
- **r**: 低秩值，通常 r ≪ min(d, k)
- **scaling**: 缩放因子 = α / r

## 📦 模块结构

```
tinyai-model-lora/
├── src/main/java/io/leavesfly/tinyai/lora/
│   ├── LoraConfig.java           # LoRA配置管理
│   ├── LoraAdapter.java          # 核心适配器实现
│   ├── LoraLinearLayer.java      # 集成LoRA的线性层
│   ├── LoraModel.java           # 完整LoRA模型
│   └── LoraDemo.java            # 使用示例
├── src/test/java/io/leavesfly/tinyai/lora/
│   ├── LoraConfigTest.java       # 配置类测试
│   ├── LoraAdapterTest.java      # 适配器测试
│   ├── LoraLinearLayerTest.java  # 线性层测试
│   └── LoraModelTest.java       # 模型测试
├── doc/
│   └── LoRA实现说明.md          # 详细技术文档
└── pom.xml                      # Maven配置
```

## 🧩 核心组件

### 1. LoraConfig - 配置管理

管理LoRA的所有超参数和配置选项：

```java
// 预设配置
LoraConfig config = LoraConfig.createMediumRank(); // rank=16, alpha=32

// 自定义配置
LoraConfig customConfig = new LoraConfig(
    8,      // rank: 低秩矩阵的秩
    16.0,   // alpha: 缩放参数
    0.1,    // dropout: 正则化参数
    true,   // enableBias: 是否启用偏置微调
    new String[]{"linear"} // targetModules: 目标模块类型
);
```

**预设配置选项**：
- `createLowRank()`: rank=4, alpha=8 (快速实验)
- `createMediumRank()`: rank=16, alpha=32 (常规微调)
- `createHighRank()`: rank=64, alpha=128 (复杂任务)

### 2. LoraAdapter - 核心适配器

实现低秩矩阵分解的核心组件：

```java
// 创建适配器
LoraAdapter adapter = new LoraAdapter(inputDim, outputDim, config);

// 前向传播
Variable output = adapter.forward(input);

// 控制启用状态
adapter.enable();   // 启用LoRA
adapter.disable();  // 禁用LoRA

// 参数统计
int paramCount = adapter.getParameterCount();
double reduction = adapter.getParameterReduction(originalParamCount);
```

### 3. LoraLinearLayer - LoRA线性层

集成了LoRA适配器的完整线性层：

```java
// 创建LoRA线性层
LoraLinearLayer layer = new LoraLinearLayer(
    "lora_layer",    // 层名称
    512,             // 输入维度
    256,             // 输出维度
    config,          // LoRA配置
    true             // 是否包含偏置
);

// 前向传播
Variable output = layer.layerForward(input);

// 权重管理
layer.freezeOriginalWeights();   // 冻结原始权重
layer.enableLora();              // 启用LoRA适配器
```

### 4. LoraModel - 完整LoRA模型

提供完整的多层LoRA模型实现：

```java
// 创建LoRA模型
int[] layerSizes = {784, 256, 128, 10}; // 网络架构
LoraModel model = new LoraModel("classifier", layerSizes, config, false);

// 批量操作
model.freezeAllOriginalWeights(); // 冻结所有原始权重
model.enableAllLora();            // 启用所有LoRA适配器

// 权重合并
List<NdArray> mergedWeights = model.mergeAllLoraWeights();
```

## 🚀 快速开始

### 1. 基础使用示例

```java
import io.leavesfly.tinyai.lora.*;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

// 创建LoRA配置
LoraConfig config = LoraConfig.createMediumRank();

// 创建LoRA线性层
LoraLinearLayer layer = new LoraLinearLayer(
    "demo_layer", 256, 128, config, true);

// 准备输入数据
NdArray inputData = NdArray.likeRandomN(Shape.of(32, 256)); // batch_size=32
Variable input = new Variable(inputData);

// 前向传播
Variable output = layer.layerForward(input);

System.out.println("输入形状: " + input.getValue().getShape());
System.out.println("输出形状: " + output.getValue().getShape());
System.out.println("参数减少率: " + layer.getParameterReduction() + "%");
```

### 2. 完整模型微调

```java
// 定义网络架构 (MNIST分类器)
int[] layerSizes = {784, 512, 256, 10};
LoraConfig config = LoraConfig.createMediumRank();

// 创建LoRA模型
LoraModel model = new LoraModel("mnist_classifier", layerSizes, config, false);

// 设置训练模式：只训练LoRA参数
model.freezeAllOriginalWeights();
model.enableAllLora();

// 模型信息
System.out.println("模型信息:");
System.out.println(model.getModelInfo());
System.out.println("可训练参数: " + model.getTrainableParameterCount());

// 测试前向传播
NdArray testInput = NdArray.likeRandomN(Shape.of(64, 784));
Variable output = model.layerForward(new Variable(testInput));
```

### 3. 预训练模型适配

```java
// 假设有预训练权重
List<NdArray> pretrainedWeights = loadPretrainedWeights();
List<NdArray> pretrainedBiases = loadPretrainedBiases();

// 从预训练权重创建LoRA模型
LoraModel fineTunedModel = LoraModel.fromPretrained(
    "finetuned_model", 
    pretrainedWeights, 
    pretrainedBiases, 
    config, 
    false
);

// 微调完成后合并权重
List<NdArray> finalWeights = fineTunedModel.mergeAllLoraWeights();
```

## 📊 性能分析

### 参数效率对比

以1024×1024的全连接层为例：

| Rank | LoRA参数 | 原始参数 | 参数减少率 | 计算开销比 |
|------|----------|----------|------------|------------|
| 4    | 8,192    | 1,048,576| 99.22%     | 0.0078x    |
| 8    | 16,384   | 1,048,576| 98.44%     | 0.0156x    |
| 16   | 32,768   | 1,048,576| 96.88%     | 0.0312x    |
| 32   | 65,536   | 1,048,576| 93.75%     | 0.0625x    |
| 64   | 131,072  | 1,048,576| 87.50%     | 0.1250x    |

### 内存使用对比

```java
// 获取参数统计信息
public void analyzeParameters() {
    LoraConfig config = LoraConfig.createMediumRank();
    LoraLinearLayer layer = new LoraLinearLayer("test", 1024, 1024, config, true);
    
    System.out.println("原始参数数量: " + (1024 * 1024));
    System.out.println("LoRA参数数量: " + layer.getLoraParameterCount());
    System.out.println("参数减少率: " + layer.getParameterReduction() + "%");
    System.out.println("内存节省: " + layer.getMemorySaving() + " MB");
}
```

## 🔧 高级功能

### 1. 动态LoRA控制

```java
// 训练时启用LoRA
model.enableAllLora();
Variable trainOutput = model.layerForward(trainInput);

// 推理时可选择禁用LoRA使用原始权重
model.disableAllLora();
Variable inferOutput = model.layerForward(testInput);

// 比较两种模式的输出差异
NdArray diff = trainOutput.getValue().sub(inferOutput.getValue());
```

### 2. 权重状态管理

```java
// 保存LoRA状态
Map<String, NdArray> loraState = model.saveLoraState();

// 恢复LoRA状态
model.loadLoraState(loraState);

// 合并权重（用于部署）
List<NdArray> mergedWeights = model.mergeAllLoraWeights();
```

### 3. 批量参数操作

```java
// 批量冻结/解冻
model.freezeAllOriginalWeights();   // 冻结所有原始权重
model.unfreezeAllOriginalWeights(); // 解冻所有原始权重

// 批量启用/禁用LoRA
model.enableAllLora();              // 启用所有LoRA适配器
model.disableAllLora();             // 禁用所有LoRA适配器
```

## 🧪 测试与验证

运行完整的测试套件：

```bash
cd /Users/yefei.yf/Qoder/TinyAI
export JAVA_HOME=/Library/Java/JavaVirtualMachines/jdk-17.jdk/Contents/Home
mvn test -pl tinyai-model-lora
```

运行演示程序：

```bash
mvn exec:java -pl tinyai-model-lora -Dexec.mainClass="io.leavesfly.tinyai.lora.LoraDemo"
```

### 测试覆盖

- ✅ **配置验证**: 参数范围和合理性检查
- ✅ **适配器功能**: 前向传播和参数统计
- ✅ **线性层集成**: LoRA与标准线性层的集成
- ✅ **模型构建**: 多层LoRA模型的完整功能
- ✅ **权重管理**: 冻结、合并、状态保存等功能
- ✅ **性能验证**: 参数减少率和计算效率

## 📚 依赖关系

本模块依赖以下TinyAI组件：

```xml
<dependencies>
    <!-- 机器学习核心模块 -->
    <dependency>
        <groupId>io.leavesfly.tinyai</groupId>
        <artifactId>tinyai-deeplearning-ml</artifactId>
    </dependency>
    
    <!-- 强化学习模块 -->
    <dependency>
        <groupId>io.leavesfly.tinyai</groupId>
        <artifactId>tinyai-deeplearning-rl</artifactId>
    </dependency>
    
    <!-- GPT模型模块 -->
    <dependency>
        <groupId>io.leavesfly.tinyai</groupId>
        <artifactId>tinyai-model-gpt</artifactId>
    </dependency>
</dependencies>
```

## 🛠️ 编译与安装

### 前置步骤

确保按依赖顺序编译相关模块：

```bash
cd /Users/yefei.yf/Qoder/TinyAI
export JAVA_HOME=/Library/Java/JavaVirtualMachines/jdk-17.jdk/Contents/Home

# 按顺序编译依赖模块
mvn install -DskipTests -pl tinyai-deeplearning-ndarr
mvn install -DskipTests -pl tinyai-deeplearning-func  
mvn install -DskipTests -pl tinyai-deeplearning-nnet
mvn install -DskipTests -pl tinyai-deeplearning-ml
mvn install -DskipTests -pl tinyai-deeplearning-rl
mvn install -DskipTests -pl tinyai-model-gpt
```

### 编译本模块

```bash
# 编译LoRA模块
mvn clean compile -pl tinyai-model-lora

# 运行测试
mvn test -pl tinyai-model-lora

# 安装到本地仓库
mvn install -pl tinyai-model-lora
```

## 📖 最佳实践

### 1. 选择合适的Rank值

```java
// 快速实验或资源受限环境
LoraConfig lowConfig = LoraConfig.createLowRank(); // rank=4

// 大多数实际应用
LoraConfig mediumConfig = LoraConfig.createMediumRank(); // rank=16

// 复杂任务或对性能要求严格
LoraConfig highConfig = LoraConfig.createHighRank(); // rank=64
```

### 2. 微调流程建议

```java
// 1. 创建模型并冻结原始权重
model.freezeAllOriginalWeights();
model.enableAllLora();

// 2. 训练LoRA参数
// ... 训练循环 ...

// 3. 评估时可尝试禁用LoRA对比
model.disableAllLora();
// ... 评估 ...

// 4. 部署时合并权重以提高推理效率
List<NdArray> finalWeights = model.mergeAllLoraWeights();
```

### 3. 内存优化

```java
// 对于大模型，可以动态控制LoRA状态以节省内存
if (isTraining) {
    model.enableAllLora();
} else {
    model.disableAllLora(); // 推理时禁用可节省内存
}
```

## 🔮 未来扩展

### 计划特性

- [ ] **动态Rank调整**: 根据任务复杂度自动调整Rank值
- [ ] **多任务LoRA**: 支持一个模型同时适配多个任务
- [ ] **量化LoRA**: 结合量化技术进一步减少内存占用
- [ ] **自适应缩放**: 智能调整缩放因子α
- [ ] **LoRA融合**: 支持多个LoRA适配器的融合

### 扩展方向

- **卷积LoRA**: 扩展到卷积层的LoRA适配
- **注意力LoRA**: 专门针对Transformer注意力机制的LoRA
- **稀疏LoRA**: 结合稀疏性进一步提高效率
- **分层LoRA**: 不同层使用不同的LoRA配置

## 📄 许可证

本项目基于 [MIT License](../LICENSE) 开源。

## 🤝 贡献指南

欢迎提交Issue和Pull Request！在贡献代码前，请确保：

1. 遵循现有的代码风格和命名规范
2. 添加适当的中文注释
3. 编写相应的单元测试
4. 更新相关文档

## 📞 技术支持

如有技术问题或建议，请通过以下方式联系：

- 提交 [GitHub Issue](../../issues)
- 查看 [详细技术文档](doc/LoRA实现说明.md)
- 参考 [项目整体文档](../README.md)

---

**TinyAI LoRA模块** - 让大模型微调变得高效而简单！ 🚀