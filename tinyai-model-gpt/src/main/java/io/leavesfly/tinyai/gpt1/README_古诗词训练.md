# GPT-1古诗词训练指南

本指南详细介绍如何使用TinyAI框架在GPT-1目录下完成古诗词语言模型的训练过程。

## 📁 文件结构

```
tinyai-model-gpt/src/main/java/io/leavesfly/tinyai/gpt1/
├── ci.txt                        # 古诗词训练数据
├── ChineseTokenizer.java         # 中文分词器
├── ChinesePoemDataSet.java       # 古诗词数据集类
├── GPT1PoemTrainer.java          # GPT-1古诗词训练器
├── GPT1PoemTrainingDemo.java     # 完整训练演示程序
├── README_古诗词训练.md           # 本文档
└── [其他GPT-1相关文件...]
```

## 🎯 训练目标

使用经典古诗词数据训练一个能够生成中文古诗的GPT-1语言模型，支持：

- 基于提示词生成古诗续写
- 不同风格和主题的诗歌创作
- 可控的文本生成（温度参数调节）
- 模型性能评估和优化

## 🚀 快速开始

### 1. 运行完整演示

直接运行演示程序体验完整的训练流程：

```bash
cd tinyai-model-gpt
java -cp target/classes io.leavesfly.tinyai.gpt1.GPT1PoemTrainingDemo
```

### 2. 自定义训练

使用训练器类进行自定义训练：

```java
// 创建训练配置
GPT1PoemTrainer.TrainingConfig config = new GPT1PoemTrainer.TrainingConfig();
config.epochs = 30;
config.batchSize = 4;
config.learningRate = 0.001;

// 创建训练器并训练
GPT1PoemTrainer trainer = new GPT1PoemTrainer(config);
trainer.initialize("src/main/java/io/leavesfly/tinyai/gpt1/ci.txt");
trainer.train();

// 生成古诗
String poem = trainer.generateText("春江", 15, 0.8);
System.out.println("生成的诗句: " + poem);
```

## 📊 训练数据

### 数据格式

`ci.txt` 文件包含经典古诗词，格式如下：

```
春江花月夜
春江潮水连海平，海上明月共潮生。
滟滟随波千万里，何处春江无月明！
江流宛转绕芳甸，月照花林皆似霰。

静夜思
床前明月光，疑是地上霜。
举头望明月，低头思故乡。
```

### 数据预处理

1. **分词策略**: 采用字符级分词，适合中文古诗词的特点
2. **词汇表构建**: 自动提取所有字符构建词汇表
3. **序列处理**: 添加特殊token（BOS/EOS/PAD/UNK）
4. **数据增强**: 创建输入-目标对用于语言模型训练

### 数据统计

训练完成后会显示数据统计信息：
- 总词汇数
- 训练序列数量
- 平均序列长度
- 长度分布情况

## 🧠 模型架构

### GPT-1配置

| 参数 | 演示配置 | 完整配置 | 说明 |
|------|----------|----------|------|
| 词汇表大小 | 自适应 | 自适应 | 根据数据自动确定 |
| 最大序列长度 | 32 | 64 | 输入序列最大长度 |
| 隐藏层维度 | 128 | 256 | Transformer隐藏层大小 |
| 层数 | 4 | 8 | Transformer层数 |
| 注意力头数 | 4 | 8 | 多头注意力头数 |

### 模型组件

1. **Token嵌入层**: 将字符转换为向量表示
2. **位置编码**: 为序列位置添加编码信息
3. **Transformer层**: 多层自注意力机制
4. **输出层**: 预测下一个字符的概率分布

## ⚙️ 训练配置

### 基础参数

```java
TrainingConfig config = new TrainingConfig();

// 训练参数
config.epochs = 20;              // 训练轮数
config.batchSize = 4;            // 批量大小
config.learningRate = 0.001;     // 学习率
config.validationRatio = 0.2;    // 验证集比例

// 生成参数
config.generateLength = 15;      // 生成文本长度
config.temperature = 0.8;        // 生成随机性控制
```

### 高级配置

- **优化器**: Adam (β1=0.9, β2=0.999, ε=1e-4)
- **损失函数**: SoftmaxCrossEntropy
- **正则化**: Dropout (可配置)
- **学习率调度**: 固定学习率（可扩展）

## 📈 训练过程

### 训练流程

1. **数据加载**: 读取古诗词文件并分词
2. **词汇表构建**: 创建字符到ID的映射
3. **数据分割**: 分为训练集和验证集
4. **模型初始化**: 创建GPT-1模型和优化器
5. **训练循环**: 
   - 前向传播计算损失
   - 反向传播更新参数
   - 定期验证和生成样本
6. **模型保存**: 保存训练好的模型

### 训练监控

训练过程中会显示：
- 训练损失和验证损失
- 生成的样本文本
- 训练进度和时间消耗
- 模型参数统计

## 🎨 文本生成

### 生成示例

```java
// 不同提示词生成
trainer.generateText("春", 12, 0.8);    // 春天主题
trainer.generateText("月", 12, 0.8);    // 月亮主题
trainer.generateText("山水", 15, 0.8);  // 山水主题

// 不同温度参数
trainer.generateText("春江", 15, 0.3);  // 更确定性
trainer.generateText("春江", 15, 1.2);  // 更随机性
```

### 温度参数说明

- **0.1-0.5**: 生成较为确定和保守的文本
- **0.6-0.8**: 平衡创新性和合理性（推荐）
- **0.9-1.2**: 更具创造性但可能不够连贯
- **>1.5**: 高度随机，可能产生不合理文本

## 📊 模型评估

### 评估指标

1. **训练损失**: 模型在训练数据上的拟合程度
2. **验证损失**: 模型的泛化能力
3. **困惑度 (Perplexity)**: 语言模型质量指标
4. **生成质量**: 主观评估生成文本的合理性

### 性能优化建议

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 训练损失高 | 模型容量不足 | 增加隐藏层维度或层数 |
| 验证损失高 | 过拟合 | 增加Dropout或减少模型复杂度 |
| 生成质量差 | 训练不充分 | 增加训练轮数或调整学习率 |
| 生成重复 | 温度参数低 | 适当提高温度参数 |

## 🔧 故障排除

### 常见问题

1. **内存不足**
   - 减少批量大小
   - 降低序列长度
   - 减少模型参数

2. **训练速度慢**
   - 使用较小的模型配置
   - 减少训练数据量
   - 优化批量大小

3. **生成质量不佳**
   - 增加训练轮数
   - 改进训练数据质量
   - 调整模型架构参数

### 调试建议

- 检查训练损失是否持续下降
- 观察验证损失避免过拟合
- 定期查看生成样本质量
- 监控模型参数梯度

## 🔮 扩展功能

### 可能的改进方向

1. **数据增强**:
   - 添加更多古诗词数据
   - 数据清洗和质量提升
   - 多样化的文本预处理

2. **模型优化**:
   - 实现学习率调度
   - 添加更多正则化技术
   - 优化注意力机制

3. **生成控制**:
   - 主题控制生成
   - 长度精确控制
   - 风格迁移

4. **评估完善**:
   - 自动化质量评估
   - 多维度性能指标
   - 人工评估框架

## 📝 使用示例

### 完整训练示例

```java
public class PoemTrainingExample {
    public static void main(String[] args) throws Exception {
        // 1. 创建配置
        TrainingConfig config = new TrainingConfig();
        config.epochs = 25;
        config.batchSize = 4;
        config.learningRate = 0.0008;
        config.maxSequenceLength = 32;
        config.hiddenSize = 128;
        
        // 2. 创建训练器
        GPT1PoemTrainer trainer = new GPT1PoemTrainer(config);
        
        // 3. 初始化和训练
        trainer.initialize("ci.txt");
        trainer.train();
        
        // 4. 测试生成
        System.out.println("春江 -> " + trainer.generateText("春江", 16, 0.8));
        System.out.println("明月 -> " + trainer.generateText("明月", 16, 0.8));
        
        // 5. 评估困惑度
        double ppl = trainer.evaluatePerplexity();
        System.out.println("困惑度: " + ppl);
    }
}
```

## 📖 参考资料

- GPT-1论文: "Improving Language Understanding by Generative Pre-Training"
- TinyAI框架文档
- Transformer架构详解
- 中文自然语言处理最佳实践

---

**注意**: 这是一个教育性质的演示项目，实际的生产级语言模型需要更大规模的数据、更复杂的架构和更多的计算资源。本项目旨在帮助理解GPT模型的基本原理和训练流程。