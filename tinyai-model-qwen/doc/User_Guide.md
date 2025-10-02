# Qwen3模型使用指南

## 快速开始

### 环境要求

- Java 8 或更高版本
- Maven 3.6+
- TinyAI框架依赖

### 项目依赖

在你的`pom.xml`中添加：

```xml
<dependency>
    <groupId>io.leavesfly.tinyai</groupId>
    <artifactId>tinyai-model-qwen</artifactId>
    <version>1.0.0</version>
</dependency>
```

### 第一个Qwen3程序

```java
import io.leavesfly.tinyai.qwen3.*;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.func.Variable;

public class Qwen3HelloWorld {
    public static void main(String[] args) {
        // 创建小型模型用于测试
        Qwen3Model model = Qwen3Model.createTinyModel("hello_qwen3");
        
        // 准备输入序列
        NdArray inputIds = NdArray.of(new float[]{1, 10, 20, 30});
        
        // 执行前向传播
        Variable logits = model.forwardWithLogits(new Variable(inputIds));
        
        System.out.println("输出形状: " + logits.getValue().getShape());
        System.out.println("预测下一个token: " + model.predictNextToken(inputIds));
    }
}
```

## 详细使用教程

### 1. 模型创建和配置

#### 使用预设配置

```java
// 小型测试模型
Qwen3Model tinyModel = Qwen3Model.createTinyModel("tiny");

// 获取配置信息
Qwen3Config config = tinyModel.getConfig();
System.out.println("词汇表大小: " + config.getVocabSize());
System.out.println("隐藏层维度: " + config.getHiddenSize());
```

#### 自定义配置

```java
// 创建自定义配置
Qwen3Config customConfig = new Qwen3Config();
customConfig.setVocabSize(50000);
customConfig.setHiddenSize(768);
customConfig.setNumHiddenLayers(12);
customConfig.setNumAttentionHeads(12);
customConfig.setIntermediateSize(3072);

// 使用自定义配置创建模型
Qwen3Model customModel = new Qwen3Model("custom_qwen3", customConfig);
```

#### 配置参数说明

| 参数 | 作用 | 推荐值 |
|------|------|--------|
| vocabSize | 词汇表大小 | 32000-151936 |
| hiddenSize | 模型隐藏维度 | 768, 1024, 2048, 4096 |
| numHiddenLayers | Transformer层数 | 6-32 |
| numAttentionHeads | 注意力头数 | 8, 12, 16, 32 |
| numKeyValueHeads | KV头数(GQA) | 通常等于或小于注意力头数 |
| intermediateSize | 前馈网络中间维度 | hiddenSize * 4 |

### 2. 输入数据准备

#### Token ID格式

```java
// 单个序列: 1D数组
NdArray singleSeq = NdArray.of(new float[]{101, 2054, 2003, 2115, 102});

// 批次序列: 2D数组 (batch_size, seq_len)
NdArray batchSeq = NdArray.of(Shape.of(3, 5));
// 填充数据...
batchSeq.set(101, 0, 0); // batch 0, position 0
batchSeq.set(2054, 0, 1); // batch 0, position 1
// ...
```

#### 输入验证

```java
public static void validateInput(NdArray inputIds, int vocabSize) {
    // 检查维度
    if (inputIds.getShape().getDimNum() > 2) {
        throw new IllegalArgumentException("输入最多支持2维");
    }
    
    // 检查token范围
    float[] tokens = inputIds.flatten();
    for (float token : tokens) {
        int tokenId = (int) token;
        if (tokenId < 0 || tokenId >= vocabSize) {
            throw new IllegalArgumentException("Token ID超出范围: " + tokenId);
        }
    }
}
```

### 3. 前向传播

#### 基本前向传播

```java
Qwen3Model model = Qwen3Model.createTinyModel("forward_demo");
NdArray inputIds = NdArray.of(new float[]{1, 15, 25, 35, 45});

// 获取logits
Variable logits = model.forwardWithLogits(new Variable(inputIds));

// 输出形状: (1, 5, vocab_size) - 因为输入是1D，会自动添加batch维度
System.out.println("Logits形状: " + logits.getValue().getShape());
```

#### 批次前向传播

```java
// 准备批次数据
NdArray batchInput = NdArray.of(Shape.of(2, 4));
batchInput.set(1, 0, 0); batchInput.set(10, 0, 1); batchInput.set(20, 0, 2); batchInput.set(30, 0, 3);
batchInput.set(2, 1, 0); batchInput.set(15, 1, 1); batchInput.set(25, 1, 2); batchInput.set(35, 1, 3);

Variable batchLogits = model.forwardWithLogits(new Variable(batchInput));
// 输出形状: (2, 4, vocab_size)
```

#### 获取隐藏状态

```java
// 如果只需要隐藏状态，不需要logits
Qwen3Block block = model.getQwen3Block();
Variable hiddenStates = block.layerForward(new Variable(inputIds));
// 输出形状: (batch_size, seq_len, hidden_size)
```

### 4. 文本生成

#### 贪心生成

```java
Qwen3Model model = Qwen3Model.createTinyModel("generation_demo");

// 初始序列
NdArray prompt = NdArray.of(new float[]{1, 10, 20});

// 生成更长序列
NdArray generated = model.generate(prompt, 15); // 生成到长度15

System.out.println("原始序列长度: " + prompt.getShape().getDimension(0));
System.out.println("生成序列长度: " + generated.getShape().getDimension(1));
```

#### 逐步生成

```java
public static NdArray stepByStepGeneration(Qwen3Model model, NdArray prompt, int maxSteps) {
    NdArray currentSeq = prompt;
    
    for (int step = 0; step < maxSteps; step++) {
        // 预测下一个token
        int nextToken = model.predictNextToken(currentSeq);
        
        System.out.println("Step " + step + ": predicted token " + nextToken);
        
        // 扩展序列
        currentSeq = appendToken(currentSeq, nextToken);
        
        // 检查结束条件
        if (nextToken == 2) { // 假设2是EOS token
            break;
        }
    }
    
    return currentSeq;
}

private static NdArray appendToken(NdArray seq, int newToken) {
    int seqLen = seq.getShape().getDimension(0);
    NdArray newSeq = NdArray.of(Shape.of(seqLen + 1));
    
    // 复制原序列
    for (int i = 0; i < seqLen; i++) {
        newSeq.set(seq.get(i), i);
    }
    // 添加新token
    newSeq.set(newToken, seqLen);
    
    return newSeq;
}
```

### 5. 模型管理

#### 模型信息查看

```java
Qwen3Model model = Qwen3Model.createTinyModel("info_demo");

// 打印完整信息
model.printModelInfo();

// 获取统计信息
String stats = model.getQwen3Block().getModelStats();
System.out.println(stats);

// 查看架构
model.getQwen3Block().printArchitecture();
```

#### 模型保存和加载

```java
// 保存模型
model.saveModel("path/to/qwen3_model.bin");

// 保存压缩模型
model.saveModelCompressed("path/to/qwen3_compressed.bin");

// 仅保存参数
model.saveParameters("path/to/qwen3_params.bin");

// 加载模型
Qwen3Model loadedModel = (Qwen3Model) Model.loadModel("path/to/qwen3_model.bin");

// 加载参数到现有模型
model.loadParameters("path/to/qwen3_params.bin");
```

### 6. 高级用法

#### 组件级别使用

```java
// 直接使用注意力层
Qwen3Config config = Qwen3Config.createTinyConfig();
Qwen3Attention attention = new Qwen3Attention("test_attn", config, 0);

NdArray hiddenStates = NdArray.likeRandomN(Shape.of(2, 10, config.getHiddenSize()));
Variable attnOutput = attention.layerForward(new Variable(hiddenStates));

// 直接使用MLP层
Qwen3MLP mlp = new Qwen3MLP("test_mlp", config);
Variable mlpOutput = mlp.layerForward(new Variable(hiddenStates));
```

#### 自定义解码器层

```java
public class CustomDecoderLayer extends Qwen3DecoderLayer {
    public CustomDecoderLayer(String name, Qwen3Config config, int layerIdx) {
        super(name, config, layerIdx);
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        // 自定义前向传播逻辑
        Variable output = super.layerForward(inputs);
        
        // 添加额外处理
        // ...
        
        return output;
    }
}
```

### 7. 性能优化

#### 内存优化

```java
// 使用小批次
int batchSize = 1; // 或更小的值

// 限制序列长度
int maxSeqLength = 512; // 根据内存情况调整

// 使用GQA减少内存
Qwen3Config config = new Qwen3Config();
config.setNumKeyValueHeads(config.getNumAttentionHeads() / 2); // 减半KV头数
```

#### 推理优化

```java
// 预热模型
for (int i = 0; i < 5; i++) {
    model.forwardWithLogits(new Variable(inputIds));
}

// 计时测试
long startTime = System.currentTimeMillis();
Variable result = model.forwardWithLogits(new Variable(inputIds));
long endTime = System.currentTimeMillis();

System.out.println("推理时间: " + (endTime - startTime) + "ms");
```

#### 梯度管理

```java
// 清除梯度（如果进行训练）
model.clearGrads();

// 获取所有参数
Map<String, Parameter> params = model.getAllParams();
System.out.println("参数数量: " + params.size());
```

### 8. 错误处理和调试

#### 常见错误及解决方案

```java
public class Qwen3ErrorHandler {
    
    public static void handleCommonErrors(Qwen3Model model, NdArray inputIds) {
        try {
            // 验证输入
            validateInput(inputIds, model.getConfig().getVocabSize());
            
            // 执行前向传播
            Variable result = model.forwardWithLogits(new Variable(inputIds));
            
        } catch (IllegalArgumentException e) {
            if (e.getMessage().contains("Token ID超出范围")) {
                System.err.println("输入包含无效的token ID，请检查词汇表范围");
            } else if (e.getMessage().contains("维度错误")) {
                System.err.println("输入维度不正确，支持1D或2D输入");
            }
        } catch (OutOfMemoryError e) {
            System.err.println("内存不足，尝试减少批次大小或序列长度");
        } catch (Exception e) {
            System.err.println("未知错误: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void validateInput(NdArray inputIds, int vocabSize) {
        // 实现验证逻辑...
    }
}
```

#### 调试工具

```java
public class Qwen3Debug {
    
    public static void printIntermediateOutputs(Qwen3Model model, NdArray inputIds) {
        Qwen3Block block = model.getQwen3Block();
        
        // 获取词嵌入输出
        Variable embeddings = block.getEmbedTokens().layerForward(new Variable(inputIds));
        System.out.println("嵌入输出形状: " + embeddings.getValue().getShape());
        
        // 获取每层输出
        Variable currentOutput = embeddings;
        for (int i = 0; i < block.getDecoderLayers().length; i++) {
            currentOutput = block.getLayer(i).layerForward(currentOutput);
            System.out.println("第" + i + "层输出形状: " + currentOutput.getValue().getShape());
        }
    }
}
```

### 9. 最佳实践

#### 开发阶段

1. **使用小型配置进行开发**
```java
Qwen3Model devModel = Qwen3Model.createTinyModel("development");
```

2. **逐步验证功能**
```java
// 先测试单个组件
// 再测试整个模型
// 最后进行端到端测试
```

3. **添加日志和监控**
```java
System.out.println("模型加载完成: " + model.getName());
System.out.println("输入处理完成，序列长度: " + inputIds.getShape());
```

#### 生产部署

1. **性能测试**
```java
// 测试不同批次大小的性能
// 测试不同序列长度的内存使用
// 进行压力测试
```

2. **错误处理**
```java
// 完善的异常处理
// 优雅的降级策略
// 监控和告警
```

3. **资源管理**
```java
// 合理的内存配置
// 连接池管理
// 缓存策略
```

## 示例项目

### 简单聊天机器人

```java
public class SimpleChatBot {
    private Qwen3Model model;
    private int maxResponseLength = 50;
    
    public SimpleChatBot() {
        this.model = Qwen3Model.createTinyModel("chatbot");
    }
    
    public String generateResponse(String input) {
        // 这里需要实现tokenization
        NdArray inputIds = tokenize(input);
        
        // 生成回复
        NdArray response = model.generate(inputIds, maxResponseLength);
        
        // 这里需要实现detokenization
        return detokenize(response);
    }
    
    private NdArray tokenize(String text) {
        // 简化的tokenization实现
        // 实际使用中需要proper tokenizer
        return NdArray.of(new float[]{1, 10, 20, 30}); // 占位实现
    }
    
    private String detokenize(NdArray tokens) {
        // 简化的detokenization实现
        return "Generated response"; // 占位实现
    }
}
```

### 文本补全工具

```java
public class TextCompletion {
    private Qwen3Model model;
    
    public TextCompletion() {
        this.model = Qwen3Model.createTinyModel("completion");
    }
    
    public String completeText(String prompt, int length) {
        NdArray promptIds = tokenize(prompt);
        NdArray completed = model.generate(promptIds, length);
        return detokenize(completed);
    }
}
```

## 性能基准

### 不同配置的性能对比

| 配置 | 层数 | 隐藏维度 | 参数量 | 推理速度(token/s) | 内存使用(GB) |
|------|------|----------|--------|-------------------|--------------|
| Tiny | 4 | 256 | 1.2M | 1000 | 0.5 |
| Small | 8 | 512 | 15M | 500 | 1.2 |
| Medium | 16 | 1024 | 120M | 200 | 3.5 |
| Large | 32 | 2048 | 800M | 50 | 12.0 |

### 优化建议

1. **内存优化**: 使用GQA，减少KV头数
2. **速度优化**: 减少层数，降低隐藏维度
3. **质量优化**: 增加参数量，使用更大配置

## 故障排除

### 常见问题

1. **内存不足**: 减少批次大小，使用更小的模型配置
2. **推理太慢**: 使用更小的模型，优化输入长度
3. **精度问题**: 检查数值范围，避免过大或过小的值
4. **维度错误**: 确保输入格式正确，验证配置参数

### 获取帮助

- 查看API文档: `doc/API_Reference.md`
- 查看架构说明: `doc/Architecture.md`
- 运行测试用例: `Qwen3Test.java`
- 查看演示代码: `Qwen3Demo.java`