# Qwen3模型开发指南

## 开发环境搭建

### 环境要求

- **JDK**: Java 8 或更高版本
- **构建工具**: Maven 3.6+
- **IDE**: IntelliJ IDEA 或 Eclipse（推荐IntelliJ IDEA）
- **内存**: 至少4GB可用内存（开发大模型推荐8GB+）

### 项目结构

```
tinyai-model-qwen/
├── doc/                          # 文档目录
│   ├── API_Reference.md          # API参考文档
│   ├── Architecture.md           # 架构设计文档
│   ├── User_Guide.md            # 使用指南
│   ├── Development_Guide.md      # 开发指南（本文档）
│   └── qwen3.py                 # Python参考实现
├── src/
│   ├── main/java/io/leavesfly/tinyai/qwen3/
│   │   ├── Qwen3Config.java     # 配置类
│   │   ├── RMSNorm.java         # RMS归一化层
│   │   ├── SiLULayer.java       # SiLU激活函数
│   │   ├── RotaryPositionalEmbedding.java  # RoPE位置编码
│   │   ├── Qwen3Attention.java  # 注意力机制
│   │   ├── Qwen3MLP.java        # 前馈网络
│   │   ├── Qwen3DecoderLayer.java # 解码器层
│   │   ├── Qwen3Block.java      # 核心网络块
│   │   ├── Qwen3Model.java      # 模型封装类
│   │   └── Qwen3Demo.java       # 演示程序
│   └── test/java/io/leavesfly/tinyai/qwen3/
│       └── Qwen3Test.java       # 单元测试
├── README.md                    # 项目说明
└── pom.xml                     # Maven配置文件
```

## 代码规范

### 命名规范

#### 1. 类命名
- **模型类**: 以`Model`结尾，如`Qwen3Model`
- **Block类**: 以`Block`结尾，如`Qwen3Block`
- **Layer类**: 以`Layer`结尾，如`SiLULayer`
- **配置类**: 以`Config`结尾，如`Qwen3Config`

#### 2. 方法命名
- **前向传播**: 使用`layerForward`（Layer/Block）或`forwardWithLogits`（Model）
- **初始化**: 使用`init`方法
- **参数获取**: 使用`get`前缀，如`getConfig()`, `getHiddenSize()`
- **工具方法**: 使用动词+名词，如`validateInput()`, `printModelInfo()`

#### 3. 变量命名
- **配置参数**: 驼峰命名，如`hiddenSize`, `numAttentionHeads`
- **张量数据**: 语义明确，如`hiddenStates`, `attentionWeights`
- **临时变量**: 简洁清晰，如`batchSize`, `seqLen`

### 注释规范

#### 1. 类注释
```java
/**
 * Qwen3多头注意力机制层
 * 
 * 实现了Qwen3模型的多头注意力机制，支持以下特性：
 * 1. 分组查询注意力（Grouped Query Attention, GQA）
 * 2. 旋转位置编码（RoPE）
 * 3. 因果掩码（用于自回归生成）
 * 4. 缩放点积注意力
 * 
 * @author 山泽
 * @version 1.0
 */
public class Qwen3Attention extends Layer {
    // ...
}
```

#### 2. 方法注释
```java
/**
 * 计算缩放点积注意力
 * 
 * @param query 查询张量，形状为(batch_size, num_heads, seq_len, head_dim)
 * @param key 键张量，形状为(batch_size, num_heads, seq_len, head_dim)
 * @param value 值张量，形状为(batch_size, num_heads, seq_len, head_dim)
 * @param batchSize 批次大小
 * @param seqLen 序列长度
 * @return 注意力输出，形状为(batch_size, seq_len, hidden_size)
 */
private Variable computeAttention(Variable query, Variable key, Variable value, int batchSize, int seqLen) {
    // 实现细节...
}
```

#### 3. 复杂逻辑注释
```java
// 重复KV张量以匹配查询头数（分组查询注意力）
// 如果nRep=1，说明KV头数等于注意力头数，无需重复
if (nRep == 1) {
    return hiddenStates;
}

// 为每个KV头创建nRep个副本
for (int h = 0; h < originalHeads; h++) {
    for (int rep = 0; rep < nRep; rep++) {
        int targetHead = h * nRep + rep;
        // 复制数据...
    }
}
```

### 代码风格

#### 1. 缩进和格式
- 使用4个空格缩进
- 大括号不换行（K&R风格）
- 方法间空一行
- 逻辑块间适当空行

#### 2. 行长度
- 每行代码不超过120字符
- 长参数列表合理换行
- 长链式调用分行书写

#### 3. 导入规范
```java
// 标准库导入
import java.util.ArrayList;
import java.util.List;

// TinyAI框架导入
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;
```

## 开发流程

### 1. 新组件开发

#### 步骤1: 设计接口
```java
// 定义组件接口和核心方法
public class NewComponent extends Layer {
    public NewComponent(String name, /* 其他参数 */) {
        super(name, inputShape, outputShape);
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        // 待实现
        return null;
    }
}
```

#### 步骤2: 实现核心逻辑
```java
@Override
public Variable layerForward(Variable... inputs) {
    Variable input = inputs[0];
    
    // 1. 输入验证
    validateInput(input);
    
    // 2. 核心计算
    Variable output = coreComputation(input);
    
    // 3. 输出验证
    validateOutput(output);
    
    return output;
}
```

#### 步骤3: 添加单元测试
```java
@Test
public void testNewComponent() {
    // 创建组件
    NewComponent component = new NewComponent("test", /* 参数 */);
    
    // 准备测试数据
    NdArray testInput = createTestInput();
    
    // 执行测试
    Variable result = component.layerForward(new Variable(testInput));
    
    // 验证结果
    assertNotNull(result);
    assertEquals(expectedShape, result.getValue().getShape());
}
```

#### 步骤4: 集成测试
```java
// 在完整模型中测试新组件
Qwen3Model model = createModelWithNewComponent();
NdArray input = createTestInput();
Variable output = model.forwardWithLogits(new Variable(input));
// 验证集成效果
```

### 2. Bug修复流程

#### 步骤1: 问题定位
```java
// 添加调试信息
System.out.println("Debug: 输入形状 = " + input.getValue().getShape());
System.out.println("Debug: 中间结果 = " + intermediate.getValue().getShape());

// 使用断言验证假设
assert input.getValue().getShape().getDimNum() == 2 : "输入应该是2维";
```

#### 步骤2: 编写复现测试
```java
@Test
public void testBugReproduction() {
    // 创建能够复现bug的最小测试用例
    // ...
    
    // 期望这个测试在修复前失败，修复后通过
}
```

#### 步骤3: 修复实现
- 最小化修改范围
- 保持API兼容性
- 添加必要的验证

#### 步骤4: 回归测试
```bash
# 运行所有相关测试
mvn test -Dtest=Qwen3Test
mvn test -Dtest=*ComponentTest
```

### 3. 性能优化流程

#### 步骤1: 性能分析
```java
// 添加性能测试
public void benchmarkComponent() {
    long startTime = System.currentTimeMillis();
    
    for (int i = 0; i < 1000; i++) {
        component.layerForward(testInput);
    }
    
    long endTime = System.currentTimeMillis();
    double avgTime = (endTime - startTime) / 1000.0;
    System.out.println("平均执行时间: " + avgTime + "ms");
}
```

#### 步骤2: 瓶颈识别
- 使用JProfiler或类似工具
- 分析内存分配模式
- 识别计算热点

#### 步骤3: 优化实现
```java
// 示例：缓存重复计算
private NdArray cachedResult = null;
private Shape lastInputShape = null;

public Variable optimizedForward(Variable input) {
    Shape currentShape = input.getValue().getShape();
    
    // 缓存机制
    if (cachedResult != null && currentShape.equals(lastInputShape)) {
        return new Variable(cachedResult);
    }
    
    // 正常计算
    Variable result = normalForward(input);
    
    // 更新缓存
    cachedResult = result.getValue();
    lastInputShape = currentShape;
    
    return result;
}
```

#### 步骤4: 性能验证
- 对比优化前后性能
- 确保功能正确性
- 进行压力测试

## 测试规范

### 1. 单元测试

#### 测试结构
```java
public class ComponentTest {
    
    @Before
    public void setUp() {
        // 测试准备
    }
    
    @Test
    public void testBasicFunctionality() {
        // 基本功能测试
    }
    
    @Test
    public void testEdgeCases() {
        // 边界情况测试
    }
    
    @Test
    public void testErrorHandling() {
        // 错误处理测试
    }
    
    @After
    public void tearDown() {
        // 清理工作
    }
}
```

#### 测试覆盖率
- **语句覆盖率**: 目标80%+
- **分支覆盖率**: 目标70%+
- **方法覆盖率**: 目标90%+

### 2. 集成测试

#### 端到端测试
```java
@Test
public void testEndToEnd() {
    // 创建完整模型
    Qwen3Model model = Qwen3Model.createTinyModel("e2e_test");
    
    // 准备真实输入
    NdArray realInput = createRealisticInput();
    
    // 执行完整流程
    Variable output = model.forwardWithLogits(new Variable(realInput));
    NdArray generated = model.generate(realInput, 20);
    
    // 验证输出质量
    validateOutput(output, generated);
}
```

#### 组件交互测试
```java
@Test
public void testComponentInteraction() {
    // 测试组件间的数据流
    // 验证接口兼容性
    // 检查数据一致性
}
```

### 3. 性能测试

#### 基准测试
```java
@Test
public void benchmarkPerformance() {
    // 测试不同配置的性能
    // 测试不同输入大小的性能
    // 记录性能基线
}
```

#### 内存测试
```java
@Test
public void testMemoryUsage() {
    Runtime runtime = Runtime.getRuntime();
    
    long beforeMemory = runtime.totalMemory() - runtime.freeMemory();
    
    // 执行操作
    performOperation();
    
    long afterMemory = runtime.totalMemory() - runtime.freeMemory();
    long memoryUsed = afterMemory - beforeMemory;
    
    assertTrue("内存使用超过预期", memoryUsed < EXPECTED_MEMORY_LIMIT);
}
```

## 调试技巧

### 1. 数据流跟踪

#### 添加中间输出
```java
public Variable debugLayerForward(Variable... inputs) {
    Variable input = inputs[0];
    System.out.println("输入形状: " + input.getValue().getShape());
    
    Variable intermediate = processInput(input);
    System.out.println("中间结果形状: " + intermediate.getValue().getShape());
    System.out.println("中间结果范围: [" + getMin(intermediate) + ", " + getMax(intermediate) + "]");
    
    Variable output = processIntermediate(intermediate);
    System.out.println("输出形状: " + output.getValue().getShape());
    
    return output;
}
```

#### 数据可视化
```java
public void visualizeAttentionWeights(NdArray weights) {
    // 打印注意力权重矩阵
    int seqLen = weights.getShape().getDimension(0);
    for (int i = 0; i < seqLen; i++) {
        for (int j = 0; j < seqLen; j++) {
            System.out.printf("%.3f ", weights.get(i, j));
        }
        System.out.println();
    }
}
```

### 2. 梯度检查

#### 数值梯度验证
```java
public void checkGradients(Layer layer, Variable input) {
    double epsilon = 1e-7;
    
    // 计算解析梯度
    Variable output = layer.layerForward(input);
    // ... 反向传播计算梯度
    
    // 计算数值梯度
    double numGrad = computeNumericalGradient(layer, input, epsilon);
    
    // 比较梯度
    double gradDiff = Math.abs(analyticalGrad - numGrad);
    assertTrue("梯度检查失败", gradDiff < 1e-6);
}
```

### 3. 错误诊断

#### 形状不匹配诊断
```java
public void diagnoseShapeMismatch(NdArray a, NdArray b, String operation) {
    if (!a.getShape().equals(b.getShape())) {
        System.err.println("形状不匹配错误:");
        System.err.println("操作: " + operation);
        System.err.println("张量A形状: " + a.getShape());
        System.err.println("张量B形状: " + b.getShape());
        
        // 提供修复建议
        suggestShapeFix(a.getShape(), b.getShape(), operation);
    }
}
```

#### 数值问题诊断
```java
public void diagnoseNumericalIssues(NdArray array, String context) {
    float[] data = array.flatten();
    
    boolean hasNaN = false;
    boolean hasInf = false;
    float min = Float.MAX_VALUE;
    float max = Float.MIN_VALUE;
    
    for (float value : data) {
        if (Float.isNaN(value)) hasNaN = true;
        if (Float.isInfinite(value)) hasInf = true;
        min = Math.min(min, value);
        max = Math.max(max, value);
    }
    
    if (hasNaN) {
        System.err.println("检测到NaN值，上下文: " + context);
    }
    if (hasInf) {
        System.err.println("检测到无穷值，上下文: " + context);
    }
    if (Math.abs(max) > 1e6 || Math.abs(min) > 1e6) {
        System.err.println("检测到极大值，可能存在数值不稳定，范围: [" + min + ", " + max + "]");
    }
}
```

## 代码审查清单

### 1. 功能性检查
- [ ] 功能符合设计要求
- [ ] 边界情况处理正确
- [ ] 错误处理完善
- [ ] 输入输出验证充分

### 2. 性能检查
- [ ] 无明显性能瓶颈
- [ ] 内存使用合理
- [ ] 算法复杂度可接受
- [ ] 缓存策略恰当

### 3. 代码质量检查
- [ ] 代码结构清晰
- [ ] 命名规范一致
- [ ] 注释充分准确
- [ ] 无重复代码

### 4. 测试检查
- [ ] 单元测试充分
- [ ] 集成测试覆盖
- [ ] 性能测试完整
- [ ] 测试用例有意义

### 5. 文档检查
- [ ] API文档更新
- [ ] 使用示例完整
- [ ] 变更记录清晰
- [ ] 设计决策记录

## 发布流程

### 1. 版本管理

#### 版本号规则
```
主版本号.次版本号.修订号
1.0.0 - 初始发布
1.1.0 - 新功能添加
1.0.1 - Bug修复
```

#### 标签管理
```bash
# 创建版本标签
git tag -a v1.0.0 -m "Qwen3模型首次发布"

# 推送标签
git push origin v1.0.0
```

### 2. 构建和测试

#### 完整测试套件
```bash
# 运行所有测试
mvn clean test

# 生成测试报告
mvn surefire-report:report

# 代码覆盖率检查
mvn jacoco:report
```

#### 构建发布包
```bash
# 编译打包
mvn clean package

# 生成文档
mvn javadoc:javadoc

# 创建发布包
mvn assembly:assembly
```

### 3. 发布检查

#### 发布前检查清单
- [ ] 所有测试通过
- [ ] 代码覆盖率达标
- [ ] 文档更新完整
- [ ] 性能基准达标
- [ ] 兼容性验证通过

#### 发布后验证
- [ ] 发布包完整性
- [ ] 安装流程验证
- [ ] 快速开始指南验证
- [ ] 示例代码验证

## 故障排除指南

### 常见问题及解决方案

1. **编译错误**
   - 检查Java版本兼容性
   - 验证依赖版本
   - 清理并重新构建

2. **运行时错误**
   - 检查输入数据格式
   - 验证模型配置
   - 分析堆栈跟踪

3. **性能问题**
   - 分析内存使用
   - 检查算法复杂度
   - 优化数据结构

4. **数值问题**
   - 检查数值范围
   - 验证数学公式
   - 调整数值精度

### 获取帮助

- **项目文档**: 查看doc目录下的文档
- **测试用例**: 参考test目录下的示例
- **社区支持**: 提交Issue或讨论
- **技术支持**: 联系维护团队