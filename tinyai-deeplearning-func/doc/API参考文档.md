# TinyAI自动微分引擎API参考文档

## 📚 API概述

本文档提供了TinyAI自动微分引擎的完整API参考，包括核心类、接口和方法的详细说明。所有API都经过充分测试，确保稳定性和可靠性。

## 🔧 核心类API

### Variable类

`Variable`是自动微分引擎的核心类，代表计算图中的变量节点。

#### 构造方法

```java
// 使用NdArray创建变量
public Variable(NdArray value)

// 使用数值创建变量
public Variable(Number number)

// 创建带名称的变量
public Variable(NdArray value, String name)

// 创建带名称和梯度控制的变量
public Variable(NdArray value, String name, boolean requireGrad)
```

**参数说明:**
- `value`: 变量的初始值，必须为非null的NdArray对象
- `number`: 数值类型的初始值，会自动转换为NdArray
- `name`: 变量名称，用于调试和可视化
- `requireGrad`: 是否需要计算梯度，默认为true

**示例:**
```java
// 创建矩阵变量
Variable x = new Variable(NdArray.of(new float[][]{{1,2},{3,4}}));

// 创建标量变量
Variable y = new Variable(3.14f);

// 创建带名称的变量
Variable weight = new Variable(NdArray.randn(new int[]{784, 10}), "weight");

// 创建不需要梯度的变量
Variable bias = new Variable(NdArray.zeros(new int[]{10}), "bias", false);
```

#### 基本属性方法

```java
// 获取变量值
public NdArray getValue()

// 设置变量值
public void setValue(NdArray value)

// 获取梯度
public NdArray getGrad()

// 设置梯度
public void setGrad(NdArray grad)

// 获取变量名称
public String getName()

// 设置变量名称
public Variable setName(String name)

// 获取创建该变量的函数
public Function getCreator()

// 设置创建该变量的函数
public void setCreator(Function creator)

// 检查是否需要计算梯度
public boolean isRequireGrad()

// 设置是否需要计算梯度
public Variable setRequireGrad(boolean requireGrad)
```

#### 反向传播方法

```java
// 递归反向传播（标准实现）
public void backward()

// 迭代反向传播（适用于深层网络，避免栈溢出）
public void backwardIterative()

// 切断计算图连接
public void unChainBackward()

// 清除梯度
public void clearGrad()
```

**使用说明:**
- `backward()`: 从当前变量开始执行反向传播，递归计算所有依赖变量的梯度
- `backwardIterative()`: 使用栈结构避免递归调用可能的栈溢出问题
- `unChainBackward()`: 断开计算图连接，常用于RNN训练中防止梯度传播过长
- `clearGrad()`: 清除当前变量的梯度，通常在每次训练迭代开始前调用

#### 四则运算API

```java
// 加法运算
public Variable add(Variable other)

// 减法运算
public Variable sub(Variable other)

// 乘法运算（支持广播）
public Variable mul(Variable other)

// 除法运算
public Variable div(Variable other)

// 取负运算
public Variable neg()
```

**特性说明:**
- 所有运算都支持自动广播机制
- 运算结果会自动构建计算图（训练模式下）
- 梯度会在反向传播时自动计算

**示例:**
```java
Variable a = new Variable(NdArray.of(new float[][]{{1,2},{3,4}}));
Variable b = new Variable(NdArray.of(new float[][]{{2,3},{4,5}}));

Variable sum = a.add(b);      // 矩阵加法
Variable product = a.mul(b);  // 逐元素乘法
Variable diff = a.sub(b);     // 矩阵减法
Variable quotient = a.div(b); // 逐元素除法
Variable negative = a.neg();  // 取负
```

#### 数学函数API

```java
// 基础数学函数
public Variable squ()                    // 平方
public Variable pow(float pow)           // 幂运算
public Variable exp()                    // 指数函数
public Variable log()                    // 自然对数
public Variable sin()                    // 正弦函数
public Variable cos()                    // 余弦函数

// 激活函数
public Variable sigmoid()                // Sigmoid激活函数
public Variable tanh()                   // Tanh激活函数
public Variable relu()                   // ReLU激活函数
public Variable softMax()                // Softmax函数

// 数值处理函数
public Variable clip(float min, float max)           // 数值裁剪
public Variable max(int axis, boolean keepdims)     // 沿轴求最大值
public Variable min(int axis, boolean keepdims)     // 沿轴求最小值
```

**参数说明:**
- `pow`: 幂指数值
- `min/max`: 裁剪的最小值和最大值
- `axis`: 操作的轴索引
- `keepdims`: 是否保持原有维度

**示例:**
```java
Variable x = new Variable(NdArray.randn(new int[]{3, 4}));

Variable squared = x.squ();              // 平方
Variable powered = x.pow(3.0f);          // 三次方
Variable activated = x.relu();           // ReLU激活
Variable normalized = x.softMax();       // Softmax归一化
Variable clipped = x.clip(-1.0f, 1.0f);  // 裁剪到[-1,1]
```

#### 矩阵运算API

```java
// 矩阵乘法
public Variable matMul(Variable other)

// 线性变换
public Variable linear(Variable weight, Variable bias)

// 形状操作
public Variable reshape(Shape shape)
public Variable transpose()
public Variable broadcastTo(Shape shape)

// 聚合操作
public Variable sum()
public Variable sumTo(Shape shape)

// 索引操作
public Variable getItem(int[] rowSlices, int[] colSlices)
```

**使用说明:**
- `matMul()`: 执行矩阵乘法操作
- `linear()`: 执行线性变换 y = xW + b，bias可以为null
- `reshape()`: 改变张量形状，总元素数必须保持不变
- `transpose()`: 矩阵转置操作
- `broadcastTo()`: 将张量广播到指定形状
- `sum()`: 对所有元素求和
- `sumTo()`: 沿指定轴求和到目标形状
- `getItem()`: 根据索引获取子张量

**示例:**
```java
Variable x = new Variable(NdArray.randn(new int[]{32, 784}));  // 输入数据
Variable W = new Variable(NdArray.randn(new int[]{784, 10}));  // 权重
Variable b = new Variable(NdArray.zeros(new int[]{10}));       // 偏置

// 线性变换
Variable output = x.linear(W, b);  // 相当于 x.matMul(W).add(b)

// 形状操作
Variable reshaped = x.reshape(new Shape(32, 28, 28));  // 重塑为图像格式
Variable transposed = W.transpose();                    // 权重转置

// 聚合操作
Variable totalSum = output.sum();                       // 总和
Variable batchSum = output.sumTo(new Shape(10));        // 批次维度求和
```

#### 损失函数API

```java
// 均方误差损失
public Variable meanSquaredError(Variable target)

// Softmax交叉熵损失
public Variable softmaxCrossEntropy(Variable target)
```

**使用说明:**
- `meanSquaredError()`: 计算与目标值的均方误差，常用于回归任务
- `softmaxCrossEntropy()`: 计算Softmax交叉熵损失，常用于分类任务

**示例:**
```java
Variable predictions = model.forward(inputs);
Variable targets = new Variable(labels);

// 回归任务
Variable mseLoss = predictions.meanSquaredError(targets);

// 分类任务
Variable crossEntropyLoss = predictions.softmaxCrossEntropy(targets);

// 反向传播
mseLoss.backward();
```

### Function抽象类

`Function`是所有数学函数的基类，定义了标准的前向和反向传播接口。

#### 核心方法

```java
// 函数调用（模板方法）
public Variable call(Variable... inputs)

// 前向传播（子类实现）
public abstract NdArray forward(NdArray... inputs)

// 反向传播（子类实现）
public abstract List<NdArray> backward(NdArray yGrad)

// 输入参数数量要求（子类实现）
public abstract int requireInputNum()

// 清理资源
public void unChain()
```

#### 属性访问

```java
// 获取输入变量
public Variable[] getInputs()

// 设置输入变量
public void setInputs(Variable[] inputs)

// 获取输出变量
public Variable getOutput()

// 设置输出变量
public void setOutput(Variable output)
```

## 🔢 具体函数类API

### 基础运算类

#### Add加法类
```java
public class Add extends Function {
    // 支持广播的加法运算
    // 输入：2个Variable
    // 输出：加法结果Variable
}
```

#### Mul乘法类
```java
public class Mul extends Function {
    // 支持广播的逐元素乘法
    // 输入：2个Variable
    // 输出：乘法结果Variable
}
```

#### Sub减法类
```java
public class Sub extends Function {
    // 支持广播的减法运算
    // 输入：2个Variable
    // 输出：减法结果Variable
}
```

#### Div除法类
```java
public class Div extends Function {
    // 支持广播的逐元素除法
    // 输入：2个Variable
    // 输出：除法结果Variable
}
```

#### Neg取负类
```java
public class Neg extends Function {
    // 取负运算
    // 输入：1个Variable
    // 输出：取负结果Variable
}
```

### 数学函数类

#### Sigmoid激活函数类
```java
public class Sigmoid extends Function {
    // Sigmoid激活函数：1/(1+e^(-x))
    // 输入：1个Variable
    // 输出：Sigmoid结果Variable
    // 值域：(0, 1)
}
```

#### ReLU激活函数类
```java
public class ReLu extends Function {
    // ReLU激活函数：max(0, x)
    // 输入：1个Variable
    // 输出：ReLU结果Variable
}
```

#### Tanh激活函数类
```java
public class Tanh extends Function {
    // Tanh激活函数
    // 输入：1个Variable
    // 输出：Tanh结果Variable
    // 值域：[-1, 1]
}
```

#### GELU激活函数类
```java
public class GELU extends Function {
    // GELU激活函数
    // 输入：1个Variable
    // 输出：GELU结果Variable
}
```

#### Exp指数函数类
```java
public class Exp extends Function {
    // 自然指数函数：e^x
    // 输入：1个Variable
    // 输出：指数结果Variable
}
```

#### Log对数函数类
```java
public class Log extends Function {
    // 自然对数函数：ln(x)
    // 输入：1个Variable
    // 输出：对数结果Variable
}
```

#### Pow幂函数类
```java
public class Pow extends Function {
    private float power;
    
    public Pow(float power) {
        this.power = power;
    }
    
    // 幂函数：x^power
    // 输入：1个Variable
    // 输出：幂运算结果Variable
}
```

#### Clip裁剪函数类
```java
public class Clip extends Function {
    private float min;
    private float max;
    
    public Clip(float min, float max) {
        this.min = min;
        this.max = max;
    }
    
    // 数值裁剪：clip(x, min, max)
    // 输入：1个Variable
    // 输出：裁剪结果Variable
}
```

### 矩阵运算类

#### MatMul矩阵乘法类
```java
public class MatMul extends Function {
    // 矩阵乘法：A × B
    // 输入：2个Variable
    // 输出：矩阵乘法结果Variable
}
```

#### Linear线性变换类
```java
public class Linear extends Function {
    // 线性变换：y = xW + b
    // 输入：2-3个Variable (x, W, 可选的b)
    // 输出：线性变换结果Variable
}
```

#### Transpose转置类
```java
public class Transpose extends Function {
    // 矩阵转置
    // 输入：1个Variable
    // 输出：转置结果Variable
}
```

#### Reshape重塑类
```java
public class Reshape extends Function {
    private Shape newShape;
    
    public Reshape(Shape newShape) {
        this.newShape = newShape;
    }
    
    // 形状重塑
    // 输入：1个Variable
    // 输出：重塑后的Variable
}
```

#### BroadcastTo广播类
```java
public class BroadcastTo extends Function {
    private Shape targetShape;
    
    public BroadcastTo(Shape targetShape) {
        this.targetShape = targetShape;
    }
    
    // 广播到指定形状
    // 输入：1个Variable
    // 输出：广播后的Variable
}
```

#### Sum求和类
```java
public class Sum extends Function {
    // 对所有元素求和
    // 输入：1个Variable
    // 输出：求和结果Variable（标量）
}
```

#### SumTo指定求和类
```java
public class SumTo extends Function {
    private Shape targetShape;
    
    public SumTo(Shape targetShape) {
        this.targetShape = targetShape;
    }
    
    // 沿指定轴求和到目标形状
    // 输入：1个Variable
    // 输出：求和结果Variable
}
```

#### SoftMax类
```java
public class SoftMax extends Function {
    // Softmax函数：exp(x) / sum(exp(x))
    // 输入：1个Variable
    // 输出：Softmax结果Variable
}
```

#### GetItem索引类
```java
public class GetItem extends Function {
    private int[] rowSlices;
    private int[] colSlices;
    
    public GetItem(int[] rowSlices, int[] colSlices) {
        this.rowSlices = rowSlices;
        this.colSlices = colSlices;
    }
    
    // 根据索引获取子张量
    // 输入：1个Variable
    // 输出：索引结果Variable
}
```

### 损失函数类

#### MeanSE均方误差类
```java
public class MeanSE extends Function {
    // 均方误差损失：MSE = Σ(predict - label)² / n
    // 输入：2个Variable（预测值，真实值）
    // 输出：均方误差损失Variable
}
```

#### SoftmaxCE Softmax交叉熵类
```java
public class SoftmaxCE extends Function {
    // Softmax交叉熵损失
    // 输入：2个Variable（logits，标签）
    // 输出：交叉熵损失Variable
}
```

#### SigmoidCE Sigmoid交叉熵类
```java
public class SigmoidCE extends Function {
    // Sigmoid交叉熵损失
    // 输入：2个Variable（logits，标签）
    // 输出：交叉熵损失Variable
}
```

## ⚙️ 工具类API

### Config配置类

```java
public class Config {
    // 训练模式开关，控制是否构建计算图
    public static boolean train = true;
    
    // 数值精度配置
    public static float epsilon = 1e-7f;
}
```

**使用说明:**
- `train`: 设置为true时构建计算图支持反向传播，设置为false时仅执行前向计算
- `epsilon`: 数值计算中的小量，用于避免除零等数值问题

**示例:**
```java
// 训练模式
Config.train = true;
Variable loss = model.forward(data).meanSquaredError(targets);
loss.backward();

// 推理模式（节省内存）
Config.train = false;
Variable predictions = model.forward(data);
```

### Utils工具类

```java
public class Utils {
    // 各种实用工具方法
    // 具体方法依据实现而定
}
```

## 🔍 异常处理

### 常见异常类型

#### RuntimeException
- **原因**: 输入参数验证失败
- **示例**: 构造Variable时传入null值
- **解决**: 检查输入参数的有效性

#### IllegalArgumentException
- **原因**: 参数值不符合要求
- **示例**: 广播操作时形状不兼容
- **解决**: 确保操作数的形状符合广播规则

#### 数值计算异常
- **原因**: 数值溢出或下溢
- **示例**: 极大值输入导致exp函数溢出
- **解决**: 使用clip函数限制输入范围

### 异常处理最佳实践

```java
try {
    // 创建变量时检查输入
    Variable x = new Variable(data);
    if (x.getValue() == null) {
        throw new RuntimeException("Invalid input data");
    }
    
    // 执行计算
    Variable result = x.sigmoid().mul(y);
    result.backward();
    
} catch (RuntimeException e) {
    System.err.println("计算错误: " + e.getMessage());
    // 错误处理逻辑
}
```

## 📊 性能考虑

### 内存优化建议

1. **及时清理梯度**
```java
// 在每次训练迭代开始前清理梯度
model.clearGrads();
```

2. **推理模式使用**
```java
// 推理时关闭计算图构建
Config.train = false;
Variable predictions = model.forward(data);
```

3. **计算图断开**
```java
// RNN训练中定期断开计算图
if (step % truncateLength == 0) {
    hiddenState.unChainBackward();
}
```

### 计算效率建议

1. **批量处理**: 尽可能使用批量数据进行计算
2. **原地操作**: 在适当时候使用原地操作减少内存分配
3. **广播优化**: 理解广播机制，避免不必要的数据复制

## 📝 版本兼容性

- **当前版本**: 1.0-SNAPSHOT
- **Java版本要求**: Java 8+
- **依赖模块**: tinyai-deeplearning-ndarr

## 🔗 相关文档

- [TinyAI技术架构设计文档](./技术架构设计文档.md)
- [TinyAI使用示例与最佳实践](./使用示例与最佳实践.md)
- [TinyAI测试修复报告](./测试修复报告.md)

---

本API参考文档涵盖了TinyAI自动微分引擎的所有公开接口。如需了解具体实现细节，请参考源代码和相关技术文档。