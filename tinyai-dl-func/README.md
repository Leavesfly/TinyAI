# TinyAI-DL-Func 函数计算模块

## 📖 模块简介

TinyAI-DL-Func 是 TinyAI 深度学习框架的核心函数计算模块，提供了完整的自动微分引擎实现。该模块是深度学习框架的数学运算基础，支持动态计算图构建、自动梯度计算和丰富的数学函数操作。

### 核心特性

- 🔧 **完整的自动微分引擎**：支持动态计算图构建和自动梯度计算
- ⚡ **丰富的数学函数库**：包含基础四则运算、激活函数、损失函数等
- 🧮 **强大的矩阵运算**：支持矩阵乘法、转置、重塑、广播等操作
- 🚀 **高效的梯度传播**：同时支持递归和迭代两种反向传播实现
- 📊 **完整的测试覆盖**：100% 单元测试通过率，确保代码可靠性

## 🏗️ 模块架构

```
tinyai-dl-func/
├── src/main/java/io/leavesfly/tinyai/
│   ├── func/
│   │   ├── Variable.java          # 自动微分变量核心类
│   │   ├── Function.java          # 函数操作抽象基类
│   │   ├── Util.java             # 工具函数集合
│   │   ├── base/                 # 基础四则运算
│   │   │   ├── Add.java          # 加法运算
│   │   │   ├── Sub.java          # 减法运算
│   │   │   ├── Mul.java          # 乘法运算（支持广播）
│   │   │   ├── Div.java          # 除法运算
│   │   │   └── Neg.java          # 取负运算
│   │   ├── math/                 # 数学函数库
│   │   │   ├── ReLu.java         # ReLU 激活函数
│   │   │   ├── Sigmoid.java      # Sigmoid 激活函数
│   │   │   ├── Tanh.java         # Tanh 激活函数
│   │   │   ├── GELU.java         # GELU 激活函数
│   │   │   ├── Exp.java          # 指数函数
│   │   │   ├── Log.java          # 对数函数
│   │   │   ├── Sin.java          # 正弦函数
│   │   │   ├── Cos.java          # 余弦函数
│   │   │   ├── Pow.java          # 幂函数
│   │   │   ├── Squ.java          # 平方函数
│   │   │   ├── Clip.java         # 裁剪函数
│   │   │   ├── Max.java          # 最大值函数
│   │   │   └── Min.java          # 最小值函数
│   │   ├── matrix/               # 矩阵运算库
│   │   │   ├── MatMul.java       # 矩阵乘法
│   │   │   ├── Linear.java       # 线性变换
│   │   │   ├── Transpose.java    # 矩阵转置
│   │   │   ├── Reshape.java      # 形状重塑
│   │   │   ├── Sum.java          # 求和运算
│   │   │   ├── SumTo.java        # 指定维度求和
│   │   │   ├── SoftMax.java      # Softmax 函数
│   │   │   ├── BroadcastTo.java  # 广播操作
│   │   │   └── GetItem.java      # 索引取值
│   │   └── loss/                 # 损失函数库
│   │       ├── MeanSE.java       # 均方误差损失
│   │       ├── SoftmaxCE.java    # Softmax 交叉熵损失
│   │       └── SigmoidCE.java    # Sigmoid 交叉熵损失
│   └── util/
│       └── Config.java           # 配置管理类
└── doc/
    └── 测试修复报告.md            # 完整的测试修复文档
```

## 🚀 快速开始

### 基础使用示例

```java
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;

// 创建变量
Variable x = new Variable(NdArray.of(new float[][]{{1, 2}, {3, 4}}), "x");
Variable y = new Variable(NdArray.of(new float[][]{{2, 3}, {4, 5}}), "y");

// 基础运算
Variable z = x.add(y);          // 矩阵加法
Variable w = x.mul(y);          // 矩阵乘法（支持广播）
Variable u = z.sigmoid();       // 激活函数

// 自动微分
u.backward();                   // 反向传播计算梯度
System.out.println("x的梯度: " + x.getGrad());
```

### 线性变换示例

```java
// 创建权重和偏置
Variable W = new Variable(NdArray.randn(new int[]{4, 3}), "weight");
Variable b = new Variable(NdArray.zeros(new int[]{4}), "bias");
Variable x = new Variable(NdArray.randn(new int[]{2, 3}), "input");

// 线性变换: y = xW^T + b
Variable y = x.linear(W, b);

// 应用激活函数
Variable output = y.relu();

// 计算损失并反向传播
Variable target = new Variable(NdArray.ones(output.getValue().getShape()));
Variable loss = output.meanSE(target);
loss.backward();
```

## 🧮 核心组件详解

### Variable 类 - 自动微分变量

`Variable` 是自动微分引擎的核心类，它不仅包含变量的值，还维护梯度信息和计算图结构。

#### 主要特性
- **值存储**：使用 `NdArray` 存储多维数组数据
- **梯度管理**：自动维护和累积梯度信息
- **计算图构建**：通过 `creator` 字段维护计算图结构
- **操作符重载**：支持直观的数学运算语法

#### 关键方法
```java
// 反向传播（递归实现）
public void backward()

// 反向传播（迭代实现，避免栈溢出）
public void backwardIterative()

// 丰富的数学运算方法
public Variable add(Variable other)
public Variable mul(Variable other)
public Variable sigmoid()
public Variable relu()
public Variable matMul(Variable other)
// ... 更多运算方法
```

### Function 类 - 函数操作基类

`Function` 是所有数学函数操作的抽象基类，定义了前向传播和反向传播的标准接口。

#### 设计模式
- **模板方法模式**：定义了 `call` 方法的执行流程
- **策略模式**：子类实现具体的 `forward` 和 `backward` 逻辑

#### 核心接口
```java
// 前向传播计算
public abstract NdArray forward(NdArray... inputs);

// 反向传播计算（求导）
public abstract List<NdArray> backward(NdArray yGrad);

// 输入参数数量要求
public abstract int requireInputNum();
```

## 🔬 数学函数库

### 基础运算 (base 包)
- **Add/Sub/Mul/Div**：四则运算，支持广播机制
- **Neg**：取负运算

### 数学函数 (math 包)
- **激活函数**：ReLU, Sigmoid, Tanh, GELU
- **基础函数**：Exp, Log, Sin, Cos, Pow, Squ
- **实用函数**：Clip, Max, Min

### 矩阵运算 (matrix 包)
- **核心运算**：MatMul, Linear, Transpose
- **形状操作**：Reshape, BroadcastTo
- **聚合操作**：Sum, SumTo, SoftMax
- **索引操作**：GetItem

### 损失函数 (loss 包)
- **MeanSE**：均方误差损失
- **SoftmaxCE**：Softmax 交叉熵损失
- **SigmoidCE**：Sigmoid 交叉熵损失

## 🧪 测试与质量保证

本模块拥有完整的单元测试覆盖，**测试通过率达到 100%**。

### 测试统计
- **总测试数**：76
- **通过测试**：76 (100%)
- **失败测试**：0
- **错误测试**：0

### 测试覆盖范围
- ✅ 所有数学函数的正确性验证
- ✅ 梯度计算的准确性测试
- ✅ 边界值和特殊情况处理
- ✅ 广播机制的完整性验证
- ✅ 矩阵运算的数值精度测试

### 运行测试
```bash
# 运行所有测试
mvn test

# 运行特定测试类
mvn test -Dtest=VariableTest
mvn test -Dtest=MathFunctionsTest
```

## 🔧 配置与依赖

### Maven 依赖
```xml
<dependency>
    <groupId>io.leavesfly.tinyai</groupId>
    <artifactId>tinyai-dl-func</artifactId>
    <version>1.0-SNAPSHOT</version>
</dependency>
```

### 模块依赖
- **tinyai-dl-ndarr**：多维数组基础库
- **jfreechart**：图表绘制支持
- **junit**：单元测试框架

### 配置选项
通过 `Config` 类可以控制模块行为：
```java
// 训练模式开关（影响计算图构建）
Config.train = true;  // 开启计算图构建
Config.train = false; // 仅执行前向计算，节省内存
```

## 📈 性能特性

### 自动微分优化
- **动态计算图**：只在训练模式下构建计算图，推理时节省内存
- **双重反向传播**：提供递归和迭代两种实现，适应不同场景
- **梯度累积**：支持梯度的自动累积和复用

### 内存管理
- **延迟计算**：只在需要时构建计算图结构
- **资源清理**：提供 `unChain()` 方法断开计算图连接
- **序列化支持**：核心类实现 `Serializable` 接口

## 🛠️ 开发与扩展

### 添加新的数学函数
1. 继承 `Function` 抽象类
2. 实现 `forward` 方法（前向计算）
3. 实现 `backward` 方法（梯度计算）
4. 指定 `requireInputNum` 返回值
5. 在 `Variable` 类中添加对应的便捷方法

示例：
```java
public class MyFunction extends Function {
    @Override
    public NdArray forward(NdArray... inputs) {
        // 实现前向计算逻辑
        return result;
    }
    
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // 实现梯度计算逻辑
        return Arrays.asList(inputGrads);
    }
    
    @Override
    public int requireInputNum() {
        return 1; // 或具体的输入参数数量
    }
}
```

### 调试技巧
- 使用 `Variable.setName()` 为变量命名，便于调试
- 利用 `Config.train` 开关控制计算图构建
- 查看 `doc/测试修复报告.md` 了解常见问题和解决方案

## 🤝 贡献指南

1. 确保所有新功能都有对应的单元测试
2. 遵循现有的代码风格和命名规范
3. 为公共方法添加详细的中文注释
4. 运行 `mvn test` 确保所有测试通过
5. 更新相关文档和示例

## 📝 更新日志

### v1.0-SNAPSHOT
- ✅ 完整的自动微分引擎实现
- ✅ 丰富的数学函数库
- ✅ 100% 单元测试覆盖
- ✅ 广播机制支持
- ✅ 双重反向传播实现
- ✅ 完整的文档和示例

## 📄 许可证

本项目采用开源许可证，详情请参阅项目根目录的 LICENSE 文件。

---

**TinyAI-DL-Func** - 为深度学习提供坚实的数学基础 🚀