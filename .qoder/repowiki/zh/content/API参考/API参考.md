# API参考

<cite>
**本文档中引用的文件**
- [Model.java](file://tinyai-deeplearning-ml/src/main/java/io/leavesfly/tinyai/ml/Model.java) - *更新了模型信息管理和序列化功能*
- [Trainer.java](file://tinyai-deeplearning-ml/src/main/java/io/leavesfly/tinyai/ml/Trainer.java) - *增强了并行训练功能*
- [Variable.java](file://tinyai-deeplearning-func/src/main/java/io/leavesfly/tinyai/func/Variable.java) - *新增了变量系统API*
- [Block.java](file://tinyai-deeplearning-nnet/src/main/java/io/leavesfly/tinyai/nnet/Block.java) - *更新了块级操作接口*
- [ModelSerializer.java](file://tinyai-deeplearning-ml/src/main/java/io/leavesfly/tinyai/ml/ModelSerializer.java) - *新增了完整的模型序列化功能*
- [ModelInfo.java](file://tinyai-deeplearning-ml/src/main/java/io/leavesfly/tinyai/ml/ModelInfo.java) - *新增了模型元数据信息类*
- [ParameterManager.java](file://tinyai-deeplearning-ml/src/main/java/io/leavesfly/tinyai/ml/ParameterManager.java) - *新增了参数管理功能*
</cite>

## 更新摘要
**变更内容**
- 新增了模型元数据管理功能，包括模型信息的详细记录和查询
- 增强了并行训练功能，支持更灵活的线程配置和资源管理
- 扩展了变量系统API，增加了更多数学运算和自动微分方法
- 更新了模型序列化功能，支持压缩保存和检查点管理
- 新增了参数管理API，支持参数复制、比较和统计

## 目录
1. [简介](#简介)
2. [Model类API](#model类api)
3. [Trainer类API](#trainer类api)
4. [Variable类API](#variable类api)
5. [Block类API](#block类api)
6. [参数管理API](#参数管理api)
7. [模型序列化API](#模型序列化api)
8. [使用示例](#使用示例)
9. [故障排除指南](#故障排除指南)
10. [总结](#总结)

## 简介

TinyAI是一个Java深度学习框架，提供了完整的机器学习和深度学习API。本文档详细介绍了框架的核心编程接口，包括Model类、Trainer类、Variable类和Block类的主要方法和使用规范。

## Model类API

Model类是TinyAI框架中模型的核心表示，提供了模型的完整生命周期管理功能。

### 构造函数

```java
public Model(String _name, Block _block)
```

**参数说明：**
- `_name`: 模型名称，用于标识和调试
- `_block`: 模型的神经网络结构，包含所有层和参数

**异常情况：**
- 抛出`RuntimeException`如果_block为null

### 核心方法

#### 前向传播
```java
public Variable forward(Variable... inputs)
```

**功能描述：** 执行模型的前向传播计算

**参数说明：**
- `inputs`: 输入变量数组

**返回值：** 输出变量

**异常情况：**
- 抛出`RuntimeException`如果输入形状不匹配

#### 参数管理
```java
public Map<String, Parameter> getAllParams()
```

**功能描述：** 获取模型中所有的参数

**返回值：** 包含所有参数的Map，键为参数名称，值为Parameter对象

#### 模型保存与加载
```java
public void saveModel(String filePath)
public static Model loadModel(String filePath)
```

**功能描述：** 保存和加载模型到文件

**参数说明：**
- `filePath`: 文件路径

**异常情况：**
- 抛出`RuntimeException`如果文件操作失败

#### 训练信息管理
```java
public void updateTrainingInfo(int epochs, double finalLoss, String optimizer, double learningRate)
public void addMetric(String metricName, double value)
```

**功能描述：** 更新模型的训练信息和性能指标

**参数说明：**
- `epochs`: 训练轮次
- `finalLoss`: 最终损失值
- `optimizer`: 优化器名称
- `learningRate`: 学习率
- `metricName`: 指标名称
- `value`: 指标值

**章节来源**
- [Model.java](file://tinyai-deeplearning-ml/src/main/java/io/leavesfly/tinyai/ml/Model.java#L1-L361)

## Trainer类API

Trainer类是TinyAI框架中模型训练的核心组件，提供了完整的训练流程管理功能。

### 构造函数

```java
public Trainer(int _maxEpoch, Monitor _monitor, Evaluator _evaluator)
public Trainer(int _maxEpoch, Monitor _monitor, Evaluator _evaluator,
               boolean enableParallel, int threadCount)
```

**参数说明：**
- `_maxEpoch`: 最大训练轮次
- `_monitor`: 监控器，用于收集训练过程信息
- `_evaluator`: 评估器，用于模型性能评估
- `enableParallel`: 是否启用并行训练
- `threadCount`: 并行线程数（0表示自动计算）

### 核心方法

#### 训练控制
```java
public void init(DataSet _dataSet, Model _model, Loss _loss, Optimizer _optimizer)
public void train(boolean shuffleData)
public void evaluate()
```

**功能描述：** 初始化训练器、执行训练和评估

**参数说明：**
- `_dataSet`: 数据集
- `_model`: 要训练的模型
- `_loss`: 损失函数
- `_optimizer`: 优化器
- `shuffleData`: 是否打乱数据

#### 并行训练配置
```java
public void configureParallelTraining(boolean enable, int threadCount)
public boolean isParallelTrainingEnabled()
public int getParallelThreadCount()
```

**功能描述：** 配置并行训练参数和查询状态

#### 资源管理
```java
public void shutdown()
```

**功能描述：** 关闭训练器并释放资源

**异常情况：**
- 抛出`RuntimeException`如果线程池关闭失败

**章节来源**
- [Trainer.java](file://tinyai-deeplearning-ml/src/main/java/io/leavesfly/tinyai/ml/Trainer.java#L1-L495)

## Variable类API

Variable类是对数学变量的抽象表示，是自动微分系统的核心组件。

### 构造函数

```java
public Variable(NdArray _value)
public Variable(NdArray _value, String _name)
public Variable(NdArray _value, String _name, boolean _requireGrad)
public Variable(Number number)
```

**参数说明：**
- `_value`: 变量的值，使用NdArray表示
- `_name`: 变量名称，用于调试和可视化
- `_requireGrad`: 是否需要计算梯度，默认为true
- `number`: 数字值，内部转换为NdArray

### 数学运算方法

#### 四则运算
```java
public Variable add(Variable other)
public Variable sub(Variable other)
public Variable mul(Variable other)
public Variable div(Variable other)
public Variable neg()
```

**功能描述：** 执行加法、减法、乘法、除法和取反运算

#### 基本数学函数
```java
public Variable squ()
public Variable pow(float pow)
public Variable exp()
public Variable sin()
public Variable cos()
public Variable log()
public Variable tanh()
public Variable sigmoid()
public Variable relu()
public Variable softMax()
```

**功能描述：** 执行各种数学函数运算

#### 张量变形操作
```java
public Variable reshape(Shape shape)
public Variable transpose()
public Variable matMul(Variable other)
public Variable sum()
public Variable broadcastTo(Shape shape)
public Variable linear(Variable w, Variable b)
```

**功能描述：** 执行张量的变形和线性变换操作

### 自动微分相关方法

#### 反向传播
```java
public void backward()
public void backwardIterative()
public void unChainBackward()
public void clearGrad()
```

**功能描述：** 执行反向传播计算、清理计算图和梯度

#### 梯度管理
```java
public NdArray getGrad()
public void setGrad(NdArray _grad)
public boolean isRequireGrad()
public Variable setRequireGrad(boolean _requireGrad)
```

**功能描述：** 获取和设置梯度，控制梯度计算

**章节来源**
- [Variable.java](file://tinyai-deeplearning-func/src/main/java/io/leavesfly/tinyai/func/Variable.java#L1-L654)

## Block类API

Block类是神经网络中用于组合多个Layer的容器类，是构建复杂神经网络结构的基础组件。

### 构造函数

```java
public Block(String _name, Shape _inputShape)
public Block(String _name, Shape _inputShape, Shape _outputShape)
```

**参数说明：**
- `_name`: Block的名称
- `_inputShape`: 输入数据的形状
- `_outputShape`: 输出数据的形状（可选）

### 核心方法

#### 层管理
```java
public void addLayer(LayerAble layerAble)
```

**功能描述：** 向Block中添加一个Layer

**参数说明：**
- `layerAble`: 要添加的Layer实例

#### 前向传播
```java
@Override
public Variable layerForward(Variable... inputs)
```

**功能描述：** 执行Block的前向传播

**参数说明：**
- `inputs`: 输入变量数组

**返回值：** 输出变量

#### 参数管理
```java
public Map<String, Parameter> getAllParams()
private void putAll(Map<String, Parameter> allParams)
```

**功能描述：** 获取Block中所有的参数

#### 状态管理
```java
public void resetState()
@Override
public void clearGrads()
```

**功能描述：** 重置Block中所有RNN层的状态和清理梯度

**章节来源**
- [Block.java](file://tinyai-deeplearning-nnet/src/main/java/io/leavesfly/tinyai/nnet/Block.java#L1-L136)

## 参数管理API

ParameterManager类提供了模型参数的高级管理功能。

### 核心方法

#### 参数复制
```java
public static int copyParameters(Model sourceModel, Model targetModel)
public static int copyParameters(Model sourceModel, Model targetModel, boolean strict)
```

**功能描述：** 将参数从一个模型复制到另一个模型

**参数说明：**
- `sourceModel`: 源模型
- `targetModel`: 目标模型
- `strict`: 是否严格模式（所有参数都必须匹配）

**返回值：** 成功复制的参数数量

#### 参数比较
```java
public static boolean compareParameters(Model model1, Model model2, double tolerance)
```

**功能描述：** 比较两个模型的参数

**参数说明：**
- `model1`: 模型1
- `model2`: 模型2
- `tolerance`: 容忍度

**返回值：** 参数是否相同

#### 参数统计
```java
public static ParameterStats getParameterStats(Map<String, Parameter> params)
public static void saveParameterStats(Map<String, Parameter> params, String filePath)
```

**功能描述：** 获取参数统计信息和保存到文件

**章节来源**
- [ParameterManager.java](file://tinyai-deeplearning-ml/src/main/java/io/leavesfly/tinyai/ml/ParameterManager.java#L1-L200)

## 模型序列化API

ModelSerializer类提供了模型的序列化和反序列化功能。

### 核心方法

#### 模型保存
```java
public static void saveModel(Model model, String filePath)
public static void saveModel(Model model, String filePath, boolean compress)
public static void saveParameters(Model model, String filePath)
```

**功能描述：** 保存模型的不同形式

**参数说明：**
- `model`: 要保存的模型
- `filePath`: 文件路径
- `compress`: 是否压缩保存

#### 模型加载
```java
public static Model loadModel(String filePath)
public static void loadParameters(Model model, String filePath)
public static Model resumeFromCheckpoint(String filePath)
```

**功能描述：** 加载模型的不同形式

#### 检查点管理
```java
public static void saveCheckpoint(Model model, int epoch, double loss, String filePath)
```

**功能描述：** 保存训练检查点

**章节来源**
- [ModelSerializer.java](file://tinyai-deeplearning-ml/src/main/java/io/leavesfly/tinyai/ml/ModelSerializer.java#L1-L396)

## 使用示例

### 基本模型训练示例

```java
// 1. 创建模型
MlpBlock mlpBlock = new MlpBlock("mlp", batchSize, Config.ActiveFunc.ReLU, 10, 64, 32, 1);
Model model = new Model("MyModel", mlpBlock);

// 2. 创建训练器
Monitor monitor = new Monitor();
Evaluator evaluator = new AccuracyEval();
Trainer trainer = new Trainer(100, monitor, evaluator);

// 3. 初始化训练器
trainer.init(dataset, model, new SoftmaxCrossEntropy(), new SGD(0.01));

// 4. 开始训练
trainer.train(true);

// 5. 保存模型
model.saveModel("my_model.model");
```

### Variable数学运算示例

```java
// 创建变量
Variable x = new Variable(NdArray.of(2.0f), "x");
Variable y = new Variable(NdArray.of(3.0f), "y");

// 执行运算
Variable sum = x.add(y);        // 加法
Variable product = x.mul(y);    // 乘法
Variable square = x.squ();      // 平方

// 自动微分
Variable loss = sum.meanSquaredError(product);
loss.backward();                // 反向传播
```

### 参数管理示例

```java
// 获取所有参数
Map<String, Parameter> params = model.getAllParams();

// 复制参数
ParameterManager.copyParameters(sourceModel, targetModel);

// 比较参数
boolean areEqual = ParameterManager.compareParameters(model1, model2, 1e-6);

// 参数统计
ParameterManager.ParameterStats stats = ParameterManager.getParameterStats(params);
System.out.println("参数总数: " + stats.totalParameters);
```

## 故障排除指南

### 常见异常及解决方案

#### 模型保存失败
**错误信息:** `RuntimeException: Model save error!`

**解决方案:**
1. 检查文件路径是否存在且可写
2. 确保磁盘空间充足
3. 验证模型对象是否完整

#### 参数形状不匹配
**错误信息:** `RuntimeException: _grad shape must equal value shape!`

**解决方案:**
1. 检查输入数据的形状是否正确
2. 验证模型架构是否匹配数据
3. 确保批处理大小一致

#### 内存不足
**错误信息:** `OutOfMemoryError`

**解决方案:**
1. 减少批处理大小
2. 使用更小的模型架构
3. 启用梯度检查点技术

### 性能优化建议

#### 并行训练配置
```java
// 启用并行训练
Trainer trainer = new Trainer(maxEpoch, monitor, evaluator, true, 4);

// 动态调整线程数
trainer.configureParallelTraining(true, Runtime.getRuntime().availableProcessors());
```

#### 内存管理
```java
// 定期清理梯度
model.clearGrads();

// 使用迭代式反向传播
variable.backwardIterative();

// 及时释放不再使用的变量
variable.unChainBackward();
```

## 总结

TinyAI提供了完整的深度学习API，涵盖了从基础的Variable类到复杂的Model类和Trainer类。主要特性包括：

1. **完整的模型生命周期管理** - 从创建到部署的全流程支持
2. **强大的自动微分系统** - 支持复杂的数学运算和梯度计算
3. **灵活的网络架构** - 支持多种网络结构和层组合
4. **高效的并行训练** - 支持多线程并行处理
5. **完善的参数管理** - 提供参数复制、比较和统计功能
6. **便捷的序列化支持** - 支持模型保存、加载和检查点

通过合理使用这些API，开发者可以快速构建和训练各种深度学习模型，满足不同的应用场景需求。