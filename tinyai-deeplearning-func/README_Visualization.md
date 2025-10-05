# 计算图可视化工具使用说明

## 概述

在`tinyai-deeplearning-func`模块的`util`包中实现了两个强大的可视化工具：

1. **ComputationGraphVisualizer** - 计算图结构可视化
2. **StepByStepVisualizer** - 逐步反向传播过程可视化

这些工具可以帮助开发者更好地理解和调试深度学习计算图的构建和反向传播过程。

## 功能特性

### ComputationGraphVisualizer

- 📊 显示变量信息：名称、形状、数值、是否需要梯度
- 🔧 显示函数信息：函数类型、输入输出关系
- 🌳 显示树形计算图结构：从输出到输入的完整依赖关系
- 🎯 自动分配ID并建立节点映射关系

### StepByStepVisualizer

- 🚀 逐步展示反向传播过程
- 📍 显示每一步的详细信息：当前变量、函数、梯度计算
- ✅ 显示梯度累加过程和最终结果
- 🧮 展示梯度在计算图中的传播路径
- 📈 汇总所有变量的最终梯度

## 使用方法

### 1. 显示计算图结构

```java
// 创建变量和计算表达式
Variable x = new Variable(NdArray.of(2.0f), "x");
Variable y = new Variable(NdArray.of(3.0f), "y");
Variable z = x.add(y);
z.setName("z = x + y");

// 显示计算图
ComputationGraphVisualizer.display(z);
```

### 2. 逐步展示反向传播

```java
// 执行反向传播（注意：这会清空creator信息）
z.backward();

// 重新创建计算图用于可视化演示
Variable x2 = new Variable(NdArray.of(2.0f), "x");
Variable y2 = new Variable(NdArray.of(3.0f), "y");
Variable z2 = x2.add(y2);
z2.setName("z = x + y");

// 显示逐步反向传播过程
StepByStepVisualizer.showBackpropagation(z2);
```

## 输出示例

### 计算图结构显示

```
=== 计算图结构 ===

📊 变量列表:
  V0: z = x + y [形状: [1,1]] [值: 5.0000] [需要梯度]
  V1: x [形状: [1,1]] [值: 2.0000] [需要梯度]
  V2: y [形状: [1,1]] [值: 3.0000] [需要梯度]

🔧 函数列表:
  F0: Add [输入: V1, V2] [输出: V0]

🌳 计算图结构 (从输出到输入):
└── V0 (z = x + y)
    └── F0 (Add)
        ├── V1 (x)
        └── V2 (y)
=== 计算图结束 ===
```

### 逐步反向传播显示

```
🚀 开始逐步反向传播演示
==============================
📍 步骤 0: 初始化根变量梯度
   📊 根变量: z = x + y
      形状: [1,1]
      数值: 5.0000
      梯度: 1.0000
      需要梯度: 是

📍 步骤 1: 开始反向传播
   📊 当前变量: z = x + y
      形状: [1,1]
      数值: 5.0000
      梯度: 1.0000
      需要梯度: 是
   🔧 函数: Add
      输入数量: 2
      输入 0: x [形状: [1,1]]
      输入 1: y [形状: [1,1]]
   🧮 计算输入变量的梯度...
   ✅ 输入变量 0 (x): 设置梯度
      梯度值: 1.0000
   ✅ 输入变量 1 (y): 设置梯度
      梯度值: 1.0000

🎉 反向传播完成！
==============================
📈 z = x + y 最终梯度: 1.0000
📈 x 最终梯度: 1.0000
📈 y 最终梯度: 1.0000
```

## 注意事项

1. **变量命名**：建议为变量设置有意义的名称，以便于查看和调试
   ```java
   Variable input = new Variable(data, "input");
   ```

2. **backward()副作用**：变量执行`backward()`后会清空`creator`信息，如需重复演示，请重新创建计算图

3. **梯度累加**：如果一个变量被多次使用，梯度会自动累加

4. **大型数组显示**：对于大型数组，只显示前几个元素和总数，避免输出过长

## 运行演示

项目中包含了完整的演示程序：

```bash
cd /path/to/TinyAI
export JAVA_HOME=/Library/Java/JavaVirtualMachines/jdk-17.jdk/Contents/Home
java -cp "tinyai-deeplearning-func/target/classes:tinyai-deeplearning-ndarr/target/classes" io.leavesfly.tinyai.util.VisualizationDemo
```

演示程序包含四个场景：
1. 简单的加法和乘法运算
2. 复杂的数学表达式
3. 矩阵运算
4. 神经网络相关计算

## 适用场景

- 🐛 **调试计算图**：查看计算图的构建是否正确
- 📚 **教学演示**：展示深度学习计算过程
- 🔍 **梯度验证**：检查梯度计算是否正确
- 📊 **性能分析**：了解计算图的复杂度

这些工具为TinyAI框架的开发和使用提供了强大的可视化支持，有助于提高开发效率和代码质量。