# 快速入门 03：搭建第一个神经网络

> **学习目标**
> 1. 理解神经网络的基本结构
> 2. 学会使用 TinyAI 构建简单的神经网络
> 3. 掌握前向传播的过程

---

## 理论速览（5分钟）

### 什么是神经网络？

神经网络是一种受生物神经元启发的计算模型。它由多层"神经元"组成，每层对输入数据进行变换。

```
输入层          隐藏层           输出层

  x1 ──→     ┌───┐
             │   │ ──→     ┌───┐
  x2 ──→     │ H │         │   │
             │   │ ──→     │ O │ ──→ 输出
  x3 ──→     └───┘         └───┘
             (激活函数)
```

### 神经网络的核心组件

| 组件 | 作用 | 类比 |
|------|------|------|
| **线性变换** | y = Wx + b | 加权求和 |
| **激活函数** | 引入非线性 | 决策/过滤 |
| **损失函数** | 衡量预测好坏 | 评分标准 |

### 激活函数的作用

没有激活函数，多层神经网络等价于单层线性变换：

```
线性 + 线性 = 线性
f(x) = W2(W1*x + b1) + b2 = (W2*W1)*x + (W2*b1 + b2) = W'*x + b'
```

激活函数引入非线性，让网络可以学习复杂的模式。

**常用激活函数：**
- **ReLU**: f(x) = max(0, x) —— 简单高效，最常用
- **Sigmoid**: f(x) = 1/(1+e^(-x)) —— 输出0~1，适合二分类
- **Tanh**: f(x) = (e^x - e^(-x))/(e^x + e^(-x)) —— 输出-1~1

---

## 代码实践

### 1. 创建第一个神经网络

```java
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

public class FirstNeuralNetwork {
    public static void main(String[] args) {
        // 网络参数
        int inputSize = 3;   // 输入特征数
        int hiddenSize = 4;  // 隐藏层神经元数
        int outputSize = 2;  // 输出类别数

        // 初始化权重（使用随机初始化）
        Variable W1 = new Variable(NdArray.randn(Shape.of(inputSize, hiddenSize)));
        Variable b1 = new Variable(NdArray.zeros(Shape.of(hiddenSize)));
        Variable W2 = new Variable(NdArray.randn(Shape.of(hiddenSize, outputSize)));
        Variable b2 = new Variable(NdArray.zeros(Shape.of(outputSize)));

        // 输入数据
        Variable x = new Variable(NdArray.of(new float[]{1.0f, 2.0f, 3.0f}));

        // 前向传播
        // 第一层：线性变换 + ReLU激活
        Variable hidden = x.dot(W1).add(b1);
        Variable hiddenActivated = hidden.relu();

        // 第二层：线性变换
        Variable output = hiddenActivated.dot(W2).add(b2);

        // 输出结果
        System.out.println("网络输出: " + output.getValue());
        System.out.println("输出形状: " + output.getValue().getShape());
    }
}
```

### 2. 封装成类

```java
public class SimpleNN {
    private Variable W1, b1, W2, b2;
    
    public SimpleNN(int inputSize, int hiddenSize, int outputSize) {
        // 初始化参数
        W1 = new Variable(NdArray.randn(Shape.of(inputSize, hiddenSize)));
        b1 = new Variable(NdArray.zeros(Shape.of(hiddenSize)));
        W2 = new Variable(NdArray.randn(Shape.of(hiddenSize, outputSize)));
        b2 = new Variable(NdArray.zeros(Shape.of(outputSize)));
        
        // 给参数命名（便于调试）
        W1.setName("W1");
        b1.setName("b1");
        W2.setName("W2");
        b2.setName("b2");
    }
    
    public Variable forward(Variable x) {
        // 第一层
        Variable hidden = x.dot(W1).add(b1).relu();
        
        // 第二层
        Variable output = hidden.dot(W2).add(b2);
        
        return output;
    }
    
    // 获取所有参数（用于优化器）
    public Variable[] getParameters() {
        return new Variable[]{W1, b1, W2, b2};
    }
}

// 使用
public class SimpleNNTest {
    public static void main(String[] args) {
        SimpleNN model = new SimpleNN(3, 4, 2);
        
        Variable input = new Variable(NdArray.of(new float[]{1.0f, 2.0f, 3.0f}));
        Variable output = model.forward(input);
        
        System.out.println("模型输出: " + output.getValue());
    }
}
```

### 3. 添加 Softmax 和预测

```java
public class ClassifierNN {
    private Variable W1, b1, W2, b2;
    
    public ClassifierNN(int inputSize, int hiddenSize, int numClasses) {
        W1 = new Variable(NdArray.randn(Shape.of(inputSize, hiddenSize)));
        b1 = new Variable(NdArray.zeros(Shape.of(hiddenSize)));
        W2 = new Variable(NdArray.randn(Shape.of(hiddenSize, numClasses)));
        b2 = new Variable(NdArray.zeros(Shape.of(numClasses)));
    }
    
    // 前向传播（返回 logits）
    public Variable forward(Variable x) {
        Variable hidden = x.dot(W1).add(b1).relu();
        Variable logits = hidden.dot(W2).add(b2);
        return logits;
    }
    
    // 预测（返回概率）
    public Variable predict(Variable x) {
        Variable logits = forward(x);
        return logits.softMax();
    }
    
    // 获取预测的类别
    public int predictClass(Variable x) {
        Variable probs = predict(x);
        return probs.getValue().argMax(-1).get(0).intValue();
    }
}

// 使用示例
public class ClassificationExample {
    public static void main(String[] args) {
        // 3分类问题，输入特征数为5
        ClassifierNN model = new ClassifierNN(5, 10, 3);
        
        // 单个样本
        Variable input = new Variable(NdArray.of(new float[]{0.5f, -0.3f, 1.2f, 0.8f, -0.1f}));
        
        // 预测概率
        Variable probs = model.predict(input);
        System.out.println("各类别概率: " + probs.getValue());
        
        // 预测类别
        int predictedClass = model.predictClass(input);
        System.out.println("预测类别: " + predictedClass);
    }
}
```

### 4. 处理批量数据

```java
public class BatchProcessing {
    public static void main(String[] args) {
        // 模型：输入4维，隐藏层8个神经元，输出3类
        SimpleNN model = new SimpleNN(4, 8, 3);
        
        // 批量输入：4个样本，每个样本4个特征
        Variable batchInput = new Variable(NdArray.of(new float[][]{
            {1.0f, 2.0f, 3.0f, 4.0f},
            {0.5f, 1.5f, 2.5f, 3.5f},
            {-1.0f, -2.0f, -3.0f, -4.0f},
            {0.0f, 0.0f, 0.0f, 0.0f}
        }));
        
        System.out.println("输入形状: " + batchInput.getValue().getShape());  // [4, 4]
        
        // 前向传播
        Variable batchOutput = model.forward(batchInput);
        
        System.out.println("输出形状: " + batchOutput.getValue().getShape());  // [4, 3]
        System.out.println("批量输出:\n" + batchOutput.getValue());
        
        // 对每个样本应用softmax
        Variable batchProbs = batchOutput.softMax();
        System.out.println("批量概率:\n" + batchProbs.getValue());
    }
}
```

### 5. 不同激活函数对比

```java
public class ActivationComparison {
    public static void main(String[] args) {
        Variable x = new Variable(NdArray.of(new float[]{-2.0f, -1.0f, 0.0f, 1.0f, 2.0f}));
        
        System.out.println("输入: " + x.getValue());
        System.out.println();
        
        // ReLU: max(0, x)
        Variable relu = x.relu();
        System.out.println("ReLU:   " + relu.getValue());
        
        // Sigmoid: 1/(1+e^(-x))
        Variable sigmoid = x.sigmoid();
        System.out.println("Sigmoid:" + sigmoid.getValue());
        
        // Tanh: (e^x - e^(-x))/(e^x + e^(-x))
        Variable tanh = x.tanh();
        System.out.println("Tanh:   " + tanh.getValue());
        
        // 对比总结
        System.out.println("\n=== 特性对比 ===");
        System.out.println("ReLU:   输出范围 [0, +∞)，计算简单，最常用");
        System.out.println("Sigmoid:输出范围 (0, 1)，适合二分类输出");
        System.out.println("Tanh:   输出范围 (-1, 1)，零中心化");
    }
}
```

### 6. 可视化网络结构

```java
public class NetworkVisualization {
    public static void visualizeNetwork(int inputSize, int hiddenSize, int outputSize) {
        System.out.println("神经网络结构可视化");
        System.out.println("==================");
        System.out.println();
        
        // 输入层
        System.out.println("输入层 (" + inputSize + " 个神经元)");
        for (int i = 0; i < inputSize; i++) {
            System.out.print("  ○  ");
        }
        System.out.println("\n");
        
        // 连接线
        System.out.println("  ↓  " + inputSize + "×" + hiddenSize + " 权重矩阵");
        System.out.println();
        
        // 隐藏层
        System.out.println("隐藏层 (" + hiddenSize + " 个神经元, ReLU激活)");
        for (int i = 0; i < hiddenSize; i++) {
            System.out.println("  ◎  ");
        }
        System.out.println();
        
        // 连接线
        System.out.println("  ↓  " + hiddenSize + "×" + outputSize + " 权重矩阵");
        System.out.println();
        
        // 输出层
        System.out.println("输出层 (" + outputSize + " 个神经元)");
        for (int i = 0; i < outputSize; i++) {
            System.out.print("  ○  ");
        }
        System.out.println();
        
        // 参数统计
        int totalParams = inputSize * hiddenSize + hiddenSize + 
                         hiddenSize * outputSize + outputSize;
        System.out.println("\n总参数量: " + totalParams);
        System.out.println("  - W1: " + (inputSize * hiddenSize));
        System.out.println("  - b1: " + hiddenSize);
        System.out.println("  - W2: " + (hiddenSize * outputSize));
        System.out.println("  - b2: " + outputSize);
    }
    
    public static void main(String[] args) {
        visualizeNetwork(784, 128, 10);
    }
}
```

---

## 练习挑战

### 练习 1：基础（必做）

创建一个神经网络，实现 XOR（异或）功能：
- 输入：2维（0或1）
- 隐藏层：4个神经元，ReLU激活
- 输出：1维，Sigmoid激活
- 测试所有4种输入组合

<details>
<summary>点击查看答案</summary>

```java
public class XorNetwork {
    private Variable W1, b1, W2, b2;
    
    public XorNetwork() {
        // 注意：这里使用固定权重来演示XOR功能
        // 实际训练时会学习这些权重
        W1 = new Variable(NdArray.of(new float[][]{
            {10, 10, -10, -10},
            {10, -10, 10, -10}
        }));
        b1 = new Variable(NdArray.of(new float[]{-5, -15, -15, 5}));
        W2 = new Variable(NdArray.of(new float[][]{
            {10}, {10}, {10}, {-10}
        }));
        b2 = new Variable(NdArray.of(new float[]{-5}));
    }
    
    public Variable forward(Variable x) {
        Variable hidden = x.dot(W1).add(b1).relu();
        Variable output = hidden.dot(W2).add(b2).sigmoid();
        return output;
    }
    
    public static void main(String[] args) {
        XorNetwork model = new XorNetwork();
        
        // 测试所有4种输入组合
        float[][] inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        
        System.out.println("XOR 测试结果:");
        for (float[] input : inputs) {
            Variable x = new Variable(NdArray.of(input));
            Variable output = model.forward(x);
            float prob = output.getValue().getNumber().floatValue();
            System.out.printf("%.0f XOR %.0f = %.3f (%.0f)%n", 
                input[0], input[1], prob, prob > 0.5 ? 1 : 0);
        }
    }
}
```

</details>

### 练习 2：进阶（推荐）

实现一个三层神经网络：
- 输入层：10维
- 隐藏层1：64维，ReLU
- 隐藏层2：32维，ReLU
- 输出层：5维，Softmax

要求：
1. 封装成类
2. 支持批量输入
3. 添加参数统计方法

<details>
<summary>点击查看答案</summary>

```java
public class ThreeLayerNN {
    private Variable W1, b1, W2, b2, W3, b3;
    
    public ThreeLayerNN(int inputSize, int h1Size, int h2Size, int outputSize) {
        W1 = new Variable(NdArray.randn(Shape.of(inputSize, h1Size)));
        b1 = new Variable(NdArray.zeros(Shape.of(h1Size)));
        W2 = new Variable(NdArray.randn(Shape.of(h1Size, h2Size)));
        b2 = new Variable(NdArray.zeros(Shape.of(h2Size)));
        W3 = new Variable(NdArray.randn(Shape.of(h2Size, outputSize)));
        b3 = new Variable(NdArray.zeros(Shape.of(outputSize)));
    }
    
    public Variable forward(Variable x) {
        Variable h1 = x.dot(W1).add(b1).relu();
        Variable h2 = h1.dot(W2).add(b2).relu();
        Variable output = h3.dot(W3).add(b3);
        return output;
    }
    
    public int getParameterCount() {
        int count = 0;
        count += W1.getValue().size();
        count += b1.getValue().size();
        count += W2.getValue().size();
        count += b2.getValue().size();
        count += W3.getValue().size();
        count += b3.getValue().size();
        return count;
    }
    
    public static void main(String[] args) {
        ThreeLayerNN model = new ThreeLayerNN(10, 64, 32, 5);
        System.out.println("总参数量: " + model.getParameterCount());
        
        // 批量测试
        Variable batch = new Variable(NdArray.randn(Shape.of(8, 10)));
        Variable output = model.forward(batch);
        System.out.println("输出形状: " + output.getValue().getShape());
    }
}
```

</details>

### 练习 3：挑战（选做）

实现一个可以动态配置层数的神经网络：
- 构造函数接收层大小数组，如 `[784, 256, 128, 10]`
- 自动创建相应的权重和偏置
- 支持选择每层的激活函数

---

## 常见问题

**Q1: 隐藏层的大小如何选择？**

A: 一般经验：
- 太小：模型欠拟合，无法学习复杂模式
- 太大：模型过拟合，计算量大
- 通常：输入层和输出层之间，或参考相似任务的文献

**Q2: 为什么要用 ReLU 而不是 Sigmoid？**

A: 
- ReLU 计算更快（没有指数运算）
- ReLU 缓解梯度消失问题（正数区域梯度为1）
- Sigmoid 在深层网络中容易导致梯度消失

**Q3: 权重为什么要随机初始化？**

A: 
- 如果全初始化为0，所有神经元学习相同的东西
- 随机初始化打破对称性，让不同神经元学习不同特征
- 通常使用小随机数（如标准正态分布乘以0.01）

**Q4: 偏置为什么要初始化为0？**

A: 偏置初始化为0是安全的，因为：
- 权重已经随机，不会对称
- 偏置会在训练中自动调整

---

## 下一步

你已经学会了构建神经网络！接下来学习：

**[04-first-training.md](04-first-training.md)** - 完成第一次模型训练。

---

## 参考资源

- [神经网络基础](../part1-deep-learning/chapter05-neural-network-blocks/)
- [激活函数详解](../deep-dive/activation-functions.md)
- [Block API 文档](../../tinyai-deeplearning/tinyai-deeplearning-nnet/)
