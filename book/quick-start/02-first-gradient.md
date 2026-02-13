# 快速入门 02：理解自动微分

> **学习目标**
> 1. 理解什么是梯度以及为什么需要它
> 2. 掌握 Variable 的基本使用
> 3. 学会使用反向传播计算梯度

---

## 理论速览（5分钟）

### 什么是梯度？

想象你在山上，想要最快地下山。梯度告诉你：**哪个方向最陡，以及有多陡**。

在数学上，梯度是函数在某一点处的导数（多元函数是偏导数）。

```
函数: f(x) = x²
导数: f'(x) = 2x

在 x = 3 处: f'(3) = 6
这意味着：如果 x 增加一点点，f(x) 会增加约 6 倍
```

### 为什么深度学习需要梯度？

神经网络训练的本质是**优化问题**：
1. 定义一个损失函数（衡量预测有多差）
2. 计算损失函数对每个参数的梯度
3. 沿着梯度反方向更新参数（梯度下降）
4. 重复直到收敛

### 自动微分 vs 手动求导

| 方法 | 优点 | 缺点 |
|------|------|------|
| 手动求导 | 精确 | 容易出错，复杂函数难以处理 |
| 数值微分 | 实现简单 | 精度低，计算量大 |
| **自动微分** | 精确、高效、通用 | 需要框架支持 |

**自动微分**是深度学习框架的核心技术。

---

## 代码实践

### 1. 创建 Variable

```java
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;

public class VariableQuickStart {
    public static void main(String[] args) {
        // 方法1：从标量创建
        Variable x = new Variable(NdArray.of(3.0f));
        x.setName("x");
        
        // 方法2：从数组创建
        Variable weights = new Variable(NdArray.randn(Shape.of(3, 4)));
        weights.setName("weights");
        
        // 方法3：创建不需要梯度的变量（如输入数据）
        Variable input = new Variable(NdArray.of(new float[]{1, 2, 3}));
        input.setRequireGrad(false);  // 输入数据不需要梯度
        
        System.out.println("x的值: " + x.getValue());
        System.out.println("x是否需要梯度: " + x.isRequireGrad());
    }
}
```

### 2. 基本运算与梯度计算

```java
public class BasicGradient {
    public static void main(String[] args) {
        // 创建变量
        Variable x = new Variable(NdArray.of(2.0f));
        Variable y = new Variable(NdArray.of(3.0f));
        
        // 前向传播：构建计算图
        // z = x * y
        Variable z = x.mul(y);
        
        // 反向传播：计算梯度
        z.backward();
        
        // 查看梯度
        System.out.println("∂z/∂x = " + x.getGrad());  // 3.0
        System.out.println("∂z/∂y = " + y.getGrad());  // 2.0
        
        // 解释：
        // z = x * y
        // ∂z/∂x = y = 3
        // ∂z/∂y = x = 2
    }
}
```

### 3. 复合函数的梯度

```java
public class CompositeFunction {
    public static void main(String[] args) {
        Variable x = new Variable(NdArray.of(2.0f));
        
        // 构建复合函数: y = (x + 1)²
        Variable temp = x.add(new Variable(NdArray.of(1.0f)));
        Variable y = temp.square();
        
        // 反向传播
        y.backward();
        
        System.out.println("y = " + y.getValue());       // 9.0
        System.out.println("∂y/∂x = " + x.getGrad());   // 6.0
        
        // 数学推导：
        // y = (x + 1)²
        // ∂y/∂x = 2 * (x + 1) = 2 * 3 = 6
    }
}
```

### 4. 链式法则

```java
public class ChainRule {
    public static void main(String[] args) {
        Variable x = new Variable(NdArray.of(2.0f));
        
        // 构建: y = (x² + 1)²
        // 分解：u = x², v = u + 1, y = v²
        Variable u = x.square();                              // u = x² = 4
        Variable v = u.add(new Variable(NdArray.of(1.0f)));   // v = u + 1 = 5
        Variable y = v.square();                              // y = v² = 25
        
        y.backward();
        
        System.out.println("y = " + y.getValue());       // 25.0
        System.out.println("∂y/∂x = " + x.getGrad());   // 40.0
        
        // 链式法则推导：
        // ∂y/∂x = ∂y/∂v * ∂v/∂u * ∂u/∂x
        //       = 2v * 1 * 2x
        //       = 2*5 * 2*2
        //       = 10 * 4 = 40
    }
}
```

### 5. 多变量函数的梯度

```java
public class MultiVariable {
    public static void main(String[] args) {
        Variable x = new Variable(NdArray.of(2.0f));
        Variable y = new Variable(NdArray.of(3.0f));
        Variable z = new Variable(NdArray.of(4.0f));
        
        // 构建: w = x*y + x*z + y*z
        Variable xy = x.mul(y);
        Variable xz = x.mul(z);
        Variable yz = y.mul(z);
        Variable w = xy.add(xz).add(yz);
        
        w.backward();
        
        System.out.println("w = " + w.getValue());       // 26.0
        System.out.println("∂w/∂x = " + x.getGrad());   // 7.0 (y + z)
        System.out.println("∂w/∂y = " + y.getGrad());   // 6.0 (x + z)
        System.out.println("∂w/∂z = " + z.getGrad());   // 5.0 (x + y)
    }
}
```

### 6. 梯度清零

```java
public class GradientAccumulation {
    public static void main(String[] args) {
        Variable x = new Variable(NdArray.of(2.0f));
        
        // 第一次前向传播
        Variable y1 = x.mul(new Variable(NdArray.of(3.0f)));
        y1.backward();
        System.out.println("第一次梯度: " + x.getGrad());  // 3.0
        
        // 第二次前向传播（梯度会累加！）
        Variable y2 = x.mul(new Variable(NdArray.of(4.0f)));
        y2.backward();
        System.out.println("第二次梯度（累加）: " + x.getGrad());  // 7.0 (3+4)
        
        // 正确做法：每次反向传播前清零梯度
        x.zeroGrad();
        Variable y3 = x.mul(new Variable(NdArray.of(5.0f)));
        y3.backward();
        System.out.println("清零后的梯度: " + x.getGrad());  // 5.0
    }
}
```

---

## 练习挑战

### 练习 1：基础（必做）

给定函数 `f(x) = x³ + 2x² + 3x + 4`，计算在 `x = 2` 处的导数。

<details>
<summary>点击查看答案</summary>

```java
Variable x = new Variable(NdArray.of(2.0f));

// f(x) = x³ + 2x² + 3x + 4
Variable x2 = x.square();           // x²
Variable x3 = x2.mul(x);            // x³
Variable term1 = x3;                // x³
Variable term2 = x2.mul(NdArray.of(2.0f));  // 2x²
Variable term3 = x.mul(NdArray.of(3.0f));   // 3x
Variable term4 = new Variable(NdArray.of(4.0f));  // 4

Variable f = term1.add(term2).add(term3).add(term4);
f.backward();

System.out.println("f(2) = " + f.getValue());      // 26.0
System.out.println("f'(2) = " + x.getGrad());      // 23.0

// 数学验证：f'(x) = 3x² + 4x + 3
// f'(2) = 3*4 + 4*2 + 3 = 12 + 8 + 3 = 23
```

</details>

### 练习 2：进阶（推荐）

实现一个简单的线性回归梯度计算：
- 模型: `y_pred = w * x + b`
- 损失: `loss = (y_pred - y_true)²`
- 给定: x=2, y_true=5, w=1, b=0
- 计算: ∂loss/∂w 和 ∂loss/∂b

<details>
<summary>点击查看答案</summary>

```java
// 数据
Variable x = new Variable(NdArray.of(2.0f));
Variable y_true = new Variable(NdArray.of(5.0f));

// 参数（需要梯度）
Variable w = new Variable(NdArray.of(1.0f));
Variable b = new Variable(NdArray.of(0.0f));

// 前向传播
Variable y_pred = x.mul(w).add(b);  // y_pred = 1*2 + 0 = 2
Variable diff = y_pred.sub(y_true); // diff = 2 - 5 = -3
Variable loss = diff.square();      // loss = 9

// 反向传播
loss.backward();

System.out.println("loss = " + loss.getValue());   // 9.0
System.out.println("∂loss/∂w = " + w.getGrad());   // -12.0
System.out.println("∂loss/∂b = " + b.getGrad());   // -6.0

// 数学推导：
// loss = (w*x + b - y_true)²
// ∂loss/∂w = 2*(w*x + b - y_true)*x = 2*(-3)*2 = -12
// ∂loss/∂b = 2*(w*x + b - y_true) = 2*(-3) = -6
```

</details>

### 练习 3：挑战（选做）

实现逻辑回归的梯度计算：
- 模型: `p = sigmoid(w*x + b)`
- 损失: `loss = -[y*log(p) + (1-y)*log(1-p)]` （二元交叉熵）
- 给定: x=1, y=1, w=0, b=0
- 计算梯度并更新参数（学习率0.1）

---

## 常见问题

**Q1: Variable 和 NdArray 有什么区别？**

A: 
- **NdArray**：纯数值容器，用于存储数据
- **Variable**：封装了 NdArray + 梯度 + 计算历史，支持自动微分

类比：
- NdArray 像 `float[]`
- Variable 像 PyTorch 的 `Tensor`（requires_grad=True）

**Q2: 为什么需要 `zeroGrad()`？**

A: 默认情况下，梯度会累加而不是替换。这在某些场景（如梯度累积）中有用，但通常每次迭代前需要清零。

**Q3: 什么时候设置 `requireGrad = false`？**

A: 当变量不需要计算梯度时，例如：
- 输入数据
- 标签（target）
- 冻结的模型参数

这样可以节省内存和计算。

**Q4: 如何理解计算图？**

A: 计算图是有向无环图（DAG），节点是变量，边是运算。反向传播时，梯度沿着图的边从输出流向输入。

```
前向：x → [×] → z → [+] → loss
            ↑           ↑
            y           w

反向：loss.grad → z.grad → x.grad
                          → y.grad
```

---

## 下一步

你已经掌握了自动微分的核心概念！接下来学习：

**[03-first-model.md](03-first-model.md)** - 搭建你的第一个神经网络。

---

## 参考资源

- [自动微分原理详解](../part1-deep-learning/chapter03-autograd-engine/)
- [反向传播的数学推导](../deep-dive/backpropagation-math.md)
- [Variable API 文档](../../tinyai-deeplearning/tinyai-deeplearning-func/)
