# 快速入门 01：10分钟上手多维数组

> **学习目标**
> 1. 理解 NdArray 是什么以及为什么需要它
> 2. 掌握 NdArray 的基本创建方法
> 3. 学会基本的数据操作和运算

---

## 理论速览（5分钟）

### 什么是 NdArray？

想象你手里有一张照片。从计算机的角度看，这张照片其实是一个**三维数组**：
- 第1维：高度（像素行数）
- 第2维：宽度（像素列数）
- 第3维：颜色通道（红、绿、蓝）

如果是一段视频，那就是**四维**：时间 + 高度 + 宽度 + 颜色通道。

**NdArray（N-dimensional Array）** 就是用来存储这种多维数据的容器。

### 维度层次

```
标量(0D)    → 向量(1D)      → 矩阵(2D)        → 张量(3D+)
  5.0          [1, 2, 3]      [[1, 2],         [[[1, 2],
                              [3, 4]]           [[3, 4]]]
```

### 为什么要用 NdArray？

| 场景 | 传统Java | NdArray |
|------|----------|---------|
| 存储100张28×28图像 | `float[100][28][28]` | `NdArray` |
| 矩阵乘法 | 三重循环 | `a.dot(b)` |
| 广播运算 | 手动扩展 | 自动处理 |
| GPU加速 | 不支持 | 支持 |

---

## 代码实践

### 1. 创建 NdArray

```java
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

public class NdArrayQuickStart {
    public static void main(String[] args) {
        // 方法1：从标量创建
        NdArray scalar = NdArray.of(3.14f);
        System.out.println("标量: " + scalar.getNumber());  // 3.14

        // 方法2：从一维数组创建（向量）
        NdArray vector = NdArray.of(new float[]{1, 2, 3, 4, 5});
        System.out.println("向量形状: " + vector.getShape());  // Shape[5]

        // 方法3：从二维数组创建（矩阵）
        NdArray matrix = NdArray.of(new float[][]{
            {1, 2, 3},
            {4, 5, 6}
        });
        System.out.println("矩阵形状: " + matrix.getShape());  // Shape[2, 3]

        // 方法4：创建特殊数组
        NdArray zeros = NdArray.zeros(Shape.of(3, 4));      // 全零
        NdArray ones = NdArray.ones(Shape.of(2, 2));        // 全一
        NdArray random = NdArray.randn(Shape.of(3, 3));     // 随机数（正态分布）
        
        System.out.println("随机矩阵:\n" + random);
    }
}
```

### 2. 基本运算

```java
public class BasicOperations {
    public static void main(String[] args) {
        NdArray a = NdArray.of(new float[][]{{1, 2}, {3, 4}});
        NdArray b = NdArray.of(new float[][]{{5, 6}, {7, 8}});

        // 元素级运算
        NdArray add = a.add(b);      // [[6, 8], [10, 12]]
        NdArray sub = a.sub(b);      // [[-4, -4], [-4, -4]]
        NdArray mul = a.mul(b);      // [[5, 12], [21, 32]]
        NdArray div = a.div(b);      // [[0.2, 0.33], [0.43, 0.5]]

        // 矩阵乘法（点积）
        NdArray c = NdArray.of(new float[][]{{1, 2}, {3, 4}, {5, 6}});  // 3×2
        NdArray d = NdArray.of(new float[][]{{7, 8, 9}, {10, 11, 12}}); // 2×3
        NdArray dot = c.dot(d);     // 结果：3×3

        System.out.println("矩阵乘法结果形状: " + dot.getShape());
    }
}
```

### 3. 形状操作

```java
public class ShapeOperations {
    public static void main(String[] args) {
        NdArray a = NdArray.of(new float[][]{{1, 2, 3}, {4, 5, 6}});
        System.out.println("原始形状: " + a.getShape());  // [2, 3]

        // 转置
        NdArray transposed = a.transpose();
        System.out.println("转置后: " + transposed.getShape());  // [3, 2]

        // 变形（元素总数不变）
        NdArray reshaped = a.reshape(Shape.of(3, 2));
        System.out.println("变形后: " + reshaped.getShape());  // [3, 2]

        // 展平
        NdArray flattened = a.flatten();
        System.out.println("展平后: " + flattened.getShape());  // [6]
    }
}
```

### 4. 聚合操作

```java
public class AggregationOperations {
    public static void main(String[] args) {
        NdArray a = NdArray.of(new float[][]{
            {1, 2, 3},
            {4, 5, 6}
        });

        // 求和
        NdArray totalSum = a.sum();           // 21
        NdArray rowSum = a.sum(1);            // [6, 15]（沿行求和）
        NdArray colSum = a.sum(0);            // [5, 7, 9]（沿列求和）

        // 均值
        NdArray mean = a.mean(0);             // [2.5, 3.5, 4.5]

        // 最大值及索引
        NdArray maxValues = a.max(1);         // [3, 6]
        NdArray maxIndices = a.argMax(1);     // [2, 2]

        System.out.println("总和: " + totalSum.getNumber());
        System.out.println("每行和: " + rowSum);
    }
}
```

### 5. 广播机制

```java
public class Broadcasting {
    public static void main(String[] args) {
        NdArray matrix = NdArray.of(new float[][]{
            {1, 2, 3},
            {4, 5, 6}
        });  // 形状: [2, 3]

        // 标量广播：矩阵 + 标量
        NdArray addScalar = matrix.add(NdArray.of(10));
        System.out.println("加标量:\n" + addScalar);
        // [[11, 12, 13],
        //  [14, 15, 16]]

        // 向量广播：矩阵 + 行向量
        NdArray rowVector = NdArray.of(new float[]{1, 2, 3});
        NdArray addRow = matrix.add(rowVector);
        System.out.println("加行向量:\n" + addRow);
        // [[2, 4, 6],
        //  [5, 7, 9]]

        // 列向量广播
        NdArray colVector = NdArray.of(new float[][]{{10}, {20}});
        NdArray addCol = matrix.add(colVector);
        System.out.println("加列向量:\n" + addCol);
        // [[11, 12, 13],
        //  [24, 25, 26]]
    }
}
```

---

## 练习挑战

### 练习 1：基础（必做）
创建一个 3×3 的单位矩阵，然后：
1. 将其每个元素乘以 5
2. 计算每行的和
3. 计算整个矩阵的均值

<details>
<summary>点击查看答案</summary>

```java
NdArray eye = NdArray.eye(Shape.of(3, 3));
NdArray scaled = eye.mul(NdArray.of(5));
NdArray rowSums = scaled.sum(1);
NdArray mean = scaled.mean(-1);

System.out.println("缩放后的矩阵:\n" + scaled);
System.out.println("每行和: " + rowSums);
System.out.println("均值: " + mean.getNumber());
```

</details>

### 练习 2：进阶（推荐）
实现一个函数，计算两个矩阵的欧几里得距离（逐元素差的平方和开根号）。

<details>
<summary>点击查看答案</summary>

```java
public static float euclideanDistance(NdArray a, NdArray b) {
    NdArray diff = a.sub(b);
    NdArray squared = diff.square();
    NdArray sum = squared.sum();
    return (float) Math.sqrt(sum.getNumber().doubleValue());
}

// 使用示例
NdArray a = NdArray.of(new float[][]{{1, 2}, {3, 4}});
NdArray b = NdArray.of(new float[][]{{2, 3}, {4, 5}});
float distance = euclideanDistance(a, b);
System.out.println("欧几里得距离: " + distance);  // 2.0
```

</details>

### 练习 3：挑战（选做）
使用 NdArray 实现一个简单的图像归一化函数：
- 输入：一批图像数据，形状为 [batch, height, width]
- 输出：归一化后的数据，每个像素的值在 [0, 1] 之间

---

## 常见问题

**Q1: NdArray 和 Java 数组有什么区别？**

A: NdArray 是专门为数值计算设计的，支持：
- 任意维度的数据
- 广播机制
- 丰富的数学运算
- 自动微分（配合 Variable 使用）

**Q2: 如何查看 NdArray 的内容？**

A: 使用 `toString()` 方法或 `getMatrix()` 方法：
```java
System.out.println(ndArray);  // 打印整个数组
float[][] matrix = ndArray.getMatrix();  // 转换为Java二维数组
```

**Q3: 广播机制的规则是什么？**

A: 从右往左比较维度，满足以下条件之一即可广播：
- 维度相等
- 其中一个维度为 1
- 其中一个数组维度较少（视为前面补1）

---

## 下一步

恭喜！你已经掌握了 NdArray 的基础知识。接下来学习：

**[02-first-gradient.md](02-first-gradient.md)** - 理解自动微分，这是深度学习的核心机制。

---

## 参考资源

- [完整 API 文档](../part1-deep-learning/chapter02-ndarray-core/2.1-ndarray-foundation.md)
- [内存布局详解](../part1-deep-learning/chapter02-ndarray-core/2.2-memory-layout.md)
- [广播机制详解](../part1-deep-learning/chapter02-ndarray-core/2.3-broadcasting.md)
