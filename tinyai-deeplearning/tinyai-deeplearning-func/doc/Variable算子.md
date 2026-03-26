# Variable 算子参考文档

> 本文档基于 `Variable.java` 实际代码整理，准确反映当前已实现的所有算子。

---

## 一、Variable 概述

`Variable` 是 TinyAI 自动微分系统的核心单元，封装了**值（NdArray）+ 梯度 + 计算图**。
每次算子调用都会构建计算图节点，支持链式自动微分。

| 特性        | NdArray       | Variable                  |
|-----------|---------------|---------------------------|
| 存储内容      | 纯数值           | 数值 + 梯度 + 计算历史             |
| 是否可微      | 否             | 是                         |
| 使用场景      | 数据存储、数值计算     | 模型参数、中间激活值                |
| 类比        | numpy.array   | torch.Tensor              |

---

## 二、构造方法

| 方法签名                                              | 说明                     |
|---------------------------------------------------|------------------------|
| `new Variable(NdArray value)`                     | 通用构造，需要梯度              |
| `new Variable(Number number)`                     | 标量构造（自动包装为NdArray）     |
| `new Variable(NdArray value, String name)`        | 带名称的构造                 |
| `new Variable(NdArray value, String name, boolean requireGrad)` | 完整构造，可指定是否需要梯度         |

---

## 三、形状与属性方法

| 方法签名                   | 说明                               | PyTorch 对应                 |
|------------------------|----------------------------------|-----------------------------|
| `ndim()`               | 获取维度数量                           | `tensor.ndim`               |
| `size(int dim)`        | 获取指定维度大小，支持负数索引（-1 表示最后一维）      | `tensor.size(dim)`          |
| `sizes()`              | 获取所有维度大小数组                       | `tensor.size()`             |
| `shape()`              | `sizes()` 的别名，返回 `int[]`         | `tensor.shape`              |
| `getElementCount()`    | 获取张量中元素的总数                       | `tensor.numel()`            |
| `isScalar()`           | 判断是否为标量（元素数等于1）                  | `tensor.dim() == 0`         |
| `isMatrix()`           | 判断是否为矩阵（2维张量）                    | `tensor.dim() == 2`         |
| `isVector()`           | 判断是否为向量（1维张量）                    | `tensor.dim() == 1`         |
| `dtype()`              | 返回数据类型描述（当前固定为 `"float32"`）     | `tensor.dtype`              |
| `getShape()`           | 获取 `Shape` 对象                    | `tensor.shape`              |
| `getValue()`           | 获取底层 `NdArray` 值               | `tensor.data`               |
| `getGrad()`            | 获取梯度                             | `tensor.grad`               |

---

## 四、自动微分

| 方法签名                   | 说明                                         |
|------------------------|--------------------------------------------|
| `backward()`           | 递归反向传播，从当前变量开始计算所有上游梯度                     |
| `backwardIterative()`  | 迭代式反向传播（栈模拟递归），适用于深层网络避免栈溢出               |
| `unChainBackward()`    | 切断计算图（RNN 截断 BPTT 时使用）                     |
| `clearGrad()`          | 将梯度置为 null，释放内存（训练迭代前调用）                   |
| `setRequireGrad(boolean)` | 设置是否需要梯度，支持链式调用                            |
| `isRequireGrad()`      | 获取是否需要梯度                                   |

---

## 五、四则运算

| 方法签名                    | 说明        | 数学表达                  |
|-------------------------|-----------|----------------------|
| `add(Variable other)`   | 加法        | `this + other`       |
| `sub(Variable other)`   | 减法        | `this - other`       |
| `mul(Variable other)`   | 乘法（逐元素）   | `this * other`       |
| `div(Variable other)`   | 除法（逐元素）   | `this / other`       |
| `neg()`                 | 取反        | `-this`              |

---

## 六、基础数学函数

| 方法签名             | 说明              | 数学表达                       |
|------------------|-----------------|----------------------------|
| `squ()`          | 平方              | `x²`                       |
| `pow(float pow)` | 幂运算             | `x^pow`                    |
| `exp()`          | 自然指数            | `e^x`                      |
| `log()`          | 自然对数            | `ln(x)`                    |
| `sqrt()`         | 平方根             | `√x`                       |
| `sin()`          | 正弦              | `sin(x)`                   |
| `cos()`          | 余弦              | `cos(x)`                   |
| `tanh()`         | 双曲正切            | `tanh(x)`                  |
| `sigmoid()`      | Sigmoid          | `1 / (1 + e^{-x})`         |
| `softMax()`      | Softmax（全局归一化）  | `e^{x_i} / Σe^{x_j}`      |

---

## 七、激活函数

| 方法签名                                   | 说明                                     |
|----------------------------------------|----------------------------------------|
| `relu()`                               | ReLU：`max(0, x)`                       |
| `gelu()`                               | GELU：高斯误差线性单元，GPT 常用                   |
| `silu()`                               | SiLU / Swish：`x * sigmoid(x)`，LLM 常用  |
| `leakyRelu(float negativeSlope)`       | LeakyReLU，负区间斜率可配置                     |
| `leakyRelu()`                          | LeakyReLU，默认负斜率 `0.01`                 |
| `elu(float alpha)`                     | ELU：`x >= 0 ? x : alpha*(e^x - 1)`    |
| `elu()`                                | ELU，默认 `alpha=1.0`                     |
| `logSoftmax(int axis)`                 | LogSoftmax，沿指定轴计算                      |
| `logSoftmax()`                         | LogSoftmax，默认 axis=-1（最后一维）            |
| `clip(float min, float max)`           | 裁剪：将值限制在 `[min, max]` 范围内              |

---

## 八、统计与归约

| 方法签名                              | 说明                    | PyTorch 对应                      |
|-----------------------------------|-----------------------|---------------------------------|
| `sum()`                           | 全局求和                  | `tensor.sum()`                  |
| `sumTo(Shape shape)`              | 归约到指定形状（广播的逆操作）       | —                               |
| `mean(int axis, boolean keepdims)` | 沿指定轴均值                | `tensor.mean(dim, keepdim)`     |
| `max(int axis, boolean keepdims)` | 沿指定轴最大值               | `tensor.max(dim, keepdim)`      |
| `min(int axis, boolean keepdims)` | 沿指定轴最小值               | `tensor.min(dim, keepdim)`      |
| `var(int axis, boolean keepdims)` | 沿指定轴方差                | `tensor.var(dim, keepdim)`      |

---

## 九、矩阵与形状操作

| 方法签名                                    | 说明                                         | PyTorch 对应                            |
|-----------------------------------------|--------------------------------------------|---------------------------------------|
| `matMul(Variable other)`                | 矩阵乘法（2D）                                   | `torch.mm` / `tensor @ other`         |
| `bmm(Variable other)`                   | 批量矩阵乘法：`[B,N,M] @ [B,M,P] -> [B,N,P]`    | `torch.bmm`                           |
| `transpose()`                           | 矩阵转置                                       | `tensor.T`                            |
| `reshape(Shape shape)`                  | 重塑形状                                       | `tensor.reshape`                      |
| `broadcastTo(Shape shape)`              | 广播到目标形状                                    | `tensor.expand`                       |
| `expand(Shape shape)`                   | 扩展维度为1的轴（不复制数据）                            | `tensor.expand`                       |
| `repeat(int... repeats)`                | 沿各维度重复（复制数据）                               | `tensor.repeat`                       |
| `squeeze()`                             | 移除所有大小为1的维度                                | `tensor.squeeze()`                    |
| `squeeze(int dim)`                      | 移除指定维度（该维度大小必须为1），支持负数索引                   | `tensor.squeeze(dim)`                 |
| `tril()`                                | 下三角矩阵，主对角线及以下保留，其余置0                      | `torch.tril`                          |
| `tril(int k)`                           | 下三角矩阵，`k` 控制对角线偏移                          | `torch.tril(input, diagonal=k)`       |
| `linear(Variable w, Variable b)`        | 线性变换 `y = xW + b`，`b` 可为 null             | `F.linear`                            |

---

## 十、索引与切片

| 方法签名                                          | 说明                                  | PyTorch 对应                            |
|-----------------------------------------------|-------------------------------------|-----------------------------------------|
| `getItem(int[] rowSlices, int[] colSlices)`   | 行列索引取子集                             | `tensor[rows, cols]`                   |
| `slice(int[] rowSlices, int[] colSlices)`     | `getItem` 的别名                       | `tensor[rows, cols]`                   |
| `select(int dim, int index)`                  | 在指定维度上选择单个索引，结果维度减1                | `tensor.select(dim, index)`            |
| `sliceRange(int dim, int start, int end)`     | 在指定维度上进行范围切片 `[start, end)`        | `tensor[start:end]`                    |
| `split(int splitSize, int dim)`               | 沿指定维度按大小分割，返回 `Variable[]`         | `torch.split`                          |
| `indexSelect(int dim, Variable index)`        | 沿指定维度按索引张量选取元素                     | `torch.index_select`                   |
| `scatterAdd(int dim, Variable index, Variable src)` | 将 `src` 按 `index` 分散累加到目标张量       | `tensor.scatter_add_`                  |

---

## 十一、条件与比较

| 方法签名                                           | 说明                          | PyTorch 对应               |
|------------------------------------------------|-----------------------------|--------------------------|
| `where(Variable cond, Variable x, Variable y)` | 静态方法：`cond` 为 true 选 x，否则选 y | `torch.where`            |
| `maskedFill(Variable mask, float value)`       | mask 为 true 的位置用 `value` 填充  | `tensor.masked_fill`     |
| `gt(Variable other)`                           | 逐元素大于比较，结果不参与梯度             | `tensor.gt`              |
| `lt(Variable other)`                           | 逐元素小于比较，结果不参与梯度             | `tensor.lt`              |
| `eq(Variable other)`                           | 逐元素等于比较，结果不参与梯度             | `tensor.eq`              |

---

## 十二、归一化

| 方法签名                                                 | 说明                                             | PyTorch 对应                 |
|------------------------------------------------------|------------------------------------------------|----------------------------|
| `rmsNorm(int[] normalizedShape, float eps, Variable weight)` | RMSNorm：`x / sqrt(mean(x²)+eps) * weight`    | `torch.nn.RMSNorm`         |
| `rmsNorm(int[] normalizedShape, Variable weight)`    | RMSNorm，默认 `eps=1e-6`                          | `torch.nn.RMSNorm`         |

---

## 十三、损失函数

| 方法签名                              | 说明                | PyTorch 对应                  |
|-------------------------------------|-------------------|-----------------------------|
| `meanSquaredError(Variable other)`  | 均方误差损失            | `F.mse_loss`                |
| `softmaxCrossEntropy(Variable other)` | Softmax 交叉熵损失    | `F.cross_entropy`           |

---

## 十四、卷积操作

| 方法签名                                          | 说明                                                      | PyTorch 对应          |
|-----------------------------------------------|---------------------------------------------------------|---------------------|
| `conv2d(Variable kernel, int stride, int padding)` | 2D卷积，输入 `[B,C,H,W]`，卷积核 `[Cout,Cin,kH,kW]` | `F.conv2d`          |
| `conv2d(Variable kernel)`                     | 2D卷积，默认 `stride=1, padding=0`                          | `F.conv2d`          |

---

## 十五、工具方法

| 方法签名                       | 说明                                | PyTorch 对应                     |
|----------------------------|------------------------------------|--------------------------------|
| `onesLike()`               | 创建与当前张量同形状的全1张量                    | `torch.ones_like`              |
| `zerosLike()`              | 创建与当前张量同形状的全0张量                    | `torch.zeros_like`             |
| `fullLike(float value)`    | 创建与当前张量同形状、以指定值填充的张量               | `torch.full_like`              |
| `fill(float value)`        | `fullLike` 的别名                     | `torch.full_like`              |
| `newLike()`                | 创建同形状空张量（内部返回全零）                   | `tensor.new_empty`             |
| `detach()`                 | 从计算图分离，停止梯度传播                      | `tensor.detach()`              |
| `clone()`                  | 深拷贝张量值，返回新叶子节点                     | `tensor.clone()`               |

---

## 十六、静态创建方法

| 方法签名                                    | 说明                          | PyTorch 对应         |
|-------------------------------------------|-----------------------------|----------------------|
| `Variable.zeros(Shape shape)`             | 全零张量                        | `torch.zeros`        |
| `Variable.ones(Shape shape)`              | 全一张量                        | `torch.ones`         |
| `Variable.full(Shape shape, float value)` | 指定值常量张量                     | `torch.full`         |
| `Variable.rand(Shape shape)`              | 均匀分布随机张量 `[0, 1)`，不需要梯度    | `torch.rand`         |
| `Variable.randn(Shape shape)`             | 标准正态分布随机张量                  | `torch.randn`        |
| `Variable.cat(Variable[] vars, int dim)`  | 沿指定维度拼接张量数组                 | `torch.cat`          |
| `Variable.where(Variable cond, Variable x, Variable y)` | 条件选择        | `torch.where`        |

---

## 十七、Function 层实现说明

以下算子的 `Function` 实现类均在 `matrix` 包下，`Variable` 已提供对应包装方法：

| Function 类           | Variable 方法                              | 等效 PyTorch API         |
|----------------------|-----------------------------------------|--------------------------|
| `Permute`            | `permute(int... order)`                 | `tensor.permute`         |
| `Unsqueeze`          | `unsqueeze(int dim)`                    | `tensor.unsqueeze`       |
| `Gather`             | `gather(Variable indices)`              | `torch.gather`           |
| `TopK`               | `topK(int k, int axis)`                 | `torch.topk`             |
| `RotaryEmbedding`    | `applyRope(int dim, int maxSeqLen)`     | 自定义实现              |
| `MaskedFill`         | `maskedFill(Variable mask, float value)`（内部用 `Where` 实现） | `tensor.masked_fill` |

**TopK 多输出说明：**
`topK` 返回 `Variable[]`，下标 `[0]` 是 values，`[1]` 是 indices：
```java
// 取最后一维 Top-5
Variable[] result = logits.topK(5, -1);
Variable values  = result[0];   // Top-5 分数
Variable indices = result[1];   // Top-5 索引

// 取最小値、不排序
Variable[] result = x.topK(3, 1, false, false);
```

**applyRope 示例：**
```java
// 3D 输入 [B, L, D]
Variable rotated = q.applyRope(headDim, maxSeqLen);

// 4D 输入 [B, H, L, D]，自定义 theta
Variable rotated = q.applyRope(headDim, maxSeqLen, 10000.0f);
```

---

## 十八、算子分类速查

```
四则运算:      add / sub / mul / div / neg
基础数学:      squ / pow / exp / log / sqrt / sin / cos / tanh / sigmoid / softMax
激活函数:      relu / gelu / silu / leakyRelu / elu / logSoftmax / clip
统计归约:      sum / sumTo / mean / max / min / var
矩阵操作:      matMul / bmm / transpose / reshape / broadcastTo / expand / repeat
维度操作:      squeeze / unsqueeze / permute / tril / linear
切片索引:      getItem / slice / select / sliceRange / split / indexSelect / scatterAdd / gather
条件操作:      where / maskedFill / gt / lt / eq
归一化:        rmsNorm
损失函数:      meanSquaredError / softmaxCrossEntropy
高级算子:      topK / applyRope
卷积:          conv2d
工具方法:      onesLike / zerosLike / fullLike / detach / clone / fill
静态创建:      zeros / ones / full / rand / randn / cat / where
```
