# TinyAI NdArray 多维数组库

## 概述

`tinyai-dl-ndarr` 是 TinyAI 深度学习框架的核心多维数组处理模块，提供了高效的 N 维数组操作功能。该模块专门设计用于支持 CPU、GPU 和 TPU 三种不同的计算设备，为深度学习和科学计算提供强大的数据结构基础。

## 核心特性

- 🎯 **统一接口设计** - 通过接口抽象实现跨设备的统一操作
- ⚡ **高性能计算** - 采用扁平化内存布局和多种优化策略
- 🔧 **功能完整** - 提供丰富的数学运算和矩阵操作功能
- 🌐 **多设备支持** - 支持 CPU、GPU、TPU 三种计算设备
- 📊 **广播机制** - 支持不同形状数组间的自动广播运算
- 🎨 **易于使用** - 提供简洁的 API 和便捷的工厂方法

## 架构设计

```mermaid
graph TB
    subgraph "核心接口层"
        NdArray[NdArray接口]
        Shape[Shape接口]
        NdArrayUtil[NdArrayUtil工具类]
    end
    
    subgraph "CPU实现"
        NdArrayCpu[NdArrayCpu]
        ShapeCpu[ShapeCpu]
    end
    
    subgraph "GPU实现"
        NdArrayGpu[NdArrayGpu]
        ShapeGpu[ShapeGpu]
    end
    
    subgraph "TPU实现"
        NdArrayTpu[NdArrayTpu]
        ShapeTpu[ShapeTpu]
    end
    
    NdArray --> NdArrayCpu
    NdArray --> NdArrayGpu
    NdArray --> NdArrayTpu
    Shape --> ShapeCpu
    Shape --> ShapeGpu
    Shape --> ShapeTpu
```

## 快速开始

### Maven 依赖

```xml
<dependency>
    <groupId>io.leavesfly.tinyai</groupId>
    <artifactId>tinyai-dl-ndarr</artifactId>
    <version>1.0-SNAPSHOT</version>
</dependency>
```

### 基础使用示例

#### 1. 创建数组

```java
import io.leavesfly.tinyai.ndarr.*;

// 从标量创建
NdArray scalar = NdArray.of(3.14f);

// 从一维数组创建
float[] data1d = {1, 2, 3, 4, 5, 6};
NdArray vector = NdArray.of(data1d);

// 从二维数组创建
float[][] data2d = {{1, 2, 3}, {4, 5, 6}};
NdArray matrix = NdArray.of(data2d);

// 使用形状创建
Shape shape = Shape.of(2, 3);
NdArray zeros = NdArray.zeros(shape);
NdArray ones = NdArray.ones(shape);
NdArray eye = NdArray.eye(Shape.of(3, 3));
```

#### 2. 数学运算

```java
// 基础四则运算
NdArray a = NdArray.of(new float[][]{{1, 2}, {3, 4}});
NdArray b = NdArray.of(new float[][]{{5, 6}, {7, 8}});

NdArray sum = a.add(b);        // 加法
NdArray diff = a.sub(b);       // 减法
NdArray prod = a.mul(b);       // 元素级乘法
NdArray quot = a.div(b);       // 元素级除法

// 矩阵乘法
NdArray matMul = a.dot(b);

// 数学函数
NdArray exp = a.exp();         // 指数函数
NdArray log = a.log();         // 对数函数
NdArray sqrt = a.sqrt();       // 平方根
NdArray pow = a.pow(2);        // 幂函数
```

#### 3. 形状操作

```java
NdArray array = NdArray.of(new float[]{1, 2, 3, 4, 5, 6});

// 重塑形状
NdArray reshaped = array.reshape(Shape.of(2, 3));

// 转置
NdArray transposed = reshaped.transpose();

// 展平
NdArray flattened = reshaped.flatten();
```

#### 4. 聚合操作

```java
NdArray data = NdArray.of(new float[][]{{1, 2, 3}, {4, 5, 6}});

// 统计函数
NdArray sum = data.sum();           // 总和
NdArray mean = data.mean();         // 平均值
NdArray max = data.max();           // 最大值
NdArray min = data.min();           // 最小值
NdArray variance = data.var();      // 方差

// 按轴聚合
NdArray rowSum = data.sum(1);       // 按行求和
NdArray colMean = data.mean(0);     // 按列求平均
```

#### 5. 广播运算

```java
// 矩阵与向量的广播运算
NdArray matrix = NdArray.of(new float[][]{{1, 2, 3}, {4, 5, 6}});
NdArray vector = NdArray.of(new float[]{10, 20, 30});

// 向量自动广播到矩阵形状进行运算
NdArray result = matrix.add(vector.broadcastTo(matrix.getShape()));
```

## 核心组件详解

### NdArray 接口

`NdArray` 是多维数组的核心接口，定义了所有数组操作的统一规范：

#### 创建方法
- **静态工厂方法**: `of()`, `zeros()`, `ones()`, `eye()`, `like()`
- **随机数组**: `likeRandomN()`, `likeRandom()`, `randn()`
- **线性空间**: `linSpace()`

#### 运算操作
- **基础运算**: `add()`, `sub()`, `mul()`, `div()`
- **数学函数**: `exp()`, `log()`, `sqrt()`, `pow()`, `sin()`, `cos()`, `tanh()`, `sigmoid()`
- **矩阵运算**: `dot()`, `transpose()`
- **比较运算**: `eq()`, `gt()`, `lt()`, `ge()`, `le()`

#### 形状操作
- **形状变换**: `reshape()`, `flatten()`, `transpose()`
- **广播操作**: `broadcastTo()`, `sumTo()`
- **切片操作**: `slice()`, `getItem()`

#### 聚合统计
- **统计函数**: `sum()`, `mean()`, `max()`, `min()`, `var()`, `std()`
- **索引操作**: `argmax()`, `argmin()`

### Shape 接口

`Shape` 接口负责管理多维数组的维度信息：

```java
// 创建形状
Shape shape2d = Shape.of(3, 4);        // 3x4 矩阵
Shape shape3d = Shape.of(2, 3, 4);     // 2x3x4 张量

// 形状信息查询
int rows = shape2d.getRow();            // 获取行数
int cols = shape2d.getColumn();         // 获取列数
int dims = shape2d.getDimNum();         // 获取维度数
int size = shape2d.size();              // 获取元素总数

// 类型判断
boolean isMatrix = shape2d.isMatrix();  // 是否为矩阵
boolean isVector = shape1d.isVector();  // 是否为向量
boolean isScalar = shape0d.isScalar();  // 是否为标量

// 索引计算
int index = shape2d.getIndex(1, 2);     // 多维索引转一维索引
```

### NdArrayUtil 工具类

提供数组操作的高级功能：

```java
// 数组合并
NdArray array1 = NdArray.of(new float[][]{{1, 2}, {3, 4}});
NdArray array2 = NdArray.of(new float[][]{{5, 6}, {7, 8}});

// 按行合并（axis=0）
NdArray merged = NdArrayUtil.merge(0, array1, array2);
// 结果: [[1, 2], [3, 4], [5, 6], [7, 8]]

// 按列合并（axis=1）
NdArray merged = NdArrayUtil.merge(1, array1, array2);
// 结果: [[1, 2, 5, 6], [3, 4, 7, 8]]
```

## 高级特性

### 广播机制

广播机制允许不同形状的数组进行运算，遵循以下规则：

1. 从右侧开始比较维度
2. 维度大小相等或其中一个为1时兼容
3. 自动扩展较小的维度

```java
// 示例：(2,3) + (3,) → (2,3)
NdArray matrix = NdArray.of(new float[][]{{1, 2, 3}, {4, 5, 6}});  // (2,3)
NdArray vector = NdArray.of(new float[]{10, 20, 30});               // (3,)

NdArray result = matrix.add(vector.broadcastTo(matrix.getShape()));
// 结果: [[11, 22, 33], [14, 25, 36]]
```

### 内存管理

- **扁平化存储**: 所有数据存储在一维数组中，提高内存访问效率
- **零拷贝操作**: 尽可能避免数据复制，通过视图操作提高性能
- **自动垃圾回收**: 依赖 JVM 的垃圾回收机制管理内存

### 设备支持

当前已实现：
- ✅ **CPU 支持**: 基于 Java 原生数组的高效实现
- 🚧 **GPU 支持**: 基于 CUDA/OpenCL 的并行计算（开发中）
- 🚧 **TPU 支持**: 基于 TPU API 的专用计算（规划中）

## 性能优化

### 建议的最佳实践

1. **预分配数组**: 对于已知大小的运算，提前创建目标数组
2. **批量操作**: 尽量使用向量化操作而非逐元素循环
3. **内存重用**: 利用 `like()` 方法创建相同形状的数组
4. **避免不必要的拷贝**: 使用视图操作代替数据拷贝

```java
// 推荐：批量操作
NdArray result = a.add(b).mul(c);

// 不推荐：逐元素操作
for (int i = 0; i < size; i++) {
    // 逐个计算...
}
```

## 测试覆盖

项目包含全面的单元测试，覆盖率达到 100%：

- **测试总数**: 78 个
- **NdArrayTest**: 40 个测试用例
- **ShapeTest**: 23 个测试用例  
- **NdArrayUtilTest**: 15 个测试用例

运行测试：

```bash
cd tinyai-dl-ndarr
mvn test
```

## 常见问题

### Q: 如何处理不同形状数组的运算？
A: 使用广播机制。调用 `broadcastTo()` 方法将较小的数组广播到目标形状，或直接进行运算（系统会自动处理兼容的广播）。

### Q: 如何优化大规模数组的计算性能？
A: 
1. 使用批量操作而非循环
2. 预分配结果数组
3. 考虑使用GPU实现（开发中）
4. 合理使用内存管理

### Q: 支持哪些数据类型？
A: 当前版本主要支持 `float` 类型。未来版本将扩展支持 `double`、`int` 等其他数值类型。

### Q: 如何扩展到其他计算设备？
A: 实现对应的 `NdArray` 和 `Shape` 接口即可。参考 `NdArrayCpu` 和 `ShapeCpu` 的实现模式。

## 开发路线图

- [x] CPU 基础实现
- [x] 完整的数学运算支持  
- [x] 广播机制
- [x] 全面单元测试
- [ ] GPU 实现（CUDA/OpenCL）
- [ ] TPU 实现
- [ ] 多数据类型支持
- [ ] 稀疏数组支持
- [ ] 分布式计算支持

## 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](../LICENSE) 文件了解详情。

## 联系我们

- 项目主页: [TinyAI](https://github.com/leavesfly/TinyAI)
- 问题反馈: [Issues](https://github.com/leavesfly/TinyAI/issues)
- 邮箱联系: [your-email@example.com]

---

*TinyAI NdArray - 为深度学习提供强大的多维数组基础 🚀*