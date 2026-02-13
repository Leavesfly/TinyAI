package io.leavesfly.tinyai.ndarr;

import io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu;
/**
 * N维数组接口 - TinyAI深度学习的核心数据结构
 *
 * <p><b>设计理念：</b>
 * NdArray（N-dimensional Array）是深度学习中所有计算的基础数据结构。
 * 无论是图像、文本还是音频，在计算机中都被表示为多维数组（张量）。
 *
 * <p><b>维度层次（从简单到复杂）：</b>
 * <pre>
 * 标量(0D) → 向量(1D) → 矩阵(2D) → 张量(3D+) → 高维张量
 *   5.0        [1,2,3]    [[1,2],   [[[1,2],     ...
 *                         [3,4]]    [[3,4]]]
 * </pre>
 *
 * <p><b>Java vs Python 对比：</b>
 * <table border="1">
 *   <tr><th>操作</th><th>NumPy (Python)</th><th>TinyAI (Java)</th></tr>
 *   <tr><td>创建全零数组</td><td>np.zeros((3,4))</td><td>NdArray.zeros(Shape.of(3,4))</td></tr>
 *   <tr><td>创建随机数组</td><td>np.random.randn(3,4)</td><td>NdArray.randn(Shape.of(3,4))</td></tr>
 *   <tr><td>矩阵乘法</td><td>a @ b 或 np.dot(a,b)</td><td>a.dot(b)</td></tr>
 *   <tr><td>数组变形</td><td>a.reshape(2,6)</td><td>a.reshape(Shape.of(2,6))</td></tr>
 *   <tr><td>沿轴求和</td><td>a.sum(axis=0)</td><td>a.sum(0)</td></tr>
 * </table>
 *
 * <p><b>快速入门示例：</b>
 * <pre>{@code
 * // 1. 创建数组
 * NdArray a = NdArray.of(new float[][]{{1, 2}, {3, 4}});
 * NdArray b = NdArray.zeros(Shape.of(2, 3));
 * NdArray c = NdArray.randn(Shape.of(3, 3));  // 标准正态分布随机数
 *
 * // 2. 基本运算
 * NdArray d = a.add(c);        // 元素级加法
 * NdArray e = a.dot(c);        // 矩阵乘法
 * NdArray f = a.transpose();   // 转置
 *
 * // 3. 形状操作
 * NdArray g = a.reshape(Shape.of(4, 1));  // 变形
 * NdArray h = a.flatten();                // 展平
 *
 * // 4. 聚合操作
 * NdArray sum = a.sum();       // 所有元素求和
 * NdArray mean = a.mean(0);    // 沿第0轴求均值
 * }</pre>
 *
 * <p><b>内存布局说明：</b>
 * TinyAI采用<b>行优先（Row-Major）</b>存储，与C/C++、NumPy一致。
 * 对于矩阵 [[1,2,3], [4,5,6]]，内存中的实际布局为：[1,2,3,4,5,6]
 *
 * <p><b>性能提示：</b>
 * <ul>
 *   <li>尽量使用批量操作而非循环遍历单个元素</li>
 *   <li>频繁的reshape操作不会复制数据（视图操作）</li>
 *   <li>矩阵乘法是大计算量操作，注意内存使用</li>
 * </ul>
 *
 * @author TinyAI Team
 * @see Shape
 * @see NdArrayCpu
 */
public interface NdArray {

    // =============================================================================
    // 1,NdArray的创建函数
    // =============================================================================

    /**
     * 从标量值创建NdArray（0维张量）
     *
     * <p><b>数学概念：</b>标量（Scalar）是单个数值，是维度最低的张量。
     * 在深度学习中，标量常用于表示损失值、学习率等单一数值。
     *
     * <p><b>使用示例：</b>
     * <pre>{@code
     * NdArray scalar = NdArray.of(3.14f);     // 创建浮点标量
     * NdArray loss = NdArray.of(0.5);        // 表示损失值
     * System.out.println(scalar.getNumber()); // 输出: 3.14
     * }</pre>
     *
     * <p><b>对比参考：</b>
     * <ul>
     *   <li>NumPy: {@code np.array(3.14)} 或 {@code np.scalar(3.14)}</li>
     *   <li>PyTorch: {@code torch.tensor(3.14)}</li>
     * </ul>
     *
     * @param number 标量值，可以是 Integer、Float、Double 等 Number 子类
     * @return 包含该标量值的0维NdArray
     * @see #of(float[])
     * @see #of(float[], Shape)
     */
    static NdArray of(Number number) {
        return new NdArrayCpu(number);
    }

    /**
     * 从一维数据数组和形状创建NdArray
     *
     * @param data  一维数据数组
     * @param shape 数组形状
     * @throws IllegalArgumentException 当数据长度与形状大小不匹配时抛出
     */
    static NdArray of(float[] data, Shape shape) {
        return new NdArrayCpu(data, shape);
    }

    /**
     * 从一维数组创建NdArray，默认形状为(1, data.length)
     *
     * @param data 一维数据数组
     */
    static NdArray of(float[] data) {
        return new NdArrayCpu(data);
    }

    /**
     * 从多维数组对象创建NdArray
     *
     * <p>支持2D、3D、4D数组的创建</p>
     *
     * @param data 多维数组对象（float[][]、float[][][]或float[][][][]）
     * @throws IllegalArgumentException 当输入类型不支持时抛出
     */
    static NdArray of(Object data) {
        return new NdArrayCpu(data);
    }

    /**
     * 从指定形状创建空的NdArray，所有元素初始化为0
     *
     * @param shape 数组形状
     */
    static NdArray of(Shape shape) {
        return new NdArrayCpu(shape);
    }


    // =============================================================================
    // 2,静态工厂方法
    // =============================================================================

    /**
     * 创建指定形状的全零数组
     *
     * <p><b>数学概念：</b>零矩阵/零张量是所有元素都为0的数组。
     * 在神经网络中常用于初始化偏置（bias）或作为占位符。
     *
     * <p><b>使用示例：</b>
     * <pre>{@code
     * // 创建2x3的零矩阵
     * NdArray zeros2x3 = NdArray.zeros(Shape.of(2, 3));
     * // 结果: [[0, 0, 0],
     * //       [0, 0, 0]]
     *
     * // 创建3维零张量（常用于表示一批灰度图像）
     * NdArray zerosImages = NdArray.zeros(Shape.of(32, 28, 28));
     * // 形状: [批次大小, 高度, 宽度] = 32张28x28的图像
     * }</pre>
     *
     * <p><b>应用场景：</b>
     * <ul>
     *   <li><b>偏置初始化：</b>神经网络层的偏置通常初始化为0</li>
     *   <li><b>占位符：</b>预分配内存用于后续计算</li>
     *   <li><b>掩码：</b>与1数组配合创建二进制掩码</li>
     * </ul>
     *
     * <p><b>对比参考：</b>
     * <ul>
     *   <li>NumPy: {@code np.zeros((2, 3))}</li>
     *   <li>PyTorch: {@code torch.zeros(2, 3)}</li>
     * </ul>
     *
     * @param shape 数组形状，使用 {@link Shape#of(int...)} 创建
     * @return 指定形状的全零数组
     * @throws IllegalArgumentException 当形状维度为0或包含负数时抛出
     * @see #ones(Shape)
     * @see #likeRandomN(Shape)
     */
    static NdArray zeros(Shape shape) {
        return NdArrayCpu.zeros(shape);
    }

    /**
     * 创建指定形状的全一数组
     *
     * @param shape 数组形状
     * @return 全一数组
     */
    static NdArray ones(Shape shape) {
        return NdArrayCpu.ones(shape);
    }

    /**
     * 创建指定形状的单位矩阵（对角矩阵）
     *
     * @param shape 矩阵形状（必须为方形矩阵）
     * @return 单位矩阵
     * @throws IllegalArgumentException 当形状不是矩阵或不是方形矩阵时抛出
     */
    static NdArray eye(Shape shape) {
        return NdArrayCpu.eye(shape);
    }

    /**
     * 创建指定形状和值的数组
     *
     * @param shape 数组形状
     * @param value 填充值
     * @return 指定值填充的数组
     */
    static NdArray like(Shape shape, Number value) {
        return NdArrayCpu.like(shape, value);
    }

    /**
     * 创建与当前数组形状相同但指定值的数组
     *
     * @param value 填充值
     * @return 指定值填充的数组
     */
    NdArray like(Number value);

    /**
     * 创建标准正态分布（均值为0，标准差为1）的随机数组
     *
     * @param shape 数组形状
     * @return 标准正态分布随机数组
     */
    static NdArray likeRandomN(Shape shape) {
        return likeRandomN(shape, 0);
    }

    /**
     * 创建标准正态分布（均值为0，标准差为1）的随机数组（可指定随机种子）
     *
     * @param shape 数组形状
     * @param seed  随机种子，0表示使用默认种子
     * @return 标准正态分布随机数组
     */
    static NdArray likeRandomN(Shape shape, long seed) {
        return NdArrayCpu.likeRandomN(shape, seed);
    }

    /**
     * 创建指定范围内的均匀分布随机数组
     *
     * @param min   最小值（包含）
     * @param max   最大值（包含）
     * @param shape 数组形状
     * @return 均匀分布随机数组
     */
    static NdArray likeRandom(float min, float max, Shape shape) {
        return likeRandom(min, max, shape, 0);
    }

    /**
     * 创建指定范围内的均匀分布随机数组（可指定随机种子）
     *
     * @param min   最小值（包含）
     * @param max   最大值（包含）
     * @param shape 数组形状
     * @param seed  随机种子，0表示使用默认种子
     * @return 均匀分布随机数组
     */
    static NdArray likeRandom(float min, float max, Shape shape, long seed) {
        return NdArrayCpu.likeRandom(min, max, shape, seed);
    }

    /**
     * 创建线性空间数组（等间距排序数组）
     *
     * @param min 起始值
     * @param max 结束值
     * @param num 元素数量
     * @return 线性空间数组
     * @throws IllegalArgumentException 当数量小于等于0时抛出
     */
    static NdArray linSpace(float min, float max, int num) {
        return NdArrayCpu.linSpace(min, max, num);
    }

    /**
     * 创建标准正态分布随机数组
     *
     * <p><b>数学概念：</b>标准正态分布 N(0,1) 是均值为0、标准差为1的正态分布。
     * 在深度学习中，这是<b>最常用的权重初始化方法</b>之一。
     *
     * <p><b>概率密度函数：</b> f(x) = (1/√(2π)) * e^(-x²/2)
     *
     * <p><b>使用示例：</b>
     * <pre>{@code
     * // 创建神经网络权重（输入256，输出128）
     * NdArray weights = NdArray.randn(Shape.of(256, 128));
     * }</pre>
     *
     * <p><b>对比参考：</b>
     * <ul>
     *   <li>NumPy: {@code np.random.randn(3, 3)}</li>
     *   <li>PyTorch: {@code torch.randn(3, 3)}</li>
     * </ul>
     *
     * @param shape 数组形状
     * @return 标准正态分布随机数组，约68%元素在[-1,1]，约95%在[-2,2]
     * @see #likeRandom(float, float, Shape)
     */
    static NdArray randn(Shape shape) {
        return NdArrayCpu.likeRandomN(shape);
    }

    // =============================================================================
    // 3,基础四则运算
    // =============================================================================

    /**
     * 数组加法运算 - 元素级相加（Element-wise Addition）
     *
     * <p><b>数学原理：</b>
     * 给定两个相同形状的数组 A 和 B，元素级加法定义为：
     * <pre>
     * C[i][j] = A[i][j] + B[i][j]
     * </pre>
     *
     * <p><b>直观理解：</b>
     * 想象你有两张相同大小的透明胶片，每张上面都有数字。
     * 将它们重叠，对应位置的数字相加，就得到了结果。
     *
     * <p><b>使用示例：</b>
     * <pre>{@code
     * NdArray a = NdArray.of(new float[][]{{1, 2}, {3, 4}});
     * NdArray b = NdArray.of(new float[][]{{5, 6}, {7, 8}});
     * NdArray c = a.add(b);
     * // 结果: [[6, 8], [10, 12]]
     *
     * // 广播加法：数组 + 标量
     * NdArray d = a.add(NdArray.of(10));
     * // 结果: [[11, 12], [13, 14]]
     * }</pre>
     *
     * <p><b>深度学习应用：</b>
     * <ul>
     *   <li><b>残差连接：</b>output = layer(input) + input（ResNet核心）</li>
     *   <li><b>偏置相加：</b>output = weights · input + bias</li>
     *   <li><b>梯度累加：</b>在反向传播中累加梯度</li>
     * </ul>
     *
     * <p><b>对比参考：</b>
     * <ul>
     *   <li>NumPy: {@code a + b} 或 {@code np.add(a, b)}</li>
     *   <li>PyTorch: {@code a + b} 或 {@code torch.add(a, b)}</li>
     * </ul>
     *
     * @param other 另一个操作数数组，形状必须与当前数组兼容（相同或可广播）
     * @return 加法运算结果，形状为广播后的形状
     * @throws IllegalArgumentException 当两个数组形状不兼容时抛出
     * @see #sub(NdArray)
     * @see #mul(NdArray)
     */
    NdArray add(NdArray other);


    /**
     * 数组减法运算，对应元素相减
     *
     * @param other 另一个操作数数组
     * @return 减法运算结果
     * @throws IllegalArgumentException 当两个数组形状不一致时抛出
     */
    NdArray sub(NdArray other);

    /**
     * 数组乘法运算，对应元素相乘
     *
     * @param other 另一个操作数数组
     * @return 乘法运算结果
     * @throws IllegalArgumentException 当两个数组形状不一致时抛出
     */
    NdArray mul(NdArray other);

    /**
     * 数组与标量相乘
     *
     * @param number 标量值
     * @return 乘法运算结果
     */
    NdArray mulNum(Number number);

    /**
     * 数组除法运算，对应元素相除
     *
     * @param other 另一个操作数数组
     * @return 除法运算结果
     * @throws IllegalArgumentException 当两个数组形状不一致时抛出
     * @throws ArithmeticException      当除数接近0时抛出
     */
    NdArray div(NdArray other);

    /**
     * 数组与标量相除
     *
     * @param number 标量值
     * @return 除法运算结果
     * @throws ArithmeticException 当除数为0时抛出
     */
    NdArray divNum(Number number);
    // =============================================================================
    // 4,逻辑运算
    // =============================================================================

    /**
     * 取反操作，对数组每个元素取负值
     *
     * @return 取反后的数组
     */
    NdArray neg();

    /**
     * 绝对值运算，对数组每个元素取绝对值
     *
     * @return 绝对值数组
     */
    NdArray abs();

    /**
     * 相等比较运算，比较两个数组对应元素是否相等
     *
     * @param other 另一个操作数数组
     * @return 比较结果数组，1.0表示相等，0.0表示不相等
     * @throws IllegalArgumentException 当两个数组形状不一致时抛出
     */
    NdArray eq(NdArray other);

    /**
     * 大于比较运算，比较当前数组元素是否大于另一个数组对应元素
     *
     * @param other 另一个操作数数组
     * @return 比较结果数组，1.0表示大于，0.0表示不大于
     * @throws IllegalArgumentException 当两个数组形状不一致时抛出
     */
    NdArray gt(NdArray other);

    /**
     * 小于比较运算，比较当前数组元素是否小于另一个数组对应元素
     *
     * @param other 另一个操作数数组
     * @return 比较结果数组，1.0表示小于，0.0表示不小于
     * @throws IllegalArgumentException 当两个数组形状不一致时抛出
     */
    NdArray lt(NdArray other);

    /**
     * 矩阵全元素大于比较，判断当前数组是否所有元素都大于另一个数组对应元素
     *
     * @param other 另一个操作数数组
     * @return 比较结果，true表示所有元素都大于，false表示存在不大于的元素
     * @throws IllegalArgumentException 当两个数组形状不一致时抛出
     */
    boolean isLar(NdArray other);

    // =============================================================================
    // 5,基本数学函数
    // =============================================================================


    /**
     * 幂运算，对数组每个元素进行幂运算
     *
     * @param number 幂指数
     * @return 幂运算结果数组
     */
    NdArray pow(Number number);

    /**
     * 平方运算，对数组每个元素进行平方运算
     *
     * @return 平方运算结果数组
     */
    NdArray square();

    /**
     * 平方根运算，对数组每个元素进行开方运算
     *
     * @return 平方根运算结果数组
     */
    NdArray sqrt();

    /**
     * 自然指数运算，对数组每个元素进行e为底的指数运算
     *
     * @return 指数运算结果数组
     */
    NdArray exp();

    /**
     * 正弦函数运算，对数组每个元素进行sin运算
     *
     * @return 正弦运算结果数组
     */
    NdArray sin();

    /**
     * 余弦函数运算，对数组每个元素进行cos运算
     *
     * @return 余弦运算结果数组
     */
    NdArray cos();

    /**
     * 双曲正切函数运算，对数组每个元素进行tanh运算
     *
     * @return 双曲正切运算结果数组
     */
    NdArray tanh();

    /**
     * Sigmoid激活函数 - 将实数映射到(0,1)区间
     *
     * <p><b>数学公式：</b>
     * <pre>
     * σ(x) = 1 / (1 + e^(-x))
     * </pre>
     *
     * <p><b>函数特性：</b>
     * <ul>
     *   <li>输出范围：(0, 1)</li>
     *   <li>σ(0) = 0.5</li>
     *   <li>当 x → +∞, σ(x) → 1</li>
     *   <li>当 x → -∞, σ(x) → 0</li>
     *   <li>导数：σ'(x) = σ(x) × (1 - σ(x))</li>
     * </ul>
     *
     * <p><b>直观理解：</b>
     * Sigmoid 像一个"概率转换器"，将任意实数转换为概率值。
     * 例如：[-3, -1, 0, 1, 3] → [0.05, 0.27, 0.5, 0.73, 0.95]
     *
     * <p><b>使用示例：</b>
     * <pre>{@code
     * // 二分类问题的输出层
     * NdArray logits = NdArray.of(new float[]{-2.0f, 0.5f, 3.0f});
     * NdArray probs = logits.sigmoid();
     * // 结果: [0.12, 0.62, 0.95] - 表示三个样本属于正类的概率
     *
     * // 门控机制（如LSTM中的遗忘门）
     * NdArray gate = weightedSum.sigmoid();  // 输出接近0表示"遗忘"，接近1表示"保留"
     * }</pre>
     *
     * <p><b>深度学习应用：</b>
     * <ul>
     *   <li><b>二分类输出层：</b>将输出转换为概率</li>
     *   <li><b>门控机制：</b>LSTM/GRU中的各种门</li>
     *   <li><b>注意力权重：</b>早期注意力机制中使用</li>
     * </ul>
     *
     * <p><b>注意事项：</b>
     * <ul>
     *   <li><b>梯度消失：</b>当|x|较大时，梯度接近0，导致深层网络训练困难</li>
     *   <li><b>非零中心化：</b>输出总是正数，影响梯度更新效率</li>
     *   <li><b>替代方案：</b>隐藏层推荐使用 ReLU 或 Swish</li>
     * </ul>
     *
     * <p><b>对比参考：</b>
     * <ul>
     *   <li>NumPy: {@code 1 / (1 + np.exp(-x))}</li>
     *   <li>PyTorch: {@code torch.sigmoid(x)}</li>
     * </ul>
     *
     * @return Sigmoid运算结果数组，每个元素在(0,1)范围内
     * @see #tanh()
     * @see #softMax()
     */
    NdArray sigmoid();

    /**
     * 自然对数运算，对数组每个元素进行ln运算
     *
     * @return 对数运算结果数组
     * @throws ArithmeticException 当输入值小于等于0时抛出
     */
    NdArray log();

    /**
     * Softmax函数 - 将任意实数向量转换为概率分布
     *
     * <p><b>数学公式：</b>
     * <pre>
     * softmax(x_i) = exp(x_i) / Σ(exp(x_j))
     * </pre>
     *
     * <p><b>直观理解：</b>
     * Softmax 像一个"竞争性归一化器"。
     * 输入值越大，输出的概率越高，但所有概率之和为1。
     * 想象一场竞赛：分数越高获奖概率越大，但所有人获奖概率加起来必须是100%。
     *
     * <p><b>数值稳定性：</b>
     * 直接使用公式可能导致指数溢出（e^100 是巨大的数）。
     * 实现采用稳定版本：先减去最大值，再计算指数。
     * <pre>
     * stable_softmax(x_i) = exp(x_i - max(x)) / Σ(exp(x_j - max(x)))
     * </pre>
     *
     * <p><b>使用示例：</b>
     * <pre>{@code
     * // 多分类问题的输出层（3个类别）
     * NdArray logits = NdArray.of(new float[][]{
     *     {2.0f, 1.0f, 0.1f},   // 样本1：类别0得分最高
     *     {0.5f, 2.5f, 0.3f}    // 样本2：类别1得分最高
     * });
     * NdArray probs = logits.softMax();
     * // 结果: [[0.70, 0.24, 0.06],   <- 样本1预测为类别0的概率70%
     * //        [0.12, 0.82, 0.06]]   <- 样本2预测为类别1的概率82%
     * }</pre>
     *
     * <p><b>深度学习应用：</b>
     * <ul>
     *   <li><b>多分类输出层：</b>将网络输出转换为类别概率</li>
     *   <li><b>注意力机制：</b>计算注意力权重（Transformer核心）</li>
     *   <li><b>强化学习：</b>策略网络输出动作概率分布</li>
     * </ul>
     *
     * <p><b>与Sigmoid的区别：</b>
     * <table border="1">
     *   <tr><th>特性</th><th>Sigmoid</th><th>Softmax</th></tr>
     *   <tr><td>输出范围</td><td>(0, 1)</td><td>(0, 1)，且和为1</td></tr>
     *   <tr><td>适用场景</td><td>二分类</td><td>多分类</td></tr>
     *   <tr><td>关系</td><td>独立处理每个元素</td><td>元素间相互竞争</td></tr>
     * </table>
     *
     * <p><b>对比参考：</b>
     * <ul>
     *   <li>NumPy: {@code np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)}</li>
     *   <li>PyTorch: {@code torch.softmax(x, dim=1)}</li>
     * </ul>
     *
     * @return Softmax运算结果数组，每行是一个概率分布（和为1）
     * @throws IllegalArgumentException 当数组不是矩阵时抛出
     * @see #softMax(int)
     * @see #sigmoid()
     */
    NdArray softMax();


    /**
     * Softmax函数运算，沿指定 axis 计算概率分布
     *
     * <p>使用数值稳定版本实现：先减去该轴上的最大值，再进行 exp 和归一化</p>
     *
     * @param axis 计算 softmax 的维度，支持负轴（-1 表示最后一维）
     * @return Softmax运算结果数组
     * @throws IllegalArgumentException 当 axis 越界时抛出
     */
    NdArray softMax(int axis);

    /**
     * 元素级最大值运算，将数组中小于指定值的元素替换为该值
     *
     * @param number 阈值
     * @return 最大值运算结果数组
     */
    NdArray maximum(Number number);

    /**
     * 掩码运算，将数组中大于指定值的元素设为1，小于等于指定值的元素设为0
     *
     * @param number 阈值
     * @return 掩码运算结果数组
     */
    NdArray mask(Number number);

    // =============================================================================
    // 6,张量的变形操作
    // =============================================================================

    /**
     * 矩阵转置操作（二维矩阵），行列互换
     *
     * @return 转置后的矩阵
     * @throws IllegalArgumentException 当数组不是矩阵时抛出
     */
    NdArray transpose();

    /**
     * 多维数组转置操作，按指定维度顺序重新排列
     *
     * @param order 新的维度顺序
     * @return 转置后的数组
     * @throws IllegalArgumentException 当维度顺序无效时抛出
     */
    NdArray transpose(int... order);


    /**
     * 数组变形操作，改变数组形状但保持元素总数不变
     *
     * @param newShape 新的数组形状
     * @return 变形后的数组
     * @throws IllegalArgumentException 当新形状大小与原形状不匹配时抛出
     */
    NdArray reshape(Shape newShape);

    /**
     * 支持广播语义的reshape（新增方法）
     * <p>
     * 允许将大小为1的维度扩展到更大的尺寸，如将[1,3]扩展为[5,3]
     * </p>
     * <p>
     * 与普通reshape的区别：
     * - 普通reshape要求元素总数相等
     * - broadcastReshape允许从维度1扩展到任意大小（通过广播实现）
     * </p>
     *
     * @param newShape 新的数组形状
     * @return 变形后的数组
     * @throws IllegalArgumentException 当形状不兼容时抛出（非1维度必须匹配）
     */
    NdArray broadcastReshape(Shape newShape);


    /**
     * 数组展平操作，将多维数组转换为一维行向量
     *
     * @return 展平后的一维行向量
     */
    NdArray flatten();

    // =============================================================================
    // 7,统计和聚合操作
    // =============================================================================

    /**
     * 元素累和运算，计算数组所有元素的总和
     *
     * @return 所有元素的总和（标量）
     */
    NdArray sum();


    /**
     * 矩阵均值运算，沿指定轴计算均值
     *
     * @param axis 聚合轴
     * @return 均值运算结果数组
     */
    NdArray mean(int axis);

    /**
     * 矩阵方差运算，沿指定轴计算方差
     *
     * @param axis 聚合轴
     * @return 方差运算结果数组
     */
    NdArray var(int axis);

    /**
     * 矩阵累和运算，沿指定轴计算累和
     *
     * @param axis 聚合轴
     * @return 累和运算结果数组
     */
    NdArray sum(int axis);

    /**
     * 按指定形状进行压缩累加运算
     *
     * <p>将当前数组按指定形状进行压缩，超出目标形状的部分会累加到对应位置</p>
     *
     * @param _shape 目标形状
     * @return 压缩累加结果数组
     * @throws IllegalArgumentException 当形状不合法时抛出
     */
    NdArray sumTo(Shape _shape);

    /**
     * 优化的sumTo实现（新增方法）
     * <p>
     * 使用轴向求和策略，性能提升2-3個。
     * 相比于普通sumTo，该方法通过识别需要求和的维度，
     * 利用高效的sum(axis)实现逐轴求和，避免了全元素遍历。
     * </p>
     *
     * @param targetShape 目标形状
     * @return 压缩结果数组
     * @throws IllegalArgumentException 当形状不合法时抛出
     */
    NdArray sumToOptimized(Shape targetShape);

    /**
     * 数组广播运算，将当前数组广播到指定形状
     *
     * <p>广播机制允许小数组与大数组进行运算，小数组会重复填充以匹配大数组的形状</p>
     *
     * @param _shape 目标广播形状
     * @return 广播结果数组
     * @throws IllegalArgumentException 当形状不合法时抛出
     */
    NdArray broadcastTo(Shape _shape);

    /**
     * 沿指定轴查找最大值的索引
     *
     * @param axis 查找轴，axis=0表示按行查找每列的最大值索引，axis=1表示按列查找每行的最大值索引
     * @return 最大值索引数组
     * @throws IllegalArgumentException 当数组不是矩阵或轴参数无效时抛出
     */
    NdArray argMax(int axis);

    /**
     * 矩阵内积运算（矩阵乘法）- 深度学习的核心操作
     *
     * <p><b>数学原理：</b>
     * 给定矩阵 A(m×n) 和 B(n×p)，结果 C(m×p) 的计算公式：
     * <pre>
     * C[i][j] = Σ(A[i][k] × B[k][j])  for k=0 to n-1
     * </pre>
     *
     * <p><b>直观理解：</b>
     * 矩阵乘法可以看作是"行与列的点积"。
     * 结果矩阵的每个元素，都是第一个矩阵的行与第二个矩阵的列对应相乘再相加。
     *
     * <p><b>形状规则：</b>
     * <pre>
     * A: [m, n]  @  B: [n, p]  =  C: [m, p]
     *      ↑_________↑
     *       内维必须匹配
     * </pre>
     *
     * <p><b>使用示例：</b>
     * <pre>{@code
     * // 简单的矩阵乘法
     * NdArray a = NdArray.of(new float[][]{{1, 2}, {3, 4}, {5, 6}});  // 3x2
     * NdArray b = NdArray.of(new float[][]{{7, 8, 9}, {10, 11, 12}}); // 2x3
     * NdArray c = a.dot(b);  // 结果: 3x3
     * // c = [[27, 30, 33],
     * //      [61, 68, 75],
     * //      [95, 106, 117]]
     *
     * // 神经网络中的线性变换: output = input · weights
     * NdArray input = NdArray.randn(Shape.of(32, 784));     // 批次32，特征784
     * NdArray weights = NdArray.randn(Shape.of(784, 256));  // 输入784，输出256
     * NdArray output = input.dot(weights);                   // 结果: [32, 256]
     * }</pre>
     *
     * <p><b>深度学习应用：</b>
     * <ul>
     *   <li><b>线性层：</b>全连接层的核心运算 y = xW + b</li>
     *   <li><b>注意力机制：</b>Q·K^T 计算注意力分数</li>
     *   <li><b>特征变换：</b>将数据从一种表示空间映射到另一种</li>
     * </ul>
     *
     * <p><b>性能提示：</b>
     * 矩阵乘法是计算密集型操作，时间复杂度为 O(m×n×p)。
     * 对于大规模矩阵，这通常是训练中的性能瓶颈。
     *
     * <p><b>对比参考：</b>
     * <ul>
     *   <li>NumPy: {@code a @ b} 或 {@code np.dot(a, b)} 或 {@code np.matmul(a, b)}</li>
     *   <li>PyTorch: {@code a @ b} 或 {@code torch.matmul(a, b)}</li>
     * </ul>
     *
     * @param other 右乘矩阵，形状为 [..., n, p]
     * @return 矩阵乘法结果，形状为 [..., m, p]
     * @throws IllegalArgumentException 当内维不匹配时抛出（即A的列数 ≠ B的行数）
     * @see #add(NdArray)
     * @see #mul(NdArray)
     */
    NdArray dot(NdArray other);

    /**
     * 获取数组的子集（切片操作）
     *
     * @param _rowSlices 行索引数组，null表示选择所有行
     * @param _colSlices 列索引数组，null表示选择所有列
     * @return 切片结果数组
     * @throws IllegalArgumentException 当数组不是矩阵或参数不合法时抛出
     */
    NdArray getItem(int[] _rowSlices, int[] _colSlices);

    /**
     * 设置数组的子集（切片赋值操作）
     *
     * @param _rowSlices 行索引数组，null表示选择所有行
     * @param _colSlices 列索引数组，null表示选择所有列
     * @param data       要设置的数据
     * @return 当前数组实例
     * @throws IllegalArgumentException 当数组不是矩阵或参数不合法时抛出
     */
    NdArray setItem(int[] _rowSlices, int[] _colSlices, float[] data);

    /**
     * 高性能连续区域赋值（新增方法）
     * <p>
     * 适用场景：Concat拼接操作、矩阵块操作
     * </p>
     *
     * @param startRow 起始行索引（包含）
     * @param endRow   结束行索引（不包含）
     * @param startCol 起始列索引（包含）
     * @param endCol   结束列索引（不包含）
     * @param data     要设置的数据，长度必须等于(endRow-startRow)*(endCol-startCol)
     * @return 当前数组实例
     * @throws IllegalArgumentException 当数组不是矩阵或参数不合法时抛出
     */
    NdArray setBlock(int startRow, int endRow, int startCol, int endCol, float[] data);

    /**
     * 行切片赋值（新增方法）
     * <p>
     * 高效设置指定行的数据
     * </p>
     *
     * @param rowIndices 行索引数组
     * @param data       要设置的数据，长度必须等于rowIndices.length * 列数
     * @return 当前数组实例
     * @throws IllegalArgumentException 当数组不是矩阵或参数不合法时抛出
     */
    NdArray setRows(int[] rowIndices, float[] data);

    /**
     * 列切片赋值（新增方法）
     * <p>
     * 高效设置指定列的数据
     * </p>
     *
     * @param colIndices 列索引数组
     * @param data       要设置的数据，长度必须等于行数 * colIndices.length
     * @return 当前数组实例
     * @throws IllegalArgumentException 当数组不是矩阵或参数不合法时抛出
     */
    NdArray setCols(int[] colIndices, float[] data);

    /**
     * 沿指定轴查找最大值
     *
     * @param axis 查找轴，axis=0表示按行查找每列的最大值，axis=1表示按列查找每行的最大值
     * @return 最大值数组
     * @throws IllegalArgumentException 当数组不是矩阵或轴参数无效时抛出
     */
    NdArray max(int axis);

    /**
     * 沿指定轴查找最小值
     *
     * @param axis 查找轴，axis=0表示按行查找每列的最小值，axis=1表示按列查找每行的最小值
     * @return 最小值数组
     * @throws IllegalArgumentException 当数组不是矩阵或轴参数无效时抛出
     */
    NdArray min(int axis);

    /**
     * 查找数组中的最大值（全局最大值）
     *
     * @return 数组中的最大值
     */
    float max();

    /**
     * 获取子数组（矩阵的子区域）
     *
     * @param startRow 起始行索引（包含）
     * @param endRow   结束行索引（不包含）
     * @param startCol 起始列索引（包含）
     * @param endCol   结束列索引（不包含）
     * @return 子数组
     * @throws IllegalArgumentException 当数组不是矩阵时抛出
     */
    NdArray subNdArray(int startRow, int endRow, int startCol, int endCol);


    /**
     * 在指定位置累加数组元素
     *
     * <p>在指定的行和列位置上累加另一个数组的元素。这个方法常用于反向传播中梯度的累积。</p>
     *
     * <p>使用示例：</p>
     * <pre>
     * NdArray a = new NdArray(new float[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
     * NdArray b = new NdArray(new float[][]{{10}, {20}});
     * NdArray result = a.addAt(new int[]{0, 2}, new int[]{1, 1}, b);
     * // 结果：在位置(0,1)和(2,1)分别累加b中的值
     * </pre>
     *
     * @param rowSlices 行索引数组，指定要累加的行位置
     * @param colSlices 列索引数组，指定要累加的列位置
     * @param other     要累加的数组
     * @return 累加结果数组
     * @throws IllegalArgumentException 当输入参数不合法时抛出
     * @throws RuntimeException         当数组不是矩阵时抛出
     */
    NdArray addAt(int[] rowSlices, int[] colSlices, NdArray other);


    /**
     * 将另一个数组累加到当前数组的指定位置
     *
     * @param i     起始行索引
     * @param j     起始列索引
     * @param other 要累加的数组
     * @return 当前数组实例
     * @throws IllegalArgumentException 当数组不是矩阵时抛出
     */
    NdArray addTo(int i, int j, NdArray other);

    /**
     * 裁剪数组元素到指定范围
     *
     * <p>将数组中小于最小值的元素设为最小值，大于最大值的元素设为最大值</p>
     *
     * @param min 最小值
     * @param max 最大值
     * @return 裁剪后的数组
     * @throws IllegalArgumentException 当最小值大于最大值时抛出
     */
    NdArray clip(float min, float max);


    // =============================================================================
    // 8,其他的运算
    // =============================================================================


    /**
     * 获取数组的第一个元素值（标量值）
     *
     * @return 第一个元素值
     */
    Number getNumber();

    Shape getShape();

    /**
     * 设置数组的形状
     *
     * <p>注意：新形状的大小必须与当前形状大小一致</p>
     *
     * @param shape 新形状
     * @throws IllegalArgumentException 当新形状大小与当前形状不匹配时抛出
     */
    void setShape(Shape shape);


    /**
     * 返回数组
     *
     * @return
     */
    float[] getArray();

    /**
     * 将数组转换为二维数组（矩阵）返回
     *
     * @return 二维数组表示
     * @throws IllegalArgumentException 当数组维度大于2时抛出
     */
    float[][] getMatrix();


    /**
     * 将数组转换为三维数组返回
     *
     * @return 三维数组表示
     * @throws IllegalArgumentException 当数组不是三维时抛出
     */
    float[][][] get3dArray();

    /**
     * 将数组转换为四维数组返回
     *
     * @return 四维数组表示
     * @throws IllegalArgumentException 当数组不是四维时抛出
     */
    float[][][][] get4dArray();


    /**
     * 按维度下标设置某一个值
     *
     * @param value      要设置的值
     * @param _dimension 维度下标数组
     * @throws IllegalArgumentException 当维度数量不匹配时抛出
     */
    void set(float value, int... _dimension);

    /**
     * 按维度下标获取某一个值
     *
     * @param _dimension 维度下标数组
     * @return 对应位置的值
     * @throws IllegalArgumentException 当维度数量不匹配时抛出
     */
    float get(int... _dimension);
}
