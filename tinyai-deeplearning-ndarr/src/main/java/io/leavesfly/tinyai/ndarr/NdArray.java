package io.leavesfly.tinyai.ndarr;

import io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu;

public interface NdArray {

    // =============================================================================
    // 1,NdArray的创建函数
    // =============================================================================

    /**
     * 从标量值创建NdArray
     *
     * @param number 标量值
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
     * @param shape 数组形状
     * @return 全零数组
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
     * @param shape 数组形状
     * @return 标准正态分布随机数组
     */
    static NdArray randn(Shape shape) {
        return NdArrayCpu.likeRandomN(shape);
    }

    // =============================================================================
    // 3,基础四则运算
    // =============================================================================

    /**
     * 数组加法运算，对应元素相加
     *
     * @param other 另一个操作数数组
     * @return 加法运算结果
     * @throws IllegalArgumentException 当两个数组形状不一致时抛出
     */
    NdArrayCpu add(NdArray other);


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
     * Sigmoid函数运算，对数组每个元素进行sigmoid运算
     *
     * <p>Sigmoid函数公式：f(x) = 1 / (1 + e^(-x))</p>
     *
     * @return Sigmoid运算结果数组
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
     * Softmax函数运算，按行计算概率分布
     *
     * <p>Softmax函数公式：softmax(x_i) = exp(x_i) / Σ(exp(x_j))</p>
     * <p>使用数值稳定版本实现，避免指数运算溢出</p>
     *
     * @return Softmax运算结果数组
     * @throws IllegalArgumentException 当数组不是矩阵时抛出
     */
    NdArray softMax();

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
     * 矩阵内积运算（矩阵乘法）
     *
     * <p>执行标准的矩阵乘法运算，要求第一个矩阵的列数等于第二个矩阵的行数</p>
     *
     * @param other 另一个矩阵
     * @return 矩阵乘法结果
     * @throws IllegalArgumentException 当数组不是矩阵或维度不匹配时抛出
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
