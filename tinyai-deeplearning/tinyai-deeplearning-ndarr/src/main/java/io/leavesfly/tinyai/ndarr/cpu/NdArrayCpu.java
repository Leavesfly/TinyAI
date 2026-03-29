package io.leavesfly.tinyai.ndarr.cpu;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.ndarr.cpu.aggregations.ReductionOperations;
import io.leavesfly.tinyai.ndarr.cpu.factories.NdArrayFactories;
import io.leavesfly.tinyai.ndarr.cpu.matrix.MatrixOperations;
import io.leavesfly.tinyai.ndarr.cpu.operations.AccumulationOperations;
import io.leavesfly.tinyai.ndarr.cpu.operations.ArithmeticOperations;
import io.leavesfly.tinyai.ndarr.cpu.operations.AxisOperations;
import io.leavesfly.tinyai.ndarr.cpu.operations.LogicalOperations;
import io.leavesfly.tinyai.ndarr.cpu.operations.MathFunctions;
import io.leavesfly.tinyai.ndarr.cpu.transformations.SlicingOperations;
import io.leavesfly.tinyai.ndarr.cpu.transformations.TransformationOperations;
import io.leavesfly.tinyai.ndarr.cpu.utils.ArrayConverter;
import io.leavesfly.tinyai.ndarr.cpu.utils.ArrayFormatter;
import io.leavesfly.tinyai.ndarr.cpu.utils.ArrayValidator;

import java.io.Serializable;
import java.util.Arrays;

/**
 * N维数组类，支持标量、向量、矩阵等多维数据结构
 *
 * 该类经过重构优化，提供更加优雅的API和更好的性能，是深度学习框架的核心数据结构。
 *
 * 主要特性：
 *  - 支持任意维度的数组操作
 *  - 高效的内存管理
 *  - 丰富的数学运算和张量操作
 *  - 广播机制支持
 */
public class NdArrayCpu implements NdArray, Serializable {

    private static final long serialVersionUID = 1L;

    /**
     * N维数组的形状，描述各维度的大小
     */
    public ShapeCpu shape;

    /**
     * 真实存储数据的一维数组，使用float32类型以节省内存并提高性能
     */
    public float[] buffer;

    // =============================================================================
    // NdArray的创建函数 - 重构后的构造方法
    // =============================================================================

    /**
     * 默认构造方法，创建空的NdArray实例
     *
     * 注意：此构造方法不会初始化shape和buffer，需要手动设置
     */
    public NdArrayCpu() {
    }

    /**
     * 从标量值创建NdArray
     *
     * @param number 标量值
     */
    public NdArrayCpu(Number number) {
        this.shape = ShapeCpu.of(1, 1);
        this.buffer = new float[1];
        this.buffer[0] = number.floatValue();
    }

    /**
     * 从一维数据数组和形状创建NdArray
     *
     * @param data  一维数据数组
     * @param shape 数组形状
     * @throws IllegalArgumentException 当数据长度与形状大小不匹配时抛出
     */
    public NdArrayCpu(float[] data, Shape shape) {
        ArrayValidator.validateDataShape(data.length, shape.size());
        this.shape = (ShapeCpu) shape;
        this.buffer = data;
    }

    /**
     * 从一维数组创建NdArray，默认形状为(1, data.length)
     *
     * @param data 一维数据数组
     */
    public NdArrayCpu(float[] data) {
        this.shape = ShapeCpu.of(1, data.length);
        this.buffer = data;
    }

    /**
     * 从多维数组对象创建NdArray
     *
     * 支持2D、3D、4D数组的创建
     *
     * @param data 多维数组对象（float[][]、float[][][]或float[][][][]）
     * @throws IllegalArgumentException 当输入类型不支持时抛出
     */
    public NdArrayCpu(Object data) {
        if (data instanceof float[][]) {
            initFromArray((float[][]) data);
        } else if (data instanceof float[][][]) {
            initFromArray((float[][][]) data);
        } else if (data instanceof float[][][][]) {
            initFromArray((float[][][][]) data);
        } else {
            throw new IllegalArgumentException("不支持的数组类型: " + data.getClass());
        }
    }

    /**
     * 从指定形状创建空的NdArray，所有元素初始化为0
     *
     * @param shape 数组形状
     */
    public NdArrayCpu(ShapeCpu shape) {
        this.shape = shape;
        this.buffer = new float[shape.size()];
    }

    /**
     * 从指定形状接口创建空的NdArray，所有元素初始化为0
     *
     * @param shape 数组形状接口
     */
    public NdArrayCpu(Shape shape) {
        this.shape = (ShapeCpu) shape;
        this.buffer = new float[shape.size()];
    }

    // 优化的初始化方法
    private void initFromArray(float[][] data) {
        ArrayValidator.validateArrayDimensions(data);
        this.shape = ShapeCpu.of(data.length, data[0].length);
        this.buffer = new float[shape.size()];
        ArrayConverter.flattenArray(data, this.buffer, 0);
    }

    private void initFromArray(float[][][] data) {
        ArrayValidator.validateArrayDimensions(data);
        this.shape = ShapeCpu.of(data.length, data[0].length, data[0][0].length);
        this.buffer = new float[shape.size()];
        ArrayConverter.flattenArray(data, this.buffer, 0);
    }

    private void initFromArray(float[][][][] data) {
        ArrayValidator.validateArrayDimensions(data);
        this.shape = ShapeCpu.of(data.length, data[0].length, data[0][0].length, data[0][0][0].length);
        this.buffer = new float[shape.size()];
        ArrayConverter.flattenArray(data, this.buffer, 0);
    }

    // =============================================================================
    // 静态工厂方法 - 优化后的创建方法
    // =============================================================================

    /**
     * 创建指定形状的全零数组
     *
     * @param shape 数组形状
     * @return 全零数组
     */
    public static NdArrayCpu zeros(Shape shape) {
        return NdArrayFactories.zeros(shape);
    }

    /**
     * 创建指定形状的全一数组
     *
     * @param shape 数组形状
     * @return 全一数组
     */
    public static NdArrayCpu ones(Shape shape) {
        return NdArrayFactories.ones(shape);
    }

    /**
     * 创建指定形状的单位矩阵（对角矩阵）
     *
     * @param shape 矩阵形状（必须为方形矩阵）
     * @return 单位矩阵
     * @throws IllegalArgumentException 当形状不是矩阵或不是方形矩阵时抛出
     */
    public static NdArrayCpu eye(Shape shape) {
        return NdArrayFactories.eye(shape);
    }

    /**
     * 创建指定形状和值的数组
     *
     * @param shape 数组形状
     * @param value 填充值
     * @return 指定值填充的数组
     */
    public static NdArrayCpu like(Shape shape, Number value) {
        return NdArrayFactories.like(shape, value);
    }

    /**
     * 创建与当前数组形状相同但指定值的数组
     *
     * @param value 填充值
     * @return 指定值填充的数组
     */
    public NdArrayCpu like(Number value) {
        return NdArrayFactories.like(this.shape, value);
    }

    /**
     * 创建标准正态分布（均值为0，标准差为1）的随机数组
     *
     * @param shape 数组形状
     * @return 标准正态分布随机数组
     */
    public static NdArrayCpu likeRandomN(Shape shape) {
        return NdArrayFactories.likeRandomN(shape);
    }

    /**
     * 创建标准正态分布（均值为0，标准差为1）的随机数组（可指定随机种子）
     *
     * @param shape 数组形状
     * @param seed  随机种子，0表示使用默认种子
     * @return 标准正态分布随机数组
     */
    public static NdArrayCpu likeRandomN(Shape shape, long seed) {
        return NdArrayFactories.likeRandomN(shape, seed);
    }

    /**
     * 创建指定范围内的均匀分布随机数组
     *
     * @param min   最小值（包含）
     * @param max   最大值（包含）
     * @param shape 数组形状
     * @return 均匀分布随机数组
     */
    public static NdArrayCpu likeRandom(float min, float max, Shape shape) {
        return NdArrayFactories.likeRandom(min, max, shape);
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
    public static NdArrayCpu likeRandom(float min, float max, Shape shape, long seed) {
        return NdArrayFactories.likeRandom(min, max, shape, seed);
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
    public static NdArrayCpu linSpace(float min, float max, int num) {
        return NdArrayFactories.linSpace(min, max, num);
    }

    // =============================================================================
    // 基础四则运算 - 重构后的统一模式
    // =============================================================================


    /**
     * 数组加法运算，对应元素相加
     *
     * @param other 另一个操作数数组
     * @return 加法运算结果
     * @throws IllegalArgumentException 当两个数组形状不一致时抛出
     */
    public NdArray add(NdArray other) {
        return ArithmeticOperations.add(this, (NdArrayCpu) other);
    }

    /**
     * 数组减法运算，对应元素相减
     *
     * @param other 另一个操作数数组
     * @return 减法运算结果
     * @throws IllegalArgumentException 当两个数组形状不一致时抛出
     */
    public NdArrayCpu sub(NdArray other) {
        return ArithmeticOperations.sub(this, (NdArrayCpu) other);
    }

    /**
     * 数组乘法运算，对应元素相乘
     *
     * @param other 另一个操作数数组
     * @return 乘法运算结果
     * @throws IllegalArgumentException 当两个数组形状不一致时抛出
     */
    public NdArrayCpu mul(NdArray other) {
        return ArithmeticOperations.mul(this, (NdArrayCpu) other);
    }

    /**
     * 数组与标量相乘
     *
     * @param number 标量值
     * @return 乘法运算结果
     */
    public NdArrayCpu mulNum(Number number) {
        return ArithmeticOperations.mulNum(this, number);
    }

    /**
     * 数组除法运算，对应元素相除
     *
     * @param other 另一个操作数数组
     * @return 除法运算结果
     * @throws IllegalArgumentException 当两个数组形状不一致时抛出
     * @throws ArithmeticException      当除数接近0时抛出
     */
    @Override
    public NdArray div(NdArray other) {
        return ArithmeticOperations.div(this, (NdArrayCpu) other);
    }

    /**
     * 数组与标量相除
     *
     * @param number 标量值
     * @return 除法运算结果
     * @throws ArithmeticException 当除数为0时抛出
     */
    public NdArray divNum(Number number) {
        return ArithmeticOperations.divNum(this, number);
    }

    // =============================================================================
    // 逻辑运算 - 重构后的统一模式
    // =============================================================================


    /**
     * 取反操作，对数组每个元素取负值
     *
     * @return 取反后的数组
     */
    public NdArrayCpu neg() {
        return LogicalOperations.neg(this);
    }

    /**
     * 绝对值运算，对数组每个元素取绝对值
     *
     * @return 绝对值数组
     */
    public NdArrayCpu abs() {
        return LogicalOperations.abs(this);
    }

    /**
     * 相等比较运算，比较两个数组对应元素是否相等
     *
     * @param other 另一个操作数数组
     * @return 比较结果数组，1.0表示相等，0.0表示不相等
     * @throws IllegalArgumentException 当两个数组形状不一致时抛出
     */
    public NdArrayCpu eq(NdArray other) {
        return LogicalOperations.eq(this, (NdArrayCpu) other);
    }

    /**
     * 大于比较运算，比较当前数组元素是否大于另一个数组对应元素
     *
     * @param other 另一个操作数数组
     * @return 比较结果数组，1.0表示大于，0.0表示不大于
     * @throws IllegalArgumentException 当两个数组形状不一致时抛出
     */
    public NdArrayCpu gt(NdArray other) {
        return LogicalOperations.gt(this, (NdArrayCpu) other);
    }

    /**
     * 小于比较运算，比较当前数组元素是否小于另一个数组对应元素
     *
     * @param other 另一个操作数数组
     * @return 比较结果数组，1.0表示小于，0.0表示不小于
     * @throws IllegalArgumentException 当两个数组形状不一致时抛出
     */
    public NdArrayCpu lt(NdArray other) {
        return LogicalOperations.lt(this, (NdArrayCpu) other);
    }

    /**
     * 全元素大于比较，判断当前数组是否所有元素都严格大于另一个数组对应元素
     *
     * @param other 另一个操作数数组
     * @return true 表示所有元素都大于，false 表示存在不大于的元素
     * @throws IllegalArgumentException 当两个数组形状不一致时抛出
     */
    public boolean isAllGreaterThan(NdArray other) {
        return LogicalOperations.isAllGreaterThan(this, (NdArrayCpu) other);
    }

    // =============================================================================
    // 基本数学函数 - 重构后的统一模式
    // =============================================================================

    /**
     * 幂运算，对数组每个元素进行幂运算
     *
     * @param number 幂指数
     * @return 幂运算结果数组
     */
    public NdArrayCpu pow(Number number) {
        return MathFunctions.pow(this, number);
    }

    /**
     * 平方运算，对数组每个元素进行平方运算
     *
     * @return 平方运算结果数组
     */
    public NdArrayCpu square() {
        return MathFunctions.square(this);
    }

    /**
     * 平方根运算，对数组每个元素进行开方运算
     *
     * @return 平方根运算结果数组
     */
    public NdArrayCpu sqrt() {
        return MathFunctions.sqrt(this);
    }

    /**
     * 自然指数运算，对数组每个元素进行e为底的指数运算
     *
     * @return 指数运算结果数组
     */
    public NdArrayCpu exp() {
        return MathFunctions.exp(this);
    }

    /**
     * 正弦函数运算，对数组每个元素进行sin运算
     *
     * @return 正弦运算结果数组
     */
    public NdArrayCpu sin() {
        return MathFunctions.sin(this);
    }

    /**
     * 余弦函数运算，对数组每个元素进行cos运算
     *
     * @return 余弦运算结果数组
     */
    public NdArrayCpu cos() {
        return MathFunctions.cos(this);
    }

    /**
     * 双曲正切函数运算，对数组每个元素进行tanh运算
     *
     * @return 双曲正切运算结果数组
     */
    public NdArrayCpu tanh() {
        return MathFunctions.tanh(this);
    }

    /**
     * Sigmoid函数运算，对数组每个元素进行sigmoid运算
     *
     * Sigmoid函数公式：f(x) = 1 / (1 + e^(-x))
     *
     * @return Sigmoid运算结果数组
     */
    public NdArrayCpu sigmoid() {
        return MathFunctions.sigmoid(this);
    }

    /**
     * 自然对数运算，对数组每个元素进行ln运算
     *
     * @return 对数运算结果数组
     * @throws ArithmeticException 当输入值小于等于0时抛出
     */
    public NdArrayCpu log() {
        return MathFunctions.log(this);
    }

    /**
     * Softmax函数运算，按行计算概率分布
     *
     * Softmax函数公式：softmax(x_i) = exp(x_i) / Σ(exp(x_j))
     * 使用数值稳定版本实现，避免指数运算溢出
     *
     * @return Softmax运算结果数组
     * @throws IllegalArgumentException 当数组不是二维矩阵时抛出
     */
    public NdArrayCpu softMax() {
        return MathFunctions.softMax(this);
    }

    /**
     * Softmax函数运算，沿指定 axis 计算概率分布
     *
     * 使用数值稳定版本实现：先减去该轴上的最大值，再进行 exp 和归一化
     *
     * @param axis 计算 softmax 的维度，支持负轴（-1 表示最后一维）
     * @return Softmax运算结果数组
     * @throws IllegalArgumentException 当 axis 越界时抛出
     */
    public NdArrayCpu softMax(int axis) {
        return MathFunctions.softMax(this, axis);
    }


    /**
     * 元素级最大值运算，将数组中小于指定值的元素替换为该值
     *
     * @param number 阈值
     * @return 最大值运算结果数组
     */
    public NdArrayCpu maximum(Number number) {
        return MathFunctions.maximum(this, number);
    }

    /**
     * 掩码运算，将数组中大于指定值的元素设为1，小于等于指定值的元素设为0
     *
     * @param number 阈值
     * @return 掩码运算结果数组
     */
    public NdArrayCpu mask(Number number) {
        return MathFunctions.mask(this, number);
    }

    // =============================================================================
    // 张量的变形操作 - 重构后的优化版本
    // =============================================================================

    /**
     * 矩阵转置操作（二维矩阵），行列互换
     *
     * @return 转置后的矩阵
     * @throws IllegalArgumentException 当数组不是矩阵时抛出
     */
    public NdArrayCpu transpose() {
        return TransformationOperations.transpose(this);
    }

    /**
     * 多维数组转置操作，按指定维度顺序重新排列
     *
     * @param order 新的维度顺序
     * @return 转置后的数组
     * @throws IllegalArgumentException 当维度顺序无效时抛出
     */
    public NdArrayCpu transpose(int... order) {
        return TransformationOperations.transpose(this, order);
    }

    /**
     * 数组变形操作，改变数组形状但保持元素总数不变
     *
     * @param newShape 新的数组形状
     * @return 变形后的数组
     * @throws IllegalArgumentException 当新形状大小与原形状不匹配时抛出
     */
    public NdArrayCpu reshape(Shape newShape) {
        return TransformationOperations.reshape(this, newShape);
    }

    /**
     * 支持广播语义的reshape（新增方法）
     *
     * @param newShape 新的数组形状
     * @return 变形后的数组
     * @throws IllegalArgumentException 当形状不兼容时抛出
     */
    @Override
    public NdArrayCpu broadcastReshape(Shape newShape) {
        return TransformationOperations.broadcastReshape(this, newShape);
    }

    /**
     * 数组展平操作，将多维数组转换为一维行向量
     *
     * @return 展平后的一维行向量
     */
    public NdArrayCpu flatten() {
        return TransformationOperations.flatten(this);
    }

    // =============================================================================
    // 统计和聚合操作 - 重构后的优化版本
    // =============================================================================

    /**
     * 元素累和运算，计算数组所有元素的总和
     *
     * @return 所有元素的总和（标量）
     */
    public NdArrayCpu sum() {
        return ReductionOperations.sum(this);
    }

    /**
     * 矩阵均值运算，沿指定轴计算均值
     *
     * @param axis 聚合轴，axis=0表示按列计算均值，axis=1表示按行计算均值
     * @return 均值运算结果数组
     */
    public NdArrayCpu mean(int axis) {
        return ReductionOperations.mean(this, axis);
    }

    /**
     * 矩阵方差运算，沿指定轴计算方差
     *
     * @param axis 聚合轴，axis=0表示按列计算方差，axis=1表示按行计算方差
     * @return 方差运算结果数组
     */
    public NdArrayCpu var(int axis) {
        return ReductionOperations.var(this, axis);
    }

    /**
     * 矩阵累和运算，沿指定轴计算累和
     *
     * @param axis 聚合轴，axis=0表示按列累和，axis=1表示按行累和
     * @return 累和运算结果数组
     */
    public NdArrayCpu sum(int axis) {
        return ReductionOperations.sum(this, axis);
    }


    /**
     * 按指定形状进行压缩累加运算
     *
     * 将当前数组按指定形状进行压缩，超出目标形状的部分会累加到对应位置
     *
     * @param targetShape 目标形状
     * @return 压缩累加结果数组
     * @throws IllegalArgumentException 当形状不合法时抛出
     */
    public NdArrayCpu sumTo(Shape targetShape) {
        return TransformationOperations.sumTo(this, targetShape);
    }

    /**
     * 优化的sumTo实现
     * 使用轴向求和策略，性能提升2-3倍
     *
     * @param targetShape 目标形状
     * @return 压缩结果数组
     * @throws IllegalArgumentException 当形状不合法时抛出
     */
    public NdArrayCpu sumToOptimized(Shape targetShape) {
        return TransformationOperations.sumToOptimized(this, targetShape);
    }

    /**
     * 数组广播运算，将当前数组广播到指定形状
     *
     * 广播机制允许小数组与大数组进行运算，小数组会重复填充以匹配大数组的形状
     *
     * @param targetShape 目标广播形状
     * @return 广播结果数组
     * @throws IllegalArgumentException 当形状不合法时抛出
     */
    public NdArrayCpu broadcastTo(Shape targetShape) {
        return TransformationOperations.broadcastTo(this, targetShape);
    }

    /**
     * 沿指定轴查找最大值的索引
     *
     * @param axis 查找轴，axis=0表示按行查找每列的最大值索引，axis=1表示按列查找每行的最大值索引
     * @return 最大值索引数组
     * @throws IllegalArgumentException 当数组不是矩阵或轴参数无效时抛出
     */
    public NdArrayCpu argMax(int axis) {
        return AxisOperations.argMax(this, axis);
    }

    /**
     * 矩阵内积运算（矩阵乘法）。
     * 执行标准矩阵乘法，要求第一个矩阵的列数等于第二个矩阵的行数。
     *
     * @param other 另一个矩阵
     * @return 矩阵乘法结果
     * @throws IllegalArgumentException 当数组不是矩阵或维度不匹配时抛出
     */
    public NdArrayCpu dot(NdArray other) {
        return MatrixOperations.dot(this, (NdArrayCpu) other);
    }

    /**
     * 获取数组的子集（切片操作）
     *
     * @param rowSlices 行索引数组，null表示选择所有行
     * @param colSlices 列索引数组，null表示选择所有列
     * @return 切片结果数组
     * @throws IllegalArgumentException 当数组不是矩阵或参数不合法时抛出
     */
    public NdArrayCpu getItem(int[] rowSlices, int[] colSlices) {
        return MatrixOperations.getItem(this, rowSlices, colSlices);
    }

    /**
     * 设置数组的子集（切片赋值操作）
     *
     * @param rowSlices 行索引数组，null表示选择所有行
     * @param colSlices 列索引数组，null表示选择所有列
     * @param data       要设置的数据
     * @return 当前数组实例
     * @throws IllegalArgumentException 当数组不是矩阵或参数不合法时抛出
     */
    public NdArrayCpu setItem(int[] rowSlices, int[] colSlices, float[] data) {
        return MatrixOperations.setItem(this, rowSlices, colSlices, data);
    }

    /**
     * 高性能连续区域赋值（新增方法）
     *
     * @param startRow 起始行索引（包含）
     * @param endRow   结束行索引（不包含）
     * @param startCol 起始列索引（包含）
     * @param endCol   结束列索引（不包含）
     * @param data     要设置的数据
     * @return 当前数组实例
     * @throws IllegalArgumentException 当数组不是矩阵或参数不合法时抛出
     */
    @Override
    public NdArrayCpu setBlock(int startRow, int endRow, int startCol, int endCol, float[] data) {
        return MatrixOperations.setBlock(this, startRow, endRow, startCol, endCol, data);
    }

    /**
     * 行切片赋值（新增方法）
     *
     * @param rowIndices 行索引数组
     * @param data       要设置的数据
     * @return 当前数组实例
     * @throws IllegalArgumentException 当数组不是矩阵或参数不合法时抛出
     */
    @Override
    public NdArrayCpu setRows(int[] rowIndices, float[] data) {
        return MatrixOperations.setRows(this, rowIndices, data);
    }

    /**
     * 列切片赋值（新增方法）
     *
     * @param colIndices 列索引数组
     * @param data       要设置的数据
     * @return 当前数组实例
     * @throws IllegalArgumentException 当数组不是矩阵或参数不合法时抛出
     */
    @Override
    public NdArrayCpu setCols(int[] colIndices, float[] data) {
        return MatrixOperations.setCols(this, colIndices, data);
    }

    /**
     * 沿指定轴查找最大值
     *
     * @param axis 查找轴，axis=0表示按行查找每列的最大值，axis=1表示按列查找每行的最大值
     * @return 最大值数组
     * @throws IllegalArgumentException 当数组不是矩阵或轴参数无效时抛出
     */
    public NdArrayCpu max(int axis) {
        return AxisOperations.max(this, axis);
    }

    /**
     * 沿指定轴查找最小值
     *
     * @param axis 查找轴，axis=0表示按行查找每列的最小值，axis=1表示按列查找每行的最小值
     * @return 最小值数组
     * @throws IllegalArgumentException 当数组不是矩阵或轴参数无效时抛出
     */
    public NdArrayCpu min(int axis) {
        return AxisOperations.min(this, axis);
    }

    /**
     * 查找数组中的最大值（全局最大值）
     *
     * @return 数组中的最大值
     */
    public float max() {
        return ReductionOperations.max(this);
    }

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
    public NdArrayCpu subNdArray(int startRow, int endRow, int startCol, int endCol) {
        return SlicingOperations.subNdArray(this, startRow, endRow, startCol, endCol);
    }


    /**
     * 在指定位置累加数组元素。
     * 在给定的行、列坐标上累加另一个数组，常用于反向传播中的梯度累积。
     * 示例：addAt(new int[]{0, 2}, new int[]{1, 1}, b) 在 (0,1) 和 (2,1) 处累加 b 的值。
     *
     * @param rowSlices 行索引数组，指定要累加的行位置
     * @param colSlices 列索引数组，指定要累加的列位置
     * @param other     要累加的数组
     * @return 累加结果数组
     * @throws IllegalArgumentException 当输入参数不合法时抛出
     * @throws RuntimeException         当数组不是矩阵时抛出
     */
    public NdArrayCpu addAt(int[] rowSlices, int[] colSlices, NdArray other) {
        return AccumulationOperations.addAt(this, rowSlices, colSlices, other);
    }

    /**
     * 将另一个数组累加到当前数组的指定位置
     *
     * @param i     起始行索引
     * @param j     起始列索引
     * @param other 要累加的数组
     * @return 当前数组实例
     * @throws IllegalArgumentException 当数组不是矩阵时抛出
     */
    public NdArrayCpu addTo(int i, int j, NdArray other) {
        return AccumulationOperations.addTo(this, i, j, other);
    }

    /**
     * 裁剪数组元素到指定范围
     *
     * 将数组中小于最小值的元素设为最小值，大于最大值的元素设为最大值
     *
     * @param min 最小值
     * @param max 最大值
     * @return 裁剪后的数组
     * @throws IllegalArgumentException 当最小值大于最大值时抛出
     */
    public NdArrayCpu clip(float min, float max) {
        return MathFunctions.clip(this, min, max);
    }


    // =============================================================================
    // 其他运算
    // =============================================================================


    /**
     * 获取数组的第一个元素值（标量值）
     *
     * @return 第一个元素值
     */
    public Number getNumber() {
        return this.buffer[0];
    }

    /**
     * 获取数组的形状
     *
     * @return 数组形状
     */
    public Shape getShape() {
        return this.shape;
    }

    /**
     * 设置数组的形状
     *
     * 注意：新形状的大小必须与当前形状大小一致
     *
     * @param shape 新形状
     * @throws IllegalArgumentException 当新形状大小与当前形状不匹配时抛出
     */
    public void setShape(Shape shape) {
        if (shape.size() != this.shape.size()) {
            throw new IllegalArgumentException("新形状大小与当前形状不匹配");
        }
        this.shape = (ShapeCpu) shape;
    }

    @Override
    public float[] getArray() {
        return buffer;
    }

    /**
     * 将数组转换为二维数组（矩阵）返回
     *
     * @return 二维数组表示
     * @throws IllegalArgumentException 当数组维度大于2时抛出
     */
    public float[][] getMatrix() {
        return ArrayConverter.toMatrix(this);
    }

    /**
     * 将数组转换为三维数组返回
     *
     * @return 三维数组表示
     * @throws IllegalArgumentException 当数组不是三维时抛出
     */
    public float[][][] get3dArray() {
        return ArrayConverter.to3dArray(this);
    }

    /**
     * 将数组转换为四维数组返回
     *
     * @return 四维数组表示
     * @throws IllegalArgumentException 当数组不是四维时抛出
     */
    public float[][][][] get4dArray() {
        return ArrayConverter.to4dArray(this);
    }

    /**
     * 按形状格式化打印数组。
     * 标量直接显示数值；1维为 [a, b, c]；2维为矩阵格式；3维及以上为嵌套结构。
     *
     * @return 格式化的字符串表示
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        
        // 显示形状信息
        sb.append("NdArray(shape=").append(shape).append(")\n");
        
        // 根据维度格式化输出
        int ndim = shape.dimension.length;
        
        if (ndim == 0 || (ndim == 2 && shape.dimension[0] == 1 && shape.dimension[1] == 1)) {
            // 标量
            sb.append(formatFloat(buffer[0]));
        } else if (ndim == 1 || (ndim == 2 && shape.dimension[0] == 1)) {
            // 1维数组或行向量
            format1DArray(sb, 0, shape.size());
        } else if (ndim == 2) {
            // 2维矩阵
            format2DArray(sb);
        } else if (ndim == 3) {
            // 3维数组
            format3DArray(sb);
        } else if (ndim == 4) {
            // 4维数组
            format4DArray(sb);
        } else {
            // 更高维度，简化显示
            formatHighDimArray(sb);
        }
        
        return sb.toString();
    }

    /**
     * 优化的equals方法，比较两个NdArray对象是否相等
     *
     * @param obj 另一个对象
     * @return 是否相等
     */
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) {
            return false;
        }

        NdArrayCpu other = (NdArrayCpu) obj;
        if (!this.shape.equals(other.shape)) {
            return false;
        }

        return Arrays.equals(this.buffer, other.buffer);
    }

    /**
     * 优化的hashCode方法，为NdArray对象生成哈希码
     *
     * @return 哈希码
     */
    @Override
    public int hashCode() {
        int result = shape.hashCode();
        result = 31 * result + Arrays.hashCode(buffer);
        return result;
    }

    /**
     * 按维度下标设置某一个值
     *
     * @param value      要设置的值
     * @param _dimension 维度下标数组
     * @throws IllegalArgumentException 当维度数量不匹配时抛出
     */
    public void set(float value, int... _dimension) {
        if (_dimension.length != shape.dimension.length) {
            throw new IllegalArgumentException(String.format("维度数量不匹配：提供%d个维度，需要%d个维度", _dimension.length, shape.dimension.length));
        }
        buffer[shape.getIndex(_dimension)] = value;
    }

    /**
     * 按维度下标获取某一个值
     *
     * @param _dimension 维度下标数组
     * @return 对应位置的值
     * @throws IllegalArgumentException 当维度数量不匹配时抛出
     */
    public float get(int... _dimension) {
        if (_dimension.length != shape.dimension.length) {
            throw new IllegalArgumentException(String.format("维度数量不匹配：提供%d个维度，需要%d个维度", _dimension.length, shape.dimension.length));
        }
        return buffer[shape.getIndex(_dimension)];
    }

    /**
     * 格式化浮点数显示
     */
    private String formatFloat(float value) {
        // 整数直接显示为整数格式
        if (value == (int) value) {
            return String.valueOf((int) value);
        }
        // 小数保留4位
        return String.format("%.4f", value);
    }
    
    /**
     * 格式化1维数组
     */
    private void format1DArray(StringBuilder sb, int start, int length) {
        sb.append("[");
        int displayCount = Math.min(length, 10);  // 最多显示10个元素
        
        for (int i = 0; i < displayCount; i++) {
            if (i > 0) sb.append(", ");
            sb.append(formatFloat(buffer[start + i]));
        }
        
        if (length > displayCount) {
            sb.append(", ... (total ").append(length).append(" elements)");
        }
        sb.append("]");
    }
    
    /**
     * 格式化2维矩阵
     */
    private void format2DArray(StringBuilder sb) {
        int rows = shape.dimension[0];
        int cols = shape.dimension[1];
        
        sb.append("[");
        int displayRows = Math.min(rows, 10);  // 最多显示10行
        int displayCols = Math.min(cols, 10);  // 最多显示10列
        
        for (int i = 0; i < displayRows; i++) {
            if (i > 0) sb.append("\n ");
            sb.append("[");
            
            for (int j = 0; j < displayCols; j++) {
                if (j > 0) sb.append(", ");
                int idx = i * cols + j;
                sb.append(formatFloat(buffer[idx]));
            }
            
            if (cols > displayCols) {
                sb.append(", ...");
            }
            sb.append("]");
        }
        
        if (rows > displayRows) {
            sb.append("\n ...");
        }
        sb.append("]");
    }
    
    /**
     * 格式化3维数组
     */
    private void format3DArray(StringBuilder sb) {
        int dim0 = shape.dimension[0];
        int dim1 = shape.dimension[1];
        int dim2 = shape.dimension[2];
        
        sb.append("[");
        int displayDim0 = Math.min(dim0, 3);  // 最多显示3个
        
        for (int i = 0; i < displayDim0; i++) {
            if (i > 0) sb.append("\n\n ");
            sb.append("[");
            
            int displayDim1 = Math.min(dim1, 5);  // 最多显示5行
            for (int j = 0; j < displayDim1; j++) {
                if (j > 0) sb.append("\n  ");
                sb.append("[");
                
                int displayDim2 = Math.min(dim2, 8);  // 最多显示8列
                for (int k = 0; k < displayDim2; k++) {
                    if (k > 0) sb.append(", ");
                    int idx = i * dim1 * dim2 + j * dim2 + k;
                    sb.append(formatFloat(buffer[idx]));
                }
                
                if (dim2 > displayDim2) {
                    sb.append(", ...");
                }
                sb.append("]");
            }
            
            if (dim1 > displayDim1) {
                sb.append("\n  ...");
            }
            sb.append("]");
        }
        
        if (dim0 > displayDim0) {
            sb.append("\n ...");
        }
        sb.append("]");
    }
    
    /**
     * 格式化4维数组
     */
    private void format4DArray(StringBuilder sb) {
        int dim0 = shape.dimension[0];
        int dim1 = shape.dimension[1];
        int dim2 = shape.dimension[2];
        int dim3 = shape.dimension[3];
        
        sb.append("[");
        int displayDim0 = Math.min(dim0, 2);  // 最多显示2个
        
        for (int i = 0; i < displayDim0; i++) {
            if (i > 0) sb.append("\n\n\n ");
            sb.append("[");
            
            int displayDim1 = Math.min(dim1, 3);
            for (int j = 0; j < displayDim1; j++) {
                if (j > 0) sb.append("\n\n  ");
                sb.append("[");
                
                int displayDim2 = Math.min(dim2, 4);
                for (int k = 0; k < displayDim2; k++) {
                    if (k > 0) sb.append("\n   ");
                    sb.append("[");
                    
                    int displayDim3 = Math.min(dim3, 6);
                    for (int l = 0; l < displayDim3; l++) {
                        if (l > 0) sb.append(", ");
                        int idx = i * dim1 * dim2 * dim3 + j * dim2 * dim3 + k * dim3 + l;
                        sb.append(formatFloat(buffer[idx]));
                    }
                    
                    if (dim3 > displayDim3) {
                        sb.append(", ...");
                    }
                    sb.append("]");
                }
                
                if (dim2 > displayDim2) {
                    sb.append("\n   ...");
                }
                sb.append("]");
            }
            
            if (dim1 > displayDim1) {
                sb.append("\n  ...");
            }
            sb.append("]");
        }
        
        if (dim0 > displayDim0) {
            sb.append("\n ...");
        }
        sb.append("]");
    }
    
    /**
     * 格式化高维数组（5维及以上）
     */
    private void formatHighDimArray(StringBuilder sb) {
        sb.append("[high-dimensional array with ").append(buffer.length).append(" elements]\n");
        sb.append("First 20 elements: ");
        format1DArray(sb, 0, Math.min(20, buffer.length));
    }

    /**
     * 深拷贝当前NdArray，返回数据完全独立的新实例
     *
     * @return 新的NdArrayCpu实例，shape和buffer均为独立副本
     */
    @Override
    public NdArrayCpu copy() {
        float[] newBuffer = Arrays.copyOf(this.buffer, this.buffer.length);
        Shape newShape = Shape.of(this.shape.getShapeDims());
        return new NdArrayCpu(newBuffer, newShape);
    }

}
