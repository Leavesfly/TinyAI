package io.leavesfly.tinyai.ndarr.cpu;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.NdArrayUtil;
import io.leavesfly.tinyai.ndarr.Shape;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;
import java.util.function.BiPredicate;
import java.util.function.BinaryOperator;
import java.util.function.UnaryOperator;

/**
 * N维数组类，支持标量、向量、矩阵等多维数据结构
 *
 * <p>该类经过重构优化，提供更加优雅的API和更好的性能，是深度学习框架的核心数据结构。</p>
 *
 * <p>主要特性：</p>
 * <ul>
 *   <li>支持任意维度的数组操作</li>
 *   <li>高效的内存管理</li>
 *   <li>丰富的数学运算和张量操作</li>
 *   <li>广播机制支持</li>
 *   <li>自动微分支持</li>
 * </ul>
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


    // 常用的数学常数，用于数值计算中的比较
    private static final float EPSILON = 1e-7f;

    // =============================================================================
    // NdArray的创建函数 - 重构后的构造方法
    // =============================================================================

    /**
     * 默认构造方法，创建空的NdArray实例
     *
     * <p>注意：此构造方法不会初始化shape和buffer，需要手动设置</p>
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
        validateDataShape(data.length, shape.size());
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
     * <p>支持2D、3D、4D数组的创建</p>
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

    // 优化的初始化方法
    private void initFromArray(float[][] data) {
        validateArrayDimensions(data);
        this.shape = ShapeCpu.of(data.length, data[0].length);
        this.buffer = new float[shape.size()];
        flattenArray(data, this.buffer, 0);
    }

    private void initFromArray(float[][][] data) {
        validateArrayDimensions(data);
        this.shape = ShapeCpu.of(data.length, data[0].length, data[0][0].length);
        this.buffer = new float[shape.size()];
        flattenArray(data, this.buffer, 0);
    }

    private void initFromArray(float[][][][] data) {
        validateArrayDimensions(data);
        this.shape = ShapeCpu.of(data.length, data[0].length, data[0][0].length, data[0][0][0].length);
        this.buffer = new float[shape.size()];
        flattenArray(data, this.buffer, 0);
    }

    // 通用数组展平方法
    private int flattenArray(Object array, float[] buffer, int index) {
        if (array instanceof float[]) {
            float[] arr = (float[]) array;
            System.arraycopy(arr, 0, buffer, index, arr.length);
            return index + arr.length;
        } else if (array.getClass().isArray()) {
            Object[] arr = (Object[]) array;
            for (Object subArray : arr) {
                index = flattenArray(subArray, buffer, index);
            }
        }
        return index;
    }

    // 验证数组维度一致性
    private void validateArrayDimensions(Object array) {
        if (array == null) {
            throw new IllegalArgumentException("数组不能为null");
        }

        if (!array.getClass().isArray()) {
            throw new IllegalArgumentException("输入必须是数组类型");
        }

        // 递归验证各维度的一致性
        validateDimensionConsistency(array, 0, new java.util.ArrayList<>());
    }

    /**
     * 递归验证多维数组各维度大小的一致性
     *
     * <p>确保多维数组在每个维度上的大小是一致的，避免不规则数组</p>
     *
     * @param array         当前数组对象
     * @param depth         当前递归深度
     * @param expectedSizes 各层级的期望大小列表
     * @throws IllegalArgumentException 当数组维度不一致或包含null元素时抛出
     */
    private void validateDimensionConsistency(Object array, int depth, java.util.List<Integer> expectedSizes) {
        if (array == null) {
            throw new IllegalArgumentException(String.format("第%d维度包含null元素", depth));
        }

        if (!array.getClass().isArray()) {
            return; // 到达最底层元素
        }

        int length = java.lang.reflect.Array.getLength(array);

        // 检查当前维度的大小是否与期望一致
        if (expectedSizes.size() <= depth) {
            // 第一次遇到这个深度，记录期望大小
            expectedSizes.add(length);
        } else if (!expectedSizes.get(depth).equals(length)) {
            // 发现大小不一致
            throw new IllegalArgumentException(String.format("第%d维度大小不一致：期望%d，实际%d", depth, expectedSizes.get(depth), length));
        }

        // 检查维度不能为0
        if (length == 0) {
            throw new IllegalArgumentException(String.format("第%d维度大小不能为0", depth));
        }

        // 递归检查下一级维度
        for (int i = 0; i < length; i++) {
            Object subArray = java.lang.reflect.Array.get(array, i);
            validateDimensionConsistency(subArray, depth + 1, expectedSizes);
        }
    }

    /**
     * 验证数据长度与形状大小是否匹配
     *
     * @param dataLength 数据数组长度
     * @param shapeSize  形状指定的总元素数量
     * @throws IllegalArgumentException 当长度不匹配时抛出
     */
    private void validateDataShape(int dataLength, int shapeSize) {
        if (dataLength != shapeSize) {
            throw new IllegalArgumentException(String.format("数据长度 %d 与形状大小 %d 不匹配", dataLength, shapeSize));
        }
    }

    /**
     * 验证形状是否为矩阵（二维数组）
     *
     * @param shape 待验证的形状
     * @throws IllegalArgumentException 当形状不是矩阵时抛出
     */
    private static void validateMatrixShape(Shape shape) {
        if (!shape.isMatrix()) {
            throw new IllegalArgumentException("操作仅适用于矩阵（二维数组）");
        }
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
        return new NdArrayCpu(shape); // 默认就是全零
    }

    /**
     * 创建指定形状的全一数组
     *
     * @param shape 数组形状
     * @return 全一数组
     */
    public static NdArrayCpu ones(Shape shape) {
        NdArrayCpu result = new NdArrayCpu(shape);
        result.fillAll(1.0f);
        return result;
    }

    /**
     * 创建指定形状的单位矩阵（对角矩阵）
     *
     * @param shape 矩阵形状（必须为方形矩阵）
     * @return 单位矩阵
     * @throws IllegalArgumentException 当形状不是矩阵或不是方形矩阵时抛出
     */
    public static NdArrayCpu eye(Shape shape) {
        validateMatrixShape(shape);
        NdArrayCpu result = new NdArrayCpu(shape);
        int minDim = Math.min(shape.getRow(), shape.getColumn());
        for (int i = 0; i < minDim; i++) {
            result.buffer[i * shape.getColumn() + i] = 1.0f;
        }
        return result;
    }

    /**
     * 创建指定形状和值的数组
     *
     * @param shape 数组形状
     * @param value 填充值
     * @return 指定值填充的数组
     */
    public static NdArrayCpu like(Shape shape, Number value) {
        NdArrayCpu result = new NdArrayCpu(shape);
        result.fillAll(value.floatValue());
        return result;
    }

    /**
     * 创建与当前数组形状相同但指定值的数组
     *
     * @param value 填充值
     * @return 指定值填充的数组
     */
    public NdArrayCpu like(Number value) {
        return NdArrayCpu.like(this.shape, value);
    }

    /**
     * 创建标准正态分布（均值为0，标准差为1）的随机数组
     *
     * @param shape 数组形状
     * @return 标准正态分布随机数组
     */
    public static NdArrayCpu likeRandomN(Shape shape) {
        return likeRandomN(shape, 0);
    }

    /**
     * 创建标准正态分布（均值为0，标准差为1）的随机数组（可指定随机种子）
     *
     * @param shape 数组形状
     * @param seed  随机种子，0表示使用默认种子
     * @return 标准正态分布随机数组
     */
    public static NdArrayCpu likeRandomN(Shape shape, long seed) {
        NdArrayCpu result = new NdArrayCpu(shape);
        Random random = new Random(seed);
        for (int i = 0; i < result.buffer.length; i++) {
            result.buffer[i] = (float) random.nextGaussian();
        }
        return result;
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
    public static NdArrayCpu likeRandom(float min, float max, Shape shape, long seed) {
        NdArrayCpu result = new NdArrayCpu(shape);
        Random random = new Random(seed);
        for (int i = 0; i < result.buffer.length; i++) {
            result.buffer[i] = random.nextFloat() * (max - min) + min;
        }
        return result;
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
        if (num <= 0) {
            throw new IllegalArgumentException("数量必须大于0");
        }
        NdArrayCpu result = likeRandom(min, max, ShapeCpu.of(1, num));
        Arrays.sort(result.buffer);
        return result;
    }

    // =============================================================================
    // 基础四则运算 - 重构后的统一模式
    // =============================================================================

    /**
     * 通用的二元运算方法，对两个相同形状的数组进行元素级运算
     *
     * @param _other        另一个操作数数组
     * @param operation     二元运算操作函数
     * @param operationName 操作名称，用于错误提示
     * @return 运算结果数组
     * @throws IllegalArgumentException 当两个数组形状不一致时抛出
     */
    private NdArrayCpu binaryOperation(NdArray _other, BinaryOperator<Float> operation, String operationName) {
        NdArrayCpu other = (NdArrayCpu) _other;
        validateShapeCompatibility(this.shape, other.shape, operationName);
        NdArrayCpu result = new NdArrayCpu(this.shape);
        for (int i = 0; i < this.buffer.length; i++) {
            result.buffer[i] = operation.apply(this.buffer[i], other.buffer[i]);
        }
        return result;
    }

    /**
     * 通用的与标量运算方法，对数组与标量进行运算
     *
     * @param scalar    标量值
     * @param operation 二元运算操作函数
     * @return 运算结果数组
     */
    private NdArrayCpu scalarOperation(Number scalar, BinaryOperator<Float> operation) {
        NdArrayCpu result = new NdArrayCpu(this.shape);
        float scalarValue = scalar.floatValue();
        for (int i = 0; i < this.buffer.length; i++) {
            result.buffer[i] = operation.apply(this.buffer[i], scalarValue);
        }
        return result;
    }

    /**
     * 验证两个数组的形状是否兼容（完全相同）
     *
     * @param shape1        第一个数组形状
     * @param shape2        第二个数组形状
     * @param operationName 操作名称，用于错误提示
     * @throws IllegalArgumentException 当形状不一致时抛出
     */
    private static void validateShapeCompatibility(Shape shape1, Shape shape2, String operationName) {
        if (!shape1.equals(shape2)) {
            throw new IllegalArgumentException(String.format("%s 操作要求形状一致：%s vs %s", operationName, shape1, shape2));
        }
    }

    /**
     * 数组加法运算，对应元素相加
     *
     * @param other 另一个操作数数组
     * @return 加法运算结果
     * @throws IllegalArgumentException 当两个数组形状不一致时抛出
     */
    public NdArrayCpu add(NdArray other) {
        return binaryOperation(other, Float::sum, "加法");
    }

    /**
     * 数组减法运算，对应元素相减
     *
     * @param other 另一个操作数数组
     * @return 减法运算结果
     * @throws IllegalArgumentException 当两个数组形状不一致时抛出
     */
    public NdArrayCpu sub(NdArray other) {
        return binaryOperation((NdArrayCpu) other, (a, b) -> a - b, "减法");
    }

    /**
     * 数组乘法运算，对应元素相乘
     *
     * @param other 另一个操作数数组
     * @return 乘法运算结果
     * @throws IllegalArgumentException 当两个数组形状不一致时抛出
     */
    public NdArrayCpu mul(NdArray other) {
        return binaryOperation((NdArrayCpu) other, (a, b) -> a * b, "乘法");
    }

    /**
     * 数组与标量相乘
     *
     * @param number 标量值
     * @return 乘法运算结果
     */
    public NdArrayCpu mulNum(Number number) {
        return scalarOperation(number, (a, b) -> a * b);
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
        return binaryOperation(other, (a, b) -> {
            if (Math.abs(b) < EPSILON) {
                throw new ArithmeticException("除数接近0");
            }
            return a / b;
        }, "除法");
    }

    /**
     * 数组与标量相除
     *
     * @param number 标量值
     * @return 除法运算结果
     * @throws ArithmeticException 当除数为0时抛出
     */
    public NdArray divNum(Number number) {
        float value = number.floatValue();
        if (Math.abs(value) < EPSILON) {
            throw new ArithmeticException("除数不能为0");
        }
        return scalarOperation(number, (a, b) -> a / b);
    }

    // =============================================================================
    // 逻辑运算 - 重构后的统一模式
    // =============================================================================

    /**
     * 通用的一元运算方法，对数组每个元素进行一元运算
     *
     * @param operation 一元运算操作函数
     * @return 运算结果数组
     */
    private NdArrayCpu unaryOperation(UnaryOperator<Float> operation) {
        NdArrayCpu result = new NdArrayCpu(this.shape);
        for (int i = 0; i < this.buffer.length; i++) {
            result.buffer[i] = operation.apply(this.buffer[i]);
        }
        return result;
    }

    /**
     * 通用的比较运算方法，对两个数组进行元素级比较
     *
     * @param other         另一个操作数数组
     * @param comparison    比较操作函数
     * @param operationName 操作名称，用于错误提示
     * @return 比较结果数组，1.0表示true，0.0表示false
     * @throws IllegalArgumentException 当两个数组形状不一致时抛出
     */
    private NdArrayCpu comparisonOperation(NdArrayCpu other, BiPredicate<Float, Float> comparison, String operationName) {
        validateShapeCompatibility(this.shape, other.shape, operationName);
        NdArrayCpu result = new NdArrayCpu(this.shape);
        for (int i = 0; i < this.buffer.length; i++) {
            boolean compResult = comparison.test(this.buffer[i], other.buffer[i]);
            result.buffer[i] = compResult ? 1.0f : 0.0f;
        }
        return result;
    }

    /**
     * 取反操作，对数组每个元素取负值
     *
     * @return 取反后的数组
     */
    public NdArrayCpu neg() {
        return unaryOperation(x -> -x);
    }

    /**
     * 绝对值运算，对数组每个元素取绝对值
     *
     * @return 绝对值数组
     */
    public NdArrayCpu abs() {
        return unaryOperation(Math::abs);
    }

    /**
     * 相等比较运算，比较两个数组对应元素是否相等
     *
     * @param other 另一个操作数数组
     * @return 比较结果数组，1.0表示相等，0.0表示不相等
     * @throws IllegalArgumentException 当两个数组形状不一致时抛出
     */
    public NdArrayCpu eq(NdArray other) {
        return comparisonOperation((NdArrayCpu) other, Float::equals, "相等比较");
    }

    /**
     * 大于比较运算，比较当前数组元素是否大于另一个数组对应元素
     *
     * @param other 另一个操作数数组
     * @return 比较结果数组，1.0表示大于，0.0表示不大于
     * @throws IllegalArgumentException 当两个数组形状不一致时抛出
     */
    public NdArrayCpu gt(NdArray other) {
        return comparisonOperation((NdArrayCpu) other, (a, b) -> a > b, "大于比较");
    }

    /**
     * 小于比较运算，比较当前数组元素是否小于另一个数组对应元素
     *
     * @param other 另一个操作数数组
     * @return 比较结果数组，1.0表示小于，0.0表示不小于
     * @throws IllegalArgumentException 当两个数组形状不一致时抛出
     */
    public NdArrayCpu lt(NdArray other) {
        return comparisonOperation((NdArrayCpu) other, (a, b) -> a < b, "小于比较");
    }

    /**
     * 矩阵全元素大于比较，判断当前数组是否所有元素都大于另一个数组对应元素
     *
     * @param _other 另一个操作数数组
     * @return 比较结果，true表示所有元素都大于，false表示存在不大于的元素
     * @throws IllegalArgumentException 当两个数组形状不一致时抛出
     */
    public boolean isLar(NdArray _other) {
        NdArrayCpu other = (NdArrayCpu) _other;
        validateShapeCompatibility(this.shape, other.shape, "全元素比较");
        for (int i = 0; i < this.buffer.length; i++) {
            if (this.buffer[i] <= other.buffer[i]) {
                return false;
            }
        }
        return true;
    }

    // =============================================================================
    // 基本数学函数 - 重构后的统一模式
    // =============================================================================

    /**
     * 通用的数学函数运算方法，对数组每个元素应用数学函数
     *
     * @param mathFunc 数学函数操作
     * @return 运算结果数组
     */
    private NdArrayCpu mathOperation(java.util.function.Function<Double, Double> mathFunc) {
        NdArrayCpu result = new NdArrayCpu(this.shape);
        for (int i = 0; i < this.buffer.length; i++) {
            result.buffer[i] = mathFunc.apply((double) this.buffer[i]).floatValue();
        }
        return result;
    }

    /**
     * 幂运算，对数组每个元素进行幂运算
     *
     * @param number 幂指数
     * @return 幂运算结果数组
     */
    public NdArrayCpu pow(Number number) {
        float exponent = number.floatValue();
        return mathOperation(x -> Math.pow(x, exponent));
    }

    /**
     * 平方运算，对数组每个元素进行平方运算
     *
     * @return 平方运算结果数组
     */
    public NdArrayCpu square() {
        return pow(2f);
    }

    /**
     * 平方根运算，对数组每个元素进行开方运算
     *
     * @return 平方根运算结果数组
     */
    public NdArrayCpu sqrt() {
        return mathOperation(Math::sqrt);
    }

    /**
     * 自然指数运算，对数组每个元素进行e为底的指数运算
     *
     * @return 指数运算结果数组
     */
    public NdArrayCpu exp() {
        return mathOperation(Math::exp);
    }

    /**
     * 正弦函数运算，对数组每个元素进行sin运算
     *
     * @return 正弦运算结果数组
     */
    public NdArrayCpu sin() {
        return mathOperation(Math::sin);
    }

    /**
     * 余弦函数运算，对数组每个元素进行cos运算
     *
     * @return 余弦运算结果数组
     */
    public NdArrayCpu cos() {
        return mathOperation(Math::cos);
    }

    /**
     * 双曲正切函数运算，对数组每个元素进行tanh运算
     *
     * @return 双曲正切运算结果数组
     */
    public NdArrayCpu tanh() {
        return mathOperation(Math::tanh);
    }

    /**
     * Sigmoid函数运算，对数组每个元素进行sigmoid运算
     *
     * <p>Sigmoid函数公式：f(x) = 1 / (1 + e^(-x))</p>
     *
     * @return Sigmoid运算结果数组
     */
    public NdArrayCpu sigmoid() {
        return mathOperation(x -> 1.0 / (1.0 + Math.exp(-x)));
    }

    /**
     * 自然对数运算，对数组每个元素进行ln运算
     *
     * @return 对数运算结果数组
     * @throws ArithmeticException 当输入值小于等于0时抛出
     */
    public NdArrayCpu log() {
        return mathOperation(x -> {
            if (x <= 0) {
                throw new ArithmeticException("对数的输入必须大于0");
            }
            return Math.log(x);
        });
    }

    /**
     * Softmax函数运算，按行计算概率分布
     *
     * <p>Softmax函数公式：softmax(x_i) = exp(x_i) / Σ(exp(x_j))</p>
     * <p>使用数值稳定版本实现，避免指数运算溢出</p>
     *
     * @return Softmax运算结果数组
     * @throws IllegalArgumentException 当数组不是矩阵时抛出
     */
    public NdArrayCpu softMax() {
        validateMatrixShape(this.shape);
        NdArrayCpu result = new NdArrayCpu(this.shape);

        // 使用数值稳定版本，避免指数爆炸
        for (int i = 0; i < shape.getRow(); i++) {
            // 找到该行的最大值用于数值稳定
            float maxVal = Float.NEGATIVE_INFINITY;
            for (int j = 0; j < shape.getColumn(); j++) {
                int index = i * shape.getColumn() + j;
                maxVal = Math.max(maxVal, this.buffer[index]);
            }

            // 计算exp和总和
            float sum = 0f;
            for (int j = 0; j < shape.getColumn(); j++) {
                int index = i * shape.getColumn() + j;
                float expVal = (float) Math.exp(this.buffer[index] - maxVal);
                result.buffer[index] = expVal;
                sum += expVal;
            }

            // 归一化
            for (int j = 0; j < shape.getColumn(); j++) {
                int index = i * shape.getColumn() + j;
                result.buffer[index] /= sum;
            }
        }
        return result;
    }

    /**
     * 元素级最大值运算，将数组中小于指定值的元素替换为该值
     *
     * @param number 阈值
     * @return 最大值运算结果数组
     */
    public NdArrayCpu maximum(Number number) {
        float threshold = number.floatValue();
        return unaryOperation(x -> Math.max(x, threshold));
    }

    /**
     * 掩码运算，将数组中大于指定值的元素设为1，小于等于指定值的元素设为0
     *
     * @param number 阈值
     * @return 掩码运算结果数组
     */
    public NdArrayCpu mask(Number number) {
        float threshold = number.floatValue();
        return unaryOperation(x -> x > threshold ? 1.0f : 0.0f);
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
        validateMatrixShape(this.shape);
        NdArrayCpu result = new NdArrayCpu(ShapeCpu.of(shape.getColumn(), shape.getRow()));

        for (int i = 0; i < shape.getRow(); i++) {
            for (int j = 0; j < shape.getColumn(); j++) {
                result.buffer[j * shape.getRow() + i] = this.buffer[i * shape.getColumn() + j];
            }
        }
        return result;
    }

    /**
     * 多维数组转置操作，按指定维度顺序重新排列
     *
     * @param order 新的维度顺序
     * @return 转置后的数组
     * @throws IllegalArgumentException 当维度顺序无效时抛出
     */
    public NdArrayCpu transpose(int... order) {
        validateTransposeOrder(order);

        int[] newDimensions = new int[shape.dimension.length];
        for (int i = 0; i < order.length; i++) {
            newDimensions[i] = shape.dimension[order[i]];
        }
        NdArrayCpu result = new NdArrayCpu(ShapeCpu.of(newDimensions));

        int[] indices = new int[shape.dimension.length];
        int totalElements = shape.size();

        for (int i = 0; i < totalElements; i++) {
            // 将一维索引转换为多维索引
            convertToMultiIndex(i, indices);

            // 计算转置后的索引
            int[] transposedIndices = new int[order.length];
            for (int j = 0; j < order.length; j++) {
                transposedIndices[j] = indices[order[j]];
            }

            // 复制数据
            result.set(this.buffer[i], transposedIndices);
        }
        return result;
    }

    /**
     * 验证转置维度顺序的有效性
     *
     * @param order 维度顺序数组
     * @throws IllegalArgumentException 当维度顺序无效时抛出
     */
    private void validateTransposeOrder(int[] order) {
        if (order.length != shape.dimension.length) {
            throw new IllegalArgumentException("转置维度数量不匹配");
        }

        boolean[] used = new boolean[shape.dimension.length];
        for (int dim : order) {
            if (dim < 0 || dim >= shape.dimension.length || used[dim]) {
                throw new IllegalArgumentException("转置维度顺序无效");
            }
            used[dim] = true;
        }
    }

    /**
     * 将一维线性索引转换为多维索引
     *
     * @param linearIndex 一维线性索引
     * @param indices     多维索引数组（输出参数）
     */
    private void convertToMultiIndex(int linearIndex, int[] indices) {
        int remaining = linearIndex;
        for (int i = shape.dimension.length - 1; i >= 0; i--) {
            if (shape.multipliers[i] == 0) {
                indices[i] = 0;
            } else {
                indices[i] = remaining / shape.multipliers[i];
                remaining %= shape.multipliers[i];
            }
        }
    }

    /**
     * 数组变形操作，改变数组形状但保持元素总数不变
     *
     * @param newShape 新的数组形状
     * @return 变形后的数组
     * @throws IllegalArgumentException 当新形状大小与原形状不匹配时抛出
     */
    public NdArrayCpu reshape(Shape newShape) {
        if (this.shape.size() != newShape.size()) {
            throw new IllegalArgumentException(String.format("形状大小不匹配：%d vs %d", this.shape.size(), newShape.size()));
        }

        // 使用共享数据的视图，避免数据复制
        NdArrayCpu result = new NdArrayCpu(newShape);
        System.arraycopy(this.buffer, 0, result.buffer, 0, this.buffer.length);
        return result;
    }

    /**
     * 数组展平操作，将多维数组转换为一维行向量
     *
     * @return 展平后的一维行向量
     */
    public NdArrayCpu flatten() {
        return this.reshape(Shape.of(1, shape.size()));
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
        float sum = 0f;
        for (float value : this.buffer) {
            sum += value;
        }
        return new NdArrayCpu(sum);
    }

    /**
     * 按轴聚合的通用方法，沿指定轴进行聚合运算
     *
     * @param axis          聚合轴，0表示按列聚合，1表示按行聚合
     * @param operation     聚合操作函数
     * @param operationName 操作名称，用于错误提示
     * @return 聚合结果数组
     * @throws IllegalArgumentException 当数组不是矩阵或轴参数无效时抛出
     */
    private NdArrayCpu axisOperation(int axis, java.util.function.Function<float[], Float> operation, String operationName) {
        validateMatrixShape(this.shape);
        validateAxis(axis);

        if (axis == 0) {
            // 按列操作
            NdArrayCpu result = new NdArrayCpu(ShapeCpu.of(1, shape.getColumn()));
            for (int j = 0; j < shape.getColumn(); j++) {
                float[] columnData = new float[shape.getRow()];
                for (int i = 0; i < shape.getRow(); i++) {
                    columnData[i] = this.buffer[i * shape.getColumn() + j];
                }
                result.buffer[j] = operation.apply(columnData);
            }
            return result;
        } else {
            // 按行操作
            NdArrayCpu result = new NdArrayCpu(Shape.of(shape.getRow(), 1));
            for (int i = 0; i < shape.getRow(); i++) {
                float[] rowData = new float[shape.getColumn()];
                for (int j = 0; j < shape.getColumn(); j++) {
                    rowData[j] = this.buffer[i * shape.getColumn() + j];
                }
                result.buffer[i] = operation.apply(rowData);
            }
            return result;
        }
    }

    /**
     * 验证轴参数的有效性
     *
     * @param axis 轴参数
     * @throws IllegalArgumentException 当轴参数无效时抛出
     */
    private void validateAxis(int axis) {
        if (axis != 0 && axis != 1) {
            throw new IllegalArgumentException("轴参数只支持 0(列) 或 1(行)");
        }
    }

    /**
     * 矩阵均值运算，沿指定轴计算均值
     *
     * @param axis 聚合轴，axis=0表示按列计算均值，axis=1表示按行计算均值
     * @return 均值运算结果数组
     */
    public NdArrayCpu mean(int axis) {
        return axisOperation(axis, values -> {
            float sum = 0f;
            for (float value : values) {
                sum += value;
            }
            return sum / values.length;
        }, "均值计算");
    }

    /**
     * 矩阵方差运算，沿指定轴计算方差
     *
     * @param axis 聚合轴，axis=0表示按列计算方差，axis=1表示按行计算方差
     * @return 方差运算结果数组
     */
    public NdArrayCpu var(int axis) {
        return axisOperation(axis, values -> {
            // 计算均值
            float mean = 0f;
            for (float value : values) {
                mean += value;
            }
            mean /= values.length;

            // 计算方差
            float variance = 0f;
            for (float value : values) {
                variance += (value - mean) * (value - mean);
            }
            return variance / values.length;
        }, "方差计算");
    }

    /**
     * 矩阵累和运算，沿指定轴计算累和
     *
     * @param axis 聚合轴，axis=0表示按列累和，axis=1表示按行累和
     * @return 累和运算结果数组
     */
    public NdArrayCpu sum(int axis) {
        return axisOperation(axis, values -> {
            float sum = 0f;
            for (float value : values) {
                sum += value;
            }
            return sum;
        }, "累和计算");
    }

    /**
     * 按指定形状进行压缩累加运算
     *
     * <p>将当前数组按指定形状进行压缩，超出目标形状的部分会累加到对应位置</p>
     *
     * @param _shape 目标形状
     * @return 压缩累加结果数组
     * @throws IllegalArgumentException 当数组不是矩阵或形状不合法时抛出
     */
    public NdArrayCpu sumTo(Shape _shape) {
        if (!shape.isMatrix()) {
            throw new IllegalArgumentException("操作仅适用于矩阵（二维数组）");
        }

        if (_shape.getRow() > this.shape.getRow() || _shape.getColumn() > this.shape.getColumn()) {
            throw new IllegalArgumentException(String.format("目标形状 %s 不能大于当前形状 %s", _shape, this.shape));
        }
        NdArrayCpu ndArray = new NdArrayCpu(ShapeCpu.of(_shape.getRow(), _shape.getColumn()));
        for (int i = 0; i < shape.getRow(); i++) {
            for (int j = 0; j < shape.getColumn(); j++) {
                ndArray.buffer[(i % _shape.getRow()) * _shape.getColumn() + j % _shape.getColumn()] += this.buffer[j + i * this.shape.getColumn()];
            }
        }
        return ndArray;
    }

    /**
     * 数组广播运算，将当前数组广播到指定形状
     *
     * <p>广播机制允许小数组与大数组进行运算，小数组会重复填充以匹配大数组的形状</p>
     *
     * @param _shape 目标广播形状
     * @return 广播结果数组
     * @throws IllegalArgumentException 当数组不是矩阵或形状不合法时抛出
     */
    public NdArrayCpu broadcastTo(Shape _shape) {
        if (!shape.isMatrix()) {
            throw new IllegalArgumentException("操作仅适用于矩阵（二维数组）");
        }

        if (_shape.getRow() < this.shape.getRow() || _shape.getColumn() < this.shape.getColumn()) {
            throw new IllegalArgumentException(String.format("目标形状 %s 不能小于当前形状 %s", _shape, this.shape));
        }
        NdArrayCpu ndArray = new NdArrayCpu(ShapeCpu.of(_shape.getRow(), _shape.getColumn()));

        for (int i = 0; i < _shape.getRow(); i++) {
            for (int j = 0; j < _shape.getColumn(); j++) {
                ndArray.buffer[i * _shape.getColumn() + j] += this.buffer[i % this.shape.getRow() * this.shape.getColumn() + j % this.shape.getColumn()];
            }
        }
        return ndArray;
    }

    /**
     * 沿指定轴查找最大值的索引
     *
     * @param axis 查找轴，axis=0表示按行查找每列的最大值索引，axis=1表示按列查找每行的最大值索引
     * @return 最大值索引数组
     * @throws IllegalArgumentException 当数组不是矩阵或轴参数无效时抛出
     */
    public NdArrayCpu argMax(int axis) {
        if (!shape.isMatrix()) {
            throw new IllegalArgumentException("操作仅适用于矩阵（二维数组）");
        }

        if (axis == 0) {
            NdArrayCpu ndArray = new NdArrayCpu(ShapeCpu.of(1, this.getShape().getColumn()));
            float[][] matrix = getMatrix();
            for (int i = 0; i < shape.getColumn(); i++) {
                float maxValue = Float.MIN_VALUE;
                int maxIndex = -1;
                for (int j = 0; j < shape.getRow(); j++) {
                    if (maxValue < matrix[j][i]) {
                        maxValue = matrix[j][i];
                        maxIndex = j;
                    }
                }
                ndArray.buffer[i] = maxIndex;
            }
            return ndArray;

        } else if (axis == 1) {
            NdArrayCpu ndArray = new NdArrayCpu(ShapeCpu.of(this.getShape().getRow(), 1));
            float[][] matrix = getMatrix();
            for (int i = 0; i < shape.getRow(); i++) {
                float maxValue = Float.MIN_VALUE;
                int maxIndex = -1;
                for (int j = 0; j < shape.getColumn(); j++) {
                    if (maxValue < matrix[i][j]) {
                        maxValue = matrix[i][j];
                        maxIndex = j;
                    }
                }
                ndArray.buffer[i] = maxIndex;
            }
            return ndArray;
        }
        throw new IllegalArgumentException(String.format("不支持的轴参数: %d，仅支持 0(列) 或 1(行)", axis));
    }

    /**
     * 矩阵内积运算（矩阵乘法）
     *
     * <p>执行标准的矩阵乘法运算，要求第一个矩阵的列数等于第二个矩阵的行数</p>
     *
     * @param _other 另一个矩阵
     * @return 矩阵乘法结果
     * @throws IllegalArgumentException 当数组不是矩阵或维度不匹配时抛出
     */
    public NdArrayCpu dot(NdArray _other) {

        NdArrayCpu other = (NdArrayCpu) _other;
        if (!shape.isMatrix()) {
            throw new IllegalArgumentException("操作仅适用于矩阵（二维数组）");
        }

        if (shape.getColumn() != other.shape.getRow()) {
            throw new IllegalArgumentException(String.format("矩阵乘法维度不匹配：%s × %s，第一个矩阵的列数(%d)必须等于第二个矩阵的行数(%d)", this.shape, other.shape, shape.getColumn(), other.shape.getRow()));
        }

        NdArrayCpu ndArray = new NdArrayCpu(ShapeCpu.of(shape.getRow(), other.shape.getColumn()));

        for (int i = 0; i < shape.getRow(); i++) {
            for (int j = 0; j < other.shape.getColumn(); j++) {

                float sum = 0f;
                for (int k = 0; k < shape.getColumn(); k++) {
//                    sum += this.get(i, k) * other.get(k, j);
                    sum += buffer[i * shape.getColumn() + k] * other.buffer[k * other.shape.getColumn() + j];
                }

//                ndArray.set(sum, i, j);
                ndArray.buffer[i * other.shape.getColumn() + j] = sum;
            }
        }
        return ndArray;
    }

    /**
     * 获取数组的子集（切片操作）
     *
     * @param _rowSlices 行索引数组，null表示选择所有行
     * @param _colSlices 列索引数组，null表示选择所有列
     * @return 切片结果数组
     * @throws IllegalArgumentException 当数组不是矩阵或参数不合法时抛出
     */
    public NdArrayCpu getItem(int[] _rowSlices, int[] _colSlices) {
        if (!shape.isMatrix()) {
            throw new IllegalArgumentException("操作仅适用于矩阵（二维数组）");
        }

        if (_rowSlices != null && _colSlices != null) {
            if (_rowSlices.length != _colSlices.length) {
                throw new IllegalArgumentException(String.format("行索引数组长度(%d)必须等于列索引数组长度(%d)", _rowSlices.length, _colSlices.length));
            }

            NdArrayCpu ndArray = new NdArrayCpu(ShapeCpu.of(1, _colSlices.length));
            for (int i = 0; i < _colSlices.length; i++) {
                ndArray.buffer[i] = buffer[_rowSlices[i] * shape.getColumn() + _colSlices[i]];
            }
            return ndArray;
        }

        if (_colSlices == null) {
            _colSlices = NdArrayUtil.getSeq(shape.getColumn());
        }
        if (_rowSlices == null) {
            _rowSlices = NdArrayUtil.getSeq(shape.getRow());
        }

        NdArrayCpu ndArray = new NdArrayCpu(ShapeCpu.of(_rowSlices.length, _colSlices.length));
        for (int i = 0; i < _rowSlices.length; i++) {
            for (int j = 0; j < _colSlices.length; j++) {
                ndArray.buffer[i * ndArray.getShape().getColumn() + j] = buffer[_rowSlices[i] * shape.getColumn() + _colSlices[j]];
            }
        }
        return ndArray;
    }

    /**
     * 设置数组的子集（切片赋值操作）
     *
     * @param _rowSlices 行索引数组，null表示选择所有行
     * @param _colSlices 列索引数组，null表示选择所有列
     * @param data       要设置的数据
     * @return 当前数组实例
     * @throws IllegalArgumentException 当数组不是矩阵或参数不合法时抛出
     */
    public NdArrayCpu setItem(int[] _rowSlices, int[] _colSlices, float[] data) {
        if (!shape.isMatrix()) {
            throw new IllegalArgumentException("操作仅适用于矩阵（二维数组）");
        }

        if (_rowSlices != null && _colSlices != null) {
            if (_rowSlices.length != _colSlices.length) {
                throw new IllegalArgumentException(String.format("行索引数组长度(%d)必须等于列索引数组长度(%d)", _rowSlices.length, _colSlices.length));
            }

            for (int i = 0; i < _colSlices.length; i++) {
                buffer[_rowSlices[i] * shape.getColumn() + _colSlices[i]] = data[i];
            }
            return this;
        }

        throw new IllegalArgumentException("功能尚未实现：不支持的参数组合");
    }

    /**
     * 沿指定轴查找最大值
     *
     * @param axis 查找轴，axis=0表示按行查找每列的最大值，axis=1表示按列查找每行的最大值
     * @return 最大值数组
     * @throws IllegalArgumentException 当数组不是矩阵或轴参数无效时抛出
     */
    public NdArrayCpu max(int axis) {
        if (!shape.isMatrix()) {
            throw new IllegalArgumentException("操作仅适用于矩阵（二维数组）");
        }

        if (axis == 1) {
            NdArrayCpu ndArray = new NdArrayCpu(ShapeCpu.of(shape.getRow(), 1));
            for (int i = 0; i < shape.getRow(); i++) {
                float max = Float.MIN_VALUE;
                for (int j = 0; j < shape.getColumn(); j++) {
                    if (max < buffer[i * shape.getColumn() + j]) {
                        max = buffer[i * shape.getColumn() + j];
                    }
                }
                ndArray.buffer[i] = max;
            }
            return ndArray;
        } else if (axis == 0) {
            // 按列找最大值（按行方向查找每列的最大值）
            NdArrayCpu ndArray = new NdArrayCpu(ShapeCpu.of(1, shape.getColumn()));
            for (int j = 0; j < shape.getColumn(); j++) {
                float max = Float.MIN_VALUE;
                for (int i = 0; i < shape.getRow(); i++) {
                    if (max < buffer[i * shape.getColumn() + j]) {
                        max = buffer[i * shape.getColumn() + j];
                    }
                }
                ndArray.buffer[j] = max;
            }
            return ndArray;
        }
        throw new IllegalArgumentException(String.format("不支持的轴参数: %d，仅支持 0(列) 或 1(行)", axis));
    }

    /**
     * 沿指定轴查找最小值
     *
     * @param axis 查找轴，axis=0表示按行查找每列的最小值，axis=1表示按列查找每行的最小值
     * @return 最小值数组
     * @throws IllegalArgumentException 当数组不是矩阵或轴参数无效时抛出
     */
    public NdArrayCpu min(int axis) {
        if (!shape.isMatrix()) {
            throw new IllegalArgumentException("操作仅适用于矩阵（二维数组）");
        }

        if (axis == 1) {
            NdArrayCpu ndArray = new NdArrayCpu(ShapeCpu.of(shape.getRow(), 1));
            for (int i = 0; i < shape.getRow(); i++) {
                float min = Float.MAX_VALUE;
                for (int j = 0; j < shape.getColumn(); j++) {
                    if (min > buffer[i * shape.getColumn() + j]) {
                        min = buffer[i * shape.getColumn() + j];
                    }
                }
                ndArray.buffer[i] = min;
            }
            return ndArray;
        } else if (axis == 0) {
            // 按列找最小值（按行方向查找每列的最小值）
            NdArrayCpu ndArray = new NdArrayCpu(ShapeCpu.of(1, shape.getColumn()));
            for (int j = 0; j < shape.getColumn(); j++) {
                float min = Float.MAX_VALUE;
                for (int i = 0; i < shape.getRow(); i++) {
                    if (min > buffer[i * shape.getColumn() + j]) {
                        min = buffer[i * shape.getColumn() + j];
                    }
                }
                ndArray.buffer[j] = min;
            }
            return ndArray;
        }
        throw new IllegalArgumentException(String.format("不支持的轴参数: %d，仅支持 0(列) 或 1(行)", axis));
    }

    /**
     * 查找数组中的最大值（全局最大值）
     *
     * @return 数组中的最大值
     */
    public float max() {
        float max = Float.MIN_VALUE;
        for (float value : this.buffer) {
            if (max < value) {
                max = value;
            }
        }
        return max;
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
        if (!shape.isMatrix()) {
            throw new IllegalArgumentException("操作仅适用于矩阵（二维数组）");
        }

        NdArrayCpu ndArray = new NdArrayCpu(ShapeCpu.of(endRow - startRow, endCol - startCol));
        for (int i = startRow; i < endRow; i++) {
            for (int j = startCol; j < endCol; j++) {
                ndArray.buffer[ndArray.shape.getColumn() * (i - startRow) + j - startCol] = this.buffer[i * this.shape.getColumn() + j];
            }
        }
        return ndArray;
    }


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
     * @param _other    要累加的数组
     * @return 累加结果数组
     * @throws IllegalArgumentException 当输入参数不合法时抛出
     * @throws RuntimeException         当数组不是矩阵时抛出
     */
    public NdArrayCpu addAt(int[] rowSlices, int[] colSlices, NdArray _other) {

        NdArrayCpu other = (NdArrayCpu) _other;
        // 验证当前数组是否为矩阵
        if (!shape.isMatrix()) {
            throw new RuntimeException("当前数组不是矩阵！");
        }

        // 处理null参数
        if (rowSlices == null) {
            rowSlices = NdArrayUtil.getSeq(shape.getRow());
        }
        if (colSlices == null) {
            colSlices = NdArrayUtil.getSeq(shape.getColumn());
        }

        // 创建结果数组的副本
        NdArrayCpu result = new NdArrayCpu(Arrays.copyOf(buffer, buffer.length), shape);

        // 验证输入参数
        validateAddAtParameters(rowSlices, colSlices, other);

        // 执行累加操作
        if (rowSlices.length == colSlices.length) {
            // 当行索引和列索引数量相等时，按对应位置累加
            for (int i = 0; i < rowSlices.length; i++) {
                int row = rowSlices[i];
                int col = colSlices[i];

                // 边界检查
                if (row < 0 || row >= shape.getRow() || col < 0 || col >= shape.getColumn()) {
                    throw new IllegalArgumentException(String.format("索引超出范围：位置(%d, %d)，数组形状%s", row, col, shape));
                }

                // 当other是一维数组时
                if (other.shape.isMatrix() && other.shape.getRow() == 1) {
                    result.buffer[row * shape.getColumn() + col] += other.buffer[i % other.buffer.length];
                } else if (other.shape.isMatrix() && other.shape.getColumn() == 1) {
                    // 当other是列向量时
                    result.buffer[row * shape.getColumn() + col] += other.buffer[i % other.buffer.length];
                } else if (other.shape.isMatrix()) {
                    // 当other是二维数组时
                    if (i < other.shape.getRow() * other.shape.getColumn()) {
                        result.buffer[row * shape.getColumn() + col] += other.buffer[i];
                    } else {
                        result.buffer[row * shape.getColumn() + col] += other.buffer[i % other.buffer.length];
                    }
                } else {
                    // 当other是标量或一维数组时
                    result.buffer[row * shape.getColumn() + col] += other.buffer[i % other.buffer.length];
                }
            }
        } else {
            // 当行索引和列索引数量不等时，对所有组合进行累加
            for (int i = 0; i < rowSlices.length; i++) {
                int row = rowSlices[i];

                // 边界检查
                if (row < 0 || row >= shape.getRow()) {
                    throw new IllegalArgumentException(String.format("行索引超出范围：%d，最大行索引%d", row, shape.getRow() - 1));
                }

                for (int j = 0; j < colSlices.length; j++) {
                    int col = colSlices[j];

                    // 边界检查
                    if (col < 0 || col >= shape.getColumn()) {
                        throw new IllegalArgumentException(String.format("列索引超出范围：%d，最大列索引%d", col, shape.getColumn() - 1));
                    }

                    // 计算other数组中的对应位置
                    int otherIndex;
                    if (other.shape.isMatrix()) {
                        // 当other是二维数组时
                        otherIndex = i * colSlices.length + j;
                        if (otherIndex >= other.buffer.length) {
                            otherIndex = otherIndex % other.buffer.length;
                        }
                    } else {
                        // 当other是一维数组时
                        otherIndex = i * colSlices.length + j;
                        if (otherIndex >= other.buffer.length) {
                            otherIndex = otherIndex % other.buffer.length;
                        }
                    }

                    result.buffer[row * shape.getColumn() + col] += other.buffer[otherIndex];
                }
            }
        }

        return result;
    }

    /**
     * 验证addAt方法的输入参数
     *
     * @param rowSlices 行索引数组
     * @param colSlices 列索引数组
     * @param other     要累加的数组
     * @throws IllegalArgumentException 当参数不合法时抛出
     */
    private void validateAddAtParameters(int[] rowSlices, int[] colSlices, NdArray other) {
        if (rowSlices.length == 0 || colSlices.length == 0) {
            throw new IllegalArgumentException("行索引数组和列索引数组不能为空");
        }

        if (other == null) {
            throw new IllegalArgumentException("要累加的数组不能为null");
        }

        if (((NdArrayCpu) other).buffer.length == 0) {
            throw new IllegalArgumentException("要累加的数组不能为空");
        }
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
        if (!shape.isMatrix()) {
            throw new IllegalArgumentException("操作仅适用于矩阵（二维数组）");
        }

        for (int _i = 0; _i < other.getShape().getRow(); _i++) {
            for (int _j = 0; _j < other.getShape().getColumn(); _j++) {
                buffer[this.getShape().getColumn() * (_i + i) + _j + j] += ((NdArrayCpu) other).buffer[other.getShape().getColumn() * _i + _j];
            }
        }
        return this;
    }

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
    public NdArrayCpu clip(float min, float max) {
        if (min > max) {
            throw new IllegalArgumentException("最小值不能大于最大值");
        }
        return unaryOperation(x -> Math.max(min, Math.min(max, x)));
    }

    /**
     * 用指定值填充整个数组
     *
     * @param number 填充值
     */
    private void fillAll(Number number) {
        float value = number.floatValue();
        Arrays.fill(this.buffer, value);
    }


    //    # =============================================================================
    //       其他的运算
    //    # =============================================================================


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
     * <p>注意：新形状的大小必须与当前形状大小一致</p>
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

    /**
     * 将数组转换为二维数组（矩阵）返回
     *
     * @return 二维数组表示
     * @throws IllegalArgumentException 当数组维度大于2时抛出
     */
    public float[][] getMatrix() {
        if (shape.isMatrix()) {
            float[][] matrix = new float[shape.dimension[0]][shape.dimension[1]];
            int k = 0;
            for (int i = 0; i < shape.dimension[0]; i++) {
                for (int j = 0; j < shape.dimension[1]; j++) {
                    matrix[i][j] = buffer[k];
                    k++;
                }
            }
            return matrix;
        } else if (shape.dimension.length == 1) {
            float[][] matrix = new float[1][shape.dimension[0]];
            matrix[0] = buffer;
            return matrix;
        } else {
            throw new IllegalArgumentException("不支持维度大于2");
        }
    }

    /**
     * 将数组转换为三维数组返回
     *
     * @return 三维数组表示
     * @throws IllegalArgumentException 当数组不是三维时抛出
     */
    public float[][][] get3dArray() {
        if (shape.dimension.length == 3) {
            float[][][] result = new float[shape.dimension[0]][shape.dimension[1]][shape.dimension[2]];
            int index = 0;
            for (int i = 0; i < shape.dimension[0]; i++) {
                for (int j = 0; j < shape.dimension[1]; j++) {
                    for (int k = 0; k < shape.dimension[2]; k++) {
                        result[i][j][k] = buffer[index];
                        index++;
                    }
                }
            }
            return result;
        } else {
            throw new IllegalArgumentException("not support!");
        }
    }

    /**
     * 将数组转换为四维数组返回
     *
     * @return 四维数组表示
     * @throws IllegalArgumentException 当数组不是四维时抛出
     */
    public float[][][][] get4dArray() {
        if (shape.dimension.length == 4) {
            float[][][][] result = new float[shape.dimension[0]][shape.dimension[1]][shape.dimension[2]][shape.dimension[3]];
            int index = 0;
            for (int i = 0; i < shape.dimension[0]; i++) {
                for (int j = 0; j < shape.dimension[1]; j++) {
                    for (int k = 0; k < shape.dimension[2]; k++) {
                        for (int l = 0; l < shape.dimension[3]; l++) {
                            result[i][j][k][l] = buffer[index];
                            index++;
                        }
                    }
                }
            }
            return result;
        } else {
            throw new IllegalArgumentException("not support!");
        }
    }

    /**
     * 优化的toString方法，提供数组的字符串表示
     *
     * <p>对于小数组会显示所有元素，对于大数组只会显示部分元素</p>
     *
     * @return 数组的字符串表示
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("NdArray{");
        sb.append("shape=").append(shape);
        sb.append(", data=");

        if (shape.size() <= 10) {
            // 小数组直接显示所有元素
            toStringHelper(sb, 0, new int[shape.dimension.length]);
        } else {
            // 大数组只显示前几个元素
            sb.append("[");
            for (int i = 0; i < Math.min(5, buffer.length); i++) {
                sb.append(String.format("%.4f", buffer[i]));
                if (i < Math.min(4, buffer.length - 1)) {
                    sb.append(", ");
                }
            }
            if (buffer.length > 5) {
                sb.append(", ..., ").append(String.format("%.4f", buffer[buffer.length - 1]));
            }
            sb.append("]");
        }

        sb.append("}");
        return sb.toString();
    }

    /**
     * 递归构建多维数组的字符串表示
     *
     * @param sb       字符串构建器
     * @param dimIndex 当前维度索引
     * @param indices  多维索引数组
     */
    private void toStringHelper(StringBuilder sb, int dimIndex, int[] indices) {
        if (dimIndex == shape.dimension.length) {
            sb.append(String.format("%.4f", get(indices)));
            return;
        }

        sb.append("[");
        for (int i = 0; i < shape.dimension[dimIndex]; i++) {
            indices[dimIndex] = i;
            toStringHelper(sb, dimIndex + 1, indices);
            if (i < shape.dimension[dimIndex] - 1) {
                sb.append(", ");
                if (dimIndex == shape.dimension.length - 2) {
                    sb.append("\n ");
                }
            }
        }
        sb.append("]");

        if (dimIndex == 0) {
            sb.append("\n");
        }
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
        if (obj == null || getClass() != obj.getClass()) return false;

        NdArrayCpu other = (NdArrayCpu) obj;
        if (!this.shape.equals(other.shape)) return false;

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

}
