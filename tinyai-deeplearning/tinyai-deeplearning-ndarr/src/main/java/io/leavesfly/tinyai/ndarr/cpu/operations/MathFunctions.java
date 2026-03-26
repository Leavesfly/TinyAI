package io.leavesfly.tinyai.ndarr.cpu.operations;

import io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu;
import io.leavesfly.tinyai.ndarr.cpu.ShapeCpu;
import io.leavesfly.tinyai.ndarr.cpu.aggregations.ReductionOperations;
import io.leavesfly.tinyai.ndarr.cpu.operations.ArithmeticOperations;
import io.leavesfly.tinyai.ndarr.cpu.transformations.TransformationOperations;

/**
 * 数学函数操作类，提供各类数学函数运算。
 */
public class MathFunctions {

    // 常用的数学常数，用于数值计算中的比较
    // 使用更小的阈值，避免优化器在添加极小epsilon时被误判为接近0而抛异常
    private static final float EPSILON = 1e-12f;

    @FunctionalInterface
    private interface FloatUnaryOp {
        float apply(float a);
    }

    /**
     * 通用的一元运算方法，对数组每个元素进行一元运算
     *
     * @param array     数组
     * @param operation 一元运算操作函数
     * @return 运算结果数组
     */
    private static NdArrayCpu unaryOperation(NdArrayCpu array, FloatUnaryOp operation) {
        NdArrayCpu result = new NdArrayCpu(array.shape);
        for (int i = 0; i < array.buffer.length; i++) {
            result.buffer[i] = operation.apply(array.buffer[i]);
        }
        return result;
    }

    /**
     * 幂运算，对数组每个元素进行幂运算
     *
     * @param array    数组
     * @param number   幂指数
     * @return 幂运算结果数组
     */
    public static NdArrayCpu pow(NdArrayCpu array, Number number) {
        float exponent = number.floatValue();
        return unaryOperation(array, x -> (float) Math.pow(x, exponent));
    }

    /**
     * 平方运算，对数组每个元素进行平方运算
     *
     * @param array 数组
     * @return 平方运算结果数组
     */
    public static NdArrayCpu square(NdArrayCpu array) {
        return pow(array, 2f);
    }

    /**
     * 平方根运算，对数组每个元素进行开方运算
     *
     * @param array 数组
     * @return 平方根运算结果数组
     */
    public static NdArrayCpu sqrt(NdArrayCpu array) {
        return unaryOperation(array, x -> (float) Math.sqrt(x));
    }

    /**
     * 自然指数运算，对数组每个元素进行e为底的指数运算
     *
     * @param array 数组
     * @return 指数运算结果数组
     */
    public static NdArrayCpu exp(NdArrayCpu array) {
        return unaryOperation(array, x -> (float) Math.exp(x));
    }

    /**
     * 正弦函数运算，对数组每个元素进行sin运算
     *
     * @param array 数组
     * @return 正弦运算结果数组
     */
    public static NdArrayCpu sin(NdArrayCpu array) {
        return unaryOperation(array, x -> (float) Math.sin(x));
    }

    /**
     * 余弦函数运算，对数组每个元素进行cos运算
     *
     * @param array 数组
     * @return 余弦运算结果数组
     */
    public static NdArrayCpu cos(NdArrayCpu array) {
        return unaryOperation(array, x -> (float) Math.cos(x));
    }

    /**
     * 双曲正切函数运算，对数组每个元素进行tanh运算
     *
     * @param array 数组
     * @return 双曲正切运算结果数组
     */
    public static NdArrayCpu tanh(NdArrayCpu array) {
        return unaryOperation(array, x -> (float) Math.tanh(x));
    }

    /**
     * Sigmoid 运算，f(x) = 1 / (1 + e^(-x))。
     *
     * @param array 数组
     * @return Sigmoid 运算结果数组
     */
    public static NdArrayCpu sigmoid(NdArrayCpu array) {
        return unaryOperation(array, x -> (float) (1.0 / (1.0 + Math.exp(-x))));
    }

    /**
     * 自然对数运算，对数组每个元素进行ln运算
     *
     * @param array 数组
     * @return 对数运算结果数组
     * @throws ArithmeticException 当输入值小于等于0时抛出
     */
    public static NdArrayCpu log(NdArrayCpu array) {
        return unaryOperation(array, x -> {
            if (Float.isNaN(x)) {
                throw new ArithmeticException("对数的输入不能为NaN");
            }
            if (x <= 0f) {
                throw new ArithmeticException("对数的输入必须大于0，当前值: " + x);
            }
            return (float) Math.log(x);
        });
    }

    /**
     * 元素级最大值运算，将数组中小于指定值的元素替换为该值
     *
     * @param array   数组
     * @param number  阈值
     * @return 最大值运算结果数组
     */
    public static NdArrayCpu maximum(NdArrayCpu array, Number number) {
        float threshold = number.floatValue();
        return unaryOperation(array, x -> Math.max(x, threshold));
    }

    /**
     * 掩码运算，将数组中大于指定值的元素设为1，小于等于指定值的元素设为0
     *
     * @param array   数组
     * @param number  阈值
     * @return 掩码运算结果数组
     */
    public static NdArrayCpu mask(NdArrayCpu array, Number number) {
        float threshold = number.floatValue();
        return unaryOperation(array, x -> x > threshold ? 1.0f : 0.0f);
    }

    /**
     * 裁剪数组元素到 [min, max] 范围。
     *
     * @param array 数组
     * @param min   最小值
     * @param max   最大值
     * @return 裁剪后的数组
     * @throws IllegalArgumentException 当 min > max 时抛出
     */
    public static NdArrayCpu clip(NdArrayCpu array, float min, float max) {
        if (min > max) {
            throw new IllegalArgumentException("最小值不能大于最大值");
        }
        return unaryOperation(array, x -> Math.max(min, Math.min(max, x)));
    }

    /**
     * Softmax 运算，softmax(x_i) = exp(x_i) / Σexp(x_j)，采用数值稳定实现。
     *
     * @param array 数组
     * @return Softmax 运算结果数组
     */
    public static NdArrayCpu softMax(NdArrayCpu array) {
        int dimNum = array.shape.getDimNum();
        int defaultAxis = (dimNum == 1) ? 0 : (dimNum - 1);
        return softMax(array, defaultAxis);
    }

    /**
     * 沿指定 axis 计算 Softmax，支持负轴（-1 表示最后一维）。
     * 数值稳定实现：先减该轴最大值，再 exp 并归一化。
     *
     * @param array 数组
     * @param axis  计算维度
     * @return Softmax 运算结果数组
     * @throws IllegalArgumentException 当 axis 越界时抛出
     */
    public static NdArrayCpu softMax(NdArrayCpu array, int axis) {
        int dimNum = array.shape.getDimNum();
        int normalizedAxis = axis < 0 ? axis + dimNum : axis;
        if (normalizedAxis < 0 || normalizedAxis >= dimNum) {
            throw new IllegalArgumentException(
                    String.format("轴参数越界: %d，应在 [0, %d) 之间", axis, dimNum)
            );
        }

        // 1) 按 axis 求最大值（数值稳定）
        NdArrayCpu maxReduced = ReductionOperations.max(array, normalizedAxis);

        // 2) keepdims（在 axis 位置插入 1），再广播回原形状
        int[] keepDims = new int[dimNum];
        for (int i = 0; i < dimNum; i++) {
            keepDims[i] = array.shape.getDimension(i);
        }
        keepDims[normalizedAxis] = 1;

        NdArrayCpu maxKeep = TransformationOperations.reshape(maxReduced, ShapeCpu.of(keepDims));
        NdArrayCpu maxBroadcast = TransformationOperations.broadcastTo(maxKeep, array.shape);

        // 3) 计算 exp(x - max)
        NdArrayCpu shifted = ArithmeticOperations.sub(array, maxBroadcast);
        NdArrayCpu expValues = exp(shifted);

        // 4) 沿 axis 求和，并 keepdims + 广播回原形状（加 EPSILON 防止除零）
        NdArrayCpu sumReduced = ReductionOperations.sum(expValues, normalizedAxis);
        NdArrayCpu sumKeep = TransformationOperations.reshape(sumReduced, ShapeCpu.of(keepDims));
        NdArrayCpu sumBroadcast = maximum(
                TransformationOperations.broadcastTo(sumKeep, array.shape),
                EPSILON
        );

        // 5) 归一化
        return ArithmeticOperations.div(expValues, sumBroadcast);
    }
}

