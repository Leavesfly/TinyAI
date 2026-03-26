package io.leavesfly.tinyai.ndarr.cpu.operations;

import io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu;
import io.leavesfly.tinyai.ndarr.cpu.utils.ArrayValidator;

/**
 * 算术运算操作类，提供四则运算（加、减、乘、除）。
 */
public class ArithmeticOperations {

    /**
     * 除零检测阈值：使用 Float.MIN_NORMAL（约 1.17e-38）作为安全下界，
     * 避免固定阈值对合法小数值（如学习率 1e-8）的误判
     */
    private static final float DIVISION_ZERO_THRESHOLD = Float.MIN_NORMAL;

    @FunctionalInterface
    private interface FloatBinaryOp {
        float apply(float a, float b);
    }

    /**
     * 通用的二元运算方法，对两个相同形状的数组进行元素级运算
     *
     * @param left         左操作数数组
     * @param right        右操作数数组
     * @param operation    二元运算操作函数
     * @param operationName 操作名称，用于错误提示
     * @return 运算结果数组
     * @throws IllegalArgumentException 当两个数组形状不一致时抛出
     */
    private static NdArrayCpu binaryOperation(NdArrayCpu left, NdArrayCpu right, FloatBinaryOp operation, String operationName) {
        ArrayValidator.validateShapeCompatibility(left.shape, right.shape, operationName);
        NdArrayCpu result = new NdArrayCpu(left.shape);
        for (int i = 0; i < left.buffer.length; i++) {
            result.buffer[i] = operation.apply(left.buffer[i], right.buffer[i]);
        }
        return result;
    }

    /**
     * 通用的与标量运算方法，对数组与标量进行运算
     *
     * @param array     数组
     * @param scalar    标量值
     * @param operation 二元运算操作函数
     * @return 运算结果数组
     */
    private static NdArrayCpu scalarOperation(NdArrayCpu array, Number scalar, FloatBinaryOp operation) {
        NdArrayCpu result = new NdArrayCpu(array.shape);
        float scalarValue = scalar.floatValue();
        for (int i = 0; i < array.buffer.length; i++) {
            result.buffer[i] = operation.apply(array.buffer[i], scalarValue);
        }
        return result;
    }

    /**
     * 数组加法运算，对应元素相加
     *
     * @param left  左操作数数组
     * @param right 右操作数数组
     * @return 加法运算结果
     * @throws IllegalArgumentException 当两个数组形状不一致时抛出
     */
    public static NdArrayCpu add(NdArrayCpu left, NdArrayCpu right) {
        return binaryOperation(left, right, Float::sum, "加法");
    }

    /**
     * 数组减法运算，对应元素相减
     *
     * @param left  左操作数数组
     * @param right 右操作数数组
     * @return 减法运算结果
     * @throws IllegalArgumentException 当两个数组形状不一致时抛出
     */
    public static NdArrayCpu sub(NdArrayCpu left, NdArrayCpu right) {
        return binaryOperation(left, right, (a, b) -> a - b, "减法");
    }

    /**
     * 数组乘法运算，对应元素相乘
     *
     * @param left  左操作数数组
     * @param right 右操作数数组
     * @return 乘法运算结果
     * @throws IllegalArgumentException 当两个数组形状不一致时抛出
     */
    public static NdArrayCpu mul(NdArrayCpu left, NdArrayCpu right) {
        return binaryOperation(left, right, (a, b) -> a * b, "乘法");
    }

    /**
     * 数组与标量相乘
     *
     * @param array  数组
     * @param number 标量值
     * @return 乘法运算结果
     */
    public static NdArrayCpu mulNum(NdArrayCpu array, Number number) {
        return scalarOperation(array, number, (a, b) -> a * b);
    }

    /**
     * 数组除法运算，对应元素相除
     *
     * @param left  左操作数数组
     * @param right 右操作数数组
     * @return 除法运算结果
     * @throws IllegalArgumentException 当两个数组形状不一致时抛出
     * @throws ArithmeticException      当除数接近0时抛出
     */
    public static NdArrayCpu div(NdArrayCpu left, NdArrayCpu right) {
        return binaryOperation(left, right, (a, b) -> {
            if (Math.abs(b) < DIVISION_ZERO_THRESHOLD) {
                throw new ArithmeticException("除数接近0，值为: " + b);
            }
            return a / b;
        }, "除法");
    }

    /**
     * 数组与标量相除
     *
     * @param array  数组
     * @param number 标量值
     * @return 除法运算结果
     * @throws ArithmeticException 当除数为0时抛出
     */
    public static NdArrayCpu divNum(NdArrayCpu array, Number number) {
        float value = number.floatValue();
        if (Math.abs(value) < DIVISION_ZERO_THRESHOLD) {
            throw new ArithmeticException("除数不能为0，值为: " + value);
        }
        return scalarOperation(array, number, (a, b) -> a / b);
    }
}

