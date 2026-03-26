package io.leavesfly.tinyai.ndarr.cpu.operations;

import io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu;
import io.leavesfly.tinyai.ndarr.cpu.utils.ArrayValidator;

/**
 * 逻辑运算操作类，提供逻辑比较和一元运算。
 */
public class LogicalOperations {

    @FunctionalInterface
    private interface FloatUnaryOp {
        float apply(float a);
    }

    @FunctionalInterface
    private interface FloatBinaryPredicate {
        boolean test(float a, float b);
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
     * 通用的比较运算方法，对两个数组进行元素级比较
     *
     * @param left         左操作数数组
     * @param right        右操作数数组
     * @param comparison   比较操作函数
     * @param operationName 操作名称，用于错误提示
     * @return 比较结果数组，1.0表示true，0.0表示false
     * @throws IllegalArgumentException 当两个数组形状不一致时抛出
     */
    private static NdArrayCpu comparisonOperation(NdArrayCpu left, NdArrayCpu right, FloatBinaryPredicate comparison, String operationName) {
        ArrayValidator.validateShapeCompatibility(left.shape, right.shape, operationName);
        NdArrayCpu result = new NdArrayCpu(left.shape);
        for (int i = 0; i < left.buffer.length; i++) {
            boolean compResult = comparison.test(left.buffer[i], right.buffer[i]);
            result.buffer[i] = compResult ? 1.0f : 0.0f;
        }
        return result;
    }

    /**
     * 取反操作，对数组每个元素取负值
     *
     * @param array 数组
     * @return 取反后的数组
     */
    public static NdArrayCpu neg(NdArrayCpu array) {
        return unaryOperation(array, x -> -x);
    }

    /**
     * 绝对值运算，对数组每个元素取绝对值
     *
     * @param array 数组
     * @return 绝对值数组
     */
    public static NdArrayCpu abs(NdArrayCpu array) {
        return unaryOperation(array, Math::abs);
    }

    /**
     * 相等比较运算，比较两个数组对应元素是否相等
     *
     * @param left  左操作数数组
     * @param right 右操作数数组
     * @return 比较结果数组，1.0表示相等，0.0表示不相等
     * @throws IllegalArgumentException 当两个数组形状不一致时抛出
     */
    public static NdArrayCpu eq(NdArrayCpu left, NdArrayCpu right) {
        return comparisonOperation(left, right, (a, b) -> a == b, "相等比较");
    }

    /**
     * 大于比较运算，比较左数组元素是否大于右数组对应元素
     *
     * @param left  左操作数数组
     * @param right 右操作数数组
     * @return 比较结果数组，1.0表示大于，0.0表示不大于
     * @throws IllegalArgumentException 当两个数组形状不一致时抛出
     */
    public static NdArrayCpu gt(NdArrayCpu left, NdArrayCpu right) {
        return comparisonOperation(left, right, (a, b) -> a > b, "大于比较");
    }

    /**
     * 小于比较运算，比较左数组元素是否小于右数组对应元素
     *
     * @param left  左操作数数组
     * @param right 右操作数数组
     * @return 比较结果数组，1.0表示小于，0.0表示不小于
     * @throws IllegalArgumentException 当两个数组形状不一致时抛出
     */
    public static NdArrayCpu lt(NdArrayCpu left, NdArrayCpu right) {
        return comparisonOperation(left, right, (a, b) -> a < b, "小于比较");
    }

    /**
     * 全元素大于比较，判断左数组是否所有元素都严格大于右数组对应元素
     *
     * @param left  左操作数数组
     * @param right 右操作数数组
     * @return true 表示所有元素都大于，false 表示存在不大于的元素
     * @throws IllegalArgumentException 当两个数组形状不一致时抛出
     */
    public static boolean isAllGreaterThan(NdArrayCpu left, NdArrayCpu right) {
        ArrayValidator.validateShapeCompatibility(left.shape, right.shape, "全元素比较");
        for (int i = 0; i < left.buffer.length; i++) {
            if (left.buffer[i] <= right.buffer[i]) {
                return false;
            }
        }
        return true;
    }
}

