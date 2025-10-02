package io.leavesfly.tinyai.func.matrix;


import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.Arrays;
import java.util.List;

/**
 * 矩阵乘法函数
 * <p>
 * 计算两个矩阵的内积（点积）。
 */
public class MatMul extends Function {
    /**
     * 前向传播计算矩阵乘法
     * <p>
     * 计算两个矩阵的内积（点积）：x * w
     *
     * @param inputs 输入的NdArray数组，长度为2
     * @return 矩阵乘法结果的NdArray
     */
    @Override
    public NdArray forward(NdArray... inputs) {
        NdArray x = inputs[0];
        NdArray w = inputs[1];

        return x.dot(w);
    }

    /**
     * 反向传播计算梯度
     * <p>
     * 对于矩阵乘法，梯度计算公式为：
     * - ∂(x*w)/∂x = yGrad * w^T
     * - ∂(x*w)/∂w = x^T * yGrad
     *
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        NdArray x = inputs[0].getValue();
        NdArray w = inputs[1].getValue();

        return Arrays.asList(yGrad.dot(w.transpose()), x.transpose().dot(yGrad));
    }

    /**
     * 获取所需输入参数个数
     * <p>
     * 矩阵乘法函数需要两个输入参数。
     *
     * @return 输入参数个数，固定为2
     */
    @Override
    public int requireInputNum() {
        return 2;
    }
}
