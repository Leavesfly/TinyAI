package io.leavesfly.tinyai.func.math;


import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.Collections;
import java.util.List;

/**
 * 正弦函数
 * <p>
 * 计算输入值的正弦值。
 */
public class Sin extends Function {
    /**
     * 前向传播计算正弦
     * <p>
     * 计算输入值的正弦值：sin(x)
     *
     * @param inputs 输入的NdArray数组，长度为1
     * @return 正弦值的NdArray
     */
    @Override
    public NdArray forward(NdArray... inputs) {
        return inputs[0].sin();
    }

    /**
     * 反向传播计算梯度
     * <p>
     * 对于正弦函数，梯度计算公式为：
     * ∂sin(x)/∂x = cos(x)
     *
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        return Collections.singletonList(inputs[0].getValue().cos().mul(yGrad));
    }

    /**
     * 获取所需输入参数个数
     * <p>
     * 正弦函数需要一个输入参数。
     *
     * @return 输入参数个数，固定为1
     */
    @Override
    public int requireInputNum() {
        return 1;
    }
}
