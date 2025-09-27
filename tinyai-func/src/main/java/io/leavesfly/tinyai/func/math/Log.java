package io.leavesfly.tinyai.func.math;


import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.Collections;
import java.util.List;

/**
 * 对数函数
 * <p>
 * 计算以e为底的对数值。
 */
public class Log extends Function {
    /**
     * 前向传播计算对数
     * <p>
     * 计算输入值的对数值：ln(x)
     *
     * @param inputs 输入的NdArray数组，长度为1
     * @return 对数值的NdArray
     */
    @Override
    public NdArray forward(NdArray... inputs) {
        return inputs[0].log();
    }

    /**
     * 反向传播计算梯度
     * <p>
     * 对于对数函数，梯度计算公式为：
     * ∂ln(x)/∂x = 1/x
     *
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        return Collections.singletonList(yGrad.div(inputs[0].getValue()));
    }

    /**
     * 获取所需输入参数个数
     * <p>
     * 对数函数需要一个输入参数。
     *
     * @return 输入参数个数，固定为1
     */
    @Override
    public int requireInputNum() {
        return 1;
    }
}
