package io.leavesfly.tinyai.func.math;


import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.Collections;
import java.util.List;

/**
 * 平方函数
 * <p>
 * 计算输入值的平方。
 */
public class Squ extends Function {

    /**
     * 前向传播计算平方
     * <p>
     * 计算输入值的平方：x²
     *
     * @param inputs 输入的NdArray数组，长度为1
     * @return 平方值的NdArray
     */
    @Override
    public NdArray forward(NdArray... inputs) {
        return inputs[0].pow(2);
    }

    /**
     * 反向传播计算梯度
     * <p>
     * 对于平方函数，梯度计算公式为：
     * ∂x²/∂x = 2x
     *
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        NdArray x = inputs[0].getValue();
        return Collections.singletonList(x.mulNum(2).mul(yGrad));
    }

    /**
     * 获取所需输入参数个数
     * <p>
     * 平方函数需要一个输入参数。
     *
     * @return 输入参数个数，固定为1
     */
    @Override
    public int requireInputNum() {
        return 1;
    }
}
