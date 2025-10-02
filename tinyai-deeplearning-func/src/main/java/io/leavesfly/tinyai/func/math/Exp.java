package io.leavesfly.tinyai.func.math;


import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.Collections;
import java.util.List;

/**
 * 指数函数
 * 
 * 计算以e为底的指数值。
 */
public class Exp extends Function {
    /**
     * 前向传播计算指数
     * 
     * 计算输入值的指数值：e^x
     * 
     * @param inputs 输入的NdArray数组，长度为1
     * @return 指数值的NdArray
     */
    @Override
    public NdArray forward(NdArray... inputs) {
        return inputs[0].exp();
    }

    /**
     * 反向传播计算梯度
     * 
     * 对于指数函数，梯度计算公式为：
     * ∂e^x/∂x = e^x
     * 
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        return Collections.singletonList(inputs[0].getValue().exp().mul(yGrad));
    }

    /**
     * 获取所需输入参数个数
     * 
     * 指数函数需要一个输入参数。
     * 
     * @return 输入参数个数，固定为1
     */
    @Override
    public int requireInputNum() {
        return 1;
    }
}
