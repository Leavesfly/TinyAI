package io.leavesfly.tinyai.func.math;


import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.Collections;
import java.util.List;

/**
 * Sigmoid激活函数
 * <p>
 * Sigmoid激活函数，用于神经网络中，将输入值映射到(0,1)区间。
 */
public class Sigmoid extends Function {

    /**
     * 前向传播计算Sigmoid
     * <p>
     * 计算Sigmoid函数值：1 / (1 + e^(-x))
     *
     * @param inputs 输入的NdArray数组，长度为1
     * @return Sigmoid函数值的NdArray
     */
    @Override
    public NdArray forward(NdArray... inputs) {
        return inputs[0].sigmoid();
    }

    /**
     * 反向传播计算梯度
     * <p>
     * 对于Sigmoid函数，梯度计算公式为：
     * ∂sigmoid(x)/∂x = sigmoid(x) * (1 - sigmoid(x))
     *
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        if (inputs == null || inputs.length == 0) {
            throw new RuntimeException("Sigmoid backward called without proper forward initialization. This usually happens when using new Sigmoid().backward() instead of reusing a forward-passed instance.");
        }
        NdArray x = inputs[0].getValue();
        NdArray sigmoidX = x.sigmoid();
        return Collections.singletonList(yGrad.mul(sigmoidX).mul(NdArray.ones(sigmoidX.getShape()).sub(sigmoidX)));
    }

    /**
     * 获取所需输入参数个数
     * <p>
     * Sigmoid函数需要一个输入参数。
     *
     * @return 输入参数个数，固定为1
     */
    @Override
    public int requireInputNum() {
        return 1;
    }
}
