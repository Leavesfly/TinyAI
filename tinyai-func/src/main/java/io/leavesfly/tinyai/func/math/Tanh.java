package io.leavesfly.tinyai.func.math;


import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.Collections;
import java.util.List;

/**
 * Tanh激活函数
 * <p>
 * 双曲正切激活函数，用于神经网络中，将输入值映射到(-1,1)区间。
 */
public class Tanh extends Function {
    /**
     * 前向传播计算Tanh
     * <p>
     * 计算Tanh函数值：(e^x - e^(-x)) / (e^x + e^(-x))
     *
     * @param inputs 输入的NdArray数组，长度为1
     * @return Tanh函数值的NdArray
     */
    @Override
    public NdArray forward(NdArray... inputs) {
        return inputs[0].tanh();
    }

    /**
     * 反向传播计算梯度
     * <p>
     * 对于Tanh函数，梯度计算公式为：
     * ∂tanh(x)/∂x = 1 - tanh(x)²
     *
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        NdArray x = inputs[0].getValue();
        NdArray tanhX = x.tanh();
        return Collections.singletonList(
                yGrad.mul(NdArray.ones(tanhX.getShape()).sub(tanhX.square())));
    }

    /**
     * 获取所需输入参数个数
     * <p>
     * Tanh函数需要一个输入参数。
     *
     * @return 输入参数个数，固定为1
     */
    @Override
    public int requireInputNum() {
        return 1;
    }
}
