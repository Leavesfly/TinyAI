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
     * 缓存前向传播的 sigmoid 输出，避免反向传播时重复计算
     */
    private NdArray cachedSigmoid;

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
        cachedSigmoid = inputs[0].sigmoid();
        return cachedSigmoid;
    }

    /**
     * 反向传播计算梯度
     * <p>
     * 对于Sigmoid函数，梯度计算公式为：
     * ∂sigmoid(x)/∂x = sigmoid(x) * (1 - sigmoid(x))
     * 复用前向传播缓存的 sigmoid 结果，避免重复计算。
     *
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        if (cachedSigmoid == null) {
            throw new RuntimeException("Sigmoid backward called without proper forward initialization.");
        }
        // 复用缓存的 sigmoid 结果：grad = yGrad * sigmoid * (1 - sigmoid)
        NdArray oneMinusSigmoid = cachedSigmoid.like(1f).sub(cachedSigmoid);
        return Collections.singletonList(yGrad.mul(cachedSigmoid).mul(oneMinusSigmoid));
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
