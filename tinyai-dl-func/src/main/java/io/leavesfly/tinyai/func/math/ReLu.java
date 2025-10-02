package io.leavesfly.tinyai.func.math;


import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.Collections;
import java.util.List;

/**
 * ReLU激活函数
 * 
 * ReLU (Rectified Linear Unit) 激活函数，用于神经网络中。
 */
public class ReLu extends Function {
    
    /**
     * 前向传播计算ReLU
     * 
     * 计算ReLU函数值：max(0, x)
     * 
     * @param inputs 输入的NdArray数组，长度为1
     * @return ReLU函数值的NdArray
     */
    @Override
    public NdArray forward(NdArray... inputs) {
        return inputs[0].maximum(0f);
    }

    /**
     * 反向传播计算梯度
     * 
     * 对于ReLU函数，梯度计算规则为：
     * - 当x > 0时，梯度为1
     * - 当x <= 0时，梯度为0
     * 
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        return Collections.singletonList(inputs[0].getValue().mask(0).mul(yGrad));
    }

    /**
     * 获取所需输入参数个数
     * 
     * ReLU函数需要一个输入参数。
     * 
     * @return 输入参数个数，固定为1
     */
    @Override
    public int requireInputNum() {
        return 1;
    }
}
