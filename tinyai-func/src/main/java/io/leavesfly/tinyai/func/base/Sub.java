package io.leavesfly.tinyai.func.base;


import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.Arrays;
import java.util.List;

/**
 * 减法函数
 * <p>
 * 实现两个变量的减法运算。
 */
public class Sub extends Function {

    /**
     * 前向传播计算减法
     * <p>
     * 执行两个NdArray的减法运算：inputs[0] - inputs[1]
     *
     * @param inputs 输入的NdArray数组，长度为2
     * @return 减法运算结果的NdArray
     */
    @Override
    public NdArray forward(NdArray... inputs) {
        return inputs[0].sub(inputs[1]);
    }

    /**
     * 反向传播计算梯度
     * <p>
     * 计算减法运算的梯度。
     * 对于 z = x - y，有：
     * - ∂z/∂x = 1
     * - ∂z/∂y = -1
     *
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        return Arrays.asList(yGrad, yGrad.neg());
    }

    /**
     * 获取所需输入参数个数
     * <p>
     * 减法运算需要两个输入参数。
     *
     * @return 输入参数个数，固定为2
     */
    @Override
    public int requireInputNum() {
        return 2;
    }
}