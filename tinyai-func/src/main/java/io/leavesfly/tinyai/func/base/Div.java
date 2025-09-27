package io.leavesfly.tinyai.func.base;


import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.Arrays;
import java.util.List;

/**
 * 除法函数
 * <p>
 * 实现两个变量的除法运算。
 */
public class Div extends Function {

    /**
     * 前向传播计算除法
     * <p>
     * 执行两个NdArray的除法运算：inputs[0] / inputs[1]
     *
     * @param inputs 输入的NdArray数组，长度为2
     * @return 除法运算结果的NdArray
     */
    @Override
    public NdArray forward(NdArray... inputs) {
        return inputs[0].div(inputs[1]);
    }

    /**
     * 反向传播计算梯度
     * <p>
     * 计算除法运算的梯度。
     * 对于 z = x / y，有：
     * - ∂z/∂x = 1/y
     * - ∂z/∂y = -x/y²
     *
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {

        NdArray ndArray0 = inputs[0].getValue();
        NdArray ndArray1 = inputs[1].getValue();

        return Arrays.asList(yGrad.div(ndArray1), yGrad.mul(ndArray0.neg().div(ndArray1.square())));
    }

    /**
     * 获取所需输入参数个数
     * <p>
     * 除法运算需要两个输入参数。
     *
     * @return 输入参数个数，固定为2
     */
    @Override
    public int requireInputNum() {
        return 2;
    }
}
