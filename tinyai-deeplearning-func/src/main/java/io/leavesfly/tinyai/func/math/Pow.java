package io.leavesfly.tinyai.func.math;


import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.Collections;
import java.util.List;

/**
 * 幂函数
 * <p>
 * 计算输入值的幂次方。
 */
public class Pow extends Function {

    private float pow = 1f;

    /**
     * 构造函数
     *
     * @param pow 幂指数
     */
    public Pow(float pow) {
        this.pow = pow;
    }

    /**
     * 前向传播计算幂函数
     * <p>
     * 计算输入值的幂次方：x^pow
     *
     * @param inputs 输入的NdArray数组，长度为1
     * @return 幂函数值的NdArray
     */
    @Override
    public NdArray forward(NdArray... inputs) {
        return inputs[0].pow(pow);
    }

    /**
     * 反向传播计算梯度
     * <p>
     * 对于幂函数，梯度计算公式为：
     * ∂x^pow/∂x = pow * x^(pow-1)
     *
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        NdArray x0 = inputs[0].getValue();
        return Collections.singletonList(x0.pow(pow - 1f).mulNum(pow).mul(yGrad));
    }

    /**
     * 获取所需输入参数个数
     * <p>
     * 幂函数需要一个输入参数。
     *
     * @return 输入参数个数，固定为1
     */
    @Override
    public int requireInputNum() {
        return 1;
    }

}
