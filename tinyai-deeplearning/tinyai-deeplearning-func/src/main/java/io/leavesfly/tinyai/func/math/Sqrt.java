package io.leavesfly.tinyai.func.math;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.Collections;
import java.util.List;

/**
 * 平方根函数
 * <p>
 * 计算输入值的平方根。
 *
 * @author leavesfly
 * @version 1.0
 */
public class Sqrt extends Function {

    /**
     * 前向传播计算平方根
     * <p>
     * 计算输入值的平方根：sqrt(x)
     *
     * @param inputs 输入的NdArray数组，长度为1
     * @return 平方根的NdArray
     */
    /**
     * 数值稳定性的最小值，防止 sqrt(负数) 和除零
     */
    private static final float EPSILON = 1e-12f;

    @Override
    public NdArray forward(NdArray... inputs) {
        // 对输入添加下限保护，防止负数输入产生 NaN
        NdArray clampedInput = inputs[0].clip(0f, Float.MAX_VALUE);
        return clampedInput.sqrt();
    }

    /**
     * 反向传播计算梯度
     * <p>
     * 对于平方根函数，梯度计算公式为：
     * d(sqrt(x))/dx = 1 / (2 * sqrt(x))
     * 对输入添加下限保护，防止 x=0 时除零产生 Infinity。
     *
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        NdArray x = inputs[0].getValue();
        // 对 x 添加下限保护，防止 sqrt(0) 导致除零
        NdArray safeX = x.clip(EPSILON, Float.MAX_VALUE);
        // 梯度: 1 / (2 * sqrt(x)) = 0.5 / sqrt(x)
        NdArray grad = safeX.sqrt().pow(-1).mulNum(0.5f).mul(yGrad);
        return Collections.singletonList(grad);
    }

    /**
     * 获取所需输入参数个数
     * <p>
     * 平方根函数需要一个输入参数。
     *
     * @return 输入参数个数，固定为1
     */
    @Override
    public int requireInputNum() {
        return 1;
    }
}
