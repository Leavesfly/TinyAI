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

    /** 防止 exp 溢出的输入裁剪上界 */
    private static final float MAX_EXP_INPUT = 88.0f;

    /** 缓存前向传播结果，避免 backward 重复计算 */
    private NdArray cachedExpResult;

    /**
     * 前向传播计算指数
     *
     * 计算输入值的指数值：e^x
     * 对输入进行裁剪防止溢出。
     *
     * @param inputs 输入的NdArray数组，长度为1
     * @return 指数值的NdArray
     */
    @Override
    public NdArray forward(NdArray... inputs) {
        NdArray clippedInput = inputs[0].clip(-MAX_EXP_INPUT, MAX_EXP_INPUT);
        cachedExpResult = clippedInput.exp();
        return cachedExpResult;
    }

    /**
     * 反向传播计算梯度
     *
     * 对于指数函数，梯度计算公式为：
     * ∂e^x/∂x = e^x
     * 复用前向传播缓存的结果，避免重复计算。
     *
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        return Collections.singletonList(cachedExpResult.mul(yGrad));
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
