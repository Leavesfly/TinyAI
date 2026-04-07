package io.leavesfly.tinyai.func.math;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.Collections;
import java.util.List;

/**
 * ELU激活函数（Exponential Linear Unit）
 * <p>
 * ELU是一种改进的ReLU激活函数，具有负值饱和特性。
 * 它可以加速学习并产生更准确的分类结果。
 * <p>
 * 公式：
 * ELU(x) = x,                  if x >= 0
 *        = alpha * (exp(x) - 1), if x < 0
 * <p>
 * 特性：
 * - 负值饱和，减少前向传播中的方差偏移
 * - 均值更接近零，加速学习
 * - 梯度更平滑，训练更稳定
 * - 默认alpha = 1.0
 *
 * @author leavesfly
 * @version 1.0
 */
public class ELU extends Function {

    /**
     * alpha参数
     * 控制负值饱和的缩放因子，默认为1.0
     */
    private final float alpha;

    /**
     * 构造函数
     *
     * @param alpha alpha参数
     */
    public ELU(float alpha) {
        this.alpha = alpha;
    }

    /**
     * 默认构造函数（alpha = 1.0）
     */
    public ELU() {
        this(1.0f);
    }

    /** 缓存前向传播的 ELU 输出，用于 backward 的等价形式 */
    private NdArray cachedOutput;

    /**
     * 前向传播计算ELU
     * <p>
     * 使用向量化操作：
     * ELU(x) = x * (x >= 0) + alpha * (exp(x) - 1) * (x < 0)
     *
     * @param inputs 输入的NdArray数组，长度为1
     * @return ELU函数值的NdArray
     */
    @Override
    public NdArray forward(NdArray... inputs) {
        NdArray x = inputs[0];
        NdArray zero = NdArray.zeros(x.getShape());
        // positiveMask: x >= 0 → gt(0) + eq(0)
        NdArray positiveMask = x.gt(zero).add(x.eq(zero));
        NdArray positivePart = x.mul(positiveMask);
        // negativeMask: x < 0
        NdArray negativeMask = zero.gt(x);
        NdArray negativePart = x.exp().sub(NdArray.ones(x.getShape())).mulNum(alpha).mul(negativeMask);
        cachedOutput = positivePart.add(negativePart);
        return cachedOutput;
    }

    /**
     * 反向传播计算梯度
     * <p>
     * 使用向量化操作和等价形式：
     * ELU'(x) = 1 (x >= 0), ELU(x) + alpha (x < 0)
     *
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        NdArray x = inputs[0].getValue();
        NdArray zero = NdArray.zeros(x.getShape());
        NdArray positiveMask = x.gt(zero).add(x.eq(zero));
        NdArray negativeMask = zero.gt(x);
        // x >= 0: grad = 1; x < 0: grad = ELU(x) + alpha
        NdArray gradMultiplier = positiveMask.add(
                cachedOutput.add(NdArray.like(cachedOutput.getShape(), alpha)).mul(negativeMask));
        return Collections.singletonList(yGrad.mul(gradMultiplier));
    }

    /**
     * 获取所需输入参数个数
     *
     * @return 输入参数个数，固定为1
     */
    @Override
    public int requireInputNum() {
        return 1;
    }

    /**
     * 获取alpha参数
     *
     * @return alpha参数值
     */
    public float getAlpha() {
        return alpha;
    }
}

