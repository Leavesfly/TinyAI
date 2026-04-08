package io.leavesfly.tinyai.nnet.layer.activation;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.core.Module;

/**
 * V2版本的ELU激活函数层（Exponential Linear Unit）
 * <p>
 * ELU是一种改进的ReLU激活函数，具有负值饱和特性。
 * 它可以加速学习并产生更准确的分类结果。
 * <p>
 * 公式：
 * ELU(x) = x,                    if x >= 0
 *        = alpha * (exp(x) - 1), if x < 0
 * <p>
 * 特性：
 * - 负值饱和，减少前向传播中的方差偏移
 * - 均值更接近零，加速学习
 * - 梯度更平滑，训练更稳定
 * - 默认alpha = 1.0
 * - 在训练和推理模式下行为一致
 *
 * @author leavesfly
 * @version 2.0
 */
public class ELU extends Module {

    /**
     * alpha参数
     * 控制负值饱和的缩放因子，默认为1.0
     */
    private final float alpha;

    /**
     * 构造函数
     *
     * @param name  层名称
     * @param alpha alpha参数
     */
    public ELU(String name, float alpha) {
        super(name);
        this.alpha = alpha;
    }

    /**
     * 构造函数（默认alpha = 1.0）
     *
     * @param name 层名称
     */
    public ELU(String name) {
        this(name, 1.0f);
    }

    /**
     * 默认构造函数
     */
    public ELU() {
        this("elu", 1.0f);
    }

    /**
     * 构造函数（指定alpha）
     *
     * @param alpha alpha参数
     */
    public ELU(float alpha) {
        this("elu", alpha);
    }

    @Override
    public Variable forward(Variable... inputs) {
        Variable x = inputs[0];
        return x.elu(alpha);
    }

    /**
     * 获取alpha参数
     *
     * @return alpha参数值
     */
    public float getAlpha() {
        return alpha;
    }

    @Override
    public String toString() {
        return "ELU{" +
                "name='" + name + '\'' +
                ", alpha=" + alpha +
                '}';
    }
}

