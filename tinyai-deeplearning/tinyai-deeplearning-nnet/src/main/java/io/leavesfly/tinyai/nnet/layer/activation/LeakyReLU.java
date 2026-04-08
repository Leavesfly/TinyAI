package io.leavesfly.tinyai.nnet.layer.activation;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.core.Module;

/**
 * V2版本的LeakyReLU激活函数层（Leaky Rectified Linear Unit）
 * <p>
 * LeakyReLU是ReLU的改进版本，解决了ReLU的"神经元死亡"问题。
 * 它允许负值以一个小的斜率通过，从而保持梯度流动。
 * <p>
 * 公式：
 * LeakyReLU(x) = x,                  if x >= 0
 *              = negative_slope * x, if x < 0
 * <p>
 * 或简写为：LeakyReLU(x) = max(negative_slope * x, x)
 * <p>
 * 特性：
 * - 解决ReLU的神经元死亡问题
 * - 负值区域有小梯度，保持梯度流动
 * - 默认negative_slope = 0.01
 * - 在训练和推理模式下行为一致
 *
 * @author leavesfly
 * @version 2.0
 */
public class LeakyReLU extends Module {

    /**
     * 负斜率参数
     * 控制负值区域的斜率，默认为0.01
     */
    private final float negativeSlope;

    /**
     * 构造函数
     *
     * @param name          层名称
     * @param negativeSlope 负斜率参数
     */
    public LeakyReLU(String name, float negativeSlope) {
        super(name);
        this.negativeSlope = negativeSlope;
    }

    /**
     * 构造函数（默认negativeSlope = 0.01）
     *
     * @param name 层名称
     */
    public LeakyReLU(String name) {
        this(name, 0.01f);
    }

    /**
     * 默认构造函数
     */
    public LeakyReLU() {
        this("leaky_relu", 0.01f);
    }

    /**
     * 构造函数（指定negativeSlope）
     *
     * @param negativeSlope 负斜率参数
     */
    public LeakyReLU(float negativeSlope) {
        this("leaky_relu", negativeSlope);
    }

    @Override
    public Variable forward(Variable... inputs) {
        Variable x = inputs[0];
        return x.leakyRelu(negativeSlope);
    }

    /**
     * 获取负斜率参数
     *
     * @return 负斜率参数值
     */
    public float getNegativeSlope() {
        return negativeSlope;
    }

    @Override
    public String toString() {
        return "LeakyReLU{" +
                "name='" + name + '\'' +
                ", negativeSlope=" + negativeSlope +
                '}';
    }
}

