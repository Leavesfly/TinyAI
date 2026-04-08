package io.leavesfly.tinyai.nnet.layer.activation;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.core.Module;

/**
 * V2版本的GELU激活函数层 (Gaussian Error Linear Unit)
 * <p>
 * GELU是一种平滑的激活函数，广泛应用于Transformer模型（如GPT、BERT、ViT等）。
 * 它通过高斯误差函数对输入进行非线性变换。
 * <p>
 * 公式：
 * GELU(x) = x * Φ(x)
 * 其中 Φ(x) 是标准正态分布的累积分布函数
 * <p>
 * 近似公式（tanh近似）：
 * GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
 * <p>
 * 特性：
 * - 无参数层
 * - 平滑的非线性变换
 * - 在训练和推理模式下行为一致
 * - 相比ReLU，对负值有更平滑的处理
 *
 * @author leavesfly
 * @version 2.0
 */
public class GELU extends Module {

    /**
     * 是否使用近似计算
     * true: 使用tanh近似公式（更快）
     * false: 使用精确计算（当前默认也是tanh近似）
     */
    private final boolean approximate;

    /**
     * 构造函数
     *
     * @param name        层名称
     * @param approximate 是否使用近似计算
     */
    public GELU(String name, boolean approximate) {
        super(name);
        this.approximate = approximate;
    }

    /**
     * 构造函数（默认使用近似计算）
     *
     * @param name 层名称
     */
    public GELU(String name) {
        this(name, true);
    }

    /**
     * 默认构造函数
     */
    public GELU() {
        this("gelu", true);
    }

    @Override
    public Variable forward(Variable... inputs) {
        Variable x = inputs[0];
        // 使用Variable的gelu方法（依赖已有的GELU Function）
        return x.gelu();
    }

    /**
     * 判断是否使用近似计算
     *
     * @return true表示使用近似计算
     */
    public boolean isApproximate() {
        return approximate;
    }

    @Override
    public String toString() {
        return "GELU{" +
                "name='" + name + '\'' +
                ", approximate=" + approximate +
                '}';
    }
}

