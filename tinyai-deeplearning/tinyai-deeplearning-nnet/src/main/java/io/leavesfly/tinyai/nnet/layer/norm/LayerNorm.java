package io.leavesfly.tinyai.nnet.layer.norm;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.core.Module;
import io.leavesfly.tinyai.nnet.core.Parameter;
import io.leavesfly.tinyai.nnet.init.Initializers;

/**
 * V2版本的LayerNorm层
 * <p>
 * Layer Normalization在特征维度上进行归一化
 * <p>
 * 公式：y = gamma * (x - mean) / sqrt(var + eps) + beta
 * <p>
 * 特性：
 * - 训练和推理模式行为一致（不需要running stats）
 * - 在最后一个维度上计算统计量
 *
 * @author leavesfly
 * @version 2.0
 */
public class LayerNorm extends Module {

    private Parameter gamma;  // 缩放参数
    private Parameter beta;   // 偏移参数
    private final int normalizedShape;
    private final float eps;

    /**
     * 构造函数
     *
     * @param name            层名称
     * @param normalizedShape 归一化的维度大小
     * @param eps             数值稳定性常数（默认1e-5）
     */
    public LayerNorm(String name, int normalizedShape, float eps) {
        super(name);
        this.normalizedShape = normalizedShape;
        this.eps = eps;

        // 创建可训练参数
        NdArray gammaData = NdArray.of(Shape.of(normalizedShape));
        NdArray betaData = NdArray.of(Shape.of(normalizedShape));

        this.gamma = registerParameter("gamma", new Parameter(gammaData));
        this.beta = registerParameter("beta", new Parameter(betaData));

        // 初始化参数
        init();
    }

    /**
     * 构造函数（默认eps=1e-5）
     *
     * @param name            层名称
     * @param normalizedShape 归一化的维度大小
     */
    public LayerNorm(String name, int normalizedShape) {
        this(name, normalizedShape, 1e-5f);
    }

    @Override
    public void resetParameters() {
        // gamma初始化为1
        Initializers.ones(gamma.data());
        // beta初始化为0
        Initializers.zeros(beta.data());
    }

    @Override
    public Variable forward(Variable... inputs) {
        Variable x = inputs[0];

        // 计算均值和方差（在最后一维）
        Variable mean = x.mean(-1, true);
        Variable variance = x.var(-1, true);

        // 归一化
        Variable normalized = x.sub(mean).div(variance.add(new Variable(eps)).sqrt());

        // 应用缩放和偏移
        return normalized.mul(gamma).add(beta);
    }

    /**
     * 获取gamma参数
     *
     * @return gamma参数
     */
    public Parameter getGamma() {
        return gamma;
    }

    /**
     * 获取beta参数
     *
     * @return beta参数
     */
    public Parameter getBeta() {
        return beta;
    }

    @Override
    public String toString() {
        return "LayerNorm{" +
                "name='" + name + '\'' +
                ", normalizedShape=" + normalizedShape +
                ", eps=" + eps +
                '}';
    }
}
