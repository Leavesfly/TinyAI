package io.leavesfly.tinyai.nnet.layer.dnn;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.core.Module;
import io.leavesfly.tinyai.nnet.core.Parameter;
import io.leavesfly.tinyai.nnet.init.Initializers;

/**
 * V2版本的线性层（全连接层）
 * <p>
 * 实现线性变换：y = xW^T + b
 * <p>
 * 特性：
 * - 使用统一的参数注册机制
 * - 支持Kaiming初始化（适配ReLU）
 * - 参数命名规范：weight、bias
 *
 * @author leavesfly
 * @version 2.0
 */
public class Linear extends Module {

    private Parameter weight;
    private Parameter bias;
    private final int inFeatures;
    private final int outFeatures;
    private final boolean useBias;

    /**
     * 构造函数
     *
     * @param name        层名称
     * @param inFeatures  输入特征数
     * @param outFeatures 输出特征数
     * @param useBias     是否使用偏置
     */
    public Linear(String name, int inFeatures, int outFeatures, boolean useBias) {
        super(name);
        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;
        this.useBias = useBias;

        // 创建参数
        NdArray weightData = NdArray.of(Shape.of(outFeatures, inFeatures));
        this.weight = registerParameter("weight", new Parameter(weightData));

        if (useBias) {
            NdArray biasData = NdArray.of(Shape.of(outFeatures));
            this.bias = registerParameter("bias", new Parameter(biasData));
        }

        // 初始化参数
        init();
    }

    /**
     * 构造函数（默认使用偏置）
     *
     * @param name        层名称
     * @param inFeatures  输入特征数
     * @param outFeatures 输出特征数
     */
    public Linear(String name, int inFeatures, int outFeatures) {
        this(name, inFeatures, outFeatures, true);
    }

    @Override
    public void resetParameters() {
        // 使用Kaiming均匀初始化（适配ReLU激活）
        Initializers.kaimingUniform(weight.data(), 0, "fan_in", "relu");
        if (bias != null) {
            Initializers.zeros(bias.data());
        }
    }

    @Override
    public Variable forward(Variable... inputs) {
        Variable x = inputs[0];

        // 线性变换：y = xW^T + b
        // x.shape: (batch, in_features)
        // weight.shape: (out_features, in_features)
        // weight需要转置: (in_features, out_features)
        Variable y = x.matMul(weight.transpose());

        if (bias != null) {
            y = y.add(bias);
        }

        return y;
    }

    /**
     * 获取权重参数
     *
     * @return 权重参数
     */
    public Parameter getWeight() {
        return weight;
    }

    /**
     * 获取偏置参数
     *
     * @return 偏置参数，如果不使用偏置则返回null
     */
    public Parameter getBias() {
        return bias;
    }

    /**
     * 获取输入特征数
     *
     * @return 输入特征数
     */
    public int getInFeatures() {
        return inFeatures;
    }

    /**
     * 获取输出特征数
     *
     * @return 输出特征数
     */
    public int getOutFeatures() {
        return outFeatures;
    }

    @Override
    public String toString() {
        return "Linear{" +
                "name='" + name + '\'' +
                ", inFeatures=" + inFeatures +
                ", outFeatures=" + outFeatures +
                ", useBias=" + useBias +
                '}';
    }
}
