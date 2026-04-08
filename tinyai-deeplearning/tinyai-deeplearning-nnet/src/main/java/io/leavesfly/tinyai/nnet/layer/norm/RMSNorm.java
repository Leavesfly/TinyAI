package io.leavesfly.tinyai.nnet.layer.norm;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.core.Module;
import io.leavesfly.tinyai.nnet.core.Parameter;
import io.leavesfly.tinyai.nnet.init.Initializers;

/**
 * V2版本的RMSNorm层（Root Mean Square Layer Normalization）
 * <p>
 * RMSNorm是LayerNorm的简化版本，去掉了均值中心化步骤，
 * 因此计算更高效。广泛应用于LLaMA、DeepSeek等现代大语言模型。
 * <p>
 * 公式：
 * RMS(x) = sqrt(mean(x^2) + eps)
 * y = x / RMS(x) * weight
 * <p>
 * 与LayerNorm的区别：
 * - 不需要计算均值（无减均值步骤）
 * - 不需要beta偏移参数
 * - 计算更高效（约减少40%计算量）
 * - 效果与LayerNorm相当
 * <p>
 * 特性：
 * - 训练和推理模式行为一致
 * - 在最后一个维度上计算RMS
 * - 常用于Transformer的Pre-LN架构
 *
 * @author leavesfly
 * @version 2.0
 */
public class RMSNorm extends Module {

    /**
     * 缩放参数（weight/gamma）
     */
    private Parameter weight;

    /**
     * 归一化的维度大小
     */
    private final int normalizedShape;

    /**
     * 数值稳定性常数
     */
    private final float eps;

    /** 缓存的 eps 常量 Variable，避免每次 forward 都创建新对象，且不参与梯度计算 */
    private transient Variable cachedEpsVar;

    /**
     * 获取 eps 常量 Variable（懒初始化，兼容反序列化场景）
     */
    private Variable getEpsVar() {
        if (cachedEpsVar == null) {
            cachedEpsVar = new Variable(NdArray.of(new float[]{eps}, Shape.of(1)));
            cachedEpsVar.setRequireGrad(false);
        }
        return cachedEpsVar;
    }

    /**
     * 构造函数
     *
     * @param name            层名称
     * @param normalizedShape 归一化的维度大小
     * @param eps             数值稳定性常数（默认1e-6）
     */
    public RMSNorm(String name, int normalizedShape, float eps) {
        super(name);
        this.normalizedShape = normalizedShape;
        this.eps = eps;

        // 创建可训练参数（只有weight，没有bias）
        NdArray weightData = NdArray.of(Shape.of(normalizedShape));
        this.weight = registerParameter("weight", new Parameter(weightData));

        // 初始化 eps 常量缓存
        this.cachedEpsVar = new Variable(NdArray.of(new float[]{eps}, Shape.of(1)));
        this.cachedEpsVar.setRequireGrad(false);

        // 初始化参数
        init();
    }

    /**
     * 构造函数（默认eps=1e-6）
     *
     * @param name            层名称
     * @param normalizedShape 归一化的维度大小
     */
    public RMSNorm(String name, int normalizedShape) {
        this(name, normalizedShape, 1e-6f);
    }

    /**
     * 构造函数（无名称）
     *
     * @param normalizedShape 归一化的维度大小
     */
    public RMSNorm(int normalizedShape) {
        this("rms_norm", normalizedShape, 1e-6f);
    }

    @Override
    public void resetParameters() {
        // weight初始化为1
        Initializers.ones(weight.data());
    }

    @Override
    public Variable forward(Variable... inputs) {
        Variable x = inputs[0];

        // 计算 x^2
        Variable xSquared = x.mul(x);

        // 计算 mean(x^2)
        Variable meanSquared = xSquared.mean(-1, true);

        // 计算 RMS = sqrt(mean(x^2) + eps)
        Variable rms = meanSquared.add(getEpsVar()).sqrt();

        // 归一化: x / RMS
        Variable normalized = x.div(rms);

        // 应用缩放: normalized * weight
        return normalized.mul(weight);
    }

    /**
     * 获取weight参数
     *
     * @return weight参数
     */
    public Parameter getWeight() {
        return weight;
    }

    /**
     * 获取归一化维度大小
     *
     * @return 归一化维度大小
     */
    public int getNormalizedShape() {
        return normalizedShape;
    }

    /**
     * 获取eps值
     *
     * @return eps值
     */
    public float getEps() {
        return eps;
    }

    @Override
    public String toString() {
        return "RMSNorm{" +
                "name='" + name + '\'' +
                ", normalizedShape=" + normalizedShape +
                ", eps=" + eps +
                '}';
    }
}

