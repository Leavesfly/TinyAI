package io.leavesfly.tinyai.nnet.init;

import io.leavesfly.tinyai.ndarr.NdArray;

/**
 * Kaiming正态初始化器（He正态初始化）
 * &lt;p&gt;
 * 适用于ReLU及其变体激活函数
 * &lt;p&gt;
 * 从正态分布 N(0, std²) 中采样，其中：
 * std = sqrt(2 / ((1 + a²) * fan))
 * &lt;p&gt;
 * fan根据mode选择fan_in或fan_out
 * &lt;p&gt;
 * 参考论文：
 * Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
 * by Kaiming He et al. (2015)
 *
 * @author leavesfly
 * @version 2.0
 */
public class KaimingNormalInitializer implements Initializer {

    private final float a;
    private final String mode;
    private final String nonlinearity;

    /**
     * 构造函数
     *
     * @param a            leaky_relu的负斜率（对于ReLU使用0）
     * @param mode         "fan_in"或"fan_out"
     * @param nonlinearity 非线性函数类型（"relu"、"leaky_relu"等）
     */
    public KaimingNormalInitializer(float a, String mode, String nonlinearity) {
        this.a = a;
        this.mode = mode;
        this.nonlinearity = nonlinearity;
    }

    /**
     * 默认构造函数（a=0, mode="fan_in", nonlinearity="relu"）
     */
    public KaimingNormalInitializer() {
        this(0, "fan_in", "relu");
    }

    @Override
    public void initialize(NdArray tensor) {
        int[] fanInOut = Initializers.calculateFanInAndFanOut(tensor.getShape());
        int fan = Initializers.getFan(fanInOut[0], fanInOut[1], mode);
        float gain = Initializers.calculateGain(a, nonlinearity);

        // 计算标准差
        float std = gain / (float) Math.sqrt(fan);

        float[] data = tensor.getArray();
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) (Initializers.getSharedRandom().nextGaussian() * std);
        }
    }
}