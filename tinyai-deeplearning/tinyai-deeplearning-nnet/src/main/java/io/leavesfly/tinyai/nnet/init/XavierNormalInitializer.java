package io.leavesfly.tinyai.nnet.init;

import io.leavesfly.tinyai.ndarr.NdArray;

/**
 * Xavier正态初始化器（Glorot正态初始化）
 * &lt;p&gt;
 * 适用于Sigmoid、Tanh等激活函数
 * &lt;p&gt;
 * 从正态分布 N(0, std²) 中采样，其中：
 * std = gain * sqrt(2 / (fan_in + fan_out))
 * &lt;p&gt;
 * 参考论文：
 * Understanding the difficulty of training deep feedforward neural networks
 * by Xavier Glorot and Yoshua Bengio (2010)
 *
 * @author leavesfly
 * @version 2.0
 */
public class XavierNormalInitializer implements Initializer {

    private final float gain;

    /**
     * 构造函数
     *
     * @param gain 增益系数（默认为1.0）
     */
    public XavierNormalInitializer(float gain) {
        this.gain = gain;
    }

    /**
     * 默认构造函数（gain=1.0）
     */
    public XavierNormalInitializer() {
        this(1.0f);
    }

    @Override
    public void initialize(NdArray tensor) {
        int[] fanInOut = Initializers.calculateFanInAndFanOut(tensor.getShape());
        int fanIn = fanInOut[0];
        int fanOut = fanInOut[1];

        // 计算标准差
        float std = gain * (float) Math.sqrt(2.0 / (fanIn + fanOut));

        float[] data = tensor.getArray();
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) (Initializers.getSharedRandom().nextGaussian() * std);
        }
    }
}