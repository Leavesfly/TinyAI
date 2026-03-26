package io.leavesfly.tinyai.nnet.v2.init;

import io.leavesfly.tinyai.ndarr.NdArray;

/**
 * Xavier均匀初始化器（Glorot均匀初始化）
 * &lt;p&gt;
 * 适用于Sigmoid、Tanh等激活函数
 * &lt;p&gt;
 * 从均匀分布 U(-a, a) 中采样，其中：
 * a = gain * sqrt(6 / (fan_in + fan_out))
 * &lt;p&gt;
 * 参考论文：
 * Understanding the difficulty of training deep feedforward neural networks
 * by Xavier Glorot and Yoshua Bengio (2010)
 *
 * @author leavesfly
 * @version 2.0
 */
public class XavierUniformInitializer implements Initializer {

    private final float gain;

    /**
     * 构造函数
     *
     * @param gain 增益系数（默认为1.0）
     */
    public XavierUniformInitializer(float gain) {
        this.gain = gain;
    }

    /**
     * 默认构造函数（gain=1.0）
     */
    public XavierUniformInitializer() {
        this(1.0f);
    }

    @Override
    public void initialize(NdArray tensor) {
        int[] fanInOut = Initializers.calculateFanInAndFanOut(tensor.getShape());
        int fanIn = fanInOut[0];
        int fanOut = fanInOut[1];

        // 计算均匀分布的边界
        float std = gain * (float) Math.sqrt(6.0 / (fanIn + fanOut));

        float[] data = tensor.getArray();
        for (int i = 0; i < data.length; i++) {
            data[i] = -std + Initializers.getSharedRandom().nextFloat() * (2 * std);
        }
    }
}