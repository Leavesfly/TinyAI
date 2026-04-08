package io.leavesfly.tinyai.nnet.init;

import io.leavesfly.tinyai.ndarr.NdArray;

/**
 * 正态分布初始化器
 * &lt;p&gt;
 * 从正态分布 N(mean, std²) 中采样初始化张量
 *
 * @author leavesfly
 * @version 2.0
 */
public class NormalInitializer implements Initializer {

    private final float mean;
    private final float std;

    /**
     * 构造函数
     *
     * @param mean 均值
     * @param std  标准差
     */
    public NormalInitializer(float mean, float std) {
        this.mean = mean;
        this.std = std;
    }

    @Override
    public void initialize(NdArray tensor) {
        float[] data = tensor.getArray();
        for (int i = 0; i < data.length; i++) {
            data[i] = mean + (float) (Initializers.getSharedRandom().nextGaussian() * std);
        }
    }
}