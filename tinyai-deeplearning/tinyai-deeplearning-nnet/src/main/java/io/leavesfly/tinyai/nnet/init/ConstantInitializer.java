package io.leavesfly.tinyai.nnet.init;

import io.leavesfly.tinyai.ndarr.NdArray;

/**
 * 常量初始化器
 * <p>
 * 将张量所有元素设置为指定常量
 *
 * @author leavesfly
 * @version 2.0
 */
public class ConstantInitializer implements Initializer {

    private final float value;

    /**
     * 构造函数
     *
     * @param value 常量值
     */
    public ConstantInitializer(float value) {
        this.value = value;
    }

    @Override
    public void initialize(NdArray tensor) {
        float[] data = tensor.getArray();
        for (int i = 0; i < data.length; i++) {
            data[i] = value;
        }
    }
}
