package io.leavesfly.tinyai.nnet.init;

import io.leavesfly.tinyai.ndarr.NdArray;

/**
 * 零初始化器
 * <p>
 * 将张量所有元素设置为0
 *
 * @author leavesfly
 * @version 2.0
 */
public class ZerosInitializer implements Initializer {

    @Override
    public void initialize(NdArray tensor) {
        float[] data = tensor.getArray();
        for (int i = 0; i < data.length; i++) {
            data[i] = 0.0f;
        }
    }
}
