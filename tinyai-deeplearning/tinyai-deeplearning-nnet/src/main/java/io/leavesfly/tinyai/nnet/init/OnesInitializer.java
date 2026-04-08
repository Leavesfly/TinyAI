package io.leavesfly.tinyai.nnet.init;

import io.leavesfly.tinyai.ndarr.NdArray;

/**
 * 全一初始化器
 * <p>
 * 将张量所有元素设置为1
 *
 * @author leavesfly
 * @version 2.0
 */
public class OnesInitializer implements Initializer {

    @Override
    public void initialize(NdArray tensor) {
        float[] data = tensor.getArray();
        for (int i = 0; i < data.length; i++) {
            data[i] = 1.0f;
        }
    }
}
