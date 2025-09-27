package io.leavesfly.tinyai.func.math;


import io.leavesfly.tinyai.ndarr.NdArray;

/**
 * 最小值函数
 * 
 * 计算输入数组沿指定轴的最小值。
 */
public class Min extends Max {

    /**
     * 构造函数
     * 
     * @param _axis 指定轴
     * @param _keepdims 是否保持维度
     */
    public Min(int _axis, boolean _keepdims) {
        super(_axis, _keepdims);
    }

    /**
     * 前向传播计算最小值
     * 
     * 计算输入数组沿指定轴的最小值。
     * 
     * @param inputs 输入的NdArray数组，长度为1
     * @return 最小值的NdArray
     */
    @Override
    public NdArray forward(NdArray... inputs) {
        if (keepdims) {
            return inputs[0].min(axis).broadcastTo(inputs[0].getShape());
        }
        return inputs[0].min(axis);
    }

}
