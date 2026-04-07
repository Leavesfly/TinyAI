package io.leavesfly.tinyai.func.math;

import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.Collections;
import java.util.List;

/**
 * 最小值函数
 * <p>
 * 计算输入数组沿指定轴的最小值。
 * 反向传播时，梯度仅传递到最小值所在位置。
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

    /**
     * 反向传播计算梯度
     * <p>
     * 对于最小值函数，梯度仅传递到最小值所在位置：
     * - 最小值位置的梯度等于上游梯度
     * - 其他位置的梯度为0
     *
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        NdArray x = inputs[0].getValue();
        NdArray y = output.getValue();

        if (!keepdims) {
            yGrad = yGrad.broadcastTo(x.getShape());
            y = y.broadcastTo(x.getShape());
        }

        // 当多个元素同时为最小值时，将梯度平均分配到这些位置
        NdArray mask = x.eq(y);
        NdArray maskCount = mask.sum(axis).broadcastTo(x.getShape());
        NdArray safeMaskCount = maskCount.clip(1f, Float.MAX_VALUE);
        return Collections.singletonList(mask.div(safeMaskCount).mul(yGrad));
    }
}
