package io.leavesfly.tinyai.func.math;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.Collections;
import java.util.List;

/**
 * 最大值函数
 * 
 * 计算输入数组沿指定轴的最大值。
 */
public class Max extends Function {
    int axis;
    boolean keepdims;

    /**
     * 构造函数
     * 
     * @param _axis 指定轴
     * @param _keepdims 是否保持维度
     */
    public Max(int _axis, boolean _keepdims) {
        this.axis = _axis;
        this.keepdims = _keepdims;
    }

    /**
     * 前向传播计算最大值
     * 
     * 计算输入数组沿指定轴的最大值。
     * 
     * @param inputs 输入的NdArray数组，长度为1
     * @return 最大值的NdArray
     */
    @Override
    public NdArray forward(NdArray... inputs) {
        if (keepdims) {
            return inputs[0].max(axis).broadcastTo(inputs[0].getShape());
        }
        return inputs[0].max(axis);
    }

    /**
     * 反向传播计算梯度
     * 
     * 对于最大值函数，梯度计算规则为：
     * - 最大值位置的梯度为1，其他位置的梯度为0
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

        // 当多个元素同时为最大值时，将梯度平均分配到这些位置
        NdArray mask = x.eq(y);

        // sum(axis) 会移除指定轴（降维），需要先 reshape 恢复维度再广播
        NdArray maskSum = mask.sum(axis);
        int[] keepDimsShape = x.getShape().getShapeDims().clone();
        keepDimsShape[axis] = 1;
        NdArray maskCount = maskSum.reshape(Shape.of(keepDimsShape)).broadcastTo(x.getShape());
        NdArray safeMaskCount = maskCount.clip(1f, Float.MAX_VALUE);
        return Collections.singletonList(mask.div(safeMaskCount).mul(yGrad));
    }

    /**
     * 获取所需输入参数个数
     * 
     * 最大值函数需要一个输入参数。
     * 
     * @return 输入参数个数，固定为1
     */
    @Override
    public int requireInputNum() {
        return 1;
    }
}
