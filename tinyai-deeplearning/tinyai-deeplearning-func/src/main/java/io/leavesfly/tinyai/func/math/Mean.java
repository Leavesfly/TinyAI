package io.leavesfly.tinyai.func.math;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.Collections;
import java.util.List;

/**
 * 均值函数
 * <p>
 * 计算输入数组沿指定轴的均值。
 *
 * @author leavesfly
 * @version 1.0
 */
public class Mean extends Function {

    private int axis;
    private boolean keepdims;
    private Shape inputShape;
    private int axisSize;

    /**
     * 构造函数
     *
     * @param axis     指定轴
     * @param keepdims 是否保持维度
     */
    public Mean(int axis, boolean keepdims) {
        this.axis = axis;
        this.keepdims = keepdims;
    }

    /**
     * 前向传播计算均值
     * <p>
     * 计算输入数组沿指定轴的均值。
     *
     * @param inputs 输入的NdArray数组，长度为1
     * @return 均值的NdArray
     */
    @Override
    public NdArray forward(NdArray... inputs) {
        inputShape = inputs[0].getShape();
        int[] shape = inputShape.getShapeDims();
        
        // 处理负数轴索引
        int actualAxis = axis;
        if (actualAxis < 0) {
            actualAxis = shape.length + actualAxis;
        }
        
        // 保存轴的大小，用于反向传播
        axisSize = shape[actualAxis];

        NdArray result = inputs[0].mean(actualAxis);

        if (keepdims) {
            // 插入被约简轴的维度为1，再广播回原形状
            int[] keepDims = shape.clone();
            keepDims[actualAxis] = 1;
            result = result.reshape(Shape.of(keepDims));
            return result.broadcastTo(inputShape);
        }
        return result;
    }

    /**
     * 反向传播计算梯度
     * <p>
     * 对于均值函数，梯度计算规则为：
     * - 将梯度值平均分配给所有元素
     *
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        if (!keepdims) {
            // keepdims=false 时，yGrad 缺少被约简的轴维度，
            // 需要先将该轴以大小 1 重新插入，使 shape 与 inputShape 可广播对齐。
            int[] shape = inputShape.getShapeDims();
            int actualAxis = axis;
            if (actualAxis < 0) {
                actualAxis = shape.length + actualAxis;
            }
            int[] gradShape = yGrad.getShape().getShapeDims();
            int[] expandedShape = new int[gradShape.length + 1];
            for (int i = 0; i < actualAxis; i++) {
                expandedShape[i] = gradShape[i];
            }
            expandedShape[actualAxis] = 1;
            for (int i = actualAxis; i < gradShape.length; i++) {
                expandedShape[i + 1] = gradShape[i];
            }
            yGrad = yGrad.reshape(Shape.of(expandedShape));
            yGrad = yGrad.broadcastTo(inputShape);
        }
        // 梯度需要除以轴的大小，因为是平均值
        NdArray grad = yGrad.divNum(axisSize);
        return Collections.singletonList(grad);
    }

    /**
     * 获取所需输入参数个数
     * <p>
     * 均值函数需要一个输入参数。
     *
     * @return 输入参数个数，固定为1
     */
    @Override
    public int requireInputNum() {
        return 1;
    }
}
