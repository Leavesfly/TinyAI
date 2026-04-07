package io.leavesfly.tinyai.func.base;


import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.func.util.BroadcastUtils;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.Arrays;
import java.util.List;

/**
 * 加法函数
 * 
 * 实现两个变量的加法运算，支持广播操作。
 * 当两个输入变量的形状不同时，会自动进行广播以匹配形状。
 */
public class Add extends Function {

    private Shape input0Shape;
    private Shape input1Shape;

    /**
     * 前向传播计算加法
     * 
     * 执行两个NdArray的加法运算。如果两个输入的形状不同，
     * 则进行广播以匹配形状。
     * 
     * @param inputs 输入的NdArray数组，长度为2
     * @return 加法运算结果的NdArray
     */
    @Override
    public NdArray forward(NdArray... inputs) {
        input0Shape = inputs[0].getShape();
        input1Shape = inputs[1].getShape();

        if (input0Shape.equals(input1Shape)) {
            return inputs[0].add(inputs[1]);
        } else if (BroadcastUtils.isBroadcastable(input1Shape, input0Shape)) {
            return inputs[0].add(inputs[1].broadcastTo(input0Shape));
        } else if (BroadcastUtils.isBroadcastable(input0Shape, input1Shape)) {
            return inputs[0].broadcastTo(input1Shape).add(inputs[1]);
        } else {
            throw new IllegalArgumentException(
                    String.format("加法操作的形状不兼容：%s vs %s", input0Shape, input1Shape));
        }
    }

    /**
     * 反向传播计算梯度
     * 
     * 计算加法运算的梯度。对于加法运算，梯度直接传递给两个输入变量。
     * 如果进行了广播操作，则需要对梯度进行相应的sumTo操作。
     * 
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        Shape yGradShape = yGrad.getShape();
        NdArray grad0 = input0Shape.equals(yGradShape) ? yGrad : BroadcastUtils.sumToShape(yGrad, input0Shape, yGradShape);
        NdArray grad1 = input1Shape.equals(yGradShape) ? yGrad : BroadcastUtils.sumToShape(yGrad, input1Shape, yGradShape);
        return Arrays.asList(grad0, grad1);
    }

    /**
     * 获取所需输入参数个数
     * 
     * 加法运算需要两个输入参数。
     * 
     * @return 输入参数个数，固定为2
     */
    @Override
    public int requireInputNum() {
        return 2;
    }
}