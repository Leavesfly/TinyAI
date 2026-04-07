package io.leavesfly.tinyai.func.base;


import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.func.util.BroadcastUtils;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.Arrays;
import java.util.List;

/**
 * 减法函数
 * <p>
 * 实现两个变量的减法运算，支持广播操作。
 */
public class Sub extends Function {

    private Shape input0Shape;
    private Shape input1Shape;

    /**
     * 前向传播计算减法
     * <p>
     * 执行两个NdArray的减法运算。如果两个输入的形状不同，
     * 则进行广播以匹配形状。
     *
     * @param inputs 输入的NdArray数组，长度为2
     * @return 减法运算结果的NdArray
     */
    @Override
    public NdArray forward(NdArray... inputs) {
        NdArray input0 = inputs[0];
        NdArray input1 = inputs[1];
        
        input0Shape = input0.getShape();
        input1Shape = input1.getShape();

        if (input0Shape.equals(input1Shape)) {
            return input0.sub(input1);
        } else if (BroadcastUtils.isBroadcastable(input1Shape, input0Shape)) {
            return input0.sub(input1.broadcastTo(input0Shape));
        } else if (BroadcastUtils.isBroadcastable(input0Shape, input1Shape)) {
            return input0.broadcastTo(input1Shape).sub(input1);
        } else {
            throw new IllegalArgumentException(
                    String.format("减法操作的形状不兼容：%s vs %s", input0Shape, input1Shape));
        }
    }

    /**
     * 反向传播计算梯度
     * <p>
     * 计算减法运算的梯度，支持广播情况。
     * 对于 z = x - y，有：
     * - ∂z/∂x = 1
     * - ∂z/∂y = -1
     * 当存在广播时，需要将梯度 sumTo 回原始形状。
     *
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        Shape yGradShape = yGrad.getShape();
        NdArray grad0 = input0Shape.equals(yGradShape) ? yGrad : BroadcastUtils.sumToShape(yGrad, input0Shape, yGradShape);
        NdArray negYGrad = yGrad.neg();
        NdArray grad1 = input1Shape.equals(yGradShape) ? negYGrad : BroadcastUtils.sumToShape(negYGrad, input1Shape, yGradShape);
        return Arrays.asList(grad0, grad1);
    }

    /**
     * 获取所需输入参数个数
     * <p>
     * 减法运算需要两个输入参数。
     *
     * @return 输入参数个数，固定为2
     */
    @Override
    public int requireInputNum() {
        return 2;
    }
}