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

    private Shape x0Shape;
    private Shape x1Shape;

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
        
        x0Shape = input0.getShape();
        x1Shape = input1.getShape();
        
        // 检查是否需要广播
        if (input0.getShape().equals(input1.getShape())) {
            // 形状相同，直接相减
            return input0.sub(input1);
        } else {
            // 需要广播
            Shape shape0 = input0.getShape();
            Shape shape1 = input1.getShape();
            
            // 判断广播方向
            if (BroadcastUtils.isBroadcastable(shape1, shape0)) {
                // input1 需要广播到 input0 的形状
                return input0.sub(input1.broadcastTo(shape0));
            } else if (BroadcastUtils.isBroadcastable(shape0, shape1)) {
                // input0 需要广播到 input1 的形状
                return input0.broadcastTo(shape1).sub(input1);
            } else {
                throw new IllegalArgumentException(
                    String.format("减法操作的形状不兼容：%s vs %s", shape0, shape1)
                );
            }
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
        NdArray gx0 = x0Shape.equals(yGradShape) ? yGrad : BroadcastUtils.sumToShape(yGrad, x0Shape, yGradShape);
        NdArray gx1 = x1Shape.equals(yGradShape) ? yGrad.neg() : BroadcastUtils.sumToShape(yGrad.neg(), x1Shape, yGradShape);
        return Arrays.asList(gx0, gx1);
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