package io.leavesfly.tinyai.func.base;


import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.func.util.BroadcastUtils;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.Arrays;
import java.util.List;

/**
 * 除法函数
 * <p>
 * 实现两个变量的除法运算，支持广播操作。
 */
public class Div extends Function {

    /**
     * 前向传播计算除法
     * <p>
     * 执行两个NdArray的除法运算。如果两个输入的形状不同，
     * 则进行广播以匹配形状。
     *
     * @param inputs 输入的NdArray数组，长度为2
     * @return 除法运算结果的NdArray
     */
    /**
     * 除零保护的 epsilon 值
     */
    private static final float EPSILON = 1e-12f;

    @Override
    public NdArray forward(NdArray... inputs) {
        NdArray input0 = inputs[0];
        NdArray input1 = inputs[1];
        
        // 检查是否需要广播
        if (input0.getShape().equals(input1.getShape())) {
            // 形状相同，直接相除（添加除零保护）
            return input0.div(safeDiv(input1));
        } else {
            // 需要广播
            Shape shape0 = input0.getShape();
            Shape shape1 = input1.getShape();
            
            // 判断广播方向
            if (BroadcastUtils.isBroadcastable(shape1, shape0)) {
                // input1 需要广播到 input0 的形状（添加除零保护）
                return input0.div(safeDiv(input1.broadcastTo(shape0)));
            } else if (BroadcastUtils.isBroadcastable(shape0, shape1)) {
                // input0 需要广播到 input1 的形状（添加除零保护）
                return input0.broadcastTo(shape1).div(safeDiv(input1));
            } else {
                throw new IllegalArgumentException(
                    String.format("除法操作的形状不兼容：%s vs %s", shape0, shape1)
                );
            }
        }
    }
    


    /**
     * 反向传播计算梯度
     * <p>
     * 计算除法运算的梯度，支持广播情况。
     * 对于 z = x / y，有：
     * - ∂z/∂x = 1/y
     * - ∂z/∂y = -x/y²
     * 当存在广播时，需要将梯度 sumTo 回原始形状。
     *
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        NdArray numerator = inputs[0].getValue();
        NdArray denominator = inputs[1].getValue();
        Shape yGradShape = yGrad.getShape();

        NdArray grad0 = computeNumeratorGrad(yGrad, denominator, numerator.getShape(), yGradShape);
        NdArray grad1 = computeDenominatorGrad(yGrad, numerator, denominator, yGradShape);
        return Arrays.asList(grad0, grad1);
    }

    /** 计算 ∂(x/y)/∂x = 1/y 的梯度 */
    private NdArray computeNumeratorGrad(NdArray yGrad, NdArray denominator, Shape targetShape, Shape yGradShape) {
        NdArray safeDenom = safeDiv(denominator);
        NdArray grad = denominator.getShape().equals(yGradShape)
                ? yGrad.div(safeDenom)
                : yGrad.div(safeDiv(denominator.broadcastTo(yGradShape)));
        if (!targetShape.equals(yGradShape)) {
            grad = BroadcastUtils.sumToShape(grad, targetShape, yGradShape);
        }
        return grad;
    }

    /** 计算 ∂(x/y)/∂y = -x/y² 的梯度 */
    private NdArray computeDenominatorGrad(NdArray yGrad, NdArray numerator, NdArray denominator, Shape yGradShape) {
        Shape denominatorShape = denominator.getShape();
        NdArray safeY2 = safeDiv(denominator.square());
        NdArray negNumerator = numerator.neg();

        NdArray negNumBroadcast = numerator.getShape().equals(yGradShape) ? negNumerator : negNumerator.broadcastTo(yGradShape);
        NdArray y2Broadcast = denominatorShape.equals(yGradShape) ? safeY2 : safeY2.broadcastTo(yGradShape);
        NdArray grad = yGrad.mul(negNumBroadcast.div(y2Broadcast));

        if (!denominatorShape.equals(yGradShape)) {
            grad = BroadcastUtils.sumToShape(grad, denominatorShape, yGradShape);
        }
        return grad;
    }
    


    /**
     * 对除数添加除零保护
     * <p>
     * 将绝对值小于 EPSILON 的元素替换为 EPSILON（保持符号），
     * 防止除零导致 Infinity 或 NaN。
     * 实现方式：abs(x) < eps 的位置用 eps 替换，其余保持不变。
     *
     * @param divisor 除数
     * @return 安全的除数
     */
    private NdArray safeDiv(NdArray divisor) {
        // mask: abs(divisor) >= EPSILON 的位置为1，否则为0
        NdArray absDivisor = divisor.abs();
        NdArray mask = absDivisor.mask(EPSILON);  // abs >= EPSILON 的位置为1
        // safeDivisor = divisor * mask + EPSILON * (1 - mask)
        NdArray epsilonFill = NdArray.like(divisor.getShape(), EPSILON);
        NdArray inverseMask = mask.like(1f).sub(mask);
        return divisor.mul(mask).add(epsilonFill.mul(inverseMask));
    }

    /**
     * 获取所需输入参数个数
     * <p>
     * 除法运算需要两个输入参数。
     *
     * @return 输入参数个数，固定为2
     */
    @Override
    public int requireInputNum() {
        return 2;
    }
}