package io.leavesfly.tinyai.func.matrix;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.Arrays;
import java.util.List;

/**
 * 条件选择算子
 * <p>
 * where(condition, x, y): condition为true选x，否则选y
 * 支持广播
 */
public class Where extends Function {

    private NdArray condition;

    @Override
    public NdArray forward(NdArray... inputs) {
        NdArray cond = inputs[0];
        NdArray x = inputs[1];
        NdArray y = inputs[2];

        condition = cond;

        // 验证形状兼容性（支持广播）
        Shape condShape = cond.getShape();
        Shape xShape = x.getShape();
        Shape yShape = y.getShape();

        // 计算输出形状（广播后的形状）
        Shape outputShape = computeBroadcastShape(condShape, xShape, yShape);

        // 执行条件选择
        return selectElements(cond, x, y, outputShape);
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // 梯度计算：
        // dX = dY * condition (广播后)
        // dY = dY * (1 - condition) (广播后)
        // dCondition = 0 (条件不可导)

        NdArray x = inputs[1].getValue();
        NdArray y = inputs[2].getValue();
        Shape xShape = x.getShape();
        Shape yShape = y.getShape();

        // 计算x和y的梯度
        NdArray ones = NdArray.ones(condition.getShape());
        NdArray invCond = ones.sub(condition); // 1 - condition

        NdArray gradX = condition.mul(yGrad.broadcastTo(xShape));
        NdArray gradY = invCond.mul(yGrad.broadcastTo(yShape));

        return Arrays.asList(
            null, // condition不可导
            gradX,
            gradY
        );
    }

    /**
     * 执行元素级条件选择
     * 简化实现：假设形状相同或可广播
     */
    private NdArray selectElements(NdArray cond, NdArray x, NdArray y, Shape outputShape) {
        // 简化实现：使用maskedFill的方式
        // result = cond * x + (1 - cond) * y
        NdArray ones = NdArray.ones(cond.getShape());
        NdArray invCond = ones.sub(cond); // 1 - cond
        
        // 广播x和y到输出形状
        NdArray xBroadcast = x.getShape().equals(outputShape) ? x : x.broadcastTo(outputShape);
        NdArray yBroadcast = y.getShape().equals(outputShape) ? y : y.broadcastTo(outputShape);
        
        // 广播cond到输出形状
        NdArray condBroadcast = cond.getShape().equals(outputShape) ? cond : cond.broadcastTo(outputShape);
        NdArray invCondBroadcast = ones.sub(condBroadcast);
        
        NdArray xPart = condBroadcast.mul(xBroadcast);
        NdArray yPart = invCondBroadcast.mul(yBroadcast);
        
        return xPart.add(yPart);
    }

    /**
     * 计算广播后的形状
     */
    private Shape computeBroadcastShape(Shape... shapes) {
        int maxDims = 0;
        for (Shape shape : shapes) {
            maxDims = Math.max(maxDims, shape.getDimNum());
        }

        int[] broadcastDims = new int[maxDims];
        for (int i = 0; i < maxDims; i++) {
            broadcastDims[i] = 1;
            for (Shape shape : shapes) {
                int[] dims = shape.getShapeDims();
                int dimIdx = dims.length - maxDims + i;
                if (dimIdx >= 0) {
                    broadcastDims[i] = Math.max(broadcastDims[i], dims[dimIdx]);
                }
            }
        }

        return Shape.of(broadcastDims);
    }

    @Override
    public int requireInputNum() {
        return 3;
    }
}

