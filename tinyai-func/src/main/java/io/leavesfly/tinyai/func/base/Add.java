package io.leavesfly.tinyai.func.base;


import io.leavesfly.tinyai.func.Function;
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

    private Shape x0Shape;
    private Shape x1Shape;

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
        x0Shape = inputs[0].getShape();
        x1Shape = inputs[1].getShape();
        
        // 检查是否需要广播
        if (x0Shape.equals(x1Shape)) {
            // 形状相同，直接相加
            return inputs[0].add(inputs[1]);
        } else {
            // 需要广播
            // 判断广播方向
            if (isBroadcastable(x1Shape, x0Shape)) {
                // input1 需要广播到 input0 的形状
                return inputs[0].add(inputs[1].broadcastTo(x0Shape));
            } else if (isBroadcastable(x0Shape, x1Shape)) {
                // input0 需要广播到 input1 的形状
                return inputs[0].broadcastTo(x1Shape).add(inputs[1]);
            } else {
                throw new IllegalArgumentException(
                    String.format("加法操作的形状不兼容：%s vs %s", x0Shape, x1Shape)
                );
            }
        }
    }
    
    /**
     * 判断一个形状是否可以广播到另一个形状
     * @param srcShape 源形状
     * @param dstShape 目标形状
     * @return 是否可以广播
     */
    private boolean isBroadcastable(Shape srcShape, Shape dstShape) {
        // 支持多维数组的广播判断
        // 从后往前检查维度是否兼容
        if (srcShape.getDimNum() <= dstShape.getDimNum()) {
            boolean compatible = true;
            for (int i = 0; i < srcShape.getDimNum(); i++) {
                int srcDimIndex = srcShape.getDimNum() - 1 - i;
                int dstDimIndex = dstShape.getDimNum() - 1 - i;
                
                int srcDim = srcShape.getDimension(srcDimIndex);
                int dstDim = dstShape.getDimension(dstDimIndex);
                
                // 广播规则：维度相等，或者源维度为1
                if (srcDim != dstDim && srcDim != 1) {
                    compatible = false;
                    break;
                }
            }
            if (compatible) {
                return true;
            }
        }
        
        return false;
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
        NdArray gx0 = yGrad;
        NdArray gx1 = x1Shape.equals(x0Shape) ? yGrad : yGrad.sumTo(x1Shape);
        return Arrays.asList(gx0, gx1);
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