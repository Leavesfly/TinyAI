package io.leavesfly.tinyai.func.base;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.Arrays;
import java.util.List;


/**
 * 乘法函数
 * <p>
 * 实现两个变量的乘法运算。
 */
public class Mul extends Function {

    /**
     * 前向传播计算乘法
     * <p>
     * 执行两个NdArray的乘法运算，支持广播机制：inputs[0] * inputs[1]
     *
     * @param inputs 输入的NdArray数组，长度为2
     * @return 乘法运算结果的NdArray
     */
    @Override
    public NdArray forward(NdArray... inputs) {
        NdArray input0 = inputs[0];
        NdArray input1 = inputs[1];
        
        // 检查是否需要广播
        if (input0.getShape().equals(input1.getShape())) {
            // 形状相同，直接相乘
            return input0.mul(input1);
        } else {
            // 需要广播
            Shape shape0 = input0.getShape();
            Shape shape1 = input1.getShape();
            
            // 判断广播方向
            if (isScalarOrBroadcastable(shape1, shape0)) {
                // input1 需要广播到 input0 的形状
                return input0.mul(input1.broadcastTo(shape0));
            } else if (isScalarOrBroadcastable(shape0, shape1)) {
                // input0 需要广播到 input1 的形状
                return input0.broadcastTo(shape1).mul(input1);
            } else {
                throw new IllegalArgumentException(
                    String.format("乘法操作的形状不兼容：%s vs %s", shape0, shape1)
                );
            }
        }
    }
    
    /**
     * 判断一个形状是否可以广播到另一个形状
     * @param smallShape 小的形状
     * @param largeShape 大的形状
     * @return 是否可以广播
     */
    private boolean isScalarOrBroadcastable(Shape smallShape, Shape largeShape) {
        // 简单的广播判断：支持标量广播
        if (smallShape.size() == 1) {
            return true;
        }
        
        // 支持多维数组的广播判断
        // 从后往前检查维度是否兼容
        if (smallShape.getDimNum() <= largeShape.getDimNum()) {
            boolean compatible = true;
            for (int i = 0; i < smallShape.getDimNum(); i++) {
                int srcDimIndex = smallShape.getDimNum() - 1 - i;
                int dstDimIndex = largeShape.getDimNum() - 1 - i;
                
                int srcDim = smallShape.getDimension(srcDimIndex);
                int dstDim = largeShape.getDimension(dstDimIndex);
                
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
     * <p>
     * 计算乘法运算的梯度。
     * 对于 z = x * y，有：
     * - ∂z/∂x = y
     * - ∂z/∂y = x
     *
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        NdArray ndArray0 = inputs[0].getValue();
        NdArray ndArray1 = inputs[1].getValue();

        return Arrays.asList(yGrad.mul(ndArray1), yGrad.mul(ndArray0));
    }

    /**
     * 获取所需输入参数个数
     * <p>
     * 乘法运算需要两个输入参数。
     *
     * @return 输入参数个数，固定为2
     */
    @Override
    public int requireInputNum() {
        return 2;
    }
}
