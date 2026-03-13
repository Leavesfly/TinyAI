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
            Shape shape0 = input0.getShape();
            Shape shape1 = input1.getShape();
            
            // 标量优化：当其中一个操作数是标量（size==1）时，
            // 直接用 mulNum 避免 broadcastTo 的维度限制
            if (shape1.size() == 1) {
                return input0.mulNum(input1.getNumber().floatValue());
            }
            if (shape0.size() == 1) {
                return input1.mulNum(input0.getNumber().floatValue());
            }
            
            // 非标量的广播
            if (isScalarOrBroadcastable(shape1, shape0)) {
                return input0.mul(input1.broadcastTo(shape0));
            } else if (isScalarOrBroadcastable(shape0, shape1)) {
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
     * 计算乘法运算的梯度，支持广播情况。
     * 对于 z = x * y，有：
     * - ∂z/∂x = y
     * - ∂z/∂y = x
     * 当存在广播时，需要将梯度 sumTo 回原始形状。
     *
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        NdArray ndArray0 = inputs[0].getValue();
        NdArray ndArray1 = inputs[1].getValue();
        
        Shape shape0 = ndArray0.getShape();
        Shape shape1 = ndArray1.getShape();
        Shape yGradShape = yGrad.getShape();
        
        // 计算 dx = yGrad * y
        NdArray grad0;
        if (shape1.size() == 1) {
            grad0 = yGrad.mulNum(ndArray1.getNumber().floatValue());
        } else if (shape1.equals(yGradShape)) {
            grad0 = yGrad.mul(ndArray1);
        } else {
            grad0 = yGrad.mul(ndArray1.broadcastTo(yGradShape));
        }
        if (!shape0.equals(yGradShape)) {
            grad0 = sumToShape(grad0, shape0, yGradShape);
        }
        
        // 计算 dy = yGrad * x
        NdArray grad1;
        if (shape0.size() == 1) {
            grad1 = yGrad.mulNum(ndArray0.getNumber().floatValue());
        } else if (shape0.equals(yGradShape)) {
            grad1 = yGrad.mul(ndArray0);
        } else {
            grad1 = yGrad.mul(ndArray0.broadcastTo(yGradShape));
        }
        if (!shape1.equals(yGradShape)) {
            grad1 = sumToShape(grad1, shape1, yGradShape);
        }
        
        return Arrays.asList(grad0, grad1);
    }
    
    /**
     * 将梯度 sumTo 回目标形状，支持维度数不同的情况
     */
    private NdArray sumToShape(NdArray grad, Shape targetShape, Shape gradShape) {
        int targetNdim = targetShape.getDimNum();
        int gradNdim = gradShape.getDimNum();
        
        // 目标是标量（size==1），直接对所有元素求和
        if (targetShape.size() == 1) {
            return NdArray.of(grad.sum().getNumber().floatValue());
        }
        
        if (targetNdim < gradNdim) {
            // 目标维度数较小，需要先扩展目标形状
            int[] targetDims = targetShape.getShapeDims();
            int[] expandedDims = new int[gradNdim];
            int offset = gradNdim - targetNdim;
            // 前面补1
            for (int i = 0; i < offset; i++) {
                expandedDims[i] = 1;
            }
            // 后面复制原始维度
            for (int i = 0; i < targetNdim; i++) {
                expandedDims[offset + i] = targetDims[i];
            }
            Shape expandedShape = Shape.of(expandedDims);
            NdArray result = grad.sumTo(expandedShape);
            // 再 reshape 回原始形状
            return result.reshape(targetShape);
        } else {
            return grad.sumTo(targetShape);
        }
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
