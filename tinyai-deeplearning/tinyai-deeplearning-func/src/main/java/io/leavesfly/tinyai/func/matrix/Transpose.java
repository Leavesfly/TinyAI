package io.leavesfly.tinyai.func.matrix;



import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.func.util.BroadcastUtils;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.Collections;
import java.util.List;

/**
 * 矩阵转置函数
 * 
 * 计算输入数组的转置。
 */
public class Transpose extends Function {
    
    private Shape inputShape;
    
    /**
     * 前向传播计算转置
     * 
     * 计算输入数组的转置。
     * 
     * @param inputs 输入的NdArray数组，长度为1
     * @return 转置后的NdArray
     */
    @Override
    public NdArray forward(NdArray... inputs) {
        inputShape = inputs[0].getShape();
        return inputs[0].transpose();
    }

    /**
     * 反向传播计算梯度
     * 
     * 对于转置操作，梯度计算通过转置操作将梯度值传播到原始形状。
     * 支持 yGrad 维度比输入高的情况（如 batch 广播）。
     * 
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        NdArray grad = yGrad.transpose();
        // 如果梯度形状与输入形状不同，需要 sumTo 回原始形状
        if (!grad.getShape().equals(inputShape)) {
            grad = BroadcastUtils.sumToShape(grad, inputShape);
        }
        return Collections.singletonList(grad);
    }

    /**
     * 获取所需输入参数个数
     * 
     * 矩阵转置函数需要一个输入参数。
     * 
     * @return 输入参数个数，固定为1
     */
    @Override
    public int requireInputNum() {
        return 1;
    }
}
