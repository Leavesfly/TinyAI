package io.leavesfly.tinyai.func.matrix;


import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.Collections;
import java.util.List;

/**
 * SoftMax函数
 * <p>
 * SoftMax激活函数，用于神经网络中，将输入值映射到概率分布。
 */
public class SoftMax extends Function {

    /**
     * 前向传播计算SoftMax
     * <p>
     * 计算SoftMax函数值：softmax(x_i) = e^(x_i) / Σ(e^(x_j))
     *
     * @param inputs 输入的NdArray数组，长度为1
     * @return SoftMax函数值的NdArray
     */
    @Override
    public NdArray forward(NdArray... inputs) {
        return inputs[0].softMax();
    }

    /**
     * 反向传播计算梯度
     * <p>
     * 对于SoftMax函数，梯度计算公式为：
     * ∂softmax(x_i)/∂x_j = softmax(x_i) * (δ_ij - softmax(x_j))
     *
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {

        NdArray y = getOutput().getValue();
        NdArray gx = y.mul(yGrad);
        NdArray sumDx = gx.sumTo(Shape.of(gx.getShape().getRow(), 1)).broadcastTo(gx.getShape());
        gx = gx.sub(y.mul(sumDx));
        return Collections.singletonList(gx);
    }

    /**
     * 获取所需输入参数个数
     * <p>
     * SoftMax函数需要一个输入参数。
     *
     * @return 输入参数个数，固定为1
     */
    @Override
    public int requireInputNum() {
        return 1;
    }
}
