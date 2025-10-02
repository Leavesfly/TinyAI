package io.leavesfly.tinyai.func.matrix;


import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.Collections;
import java.util.List;

/**
 * 求和函数
 * <p>
 * 计算输入数组所有元素的和。
 */
public class Sum extends Function {

    private Shape inputShape;

    /**
     * 前向传播计算求和
     * <p>
     * 计算输入数组所有元素的和。
     *
     * @param inputs 输入的NdArray数组，长度为1
     * @return 求和结果的NdArray
     */
    @Override
    public NdArray forward(NdArray... inputs) {
        inputShape = inputs[0].getShape();
        return inputs[0].sum();
    }

    /**
     * 反向传播计算梯度
     * <p>
     * 对于求和操作，梯度计算通过广播操作将梯度值传播到所有元素。
     *
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {

        return Collections.singletonList(yGrad.broadcastTo(inputShape));
    }

    /**
     * 获取所需输入参数个数
     * <p>
     * 求和函数需要一个输入参数。
     *
     * @return 输入参数个数，固定为1
     */
    @Override
    public int requireInputNum() {
        return 1;
    }
}
