package io.leavesfly.tinyai.func.matrix;


import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.Collections;
import java.util.List;

/**
 * 广播函数
 * <p>
 * 将输入数组广播到指定形状。
 */
public class BroadcastTo extends Function {

    private Shape shape;
    private Shape inputShape;

    /**
     * 构造函数
     *
     * @param _shape 目标形状
     */
    public BroadcastTo(Shape _shape) {
        this.shape = _shape;
    }

    /**
     * 前向传播计算广播
     * <p>
     * 将输入数组广播到指定形状。
     *
     * @param inputs 输入的NdArray数组，长度为1
     * @return 广播后的NdArray
     */
    @Override
    public NdArray forward(NdArray... inputs) {
        inputShape = inputs[0].getShape();
        return inputs[0].broadcastTo(shape);

    }

    /**
     * 反向传播计算梯度
     * <p>
     * 对于广播操作，梯度计算通过sumTo操作还原到原始形状。
     *
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        return Collections.singletonList(yGrad.sumTo(inputShape));
    }

    /**
     * 获取所需输入参数个数
     * <p>
     * 广播函数需要一个输入参数。
     *
     * @return 输入参数个数，固定为1
     */
    @Override
    public int requireInputNum() {
        return 1;
    }
}
