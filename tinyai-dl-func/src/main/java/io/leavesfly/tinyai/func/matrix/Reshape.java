package io.leavesfly.tinyai.func.matrix;


import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.Collections;
import java.util.List;

/**
 * 矩阵重塑函数
 * <p>
 * 将输入数组重塑为指定形状。
 */
public class Reshape extends Function {

    private Shape shape;
    private Shape inputShape;

    /**
     * 构造函数
     *
     * @param _shape 目标形状
     */
    public Reshape(Shape _shape) {
        this.shape = _shape;
    }

    /**
     * 前向传播计算重塑
     * <p>
     * 将输入数组重塑为指定形状。
     *
     * @param inputs 输入的NdArray数组，长度为1
     * @return 重塑后的NdArray
     */
    @Override
    public NdArray forward(NdArray... inputs) {
        inputShape = inputs[0].getShape();
        return inputs[0].reshape(shape);

    }

    /**
     * 反向传播计算梯度
     * <p>
     * 对于重塑操作，梯度计算通过reshape操作还原到原始形状。
     *
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        return Collections.singletonList(yGrad.reshape(inputShape));
    }

    /**
     * 获取所需输入参数个数
     * <p>
     * 矩阵重塑函数需要一个输入参数。
     *
     * @return 输入参数个数，固定为1
     */
    @Override
    public int requireInputNum() {
        return 1;
    }
}
