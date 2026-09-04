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
     * <p>
     * 形状处理说明：{@code NdArray.sum()} 的输出恒为二维 {@code [1,1]}，而
     * {@code broadcastTo} 要求源形状的维度数不得超过目标形状。当输入是一维
     * （如损失归约中常见的逐样本向量 {@code [batch]}）时，{@code [1,1] -> [batch]}
     * 会直接抛出"源形状维度不能大于目标形状维度"。
     * <p>
     * 因此先把梯度重塑成与输入同秩的全 1 形状再广播。求和的输出只有一个元素，
     * 回传梯度也只含一个元素，重塑总是合法；对原本就能工作的二维/三维输入，
     * 重塑前后形状一致，行为不变。
     *
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        int[] inputDims = inputShape.getShapeDims();
        int[] onesShape = new int[inputDims.length];
        java.util.Arrays.fill(onesShape, 1);

        NdArray grad = yGrad;
        if (grad.getShape().getShapeDims().length != onesShape.length) {
            grad = grad.reshape(Shape.of(onesShape));
        }

        return Collections.singletonList(grad.broadcastTo(inputShape));
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
