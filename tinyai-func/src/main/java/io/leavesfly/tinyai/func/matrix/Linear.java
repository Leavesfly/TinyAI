package io.leavesfly.tinyai.func.matrix;


import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.Arrays;
import java.util.List;

/**
 * 线性变换函数
 * <p>
 * 执行线性变换操作：y = x * w + b，其中b为可选偏置项。
 */
public class Linear extends Function {

    /**
     * 前向传播计算线性变换
     * <p>
     * 执行线性变换操作：y = x * w + b，其中b为可选偏置项。
     *
     * @param inputs 输入的NdArray数组，长度为2或3（x, w, [b]）
     * @return 线性变换后的NdArray
     */
    @Override
    public NdArray forward(NdArray... inputs) {

        NdArray y = inputs[0].dot(inputs[1]);
        if (inputs.length == 2) {
            return y;
        }
        return y.add(inputs[2].broadcastTo(y.getShape()));
    }

    /**
     * 反向传播计算梯度
     * <p>
     * 对于线性变换，梯度计算公式为：
     * - ∂y/∂x = yGrad * w^T
     * - ∂y/∂w = x^T * yGrad
     * - ∂y/∂b = sum(yGrad)
     *
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {

        NdArray x = inputs[0].getValue();
        NdArray w = inputs[1].getValue();

        if (inputs.length == 2) {
            return Arrays.asList(yGrad.dot(w.transpose()), x.transpose().dot(yGrad));
        } else {
            NdArray b = inputs[2].getValue();
            return Arrays.asList(yGrad.dot(w.transpose()), x.transpose().dot(yGrad), yGrad.sumTo(b.getShape()));
        }
    }

    /**
     * 获取所需输入参数个数
     * <p>
     * 线性变换函数可以接受2个或3个输入参数（x, w, [b]）。
     *
     * @return 输入参数个数，-1表示可变参数
     */
    @Override
    public int requireInputNum() {
        return -1;
    }
}
