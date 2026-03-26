package io.leavesfly.tinyai.func.math;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.Arrays;
import java.util.List;

/**
 * 相等比较函数
 * <p>
 * 逐元素比较两个张量是否相等，返回 0/1 掩码。
 * 比较操作不可导，反向传播时梯度为零。
 */
public class Equal extends Function {

    /**
     * 前向传播：逐元素相等比较
     *
     * @param inputs 输入的NdArray数组，长度为2
     * @return 比较结果的 0/1 掩码
     */
    @Override
    public NdArray forward(NdArray... inputs) {
        return inputs[0].eq(inputs[1]);
    }

    /**
     * 反向传播：比较操作不可导，梯度为零
     *
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表（全零）
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        NdArray grad0 = NdArray.zeros(inputs[0].getValue().getShape());
        NdArray grad1 = NdArray.zeros(inputs[1].getValue().getShape());
        return Arrays.asList(grad0, grad1);
    }

    @Override
    public int requireInputNum() {
        return 2;
    }
}

