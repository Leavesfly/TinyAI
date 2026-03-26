package io.leavesfly.tinyai.func.math;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.Collections;
import java.util.List;

/**
 * SiLU激活函数（Sigmoid Linear Unit，又称Swish）
 * <p>
 * SiLU是一种自门控激活函数，由Google在2017年提出。
 * 它结合了线性和非线性的特性，在许多深度学习任务中表现优异。
 * <p>
 * 公式：SiLU(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))
 * <p>
 * 特性：
 * - 平滑且非单调
 * - 具有自门控特性
 * - 梯度比ReLU更稳定
 * - 广泛应用于EfficientNet、YOLOv5等模型
 *
 * @author leavesfly
 * @version 1.0
 */
public class SiLU extends Function {

    /**
     * 缓存前向传播的 sigmoid 结果，避免反向传播时重复计算
     */
    private NdArray cachedSigmoid;

    /**
     * 前向传播计算SiLU
     * <p>
     * 计算公式：SiLU(x) = x * sigmoid(x)
     *
     * @param inputs 输入的NdArray数组，长度为1
     * @return SiLU函数值的NdArray
     */
    @Override
    public NdArray forward(NdArray... inputs) {
        NdArray x = inputs[0];

        // sigmoid(x) = 1 / (1 + exp(-x))，缓存供 backward 使用
        cachedSigmoid = x.sigmoid();

        // SiLU(x) = x * sigmoid(x)
        return x.mul(cachedSigmoid);
    }

    /**
     * 反向传播计算梯度
     * <p>
     * SiLU'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
     *          = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
     * 复用前向传播缓存的 sigmoid 结果。
     *
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        NdArray x = inputs[0].getValue();

        // 复用缓存的 sigmoid 结果
        // 1 - sigmoid(x)
        NdArray oneMinusSigmoid = cachedSigmoid.like(1f).sub(cachedSigmoid);

        // 1 + x * (1 - sigmoid(x))
        NdArray onePlusXTimes = cachedSigmoid.like(1f).add(x.mul(oneMinusSigmoid));

        // sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        NdArray grad = cachedSigmoid.mul(onePlusXTimes);

        return Collections.singletonList(yGrad.mul(grad));
    }

    /**
     * 获取所需输入参数个数
     *
     * @return 输入参数个数，固定为1
     */
    @Override
    public int requireInputNum() {
        return 1;
    }
}

