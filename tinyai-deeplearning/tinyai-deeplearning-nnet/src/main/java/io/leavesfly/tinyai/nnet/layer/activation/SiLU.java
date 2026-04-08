package io.leavesfly.tinyai.nnet.layer.activation;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.core.Module;

/**
 * V2版本的SiLU激活函数层（Sigmoid Linear Unit，又称Swish）
 * <p>
 * SiLU是一种自门控激活函数，由Google在2017年提出。
 * 它结合了线性和非线性的特性，在许多深度学习任务中表现优异。
 * <p>
 * 公式：SiLU(x) = x * sigmoid(x)
 * <p>
 * 特性：
 * - 无参数层
 * - 平滑且非单调
 * - 具有自门控特性
 * - 梯度比ReLU更稳定
 * - 在训练和推理模式下行为一致
 * <p>
 * 应用场景：
 * - EfficientNet
 * - YOLOv5
 * - 各种现代CNN和Transformer架构
 *
 * @author leavesfly
 * @version 2.0
 */
public class SiLU extends Module {

    /**
     * 构造函数
     *
     * @param name 层名称
     */
    public SiLU(String name) {
        super(name);
    }

    /**
     * 默认构造函数
     */
    public SiLU() {
        super("silu");
    }

    @Override
    public Variable forward(Variable... inputs) {
        Variable x = inputs[0];
        // 使用Variable的silu方法
        return x.silu();
    }

    @Override
    public String toString() {
        return "SiLU{name='" + name + "'}";
    }
}

