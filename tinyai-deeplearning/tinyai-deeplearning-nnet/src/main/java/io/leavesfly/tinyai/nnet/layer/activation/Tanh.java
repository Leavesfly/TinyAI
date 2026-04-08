package io.leavesfly.tinyai.nnet.layer.activation;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.core.Module;

/**
 * V2版本的Tanh激活函数层
 * <p>
 * Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
 * <p>
 * 特性：
 * - 无参数层
 * - 在训练和推理模式下行为一致
 * - 输出范围：(-1, 1)
 *
 * @author leavesfly
 * @version 2.0
 */
public class Tanh extends Module {

    /**
     * 构造函数
     *
     * @param name 层名称
     */
    public Tanh(String name) {
        super(name);
    }

    /**
     * 默认构造函数
     */
    public Tanh() {
        super("tanh");
    }

    @Override
    public Variable forward(Variable... inputs) {
        Variable x = inputs[0];
        // 使用Variable的tanh方法
        return x.tanh();
    }

    @Override
    public String toString() {
        return "Tanh{name='" + name + "'}";
    }
}
