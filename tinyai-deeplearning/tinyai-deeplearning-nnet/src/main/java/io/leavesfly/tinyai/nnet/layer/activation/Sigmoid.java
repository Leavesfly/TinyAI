package io.leavesfly.tinyai.nnet.layer.activation;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.core.Module;

/**
 * V2版本的Sigmoid激活函数层
 * <p>
 * Sigmoid(x) = 1 / (1 + exp(-x))
 * <p>
 * 特性：
 * - 无参数层
 * - 在训练和推理模式下行为一致
 * - 输出范围：(0, 1)
 *
 * @author leavesfly
 * @version 2.0
 */
public class Sigmoid extends Module {

    /**
     * 构造函数
     *
     * @param name 层名称
     */
    public Sigmoid(String name) {
        super(name);
    }

    /**
     * 默认构造函数
     */
    public Sigmoid() {
        super("sigmoid");
    }

    @Override
    public Variable forward(Variable... inputs) {
        Variable x = inputs[0];
        // 使用Variable的sigmoid方法
        return x.sigmoid();
    }

    @Override
    public String toString() {
        return "Sigmoid{name='" + name + "'}";
    }
}
