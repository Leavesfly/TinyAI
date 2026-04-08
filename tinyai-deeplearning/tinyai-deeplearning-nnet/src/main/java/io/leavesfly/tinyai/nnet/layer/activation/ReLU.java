package io.leavesfly.tinyai.nnet.layer.activation;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.core.Module;

/**
 * V2版本的ReLU激活函数层
 * <p>
 * ReLU(x) = max(0, x)
 * <p>
 * 特性：
 * - 无参数层
 * - 在训练和推理模式下行为一致
 *
 * @author leavesfly
 * @version 2.0
 */
public class ReLU extends Module {

    /**
     * 构造函数
     *
     * @param name 层名称
     */
    public ReLU(String name) {
        super(name);
    }

    /**
     * 默认构造函数
     */
    public ReLU() {
        super("relu");
    }

    @Override
    public Variable forward(Variable... inputs) {
        Variable x = inputs[0];
        // 使用Variable的relu方法（依赖已有的ReLU Function）
        return x.relu();
    }

    @Override
    public String toString() {
        return "ReLU{name='" + name + "'}";
    }
}
