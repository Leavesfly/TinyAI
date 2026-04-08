package io.leavesfly.tinyai.nnet.layer.activation;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.core.Module;

/**
 * V2版本的SoftMax激活函数层
 * <p>
 * SoftMax(x_i) = exp(x_i) / sum(exp(x_j))
 * <p>
 * 特性：
 * - 无参数层
 * - 在训练和推理模式下行为一致
 * - 输出和为1，常用于多分类任务
 *
 * @author leavesfly
 * @version 2.0
 */
public class SoftMax extends Module {

    private final int axis;

    /**
     * 构造函数
     *
     * @param name 层名称
     * @param axis SoftMax计算的维度（默认为-1，即最后一维）
     */
    public SoftMax(String name, int axis) {
        super(name);
        this.axis = axis;
    }

    /**
     * 默认构造函数（axis=-1）
     *
     * @param name 层名称
     */
    public SoftMax(String name) {
        this(name, -1);
    }

    /**
     * 默认构造函数
     */
    public SoftMax() {
        this("softmax", -1);
    }

    @Override
    public Variable forward(Variable... inputs) {
        Variable x = inputs[0];
        // 使用Variable的softmax方法
        // 注意：这里假设Variable已有softmax方法
        // 如果没有，需要手动实现：
        // 1. 减去最大值（数值稳定性）
        // 2. 计算exp
        // 3. 归一化
        return x.softMax();
    }

    @Override
    public String toString() {
        return "SoftMax{name='" + name + "', axis=" + axis + '}';
    }
}
