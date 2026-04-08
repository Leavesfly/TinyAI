package io.leavesfly.tinyai.nnet.layer.activation;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.core.Module;

/**
 * V2版本的LogSoftmax激活函数层
 * <p>
 * LogSoftmax是Softmax的对数形式，常用于NLLLoss组合。
 * 相比先Softmax再Log，直接使用LogSoftmax更数值稳定。
 * <p>
 * 公式：
 * LogSoftmax(x) = log(softmax(x)) = x - log(sum(exp(x)))
 * <p>
 * 使用log-sum-exp技巧保证数值稳定性：
 * LogSoftmax(x) = x - max(x) - log(sum(exp(x - max(x))))
 * <p>
 * 特性：
 * - 数值稳定，避免溢出
 * - 常与NLLLoss配合使用（等效于CrossEntropyLoss）
 * - 默认在最后一维进行计算
 * - 在训练和推理模式下行为一致
 *
 * @author leavesfly
 * @version 2.0
 */
public class LogSoftmax extends Module {

    /**
     * 计算轴
     * -1 表示最后一维
     */
    private final int axis;

    /**
     * 构造函数
     *
     * @param name 层名称
     * @param axis 计算轴
     */
    public LogSoftmax(String name, int axis) {
        super(name);
        this.axis = axis;
    }

    /**
     * 构造函数（默认axis = -1）
     *
     * @param name 层名称
     */
    public LogSoftmax(String name) {
        this(name, -1);
    }

    /**
     * 默认构造函数
     */
    public LogSoftmax() {
        this("log_softmax", -1);
    }

    /**
     * 构造函数（指定axis）
     *
     * @param axis 计算轴
     */
    public LogSoftmax(int axis) {
        this("log_softmax", axis);
    }

    @Override
    public Variable forward(Variable... inputs) {
        Variable x = inputs[0];
        return x.logSoftmax(axis);
    }

    /**
     * 获取计算轴
     *
     * @return 计算轴
     */
    public int getAxis() {
        return axis;
    }

    @Override
    public String toString() {
        return "LogSoftmax{" +
                "name='" + name + '\'' +
                ", axis=" + axis +
                '}';
    }
}

