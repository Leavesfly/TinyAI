package io.leavesfly.tinyai.ml.loss;

import io.leavesfly.tinyai.func.Variable;

/**
 * 损失函数抽象类
 * <p>
 * 该类是所有损失函数实现的基类，定义了计算损失值的基本接口。
 * 子类需要实现具体的损失计算逻辑。
 *
 * @author TinyDL
 * @version 1.0
 */
public abstract class Loss {
    /**
     * 计算损失值
     *
     * @param y       真实标签
     * @param predict 预测值
     * @return 损失值变量
     */
    public abstract Variable loss(Variable y, Variable predict);
}