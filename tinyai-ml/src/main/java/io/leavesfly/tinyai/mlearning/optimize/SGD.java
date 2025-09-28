package io.leavesfly.tinyai.mlearning.optimize;

import io.leavesfly.tinyai.mlearning.Model;
import io.leavesfly.tinyai.nnet.Parameter;

/**
 * 随机梯度下降优化器
 * 
 * 实现了经典的随机梯度下降算法，用于更新模型参数。
 * 更新公式：θ = θ - lr * ∇J(θ)
 * 
 * @author TinyDL
 * @version 1.0
 */
public class SGD extends Optimizer {

    private float lr;

    /**
     * 构造函数
     * @param target 目标模型
     * @param learnRate 学习率
     */
    public SGD(Model target, float learnRate) {
        super(target);
        lr = learnRate;
    }

    @Override
    public void updateOne(Parameter parameter) {
        parameter.setValue(parameter.getValue().sub(parameter.getGrad().mulNum(lr)));
    }
}