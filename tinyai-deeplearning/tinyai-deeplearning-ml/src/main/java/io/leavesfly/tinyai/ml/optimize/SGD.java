package io.leavesfly.tinyai.ml.optimize;

import io.leavesfly.tinyai.ml.model.Model;
import io.leavesfly.tinyai.nnet.core.Parameter;

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
        // 检查参数的梯度是否为null，如果为null则跳过更新
        if (parameter.getGrad() == null) {
            return;
        }
        parameter.setValue(parameter.getValue().sub(parameter.getGrad().mulNum(lr)));
    }
    
    /**
     * 设置学习率（用于学习率调度器）
     * 
     * @param learningRate 新的学习率
     */
    public void setLearningRate(float learningRate) {
        this.lr = learningRate;
    }
    
    /**
     * 获取当前学习率
     * 
     * @return 当前学习率
     */
    public float getLearningRate() {
        return lr;
    }
}