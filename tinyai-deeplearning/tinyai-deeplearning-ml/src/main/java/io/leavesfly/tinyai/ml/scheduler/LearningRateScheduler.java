package io.leavesfly.tinyai.ml.scheduler;

import io.leavesfly.tinyai.ml.optimize.Optimizer;

/**
 * 学习率调度器接口，用于在训练过程中动态调整学习率。
 *
 * @author TinyAI
 * @version 1.0
 */
public interface LearningRateScheduler {
    
    /**
     * 获取当前学习率
     * 
     * @param epoch 当前训练轮次
     * @param step 当前步数（批次索引）
     * @return 当前学习率
     */
    float getLearningRate(int epoch, int step);
    
    /**
     * 更新优化器的学习率
     * 
     * @param optimizer 优化器
     */
    void update(Optimizer optimizer);
    
    /**
     * 重置调度器状态
     */
    void reset();
}

