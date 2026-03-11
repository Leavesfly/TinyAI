package io.leavesfly.tinyai.ml.scheduler;

import io.leavesfly.tinyai.ml.optimize.Optimizer;
import io.leavesfly.tinyai.ml.visual.ValidationUtils;

/**
 * 阶梯式学习率调度器，每隔 stepSize 个 epoch，学习率乘以 gamma。
 *
 * @author TinyAI
 * @version 1.0
 */
public class StepLR implements LearningRateScheduler {
    
    private final float initialLR;
    private final int stepSize;
    private final float gamma;
    private float currentLR;
    
    /**
     * 构造函数
     * 
     * @param initialLR 初始学习率
     * @param stepSize 每隔多少个epoch降低一次学习率
     * @param gamma 学习率衰减因子
     */
    public StepLR(float initialLR, int stepSize, float gamma) {
        ValidationUtils.requirePositive(initialLR, "initialLR");
        ValidationUtils.requirePositive(stepSize, "stepSize");
        ValidationUtils.requirePositive(gamma, "gamma");
        
        this.initialLR = initialLR;
        this.stepSize = stepSize;
        this.gamma = gamma;
        this.currentLR = initialLR;
    }
    
    @Override
    public float getLearningRate(int epoch, int step) {
        // 计算应该降低多少次
        int steps = epoch / stepSize;
        currentLR = initialLR * (float) Math.pow(gamma, steps);
        return currentLR;
    }
    
    @Override
    public void update(Optimizer optimizer) {
        // 如果优化器支持动态设置学习率，则更新
        if (optimizer instanceof io.leavesfly.tinyai.ml.optimize.SGD) {
            ((io.leavesfly.tinyai.ml.optimize.SGD) optimizer).setLearningRate(currentLR);
        } else if (optimizer instanceof io.leavesfly.tinyai.ml.optimize.Adam) {
            ((io.leavesfly.tinyai.ml.optimize.Adam) optimizer).setLearningRate(currentLR);
        }
    }
    
    @Override
    public void reset() {
        this.currentLR = initialLR;
    }
    
    public float getCurrentLR() {
        return currentLR;
    }
}

