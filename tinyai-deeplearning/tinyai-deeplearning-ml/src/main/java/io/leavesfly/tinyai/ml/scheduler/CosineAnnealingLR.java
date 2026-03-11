package io.leavesfly.tinyai.ml.scheduler;

import io.leavesfly.tinyai.ml.optimize.Optimizer;
import io.leavesfly.tinyai.ml.visual.ValidationUtils;

/**
 * 余弦退火学习率调度器，学习率按余弦函数从初始值衰减到最小值。
 *
 * @author TinyAI
 * @version 1.0
 */
public class CosineAnnealingLR implements LearningRateScheduler {
    
    private final float initialLR;
    private final float minLR;
    private final int TMax; // 周期长度
    private float currentLR;
    
    /**
     * 构造函数
     * 
     * @param initialLR 初始学习率
     * @param minLR 最小学习率
     * @param TMax 周期长度（epoch数）
     */
    public CosineAnnealingLR(float initialLR, float minLR, int TMax) {
        ValidationUtils.requirePositive(initialLR, "initialLR");
        ValidationUtils.requireNonNegative(minLR, "minLR");
        ValidationUtils.requirePositive(TMax, "TMax");
        
        if (minLR >= initialLR) {
            throw new IllegalArgumentException("minLR must be less than initialLR");
        }
        
        this.initialLR = initialLR;
        this.minLR = minLR;
        this.TMax = TMax;
        this.currentLR = initialLR;
    }
    
    @Override
    public float getLearningRate(int epoch, int step) {
        // 计算当前周期内的位置
        int epochInCycle = epoch % TMax;
        
        // 余弦退火公式
        currentLR = minLR + (initialLR - minLR) * 
                   (1 + (float) Math.cos(Math.PI * epochInCycle / TMax)) / 2.0f;
        
        return currentLR;
    }
    
    @Override
    public void update(Optimizer optimizer) {
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

