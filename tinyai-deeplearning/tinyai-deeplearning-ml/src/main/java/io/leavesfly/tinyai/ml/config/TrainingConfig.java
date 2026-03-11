package io.leavesfly.tinyai.ml.config;

import io.leavesfly.tinyai.ml.visual.ValidationUtils;

/**
 * 训练配置类，统一管理训练相关的所有配置参数，提升易用性和可维护性。
 *
 * @author TinyAI
 * @version 1.0
 */
public class TrainingConfig {
    
    // 基础训练配置
    private int maxEpochs;
    private int batchSize;
    private float learningRate;
    private boolean shuffleData;
    
    // 并行训练配置
    private boolean enableParallelTraining;
    private int parallelThreadCount;
    
    // 检查点配置
    private String checkpointDir;
    private int checkpointInterval; // 每N个epoch保存一次检查点
    
    // 验证配置
    private boolean enableValidation;
    private int validationInterval; // 每N个epoch进行一次验证
    
    // 早停配置
    private boolean enableEarlyStopping;
    private int earlyStoppingPatience;
    private float earlyStoppingMinDelta;
    
    // 梯度裁剪配置
    private boolean enableGradientClipping;
    private float gradientClipValue;
    
    /**
     * 私有构造函数，使用Builder模式
     */
    private TrainingConfig() {
        // 设置默认值
        this.maxEpochs = 100;
        this.batchSize = 32;
        this.learningRate = 0.001f;
        this.shuffleData = true;
        this.enableParallelTraining = false;
        this.parallelThreadCount = 0; // 0表示自动计算
        this.checkpointDir = null;
        this.checkpointInterval = 0; // 0表示不保存检查点
        this.enableValidation = false;
        this.validationInterval = 1;
        this.enableEarlyStopping = false;
        this.earlyStoppingPatience = 10;
        this.earlyStoppingMinDelta = 0.0f;
        this.enableGradientClipping = false;
        this.gradientClipValue = 1.0f;
    }
    
    // =========== Getter方法 ===========
    
    public int getMaxEpochs() {
        return maxEpochs;
    }
    
    public int getBatchSize() {
        return batchSize;
    }
    
    public float getLearningRate() {
        return learningRate;
    }
    
    public boolean isShuffleData() {
        return shuffleData;
    }
    
    public boolean isEnableParallelTraining() {
        return enableParallelTraining;
    }
    
    public int getParallelThreadCount() {
        return parallelThreadCount;
    }
    
    public String getCheckpointDir() {
        return checkpointDir;
    }
    
    public int getCheckpointInterval() {
        return checkpointInterval;
    }
    
    public boolean isEnableValidation() {
        return enableValidation;
    }
    
    public int getValidationInterval() {
        return validationInterval;
    }
    
    public boolean isEnableEarlyStopping() {
        return enableEarlyStopping;
    }
    
    public int getEarlyStoppingPatience() {
        return earlyStoppingPatience;
    }
    
    public float getEarlyStoppingMinDelta() {
        return earlyStoppingMinDelta;
    }
    
    public boolean isEnableGradientClipping() {
        return enableGradientClipping;
    }
    
    public float getGradientClipValue() {
        return gradientClipValue;
    }
    
    // =========== Builder类 ===========
    
    /**
     * Builder类，用于构建TrainingConfig
     */
    public static class Builder {
        private final TrainingConfig config;
        
        public Builder() {
            this.config = new TrainingConfig();
        }
        
        public Builder maxEpochs(int maxEpochs) {
            ValidationUtils.requirePositive(maxEpochs, "maxEpochs");
            config.maxEpochs = maxEpochs;
            return this;
        }
        
        public Builder batchSize(int batchSize) {
            ValidationUtils.requirePositive(batchSize, "batchSize");
            config.batchSize = batchSize;
            return this;
        }
        
        public Builder learningRate(float learningRate) {
            ValidationUtils.requirePositive(learningRate, "learningRate");
            config.learningRate = learningRate;
            return this;
        }
        
        public Builder shuffleData(boolean shuffleData) {
            config.shuffleData = shuffleData;
            return this;
        }
        
        public Builder enableParallelTraining(boolean enable, int threadCount) {
            config.enableParallelTraining = enable;
            if (threadCount > 0) {
                ValidationUtils.requirePositive(threadCount, "threadCount");
                config.parallelThreadCount = threadCount;
            }
            return this;
        }
        
        public Builder checkpoint(String dir, int interval) {
            config.checkpointDir = dir;
            ValidationUtils.requireNonNegative(interval, "checkpointInterval");
            config.checkpointInterval = interval;
            return this;
        }
        
        public Builder validation(boolean enable, int interval) {
            config.enableValidation = enable;
            if (enable) {
                ValidationUtils.requirePositive(interval, "validationInterval");
                config.validationInterval = interval;
            }
            return this;
        }
        
        public Builder earlyStopping(boolean enable, int patience, float minDelta) {
            config.enableEarlyStopping = enable;
            if (enable) {
                ValidationUtils.requirePositive(patience, "patience");
                ValidationUtils.requireNonNegative(minDelta, "minDelta");
                config.earlyStoppingPatience = patience;
                config.earlyStoppingMinDelta = minDelta;
            }
            return this;
        }
        
        public Builder gradientClipping(boolean enable, float clipValue) {
            config.enableGradientClipping = enable;
            if (enable) {
                ValidationUtils.requirePositive(clipValue, "clipValue");
                config.gradientClipValue = clipValue;
            }
            return this;
        }
        
        public TrainingConfig build() {
            return config;
        }
    }
    
    /**
     * 创建默认配置的Builder
     * 
     * @return Builder实例
     */
    public static Builder builder() {
        return new Builder();
    }
}

