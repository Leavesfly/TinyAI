package io.leavesfly.tinyai.minimind.training.distillation;

import java.io.Serializable;

/**
 * 知识蒸馏训练配置类
 * <p>
 * 对标 Python minimind3 train_distillation.py 的蒸馏参数配置。
 * <p>
 * 知识蒸馏通过教师模型（大模型）指导学生模型（小模型）学习，
 * 混合损失 = alpha * CE_loss + (1 - alpha) * KL_loss
 * <p>
 * 关键概念：
 * - temperature：温度缩放参数，软化概率分布以暴露"暗知识"
 * - alpha：CE 损失权重，总损失 = alpha * CE + (1-alpha) * KL
 * - KL 散度乘以 T²，保持梯度尺度一致
 *
 * @author TinyAI Team
 * @since 2025
 */
public class DistillationConfig implements Serializable {

    private static final long serialVersionUID = 1L;

    /**
     * CE 损失权重
     * 总损失 = alpha * CE + (1-alpha) * KL
     * <p>
     * 对标 Python: alpha=0.5
     */
    private float alpha = 0.5f;

    /**
     * 蒸馏温度
     * <p>
     * T > 1: 软化概率分布，暴露更多暗知识
     * T = 1: 标准 softmax，无软化效果
     * T < 1: 锐化概率分布，强调高概率类别
     * <p>
     * 对标 Python: temperature=1.5（推荐范围 1.0-2.0）
     */
    private float temperature = 1.5f;

    /**
     * 学习率
     * <p>
     * 蒸馏通常基于 SFT 权重微调，使用较小学习率
     * 对标 Python: learning_rate=5e-6
     */
    private float learningRate = 5e-6f;

    /**
     * 最大训练轮数
     * 对标 Python: epochs=6
     */
    private int maxEpochs = 6;

    /**
     * 梯度累积步数
     * 对标 Python: accumulation_steps=16
     */
    private int accumulationSteps = 16;

    /**
     * 梯度裁剪阈值
     * 对标 Python: grad_clip=1.0
     */
    private float gradClip = 1.0f;

    /**
     * 日志打印间隔
     * 对标 Python: log_interval=100
     */
    private int logInterval = 100;

    /**
     * 模型保存间隔
     * 对标 Python: save_interval=100
     */
    private int saveInterval = 100;

    // ========== 构造函数 ==========

    public DistillationConfig() {
    }

    public DistillationConfig(float alpha, float temperature) {
        this.alpha = alpha;
        this.temperature = temperature;
    }

    // ========== 工厂方法 ==========

    /**
     * 创建默认配置（对标 Python train_distillation.py 默认参数）
     */
    public static DistillationConfig createDefault() {
        return new DistillationConfig();
    }

    /**
     * 创建纯蒸馏配置（alpha=0，完全依赖教师指导）
     */
    public static DistillationConfig createPureDistillation() {
        DistillationConfig config = new DistillationConfig();
        config.setAlpha(0.0f);
        config.setTemperature(2.0f);
        return config;
    }

    /**
     * 创建平衡配置（alpha=0.5，CE 和 KL 等权重）
     */
    public static DistillationConfig createBalanced() {
        return new DistillationConfig(0.5f, 1.5f);
    }

    /**
     * 创建 Micro Demo 配置（用于教学演示的超小规模配置）
     */
    public static DistillationConfig createMicroDemo() {
        DistillationConfig config = new DistillationConfig();
        config.setAlpha(0.5f);
        config.setTemperature(1.5f);
        config.setLearningRate(1e-3f);      // Demo 较大学习率
        config.setMaxEpochs(10);
        config.setAccumulationSteps(1);
        config.setLogInterval(10);
        config.setSaveInterval(1000);
        config.setGradClip(1.0f);
        return config;
    }

    // ========== 验证 ==========

    /**
     * 验证配置有效性
     */
    public void validate() {
        if (alpha < 0 || alpha > 1) {
            throw new IllegalArgumentException("alpha must be in [0, 1], got: " + alpha);
        }
        if (temperature <= 0) {
            throw new IllegalArgumentException("temperature must be positive, got: " + temperature);
        }
        if (learningRate <= 0) {
            throw new IllegalArgumentException("learningRate must be positive, got: " + learningRate);
        }
        if (maxEpochs <= 0) {
            throw new IllegalArgumentException("maxEpochs must be positive, got: " + maxEpochs);
        }
        if (accumulationSteps <= 0) {
            throw new IllegalArgumentException("accumulationSteps must be positive, got: " + accumulationSteps);
        }
        if (gradClip <= 0) {
            throw new IllegalArgumentException("gradClip must be positive, got: " + gradClip);
        }
    }

    // ========== Getter/Setter ==========

    public float getAlpha() {
        return alpha;
    }

    public void setAlpha(float alpha) {
        this.alpha = alpha;
    }

    public float getTemperature() {
        return temperature;
    }

    public void setTemperature(float temperature) {
        this.temperature = temperature;
    }

    public float getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(float learningRate) {
        this.learningRate = learningRate;
    }

    public int getMaxEpochs() {
        return maxEpochs;
    }

    public void setMaxEpochs(int maxEpochs) {
        this.maxEpochs = maxEpochs;
    }

    public int getAccumulationSteps() {
        return accumulationSteps;
    }

    public void setAccumulationSteps(int accumulationSteps) {
        this.accumulationSteps = accumulationSteps;
    }

    public float getGradClip() {
        return gradClip;
    }

    public void setGradClip(float gradClip) {
        this.gradClip = gradClip;
    }

    public int getLogInterval() {
        return logInterval;
    }

    public void setLogInterval(int logInterval) {
        this.logInterval = logInterval;
    }

    public int getSaveInterval() {
        return saveInterval;
    }

    public void setSaveInterval(int saveInterval) {
        this.saveInterval = saveInterval;
    }

    @Override
    public String toString() {
        return String.format(
            "DistillationConfig{alpha=%.2f, temperature=%.2f, lr=%.2e, " +
            "epochs=%d, accumSteps=%d, gradClip=%.1f}",
            alpha, temperature, learningRate, maxEpochs, accumulationSteps, gradClip
        );
    }
}
