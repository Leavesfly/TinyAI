package io.leavesfly.tinyai.minimind.training.agent;

import java.io.Serializable;

/**
 * Agent 强化学习训练配置类
 * <p>
 * 对标 Python minimind3 train_agent.py 的训练参数配置。
 * <p>
 * Agent RL 训练通过工具调用强化学习优化模型的工具使用能力：
 * - 使用 GRPO 算法进行策略优化
 * - 通过 KL 散度约束防止策略偏离过远
 * - 多维度奖励计算（工具调用正确性、参数校验、GT 验证等）
 * - 支持多轮工具交互的 Rollout 生成
 *
 * @author TinyAI Team
 * @since 2025
 */
public class AgentConfig implements Serializable {

    private static final long serialVersionUID = 1L;

    /**
     * 学习率
     * 对标 Python: learning_rate=3e-7（大模型），Demo 使用较大值
     */
    private float learningRate = 3e-7f;

    /**
     * 每个 prompt 生成的候选数量
     * 对标 Python: num_generations=4
     */
    private int numGenerations = 4;

    /**
     * 最大工具交互轮数
     * 对标 Python: max_turns=3
     */
    private int maxTurns = 3;

    /**
     * 单次最大生成 token 长度
     * 对标 Python: max_gen_len=768
     */
    private int maxGenLen = 768;

    /**
     * 采样温度
     * 对标 Python: temperature=0.8
     */
    private float temperature = 0.8f;

    /**
     * KL 散度惩罚系数
     * 对标 Python: beta=0.1
     */
    private float beta = 0.1f;

    /**
     * PPO clip 范围 epsilon
     * 对标 Python: epsilon=0.2
     */
    private float epsilon = 0.2f;

    /**
     * 思考模式开启概率 (0.0~1.0)
     * 对标 Python: thinking_ratio=0.1
     */
    private float thinkingRatio = 0.1f;

    /**
     * 梯度累积步数
     * 对标 Python: accumulation_steps=2
     */
    private int accumulationSteps = 2;

    /**
     * 最大训练轮数
     * 对标 Python: epochs=1
     */
    private int maxEpochs = 1;

    /**
     * 梯度裁剪阈值
     * 对标 Python: grad_clip=1.0
     */
    private float gradClip = 1.0f;

    /**
     * 日志打印间隔
     * 对标 Python: log_interval=1
     */
    private int logInterval = 1;

    /**
     * 模型保存间隔
     * 对标 Python: save_interval=10
     */
    private int saveInterval = 10;

    // ========== 构造函数 ==========

    public AgentConfig() {
    }

    // ========== 工厂方法 ==========

    /**
     * 创建默认配置（对标 Python train_agent.py 默认参数）
     */
    public static AgentConfig createDefault() {
        return new AgentConfig();
    }

    /**
     * 创建 Micro Demo 配置（用于教学演示的超小规模配置）
     * <p>
     * 适配 Demo 场景：小词表、短序列、少量数据
     */
    public static AgentConfig createMicroDemo() {
        AgentConfig config = new AgentConfig();
        config.setLearningRate(1e-3f);        // Demo 较大学习率
        config.setNumGenerations(2);           // 每 prompt 2 个候选
        config.setMaxTurns(2);                 // 最多 2 轮工具交互
        config.setMaxGenLen(30);               // 短序列生成
        config.setTemperature(0.8f);
        config.setBeta(0.1f);
        config.setEpsilon(0.2f);
        config.setThinkingRatio(0.0f);         // Demo 关闭思考模式
        config.setAccumulationSteps(1);        // 无梯度累积
        config.setMaxEpochs(5);
        config.setGradClip(1.0f);
        config.setLogInterval(5);
        config.setSaveInterval(1000);
        return config;
    }

    // ========== 验证 ==========

    /**
     * 验证配置有效性
     */
    public void validate() {
        if (learningRate <= 0) {
            throw new IllegalArgumentException("learningRate must be positive, got: " + learningRate);
        }
        if (numGenerations < 1) {
            throw new IllegalArgumentException("numGenerations must be >= 1, got: " + numGenerations);
        }
        if (maxTurns < 1) {
            throw new IllegalArgumentException("maxTurns must be >= 1, got: " + maxTurns);
        }
        if (maxGenLen < 1) {
            throw new IllegalArgumentException("maxGenLen must be >= 1, got: " + maxGenLen);
        }
        if (temperature <= 0) {
            throw new IllegalArgumentException("temperature must be positive, got: " + temperature);
        }
        if (beta < 0) {
            throw new IllegalArgumentException("beta must be non-negative, got: " + beta);
        }
        if (epsilon <= 0 || epsilon >= 1) {
            throw new IllegalArgumentException("epsilon must be in (0, 1), got: " + epsilon);
        }
        if (maxEpochs <= 0) {
            throw new IllegalArgumentException("maxEpochs must be positive, got: " + maxEpochs);
        }
        if (gradClip <= 0) {
            throw new IllegalArgumentException("gradClip must be positive, got: " + gradClip);
        }
    }

    // ========== Getter/Setter ==========

    public float getLearningRate() { return learningRate; }
    public void setLearningRate(float learningRate) { this.learningRate = learningRate; }

    public int getNumGenerations() { return numGenerations; }
    public void setNumGenerations(int numGenerations) { this.numGenerations = numGenerations; }

    public int getMaxTurns() { return maxTurns; }
    public void setMaxTurns(int maxTurns) { this.maxTurns = maxTurns; }

    public int getMaxGenLen() { return maxGenLen; }
    public void setMaxGenLen(int maxGenLen) { this.maxGenLen = maxGenLen; }

    public float getTemperature() { return temperature; }
    public void setTemperature(float temperature) { this.temperature = temperature; }

    public float getBeta() { return beta; }
    public void setBeta(float beta) { this.beta = beta; }

    public float getEpsilon() { return epsilon; }
    public void setEpsilon(float epsilon) { this.epsilon = epsilon; }

    public float getThinkingRatio() { return thinkingRatio; }
    public void setThinkingRatio(float thinkingRatio) { this.thinkingRatio = thinkingRatio; }

    public int getAccumulationSteps() { return accumulationSteps; }
    public void setAccumulationSteps(int accumulationSteps) { this.accumulationSteps = accumulationSteps; }

    public int getMaxEpochs() { return maxEpochs; }
    public void setMaxEpochs(int maxEpochs) { this.maxEpochs = maxEpochs; }

    public float getGradClip() { return gradClip; }
    public void setGradClip(float gradClip) { this.gradClip = gradClip; }

    public int getLogInterval() { return logInterval; }
    public void setLogInterval(int logInterval) { this.logInterval = logInterval; }

    public int getSaveInterval() { return saveInterval; }
    public void setSaveInterval(int saveInterval) { this.saveInterval = saveInterval; }

    @Override
    public String toString() {
        return String.format(
            "AgentConfig{lr=%.2e, numGen=%d, maxTurns=%d, maxGenLen=%d, " +
            "temp=%.1f, beta=%.2f, eps=%.2f, epochs=%d, accumSteps=%d}",
            learningRate, numGenerations, maxTurns, maxGenLen,
            temperature, beta, epsilon, maxEpochs, accumulationSteps
        );
    }
}
