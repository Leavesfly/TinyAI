package io.leavesfly.tinyai.deepseek.r1;

import io.leavesfly.tinyai.deepseek.base.DeepSeekBaseConfig;

/**
 * DeepSeek-R1模型配置类
 * 
 * DeepSeek-R1基于DeepSeek-V3-Base架构，通过强化学习（RL）驱动推理能力的涌现。
 * 根据官方论文（arXiv:2501.12948），R1与V3使用相同的MoE架构（671B参数，8专家，Top-2路由），
 * 区别在于训练方式：
 * - V3: 标准预训练 + 后训练（SFT）
 * - R1: V3-Base + 纯强化学习（RL），无需大量SFT数据即可涌现推理能力
 * 
 * R1核心特点：
 * 1. MoE架构 - 与V3完全相同的混合专家模型
 * 2. RL驱动推理 - 通过强化学习自然涌现思维链推理能力
 * 3. 无显式推理模块 - 推理能力在RL训练中自发学习，非人为设计
 * 4. 多阶段训练 - 冷启动数据 + RL + 拒绝采样SFT + 二次RL
 * 
 * 本实现继承DeepSeekBaseConfig共享基础架构，仅扩展R1特有的RL训练配置。
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekR1Config extends DeepSeekBaseConfig {
    
    private static final long serialVersionUID = 1L;
    
    // ==================== RL训练配置（R1特有）====================
    
    /** RL训练的探索率，默认0.1 */
    private double rlExplorationRate = 0.1;
    
    /** RL奖励折扣因子，默认0.99 */
    private double rlDiscountFactor = 0.99;
    
    /** RL策略学习率，默认1e-5 */
    private double rlPolicyLearningRate = 1e-5;
    
    /** RL值函数学习率，默认1e-4 */
    private double rlValueLearningRate = 1e-4;
    
    /** RL训练批次大小，默认32 */
    private int rlBatchSize = 32;
    
    /** RL PPO clip范围，默认0.2 */
    private double rlClipRange = 0.2;
    
    /** RL熵系数（鼓励探索），默认0.01 */
    private double rlEntropyCoefficient = 0.01;
    
    /** RL最大梯度范数（梯度裁剪），默认1.0 */
    private double rlMaxGradNorm = 1.0;
    
    /** RL奖励缩放因子，默认1.0 */
    private double rlRewardScale = 1.0;
    
    /**
     * 默认构造函数，创建标准DeepSeek-R1配置
     * 使用与V3相同的基础架构（MoE）
     */
    public DeepSeekR1Config() {
        super(); // 调用父类构造函数
    }
    
    /**
     * 完整配置构造函数
     */
    public DeepSeekR1Config(int vocabSize, int nPositions, int nEmbd, int nLayer,
                            int nHead, int nInner, String activationFunction,
                            int numExperts, int topK, int expertHiddenDim,
                            double loadBalanceLossWeight, double expertDropout,
                            boolean enableTaskAwareRouting, int taskEmbedDim,
                            int taskClassifierHiddenDim, int numTaskTypes,
                            double residPdrop, double embdPdrop, double attnPdrop,
                            double layerNormEpsilon, double initializerRange,
                            double rlExplorationRate, double rlDiscountFactor,
                            double rlPolicyLearningRate, double rlValueLearningRate,
                            int rlBatchSize, double rlClipRange,
                            double rlEntropyCoefficient, double rlMaxGradNorm,
                            double rlRewardScale) {
        super(vocabSize, nPositions, nEmbd, nLayer, nHead, nInner, activationFunction,
              numExperts, topK, expertHiddenDim, loadBalanceLossWeight, expertDropout,
              enableTaskAwareRouting, taskEmbedDim, taskClassifierHiddenDim, numTaskTypes,
              residPdrop, embdPdrop, attnPdrop, layerNormEpsilon, initializerRange);
        
        // 设置R1特有的RL配置
        this.rlExplorationRate = rlExplorationRate;
        this.rlDiscountFactor = rlDiscountFactor;
        this.rlPolicyLearningRate = rlPolicyLearningRate;
        this.rlValueLearningRate = rlValueLearningRate;
        this.rlBatchSize = rlBatchSize;
        this.rlClipRange = rlClipRange;
        this.rlEntropyCoefficient = rlEntropyCoefficient;
        this.rlMaxGradNorm = rlMaxGradNorm;
        this.rlRewardScale = rlRewardScale;
    }
    
    // ==================== 预设配置工厂方法 ====================
    
    /**
     * 创建标准DeepSeek-R1配置（继承V3的MoE架构）
     * 配置：768维, 12层, 12头, 8专家, Top-2路由, 序列长度2048
     */
    public static DeepSeekR1Config createStandardConfig() {
        return new DeepSeekR1Config(); // 使用默认值即可
    }
    
    /**
     * 创建微型DeepSeek-R1配置（用于快速测试）
     * 配置：256维, 6层, 8头, 4专家, Top-2路由, 序列长度512
     */
    public static DeepSeekR1Config createTinyConfig() {
        DeepSeekR1Config config = new DeepSeekR1Config();
        config.setNEmbd(256);
        config.setNLayer(6);
        config.setNHead(8);
        config.setNInner(1024);
        config.setNPositions(512);
        config.setNumExperts(4);
        config.setTopK(2);
        config.setExpertHiddenDim(1024);
        return config;
    }
    
    /**
     * 创建小型DeepSeek-R1配置（用于学习和实验）
     * 配置：512维, 8层, 8头, 8专家, Top-2路由, 序列长度1024
     */
    public static DeepSeekR1Config createSmallConfig() {
        DeepSeekR1Config config = new DeepSeekR1Config();
        config.setNEmbd(512);
        config.setNLayer(8);
        config.setNHead(8);
        config.setNInner(2048);
        config.setNPositions(1024);
        config.setNumExperts(8);
        config.setTopK(2);
        config.setExpertHiddenDim(2048);
        return config;
    }
    
    // ==================== Getter和Setter方法（R1特有的RL配置）====================
    
    public double getRlExplorationRate() {
        return rlExplorationRate;
    }
    
    public void setRlExplorationRate(double rlExplorationRate) {
        this.rlExplorationRate = rlExplorationRate;
    }
    
    public double getRlDiscountFactor() {
        return rlDiscountFactor;
    }
    
    public void setRlDiscountFactor(double rlDiscountFactor) {
        this.rlDiscountFactor = rlDiscountFactor;
    }
    
    public double getRlPolicyLearningRate() {
        return rlPolicyLearningRate;
    }
    
    public void setRlPolicyLearningRate(double rlPolicyLearningRate) {
        this.rlPolicyLearningRate = rlPolicyLearningRate;
    }
    
    public double getRlValueLearningRate() {
        return rlValueLearningRate;
    }
    
    public void setRlValueLearningRate(double rlValueLearningRate) {
        this.rlValueLearningRate = rlValueLearningRate;
    }
    
    public int getRlBatchSize() {
        return rlBatchSize;
    }
    
    public void setRlBatchSize(int rlBatchSize) {
        this.rlBatchSize = rlBatchSize;
    }
    
    public double getRlClipRange() {
        return rlClipRange;
    }
    
    public void setRlClipRange(double rlClipRange) {
        this.rlClipRange = rlClipRange;
    }
    
    public double getRlEntropyCoefficient() {
        return rlEntropyCoefficient;
    }
    
    public void setRlEntropyCoefficient(double rlEntropyCoefficient) {
        this.rlEntropyCoefficient = rlEntropyCoefficient;
    }
    
    public double getRlMaxGradNorm() {
        return rlMaxGradNorm;
    }
    
    public void setRlMaxGradNorm(double rlMaxGradNorm) {
        this.rlMaxGradNorm = rlMaxGradNorm;
    }
    
    public double getRlRewardScale() {
        return rlRewardScale;
    }
    
    public void setRlRewardScale(double rlRewardScale) {
        this.rlRewardScale = rlRewardScale;
    }
    
    // ==================== toString方法 ====================
    
    @Override
    public String toString() {
        return String.format(
            "DeepSeekR1Config {\n" +
            "  基础架构: %s\n" +
            "  RL训练配置:\n" +
            "    探索率: %.3f\n" +
            "    折扣因子: %.3f\n" +
            "    策略学习率: %.1e\n" +
            "    值函数学习率: %.1e\n" +
            "    批次大小: %d\n" +
            "    PPO clip范围: %.2f\n" +
            "    熵系数: %.3f\n" +
            "    最大梯度范数: %.2f\n" +
            "    奖励缩放: %.2f\n" +
            "}",
            super.toString(),
            rlExplorationRate, rlDiscountFactor, rlPolicyLearningRate,
            rlValueLearningRate, rlBatchSize, rlClipRange,
            rlEntropyCoefficient, rlMaxGradNorm, rlRewardScale
        );
    }
}
