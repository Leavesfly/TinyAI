package io.leavesfly.tinyai.deepseek.base;

import java.io.Serializable;

/**
 * DeepSeek基础模型配置类
 * 
 * 本配置类定义了DeepSeek-R1和DeepSeek-V3的共享基础架构。
 * 根据DeepSeek官方论文（arXiv:2501.12948），R1和V3使用相同的MoE架构，
 * 仅在训练方式上有所不同：
 * - V3: 标准预训练 + 后训练（SFT）
 * - R1: 基于V3-Base + 强化学习（RL）驱动推理能力涌现
 * 
 * 核心特点：
 * 1. Mixture-of-Experts (MoE) - 8个专家网络，Top-2路由选择
 * 2. Multi-Head Latent Attention (MLA) - 高效注意力机制（本实现使用标准MHA）
 * 3. Pre-LayerNorm架构 - 提升训练稳定性
 * 4. 参数高效 - 每次仅激活约25%的参数（2/8专家）
 * 
 * 官方架构参数：
 * - 总参数: 671B (6710亿)
 * - 激活参数: 37B/token
 * - 层数: 61层Transformer
 * - 上下文长度: 128K tokens
 * 
 * TinyAI实现说明：
 * - 本实现为教学版本，参数规模缩小至 20M-350M
 * - 使用标准MultiHeadAttention替代MLA
 * - 保留核心MoE架构和任务感知路由机制
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekBaseConfig implements Serializable {
    
    private static final long serialVersionUID = 1L;
    
    // ==================== 基础模型配置 ====================
    
    /** 词汇表大小，默认50257 */
    protected int vocabSize = 50257;
    
    /** 最大位置数（序列长度），默认2048 */
    protected int nPositions = 2048;
    
    /** 嵌入维度，默认768 */
    protected int nEmbd = 768;
    
    /** Transformer层数，默认12 */
    protected int nLayer = 12;
    
    /** 注意力头数，默认12 */
    protected int nHead = 12;
    
    /** 前馈网络中间层维度，默认4倍嵌入维度 */
    protected int nInner = 3072;
    
    /** 激活函数类型，默认"gelu" */
    protected String activationFunction = "gelu";
    
    // ==================== MoE配置（核心架构）====================
    
    /** 专家数量，默认8个专家（与官方一致） */
    protected int numExperts = 8;
    
    /** Top-K专家选择数量，默认选择2个专家（与官方一致，激活率25%） */
    protected int topK = 2;
    
    /**
     * 共享专家数量（DeepSeekMoE核心创新）
     * 共享专家每次必被激活，与路由专家共同作用。
     * 论文原版：1个共享专家 + 256个细粒度路由专家
     * 教学简化：1个共享专家 + numExperts个路由专家
     */
    protected int numSharedExperts = 1;
    
    /** 每个专家的隐藏层维度，默认与nInner相同 */
    protected int expertHiddenDim = 3072;
    
    /** 负载均衡损失权重，默认0.01 */
    protected double loadBalanceLossWeight = 0.01;
    
    /** 专家dropout概率，默认0.1 */
    protected double expertDropout = 0.1;
    
    // ==================== 任务感知配置（占位，当前实现未真正启用）====================
    //
    // ⚠️ 重要说明：
    //   本教学实现的 DeepSeekV3MoEBlock / DeepSeekR1Block 的 gating 计算是 **标准 Sigmoid 路由**，
    //   并未将 taskType embedding 融入 gating logits。下列 4 个字段当前仅作为：
    //     1) 元信息保留（便于输出 toString / estimateParameterCount 对齐论文）
    //     2) 未来扩展位：若要接入 DeepSeekMoE 论文原版任务感知路由，需改动 MoE forward 签名
    //   暂时不移除这些字段以保持与现有 V3/R1 Config 构造签名的二进制兼容。
    //
    //   用户若希望真正启用任务感知路由，需要：
    //   - 在 MoEBlock 中新增 forward(Variable input, int taskType)
    //   - 把 taskEmbed[taskType] 与 gating logits 相加后再走 Top-K
    //   - 对应调整 Dataset.Batch 传入 taskType
    //   本项目目前未做此扩展，setEnableTaskAwareRouting(true) 不会改变前向行为。

    /**
     * 是否启用任务感知路由。
     * <p><b>当前实现未真正启用：</b>参见类级注释。
     */
    protected boolean enableTaskAwareRouting = true;

    /** 任务类型嵌入维度，默认128（当前未被 MoE 消费） */
    protected int taskEmbedDim = 128;

    /** 任务识别器隐藏层维度，默认256（当前未被 MoE 消费） */
    protected int taskClassifierHiddenDim = 256;

    /** 支持的任务类型数量，默认5种：推理、代码、数学、通用、多模态 */
    protected int numTaskTypes = 5;
    
    // ==================== Dropout配置 ====================
    
    /** 残差dropout概率，默认0.1 */
    protected double residPdrop = 0.1;
    
    /** 嵌入dropout概率，默认0.1 */
    protected double embdPdrop = 0.1;
    
    /** 注意力dropout概率，默认0.1 */
    protected double attnPdrop = 0.1;
    
    // ==================== 初始化配置 ====================
    
    /** 层归一化epsilon，默认1e-5 */
    protected double layerNormEpsilon = 1e-5;
    
    /** 权重初始化范围，默认0.02 */
    protected double initializerRange = 0.02;

    // ==================== 缓存字段（volatile 保证多线程可见性）====================

    /** 参数量缓存 */
    private volatile Long cachedParameterCount;

    /** 激活参数量缓存 */
    private volatile Long cachedActiveParameterCount;

    /** 激活率缓存 */
    private volatile Double cachedActivationRatio;

    /**
     * 默认构造函数，创建标准DeepSeek基础配置
     */
    public DeepSeekBaseConfig() {
        // 使用默认值
    }
    
    /**
     * 完整配置构造函数
     */
    public DeepSeekBaseConfig(int vocabSize, int nPositions, int nEmbd, int nLayer,
                             int nHead, int nInner, String activationFunction,
                             int numExperts, int topK, int expertHiddenDim,
                             double loadBalanceLossWeight, double expertDropout,
                             boolean enableTaskAwareRouting, int taskEmbedDim,
                             int taskClassifierHiddenDim, int numTaskTypes,
                             double residPdrop, double embdPdrop, double attnPdrop,
                             double layerNormEpsilon, double initializerRange) {
        this.vocabSize = vocabSize;
        this.nPositions = nPositions;
        this.nEmbd = nEmbd;
        this.nLayer = nLayer;
        this.nHead = nHead;
        this.nInner = nInner;
        this.activationFunction = activationFunction;
        this.numExperts = numExperts;
        this.topK = topK;
        this.expertHiddenDim = expertHiddenDim;
        this.loadBalanceLossWeight = loadBalanceLossWeight;
        this.expertDropout = expertDropout;
        this.enableTaskAwareRouting = enableTaskAwareRouting;
        this.taskEmbedDim = taskEmbedDim;
        this.taskClassifierHiddenDim = taskClassifierHiddenDim;
        this.numTaskTypes = numTaskTypes;
        this.residPdrop = residPdrop;
        this.embdPdrop = embdPdrop;
        this.attnPdrop = attnPdrop;
        this.layerNormEpsilon = layerNormEpsilon;
        this.initializerRange = initializerRange;
    }
    
    // ==================== 预设配置工厂方法 ====================
    
    /**
     * 创建微型配置（快速测试，参数量约20M）
     * 
     * 配置特点：
     * - 256维嵌入
     * - 6层Transformer
     * - 8个注意力头
     * - 4个专家（Top-2路由）
     * - 512序列长度
     */
    public static DeepSeekBaseConfig createTinyConfig() {
        DeepSeekBaseConfig config = new DeepSeekBaseConfig();
        config.nEmbd = 256;
        config.nLayer = 6;
        config.nHead = 8;
        config.nInner = 1024;
        config.nPositions = 512;
        config.numExperts = 4;
        config.topK = 2;
        config.expertHiddenDim = 1024;
        return config;
    }
    
    /**
     * 创建标准配置（标准应用，参数量约100M）
     * 
     * 配置特点：
     * - 768维嵌入
     * - 12层Transformer
     * - 12个注意力头
     * - 8个专家（Top-2路由）
     * - 2048序列长度
     */
    public static DeepSeekBaseConfig createStandardConfig() {
        return new DeepSeekBaseConfig(); // 使用默认值
    }
    
    /**
     * 创建大型配置（高性能应用，参数量约350M）
     * 
     * 配置特点：
     * - 1024维嵌入
     * - 24层Transformer
     * - 16个注意力头
     * - 8个专家（Top-2路由）
     * - 2048序列长度
     */
    public static DeepSeekBaseConfig createLargeConfig() {
        DeepSeekBaseConfig config = new DeepSeekBaseConfig();
        config.nEmbd = 1024;
        config.nLayer = 24;
        config.nHead = 16;
        config.nInner = 4096;
        config.nPositions = 2048;
        config.numExperts = 8;
        config.topK = 2;
        config.expertHiddenDim = 4096;
        return config;
    }
    
    // ==================== 参数估算方法 ====================
    
    /**
     * 估算模型总参数量
     * 
     * @return 估算的总参数数量
     */
    public long estimateParameterCount() {
        if (cachedParameterCount != null) {
            return cachedParameterCount;
        }

        // Token嵌入: vocabSize * nEmbd
        long tokenEmbed = (long) vocabSize * nEmbd;

        // 位置嵌入: nPositions * nEmbd
        long posEmbed = (long) nPositions * nEmbd;

        // 每个Transformer层的参数
        long attnParams = calculateAttentionParams();
        long ln1Params = calculateLayerNormParams();
        long moeParams = calculateMoEParams();
        long ln2Params = calculateLayerNormParams();

        // 每层总参数
        long paramsPerLayer = attnParams + ln1Params + moeParams + ln2Params;

        // 所有Transformer层的参数
        long allLayersParams = paramsPerLayer * nLayer;

        // 最终LayerNorm: gamma(nEmbd) + beta(nEmbd)
        long finalLnParams = 2L * nEmbd;

        // 总参数 = 嵌入 + 所有Transformer层 + 最终LN
        cachedParameterCount = tokenEmbed + posEmbed + allLayersParams + finalLnParams;
        return cachedParameterCount;
    }

    /**
     * 计算注意力层参数
     * 
     * @return 注意力层参数数量
     */
    private long calculateAttentionParams() {
        // Q: nEmbd * nEmbd + nEmbd
        // K: nEmbd * nEmbd + nEmbd
        // V: nEmbd * nEmbd + nEmbd
        // O: nEmbd * nEmbd + nEmbd
        return 4L * ((long) nEmbd * nEmbd + nEmbd);
    }

    /**
     * 计算LayerNorm层参数
     * 
     * @return LayerNorm层参数数量
     */
    private long calculateLayerNormParams() {
        // gamma(nEmbd) + beta(nEmbd)
        return 2L * nEmbd;
    }

    /**
     * 计算MoE层参数
     * 
     * 注意：实际 DeepSeek V3 的专家为 SwiGLU 结构（gate_proj + up_proj + down_proj），
     * 单个专家有 3 个 Linear 而非 2 个，因此参数量估算使用 3 倍 FFN。
     * 同时包含路由专家(numExperts) + 共享专家(numSharedExperts) + 专家 bias 参数。
     * 
     * @return MoE层参数数量
     */
    private long calculateMoEParams() {
        // 门控网络: nEmbd * numExperts（不带 bias，bias 由 expertBias 独立管理）
        long gatingParams = (long) nEmbd * numExperts;

        // 专家路由 bias（无辅助损失负载均衡）
        long expertBiasParams = numExperts;

        // 每个专家（SwiGLU）的参数: gate_proj + up_proj + down_proj，均带 bias
        // gate_proj: nEmbd * expertHiddenDim + expertHiddenDim
        // up_proj:   nEmbd * expertHiddenDim + expertHiddenDim
        // down_proj: expertHiddenDim * nEmbd + nEmbd
        long paramsPerExpert = 2L * ((long) nEmbd * expertHiddenDim + expertHiddenDim) +
                                (long) expertHiddenDim * nEmbd + nEmbd;

        // 路由专家总参数
        long routedExpertsParams = paramsPerExpert * numExperts;

        // 共享专家总参数（DeepSeekMoE 核心创新）
        long sharedExpertsParams = paramsPerExpert * numSharedExperts;

        return gatingParams + expertBiasParams + routedExpertsParams + sharedExpertsParams;
    }
    
    /**
     * 计算激活参数数量（仅激活Top-K个专家）
     * 
     * @return 激活的参数数量
     */
    public long estimateActiveParameterCount() {
        if (cachedActiveParameterCount != null) {
            return cachedActiveParameterCount;
        }

        // Token嵌入
        long tokenEmbed = (long) vocabSize * nEmbd;

        // 位置嵌入
        long posEmbed = (long) nPositions * nEmbd;

        // 每层的激活参数
        long attnParams = calculateAttentionParams();
        long ln1Params = calculateLayerNormParams();
        long activeMoeParams = calculateActiveMoEParams();
        long ln2Params = calculateLayerNormParams();

        // 每层激活参数
        long activeParamsPerLayer = attnParams + ln1Params + activeMoeParams + ln2Params;

        // 所有层的激活参数
        long allLayersActiveParams = activeParamsPerLayer * nLayer;

        // 最终LayerNorm
        long finalLnParams = 2L * nEmbd;

        cachedActiveParameterCount = tokenEmbed + posEmbed + allLayersActiveParams + finalLnParams;
        return cachedActiveParameterCount;
    }

    /**
     * 计算激活的MoE层参数（仅Top-K个专家）
     * 
     * @return 激活的MoE层参数数量
     */
    private long calculateActiveMoEParams() {
        // 门控网络: nEmbd * numExperts + numExperts
        long gatingParams = (long) nEmbd * numExperts + numExperts;

        // 每个专家的FFN: fc1(nEmbd * expertHiddenDim + expertHiddenDim) + fc2(expertHiddenDim * nEmbd + nEmbd)
        long paramsPerExpert = (long) nEmbd * expertHiddenDim + expertHiddenDim +
                                (long) expertHiddenDim * nEmbd + nEmbd;

        // 仅激活Top-K个专家
        long activeExpertsParams = paramsPerExpert * topK;

        // 激活的MoE总参数
        return gatingParams + activeExpertsParams;
    }
    
    /**
     * 计算参数激活率（激活参数/总参数）
     * 
     * @return 激活率百分比（0-100之间）
     */
    public double getActivationRatio() {
        if (cachedActivationRatio != null) {
            return cachedActivationRatio;
        }

        cachedActivationRatio = (double) estimateActiveParameterCount() / estimateParameterCount() * 100;
        return cachedActivationRatio;
    }

    /**
     * 清除缓存
     * 
     * 当配置参数发生变化时调用此方法清除缓存
     */
    public void clearCache() {
        cachedParameterCount = null;
        cachedActiveParameterCount = null;
        cachedActivationRatio = null;
    }
    
    // ==================== Getter和Setter方法 ====================
    
    public int getVocabSize() {
        return vocabSize;
    }
    
    public void setVocabSize(int vocabSize) {
        this.vocabSize = vocabSize;
        clearCache();
    }
    
    public int getNPositions() {
        return nPositions;
    }
    
    public void setNPositions(int nPositions) {
        this.nPositions = nPositions;
        clearCache();
    }
    
    public int getNEmbd() {
        return nEmbd;
    }
    
    public void setNEmbd(int nEmbd) {
        this.nEmbd = nEmbd;
        clearCache();
    }
    
    public int getNLayer() {
        return nLayer;
    }
    
    public void setNLayer(int nLayer) {
        this.nLayer = nLayer;
        clearCache();
    }
    
    public int getNHead() {
        return nHead;
    }
    
    public void setNHead(int nHead) {
        this.nHead = nHead;
        clearCache();
    }
    
    public int getNInner() {
        return nInner;
    }
    
    public void setNInner(int nInner) {
        this.nInner = nInner;
        clearCache();
    }
    
    public String getActivationFunction() {
        return activationFunction;
    }
    
    public void setActivationFunction(String activationFunction) {
        this.activationFunction = activationFunction;
    }
    
    public int getNumExperts() {
        return numExperts;
    }
    
    public void setNumExperts(int numExperts) {
        this.numExperts = numExperts;
        clearCache();
    }
    
    public int getTopK() {
        return topK;
    }
    
    public void setTopK(int topK) {
        this.topK = topK;
        clearCache();
    }
    
    public int getNumSharedExperts() {
        return numSharedExperts;
    }
    
    public void setNumSharedExperts(int numSharedExperts) {
        this.numSharedExperts = numSharedExperts;
        clearCache();
    }
    
    public int getExpertHiddenDim() {
        return expertHiddenDim;
    }
    
    public void setExpertHiddenDim(int expertHiddenDim) {
        this.expertHiddenDim = expertHiddenDim;
        clearCache();
    }
    
    public double getLoadBalanceLossWeight() {
        return loadBalanceLossWeight;
    }

    public void setLoadBalanceLossWeight(double loadBalanceLossWeight) {
        this.loadBalanceLossWeight = loadBalanceLossWeight;
        // 不影响参数量，无需清缓存
    }

    public double getExpertDropout() {
        return expertDropout;
    }

    public void setExpertDropout(double expertDropout) {
        this.expertDropout = expertDropout;
        // 不影响参数量，无需清缓存
    }

    public boolean isEnableTaskAwareRouting() {
        return enableTaskAwareRouting;
    }

    public void setEnableTaskAwareRouting(boolean enableTaskAwareRouting) {
        this.enableTaskAwareRouting = enableTaskAwareRouting;
        clearCache();
    }
    
    public int getTaskEmbedDim() {
        return taskEmbedDim;
    }

    public void setTaskEmbedDim(int taskEmbedDim) {
        this.taskEmbedDim = taskEmbedDim;
        clearCache();
    }

    public int getTaskClassifierHiddenDim() {
        return taskClassifierHiddenDim;
    }

    public void setTaskClassifierHiddenDim(int taskClassifierHiddenDim) {
        this.taskClassifierHiddenDim = taskClassifierHiddenDim;
        clearCache();
    }

    public int getNumTaskTypes() {
        return numTaskTypes;
    }

    public void setNumTaskTypes(int numTaskTypes) {
        this.numTaskTypes = numTaskTypes;
        clearCache();
    }
    
    public double getResidPdrop() {
        return residPdrop;
    }
    
    public void setResidPdrop(double residPdrop) {
        this.residPdrop = residPdrop;
    }
    
    public double getEmbdPdrop() {
        return embdPdrop;
    }
    
    public void setEmbdPdrop(double embdPdrop) {
        this.embdPdrop = embdPdrop;
    }
    
    public double getAttnPdrop() {
        return attnPdrop;
    }
    
    public void setAttnPdrop(double attnPdrop) {
        this.attnPdrop = attnPdrop;
    }
    
    public double getLayerNormEpsilon() {
        return layerNormEpsilon;
    }
    
    public void setLayerNormEpsilon(double layerNormEpsilon) {
        this.layerNormEpsilon = layerNormEpsilon;
    }
    
    public double getInitializerRange() {
        return initializerRange;
    }
    
    public void setInitializerRange(double initializerRange) {
        this.initializerRange = initializerRange;
    }
    
    // ==================== toString方法 ====================
    
    @Override
    public String toString() {
        return String.format(
            "DeepSeekBaseConfig {\n" +
            "  基础架构:\n" +
            "    词汇表大小: %,d\n" +
            "    嵌入维度: %d\n" +
            "    Transformer层数: %d\n" +
            "    注意力头数: %d\n" +
            "    前馈网络维度: %d\n" +
            "    最大序列长度: %d\n" +
            "  MoE配置:\n" +
            "    专家数量: %d\n" +
            "    共享专家数: %d\n" +
            "    Top-K选择: %d\n" +
            "    专家隐藏维度: %d\n" +
            "    激活率: %.1f%%\n" +
            "  任务感知:\n" +
            "    启用任务感知路由: %s\n" +
            "    支持任务类型数: %d\n" +
            "  参数统计:\n" +
            "    总参数量: %,d (%.2fM)\n" +
            "    激活参数量: %,d (%.2fM)\n" +
            "  Dropout:\n" +
            "    残差dropout: %.2f\n" +
            "    注意力dropout: %.2f\n" +
            "}",
            vocabSize, nEmbd, nLayer, nHead, nInner, nPositions,
            numExperts, numSharedExperts, topK, expertHiddenDim, getActivationRatio(),
            enableTaskAwareRouting, numTaskTypes,
            estimateParameterCount(), estimateParameterCount() / 1_000_000.0,
            estimateActiveParameterCount(), estimateActiveParameterCount() / 1_000_000.0,
            residPdrop, attnPdrop
        );
    }
}
