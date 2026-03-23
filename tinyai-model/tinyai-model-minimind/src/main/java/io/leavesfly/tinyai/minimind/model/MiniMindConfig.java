package io.leavesfly.tinyai.minimind.model;

import java.io.Serializable;

/**
 * MiniMind 模型配置类
 * <p>
 * 定义 MiniMind 模型的所有超参数配置,包括模型规模、架构参数、训练参数等。
 * 提供三种预设配置:Small(26M)、Medium(108M)、MoE(145M)
 * </p>
 *
 * @author TinyAI Team
 * @version 1.0
 * @since 2025-01-01
 */
public class MiniMindConfig implements Serializable {

    private static final long serialVersionUID = 1L;

    /**
     * EOS token ID (默认为 2)
     */
    private int eosTokenId = 2;

    // ========== 基础配置 ==========

    /**
     * 词汇表大小
     */
    private int vocabSize = 6400;

    /**
     * 最大序列长度
     */
    private int maxSeqLen = 512;

    /**
     * 隐藏层维度(d_model)
     */
    private int hiddenSize = 512;

    /**
     * Transformer 层数
     */
    private int numLayers = 8;

    /**
     * 注意力头数
     */
    private int numHeads = 16;

    /**
     * 前馈网络隐藏层维度(FFN hidden size)
     */
    private int ffnHiddenSize = 1024;

    // ========== 正则化参数 ==========

    /**
     * Dropout 比例
     */
    private float dropout = 0.1f;

    /**
     * 注意力 Dropout 比例
     */
    private float attentionDropout = 0.1f;

    /**
     * LayerNorm epsilon 值
     */
    private float epsilon = 1e-5f;

    // ========== 架构特性 ==========

    /**
     * 激活函数类型 ("silu", "gelu", "relu")
     */
    private String activationFunction = "silu";

    /**
     * 是否使用 RoPE 位置编码
     */
    private boolean useRoPE = true;

    /**
     * 是否使用 Pre-LayerNorm (true: Pre-LN, false: Post-LN)
     */
    private boolean preLayerNorm = true;

    /**
     * RoPE 的 theta 参数(用于计算频率)
     */
    private float ropeTheta = 10000.0f;

    /**
     * 是否使用 Bias(默认不使用,遵循现代 LLM 设计)
     */
    private boolean useBias = false;

    // ========== MoE 相关配置 ==========

    /**
     * 是否启用 MoE 架构
     */
    private boolean useMoE = false;

    /**
     * MoE 专家数量
     */
    private int numExperts = 4;

    /**
     * 每个 Token 激活的专家数量(Top-K)
     */
    private int numExpertsPerToken = 2;

    /**
     * MoE 负载均衡损失系数
     */
    private float moeLoadBalanceWeight = 0.01f;

    // ========== 训练相关配置 ==========

    /**
     * 初始化标准差(Xavier/He 初始化)
     */
    private float initStd = 0.02f;

    /**
     * 是否使用梯度检查点(降低显存占用)
     */
    private boolean useGradientCheckpointing = false;

    // ========== 预设配置工厂方法 ==========

    /**
     * 创建 Small 模型配置 (26M 参数)
     * <p>
     * 参数配置:
     * - 词汇表: 6400
     * - 序列长度: 512
     * - 隐藏维度: 512
     * - 层数: 8
     * - 注意力头数: 16
     * - FFN 维度: 1024
     * </p>
     *
     * @return Small 模型配置
     */
    public static MiniMindConfig createSmallConfig() {
        MiniMindConfig config = new MiniMindConfig();
        config.vocabSize = 6400;
        config.maxSeqLen = 512;
        config.hiddenSize = 512;
        config.numLayers = 8;
        config.numHeads = 16;
        config.ffnHiddenSize = 1024;
        config.dropout = 0.1f;
        config.attentionDropout = 0.1f;
        config.epsilon = 1e-5f;
        config.ropeTheta = 10000.0f;
        config.useBias = false;
        config.activationFunction = "silu";
        config.useRoPE = true;
        config.preLayerNorm = true;
        return config;
    }

    /**
     * 创建 Medium 模型配置 (108M 参数)
     * <p>
     * 参数配置:
     * - 词汇表: 6400
     * - 序列长度: 512
     * - 隐藏维度: 768
     * - 层数: 16
     * - 注意力头数: 16
     * - FFN 维度: 2048
     * </p>
     *
     * @return Medium 模型配置
     */
    public static MiniMindConfig createMediumConfig() {
        MiniMindConfig config = new MiniMindConfig();
        config.vocabSize = 6400;
        config.maxSeqLen = 512;
        config.hiddenSize = 768;
        config.numLayers = 16;
        config.numHeads = 16;
        config.ffnHiddenSize = 2048;
        config.dropout = 0.1f;
        config.attentionDropout = 0.1f;
        config.epsilon = 1e-5f;
        config.ropeTheta = 10000.0f;
        config.useBias = false;
        config.activationFunction = "silu";
        config.useRoPE = true;
        config.preLayerNorm = true;
        return config;
    }

    /**
     * 创建 MoE 模型配置 (145M 参数, 4 专家)
     * <p>
     * 参数配置:
     * - 词汇表: 6400
     * - 序列长度: 512
     * - 隐藏维度: 512
     * - 层数: 8
     * - 注意力头数: 16
     * - FFN 维度: 1024
     * - 专家数量: 4
     * - 每次激活: 2 个专家
     * </p>
     *
     * @return MoE 模型配置
     */
    public static MiniMindConfig createMoEConfig() {
        MiniMindConfig config = createSmallConfig();
        config.useMoE = true;
        config.numExperts = 4;
        config.numExpertsPerToken = 2;
        config.moeLoadBalanceWeight = 0.01f;
        return config;
    }

    // ========== 辅助方法 ==========

    /**
     * 获取每个注意力头的维度
     *
     * @return 头维度 (hiddenSize / numHeads)
     */
    public int getHeadDim() {
        if (hiddenSize % numHeads != 0) {
            throw new IllegalStateException(
                    "hiddenSize(" + hiddenSize + ") must be divisible by numHeads(" + numHeads + ")"
            );
        }
        return hiddenSize / numHeads;
    }

    /**
     * 验证配置的有效性
     *
     * @throws IllegalStateException 如果配置无效
     */
    public void validate() {
        if (vocabSize <= 0) {
            throw new IllegalStateException("vocabSize must be positive");
        }
        if (maxSeqLen <= 0) {
            throw new IllegalStateException("maxSeqLen must be positive");
        }
        if (hiddenSize <= 0) {
            throw new IllegalStateException("hiddenSize must be positive");
        }
        if (numLayers <= 0) {
            throw new IllegalStateException("numLayers must be positive");
        }
        if (numHeads <= 0) {
            throw new IllegalStateException("numHeads must be positive");
        }
        if (hiddenSize % numHeads != 0) {
            throw new IllegalStateException("hiddenSize must be divisible by numHeads");
        }
        if (ffnHiddenSize <= 0) {
            throw new IllegalStateException("ffnHiddenSize must be positive");
        }
        if (dropout < 0 || dropout >= 1) {
            throw new IllegalStateException("dropout must be in [0, 1)");
        }
        if (attentionDropout < 0 || attentionDropout >= 1) {
            throw new IllegalStateException("attentionDropout must be in [0, 1)");
        }
        if (epsilon <= 0) {
            throw new IllegalStateException("epsilon must be positive");
        }
        if (ropeTheta <= 0) {
            throw new IllegalStateException("ropeTheta must be positive");
        }
        if (initStd <= 0) {
            throw new IllegalStateException("initStd must be positive");
        }
        if (useMoE) {
            if (numExperts <= 0) {
                throw new IllegalStateException("numExperts must be positive when using MoE");
            }
            if (numExpertsPerToken <= 0 || numExpertsPerToken > numExperts) {
                throw new IllegalStateException("numExpertsPerToken must be in (0, numExperts]");
            }
        }
    }

    /**
     * 获取模型规模描述
     *
     * @return 模型规模字符串
     */
    public String getModelSize() {
        if (useMoE) {
            return String.format("MoE-%dM (%d Experts)", estimateParameters() / 1_000_000, numExperts);
        } else if (hiddenSize == 512 && numLayers == 8) {
            return "Small-26M";
        } else if (hiddenSize == 768 && numLayers == 16) {
            return "Medium-108M";
        } else {
            return String.format("Custom-%dM", estimateParameters() / 1_000_000);
        }
    }

    /**
     * 估算模型参数量(粗略计算)
     *
     * @return 估算的参数数量
     */
    public long estimateParameters() {
        long params = 0;

        // Token Embedding: vocabSize * hiddenSize
        params += (long) vocabSize * hiddenSize;

        // Transformer Layers
        for (int i = 0; i < numLayers; i++) {
            // Attention: QKV projections + Output projection
            // QKV: 3 * hiddenSize * hiddenSize (with bias: + 3 * hiddenSize)
            // Output: hiddenSize * hiddenSize (with bias: + hiddenSize)
            params += (long) hiddenSize * hiddenSize * 4;
            if (useBias) {
                params += (long) hiddenSize * 4;
            }

            // FFN
            if (useMoE) {
                // MoE: numExperts * (hiddenSize * ffnHiddenSize * 2)
                params += (long) numExperts * hiddenSize * ffnHiddenSize * 2;
                if (useBias) {
                    params += (long) numExperts * (ffnHiddenSize + hiddenSize) * 2;
                }
                // Router: hiddenSize * numExperts
                params += (long) hiddenSize * numExperts;
                if (useBias) {
                    params += (long) numExperts;
                }
            } else {
                // Standard FFN: hiddenSize * ffnHiddenSize * 2
                params += (long) hiddenSize * ffnHiddenSize * 2;
                if (useBias) {
                    params += (long) ffnHiddenSize + hiddenSize;
                }
            }

            // LayerNorm: 2 * hiddenSize (gamma + beta)
            params += (long) hiddenSize * 2 * 2;
        }

        // Final LayerNorm: hiddenSize
        params += hiddenSize;

        // LM Head: hiddenSize * vocabSize (may share with embedding)
        params += (long) hiddenSize * vocabSize;
        if (useBias) {
            params += (long) vocabSize;
        }

        return params;
    }

    // ========== Getter 和 Setter 方法 ==========

    public int getVocabSize() { return vocabSize; }
    public void setVocabSize(int vocabSize) { this.vocabSize = vocabSize; }

    public int getMaxSeqLen() { return maxSeqLen; }
    public void setMaxSeqLen(int maxSeqLen) { this.maxSeqLen = maxSeqLen; }

    public int getHiddenSize() { return hiddenSize; }
    public void setHiddenSize(int hiddenSize) { this.hiddenSize = hiddenSize; }

    public int getNumLayers() { return numLayers; }
    public void setNumLayers(int numLayers) { this.numLayers = numLayers; }

    public int getNumHeads() { return numHeads; }
    public void setNumHeads(int numHeads) { this.numHeads = numHeads; }

    public int getFfnHiddenSize() { return ffnHiddenSize; }
    public void setFfnHiddenSize(int ffnHiddenSize) { this.ffnHiddenSize = ffnHiddenSize; }

    public float getDropout() { return dropout; }
    public void setDropout(float dropout) { this.dropout = dropout; }

    public float getAttentionDropout() { return attentionDropout; }
    public void setAttentionDropout(float attentionDropout) { this.attentionDropout = attentionDropout; }

    public float getEpsilon() { return epsilon; }
    public void setEpsilon(float epsilon) { this.epsilon = epsilon; }

    public String getActivationFunction() { return activationFunction; }
    public void setActivationFunction(String activationFunction) { this.activationFunction = activationFunction; }

    public boolean isUseRoPE() { return useRoPE; }
    public void setUseRoPE(boolean useRoPE) { this.useRoPE = useRoPE; }

    public boolean isPreLayerNorm() { return preLayerNorm; }
    public void setPreLayerNorm(boolean preLayerNorm) { this.preLayerNorm = preLayerNorm; }

    public float getRopeTheta() { return ropeTheta; }
    public void setRopeTheta(float ropeTheta) { this.ropeTheta = ropeTheta; }

    public boolean isUseBias() { return useBias; }
    public void setUseBias(boolean useBias) { this.useBias = useBias; }

    public boolean isUseMoE() { return useMoE; }
    public void setUseMoE(boolean useMoE) { this.useMoE = useMoE; }

    public int getNumExperts() { return numExperts; }
    public void setNumExperts(int numExperts) { this.numExperts = numExperts; }

    public int getNumExpertsPerToken() { return numExpertsPerToken; }
    public void setNumExpertsPerToken(int numExpertsPerToken) { this.numExpertsPerToken = numExpertsPerToken; }

    public float getMoeLoadBalanceWeight() { return moeLoadBalanceWeight; }
    public void setMoeLoadBalanceWeight(float moeLoadBalanceWeight) { this.moeLoadBalanceWeight = moeLoadBalanceWeight; }

    public float getInitStd() { return initStd; }
    public void setInitStd(float initStd) { this.initStd = initStd; }

    public boolean isUseGradientCheckpointing() { return useGradientCheckpointing; }
    public void setUseGradientCheckpointing(boolean useGradientCheckpointing) {
        this.useGradientCheckpointing = useGradientCheckpointing;
    }

    public int getEosTokenId() { return eosTokenId; }
    public void setEosTokenId(int eosTokenId) { this.eosTokenId = eosTokenId; }

    @Override
    public String toString() {
        return "MiniMindConfig{" +
                "modelSize=" + getModelSize() +
                ", vocabSize=" + vocabSize +
                ", maxSeqLen=" + maxSeqLen +
                ", hiddenSize=" + hiddenSize +
                ", numLayers=" + numLayers +
                ", numHeads=" + numHeads +
                ", ffnHiddenSize=" + ffnHiddenSize +
                ", dropout=" + dropout +
                ", activation='" + activationFunction + '\'' +
                ", useRoPE=" + useRoPE +
                ", useMoE=" + useMoE +
                ", estimatedParams=" + estimateParameters() +
                '}';
    }
}