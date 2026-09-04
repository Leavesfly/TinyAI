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
     * BOS token ID (默认为 1)
     */
    private int bosTokenId = 1;

    /**
     * EOS token ID (默认为 2)
     */
    private int eosTokenId = 2;

    // ========== 基础配置（对标 Python MiniMindConfig 默认值） ==========

    /**
     * 词汇表大小
     */
    private int vocabSize = 6400;

    /**
     * 最大序列长度（训练时使用，对标 Python max_position_embeddings=32768）
     */
    private int maxSeqLen = 512;

    /**
     * 最大位置编码长度（对标 Python max_position_embeddings）
     */
    private int maxPositionEmbeddings = 32768;

    /**
     * 隐藏层维度(d_model)，对标 Python hidden_size=768
     */
    private int hiddenSize = 768;

    /**
     * Transformer 层数，对标 Python num_hidden_layers=8
     */
    private int numLayers = 8;

    /**
     * 注意力头数（Q 头数），对标 Python num_attention_heads=8
     */
    private int numHeads = 8;

    /**
     * GQA 中的 KV 头数，对标 Python num_key_value_heads=4
     * 当 numKVHeads < numHeads 时启用 GQA（分组查询注意力）
     */
    private int numKVHeads = 4;

    /**
     * 每个注意力头的维度，默认 hiddenSize / numHeads
     */
    private int headDim = -1; // -1 表示自动计算

    /**
     * 前馈网络中间层维度（SwiGLU intermediate size）
     * 对标 Python: intermediate_size = ceil(hidden_size * PI / 64) * 64
     * 若为 -1 则自动按公式计算
     */
    private int intermediateSize = -1;

    /**
     * 前馈网络隐藏层维度（兼容旧接口，优先使用 intermediateSize）
     * @deprecated 使用 {@link #intermediateSize} 代替
     */
    private int ffnHiddenSize = -1;

    // ========== 正则化参数 ==========

    /**
     * Dropout 比例，对标 Python dropout=0.0
     */
    private float dropout = 0.0f;

    /**
     * 注意力 Dropout 比例
     */
    private float attentionDropout = 0.0f;

    /**
     * RMSNorm epsilon 值，对标 Python rms_norm_eps=1e-6
     */
    private float epsilon = 1e-6f;

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
     * RoPE 的 theta 参数，对标 Python rope_theta=1e6
     */
    private float ropeTheta = 1000000.0f;

    /**
     * 是否使用 Flash Attention（Java 中仅做标记）
     */
    private boolean flashAttn = true;

    /**
     * 是否使用 Bias(默认不使用,遵循现代 LLM 设计)
     */
    private boolean useBias = false;

    // ========== MoE 相关配置 ==========

    /**
     * 是否启用 MoE 架构，对标 Python use_moe=False
     */
    private boolean useMoE = false;

    /**
     * MoE 专家数量
     */
    private int numExperts = 4;

    /**
     * 每个 Token 激活的专家数量(Top-K)，对标 Python num_experts_per_tok=1
     */
    private int numExpertsPerToken = 1;

    /**
     * MoE 路由辅助损失系数，对标 Python router_aux_loss_coef=5e-4
     */
    private float routerAuxLossCoef = 5e-4f;

    /**
     * MoE 路由噪声因子
     */
    private float moeNoiseFactor = 0.1f;

    /**
     * MoE 专家中间层维度，默认等于 intermediateSize
     * 对标 Python moe_intermediate_size
     */
    private int moeIntermediateSize = -1;

    /**
     * 是否归一化 top-k 概率，对标 Python norm_topk_prob=True
     */
    private boolean normTopkProb = true;

    /**
     * MoE 重要性损失系数（兼容旧接口）
     */
    private float moeImportanceCoef = 0.01f;

    /**
     * MoE 负载损失系数（兼容旧接口）
     */
    private float moeLoadCoef = 0.01f;

    /**
     * 是否使用共享专家
     */
    private boolean moeSharedExperts = false;

    /**
     * 是否启用负载均衡
     */
    private boolean moeEnableLoadBalance = true;

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
     * 创建默认配置（对标 Python MiniMindConfig 默认值）
     * <p>
     * 参数配置（对标 minimind3）:
     * - 词汇表: 6400
     * - 隐藏维度: 768
     * - 层数: 8
     * - Q 注意力头数: 8
     * - KV 注意力头数: 4（GQA）
     * - intermediate_size: ceil(768 * PI / 64) * 64 = 2432
     * - dropout: 0.0
     * - rms_norm_eps: 1e-6
     * - rope_theta: 1e6
     * </p>
     *
     * @return 默认模型配置
     */
    public static MiniMindConfig createDefaultConfig() {
        MiniMindConfig config = new MiniMindConfig();
        // 所有默认值已在字段声明中对标 Python
        return config;
    }

    /**
     * 创建 Small 模型配置（对标 Python 默认配置）
     *
     * @return Small 模型配置
     */
    public static MiniMindConfig createSmallConfig() {
        return createDefaultConfig();
    }

    /**
     * 创建 Medium 模型配置
     * <p>
     * 参数配置:
     * - 隐藏维度: 1024
     * - 层数: 16
     * - Q 注意力头数: 16
     * - KV 注意力头数: 8
     * </p>
     *
     * @return Medium 模型配置
     */
    public static MiniMindConfig createMediumConfig() {
        MiniMindConfig config = new MiniMindConfig();
        config.hiddenSize = 1024;
        config.numLayers = 16;
        config.numHeads = 16;
        config.numKVHeads = 8;
        // intermediateSize 自动计算
        return config;
    }

    /**
     * 创建 MoE 模型配置（对标 Python use_moe=True）
     * <p>
     * 参数配置:
     * - 基础: 默认配置
     * - 专家数量: 4
     * - 每次激活: 1 个专家
     * - 辅助损失系数: 5e-4
     * </p>
     *
     * @return MoE 模型配置
     */
    public static MiniMindConfig createMoEConfig() {
        MiniMindConfig config = createDefaultConfig();
        config.useMoE = true;
        config.numExperts = 4;
        config.numExpertsPerToken = 1;
        config.routerAuxLossCoef = 5e-4f;
        config.normTopkProb = true;
        config.moeEnableLoadBalance = true;
        return config;
    }

    // ========== 辅助方法 ==========

    /**
     * 获取每个注意力头的维度
     *
     * @return 头维度 (hiddenSize / numHeads)
     */
    public int getHeadDim() {
        if (headDim > 0) {
            return headDim;
        }
        if (hiddenSize % numHeads != 0) {
            throw new IllegalStateException(
                    "hiddenSize(" + hiddenSize + ") must be divisible by numHeads(" + numHeads + ")"
            );
        }
        return hiddenSize / numHeads;
    }

    /**
     * 获取 FFN 中间层维度（SwiGLU intermediate size）
     * 若未显式设置，按 Python 公式自动计算：ceil(hiddenSize * PI / 64) * 64
     *
     * @return 中间层维度
     */
    public int getIntermediateSize() {
        if (intermediateSize > 0) {
            return intermediateSize;
        }
        if (ffnHiddenSize > 0) {
            return ffnHiddenSize;
        }
        return (int) Math.ceil(hiddenSize * Math.PI / 64) * 64;
    }

    /**
     * 获取 MoE 专家的中间层维度
     * 默认等于 intermediateSize
     *
     * @return MoE 中间层维度
     */
    public int getMoeIntermediateSize() {
        if (moeIntermediateSize > 0) {
            return moeIntermediateSize;
        }
        return getIntermediateSize();
    }

    /**
     * 获取 GQA 中 KV 头的重复倍数
     * numHeads / numKVHeads
     *
     * @return KV 头重复倍数
     */
    public int getNumKVGroups() {
        return numHeads / numKVHeads;
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
        if (getIntermediateSize() <= 0) {
            throw new IllegalStateException("intermediateSize must be positive");
        }
        if (numKVHeads <= 0 || numKVHeads > numHeads || numHeads % numKVHeads != 0) {
            throw new IllegalStateException(
                    "numHeads(" + numHeads + ") 必须是 numKVHeads(" + numKVHeads
                            + ") 的整数倍且不小于它");
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
        } else if (hiddenSize == 768 && numLayers == 8) {
            return "Small";
        } else if (hiddenSize == 1024 && numLayers == 16) {
            return "Medium";
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
            // Attention: Q projection + KV projections + Output projection
            // GQA: Q uses numHeads, K/V use numKVHeads
            int kvDim = numKVHeads * getHeadDim();
            params += (long) hiddenSize * hiddenSize; // Q proj
            params += (long) hiddenSize * kvDim * 2;  // K, V proj
            params += (long) hiddenSize * hiddenSize; // Output proj
            // QK Norm: 2 * headDim
            params += (long) getHeadDim() * 2;

            // SwiGLU FFN: gate_proj + up_proj + down_proj
            int ffnDim = getIntermediateSize();
            if (useMoE) {
                int moeFfnDim = getMoeIntermediateSize();
                // MoE: numExperts * (gate + up + down)
                params += (long) numExperts * (hiddenSize * moeFfnDim * 3);
                // Router: hiddenSize * numExperts
                params += (long) hiddenSize * numExperts;
            } else {
                // Standard SwiGLU: gate_proj + up_proj + down_proj
                params += (long) hiddenSize * ffnDim * 3;
            }

            // RMSNorm: 2 * hiddenSize (attention_norm + ffn_norm, only weight, no bias)
            params += (long) hiddenSize * 2;
        }

        // Final RMSNorm: hiddenSize
        params += hiddenSize;

        // LM Head shares weight with Embedding, so not counted separately

        return params;
    }

    // ========== Getter 和 Setter 方法 ==========

    public int getVocabSize() {
        return vocabSize;
    }

    public void setVocabSize(int vocabSize) {
        this.vocabSize = vocabSize;
    }

    public int getMaxSeqLen() {
        return maxSeqLen;
    }

    public void setMaxSeqLen(int maxSeqLen) {
        this.maxSeqLen = maxSeqLen;
    }

    public int getHiddenSize() {
        return hiddenSize;
    }

    public void setHiddenSize(int hiddenSize) {
        this.hiddenSize = hiddenSize;
    }

    public int getNumLayers() {
        return numLayers;
    }

    public void setNumLayers(int numLayers) {
        this.numLayers = numLayers;
    }

    public int getNumHeads() {
        return numHeads;
    }

    /**
     * 设置 Q 头数
     * <p>
     * 同时维护 GQA 不变量：{@code numKVHeads} 必须为正、不超过 {@code numHeads}，
     * 且能整除 {@code numHeads}。单独修改 numHeads 很容易破坏它——例如只把默认的 8
     * 改成 2，而 numKVHeads 仍是默认的 4，此时 {@code numKVGroups = numHeads / numKVHeads = 0}，
     * 注意力里的 {@code repeatKV}（条件为 numKVGroups > 1）被跳过，K/V 的头数多于 Q，
     * 最终在 {@code batchedMatMul} 的 reshape 处抛出难以定位的"形状大小不匹配"。
     * <p>
     * 不变量被破坏时<b>静默</b>退化为 MHA（numKVHeads = numHeads）：
     * "先 setNumHeads 再 setNumKVHeads" 是完全正常的配置序列（中途必然存在不一致的
     * 瞬态），在这里告警会对合法用法误报；而退化后的值会直接体现在
     * {@link #toString()} 与 {@link #validate()} 里，并不隐蔽。若显式设置非法的
     * numKVHeads，{@link #setNumKVHeads(int)} 会直接报错。
     */
    public void setNumHeads(int numHeads) {
        this.numHeads = numHeads;
        if (numHeads > 0 && (this.numKVHeads <= 0 || this.numKVHeads > numHeads
                || numHeads % this.numKVHeads != 0)) {
            this.numKVHeads = numHeads;
        }
    }

    public int getFfnHiddenSize() {
        return getIntermediateSize();
    }

    public void setFfnHiddenSize(int ffnHiddenSize) {
        this.ffnHiddenSize = ffnHiddenSize;
        this.intermediateSize = ffnHiddenSize;
    }

    public void setIntermediateSize(int intermediateSize) {
        this.intermediateSize = intermediateSize;
    }

    public int getNumKVHeads() {
        return numKVHeads;
    }

    /**
     * 设置 KV 头数（GQA）
     * <p>
     * 必须满足 {@code 0 < numKVHeads <= numHeads} 且 {@code numHeads % numKVHeads == 0}，
     * 否则直接报错：这是调用方的显式意图，不适合静默纠正。
     *
     * @throws IllegalArgumentException 不构成合法 GQA 配置
     */
    public void setNumKVHeads(int numKVHeads) {
        if (numKVHeads <= 0) {
            throw new IllegalArgumentException("numKVHeads must be positive, got " + numKVHeads);
        }
        if (numHeads > 0 && (numKVHeads > numHeads || numHeads % numKVHeads != 0)) {
            throw new IllegalArgumentException(
                    "numHeads(" + numHeads + ") 必须是 numKVHeads(" + numKVHeads
                            + ") 的整数倍且不小于它；请先调用 setNumHeads 再设置 numKVHeads");
        }
        this.numKVHeads = numKVHeads;
    }

    public void setHeadDim(int headDim) {
        this.headDim = headDim;
    }

    public int getMaxPositionEmbeddings() {
        return maxPositionEmbeddings;
    }

    public void setMaxPositionEmbeddings(int maxPositionEmbeddings) {
        this.maxPositionEmbeddings = maxPositionEmbeddings;
    }

    public boolean isFlashAttn() {
        return flashAttn;
    }

    public void setFlashAttn(boolean flashAttn) {
        this.flashAttn = flashAttn;
    }

    public float getRouterAuxLossCoef() {
        return routerAuxLossCoef;
    }

    public void setRouterAuxLossCoef(float routerAuxLossCoef) {
        this.routerAuxLossCoef = routerAuxLossCoef;
    }

    public boolean isNormTopkProb() {
        return normTopkProb;
    }

    public void setNormTopkProb(boolean normTopkProb) {
        this.normTopkProb = normTopkProb;
    }

    public void setMoeIntermediateSize(int moeIntermediateSize) {
        this.moeIntermediateSize = moeIntermediateSize;
    }

    public int getBosTokenId() {
        return bosTokenId;
    }

    public void setBosTokenId(int bosTokenId) {
        this.bosTokenId = bosTokenId;
    }

    public float getDropout() {
        return dropout;
    }

    public void setDropout(float dropout) {
        this.dropout = dropout;
    }

    public float getAttentionDropout() {
        return attentionDropout;
    }

    public void setAttentionDropout(float attentionDropout) {
        this.attentionDropout = attentionDropout;
    }

    public float getEpsilon() {
        return epsilon;
    }

    public void setEpsilon(float epsilon) {
        this.epsilon = epsilon;
    }

    public String getActivationFunction() {
        return activationFunction;
    }

    public void setActivationFunction(String activationFunction) {
        this.activationFunction = activationFunction;
    }

    public boolean isUseRoPE() {
        return useRoPE;
    }

    public void setUseRoPE(boolean useRoPE) {
        this.useRoPE = useRoPE;
    }

    public boolean isPreLayerNorm() {
        return preLayerNorm;
    }

    public void setPreLayerNorm(boolean preLayerNorm) {
        this.preLayerNorm = preLayerNorm;
    }

    public float getRopeTheta() {
        return ropeTheta;
    }

    public void setRopeTheta(float ropeTheta) {
        this.ropeTheta = ropeTheta;
    }

    public boolean isUseBias() {
        return useBias;
    }

    public void setUseBias(boolean useBias) {
        this.useBias = useBias;
    }

    public boolean isUseMoE() {
        return useMoE;
    }

    public void setUseMoE(boolean useMoE) {
        this.useMoE = useMoE;
    }

    public int getNumExperts() {
        return numExperts;
    }

    public void setNumExperts(int numExperts) {
        this.numExperts = numExperts;
    }

    public int getNumExpertsPerToken() {
        return numExpertsPerToken;
    }

    public void setNumExpertsPerToken(int numExpertsPerToken) {
        this.numExpertsPerToken = numExpertsPerToken;
    }

    public float getMoeLoadBalanceWeight() {
        return routerAuxLossCoef;
    }

    public void setMoeLoadBalanceWeight(float moeLoadBalanceWeight) {
        this.routerAuxLossCoef = moeLoadBalanceWeight;
    }

    public float getMoeNoiseFactor() {
        return moeNoiseFactor;
    }

    public void setMoeNoiseFactor(float moeNoiseFactor) {
        this.moeNoiseFactor = moeNoiseFactor;
    }

    public float getMoeImportanceCoef() {
        return moeImportanceCoef;
    }

    public void setMoeImportanceCoef(float moeImportanceCoef) {
        this.moeImportanceCoef = moeImportanceCoef;
    }

    public float getMoeLoadCoef() {
        return moeLoadCoef;
    }

    public void setMoeLoadCoef(float moeLoadCoef) {
        this.moeLoadCoef = moeLoadCoef;
    }

    public boolean isMoeSharedExperts() {
        return moeSharedExperts;
    }

    public void setMoeSharedExperts(boolean moeSharedExperts) {
        this.moeSharedExperts = moeSharedExperts;
    }

    public boolean isMoeEnableLoadBalance() {
        return moeEnableLoadBalance;
    }

    public void setMoeEnableLoadBalance(boolean moeEnableLoadBalance) {
        this.moeEnableLoadBalance = moeEnableLoadBalance;
    }

    public int getEosTokenId() {
        return eosTokenId;
    }

    public void setEosTokenId(int eosTokenId) {
        this.eosTokenId = eosTokenId;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("MiniMindConfig{");
        sb.append("modelSize=").append(getModelSize());
        sb.append(", vocabSize=").append(vocabSize);
        sb.append(", maxSeqLen=").append(maxSeqLen);
        sb.append(", hiddenSize=").append(hiddenSize);
        sb.append(", numLayers=").append(numLayers);
        sb.append(", numHeads=").append(numHeads);
        sb.append(", intermediateSize=").append(getIntermediateSize());
        sb.append(", numKVHeads=").append(numKVHeads);
        sb.append(", dropout=").append(dropout);
        sb.append(", activation='").append(activationFunction).append('\'');
        sb.append(", useRoPE=").append(useRoPE);
        sb.append(", useMoE=").append(useMoE);

        if (useMoE) {
            sb.append(", numExperts=").append(numExperts);
            sb.append(", numExpertsPerToken=").append(numExpertsPerToken);
            sb.append(", routerAuxLossCoef=").append(routerAuxLossCoef);
            sb.append(", normTopkProb=").append(normTopkProb);
            sb.append(", moeEnableLoadBalance=").append(moeEnableLoadBalance);
        }

        sb.append(", estimatedParams=").append(estimateParameters());
        sb.append('}');
        return sb.toString();
    }
}