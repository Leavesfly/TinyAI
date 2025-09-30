package io.leavesfly.tinyai.nlp.deepseekV3;

/**
 * DeepSeek-V3 模型配置类
 * 
 * 封装了DeepSeek-V3模型的所有配置参数，基于DeepSeek-V3技术报告的架构设计。
 * 
 * DeepSeek-V3的核心特性：
 * 1. Multi-head Latent Attention (MLA) - 减少KV缓存内存占用
 * 2. DeepSeekMoE架构 - 高效的专家混合机制
 * 3. FP8混合精度训练 - 提升训练效率
 * 4. 671B总参数，37B激活参数
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekV3Config {
    
    // ========== 基础模型参数 ==========
    private int vocabSize;              // 词汇表大小
    private int dModel;                 // 模型维度
    private int numLayers;              // Transformer层数
    private int numHeads;               // 注意力头数
    private int maxSeqLength;           // 最大序列长度
    
    // ========== MLA相关参数 ==========
    private int dMLA;                   // MLA潜在维度
    private int qkNormDim;              // QK归一化维度
    private boolean useMLACache;        // 是否使用MLA缓存压缩
    private double cacheCompressionRatio; // 缓存压缩比率
    
    // ========== MoE相关参数 ==========
    private int numExperts;             // 每层专家数量
    private int dExpert;                // 专家隐藏层维度
    private int topK;                   // Top-K专家选择
    private boolean useSharedExperts;   // 是否使用共享专家
    private int numSharedExperts;       // 共享专家数量
    private double expertLoadBalanceWeight; // 专家负载均衡权重
    
    // ========== DeepSeekMoE增强参数 ==========
    private boolean useExpertRouting;   // 是否使用专家路由优化
    private boolean useAuxiliaryLoss;   // 是否使用辅助损失
    private double routingProbabilityThreshold; // 路由概率阈值
    private boolean useExpertDropout;   // 是否使用专家dropout
    private double expertDropoutRate;   // 专家dropout率
    
    // ========== 训练优化参数 ==========
    private boolean useFP8Training;     // 是否使用FP8混合精度训练
    private double dropoutRate;         // 标准Dropout比率
    private double learningRate;        // 学习率
    private int batchSize;             // 批大小
    private int numEpochs;             // 训练轮数
    private boolean useGradientCheckpointing; // 是否使用梯度检查点
    
    // ========== RoPE位置编码参数 ==========
    private boolean useRoPE;            // 是否使用RoPE位置编码
    private double ropeTheta;           // RoPE的theta参数
    private int ropeMaxPosition;        // RoPE最大位置
    
    // ========== 层归一化参数 ==========
    private boolean useLayerNorm;       // 是否使用层归一化
    private boolean useRMSNorm;         // 是否使用RMSNorm
    private double layerNormEps;        // 层归一化epsilon
    
    // ========== 激活函数参数 ==========
    private String activationFunction;  // 激活函数类型 (SwiGLU, GELU, etc.)
    private boolean useBias;            // 是否使用偏置
    
    // ========== 推理优化参数 ==========
    private boolean useFlashAttention;  // 是否使用FlashAttention
    private boolean useKVCache;         // 是否使用KV缓存
    private int kvCacheMaxTokens;       // KV缓存最大token数
    
    /**
     * 默认构造函数 - 使用DeepSeek-V3标准配置
     */
    public DeepSeekV3Config() {
        // 基础参数 (类似GPT-2基础配置)
        this.vocabSize = 102400;        // DeepSeek系列常用词汇表大小
        this.dModel = 2048;             // 中等规模模型维度
        this.numLayers = 28;            // DeepSeek-V3的层数
        this.numHeads = 16;             // 注意力头数
        this.maxSeqLength = 4096;       // 最大序列长度
        
        // MLA参数
        this.dMLA = 512;                // MLA潜在维度
        this.qkNormDim = 256;           // QK归一化维度
        this.useMLACache = true;        // 启用MLA缓存压缩
        this.cacheCompressionRatio = 0.25; // 4倍压缩
        
        // MoE参数
        this.numExperts = 64;           // 专家数量
        this.dExpert = 1408;            // 专家隐藏层维度
        this.topK = 6;                  // Top-6专家选择
        this.useSharedExperts = true;   // 使用共享专家
        this.numSharedExperts = 2;      // 2个共享专家
        this.expertLoadBalanceWeight = 0.001; // 负载均衡权重
        
        // DeepSeekMoE增强
        this.useExpertRouting = true;
        this.useAuxiliaryLoss = true;
        this.routingProbabilityThreshold = 0.01;
        this.useExpertDropout = true;
        this.expertDropoutRate = 0.1;
        
        // 训练参数
        this.useFP8Training = true;     // 启用FP8训练
        this.dropoutRate = 0.1;
        this.learningRate = 0.0001;
        this.batchSize = 32;
        this.numEpochs = 100;
        this.useGradientCheckpointing = true;
        
        // RoPE参数
        this.useRoPE = true;
        this.ropeTheta = 10000.0;
        this.ropeMaxPosition = 4096;
        
        // 归一化参数
        this.useLayerNorm = false;      // DeepSeek-V3使用RMSNorm
        this.useRMSNorm = true;
        this.layerNormEps = 1e-6;
        
        // 激活函数
        this.activationFunction = "SwiGLU"; // DeepSeek-V3使用SwiGLU
        this.useBias = false;           // 现代架构通常不使用偏置
        
        // 推理优化
        this.useFlashAttention = true;
        this.useKVCache = true;
        this.kvCacheMaxTokens = 8192;
    }
    
    /**
     * 自定义构造函数
     */
    public DeepSeekV3Config(int vocabSize, int dModel, int numLayers, int numHeads,
                           int maxSeqLength, int numExperts, int topK) {
        this();
        this.vocabSize = vocabSize;
        this.dModel = dModel;
        this.numLayers = numLayers;
        this.numHeads = numHeads;
        this.maxSeqLength = maxSeqLength;
        this.numExperts = numExperts;
        this.topK = topK;
        
        // 自适应调整相关参数
        adjustMLADimensions();
        adjustExpertDimensions();
        
        validate();
    }
    
    /**
     * 创建迷你版配置 - 用于测试和开发
     */
    public static DeepSeekV3Config createTinyConfig() {
        DeepSeekV3Config config = new DeepSeekV3Config();
        config.vocabSize = 1024;
        config.dModel = 256;
        config.numLayers = 4;
        config.numHeads = 4;
        config.maxSeqLength = 128;
        config.dMLA = 64;
        config.qkNormDim = 32;
        config.numExperts = 4;
        config.dExpert = 512;
        config.topK = 2;
        config.batchSize = 8;
        
        config.adjustMLADimensions();
        config.adjustExpertDimensions();
        return config;
    }
    
    /**
     * 创建小型配置
     */
    public static DeepSeekV3Config createSmallConfig() {
        DeepSeekV3Config config = new DeepSeekV3Config();
        config.vocabSize = 32000;
        config.dModel = 768;
        config.numLayers = 12;
        config.numHeads = 12;
        config.maxSeqLength = 2048;
        config.dMLA = 192;
        config.qkNormDim = 96;
        config.numExperts = 16;
        config.dExpert = 2048;
        config.topK = 4;
        config.batchSize = 16;
        
        config.adjustMLADimensions();
        config.adjustExpertDimensions();
        return config;
    }
    
    /**
     * 创建标准配置 - 接近真实DeepSeek-V3规模
     */
    public static DeepSeekV3Config createStandardConfig() {
        return new DeepSeekV3Config(); // 使用默认配置
    }
    
    /**
     * 自适应调整MLA维度
     */
    private void adjustMLADimensions() {
        // MLA潜在维度通常是模型维度的1/4到1/8
        this.dMLA = Math.max(64, this.dModel / 4);
        this.qkNormDim = Math.max(32, this.dMLA / 2);
    }
    
    /**
     * 自适应调整专家维度
     */
    private void adjustExpertDimensions() {
        // 专家隐藏层维度通常是模型维度的2.7倍（DeepSeek惯例）
        this.dExpert = (int) (this.dModel * 2.7);
    }
    
    /**
     * 参数验证
     */
    public void validate() {
        if (vocabSize <= 0 || dModel <= 0 || numLayers <= 0 || numHeads <= 0 ||
            maxSeqLength <= 0 || numExperts <= 0 || topK <= 0) {
            throw new IllegalArgumentException("所有大小参数必须大于0");
        }
        
        if (dModel % numHeads != 0) {
            throw new IllegalArgumentException("dModel必须能被numHeads整除");
        }
        
        if (topK > numExperts) {
            throw new IllegalArgumentException("topK不能超过专家总数");
        }
        
        if (dropoutRate < 0.0 || dropoutRate > 1.0) {
            throw new IllegalArgumentException("dropout比率必须在0.0到1.0之间");
        }
        
        if (expertLoadBalanceWeight < 0.0) {
            throw new IllegalArgumentException("专家负载均衡权重必须非负");
        }
        
        if (numSharedExperts < 0 || numSharedExperts > numExperts) {
            throw new IllegalArgumentException("共享专家数量必须在0到专家总数之间");
        }
    }
    
    /**
     * 计算有效参数数（激活参数）
     */
    public long getActiveParameterCount() {
        long activeParams = 0;
        
        // Token嵌入
        activeParams += (long) vocabSize * dModel;
        
        // 每层的激活参数
        for (int i = 0; i < numLayers; i++) {
            // MLA注意力参数（全部激活）
            activeParams += (long) dModel * dMLA; // Query投影
            activeParams += (long) dMLA * numHeads; // Key/Value投影（压缩）
            activeParams += (long) numHeads * dModel; // 输出投影
            
            // MoE参数（只有topK个专家激活）
            long expertParams = 2L * dModel * dExpert; // 每个专家的参数
            activeParams += topK * expertParams; // 只有topK个专家激活
            
            // 共享专家（全部激活）
            if (useSharedExperts) {
                activeParams += numSharedExperts * expertParams;
            }
            
            // 层归一化和其他参数
            activeParams += 2L * dModel * 2; // 两个层归一化
        }
        
        // 输出头
        activeParams += (long) dModel * vocabSize;
        
        return activeParams;
    }
    
    /**
     * 计算总参数数
     */
    public long getTotalParameterCount() {
        long totalParams = 0;
        
        // Token嵌入
        totalParams += (long) vocabSize * dModel;
        
        // 每层参数
        for (int i = 0; i < numLayers; i++) {
            // MLA注意力参数
            totalParams += (long) dModel * dMLA; // Query投影
            totalParams += (long) dMLA * numHeads; // Key/Value投影
            totalParams += (long) numHeads * dModel; // 输出投影
            
            // MoE参数（所有专家）
            long expertParams = 2L * dModel * dExpert;
            totalParams += numExperts * expertParams;
            
            // 共享专家
            if (useSharedExperts) {
                totalParams += numSharedExperts * expertParams;
            }
            
            // 门控网络
            totalParams += (long) dModel * numExperts;
            
            // 层归一化
            totalParams += 2L * dModel * 2;
        }
        
        // 输出头
        totalParams += (long) dModel * vocabSize;
        
        return totalParams;
    }
    
    /**
     * 获取配置摘要
     */
    public String getSummary() {
        return String.format(
            "DeepSeek-V3 Config Summary:\n" +
            "  Architecture: %d layers, %d dim, %d heads, vocab=%d\n" +
            "  MLA: latent_dim=%d, qk_norm=%d, cache_compress=%.2f\n" +
            "  MoE: %d experts, top-%d, expert_dim=%d, shared=%d\n" +
            "  Sequence: max_len=%d\n" +
            "  Training: fp8=%s, dropout=%.2f, lr=%.4f\n" +
            "  Parameters: total=%s, active=%s (%.1f%%)",
            numLayers, dModel, numHeads, vocabSize,
            dMLA, qkNormDim, cacheCompressionRatio,
            numExperts, topK, dExpert, numSharedExperts,
            maxSeqLength,
            useFP8Training ? "ON" : "OFF", dropoutRate, learningRate,
            formatNumber(getTotalParameterCount()),
            formatNumber(getActiveParameterCount()),
            (double) getActiveParameterCount() / getTotalParameterCount() * 100
        );
    }
    
    /**
     * 打印详细配置信息
     */
    public void printConfig() {
        System.out.println("\n========== DeepSeek-V3 Configuration ==========");
        System.out.println(getSummary());
        System.out.println("\nDetailed Configuration:");
        System.out.printf("  - RoPE: enabled=%s, theta=%.1f\n", useRoPE, ropeTheta);
        System.out.printf("  - Normalization: RMSNorm=%s, eps=%.1e\n", useRMSNorm, layerNormEps);
        System.out.printf("  - Activation: %s, bias=%s\n", activationFunction, useBias);
        System.out.printf("  - FlashAttention: %s\n", useFlashAttention ? "enabled" : "disabled");
        System.out.printf("  - Expert Routing: %s\n", useExpertRouting ? "enabled" : "disabled");
        System.out.printf("  - Auxiliary Loss: %s\n", useAuxiliaryLoss ? "enabled" : "disabled");
        System.out.println("===============================================\n");
    }
    
    /**
     * 格式化数字显示
     */
    private String formatNumber(long number) {
        if (number >= 1_000_000_000L) {
            return String.format("%.1fB", number / 1_000_000_000.0);
        } else if (number >= 1_000_000L) {
            return String.format("%.1fM", number / 1_000_000.0);
        } else if (number >= 1_000L) {
            return String.format("%.1fK", number / 1_000.0);
        } else {
            return String.valueOf(number);
        }
    }
    
    // ========== Getter and Setter Methods ==========
    
    public int getVocabSize() { return vocabSize; }
    public void setVocabSize(int vocabSize) { this.vocabSize = vocabSize; }
    
    public int getDModel() { return dModel; }
    public void setDModel(int dModel) { this.dModel = dModel; }
    
    public int getNumLayers() { return numLayers; }
    public void setNumLayers(int numLayers) { this.numLayers = numLayers; }
    
    public int getNumHeads() { return numHeads; }
    public void setNumHeads(int numHeads) { this.numHeads = numHeads; }
    
    public int getMaxSeqLength() { return maxSeqLength; }
    public void setMaxSeqLength(int maxSeqLength) { this.maxSeqLength = maxSeqLength; }
    
    public int getDMLA() { return dMLA; }
    public void setDMLA(int dMLA) { this.dMLA = dMLA; }
    
    public int getQkNormDim() { return qkNormDim; }
    public void setQkNormDim(int qkNormDim) { this.qkNormDim = qkNormDim; }
    
    public boolean isUseMLACache() { return useMLACache; }
    public void setUseMLACache(boolean useMLACache) { this.useMLACache = useMLACache; }
    
    public double getCacheCompressionRatio() { return cacheCompressionRatio; }
    public void setCacheCompressionRatio(double cacheCompressionRatio) { this.cacheCompressionRatio = cacheCompressionRatio; }
    
    public int getNumExperts() { return numExperts; }
    public void setNumExperts(int numExperts) { this.numExperts = numExperts; }
    
    public int getDExpert() { return dExpert; }
    public void setDExpert(int dExpert) { this.dExpert = dExpert; }
    
    public int getTopK() { return topK; }
    public void setTopK(int topK) { this.topK = topK; }
    
    public boolean isUseSharedExperts() { return useSharedExperts; }
    public void setUseSharedExperts(boolean useSharedExperts) { this.useSharedExperts = useSharedExperts; }
    
    public int getNumSharedExperts() { return numSharedExperts; }
    public void setNumSharedExperts(int numSharedExperts) { this.numSharedExperts = numSharedExperts; }
    
    public double getExpertLoadBalanceWeight() { return expertLoadBalanceWeight; }
    public void setExpertLoadBalanceWeight(double expertLoadBalanceWeight) { this.expertLoadBalanceWeight = expertLoadBalanceWeight; }
    
    public boolean isUseExpertRouting() { return useExpertRouting; }
    public void setUseExpertRouting(boolean useExpertRouting) { this.useExpertRouting = useExpertRouting; }
    
    public boolean isUseAuxiliaryLoss() { return useAuxiliaryLoss; }
    public void setUseAuxiliaryLoss(boolean useAuxiliaryLoss) { this.useAuxiliaryLoss = useAuxiliaryLoss; }
    
    public double getRoutingProbabilityThreshold() { return routingProbabilityThreshold; }
    public void setRoutingProbabilityThreshold(double routingProbabilityThreshold) { this.routingProbabilityThreshold = routingProbabilityThreshold; }
    
    public boolean isUseExpertDropout() { return useExpertDropout; }
    public void setUseExpertDropout(boolean useExpertDropout) { this.useExpertDropout = useExpertDropout; }
    
    public double getExpertDropoutRate() { return expertDropoutRate; }
    public void setExpertDropoutRate(double expertDropoutRate) { this.expertDropoutRate = expertDropoutRate; }
    
    public boolean isUseFP8Training() { return useFP8Training; }
    public void setUseFP8Training(boolean useFP8Training) { this.useFP8Training = useFP8Training; }
    
    public double getDropoutRate() { return dropoutRate; }
    public void setDropoutRate(double dropoutRate) { this.dropoutRate = dropoutRate; }
    
    public double getLearningRate() { return learningRate; }
    public void setLearningRate(double learningRate) { this.learningRate = learningRate; }
    
    public int getBatchSize() { return batchSize; }
    public void setBatchSize(int batchSize) { this.batchSize = batchSize; }
    
    public int getNumEpochs() { return numEpochs; }
    public void setNumEpochs(int numEpochs) { this.numEpochs = numEpochs; }
    
    public boolean isUseGradientCheckpointing() { return useGradientCheckpointing; }
    public void setUseGradientCheckpointing(boolean useGradientCheckpointing) { this.useGradientCheckpointing = useGradientCheckpointing; }
    
    public boolean isUseRoPE() { return useRoPE; }
    public void setUseRoPE(boolean useRoPE) { this.useRoPE = useRoPE; }
    
    public double getRopeTheta() { return ropeTheta; }
    public void setRopeTheta(double ropeTheta) { this.ropeTheta = ropeTheta; }
    
    public int getRopeMaxPosition() { return ropeMaxPosition; }
    public void setRopeMaxPosition(int ropeMaxPosition) { this.ropeMaxPosition = ropeMaxPosition; }
    
    public boolean isUseLayerNorm() { return useLayerNorm; }
    public void setUseLayerNorm(boolean useLayerNorm) { this.useLayerNorm = useLayerNorm; }
    
    public boolean isUseRMSNorm() { return useRMSNorm; }
    public void setUseRMSNorm(boolean useRMSNorm) { this.useRMSNorm = useRMSNorm; }
    
    public double getLayerNormEps() { return layerNormEps; }
    public void setLayerNormEps(double layerNormEps) { this.layerNormEps = layerNormEps; }
    
    public String getActivationFunction() { return activationFunction; }
    public void setActivationFunction(String activationFunction) { this.activationFunction = activationFunction; }
    
    public boolean isUseBias() { return useBias; }
    public void setUseBias(boolean useBias) { this.useBias = useBias; }
    
    public boolean isUseFlashAttention() { return useFlashAttention; }
    public void setUseFlashAttention(boolean useFlashAttention) { this.useFlashAttention = useFlashAttention; }
    
    public boolean isUseKVCache() { return useKVCache; }
    public void setUseKVCache(boolean useKVCache) { this.useKVCache = useKVCache; }
    
    public int getKvCacheMaxTokens() { return kvCacheMaxTokens; }
    public void setKvCacheMaxTokens(int kvCacheMaxTokens) { this.kvCacheMaxTokens = kvCacheMaxTokens; }
}