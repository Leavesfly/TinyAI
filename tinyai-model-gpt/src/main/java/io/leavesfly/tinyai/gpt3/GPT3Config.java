package io.leavesfly.tinyai.gpt3;

/**
 * GPT-3模型配置类
 * 
 * 定义GPT-3模型的超参数和配置选项
 * 支持不同规模的GPT-3模型配置
 * 
 * @author 山泽
 * @version 1.0
 */
public class GPT3Config {
    
    // 基础模型参数
    private int vocabSize = 50257;          // 词汇表大小
    private int nPositions = 2048;          // 最大序列长度
    private int nEmbd = 12288;              // 嵌入维度（GPT-3 175B）
    private int nLayer = 96;                // Transformer层数（GPT-3 175B）
    private int nHead = 96;                 // 注意力头数（GPT-3 175B）
    private int nInner;                     // 前馈网络隐藏层维度（通常是nEmbd的4倍）
    
    // 激活函数和正则化
    private String activationFunction = "gelu_new";  // 激活函数类型
    private double residDropout = 0.1;              // 残差连接dropout率
    private double embdDropout = 0.1;               // 嵌入层dropout率
    private double attnDropout = 0.1;               // 注意力dropout率
    private double layerNormEpsilon = 1e-5;         // 层归一化epsilon
    private double initializerRange = 0.02;         // 参数初始化范围
    
    // GPT-3特有参数
    private boolean sparseAttention = false;        // 是否使用稀疏注意力
    private boolean gradientCheckpointing = true;   // 梯度检查点（节省内存）
    private boolean parallelAttention = true;       // 并行注意力和MLP计算
    private double rotaryPct = 0.25;               // 旋转位置编码比例
    private boolean useCache = true;               // 是否使用KV缓存
    
    /**
     * 默认构造函数，使用GPT-3 175B配置
     */
    public GPT3Config() {
        this.nInner = 4 * nEmbd;  // 默认是嵌入维度的4倍
    }
    
    /**
     * 自定义构造函数
     */
    public GPT3Config(int vocabSize, int nPositions, int nEmbd, int nLayer, int nHead) {
        this.vocabSize = vocabSize;
        this.nPositions = nPositions;
        this.nEmbd = nEmbd;
        this.nLayer = nLayer;
        this.nHead = nHead;
        this.nInner = 4 * nEmbd;
        
        // 验证配置
        validate();
    }
    
    /**
     * 创建小型GPT-3配置（125M参数）
     */
    public static GPT3Config createSmallConfig() {
        GPT3Config config = new GPT3Config();
        config.nEmbd = 768;
        config.nLayer = 12;
        config.nHead = 12;
        config.nInner = 4 * config.nEmbd;
        config.nPositions = 1024;
        config.sparseAttention = false;
        config.gradientCheckpointing = false;
        return config;
    }
    
    /**
     * 创建中型GPT-3配置（350M参数）
     */
    public static GPT3Config createMediumConfig() {
        GPT3Config config = new GPT3Config();
        config.nEmbd = 1024;
        config.nLayer = 24;
        config.nHead = 16;
        config.nInner = 4 * config.nEmbd;
        config.nPositions = 1024;
        config.sparseAttention = false;
        config.gradientCheckpointing = false;
        return config;
    }
    
    /**
     * 创建大型GPT-3配置（1.3B参数）
     */
    public static GPT3Config createLargeConfig() {
        GPT3Config config = new GPT3Config();
        config.nEmbd = 2048;
        config.nLayer = 24;
        config.nHead = 32;
        config.nInner = 4 * config.nEmbd;
        config.nPositions = 2048;
        config.sparseAttention = true;
        config.gradientCheckpointing = true;
        return config;
    }
    
    /**
     * 创建超大型GPT-3配置（175B参数）
     */
    public static GPT3Config createXLConfig() {
        GPT3Config config = new GPT3Config();
        config.nEmbd = 12288;
        config.nLayer = 96;
        config.nHead = 96;
        config.nInner = 4 * config.nEmbd;
        config.nPositions = 2048;
        config.sparseAttention = true;
        config.gradientCheckpointing = true;
        config.parallelAttention = true;
        return config;
    }
    
    /**
     * 验证配置的有效性
     */
    public void validate() {
        if (nEmbd <= 0) {
            throw new IllegalArgumentException("嵌入维度必须大于0");
        }
        if (nLayer <= 0) {
            throw new IllegalArgumentException("层数必须大于0");
        }
        if (nHead <= 0) {
            throw new IllegalArgumentException("注意力头数必须大于0");
        }
        if (nEmbd % nHead != 0) {
            throw new IllegalArgumentException("嵌入维度必须能被注意力头数整除");
        }
        if (vocabSize <= 0) {
            throw new IllegalArgumentException("词汇表大小必须大于0");
        }
        if (nPositions <= 0) {
            throw new IllegalArgumentException("最大序列长度必须大于0");
        }
        if (nInner <= 0) {
            nInner = 4 * nEmbd;  // 自动设置为默认值
        }
    }
    
    /**
     * 计算估算的参数数量
     */
    public long estimateParameterCount() {
        // 嵌入层参数：词汇表嵌入 + 位置嵌入
        long embeddingParams = (long) vocabSize * nEmbd + (long) nPositions * nEmbd;
        
        // 每个Transformer层的参数
        long layerParams = 
            // 注意力层：Q、K、V投影 + 输出投影
            4L * nEmbd * nEmbd +
            // 前馈网络：两个线性层
            (long) nEmbd * nInner + (long) nInner * nEmbd +
            // 层归一化：两个LayerNorm，每个有gamma和beta
            4L * nEmbd;
        
        // 总参数 = 嵌入层 + 所有Transformer层 + 最终LayerNorm
        return embeddingParams + nLayer * layerParams + 2L * nEmbd;
    }
    
    @Override
    public String toString() {
        return String.format(
            "GPT3Config{" +
            "vocabSize=%d, nPositions=%d, nEmbd=%d, nLayer=%d, nHead=%d, " +
            "nInner=%d, activationFunction='%s', sparseAttention=%s, " +
            "estimatedParams=%,d}",
            vocabSize, nPositions, nEmbd, nLayer, nHead, nInner, 
            activationFunction, sparseAttention, estimateParameterCount()
        );
    }
    
    // ==================== Getter和Setter方法 ====================
    
    public int getVocabSize() { return vocabSize; }
    public void setVocabSize(int vocabSize) { this.vocabSize = vocabSize; }
    
    public int getNPositions() { return nPositions; }
    public void setNPositions(int nPositions) { this.nPositions = nPositions; }
    
    public int getNEmbd() { return nEmbd; }
    public void setNEmbd(int nEmbd) { this.nEmbd = nEmbd; }
    
    public int getNLayer() { return nLayer; }
    public void setNLayer(int nLayer) { this.nLayer = nLayer; }
    
    public int getNHead() { return nHead; }
    public void setNHead(int nHead) { this.nHead = nHead; }
    
    public int getNInner() { return nInner; }
    public void setNInner(int nInner) { this.nInner = nInner; }
    
    public String getActivationFunction() { return activationFunction; }
    public void setActivationFunction(String activationFunction) { this.activationFunction = activationFunction; }
    
    public double getResidDropout() { return residDropout; }
    public void setResidDropout(double residDropout) { this.residDropout = residDropout; }
    
    public double getEmbdDropout() { return embdDropout; }
    public void setEmbdDropout(double embdDropout) { this.embdDropout = embdDropout; }
    
    public double getAttnDropout() { return attnDropout; }
    public void setAttnDropout(double attnDropout) { this.attnDropout = attnDropout; }
    
    public double getLayerNormEpsilon() { return layerNormEpsilon; }
    public void setLayerNormEpsilon(double layerNormEpsilon) { this.layerNormEpsilon = layerNormEpsilon; }
    
    public double getInitializerRange() { return initializerRange; }
    public void setInitializerRange(double initializerRange) { this.initializerRange = initializerRange; }
    
    public boolean isSparseAttention() { return sparseAttention; }
    public void setSparseAttention(boolean sparseAttention) { this.sparseAttention = sparseAttention; }
    
    public boolean isGradientCheckpointing() { return gradientCheckpointing; }
    public void setGradientCheckpointing(boolean gradientCheckpointing) { this.gradientCheckpointing = gradientCheckpointing; }
    
    public boolean isParallelAttention() { return parallelAttention; }
    public void setParallelAttention(boolean parallelAttention) { this.parallelAttention = parallelAttention; }
    
    public double getRotaryPct() { return rotaryPct; }
    public void setRotaryPct(double rotaryPct) { this.rotaryPct = rotaryPct; }
    
    public boolean isUseCache() { return useCache; }
    public void setUseCache(boolean useCache) { this.useCache = useCache; }
}