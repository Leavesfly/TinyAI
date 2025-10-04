package io.leavesfly.tinyai.qwen3;

/**
 * Qwen3模型配置类
 * 
 * @author 山泽
 * @version 1.0
 */
public class Qwen3Config {
    
    /** 词汇表大小 */
    private int vocabSize = 32000;
    
    /** 隐藏层维度 */
    private int hiddenSize = 2048;
    
    /** 前馈网络中间层维度 */
    private int intermediateSize = 5632;
    
    /** 隐藏层数量 */
    private int numHiddenLayers = 24;
    
    /** 注意力头数量 */
    private int numAttentionHeads = 16;
    
    /** 键值头数量（用于分组查询注意力） */
    private int numKeyValueHeads = 16;
    
    /** 最大位置编码长度 */
    private int maxPositionEmbeddings = 8192;
    
    /** RoPE基础频率 */
    private double ropeTheta = 10000.0;
    
    /** RMSNorm的epsilon */
    private double rmsNormEps = 1e-6;
    
    /** 填充标记ID */
    private int padTokenId = 0;
    
    /** 开始标记ID */
    private int bosTokenId = 1;
    
    /** 结束标记ID */
    private int eosTokenId = 2;
    
    /** 是否共享输入输出嵌入权重 */
    private boolean tieWordEmbeddings = false;
    
    /** 默认构造函数 */
    public Qwen3Config() {
    }
    
    /**
     * 创建小型配置用于测试
     */
    public static Qwen3Config createSmallConfig() {
        Qwen3Config config = new Qwen3Config();
        config.vocabSize = 1000;
        config.hiddenSize = 512;
        config.intermediateSize = 1024;
        config.numHiddenLayers = 4;
        config.numAttentionHeads = 8;
        config.numKeyValueHeads = 8;
        config.maxPositionEmbeddings = 1024;
        return config;
    }
    
    /**
     * 创建演示配置
     */
    public static Qwen3Config createDemoConfig() {
        Qwen3Config config = new Qwen3Config();
        config.vocabSize = 32000;
        config.hiddenSize = 512;
        config.intermediateSize = 1024;
        config.numHiddenLayers = 6;
        config.numAttentionHeads = 8;
        config.numKeyValueHeads = 8;
        config.maxPositionEmbeddings = 2048;
        return config;
    }
    
    // Getter和Setter方法
    public int getVocabSize() {
        return vocabSize;
    }
    
    public void setVocabSize(int vocabSize) {
        this.vocabSize = vocabSize;
    }
    
    public int getHiddenSize() {
        return hiddenSize;
    }
    
    public void setHiddenSize(int hiddenSize) {
        this.hiddenSize = hiddenSize;
    }
    
    public int getIntermediateSize() {
        return intermediateSize;
    }
    
    public void setIntermediateSize(int intermediateSize) {
        this.intermediateSize = intermediateSize;
    }
    
    public int getNumHiddenLayers() {
        return numHiddenLayers;
    }
    
    public void setNumHiddenLayers(int numHiddenLayers) {
        this.numHiddenLayers = numHiddenLayers;
    }
    
    public int getNumAttentionHeads() {
        return numAttentionHeads;
    }
    
    public void setNumAttentionHeads(int numAttentionHeads) {
        this.numAttentionHeads = numAttentionHeads;
    }
    
    public int getNumKeyValueHeads() {
        return numKeyValueHeads;
    }
    
    public void setNumKeyValueHeads(int numKeyValueHeads) {
        this.numKeyValueHeads = numKeyValueHeads;
    }
    
    public int getMaxPositionEmbeddings() {
        return maxPositionEmbeddings;
    }
    
    public void setMaxPositionEmbeddings(int maxPositionEmbeddings) {
        this.maxPositionEmbeddings = maxPositionEmbeddings;
    }
    
    public double getRopeTheta() {
        return ropeTheta;
    }
    
    public void setRopeTheta(double ropeTheta) {
        this.ropeTheta = ropeTheta;
    }
    
    public double getRmsNormEps() {
        return rmsNormEps;
    }
    
    public void setRmsNormEps(double rmsNormEps) {
        this.rmsNormEps = rmsNormEps;
    }
    
    public int getPadTokenId() {
        return padTokenId;
    }
    
    public void setPadTokenId(int padTokenId) {
        this.padTokenId = padTokenId;
    }
    
    public int getBosTokenId() {
        return bosTokenId;
    }
    
    public void setBosTokenId(int bosTokenId) {
        this.bosTokenId = bosTokenId;
    }
    
    public int getEosTokenId() {
        return eosTokenId;
    }
    
    public void setEosTokenId(int eosTokenId) {
        this.eosTokenId = eosTokenId;
    }
    
    public boolean isTieWordEmbeddings() {
        return tieWordEmbeddings;
    }
    
    public void setTieWordEmbeddings(boolean tieWordEmbeddings) {
        this.tieWordEmbeddings = tieWordEmbeddings;
    }
    
    /**
     * 计算头维度
     */
    public int getHeadDim() {
        return hiddenSize / numAttentionHeads;
    }
    
    /**
     * 计算键值组数
     */
    public int getNumKeyValueGroups() {
        return numAttentionHeads / numKeyValueHeads;
    }
    
    /**
     * 验证配置有效性
     */
    public void validate() {
        if (hiddenSize % numAttentionHeads != 0) {
            throw new IllegalArgumentException(
                String.format("hiddenSize (%d) 必须能被 numAttentionHeads (%d) 整除", 
                    hiddenSize, numAttentionHeads));
        }
        
        if (numAttentionHeads % numKeyValueHeads != 0) {
            throw new IllegalArgumentException(
                String.format("numAttentionHeads (%d) 必须能被 numKeyValueHeads (%d) 整除", 
                    numAttentionHeads, numKeyValueHeads));
        }
        
        if (vocabSize <= 0) {
            throw new IllegalArgumentException("vocabSize 必须大于 0");
        }
        
        if (hiddenSize <= 0) {
            throw new IllegalArgumentException("hiddenSize 必须大于 0");
        }
        
        if (numHiddenLayers <= 0) {
            throw new IllegalArgumentException("numHiddenLayers 必须大于 0");
        }
    }
    
    @Override
    public String toString() {
        return String.format(
            "Qwen3Config{vocabSize=%d, hiddenSize=%d, intermediateSize=%d, " +
            "numHiddenLayers=%d, numAttentionHeads=%d, numKeyValueHeads=%d, " +
            "maxPositionEmbeddings=%d, ropeTheta=%.1f, rmsNormEps=%.0e}",
            vocabSize, hiddenSize, intermediateSize, numHiddenLayers,
            numAttentionHeads, numKeyValueHeads, maxPositionEmbeddings,
            ropeTheta, rmsNormEps
        );
    }
}