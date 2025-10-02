package io.leavesfly.tinyai.qwen3;

/**
 * Qwen3模型配置类
 * 
 * 该类定义了Qwen3大语言模型的所有关键配置参数，
 * 包括词汇表大小、模型维度、层数、注意力头数等。
 * 
 * @author 山泽
 * @version 1.0
 */
public class Qwen3Config {
    
    /**
     * 词汇表大小
     */
    private int vocabSize = 151936;
    
    /**
     * 隐藏层维度
     */
    private int hiddenSize = 4096;
    
    /**
     * Transformer层数
     */
    private int numHiddenLayers = 32;
    
    /**
     * 注意力头数
     */
    private int numAttentionHeads = 32;
    
    /**
     * KV头数（用于分组查询注意力GQA）
     */
    private int numKeyValueHeads = 32;
    
    /**
     * 前馈网络中间层维度
     */
    private int intermediateSize = 11008;
    
    /**
     * 最大位置编码长度
     */
    private int maxPositionEmbeddings = 32768;
    
    /**
     * RMSNorm的epsilon值
     */
    private float rmsNormEps = 1e-6f;
    
    /**
     * RoPE的theta参数
     */
    private float ropeTheta = 1000000.0f;
    
    /**
     * 注意力dropout概率
     */
    private float attentionDropout = 0.0f;
    
    /**
     * 隐藏层dropout概率
     */
    private float hiddenDropout = 0.0f;
    
    /**
     * 是否使用KV缓存
     */
    private boolean useCache = true;
    
    /**
     * padding token id
     */
    private int padTokenId = 151643;
    
    /**
     * 开始token id
     */
    private int bosTokenId = 151643;
    
    /**
     * 结束token id
     */
    private int eosTokenId = 151645;

    /**
     * 默认构造函数
     */
    public Qwen3Config() {
    }

    /**
     * 创建小型配置用于演示和测试
     * 
     * @return 小型Qwen3配置
     */
    public static Qwen3Config createTinyConfig() {
        Qwen3Config config = new Qwen3Config();
        config.vocabSize = 1000;
        config.hiddenSize = 256;
        config.numHiddenLayers = 4;
        config.numAttentionHeads = 8;
        config.numKeyValueHeads = 8;
        config.intermediateSize = 512;
        config.maxPositionEmbeddings = 512;
        return config;
    }

    // Getters and Setters
    
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

    public int getIntermediateSize() {
        return intermediateSize;
    }

    public void setIntermediateSize(int intermediateSize) {
        this.intermediateSize = intermediateSize;
    }

    public int getMaxPositionEmbeddings() {
        return maxPositionEmbeddings;
    }

    public void setMaxPositionEmbeddings(int maxPositionEmbeddings) {
        this.maxPositionEmbeddings = maxPositionEmbeddings;
    }

    public float getRmsNormEps() {
        return rmsNormEps;
    }

    public void setRmsNormEps(float rmsNormEps) {
        this.rmsNormEps = rmsNormEps;
    }

    public float getRopeTheta() {
        return ropeTheta;
    }

    public void setRopeTheta(float ropeTheta) {
        this.ropeTheta = ropeTheta;
    }

    public float getAttentionDropout() {
        return attentionDropout;
    }

    public void setAttentionDropout(float attentionDropout) {
        this.attentionDropout = attentionDropout;
    }

    public float getHiddenDropout() {
        return hiddenDropout;
    }

    public void setHiddenDropout(float hiddenDropout) {
        this.hiddenDropout = hiddenDropout;
    }

    public boolean isUseCache() {
        return useCache;
    }

    public void setUseCache(boolean useCache) {
        this.useCache = useCache;
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

    /**
     * 获取每个注意力头的维度
     * 
     * @return 注意力头维度
     */
    public int getHeadDim() {
        return hiddenSize / numAttentionHeads;
    }

    /**
     * 获取KV头维度
     * 
     * @return KV头维度
     */
    public int getKvHeadDim() {
        return hiddenSize / numKeyValueHeads;
    }

    /**
     * 获取注意力头分组数（用于分组查询注意力）
     * 
     * @return 分组数
     */
    public int getNumKeyValueGroups() {
        return numAttentionHeads / numKeyValueHeads;
    }

    @Override
    public String toString() {
        return "Qwen3Config{" +
                "vocabSize=" + vocabSize +
                ", hiddenSize=" + hiddenSize +
                ", numHiddenLayers=" + numHiddenLayers +
                ", numAttentionHeads=" + numAttentionHeads +
                ", numKeyValueHeads=" + numKeyValueHeads +
                ", intermediateSize=" + intermediateSize +
                ", maxPositionEmbeddings=" + maxPositionEmbeddings +
                ", rmsNormEps=" + rmsNormEps +
                ", ropeTheta=" + ropeTheta +
                ", attentionDropout=" + attentionDropout +
                ", hiddenDropout=" + hiddenDropout +
                ", useCache=" + useCache +
                ", padTokenId=" + padTokenId +
                ", bosTokenId=" + bosTokenId +
                ", eosTokenId=" + eosTokenId +
                '}';
    }
}