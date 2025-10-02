package io.leavesfly.tinyai.gpt2;

/**
 * GPT-2模型配置类
 * 
 * 包含GPT-2模型的所有超参数配置，对应Python实现中的GPT2Config
 * 
 * @author 山泽
 * @version 1.0
 */
public class GPT2Config {
    
    /** 词汇表大小，默认50257 (GPT-2默认值) */
    private int vocabSize = 50257;
    
    /** 最大位置数，默认1024 */
    private int nPositions = 1024;
    
    /** 嵌入维度，默认768 */
    private int nEmbd = 768;
    
    /** Transformer层数，默认12 */
    private int nLayer = 12;
    
    /** 注意力头数，默认12 */
    private int nHead = 12;
    
    /** 前馈网络中间层维度，默认4倍嵌入维度 */
    private int nInner = 3072;
    
    /** 激活函数类型，默认"gelu" */
    private String activationFunction = "gelu";
    
    /** 残差dropout概率，默认0.1 */
    private double residPdrop = 0.1;
    
    /** 嵌入dropout概率，默认0.1 */
    private double embdPdrop = 0.1;
    
    /** 注意力dropout概率，默认0.1 */
    private double attnPdrop = 0.1;
    
    /** 层归一化epsilon，默认1e-5 */
    private double layerNormEpsilon = 1e-5;
    
    /** 权重初始化范围，默认0.02 */
    private double initializerRange = 0.02;
    
    /**
     * 默认构造函数，创建标准GPT-2配置
     */
    public GPT2Config() {
        // 使用默认值
    }
    
    /**
     * 完整配置构造函数
     */
    public GPT2Config(int vocabSize, int nPositions, int nEmbd, int nLayer, 
                     int nHead, int nInner, String activationFunction, 
                     double residPdrop, double embdPdrop, double attnPdrop, 
                     double layerNormEpsilon, double initializerRange) {
        this.vocabSize = vocabSize;
        this.nPositions = nPositions;
        this.nEmbd = nEmbd;
        this.nLayer = nLayer;
        this.nHead = nHead;
        this.nInner = nInner;
        this.activationFunction = activationFunction;
        this.residPdrop = residPdrop;
        this.embdPdrop = embdPdrop;
        this.attnPdrop = attnPdrop;
        this.layerNormEpsilon = layerNormEpsilon;
        this.initializerRange = initializerRange;
    }
    
    /**
     * 创建小型GPT-2配置（用于测试和快速训练）
     */
    public static GPT2Config createSmallConfig() {
        GPT2Config config = new GPT2Config();
        config.vocabSize = 5000;     // 较小的词汇表
        config.nPositions = 256;     // 较短的序列长度
        config.nEmbd = 256;          // 较小的嵌入维度
        config.nLayer = 6;           // 较少的层数
        config.nHead = 8;            // 较少的注意力头
        config.nInner = 1024;        // 对应的前馈维度
        return config;
    }
    
    /**
     * 创建中型GPT-2配置
     */
    public static GPT2Config createMediumConfig() {
        GPT2Config config = new GPT2Config();
        config.nEmbd = 1024;
        config.nLayer = 24;
        config.nHead = 16;
        config.nInner = 4096;
        return config;
    }
    
    /**
     * 创建大型GPT-2配置
     */
    public static GPT2Config createLargeConfig() {
        GPT2Config config = new GPT2Config();
        config.nEmbd = 1280;
        config.nLayer = 36;
        config.nHead = 20;
        config.nInner = 5120;
        return config;
    }
    
    /**
     * 验证配置参数的有效性
     */
    public void validate() {
        if (vocabSize <= 0) {
            throw new IllegalArgumentException("词汇表大小必须大于0");
        }
        if (nPositions <= 0) {
            throw new IllegalArgumentException("最大位置数必须大于0");
        }
        if (nEmbd <= 0) {
            throw new IllegalArgumentException("嵌入维度必须大于0");
        }
        if (nLayer <= 0) {
            throw new IllegalArgumentException("Transformer层数必须大于0");
        }
        if (nHead <= 0) {
            throw new IllegalArgumentException("注意力头数必须大于0");
        }
        if (nEmbd % nHead != 0) {
            throw new IllegalArgumentException("嵌入维度必须能被注意力头数整除");
        }
        if (nInner <= 0) {
            throw new IllegalArgumentException("前馈网络维度必须大于0");
        }
        if (residPdrop < 0 || residPdrop >= 1) {
            throw new IllegalArgumentException("残差dropout概率必须在[0,1)范围内");
        }
        if (embdPdrop < 0 || embdPdrop >= 1) {
            throw new IllegalArgumentException("嵌入dropout概率必须在[0,1)范围内");
        }
        if (attnPdrop < 0 || attnPdrop >= 1) {
            throw new IllegalArgumentException("注意力dropout概率必须在[0,1)范围内");
        }
        if (layerNormEpsilon <= 0) {
            throw new IllegalArgumentException("层归一化epsilon必须大于0");
        }
        if (initializerRange <= 0) {
            throw new IllegalArgumentException("初始化范围必须大于0");
        }
    }
    
    // Getter和Setter方法
    
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
    
    public double getResidPdrop() { return residPdrop; }
    public void setResidPdrop(double residPdrop) { this.residPdrop = residPdrop; }
    
    public double getEmbdPdrop() { return embdPdrop; }
    public void setEmbdPdrop(double embdPdrop) { this.embdPdrop = embdPdrop; }
    
    public double getAttnPdrop() { return attnPdrop; }
    public void setAttnPdrop(double attnPdrop) { this.attnPdrop = attnPdrop; }
    
    public double getLayerNormEpsilon() { return layerNormEpsilon; }
    public void setLayerNormEpsilon(double layerNormEpsilon) { this.layerNormEpsilon = layerNormEpsilon; }
    
    public double getInitializerRange() { return initializerRange; }
    public void setInitializerRange(double initializerRange) { this.initializerRange = initializerRange; }
    
    @Override
    public String toString() {
        return String.format("GPT2Config{vocabSize=%d, nPositions=%d, nEmbd=%d, nLayer=%d, " +
                           "nHead=%d, nInner=%d, activationFunction='%s', " +
                           "residPdrop=%.3f, embdPdrop=%.3f, attnPdrop=%.3f, " +
                           "layerNormEpsilon=%.6f, initializerRange=%.3f}",
                           vocabSize, nPositions, nEmbd, nLayer, nHead, nInner, 
                           activationFunction, residPdrop, embdPdrop, attnPdrop, 
                           layerNormEpsilon, initializerRange);
    }
}