package io.leavesfly.tinyai.gpt1;

import java.io.Serializable;

/**
 * GPT-1模型配置类
 * 
 * GPT-1是OpenAI于2018年发布的第一代生成式预训练Transformer模型，
 * 开创了"预训练+微调"的范式，为后续GPT系列奠定了基础。
 * 
 * 核心特点：
 * 1. Post-LayerNorm架构 - 在子层之后应用层归一化
 * 2. 标准的Transformer解码器结构
 * 3. 因果掩码的自注意力机制
 * 4. 较短的序列长度（512）
 * 
 * 本实现完全基于TinyAI框架的V2 API，保持独立性，不依赖GPT-2/GPT-3模块。
 * 
 * @author leavesfly
 * @version 1.0
 */
public class GPT1Config implements Serializable {
    
    private static final long serialVersionUID = 1L;
    
    // ==================== 基础模型配置 ====================
    
    /** 词汇表大小，默认40478（GPT-1原始值，BPE编码） */
    private int vocabSize = 40478;
    
    /** 最大位置数（序列长度），默认512 */
    private int nPositions = 512;
    
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
    
    // ==================== Dropout配置 ====================
    
    /** 残差dropout概率，默认0.1 */
    private double residPdrop = 0.1;
    
    /** 嵌入dropout概率，默认0.1 */
    private double embdPdrop = 0.1;
    
    /** 注意力dropout概率，默认0.1 */
    private double attnPdrop = 0.1;
    
    // ==================== 初始化配置 ====================
    
    /** 层归一化epsilon，默认1e-5 */
    private double layerNormEpsilon = 1e-5;
    
    /** 权重初始化范围，默认0.02 */
    private double initializerRange = 0.02;
    
    /**
     * 默认构造函数，创建标准GPT-1配置
     */
    public GPT1Config() {
        // 使用默认值
    }
    
    /**
     * 完整配置构造函数
     */
    public GPT1Config(int vocabSize, int nPositions, int nEmbd, int nLayer,
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
    
    // ==================== 预设配置工厂方法 ====================
    
    /**
     * 创建标准GPT-1配置（117M参数）
     * 配置：768维, 12层, 12头, 序列长度512
     * 这是GPT-1论文中的标准配置
     */
    public static GPT1Config createStandardConfig() {
        GPT1Config config = new GPT1Config();
        config.setVocabSize(40478);
        config.setNEmbd(768);
        config.setNLayer(12);
        config.setNHead(12);
        config.setNInner(3072);
        config.setNPositions(512);
        return config;
    }
    
    /**
     * 创建微型GPT-1配置（用于快速测试）
     * 配置：256维, 6层, 8头, 序列长度128
     */
    public static GPT1Config createTinyConfig() {
        GPT1Config config = new GPT1Config();
        config.setVocabSize(10000);
        config.setNEmbd(256);
        config.setNLayer(6);
        config.setNHead(8);
        config.setNInner(1024);
        config.setNPositions(128);
        return config;
    }
    
    /**
     * 创建小型GPT-1配置（用于学习和实验）
     * 配置：512维, 8层, 8头, 序列长度256
     */
    public static GPT1Config createSmallConfig() {
        GPT1Config config = new GPT1Config();
        config.setVocabSize(20000);
        config.setNEmbd(512);
        config.setNLayer(8);
        config.setNHead(8);
        config.setNInner(2048);
        config.setNPositions(256);
        return config;
    }
    
    // ==================== 验证方法 ====================
    
    /**
     * 验证配置的有效性
     * 
     * @throws IllegalArgumentException 如果配置无效
     */
    public void validate() {
        if (vocabSize <= 0) {
            throw new IllegalArgumentException("词汇表大小必须大于0，实际: " + vocabSize);
        }
        if (nPositions <= 0) {
            throw new IllegalArgumentException("最大位置数必须大于0，实际: " + nPositions);
        }
        if (nEmbd <= 0) {
            throw new IllegalArgumentException("嵌入维度必须大于0，实际: " + nEmbd);
        }
        if (nLayer <= 0) {
            throw new IllegalArgumentException("层数必须大于0，实际: " + nLayer);
        }
        if (nHead <= 0) {
            throw new IllegalArgumentException("注意力头数必须大于0，实际: " + nHead);
        }
        if (nEmbd % nHead != 0) {
            throw new IllegalArgumentException(
                String.format("嵌入维度(%d)必须能被注意力头数(%d)整除", nEmbd, nHead)
            );
        }
        if (nInner <= 0) {
            throw new IllegalArgumentException("前馈网络维度必须大于0，实际: " + nInner);
        }
        if (residPdrop < 0 || residPdrop >= 1) {
            throw new IllegalArgumentException("残差dropout概率必须在[0,1)范围内，实际: " + residPdrop);
        }
        if (embdPdrop < 0 || embdPdrop >= 1) {
            throw new IllegalArgumentException("嵌入dropout概率必须在[0,1)范围内，实际: " + embdPdrop);
        }
        if (attnPdrop < 0 || attnPdrop >= 1) {
            throw new IllegalArgumentException("注意力dropout概率必须在[0,1)范围内，实际: " + attnPdrop);
        }
    }
    
    /**
     * 估算模型参数数量
     * 
     * @return 估算的参数数量
     */
    public long estimateParameterCount() {
        // Token嵌入: vocabSize * nEmbd
        long tokenEmbed = (long) vocabSize * nEmbd;
        
        // 位置嵌入: nPositions * nEmbd
        long posEmbed = (long) nPositions * nEmbd;
        
        // 每个Transformer块的参数
        // 注意力: QKV投影(3 * nEmbd * nEmbd) + 输出投影(nEmbd * nEmbd) + 偏置(4 * nEmbd)
        long attnParams = 4L * nEmbd * nEmbd + 4L * nEmbd;
        
        // LayerNorm1: gamma(nEmbd) + beta(nEmbd)
        long ln1Params = 2L * nEmbd;
        
        // 前馈网络: fc1(nEmbd * nInner + nInner) + fc2(nInner * nEmbd + nEmbd)
        long ffnParams = (long) nEmbd * nInner + nInner + (long) nInner * nEmbd + nEmbd;
        
        // LayerNorm2: gamma(nEmbd) + beta(nEmbd)
        long ln2Params = 2L * nEmbd;
        
        // 每层总参数
        long paramsPerLayer = attnParams + ln1Params + ffnParams + ln2Params;
        
        // 所有层的参数
        long allLayersParams = paramsPerLayer * nLayer;
        
        // 最终LayerNorm: gamma(nEmbd) + beta(nEmbd)
        long finalLnParams = 2L * nEmbd;
        
        // 输出投影: nEmbd * vocabSize（通常与token嵌入权重共享）
        // 这里不重复计算
        
        // 总参数 = 嵌入 + 所有层 + 最终LN
        return tokenEmbed + posEmbed + allLayersParams + finalLnParams;
    }
    
    // ==================== Getter和Setter方法 ====================
    
    public int getVocabSize() {
        return vocabSize;
    }
    
    public void setVocabSize(int vocabSize) {
        this.vocabSize = vocabSize;
    }
    
    public int getNPositions() {
        return nPositions;
    }
    
    public void setNPositions(int nPositions) {
        this.nPositions = nPositions;
    }
    
    public int getNEmbd() {
        return nEmbd;
    }
    
    public void setNEmbd(int nEmbd) {
        this.nEmbd = nEmbd;
    }
    
    public int getNLayer() {
        return nLayer;
    }
    
    public void setNLayer(int nLayer) {
        this.nLayer = nLayer;
    }
    
    public int getNHead() {
        return nHead;
    }
    
    public void setNHead(int nHead) {
        this.nHead = nHead;
    }
    
    public int getNInner() {
        return nInner;
    }
    
    public void setNInner(int nInner) {
        this.nInner = nInner;
    }
    
    public String getActivationFunction() {
        return activationFunction;
    }
    
    public void setActivationFunction(String activationFunction) {
        this.activationFunction = activationFunction;
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
    
    @Override
    public String toString() {
        return String.format(
            "GPT1Config{\n" +
            "  vocabSize=%d,\n" +
            "  nPositions=%d,\n" +
            "  nEmbd=%d,\n" +
            "  nLayer=%d,\n" +
            "  nHead=%d,\n" +
            "  nInner=%d,\n" +
            "  activationFunction='%s',\n" +
            "  residPdrop=%.3f,\n" +
            "  embdPdrop=%.3f,\n" +
            "  attnPdrop=%.3f,\n" +
            "  layerNormEpsilon=%.1e,\n" +
            "  initializerRange=%.3f,\n" +
            "  estimatedParams=%s\n" +
            "}",
            vocabSize, nPositions, nEmbd, nLayer, nHead, nInner,
            activationFunction, residPdrop, embdPdrop, attnPdrop,
            layerNormEpsilon, initializerRange,
            formatParamCount(estimateParameterCount())
        );
    }
    
    /**
     * 格式化参数数量显示
     */
    private String formatParamCount(long count) {
        if (count >= 1_000_000_000) {
            return String.format("%.2fB", count / 1_000_000_000.0);
        } else if (count >= 1_000_000) {
            return String.format("%.2fM", count / 1_000_000.0);
        } else if (count >= 1_000) {
            return String.format("%.2fK", count / 1_000.0);
        } else {
            return String.format("%d", count);
        }
    }
}
