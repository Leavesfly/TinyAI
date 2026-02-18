package io.leavesfly.tinyai.deepseek.v3;

import io.leavesfly.tinyai.deepseek.base.DeepSeekBaseConfig;

/**
 * DeepSeek-V3模型配置类
 * 
 * DeepSeek-V3是DeepSeek系列的基础模型，采用MoE架构（671B参数，8专家，Top-2路由）。
 * V3通过标准的预训练+后训练（SFT）方式训练，专注于任务感知路由和代码生成优化。
 * 
 * V3与R1的关系：
 * - 共享架构：都使用相同的MoE基础架构（DeepSeekBaseConfig）
 * - 训练差异：V3使用标准训练，R1在V3-Base基础上使用强化学习
 * 
 * V3核心特点：
 * 1. 标准训练流程 - 预训练 + 后训练（SFT）
 * 2. 任务感知路由 - 支持推理、代码、数学、通用、多模态5种任务类型
 * 3. 代码生成优化 - 专门针对代码任务的质量评估和语言识别
 * 4. 推理增强 - 内置推理模块和自我纠错机制
 * 
 * 本实现继承DeepSeekBaseConfig共享基础架构，仅扩展V3特有的代码和推理配置。
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekV3Config extends DeepSeekBaseConfig {
    
    private static final long serialVersionUID = 1L;
    
    // ==================== 推理增强配置（V3特有）====================
    
    /** 推理隐藏层维度，默认为嵌入维度的2倍 */
    private int reasoningHiddenDim = 1536;
    
    /** 推理置信度阈值，默认0.75（V3比R1更严格） */
    private double confidenceThreshold = 0.75;
    
    /** 是否启用自我纠错机制，默认启用 */
    private boolean enableSelfCorrection = true;
    
    // ==================== 代码生成配置（V3特有）====================
    
    /** 代码质量评估维度数量，默认4个维度（语法、结构、可读性、性能） */
    private int codeQualityDim = 4;
    
    /** 支持的编程语言数量，默认10种 */
    private int numProgrammingLanguages = 10;
    
    /** 代码分析隐藏层维度，默认512 */
    private int codeAnalysisHiddenDim = 512;
    
    /** 语法验证器隐藏层维度，默认256 */
    private int syntaxValidatorHiddenDim = 256;
    
    /**
     * 默认构造函数，创建标准DeepSeek-V3配置
     * 使用与R1相同的基础MoE架构
     */
    public DeepSeekV3Config() {
        super(); // 调用父类构造函数
    }
    
    /**
     * 完整配置构造函数
     */
    public DeepSeekV3Config(int vocabSize, int nPositions, int nEmbd, int nLayer,
                           int nHead, int nInner, String activationFunction,
                           int numExperts, int topK, int expertHiddenDim,
                           double loadBalanceLossWeight, double expertDropout,
                           boolean enableTaskAwareRouting, int taskEmbedDim,
                           int taskClassifierHiddenDim, int numTaskTypes,
                           double residPdrop, double embdPdrop, double attnPdrop,
                           double layerNormEpsilon, double initializerRange,
                           int reasoningHiddenDim, double confidenceThreshold,
                           boolean enableSelfCorrection, int codeQualityDim,
                           int numProgrammingLanguages, int codeAnalysisHiddenDim,
                           int syntaxValidatorHiddenDim) {
        super(vocabSize, nPositions, nEmbd, nLayer, nHead, nInner, activationFunction,
              numExperts, topK, expertHiddenDim, loadBalanceLossWeight, expertDropout,
              enableTaskAwareRouting, taskEmbedDim, taskClassifierHiddenDim, numTaskTypes,
              residPdrop, embdPdrop, attnPdrop, layerNormEpsilon, initializerRange);
        
        // 设置V3特有的推理和代码配置
        this.reasoningHiddenDim = reasoningHiddenDim;
        this.confidenceThreshold = confidenceThreshold;
        this.enableSelfCorrection = enableSelfCorrection;
        this.codeQualityDim = codeQualityDim;
        this.numProgrammingLanguages = numProgrammingLanguages;
        this.codeAnalysisHiddenDim = codeAnalysisHiddenDim;
        this.syntaxValidatorHiddenDim = syntaxValidatorHiddenDim;
    }
    
    // ==================== 预设配置工厂方法 ====================
    
    /**
     * 创建标准DeepSeek-V3配置（继承基础MoE架构）
     * 配置：768维, 12层, 12头, 8专家, Top-2路由, 序列长度2048
     */
    public static DeepSeekV3Config createStandardConfig() {
        return new DeepSeekV3Config(); // 使用默认值即可
    }
    
    /**
     * 创建微型DeepSeek-V3配置（用于快速测试）
     * 配置：256维, 6层, 8头, 4专家, Top-2路由, 序列长度512
     */
    public static DeepSeekV3Config createTinyConfig() {
        DeepSeekV3Config config = new DeepSeekV3Config();
        config.setNEmbd(256);
        config.setNLayer(6);
        config.setNHead(8);
        config.setNInner(1024);
        config.setNPositions(512);
        config.setNumExperts(4);
        config.setTopK(2);
        config.setExpertHiddenDim(1024);
        config.setReasoningHiddenDim(512);
        config.setCodeAnalysisHiddenDim(256);
        return config;
    }
    
    /**
     * 创建极小型DeepSeek-V3配置（用于默认JVM堆内存下测试）
     * 配置：64维, 2层, 2头, 2专家, Top-1路由, 序列长度32
     * 参数量：约200K，适合在默认JVM内存下运行
     */
    public static DeepSeekV3Config createMicroConfig() {
        DeepSeekV3Config config = new DeepSeekV3Config();
        config.setVocabSize(1000);
        config.setNEmbd(64);
        config.setNLayer(2);
        config.setNHead(2);
        config.setNInner(128);
        config.setNPositions(32);
        config.setNumExperts(2);
        config.setTopK(1);
        config.setExpertHiddenDim(128);
        config.setReasoningHiddenDim(64);
        config.setCodeAnalysisHiddenDim(32);
        return config;
    }
    
    /**
     * 创建小型DeepSeek-V3配置（用于学习和实验）
     * 配置：512维, 8层, 8头, 6专家, Top-2路由, 序列长度1024
     */
    public static DeepSeekV3Config createSmallConfig() {
        DeepSeekV3Config config = new DeepSeekV3Config();
        config.setNEmbd(512);
        config.setNLayer(8);
        config.setNHead(8);
        config.setNInner(2048);
        config.setNPositions(1024);
        config.setNumExperts(6);
        config.setTopK(2);
        config.setExpertHiddenDim(2048);
        config.setReasoningHiddenDim(1024);
        config.setCodeAnalysisHiddenDim(384);
        return config;
    }
    
    // ==================== Getter和Setter方法（V3特有配置）====================
    
    public int getReasoningHiddenDim() {
        return reasoningHiddenDim;
    }
    
    public void setReasoningHiddenDim(int reasoningHiddenDim) {
        this.reasoningHiddenDim = reasoningHiddenDim;
    }
    
    public double getConfidenceThreshold() {
        return confidenceThreshold;
    }
    
    public void setConfidenceThreshold(double confidenceThreshold) {
        this.confidenceThreshold = confidenceThreshold;
    }
    
    public boolean isEnableSelfCorrection() {
        return enableSelfCorrection;
    }
    
    public void setEnableSelfCorrection(boolean enableSelfCorrection) {
        this.enableSelfCorrection = enableSelfCorrection;
    }
    
    public int getCodeQualityDim() {
        return codeQualityDim;
    }
    
    public void setCodeQualityDim(int codeQualityDim) {
        this.codeQualityDim = codeQualityDim;
    }
    
    public int getNumProgrammingLanguages() {
        return numProgrammingLanguages;
    }
    
    public void setNumProgrammingLanguages(int numProgrammingLanguages) {
        this.numProgrammingLanguages = numProgrammingLanguages;
    }
    
    public int getCodeAnalysisHiddenDim() {
        return codeAnalysisHiddenDim;
    }
    
    public void setCodeAnalysisHiddenDim(int codeAnalysisHiddenDim) {
        this.codeAnalysisHiddenDim = codeAnalysisHiddenDim;
    }
    
    public int getSyntaxValidatorHiddenDim() {
        return syntaxValidatorHiddenDim;
    }
    
    public void setSyntaxValidatorHiddenDim(int syntaxValidatorHiddenDim) {
        this.syntaxValidatorHiddenDim = syntaxValidatorHiddenDim;
    }
    
    // ==================== toString方法 ====================
    
    @Override
    public String toString() {
        return String.format(
            "DeepSeekV3Config {\n" +
            "  基础架构: %s\n" +
            "  V3特有配置:\n" +
            "    推理隐藏维度: %d\n" +
            "    置信度阈值: %.2f\n" +
            "    自我纠错: %s\n" +
            "    代码质量维度: %d\n" +
            "    支持编程语言数: %d\n" +
            "    代码分析维度: %d\n" +
            "}",
            super.toString(),
            reasoningHiddenDim, confidenceThreshold,
            enableSelfCorrection ? "启用" : "禁用",
            codeQualityDim, numProgrammingLanguages, codeAnalysisHiddenDim
        );
    }
}
