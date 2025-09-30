package io.leavesfly.tinyai.nlp.deepseekR1;

/**
 * DeepSeek-R1 模型配置类
 * 
 * 这个类封装了DeepSeek-R1模型的所有配置参数，
 * 提供了预定义的配置模板和自定义配置功能。
 * 
 * 配置参数包括：
 * 1. 基础模型参数：词汇表大小、模型维度、层数等
 * 2. 推理参数：最大推理步骤、置信度阈值等
 * 3. 训练参数：学习率、dropout比率等
 * 4. 优化参数：自适应阈值、探索参数等
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekR1Config {
    
    // 基础模型参数
    private int vocabSize;              // 词汇表大小
    private int dModel;                 // 模型维度
    private int numLayers;              // Transformer层数
    private int numHeads;               // 注意力头数
    private int dFF;                    // 前馈网络维度
    private int maxSeqLength;           // 最大序列长度
    
    // 推理相关参数
    private int maxReasoningSteps;      // 最大推理步骤数
    private double reasoningThreshold;   // 推理置信度阈值
    private double adaptiveThreshold;    // 自适应调整阈值
    private boolean enableReasoning;     // 是否启用推理模式
    private boolean enableCoT;           // 是否启用思维链
    
    // 训练参数
    private double dropoutRate;         // Dropout比率
    private double learningRate;        // 学习率
    private int batchSize;             // 批大小
    private int numEpochs;             // 训练轮数
    
    // 优化参数
    private boolean useLayerNorm;       // 是否使用层归一化
    private boolean useBias;            // 是否使用偏置
    private double weightDecay;         // 权重衰减
    private double gradientClipping;    // 梯度裁剪
    
    // 推理优化参数
    private boolean enableEarlyStopping; // 是否启用早停
    private boolean enableExploration;   // 是否启用探索
    private double explorationRate;      // 探索率
    private int warmupSteps;            // 预热步骤数
    
    /**
     * 默认构造函数 - 使用中等规模配置
     */
    public DeepSeekR1Config() {
        this.vocabSize = 50257;
        this.dModel = 768;
        this.numLayers = 12;
        this.numHeads = 12;
        this.dFF = 3072;
        this.maxSeqLength = 1024;
        
        this.maxReasoningSteps = 10;
        this.reasoningThreshold = 0.7;
        this.adaptiveThreshold = 0.1;
        this.enableReasoning = true;
        this.enableCoT = true;
        
        this.dropoutRate = 0.1;
        this.learningRate = 0.0001;
        this.batchSize = 32;
        this.numEpochs = 100;
        
        this.useLayerNorm = true;
        this.useBias = true;
        this.weightDecay = 0.01;
        this.gradientClipping = 1.0;
        
        this.enableEarlyStopping = true;
        this.enableExploration = true;
        this.explorationRate = 0.1;
        this.warmupSteps = 1000;
    }
    
    /**
     * 完整参数构造函数
     */
    public DeepSeekR1Config(int vocabSize, int dModel, int numLayers, int numHeads,
                           int dFF, int maxSeqLength, int maxReasoningSteps,
                           double reasoningThreshold, double dropoutRate) {
        this();
        this.vocabSize = vocabSize;
        this.dModel = dModel;
        this.numLayers = numLayers;
        this.numHeads = numHeads;
        this.dFF = dFF;
        this.maxSeqLength = maxSeqLength;
        this.maxReasoningSteps = maxReasoningSteps;
        this.reasoningThreshold = reasoningThreshold;
        this.dropoutRate = dropoutRate;
        
        // 验证参数
        validate();
    }
    
    /**
     * 创建小型模型配置
     * 适用于快速原型验证和资源受限环境
     */
    public static DeepSeekR1Config createTinyConfig() {
        DeepSeekR1Config config = new DeepSeekR1Config();
        config.dModel = 384;
        config.numLayers = 6;
        config.numHeads = 6;
        config.dFF = 1536;
        config.maxSeqLength = 512;
        config.maxReasoningSteps = 5;
        config.batchSize = 16;
        return config;
    }
    
    /**
     * 创建中型模型配置
     * 平衡性能和资源消耗的标准配置
     */
    public static DeepSeekR1Config createMediumConfig() {
        return new DeepSeekR1Config(); // 使用默认配置
    }
    
    /**
     * 创建大型模型配置
     * 追求最佳性能，适用于高性能计算环境
     */
    public static DeepSeekR1Config createLargeConfig() {
        DeepSeekR1Config config = new DeepSeekR1Config();
        config.dModel = 1536;
        config.numLayers = 24;
        config.numHeads = 24;
        config.dFF = 6144;
        config.maxSeqLength = 2048;
        config.maxReasoningSteps = 20;
        config.batchSize = 8;
        config.learningRate = 0.00005;
        return config;
    }
    
    /**
     * 创建推理优化配置
     * 专门针对推理任务优化的配置
     */
    public static DeepSeekR1Config createReasoningConfig() {
        DeepSeekR1Config config = new DeepSeekR1Config();
        config.maxReasoningSteps = 15;
        config.reasoningThreshold = 0.8;
        config.adaptiveThreshold = 0.05;
        config.enableEarlyStopping = true;
        config.enableExploration = true;
        config.explorationRate = 0.15;
        return config;
    }
    
    /**
     * 创建训练优化配置
     * 专门针对训练效率优化的配置
     */
    public static DeepSeekR1Config createTrainingConfig() {
        DeepSeekR1Config config = new DeepSeekR1Config();
        config.dropoutRate = 0.15;
        config.learningRate = 0.0002;
        config.batchSize = 64;
        config.gradientClipping = 0.5;
        config.weightDecay = 0.02;
        config.warmupSteps = 2000;
        return config;
    }
    
    /**
     * 创建调试配置
     * 用于开发和调试的小规模快速配置
     */
    public static DeepSeekR1Config createDebugConfig() {
        DeepSeekR1Config config = new DeepSeekR1Config();
        config.vocabSize = 1000;
        config.dModel = 128;
        config.numLayers = 2;
        config.numHeads = 4;
        config.dFF = 512;
        config.maxSeqLength = 64;
        config.maxReasoningSteps = 3;
        config.batchSize = 4;
        config.numEpochs = 5;
        return config;
    }
    
    /**
     * 配置参数验证
     */
    public void validate() {
        if (vocabSize <= 0 || dModel <= 0 || numLayers <= 0 || numHeads <= 0 ||
            dFF <= 0 || maxSeqLength <= 0 || maxReasoningSteps <= 0) {
            throw new IllegalArgumentException("所有大小参数必须大于0");
        }
        
        if (dModel % numHeads != 0) {
            throw new IllegalArgumentException("dModel必须能被numHeads整除");
        }
        
        if (dropoutRate < 0.0 || dropoutRate > 1.0) {
            throw new IllegalArgumentException("dropout比率必须在0.0到1.0之间");
        }
        
        if (reasoningThreshold < 0.0 || reasoningThreshold > 1.0) {
            throw new IllegalArgumentException("推理置信度阈值必须在0.0到1.0之间");
        }
        
        if (learningRate <= 0.0) {
            throw new IllegalArgumentException("学习率必须大于0");
        }
        
        if (batchSize <= 0) {
            throw new IllegalArgumentException("批大小必须大于0");
        }
    }
    
    /**
     * 复制配置
     */
    public DeepSeekR1Config copy() {
        DeepSeekR1Config config = new DeepSeekR1Config();
        
        // 基础模型参数
        config.vocabSize = this.vocabSize;
        config.dModel = this.dModel;
        config.numLayers = this.numLayers;
        config.numHeads = this.numHeads;
        config.dFF = this.dFF;
        config.maxSeqLength = this.maxSeqLength;
        
        // 推理参数
        config.maxReasoningSteps = this.maxReasoningSteps;
        config.reasoningThreshold = this.reasoningThreshold;
        config.adaptiveThreshold = this.adaptiveThreshold;
        config.enableReasoning = this.enableReasoning;
        config.enableCoT = this.enableCoT;
        
        // 训练参数
        config.dropoutRate = this.dropoutRate;
        config.learningRate = this.learningRate;
        config.batchSize = this.batchSize;
        config.numEpochs = this.numEpochs;
        
        // 优化参数
        config.useLayerNorm = this.useLayerNorm;
        config.useBias = this.useBias;
        config.weightDecay = this.weightDecay;
        config.gradientClipping = this.gradientClipping;
        
        // 推理优化参数
        config.enableEarlyStopping = this.enableEarlyStopping;
        config.enableExploration = this.enableExploration;
        config.explorationRate = this.explorationRate;
        config.warmupSteps = this.warmupSteps;
        
        return config;
    }
    
    /**
     * 获取配置摘要字符串
     */
    public String getSummary() {
        return String.format(
            "DeepSeek-R1 Config Summary:\n" +
            "  Model: %d layers, %d dim, %d heads, vocab=%d\n" +
            "  Sequence: max_len=%d\n" +
            "  Reasoning: max_steps=%d, threshold=%.2f\n" +
            "  Training: dropout=%.2f, lr=%.4f, batch=%d\n" +
            "  Features: reasoning=%s, CoT=%s, early_stop=%s",
            numLayers, dModel, numHeads, vocabSize,
            maxSeqLength,
            maxReasoningSteps, reasoningThreshold,
            dropoutRate, learningRate, batchSize,
            enableReasoning ? "ON" : "OFF",
            enableCoT ? "ON" : "OFF",
            enableEarlyStopping ? "ON" : "OFF"
        );
    }
    
    /**
     * 打印配置信息
     */
    public void printConfig() {
        System.out.println("\n=== DeepSeek-R1 Configuration ===");
        System.out.println(getSummary());
        System.out.println("==================================\n");
    }
    
    // Getter和Setter方法
    public int getVocabSize() { return vocabSize; }
    public void setVocabSize(int vocabSize) { this.vocabSize = vocabSize; }
    
    public int getDModel() { return dModel; }
    public void setDModel(int dModel) { this.dModel = dModel; }
    
    public int getNumLayers() { return numLayers; }
    public void setNumLayers(int numLayers) { this.numLayers = numLayers; }
    
    public int getNumHeads() { return numHeads; }
    public void setNumHeads(int numHeads) { this.numHeads = numHeads; }
    
    public int getDFF() { return dFF; }
    public void setDFF(int dFF) { this.dFF = dFF; }
    
    public int getMaxSeqLength() { return maxSeqLength; }
    public void setMaxSeqLength(int maxSeqLength) { this.maxSeqLength = maxSeqLength; }
    
    public int getMaxReasoningSteps() { return maxReasoningSteps; }
    public void setMaxReasoningSteps(int maxReasoningSteps) { this.maxReasoningSteps = maxReasoningSteps; }
    
    public double getReasoningThreshold() { return reasoningThreshold; }
    public void setReasoningThreshold(double reasoningThreshold) { this.reasoningThreshold = reasoningThreshold; }
    
    public double getAdaptiveThreshold() { return adaptiveThreshold; }
    public void setAdaptiveThreshold(double adaptiveThreshold) { this.adaptiveThreshold = adaptiveThreshold; }
    
    public boolean isEnableReasoning() { return enableReasoning; }
    public void setEnableReasoning(boolean enableReasoning) { this.enableReasoning = enableReasoning; }
    
    public boolean isEnableCoT() { return enableCoT; }
    public void setEnableCoT(boolean enableCoT) { this.enableCoT = enableCoT; }
    
    public double getDropoutRate() { return dropoutRate; }
    public void setDropoutRate(double dropoutRate) { this.dropoutRate = dropoutRate; }
    
    public double getLearningRate() { return learningRate; }
    public void setLearningRate(double learningRate) { this.learningRate = learningRate; }
    
    public int getBatchSize() { return batchSize; }
    public void setBatchSize(int batchSize) { this.batchSize = batchSize; }
    
    public int getNumEpochs() { return numEpochs; }
    public void setNumEpochs(int numEpochs) { this.numEpochs = numEpochs; }
    
    public boolean isUseLayerNorm() { return useLayerNorm; }
    public void setUseLayerNorm(boolean useLayerNorm) { this.useLayerNorm = useLayerNorm; }
    
    public boolean isUseBias() { return useBias; }
    public void setUseBias(boolean useBias) { this.useBias = useBias; }
    
    public double getWeightDecay() { return weightDecay; }
    public void setWeightDecay(double weightDecay) { this.weightDecay = weightDecay; }
    
    public double getGradientClipping() { return gradientClipping; }
    public void setGradientClipping(double gradientClipping) { this.gradientClipping = gradientClipping; }
    
    public boolean isEnableEarlyStopping() { return enableEarlyStopping; }
    public void setEnableEarlyStopping(boolean enableEarlyStopping) { this.enableEarlyStopping = enableEarlyStopping; }
    
    public boolean isEnableExploration() { return enableExploration; }
    public void setEnableExploration(boolean enableExploration) { this.enableExploration = enableExploration; }
    
    public double getExplorationRate() { return explorationRate; }
    public void setExplorationRate(double explorationRate) { this.explorationRate = explorationRate; }
    
    public int getWarmupSteps() { return warmupSteps; }
    public void setWarmupSteps(int warmupSteps) { this.warmupSteps = warmupSteps; }
}