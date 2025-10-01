package io.leavesfly.tinyai.deepseek.v3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.Model;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Parameter;

import java.util.List;
import java.util.Map;

/**
 * DeepSeek V3模型
 * 
 * 继承自TinyAI的Model类，提供了DeepSeek V3的完整功能，包括：
 * 1. 模型的初始化和配置
 * 2. 前向传播和推理
 * 3. 任务类型感知的生成
 * 4. 推理过程跟踪
 * 5. 代码生成专门功能
 * 6. 模型状态管理
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekV3Model extends Model {
    
    /**
     * V3模型核心Block
     */
    private final DeepSeekV3Block deepSeekV3Block;
    
    /**
     * 模型配置
     */
    private final V3ModelConfig config;
    
    /**
     * 构造函数
     * 
     * @param name 模型名称
     * @param config V3模型配置
     */
    public DeepSeekV3Model(String name, V3ModelConfig config) {
        super(name, createDeepSeekV3Block(name, config));
        
        this.config = config;
        this.deepSeekV3Block = (DeepSeekV3Block) getBlock();
        
        // 设置模型描述
        setDescription("DeepSeek V3 - 混合专家大语言模型，具备增强推理和代码生成能力");
        
        // 更新模型信息
        updateModelInfo();
    }
    
    /**
     * 默认构造函数 - 使用标准配置
     */
    public DeepSeekV3Model(String name) {
        this(name, V3ModelConfig.getDefaultConfig());
    }
    
    /**
     * 创建DeepSeek V3 Block
     */
    private static DeepSeekV3Block createDeepSeekV3Block(String name, V3ModelConfig config) {
        return new DeepSeekV3Block(
            name + "_v3_block",
            config.vocabSize,
            config.dModel,
            config.numLayers,
            config.numHeads,
            config.dFF,
            config.numExperts,
            config.maxSeqLen,
            config.dropout
        );
    }
    
    /**
     * 更新模型信息
     */
    private void updateModelInfo() {
        if (getModelInfo() != null) {
            getModelInfo().setArchitectureType("DeepSeek V3 - MoE Transformer");
            getModelInfo().addMetric("num_experts", config.numExperts);
            getModelInfo().addMetric("max_sequence_length", config.maxSeqLen);
            getModelInfo().addMetric("vocabulary_size", config.vocabSize);
            getModelInfo().addMetric("transformer_layers", config.numLayers);
        }
    }
    
    /**
     * 执行V3推理生成
     * 
     * @param inputIds 输入token IDs
     * @param taskType 任务类型
     * @return V3模型输出
     */
    public DeepSeekV3Block.DeepSeekV3Output generateWithTaskType(NdArray inputIds, TaskType taskType) {
        Variable inputVar = new Variable(inputIds);
        return deepSeekV3Block.forwardWithTaskType(inputVar, null, taskType);
    }
    
    /**
     * 通用推理生成（自动识别任务类型）
     * 
     * @param inputIds 输入token IDs
     * @return V3模型输出
     */
    public DeepSeekV3Block.DeepSeekV3Output generate(NdArray inputIds) {
        return generateWithTaskType(inputIds, TaskType.GENERAL);
    }
    
    /**
     * 代码生成专门接口
     * 
     * @param inputIds 输入token IDs
     * @return 代码生成结果
     */
    public CodeGenerationResult generateCode(NdArray inputIds) {
        DeepSeekV3Block.DeepSeekV3Output output = generateWithTaskType(inputIds, TaskType.CODING);
        return new CodeGenerationResult(output);
    }
    
    /**
     * 推理任务专门接口
     * 
     * @param inputIds 输入token IDs
     * @return 推理结果
     */
    public ReasoningResult performReasoning(NdArray inputIds) {
        DeepSeekV3Block.DeepSeekV3Output output = generateWithTaskType(inputIds, TaskType.REASONING);
        return new ReasoningResult(output);
    }
    
    /**
     * 数学计算专门接口
     * 
     * @param inputIds 输入token IDs
     * @return 数学计算结果
     */
    public MathResult solveMath(NdArray inputIds) {
        DeepSeekV3Block.DeepSeekV3Output output = generateWithTaskType(inputIds, TaskType.MATH);
        return new MathResult(output);
    }
    
    /**
     * 批量生成
     * 
     * @param batchInputIds 批量输入
     * @param taskType 任务类型
     * @return 批量输出结果
     */
    public BatchGenerationResult generateBatch(NdArray batchInputIds, TaskType taskType) {
        DeepSeekV3Block.DeepSeekV3Output output = generateWithTaskType(batchInputIds, taskType);
        return new BatchGenerationResult(output, batchInputIds.getShape().getDimension(0));
    }
    
    /**
     * 获取模型统计信息
     */
    public V3ModelStats getModelStats() {
        DeepSeekV3Block.DeepSeekV3Output lastOutput = deepSeekV3Block.getLastOutput();
        
        V3ModelStats stats = new V3ModelStats();
        stats.totalParameters = getAllParams().size();
        stats.vocabSize = config.vocabSize;
        stats.dModel = config.dModel;
        stats.numLayers = config.numLayers;
        stats.numExperts = config.numExperts;
        stats.maxSeqLen = config.maxSeqLen;
        
        if (lastOutput != null) {
            stats.lastMoeLoss = lastOutput.moeLoss;
            stats.lastReasoningQuality = lastOutput.getReasoningQuality();
            stats.lastCodeConfidence = lastOutput.getCodeConfidence();
            stats.expertUsageStats = lastOutput.getExpertUsageStats();
        }
        
        return stats;
    }
    
    /**
     * 重置模型状态
     */
    @Override
    public void resetState() {
        super.resetState();
        deepSeekV3Block.resetAllStates();
    }
    
    /**
     * 获取模型配置
     */
    public V3ModelConfig getConfig() {
        return config;
    }
    
    /**
     * 获取最后一次推理的详细信息
     */
    public DetailedInferenceInfo getLastInferenceDetails() {
        DeepSeekV3Block.DeepSeekV3Output lastOutput = deepSeekV3Block.getLastOutput();
        if (lastOutput == null) {
            return null;
        }
        
        return new DetailedInferenceInfo(lastOutput);
    }
    
    /**
     * 打印模型架构信息
     */
    public void printArchitecture() {
        System.out.println("=== DeepSeek V3 模型架构 ===");
        System.out.println("模型名称: " + getName());
        System.out.println("词汇表大小: " + config.vocabSize);
        System.out.println("模型维度: " + config.dModel);
        System.out.println("Transformer层数: " + config.numLayers);
        System.out.println("注意力头数: " + config.numHeads);
        System.out.println("专家数量: " + config.numExperts);
        System.out.println("最大序列长度: " + config.maxSeqLen);
        System.out.println("前馈网络维度: " + config.dFF);
        System.out.println("Dropout概率: " + config.dropout);
        
        long totalParams = 0;
        for (Parameter param : getAllParams().values()) {
            totalParams += param.getValue().getShape().size();
        }
        System.out.println("总参数量: " + formatParameterCount(totalParams));
        System.out.println("========================");
    }
    
    /**
     * 格式化参数数量显示
     */
    private String formatParameterCount(long count) {
        if (count >= 1_000_000_000) {
            return String.format("%.1fB", count / 1_000_000_000.0);
        } else if (count >= 1_000_000) {
            return String.format("%.1fM", count / 1_000_000.0);
        } else if (count >= 1_000) {
            return String.format("%.1fK", count / 1_000.0);
        } else {
            return String.valueOf(count);
        }
    }
    
    // 内部结果类
    
    /**
     * 代码生成结果
     */
    public static class CodeGenerationResult {
        public final Variable logits;
        public final String detectedLanguage;
        public final float syntaxScore;
        public final float qualityScore;
        public final float codeConfidence;
        public final List<V3ReasoningStep> reasoningSteps;
        
        public CodeGenerationResult(DeepSeekV3Block.DeepSeekV3Output output) {
            this.logits = output.logits;
            this.reasoningSteps = output.reasoningSteps;
            
            if (output.codeInfo != null) {
                this.detectedLanguage = output.codeInfo.getDetectedLanguage();
                this.syntaxScore = output.codeInfo.getSyntaxScore();
                this.qualityScore = output.codeInfo.getQualityScore();
                this.codeConfidence = output.codeInfo.getCodeConfidence();
            } else {
                this.detectedLanguage = "Unknown";
                this.syntaxScore = 0.0f;
                this.qualityScore = 0.0f;
                this.codeConfidence = 0.0f;
            }
        }
    }
    
    /**
     * 推理结果
     */
    public static class ReasoningResult {
        public final Variable logits;
        public final List<V3ReasoningStep> reasoningSteps;
        public final float averageConfidence;
        public final TaskType identifiedTaskType;
        
        public ReasoningResult(DeepSeekV3Block.DeepSeekV3Output output) {
            this.logits = output.logits;
            this.reasoningSteps = output.reasoningSteps;
            this.identifiedTaskType = output.identifiedTaskType;
            
            this.averageConfidence = (float) reasoningSteps.stream()
                                                         .mapToDouble(V3ReasoningStep::getConfidence)
                                                         .average()
                                                         .orElse(0.0);
        }
    }
    
    /**
     * 数学计算结果
     */
    public static class MathResult {
        public final Variable logits;
        public final List<V3ReasoningStep> reasoningSteps;
        public final float mathConfidence;
        
        public MathResult(DeepSeekV3Block.DeepSeekV3Output output) {
            this.logits = output.logits;
            this.reasoningSteps = output.reasoningSteps;
            
            // 计算数学推理的置信度
            this.mathConfidence = (float) reasoningSteps.stream()
                                                      .filter(step -> step.getTaskType() == TaskType.MATH)
                                                      .mapToDouble(V3ReasoningStep::getConfidence)
                                                      .average()
                                                      .orElse(0.0);
        }
    }
    
    /**
     * 批量生成结果
     */
    public static class BatchGenerationResult {
        public final Variable batchLogits;
        public final int batchSize;
        public final List<V3ReasoningStep> reasoningSteps;
        public final float averageReasoningQuality;
        
        public BatchGenerationResult(DeepSeekV3Block.DeepSeekV3Output output, int batchSize) {
            this.batchLogits = output.logits;
            this.batchSize = batchSize;
            this.reasoningSteps = output.reasoningSteps;
            this.averageReasoningQuality = output.getReasoningQuality();
        }
    }
    
    /**
     * 模型统计信息
     */
    public static class V3ModelStats {
        public long totalParameters;
        public int vocabSize;
        public int dModel;
        public int numLayers;
        public int numExperts;
        public int maxSeqLen;
        public float lastMoeLoss;
        public float lastReasoningQuality;
        public float lastCodeConfidence;
        public Map<String, Integer> expertUsageStats;
        
        @Override
        public String toString() {
            return String.format("V3ModelStats{params=%d, experts=%d, moeLoss=%.4f, reasoning=%.3f}", 
                               totalParameters, numExperts, lastMoeLoss, lastReasoningQuality);
        }
    }
    
    /**
     * 详细推理信息
     */
    public static class DetailedInferenceInfo {
        public final TaskType requestedTaskType;
        public final TaskType identifiedTaskType;
        public final List<V3ReasoningStep> reasoningSteps;
        public final float moeLoss;
        public final float reasoningQuality;
        public final float codeConfidence;
        public final Map<String, Integer> expertUsage;
        
        public DetailedInferenceInfo(DeepSeekV3Block.DeepSeekV3Output output) {
            this.requestedTaskType = output.requestedTaskType;
            this.identifiedTaskType = output.identifiedTaskType;
            this.reasoningSteps = output.reasoningSteps;
            this.moeLoss = output.moeLoss;
            this.reasoningQuality = output.getReasoningQuality();
            this.codeConfidence = output.getCodeConfidence();
            this.expertUsage = output.getExpertUsageStats();
        }
        
        public void printSummary() {
            System.out.println("=== 推理详情 ===");
            System.out.println("请求任务类型: " + requestedTaskType);
            System.out.println("识别任务类型: " + identifiedTaskType);
            System.out.println("推理步骤数: " + reasoningSteps.size());
            System.out.println("推理质量: " + String.format("%.3f", reasoningQuality));
            System.out.println("MoE损失: " + String.format("%.4f", moeLoss));
            if (codeConfidence > 0) {
                System.out.println("代码置信度: " + String.format("%.3f", codeConfidence));
            }
            System.out.println("专家使用统计: " + expertUsage);
            System.out.println("===============");
        }
    }
    
    /**
     * V3模型配置类
     */
    public static class V3ModelConfig {
        public final int vocabSize;
        public final int dModel;
        public final int numLayers;
        public final int numHeads;
        public final int dFF;
        public final int numExperts;
        public final int maxSeqLen;
        public final float dropout;
        
        public V3ModelConfig(int vocabSize, int dModel, int numLayers, int numHeads, 
                           int dFF, int numExperts, int maxSeqLen, float dropout) {
            this.vocabSize = vocabSize;
            this.dModel = dModel;
            this.numLayers = numLayers;
            this.numHeads = numHeads;
            this.dFF = dFF;
            this.numExperts = numExperts;
            this.maxSeqLen = maxSeqLen;
            this.dropout = dropout;
        }
        
        public static V3ModelConfig getDefaultConfig() {
            return new V3ModelConfig(32000, 768, 12, 12, 3072, 8, 8192, 0.1f);
        }
        
        public static V3ModelConfig getLargeConfig() {
            return new V3ModelConfig(50000, 1024, 24, 16, 4096, 16, 16384, 0.1f);
        }
        
        public static V3ModelConfig getSmallConfig() {
            return new V3ModelConfig(16000, 512, 6, 8, 2048, 4, 4096, 0.1f);
        }
    }
}