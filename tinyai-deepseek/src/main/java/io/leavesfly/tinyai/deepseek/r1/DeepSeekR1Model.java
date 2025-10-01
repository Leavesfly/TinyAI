package io.leavesfly.tinyai.deepseek.r1;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.Model;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * DeepSeek R1模型 - 继承自TinyAI的Model类
 * 
 * 该模型提供完整的DeepSeek R1功能，包括：
 * 1. 标准的模型接口
 * 2. 推理和反思能力
 * 3. 文本生成功能
 * 4. 思维链推理
 * 5. 模型状态管理
 * 
 * 基于Python实现重新设计，符合TinyAI框架规范
 */
public class DeepSeekR1Model extends Model {
    
    private DeepSeekR1Block deepseekR1Block;
    
    // 模型配置参数
    private int vocabSize;
    private int dModel;
    private int numLayers;
    private int numHeads;
    private int dFF;
    private int maxSeqLen;
    private double dropout;
    
    /**
     * 构造DeepSeek R1模型
     * 
     * @param name 模型名称
     * @param vocabSize 词汇表大小
     * @param dModel 模型维度
     * @param numLayers Transformer层数
     * @param numHeads 注意力头数
     * @param dFF 前馈网络隐藏维度
     * @param maxSeqLen 最大序列长度
     * @param dropout Dropout比率
     */
    public DeepSeekR1Model(String name, int vocabSize, int dModel, int numLayers, 
                          int numHeads, int dFF, int maxSeqLen, double dropout) {
        super(name, new DeepSeekR1Block(name + "_block", vocabSize, dModel, numLayers, 
                                       numHeads, dFF, maxSeqLen, dropout));
        
        this.deepseekR1Block = (DeepSeekR1Block) getBlock();
        this.vocabSize = vocabSize;
        this.dModel = dModel;
        this.numLayers = numLayers;
        this.numHeads = numHeads;
        this.dFF = dFF;
        this.maxSeqLen = maxSeqLen;
        this.dropout = dropout;
        
        // 设置模型描述信息
        setDescription("DeepSeek R1 - 具有推理和反思能力的大语言模型");
        
        // 更新模型信息
        updateModelInfo();
    }
    
    /**
     * 使用默认参数的构造函数
     * 适合快速原型开发和测试
     */
    public DeepSeekR1Model(String name, int vocabSize, int dModel) {
        this(name, vocabSize, dModel, 6, 8, dModel * 4, 512, 0.1);
    }
    
    /**
     * 五参数构造函数，用于测试
     */
    public DeepSeekR1Model(String name, int vocabSize, int dModel, int numLayers, int numHeads) {
        this(name, vocabSize, dModel, numLayers, numHeads, dModel * 4, 512, 0.1);
    }
    
    /**
     * 更新模型信息
     */
    private void updateModelInfo() {
        if (getModelInfo() != null) {
            getModelInfo().setArchitectureType("DeepSeek-R1");
            getModelInfo().addMetric("vocab_size", vocabSize);
            getModelInfo().addMetric("d_model", dModel);
            getModelInfo().addMetric("num_layers", numLayers);
            getModelInfo().addMetric("num_heads", numHeads);
            getModelInfo().addMetric("max_seq_len", maxSeqLen);
        }
    }
    
    /**
     * 执行推理，返回logits
     * 
     * @param inputIds 输入token序列
     * @return 输出logits
     */
    public Variable inference(NdArray inputIds) {
        Variable inputVar = new Variable(inputIds);
        return forward(inputVar);
    }
    
    /**
     * 执行推理，返回logits（带注意力掩码）
     * 
     * @param inputIds 输入token序列
     * @param attentionMask 注意力掩码
     * @return 输出logits
     */
    public Variable inference(NdArray inputIds, NdArray attentionMask) {
        Variable inputVar = new Variable(inputIds);
        Variable maskVar = new Variable(attentionMask);
        return deepseekR1Block.layerForward(inputVar, maskVar);
    }
    
    /**
     * 执行完整推理，包含推理细节
     * 
     * @param inputIds 输入token序列
     * @param attentionMask 注意力掩码（可为null）
     * @return 完整的推理结果
     */
    public DeepSeekR1Block.DeepSeekR1Result inferenceWithDetails(NdArray inputIds, NdArray attentionMask) {
        Variable inputVar = new Variable(inputIds);
        Variable maskVar = attentionMask != null ? new Variable(attentionMask) : null;
        return deepseekR1Block.forwardWithReasoningDetails(inputVar, maskVar);
    }
    
    /**
     * 文本生成
     * 
     * @param inputTokens 输入token列表
     * @param maxNewTokens 最大生成token数
     * @param temperature 采样温度
     * @param topK top-k采样参数
     * @return 生成的token序列
     */
    public List<Integer> generateText(List<Integer> inputTokens, int maxNewTokens, 
                                    float temperature, int topK) {
        return deepseekR1Block.generateSequence(inputTokens, maxNewTokens, temperature, topK);
    }
    
    /**
     * 文本生成（使用默认参数）
     * 
     * @param inputTokens 输入token列表
     * @param maxNewTokens 最大生成token数
     * @return 生成的token序列
     */
    public List<Integer> generateText(List<Integer> inputTokens, int maxNewTokens) {
        return generateText(inputTokens, maxNewTokens, 1.0f, 50);
    }
    
    /**
     * 思维链推理生成
     * 该方法专门用于展示推理过程
     * 
     * @param inputTokens 输入token列表
     * @param maxSteps 最大推理步骤
     * @return 思维链推理结果
     */
    public ChainOfThoughtResult chainOfThoughtReasoning(List<Integer> inputTokens, int maxSteps) {
        NdArray inputIds = createInputArray(inputTokens);
        DeepSeekR1Block.DeepSeekR1Result result = inferenceWithDetails(inputIds, null);
        
        // 构建思维链结果
        List<ReasoningStep> steps = new ArrayList<>();
        
        // 从推理模块获取推理步骤（简化版）
        for (int i = 0; i < Math.min(maxSteps, deepseekR1Block.getReasoningModule().getNumReasoningSteps()); i++) {
            ReasoningStep step = new ReasoningStep(
                i + 1,
                "推理步骤 " + (i + 1) + ": 分析输入并推导结论",
                "基于前面的分析，采取相应的行动",
                0.8f // 模拟置信度
            );
            steps.add(step);
        }
        
        return new ChainOfThoughtResult(
            steps,
            result.getReflectionResult(),
            generateFinalAnswer(result.getLogits())
        );
    }
    
    /**
     * 创建输入数组
     */
    private NdArray createInputArray(List<Integer> tokens) {
        int seqLen = Math.min(tokens.size(), maxSeqLen);
        NdArray inputIds = NdArray.zeros(Shape.of(1, seqLen));
        
        for (int i = 0; i < seqLen; i++) {
            inputIds.set(tokens.get(i), 0, i);
        }
        
        return inputIds;
    }
    
    /**
     * 根据logits生成最终答案（简化版）
     */
    private String generateFinalAnswer(Variable logits) {
        // 这里应该有一个完整的解码过程
        // 简化实现：返回模拟答案
        return "基于推理分析，得出的最终结论。";
    }
    
    /**
     * 获取模型状态统计
     * 
     * @return 模型统计信息
     */
    public Map<String, Object> getModelStatistics() {
        return deepseekR1Block.getModelStatistics();
    }
    
    /**
     * 打印模型详细信息
     */
    public void printModelDetails() {
        System.out.println("=== DeepSeek R1 模型详情 ===");
        System.out.println("模型名称: " + getName());
        System.out.println("架构类型: DeepSeek-R1");
        System.out.println("词汇表大小: " + vocabSize);
        System.out.println("模型维度: " + dModel);
        System.out.println("Transformer层数: " + numLayers);
        System.out.println("注意力头数: " + numHeads);
        System.out.println("前馈网络维度: " + dFF);
        System.out.println("最大序列长度: " + maxSeqLen);
        System.out.println("Dropout比率: " + dropout);
        
        Map<String, Object> stats = getModelStatistics();
        System.out.println("总参数数量: " + stats.get("total_parameters"));
        System.out.println("推理步骤数: " + stats.get("reasoning_steps"));
        System.out.println("反思质量阈值: " + stats.get("reflection_threshold"));
        System.out.println("================================");
    }
    
    /**
     * 检查模型配置的有效性
     * 
     * @return 配置是否有效
     */
    public boolean validateConfiguration() {
        if (vocabSize <= 0) {
            System.err.println("错误: 词汇表大小必须大于0");
            return false;
        }
        
        if (dModel <= 0 || dModel % numHeads != 0) {
            System.err.println("错误: 模型维度必须大于0且能被注意力头数整除");
            return false;
        }
        
        if (numLayers <= 0) {
            System.err.println("错误: Transformer层数必须大于0");
            return false;
        }
        
        if (numHeads <= 0) {
            System.err.println("错误: 注意力头数必须大于0");
            return false;
        }
        
        if (maxSeqLen <= 0) {
            System.err.println("错误: 最大序列长度必须大于0");
            return false;
        }
        
        return true;
    }
    
    /**
     * 推理步骤类
     */
    public static class ReasoningStep {
        private int stepNumber;
        private String thought;
        private String action;
        private float confidence;
        
        public ReasoningStep(int stepNumber, String thought, String action, float confidence) {
            this.stepNumber = stepNumber;
            this.thought = thought;
            this.action = action;
            this.confidence = confidence;
        }
        
        // Getters
        public int getStepNumber() { return stepNumber; }
        public String getThought() { return thought; }
        public String getAction() { return action; }
        public float getConfidence() { return confidence; }
        
        @Override
        public String toString() {
            return String.format("步骤%d: %s -> %s (置信度: %.2f)", 
                               stepNumber, thought, action, confidence);
        }
    }
    
    /**
     * 思维链推理结果类
     */
    public static class ChainOfThoughtResult {
        private List<ReasoningStep> reasoningSteps;
        private ReflectionBlock.ReflectionResult reflectionResult;
        private String finalAnswer;
        
        public ChainOfThoughtResult(List<ReasoningStep> reasoningSteps, 
                                  ReflectionBlock.ReflectionResult reflectionResult,
                                  String finalAnswer) {
            this.reasoningSteps = reasoningSteps;
            this.reflectionResult = reflectionResult;
            this.finalAnswer = finalAnswer;
        }
        
        // Getters
        public List<ReasoningStep> getReasoningSteps() { return reasoningSteps; }
        public ReflectionBlock.ReflectionResult getReflectionResult() { return reflectionResult; }
        public String getFinalAnswer() { return finalAnswer; }
        
        /**
         * 打印完整的思维链过程
         */
        public void printChainOfThought() {
            System.out.println("=== 思维链推理过程 ===");
            for (ReasoningStep step : reasoningSteps) {
                System.out.println(step);
            }
            System.out.println("\n=== 反思结果 ===");
            System.out.println(reflectionResult);
            System.out.println("\n=== 最终答案 ===");
            System.out.println(finalAnswer);
            System.out.println("=====================");
        }
    }
    
    // Getters
    public DeepSeekR1Block getDeepSeekR1Block() { return deepseekR1Block; }
    public int getVocabSize() { return vocabSize; }
    public int getDModel() { return dModel; }
    public int getNumLayers() { return numLayers; }
    public int getNumHeads() { return numHeads; }
    public int getDFF() { return dFF; }
    public int getMaxSeqLen() { return maxSeqLen; }
    public double getDropoutRate() { return dropout; }
}