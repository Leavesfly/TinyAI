package io.leavesfly.tinyai.deepseek.v3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Block;
import io.leavesfly.tinyai.nnet.Parameter;
import io.leavesfly.tinyai.nnet.layer.dnn.LinearLayer;
import io.leavesfly.tinyai.nnet.layer.transformer.LayerNorm;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * DeepSeek V3主模型Block
 * 
 * 集成了DeepSeek V3的所有核心组件，包括：
 * 1. Token和位置嵌入层
 * 2. 多层V3 Transformer块（带MoE）
 * 3. V3增强推理模块
 * 4. 代码生成专门模块
 * 5. 多任务输出头
 * 6. 层归一化和Dropout
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekV3Block extends Block {
    
    // 模型配置参数
    private final int vocabSize;
    private final int dModel;
    private final int numLayers;
    private final int numHeads;
    private final int dFF;
    private final int numExperts;
    private final int maxSeqLen;
    private final float dropout;
    
    // 嵌入层
    private Parameter tokenEmbedding;
    private Parameter positionEmbedding;
    
    // V3 Transformer层
    private List<V3TransformerBlock> transformerLayers;
    
    // V3增强推理模块
    private V3ReasoningBlock reasoningModule;
    
    // 代码生成专门模块
    private CodeGenerationBlock codeGeneration;
    
    // 最终层归一化
    private LayerNorm finalNorm;
    
    // 多任务输出头
    private Map<TaskType, LinearLayer> outputHeads;
    
    // 最后一次前向传播的状态
    private DeepSeekV3Output lastOutput;
    
    /**
     * 构造函数
     * 
     * @param name 模型名称
     * @param vocabSize 词汇表大小
     * @param dModel 模型维度
     * @param numLayers Transformer层数
     * @param numHeads 注意力头数
     * @param dFF 前馈网络维度
     * @param numExperts 专家数量
     * @param maxSeqLen 最大序列长度
     * @param dropout Dropout概率
     */
    public DeepSeekV3Block(String name, int vocabSize, int dModel, int numLayers, 
                          int numHeads, int dFF, int numExperts, int maxSeqLen, float dropout) {
        super(name, Shape.of(-1, -1), Shape.of(-1, -1, dModel));
        
        this.vocabSize = vocabSize;
        this.dModel = dModel;
        this.numLayers = numLayers;
        this.numHeads = numHeads;
        this.dFF = dFF;
        this.numExperts = numExperts;
        this.maxSeqLen = maxSeqLen;
        this.dropout = dropout;
        
        init();
    }
    
    /**
     * 默认构造函数 - 使用标准配置
     */
    public DeepSeekV3Block(String name, int vocabSize, int dModel) {
        this(name, vocabSize, dModel, 12, 12, dModel * 4, 8, 8192, 0.1f);
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            initEmbeddings();
            initTransformerLayers();
            initReasoningModule();
            initCodeGenerationModule();
            initOutputHeads();
            initFinalNormalization();
            
            alreadyInit = true;
        }
    }
    
    /**
     * 初始化嵌入层
     */
    private void initEmbeddings() {
        // Token嵌入 - 使用Xavier初始化
        NdArray tokenEmbeddingData = NdArray.likeRandomN(Shape.of(vocabSize, dModel))
                                          .mulNum(Math.sqrt(1.0 / dModel));
        tokenEmbedding = new Parameter(tokenEmbeddingData);
        tokenEmbedding.setName(name + "_token_embedding");
        addParam(tokenEmbedding.getName(), tokenEmbedding);
        
        // 位置嵌入
        NdArray positionEmbeddingData = NdArray.likeRandomN(Shape.of(maxSeqLen, dModel))
                                             .mulNum(Math.sqrt(1.0 / dModel));
        positionEmbedding = new Parameter(positionEmbeddingData);
        positionEmbedding.setName(name + "_position_embedding");
        addParam(positionEmbedding.getName(), positionEmbedding);
    }
    
    /**
     * 初始化Transformer层
     */
    private void initTransformerLayers() {
        transformerLayers = new ArrayList<>();
        
        for (int i = 0; i < numLayers; i++) {
            V3TransformerBlock layer = new V3TransformerBlock(
                name + "_transformer_" + i, 
                dModel, numHeads, dFF, numExperts, dropout
            );
            transformerLayers.add(layer);
            addLayer(layer);
        }
    }
    
    /**
     * 初始化推理模块
     */
    private void initReasoningModule() {
        reasoningModule = new V3ReasoningBlock(name + "_reasoning", dModel, 7);
        addLayer(reasoningModule);
    }
    
    /**
     * 初始化代码生成模块
     */
    private void initCodeGenerationModule() {
        codeGeneration = new CodeGenerationBlock(name + "_code_gen", dModel, 10);
        addLayer(codeGeneration);
    }
    
    /**
     * 初始化多任务输出头
     */
    private void initOutputHeads() {
        outputHeads = new HashMap<>();
        
        for (TaskType taskType : TaskType.values()) {
            LinearLayer outputHead = new LinearLayer(
                name + "_output_" + taskType.getValue(), 
                dModel, vocabSize, false
            );
            outputHeads.put(taskType, outputHead);
            addLayer(outputHead);
        }
    }
    
    /**
     * 初始化最终层归一化
     */
    private void initFinalNormalization() {
        finalNorm = new LayerNorm(name + "_final_norm", dModel);
        addLayer(finalNorm);
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable inputIds = inputs[0];
        TaskType taskType = TaskType.GENERAL; // 默认任务类型
        NdArray attentionMask = null;
        
        // 解析额外参数
        if (inputs.length > 1) {
            // 可以扩展参数传递机制
        }
        
        // 执行V3前向传播
        DeepSeekV3Output output = forwardWithTaskType(inputIds, attentionMask, taskType);
        lastOutput = output;
        
        return output.logits;
    }
    
    /**
     * 执行带任务类型的V3前向传播
     * 
     * @param inputIds 输入token IDs
     * @param attentionMask 注意力掩码（可为null）
     * @param taskType 任务类型
     * @return V3模型输出
     */
    public DeepSeekV3Output forwardWithTaskType(Variable inputIds, NdArray attentionMask, TaskType taskType) {
        NdArray inputData = inputIds.getValue();
        int batchSize = inputData.getShape().getDimension(0);
        int seqLen = inputData.getShape().getDimension(1);
        
        // 嵌入处理
        Variable embeddings = processEmbeddings(inputData, batchSize, seqLen);
        
        // 通过V3 Transformer层
        TransformerResult transformerResult = processTransformerLayers(embeddings, attentionMask, taskType);
        
        // V3推理模块
        V3ReasoningBlock.ReasoningResult reasoningResult = reasoningModule.performV3Reasoning(transformerResult.output);
        
        // 代码生成分析（如果是代码任务）
        CodeGenerationBlock.CodeGenerationResult codeResult = null;
        if (taskType == TaskType.CODING) {
            codeResult = codeGeneration.performCodeGenerationAnalysis(reasoningResult.finalOutput);
        }
        
        // 最终层归一化
        Variable normalizedOutput = finalNorm.layerForward(reasoningResult.finalOutput);
        
        // 选择输出头
        LinearLayer outputHead = outputHeads.getOrDefault(taskType, outputHeads.get(TaskType.GENERAL));
        Variable finalLogits = outputHead.layerForward(normalizedOutput);
        
        // 计算总的MoE损失
        float totalMoELoss = transformerResult.allRoutingInfo.stream()
                                                           .map(ExpertRoutingInfo::getTotalMoELoss)
                                                           .reduce(0.0f, Float::sum);
        
        // 创建输出对象
        return new DeepSeekV3Output(
            finalLogits,
            reasoningResult.reasoningSteps,
            codeResult,
            reasoningResult.finalOutput,
            totalMoELoss,
            transformerResult.allRoutingInfo,
            taskType,
            reasoningResult.taskType
        );
    }
    
    /**
     * 处理嵌入
     */
    private Variable processEmbeddings(NdArray inputData, int batchSize, int seqLen) {
        // 创建位置索引
        NdArray positions = NdArray.of(Shape.of(batchSize, seqLen));
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                positions.set(s, b, s);
            }
        }
        
        // Token嵌入
        NdArray tokenEmb = embedTokens(inputData);
        
        // 位置嵌入
        NdArray posEmb = embedPositions(positions);
        
        // 组合嵌入
        NdArray combinedEmb = tokenEmb.add(posEmb);
        
        // 应用Dropout（简化为不操作）
        return new Variable(combinedEmb);
    }
    
    /**
     * Token嵌入
     */
    private NdArray embedTokens(NdArray inputIds) {
        int batchSize = inputIds.getShape().getDimension(0);
        int seqLen = inputIds.getShape().getDimension(1);
        NdArray tokenEmbeddingData = tokenEmbedding.getValue();
        
        NdArray result = NdArray.of(Shape.of(batchSize, seqLen, dModel));
        
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                int tokenId = (int) inputIds.get(b, s);
                if (tokenId >= 0 && tokenId < vocabSize) {
                    for (int d = 0; d < dModel; d++) {
                        float embValue = tokenEmbeddingData.get(tokenId, d);
                        result.set(embValue, b, s, d);
                    }
                }
            }
        }
        
        return result;
    }
    
    /**
     * 位置嵌入
     */
    private NdArray embedPositions(NdArray positions) {
        int batchSize = positions.getShape().getDimension(0);
        int seqLen = positions.getShape().getDimension(1);
        NdArray positionEmbeddingData = positionEmbedding.getValue();
        
        NdArray result = NdArray.of(Shape.of(batchSize, seqLen, dModel));
        
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                int pos = (int) positions.get(b, s);
                if (pos >= 0 && pos < maxSeqLen) {
                    for (int d = 0; d < dModel; d++) {
                        float posValue = positionEmbeddingData.get(pos, d);
                        result.set(posValue, b, s, d);
                    }
                }
            }
        }
        
        return result;
    }
    
    /**
     * 处理Transformer层
     */
    private TransformerResult processTransformerLayers(Variable x, NdArray attentionMask, TaskType taskType) {
        Variable currentOutput = x;
        List<ExpertRoutingInfo> allRoutingInfo = new ArrayList<>();
        
        for (V3TransformerBlock layer : transformerLayers) {
            currentOutput = layer.forwardWithTaskType(currentOutput, attentionMask, taskType);
            
            // 收集MoE路由信息
            ExpertRoutingInfo routingInfo = layer.getLastRoutingInfo();
            if (routingInfo != null) {
                allRoutingInfo.add(routingInfo);
            }
        }
        
        return new TransformerResult(currentOutput, allRoutingInfo);
    }
    
    /**
     * 获取最后一次前向传播的输出
     */
    public DeepSeekV3Output getLastOutput() {
        return lastOutput;
    }
    
    /**
     * 获取总的MoE损失
     */
    public float getTotalMoELoss() {
        if (lastOutput != null) {
            return lastOutput.moeLoss;
        }
        return 0.0f;
    }
    
    /**
     * 重置所有状态
     */
    public void resetAllStates() {
        lastOutput = null;
        for (V3TransformerBlock layer : transformerLayers) {
            layer.resetRoutingInfo();
        }
    }
    
    // 内部辅助类
    private static class TransformerResult {
        final Variable output;
        final List<ExpertRoutingInfo> allRoutingInfo;
        
        TransformerResult(Variable output, List<ExpertRoutingInfo> allRoutingInfo) {
            this.output = output;
            this.allRoutingInfo = allRoutingInfo;
        }
    }
    
    /**
     * DeepSeek V3输出包装类
     */
    public static class DeepSeekV3Output {
        public final Variable logits;
        public final List<V3ReasoningStep> reasoningSteps;
        public final CodeGenerationBlock.CodeGenerationResult codeInfo;
        public final Variable hiddenStates;
        public final float moeLoss;
        public final List<ExpertRoutingInfo> routingInfo;
        public final TaskType requestedTaskType;
        public final TaskType identifiedTaskType;
        
        public DeepSeekV3Output(Variable logits, List<V3ReasoningStep> reasoningSteps,
                               CodeGenerationBlock.CodeGenerationResult codeInfo, Variable hiddenStates,
                               float moeLoss, List<ExpertRoutingInfo> routingInfo,
                               TaskType requestedTaskType, TaskType identifiedTaskType) {
            this.logits = logits;
            this.reasoningSteps = reasoningSteps;
            this.codeInfo = codeInfo;
            this.hiddenStates = hiddenStates;
            this.moeLoss = moeLoss;
            this.routingInfo = routingInfo;
            this.requestedTaskType = requestedTaskType;
            this.identifiedTaskType = identifiedTaskType;
        }
        
        /**
         * 获取推理质量评分
         */
        public float getReasoningQuality() {
            if (reasoningSteps.isEmpty()) {
                return 0.0f;
            }
            
            return (float) reasoningSteps.stream()
                                       .mapToDouble(V3ReasoningStep::getConfidence)
                                       .average()
                                       .orElse(0.0);
        }
        
        /**
         * 获取代码生成置信度
         */
        public float getCodeConfidence() {
            if (codeInfo != null) {
                return codeInfo.getCodeConfidence();
            }
            return 0.0f;
        }
        
        /**
         * 获取专家使用统计
         */
        public Map<String, Integer> getExpertUsageStats() {
            Map<String, Integer> stats = new HashMap<>();
            
            for (ExpertRoutingInfo info : routingInfo) {
                for (Integer expertId : info.getSelectedExperts()) {
                    String key = "expert_" + expertId;
                    stats.put(key, stats.getOrDefault(key, 0) + 1);
                }
            }
            
            return stats;
        }
    }
    
    // Getters
    public int getVocabSize() {
        return vocabSize;
    }
    
    public int getDModel() {
        return dModel;
    }
    
    public int getNumLayers() {
        return numLayers;
    }
    
    public int getNumHeads() {
        return numHeads;
    }
    
    public int getDFF() {
        return dFF;
    }
    
    public int getNumExperts() {
        return numExperts;
    }
    
    public int getMaxSeqLen() {
        return maxSeqLen;
    }
    
    public float getDropout() {
        return dropout;
    }
    
    public List<V3TransformerBlock> getTransformerLayers() {
        return transformerLayers;
    }
    
    public V3ReasoningBlock getReasoningModule() {
        return reasoningModule;
    }
    
    public CodeGenerationBlock getCodeGeneration() {
        return codeGeneration;
    }
    
    public Map<TaskType, LinearLayer> getOutputHeads() {
        return outputHeads;
    }
    
    @Override
    public String toString() {
        return String.format("DeepSeekV3Block{name='%s', layers=%d, dModel=%d, numExperts=%d, vocabSize=%d}", 
                           name, numLayers, dModel, numExperts, vocabSize);
    }
}