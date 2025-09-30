package io.leavesfly.tinyai.deepseek.v3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.Model;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Block;
import io.leavesfly.tinyai.nnet.layer.dnn.LinearLayer;
import io.leavesfly.tinyai.nnet.layer.embedd.Embedding;
import io.leavesfly.tinyai.nnet.layer.transformer.LayerNorm;

import java.util.*;

/**
 * DeepSeek V3 主模型实现
 * 
 * 基于TinyAI架构的DeepSeek V3模型，集成了以下核心特性：
 * 1. Token嵌入和位置嵌入
 * 2. 多层V3TransformerBlock（包含MoE）
 * 3. V3增强推理模块
 * 4. 代码生成专门模块
 * 5. 多任务输出头
 * 6. 任务类型感知处理
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekV3Model extends Block {
    
    // ========== 配置参数 ==========
    private int vocabSize;                    // 词汇表大小
    private int dModel;                       // 模型维度
    private int numLayers;                    // Transformer层数
    private int numHeads;                     // 注意力头数
    private int numExperts;                   // 专家数量
    private int maxSeqLen;                    // 最大序列长度
    private double dropout;                   // dropout比率
    
    // ========== 嵌入层 ==========
    private Embedding tokenEmbedding;         // Token嵌入
    private Embedding positionEmbedding;      // 位置嵌入
    
    // ========== Transformer层 ==========
    private List<V3TransformerBlock> transformerLayers; // V3 Transformer层列表
    
    // ========== V3特有模块 ==========
    private V3ReasoningModule reasoningModule;           // V3推理模块
    private CodeGenerationModule codeGeneration;         // 代码生成模块
    
    // ========== 输出层 ==========
    private LayerNorm finalLayerNorm;                    // 最终层归一化
    private Map<TaskType, Linear> outputHeads;           // 多任务输出头
    
    // ========== 运行时状态 ==========
    private List<ExpertRoutingInfo> allRoutingInfo;      // 所有层的路由信息
    private List<V3ReasoningStep> currentReasoningChain; // 当前推理链
    private CodeGenerationModule.CodeGenerationResult codeInfo; // 代码信息
    private TaskType currentTaskType;                     // 当前任务类型
    private long totalTokensProcessed;                    // 处理的总token数
    
    /**
     * 构造函数
     * 
     * @param name 模型名称
     * @param vocabSize 词汇表大小
     * @param dModel 模型维度
     * @param numLayers Transformer层数
     * @param numHeads 注意力头数
     * @param numExperts 专家数量
     * @param maxSeqLen 最大序列长度
     * @param dropout dropout比率
     */
    public DeepSeekV3Model(String name, int vocabSize, int dModel, int numLayers,
                          int numHeads, int numExperts, int maxSeqLen, double dropout) {
        super(name, 
              Shape.of(-1, maxSeqLen),           // 输入形状: [batch, seq]
              Shape.of(-1, maxSeqLen, vocabSize)); // 输出形状: [batch, seq, vocab]
        
        this.vocabSize = vocabSize;
        this.dModel = dModel;
        this.numLayers = numLayers;
        this.numHeads = numHeads;
        this.numExperts = numExperts;
        this.maxSeqLen = maxSeqLen;
        this.dropout = dropout;
        
        // 初始化运行时状态
        this.allRoutingInfo = new ArrayList<>();
        this.currentReasoningChain = new ArrayList<>();
        this.totalTokensProcessed = 0L;
        
        init();
    }
    
    /**
     * 简化构造函数
     * 
     * @param name 模型名称
     * @param vocabSize 词汇表大小
     * @param dModel 模型维度
     * @param numLayers Transformer层数
     */
    public DeepSeekV3Model(String name, int vocabSize, int dModel, int numLayers) {
        this(name, vocabSize, dModel, numLayers, 
             dModel / 64,  // 默认头数
             8,            // 默认专家数
             2048,         // 默认最大序列长度
             0.1);         // 默认dropout
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            // 1. 初始化嵌入层
            tokenEmbedding = new Embedding(name + "_token_emb", vocabSize, dModel);
            positionEmbedding = new Embedding(name + "_pos_emb", maxSeqLen, dModel);
            tokenEmbedding.init();
            positionEmbedding.init();
            
            // 2. 初始化Transformer层
            transformerLayers = new ArrayList<>();
            for (int i = 0; i < numLayers; i++) {
                V3TransformerBlock block = new V3TransformerBlock(
                    name + "_layer_" + i, dModel, numHeads, numExperts);
                transformerLayers.add(block);
                addLayer(block); // 添加到Block的层管理中
            }
            
            // 3. 初始化V3推理模块
            reasoningModule = new V3ReasoningModule(name + "_reasoning", dModel);
            reasoningModule.init();
            addLayer(reasoningModule);
            
            // 4. 初始化代码生成模块
            codeGeneration = new CodeGenerationModule(name + "_codegen", dModel);
            codeGeneration.init();
            addLayer(codeGeneration);
            
            // 5. 初始化最终层归一化
            finalLayerNorm = new LayerNorm(name + "_final_ln", dModel);
            finalLayerNorm.init();
            addLayer(finalLayerNorm);
            
            // 6. 初始化多任务输出头
            outputHeads = new HashMap<>();
            for (TaskType taskType : TaskType.values()) {
                Linear outputHead = new Linear(name + "_head_" + taskType.getValue(), 
                                             dModel, vocabSize, false);
                outputHead.init();
                outputHeads.put(taskType, outputHead);
                addLayer(outputHead);
            }
            
            alreadyInit = true;
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable inputIds = inputs[0];  // [batch_size, seq_len]
        Variable attentionMask = inputs.length > 1 ? inputs[1] : null;
        TaskType taskType = extractTaskType(inputs);
        
        this.currentTaskType = taskType;
        
        NdArray inputData = inputIds.getValue();
        Shape inputShape = inputData.getShape();
        int batchSize = inputShape.getDimension(0);
        int seqLen = inputShape.getDimension(1);
        
        // 验证序列长度
        if (seqLen > maxSeqLen) {
            throw new IllegalArgumentException(
                String.format("输入序列长度 %d 超过最大长度 %d", seqLen, maxSeqLen));
        }
        
        // 更新token处理统计
        totalTokensProcessed += (long) batchSize * seqLen;
        
        // 1. Token嵌入 + 位置嵌入
        Variable x = computeEmbeddings(inputIds, seqLen);
        
        // 2. 通过所有V3 Transformer层
        allRoutingInfo.clear();
        for (V3TransformerBlock layer : transformerLayers) {
            x = layer.layerForward(x, attentionMask, createTaskTypeInput(taskType));
            
            // 收集路由信息
            ExpertRoutingInfo routingInfo = layer.getLastRoutingInfo();
            if (routingInfo != null) {
                allRoutingInfo.add(routingInfo);
            }
        }
        
        // 3. 最终层归一化
        x = finalLayerNorm.layerForward(x);
        
        // 保存Transformer输出用于推理
        Variable transformerOutput = x;
        
        // 4. V3推理模块处理
        Variable reasoningOutput = reasoningModule.layerForward(transformerOutput);
        this.currentReasoningChain = reasoningModule.getCurrentReasoningChain();
        
        // 5. 代码生成分析（如果是代码任务）
        if (taskType == TaskType.CODING) {
            this.codeInfo = codeGeneration.analyzeCodeGeneration(reasoningOutput);
        }
        
        // 6. 选择适当的输出头
        Linear outputHead = selectOutputHead(taskType);
        Variable finalLogits = outputHead.layerForward(reasoningOutput.reshape(
            Shape.of(batchSize, dModel)));
        
        // 7. 扩展到序列维度
        finalLogits = expandToSequence(finalLogits, seqLen);
        
        return finalLogits;
    }
    
    /**
     * 计算嵌入（Token + 位置）
     * 
     * @param inputIds 输入token IDs
     * @param seqLen 序列长度
     * @return 嵌入结果
     */
    private Variable computeEmbeddings(Variable inputIds, int seqLen) {
        // Token嵌入
        Variable tokenEmb = tokenEmbedding.layerForward(inputIds);
        
        // 位置嵌入
        NdArray posIds = createPositionIds(inputIds.getValue().getShape().getDimension(0), seqLen);
        Variable positionIds = new Variable(posIds);
        Variable posEmb = positionEmbedding.layerForward(positionIds);
        
        // 相加得到最终嵌入
        return tokenEmb.add(posEmb);
    }
    
    /**
     * 创建位置ID
     * 
     * @param batchSize 批大小
     * @param seqLen 序列长度
     * @return 位置ID数组
     */
    private NdArray createPositionIds(int batchSize, int seqLen) {
        double[] posData = new double[batchSize * seqLen];
        
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                posData[b * seqLen + s] = s;
            }
        }
        
        return NdArray.of(posData).reshape(Shape.of(batchSize, seqLen));
    }
    
    /**
     * 从输入中提取任务类型
     * 
     * @param inputs 输入数组
     * @return 任务类型
     */
    private TaskType extractTaskType(Variable[] inputs) {
        if (inputs.length > 2 && inputs[2] != null) {
            // 可以从第三个输入参数中提取任务类型信息
            // 这里简化处理，返回默认任务类型
            return TaskType.GENERAL;
        }
        return TaskType.GENERAL;
    }
    
    /**
     * 创建任务类型输入
     * 
     * @param taskType 任务类型
     * @return 任务类型变量
     */
    private Variable createTaskTypeInput(TaskType taskType) {
        if (taskType == null) {
            return null;
        }
        
        // 创建任务类型的one-hot编码
        double[] encoding = new double[TaskType.values().length];
        encoding[taskType.ordinal()] = 1.0;
        
        return new Variable(NdArray.of(encoding).reshape(Shape.of(1, TaskType.values().length)));
    }
    
    /**
     * 选择输出头
     * 
     * @param taskType 任务类型
     * @return 对应的输出头
     */
    private Linear selectOutputHead(TaskType taskType) {
        if (taskType != null && outputHeads.containsKey(taskType)) {
            return outputHeads.get(taskType);
        }
        return outputHeads.get(TaskType.GENERAL);
    }
    
    /**
     * 将输出扩展到序列维度
     * 
     * @param output 输出张量 [batch, dModel]
     * @param seqLen 序列长度
     * @return 扩展后的张量 [batch, seq, vocab]
     */
    private Variable expandToSequence(Variable output, int seqLen) {
        NdArray outputData = output.getValue();
        Shape outputShape = outputData.getShape();
        int batchSize = outputShape.getDimension(0);
        int vocabSize = outputShape.getDimension(1);
        
        // 简化实现：复制输出到每个序列位置
        double[] outputArray = outputData.toDoubleArray();
        double[] expandedArray = new double[batchSize * seqLen * vocabSize];
        
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                System.arraycopy(outputArray, b * vocabSize,
                               expandedArray, (b * seqLen + s) * vocabSize,
                               vocabSize);
            }
        }
        
        return new Variable(NdArray.of(expandedArray).reshape(Shape.of(batchSize, seqLen, vocabSize)));
    }
    
    /**
     * 计算总的MoE负载均衡损失
     * 
     * @return 总负载均衡损失
     */
    public double computeTotalLoadBalancingLoss() {
        double totalLoss = 0.0;
        for (V3TransformerBlock layer : transformerLayers) {
            totalLoss += layer.computeLoadBalancingLoss();
        }
        return totalLoss;
    }
    
    /**
     * 重置所有MoE统计信息
     */
    public void resetAllMoEStats() {
        for (V3TransformerBlock layer : transformerLayers) {
            layer.resetMoEStats();
        }
    }
    
    /**
     * 获取所有层的专家使用统计
     * 
     * @return 专家使用统计列表
     */
    public List<long[]> getAllLayersExpertUsage() {
        List<long[]> allUsage = new ArrayList<>();
        for (V3TransformerBlock layer : transformerLayers) {
            allUsage.add(layer.getExpertUsageCount());
        }
        return allUsage;
    }
    
    /**
     * 计算模型总参数数量
     * 
     * @return 总参数数
     */
    public long getTotalParameterCount() {
        long totalParams = 0;
        
        // 嵌入层参数
        totalParams += (long) vocabSize * dModel;    // Token嵌入
        totalParams += (long) maxSeqLen * dModel;    // 位置嵌入
        
        // Transformer层参数
        for (V3TransformerBlock layer : transformerLayers) {
            totalParams += layer.getTotalParameterCount();
        }
        
        // 推理模块参数（估算）
        totalParams += (long) dModel * dModel * 10;  // 简化估算
        
        // 代码生成模块参数（估算）
        totalParams += (long) dModel * 1000;         // 简化估算
        
        // 最终LayerNorm参数
        totalParams += 2L * dModel;
        
        // 输出头参数
        totalParams += (long) TaskType.values().length * dModel * vocabSize;
        
        return totalParams;
    }
    
    /**
     * 计算激活参数数量
     * 
     * @return 激活参数数
     */
    public long getActiveParameterCount() {
        long activeParams = 0;
        
        // 嵌入层参数（全部激活）
        activeParams += (long) vocabSize * dModel;
        activeParams += (long) maxSeqLen * dModel;
        
        // Transformer层激活参数
        for (V3TransformerBlock layer : transformerLayers) {
            activeParams += layer.getActiveParameterCount();
        }
        
        // 推理模块参数（全部激活）
        activeParams += (long) dModel * dModel * 10;
        
        // 代码生成模块参数（按需激活）
        if (currentTaskType == TaskType.CODING) {
            activeParams += (long) dModel * 1000;
        }
        
        // 最终LayerNorm参数
        activeParams += 2L * dModel;
        
        // 输出头参数（只激活当前任务的输出头）
        activeParams += (long) dModel * vocabSize;
        
        return activeParams;
    }
    
    /**
     * 获取V3模型信息摘要
     * 
     * @return 模型信息字符串
     */
    public String getV3ModelInfo() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== DeepSeek V3 模型信息 ===\n");
        sb.append(String.format("模型名称: %s\n", name));
        sb.append(String.format("词汇表大小: %d\n", vocabSize));
        sb.append(String.format("模型维度: %d\n", dModel));
        sb.append(String.format("Transformer层数: %d\n", numLayers));
        sb.append(String.format("注意力头数: %d\n", numHeads));
        sb.append(String.format("专家数量: %d\n", numExperts));
        sb.append(String.format("最大序列长度: %d\n", maxSeqLen));
        
        sb.append("\n模型统计:\n");
        sb.append(String.format("  - 总参数数: %,d\n", getTotalParameterCount()));
        sb.append(String.format("  - 激活参数数: %,d\n", getActiveParameterCount()));
        sb.append(String.format("  - 参数效率: %.1f%%\n", 
                 (double) getActiveParameterCount() / getTotalParameterCount() * 100));
        sb.append(String.format("  - 处理Token总数: %,d\n", totalTokensProcessed));
        sb.append(String.format("  - 当前任务类型: %s\n", 
                 currentTaskType != null ? currentTaskType.getValue() : "未知"));
        
        sb.append(String.format("\nMoE统计:\n"));
        sb.append(String.format("  - 总负载均衡损失: %.6f\n", computeTotalLoadBalancingLoss()));
        
        if (!currentReasoningChain.isEmpty()) {
            double avgConfidence = currentReasoningChain.stream()
                    .mapToDouble(V3ReasoningStep::getConfidence)
                    .average()
                    .orElse(0.0);
            sb.append(String.format("\n推理统计:\n"));
            sb.append(String.format("  - 推理步骤数: %d\n", currentReasoningChain.size()));
            sb.append(String.format("  - 平均置信度: %.3f\n", avgConfidence));
        }
        
        if (codeInfo != null) {
            sb.append(String.format("\n代码生成统计:\n"));
            sb.append(String.format("  - 检测语言: %s\n", 
                     codeInfo.getLanguageResult().getDetectedLanguage()));
            sb.append(String.format("  - 代码置信度: %.3f\n", codeInfo.getCodeConfidence()));
        }
        
        sb.append("========================");
        return sb.toString();
    }
    
    /**
     * 获取负载均衡报告
     * 
     * @return 负载均衡详细报告
     */
    public String getLoadBalancingReport() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== DeepSeek V3 负载均衡报告 ===\n");
        
        for (int i = 0; i < transformerLayers.size(); i++) {
            V3TransformerBlock layer = transformerLayers.get(i);
            sb.append(String.format("第%d层:\n", i + 1));
            sb.append(String.format("  总Token数: %d\n", layer.getTotalTokens()));
            sb.append(String.format("  负载均衡损失: %.6f\n", layer.computeLoadBalancingLoss()));
            
            long[] usage = layer.getExpertUsageCount();
            sb.append("  专家使用统计: [");
            for (int j = 0; j < usage.length; j++) {
                if (j > 0) sb.append(", ");
                sb.append(usage[j]);
            }
            sb.append("]\n");
        }
        
        sb.append("==============================");
        return sb.toString();
    }
    
    /**
     * 获取推理链报告
     * 
     * @return 推理链详细报告
     */
    public String getReasoningChainReport() {
        if (reasoningModule != null) {
            return reasoningModule.getReasoningChainSummary();
        }
        return "推理模块未初始化";
    }
    
    /**
     * 获取代码生成报告
     * 
     * @return 代码生成详细报告
     */
    public String getCodeGenerationReport() {
        if (codeGeneration != null) {
            return codeGeneration.getCodeGenerationReport();
        }
        return "代码生成模块未初始化";
    }
    
    /**
     * 重置模型状态
     */
    public void resetState() {
        super.resetState();
        resetAllMoEStats();
        allRoutingInfo.clear();
        currentReasoningChain.clear();
        codeInfo = null;
        currentTaskType = null;
        totalTokensProcessed = 0L;
    }
    
    // ========== 工厂方法 ==========
    
    /**
     * 创建迷你V3模型
     * 
     * @param name 模型名称
     * @return 迷你V3模型
     */
    public static DeepSeekV3Model createTinyV3(String name) {
        return new DeepSeekV3Model(name, 1024, 256, 4, 4, 4, 128, 0.1);
    }
    
    /**
     * 创建小型V3模型
     * 
     * @param name 模型名称
     * @return 小型V3模型
     */
    public static DeepSeekV3Model createSmallV3(String name) {
        return new DeepSeekV3Model(name, 32000, 768, 12, 12, 8, 2048, 0.1);
    }
    
    /**
     * 创建标准V3模型
     * 
     * @param name 模型名称
     * @return 标准V3模型
     */
    public static DeepSeekV3Model createStandardV3(String name) {
        return new DeepSeekV3Model(name, 102400, 2048, 28, 32, 64, 4096, 0.1);
    }
    
    // ========== Getter Methods ==========
    
    public int getVocabSize() { return vocabSize; }
    public int getDModel() { return dModel; }
    public int getNumLayers() { return numLayers; }
    public int getNumHeads() { return numHeads; }
    public int getNumExperts() { return numExperts; }
    public int getMaxSeqLen() { return maxSeqLen; }
    public double getDropout() { return dropout; }
    
    public List<V3TransformerBlock> getTransformerLayers() { return transformerLayers; }
    public V3ReasoningModule getReasoningModule() { return reasoningModule; }
    public CodeGenerationModule getCodeGeneration() { return codeGeneration; }
    
    public List<ExpertRoutingInfo> getAllRoutingInfo() { return allRoutingInfo; }
    public List<V3ReasoningStep> getCurrentReasoningChain() { return currentReasoningChain; }
    public CodeGenerationModule.CodeGenerationResult getCodeInfo() { return codeInfo; }
    public TaskType getCurrentTaskType() { return currentTaskType; }
    public long getTotalTokensProcessed() { return totalTokensProcessed; }
}