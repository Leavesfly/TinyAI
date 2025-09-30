package io.leavesfly.tinyai.deepseek.r1;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.func.matrix.MatMul;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.Parameter;
import io.leavesfly.tinyai.nnet.layer.transformer.LayerNorm;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * DeepSeek R1 主模型
 * 
 * 整合了思维链推理、多步推理和自我反思等核心功能的
 * 完整DeepSeek R1模型实现。
 * 
 * 模型架构：
 * Token Embedding + Position Embedding
 * → N × TransformerBlock
 * → ReasoningModule (推理模块)
 * → ReflectionModule (反思模块)
 * → LayerNorm
 * → Output Projection (输出投影)
 * 
 * 核心特性：
 * 1. 多步思维链推理
 * 2. 自我反思和纠错
 * 3. 置信度评估
 * 4. 可解释的推理过程
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekR1Model extends Layer {
    
    // 模型超参数
    private int vocabSize;              // 词汇表大小
    private int dModel;                 // 模型维度
    private int numLayers;              // Transformer层数
    private int numHeads;               // 注意力头数
    private int dFF;                    // 前馈网络维度
    private int maxSeqLength;           // 最大序列长度
    private int maxReasoningSteps;      // 最大推理步骤数
    private double dropoutRate;         // Dropout比率
    private double reasoningThreshold;  // 推理置信度阈值
    
    // 嵌入层权重
    private Parameter tokenEmbedding;   // Token嵌入权重
    private Parameter positionEmbedding; // 位置嵌入权重
    
    // 模型组件
    private List<TransformerBlock> transformerBlocks;  // Transformer块列表
    private ReasoningModule reasoningModule;           // 推理模块
    private ReflectionModule reflectionModule;         // 反思模块
    private LayerNorm finalLayerNorm;                 // 最终层归一化
    
    // 输出投影
    private Parameter outputProjectionW;  // 输出投影权重
    private Parameter outputProjectionB;  // 输出投影偏置
    
    // 推理状态
    private boolean isReasoningMode;      // 是否处于推理模式
    private ReasoningChain currentChain;  // 当前推理链
    
    /**
     * 构造DeepSeek R1模型
     * 
     * @param name 模型名称
     * @param vocabSize 词汇表大小
     * @param dModel 模型维度
     * @param numLayers Transformer层数
     * @param numHeads 注意力头数
     * @param dFF 前馈网络维度
     * @param maxSeqLength 最大序列长度
     * @param maxReasoningSteps 最大推理步骤数
     * @param dropoutRate Dropout比率
     * @param reasoningThreshold 推理置信度阈值
     */
    public DeepSeekR1Model(String name, int vocabSize, int dModel, int numLayers, 
                          int numHeads, int dFF, int maxSeqLength, 
                          int maxReasoningSteps, double dropoutRate, 
                          double reasoningThreshold) {
        super(name, Shape.of(-1, maxSeqLength), Shape.of(-1, maxSeqLength, vocabSize));
        
        validateParameters(vocabSize, dModel, numLayers, numHeads, dFF, 
                          maxSeqLength, maxReasoningSteps, dropoutRate, reasoningThreshold);
        
        this.vocabSize = vocabSize;
        this.dModel = dModel;
        this.numLayers = numLayers;
        this.numHeads = numHeads;
        this.dFF = dFF;
        this.maxSeqLength = maxSeqLength;
        this.maxReasoningSteps = maxReasoningSteps;
        this.dropoutRate = dropoutRate;
        this.reasoningThreshold = reasoningThreshold;
        this.isReasoningMode = true;
        
        init();
    }
    
    /**
     * 简化构造函数
     */
    public DeepSeekR1Model(String name, int vocabSize, int dModel, int numLayers) {
        this(name, vocabSize, dModel, numLayers, 
             Math.max(8, dModel / 64), // numHeads
             dModel * 4,              // dFF
             512,                     // maxSeqLength
             10,                      // maxReasoningSteps
             0.1,                     // dropoutRate
             0.7);                    // reasoningThreshold
    }
    
    /**
     * 参数验证
     */
    private void validateParameters(int vocabSize, int dModel, int numLayers, int numHeads, 
                                   int dFF, int maxSeqLength, int maxReasoningSteps, 
                                   double dropoutRate, double reasoningThreshold) {
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
    }
    
    /**
     * 初始化权重矩阵
     */
    private NdArray initializeWeights(Shape shape) {
        NdArray weights = NdArray.of(shape);
        double fanIn = shape.getDimension(0);
        double fanOut = shape.getDimension(1);
        double limit = Math.sqrt(6.0 / (fanIn + fanOut));
        
        for (int i = 0; i < weights.getShape().size(); i++) {
            double value = (Math.random() * 2.0 - 1.0) * limit;
            weights.set((float) value, i);
        }
        
        return weights;
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            // 1. 初始化Token嵌入
            tokenEmbedding = new Parameter(initializeWeights(Shape.of(vocabSize, dModel)));
            params.put("tokenEmbedding", tokenEmbedding);
            
            // 2. 初始化位置嵌入
            positionEmbedding = new Parameter(initializeWeights(Shape.of(maxSeqLength, dModel)));
            params.put("positionEmbedding", positionEmbedding);
            
            // 3. 初始化Transformer块
            transformerBlocks = new ArrayList<>();
            for (int i = 0; i < numLayers; i++) {
                TransformerBlock block = new TransformerBlock(
                    name + "_transformer_" + i,
                    dModel,
                    numHeads,
                    dFF,
                    dropoutRate
                );
                transformerBlocks.add(block);
            }
            
            // 4. 初始化推理模块
            reasoningModule = new ReasoningModule(
                name + "_reasoning",
                dModel,
                maxReasoningSteps,
                reasoningThreshold
            );
            
            // 5. 初始化反思模块
            reflectionModule = new ReflectionModule(
                name + "_reflection",
                dModel
            );
            
            // 6. 初始化最终层归一化
            finalLayerNorm = new LayerNorm(name + "_final_norm", dModel);
            
            // 7. 初始化输出投影
            outputProjectionW = new Parameter(initializeWeights(Shape.of(dModel, vocabSize)));
            outputProjectionB = new Parameter(NdArray.zeros(Shape.of(vocabSize)));
            params.put("outputProjectionW", outputProjectionW);
            params.put("outputProjectionB", outputProjectionB);
            
            alreadyInit = true;
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable inputIds = inputs[0];  // shape: (batch_size, seq_len)
        Variable mask = inputs.length > 1 ? inputs[1] : null;  // 可选的注意力掩码
        
        // 验证输入序列长度
        int seqLen = inputIds.getValue().getShape().getDimension(1);
        if (seqLen > maxSeqLength) {
            throw new IllegalArgumentException(
                String.format("输入序列长度 %d 超过最大长度 %d", seqLen, maxSeqLength)
            );
        }
        
        // 开始推理链
        if (isReasoningMode) {
            currentChain = ReasoningChain.create("DeepSeek R1推理任务");
        }
        
        // 1. Token嵌入 + 位置嵌入
        Variable x = performEmbedding(inputIds);  // shape: (batch_size, seq_len, dModel)
        Variable originalInput = meanPooling(x);  // 保存原始输入用于反思
        
        // 2. 通过所有Transformer块
        for (TransformerBlock block : transformerBlocks) {
            x = block.layerForward(x, mask);
        }
        
        // 3. 推理模块（如果启用推理模式）
        Variable reasoningOutput = x;
        if (isReasoningMode) {
            reasoningOutput = reasoningModule.layerForward(x);
            
            // 记录推理步骤到推理链
            List<ReasoningStep> steps = reasoningModule.getReasoningSteps();
            for (ReasoningStep step : steps) {
                currentChain.addStep(step);
            }
        }
        
        // 4. 最终层归一化
        Variable normalized = finalLayerNorm.layerForward(reasoningOutput);
        
        // 5. 输出投影
        Variable output = outputProjection(normalized);
        
        // 6. 自我反思（如果启用推理模式）
        if (isReasoningMode) {
            Variable reasoningPooled = meanPooling(reasoningOutput);
            ReflectionModule.ReflectionResult reflection = 
                reflectionModule.performReflection(reasoningPooled, originalInput);
            
            // 完成推理链
            currentChain.complete(
                "DeepSeek R1推理完成",
                reflection.getReflectionReport()
            );
            currentChain.addMetadata("reflection_result", reflection);
        }
        
        return output;
    }
    
    /**
     * Token嵌入 + 位置嵌入
     */
    private Variable performEmbedding(Variable inputIds) {
        NdArray inputArray = inputIds.getValue();
        int batchSize = inputArray.getShape().getDimension(0);
        int seqLen = inputArray.getShape().getDimension(1);
        
        NdArray result = NdArray.of(Shape.of(batchSize, seqLen, dModel));
        
        // Token嵌入 + 位置嵌入
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < seqLen; t++) {
                int tokenId = (int) inputArray.get(b, t);
                tokenId = Math.max(0, Math.min(tokenId, vocabSize - 1)); // 边界检查
                
                for (int d = 0; d < dModel; d++) {
                    float tokenEmb = tokenEmbedding.getValue().get(tokenId, d);
                    float posEmb = positionEmbedding.getValue().get(t, d);
                    result.set(tokenEmb + posEmb, b, t, d);
                }
            }
        }
        
        return new Variable(result);
    }
    
    /**
     * 输出投影
     */
    private Variable outputProjection(Variable input) {
        MatMul matMul = new MatMul();
        NdArray projected = matMul.forward(input.getValue(), outputProjectionW.getValue());
        
        // 添加偏置
        return addBias(new Variable(projected), outputProjectionB);
    }
    
    /**
     * 平均池化
     */
    private Variable meanPooling(Variable input) {
        NdArray inputArray = input.getValue();
        int batchSize = inputArray.getShape().getDimension(0);
        int seqLen = inputArray.getShape().getDimension(1);
        int dModel = inputArray.getShape().getDimension(2);
        
        NdArray result = NdArray.of(Shape.of(batchSize, dModel));
        
        for (int b = 0; b < batchSize; b++) {
            for (int d = 0; d < dModel; d++) {
                float sum = 0.0f;
                for (int t = 0; t < seqLen; t++) {
                    sum += inputArray.get(b, t, d);
                }
                result.set(sum / seqLen, b, d);
            }
        }
        
        return new Variable(result);
    }
    
    /**
     * 添加偏置
     */
    private Variable addBias(Variable input, Parameter bias) {
        NdArray inputArray = input.getValue();
        NdArray biasArray = bias.getValue();
        NdArray result = NdArray.of(inputArray.getShape());
        
        int batchSize = inputArray.getShape().getDimension(0);
        int seqLen = inputArray.getShape().getDimension(1);
        int vocabSize = inputArray.getShape().getDimension(2);
        
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < seqLen; t++) {
                for (int v = 0; v < vocabSize; v++) {
                    float value = inputArray.get(b, t, v) + biasArray.get(v);
                    result.set(value, b, t, v);
                }
            }
        }
        
        return new Variable(result);
    }
    
    /**
     * 执行推理任务
     * 
     * @param inputIds 输入token序列
     * @param question 问题描述（可选）
     * @return 推理结果
     */
    public DeepSeekR1Result performReasoning(NdArray inputIds, String question) {
        // 启用推理模式
        setReasoningMode(true);
        
        if (question != null && currentChain != null) {
            currentChain.setOriginalQuestion(question);
        }
        
        // 前向传播
        Variable inputVar = new Variable(inputIds);
        Variable output = layerForward(inputVar);
        
        // 构建结果
        DeepSeekR1Result result = new DeepSeekR1Result();
        result.setModelOutput(output.getValue());
        result.setReasoningChain(currentChain);
        
        // 从推理链元数据中获取反思结果
        if (currentChain != null && currentChain.getMetadata().containsKey("reflection_result")) {
            ReflectionModule.ReflectionResult reflection = 
                (ReflectionModule.ReflectionResult) currentChain.getMetadata().get("reflection_result");
            result.setReflectionResult(reflection);
        }
        
        // 计算最终置信度
        if (currentChain != null) {
            result.setFinalConfidence(currentChain.getTotalConfidence());
        }
        
        return result;
    }
    
    /**
     * 生成回复
     * 
     * @param inputIds 输入序列
     * @param maxLength 最大生成长度
     * @return 生成的序列
     */
    public NdArray generateResponse(NdArray inputIds, int maxLength) {
        // 简化的生成实现，实际应用中需要更复杂的解码策略
        Variable input = new Variable(inputIds);
        Variable output = layerForward(input);
        
        // 使用贪心解码获取最可能的token
        return greedyDecode(output.getValue(), maxLength);
    }
    
    /**
     * 贪心解码
     */
    private NdArray greedyDecode(NdArray logits, int maxLength) {
        int batchSize = logits.getShape().getDimension(0);
        int seqLen = logits.getShape().getDimension(1);
        
        // 简化实现：返回最后一个时间步的最大概率token
        NdArray result = NdArray.of(Shape.of(batchSize, 1));
        
        for (int b = 0; b < batchSize; b++) {
            float maxProb = Float.NEGATIVE_INFINITY;
            int bestToken = 0;
            
            for (int v = 0; v < vocabSize; v++) {
                float prob = logits.get(b, seqLen - 1, v);
                if (prob > maxProb) {
                    maxProb = prob;
                    bestToken = v;
                }
            }
            
            result.set(bestToken, b, 0);
        }
        
        return result;
    }
    
    /**
     * 设置推理模式
     */
    public void setReasoningMode(boolean reasoningMode) {
        this.isReasoningMode = reasoningMode;
    }
    
    /**
     * 获取模型配置信息
     */
    public Map<String, Object> getModelConfig() {
        Map<String, Object> config = new HashMap<>();
        config.put("vocabSize", vocabSize);
        config.put("dModel", dModel);
        config.put("numLayers", numLayers);
        config.put("numHeads", numHeads);
        config.put("dFF", dFF);
        config.put("maxSeqLength", maxSeqLength);
        config.put("maxReasoningSteps", maxReasoningSteps);
        config.put("dropoutRate", dropoutRate);
        config.put("reasoningThreshold", reasoningThreshold);
        config.put("isReasoningMode", isReasoningMode);
        return config;
    }
    
    /**
     * 打印模型信息
     */
    public void printModelInfo() {
        System.out.println("\n=== DeepSeek R1 Model Information ===");
        System.out.println("架构配置:");
        System.out.printf("  词汇表大小: %d\n", vocabSize);
        System.out.printf("  模型维度: %d\n", dModel);
        System.out.printf("  Transformer层数: %d\n", numLayers);
        System.out.printf("  注意力头数: %d\n", numHeads);
        System.out.printf("  前馈网络维度: %d\n", dFF);
        System.out.printf("  最大序列长度: %d\n", maxSeqLength);
        System.out.printf("  最大推理步骤: %d\n", maxReasoningSteps);
        System.out.printf("  Dropout比率: %.3f\n", dropoutRate);
        System.out.printf("  推理阈值: %.3f\n", reasoningThreshold);
        System.out.printf("  推理模式: %s\n", isReasoningMode ? "启用" : "禁用");
        
        System.out.println("\n核心能力:");
        System.out.println("  ✓ 多步思维链推理");
        System.out.println("  ✓ 自我反思和纠错");
        System.out.println("  ✓ 置信度评估");
        System.out.println("  ✓ 可解释推理过程");
        System.out.println("  ✓ 自适应推理深度");
        
        int totalParams = calculateTotalParameters();
        System.out.printf("\n总参数量: %,d\n", totalParams);
        System.out.println("=====================================\n");
    }
    
    /**
     * 计算总参数数量
     */
    private int calculateTotalParameters() {
        int embeddingParams = vocabSize * dModel + maxSeqLength * dModel;
        int transformerParams = numLayers * (
            4 * dModel * dModel + 4 * dModel +  // 注意力层
            dModel * dFF + dFF + dFF * dModel + dModel +  // 前馈层
            2 * 2 * dModel  // LayerNorm层
        );
        int reasoningParams = estimateReasoningParams();
        int reflectionParams = estimateReflectionParams();
        int outputParams = dModel * vocabSize + vocabSize;
        
        return embeddingParams + transformerParams + reasoningParams + reflectionParams + outputParams;
    }
    
    private int estimateReasoningParams() {
        // 推理模块参数估算
        return dModel * dModel * 8 + dModel * 8;  // 简化估算
    }
    
    private int estimateReflectionParams() {
        // 反思模块参数估算
        return dModel * dModel * 6 + dModel * 6;  // 简化估算
    }
    
    // Getter方法
    public int getVocabSize() { return vocabSize; }
    public int getDModel() { return dModel; }
    public int getNumLayers() { return numLayers; }
    public int getNumHeads() { return numHeads; }
    public int getDFF() { return dFF; }
    public int getMaxSeqLength() { return maxSeqLength; }
    public int getMaxReasoningSteps() { return maxReasoningSteps; }
    public double getDropoutRate() { return dropoutRate; }
    public double getReasoningThreshold() { return reasoningThreshold; }
    public boolean isReasoningMode() { return isReasoningMode; }
    public ReasoningChain getCurrentChain() { return currentChain; }
    
    public List<TransformerBlock> getTransformerBlocks() { return transformerBlocks; }
    public ReasoningModule getReasoningModule() { return reasoningModule; }
    public ReflectionModule getReflectionModule() { return reflectionModule; }
    public LayerNorm getFinalLayerNorm() { return finalLayerNorm; }
    
    @Override
    public String toString() {
        return String.format("DeepSeekR1Model(vocab=%d, dModel=%d, layers=%d, reasoning=%s)", 
                           vocabSize, dModel, numLayers, isReasoningMode ? "ON" : "OFF");
    }
}