package io.leavesfly.tinyai.nlp.deepseekR1;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Block;
import io.leavesfly.tinyai.nnet.block.transformer.GPT2Block;
import io.leavesfly.tinyai.nnet.layer.transformer.GPT2OutputHead;
import io.leavesfly.tinyai.nnet.layer.transformer.GPT2TokenEmbedding;
import io.leavesfly.tinyai.nnet.layer.transformer.LayerNorm;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.HashMap;

/**
 * DeepSeek-R1 推理模型实现
 * 
 * DeepSeek-R1是一个专门针对复杂推理任务优化的大语言模型，
 * 核心特点是引入了思维链(Chain of Thought)推理机制，
 * 能够在给出最终答案前生成详细的推理过程。
 * 
 * 模型架构特点：
 * 1. 基于Transformer解码器架构
 * 2. 集成思维链(CoT)推理层
 * 3. 多步推理机制
 * 4. 强化学习优化的推理能力
 * 5. 可解释的推理过程输出
 * 
 * 模型结构：
 * Token Embedding + Position Embedding
 * → N × TransformerBlock  
 * → CoT Reasoning Layer (思维链推理层)
 * → Multi-Step Reasoning Engine (多步推理引擎)
 * → Final LayerNorm
 * → Reasoning Output Head (推理输出头)
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekR1Model extends Block {
    
    // 模型超参数
    private int vocabSize;          // 词汇表大小
    private int dModel;             // 模型维度
    private int numLayers;          // Transformer块数量
    private int numHeads;           // 注意力头数量
    private int dFF;                // 前馈网络隐藏维度
    private int maxSeqLength;       // 最大序列长度
    private double dropoutRate;     // Dropout比率
    private int maxReasoningSteps;  // 最大推理步骤数
    private double reasoningThreshold; // 推理置信度阈值
    
    // 模型组件
    private GPT2TokenEmbedding tokenEmbedding;      // Token嵌入层
    private List<GPT2Block> transformerBlocks;      // Transformer块列表
    private CoTReasoningLayer cotReasoningLayer;    // 思维链推理层
    private MultiStepReasoningEngine reasoningEngine; // 多步推理引擎
    private LayerNorm finalLayerNorm;               // 最终层归一化
    private ReasoningOutputHead reasoningOutputHead; // 推理输出头
    
    // 推理状态管理
    private boolean isReasoningMode;                // 是否处于推理模式
    private List<String> reasoningSteps;           // 推理步骤记录
    private Map<String, Object> reasoningContext;  // 推理上下文
    
    /**
     * 构造DeepSeek-R1模型
     * 
     * @param name 模型名称
     * @param vocabSize 词汇表大小
     * @param dModel 模型维度
     * @param numLayers Transformer块数量
     * @param numHeads 注意力头数量
     * @param dFF 前馈网络隐藏维度
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
        
        // 参数验证
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
        
        // 初始化推理状态
        this.isReasoningMode = true;
        this.reasoningSteps = new ArrayList<>();
        this.reasoningContext = new HashMap<>();
        
        init();
    }
    
    /**
     * 使用默认参数的构造函数
     */
    public DeepSeekR1Model(String name, int vocabSize, int dModel, int numLayers, int maxSeqLength) {
        this(name, vocabSize, dModel, numLayers, 
             Math.max(8, dModel / 64), // numHeads: 根据dModel自适应
             dModel * 4,              // dFF: 标准的4倍扩展
             maxSeqLength, 
             10,                      // maxReasoningSteps: 默认10步
             0.1,                     // dropoutRate: 标准dropout
             0.7);                    // reasoningThreshold: 推理置信度阈值
    }
    
    /**
     * 中等规模DeepSeek-R1配置的构造函数
     */
    public DeepSeekR1Model(String name, int vocabSize, int maxSeqLength) {
        this(name, vocabSize, 768, 12, maxSeqLength);
    }
    
    /**
     * 兼容原有构造函数
     */
    public DeepSeekR1Model(String _name, Shape _inputShape) {
        super(_name, _inputShape);
        // 使用默认配置
        this.vocabSize = 50257;      // GPT-2默认词汇表大小
        this.dModel = 768;
        this.numLayers = 12;
        this.numHeads = 12;
        this.dFF = 3072;
        this.maxSeqLength = 1024;
        this.maxReasoningSteps = 10;
        this.dropoutRate = 0.1;
        this.reasoningThreshold = 0.7;
        
        // 初始化推理状态
        this.isReasoningMode = true;
        this.reasoningSteps = new ArrayList<>();
        this.reasoningContext = new HashMap<>();
        
        init();
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

    @Override
    public void init() {
        if (!alreadyInit) {
            // 1. 初始化Token嵌入层
            tokenEmbedding = new GPT2TokenEmbedding(
                name + "_token_embedding", 
                vocabSize, 
                dModel, 
                maxSeqLength, 
                true,  // 使用位置嵌入
                dropoutRate
            );
            addLayer(tokenEmbedding);
            
            // 2. 初始化Transformer块列表
            transformerBlocks = new ArrayList<>();
            for (int i = 0; i < numLayers; i++) {
                GPT2Block block = new GPT2Block(
                    name + "_block_" + i,
                    dModel,
                    numHeads,
                    dFF,
                    dropoutRate
                );
                transformerBlocks.add(block);
                addLayer(block);
            }
            
            // 3. 初始化思维链推理层
            cotReasoningLayer = new CoTReasoningLayer(
                name + "_cot_reasoning",
                dModel,
                maxReasoningSteps,
                reasoningThreshold
            );
            addLayer(cotReasoningLayer);
            
            // 4. 初始化多步推理引擎
            reasoningEngine = new MultiStepReasoningEngine(
                name + "_reasoning_engine",
                dModel,
                maxReasoningSteps,
                reasoningThreshold
            );
            addLayer(reasoningEngine);
            
            // 5. 初始化最终层归一化
            finalLayerNorm = new LayerNorm(name + "_final_ln", dModel);
            addLayer(finalLayerNorm);
            
            // 6. 初始化推理输出头
            reasoningOutputHead = new ReasoningOutputHead(
                name + "_reasoning_output_head",
                dModel,
                vocabSize,
                maxReasoningSteps
            );
            addLayer(reasoningOutputHead);
            
            alreadyInit = true;
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable input = inputs[0];  // shape: (batch_size, seq_len)
        
        // 验证输入序列长度
        int seqLen = input.getValue().getShape().getDimension(1);
        if (seqLen > maxSeqLength) {
            throw new IllegalArgumentException(
                String.format("输入序列长度 %d 超过最大长度 %d", seqLen, maxSeqLength)
            );
        }
        
        // 清空推理步骤记录
        reasoningSteps.clear();
        reasoningContext.clear();
        
        // 1. Token嵌入 + 位置嵌入
        Variable x = tokenEmbedding.layerForward(input);  // shape: (batch_size, seq_len, dModel)
        
        // 2. 通过所有Transformer块
        for (GPT2Block block : transformerBlocks) {
            x = block.layerForward(x);
        }
        
        // 3. 如果处于推理模式，通过思维链推理层
        if (isReasoningMode) {
            x = cotReasoningLayer.layerForward(x);
            
            // 4. 多步推理引擎处理
            x = reasoningEngine.layerForward(x);
        }
        
        // 5. 最终层归一化
        x = finalLayerNorm.layerForward(x);
        
        // 6. 推理输出头得到结果
        Variable output = reasoningOutputHead.layerForward(x);
        
        return output;
    }
    
    /**
     * 设置推理模式
     * 
     * @param isReasoningMode 是否启用推理模式
     */
    public void setReasoningMode(boolean isReasoningMode) {
        this.isReasoningMode = isReasoningMode;
    }
    
    /**
     * 获取推理步骤记录
     * 
     * @return 推理步骤列表
     */
    public List<String> getReasoningSteps() {
        return new ArrayList<>(reasoningSteps);
    }
    
    /**
     * 获取推理上下文
     * 
     * @return 推理上下文信息
     */
    public Map<String, Object> getReasoningContext() {
        return new HashMap<>(reasoningContext);
    }
    
    /**
     * 添加推理步骤
     * 
     * @param step 推理步骤描述
     */
    public void addReasoningStep(String step) {
        reasoningSteps.add(step);
    }
    
    /**
     * 更新推理上下文
     * 
     * @param key 上下文键
     * @param value 上下文值
     */
    public void updateReasoningContext(String key, Object value) {
        reasoningContext.put(key, value);
    }
    
    /**
     * 执行复杂推理任务
     * 
     * @param input 输入问题
     * @return 推理结果，包含思维链过程
     */
    public ReasoningResult performReasoning(NdArray input) {
        // 启用推理模式
        setReasoningMode(true);
        
        try {
            // 前向传播
            Variable inputVar = new Variable(input);
            Variable output = layerForward(inputVar);
            
            // 构建推理结果
            ReasoningResult result = new ReasoningResult();
            result.setFinalAnswer(output.getValue());
            result.setReasoningSteps(new ArrayList<>(reasoningSteps));
            result.setConfidenceScore(calculateConfidenceScore(output.getValue()));
            result.setReasoningContext(new HashMap<>(reasoningContext));
            
            return result;
            
        } catch (Exception e) {
            System.err.println("推理过程中发生错误: " + e.getMessage());
            
            // 返回错误结果
            ReasoningResult errorResult = new ReasoningResult();
            errorResult.setError(true);
            errorResult.setErrorMessage(e.getMessage());
            return errorResult;
        }
    }
    
    /**
     * 计算置信度分数
     * 
     * @param output 模型输出
     * @return 置信度分数
     */
    private double calculateConfidenceScore(NdArray output) {
        // 简单的置信度计算：基于输出概率的最大值
        int seqLen = output.getShape().getDimension(1);
        int lastTimeStep = seqLen - 1;
        
        float maxProb = 0.0f;
        for (int i = 0; i < vocabSize; i++) {
            float prob = output.get(0, lastTimeStep, i);
            maxProb = Math.max(maxProb, prob);
        }
        
        return maxProb;
    }
    
    /**
     * 获取模型配置信息
     */
    public String getModelConfig() {
        return String.format(
            "DeepSeek-R1 Model Config:\n" +
            "  - Vocab Size: %d\n" +
            "  - Model Dim: %d\n" +
            "  - Num Layers: %d\n" +
            "  - Num Heads: %d\n" +
            "  - FFN Dim: %d\n" +
            "  - Max Seq Length: %d\n" +
            "  - Max Reasoning Steps: %d\n" +
            "  - Dropout Rate: %.2f\n" +
            "  - Reasoning Threshold: %.2f\n" +
            "  - Reasoning Mode: %s",
            vocabSize, dModel, numLayers, numHeads, dFF, maxSeqLength, 
            maxReasoningSteps, dropoutRate, reasoningThreshold,
            isReasoningMode ? "Enabled" : "Disabled"
        );
    }
    
    /**
     * 打印模型信息
     */
    public void printModelInfo() {
        System.out.println("\n=== DeepSeek-R1 Model Information ===");
        System.out.println(getModelConfig());
        System.out.println("Reasoning Capabilities:");
        System.out.println("  - Chain of Thought (CoT) Reasoning: ✓");
        System.out.println("  - Multi-Step Reasoning: ✓");
        System.out.println("  - Explainable Reasoning Process: ✓");
        System.out.println("  - Confidence Score Calculation: ✓");
        System.out.println("========================================\n");
    }
    
    /**
     * 创建小型DeepSeek-R1模型的工厂方法
     */
    public static DeepSeekR1Model createTinyModel(String name, int vocabSize) {
        return new DeepSeekR1Model(
            name, vocabSize, 384, 6, 512  // 小型配置
        );
    }
    
    /**
     * 创建中型DeepSeek-R1模型的工厂方法
     */
    public static DeepSeekR1Model createMediumModel(String name, int vocabSize) {
        return new DeepSeekR1Model(
            name, vocabSize, 768, 12, 1024  // 中型配置
        );
    }
    
    /**
     * 创建大型DeepSeek-R1模型的工厂方法
     */
    public static DeepSeekR1Model createLargeModel(String name, int vocabSize) {
        return new DeepSeekR1Model(
            name, vocabSize, 1536, 24, 2048  // 大型配置
        );
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
    
    public GPT2TokenEmbedding getTokenEmbedding() { return tokenEmbedding; }
    public List<GPT2Block> getTransformerBlocks() { return transformerBlocks; }
    public CoTReasoningLayer getCotReasoningLayer() { return cotReasoningLayer; }
    public MultiStepReasoningEngine getReasoningEngine() { return reasoningEngine; }
    public LayerNorm getFinalLayerNorm() { return finalLayerNorm; }
    public ReasoningOutputHead getReasoningOutputHead() { return reasoningOutputHead; }
}