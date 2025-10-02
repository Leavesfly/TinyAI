package io.leavesfly.tinyai.deepseek.r1;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.gpt2.GPT2TokenEmbedding;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Block;
import io.leavesfly.tinyai.nnet.layer.dnn.LinearLayer;
import io.leavesfly.tinyai.nnet.layer.transformer.PositionalEncoding;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * DeepSeek R1主模型Block - 整合所有组件的核心实现
 * 
 * 该Block包含：
 * 1. Token嵌入和位置编码
 * 2. 多层Transformer块
 * 3. 推理模块(ReasoningBlock)
 * 4. 反思模块(ReflectionBlock)
 * 5. 输出投影层
 * 
 * 基于Python实现中的DeepSeekR1Model，使用TinyAI架构重新实现
 */
public class DeepSeekR1Block extends Block {
    
    // 模型配置参数
    private int vocabSize;
    private int dModel;
    private int numLayers;
    private int numHeads;
    private int dFF;
    private int maxSeqLen;
    private double dropout;
    
    // 嵌入层组件
    private GPT2TokenEmbedding tokenEmbedding;
    private PositionalEncoding positionalEncoding;
    
    // Transformer层
    private List<TransformerBlock> transformerLayers;
    
    // DeepSeek R1特有组件
    private ReasoningBlock reasoningModule;
    private ReflectionBlock reflectionModule;
    
    // 输出层
    private LinearLayer outputProjection;
    
    /**
     * 构造DeepSeek R1 Block
     * 
     * @param name DeepSeek R1块名称
     * @param vocabSize 词汇表大小
     * @param dModel 模型维度
     * @param numLayers Transformer层数
     * @param numHeads 注意力头数
     * @param dFF 前馈网络隐藏维度
     * @param maxSeqLen 最大序列长度
     * @param dropout Dropout比率
     */
    public DeepSeekR1Block(String name, int vocabSize, int dModel, int numLayers, 
                          int numHeads, int dFF, int maxSeqLen, double dropout) {
        super(name, Shape.of(-1, maxSeqLen), Shape.of(-1, vocabSize));
        
        this.vocabSize = vocabSize;
        this.dModel = dModel;
        this.numLayers = numLayers;
        this.numHeads = numHeads;
        this.dFF = dFF;
        this.maxSeqLen = maxSeqLen;
        this.dropout = dropout;
        
        init();
    }
    
    /**
     * 使用默认参数的构造函数
     */
    public DeepSeekR1Block(String name, int vocabSize, int dModel, int numLayers, int numHeads) {
        this(name, vocabSize, dModel, numLayers, numHeads, dModel * 4, 512, 0.1);
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            // 1. 初始化嵌入层
            tokenEmbedding = new GPT2TokenEmbedding(name + "_token_embed", vocabSize, dModel, maxSeqLen);
            positionalEncoding = new PositionalEncoding(name + "_pos_encode", dModel, maxSeqLen, dropout);
            
            addLayer(tokenEmbedding);
            addLayer(positionalEncoding);
            
            // 2. 初始化Transformer层
            transformerLayers = new ArrayList<>();
            for (int i = 0; i < numLayers; i++) {
                TransformerBlock transformerLayer = new TransformerBlock(
                    name + "_transformer_" + i, dModel, numHeads, dFF, dropout
                );
                transformerLayers.add(transformerLayer);
                addLayer(transformerLayer);
            }
            
            // 3. 初始化推理模块
            reasoningModule = new ReasoningBlock(name + "_reasoning", dModel);
            addLayer(reasoningModule);
            
            // 4. 初始化反思模块
            reflectionModule = new ReflectionBlock(name + "_reflection", dModel);
            addLayer(reflectionModule);
            
            // 5. 初始化输出投影层
            outputProjection = new LinearLayer(name + "_output_proj", dModel, vocabSize, false);
            addLayer(outputProjection);
            
            alreadyInit = true;
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable inputIds = inputs[0];
        Variable attentionMask = inputs.length > 1 ? inputs[1] : null;
        
        // 1. Token嵌入和位置编码
        Variable tokenEmbeds = tokenEmbedding.layerForward(inputIds);
        Variable x = positionalEncoding.layerForward(tokenEmbeds);
        
        // 2. 通过Transformer层
        for (TransformerBlock transformerLayer : transformerLayers) {
            x = transformerLayer.layerForward(x, attentionMask);
        }
        
        // 保存原始Transformer输出用于反思
        Variable transformerOutput = x;
        
        // 计算序列平均值作为推理模块的输入
        Variable transformerMean = meanAlongSequence(transformerOutput);
        
        // 3. 推理模块处理
        Variable reasoningOutput = reasoningModule.layerForward(transformerOutput);
        
        // 4. 反思模块处理（需要推理输出和原始Transformer输出）
        Variable improvementSuggestion = reflectionModule.layerForward(reasoningOutput, transformerMean);
        
        // 5. 输出投影
        Variable finalOutput = outputProjection.layerForward(reasoningOutput);
        
        return finalOutput;
    }
    
    /**
     * 执行完整的推理过程，返回详细结果
     */
    public DeepSeekR1Result forwardWithReasoningDetails(Variable inputIds, Variable attentionMask) {
        // 1. Token嵌入和位置编码
        Variable tokenEmbeds = tokenEmbedding.layerForward(inputIds);
        Variable x = positionalEncoding.layerForward(tokenEmbeds);
        
        // 2. 通过Transformer层
        for (TransformerBlock transformerLayer : transformerLayers) {
            x = transformerLayer.layerForward(x, attentionMask);
        }
        
        Variable transformerOutput = x;
        Variable transformerMean = meanAlongSequence(transformerOutput);
        
        // 3. 推理模块处理
        Variable reasoningOutput = reasoningModule.layerForward(transformerOutput);
        
        // 4. 反思模块处理
        ReflectionBlock.ReflectionResult reflectionResult = 
            reflectionModule.performReflection(reasoningOutput, transformerMean);
        
        // 5. 输出投影
        Variable logits = outputProjection.layerForward(reasoningOutput);
        
        return new DeepSeekR1Result(
            logits,
            reasoningOutput,
            reflectionResult,
            transformerOutput
        );
    }
    
    /**
     * 计算序列维度的平均值
     */
    private Variable meanAlongSequence(Variable input) {
        NdArray inputData = input.getValue();
        Shape inputShape = inputData.getShape();
        
        int batchSize = inputShape.getDimension(0);
        int seqLen = inputShape.getDimension(1);
        int dModel = inputShape.getDimension(2);
        
        // 创建输出数组：[batch_size, d_model]
        NdArray output = NdArray.zeros(Shape.of(batchSize, dModel));
        
        // 计算每个batch的序列平均值
        for (int b = 0; b < batchSize; b++) {
            for (int d = 0; d < dModel; d++) {
                float sum = 0.0f;
                for (int s = 0; s < seqLen; s++) {
                    sum += inputData.get(b, s, d);
                }
                output.set(sum / seqLen, b, d);
            }
        }
        
        return new Variable(output);
    }
    
    /**
     * 生成文本序列（简化版本）
     */
    public List<Integer> generateSequence(List<Integer> inputTokens, int maxNewTokens, 
                                        float temperature, int topK) {
        List<Integer> generatedTokens = new ArrayList<>(inputTokens);
        
        for (int i = 0; i < maxNewTokens; i++) {
            // 准备输入
            NdArray inputIds = createInputIds(generatedTokens);
            Variable inputVar = new Variable(inputIds);
            
            // 前向传播
            DeepSeekR1Result result = forwardWithReasoningDetails(inputVar, null);
            Variable logits = result.getLogits();
            
            // 获取最后一个时间步的logits
            int nextToken = sampleNextToken(logits, temperature, topK);
            generatedTokens.add(nextToken);
            
            // 如果生成了结束token，停止生成
            if (isEndToken(nextToken)) {
                break;
            }
        }
        
        return generatedTokens;
    }
    
    /**
     * 创建输入ID数组
     */
    private NdArray createInputIds(List<Integer> tokens) {
        int seqLen = Math.min(tokens.size(), maxSeqLen);
        NdArray inputIds = NdArray.zeros(Shape.of(1, seqLen));
        
        for (int i = 0; i < seqLen; i++) {
            inputIds.set(tokens.get(tokens.size() - seqLen + i), 0, i);
        }
        
        return inputIds;
    }
    
    /**
     * 采样下一个token
     */
    private int sampleNextToken(Variable logits, float temperature, int topK) {
        NdArray logitsData = logits.getValue();
        int batchSize = logitsData.getShape().getDimension(0);
        int seqLen = logitsData.getShape().getDimension(1);
        int vocabSize = logitsData.getShape().getDimension(2);
        
        // 获取最后一个时间步的logits
        NdArray lastLogits = NdArray.zeros(Shape.of(vocabSize));
        for (int v = 0; v < vocabSize; v++) {
            lastLogits.set(logitsData.get(0, seqLen - 1, v), v);
        }
        
        // 温度缩放
        if (temperature != 1.0f) {
            for (int v = 0; v < vocabSize; v++) {
                lastLogits.set(lastLogits.get(v) / temperature, v);
            }
        }
        
        // Softmax转换为概率
        NdArray probs = lastLogits.softMax();
        
        // 简化的采样：选择概率最高的token
        int bestToken = 0;
        float bestProb = probs.get(0);
        for (int v = 1; v < vocabSize; v++) {
            if (probs.get(v) > bestProb) {
                bestProb = probs.get(v);
                bestToken = v;
            }
        }
        
        return bestToken;
    }
    
    /**
     * 判断是否是结束token
     */
    private boolean isEndToken(int tokenId) {
        // 简化实现，实际应该根据具体的tokenizer定义
        return tokenId == 0 || tokenId == 1; // 假设0和1是特殊的结束token
    }
    
    /**
     * 获取模型统计信息
     */
    public Map<String, Object> getModelStatistics() {
        Map<String, Object> stats = new HashMap<>();
        
        // 基础配置
        stats.put("vocab_size", vocabSize);
        stats.put("d_model", dModel);
        stats.put("num_layers", numLayers);
        stats.put("num_heads", numHeads);
        stats.put("d_ff", dFF);
        stats.put("max_seq_len", maxSeqLen);
        stats.put("dropout", dropout);
        
        // 参数统计
        Map<String, io.leavesfly.tinyai.nnet.Parameter> allParams = getAllParams();
        long totalParams = 0;
        for (io.leavesfly.tinyai.nnet.Parameter param : allParams.values()) {
            totalParams += param.getValue().getShape().size();
        }
        stats.put("total_parameters", totalParams);
        
        // 层统计
        stats.put("transformer_layers", transformerLayers.size());
        stats.put("reasoning_steps", reasoningModule.getNumReasoningSteps());
        stats.put("reflection_threshold", reflectionModule.getQualityThreshold());
        
        return stats;
    }
    
    /**
     * DeepSeek R1推理结果类
     */
    public static class DeepSeekR1Result {
        private Variable logits;
        private Variable reasoningOutput;
        private ReflectionBlock.ReflectionResult reflectionResult;
        private Variable transformerOutput;
        
        public DeepSeekR1Result(Variable logits, Variable reasoningOutput, 
                               ReflectionBlock.ReflectionResult reflectionResult,
                               Variable transformerOutput) {
            this.logits = logits;
            this.reasoningOutput = reasoningOutput;
            this.reflectionResult = reflectionResult;
            this.transformerOutput = transformerOutput;
        }
        
        // Getters
        public Variable getLogits() { return logits; }
        public Variable getReasoningOutput() { return reasoningOutput; }
        public ReflectionBlock.ReflectionResult getReflectionResult() { return reflectionResult; }
        public Variable getTransformerOutput() { return transformerOutput; }
        
        @Override
        public String toString() {
            return String.format("DeepSeekR1Result{反思质量=%.3f, 需要改进=%s}",
                    reflectionResult.getQualityScore(), 
                    reflectionResult.needsRefinement() ? "是" : "否");
        }
    }
    
    // Getters
    public int getVocabSize() { return vocabSize; }
    public int getDModel() { return dModel; }
    public int getNumLayers() { return numLayers; }
    public int getNumHeads() { return numHeads; }
    public int getDFF() { return dFF; }
    public int getMaxSeqLen() { return maxSeqLen; }
    public double getDropout() { return dropout; }
    public List<TransformerBlock> getTransformerLayers() { return transformerLayers; }
    public ReasoningBlock getReasoningModule() { return reasoningModule; }
    public ReflectionBlock getReflectionModule() { return reflectionModule; }
}