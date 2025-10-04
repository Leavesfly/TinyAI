package io.leavesfly.tinyai.qwen3.layer;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.layer.dnn.LinearLayer;
import io.leavesfly.tinyai.qwen3.Qwen3Config;

/**
 * Qwen3多头注意力机制层
 * 
 * 支持分组查询注意力 (Grouped Query Attention, GQA)：
 * - 查询头数量通常大于键值头数量
 * - 通过重复键值头来匹配查询头数量
 * - 减少KV缓存内存占用40-60%
 * - 保持模型表达能力
 * 
 * 注意力计算公式：
 * Attention(Q,K,V) = softmax(QK^T/√d_k)V
 * 
 * @author 山泽
 * @version 1.0
 */
public class Qwen3AttentionLayer extends Layer {
    
    /** 配置对象 */
    private Qwen3Config config;
    
    /** 注意力头数量 */
    private int numHeads;
    
    /** 键值头数量 */
    private int numKeyValueHeads;
    
    /** 头维度 */
    private int headDim;
    
    /** 键值组数（numHeads / numKeyValueHeads） */
    private int numKeyValueGroups;
    
    /** 查询投影层 */
    private LinearLayer queryProjection;
    
    /** 键投影层 */
    private LinearLayer keyProjection;
    
    /** 值投影层 */
    private LinearLayer valueProjection;
    
    /** 输出投影层 */
    private LinearLayer outputProjection;
    
    /** 旋转位置编码层 */
    private RotaryPositionalEmbeddingLayer rotary;
    
    /**
     * 构造Qwen3注意力层
     * 
     * @param name 层名称
     * @param config Qwen3配置
     */
    public Qwen3AttentionLayer(String name, Qwen3Config config) {
        super(name, Shape.of(-1, -1, config.getHiddenSize()), 
              Shape.of(-1, -1, config.getHiddenSize()));
        
        this.config = config;
        this.numHeads = config.getNumAttentionHeads();
        this.numKeyValueHeads = config.getNumKeyValueHeads();
        this.headDim = config.getHeadDim();
        this.numKeyValueGroups = config.getNumKeyValueGroups();
        
        // 验证配置
        if (config.getHiddenSize() % numHeads != 0) {
            throw new IllegalArgumentException(
                String.format("hiddenSize (%d) 必须能被 numHeads (%d) 整除",
                    config.getHiddenSize(), numHeads));
        }
        
        if (numHeads % numKeyValueHeads != 0) {
            throw new IllegalArgumentException(
                String.format("numHeads (%d) 必须能被 numKeyValueHeads (%d) 整除",
                    numHeads, numKeyValueHeads));
        }
        
        init();
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            int hiddenSize = config.getHiddenSize();
            
            // 初始化线性投影层
            queryProjection = new LinearLayer(
                name + "_query", hiddenSize, numHeads * headDim, false);
            keyProjection = new LinearLayer(
                name + "_key", hiddenSize, numKeyValueHeads * headDim, false);
            valueProjection = new LinearLayer(
                name + "_value", hiddenSize, numKeyValueHeads * headDim, false);
            outputProjection = new LinearLayer(
                name + "_output", numHeads * headDim, hiddenSize, false);
            
            // 初始化旋转位置编码
            rotary = new RotaryPositionalEmbeddingLayer(
                name + "_rope", headDim, config.getMaxPositionEmbeddings(), config.getRopeTheta());
            
            alreadyInit = true;
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable hiddenStates = inputs[0];
        
        // 可选的注意力掩码（因果掩码）
        NdArray attentionMask = null;
        if (inputs.length > 1 && inputs[1] != null) {
            attentionMask = inputs[1].getValue();
        }
        
        return forwardAttention(hiddenStates, attentionMask);
    }
    
    /**
     * 前向注意力计算
     * 
     * @param hiddenStates 输入隐藏状态 [batch_size, seq_len, hidden_size]
     * @param attentionMask 注意力掩码
     * @return 注意力输出
     */
    private Variable forwardAttention(Variable hiddenStates, NdArray attentionMask) {
        NdArray input = hiddenStates.getValue();
        Shape inputShape = input.getShape();
        
        int batchSize = inputShape.getDimension(0);
        int seqLen = inputShape.getDimension(1);
        int hiddenSize = inputShape.getDimension(2);
        
        // 1. 线性投影生成Q、K、V
        // 将3D输入重塑为2D进行线性变换
        NdArray input2D = reshape3DTo2D(input, batchSize, seqLen, hiddenSize);
        
        Variable queryStates = queryProjection.layerForward(new Variable(input2D));
        Variable keyStates = keyProjection.layerForward(new Variable(input2D));
        Variable valueStates = valueProjection.layerForward(new Variable(input2D));
        
        // 2. 重塑为多头形式
        NdArray query = reshape2DToMultiHead(queryStates.getValue(), batchSize, seqLen, numHeads, headDim);
        NdArray key = reshape2DToMultiHead(keyStates.getValue(), batchSize, seqLen, numKeyValueHeads, headDim);
        NdArray value = reshape2DToMultiHead(valueStates.getValue(), batchSize, seqLen, numKeyValueHeads, headDim);
        
        // 3. 应用旋转位置编码
        NdArray[] rotatedQK = rotary.applyRotaryPosEmb(query, key, seqLen);
        NdArray rotatedQuery = rotatedQK[0];
        NdArray rotatedKey = rotatedQK[1];
        
        // 4. 重复键值头以匹配查询头数量（分组查询注意力）
        NdArray expandedKey = repeatKeyValueHeads(rotatedKey, numKeyValueGroups);
        NdArray expandedValue = repeatKeyValueHeads(value, numKeyValueGroups);
        
        // 5. 计算缩放点积注意力
        NdArray attentionOutput = computeScaledDotProductAttention(
            rotatedQuery, expandedKey, expandedValue, attentionMask, batchSize, seqLen);
        
        // 6. 合并多头结果
        NdArray concatenated = concatenateHeads(attentionOutput, batchSize, seqLen, numHeads, headDim);
        
        // 7. 输出投影
        NdArray concat2D = reshape3DTo2D(concatenated, batchSize, seqLen, numHeads * headDim);
        Variable output = outputProjection.layerForward(new Variable(concat2D));
        
        // 8. 重塑回3D
        NdArray result = reshape2DTo3D(output.getValue(), batchSize, seqLen, hiddenSize);
        
        return new Variable(result);
    }
    
    /**
     * 将3D张量重塑为2D用于线性变换
     */
    private NdArray reshape3DTo2D(NdArray input, int batchSize, int seqLen, int hiddenSize) {
        NdArray result = NdArray.of(Shape.of(batchSize * seqLen, hiddenSize));
        
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                for (int h = 0; h < hiddenSize; h++) {
                    result.set(input.get(b, s, h), b * seqLen + s, h);
                }
            }
        }
        
        return result;
    }
    
    /**
     * 将2D张量重塑回3D
     */
    private NdArray reshape2DTo3D(NdArray input, int batchSize, int seqLen, int hiddenSize) {
        NdArray result = NdArray.of(Shape.of(batchSize, seqLen, hiddenSize));
        
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                for (int h = 0; h < hiddenSize; h++) {
                    result.set(input.get(b * seqLen + s, h), b, s, h);
                }
            }
        }
        
        return result;
    }
    
    /**
     * 将2D投影结果重塑为多头形式
     */
    private NdArray reshape2DToMultiHead(NdArray input, int batchSize, int seqLen, int numHeads, int headDim) {
        NdArray result = NdArray.of(Shape.of(batchSize, numHeads, seqLen, headDim));
        
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                for (int h = 0; h < numHeads; h++) {
                    for (int d = 0; d < headDim; d++) {
                        float value = input.get(b * seqLen + s, h * headDim + d);
                        result.set(value, b, h, s, d);
                    }
                }
            }
        }
        
        return result;
    }
    
    /**
     * 重复键值头以匹配查询头数量
     */
    private NdArray repeatKeyValueHeads(NdArray kv, int numRepeats) {
        if (numRepeats == 1) {
            return kv;
        }
        
        Shape kvShape = kv.getShape();
        int batchSize = kvShape.getDimension(0);
        int kvHeads = kvShape.getDimension(1);
        int seqLen = kvShape.getDimension(2);
        int headDim = kvShape.getDimension(3);
        
        NdArray expanded = NdArray.of(Shape.of(batchSize, kvHeads * numRepeats, seqLen, headDim));
        
        for (int b = 0; b < batchSize; b++) {
            for (int kvh = 0; kvh < kvHeads; kvh++) {
                for (int r = 0; r < numRepeats; r++) {
                    int expandedHead = kvh * numRepeats + r;
                    for (int s = 0; s < seqLen; s++) {
                        for (int d = 0; d < headDim; d++) {
                            float value = kv.get(b, kvh, s, d);
                            expanded.set(value, b, expandedHead, s, d);
                        }
                    }
                }
            }
        }
        
        return expanded;
    }
    
    /**
     * 计算缩放点积注意力
     */
    private NdArray computeScaledDotProductAttention(NdArray query, NdArray key, NdArray value, 
                                                   NdArray attentionMask, int batchSize, int seqLen) {
        double scale = 1.0 / Math.sqrt(headDim);
        NdArray attentionOutput = NdArray.of(Shape.of(batchSize, numHeads, seqLen, headDim));
        
        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < numHeads; h++) {
                // 计算注意力分数：Q * K^T
                NdArray scores = NdArray.of(Shape.of(seqLen, seqLen));
                for (int i = 0; i < seqLen; i++) {
                    for (int j = 0; j < seqLen; j++) {
                        float score = 0.0f;
                        for (int d = 0; d < headDim; d++) {
                            score += query.get(b, h, i, d) * key.get(b, h, j, d);
                        }
                        scores.set((float) (score * scale), i, j);
                    }
                }
                
                // 应用因果掩码
                applyCausalMask(scores, seqLen);
                
                // 应用额外掩码（如填充掩码）
                if (attentionMask != null) {
                    applyAttentionMask(scores, attentionMask, b, seqLen);
                }
                
                // Softmax
                applySoftmax(scores, seqLen);
                
                // 注意力加权求和：Attention * V
                for (int i = 0; i < seqLen; i++) {
                    for (int d = 0; d < headDim; d++) {
                        float output = 0.0f;
                        for (int j = 0; j < seqLen; j++) {
                            output += scores.get(i, j) * value.get(b, h, j, d);
                        }
                        attentionOutput.set(output, b, h, i, d);
                    }
                }
            }
        }
        
        return attentionOutput;
    }
    
    /**
     * 应用因果掩码（下三角掩码）
     */
    private void applyCausalMask(NdArray scores, int seqLen) {
        for (int i = 0; i < seqLen; i++) {
            for (int j = i + 1; j < seqLen; j++) {
                scores.set(Float.NEGATIVE_INFINITY, i, j);
            }
        }
    }
    
    /**
     * 应用注意力掩码
     */
    private void applyAttentionMask(NdArray scores, NdArray mask, int batchIdx, int seqLen) {
        // 简化实现：假设mask是[batch_size, seq_len]形式
        for (int i = 0; i < seqLen; i++) {
            if (mask.get(batchIdx, i) == 0) {
                for (int j = 0; j < seqLen; j++) {
                    scores.set(Float.NEGATIVE_INFINITY, i, j);
                }
            }
        }
    }
    
    /**
     * 应用Softmax
     */
    private void applySoftmax(NdArray scores, int seqLen) {
        for (int i = 0; i < seqLen; i++) {
            // 计算最大值用于数值稳定性
            float maxVal = Float.NEGATIVE_INFINITY;
            for (int j = 0; j < seqLen; j++) {
                float val = scores.get(i, j);
                if (!Float.isInfinite(val) && val > maxVal) {
                    maxVal = val;
                }
            }
            
            // 计算exp和sum
            float sum = 0.0f;
            for (int j = 0; j < seqLen; j++) {
                float val = scores.get(i, j);
                if (Float.isInfinite(val)) {
                    scores.set(0.0f, i, j);
                } else {
                    float expVal = (float) Math.exp(val - maxVal);
                    scores.set(expVal, i, j);
                    sum += expVal;
                }
            }
            
            // 归一化
            if (sum > 0) {
                for (int j = 0; j < seqLen; j++) {
                    scores.set(scores.get(i, j) / sum, i, j);
                }
            }
        }
    }
    
    /**
     * 合并多头结果
     */
    private NdArray concatenateHeads(NdArray attention, int batchSize, int seqLen, int numHeads, int headDim) {
        NdArray result = NdArray.of(Shape.of(batchSize, seqLen, numHeads * headDim));
        
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                for (int h = 0; h < numHeads; h++) {
                    for (int d = 0; d < headDim; d++) {
                        float value = attention.get(b, h, s, d);
                        result.set(value, b, s, h * headDim + d);
                    }
                }
            }
        }
        
        return result;
    }
    
    // Getter方法
    public int getNumHeads() { return numHeads; }
    public int getNumKeyValueHeads() { return numKeyValueHeads; }
    public int getHeadDim() { return headDim; }
    public int getNumKeyValueGroups() { return numKeyValueGroups; }
}