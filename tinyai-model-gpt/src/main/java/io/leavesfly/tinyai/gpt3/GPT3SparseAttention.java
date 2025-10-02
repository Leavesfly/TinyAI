package io.leavesfly.tinyai.gpt3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.layer.dnn.LinearLayer;

/**
 * GPT-3稀疏注意力机制实现
 * 
 * 稀疏注意力是GPT-3处理长序列的关键技术，主要优势：
 * 1. 降低注意力计算复杂度从O(n²)到O(n√n)或O(n log n)
 * 2. 支持更长的上下文窗口
 * 3. 减少内存使用
 * 4. 保持模型性能
 * 
 * 实现策略：
 * - 局部注意力：每个位置只关注邻近的位置
 * - 稀疏全局注意力：在特定位置使用全局注意力
 * - 步长注意力：以固定步长关注远距离位置
 * 
 * @author 山泽
 * @version 1.0
 */
public class GPT3SparseAttention extends Layer {
    
    /** 注意力头数 */
    private int numHeads;
    
    /** 模型维度 */
    private int dModel;
    
    /** 每个头的维度 */
    private int headDim;
    
    /** 局部注意力窗口大小 */
    private int localWindowSize;
    
    /** 全局注意力步长 */
    private int globalStride;
    
    /** 是否启用稀疏模式 */
    private boolean sparseMode;
    
    /** 层索引（用于确定稀疏模式） */
    private int layerIndex;
    
    // 线性变换层
    private LinearLayer queryLayer;
    private LinearLayer keyLayer;
    private LinearLayer valueLayer;
    private LinearLayer outputLayer;
    
    /** 旋转位置编码 */
    private GPT3RotaryEmbedding rotaryEmbedding;
    
    /** 旋转维度比例 */
    private double rotaryPct;
    
    /**
     * 构造稀疏注意力层
     * 
     * @param name 层名称
     * @param dModel 模型维度
     * @param numHeads 注意力头数
     * @param layerIndex 层索引
     * @param config GPT-3配置
     */
    public GPT3SparseAttention(String name, int dModel, int numHeads, int layerIndex, GPT3Config config) {
        super(name, Shape.of(-1, -1, dModel), Shape.of(-1, -1, dModel));
        
        if (dModel % numHeads != 0) {
            throw new IllegalArgumentException("dModel必须能被numHeads整除");
        }
        
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.headDim = dModel / numHeads;
        this.layerIndex = layerIndex;
        this.sparseMode = config.isSparseAttention();
        this.rotaryPct = config.getRotaryPct();
        
        // 稀疏注意力参数
        this.localWindowSize = Math.min(128, config.getNPositions() / 8);  // 自适应窗口大小
        this.globalStride = Math.max(8, config.getNPositions() / 256);     // 自适应步长
        
        init();
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            // 初始化线性变换层
            queryLayer = new LinearLayer(name + "_q_proj", dModel, dModel, false);
            keyLayer = new LinearLayer(name + "_k_proj", dModel, dModel, false);
            valueLayer = new LinearLayer(name + "_v_proj", dModel, dModel, false);
            outputLayer = new LinearLayer(name + "_out_proj", dModel, dModel, false);
            
            // 初始化旋转位置编码
            int rotaryDim = (int) (headDim * rotaryPct);
            if (rotaryDim > 0 && rotaryDim % 2 == 0) {
                rotaryEmbedding = new GPT3RotaryEmbedding(
                    name + "_rope", 
                    rotaryDim, 
                    8192  // 支持较长序列
                );
            }
            
            alreadyInit = true;
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable query = inputs[0];
        Variable key = inputs.length > 1 ? inputs[1] : query;
        Variable value = inputs.length > 2 ? inputs[2] : key;
        
        return sparseMode ? 
            computeSparseAttention(query, key, value) : 
            computeFullAttention(query, key, value);
    }
    
    /**
     * 计算稀疏注意力
     */
    private Variable computeSparseAttention(Variable query, Variable key, Variable value) {
        NdArray queryData = query.getValue();
        int batchSize = queryData.getShape().getDimension(0);
        int seqLen = queryData.getShape().getDimension(1);
        
        // 1. 线性变换
        Variable Q = queryLayer.layerForward(query);
        Variable K = keyLayer.layerForward(key);
        Variable V = valueLayer.layerForward(value);
        
        // 2. 重塑为多头形式
        NdArray qHeads = reshapeToHeads(Q.getValue(), batchSize, seqLen);
        NdArray kHeads = reshapeToHeads(K.getValue(), batchSize, seqLen);
        NdArray vHeads = reshapeToHeads(V.getValue(), batchSize, seqLen);
        
        // 3. 应用旋转位置编码（如果启用）
        if (rotaryEmbedding != null) {
            Variable[] rotatedQK = rotaryEmbedding.applyRotaryPositionEmbedding(
                new Variable(qHeads), new Variable(kHeads), seqLen
            );
            qHeads = rotatedQK[0].getValue();
            kHeads = rotatedQK[1].getValue();
        }
        
        // 4. 计算稀疏注意力
        NdArray attention = computeSparseAttentionScores(qHeads, kHeads, vHeads, batchSize, seqLen);
        
        // 5. 合并多头结果
        NdArray concatenated = concatenateHeads(attention, batchSize, seqLen);
        
        // 6. 输出投影
        Variable output = outputLayer.layerForward(new Variable(concatenated));
        
        return output;
    }
    
    /**
     * 计算完整注意力（回退模式）
     */
    private Variable computeFullAttention(Variable query, Variable key, Variable value) {
        // 使用标准的多头注意力实现作为回退
        NdArray queryData = query.getValue();
        int batchSize = queryData.getShape().getDimension(0);
        int seqLen = queryData.getShape().getDimension(1);
        
        // 线性变换
        Variable Q = queryLayer.layerForward(query);
        Variable K = keyLayer.layerForward(key);
        Variable V = valueLayer.layerForward(value);
        
        // 计算标准注意力
        NdArray qHeads = reshapeToHeads(Q.getValue(), batchSize, seqLen);
        NdArray kHeads = reshapeToHeads(K.getValue(), batchSize, seqLen);
        NdArray vHeads = reshapeToHeads(V.getValue(), batchSize, seqLen);
        
        NdArray attention = computeFullAttentionScores(qHeads, kHeads, vHeads, batchSize, seqLen);
        NdArray concatenated = concatenateHeads(attention, batchSize, seqLen);
        
        return outputLayer.layerForward(new Variable(concatenated));
    }
    
    /**
     * 计算稀疏注意力分数
     */
    private NdArray computeSparseAttentionScores(NdArray query, NdArray key, NdArray value, 
                                               int batchSize, int seqLen) {
        NdArray attention = NdArray.of(Shape.of(batchSize, numHeads, seqLen, headDim));
        double scale = 1.0 / Math.sqrt(headDim);
        
        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < numHeads; h++) {
                // 为每个头计算稀疏注意力
                computeHeadSparseAttention(query, key, value, attention, b, h, seqLen, scale);
            }
        }
        
        return attention;
    }
    
    /**
     * 为单个注意力头计算稀疏注意力
     */
    private void computeHeadSparseAttention(NdArray query, NdArray key, NdArray value, 
                                          NdArray attention, int batch, int head, 
                                          int seqLen, double scale) {
        for (int i = 0; i < seqLen; i++) {
            // 获取当前位置可以关注的位置
            boolean[] attentionMask = createSparseAttentionMask(i, seqLen);
            
            // 计算注意力权重
            float[] weights = new float[seqLen];
            float maxScore = Float.NEGATIVE_INFINITY;
            
            // 计算分数
            for (int j = 0; j < seqLen; j++) {
                if (attentionMask[j]) {
                    float score = 0.0f;
                    for (int d = 0; d < headDim; d++) {
                        score += query.get(batch, head, i, d) * key.get(batch, head, j, d);
                    }
                    weights[j] = (float) (score * scale);
                    maxScore = Math.max(maxScore, weights[j]);
                } else {
                    weights[j] = Float.NEGATIVE_INFINITY;
                }
            }
            
            // Softmax
            float sumExp = 0.0f;
            for (int j = 0; j < seqLen; j++) {
                if (weights[j] != Float.NEGATIVE_INFINITY) {
                    weights[j] = (float) Math.exp(weights[j] - maxScore);
                    sumExp += weights[j];
                } else {
                    weights[j] = 0.0f;
                }
            }
            
            // 归一化
            if (sumExp > 0) {
                for (int j = 0; j < seqLen; j++) {
                    weights[j] /= sumExp;
                }
            }
            
            // 应用权重到值
            for (int d = 0; d < headDim; d++) {
                float output = 0.0f;
                for (int j = 0; j < seqLen; j++) {
                    if (attentionMask[j]) {
                        output += weights[j] * value.get(batch, head, j, d);
                    }
                }
                attention.set(output, batch, head, i, d);
            }
        }
    }
    
    /**
     * 创建稀疏注意力掩码
     */
    private boolean[] createSparseAttentionMask(int queryPos, int seqLen) {
        boolean[] mask = new boolean[seqLen];
        
        // 1. 局部注意力：关注附近位置
        int localStart = Math.max(0, queryPos - localWindowSize / 2);
        int localEnd = Math.min(seqLen, queryPos + localWindowSize / 2 + 1);
        for (int j = localStart; j < localEnd; j++) {
            if (j <= queryPos) {  // 因果掩码
                mask[j] = true;
            }
        }
        
        // 2. 全局注意力：以固定步长关注远距离位置
        for (int j = 0; j <= queryPos; j += globalStride) {
            mask[j] = true;
        }
        
        // 3. 确保总是能看到当前位置
        mask[queryPos] = true;
        
        return mask;
    }
    
    /**
     * 计算完整注意力分数（回退实现）
     */
    private NdArray computeFullAttentionScores(NdArray query, NdArray key, NdArray value, 
                                             int batchSize, int seqLen) {
        // 简化的完整注意力实现
        NdArray attention = NdArray.of(Shape.of(batchSize, numHeads, seqLen, headDim));
        double scale = 1.0 / Math.sqrt(headDim);
        
        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < numHeads; h++) {
                for (int i = 0; i < seqLen; i++) {
                    for (int d = 0; d < headDim; d++) {
                        float output = 0.0f;
                        float sumWeights = 0.0f;
                        
                        for (int j = 0; j <= i; j++) {  // 因果掩码
                            float score = 0.0f;
                            for (int k = 0; k < headDim; k++) {
                                score += query.get(b, h, i, k) * key.get(b, h, j, k);
                            }
                            float weight = (float) Math.exp(score * scale);
                            output += weight * value.get(b, h, j, d);
                            sumWeights += weight;
                        }
                        
                        if (sumWeights > 0) {
                            output /= sumWeights;
                        }
                        attention.set(output, b, h, i, d);
                    }
                }
            }
        }
        
        return attention;
    }
    
    /**
     * 重塑为多头形式
     */
    private NdArray reshapeToHeads(NdArray input, int batchSize, int seqLen) {
        NdArray reshaped = NdArray.of(Shape.of(batchSize, numHeads, seqLen, headDim));
        
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                for (int h = 0; h < numHeads; h++) {
                    for (int d = 0; d < headDim; d++) {
                        float value = input.get(b, s, h * headDim + d);
                        reshaped.set(value, b, h, s, d);
                    }
                }
            }
        }
        
        return reshaped;
    }
    
    /**
     * 合并多头结果
     */
    private NdArray concatenateHeads(NdArray multiHeadOutput, int batchSize, int seqLen) {
        NdArray concatenated = NdArray.of(Shape.of(batchSize, seqLen, dModel));
        
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                for (int h = 0; h < numHeads; h++) {
                    for (int d = 0; d < headDim; d++) {
                        float value = multiHeadOutput.get(b, h, s, d);
                        concatenated.set(value, b, s, h * headDim + d);
                    }
                }
            }
        }
        
        return concatenated;
    }
    
    // ==================== Getter方法 ====================
    
    public int getNumHeads() { return numHeads; }
    public int getDModel() { return dModel; }
    public int getHeadDim() { return headDim; }
    public int getLocalWindowSize() { return localWindowSize; }
    public int getGlobalStride() { return globalStride; }
    public boolean isSparseMode() { return sparseMode; }
    public int getLayerIndex() { return layerIndex; }
    public GPT3RotaryEmbedding getRotaryEmbedding() { return rotaryEmbedding; }
}