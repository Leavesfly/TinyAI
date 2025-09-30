package io.leavesfly.tinyai.nlp.deepseekV3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.layer.dnn.LinearLayer;
import io.leavesfly.tinyai.nnet.layer.transformer.LayerNorm;

/**
 * Multi-head Latent Attention (MLA) 实现
 * 
 * MLA是DeepSeek-V3的核心创新之一，通过潜在空间压缩显著减少KV缓存的内存占用。
 * 
 * 核心思想：
 * 1. 将Keys和Values投影到低维潜在空间中进行缓存
 * 2. 在计算注意力时再将潜在表示解压回完整维度
 * 3. 大幅减少推理时的内存占用（约4-8倍压缩）
 * 
 * 工作流程：
 * 1. Query: 直接从输入计算，维度为 [seq_len, dModel]
 * 2. Key/Value压缩: 输入 -> 潜在空间 [seq_len, dMLA]
 * 3. Key/Value解压: 潜在空间 -> 多头空间 [seq_len, numHeads, headDim]
 * 4. 注意力计算: 使用解压后的K,V与Q计算注意力
 * 5. 输出投影: 注意力输出投影回dModel维度
 * 
 * @author leavesfly
 * @version 1.0
 */
public class MultiHeadLatentAttention extends Layer {
    
    // ========== 基础参数 ==========
    private int dModel;                 // 模型维度
    private int numHeads;               // 注意力头数
    private int headDim;                // 每个头的维度
    private int dMLA;                   // MLA潜在维度
    private int qkNormDim;              // QK归一化维度
    
    // ========== 网络层 ==========
    private LinearLayer queryProjection;    // Query投影层
    private LinearLayer keyLatentProjection; // Key潜在空间投影
    private LinearLayer valueLatentProjection; // Value潜在空间投影
    private LinearLayer keyDecompression;    // Key解压层
    private LinearLayer valueDecompression;  // Value解压层
    private LinearLayer outputProjection;   // 输出投影层
    
    // ========== 归一化层 ==========
    private LayerNorm qkNorm;           // QK归一化（用于训练稳定性）
    
    // ========== 配置参数 ==========
    private boolean useQKNorm;          // 是否使用QK归一化
    private boolean useFlashAttention;  // 是否使用FlashAttention优化
    private double scaleFactor;         // 注意力缩放因子
    private double dropoutRate;         // Dropout比率
    
    // ========== 缓存相关 ==========
    private boolean useKVCache;         // 是否使用KV缓存
    private NdArray cachedKeyLatent;    // 缓存的Key潜在表示
    private NdArray cachedValueLatent;  // 缓存的Value潜在表示
    private int cacheSeqLen;            // 缓存的序列长度
    
    /**
     * 构造Multi-head Latent Attention层
     * 
     * @param name 层名称
     * @param dModel 模型维度
     * @param numHeads 注意力头数
     * @param dMLA MLA潜在维度
     * @param qkNormDim QK归一化维度
     * @param dropoutRate Dropout比率
     * @param useQKNorm 是否使用QK归一化
     */
    public MultiHeadLatentAttention(String name, int dModel, int numHeads, 
                                   int dMLA, int qkNormDim, double dropoutRate, 
                                   boolean useQKNorm) {
        super(name, Shape.of(-1, -1, dModel), Shape.of(-1, -1, dModel));
        
        // 参数验证
        if (dModel <= 0 || numHeads <= 0 || dMLA <= 0) {
            throw new IllegalArgumentException("所有维度参数必须大于0");
        }
        if (dModel % numHeads != 0) {
            throw new IllegalArgumentException("dModel必须能被numHeads整除");
        }
        if (dropoutRate < 0.0 || dropoutRate > 1.0) {
            throw new IllegalArgumentException("Dropout比率必须在0.0到1.0之间");
        }
        
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.headDim = dModel / numHeads;
        this.dMLA = dMLA;
        this.qkNormDim = qkNormDim;
        this.dropoutRate = dropoutRate;
        this.useQKNorm = useQKNorm;
        this.useFlashAttention = false; // 简化实现，后续可扩展
        this.useKVCache = false;
        
        // 计算注意力缩放因子
        this.scaleFactor = 1.0 / Math.sqrt(headDim);
        
        // 初始化缓存
        this.cachedKeyLatent = null;
        this.cachedValueLatent = null;
        this.cacheSeqLen = 0;
        
        init();
    }
    
    /**
     * 简化构造函数
     */
    public MultiHeadLatentAttention(String name, int dModel, int numHeads, int dMLA) {
        this(name, dModel, numHeads, dMLA, Math.max(32, dMLA / 2), 0.1, true);
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            // 1. Query投影层：dModel -> dModel (保持完整维度)
            queryProjection = new LinearLayer(
                name + "_query_proj", 
                dModel, 
                dModel, 
                false  // 不使用偏置
            );
            
            // 2. Key潜在空间投影：dModel -> dMLA (压缩)
            keyLatentProjection = new LinearLayer(
                name + "_key_latent_proj", 
                dModel, 
                dMLA, 
                false
            );
            
            // 3. Value潜在空间投影：dModel -> dMLA (压缩)
            valueLatentProjection = new LinearLayer(
                name + "_value_latent_proj", 
                dModel, 
                dMLA, 
                false
            );
            
            // 4. Key解压层：dMLA -> numHeads * headDim
            keyDecompression = new LinearLayer(
                name + "_key_decomp", 
                dMLA, 
                numHeads * headDim, 
                false
            );
            
            // 5. Value解压层：dMLA -> numHeads * headDim
            valueDecompression = new LinearLayer(
                name + "_value_decomp", 
                dMLA, 
                numHeads * headDim, 
                false
            );
            
            // 6. 输出投影层：dModel -> dModel
            outputProjection = new LinearLayer(
                name + "_output_proj", 
                dModel, 
                dModel, 
                false
            );
            
            // 7. QK归一化层（如果启用）
            if (useQKNorm) {
                qkNorm = new LayerNorm(name + "_qk_norm", qkNormDim);
            }
            
            alreadyInit = true;
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable input = inputs[0];  // shape: [batchSize, seqLen, dModel]
        NdArray inputData = input.getValue();
        
        int batchSize = inputData.getShape().getDimension(0);
        int seqLen = inputData.getShape().getDimension(1);
        
        // 验证输入维度
        if (inputData.getShape().getDimension(2) != dModel) {
            throw new IllegalArgumentException(
                String.format("MLA输入维度不匹配。期望%d，实际%d", 
                             dModel, inputData.getShape().getDimension(2))
            );
        }
        
        // 1. 计算Query（保持完整维度）
        Variable query = queryProjection.layerForward(input);
        // shape: [batchSize, seqLen, dModel]
        
        // 2. 计算Key和Value的潜在表示（压缩）
        Variable keyLatent, valueLatent;
        
        if (useKVCache && cachedKeyLatent != null && cachedValueLatent != null) {
            // 使用缓存的K,V（仅处理新的tokens）
            keyLatent = new Variable(cachedKeyLatent);
            valueLatent = new Variable(cachedValueLatent);
        } else {
            // 计算新的K,V潜在表示
            keyLatent = keyLatentProjection.layerForward(input);
            valueLatent = valueLatentProjection.layerForward(input);
            // shape: [batchSize, seqLen, dMLA]
            
            // 更新缓存
            if (useKVCache) {
                // 创建副本用于缓存
                cachedKeyLatent = NdArray.of(keyLatent.getValue().getMatrix()).reshape(keyLatent.getValue().getShape());
                cachedValueLatent = NdArray.of(valueLatent.getValue().getMatrix()).reshape(valueLatent.getValue().getShape());
                cacheSeqLen = seqLen;
            }
        }
        
        // 3. 解压Key和Value到多头空间
        Variable keyFull = keyDecompression.layerForward(keyLatent);
        Variable valueFull = valueDecompression.layerForward(valueLatent);
        // shape: [batchSize, seqLen, numHeads * headDim]
        
        // 4. 重塑为多头格式
        NdArray queryReshaped = reshapeToMultiHead(query.getValue(), batchSize, seqLen);
        NdArray keyReshaped = reshapeToMultiHead(keyFull.getValue(), batchSize, seqLen);
        NdArray valueReshaped = reshapeToMultiHead(valueFull.getValue(), batchSize, seqLen);
        // shape: [batchSize, numHeads, seqLen, headDim]
        
        // 5. 计算注意力得分
        NdArray attentionOutput = computeAttention(queryReshaped, keyReshaped, valueReshaped, 
                                                  batchSize, seqLen);
        // shape: [batchSize, seqLen, dModel]
        
        // 6. 输出投影
        Variable output = outputProjection.layerForward(new Variable(attentionOutput));
        
        return output;
    }
    
    /**
     * 将张量重塑为多头格式
     * 
     * @param tensor 输入张量 [batchSize, seqLen, dModel]
     * @param batchSize 批大小
     * @param seqLen 序列长度
     * @return 重塑后的张量 [batchSize, numHeads, seqLen, headDim]
     */
    private NdArray reshapeToMultiHead(NdArray tensor, int batchSize, int seqLen) {
        // 简化实现：重塑张量维度
        NdArray reshaped = NdArray.zeros(Shape.of(batchSize, numHeads, seqLen, headDim));
        
        // 重新排列数据
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                for (int h = 0; h < numHeads; h++) {
                    for (int d = 0; d < headDim; d++) {
                        int sourceIdx = h * headDim + d;
                        if (sourceIdx < dModel) {
                            float value = tensor.get(b, s, sourceIdx);
                            reshaped.set(value, b, h, s, d);
                        }
                    }
                }
            }
        }
        
        return reshaped;
    }
    
    /**
     * 计算多头注意力
     * 
     * @param query Query张量 [batchSize, numHeads, seqLen, headDim]
     * @param key Key张量 [batchSize, numHeads, seqLen, headDim]
     * @param value Value张量 [batchSize, numHeads, seqLen, headDim]
     * @param batchSize 批大小
     * @param seqLen 序列长度
     * @return 注意力输出 [batchSize, seqLen, dModel]
     */
    private NdArray computeAttention(NdArray query, NdArray key, NdArray value, 
                                   int batchSize, int seqLen) {
        // 简化的注意力计算实现
        NdArray output = NdArray.zeros(Shape.of(batchSize, seqLen, dModel));
        
        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < numHeads; h++) {
                // 计算注意力得分：Q * K^T
                NdArray scores = NdArray.zeros(Shape.of(seqLen, seqLen));
                
                for (int i = 0; i < seqLen; i++) {
                    for (int j = 0; j < seqLen; j++) {
                        float score = 0.0f;
                        for (int d = 0; d < headDim; d++) {
                            float q_val = query.get(b, h, i, d);
                            float k_val = key.get(b, h, j, d);
                            score += q_val * k_val;
                        }
                        scores.set(score * (float) scaleFactor, i, j);
                    }
                }
                
                // 应用softmax
                softmaxInPlace(scores, seqLen);
                
                // 计算加权Value：Attention * V
                for (int i = 0; i < seqLen; i++) {
                    for (int d = 0; d < headDim; d++) {
                        float weightedSum = 0.0f;
                        for (int j = 0; j < seqLen; j++) {
                            float attention_weight = scores.get(i, j);
                            float v_val = value.get(b, h, j, d);
                            weightedSum += attention_weight * v_val;
                        }
                        
                        // 写入输出张量
                        int outputIdx = h * headDim + d;
                        if (outputIdx < dModel) {
                            output.set(weightedSum, b, i, outputIdx);
                        }
                    }
                }
            }
        }
        
        return output;
    }
    
    /**
     * 就地Softmax计算
     */
    private void softmaxInPlace(NdArray scores, int seqLen) {
        for (int i = 0; i < seqLen; i++) {
            // 找到最大值（数值稳定性）
            float maxVal = Float.NEGATIVE_INFINITY;
            for (int j = 0; j < seqLen; j++) {
                maxVal = Math.max(maxVal, scores.get(i, j));
            }
            
            // 计算exp和sum
            float sum = 0.0f;
            for (int j = 0; j < seqLen; j++) {
                float expVal = (float) Math.exp(scores.get(i, j) - maxVal);
                scores.set(expVal, i, j);
                sum += expVal;
            }
            
            // 归一化
            for (int j = 0; j < seqLen; j++) {
                scores.set(scores.get(i, j) / sum, i, j);
            }
        }
    }
    
    /**
     * 启用KV缓存
     */
    public void enableKVCache() {
        this.useKVCache = true;
    }
    
    /**
     * 禁用KV缓存
     */
    public void disableKVCache() {
        this.useKVCache = false;
        clearKVCache();
    }
    
    /**
     * 清空KV缓存
     */
    public void clearKVCache() {
        this.cachedKeyLatent = null;
        this.cachedValueLatent = null;
        this.cacheSeqLen = 0;
    }
    
    /**
     * 计算缓存压缩比
     */
    public double getCacheCompressionRatio() {
        // 传统注意力缓存大小：2 * seqLen * dModel
        // MLA缓存大小：2 * seqLen * dMLA
        return (double) dMLA / dModel;
    }
    
    /**
     * 计算内存节省量（以字节为单位）
     */
    public long getMemorySavingsBytes(int seqLen) {
        // 传统KV缓存：2 * seqLen * dModel * 4 bytes (float)
        long traditionalCacheSize = 2L * seqLen * dModel * 4;
        
        // MLA缓存：2 * seqLen * dMLA * 4 bytes (float)
        long mlaCacheSize = 2L * seqLen * dMLA * 4;
        
        return traditionalCacheSize - mlaCacheSize;
    }
    
    /**
     * 获取注意力统计信息
     */
    public String getAttentionStats() {
        return String.format(
            "MLA Attention Stats:\n" +
            "  - Model Dim: %d\n" +
            "  - Num Heads: %d\n" +
            "  - Head Dim: %d\n" +
            "  - MLA Latent Dim: %d\n" +
            "  - Cache Compression: %.2fx\n" +
            "  - KV Cache: %s\n" +
            "  - QK Norm: %s\n" +
            "  - Memory Savings (1K seq): %.1f KB",
            dModel, numHeads, headDim, dMLA,
            1.0 / getCacheCompressionRatio(),
            useKVCache ? "enabled" : "disabled",
            useQKNorm ? "enabled" : "disabled",
            getMemorySavingsBytes(1024) / 1024.0
        );
    }
    
    /**
     * 打印注意力配置信息
     */
    public void printAttentionInfo() {
        System.out.println("\n=== Multi-head Latent Attention Info ===");
        System.out.println(getAttentionStats());
        System.out.println("=========================================\n");
    }
    
    @Override
    public int requireInputNum() {
        return 1;
    }
    
    // ========== Getter Methods ==========
    public int getDModel() { return dModel; }
    public int getNumHeads() { return numHeads; }
    public int getHeadDim() { return headDim; }
    public int getDMLA() { return dMLA; }
    public int getQkNormDim() { return qkNormDim; }
    public boolean isUseQKNorm() { return useQKNorm; }
    public boolean isUseKVCache() { return useKVCache; }
    public double getDropoutRate() { return dropoutRate; }
    public int getCacheSeqLen() { return cacheSeqLen; }
    
    public LinearLayer getQueryProjection() { return queryProjection; }
    public LinearLayer getKeyLatentProjection() { return keyLatentProjection; }
    public LinearLayer getValueLatentProjection() { return valueLatentProjection; }
    public LinearLayer getKeyDecompression() { return keyDecompression; }
    public LinearLayer getValueDecompression() { return valueDecompression; }
    public LinearLayer getOutputProjection() { return outputProjection; }
    public LayerNorm getQkNorm() { return qkNorm; }
}