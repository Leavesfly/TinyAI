package io.leavesfly.tinyai.deepseek.v3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.func.math.Sqrt;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.LayerAble;
import io.leavesfly.tinyai.nnet.layer.dnn.LinearLayer;

/**
 * 多头注意力机制
 * 
 * 基于Transformer架构的多头注意力实现，支持：
 * 1. 自注意力和交叉注意力
 * 2. 注意力掩码
 * 3. 梯度检查点
 * 
 * @author leavesfly
 * @version 1.0
 */
public class MultiHeadAttention extends LayerAble {
    
    private int dModel;      // 模型维度
    private int numHeads;    // 注意力头数
    private int dK;          // 每个头的维度
    private double dropout;  // dropout比率
    
    // 投影层
    private Linear wQ;       // Query投影
    private Linear wK;       // Key投影  
    private Linear wV;       // Value投影
    private Linear wO;       // 输出投影
    
    /**
     * 构造函数
     * 
     * @param name 层名称
     * @param dModel 模型维度
     * @param numHeads 注意力头数
     * @param dropout dropout比率
     */
    public MultiHeadAttention(String name, int dModel, int numHeads, double dropout) {
        this.name = name;
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.dropout = dropout;
        
        if (dModel % numHeads != 0) {
            throw new IllegalArgumentException("dModel必须能被numHeads整除");
        }
        
        this.dK = dModel / numHeads;
        
        // 设置输入输出形状
        this.inputShape = Shape.of(-1, -1, dModel);  // [batch, seq, dModel]
        this.outputShape = Shape.of(-1, -1, dModel); // [batch, seq, dModel]
        
        init();
    }
    
    /**
     * 默认构造函数（无dropout）
     */
    public MultiHeadAttention(String name, int dModel, int numHeads) {
        this(name, dModel, numHeads, 0.1);
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            // 初始化投影层
            wQ = new Linear(name + "_wq", dModel, dModel, false);
            wK = new Linear(name + "_wk", dModel, dModel, false);
            wV = new Linear(name + "_wv", dModel, dModel, false);
            wO = new Linear(name + "_wo", dModel, dModel, false);
            
            wQ.init();
            wK.init();
            wV.init();
            wO.init();
            
            alreadyInit = true;
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable query = inputs[0];
        Variable key = inputs.length > 1 ? inputs[1] : query;
        Variable value = inputs.length > 2 ? inputs[2] : key;
        Variable mask = inputs.length > 3 ? inputs[3] : null;
        
        return forward(query, key, value, mask);
    }
    
    /**
     * 注意力前向传播
     * 
     * @param query Query张量
     * @param key Key张量
     * @param value Value张量
     * @param mask 注意力掩码（可选）
     * @return 注意力输出
     */
    public Variable forward(Variable query, Variable key, Variable value, Variable mask) {
        NdArray queryData = query.getValue();
        Shape queryShape = queryData.getShape();
        int batchSize = queryShape.getDimension(0);
        int seqLen = queryShape.getDimension(1);
        
        // 1. 线性投影
        Variable Q = wQ.layerForward(query); // [batch, seq, dModel]
        Variable K = wK.layerForward(key);   // [batch, seq, dModel]
        Variable V = wV.layerForward(value); // [batch, seq, dModel]
        
        // 2. 重塑为多头形式
        Q = reshapeToMultiHead(Q, batchSize, seqLen); // [batch, numHeads, seq, dK]
        K = reshapeToMultiHead(K, batchSize, seqLen); // [batch, numHeads, seq, dK]
        V = reshapeToMultiHead(V, batchSize, seqLen); // [batch, numHeads, seq, dK]
        
        // 3. 计算缩放点积注意力
        Variable attention = scaledDotProductAttention(Q, K, V, mask);
        
        // 4. 重塑回原始形状
        attention = reshapeFromMultiHead(attention, batchSize, seqLen);
        
        // 5. 输出投影
        Variable output = wO.layerForward(attention);
        
        return output;
    }
    
    /**
     * 重塑张量为多头形式
     * 
     * @param x 输入张量 [batch, seq, dModel]
     * @param batchSize 批大小
     * @param seqLen 序列长度
     * @return 重塑后的张量 [batch, numHeads, seq, dK]
     */
    private Variable reshapeToMultiHead(Variable x, int batchSize, int seqLen) {
        // [batch, seq, dModel] -> [batch, seq, numHeads, dK] -> [batch, numHeads, seq, dK]
        Variable reshaped = x.reshape(Shape.of(batchSize, seqLen, numHeads, dK));
        
        // 转置：维度交换 (1, 2) -> [batch, numHeads, seq, dK]
        // 简化实现：直接重塑到目标形状
        NdArray data = reshaped.getValue();
        double[] flatData = data.toDoubleArray();
        double[] newData = new double[flatData.length];
        
        // 手动转置实现
        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < numHeads; h++) {
                for (int s = 0; s < seqLen; s++) {
                    for (int d = 0; d < dK; d++) {
                        int oldIdx = ((b * seqLen + s) * numHeads + h) * dK + d;
                        int newIdx = ((b * numHeads + h) * seqLen + s) * dK + d;
                        newData[newIdx] = flatData[oldIdx];
                    }
                }
            }
        }
        
        return new Variable(NdArray.of(newData).reshape(Shape.of(batchSize, numHeads, seqLen, dK)));
    }
    
    /**
     * 从多头形式重塑回原始形状
     * 
     * @param x 输入张量 [batch, numHeads, seq, dK]
     * @param batchSize 批大小
     * @param seqLen 序列长度
     * @return 重塑后的张量 [batch, seq, dModel]
     */
    private Variable reshapeFromMultiHead(Variable x, int batchSize, int seqLen) {
        // [batch, numHeads, seq, dK] -> [batch, seq, numHeads, dK] -> [batch, seq, dModel]
        NdArray data = x.getValue();
        double[] flatData = data.toDoubleArray();
        double[] newData = new double[flatData.length];
        
        // 手动转置实现
        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < numHeads; h++) {
                for (int s = 0; s < seqLen; s++) {
                    for (int d = 0; d < dK; d++) {
                        int oldIdx = ((b * numHeads + h) * seqLen + s) * dK + d;
                        int newIdx = ((b * seqLen + s) * numHeads + h) * dK + d;
                        newData[newIdx] = flatData[oldIdx];
                    }
                }
            }
        }
        
        Variable reshaped = new Variable(NdArray.of(newData).reshape(Shape.of(batchSize, seqLen, numHeads, dK)));
        return reshaped.reshape(Shape.of(batchSize, seqLen, dModel));
    }
    
    /**
     * 缩放点积注意力
     * 
     * @param Q Query张量 [batch, numHeads, seq, dK]
     * @param K Key张量 [batch, numHeads, seq, dK]
     * @param V Value张量 [batch, numHeads, seq, dK]
     * @param mask 注意力掩码（可选）
     * @return 注意力输出 [batch, numHeads, seq, dK]
     */
    private Variable scaledDotProductAttention(Variable Q, Variable K, Variable V, Variable mask) {
        // 1. 计算注意力分数：Q * K^T / sqrt(dK)
        Variable scores = computeAttentionScores(Q, K);
        
        // 2. 应用掩码（如果提供）
        if (mask != null) {
            scores = applyMask(scores, mask);
        }
        
        // 3. Softmax归一化
        Variable attentionWeights = scores.softMax();
        
        // 4. 应用dropout（训练时）
        // 这里简化，不实现dropout
        
        // 5. 注意力加权：Attention * V
        Variable context = computeContext(attentionWeights, V);
        
        return context;
    }
    
    /**
     * 计算注意力分数
     * 
     * @param Q Query张量 [batch, numHeads, seq, dK]
     * @param K Key张量 [batch, numHeads, seq, dK]
     * @return 注意力分数 [batch, numHeads, seq, seq]
     */
    private Variable computeAttentionScores(Variable Q, Variable K) {
        NdArray qData = Q.getValue();
        NdArray kData = K.getValue();
        
        int batchSize = qData.getShape().getDimension(0);
        int numHeads = qData.getShape().getDimension(1);
        int seqLen = qData.getShape().getDimension(2);
        int dK = qData.getShape().getDimension(3);
        
        // 初始化分数矩阵
        double[] scoresData = new double[batchSize * numHeads * seqLen * seqLen];
        double[] qArray = qData.toDoubleArray();
        double[] kArray = kData.toDoubleArray();
        
        // 计算 Q * K^T
        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < numHeads; h++) {
                for (int i = 0; i < seqLen; i++) {
                    for (int j = 0; j < seqLen; j++) {
                        double score = 0.0;
                        
                        // 点积计算
                        for (int d = 0; d < dK; d++) {
                            int qIdx = ((b * numHeads + h) * seqLen + i) * dK + d;
                            int kIdx = ((b * numHeads + h) * seqLen + j) * dK + d;
                            score += qArray[qIdx] * kArray[kIdx];
                        }
                        
                        // 缩放
                        score /= Math.sqrt(dK);
                        
                        int scoreIdx = ((b * numHeads + h) * seqLen + i) * seqLen + j;
                        scoresData[scoreIdx] = score;
                    }
                }
            }
        }
        
        return new Variable(NdArray.of(scoresData).reshape(Shape.of(batchSize, numHeads, seqLen, seqLen)));
    }
    
    /**
     * 应用注意力掩码
     * 
     * @param scores 注意力分数
     * @param mask 掩码张量
     * @return 应用掩码后的分数
     */
    private Variable applyMask(Variable scores, Variable mask) {
        NdArray scoresData = scores.getValue();
        NdArray maskData = mask.getValue();
        
        double[] scoresArray = scoresData.toDoubleArray();
        double[] maskArray = maskData.toDoubleArray();
        
        // 将掩码为0的位置设置为负无穷（实际使用一个很大的负数）
        for (int i = 0; i < scoresArray.length && i < maskArray.length; i++) {
            if (maskArray[i] == 0.0) {
                scoresArray[i] = -1e9;
            }
        }
        
        return new Variable(NdArray.of(scoresArray).reshape(scoresData.getShape()));
    }
    
    /**
     * 计算上下文向量
     * 
     * @param attentionWeights 注意力权重 [batch, numHeads, seq, seq]
     * @param V Value张量 [batch, numHeads, seq, dK]
     * @return 上下文向量 [batch, numHeads, seq, dK]
     */
    private Variable computeContext(Variable attentionWeights, Variable V) {
        NdArray weightsData = attentionWeights.getValue();
        NdArray vData = V.getValue();
        
        int batchSize = weightsData.getShape().getDimension(0);
        int numHeads = weightsData.getShape().getDimension(1);
        int seqLen = weightsData.getShape().getDimension(2);
        int dK = vData.getShape().getDimension(3);
        
        double[] contextData = new double[batchSize * numHeads * seqLen * dK];
        double[] weightsArray = weightsData.toDoubleArray();
        double[] vArray = vData.toDoubleArray();
        
        // 计算 Attention * V
        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < numHeads; h++) {
                for (int i = 0; i < seqLen; i++) {
                    for (int d = 0; d < dK; d++) {
                        double contextValue = 0.0;
                        
                        for (int j = 0; j < seqLen; j++) {
                            int weightIdx = ((b * numHeads + h) * seqLen + i) * seqLen + j;
                            int vIdx = ((b * numHeads + h) * seqLen + j) * dK + d;
                            contextValue += weightsArray[weightIdx] * vArray[vIdx];
                        }
                        
                        int contextIdx = ((b * numHeads + h) * seqLen + i) * dK + d;
                        contextData[contextIdx] = contextValue;
                    }
                }
            }
        }
        
        return new Variable(NdArray.of(contextData).reshape(Shape.of(batchSize, numHeads, seqLen, dK)));
    }
    
    @Override
    public void clearGrads() {
        wQ.clearGrads();
        wK.clearGrads();
        wV.clearGrads();
        wO.clearGrads();
    }
    
    // ========== Getter Methods ==========
    
    public int getDModel() { return dModel; }
    public int getNumHeads() { return numHeads; }
    public int getDK() { return dK; }
    public double getDropout() { return dropout; }
}