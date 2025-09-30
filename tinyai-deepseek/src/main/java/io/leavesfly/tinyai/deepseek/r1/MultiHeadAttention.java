package io.leavesfly.tinyai.deepseek.r1;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.func.math.Sigmoid;
import io.leavesfly.tinyai.func.matrix.MatMul;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.Parameter;

/**
 * 多头注意力机制实现
 * 
 * 基于TinyAI框架实现的多头注意力层，支持缩放点积注意力，
 * 是Transformer架构的核心组件。
 * 
 * 注意力机制计算公式：
 * Attention(Q,K,V) = softmax(QK^T/√d_k)V
 * 
 * 多头注意力将输入分成多个头，并行计算注意力，最后合并结果。
 * 
 * @author leavesfly
 * @version 1.0
 */
public class MultiHeadAttention extends Layer {
    
    private int dModel;      // 模型维度
    private int numHeads;    // 注意力头数量
    private int dK;          // 每个头的维度
    private double dropout;  // Dropout比率
    
    // 权重参数
    private Parameter wQ;    // Query权重矩阵
    private Parameter wK;    // Key权重矩阵  
    private Parameter wV;    // Value权重矩阵
    private Parameter wO;    // 输出投影权重矩阵
    
    // 偏置参数
    private Parameter bQ;    // Query偏置
    private Parameter bK;    // Key偏置
    private Parameter bV;    // Value偏置
    private Parameter bO;    // 输出投影偏置
    
    /**
     * 构造多头注意力层
     * 
     * @param name 层名称
     * @param dModel 模型维度
     * @param numHeads 注意力头数量
     * @param dropout Dropout比率
     */
    public MultiHeadAttention(String name, int dModel, int numHeads, double dropout) {
        super(name, Shape.of(-1, -1, dModel), Shape.of(-1, -1, dModel));
        
        if (dModel <= 0 || numHeads <= 0) {
            throw new IllegalArgumentException("dModel和numHeads必须大于0");
        }
        if (dModel % numHeads != 0) {
            throw new IllegalArgumentException("dModel必须能被numHeads整除");
        }
        if (dropout < 0.0 || dropout > 1.0) {
            throw new IllegalArgumentException("dropout必须在0.0到1.0之间");
        }
        
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.dK = dModel / numHeads;
        this.dropout = dropout;
        
        init();
    }
    
    /**
     * 默认构造函数（无dropout）
     */
    public MultiHeadAttention(String name, int dModel, int numHeads) {
        this(name, dModel, numHeads, 0.1);
    }
    
    /**
     * 初始化权重矩阵
     * 使用Xavier/Glorot初始化
     */
    private NdArray initializeWeights(Shape shape) {
        NdArray weights = NdArray.of(shape);
        double fanIn = shape.getDimension(0);
        double fanOut = shape.getDimension(1);
        double limit = Math.sqrt(6.0 / (fanIn + fanOut));
        
        // 随机初始化 [-limit, limit]
        for (int i = 0; i < weights.getShape().size(); i++) {
            double value = (Math.random() * 2.0 - 1.0) * limit;
            weights.set((float) value, i);
        }
        
        return weights;
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            // 初始化Query权重和偏置
            wQ = new Parameter(initializeWeights(Shape.of(dModel, dModel)));
            bQ = new Parameter(NdArray.zeros(Shape.of(dModel)));
            params.put("wQ", wQ);
            params.put("bQ", bQ);
            
            // 初始化Key权重和偏置
            wK = new Parameter(initializeWeights(Shape.of(dModel, dModel)));
            bK = new Parameter(NdArray.zeros(Shape.of(dModel)));
            params.put("wK", wK);
            params.put("bK", bK);
            
            // 初始化Value权重和偏置
            wV = new Parameter(initializeWeights(Shape.of(dModel, dModel)));
            bV = new Parameter(NdArray.zeros(Shape.of(dModel)));
            params.put("wV", wV);
            params.put("bV", bV);
            
            // 初始化输出投影权重和偏置
            wO = new Parameter(initializeWeights(Shape.of(dModel, dModel)));
            bO = new Parameter(NdArray.zeros(Shape.of(dModel)));
            params.put("wO", wO);
            params.put("bO", bO);
            
            alreadyInit = true;
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable query = inputs[0];  // shape: (batch_size, seq_len, dModel)
        Variable key = inputs.length > 1 ? inputs[1] : query;    // 自注意力时key=query
        Variable value = inputs.length > 2 ? inputs[2] : key;    // 自注意力时value=key
        Variable mask = inputs.length > 3 ? inputs[3] : null;    // 可选的注意力掩码
        
        return forward(query, key, value, mask);
    }
    
    /**
     * 多头注意力前向传播
     * 
     * @param query Query向量
     * @param key Key向量
     * @param value Value向量
     * @param mask 注意力掩码（可选）
     * @return 注意力输出
     */
    public Variable forward(Variable query, Variable key, Variable value, Variable mask) {
        int batchSize = query.getValue().getShape().getDimension(0);
        int seqLen = query.getValue().getShape().getDimension(1);
        
        // 1. 线性变换得到Q, K, V
        Variable Q = linearTransform(query, wQ, bQ);  // (batch_size, seq_len, dModel)
        Variable K = linearTransform(key, wK, bK);
        Variable V = linearTransform(value, wV, bV);
        
        // 2. 重塑为多头格式
        Q = reshapeForMultiHead(Q, batchSize, seqLen);  // (batch_size, num_heads, seq_len, d_k)
        K = reshapeForMultiHead(K, batchSize, seqLen);
        V = reshapeForMultiHead(V, batchSize, seqLen);
        
        // 3. 计算缩放点积注意力
        Variable attention = scaledDotProductAttention(Q, K, V, mask);
        
        // 4. 重塑并合并多头结果
        attention = reshapeFromMultiHead(attention, batchSize, seqLen);  // (batch_size, seq_len, dModel)
        
        // 5. 输出投影
        Variable output = linearTransform(attention, wO, bO);
        
        return output;
    }
    
    /**
     * 线性变换: x @ weight + bias
     */
    private Variable linearTransform(Variable input, Parameter weight, Parameter bias) {
        MatMul matMul = new MatMul();
        NdArray result = matMul.forward(input.getValue(), weight.getValue());
        
        // 添加偏置
        return addBias(new Variable(result), bias);
    }
    
    /**
     * 重塑张量以适应多头格式
     * (batch_size, seq_len, dModel) -> (batch_size, num_heads, seq_len, d_k)
     */
    private Variable reshapeForMultiHead(Variable input, int batchSize, int seqLen) {
        NdArray inputArray = input.getValue();
        NdArray reshaped = NdArray.of(Shape.of(batchSize, numHeads, seqLen, dK));
        
        // 重新排列数据
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < seqLen; t++) {
                for (int h = 0; h < numHeads; h++) {
                    for (int d = 0; d < dK; d++) {
                        int originalIndex = h * dK + d;
                        float value = inputArray.get(b, t, originalIndex);
                        reshaped.set(value, b, h, t, d);
                    }
                }
            }
        }
        
        return new Variable(reshaped);
    }
    
    /**
     * 从多头格式重塑回原始格式
     * (batch_size, num_heads, seq_len, d_k) -> (batch_size, seq_len, dModel)
     */
    private Variable reshapeFromMultiHead(Variable input, int batchSize, int seqLen) {
        NdArray inputArray = input.getValue();
        NdArray reshaped = NdArray.of(Shape.of(batchSize, seqLen, dModel));
        
        // 重新排列数据
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < seqLen; t++) {
                for (int h = 0; h < numHeads; h++) {
                    for (int d = 0; d < dK; d++) {
                        float value = inputArray.get(b, h, t, d);
                        int targetIndex = h * dK + d;
                        reshaped.set(value, b, t, targetIndex);
                    }
                }
            }
        }
        
        return new Variable(reshaped);
    }
    
    /**
     * 缩放点积注意力计算
     * Attention(Q,K,V) = softmax(QK^T/√d_k)V
     */
    private Variable scaledDotProductAttention(Variable Q, Variable K, Variable V, Variable mask) {
        // 计算注意力分数: Q @ K^T
        Variable scores = computeAttentionScores(Q, K);
        
        // 缩放
        scores = scaleAttentionScores(scores);
        
        // 应用掩码（如果有）
        if (mask != null) {
            scores = applyMask(scores, mask);
        }
        
        // Softmax
        Variable attentionWeights = softmax(scores);
        
        // 应用Dropout（训练时）
        if (dropout > 0.0) {
            attentionWeights = applyDropout(attentionWeights);
        }
        
        // 计算输出: attention_weights @ V
        Variable output = applyAttentionToValues(attentionWeights, V);
        
        return output;
    }
    
    /**
     * 计算注意力分数 Q @ K^T
     */
    private Variable computeAttentionScores(Variable Q, Variable K) {
        NdArray qArray = Q.getValue();
        NdArray kArray = K.getValue();
        
        int batchSize = qArray.getShape().getDimension(0);
        int numHeads = qArray.getShape().getDimension(1);
        int seqLen = qArray.getShape().getDimension(2);
        int dK = qArray.getShape().getDimension(3);
        
        NdArray scores = NdArray.of(Shape.of(batchSize, numHeads, seqLen, seqLen));
        
        // 计算 Q @ K^T
        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < numHeads; h++) {
                for (int i = 0; i < seqLen; i++) {
                    for (int j = 0; j < seqLen; j++) {
                        float score = 0.0f;
                        for (int d = 0; d < dK; d++) {
                            score += qArray.get(b, h, i, d) * kArray.get(b, h, j, d);
                        }
                        scores.set(score, b, h, i, j);
                    }
                }
            }
        }
        
        return new Variable(scores);
    }
    
    /**
     * 缩放注意力分数
     */
    private Variable scaleAttentionScores(Variable scores) {
        double scale = 1.0 / Math.sqrt(dK);
        return multiplyByScalar(scores, scale);
    }
    
    /**
     * 应用注意力掩码
     */
    private Variable applyMask(Variable scores, Variable mask) {
        // 简化实现：将掩码为0的位置设为大负数
        NdArray scoresArray = scores.getValue();
        NdArray maskArray = mask.getValue();
        NdArray result = NdArray.of(scoresArray.getShape());
        
        for (int i = 0; i < scoresArray.getShape().size(); i++) {
            float maskValue = maskArray.get(i);
            float scoreValue = scoresArray.get(i);
            result.set(maskValue == 0.0f ? -1e9f : scoreValue, i);
        }
        
        return new Variable(result);
    }
    
    /**
     * Softmax激活函数
     */
    private Variable softmax(Variable input) {
        NdArray inputArray = input.getValue();
        NdArray result = NdArray.of(inputArray.getShape());
        
        int batchSize = inputArray.getShape().getDimension(0);
        int numHeads = inputArray.getShape().getDimension(1);
        int seqLen = inputArray.getShape().getDimension(2);
        
        // 对最后一个维度进行softmax
        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < numHeads; h++) {
                for (int i = 0; i < seqLen; i++) {
                    // 找到最大值以避免数值溢出
                    float maxVal = Float.NEGATIVE_INFINITY;
                    for (int j = 0; j < seqLen; j++) {
                        maxVal = Math.max(maxVal, inputArray.get(b, h, i, j));
                    }
                    
                    // 计算指数和
                    float expSum = 0.0f;
                    for (int j = 0; j < seqLen; j++) {
                        expSum += Math.exp(inputArray.get(b, h, i, j) - maxVal);
                    }
                    
                    // 计算softmax
                    for (int j = 0; j < seqLen; j++) {
                        float prob = (float) (Math.exp(inputArray.get(b, h, i, j) - maxVal) / expSum);
                        result.set(prob, b, h, i, j);
                    }
                }
            }
        }
        
        return new Variable(result);
    }
    
    /**
     * 应用Dropout（简化实现）
     */
    private Variable applyDropout(Variable input) {
        // 在实际训练中应该根据训练/评估模式决定是否应用dropout
        // 这里简化为不应用dropout
        return input;
    }
    
    /**
     * 将注意力权重应用到Value上
     */
    private Variable applyAttentionToValues(Variable attentionWeights, Variable V) {
        NdArray attArray = attentionWeights.getValue();
        NdArray vArray = V.getValue();
        
        int batchSize = attArray.getShape().getDimension(0);
        int numHeads = attArray.getShape().getDimension(1);
        int seqLen = attArray.getShape().getDimension(2);
        int dK = vArray.getShape().getDimension(3);
        
        NdArray result = NdArray.of(Shape.of(batchSize, numHeads, seqLen, dK));
        
        // 计算 attention_weights @ V
        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < numHeads; h++) {
                for (int i = 0; i < seqLen; i++) {
                    for (int d = 0; d < dK; d++) {
                        float sum = 0.0f;
                        for (int j = 0; j < seqLen; j++) {
                            sum += attArray.get(b, h, i, j) * vArray.get(b, h, j, d);
                        }
                        result.set(sum, b, h, i, d);
                    }
                }
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
        int features = inputArray.getShape().getDimension(2);
        
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < seqLen; t++) {
                for (int f = 0; f < features; f++) {
                    float value = inputArray.get(b, t, f) + biasArray.get(f);
                    result.set(value, b, t, f);
                }
            }
        }
        
        return new Variable(result);
    }
    
    /**
     * 标量乘法
     */
    private Variable multiplyByScalar(Variable input, double scalar) {
        NdArray inputArray = input.getValue();
        NdArray result = NdArray.of(inputArray.getShape());
        
        for (int i = 0; i < inputArray.getShape().size(); i++) {
            result.set((float)(inputArray.get(i) * scalar), i);
        }
        
        return new Variable(result);
    }
    
    // Getter方法
    public int getDModel() { return dModel; }
    public int getNumHeads() { return numHeads; }
    public int getDK() { return dK; }
    public double getDropout() { return dropout; }
    
    @Override
    public String toString() {
        return String.format("MultiHeadAttention(dModel=%d, numHeads=%d, dK=%d, dropout=%.2f)", 
                           dModel, numHeads, dK, dropout);
    }
}