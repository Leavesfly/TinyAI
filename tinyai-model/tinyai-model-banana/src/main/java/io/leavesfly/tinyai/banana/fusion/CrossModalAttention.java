package io.leavesfly.tinyai.banana.fusion;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.func.matrix.Permute;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Dropout;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;

/**
 * 跨模态注意力层 (Cross-Modal Attention)
 * 
 * 实现文本-图像特征之间的跨模态交互,这是多模态模型的核心机制。
 * 
 * 核心思想:
 * - Query来自一个模态(如文本特征)
 * - Key和Value来自另一个模态(如图像特征)
 * - 通过注意力机制实现跨模态信息融合
 * 
 * 计算流程:
 * 1. Q = textFeatures @ W_Q
 * 2. K = imageFeatures @ W_K
 * 3. V = imageFeatures @ W_V
 * 4. Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
 * 5. Output = Attention @ W_O
 * 
 * 应用场景:
 * - 文本引导的图像理解
 * - 图像引导的文本生成
 * - 多模态特征融合
 * 
 * @author leavesfly
 * @version 1.0
 */
public class CrossModalAttention extends Module {
    
    private final int hiddenSize;    // 隐藏层维度
    private final int numHeads;      // 注意力头数
    private final int headDim;       // 每个头的维度
    private final float dropout;     // Dropout比率
    
    // 投影层
    private final Linear queryProj;   // Query投影(来自模态1)
    private final Linear keyProj;     // Key投影(来自模态2)
    private final Linear valueProj;   // Value投影(来自模态2)
    private final Linear outputProj;  // 输出投影
    
    // Dropout层
    private final Dropout attnDropout;
    
    /**
     * 构造函数
     * 
     * @param name 模块名称
     * @param hiddenSize 隐藏层维度
     * @param numHeads 注意力头数
     * @param dropout Dropout比率
     */
    public CrossModalAttention(String name, int hiddenSize, int numHeads, float dropout) {
        super(name);
        
        if (hiddenSize % numHeads != 0) {
            throw new IllegalArgumentException(
                "hiddenSize必须能被numHeads整除: " + hiddenSize + " % " + numHeads + " != 0"
            );
        }
        
        this.hiddenSize = hiddenSize;
        this.numHeads = numHeads;
        this.headDim = hiddenSize / numHeads;
        this.dropout = dropout;
        
        // 初始化投影层
        this.queryProj = new Linear(name + "_q_proj", hiddenSize, hiddenSize, true);
        this.keyProj = new Linear(name + "_k_proj", hiddenSize, hiddenSize, true);
        this.valueProj = new Linear(name + "_v_proj", hiddenSize, hiddenSize, true);
        this.outputProj = new Linear(name + "_o_proj", hiddenSize, hiddenSize, true);
        
        registerModule("q_proj", queryProj);
        registerModule("k_proj", keyProj);
        registerModule("v_proj", valueProj);
        registerModule("o_proj", outputProj);
        
        // 初始化Dropout
        this.attnDropout = new Dropout(name + "_attn_dropout", dropout);
        registerModule("attn_dropout", attnDropout);
        
        init();
    }
    
    @Override
    public void resetParameters() {
        // 子模块的参数由其自己初始化
    }
    
    /**
     * 前向传播
     * 
     * @param inputs inputs[0]: query特征(如文本) [batch, query_len, hidden_size]
     *               inputs[1]: key/value特征(如图像) [batch, kv_len, hidden_size]
     * @return 跨模态融合后的特征 [batch, query_len, hidden_size]
     */
    @Override
    public Variable forward(Variable... inputs) {
        if (inputs == null || inputs.length < 2) {
            throw new IllegalArgumentException(
                "CrossModalAttention需要2个输入: queryFeatures和kvFeatures"
            );
        }
        
        Variable queryFeatures = inputs[0];  // 模态1特征(如文本)
        Variable kvFeatures = inputs[1];     // 模态2特征(如图像)
        
        // 获取形状信息
        int[] queryShape = queryFeatures.getValue().getShape().getShapeDims();
        int[] kvShape = kvFeatures.getValue().getShape().getShapeDims();
        
        int batchSize = queryShape[0];
        int queryLen = queryShape[1];
        int kvLen = kvShape[1];
        
        // 1. 投影Q, K, V
        Variable Q = queryProj.forward(queryFeatures);  // [batch, query_len, hidden_size]
        Variable K = keyProj.forward(kvFeatures);       // [batch, kv_len, hidden_size]
        Variable V = valueProj.forward(kvFeatures);     // [batch, kv_len, hidden_size]
        
        // 2. 分割成多头
        Q = splitHeads(Q, batchSize, queryLen);         // [batch, num_heads, query_len, head_dim]
        K = splitHeads(K, batchSize, kvLen);            // [batch, num_heads, kv_len, head_dim]
        V = splitHeads(V, batchSize, kvLen);            // [batch, num_heads, kv_len, head_dim]
        
        // 3. 计算跨模态注意力
        Variable attnOutput = scaledDotProductAttention(Q, K, V);
        
        // 4. 合并多头
        Variable merged = mergeHeads(attnOutput, batchSize, queryLen);
        
        // 5. 输出投影
        Variable output = outputProj.forward(merged);
        
        return output;
    }
    
    /**
     * 分割成多头
     * 
     * [batch, seq_len, hidden_size] -> [batch, num_heads, seq_len, head_dim]
     */
    private Variable splitHeads(Variable x, int batchSize, int seqLen) {
        // [batch, seq_len, hidden_size] -> [batch, seq_len, num_heads, head_dim]
        Variable reshaped = x.reshape(Shape.of(batchSize, seqLen, numHeads, headDim));
        
        // [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
        return new Permute(0, 2, 1, 3).call(reshaped);
    }
    
    /**
     * 合并多头
     * 
     * [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, hidden_size]
     */
    private Variable mergeHeads(Variable x, int batchSize, int seqLen) {
        // [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, num_heads, head_dim]
        Variable permuted = new Permute(0, 2, 1, 3).call(x);
        
        // [batch, seq_len, num_heads, head_dim] -> [batch, seq_len, hidden_size]
        return permuted.reshape(Shape.of(batchSize, seqLen, hiddenSize));
    }
    
    /**
     * 缩放点积注意力
     * 
     * Attention(Q, K, V) = softmax(QK^T / sqrt(head_dim)) V
     * 
     * @param Q Query张量 [batch, num_heads, query_len, head_dim]
     * @param K Key张量 [batch, num_heads, kv_len, head_dim]
     * @param V Value张量 [batch, num_heads, kv_len, head_dim]
     * @return 注意力输出 [batch, num_heads, query_len, head_dim]
     */
    private Variable scaledDotProductAttention(Variable Q, Variable K, Variable V) {
        // 1. 计算Q * K^T
        // K: [batch, num_heads, kv_len, head_dim]
        // K^T: [batch, num_heads, head_dim, kv_len]
        Variable KT = new Permute(0, 1, 3, 2).call(K);
        
        // 2. Q @ K^T
        // [batch, num_heads, query_len, head_dim] @ [batch, num_heads, head_dim, kv_len]
        // -> [batch, num_heads, query_len, kv_len]
        Variable scores = Q.matMul(KT);
        
        // 3. 缩放
        double scale = Math.sqrt(headDim);
        Variable scaledScores = scores.div(new Variable((float) scale));
        
        // 4. Softmax(在最后一维kv_len上实施，softMax()默认在最后一维上计算)
        Variable attnWeights = scaledScores.softMax();
        
        // 5. 应用Dropout(训练时)
        if (isTraining() && dropout > 0) {
            attnWeights = attnDropout.forward(attnWeights);
        }
        
        // 6. 计算注意力输出: attn_weights @ V
        // [batch, num_heads, query_len, kv_len] @ [batch, num_heads, kv_len, head_dim]
        // -> [batch, num_heads, query_len, head_dim]
        Variable output = attnWeights.matMul(V);
        
        return output;
    }
    
    // ==================== Getter方法 ====================
    
    public int getHiddenSize() {
        return hiddenSize;
    }
    
    public int getNumHeads() {
        return numHeads;
    }
    
    public int getHeadDim() {
        return headDim;
    }
    
    public float getDropout() {
        return dropout;
    }
    
    @Override
    public String toString() {
        return String.format(
            "CrossModalAttention{hiddenSize=%d, numHeads=%d, headDim=%d, dropout=%.2f}",
            hiddenSize, numHeads, headDim, dropout
        );
    }
}
