package io.leavesfly.tinyai.nnet.layer.transformer;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.layer.dnn.LinearLayer;

import java.util.ArrayList;
import java.util.List;

/**
 * 多头注意力机制层实现
 * <p>
 * 多头注意力是Transformer的核心组件，通过并行计算多个注意力头来捕获不同子空间的信息。
 * <p>
 * Attention(Q,K,V) = softmax(QK^T/√d_k)V
 * MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O
 * <p>
 * 其中 head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
 */
public class MultiHeadAttention extends Layer {

    private int numHeads;
    private int dModel;
    private int dK;
    private int dV;

    // 线性变换层
    private LinearLayer queryLayer;
    private LinearLayer keyLayer;
    private LinearLayer valueLayer;
    private LinearLayer outputLayer;

    private boolean useMask;

    /**
     * 构造多头注意力层
     *
     * @param name     层名称
     * @param dModel   模型维度
     * @param numHeads 注意力头数
     * @param useMask  是否使用掩码（解码器中需要）
     */
    public MultiHeadAttention(String name, int dModel, int numHeads, boolean useMask) {
        super(name, Shape.of(-1, -1, dModel), Shape.of(-1, -1, dModel));

        if (dModel % numHeads != 0) {
            throw new IllegalArgumentException("dModel must be divisible by numHeads");
        }

        this.dModel = dModel;
        this.numHeads = numHeads;
        this.dK = dModel / numHeads;
        this.dV = dModel / numHeads;
        this.useMask = useMask;

        init();
    }

    @Override
    public void init() {
        if (!alreadyInit) {
            // 初始化线性变换层
            queryLayer = new LinearLayer(name + "_query", dModel, dModel, false);
            keyLayer = new LinearLayer(name + "_key", dModel, dModel, false);
            valueLayer = new LinearLayer(name + "_value", dModel, dModel, false);
            outputLayer = new LinearLayer(name + "_output", dModel, dModel, false);

            alreadyInit = true;
        }
    }

    @Override
    public Variable layerForward(Variable... inputs) {
        Variable query = inputs[0];
        Variable key = inputs.length > 1 ? inputs[1] : query;
        Variable value = inputs.length > 2 ? inputs[2] : key;

        NdArray queryData = query.getValue();
        NdArray keyData = key.getValue();
        NdArray valueData = value.getValue();

        int batchSize = queryData.getShape().getDimension(0);
        int querySeqLen = queryData.getShape().getDimension(1);
        int keySeqLen = keyData.getShape().getDimension(1);
        int valueSeqLen = valueData.getShape().getDimension(1);
        
        // 验证key和value的序列长度必须相同
        if (keySeqLen != valueSeqLen) {
            throw new IllegalArgumentException(
                String.format("Key序列长度(%d)必须与Value序列长度(%d)相同", keySeqLen, valueSeqLen)
            );
        }

        // 将三维张量重塑为二维矩阵进行线性变换
        NdArray queryReshaped = reshapeTo2D(queryData);
        NdArray keyReshaped = reshapeTo2D(keyData);
        NdArray valueReshaped = reshapeTo2D(valueData);

        // 线性变换：Q, K, V
        Variable Q = queryLayer.layerForward(new Variable(queryReshaped));
        Variable K = keyLayer.layerForward(new Variable(keyReshaped));
        Variable V = valueLayer.layerForward(new Variable(valueReshaped));

        // 重塑回三维
        NdArray qData = reshapeFrom2D(Q.getValue(), batchSize, querySeqLen, dModel);
        NdArray kData = reshapeFrom2D(K.getValue(), batchSize, keySeqLen, dModel);
        NdArray vData = reshapeFrom2D(V.getValue(), batchSize, valueSeqLen, dModel);

        // 重塑为多头形式：(batch_size, seq_len, num_heads, d_k)
        NdArray qHeads = reshapeForHeads(qData, batchSize, querySeqLen, numHeads, dK);
        NdArray kHeads = reshapeForHeads(kData, batchSize, keySeqLen, numHeads, dK);
        NdArray vHeads = reshapeForHeads(vData, batchSize, valueSeqLen, numHeads, dV);

        // 计算注意力
        NdArray attention = computeAttention(qHeads, kHeads, vHeads, batchSize, querySeqLen, keySeqLen);

        // 合并多头结果
        NdArray concatenated = concatenateHeads(attention, batchSize, querySeqLen);

        // 输出投影
        NdArray concatReshaped = reshapeTo2D(concatenated);
        Variable output = outputLayer.layerForward(new Variable(concatReshaped));

        // 重塑回三维
        NdArray result = reshapeFrom2D(output.getValue(), batchSize, querySeqLen, dModel);
        return new Variable(result);
    }

    /**
     * 将三维张量重塑为二维矩阵以用于线性变换
     */
    private NdArray reshapeTo2D(NdArray input) {
        // input shape: (batch_size, seq_len, feature_dim)
        // output shape: (batch_size * seq_len, feature_dim)
        int batchSize = input.getShape().getDimension(0);
        int seqLen = input.getShape().getDimension(1);
        int featureDim = input.getShape().getDimension(2);

        return input.reshape(Shape.of(batchSize * seqLen, featureDim));
    }

    /**
     * 将二维矩阵重塑回三维张量
     */
    private NdArray reshapeFrom2D(NdArray input, int batchSize, int seqLen, int featureDim) {
        // input shape: (batch_size * seq_len, feature_dim)
        // output shape: (batch_size, seq_len, feature_dim)
        return input.reshape(Shape.of(batchSize, seqLen, featureDim));
    }

    /**
     * 重塑张量为多头形式
     */
    private NdArray reshapeForHeads(NdArray input, int batchSize, int seqLen, int numHeads, int headDim) {
        // input shape: (batch_size, seq_len, d_model)
        // output shape: (batch_size, num_heads, seq_len, head_dim)
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
     * 计算缩放点积注意力
     */
    private NdArray computeAttention(NdArray query, NdArray key, NdArray value, int batchSize, int querySeqLen, int keySeqLen) {
        // query shape: (batch_size, num_heads, query_seq_len, head_dim)
        // key, value shape: (batch_size, num_heads, key_seq_len, head_dim)
        NdArray attention = NdArray.of(Shape.of(batchSize, numHeads, querySeqLen, dV));

        double scale = 1.0 / Math.sqrt(dK);

        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < numHeads; h++) {
                // 计算attention scores: Q * K^T
                NdArray scores = NdArray.of(Shape.of(querySeqLen, keySeqLen));
                for (int i = 0; i < querySeqLen; i++) {
                    for (int j = 0; j < keySeqLen; j++) {
                        float score = 0.0f;
                        for (int d = 0; d < dK; d++) {
                            score += query.get(b, h, i, d) * key.get(b, h, j, d);
                        }
                        scores.set((float) (score * scale), i, j);
                    }
                }

                // 应用掩码（如果需要）
                if (useMask) {
                    applyMask(scores, querySeqLen, keySeqLen);
                }

                // Softmax
                NdArray attentionWeights = scores.softMax();

                // 应用权重到values
                for (int i = 0; i < querySeqLen; i++) {
                    for (int d = 0; d < dV; d++) {
                        float output = 0.0f;
                        for (int j = 0; j < keySeqLen; j++) {
                            output += attentionWeights.get(i, j) * value.get(b, h, j, d);
                        }
                        attention.set(output, b, h, i, d);
                    }
                }
            }
        }

        return attention;
    }

    /**
     * 应用因果掩码（用于解码器）
     */
    private void applyMask(NdArray scores, int querySeqLen, int keySeqLen) {
        // 对于因果掩码，只在查询位置i不能看到键位置j > i的情况下应用
        // 这里实现简单的因果掩码：对于query的每个位置i，只能看到key的前i+1个位置
        for (int i = 0; i < querySeqLen; i++) {
            for (int j = i + 1; j < Math.min(keySeqLen, querySeqLen); j++) {
                if (j < keySeqLen) {
                    scores.set(Float.NEGATIVE_INFINITY, i, j);
                }
            }
        }
    }

    /**
     * 合并多头结果
     */
    private NdArray concatenateHeads(NdArray multiHeadOutput, int batchSize, int seqLen) {
        // input shape: (batch_size, num_heads, seq_len, head_dim)
        // output shape: (batch_size, seq_len, d_model)
        NdArray concatenated = NdArray.of(Shape.of(batchSize, seqLen, dModel));

        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                for (int h = 0; h < numHeads; h++) {
                    for (int d = 0; d < dV; d++) {
                        float value = multiHeadOutput.get(b, h, s, d);
                        concatenated.set(value, b, s, h * dV + d);
                    }
                }
            }
        }

        return concatenated;
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        Variable[] variables = new Variable[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            variables[i] = new Variable(inputs[i]);
        }
        return layerForward(variables).getValue();
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // 简化的反向传播实现
        List<NdArray> result = new ArrayList<>();
        result.add(yGrad);
        return result;
    }

    @Override
    public int requireInputNum() {
        return 3; // query, key, value
    }
}