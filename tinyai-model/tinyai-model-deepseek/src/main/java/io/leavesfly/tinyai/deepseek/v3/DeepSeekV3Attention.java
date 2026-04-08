package io.leavesfly.tinyai.deepseek.v3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.func.matrix.Permute;
import io.leavesfly.tinyai.func.matrix.RotaryEmbedding;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.core.Module;
import io.leavesfly.tinyai.nnet.layer.dnn.Dropout;
import io.leavesfly.tinyai.nnet.layer.dnn.Linear;

/**
 * DeepSeek-V3 自注意力层（内置 RoPE 旋转位置编码）
 *
 * 对标 DeepSeek-V3 论文，在 Q/K 线性投影后、注意力计算前应用 RoPE，
 * 替代传统的学习式绝对位置嵌入。
 *
 * 计算流程：
 * 1. Q = W_q(x), K = W_k(x), V = W_v(x)
 * 2. 分割多头: [B, L, D] -> [B, H, L, D_k]
 * 3. 对 Q, K 应用 RoPE 旋转位置编码
 * 4. Attention = softmax(Q·K^T / √d_k + causal_mask) · V
 * 5. 合并多头 + 输出投影
 *
 * @author leavesfly
 * @version 2.0
 */
public class DeepSeekV3Attention extends Module {

    private final int dModel;
    private final int numHeads;
    private final int headDim;

    // 线性投影层
    private final Linear queryProjection;
    private final Linear keyProjection;
    private final Linear valueProjection;
    private final Linear outputProjection;
    private final Dropout attnDropout;

    // RoPE 旋转位置编码（作用于每个 head 的维度）
    private final RotaryEmbedding rotaryEmbedding;

    // Permute 对象缓存（避免每次 forward 都创建新对象）
    private static final Permute SPLIT_HEADS_PERMUTE = new Permute(0, 2, 1, 3);
    private static final Permute MERGE_HEADS_PERMUTE = new Permute(0, 2, 1, 3);
    private static final Permute KEY_TRANSPOSE_PERMUTE = new Permute(0, 1, 3, 2);

    /**
     * 构造函数
     *
     * @param name       模块名称
     * @param dModel     模型维度
     * @param numHeads   注意力头数
     * @param maxSeqLen  最大序列长度（用于 RoPE 预计算）
     * @param dropout    dropout 比率
     * @param ropeTheta  RoPE 频率基数（DeepSeek-V3 默认 10000）
     */
    public DeepSeekV3Attention(String name, int dModel, int numHeads,
                               int maxSeqLen, float dropout, float ropeTheta) {
        super(name);

        if (dModel % numHeads != 0) {
            throw new IllegalArgumentException(
                    String.format("d_model (%d) must be divisible by num_heads (%d)", dModel, numHeads));
        }

        this.dModel = dModel;
        this.numHeads = numHeads;
        this.headDim = dModel / numHeads;

        // Q/K/V/O 线性投影
        queryProjection = new Linear("q_proj", dModel, dModel, true);
        keyProjection = new Linear("k_proj", dModel, dModel, true);
        valueProjection = new Linear("v_proj", dModel, dModel, true);
        outputProjection = new Linear("o_proj", dModel, dModel, true);
        attnDropout = new Dropout("attn_dropout", dropout);

        registerModule("q_proj", queryProjection);
        registerModule("k_proj", keyProjection);
        registerModule("v_proj", valueProjection);
        registerModule("o_proj", outputProjection);
        registerModule("attn_dropout", attnDropout);

        // 初始化 RoPE（作用于 headDim 维度，支持 4D 输入 [B, H, L, D_k]）
        rotaryEmbedding = new RotaryEmbedding(headDim, maxSeqLen, ropeTheta);
    }

    /**
     * 前向传播
     *
     * @param inputs inputs[0]: 输入张量 [batch_size, seq_len, d_model]
     *               inputs[1]: 因果掩码（可选）[1, 1, seq_len, seq_len]
     * @return 注意力输出 [batch_size, seq_len, d_model]
     */
    @Override
    public Variable forward(Variable... inputs) {
        Variable x = inputs[0];
        Variable causalMask = inputs.length > 1 ? inputs[1] : null;

        int[] shape = x.getValue().getShape().getShapeDims();
        int batchSize = shape[0];
        int seqLen = shape[1];

        // 1. 线性投影 Q, K, V
        Variable queryOutput = queryProjection.forward(x);
        Variable keyOutput = keyProjection.forward(x);
        Variable valueOutput = valueProjection.forward(x);

        // 2. 分割多头: [B, L, D] -> [B, H, L, D_k]
        queryOutput = splitHeads(queryOutput, batchSize, seqLen);
        keyOutput = splitHeads(keyOutput, batchSize, seqLen);
        valueOutput = splitHeads(valueOutput, batchSize, seqLen);

        // 3. 对 Q, K 应用 RoPE（4D 输入 [B, H, L, D_k]）
        queryOutput = rotaryEmbedding.call(queryOutput);
        keyOutput = rotaryEmbedding.call(keyOutput);

        // 4. 缩放点积注意力
        Variable attention = scaledDotProductAttention(queryOutput, keyOutput, valueOutput, causalMask);

        // 5. 合并多头: [B, H, L, D_k] -> [B, L, D]
        Variable merged = mergeHeads(attention, batchSize, seqLen);

        // 6. 输出投影
        return outputProjection.forward(merged);
    }

    /**
     * 分割多头: [B, L, D] -> [B, H, L, D_k]
     */
    private Variable splitHeads(Variable x, int batchSize, int seqLen) {
        Variable reshaped = x.reshape(Shape.of(batchSize, seqLen, numHeads, headDim));
        return SPLIT_HEADS_PERMUTE.call(reshaped);
    }

    /**
     * 合并多头: [B, H, L, D_k] -> [B, L, D]
     */
    private Variable mergeHeads(Variable x, int batchSize, int seqLen) {
        Variable permuted = MERGE_HEADS_PERMUTE.call(x);
        return permuted.reshape(Shape.of(batchSize, seqLen, dModel));
    }

    /**
     * 缩放点积注意力
     *
     * Attention(Q, K, V) = softmax(Q·K^T / √d_k + mask) · V
     *
     * @param queryTensor  [B, H, L, D_k]（已应用 RoPE）
     * @param keyTensor    [B, H, L, D_k]（已应用 RoPE）
     * @param valueTensor  [B, H, L, D_k]
     * @param causalMask   [1, 1, L, L]（可选）
     * @return 注意力输出 [B, H, L, D_k]
     */
    private Variable scaledDotProductAttention(Variable queryTensor, Variable keyTensor,
                                               Variable valueTensor, Variable causalMask) {
        // Q · K^T
        Variable keyTransposed = KEY_TRANSPOSE_PERMUTE.call(keyTensor);
        Variable scores = queryTensor.matMul(keyTransposed);

        // 缩放
        float scale = (float) Math.sqrt(headDim);
        Variable scaledScores = scores.div(new Variable(scale));

        // 应用因果掩码
        if (causalMask != null) {
            scaledScores = scaledScores.add(causalMask);
        }

        // Softmax
        Variable attentionWeights = scaledScores.softMax();

        // Dropout
        if (isTraining()) {
            attentionWeights = attnDropout.forward(attentionWeights);
        }

        // 加权求和
        return attentionWeights.matMul(valueTensor);
    }

    /**
     * 生成因果掩码（批量版本）
     *
     * @param seqLen 序列长度
     * @return 因果掩码 [1, 1, seqLen, seqLen]
     */
    public static Variable generateCausalMask(int seqLen) {
        float[] mask = new float[seqLen * seqLen];
        for (int i = 0; i < seqLen; i++) {
            for (int j = 0; j < seqLen; j++) {
                mask[i * seqLen + j] = (j > i) ? -1e9f : 0f;
            }
        }
        Variable maskVar = new Variable(NdArray.of(mask, Shape.of(1, 1, seqLen, seqLen)));
        maskVar.setRequireGrad(false);
        return maskVar;
    }
}