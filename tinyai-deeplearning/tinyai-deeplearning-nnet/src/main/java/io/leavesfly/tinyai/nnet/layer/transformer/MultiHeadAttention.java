package io.leavesfly.tinyai.nnet.layer.transformer;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.func.matrix.Permute;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.core.Module;
import io.leavesfly.tinyai.nnet.layer.dnn.Dropout;
import io.leavesfly.tinyai.nnet.layer.dnn.Linear;

/**
 * V2版本的MultiHeadAttention层（增强版）
 * <p>
 * 多头注意力机制是Transformer架构的核心组件。
 * <p>
 * 核心思想：
 * 1. 将查询(Q)、键(K)、值(V)分别投影到多个子空间
 * 2. 在每个子空间独立计算注意力
 * 3. 将多头结果拼接后再进行一次线性变换
 * <p>
 * 公式：
 * MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
 * head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
 * Attention(Q, K, V) = softmax(QK^T / sqrt(d_k) + mask) V
 * <p>
 * 其中：
 * - h: 注意力头数
 * - d_k: 每个头的维度 (d_model / h)
 * - W^Q, W^K, W^V: Query、Key、Value的投影矩阵
 * - W^O: 输出投影矩阵
 * - mask: 可选的注意力掩码（如因果掩码、padding掩码）
 * <p>
 * V2增强特性：
 * - 支持attnMask（注意力掩码，如因果掩码）
 * - 支持keyPaddingMask（键填充掩码）
 * - 提供生成因果掩码的静态方法
 *
 * @author leavesfly
 * @version 2.1
 */
public class MultiHeadAttention extends Module {

    private final int dModel;      // 模型维度
    private final int numHeads;    // 注意力头数
    private final int dK;          // 每个头的键/查询维度
    private final int dV;          // 每个头的值维度
    private final float dropout;   // dropout比率

    // 线性投影层
    private Linear queryProjection;  // Q投影
    private Linear keyProjection;    // K投影
    private Linear valueProjection;  // V投影
    private Linear outputProjection; // 输出投影
    private final Dropout attnDropout; // 注意力权重dropout

    /**
     * 构造函数
     *
     * @param name     层名称
     * @param dModel   模型维度
     * @param numHeads 注意力头数
     * @param dropout  dropout比率
     */
    public MultiHeadAttention(String name, int dModel, int numHeads, float dropout) {
        super(name);

        if (dModel % numHeads != 0) {
            throw new IllegalArgumentException(
                    String.format("d_model (%d) must be divisible by num_heads (%d)", dModel, numHeads));
        }

        this.dModel = dModel;
        this.numHeads = numHeads;
        this.dK = dModel / numHeads;
        this.dV = dModel / numHeads;
        this.dropout = dropout;

        // 创建投影层
        queryProjection = new Linear("q_proj", dModel, dModel, true);
        keyProjection = new Linear("k_proj", dModel, dModel, true);
        valueProjection = new Linear("v_proj", dModel, dModel, true);
        outputProjection = new Linear("o_proj", dModel, dModel, true);
        attnDropout = new Dropout("attn_dropout", dropout);

        // 注册子模块
        registerModule("q_proj", queryProjection);
        registerModule("k_proj", keyProjection);
        registerModule("v_proj", valueProjection);
        registerModule("o_proj", outputProjection);
        registerModule("attn_dropout", attnDropout);

        init();
    }

    /**
     * 构造函数（默认dropout=0.1）
     *
     * @param name     层名称
     * @param dModel   模型维度
     * @param numHeads 注意力头数
     */
    public MultiHeadAttention(String name, int dModel, int numHeads) {
        this(name, dModel, numHeads, 0.1f);
    }

    @Override
    public void resetParameters() {
        // 子模块的参数由其自己初始化
    }

    @Override
    public Variable forward(Variable... inputs) {
        // 输入：query, key, value, [attnMask], [keyPaddingMask]
        Variable query = inputs[0];
        Variable key = inputs.length > 1 ? inputs[1] : query;
        Variable value = inputs.length > 2 ? inputs[2] : key;
        Variable attnMask = inputs.length > 3 ? inputs[3] : null;
        Variable keyPaddingMask = inputs.length > 4 ? inputs[4] : null;

        int[] queryShape = query.getValue().getShape().getShapeDims();
        int batchSize = queryShape[0];
        int seqLen = queryShape[1];
        int keySeqLen = key.getValue().getShape().getShapeDims()[1];

        // 1. 线性投影 Q, K, V
        Variable Q = queryProjection.forward(query);   // (batch, seq_len, d_model)
        Variable K = keyProjection.forward(key);       // (batch, key_seq_len, d_model)
        Variable V = valueProjection.forward(value);   // (batch, key_seq_len, d_model)

        // 2. 分割成多头
        // 需要将 (batch, seq_len, d_model) 重塑为 (batch, seq_len, num_heads, d_k)
        // 然后转置为 (batch, num_heads, seq_len, d_k)
        Q = splitHeads(Q, batchSize, seqLen);
        K = splitHeads(K, batchSize, keySeqLen);
        V = splitHeads(V, batchSize, keySeqLen);

        // 3. 计算缩放点积注意力（带掩码）
        Variable attention = scaledDotProductAttention(Q, K, V, attnMask, keyPaddingMask);

        // 4. 合并多头
        Variable concat = mergeHeads(attention, batchSize, seqLen);

        // 5. 输出投影
        Variable output = outputProjection.forward(concat);

        return output;
    }

    /**
     * 简化的前向传播（仅自注意力）
     *
     * @param x        输入张量 (batch, seq_len, d_model)
     * @param attnMask 可选的注意力掩码
     * @return 自注意力输出
     */
    public Variable forward(Variable x, Variable attnMask) {
        if (attnMask != null) {
            return forward(x, x, x, attnMask);
        } else {
            return forward(x, x, x);
        }
    }

    /**
     * 分割成多头
     * <p>
     * 输入: (batch, seq_len, d_model)
     * 输出: (batch, num_heads, seq_len, d_k)
     *
     * @param x         输入变量
     * @param batchSize 批次大小
     * @param seqLen    序列长度
     * @return 分割后的张量
     */
    private Variable splitHeads(Variable x, int batchSize, int seqLen) {
        // (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
        Variable reshaped = x.reshape(Shape.of(batchSize, seqLen, numHeads, dK));
        return new Permute(0, 2, 1, 3).call(reshaped);
    }

    /**
     * 合并多头
     * <p>
     * 输入: (batch, num_heads, seq_len, d_v)
     * 输出: (batch, seq_len, d_model)
     *
     * @param x         输入变量
     * @param batchSize 批次大小
     * @param seqLen    序列长度
     * @return 合并后的张量
     */
    private Variable mergeHeads(Variable x, int batchSize, int seqLen) {
        // (batch, num_heads, seq_len, d_v) -> (batch, seq_len, d_model)
        Variable permuted = new Permute(0, 2, 1, 3).call(x);
        return permuted.reshape(Shape.of(batchSize, seqLen, dModel));
    }

    /**
     * 缩放点积注意力（带掩码）
     * <p>
     * Attention(Q, K, V) = softmax(QK^T / sqrt(d_k) + mask) V
     *
     * @param Q              查询张量 (batch, heads, seq_len, d_k)
     * @param K              键张量 (batch, heads, key_seq_len, d_k)
     * @param V              值张量 (batch, heads, key_seq_len, d_v)
     * @param attnMask       注意力掩码（可选）- 添加到注意力分数上
     * @param keyPaddingMask 键填充掩码（可选）- 标记padding位置
     * @return 注意力输出
     */
    private Variable scaledDotProductAttention(Variable Q, Variable K, Variable V,
                                                Variable attnMask, Variable keyPaddingMask) {
        // 1. 计算 Q * K^T
        Variable kTransposed = new Permute(0, 1, 3, 2).call(K);
        Variable scores = Q.matMul(kTransposed);  // (batch, heads, seq_len, key_seq_len)

        // 2. 缩放
        double scale = Math.sqrt(dK);
        Variable scaledScores = scores.div(new Variable((float) scale));

        // 3. 应用注意力掩码（如因果掩码）
        // attnMask: 通常是 (seq_len, key_seq_len) 或 (batch, heads, seq_len, key_seq_len)
        // 掩码位置应该是 -inf（或很大的负数），softmax后会变成0
        if (attnMask != null) {
            scaledScores = scaledScores.add(attnMask);
        }

        // 4. 应用键填充掩码
        // keyPaddingMask: (batch, key_seq_len)，标记哪些位置是padding
        // 需要扩展为 (batch, 1, 1, key_seq_len) 以便广播
        if (keyPaddingMask != null) {
            // 将padding位置设为很大的负数
            Variable paddingMask = createPaddingMaskFromBoolean(keyPaddingMask);
            scaledScores = scaledScores.add(paddingMask);
        }

        // 5. Softmax（在最后一维）— 使用数值稳定的实现
        // 先减去最大值防止 exp() 溢出：softmax(x) = softmax(x - max(x))
        // 注意：Max 算子不支持负数轴，需要转换为正数轴
        int lastAxis = scaledScores.getShape().getDimNum() - 1;
        Variable maxScores = scaledScores.max(lastAxis, true);
        Variable stableScores = scaledScores.sub(maxScores);
        Variable attentionWeights = stableScores.softMax();

        // 6. 应用dropout（训练模式）
        if (isTraining() && dropout > 0) {
            attentionWeights = attnDropout.forward(attentionWeights);
        }

        // 7. 计算注意力输出: attention_weights * V
        Variable output = attentionWeights.matMul(V);

        return output;
    }

    /**
     * 将布尔掩码转换为加法掩码
     * <p>
     * 输入掩码中True表示需要被屏蔽的位置
     * <p>
     * 注：这里需要直接操作NdArray来生成掩码，因为这是条件转换逻辑
     *
     * @param boolMask 布尔掩码
     * @return 加法掩码（被屏蔽位置为-1e9，其他位置为0）
     */
    private Variable createPaddingMaskFromBoolean(Variable boolMask) {
        // 这是一个工具方法，需要处理条件逻辑，直接操作NdArray是合理的
        NdArray maskData = boolMask.getValue();
        float[] data = maskData.getArray();
        float[] result = new float[data.length];

        for (int i = 0; i < data.length; i++) {
            // 如果是1（True，需要屏蔽），则设为-1e9
            result[i] = data[i] > 0.5f ? -1e9f : 0f;
        }

        Variable resultVar = new Variable(NdArray.of(result, maskData.getShape()));
        resultVar.setRequireGrad(false);  // 掩码不需要梯度
        return resultVar;
    }

    public int getDModel() {
        return dModel;
    }

    public int getNumHeads() {
        return numHeads;
    }

    public int getDK() {
        return dK;
    }

    public int getDV() {
        return dV;
    }

    public float getDropout() {
        return dropout;
    }

    @Override
    public String toString() {
        return "MultiHeadAttention{" +
                "name='" + name + '\'' +
                ", dModel=" + dModel +
                ", numHeads=" + numHeads +
                ", dK=" + dK +
                ", dV=" + dV +
                ", dropout=" + dropout +
                '}';
    }

    /* ===== 静态工具方法 ===== */

    /**
     * 生成因果掩码（Causal Mask / Look-ahead Mask）
     * <p>
     * 用于自回归解码，确保每个位置只能看到之前的位置。
     * 掩码的上三角部分为-1e9（屏蔽未来位置），对角线及以下为0。
     * <p>
     * 示例（seqLen=4）：
     * <pre>
     *   [[ 0,   -inf, -inf, -inf],
     *    [ 0,    0,   -inf, -inf],
     *    [ 0,    0,    0,   -inf],
     *    [ 0,    0,    0,    0  ]]
     * </pre>
     *
     * @param seqLen 序列长度
     * @return 因果掩码 (seqLen, seqLen)
     */
    public static Variable generateCausalMask(int seqLen) {
        float[][] mask = new float[seqLen][seqLen];

        for (int i = 0; i < seqLen; i++) {
            for (int j = 0; j < seqLen; j++) {
                // j > i 表示是未来位置，需要屏蔽
                mask[i][j] = (j > i) ? -1e9f : 0f;
            }
        }

        return new Variable(NdArray.of(mask));
    }

    /**
     * 生成因果掩码（批量版本）
     * <p>
     * 返回形状为 (1, 1, seqLen, seqLen) 的掩码，可广播到 (batch, heads, seqLen, seqLen)
     *
     * @param seqLen 序列长度
     * @return 可广播的因果掩码 (1, 1, seqLen, seqLen)
     */
    public static Variable generateCausalMaskBatched(int seqLen) {
        float[] mask = new float[seqLen * seqLen];

        for (int i = 0; i < seqLen; i++) {
            for (int j = 0; j < seqLen; j++) {
                mask[i * seqLen + j] = (j > i) ? -1e9f : 0f;
            }
        }

        return new Variable(NdArray.of(mask, Shape.of(1, 1, seqLen, seqLen)));
    }

    /**
     * 生成填充掩码
     * <p>
     * 根据实际序列长度生成填充掩码。
     *
     * @param batchSize       批次大小
     * @param maxLen          最大序列长度
     * @param actualLengths   每个样本的实际长度数组
     * @return 填充掩码 (batch, maxLen)，padding位置为1，有效位置为0
     */
    public static Variable generatePaddingMask(int batchSize, int maxLen, int[] actualLengths) {
        if (actualLengths.length != batchSize) {
            throw new IllegalArgumentException("actualLengths length must equal batchSize");
        }

        float[][] mask = new float[batchSize][maxLen];

        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < maxLen; j++) {
                // j >= actualLengths[i] 表示是padding位置
                mask[i][j] = (j >= actualLengths[i]) ? 1f : 0f;
            }
        }

        return new Variable(NdArray.of(mask));
    }

    /**
     * 组合因果掩码和填充掩码
     * <p>
     * 用于解码器的自注意力，同时屏蔽未来位置和padding位置。
     *
     * @param seqLen        序列长度
     * @param paddingMask   填充掩码 (batch, seqLen)
     * @return 组合后的掩码 (batch, 1, seqLen, seqLen)
     */
    public static Variable combineCausalAndPaddingMask(int seqLen, Variable paddingMask) {
        // 首先生成因果掩码
        Variable causalMask = generateCausalMaskBatched(seqLen);

        if (paddingMask == null) {
            return causalMask;
        }

        // 将paddingMask转换为适当的形状
        // paddingMask: (batch, seqLen) -> 需要扩展并转换为加法掩码
        NdArray padData = paddingMask.getValue();
        int batchSize = padData.getShape().getShapeDims()[0];

        float[] combinedData = new float[batchSize * seqLen * seqLen];
        float[] causalData = causalMask.getValue().getArray();
        float[] padArray = padData.getArray();

        for (int b = 0; b < batchSize; b++) {
            for (int i = 0; i < seqLen; i++) {
                for (int j = 0; j < seqLen; j++) {
                    int idx = b * seqLen * seqLen + i * seqLen + j;
                    // 因果掩码 + padding掩码（如果j位置是padding，则屏蔽）
                    float causalVal = causalData[i * seqLen + j];
                    float padVal = padArray[b * seqLen + j] > 0.5f ? -1e9f : 0f;
                    combinedData[idx] = Math.min(causalVal + padVal, 0f);  // 保证不超过0
                    if (causalVal < -1e8f || padVal < -1e8f) {
                        combinedData[idx] = -1e9f;
                    }
                }
            }
        }

        return new Variable(NdArray.of(combinedData, Shape.of(batchSize, 1, seqLen, seqLen)));
    }
}
