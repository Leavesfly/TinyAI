package io.leavesfly.tinyai.gpt3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.func.matrix.Permute;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Dropout;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;

/**
 * GPT-3 增强型多头注意力机制
 *
 * 在标准多头注意力基础上，整合三项核心技术：
 *
 * 1. RoPE（旋转位置编码）
 *    - 对 Q/K 的前 rotaryDim 个维度应用旋转位置编码
 *    - rotaryDim = headDim × rotaryPct（如 0.25 表示 25% 的维度）
 *    - 旋转公式：[x0, x1] -> [x0·cos - x1·sin, x0·sin + x1·cos]
 *    - 频率：freq[i] = 1 / (rotaryBase ^ (2i / rotaryDim))
 *
 * 2. 稀疏注意力（Sparse Attention）
 *    - 局部窗口：每个 Token 只关注距离在 sparseLocalWindow 以内的历史 Token
 *    - 步长全局：额外关注位置编号是 sparseStrideSize 倍数的 Token（充当"锚点"）
 *    - 复杂度从 O(n²) 降至 O(n × (w + n/s))
 *
 * 3. KV Cache（键值缓存）
 *    - 通过 forwardWithCache() 在推理时复用历史 K/V
 *    - 每步仅计算新 Token，大幅减少重复计算
 *
 * 计算流程：
 * <pre>
 * x -> Q/K/V proj -> split heads -> [RoPE on Q&K] ->
 * [KV Cache update] -> scores = Q @ K^T / sqrt(d) ->
 * [sparse/causal mask] -> softmax -> dropout -> attn @ V ->
 * merge heads -> O proj -> output
 * </pre>
 *
 * @author leavesfly
 * @version 1.0
 */
public class GPT3Attention extends Module {

    /** 注意力掩码中用于屏蔽位置的极大负值，softmax 后趋近于 0 */
    private static final float MASK_NEG_INF = -1e9f;

    private final GPT3Config config;
    private final int dModel;
    private final int numHeads;
    private final int headDim;

    /**
     * 实际应用 RoPE 的维度数（headDim 的 rotaryPct 比例，必须为偶数）
     * 例如：headDim=64, rotaryPct=0.25 => rotaryDim=16
     */
    private final int rotaryDim;
    private final boolean useRoPE;

    // Q/K/V/O 线性投影层
    private final Linear qProj;
    private final Linear kProj;
    private final Linear vProj;
    private final Linear oProj;

    /** 注意力权重 Dropout（应用于 softmax 后的权重） */
    private final Dropout attnDropout;

    /**
     * RoPE 预计算的 cos/sin 表
     * 形状逻辑：[maxSeqLen, rotaryDim/2]（展平存储）
     * 访问方式：cosTable[pos * halfRotary + i]
     */
    private final float[] cosTable;
    private final float[] sinTable;

    /**
     * 构造 GPT-3 增强注意力层
     *
     * @param name   层名称
     * @param config GPT-3 配置（包含 RoPE、稀疏注意力等开关）
     */
    public GPT3Attention(String name, GPT3Config config) {
        super(name);
        this.config = config;
        this.dModel = config.getNEmbd();
        this.numHeads = config.getNHead();
        this.headDim = dModel / numHeads;
        this.useRoPE = config.isUseRotaryEmbedding();

        // 计算 RoPE 作用的维度数（必须为偶数）
        int rawRotaryDim = (int) (headDim * config.getRotaryPct());
        this.rotaryDim = (rawRotaryDim > 0 && rawRotaryDim % 2 != 0)
                ? rawRotaryDim - 1
                : rawRotaryDim;

        // 创建 Q/K/V/O 投影层
        this.qProj = new Linear("q_proj", dModel, dModel, true);
        this.kProj = new Linear("k_proj", dModel, dModel, true);
        this.vProj = new Linear("v_proj", dModel, dModel, true);
        this.oProj = new Linear("o_proj", dModel, dModel, true);
        this.attnDropout = new Dropout("attn_dropout", (float) config.getAttnPdrop());

        registerModule("q_proj", qProj);
        registerModule("k_proj", kProj);
        registerModule("v_proj", vProj);
        registerModule("o_proj", oProj);
        registerModule("attn_dropout", attnDropout);

        // 预计算 RoPE 的 cos/sin 频率表
        if (useRoPE && rotaryDim > 0) {
            int halfDim = rotaryDim / 2;
            int maxSeqLen = config.getNPositions();
            float base = (float) config.getRotaryBase();
            cosTable = new float[maxSeqLen * halfDim];
            sinTable = new float[maxSeqLen * halfDim];

            for (int pos = 0; pos < maxSeqLen; pos++) {
                for (int i = 0; i < halfDim; i++) {
                    // freq[i] = 1 / (base ^ (2i / rotaryDim))
                    float freq = (float) (1.0 / Math.pow(base, (2.0 * i) / rotaryDim));
                    float angle = pos * freq;
                    cosTable[pos * halfDim + i] = (float) Math.cos(angle);
                    sinTable[pos * halfDim + i] = (float) Math.sin(angle);
                }
            }
        } else {
            cosTable = null;
            sinTable = null;
        }

        init();
    }

    /**
     * 标准前向传播（不使用 KV Cache）
     *
     * 兼容旧接口：支持 forward(x) 或 forward(x, x, x, mask, null) 两种调用形式。
     * 当传入多个参数时（旧 MultiHeadAttention 接口），只使用 inputs[0] 作为输入，
     * 忽略外部传入的 mask（GPT3Attention 内部自动生成正确的掩码）。
     *
     * @param inputs inputs[0] 为输入张量 (batch, seqLen, dModel)
     * @return 注意力输出 (batch, seqLen, dModel)
     */
    @Override
    public Variable forward(Variable... inputs) {
        Variable x = inputs[0];
        return forwardWithCache(x, null, 0);
    }

    /**
     * 带 KV Cache 的前向传播（用于自回归推理加速）
     *
     * @param x        输入张量 (batch, newSeqLen, dModel)
     * @param cache    KV缓存对象（null 表示不使用缓存）
     * @param startPos 当前输入在完整序列中的起始位置（用于 RoPE 和因果掩码）
     * @return 注意力输出 (batch, newSeqLen, dModel)
     */
    public Variable forwardWithCache(Variable x, GPT3KVCache cache, int startPos) {
        int[] shape = x.getValue().getShape().getShapeDims();
        int batchSize = shape[0];
        int seqLen = shape[1];

        // 1. Q/K/V 投影: (batch, seq, dModel)
        Variable Q = qProj.forward(x);
        Variable K = kProj.forward(x);
        Variable V = vProj.forward(x);

        // 2. 多头分割: (batch, seq, dModel) -> (batch, numHeads, seq, headDim)
        Q = splitHeads(Q, batchSize, seqLen);
        K = splitHeads(K, batchSize, seqLen);
        V = splitHeads(V, batchSize, seqLen);

        // 3. 对 Q 和 K 应用 RoPE（仅作用于前 rotaryDim 个维度）
        if (useRoPE && rotaryDim > 0) {
            Q = applyPartialRoPE(Q, batchSize, seqLen, startPos);
            K = applyPartialRoPE(K, batchSize, seqLen, startPos);
        }

        // 4. KV Cache 更新（推理加速）
        if (cache != null) {
            NdArray[] updated = cache.update(K.getValue(), V.getValue());
            K = new Variable(updated[0]);
            V = new Variable(updated[1]);
            K.setRequireGrad(false);
            V.setRequireGrad(false);
        }

        int kvSeqLen = K.getValue().getShape().getShapeDims()[2];

        // 5. 计算缩放点积注意力分数：scores = Q @ K^T / sqrt(headDim)
        //    Q: (batch, numHeads, seqLen, headDim)
        //    K^T: (batch, numHeads, headDim, kvSeqLen)
        //    -> scores: (batch, numHeads, seqLen, kvSeqLen)
        Variable KT = new Permute(0, 1, 3, 2).call(K);
        Variable scores = Q.matMul(KT);
        float scale = (float) (1.0 / Math.sqrt(headDim));
        Variable scaleVar = new Variable(scale);
        scaleVar.setRequireGrad(false);
        scores = scores.mul(scaleVar);

        // 6. 应用注意力掩码
        if (config.isSparseAttention()) {
            // 稀疏注意力掩码：局部窗口 + 步长全局
            scores = applySparseAttentionMask(scores, batchSize, seqLen, kvSeqLen, startPos);
        } else {
            // 标准因果掩码（下三角）
            scores = applyCausalMask(scores, batchSize, seqLen, kvSeqLen, startPos);
        }

        // 7. Softmax 归一化（在 kvSeqLen 维度上）
        scores = softmaxOnLastDim(scores, batchSize, seqLen, kvSeqLen);

        // 8. 注意力权重 Dropout（仅训练时）
        if (isTraining() && config.getAttnPdrop() > 0) {
            scores = attnDropout.forward(scores);
        }

        // 9. 加权求和：attended = scores @ V
        //    scores: (batch, numHeads, seqLen, kvSeqLen)
        //    V: (batch, numHeads, kvSeqLen, headDim)
        //    -> (batch, numHeads, seqLen, headDim)
        Variable attended = scores.matMul(V);

        // 10. 合并多头: (batch, numHeads, seqLen, headDim) -> (batch, seqLen, dModel)
        Variable merged = mergeHeads(attended, batchSize, seqLen);

        // 11. 输出投影
        return oProj.forward(merged);
    }

    // ========================== 辅助方法 ==========================

    /**
     * 多头分割
     * (batch, seq, dModel) -> (batch, numHeads, seq, headDim)
     */
    private Variable splitHeads(Variable x, int batchSize, int seqLen) {
        Variable reshaped = x.reshape(Shape.of(batchSize, seqLen, numHeads, headDim));
        return new Permute(0, 2, 1, 3).call(reshaped);
    }

    /**
     * 多头合并
     * (batch, numHeads, seq, headDim) -> (batch, seq, dModel)
     */
    private Variable mergeHeads(Variable x, int batchSize, int seqLen) {
        Variable permuted = new Permute(0, 2, 1, 3).call(x);
        return permuted.reshape(Shape.of(batchSize, seqLen, dModel));
    }

    /**
     * 对 Q 或 K 应用部分 RoPE（仅旋转前 rotaryDim 个维度）
     *
     * 旋转公式（对每对相邻维度 [x0, x1]）：
     *   x0' = x0 * cos(θ) - x1 * sin(θ)
     *   x1' = x0 * sin(θ) + x1 * cos(θ)
     * 其中 θ = position * freq[i]
     *
     * 说明：RoPE 旋转变换保持向量模长不变（是等距变换），
     * 因此梯度近似（直接操作 NdArray）在训练中误差可忽略。
     *
     * @param qk       Q 或 K 张量，Shape: (batch, numHeads, seq, headDim)
     * @param batchSize 批次大小
     * @param seqLen   序列长度
     * @param startPos 在完整序列中的起始位置（KV Cache 增量推理时使用）
     */
    private Variable applyPartialRoPE(Variable qk, int batchSize, int seqLen, int startPos) {
        float[] data = qk.getValue().getArray();
        // 避免全量 clone：直接分配新数组，仅写入需要修改的部分，
        // 未旋转的维度 [rotaryDim, headDim) 通过 System.arraycopy 批量复制
        int totalElements = data.length;
        float[] output = new float[totalElements];

        int halfRotary = rotaryDim / 2;
        int unrotatedDims = headDim - rotaryDim;

        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < numHeads; h++) {
                for (int s = 0; s < seqLen; s++) {
                    int pos = startPos + s;
                    int baseOffset = ((b * numHeads + h) * seqLen + s) * headDim;

                    // 对前 rotaryDim 个维度做旋转
                    for (int i = 0; i < halfRotary; i++) {
                        float x0 = data[baseOffset + 2 * i];
                        float x1 = data[baseOffset + 2 * i + 1];
                        float cos = cosTable[pos * halfRotary + i];
                        float sin = sinTable[pos * halfRotary + i];

                        output[baseOffset + 2 * i]     = x0 * cos - x1 * sin;
                        output[baseOffset + 2 * i + 1] = x0 * sin + x1 * cos;
                    }

                    // 批量复制未旋转的维度 [rotaryDim, headDim)
                    if (unrotatedDims > 0) {
                        System.arraycopy(data, baseOffset + rotaryDim,
                                output, baseOffset + rotaryDim, unrotatedDims);
                    }
                }
            }
        }

        Variable result = new Variable(NdArray.of(output, qk.getValue().getShape()));
        result.setRequireGrad(qk.isRequireGrad());
        return result;
    }

    /**
     * 标准因果掩码（自回归下三角掩码）
     *
     * 位置 j 只有满足 j <= qPos 才可被关注（qPos = startPos + i）
     * 屏蔽位置填充 -1e9，softmax 后趋近于 0
     */
    private Variable applyCausalMask(Variable scores, int batchSize, int seqLen,
                                     int kvSeqLen, int startPos) {
        float[] maskData = new float[batchSize * numHeads * seqLen * kvSeqLen];
        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < numHeads; h++) {
                for (int i = 0; i < seqLen; i++) {
                    int qPos = startPos + i;
                    for (int j = 0; j < kvSeqLen; j++) {
                        if (j > qPos) {
                            int idx = ((b * numHeads + h) * seqLen + i) * kvSeqLen + j;
                            maskData[idx] = MASK_NEG_INF;
                        }
                    }
                }
            }
        }
        Variable mask = new Variable(NdArray.of(maskData,
                Shape.of(batchSize, numHeads, seqLen, kvSeqLen)));
        mask.setRequireGrad(false);
        return scores.add(mask);
    }

    /**
     * 稀疏注意力掩码（局部窗口 + 步长全局）
     *
     * 每个 Query 位置 qPos 可以关注 Key 位置 j，当且仅当：
     *   条件1（局部窗口）：qPos - j < sparseLocalWindow
     *   条件2（步长全局）：j % sparseStrideSize == 0
     *   基础条件（因果）：j <= qPos
     *
     * 不满足任何条件的位置填充 -1e9（被屏蔽）。
     *
     * 示例（localWindow=3, strideSize=4, seqLen=8）：
     *   位置 6 可以关注：{0,4}（步长全局） ∪ {4,5,6}（局部）= {0,4,5,6}
     */
    private Variable applySparseAttentionMask(Variable scores, int batchSize, int seqLen,
                                              int kvSeqLen, int startPos) {
        int localWindow = config.getSparseLocalWindow();
        int strideSize = config.getSparseStrideSize();

        // 初始化为全部屏蔽（-1e9）
        float[] maskData = new float[batchSize * numHeads * seqLen * kvSeqLen];
        for (int idx = 0; idx < maskData.length; idx++) {
            maskData[idx] = MASK_NEG_INF;
        }

        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < numHeads; h++) {
                for (int i = 0; i < seqLen; i++) {
                    int qPos = startPos + i;
                    for (int j = 0; j < kvSeqLen; j++) {
                        // 因果约束：不能看未来
                        if (j > qPos) continue;

                        // 局部窗口：qPos 距 j 在 localWindow 以内
                        boolean inLocalWindow = (localWindow <= 0) || (qPos - j < localWindow);
                        // 步长全局：j 是 strideSize 的倍数（充当全局"锚点"）
                        boolean isStrideGlobal = (strideSize > 0) && (j % strideSize == 0);

                        if (inLocalWindow || isStrideGlobal) {
                            // 允许关注此位置（mask = 0，即不屏蔽）
                            int idx = ((b * numHeads + h) * seqLen + i) * kvSeqLen + j;
                            maskData[idx] = 0f;
                        }
                    }
                }
            }
        }

        Variable mask = new Variable(NdArray.of(maskData,
                Shape.of(batchSize, numHeads, seqLen, kvSeqLen)));
        mask.setRequireGrad(false);
        return scores.add(mask);
    }

    /**
     * 在最后一个维度（kvSeqLen）上应用 Softmax
     * 输入: (batch, numHeads, seqLen, kvSeqLen)
     * 输出: same shape，每行之和为 1
     */
    private Variable softmaxOnLastDim(Variable input, int batchSize, int seqLen, int kvSeqLen) {
        // 展平前三维度，在 kvSeqLen 上做 softmax
        Variable reshaped = input.reshape(Shape.of(batchSize * numHeads * seqLen, kvSeqLen));
        Variable softmaxed = reshaped.softMax();
        return softmaxed.reshape(Shape.of(batchSize, numHeads, seqLen, kvSeqLen));
    }

    // ========================== Getter ==========================

    public int getDModel() { return dModel; }
    public int getNumHeads() { return numHeads; }
    public int getHeadDim() { return headDim; }
    public int getRotaryDim() { return rotaryDim; }
    public boolean isUseRoPE() { return useRoPE; }

    @Override
    public String toString() {
        return String.format(
                "GPT3Attention{name='%s', dModel=%d, heads=%d, headDim=%d, " +
                "RoPE=%s(dim=%d), sparse=%s, localWin=%d, stride=%d}",
                name, dModel, numHeads, headDim,
                useRoPE, rotaryDim,
                config.isSparseAttention(),
                config.getSparseLocalWindow(),
                config.getSparseStrideSize());
    }
}
