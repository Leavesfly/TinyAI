package io.leavesfly.tinyai.gpt3;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

/**
 * GPT-3 KV缓存（Key-Value Cache）
 *
 * 用于自回归文本生成时加速推理：
 * - 缓存历史Token的 Key 和 Value 向量
 * - 每次只计算新Token的Q/K/V，K/V和历史缓存拼接后计算注意力
 * - 计算量从 O(n²) 降低至 O(n) 每步
 *
 * 缓存形状：[batchSize, numHeads, seqLen, headDim]
 *
 * @author leavesfly
 * @version 1.0
 */
public class GPT3KVCache {

    /** 缓存的 Key 向量，Shape: [batch, numHeads, cachedLen, headDim] */
    private NdArray cachedK;

    /** 缓存的 Value 向量，Shape: [batch, numHeads, cachedLen, headDim] */
    private NdArray cachedV;

    /** 当前缓存的序列长度 */
    private int currentSeqLen;

    private final int batchSize;
    private final int numHeads;
    private final int headDim;
    private final int maxCacheLen;

    /**
     * 是否使用 RoPE 位置编码。
     * 当启用 RoPE 时，K 向量中已嵌入绝对位置信息，滑动窗口截断会破坏位置关系，
     * 因此启用 RoPE 时禁用滑动窗口截断。
     */
    private final boolean useRoPE;

    /** 滑动窗口截断警告是否已输出（避免重复打印） */
    private boolean slidingWindowWarned = false;

    /**
     * 创建KV缓存
     *
     * @param batchSize   批次大小
     * @param numHeads    注意力头数
     * @param headDim     每个头的维度
     * @param maxCacheLen 最大缓存序列长度（通常等于 nPositions）
     */
    public GPT3KVCache(int batchSize, int numHeads, int headDim, int maxCacheLen) {
        this(batchSize, numHeads, headDim, maxCacheLen, false);
    }

    /**
     * 创建KV缓存（支持 RoPE 安全模式）
     *
     * @param batchSize   批次大小
     * @param numHeads    注意力头数
     * @param headDim     每个头的维度
     * @param maxCacheLen 最大缓存序列长度（通常等于 nPositions）
     * @param useRoPE     是否使用 RoPE 位置编码（启用时禁止滑动窗口截断）
     */
    public GPT3KVCache(int batchSize, int numHeads, int headDim, int maxCacheLen, boolean useRoPE) {
        this.batchSize = batchSize;
        this.numHeads = numHeads;
        this.headDim = headDim;
        this.maxCacheLen = maxCacheLen;
        this.useRoPE = useRoPE;
        this.currentSeqLen = 0;
        this.cachedK = null;
        this.cachedV = null;
    }

    /**
     * 更新缓存，将新的 K/V 追加到历史缓存，返回完整的 K/V 序列
     *
     * @param newK 新Token的Key，Shape: [batch, numHeads, newSeqLen, headDim]
     * @param newV 新Token的Value，Shape: [batch, numHeads, newSeqLen, headDim]
     * @return 完整的 [cachedK, cachedV]，包含历史+新Token
     */
    public NdArray[] update(NdArray newK, NdArray newV) {
        if (cachedK == null || cachedV == null) {
            // 首次填充
            cachedK = newK;
            cachedV = newV;
            currentSeqLen = newK.getShape().getDimension(2);
        } else {
            // 拼接历史缓存与新的K/V
            cachedK = concatenateOnSeqDim(cachedK, newK);
            cachedV = concatenateOnSeqDim(cachedV, newV);
            currentSeqLen += newK.getShape().getDimension(2);

            // 超出最大长度时的处理
            if (currentSeqLen > maxCacheLen) {
                if (useRoPE) {
                    // RoPE 模式下禁止滑动窗口截断：K 向量中已嵌入绝对位置信息，
                    // 截断后剩余 K 的位置编码与实际索引不一致，会导致注意力计算错误。
                    // 此处仅输出警告，保留完整缓存以保证正确性。
                    if (!slidingWindowWarned) {
                        System.err.println("[GPT3KVCache] 警告: 缓存长度(" + currentSeqLen
                                + ")超过最大限制(" + maxCacheLen
                                + ")，但因启用 RoPE 无法安全截断，将保留完整缓存。"
                                + "建议增大 nPositions 或缩短生成长度。");
                        slidingWindowWarned = true;
                    }
                } else {
                    // 非 RoPE 模式：安全执行滑动窗口截断
                    int excess = currentSeqLen - maxCacheLen;
                    cachedK = sliceOnSeqDim(cachedK, excess, currentSeqLen);
                    cachedV = sliceOnSeqDim(cachedV, excess, currentSeqLen);
                    currentSeqLen = maxCacheLen;
                }
            }
        }
        return new NdArray[]{cachedK, cachedV};
    }

    /**
     * 在序列维度（dim=2）上拼接两个 NdArray
     * cached: [batch, heads, oldSeq, dim]
     * newData: [batch, heads, newSeq, dim]
     * -> [batch, heads, oldSeq+newSeq, dim]
     */
    private NdArray concatenateOnSeqDim(NdArray cached, NdArray newData) {
        int[] cs = cached.getShape().getShapeDims();
        int[] ns = newData.getShape().getShapeDims();

        int batch = cs[0], heads = cs[1], oldSeq = cs[2], dim = cs[3];
        int newSeq = ns[2];
        int totalSeq = oldSeq + newSeq;

        float[] result = new float[batch * heads * totalSeq * dim];
        float[] cData = cached.getArray();
        float[] nData = newData.getArray();

        for (int b = 0; b < batch; b++) {
            for (int h = 0; h < heads; h++) {
                int headIndex = b * heads + h;
                // 批量复制历史缓存（用 System.arraycopy 替代逐元素循环）
                int cachedSrcOffset = headIndex * oldSeq * dim;
                int resultDstOffset = headIndex * totalSeq * dim;
                System.arraycopy(cData, cachedSrcOffset, result, resultDstOffset, oldSeq * dim);
                // 批量追加新Token的K/V
                int newSrcOffset = headIndex * newSeq * dim;
                int newDstOffset = resultDstOffset + oldSeq * dim;
                System.arraycopy(nData, newSrcOffset, result, newDstOffset, newSeq * dim);
            }
        }
        return NdArray.of(result, Shape.of(batch, heads, totalSeq, dim));
    }

    /**
     * 在序列维度上切片 [start, end)
     */
    private NdArray sliceOnSeqDim(NdArray data, int start, int end) {
        int[] shape = data.getShape().getShapeDims();
        int batch = shape[0], heads = shape[1], seqLen = shape[2], dim = shape[3];
        int newSeq = end - start;

        float[] result = new float[batch * heads * newSeq * dim];
        float[] srcArray = data.getArray();

        for (int b = 0; b < batch; b++) {
            for (int h = 0; h < heads; h++) {
                int headIndex = b * heads + h;
                int srcOffset = (headIndex * seqLen + start) * dim;
                int dstOffset = headIndex * newSeq * dim;
                System.arraycopy(srcArray, srcOffset, result, dstOffset, newSeq * dim);
            }
        }
        return NdArray.of(result, Shape.of(batch, heads, newSeq, dim));
    }

    /** 清空缓存（开始新的生成任务） */
    public void clear() {
        cachedK = null;
        cachedV = null;
        currentSeqLen = 0;
    }

    /** 当前缓存的序列长度 */
    public int getCurrentSeqLen() {
        return currentSeqLen;
    }

    /** 缓存是否为空 */
    public boolean isEmpty() {
        return cachedK == null;
    }

    public NdArray getCachedK() {
        return cachedK;
    }

    public NdArray getCachedV() {
        return cachedV;
    }

    @Override
    public String toString() {
        return String.format("GPT3KVCache{batch=%d, heads=%d, headDim=%d, currentLen=%d, maxLen=%d}",
                batchSize, numHeads, headDim, currentSeqLen, maxCacheLen);
    }
}
