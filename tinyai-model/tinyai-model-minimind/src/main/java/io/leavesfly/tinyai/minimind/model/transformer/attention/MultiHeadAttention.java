package io.leavesfly.tinyai.minimind.model.transformer.attention;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.minimind.model.MiniMindConfig;
import io.leavesfly.tinyai.minimind.model.embedding.RotaryPositionEmbedding;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.core.Module;
import io.leavesfly.tinyai.nnet.layer.dnn.Dropout;
import io.leavesfly.tinyai.nnet.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.layer.norm.RMSNorm;

/**
 * 多头注意力机制（Multi-Head Attention，对标 Python Attention）
 * <p>
 * 支持 GQA（Grouped Query Attention）和 QK Normalization。
 * <p>
 * 计算流程（对标 Python）：
 * 1. Q = X @ W_Q, K = X @ W_K, V = X @ W_V
 * 2. 多头分割：Q -> [batch, seqLen, numHeads, headDim], K/V -> [batch, seqLen, numKVHeads, headDim]
 * 3. QK Normalization: Q = RMSNorm(Q), K = RMSNorm(K)
 * 4. 应用 RoPE 位置编码
 * 5. repeat_kv: 将 KV 头扩展到与 Q 头一致
 * 6. 转置为 [batch, numHeads, seqLen, headDim]
 * 7. 计算注意力分数：scores = (Q @ K^T) / sqrt(headDim)
 * 8. 应用因果掩码
 * 9. Softmax 归一化 + Dropout
 * 10. 应用注意力权重：output = scores @ V
 * 11. 多头合并 + 输出投影
 *
 * @author leavesfly
 * @version 2.0
 */
public class MultiHeadAttention extends Module {

    /** 隐藏层维度 */
    private final int hiddenSize;

    /** Q 注意力头数 */
    private final int numHeads;

    /** KV 注意力头数（GQA） */
    private final int numKVHeads;

    /** 每个头的维度 */
    private final int headDim;

    /** KV 头重复倍数 (numHeads / numKVHeads) */
    private final int numKVGroups;

    /** Query 投影层（支持 LoRA 替换） */
    private Module queryProj;

    /** Key 投影层 */
    private Module keyProj;

    /** Value 投影层（支持 LoRA 替换） */
    private Module valueProj;

    /** 输出投影层 */
    private Module outputProj;

    /** QK Normalization - Q 归一化（对标 Python self.q_norm） */
    private final RMSNorm qNorm;

    /** QK Normalization - K 归一化（对标 Python self.k_norm） */
    private final RMSNorm kNorm;

    /** RoPE 位置编码 */
    private final RotaryPositionEmbedding rope;

    /** Dropout 比例 */
    private final float dropoutRate;

    /** 注意力权重的 Dropout 层 */
    private final Dropout attnDropout;

    /** 是否处于训练模式 */
    private boolean training = true;

    /** 掩码缓存 */
    private Variable cachedMask;
    private long cachedMaskSize = -1;

    /**
     * 使用 MiniMindConfig 构造（推荐，支持 GQA + QK Norm）
     *
     * @param name   层名称
     * @param config 模型配置
     */
    public MultiHeadAttention(String name, MiniMindConfig config) {
        super(name);

        this.hiddenSize = config.getHiddenSize();
        this.numHeads = config.getNumHeads();
        this.numKVHeads = config.getNumKVHeads();
        this.headDim = config.getHeadDim();
        this.numKVGroups = numHeads / numKVHeads;
        this.dropoutRate = config.getDropout();

        // GQA 不变量兜底校验：若 numKVGroups 退化为 0，repeatKV（条件 numKVGroups > 1）会被跳过，
        // K/V 的头数就会多于 Q，直到 batchedMatMul 的 reshape 才报出"形状大小不匹配"。
        // 在这里立即报错，把难以定位的形状异常换成可操作的配置提示。
        if (numKVHeads <= 0 || numKVHeads > numHeads || numHeads % numKVHeads != 0) {
            throw new IllegalArgumentException(
                "非法的 GQA 配置: numHeads(" + numHeads + ") 必须是 numKVHeads(" + numKVHeads
                + ") 的整数倍且不小于它");
        }

        // Q 投影: hiddenSize -> numHeads * headDim
        this.queryProj = new Linear("query_proj", hiddenSize, numHeads * headDim, false);
        // K 投影: hiddenSize -> numKVHeads * headDim（GQA: KV 维度更小）
        this.keyProj = new Linear("key_proj", hiddenSize, numKVHeads * headDim, false);
        // V 投影: hiddenSize -> numKVHeads * headDim
        this.valueProj = new Linear("value_proj", hiddenSize, numKVHeads * headDim, false);
        // 输出投影: numHeads * headDim -> hiddenSize
        this.outputProj = new Linear("output_proj", numHeads * headDim, hiddenSize, false);

        registerModule("query_proj", queryProj);
        registerModule("key_proj", keyProj);
        registerModule("value_proj", valueProj);
        registerModule("output_proj", outputProj);

        // QK Normalization（对标 Python self.q_norm / self.k_norm）
        this.qNorm = new RMSNorm("q_norm", headDim, config.getEpsilon());
        this.kNorm = new RMSNorm("k_norm", headDim, config.getEpsilon());
        registerModule("q_norm", qNorm);
        registerModule("k_norm", kNorm);

        // Dropout
        this.attnDropout = new Dropout("attn_dropout", dropoutRate);
        registerModule("attn_dropout", attnDropout);

        // RoPE（使用 config 中的 theta 和 maxPositionEmbeddings）
        this.rope = new RotaryPositionEmbedding(headDim, config.getMaxPositionEmbeddings(),
                config.getRopeTheta());
        registerModule("rope", rope);

        init();
    }

    /**
     * 兼容旧接口的构造函数（不支持 GQA，numKVHeads = numHeads，无 QK Norm）
     */
    public MultiHeadAttention(String name, int hiddenSize, int numHeads, int maxSeqLen, float dropoutRate) {
        super(name);

        if (hiddenSize % numHeads != 0) {
            throw new IllegalArgumentException("hiddenSize must be divisible by numHeads");
        }

        this.hiddenSize = hiddenSize;
        this.numHeads = numHeads;
        this.numKVHeads = numHeads; // 无 GQA
        this.headDim = hiddenSize / numHeads;
        this.numKVGroups = 1;
        this.dropoutRate = dropoutRate;

        this.queryProj = new Linear("query_proj", hiddenSize, hiddenSize, false);
        this.keyProj = new Linear("key_proj", hiddenSize, hiddenSize, false);
        this.valueProj = new Linear("value_proj", hiddenSize, hiddenSize, false);
        this.outputProj = new Linear("output_proj", hiddenSize, hiddenSize, false);

        registerModule("query_proj", queryProj);
        registerModule("key_proj", keyProj);
        registerModule("value_proj", valueProj);
        registerModule("output_proj", outputProj);

        // 旧接口无 QK Norm，使用 identity-like RMSNorm
        this.qNorm = null;
        this.kNorm = null;

        this.attnDropout = new Dropout("attn_dropout", dropoutRate);
        registerModule("attn_dropout", attnDropout);

        this.rope = new RotaryPositionEmbedding(headDim, maxSeqLen);
        registerModule("rope", rope);

        init();
    }

    @Override
    public Variable forward(Variable... inputs) {
        // 默认不使用 KV-Cache
        return forwardWithCache(inputs[0], null, 0);
    }

    /**
     * 带 KV-Cache 的前向传播（对标 Python Attention.forward）
     *
     * @param x        输入 Variable [batch, seqLen, hiddenSize]
     * @param kvCache  KV-Cache 对象（可为 null）
     * @param startPos 起始位置（用于 RoPE 和因果掩码）
     * @return 输出 Variable [batch, seqLen, hiddenSize]
     */
    public Variable forwardWithCache(Variable x, KVCache kvCache, int startPos) {

        int[] xShape = x.getValue().getShape().getShapeDims();
        int batchSize = xShape[0];
        int seqLen = xShape[1];

        // 1. Q、K、V 投影
        Variable Q = queryProj.forward(x);  // [batch, seqLen, numHeads * headDim]
        Variable K = keyProj.forward(x);    // [batch, seqLen, numKVHeads * headDim]
        Variable V = valueProj.forward(x);  // [batch, seqLen, numKVHeads * headDim]

        // 2. 多头分割
        // Q: [batch, seqLen, numHeads, headDim]
        Variable qSplit = Q.reshape(Shape.of(batchSize, seqLen, numHeads, headDim));
        // K: [batch, seqLen, numKVHeads, headDim]
        Variable kSplit = K.reshape(Shape.of(batchSize, seqLen, numKVHeads, headDim));
        // V: [batch, seqLen, numKVHeads, headDim]
        Variable vSplit = V.reshape(Shape.of(batchSize, seqLen, numKVHeads, headDim));

        // 3. QK Normalization（对标 Python: xq, xk = self.q_norm(xq), self.k_norm(xk)）
        if (qNorm != null && kNorm != null) {
            qSplit = qNorm.forward(qSplit);
            kSplit = kNorm.forward(kSplit);
        }

        // 4. 转置为 [batch, numHeads/numKVHeads, seqLen, headDim] 以便应用 RoPE
        qSplit = qSplit.permute(0, 2, 1, 3);
        kSplit = kSplit.permute(0, 2, 1, 3);
        vSplit = vSplit.permute(0, 2, 1, 3);

        // 5. 应用 RoPE 位置编码
        qSplit = rope.forward(qSplit, new Variable(NdArray.of(new float[]{startPos})));
        kSplit = rope.forward(kSplit, new Variable(NdArray.of(new float[]{startPos})));

        // 6. KV-Cache 处理（推理时使用）
        if (kvCache != null) {
            NdArray[] updated = kvCache.update(kSplit.getValue(), vSplit.getValue());
            kSplit = new Variable(updated[0]);
            kSplit.setRequireGrad(false);
            vSplit = new Variable(updated[1]);
            vSplit.setRequireGrad(false);
        }

        // 7. repeat_kv: 将 KV 头扩展到与 Q 头一致（GQA 核心）
        if (numKVGroups > 1) {
            kSplit = repeatKV(kSplit, numKVGroups);
            vSplit = repeatKV(vSplit, numKVGroups);
        }

        int kvSeqLen = kSplit.getShape().getShapeDims()[2];

        // 8. 注意力计算
        Variable attnOutput = computeAttentionWithVariable(qSplit, kSplit, vSplit,
                batchSize, seqLen, kvSeqLen, startPos,
                kvCache == null);

        // 9. 多头合并：[batch, numHeads, seqLen, headDim] -> [batch, seqLen, numHeads * headDim]
        Variable merged = mergeMultiHead(attnOutput, batchSize, seqLen);

        // 10. 输出投影
        return outputProj.forward(merged);
    }

    /**
     * 使用 Variable 层面操作计算注意力
     */
    private Variable computeAttentionWithVariable(Variable Q, Variable K, Variable V,
                                         int batchSize, int seqLen, int kvSeqLen, int startPos,
                                         boolean applyMask) {
        // Q: [batch, numHeads, seqLen, headDim]
        // K: [batch, numHeads, kvSeqLen, headDim]
        // V: [batch, numHeads, kvSeqLen, headDim]
        
        // 5. 计算注意力分数：scores = (Q @ K^T) / sqrt(headDim)
        // K^T: [batch, numHeads, headDim, kvSeqLen]
        Variable KT = transposeLastTwoDims(K);  // [batch, numHeads, headDim, kvSeqLen]
        
        // 批量矩阵乘法: [batch*numHeads, seqLen, headDim] @ [batch*numHeads, headDim, kvSeqLen]
        // -> [batch*numHeads, seqLen, kvSeqLen]
        Variable scores = batchedMatMul(Q, KT, batchSize, numHeads, seqLen, headDim, kvSeqLen);
        
        // 缩放
        float scale = (float) (1.0 / Math.sqrt(headDim));
        Variable scaleVar = new Variable(scale);
        scaleVar.setRequireGrad(false);
        scores = scores.mul(scaleVar);
        
        // 6. 应用因果掩码
        if (training || applyMask) {
            scores = applyCausalMaskVar(scores, batchSize, numHeads, seqLen, kvSeqLen, startPos);
        }
        
        // 7. Softmax 归一化 (在最后一个维度上)
        Variable attnWeights = softmaxLastDim(scores, batchSize, numHeads, seqLen, kvSeqLen);
        
        // 8. Dropout（训练时）
        if (training) {
            attnWeights = attnDropout.forward(attnWeights);
        }
        
        // 9. 应用注意力权重：output = attnWeights @ V
        // [batch*numHeads, seqLen, kvSeqLen] @ [batch*numHeads, kvSeqLen, headDim]
        // -> [batch*numHeads, seqLen, headDim]
        Variable attended = batchedMatMul(attnWeights, V, batchSize, numHeads, seqLen, kvSeqLen, headDim);
        
        return attended;
    }

    /**
     * repeat_kv: 将 KV 头扩展到与 Q 头一致（对标 Python repeat_kv）
     * <p>
     * 输入: [batch, numKVHeads, seqLen, headDim]
     * 输出: [batch, numKVHeads * nRep, seqLen, headDim]
     */
    private Variable repeatKV(Variable x, int nRep) {
        if (nRep == 1) return x;
        int[] shape = x.getShape().getShapeDims();
        int bs = shape[0], nKVHeads = shape[1], slen = shape[2], hd = shape[3];
        // [bs, nKVHeads, slen, headDim] -> [bs, nKVHeads, 1, slen, headDim]
        Variable expanded = x.reshape(Shape.of(bs, nKVHeads, 1, slen, hd));
        // broadcast to [bs, nKVHeads, nRep, slen, headDim]
        expanded = expanded.broadcastTo(Shape.of(bs, nKVHeads, nRep, slen, hd));
        // reshape to [bs, nKVHeads * nRep, slen, headDim]
        return expanded.reshape(Shape.of(bs, nKVHeads * nRep, slen, hd));
    }

    /**
     * 设置训练模式
     * <p>
     * 同时传播到所有投影层子模块（包括 LoRALinear 等需要感知训练模式的层）
     */
    public void setTraining(boolean training) {
        this.training = training;
        // 传播到投影层子模块（LoRALinear 的 Dropout 依赖 _training 状态）
        if (queryProj != null) {
            queryProj.train(training);
        }
        if (keyProj != null) {
            keyProj.train(training);
        }
        if (valueProj != null) {
            valueProj.train(training);
        }
        if (outputProj != null) {
            outputProj.train(training);
        }
        // 同步注意力 Dropout 的训练状态
        attnDropout.train(training);
    }

    /**
     * 获取注意力头数
     */
    public int getNumHeads() {
        return numHeads;
    }

    /**
     * 获取每个头的维度
     */
    public int getHeadDim() {
        return headDim;
    }
    
    /**
     * 获取 Query 投影层
     */
    public Module getQueryProj() {
        return queryProj;
    }
    
    /**
     * 获取 Value 投影层
     */
    public Module getValueProj() {
        return valueProj;
    }
    
    /**
     * 获取 Key 投影层
     */
    public Module getKeyProj() {
        return keyProj;
    }
    
    /**
     * 获取 Output 投影层
     */
    public Module getOutputProj() {
        return outputProj;
    }
    
    /**
     * 设置 Query 投影层（用于 LoRA 注入）
     */
    public void setQueryProj(Module queryProj) {
        this.queryProj = queryProj;
        // 直接替换模块（绕过 registerModule 的重复检查）
        _modules.put("query_proj", queryProj);
        if (queryProj != null) {
            queryProj.setParent(this);
        }
    }
    
    /**
     * 设置 Value 投影层（用于 LoRA 注入）
     */
    public void setValueProj(Module valueProj) {
        this.valueProj = valueProj;
        // 直接替换模块（绕过 registerModule 的重复检查）
        _modules.put("value_proj", valueProj);
        if (valueProj != null) {
            valueProj.setParent(this);
        }
    }
    
    /**
     * 设置 Key 投影层（用于 LoRA 注入）
     */
    public void setKeyProj(Module keyProj) {
        this.keyProj = keyProj;
        // 直接替换模块（绕过 registerModule 的重复检查）
        _modules.put("key_proj", keyProj);
        if (keyProj != null) {
            keyProj.setParent(this);
        }
    }
    
    /**
     * 设置 Output 投影层（用于 LoRA 注入）
     */
    public void setOutputProj(Module outputProj) {
        this.outputProj = outputProj;
        // 直接替换模块（绕过 registerModule 的重复检查）
        _modules.put("output_proj", outputProj);
        if (outputProj != null) {
            outputProj.setParent(this);
        }
    }
    
    /**
     * 获取隐藏层维度
     */
    public int getHiddenSize() {
        return hiddenSize;
    }
    
    // =============================================================================
    // Variable 层面的辅助方法
    // =============================================================================
    
    /**
     * 多头合并（使用 Variable.permute + reshape，保持计算图连通）
     * [batch, numHeads, seqLen, headDim] -> [batch, seqLen, numHeads * headDim]
     */
    private Variable mergeMultiHead(Variable input, int batchSize, int seqLen) {
        Variable transposed = input.permute(0, 2, 1, 3);
        return transposed.reshape(Shape.of(batchSize, seqLen, numHeads * headDim));
    }
    
    /**
     * 转置张量的最后两个维度（通过 permute 保持计算图连通）
     */
    private Variable transposeLastTwoDims(Variable input) {
        int[] shape = input.getShape().getShapeDims();
        int ndim = shape.length;
        int[] perm = new int[ndim];
        for (int i = 0; i < ndim - 2; i++) {
            perm[i] = i;
        }
        perm[ndim - 2] = ndim - 1;
        perm[ndim - 1] = ndim - 2;
        return input.permute(perm);
    }
    
    /**
     * 批量矩阵乘法（使用 Variable.bmm）
     */
    private Variable batchedMatMul(Variable a, Variable b, int batchSize, int numHeads,
                                   int m, int k, int n) {
        // a: [batch, numHeads, m, k]
        // b: [batch, numHeads, k, n]
        // -> [batch, numHeads, m, n]
        
        // Reshape 为 3D: [batch*numHeads, m, k] 和 [batch*numHeads, k, n]
        Variable a3d = a.reshape(Shape.of(batchSize * numHeads, m, k));
        Variable b3d = b.reshape(Shape.of(batchSize * numHeads, k, n));
        
        // 批量矩阵乘法
        Variable result3d = a3d.bmm(b3d);  // [batch*numHeads, m, n]
        
        // Reshape 回 4D
        return result3d.reshape(Shape.of(batchSize, numHeads, m, n));
    }
    
    /**
     * 应用因果掩码（使用 Variable.maskedFill）
     * 优化：添加掩码缓存，只在序列长度变化时重新创建
     */
    private Variable applyCausalMaskVar(Variable scores, int batchSize, int numHeads,
                                        int qSeqLen, int kvSeqLen, int startPos) {
        // scores: [batch, numHeads, qSeqLen, kvSeqLen]
        
        // 使用 Long 类型的缓存键，使用更大的乘数避免冲突
        long cacheKey = (long) batchSize * 1_000_000L * 1_000_000L + 
                        (long) qSeqLen * 1_000_000L + 
                        (long) kvSeqLen;
        
        // 检查缓存是否有效
        if (cachedMask != null && cachedMaskSize == cacheKey) {
            // 使用缓存的掩码
            return scores.maskedFill(cachedMask, -1e9f);
        }
        
        // 创建新的因果掩码矩阵
        NdArray maskData = NdArray.zeros(Shape.of(batchSize, numHeads, qSeqLen, kvSeqLen));
        float[] maskBuffer = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) maskData).buffer;
        
        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < numHeads; h++) {
                for (int i = 0; i < qSeqLen; i++) {
                    for (int j = 0; j < kvSeqLen; j++) {
                        int currentPos = startPos + i;
                        if (j > currentPos) {
                            int idx = ((b * numHeads + h) * qSeqLen + i) * kvSeqLen + j;
                            maskBuffer[idx] = 1.0f;  // 标记需要掩码的位置
                        }
                    }
                }
            }
        }
        
        // 缓存新创建的掩码
        cachedMask = new Variable(maskData);
        cachedMask.setRequireGrad(false);
        cachedMaskSize = cacheKey;
        
        // 使用 maskedFill 将掩码位置填充为较大的负数（避免使用负无穷导致 NaN）
        return scores.maskedFill(cachedMask, -1e9f);
    }
    
    /**
     * 在最后一个维度上应用 Softmax
     */
    private Variable softmaxLastDim(Variable input, int batchSize, int numHeads,
                                    int qSeqLen, int kvSeqLen) {
        // input: [batch, numHeads, qSeqLen, kvSeqLen]
        // 从实际张量形状获取维度，避免参数与实际形状不一致
        int[] actualShape = input.getValue().getShape().getShapeDims();
        int actualRows = actualShape[0] * actualShape[1] * actualShape[2];
        int actualCols = actualShape[3];
        
        // Reshape 为 2D: [batch*numHeads*qSeqLen, kvSeqLen]
        Variable reshaped = input.reshape(Shape.of(actualRows, actualCols));
        
        // 应用 softmax
        Variable softmaxed = reshaped.softMax();
        
        // Reshape 回 4D（使用实际形状）
        return softmaxed.reshape(Shape.of(actualShape[0], actualShape[1], actualShape[2], actualShape[3]));
    }
}