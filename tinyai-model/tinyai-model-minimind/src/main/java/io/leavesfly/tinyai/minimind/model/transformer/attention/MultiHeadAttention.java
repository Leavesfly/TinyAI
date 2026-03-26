package io.leavesfly.tinyai.minimind.model.transformer.attention;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.minimind.model.embedding.RotaryPositionEmbedding;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Dropout;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;

/**
 * 多头注意力机制（Multi-Head Attention）
 * <p>
 * 实现功能：
 * - Q、K、V 投影使用 V2 Linear 层
 * - 集成 RoPE 旋转位置编码
 * - 支持因果掩码(Causal Mask)
 * - 支持 KV-Cache 增量推理
 * - Scaled Dot-Product Attention
 * <p>
 * 计算流程：
 * 1. Q = X @ W_Q, K = X @ W_K, V = X @ W_V
 * 2. 应用 RoPE 位置编码到 Q、K
 * 3. 多头分割：reshape 为 [batch, numHeads, seqLen, headDim]
 * 4. 计算注意力分数：scores = (Q @ K^T) / sqrt(headDim)
 * 5. 应用因果掩码
 * 6. Softmax 归一化
 * 7. 应用注意力权重：output = scores @ V
 * 8. 多头合并：reshape 为 [batch, seqLen, hiddenSize]
 * 9. 输出投影：output @ W_O
 *
 * @author leavesfly
 * @version 1.0
 */
public class MultiHeadAttention extends Module {

    /**
     * 隐藏层维度
     */
    private final int hiddenSize;

    /**
     * 注意力头数
     */
    private final int numHeads;

    /**
     * 每个头的维度
     */
    private final int headDim;

    /**
     * Query 投影层（支持 LoRA 替换）
     */
    private Module queryProj;

    /**
     * Key 投影层
     */
    private Module keyProj;

    /**
     * Value 投影层（支持 LoRA 替换）
     */
    private Module valueProj;

    /**
     * 输出投影层
     */
    private Module outputProj;

    /**
     * RoPE 位置编码
     */
    private final RotaryPositionEmbedding rope;

    /**
     * Dropout 比例
     */
    private final float dropoutRate;

    /**
     * 注意力权重的 Dropout 层
     */
    private final Dropout attnDropout;

    /**
     * 是否处于训练模式
     */
    private boolean training = true;
    
    /**
     * 掩码缓存：缓存已创建的因果掩码
     */
    private Variable cachedMask;
    
    /**
     * 缓存的掩码大小：记录缓存的掩码对应的序列长度
     */
    private int cachedMaskSize = -1;

    /**
     * 构造多头注意力层
     *
     * @param name        层名称
     * @param hiddenSize  隐藏层维度
     * @param numHeads    注意力头数
     * @param maxSeqLen   最大序列长度
     * @param dropoutRate Dropout 比例
     */
    public MultiHeadAttention(String name, int hiddenSize, int numHeads, int maxSeqLen, float dropoutRate) {
        super(name);

        if (hiddenSize % numHeads != 0) {
            throw new IllegalArgumentException("hiddenSize must be divisible by numHeads");
        }

        this.hiddenSize = hiddenSize;
        this.numHeads = numHeads;
        this.headDim = hiddenSize / numHeads;
        this.dropoutRate = dropoutRate;

        // 创建投影层（使用 V2 Linear）
        this.queryProj = new Linear("query_proj", hiddenSize, hiddenSize, false);
        this.keyProj = new Linear("key_proj", hiddenSize, hiddenSize, false);
        this.valueProj = new Linear("value_proj", hiddenSize, hiddenSize, false);
        this.outputProj = new Linear("output_proj", hiddenSize, hiddenSize, false);

        // 注册子模块
        registerModule("query_proj", queryProj);
        registerModule("key_proj", keyProj);
        registerModule("value_proj", valueProj);
        registerModule("output_proj", outputProj);

        // 创建注意力 Dropout
        this.attnDropout = new Dropout("attn_dropout", dropoutRate);
        registerModule("attn_dropout", attnDropout);

        // 创建 RoPE 位置编码
        this.rope = new RotaryPositionEmbedding(headDim, maxSeqLen);
        registerModule("rope", rope);

        // 初始化参数
        init();
    }

    @Override
    public Variable forward(Variable... inputs) {
        // 默认不使用 KV-Cache
        return forwardWithCache(inputs[0], null, 0);
    }

    /**
     * 带 KV-Cache 的前向传播
     *
     * @param x        输入 Variable
     * @param kvCache  KV-Cache 对象（可为 null）
     * @param startPos 起始位置（用于 RoPE 和因果掩码）
     * @return 输出 Variable
     */
    public Variable forwardWithCache(Variable x, KVCache kvCache, int startPos) {

        // 1. Q、K、V 投影
        Variable Q = queryProj.forward(x);
        Variable K = keyProj.forward(x);
        Variable V = valueProj.forward(x);

        // 获取输入形状
        int[] qShape = Q.getValue().getShape().getShapeDims();
        int batchSize = qShape[0];
        int seqLen = qShape[1];

        // 2. 多头分割：[batch, seqLen, hiddenSize] -> [batch, numHeads, seqLen, headDim]
        //    先分割，RoPE 需要 headDim 维度的输入
        Variable qSplit = reshapeForMultiHead(Q, batchSize, seqLen);
        Variable kSplit = reshapeForMultiHead(K, batchSize, seqLen);
        Variable vSplit = reshapeForMultiHead(V, batchSize, seqLen);

        // 3. 应用 RoPE 位置编码（在 headDim 维度上）
        qSplit = rope.forward(qSplit, new Variable(NdArray.of(new float[]{startPos})));
        kSplit = rope.forward(kSplit, new Variable(NdArray.of(new float[]{startPos})));

        // 4. KV-Cache 处理
        if (kvCache != null) {
            NdArray[] updated = kvCache.update(kSplit.getValue(), vSplit.getValue());
            kSplit = new Variable(updated[0]);
            vSplit = new Variable(updated[1]);
        }

        int kvSeqLen = kSplit.getShape().getShapeDims()[2];

        // 5-9. 注意力计算：使用 Variable 层面操作
        Variable attnOutput = computeAttentionWithVariable(qSplit, kSplit, vSplit, 
                                                   batchSize, seqLen, kvSeqLen, startPos, 
                                                   kvCache == null);

        // 10. 多头合并：[batch, numHeads, seqLen, headDim] -> [batch, seqLen, hiddenSize]
        Variable merged = mergeMultiHead(attnOutput, batchSize, seqLen);

        // 11. 输出投影
        Variable output = outputProj.forward(merged);

        return output;
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

    // 已删除旧的 NdArray 直接操作方法，改用 Variable 算子

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
     * 获取隐藏层维度
     */
    public int getHiddenSize() {
        return hiddenSize;
    }
    
    // =============================================================================
    // Variable 层面的辅助方法
    // =============================================================================
    
    /**
     * 多头分割（使用 Variable.reshape）
     * [batch, seqLen, hiddenSize] -> [batch, numHeads, seqLen, headDim]
     */
    private Variable reshapeForMultiHead(Variable input, int batchSize, int seqLen) {
        // [batch, seqLen, hiddenSize] -> [batch, seqLen, numHeads, headDim]
        Variable reshaped1 = input.reshape(Shape.of(batchSize, seqLen, numHeads, headDim));
        // 转置为 [batch, numHeads, seqLen, headDim]
        // 由于 Variable 暂不支持多维转置，使用 NdArray 操作
        NdArray data = reshaped1.getValue();
        NdArray transposed = transposeAxes(data, new int[]{0, 2, 1, 3});
        return new Variable(transposed);
    }
    
    /**
     * 多头合并（使用 Variable.reshape）
     * [batch, numHeads, seqLen, headDim] -> [batch, seqLen, hiddenSize]
     */
    private Variable mergeMultiHead(Variable input, int batchSize, int seqLen) {
        // [batch, numHeads, seqLen, headDim] -> [batch, seqLen, numHeads, headDim]
        NdArray data = input.getValue();
        NdArray transposed = transposeAxes(data, new int[]{0, 2, 1, 3});
        Variable transposedVar = new Variable(transposed);
        // [batch, seqLen, numHeads, headDim] -> [batch, seqLen, hiddenSize]
        return transposedVar.reshape(Shape.of(batchSize, seqLen, hiddenSize));
    }
    
    /**
     * 转置张量的最后两个维度
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
        
        NdArray transposed = transposeAxes(input.getValue(), perm);
        return new Variable(transposed);
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
        
        // 缓存键必须包含 batchSize，避免不同 batch 大小时复用错误形状的掩码
        int cacheKey = batchSize * 100000000 + qSeqLen * 10000 + kvSeqLen;
        
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
    
    /**
     * 转置张量的轴
     */
    private NdArray transposeAxes(NdArray input, int[] perm) {
        int[] shape = input.getShape().getShapeDims();
        int[] newShape = new int[perm.length];
        for (int i = 0; i < perm.length; i++) {
            newShape[i] = shape[perm[i]];
        }
        
        float[] inputBuffer = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) input).buffer;
        float[] outputBuffer = new float[inputBuffer.length];
        
        // 计算步长
        int[] strides = new int[perm.length];
        strides[perm.length - 1] = 1;
        for (int i = perm.length - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        
        int[] newStrides = new int[perm.length];
        newStrides[perm.length - 1] = 1;
        for (int i = perm.length - 2; i >= 0; i--) {
            newStrides[i] = newStrides[i + 1] * newShape[i + 1];
        }
        
        // 转置
        int totalSize = inputBuffer.length;
        for (int i = 0; i < totalSize; i++) {
            int[] indices = new int[perm.length];
            int remainder = i;
            for (int j = 0; j < perm.length; j++) {
                indices[j] = remainder / strides[j];
                remainder %= strides[j];
            }
            
            int[] newIndices = new int[perm.length];
            for (int j = 0; j < perm.length; j++) {
                newIndices[j] = indices[perm[j]];
            }
            
            int newIdx = 0;
            for (int j = 0; j < perm.length; j++) {
                newIdx += newIndices[j] * newStrides[j];
            }
            
            outputBuffer[newIdx] = inputBuffer[i];
        }
        
        return NdArray.of(outputBuffer, Shape.of(newShape));
    }
}