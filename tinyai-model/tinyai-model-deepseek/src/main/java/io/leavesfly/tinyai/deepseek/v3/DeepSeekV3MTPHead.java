package io.leavesfly.tinyai.deepseek.v3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.loss.SoftmaxCrossEntropy;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.core.Module;
import io.leavesfly.tinyai.nnet.core.Parameter;
import io.leavesfly.tinyai.nnet.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.layer.norm.RMSNorm;

import java.util.ArrayList;
import java.util.List;

/**
 * DeepSeek-V3 Multi-Token Prediction (MTP) 预测头
 * 
 * 对标 DeepSeek-V3 论文的 MTP 机制：
 * 除了标准的 next-token prediction，额外预测后续 D 个 token（D 为 MTP 深度）。
 * 
 * 每个 MTP 深度 k 的预测流程：
 * 1. 将 Transformer 隐藏状态 h_i 与第 i+k 个 token 的嵌入向量拼接
 * 2. 通过线性投影层将拼接向量映射回 nEmbd 维度
 * 3. 通过 RMSNorm 归一化
 * 4. 通过共享的输出投影层（lm_head）得到 logits
 * 5. 预测目标是第 i+k+1 个 token
 * 
 * 关键设计：
 * - 每个深度有独立的投影层和 RMSNorm
 * - 输出投影层（lm_head）与主模型共享，减少参数量
 * - 训练时 MTP loss = 各深度 loss 的均值 × mtpLossWeight
 * - 推理时 MTP 头可用于推测解码（speculative decoding）加速
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekV3MTPHead extends Module {

    private final DeepSeekV3Config config;
    private final int mtpDepth;

    // 每个深度的投影层：将 [nEmbd + nEmbd] 拼接向量映射回 nEmbd
    private final List<Linear> projectionLayers;

    // 每个深度的 RMSNorm
    private final List<RMSNorm> normLayers;

    // 共享的输出投影层（lm_head），由外部传入
    private Linear sharedOutputProjection;

    // 共享的 token 嵌入参数，由外部传入
    private Parameter sharedTokenEmbeddings;

    /**
     * 构造函数
     * 
     * @param name                   模块名称
     * @param config                 V3 配置对象
     * @param sharedOutputProjection 与主模型共享的输出投影层（lm_head）
     * @param sharedTokenEmbeddings  与主模型共享的 token 嵌入参数
     */
    public DeepSeekV3MTPHead(String name, DeepSeekV3Config config,
                             Linear sharedOutputProjection, Parameter sharedTokenEmbeddings) {
        super(name);
        this.config = config;
        this.mtpDepth = config.getMtpDepth();
        this.sharedOutputProjection = sharedOutputProjection;
        this.sharedTokenEmbeddings = sharedTokenEmbeddings;

        this.projectionLayers = new ArrayList<>();
        this.normLayers = new ArrayList<>();

        initializeComponents();
    }

    /**
     * 初始化每个深度的投影层和 RMSNorm
     */
    private void initializeComponents() {
        int nEmbd = config.getNEmbd();

        for (int depth = 0; depth < mtpDepth; depth++) {
            // 投影层：将 [h_i; embed_{i+k}] 的拼接（2 * nEmbd）映射回 nEmbd
            Linear projection = new Linear(
                name + "_mtp_proj_" + depth,
                2 * nEmbd,
                nEmbd,
                false
            );
            projectionLayers.add(projection);
            registerModule("mtp_proj_" + depth, projection);

            // RMSNorm 归一化
            RMSNorm norm = new RMSNorm(
                name + "_mtp_norm_" + depth,
                nEmbd,
                (float) config.getLayerNormEpsilon()
            );
            normLayers.add(norm);
            registerModule("mtp_norm_" + depth, norm);
        }
    }

    /**
     * 前向传播（不直接使用，MTP 通过 computeMTPLogits 调用）
     */
    @Override
    public Variable forward(Variable... inputs) {
        throw new UnsupportedOperationException(
            "MTP 头不支持直接 forward 调用，请使用 computeMTPLoss 方法");
    }

    /**
     * 计算指定深度的 MTP logits
     * 
     * 对标 DeepSeek-V3 论文 MTP 流程：
     * 1. 取 Transformer 隐藏状态 h[0..validLen-1]
     * 2. 取对应位置偏移 offset 的 token 嵌入 embed[offset..seqLen-1]
     * 3. 拼接 [h_i; embed_{i+offset}] → 投影 → RMSNorm → 共享 lm_head → logits
     * 
     * 所有操作基于 Variable 完成，保持计算图连通以支持梯度回传。
     * 
     * @param hiddenStates Transformer 最后一层的隐藏状态 [batch, seqLen, nEmbd]
     * @param tokenIds     输入 token ID 序列 [batch, seqLen]
     * @param depth        MTP 深度（0-indexed，第 0 层预测 i+2，第 1 层预测 i+3...）
     * @return logits [batch, validLen, vocabSize]，validLen = seqLen - depth - 1
     */
    public Variable computeMTPLogits(Variable hiddenStates, Variable tokenIds, int depth) {
        if (depth < 0 || depth >= mtpDepth) {
            throw new IllegalArgumentException(
                String.format("MTP depth %d 超出范围 [0, %d)", depth, mtpDepth));
        }

        int seqLen = hiddenStates.getValue().getShape().getDimension(1);
        int offset = depth + 1;
        int validLen = seqLen - offset;
        if (validLen <= 0) {
            int batchSize = hiddenStates.getValue().getShape().getDimension(0);
            return new Variable(NdArray.zeros(Shape.of(batchSize, 0, config.getVocabSize())));
        }

        // 1. 截取隐藏状态 h[0..validLen-1]（保持计算图）
        Variable truncatedHidden = hiddenStates.sliceRange(1, 0, validLen);

        // 2. 获取偏移位置的 token 嵌入 embed[offset..seqLen-1]（保持计算图）
        Variable offsetTokenIds = tokenIds.sliceRange(1, offset, seqLen);
        Variable embedParam = new Variable(sharedTokenEmbeddings.data());
        // 将 [batch, validLen] 展平为 [batch*validLen]，做 indexSelect，再 reshape 回 [batch, validLen, nEmbd]
        int batchSize = offsetTokenIds.getValue().getShape().getDimension(0);
        int nEmbd = config.getNEmbd();
        Variable flatTokenIds = offsetTokenIds.reshape(Shape.of(batchSize * validLen));
        Variable flatEmbeddings = embedParam.indexSelect(0, flatTokenIds);
        Variable offsetEmbeddings = flatEmbeddings.reshape(Shape.of(batchSize, validLen, nEmbd));

        // 3. 拼接 [h_i; embed_{i+offset}]，形状 [batch, validLen, 2*nEmbd]（保持计算图）
        Variable concatenated = Variable.cat(new Variable[]{truncatedHidden, offsetEmbeddings}, 2);

        // 4. 投影 → RMSNorm → 共享 lm_head
        Variable projected = projectionLayers.get(depth).forward(concatenated);
        Variable normalized = normLayers.get(depth).forward(projected);
        return sharedOutputProjection.forward(normalized);
    }

    /**
     * 计算 MTP 总损失
     * 
     * 与 DeepSeekV3Pretrainer.reshapeForLoss 保持一致的格式：
     * - logits reshape 为 [N, vocabSize]
     * - targets reshape 为 [N, 1]
     * 
     * @param hiddenStates Transformer 最后一层的隐藏状态 [batch, seqLen, nEmbd]
     * @param tokenIds     输入 token ID 序列 [batch, seqLen]
     * @param targetIds    目标 token ID 序列 [batch, seqLen]（即 tokenIds 左移一位）
     * @param lossFunction 损失函数
     * @return MTP 总损失（各深度 loss 的均值 × mtpLossWeight）
     */
    public Variable computeMTPLoss(Variable hiddenStates, Variable tokenIds,
                                   NdArray targetIds, SoftmaxCrossEntropy lossFunction) {
        if (mtpDepth <= 0) {
            return new Variable(NdArray.of(new float[]{0.0f}));
        }

        int batchSize = targetIds.getShape().getDimension(0);
        int seqLen = targetIds.getShape().getDimension(1);
        Variable totalLoss = null;

        for (int depth = 0; depth < mtpDepth; depth++) {
            int offset = depth + 1;
            int validLen = seqLen - offset;
            if (validLen <= 0) {
                continue;
            }

            // 计算该深度的 logits
            Variable logits = computeMTPLogits(hiddenStates, tokenIds, depth);

            // 构造该深度的目标：target[offset..seqLen-1]
            float[][] depthTargets = new float[batchSize][validLen];
            for (int b = 0; b < batchSize; b++) {
                for (int t = 0; t < validLen; t++) {
                    depthTargets[b][t] = targetIds.get(b, t + offset);
                }
            }

            // reshape 与 Pretrain.reshapeForLoss 保持一致：
            // logits: [batch, validLen, vocabSize] → [batch*validLen, vocabSize]
            // targets: [batch, validLen] → [batch*validLen, 1]
            int totalTokens = batchSize * validLen;
            Variable reshapedLogits = logits.reshape(
                Shape.of(totalTokens, config.getVocabSize()));
            Variable reshapedTargets = new Variable(
                NdArray.of(depthTargets)).reshape(Shape.of(totalTokens, 1));

            Variable depthLoss = lossFunction.loss(reshapedTargets, reshapedLogits);

            if (totalLoss == null) {
                totalLoss = depthLoss;
            } else {
                totalLoss = totalLoss.add(depthLoss);
            }
        }

        if (totalLoss == null) {
            return new Variable(NdArray.of(new float[]{0.0f}));
        }

        // 各深度 loss 的均值 × mtpLossWeight
        float weight = (float) (config.getMtpLossWeight() / mtpDepth);
        return totalLoss.mul(new Variable(NdArray.of(new float[]{weight})));
    }

    /**
     * 获取 MTP 深度
     */
    public int getMtpDepth() {
        return mtpDepth;
    }
}
