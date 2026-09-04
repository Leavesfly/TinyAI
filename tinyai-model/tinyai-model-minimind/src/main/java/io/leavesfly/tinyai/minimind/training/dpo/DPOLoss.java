package io.leavesfly.tinyai.minimind.training.dpo;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.minimind.training.BaseTrainer;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

/**
 * DPO (Direct Preference Optimization) 损失函数
 * 
 * DPO损失公式:
 * L_DPO = -log(σ(β * log(π_θ(y_w|x)/π_ref(y_w|x)) - β * log(π_θ(y_l|x)/π_ref(y_l|x))))
 * 
 * 其中:
 * - π_θ: 策略模型(被训练的模型)
 * - π_ref: 参考模型(冻结的模型)
 * - y_w: 更好的响应(chosen/winner)
 * - y_l: 较差的响应(rejected/loser)
 * - β: KL散度惩罚系数
 * - σ: Sigmoid函数
 * 
 * DPO直接优化偏好,无需奖励模型
 * <p>
 * 归约口径：先按"每条序列"计算对数概率与损失，再对 batch 求平均。
 * 不可把整个 batch 的 token 混在一起求平均——那样长短序列的权重会被混淆，
 * 且一个 batch 只能产出一个偏好比较结果（准确率退化成 1 bit）。
 * 
 * @author leavesfly
 * @since 2024
 */
public class DPOLoss {
    
    /**
     * softplus 输入的裁剪范围
     * <p>
     * {@code softplus(z) = log(1 + exp(z))} 在 z 较大时 exp 会溢出成 Inf，
     * 策略一旦跑飞损失立刻变成 NaN。裁剪到 ±30 后 exp(30)≈1e13 仍在 float 范围内，
     * 且 Clip 的反向传播对越界位置返回 0 梯度，等价于对极端样本停止更新。
     */
    private static final float LOGIT_CLIP = 30.0f;
    
    /**
     * 标签忽略位标记（对齐 SoftmaxCE 的 ignore_index 约定）
     */
    private static final int IGNORE_INDEX = -100;
    
    private final float beta;
    private final float labelSmoothing;
    private final boolean lengthNormalization;
    
    /**
     * 构造函数（默认做长度归一化，即使用每 token 平均对数概率）
     * 
     * @param beta KL散度惩罚系数
     * @param labelSmoothing 标签平滑系数
     */
    public DPOLoss(float beta, float labelSmoothing) {
        this(beta, labelSmoothing, true);
    }
    
    /**
     * 构造函数
     * 
     * @param beta KL散度惩罚系数
     * @param labelSmoothing 标签平滑系数
     * @param lengthNormalization true 表示使用每 token 平均对数概率；
     *                            false 表示使用序列对数概率之和（长回答权重更大）
     */
    public DPOLoss(float beta, float labelSmoothing, boolean lengthNormalization) {
        this.beta = beta;
        this.labelSmoothing = labelSmoothing;
        this.lengthNormalization = lengthNormalization;
    }
    
    /**
     * 计算DPO损失
     * 
     * @param chosenLogProbs 策略模型在chosen响应上的对数概率 [batchSize] (在计算图中)
     * @param rejectedLogProbs 策略模型在rejected响应上的对数概率 [batchSize] (在计算图中)
     * @param refChosenLogProbs 参考模型在chosen响应上的对数概率 [batchSize] (常量)
     * @param refRejectedLogProbs 参考模型在rejected响应上的对数概率 [batchSize] (常量)
     * @return DPO损失（标量，batch 平均）
     */
    public Variable loss(Variable chosenLogProbs, Variable rejectedLogProbs,
                        float[] refChosenLogProbs, float[] refRejectedLogProbs) {
        
        int batchSize = chosenLogProbs.getValue().getShape().getDimension(0);
        
        // 计算策略模型的log ratio (在计算图中，可反向传播)
        // log(π_θ(y_w|x)) - log(π_θ(y_l|x))
        Variable policyLogRatio = chosenLogProbs.sub(rejectedLogProbs);
        
        // 参考模型的log ratio (常量，不参与梯度)
        float[] refLogRatio = new float[batchSize];
        for (int i = 0; i < batchSize; i++) {
            refLogRatio[i] = refChosenLogProbs[i] - refRejectedLogProbs[i];
        }
        
        // 隐式奖励: policy_log_ratio - ref_log_ratio，再乘以 β
        Variable implicitReward = policyLogRatio.sub(constant(refLogRatio));
        Variable scaledReward = implicitReward.mul(constant(fill(batchSize, beta)));
        
        // 计算sigmoid损失: -log(σ(x)) = softplus(-x)
        Variable negScaledReward = scaledReward.mul(constant(fill(batchSize, -1.0f)));
        Variable perSampleLoss = softplus(negScaledReward.clip(-LOGIT_CLIP, LOGIT_CLIP));
        
        // 应用标签平滑（标准 cDPO 形式）:
        // loss = (1-ε) * (-log σ(x)) + ε * (-log σ(-x))
        // 旧实现用 -ε/2 * (logπ_w + logπ_l) 作为正则项，那是无条件的 NLL 惩罚，
        // 会同时压低 chosen 与 rejected 的概率，与标签平滑的语义不符
        if (labelSmoothing > 0) {
            Variable smoothedBranch = softplus(scaledReward.clip(-LOGIT_CLIP, LOGIT_CLIP));
            perSampleLoss = perSampleLoss.mul(constant(fill(batchSize, 1.0f - labelSmoothing)))
                .add(smoothedBranch.mul(constant(fill(batchSize, labelSmoothing))));
        }
        
        // 对 batch 求平均，得到标量损失
        // 不能用 mean(axis, keepdims)：该实现会把结果广播回原形状，
        // 导致损失不是标量、backward 时按元素求和而放大梯度
        return BaseTrainer.toScalar(perSampleLoss.sum().mul(constant(1.0f / batchSize)));
    }
    
    /**
     * Softplus函数: softplus(x) = log(1 + exp(x))
     * <p>
     * 调用方需先用 {@link #LOGIT_CLIP} 裁剪输入，否则 exp 会溢出。
     * 
     * @param x 输入
     * @return softplus(x)
     */
    private Variable softplus(Variable x) {
        Variable expX = x.exp();
        Variable onePlusExp = expX.add(constant(fill(
            expX.getValue().getShape().getDimension(0), 1.0f)));
        return onePlusExp.log();
    }
    
    /**
     * 计算每条序列的对数概率 (保持计算图，用于策略模型)
     * <p>
     * 实现方式：逐样本切出 logits 后交给 {@code softmaxCrossEntropy}。
     * 该算子内部完成 logSoftmax → gather(label) → 对有效 token 求平均 NLL，
     * 并且天然支持 ignore_index(-100)：前向跳过、反向该行梯度置零。
     * <p>
     * 两个刻意的取舍：
     * 1. 用 {@code sliceRange} 而不是 {@code split} 切片。split 是多输出函数，
     *    当多个分片汇入同一个损失时，先被访问的分片梯度会被后一个分片的
     *    multi-output backward 重复计入（其 .grad 未被清理），造成梯度翻倍。
     * 2. 不再构造 [totalTokens, vocabSize] 的 one-hot 稠密矩阵来做 gather。
     *    那会带来 tokens*vocab*4B 的额外内存（大词表下轻易上百 MB）。
     * 
     * @param logits 模型输出logits [batch, seq_len, vocab_size]
     * @param labels 标签 Variable [batch, seq_len]
     * @param mask 掩码 Variable [batch, seq_len], 1表示计算(response)，0表示忽略(prompt)
     * @return 每条序列的对数概率 [batch]（保持计算图）
     */
    public Variable computeLogProbs(Variable logits, Variable labels, Variable mask) {
        int[] logitsShape = logits.getValue().getShape().getShapeDims();
        int batchSize = logitsShape[0];
        int seqLen = logitsShape[1];
        int vocabSize = logitsShape[2];
        
        float[] labelsData = labels.getValue().getArray();
        float[] maskData = mask.getValue().getArray();
        
        Variable[] perSampleLogProbs = new Variable[batchSize];
        
        for (int b = 0; b < batchSize; b++) {
            // 构造该样本的标签列：被掩码或越界的位置置为 ignore_index
            float[] labelColumn = new float[seqLen];
            int validTokenCount = 0;
            for (int s = 0; s < seqLen; s++) {
                int flatIdx = b * seqLen + s;
                int labelIdx = (int) labelsData[flatIdx];
                if (maskData[flatIdx] > 0.5f && labelIdx >= 0 && labelIdx < vocabSize) {
                    labelColumn[s] = labelIdx;
                    validTokenCount++;
                } else {
                    labelColumn[s] = IGNORE_INDEX;
                }
            }
            
            Variable sampleLogits = logits.sliceRange(0, b, b + 1)
                .reshape(Shape.of(seqLen, vocabSize));
            Variable sampleLabels = new Variable(
                NdArray.of(labelColumn, Shape.of(seqLen, 1)));
            sampleLabels.setRequireGrad(false);
            
            // softmaxCrossEntropy 返回有效 token 的平均 NLL，取负即为平均对数概率。
            // 该算子的标量输出形状是 [1,1]，必须 reshape 成 [1] 后再 cat：
            // 否则 cat(dim=0) 得到 [batch, 1]，与形状 [batch] 的参考 log ratio 常量
            // 相减时会因广播不兼容直接报错。
            Variable meanLogProb = sampleLogits.softmaxCrossEntropy(sampleLabels).neg()
                .reshape(Shape.of(1));
            
            if (lengthNormalization || validTokenCount == 0) {
                perSampleLogProbs[b] = meanLogProb;
            } else {
                // 不做长度归一化：还原为整条序列的对数概率之和
                perSampleLogProbs[b] = meanLogProb.mul(constant(validTokenCount));
            }
        }
        
        // 拼接为 [batch]，Concat 的反向传播会把梯度按分片切回各自的位置
        return Variable.cat(perSampleLogProbs, 0);
    }
    
    /**
     * 计算每条序列的对数概率 (不保持计算图，用于参考模型)
     * 
     * 参考模型是冻结的，不需要梯度，直接数值计算更高效。
     * 
     * @param logits 模型输出logits [batch, seq_len, vocab_size]（应已 detach）
     * @param labels 标签 [batch, seq_len]
     * @param mask 掩码 [batch, seq_len], 1表示计算(response)，0表示忽略(prompt)
     * @return 每条序列的对数概率 [batch]
     */
    public float[] computeLogProbsDetached(Variable logits, NdArray labels, NdArray mask) {
        int[] logitsShape = logits.getValue().getShape().getShapeDims();
        int batchSize = logitsShape[0];
        int seqLen = logitsShape[1];
        int vocabSize = logitsShape[2];
        
        float[] logitsData = logits.getValue().getArray();
        float[] labelsData = labels.getArray();
        float[] maskData = mask.getArray();
        
        float[] result = new float[batchSize];
        
        for (int b = 0; b < batchSize; b++) {
            float totalLogProb = 0.0f;
            int validTokens = 0;
            
            for (int s = 0; s < seqLen; s++) {
                int flatIdx = b * seqLen + s;
                if (maskData[flatIdx] <= 0.5f) {
                    continue;
                }
                int labelIdx = (int) labelsData[flatIdx];
                if (labelIdx < 0 || labelIdx >= vocabSize) {
                    continue;
                }
                
                int logitsOffset = flatIdx * vocabSize;
                
                // 数值稳定的 log softmax
                float maxLogit = Float.NEGATIVE_INFINITY;
                for (int v = 0; v < vocabSize; v++) {
                    maxLogit = Math.max(maxLogit, logitsData[logitsOffset + v]);
                }
                
                float sumExp = 0.0f;
                for (int v = 0; v < vocabSize; v++) {
                    sumExp += (float) Math.exp(logitsData[logitsOffset + v] - maxLogit);
                }
                float logSumExp = maxLogit + (float) Math.log(sumExp);
                
                totalLogProb += logitsData[logitsOffset + labelIdx] - logSumExp;
                validTokens++;
            }
            
            if (validTokens == 0) {
                result[b] = 0.0f;
            } else if (lengthNormalization) {
                result[b] = totalLogProb / validTokens;
            } else {
                result[b] = totalLogProb;
            }
        }
        
        return result;
    }
    
    // ==================== 工具方法 ====================
    
    /**
     * 构造不参与梯度的常量 Variable
     * <p>
     * 委派给 {@link BaseTrainer#constant}，与 RL / 蒸馏 / SFT 各处共用同一份实现。
     */
    private static Variable constant(float[] values) {
        return BaseTrainer.constant(values);
    }
    
    private static Variable constant(float value) {
        return BaseTrainer.constant(value);
    }
    
    /**
     * 生成填充了同一值的数组
     */
    private static float[] fill(int length, float value) {
        float[] values = new float[length];
        for (int i = 0; i < length; i++) {
            values[i] = value;
        }
        return values;
    }
    
    public float getBeta() {
        return beta;
    }
    
    public float getLabelSmoothing() {
        return labelSmoothing;
    }
    
    public boolean isLengthNormalization() {
        return lengthNormalization;
    }
}
