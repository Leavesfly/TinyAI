package io.leavesfly.tinyai.minimind.training.dpo;

import io.leavesfly.tinyai.func.Variable;
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
 * 
 * @author leavesfly
 * @since 2024
 */
public class DPOLoss {
    
    private final float beta;
    private final float labelSmoothing;
    
    /**
     * 构造函数
     * 
     * @param beta KL散度惩罚系数
     * @param labelSmoothing 标签平滑系数
     */
    public DPOLoss(float beta, float labelSmoothing) {
        this.beta = beta;
        this.labelSmoothing = labelSmoothing;
    }
    
    /**
     * 计算DPO损失
     * 
     * @param chosenLogProbs 策略模型在chosen响应上的对数概率 (在计算图中)
     * @param rejectedLogProbs 策略模型在rejected响应上的对数概率 (在计算图中)
     * @param refChosenLogProbValue 参考模型在chosen响应上的对数概率 (标量常量)
     * @param refRejectedLogProbValue 参考模型在rejected响应上的对数概率 (标量常量)
     * @return DPO损失
     */
    public Variable loss(Variable chosenLogProbs, Variable rejectedLogProbs,
                        float refChosenLogProbValue, float refRejectedLogProbValue) {
        
        // 计算策略模型的log ratio (在计算图中，可反向传播)
        // log(π_θ(y_w|x)) - log(π_θ(y_l|x))
        Variable policyLogRatio = chosenLogProbs.sub(rejectedLogProbs);
        
        // 参考模型的log ratio (常量，不参与梯度)
        float refLogRatioValue = refChosenLogProbValue - refRejectedLogProbValue;
        
        // 隐式奖励: policy_log_ratio - ref_log_ratio
        Variable implicitReward = policyLogRatio.sub(new Variable(NdArray.of(refLogRatioValue)));
        Variable scaledReward = implicitReward.mul(new Variable(NdArray.of(beta)));
        
        // 计算sigmoid损失: -log(σ(scaled_reward)) = softplus(-scaled_reward)
        Variable negScaledReward = scaledReward.mul(new Variable(NdArray.of(-1.0f)));
        Variable dpoLoss = softplus(negScaledReward);
        
        // 应用标签平滑
        if (labelSmoothing > 0) {
            Variable regularization = chosenLogProbs.add(rejectedLogProbs).mul(
                new Variable(NdArray.of(-labelSmoothing * 0.5f))
            );
            dpoLoss = dpoLoss.add(regularization);
        }
        
        return dpoLoss;
    }
    
    /**
     * Softplus函数: softplus(x) = log(1 + exp(x))
     * 
     * @param x 输入
     * @return softplus(x)
     */
    private Variable softplus(Variable x) {
        Variable expX = x.exp();
        Variable onePlusExp = expX.add(new Variable(NdArray.of(1.0f)));
        return onePlusExp.log();
    }
    
    /**
     * 计算序列的对数概率 (保持计算图，用于策略模型)
     * 
     * 通过 logSoftmax + one-hot gather + mask 实现，保持完整的计算图以支持反向传播。
     * 
     * @param logits 模型输出logits [batch, seq_len, vocab_size]
     * @param labels 标签 [batch, seq_len]
     * @param mask 掩码 [batch, seq_len], 1表示计算(response)，0表示忽略(prompt)
     * @return 序列的平均对数概率 (标量Variable，保持计算图)
     */
    public Variable computeLogProbs(Variable logits, Variable labels, Variable mask) {
        int[] logitsShape = logits.getValue().getShape().getShapeDims();
        int batchSize = logitsShape[0];
        int seqLen = logitsShape[1];
        int vocabSize = logitsShape[2];
        int totalTokens = batchSize * seqLen;
        
        // 1. reshape logits 为 [totalTokens, vocabSize]
        Variable logitsFlat = logits.reshape(Shape.of(totalTokens, vocabSize));
        
        // 2. 计算 logSoftmax (在计算图中，支持反向传播)
        Variable logProbs = logitsFlat.logSoftmax(-1);  // [totalTokens, vocabSize]
        
        // 3. 构造 one-hot 矩阵来 gather 目标 token 的 log prob
        float[] labelsData = labels.getValue().getArray();
        float[] maskData = mask.getValue().getArray();
        float[] oneHotMasked = new float[totalTokens * vocabSize];
        int validTokenCount = 0;
        
        for (int i = 0; i < totalTokens; i++) {
            if (maskData[i] > 0.5f) {
                int labelIdx = (int) labelsData[i];
                if (labelIdx >= 0 && labelIdx < vocabSize) {
                    oneHotMasked[i * vocabSize + labelIdx] = 1.0f;
                    validTokenCount++;
                }
            }
        }
        
        // 4. 用 one-hot 乘以 logProbs，然后 sum 得到每个 token 的 log prob
        // logProbs * oneHot 只保留目标 token 位置的值，其余为 0
        Variable oneHotVar = new Variable(NdArray.of(oneHotMasked, Shape.of(totalTokens, vocabSize)));
        Variable selectedLogProbs = logProbs.mul(oneHotVar);  // [totalTokens, vocabSize]
        
        // 5. 对 vocabSize 维度求和，得到每个 token 的 log prob
        Variable tokenLogProbs = selectedLogProbs.sum();  // 标量：所有 masked token 的 log prob 之和
        
        // 6. 除以有效 token 数得到平均 log prob
        if (validTokenCount > 0) {
            Variable divisor = new Variable(NdArray.of(1.0f / validTokenCount));
            return tokenLogProbs.mul(divisor);
        }
        
        return tokenLogProbs;
    }
    
    /**
     * 计算序列的对数概率 (不保持计算图，用于参考模型)
     * 
     * 参考模型是冻结的，不需要梯度，直接手动计算更高效。
     * 
     * @param logits 模型输出logits [batch, seq_len, vocab_size]
     * @param labels 标签 [batch, seq_len]
     * @param mask 掩码 [batch, seq_len], 1表示计算(response)，0表示忽略(prompt)
     * @return 平均对数概率 (float 标量)
     */
    public float computeLogProbsDetached(Variable logits, Variable labels, Variable mask) {
        int[] logitsShape = logits.getValue().getShape().getShapeDims();
        int batchSize = logitsShape[0];
        int seqLen = logitsShape[1];
        int vocabSize = logitsShape[2];
        
        float[] logitsData = logits.getValue().getArray();
        float[] labelsData = labels.getValue().getArray();
        float[] maskData = mask.getValue().getArray();
        
        float totalLogProb = 0.0f;
        int validTokens = 0;
        
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                int flatIdx = b * seqLen + s;
                
                if (maskData[flatIdx] > 0.5f) {
                    int labelIdx = (int) labelsData[flatIdx];
                    int logitsOffset = (b * seqLen + s) * vocabSize;
                    
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
                    
                    float logProb = logitsData[logitsOffset + labelIdx] - logSumExp;
                    totalLogProb += logProb;
                    validTokens++;
                }
            }
        }
        
        return validTokens > 0 ? totalLogProb / validTokens : 0.0f;
    }
}
