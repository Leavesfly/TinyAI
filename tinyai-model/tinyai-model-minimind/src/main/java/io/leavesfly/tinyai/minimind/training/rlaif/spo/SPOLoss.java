package io.leavesfly.tinyai.minimind.training.rlaif.spo;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;

/**
 * SPO (Simplified Policy Optimization) 损失函数
 * 
 * SPO算法核心思想:
 * 1. 生成K个候选回答
 * 2. 计算每个候选的奖励R(y_i)
 * 3. 计算相对优势: A(y_i) = R(y_i) - mean(R)
 * 4. 策略梯度: ∇L = -∑ A(y_i) * log π_θ(y_i|x)
 * 5. 添加熵正则化鼓励探索
 * 
 * 无需Critic网络,直接使用奖励信号优化策略
 * 
 * 计算图完整性保证:
 * - 所有可微操作均通过Variable API完成,保持计算图连通
 * - 奖励/优势作为外部信号,使用detach()标记为不需要梯度的常量
 * - logSoftmax/softmax使用框架内置实现,避免手写导致的维度错误
 * - 通过softmaxCrossEntropy正确计算token级别的对数概率
 * 
 * @author leavesfly
 * @since 2024
 */
public class SPOLoss {
    
    private final SPOConfig config;
    
    /**
     * 构造函数
     * 
     * @param config SPO配置
     */
    public SPOLoss(SPOConfig config) {
        this.config = config;
    }
    
    /**
     * 计算SPO损失
     * 
     * 计算图流向: logits → softmaxCrossEntropy(labels) → logProb → weighted by advantage → policyLoss
     *            logits → logSoftmax → softMax(detach) → entropy → entropyLoss
     *            totalLoss = policyLoss - entropyCoef * entropyLoss
     * 
     * @param logits 模型输出的logits [K, batch_size, seq_len, vocab_size]
     * @param labels 标签 [K, batch_size, seq_len]
     * @param rewards 奖励 [batch_size, K] (外部信号,不参与反向传播)
     * @return SPO损失
     */
    public Variable computeLoss(Variable[] logits, Variable[] labels, float[][] rewards) {
        int numCandidates = logits.length;
        
        // 1. 归一化奖励(纯数值计算,不进入计算图)
        float[][] normalizedRewards = normalizeRewards(rewards);
        
        // 2. 计算优势函数: A(y_i) = R(y_i) - mean(R)
        float[][] advantages = computeAdvantages(normalizedRewards);
        
        // 3. 计算每个候选的对数概率(通过softmaxCrossEntropy保持计算图连通)
        Variable[] logProbs = new Variable[numCandidates];
        for (int k = 0; k < numCandidates; k++) {
            logProbs[k] = computeLogProb(logits[k], labels[k]);
        }
        
        // 4. 计算策略梯度损失: L = -∑ A(y_i) * log π(y_i|x)
        Variable policyLoss = computePolicyGradientLoss(logProbs, advantages);
        
        // 5. 添加熵正则化(鼓励探索)
        Variable entropyLoss = computeEntropyLoss(logits);
        
        // 6. 总损失 = 策略损失 - 熵系数 * 熵损失
        Variable entropyCoef = new Variable(NdArray.of(config.getEntropyCoef()));
        entropyCoef.setRequireGrad(false);
        Variable totalLoss = policyLoss.sub(entropyLoss.mul(entropyCoef));
        
        return totalLoss;
    }
    
    /**
     * 归一化奖励(纯数值计算,不进入计算图)
     */
    private float[][] normalizeRewards(float[][] rewards) {
        int batchSize = rewards.length;
        int numCandidates = rewards[0].length;
        float[][] normalized = new float[batchSize][numCandidates];
        
        switch (config.getRewardNormalization()) {
            case NONE:
                for (int i = 0; i < batchSize; i++) {
                    System.arraycopy(rewards[i], 0, normalized[i], 0, numCandidates);
                }
                break;
                
            case STANDARDIZE:
                for (int i = 0; i < batchSize; i++) {
                    float mean = computeMean(rewards[i]);
                    float std = computeStd(rewards[i], mean);
                    for (int k = 0; k < numCandidates; k++) {
                        normalized[i][k] = (rewards[i][k] - mean) / std;
                    }
                }
                break;
                
            case NORMALIZE:
                for (int i = 0; i < batchSize; i++) {
                    float min = Float.MAX_VALUE;
                    float max = -Float.MAX_VALUE;
                    for (float r : rewards[i]) {
                        min = Math.min(min, r);
                        max = Math.max(max, r);
                    }
                    float range = max - min + 1e-8f;
                    for (int k = 0; k < numCandidates; k++) {
                        normalized[i][k] = (rewards[i][k] - min) / range;
                    }
                }
                break;
                
            case WHITENING:
                for (int i = 0; i < batchSize; i++) {
                    float mean = computeMean(rewards[i]);
                    float std = computeStd(rewards[i], mean);
                    for (int k = 0; k < numCandidates; k++) {
                        float value = (rewards[i][k] - mean) / std;
                        normalized[i][k] = Math.max(-3.0f, Math.min(3.0f, value));
                    }
                }
                break;
        }
        
        return normalized;
    }
    
    /**
     * 计算优势函数: A(y_i) = R(y_i) - mean(R)
     */
    private float[][] computeAdvantages(float[][] rewards) {
        int batchSize = rewards.length;
        int numCandidates = rewards[0].length;
        float[][] advantages = new float[batchSize][numCandidates];
        
        for (int i = 0; i < batchSize; i++) {
            float meanReward = computeMean(rewards[i]);
            
            for (int k = 0; k < numCandidates; k++) {
                advantages[i][k] = rewards[i][k] - meanReward;
            }
            
            if (config.isNormalizeAdvantage()) {
                float std = 0.0f;
                for (float a : advantages[i]) {
                    std += a * a;
                }
                std = (float) Math.sqrt(std / numCandidates + 1e-8f);
                
                for (int k = 0; k < numCandidates; k++) {
                    advantages[i][k] /= std;
                }
            }
        }
        
        return advantages;
    }
    
    /**
     * 计算对数概率: log π(y|x)
     * 
     * 使用框架内置的softmaxCrossEntropy保持计算图完整连通:
     * logits → softmaxCrossEntropy(labels) → 交叉熵损失(即NLL)
     * 对数概率 = -交叉熵损失
     * 
     * @param logits 模型输出 [batch_size, seq_len, vocab_size]
     * @param labels 目标标签 [batch_size, seq_len]
     * @return 平均对数概率(标量Variable,保持计算图连通)
     */
    private Variable computeLogProb(Variable logits, Variable labels) {
        // softmaxCrossEntropy内部完成: logSoftmax → gather(labels) → 取负均值
        // 返回的是交叉熵损失(正值), 对数概率 = -crossEntropy
        Variable crossEntropy = logits.softmaxCrossEntropy(labels);
        return crossEntropy.neg();
    }
    
    /**
     * 计算策略梯度损失: L = -∑ A(y_i) * log π(y_i|x)
     * 
     * advantage作为外部奖励信号,不需要梯度回传,使用detach()标记。
     * 梯度只通过logProbs流回模型参数。
     */
    private Variable computePolicyGradientLoss(Variable[] logProbs, float[][] advantages) {
        Variable totalLoss = null;
        
        int batchSize = advantages.length;
        int numCandidates = logProbs.length;
        
        for (int k = 0; k < numCandidates; k++) {
            float[] advantageWeights = new float[batchSize];
            for (int i = 0; i < batchSize; i++) {
                advantageWeights[i] = advantages[i][k];
            }
            
            // advantage是外部信号常量,不参与反向传播
            Variable advantageVar = new Variable(NdArray.of(advantageWeights));
            advantageVar.setRequireGrad(false);
            
            Variable weightedLogProb = logProbs[k].mul(advantageVar);
            
            if (totalLoss == null) {
                totalLoss = weightedLogProb;
            } else {
                totalLoss = totalLoss.add(weightedLogProb);
            }
        }
        
        // 取负均值: 最大化 A*logπ 等价于最小化 -A*logπ
        Variable scale = new Variable(NdArray.of(-1.0f / numCandidates));
        scale.setRequireGrad(false);
        return totalLoss.mul(scale);
    }
    
    /**
     * 计算熵正则化损失(鼓励探索)
     * H = -∑ p * log(p)
     * 
     * 使用框架内置的logSoftmax和softMax,保持计算图连通。
     * softMax结果用detach()阻断梯度,避免熵损失对概率分布产生双重梯度。
     * 熵的梯度仅通过logSoftmax路径回传。
     */
    private Variable computeEntropyLoss(Variable[] logits) {
        Variable totalEntropy = null;
        
        for (Variable logit : logits) {
            // 使用框架内置的logSoftmax,在vocab维度(axis=-1)上计算
            Variable logProbs = logit.logSoftmax();
            
            // softMax结果detach,避免梯度通过两条路径重复回传
            Variable probs = logit.softMax().detach();
            
            // 熵: H = -∑ p * log(p), p已detach,梯度只通过logProbs流回
            Variable negEntropy = probs.mul(logProbs);
            Variable entropy = negEntropy.neg();
            
            // 沿vocab维度求均值
            Variable meanEntropy = entropy.mean(-1, true);
            
            if (totalEntropy == null) {
                totalEntropy = meanEntropy;
            } else {
                totalEntropy = totalEntropy.add(meanEntropy);
            }
        }
        
        Variable scale = new Variable(NdArray.of(1.0f / logits.length));
        scale.setRequireGrad(false);
        return totalEntropy.mul(scale);
    }
    
    // ==================== 工具方法 ====================
    
    private float computeMean(float[] values) {
        float sum = 0.0f;
        for (float v : values) {
            sum += v;
        }
        return sum / values.length;
    }
    
    private float computeStd(float[] values, float mean) {
        float variance = 0.0f;
        for (float v : values) {
            variance += (v - mean) * (v - mean);
        }
        return (float) Math.sqrt(variance / values.length + 1e-8f);
    }
}
