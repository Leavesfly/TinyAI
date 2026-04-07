package io.leavesfly.tinyai.minimind.training.rlaif.grpo;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu;

/**
 * GRPO (Group Relative Policy Optimization) 损失函数
 * <p>
 * GRPO核心思想:
 * 1. 组相对优势:将K个候选分组,计算组内相对优势
 * 2. Clipped Surrogate Objective
 * 3. 可选的组间对比损失
 * <p>
 * 核心公式:
 * 组内相对优势: A_relative(y_i) = R(y_i) - mean_group(R)
 * L^{CLIP} = min(r_t*A_t, clip(r_t, 1-ε, 1+ε)*A_t)
 *
 * @author leavesfly
 * @since 2024
 */
public class GRPOLoss {

    private final GRPOConfig config;

    /**
     * 构造函数
     */
    public GRPOLoss(GRPOConfig config) {
        this.config = config;
    }

    /**
     * 计算GRPO总损失
     *
     * @param newLogProbs 新策略对数概率 [batch_size * K]
     * @param oldLogProbs 旧策略对数概率 [batch_size * K]
     * @param rewards     奖励 [batch_size, K]
     * @param logits      模型输出(用于计算熵)
     * @return 总损失
     */
    public Variable computeTotalLoss(Variable newLogProbs, Variable oldLogProbs, float[][] rewards, Variable logits) {
        // 1. 计算组相对优势
        float[][] groupAdvantages = computeGroupRelativeAdvantages(rewards);

        // 2. 展平优势数组
        float[] flatAdvantages = flattenAdvantages(groupAdvantages);

        // 3. 计算策略损失(Clipped)
        Variable policyLoss = computeClippedPolicyLoss(newLogProbs, oldLogProbs, flatAdvantages);

        // 4. 计算熵损失
        Variable entropyLoss = computeEntropyLoss(logits);

        // 5. 可选:组间对比损失
        Variable contrastLoss = null;
        if (config.isUseGroupContrast()) {
            contrastLoss = computeGroupContrastLoss(rewards, groupAdvantages);
        }

        // 6. 总损失
        Variable totalLoss = policyLoss.sub(
                entropyLoss.mul(new Variable(NdArray.of(config.getEntropyCoef())))
        );

        if (contrastLoss != null) {
            totalLoss = totalLoss.add(contrastLoss.mul(new Variable(NdArray.of(0.1f))));
        }

        return totalLoss;
    }

    /**
     * 计算组相对优势
     * <p>
     * 对于每组,计算: A_relative(y_i) = R(y_i) - mean_group(R)
     */
    private float[][] computeGroupRelativeAdvantages(float[][] rewards) {
        int batchSize = rewards.length;
        int numCandidates = rewards[0].length;
        int groupSize = config.getGroupSize();

        float[][] advantages = new float[batchSize][numCandidates];

        // 归一化奖励(可选)
        float[][] normalizedRewards = normalizeRewards(rewards);

        for (int i = 0; i < batchSize; i++) {
            // 按组处理
            int numGroups = (numCandidates + groupSize - 1) / groupSize;

            for (int g = 0; g < numGroups; g++) {
                int groupStart = g * groupSize;
                int groupEnd = Math.min(groupStart + groupSize, numCandidates);
                int actualGroupSize = groupEnd - groupStart;

                // 计算组内平均奖励
                float groupMeanReward = 0.0f;
                for (int k = groupStart; k < groupEnd; k++) {
                    groupMeanReward += normalizedRewards[i][k];
                }
                groupMeanReward /= actualGroupSize;

                // 计算组内相对优势
                for (int k = groupStart; k < groupEnd; k++) {
                    advantages[i][k] = normalizedRewards[i][k] - groupMeanReward;
                }
            }

            // 可选:归一化优势
            if (config.isNormalizeAdvantage()) {
                float mean = 0.0f;
                for (float a : advantages[i]) {
                    mean += a;
                }
                mean /= numCandidates;

                float std = 0.0f;
                for (float a : advantages[i]) {
                    std += (a - mean) * (a - mean);
                }
                std = (float) Math.sqrt(std / numCandidates + 1e-8f);

                for (int k = 0; k < numCandidates; k++) {
                    advantages[i][k] = (advantages[i][k] - mean) / std;
                }
            }
        }

        return advantages;
    }

    /**
     * 归一化奖励
     */
    private float[][] normalizeRewards(float[][] rewards) {
        int batchSize = rewards.length;
        int numCandidates = rewards[0].length;
        float[][] normalized = new float[batchSize][numCandidates];

        GRPOConfig.RewardNormalization normType = config.getRewardNormalization();

        switch (normType) {
            case NONE:
                for (int i = 0; i < batchSize; i++) {
                    System.arraycopy(rewards[i], 0, normalized[i], 0, numCandidates);
                }
                break;

            case STANDARDIZE:
                for (int i = 0; i < batchSize; i++) {
                    float mean = 0.0f;
                    for (float r : rewards[i]) mean += r;
                    mean /= numCandidates;

                    float std = 0.0f;
                    for (float r : rewards[i]) std += (r - mean) * (r - mean);
                    std = (float) Math.sqrt(std / numCandidates + 1e-8f);

                    for (int k = 0; k < numCandidates; k++) {
                        normalized[i][k] = (rewards[i][k] - mean) / std;
                    }
                }
                break;

            case NORMALIZE:
                for (int i = 0; i < batchSize; i++) {
                    float min = Float.MAX_VALUE;
                    float max = Float.MIN_VALUE;
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
                    float mean = 0.0f;
                    for (float r : rewards[i]) mean += r;
                    mean /= numCandidates;

                    float std = 0.0f;
                    for (float r : rewards[i]) std += (r - mean) * (r - mean);
                    std = (float) Math.sqrt(std / numCandidates + 1e-8f);

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
     * 计算Clipped策略损失
     */
    private Variable computeClippedPolicyLoss(Variable newLogProbs, Variable oldLogProbs,
                                              float[] advantages) {
        // 概率比: r = exp(log π_new - log π_old)
        Variable logRatio = newLogProbs.sub(oldLogProbs);
        Variable ratio = logRatio.exp();

        NdArray ratioData = ratio.getValue();
        float[] ratioBuffer = ((NdArrayCpu) ratioData).buffer;

        float clipEpsilon = config.getClipEpsilon();
        float[] minSurrogate = new float[ratioBuffer.length];

        for (int i = 0; i < ratioBuffer.length; i++) {
            float r = ratioBuffer[i];
            float adv = advantages[i];

            // surrogate1 = r * A
            float surrogate1 = r * adv;

            // surrogate2 = clip(r, 1-ε, 1+ε) * A
            float clippedRatio = Math.max(1.0f - clipEpsilon,
                    Math.min(1.0f + clipEpsilon, r));
            float surrogate2 = clippedRatio * adv;

            minSurrogate[i] = Math.min(surrogate1, surrogate2);
        }

        // 负均值
        float loss = 0.0f;
        for (float s : minSurrogate) loss += s;
        loss = -loss / minSurrogate.length;

        return new Variable(NdArray.of(loss));
    }

    /**
     * 计算熵损失
     */
    private Variable computeEntropyLoss(Variable logits) {
        Variable probs = softmax(logits);
        Variable logProbs = logSoftmax(logits);
        Variable entropy = probs.mul(logProbs).mul(new Variable(NdArray.of(-1.0f)));
        return entropy.mean(0, true);
    }

    /**
     * 计算组间对比损失
     * <p>
     * 鼓励高奖励组的策略概率高于低奖励组
     */
    private Variable computeGroupContrastLoss(float[][] rewards, float[][] advantages) {
        int batchSize = rewards.length;
        int numCandidates = rewards[0].length;
        int groupSize = config.getGroupSize();
        int numGroups = (numCandidates + groupSize - 1) / groupSize;

        if (numGroups < 2) {
            return new Variable(NdArray.of(0.0f));
        }

        // 计算每组的平均奖励
        float[] groupMeanRewards = new float[numGroups];
        for (int i = 0; i < batchSize; i++) {
            for (int g = 0; g < numGroups; g++) {
                int groupStart = g * groupSize;
                int groupEnd = Math.min(groupStart + groupSize, numCandidates);
                int actualGroupSize = groupEnd - groupStart;

                float groupSum = 0.0f;
                for (int k = groupStart; k < groupEnd; k++) {
                    groupSum += rewards[i][k];
                }
                groupMeanRewards[g] += groupSum / actualGroupSize;
            }
        }

        // 归一化
        for (int g = 0; g < numGroups; g++) {
            groupMeanRewards[g] /= batchSize;
        }

        // 计算对比损失(简化实现)
        float contrastLoss = 0.0f;
        for (int g1 = 0; g1 < numGroups; g1++) {
            for (int g2 = g1 + 1; g2 < numGroups; g2++) {
                float diff = groupMeanRewards[g1] - groupMeanRewards[g2];
                contrastLoss += Math.abs(diff);
            }
        }

        return new Variable(NdArray.of(contrastLoss / (numGroups * (numGroups - 1) / 2)));
    }

    /**
     * 展平优势数组
     */
    private float[] flattenAdvantages(float[][] advantages) {
        int batchSize = advantages.length;
        int numCandidates = advantages[0].length;
        float[] flat = new float[batchSize * numCandidates];

        int idx = 0;
        for (int i = 0; i < batchSize; i++) {
            for (int k = 0; k < numCandidates; k++) {
                flat[idx++] = advantages[i][k];
            }
        }

        return flat;
    }

    /**
     * Softmax实现
     */
    private Variable softmax(Variable x) {
        Variable expX = x.exp();
        Variable sumExp = expX.sum();
        return expX.div(sumExp);
    }

    /**
     * Log Softmax实现
     */
    private Variable logSoftmax(Variable x) {
        Variable expX = x.exp();
        Variable sumExp = expX.sum();
        Variable logSumExp = sumExp.log();
        return x.sub(logSumExp);
    }
}
