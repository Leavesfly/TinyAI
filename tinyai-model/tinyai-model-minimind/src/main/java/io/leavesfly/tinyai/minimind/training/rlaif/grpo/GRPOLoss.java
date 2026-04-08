package io.leavesfly.tinyai.minimind.training.rlaif.grpo;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;

/**
 * GRPO (Group Relative Policy Optimization) 损失函数
 * <p>
 * GRPO核心思想:
 * 1. 组相对优势: 将K个候选分组, 计算组内相对优势
 * 2. Clipped Surrogate Objective: 限制策略更新幅度
 * 3. 熵正则化: 鼓励探索
 * <p>
 * 核心公式:
 * 组内相对优势: A_relative(y_i) = R(y_i) - mean_group(R)
 * L^{CLIP} = -min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)
 * <p>
 * 注意: 所有涉及策略概率的计算必须通过 Variable 算子完成,
 * 以保持计算图连通, 确保梯度能正确回传到 actor 参数。
 *
 * @author leavesfly
 * @since 2024
 */
public class GRPOLoss {

    private final GRPOConfig config;

    public GRPOLoss(GRPOConfig config) {
        this.config = config;
    }

    /**
     * 计算单个候选的 Clipped 策略损失（保持计算图连通）
     * <p>
     * 对于单个候选 k, 计算:
     * ratio = exp(newLogProb - oldLogProb)
     * surrogate1 = ratio * advantage
     * surrogate2 = clip(ratio, 1-ε, 1+ε) * advantage
     * loss = -min(surrogate1, surrogate2)
     *
     * @param newLogProb 新策略对数概率（Variable, 保持计算图）
     * @param oldLogProb 旧策略对数概率（float, 已 detach）
     * @param advantage  组相对优势值（float, 常量）
     * @return 该候选的策略损失（Variable, 计算图连通到 actor）
     */
    public Variable computeCandidateLoss(Variable newLogProb, float oldLogProb, float advantage) {
        // ratio = exp(log π_new - log π_old)，计算图连通
        // oldLogProb是旧策略的值,不需要梯度
        Variable oldLogProbVar = new Variable(NdArray.of(oldLogProb));
        oldLogProbVar.setRequireGrad(false);
        Variable logRatio = newLogProb.sub(oldLogProbVar);
        Variable ratio = logRatio.exp();

        // surrogate1 = ratio * advantage，advantage是外部信号常量
        Variable advantageVar = new Variable(NdArray.of(advantage));
        advantageVar.setRequireGrad(false);
        Variable surrogate1 = ratio.mul(advantageVar);

        // surrogate2 = clip(ratio, 1-ε, 1+ε) * advantage，计算图连通
        float clipEpsilon = config.getClipEpsilon();
        Variable clippedRatio = ratio.clip(1.0f - clipEpsilon, 1.0f + clipEpsilon);
        Variable surrogate2 = clippedRatio.mul(advantageVar);

        // min(surrogate1, surrogate2): 通过 Variable.where 实现
        // 当 surrogate1 < surrogate2 时取 surrogate1, 否则取 surrogate2
        Variable condition = surrogate1.lt(surrogate2);
        Variable minSurrogate = Variable.where(condition, surrogate1, surrogate2);

        // 取负值（最大化 surrogate → 最小化 -surrogate）
        return minSurrogate.neg();
    }

    /**
     * 计算组相对优势
     * <p>
     * 对于每组, 计算: A_relative(y_i) = R(y_i) - mean_group(R)
     * 优势值是纯数值计算, 不需要梯度, 使用 float 即可。
     *
     * @param rewards 奖励 [batchSize, numCandidates]
     * @return 组相对优势 [batchSize, numCandidates]
     */
    public float[][] computeGroupRelativeAdvantages(float[][] rewards) {
        int batchSize = rewards.length;
        int numCandidates = rewards[0].length;
        int groupSize = config.getGroupSize();

        float[][] advantages = new float[batchSize][numCandidates];
        float[][] normalizedRewards = normalizeRewards(rewards);

        for (int i = 0; i < batchSize; i++) {
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

                // 组内相对优势
                for (int k = groupStart; k < groupEnd; k++) {
                    advantages[i][k] = normalizedRewards[i][k] - groupMeanReward;
                }
            }

            // 可选: 归一化优势
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
     * 计算熵正则化损失（保持计算图连通）
     *
     * @param logits 模型输出 logits（Variable）
     * @return 熵损失
     */
    public Variable computeEntropyLoss(Variable logits) {
        // 使用框架内置的logSoftmax,在vocab维度(axis=-1)上计算
        Variable logProbs = logits.logSoftmax();
        // softMax结果detach,避免梯度通过两条路径重复回传
        Variable probs = logits.softMax().detach();
        // entropy = -sum(p * log(p)), p已detach,梯度只通过logProbs流回
        Variable entropy = probs.mul(logProbs).neg();
        return entropy.mean(-1, true);
    }

    /**
     * 归一化奖励（纯数值计算, 不涉及计算图）
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
}
