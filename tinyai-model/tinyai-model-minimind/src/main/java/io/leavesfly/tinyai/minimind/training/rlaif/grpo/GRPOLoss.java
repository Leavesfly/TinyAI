package io.leavesfly.tinyai.minimind.training.rlaif.grpo;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.minimind.training.rlaif.BaseRLTrainer;

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
 * <p>
 * 形状约定: 优势与 logProb 均以 [batchSize] 为单位逐样本计算。
 * 绝不能把整个 batch 的 logProb 归约成标量后再乘各样本的优势——那样梯度会退化成
 * (Σ_i A_i)·∂f(batch级logprob)/∂θ，per-sample 的 credit assignment 完全丢失。
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
     * 计算单个候选的 Clipped 策略损失（保持计算图连通，逐样本）
     * <p>
     * 对于单个候选 k, 对 batch 内每个样本 i 计算:
     * ratio_i = exp(newLogProb_i - oldLogProb_i)
     * surrogate1_i = ratio_i * advantage_i
     * surrogate2_i = clip(ratio_i, 1-ε, 1+ε) * advantage_i
     * loss = -mean_i(min(surrogate1_i, surrogate2_i))
     *
     * @param newLogProbs 新策略逐样本对数概率 [batchSize]（Variable, 保持计算图）
     * @param oldLogProbs 旧策略逐样本对数概率 [batchSize]（float, 已 detach）
     * @param advantages  该候选的组相对优势 [batchSize]（float, 常量）
     * @return 该候选的策略损失（标量 Variable, 计算图连通到 actor）
     */
    public Variable computeCandidateLoss(Variable newLogProbs, float[] oldLogProbs,
                                         float[] advantages) {
        // 委派给 BaseRLTrainer 的权威实现，GRPO / PPO / Agent RL 共用同一份
        // clipped surrogate 逻辑，避免多处实现不一致（例如漏掉 clip、归约成非标量）
        return BaseRLTrainer.clippedSurrogateLoss(
                newLogProbs, oldLogProbs, advantages, config.getClipEpsilon());
    }

    /**
     * 计算优势
     * <p>
     * useGroupContrast = true（默认，GRPO）:
     * 对每个 prompt 的 K 个候选按 groupSize 分组，A[i][k] = R̃[i][k] - mean_group(R̃[i])
     * <p>
     * useGroupContrast = false（REINFORCE + 全局基线）:
     * A[i][k] = R̃[i][k] - mean(全 batch 所有候选的 R̃)
     *
     * @param rewards 奖励 [batchSize, numCandidates]
     * @return 优势 [batchSize, numCandidates]
     */
    public float[][] computeGroupRelativeAdvantages(float[][] rewards) {
        int batchSize = rewards.length;
        if (batchSize == 0) {
            return new float[0][0];
        }
        int numCandidates = rewards[0].length;

        float[][] advantages = new float[batchSize][numCandidates];
        float[][] normalizedRewards = normalizeRewards(rewards);

        if (config.isUseGroupContrast()) {
            int groupSize = Math.max(1, config.getGroupSize());
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
            }
        } else {
            // 全局基线：所有样本所有候选的均值
            float globalMean = 0.0f;
            int total = 0;
            for (int i = 0; i < batchSize; i++) {
                for (int k = 0; k < numCandidates; k++) {
                    globalMean += normalizedRewards[i][k];
                    total++;
                }
            }
            globalMean = total > 0 ? globalMean / total : 0.0f;
            for (int i = 0; i < batchSize; i++) {
                for (int k = 0; k < numCandidates; k++) {
                    advantages[i][k] = normalizedRewards[i][k] - globalMean;
                }
            }
        }

        // 可选: 逐样本归一化优势
        if (config.isNormalizeAdvantage()) {
            for (int i = 0; i < batchSize; i++) {
                float mean = 0.0f;
                for (float a : advantages[i]) {
                    mean += a;
                }
                mean /= numCandidates;

                float variance = 0.0f;
                for (float a : advantages[i]) {
                    variance += (a - mean) * (a - mean);
                }
                float std = (float) Math.sqrt(variance / numCandidates + 1e-8f);

                for (int k = 0; k < numCandidates; k++) {
                    advantages[i][k] = (advantages[i][k] - mean) / std;
                }
            }
        }

        return advantages;
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
