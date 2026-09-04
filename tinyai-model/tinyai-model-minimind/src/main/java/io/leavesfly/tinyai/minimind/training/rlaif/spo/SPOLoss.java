package io.leavesfly.tinyai.minimind.training.rlaif.spo;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.minimind.training.BaseTrainer;
import io.leavesfly.tinyai.minimind.training.rlaif.BaseRLTrainer;

/**
 * SPO (Simplified Policy Optimization) 损失函数
 * <p>
 * SPO算法核心思想:
 * 1. 生成K个候选回答
 * 2. 计算每个候选的奖励R(y_i)
 * 3. 计算相对优势: A(y_i) = R(y_i) - mean(R)
 * 4. 策略梯度: ∇L = -∑ A(y_i) * log π_θ(y_i|x)
 * 5. 添加熵正则化鼓励探索
 * <p>
 * 无需Critic网络,直接使用奖励信号优化策略（即带基线的 REINFORCE）。
 * <p>
 * 计算图完整性保证:
 * - 所有可微操作均通过 Variable API 完成,保持计算图连通
 * - 奖励/优势作为外部信号,以 requireGrad=false 的常量参与,不回传梯度
 * - logProb / 熵统一复用 {@link BaseRLTrainer} 的实现,与 GRPO / PPO / Agent RL 口径一致
 * <p>
 * 形状约定（重要）:
 * - logProb 必须是逐样本的 [batch]，不能是整个 batch 归约后的标量。
 *   优势 A[i][k] 是"第 i 个 prompt 的第 k 个候选"的信号，只有与该样本自己的
 *   log π(y_i|x_i) 相乘才有意义；若先把 batch 压成标量再乘以优势向量，
 *   梯度会退化成 (Σ_i A[i][k]) · ∂f(batch级logprob)/∂θ，credit assignment 完全丢失。
 * - 最终损失必须是标量。不能用 mean(axis, keepdims=true)——该实现会把结果广播回
 *   原形状，backward() 对非标量损失用 ones(shape) 初始化梯度，等价于按元素求和，
 *   熵项的等效权重会被放大 batch*seq 倍；日志里的 getNumber() 也只取 buffer[0]。
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
     * 计算单个候选的策略梯度损失（推荐入口）
     * <p>
     * L_k = -mean_i( A[i][k] * log π(y_i^k | x_i) )
     * <p>
     * 逐候选调用 + 逐候选 backward 可以让梯度在参数上自然累积，避免同时持有 K 份
     * [batch, seq, vocab] 的计算图（K 较大时内存会线性膨胀）。调用方需自行按 1/K 加权。
     *
     * @param logits     该候选的模型输出 [batch, seq, vocab]（保持计算图）
     * @param labels     该候选的标签 [batch, seq]，忽略位为负数
     * @param advantages 该候选的逐样本优势 [batch]（常量）
     * @return 标量损失
     */
    public Variable computeCandidateLoss(Variable logits, Variable labels, float[] advantages) {
        int batchSize = advantages.length;

        // 逐样本 logProb [batch]，计算图连通
        Variable logProbs = BaseRLTrainer.perSampleLogProbs(logits, labels);

        // 优势是外部奖励信号，不参与反向传播
        Variable weighted = logProbs.mul(BaseTrainer.constant(advantages));

        // 最大化 A*logπ 等价于最小化 -A*logπ；归约到标量（sum 后乘 -1/batch）
        float scale = batchSize > 0 ? -1.0f / batchSize : 0.0f;
        return BaseTrainer.toScalar(weighted.sum().mul(BaseTrainer.constant(scale)));
    }

    /**
     * 计算标量平均熵（鼓励探索）
     * <p>
     * H = -Σ_v p·log p，先按 token 归约再对全部 token 求平均，返回形状为 [1] 的标量。
     * 委派给 {@link BaseRLTrainer#scalarEntropy(Variable)}。
     *
     * @param logits 模型输出 [batch, seq, vocab]
     * @return 标量熵
     */
    public Variable computeEntropy(Variable logits) {
        return BaseRLTrainer.scalarEntropy(logits);
    }

    /**
     * 计算完整的 SPO 损失（一次性入口）
     * <p>
     * totalLoss = -(1/K) Σ_k mean_i( A[i][k]·logπ_i^k ) - entropyCoef · (1/K) Σ_k H_k
     * <p>
     * 注意：本方法会同时持有 K 份计算图，K 或 batch 较大时内存开销明显。
     * 训练器应优先使用 {@link #computeCandidateLoss} 逐候选累积梯度。
     *
     * @param logits  模型输出数组，每项形状 [batch, seq, vocab]
     * @param labels  标签数组，每项形状 [batch, seq]
     * @param rewards 奖励 [batch, K]（外部信号,不参与反向传播）
     * @return 标量总损失
     */
    public Variable computeLoss(Variable[] logits, Variable[] labels, float[][] rewards) {
        int numCandidates = logits.length;
        if (numCandidates == 0) {
            throw new IllegalArgumentException("SPO 至少需要一个候选回答");
        }
        if (labels.length != numCandidates) {
            throw new IllegalArgumentException(
                "logits 与 labels 数量不一致: " + numCandidates + " vs " + labels.length);
        }

        // 1. 归一化奖励并计算优势（纯数值计算,不进入计算图）
        float[][] advantages = computeAdvantages(normalizeRewards(rewards));

        int batchSize = advantages.length;
        float entropyCoef = config.getEntropyCoef();

        Variable policyLoss = null;
        Variable entropySum = null;

        for (int k = 0; k < numCandidates; k++) {
            float[] advantageColumn = column(advantages, k, batchSize);

            Variable candidateLoss = computeCandidateLoss(logits[k], labels[k], advantageColumn);
            policyLoss = (policyLoss == null) ? candidateLoss : policyLoss.add(candidateLoss);

            Variable entropy = computeEntropy(logits[k]);
            entropySum = (entropySum == null) ? entropy : entropySum.add(entropy);
        }

        // 对候选维度求平均
        Variable invCandidates = BaseTrainer.constant(1.0f / numCandidates);
        Variable avgPolicyLoss = policyLoss.mul(invCandidates);
        Variable avgEntropy = entropySum.mul(invCandidates);

        // 总损失 = 策略损失 - 熵系数 * 熵（最大化熵 → 最小化时取负）
        return BaseTrainer.toScalar(
                avgPolicyLoss.sub(avgEntropy.mul(BaseTrainer.constant(entropyCoef))));
    }

    // ==================== 奖励与优势（纯数值） ====================

    /**
     * 计算归一化后的优势矩阵
     *
     * @param rewards 原始奖励 [batch, K]
     * @return 优势 [batch, K]
     */
    public float[][] computeNormalizedAdvantages(float[][] rewards) {
        return computeAdvantages(normalizeRewards(rewards));
    }

    /**
     * 归一化奖励(纯数值计算,不进入计算图)
     * <p>
     * 先按配置做奖励裁剪，再按配置做归一化。裁剪可以抑制离群奖励把优势尺度整体拉大。
     */
    private float[][] normalizeRewards(float[][] rewards) {
        int batchSize = rewards.length;
        int numCandidates = batchSize > 0 ? rewards[0].length : 0;
        float[][] normalized = new float[batchSize][numCandidates];

        float clipMin = config.getRewardClipMin();
        float clipMax = config.getRewardClipMax();
        boolean clipEnabled = clipMax > clipMin;

        switch (config.getRewardNormalization()) {
            case NONE:
                for (int i = 0; i < batchSize; i++) {
                    for (int k = 0; k < numCandidates; k++) {
                        normalized[i][k] = clip(rewards[i][k], clipMin, clipMax, clipEnabled);
                    }
                }
                break;

            case STANDARDIZE:
                for (int i = 0; i < batchSize; i++) {
                    float mean = computeMean(rewards[i]);
                    float std = computeStd(rewards[i], mean);
                    for (int k = 0; k < numCandidates; k++) {
                        float value = (rewards[i][k] - mean) / std;
                        normalized[i][k] = clip(value, clipMin, clipMax, clipEnabled);
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
                        float value = (rewards[i][k] - min) / range;
                        normalized[i][k] = clip(value, clipMin, clipMax, clipEnabled);
                    }
                }
                break;

            case WHITENING:
                for (int i = 0; i < batchSize; i++) {
                    float mean = computeMean(rewards[i]);
                    float std = computeStd(rewards[i], mean);
                    for (int k = 0; k < numCandidates; k++) {
                        float value = (rewards[i][k] - mean) / std;
                        // 白化额外做 ±3σ 截断，再叠加配置的奖励裁剪
                        value = Math.max(-3.0f, Math.min(3.0f, value));
                        normalized[i][k] = clip(value, clipMin, clipMax, clipEnabled);
                    }
                }
                break;

            default:
                for (int i = 0; i < batchSize; i++) {
                    System.arraycopy(rewards[i], 0, normalized[i], 0,
                        Math.min(numCandidates, rewards[i].length));
                }
                break;
        }

        return normalized;
    }

    /**
     * 计算优势函数: A(y_i) = R(y_i) - mean(R)
     * <p>
     * 基线取"同一 prompt 下 K 个候选的平均奖励"，这正是无 Critic 时方差最小的常见选择。
     */
    private float[][] computeAdvantages(float[][] rewards) {
        int batchSize = rewards.length;
        int numCandidates = batchSize > 0 ? rewards[0].length : 0;
        float[][] advantages = new float[batchSize][numCandidates];

        for (int i = 0; i < batchSize; i++) {
            float meanReward = computeMean(rewards[i]);

            for (int k = 0; k < numCandidates; k++) {
                advantages[i][k] = rewards[i][k] - meanReward;
            }

            if (config.isNormalizeAdvantage() && numCandidates > 1) {
                float variance = 0.0f;
                for (float a : advantages[i]) {
                    variance += a * a;
                }
                float std = (float) Math.sqrt(variance / numCandidates + 1e-8f);

                for (int k = 0; k < numCandidates; k++) {
                    advantages[i][k] /= std;
                }
            }
        }

        return advantages;
    }

    // ==================== 工具方法 ====================

    private static float clip(float value, float min, float max, boolean enabled) {
        return enabled ? Math.max(min, Math.min(max, value)) : value;
    }

    /**
     * 取出 [batchSize][numCandidates] 矩阵的第 k 列
     */
    static float[] column(float[][] matrix, int k, int batchSize) {
        float[] column = new float[batchSize];
        for (int i = 0; i < batchSize; i++) {
            column[i] = (i < matrix.length && k < matrix[i].length) ? matrix[i][k] : 0.0f;
        }
        return column;
    }

    private float computeMean(float[] values) {
        if (values.length == 0) {
            return 0.0f;
        }
        float sum = 0.0f;
        for (float v : values) {
            sum += v;
        }
        return sum / values.length;
    }

    private float computeStd(float[] values, float mean) {
        if (values.length == 0) {
            return 1.0f;
        }
        float variance = 0.0f;
        for (float v : values) {
            variance += (v - mean) * (v - mean);
        }
        return (float) Math.sqrt(variance / values.length + 1e-8f);
    }

    public SPOConfig getConfig() {
        return config;
    }
}
