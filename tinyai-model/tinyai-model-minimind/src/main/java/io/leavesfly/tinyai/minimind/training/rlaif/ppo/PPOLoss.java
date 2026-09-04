package io.leavesfly.tinyai.minimind.training.rlaif.ppo;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.minimind.training.BaseTrainer;
import io.leavesfly.tinyai.minimind.training.rlaif.BaseRLTrainer;
import io.leavesfly.tinyai.ndarr.Shape;

/**
 * PPO (Proximal Policy Optimization) 损失函数
 * <p>
 * PPO核心思想:
 * 1. Clipped Surrogate Objective防止策略更新过大
 * 2. 价值函数损失
 * 3. 熵正则化鼓励探索
 * <p>
 * 核心公式:
 * L^{CLIP}(θ) = E_i[min(r_i(θ)*A_i, clip(r_i(θ), 1-ε, 1+ε)*A_i)]
 * 其中:
 * - r_i(θ) = π_θ(y_i|x_i) / π_θ_old(y_i|x_i) (概率比)
 * - A_i = 优势函数
 * - ε = clip范围
 * <p>
 * 总损失: L_total = L_policy + c1*L_value - c2*L_entropy
 * <p>
 * 关于优势估计的口径（重要）：
 * 本模块的 rollout 是"单步终止"的 bandit 形式——每个候选回答是一次独立的完整采样，
 * 采样结束即拿到终端奖励，候选之间没有时序先后关系。因此优势用
 * {@code A_i = R_i - V_i}（价值基线），价值目标就是奖励本身 {@code return_i = R_i}。
 * <p>
 * 不要在这里使用 GAE：GAE 的 {@code δ_t = r_t + γV(s_{t+1}) - V(s_t)} 递推要求
 * 数组下标 t 是同一条轨迹上相邻的时间步。把"互相独立的候选样本"按列表顺序喂给 GAE，
 * 会把第 k 个候选当成第 k-1 个候选的下一时刻，优势估计在数学上没有意义。
 * 只有当 rollout 真正产出多轮时序轨迹（如 Agent 多轮工具调用按轮次展开）时才应引入 GAE。
 *
 * @author leavesfly
 * @since 2024
 */
public class PPOLoss {

    private final PPOConfig config;

    /**
     * 构造函数
     */
    public PPOLoss(PPOConfig config) {
        this.config = config;
    }

    /**
     * 计算策略损失(Clipped Surrogate Objective)
     * <p>
     * L^{CLIP} = -mean_i[min(r_i*A_i, clip(r_i, 1-ε, 1+ε)*A_i)]
     * <p>
     * 委派给 {@link BaseRLTrainer#clippedSurrogateLoss}，与 GRPO / Agent RL 共用同一份实现。
     *
     * @param newLogProbs 新策略逐样本对数概率 [batchSize]（保持计算图）
     * @param oldLogProbs 旧策略逐样本对数概率 [batchSize]（常量）
     * @param advantages  逐样本优势 [batchSize]（常量）
     * @return 标量策略损失
     */
    public Variable computePolicyLoss(Variable newLogProbs, float[] oldLogProbs,
                                      float[] advantages) {
        return BaseRLTrainer.clippedSurrogateLoss(
                newLogProbs, oldLogProbs, advantages, config.getClipEpsilon());
    }

    /**
     * 计算价值损失
     * <p>
     * L_value = 0.5 * mean_i (V_i - return_i)^2
     * <p>
     * 形状归一：Critic 输出常为 [batchSize, 1]，而 returns 是 [batchSize]。
     * 直接相减会按广播规则得到 [batchSize, batchSize] 的错误结果，因此先把
     * [batchSize, 1] reshape 成 [batchSize]（reshape 保持计算图连通）。
     * <p>
     * 归约到标量：先平方、再 sum、再乘 0.5/batchSize。
     * 不用 mean(axis, keepdims)——该实现会把结果广播回原形状，使损失非标量，
     * backward 时按元素求和而放大梯度。
     *
     * @param values  Critic 输出的逐样本价值 [batchSize] 或 [batchSize, 1]（保持计算图）
     * @param returns 价值目标 [batchSize]（bandit 形式下即奖励，常量）
     * @return 标量价值损失
     */
    public Variable computeValueLoss(Variable values, float[] returns) {
        int batchSize = returns.length;

        int[] dims = values.getValue().getShape().getShapeDims();
        if (dims.length == 2 && dims[1] == 1 && dims[0] == batchSize) {
            values = values.reshape(Shape.of(batchSize));
        } else if (dims.length != 1 || dims[0] != batchSize) {
            throw new IllegalArgumentException(
                "价值损失需要形状 [" + batchSize + "] 或 [" + batchSize + ", 1] 的 value，实际: "
                    + java.util.Arrays.toString(dims));
        }

        Variable diff = values.sub(BaseTrainer.constant(returns));
        float scale = batchSize > 0 ? 0.5f / batchSize : 0.0f;
        return BaseTrainer.toScalar(diff.squ().sum().mul(BaseTrainer.constant(scale)));
    }

    /**
     * 计算 bandit 形式的优势与价值目标
     * <p>
     * A_i = R_i - V_i，return_i = R_i（单步终止，无折扣、无 bootstrap）
     *
     * @param rewards 奖励 [batchSize, numCandidates]
     * @param values  Critic 价值估计 [batchSize, numCandidates]
     * @return [advantages, returns]，两者形状均为 [batchSize, numCandidates]
     */
    public float[][][] computeBanditAdvantages(float[][] rewards, float[][] values) {
        int batchSize = rewards.length;
        int numCandidates = batchSize > 0 ? rewards[0].length : 0;

        float[][] advantages = new float[batchSize][numCandidates];
        float[][] returns = new float[batchSize][numCandidates];

        for (int i = 0; i < batchSize; i++) {
            for (int k = 0; k < numCandidates; k++) {
                float reward = rewards[i][k];
                float value = (i < values.length && k < values[i].length) ? values[i][k] : 0.0f;
                advantages[i][k] = reward - value;
                returns[i][k] = reward;
            }
        }

        if (config.isNormalizeAdvantage()) {
            // 全 batch 归一化（候选之间相互独立，不存在"组内"概念）
            float mean = 0.0f;
            int count = 0;
            for (int i = 0; i < batchSize; i++) {
                for (int k = 0; k < numCandidates; k++) {
                    mean += advantages[i][k];
                    count++;
                }
            }
            mean = count > 0 ? mean / count : 0.0f;

            float variance = 0.0f;
            for (int i = 0; i < batchSize; i++) {
                for (int k = 0; k < numCandidates; k++) {
                    variance += (advantages[i][k] - mean) * (advantages[i][k] - mean);
                }
            }
            float std = count > 0 ? (float) Math.sqrt(variance / count + 1e-8f) : 1.0f;

            for (int i = 0; i < batchSize; i++) {
                for (int k = 0; k < numCandidates; k++) {
                    advantages[i][k] = (advantages[i][k] - mean) / std;
                }
            }
        }

        return new float[][][]{advantages, returns};
    }

    public PPOConfig getConfig() {
        return config;
    }
}
