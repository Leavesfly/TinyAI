package io.leavesfly.tinyai.minimind.training.rlaif.ppo;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.minimind.model.MiniMindBlock;
import io.leavesfly.tinyai.minimind.model.MiniMindModel;
import io.leavesfly.tinyai.minimind.training.dataset.RLAIFDataset;
import io.leavesfly.tinyai.minimind.training.rlaif.BaseRLTrainer;
import io.leavesfly.tinyai.ml.model.Model;
import io.leavesfly.tinyai.ml.optimize.Adam;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * PPO (Proximal Policy Optimization) 训练器
 * <p>
 * Actor-Critic 架构:
 * - Actor: 策略网络 (MiniMindModel)
 * - Critic: 价值网络 (ValueNetwork)，独立的参数与独立的优化器
 * <p>
 * 训练流程:
 * 1. 采集经验: 用旧策略对 K 个候选各做一次前向，记录逐样本 logProb 与价值估计（均为 detach 的数值）
 * 2. 计算优势与价值目标: bandit 形式 A_i = R_i - V_i, return_i = R_i
 * 3. 多轮 PPO 更新: 复用同一批 rollout，用新策略重新前向并构建有梯度的计算图
 * 4. 分别更新 Actor 与 Critic
 * <p>
 * 计算图连通性保证（Actor 侧）:
 * {@code forwardWithAux(input) → computePerSampleLogProbs(logits, label) → computePolicyLoss → backward}
 * 整条链路全部通过 Variable 算子连接，梯度可以正确回传到 Actor 参数。
 * <p>
 * Critic 与 Actor 的解耦（刻意设计）:
 * Critic 的输入是从 Actor 隐藏状态里"取值后重建"的常量（requireGrad=false），因此价值损失
 * 只更新 Critic 参数，不会回传进 Actor。若让 value loss 流入 Actor，Actor 会被优化成
 * "产出容易被价值头预测的隐藏状态"，这与策略梯度目标冲突；标准做法是共享主干时才让
 * 价值头参与主干更新，且此时应使用同一个优化器与同一份裁剪范数。本实现是独立 Critic，故 detach。
 * <p>
 * 内存与正确性策略: 每个候选单独 backward（梯度在各自参数上自然累积），全部候选完成后
 * 再统一裁剪与更新。这样既避免同时持有 K 份计算图，也让 K 个候选的梯度按 1/K 等权合并。
 * <p>
 * 训练循环、日志、检查点、NaN 保护、结束后复位 eval 模式均复用基类实现，本类不再重复声明
 * maxEpochs/logInterval/currentEpoch/currentStep 等状态字段（重复声明会遮蔽基类字段，
 * 导致基类的检查点保存与日志读到永远为 0/空的另一份状态）。
 * <p>
 * 检查点: {@link #saveCheckpoint()} 在保存 Actor 的同时，会把 Critic 状态一并落盘为
 * {@code <prefix>_critic_epoch<E>_step<S>.state}；断点续训时用 {@link #loadCriticState(String)}
 * 恢复 Critic，与 Actor 检查点配套使用，避免续训时价值网络被重置。
 * <p>
 * 优势语义（刻意设计）: 本实现是 contextual-bandit 形式的 PPO —— 优势 {@code A_i = R_i - V_i}、
 * 回报 {@code return_i = R_i}，不含 GAE / 多步时序信用分配。适用于“整段回答给一个标量奖励”的
 * 候选排序场景；若需要序列决策的逐步信用分配，需另行引入 GAE。
 *
 * @author leavesfly
 * @since 2024
 */
public class PPOTrainer extends BaseRLTrainer {

    private final MiniMindModel actor;           // Actor策略网络
    private final ValueNetwork critic;           // Critic价值网络
    private final RLAIFDataset dataset;
    private final PPOConfig config;
    private final PPOLoss ppoLoss;

    private final Adam actorOptimizer;
    private final Adam criticOptimizer;

    private final List<Float> policyLossHistory;
    private final List<Float> valueLossHistory;
    private final List<Float> rewardHistory;

    /**
     * 构造函数
     *
     * @param actor   策略网络
     * @param critic  价值网络（隐藏层维度应与模型 hiddenSize 一致）
     * @param dataset RLAIF 数据集
     * @param config  PPO 配置
     */
    public PPOTrainer(MiniMindModel actor, ValueNetwork critic,
                      RLAIFDataset dataset, PPOConfig config) {
        super(actor);
        this.actor = actor;
        this.critic = critic;
        this.dataset = dataset;
        this.config = config;
        this.ppoLoss = new PPOLoss(config);

        // 创建优化器
        this.actorOptimizer = new Adam(actor, config.getActorLearningRate(),
                0.9f, 0.999f, 1e-8f);
        // ValueNetwork 是 Module 而非 Model，包装后再交给 Adam
        Model criticModel = new Model("critic", critic);
        this.criticOptimizer = new Adam(criticModel, config.getCriticLearningRate(),
                0.9f, 0.999f, 1e-8f);

        // 训练配置写入基类字段，保证基类的训练循环/日志/检查点逻辑读到同一份状态
        this.maxEpochs = 1;
        this.logInterval = 10;
        this.saveInterval = 500;
        this.maxGradNorm = config.getMaxGradNorm();
        this.checkpointDir = "./checkpoints/minimind/ppo";

        this.policyLossHistory = new ArrayList<>();
        this.valueLossHistory = new ArrayList<>();
        this.rewardHistory = new ArrayList<>();
    }

    /**
     * 配置训练
     */
    public PPOTrainer configure(int maxEpochs, int logInterval) {
        this.maxEpochs = maxEpochs;
        this.logInterval = logInterval;
        return this;
    }

    // ==================== 核心训练逻辑 ====================

    /**
     * 训练一步：采集 rollout，再执行 ppoEpochs 轮更新
     */
    @Override
    protected float trainStep(Object batch) {
        RLAIFDataset.Batch rlBatch = (RLAIFDataset.Batch) batch;

        // 1. 用旧策略 + 旧 Critic 采集经验，并计算优势与价值目标
        Rollout rollout = collectRollout(rlBatch);

        // 记录平均奖励，便于观察 RL 是否真的在提升
        rewardHistory.add(averageReward(rollout.rewards));

        // 2. 多轮 PPO 更新（复用同一批 rollout，每轮各自完成一次参数更新）
        int innerEpochs = Math.max(1, config.getPpoEpochs());
        float totalLoss = 0.0f;
        for (int epoch = 0; epoch < innerEpochs; epoch++) {
            totalLoss += ppoUpdate(rollout);
        }
        return totalLoss / innerEpochs;
    }

    /**
     * 采集一次 rollout（旧策略 + 旧 Critic，全部 detach）
     * <p>
     * 每个候选只做一次前向，得到该候选下 batch 内每个样本的 logProb 与 value。
     * 必须保留 batch 维：优势是 per-sample 信号，与该样本自己的 logProb 相乘才有意义；
     * 若归约成标量，整个 batch 会退化成"一个样本"，credit assignment 完全丢失。
     */
    private Rollout collectRollout(RLAIFDataset.Batch batch) {
        // rollout 阶段关闭 dropout，保证旧策略 logProb 是可复现的参考值
        actor.setTraining(false);

        int numCandidates = batch.getNumCandidates();
        int batchSize = batch.getBatchSize();
        NdArray[] candidateInputs = batch.getCandidateInputs();
        NdArray[] candidateLabels = batch.getCandidateLabels();
        float[][] rewards = batch.getRewards();

        float[][] oldLogProbs = new float[batchSize][numCandidates];
        float[][] oldValues = new float[batchSize][numCandidates];

        for (int k = 0; k < numCandidates; k++) {
            Variable inputVar = new Variable(candidateInputs[k]);
            Variable labelVar = new Variable(candidateLabels[k]);
            labelVar.setRequireGrad(false);

            // 同时取出 logits 与 lm_head 之前的隐藏状态（Critic 的输入）
            MiniMindBlock.MoEOutput output = forwardWithAux(inputVar);
            Variable logits = output.getOutput();

            // 旧策略 logProb：detach 后计算，只保留数值
            Variable detachedLogits = logits.detach();
            Variable logProbVar = computePerSampleLogProbs(detachedLogits, labelVar);
            float[] perSampleLogProb = logProbVar.getValue().getArray().clone();
            logProbVar.unChainBackward();

            // 旧 Critic 价值：输入是重建出来的常量，天然与 Actor 计算图断开
            Variable criticInput = buildCriticInput(output.getHiddenState(), candidateLabels[k]);
            Variable valueVar = critic.forward(criticInput);
            float[] perSampleValue = valueVar.getValue().getArray().clone();
            valueVar.unChainBackward();

            for (int i = 0; i < batchSize; i++) {
                if (i < perSampleLogProb.length) {
                    oldLogProbs[i][k] = perSampleLogProb[i];
                }
                if (i < perSampleValue.length) {
                    oldValues[i][k] = perSampleValue[i];
                }
            }

            // 释放本次前向的计算图
            logits.unChainBackward();
        }

        // bandit 形式的优势与价值目标（纯数值，不参与梯度）
        float[][][] advantageAndReturn = ppoLoss.computeBanditAdvantages(rewards, oldValues);

        return new Rollout(candidateInputs, candidateLabels, rewards,
                oldLogProbs, oldValues, advantageAndReturn[0], advantageAndReturn[1],
                batchSize, numCandidates);
    }

    /**
     * 一轮 PPO 更新（计算图连通版本）
     * <p>
     * 关键: 新策略的 logProb 必须保持为 Variable 且保留 batch 维，不能提取为 float，
     * 否则计算图断裂，梯度无法回传到 Actor 参数。
     *
     * @return 本轮的策略损失 + 价值损失
     */
    private float ppoUpdate(Rollout rollout) {
        actor.setTraining(true);
        actor.clearGrads();
        critic.clearGrads();

        int numCandidates = rollout.numCandidates;
        int batchSize = rollout.batchSize;
        float entropyCoef = config.getEntropyCoef();
        float valueLossCoef = config.getValueLossCoef();

        float policyLossSum = 0.0f;
        float valueLossSum = 0.0f;
        int validCandidates = 0;

        for (int k = 0; k < numCandidates; k++) {
            Variable inputVar = new Variable(rollout.candidateInputs[k]);
            Variable labelVar = new Variable(rollout.candidateLabels[k]);
            labelVar.setRequireGrad(false);

            // ===== Actor: 新策略前向（同时取出 MoE 负载均衡损失与隐藏状态）=====
            MiniMindBlock.MoEOutput output = forwardWithAux(inputVar);
            Variable logits = output.getOutput();

            // 逐样本 logProb [batch]，计算图连通
            Variable newLogProbs = computePerSampleLogProbs(logits, labelVar);

            // clipped surrogate（与 GRPO / Agent RL 共用同一份权威实现）
            Variable policyLoss = ppoLoss.computePolicyLoss(newLogProbs,
                    column(rollout.oldLogProbs, k, batchSize),
                    column(rollout.advantages, k, batchSize));

            // 熵正则化（鼓励探索）：标量平均熵，避免损失退化为非标量张量
            Variable entropy = computeScalarEntropy(logits);
            Variable actorLoss = policyLoss.sub(entropy.mul(constant(entropyCoef)));

            // 并入 MoE 辅助损失，并按候选数等权平均
            actorLoss = withMoeAuxLoss(actorLoss, output).mul(constant(1.0f / numCandidates));

            float policyLossValue = actorLoss.getValue().getNumber().floatValue();
            if (!Float.isFinite(policyLossValue)) {
                System.err.printf("警告: PPO 候选 %d 策略损失异常(%s)，已跳过%n",
                        k, Float.isNaN(policyLossValue) ? "NaN" : "Inf");
                actorLoss.unChainBackward();
                continue;
            }

            // 逐候选反向传播：梯度在 Actor 参数上累积，避免同时持有 K 份计算图
            actorLoss.backward();

            // ===== Critic: 输入 detach，价值损失不污染 Actor =====
            Variable criticInput = buildCriticInput(output.getHiddenState(),
                    rollout.candidateLabels[k]);
            Variable newValues = critic.forward(criticInput);
            Variable valueLoss = ppoLoss
                    .computeValueLoss(newValues, column(rollout.returns, k, batchSize))
                    .mul(constant(valueLossCoef / numCandidates));

            float valueLossValue = valueLoss.getValue().getNumber().floatValue();
            if (Float.isFinite(valueLossValue)) {
                valueLoss.backward();
                valueLossSum += valueLossValue;
            } else {
                System.err.printf("警告: PPO 候选 %d 价值损失异常(%s)，已跳过 Critic 累积%n",
                        k, Float.isNaN(valueLossValue) ? "NaN" : "Inf");
            }

            // 释放本候选的计算图
            valueLoss.unChainBackward();
            actorLoss.unChainBackward();

            policyLossSum += policyLossValue;
            validCandidates++;
        }

        // 全部候选累积完成后统一裁剪与更新（Actor / Critic 各自独立裁剪，范数互不干扰）
        clipGradients(actor, config.getMaxGradNorm());
        clipGradients(critic, config.getMaxGradNorm());
        actorOptimizer.update();
        criticOptimizer.update();
        actor.clearGrads();
        critic.clearGrads();

        float avgPolicyLoss = validCandidates > 0 ? policyLossSum / validCandidates : 0.0f;
        float avgValueLoss = validCandidates > 0 ? valueLossSum / validCandidates : 0.0f;
        policyLossHistory.add(avgPolicyLoss);
        valueLossHistory.add(avgValueLoss);

        return avgPolicyLoss + avgValueLoss;
    }

    // ==================== Critic 输入构造 ====================

    /**
     * 从 Actor 隐藏状态构造 Critic 的输入 [batch, hidden]
     * <p>
     * 取每个样本"最后一个被监督位置"的隐藏状态——该位置是模型看完整条回答之后的状态，
     * 是 bandit 形式下最自然的 V(s)。padding 位置（标签为 ignore_index）不参与选取，
     * 避免 Critic 学到的是 pad token 的表征。
     * <p>
     * 返回的 Variable 由复制出来的数据重建，requireGrad=false，因此与 Actor 的计算图完全断开：
     * 价值损失只更新 Critic 参数。
     *
     * @param hiddenState lm_head 之前的隐藏状态 [batch, seq, hidden]（可能为 [batch, hidden]）
     * @param labels      该候选的标签 [batch, seq]，用于定位最后一个有效位置
     * @return Critic 输入 [batch, hidden]
     */
    private Variable buildCriticInput(Variable hiddenState, NdArray labels) {
        if (hiddenState == null) {
            throw new IllegalStateException(
                    "PPO 需要模型暴露 lm_head 之前的隐藏状态，但 MoEOutput.hiddenState 为 null；"
                            + "请确认前向走的是 MiniMindBlock.forwardWithMoEOutput");
        }

        int[] dims = hiddenState.getValue().getShape().getShapeDims();
        float[] hiddenData = hiddenState.getValue().getArray();

        if (dims.length == 2) {
            // 已经是 [batch, hidden]
            return constantMatrix(hiddenData.clone(), dims[0], dims[1]);
        }
        if (dims.length != 3) {
            throw new IllegalStateException(
                    "隐藏状态应为 [batch, seq, hidden] 或 [batch, hidden]，实际维度: " + dims.length);
        }

        int batchSize = dims[0];
        int seqLen = dims[1];
        int hiddenDim = dims[2];
        float[] labelData = labels.getArray();

        float[] gathered = new float[batchSize * hiddenDim];
        for (int i = 0; i < batchSize; i++) {
            int position = lastSupervisedPosition(labelData, i, seqLen);
            System.arraycopy(hiddenData, (i * seqLen + position) * hiddenDim,
                    gathered, i * hiddenDim, hiddenDim);
        }
        return constantMatrix(gathered, batchSize, hiddenDim);
    }

    /**
     * 找出该样本最后一个被监督的位置（标签 >= 0 的最大下标）
     * <p>
     * 全部位置都被忽略时退回最后一个位置，避免返回非法下标。
     */
    private static int lastSupervisedPosition(float[] labelData, int batchIdx, int seqLen) {
        int offset = batchIdx * seqLen;
        for (int s = seqLen - 1; s >= 0; s--) {
            if (offset + s < labelData.length && labelData[offset + s] >= 0) {
                return s;
            }
        }
        return Math.max(0, seqLen - 1);
    }

    /**
     * 构造不参与梯度的矩阵常量 [rows, cols]
     */
    private static Variable constantMatrix(float[] data, int rows, int cols) {
        Variable var = new Variable(NdArray.of(data, Shape.of(rows, cols)));
        var.setRequireGrad(false);
        return var;
    }

    // ==================== 工具方法 ====================

    /**
     * 取出 [batchSize][numCandidates] 矩阵的第 k 列
     */
    private static float[] column(float[][] matrix, int k, int batchSize) {
        float[] column = new float[batchSize];
        for (int i = 0; i < batchSize; i++) {
            column[i] = (i < matrix.length && k < matrix[i].length) ? matrix[i][k] : 0.0f;
        }
        return column;
    }

    /**
     * 计算平均奖励
     */
    private static float averageReward(float[][] rewards) {
        float sum = 0.0f;
        int count = 0;
        for (float[] row : rewards) {
            for (float r : row) {
                sum += r;
                count++;
            }
        }
        return count > 0 ? sum / count : 0.0f;
    }

    public List<Float> getPolicyLossHistory() {
        return new ArrayList<>(policyLossHistory);
    }

    public List<Float> getValueLossHistory() {
        return new ArrayList<>(valueLossHistory);
    }

    /**
     * 总损失历史
     * <p>
     * 直接复用基类的 {@code lossHistory}，避免维护两份可能不同步的状态。
     */
    public List<Float> getTotalLossHistory() {
        return getLossHistory();
    }

    public List<Float> getRewardHistory() {
        return new ArrayList<>(rewardHistory);
    }

    // ==================== BaseTrainer 抽象方法实现 ====================

    @Override
    protected String getTrainerName() {
        return "PPO";
    }

    @Override
    protected void printTrainingInfo() {
        System.out.println("=".repeat(70));
        System.out.println("开始PPO训练");
        System.out.println("配置: " + config);
        System.out.println("样本数: " + dataset.getSampleCount());
        System.out.println("最大轮次: " + maxEpochs);
        System.out.println("Critic: " + critic);
        System.out.println("=".repeat(70));
    }

    @Override
    protected void printTrainingLog() {
        double avgPolicy = averageOfLast(policyLossHistory, logInterval);
        double avgValue = averageOfLast(valueLossHistory, logInterval);
        double avgReward = averageOfLast(rewardHistory, logInterval);

        System.out.printf("Epoch %d/%d | Step %d | Policy: %.4f | Value: %.4f | Reward: %.4f%n",
                currentEpoch + 1, maxEpochs, currentStep, avgPolicy, avgValue, avgReward);
    }

    private static double averageOfLast(List<Float> history, int window) {
        return history.stream()
                .skip(Math.max(0, history.size() - window))
                .mapToDouble(Float::doubleValue)
                .average()
                .orElse(0.0);
    }

    /**
     * epoch 收尾：同时清掉 Actor 与 Critic 的残留梯度
     */
    @Override
    protected void onEpochEnd() {
        actor.clearGrads();
        critic.clearGrads();
    }

    @Override
    protected void prepareDataset() {
        dataset.prepare(true);
    }

    @Override
    protected boolean hasNextBatch() {
        return dataset.hasNext();
    }

    @Override
    protected Object getNextBatch() {
        return dataset.nextBatch();
    }

    @Override
    protected void resetDataset() {
        dataset.reset();
    }

    @Override
    protected String getCheckpointPrefix() {
        return "ppo";
    }

    /**
     * 保存检查点：Actor（沿用基类 Model 序列化）+ Critic 状态
     * <p>
     * 基类只序列化 Actor，独立 Critic 的价值权重会丢失，导致续训时价值估计从零开始、
     * 优势 A=R-V 失真。这里在 Actor 检查点旁额外落盘一份 Critic 状态。
     */
    @Override
    protected void saveCheckpoint() {
        super.saveCheckpoint();
        saveCriticState();
    }

    /**
     * 将 Critic 状态写入 {@code <prefix>_critic_epoch<E>_step<S>.state}
     */
    private void saveCriticState() {
        String filename = String.format("%s_critic_epoch%d_step%d.state",
                getCheckpointPrefix(), currentEpoch, currentStep);
        String filepath = Paths.get(checkpointDir, filename).toString();
        try (ObjectOutputStream oos =
                     new ObjectOutputStream(new FileOutputStream(filepath))) {
            oos.writeObject(critic.getState());
            System.out.println(getTrainerName() + " Critic 状态已保存: " + filepath);
        } catch (Exception e) {
            System.err.println("保存 Critic 状态失败: " + filepath);
            e.printStackTrace();
        }
    }

    /**
     * 从文件恢复 Critic 状态（断点续训时与 Actor 检查点配套调用）
     *
     * @param filepath {@link #saveCriticState()} 生成的 {@code .state} 文件路径
     */
    public void loadCriticState(String filepath) {
        try (ObjectInputStream ois =
                     new ObjectInputStream(new FileInputStream(filepath))) {
            @SuppressWarnings("unchecked")
            Map<String, float[]> state = (Map<String, float[]>) ois.readObject();
            critic.loadState(state);
            System.out.println(getTrainerName() + " Critic 状态已恢复: " + filepath);
        } catch (Exception e) {
            System.err.println("恢复 Critic 状态失败: " + filepath);
            e.printStackTrace();
        }
    }

    /**
     * 一次 rollout 的快照
     * <p>
     * 只保存数值（float）与原始输入/标签，ppoUpdate 时会重新前向以构建有梯度的计算图。
     * 所有 [batchSize][numCandidates] 矩阵的第 k 列对应第 k 个候选。
     */
    private static final class Rollout {
        final NdArray[] candidateInputs;
        final NdArray[] candidateLabels;
        final float[][] rewards;
        final float[][] oldLogProbs;
        final float[][] oldValues;
        final float[][] advantages;
        final float[][] returns;
        final int batchSize;
        final int numCandidates;

        Rollout(NdArray[] candidateInputs, NdArray[] candidateLabels, float[][] rewards,
                float[][] oldLogProbs, float[][] oldValues,
                float[][] advantages, float[][] returns,
                int batchSize, int numCandidates) {
            this.candidateInputs = candidateInputs;
            this.candidateLabels = candidateLabels;
            this.rewards = rewards;
            this.oldLogProbs = oldLogProbs;
            this.oldValues = oldValues;
            this.advantages = advantages;
            this.returns = returns;
            this.batchSize = batchSize;
            this.numCandidates = numCandidates;
        }
    }
}
