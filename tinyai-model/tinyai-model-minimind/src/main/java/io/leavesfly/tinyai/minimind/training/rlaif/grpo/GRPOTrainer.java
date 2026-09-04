package io.leavesfly.tinyai.minimind.training.rlaif.grpo;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.minimind.model.MiniMindBlock;
import io.leavesfly.tinyai.minimind.model.MiniMindModel;
import io.leavesfly.tinyai.minimind.training.dataset.RLAIFDataset;
import io.leavesfly.tinyai.minimind.training.rlaif.BaseRLTrainer;
import io.leavesfly.tinyai.ml.optimize.Adam;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.ArrayList;
import java.util.List;

/**
 * GRPO (Group Relative Policy Optimization) 训练器
 * <p>
 * GRPO 相比 PPO 的核心区别:
 * 1. 不需要 Critic 网络, 通过组相对优势替代价值估计
 * 2. 将 K 个候选分组, 计算组内相对优势, 减少奖励估计方差
 * 3. 适合大规模候选场景 (K >> 2)
 * <p>
 * 训练流程:
 * 1. 收集 K 个候选回答, 用旧策略计算逐样本 logProb (detach, 不需要梯度)
 * 2. 计算组相对优势 [batchSize, K] (纯数值, 不需要梯度)
 * 3. 多轮 GRPO 更新: 用新策略前向传播, 逐样本计算 clipped surrogate, 反向传播更新 actor
 * <p>
 * 计算图连通性保证:
 * actor.predict() → computePerSampleLogProbs() → grpoLoss.computeCandidateLoss() → loss.backward()
 * 整条链路全部通过 Variable 算子连接, 梯度可以正确回传到 actor 参数。
 * <p>
 * 内存与正确性策略: 每个候选单独 backward（梯度在 actor 参数上自然累积），
 * 全部候选完成后再统一裁剪与更新。这样既避免同时持有 K 份计算图，
 * 也让 K 个候选的梯度按 1/K 等权合并，等价于对候选维度求平均。
 * <p>
 * 训练循环、日志、检查点、NaN 保护、结束后复位 eval 模式均复用 {@link BaseRLTrainer}
 * 的基类实现，本类不再重复声明 maxEpochs/currentStep/lossHistory 等状态字段
 * （重复声明会遮蔽基类字段，导致基类的检查点保存与日志读到永远为 0/空的另一份状态）。
 *
 * @author leavesfly
 * @since 2024
 */
public class GRPOTrainer extends BaseRLTrainer {

    private final MiniMindModel actor;
    private final RLAIFDataset dataset;
    private final GRPOConfig config;
    private final GRPOLoss grpoLoss;

    private final Adam actorOptimizer;

    private final List<Float> rewardHistory;

    /**
     * 是否已经就"全零奖励"告过警（只警告一次，避免刷屏）
     */
    private boolean zeroRewardWarned = false;

    /**
     * 构造函数
     *
     * @param actor   策略网络 (GRPO 不需要 Critic)
     * @param dataset RLAIF 数据集
     * @param config  GRPO 配置
     */
    public GRPOTrainer(MiniMindModel actor, RLAIFDataset dataset, GRPOConfig config) {
        super(actor);
        this.actor = actor;
        this.dataset = dataset;
        this.config = config;
        this.grpoLoss = new GRPOLoss(config);

        this.actorOptimizer = new Adam(actor, config.getActorLearningRate(),
                0.9f, 0.999f, 1e-8f);

        // 训练配置写入基类字段，保证基类的训练循环/日志/检查点逻辑读到同一份状态
        this.maxEpochs = 1;
        this.logInterval = 10;
        this.saveInterval = 500;
        this.maxGradNorm = config.getMaxGradNorm();
        this.checkpointDir = "./checkpoints/minimind/grpo";

        this.rewardHistory = new ArrayList<>();
    }

    /**
     * 配置训练
     */
    public GRPOTrainer configure(int maxEpochs, int logInterval) {
        this.maxEpochs = maxEpochs;
        this.logInterval = logInterval;
        return this;
    }

    // ==================== 核心训练逻辑 ====================

    /**
     * 训练一步：采集旧策略 logProb 与优势，再执行 grpoEpochs 轮更新
     */
    @Override
    protected float trainStep(Object batch) {
        RLAIFDataset.Batch rlBatch = (RLAIFDataset.Batch) batch;

        // 1. 用旧策略收集逐样本 logProb（detach, 不需要梯度）
        float[][] oldLogProbs = collectOldLogProbs(rlBatch);

        // 2. 计算组相对优势（纯数值, 不需要梯度）
        float[][] rewards = rlBatch.getRewards();
        warnIfAllRewardsZero(rewards);
        float[][] advantages = grpoLoss.computeGroupRelativeAdvantages(rewards);

        // 记录平均奖励，便于观察 RL 是否真的在提升
        rewardHistory.add(averageReward(rewards));

        // 3. 多轮 GRPO 更新（复用同一批 rollout，每轮各自完成一次参数更新）
        int innerEpochs = Math.max(1, config.getGrpoEpochs());
        float totalLoss = 0.0f;
        for (int epoch = 0; epoch < innerEpochs; epoch++) {
            totalLoss += grpoUpdate(rlBatch, oldLogProbs, advantages);
        }
        return totalLoss / innerEpochs;
    }

    /**
     * 收集旧策略的逐样本对数概率（detach, 不参与计算图）
     * <p>
     * 旧策略的 logProb 只需要数值, 不需要梯度。
     * 必须保留 batch 维：优势是 per-sample 信号，与该样本自己的 logProb 相乘才有意义。
     *
     * @return [batchSize][numCandidates]
     */
    private float[][] collectOldLogProbs(RLAIFDataset.Batch batch) {
        actor.setTraining(false);

        int numCandidates = batch.getNumCandidates();
        int batchSize = batch.getBatchSize();
        NdArray[] candidateInputs = batch.getCandidateInputs();
        NdArray[] candidateLabels = batch.getCandidateLabels();

        float[][] oldLogProbs = new float[batchSize][numCandidates];

        for (int k = 0; k < numCandidates; k++) {
            Variable inputVar = new Variable(candidateInputs[k]);
            Variable labelVar = new Variable(candidateLabels[k]);
            labelVar.setRequireGrad(false);

            // detach：旧策略只需要数值，不必保留计算图
            Variable logits = actor.predict(inputVar).detach();
            float[] perSample = computePerSampleLogProbValues(logits, labelVar);
            for (int i = 0; i < perSample.length && i < batchSize; i++) {
                oldLogProbs[i][k] = perSample[i];
            }
            logits.unChainBackward();
        }

        return oldLogProbs;
    }

    /**
     * GRPO 更新（计算图连通版本）
     * <p>
     * 关键: 新策略的 logProb 必须保持为 Variable（且保留 batch 维），不能提取为 float,
     * 否则计算图断裂, 梯度无法回传到 actor 参数。
     * <p>
     * 计算图链路:
     * actor.predict(input) → logits [Variable, 有 creator]
     *   → computePerSampleLogProbs(logits, label) → newLogProbs [batch] [Variable, 有 creator]
     *     → grpoLoss.computeCandidateLoss(newLogProbs, oldLogProbs, advantages) → 标量损失
     *       → loss.backward() → 梯度回传到 actor 参数 ✓
     *
     * @param batch       当前批次数据
     * @param oldLogProbs 旧策略的逐样本 logProb [batchSize][numCandidates]（已 detach）
     * @param advantages  组相对优势 [batchSize][numCandidates]
     * @return 损失值
     */
    private float grpoUpdate(RLAIFDataset.Batch batch, float[][] oldLogProbs,
                             float[][] advantages) {
        actor.setTraining(true);
        actor.clearGrads();

        int numCandidates = batch.getNumCandidates();
        int batchSize = batch.getBatchSize();
        NdArray[] candidateInputs = batch.getCandidateInputs();
        NdArray[] candidateLabels = batch.getCandidateLabels();

        float entropyCoef = config.getEntropyCoef();
        float lossSum = 0.0f;
        int validCandidates = 0;

        for (int k = 0; k < numCandidates; k++) {
            Variable inputVar = new Variable(candidateInputs[k]);
            Variable labelVar = new Variable(candidateLabels[k]);
            labelVar.setRequireGrad(false);

            // 新策略前向传播（同时取出 MoE 负载均衡损失；Dense 模式为 0 常量）
            MiniMindBlock.MoEOutput output = forwardWithAux(inputVar);
            Variable logits = output.getOutput();

            // 逐样本 logProb [batch]，计算图连通
            Variable newLogProbs = computePerSampleLogProbs(logits, labelVar);

            // 该候选的旧策略 logProb 与优势（均为 per-sample 常量）
            float[] oldLogProbsK = column(oldLogProbs, k, batchSize);
            float[] advantagesK = column(advantages, k, batchSize);

            Variable policyLoss = grpoLoss.computeCandidateLoss(
                    newLogProbs, oldLogProbsK, advantagesK);

            // 熵正则化（鼓励探索）：标量平均熵，避免损失退化为非标量张量
            Variable entropy = computeScalarEntropy(logits);
            Variable loss = policyLoss.sub(entropy.mul(constant(entropyCoef)));

            // 并入 MoE 辅助损失，并按候选数等权平均
            loss = withMoeAuxLoss(loss, output).mul(constant(1.0f / numCandidates));

            float lossValue = loss.getValue().getNumber().floatValue();
            if (Float.isNaN(lossValue) || Float.isInfinite(lossValue)) {
                System.err.println("警告: GRPO 候选 " + k + " 损失异常 (" + lossValue + "), 跳过");
                loss.unChainBackward();
                continue;
            }

            // 逐候选反向传播：梯度在 actor 参数上累积，避免同时持有 K 份计算图
            loss.backward();
            loss.unChainBackward();

            lossSum += lossValue;
            validCandidates++;
        }

        // 全部候选累积完成后统一裁剪与更新
        clipGradients(actor, config.getMaxGradNorm());
        actorOptimizer.update();
        actor.clearGrads();

        return validCandidates > 0 ? lossSum / validCandidates : 0.0f;
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
     * 全零奖励时给出一次性警告
     * <p>
     * GRPO 的优势来自组内奖励差异：若整批奖励都是 0（典型原因是
     * {@code RLAIFDataset.addSample(prompt, candidates)} 没有传奖励，数据集会填 0.0f），
     * 优势全为 0，策略梯度不会产生任何有效更新——训练看起来在跑，实际是空转。
     */
    private void warnIfAllRewardsZero(float[][] rewards) {
        if (zeroRewardWarned) {
            return;
        }
        for (float[] row : rewards) {
            for (float r : row) {
                if (r != 0.0f) {
                    return;
                }
            }
        }
        zeroRewardWarned = true;
        System.err.println("警告: GRPO 本批次全部奖励为 0，组相对优势将全为 0，策略梯度不会产生任何更新；"
                + "请确认 RLAIFDataset.addSample 传入了奖励，或改用带规则奖励兜底的 SPOTrainer");
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

    public List<Float> getRewardHistory() {
        return new ArrayList<>(rewardHistory);
    }

    // ==================== BaseTrainer 抽象方法实现 ====================

    @Override
    protected String getTrainerName() {
        return "GRPO";
    }

    @Override
    protected void printTrainingInfo() {
        System.out.println("=".repeat(70));
        System.out.println("开始GRPO训练");
        System.out.println("配置: " + config);
        System.out.println("样本数: " + dataset.getSampleCount());
        System.out.println("最大轮次: " + maxEpochs);
        System.out.println("=".repeat(70));
    }

    @Override
    protected void printTrainingLog() {
        double avgLoss = lossHistory.stream()
                .skip(Math.max(0, lossHistory.size() - logInterval))
                .mapToDouble(Float::doubleValue)
                .average()
                .orElse(0.0);

        double avgReward = rewardHistory.stream()
                .skip(Math.max(0, rewardHistory.size() - logInterval))
                .mapToDouble(Float::doubleValue)
                .average()
                .orElse(0.0);

        System.out.printf("Epoch %d/%d | Step %d | Loss: %.4f | Reward: %.4f%n",
                currentEpoch + 1, maxEpochs, currentStep, avgLoss, avgReward);
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
        return "grpo";
    }
}
