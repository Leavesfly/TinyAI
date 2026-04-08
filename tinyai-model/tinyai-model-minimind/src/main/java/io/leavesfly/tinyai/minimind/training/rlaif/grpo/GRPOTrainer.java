package io.leavesfly.tinyai.minimind.training.rlaif.grpo;

import io.leavesfly.tinyai.func.Variable;
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
 * 1. 收集 K 个候选回答, 用旧策略计算 logProb (detach, 不需要梯度)
 * 2. 计算组相对优势 (纯数值, 不需要梯度)
 * 3. 多轮 GRPO 更新: 用新策略前向传播, 通过 Variable 算子计算损失, 反向传播更新 actor
 * <p>
 * 计算图连通性保证:
 * actor.predict() → computeLogProb() → grpoLoss.computeCandidateLoss() → totalLoss.backward()
 * 整条链路全部通过 Variable 算子连接, 梯度可以正确回传到 actor 参数。
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

    private int maxEpochs;
    private int logInterval;
    private int currentEpoch;
    private int currentStep;

    private final List<Float> lossHistory;
    private final List<Float> rewardHistory;

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

        this.maxEpochs = 1;
        this.logInterval = 10;
        this.currentEpoch = 0;
        this.currentStep = 0;

        this.lossHistory = new ArrayList<>();
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

    /**
     * 设置检查点目录
     */
    public GRPOTrainer setCheckpointDir(String checkpointDir) {
        this.checkpointDir = checkpointDir;
        return this;
    }

    /**
     * 训练
     */
    public void train() {
        System.out.println("=".repeat(70));
        System.out.println("开始GRPO训练");
        System.out.println("配置: " + config);
        System.out.println("样本数: " + dataset.getSampleCount());
        System.out.println("=".repeat(70));

        createCheckpointDir();

        for (currentEpoch = 0; currentEpoch < maxEpochs; currentEpoch++) {
            trainOneEpoch();
        }

        System.out.println("\nGRPO训练完成!");
    }

    /**
     * 训练一个 epoch
     */
    protected void trainOneEpoch() {
        dataset.prepare(true);
        float epochLoss = 0.0f;
        int batchCount = 0;

        while (dataset.hasNext()) {
            RLAIFDataset.Batch batch = dataset.nextBatch();

            // 1. 用旧策略收集 logProb（detach, 不需要梯度）
            float[] oldLogProbs = collectOldLogProbs(batch);

            // 2. 计算组相对优势（纯数值, 不需要梯度）
            float[][] rewards = batch.getRewards();
            float[][] advantages = grpoLoss.computeGroupRelativeAdvantages(rewards);

            // 3. 多轮 GRPO 更新
            float avgLoss = 0.0f;
            for (int epoch = 0; epoch < config.getGrpoEpochs(); epoch++) {
                float loss = grpoUpdate(batch, oldLogProbs, advantages);
                avgLoss += loss;
            }
            avgLoss /= config.getGrpoEpochs();

            epochLoss += avgLoss;
            batchCount++;
            currentStep++;
            lossHistory.add(avgLoss);

            if (currentStep % logInterval == 0) {
                System.out.printf("Epoch %d | Step %d | Loss: %.4f%n",
                        currentEpoch + 1, currentStep, avgLoss);
            }
        }

        System.out.printf("Epoch %d 完成 | 平均损失: %.4f%n",
                currentEpoch + 1, epochLoss / batchCount);

        dataset.reset();
    }

    /**
     * 收集旧策略的对数概率（detach, 不参与计算图）
     * <p>
     * 旧策略的 logProb 只需要数值, 不需要梯度,
     * 所以提取为 float 是正确的。
     */
    private float[] collectOldLogProbs(RLAIFDataset.Batch batch) {
        actor.setTraining(false);

        int numCandidates = batch.getNumCandidates();
        NdArray[] candidateInputs = batch.getCandidateInputs();
        NdArray[] candidateLabels = batch.getCandidateLabels();

        float[] oldLogProbs = new float[numCandidates];

        for (int k = 0; k < numCandidates; k++) {
            Variable inputVar = new Variable(candidateInputs[k]);
            Variable labelVar = new Variable(candidateLabels[k]);

            Variable logits = actor.predict(inputVar);
            Variable logProb = computeLogProb(logits, labelVar);

            oldLogProbs[k] = logProb.getValue().getNumber().floatValue();
        }

        return oldLogProbs;
    }

    /**
     * GRPO 更新（计算图连通版本）
     * <p>
     * 关键修复: 新策略的 logProb 必须保持为 Variable, 不能提取为 float,
     * 否则计算图断裂, 梯度无法回传到 actor 参数。
     * <p>
     * 计算图链路:
     * actor.predict(input) → logits [Variable, 有 creator]
     *   → computeLogProb(logits, label) → newLogProb [Variable, 有 creator]
     *     → grpoLoss.computeCandidateLoss(newLogProb, oldLogProb, advantage) → candidateLoss [Variable, 有 creator]
     *       → totalLoss.backward() → 梯度回传到 actor 参数 ✓
     *
     * @param batch       当前批次数据
     * @param oldLogProbs 旧策略的 logProb（float[], 已 detach）
     * @param advantages  组相对优势 [batchSize, numCandidates]
     * @return 损失值
     */
    private float grpoUpdate(RLAIFDataset.Batch batch, float[] oldLogProbs,
                             float[][] advantages) {
        actor.setTraining(true);

        int numCandidates = batch.getNumCandidates();
        int batchSize = batch.getBatchSize();
        NdArray[] candidateInputs = batch.getCandidateInputs();
        NdArray[] candidateLabels = batch.getCandidateLabels();

        // 累加所有候选的策略损失, 保持计算图连通
        Variable totalPolicyLoss = null;
        Variable lastLogits = null;
        int candidateCount = 0;

        for (int k = 0; k < numCandidates; k++) {
            Variable inputVar = new Variable(candidateInputs[k]);
            Variable labelVar = new Variable(candidateLabels[k]);

            // 新策略前向传播, logits 是 Variable, 有 creator, 计算图连通
            Variable logits = actor.predict(inputVar);
            lastLogits = logits;

            // computeLogProb 返回 Variable, 计算图连通
            Variable newLogProb = computeLogProb(logits, labelVar);

            // 对 batch 中每个样本计算损失并累加
            for (int i = 0; i < batchSize; i++) {
                float oldLogProb = oldLogProbs[k];
                float advantage = advantages[i][k];

                // computeCandidateLoss 全部通过 Variable 算子, 计算图连通
                Variable candidateLoss = grpoLoss.computeCandidateLoss(
                        newLogProb, oldLogProb, advantage);

                totalPolicyLoss = (totalPolicyLoss == null)
                        ? candidateLoss
                        : totalPolicyLoss.add(candidateLoss);
                candidateCount++;
            }
        }

        // 平均策略损失
        Variable count = new Variable(NdArray.of((float) candidateCount));
        count.setRequireGrad(false);
        Variable avgPolicyLoss = totalPolicyLoss.div(count);

        // 熵正则化（鼓励探索, 使用最后一个 logits 的熵作为近似）
        Variable entropyLoss = grpoLoss.computeEntropyLoss(lastLogits);
        Variable entropyCoef = new Variable(NdArray.of(config.getEntropyCoef()));
        entropyCoef.setRequireGrad(false);
        Variable totalLoss = avgPolicyLoss.sub(entropyLoss.mul(entropyCoef));

        // 反向传播: totalLoss → avgPolicyLoss → candidateLoss → newLogProb → logits → actor 参数
        actor.clearGrads();
        totalLoss.backward();

        // 梯度裁剪
        clipGradients(actor, config.getMaxGradNorm());

        // 更新 actor 参数
        actorOptimizer.update();

        float lossValue = totalLoss.getValue().getNumber().floatValue();
        totalLoss.unChainBackward();

        return lossValue;
    }

    public List<Float> getLossHistory() {
        return new ArrayList<>(lossHistory);
    }

    public List<Float> getRewardHistory() {
        return new ArrayList<>(rewardHistory);
    }

    // ==================== BaseTrainer 抽象方法实现 ====================

    @Override
    protected float trainStep(Object batch) {
        RLAIFDataset.Batch rlBatch = (RLAIFDataset.Batch) batch;
        float[] oldLogProbs = collectOldLogProbs(rlBatch);
        float[][] advantages = grpoLoss.computeGroupRelativeAdvantages(rlBatch.getRewards());
        return grpoUpdate(rlBatch, oldLogProbs, advantages);
    }

    @Override
    protected String getTrainerName() {
        return "GRPO";
    }

    @Override
    protected void printTrainingInfo() {
        System.out.println("GRPO训练器 | 配置: " + config);
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
