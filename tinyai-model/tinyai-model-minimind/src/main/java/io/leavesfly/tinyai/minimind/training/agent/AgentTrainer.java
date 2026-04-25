package io.leavesfly.tinyai.minimind.training.agent;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.minimind.model.MiniMindModel;
import io.leavesfly.tinyai.minimind.tokenizer.MiniMindTokenizer;
import io.leavesfly.tinyai.minimind.training.rlaif.BaseRLTrainer;
import io.leavesfly.tinyai.ml.optimize.Adam;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.*;

/**
 * Agent 强化学习训练器
 * <p>
 * 对标 Python minimind3 train_agent.py rl_train_epoch (L414-L577)。
 * <p>
 * 训练流程：
 * 1. 从 AgentDataset 加载 batch（prompt + tools + gt）
 * 2. 使用 AgentRolloutEngine 生成多轮 rollout（每个 prompt 生成 numGenerations 个候选）
 * 3. 使用 AgentRewardCalculator 计算多维度奖励
 * 4. 计算组相对优势（GRPO）
 * 5. PPO Clipped Surrogate + KL 散度约束更新策略
 * 6. 余弦退火学习率 + 梯度裁剪 + 梯度累积
 * <p>
 * 核心公式（对标 Python L521-L531）：
 * - kl_div = ref_logps - per_token_logps
 * - per_token_kl = exp(kl_div) - kl_div - 1
 * - ratio = exp(new_logps - old_logps)
 * - clipped_ratio = clip(ratio, 1-eps, 1+eps)
 * - per_token_loss = -(min(ratio*adv, clipped_ratio*adv) - beta*per_token_kl)
 *
 * @author TinyAI Team
 * @since 2025
 */
public class AgentTrainer extends BaseRLTrainer {

    private final MiniMindModel policyModel;       // 策略模型（训练）
    private final MiniMindModel refModel;           // 参考模型（冻结）
    private final AgentDataset dataset;
    private final AgentConfig config;
    private final AgentRolloutEngine rolloutEngine;
    private final MiniMindTokenizer tokenizer;
    private final Adam optimizer;

    // 训练状态
    private float currentLearningRate;
    private int accumulationCounter = 0;

    // 统计
    private final List<Float> rewardHistory;
    private final List<Float> klHistory;

    /**
     * 构造函数
     *
     * @param policyModel 策略模型（将被训练）
     * @param refModel    参考模型（冻结，用于 KL 约束）
     * @param dataset     Agent 数据集
     * @param config      Agent RL 配置
     * @param tokenizer   分词器
     */
    public AgentTrainer(MiniMindModel policyModel, MiniMindModel refModel,
                         AgentDataset dataset, AgentConfig config,
                         MiniMindTokenizer tokenizer) {
        super(policyModel);
        this.policyModel = policyModel;
        this.refModel = refModel;
        this.dataset = dataset;
        this.config = config;
        this.tokenizer = tokenizer;
        this.rolloutEngine = new AgentRolloutEngine(policyModel, tokenizer);

        // 验证配置
        config.validate();

        // 冻结参考模型
        refModel.setTraining(false);

        // 训练参数
        this.maxEpochs = config.getMaxEpochs();
        this.maxGradNorm = config.getGradClip();
        this.logInterval = config.getLogInterval();
        this.saveInterval = config.getSaveInterval();
        this.checkpointDir = "./checkpoints/minimind/agent";
        this.currentLearningRate = config.getLearningRate();

        // 优化器
        this.optimizer = new Adam(policyModel, config.getLearningRate(),
                0.9f, 0.999f, 1e-8f);

        // 统计
        this.rewardHistory = new ArrayList<>();
        this.klHistory = new ArrayList<>();
    }

    // ==================== 核心训练逻辑 ====================

    @Override
    protected float trainStep(Object batch) {
        // 更新学习率
        updateLearningRate();

        AgentDataset.Batch agentBatch = (AgentDataset.Batch) batch;
        List<List<Map<String, String>>> messagesBatch = agentBatch.getMessagesBatch();
        List<List<String>> toolsBatch = agentBatch.getToolsBatch();
        List<List<String>> gtBatch = agentBatch.getGtBatch();
        int batchSize = agentBatch.getBatchSize();
        int numGen = config.getNumGenerations();

        // ========== 1. Rollout 生成 ==========
        policyModel.setTraining(false);
        List<AgentRolloutEngine.RolloutResult> rolloutResults = rolloutEngine.rolloutBatch(
                messagesBatch, toolsBatch, numGen,
                config.getMaxTurns(), config.getMaxGenLen(), config.getTemperature());
        policyModel.setTraining(true);

        // 收集生成结果
        List<String> completions = new ArrayList<>();
        List<List<String>> turnOutputsBatch = new ArrayList<>();
        List<Boolean> unfinishedBatch = new ArrayList<>();

        for (AgentRolloutEngine.RolloutResult r : rolloutResults) {
            completions.add(r.getFinalOutput());
            turnOutputsBatch.add(r.getTurnOutputs());
            unfinishedBatch.add(r.isUnfinished());
        }

        // ========== 2. 计算奖励 ==========
        float[] rewards = AgentRewardCalculator.calculateRewards(
                completions, gtBatch, toolsBatch, numGen,
                turnOutputsBatch, unfinishedBatch);

        // 记录奖励
        float avgReward = 0.0f;
        for (float r : rewards) avgReward += r;
        avgReward /= rewards.length;
        rewardHistory.add(avgReward);

        // ========== 3. 计算组相对优势（对标 Python L516-L519） ==========
        float[] advantages = computeGroupAdvantages(rewards, numGen);

        // ========== 4. 策略更新 ==========
        // 为每个 completion 计算策略损失
        float totalLoss = 0.0f;
        int validCount = 0;

        for (int idx = 0; idx < completions.size(); idx++) {
            String completion = completions.get(idx);
            if (completion.isEmpty()) continue;

            // 编码 completion 为 token
            List<Integer> tokens = tokenizer.encode(completion);
            if (tokens.size() < 2) continue;

            int seqLen = Math.min(tokens.size(), policyModel.getConfig().getMaxSeqLen());
            int[] inputIds = new int[seqLen];
            int[] targetIds = new int[seqLen];

            for (int i = 0; i < seqLen - 1; i++) {
                inputIds[i] = tokens.get(i);
                targetIds[i] = tokens.get(i + 1);
            }
            inputIds[seqLen - 1] = tokens.get(seqLen - 1);
            targetIds[seqLen - 1] = tokens.get(Math.min(seqLen, tokens.size() - 1));

            // 构建 NdArray [1, seqLen]
            float[] inputData = new float[seqLen];
            float[] targetData = new float[seqLen];
            for (int i = 0; i < seqLen; i++) {
                inputData[i] = inputIds[i];
                targetData[i] = targetIds[i];
            }
            NdArray inputNd = NdArray.of(inputData, Shape.of(1, seqLen));
            NdArray targetNd = NdArray.of(targetData, Shape.of(1, seqLen));

            // 新策略前向传播
            policyModel.setTraining(true);
            Variable inputVar = new Variable(inputNd);
            Variable newLogits = policyModel.predict(inputVar);

            // 计算新策略 logProb
            Variable labelVar = new Variable(targetNd);
            Variable newLogProb = computeLogProb(newLogits, labelVar);

            // 参考模型前向传播（无梯度）
            refModel.setTraining(false);
            Variable refInputVar = new Variable(inputNd.copy());
            Variable refLogits = refModel.predict(refInputVar);
            refLogits = refLogits.detach();
            Variable refLogProb = computeLogProb(refLogits, new Variable(targetNd.copy()));
            float refLogProbValue = refLogProb.getValue().getNumber().floatValue();
            refLogits.unChainBackward();

            // 旧策略 logProb（使用 rollout 时的值近似为当前值的 detach）
            float oldLogProb = newLogProb.getValue().getNumber().floatValue();

            // KL 散度约束（对标 Python L521-L522）
            float klDiv = refLogProbValue - newLogProb.getValue().getNumber().floatValue();
            float perTokenKl = (float) (Math.exp(klDiv) - klDiv - 1.0);

            // PPO Clipped Surrogate Loss（对标 Python L528-L531）
            float advantage = advantages[idx];
            Variable candidateLoss = computeAgentLoss(newLogProb, oldLogProb, advantage,
                    perTokenKl, config.getBeta(), config.getEpsilon());

            // 累加损失
            float lossVal = candidateLoss.getValue().getNumber().floatValue();

            // 梯度累积缩放
            int accumSteps = config.getAccumulationSteps();
            if (accumSteps > 1) {
                Variable accumScale = new Variable(1.0f / accumSteps);
                accumScale.setRequireGrad(false);
                candidateLoss = candidateLoss.mul(accumScale);
            }

            // 反向传播
            candidateLoss.backward();
            candidateLoss.unChainBackward();

            totalLoss += lossVal;
            validCount++;
        }

        // 梯度累积更新
        accumulationCounter++;
        if (accumulationCounter % config.getAccumulationSteps() == 0) {
            clipGradients(policyModel, config.getGradClip());
            optimizer.update();
            policyModel.clearGrads();
            accumulationCounter = 0;
        }

        return validCount > 0 ? totalLoss / validCount : 0.0f;
    }

    /**
     * 计算 Agent PPO Clipped Loss + KL 惩罚
     * <p>
     * 对标 Python L528-L531 (GRPO 模式):
     * clipped_ratio = clip(ratio, 1-eps, 1+eps)
     * per_token_loss = -(min(ratio*adv, clipped_ratio*adv) - beta*per_token_kl)
     *
     * @param newLogProb 新策略 logProb（Variable, 保持梯度）
     * @param oldLogProb 旧策略 logProb（float, 已 detach）
     * @param advantage  组相对优势
     * @param kl         KL 散度
     * @param beta       KL 惩罚系数
     * @param epsilon    PPO clip 范围
     * @return 策略损失
     */
    private Variable computeAgentLoss(Variable newLogProb, float oldLogProb,
                                       float advantage, float kl,
                                       float beta, float epsilon) {
        // ratio = exp(newLogProb - oldLogProb)
        Variable oldLogProbVar = new Variable(NdArray.of(oldLogProb));
        oldLogProbVar.setRequireGrad(false);
        Variable logRatio = newLogProb.sub(oldLogProbVar);
        Variable ratio = logRatio.exp();

        // surrogate1 = ratio * advantage
        Variable advVar = new Variable(NdArray.of(advantage));
        advVar.setRequireGrad(false);
        Variable surrogate1 = ratio.mul(advVar);

        // surrogate2 = clip(ratio, 1-eps, 1+eps) * advantage
        Variable clippedRatio = ratio.clip(1.0f - epsilon, 1.0f + epsilon);
        Variable surrogate2 = clippedRatio.mul(advVar);

        // min(surrogate1, surrogate2)
        Variable condition = surrogate1.lt(surrogate2);
        Variable minSurrogate = Variable.where(condition, surrogate1, surrogate2);

        // KL 惩罚
        Variable klPenalty = new Variable(NdArray.of(beta * kl));
        klPenalty.setRequireGrad(false);

        // loss = -(minSurrogate - klPenalty)
        Variable loss = minSurrogate.sub(klPenalty).neg();

        return loss;
    }

    /**
     * 计算组相对优势（对标 Python L516-L519）
     * <p>
     * advantages = (rewards - group_mean) / (group_std + 1e-4)
     */
    private float[] computeGroupAdvantages(float[] rewards, int numGen) {
        float[] advantages = new float[rewards.length];
        int numGroups = rewards.length / numGen;

        for (int g = 0; g < numGroups; g++) {
            int start = g * numGen;
            int end = Math.min(start + numGen, rewards.length);
            int groupSize = end - start;

            // 组内均值
            float mean = 0.0f;
            for (int i = start; i < end; i++) {
                mean += rewards[i];
            }
            mean /= groupSize;

            // 组内标准差
            float std = 0.0f;
            for (int i = start; i < end; i++) {
                std += (rewards[i] - mean) * (rewards[i] - mean);
            }
            std = (float) Math.sqrt(std / groupSize);

            // 归一化优势
            for (int i = start; i < end; i++) {
                advantages[i] = (rewards[i] - mean) / (std + 1e-4f);
            }
        }

        return advantages;
    }

    /**
     * 更新学习率（余弦退火 with 10% floor，对标 Python CosineAnnealingLR）
     */
    private void updateLearningRate() {
        int totalSteps = maxEpochs * dataset.getBatchCount();
        double cosineDecay = 0.1 + 0.45 * (1 + Math.cos(Math.PI * currentStep / Math.max(totalSteps, 1)));
        currentLearningRate = config.getLearningRate() * (float) cosineDecay;
        optimizer.setLearningRate(currentLearningRate);
    }

    // ==================== 训练入口 ====================

    @Override
    public void train() {
        printTrainingInfo();
        createCheckpointDir();

        for (currentEpoch = 0; currentEpoch < maxEpochs; currentEpoch++) {
            trainOneEpoch();
        }

        System.out.println("\nAgent RL 训练完成!");
        printAgentStats();
    }

    /**
     * 打印 Agent 训练统计
     */
    private void printAgentStats() {
        System.out.println("=".repeat(70));
        System.out.println("Agent RL 训练统计");
        System.out.println("=".repeat(70));

        if (!lossHistory.isEmpty()) {
            double avgLoss = lossHistory.stream().mapToDouble(Float::doubleValue).average().orElse(0.0);
            System.out.printf("  平均损失: %.4f%n", avgLoss);
            System.out.printf("  最终损失: %.4f%n", lossHistory.get(lossHistory.size() - 1));
        }

        if (!rewardHistory.isEmpty()) {
            double avgReward = rewardHistory.stream().mapToDouble(Float::doubleValue).average().orElse(0.0);
            System.out.printf("  平均奖励: %.4f%n", avgReward);
            System.out.printf("  最终奖励: %.4f%n", rewardHistory.get(rewardHistory.size() - 1));
        }

        System.out.printf("  总训练步数: %d%n", currentStep);
        System.out.printf("  配置: %s%n", config);
        System.out.println("=".repeat(70));
    }

    // ==================== BaseTrainer 抽象方法实现 ====================

    @Override
    protected String getTrainerName() {
        return "Agent RL";
    }

    @Override
    protected void printTrainingInfo() {
        System.out.println("=".repeat(70));
        System.out.println("开始 Agent RL 训练 (工具调用强化学习)");
        System.out.println("=".repeat(70));
        System.out.println("  配置: " + config);
        System.out.printf("  训练样本数: %d%n", dataset.getSampleCount());
        System.out.printf("  批次数量: %d%n", dataset.getBatchCount());
        System.out.printf("  每 prompt 候选数: %d%n", config.getNumGenerations());
        System.out.printf("  最大工具交互轮数: %d%n", config.getMaxTurns());
        System.out.printf("  学习率: %.2e%n", config.getLearningRate());
        System.out.printf("  Beta (KL): %.2f, Epsilon (clip): %.2f%n",
                config.getBeta(), config.getEpsilon());
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

        System.out.printf("Epoch %d/%d | Step %d | Loss: %.4f | Reward: %.4f | LR: %.6f%n",
                currentEpoch + 1, maxEpochs, currentStep, avgLoss, avgReward, currentLearningRate);
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
        return "agent";
    }

    // ==================== Getter ====================

    public List<Float> getRewardHistory() {
        return new ArrayList<>(rewardHistory);
    }

    public void setCheckpointDir(String checkpointDir) {
        this.checkpointDir = checkpointDir;
    }
}
