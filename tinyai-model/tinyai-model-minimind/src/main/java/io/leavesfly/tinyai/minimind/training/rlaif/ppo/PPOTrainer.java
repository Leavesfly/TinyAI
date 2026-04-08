package io.leavesfly.tinyai.minimind.training.rlaif.ppo;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.minimind.model.MiniMindModel;

import io.leavesfly.tinyai.minimind.training.dataset.RLAIFDataset;
import io.leavesfly.tinyai.minimind.training.rlaif.BaseRLTrainer;
import io.leavesfly.tinyai.ml.model.Model;
import io.leavesfly.tinyai.ml.optimize.Adam;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.ArrayList;
import java.util.List;

/**
 * PPO (Proximal Policy Optimization) 训练器
 * <p>
 * Actor-Critic架构:
 * - Actor: 策略网络(MiniMindModel)
 * - Critic: 价值网络(ValueNetwork)
 * <p>
 * 训练流程:
 * 1. 收集经验(生成K个候选回答)
 * 2. 计算GAE优势和回报
 * 3. 多轮PPO更新(使用经验重放)
 * 4. 更新Actor和Critic
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

    private int maxEpochs;
    private int logInterval;
    private int currentEpoch;
    private int currentStep;

    private final List<Float> policyLossHistory;
    private final List<Float> valueLossHistory;
    private final List<Float> totalLossHistory;

    /**
     * 构造函数
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
        // 修复: 将 ValueNetwork 包装为 Model 后创建优化器
        Model criticModel = new Model("critic", critic);
        this.criticOptimizer = new Adam(criticModel, config.getCriticLearningRate(),
                0.9f, 0.999f, 1e-8f);

        this.maxEpochs = 1;
        this.logInterval = 10;
        this.currentEpoch = 0;
        this.currentStep = 0;

        this.policyLossHistory = new ArrayList<>();
        this.valueLossHistory = new ArrayList<>();
        this.totalLossHistory = new ArrayList<>();
    }

    /**
     * 配置训练
     */
    public PPOTrainer configure(int maxEpochs, int logInterval) {
        this.maxEpochs = maxEpochs;
        this.logInterval = logInterval;
        return this;
    }

    /**
     * 训练
     */
    public void train() {
        System.out.println("=".repeat(70));
        System.out.println("开始PPO训练");
        System.out.println("配置: " + config);
        System.out.println("样本数: " + dataset.getSampleCount());
        System.out.println("=".repeat(70));

        for (currentEpoch = 0; currentEpoch < maxEpochs; currentEpoch++) {
            trainOneEpoch();
        }

        System.out.println("\nPPO训练完成!");
    }

    /**
     * 训练一个epoch
     */
    protected void trainOneEpoch() {
        dataset.prepare(true);
        float epochLoss = 0.0f;
        int batchCount = 0;

        while (dataset.hasNext()) {
            RLAIFDataset.Batch batch = dataset.nextBatch();

            // 1. 收集经验并计算优势
            ExperienceBuffer experience = collectExperience(batch);

            // 2. 多轮PPO更新
            float avgLoss = 0.0f;
            for (int epoch = 0; epoch < config.getPpoEpochs(); epoch++) {
                float loss = ppoUpdate(experience);
                avgLoss += loss;
            }
            avgLoss /= config.getPpoEpochs();

            epochLoss += avgLoss;
            batchCount++;
            currentStep++;
            totalLossHistory.add(avgLoss);

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
     * 收集经验（旧策略，detach，不需要梯度）
     *
     * 只收集旧策略的 logProb 和 value 的 float 值，以及原始输入/标签/奖励数据。
     * 后续 ppoUpdate 会重新前向传播来构建有梯度的计算图。
     */
    private ExperienceBuffer collectExperience(RLAIFDataset.Batch batch) {
        actor.setTraining(false);

        int numCandidates = batch.getNumCandidates();
        int batchSize = batch.getBatchSize();
        NdArray[] candidateInputs = batch.getCandidateInputs();
        NdArray[] candidateLabels = batch.getCandidateLabels();
        float[][] rewards = batch.getRewards();

        ExperienceBuffer buffer = new ExperienceBuffer(batchSize * numCandidates);

        for (int k = 0; k < numCandidates; k++) {
            Variable inputVar = new Variable(candidateInputs[k]);
            Variable labelVar = new Variable(candidateLabels[k]);

            // 旧策略前向传播（detach，仅取 float 值）
            Variable logits = actor.predict(inputVar);
            Variable logProb = computeLogProb(logits, labelVar);
            float oldLogProbValue = logProb.getValue().getNumber().floatValue();

            // 旧 Critic 前向传播（detach，仅取 float 值）
            Variable hidden = extractHiddenState(logits);
            Variable value = critic.forward(hidden);
            float oldValueFloat = value.getValue().getNumber().floatValue();

            // 存储原始输入数据和 detach 的旧策略值
            for (int i = 0; i < batchSize; i++) {
                buffer.add(
                        oldLogProbValue,
                        oldValueFloat,
                        rewards[i][k],
                        candidateInputs[k],
                        candidateLabels[k]
                );
            }
        }

        // 计算 GAE 优势和回报
        buffer.computeAdvantages(ppoLoss, config);

        return buffer;
    }

    /**
     * PPO更新
     *
     * 关键修复: 重新通过 actor/critic 前向传播，保持计算图连通。
     *
     * 计算图链路:
     * actor.predict(input) → logits [Variable, 有 creator]
     *   → computeLogProb(logits, label) → newLogProb [Variable, 有 creator]
     *     → ppoLoss.computeTotalLoss(...) → totalLoss [Variable, 有 creator]
     *       → totalLoss.backward() → 梯度回传到 actor/critic 参数 ✓
     */
    private float ppoUpdate(ExperienceBuffer experience) {
        actor.setTraining(true);

        int experienceSize = experience.size();

        // 累加所有经验的损失，保持计算图连通
        Variable totalPolicyLoss = null;
        Variable totalValueLoss = null;
        Variable lastLogits = null;

        for (int i = 0; i < experienceSize; i++) {
            NdArray inputData = experience.inputDataList.get(i);
            NdArray labelData = experience.labelDataList.get(i);

            // ===== Actor 重新前向传播（计算图连通）=====
            Variable inputVar = new Variable(inputData);
            Variable labelVar = new Variable(labelData);
            Variable logits = actor.predict(inputVar);
            lastLogits = logits;

            // computeLogProb 返回 Variable，计算图连通
            Variable newLogProb = computeLogProb(logits, labelVar);

            // 计算概率比: r_t = exp(log π_new - log π_old)
            // oldLogProb是旧策略的值,不需要梯度
            Variable oldLogProbVar = new Variable(NdArray.of(experience.logProbs[i]));
            oldLogProbVar.setRequireGrad(false);
            Variable logRatio = newLogProb.sub(oldLogProbVar);
            Variable ratio = logRatio.exp();

            // 计算 clipped surrogate objective
            // advantage是外部信号常量,不参与反向传播
            Variable advVar = new Variable(NdArray.of(experience.advantages[i]));
            advVar.setRequireGrad(false);
            Variable surrogate1 = ratio.mul(advVar);

            // 使用Variable.clip()保持计算图连通
            float clipEps = config.getClipEpsilon();
            Variable clippedRatio = ratio.clip(1.0f - clipEps, 1.0f + clipEps);
            Variable surrogate2 = clippedRatio.mul(advVar);

            // min(surrogate1, surrogate2)，取负值（最大化 → 最小化）
            Variable condition = surrogate1.lt(surrogate2);
            Variable candidatePolicyLoss = Variable.where(condition, surrogate1, surrogate2).neg();

            totalPolicyLoss = (totalPolicyLoss == null)
                    ? candidatePolicyLoss
                    : totalPolicyLoss.add(candidatePolicyLoss);

            // ===== Critic 重新前向传播（计算图连通）=====
            Variable hidden = extractHiddenState(logits);
            Variable newValue = critic.forward(hidden);

            // 价值损失: 0.5 * (V - R)^2, returns是外部信号不需要梯度
            Variable returnVar = new Variable(NdArray.of(experience.returns[i]));
            returnVar.setRequireGrad(false);
            Variable valueDiff = newValue.sub(returnVar);
            Variable half = new Variable(NdArray.of(0.5f));
            half.setRequireGrad(false);
            Variable candidateValueLoss = valueDiff.squ().mul(half);

            totalValueLoss = (totalValueLoss == null)
                    ? candidateValueLoss
                    : totalValueLoss.add(candidateValueLoss);
        }

        // 平均损失
        Variable count = new Variable(NdArray.of((float) experienceSize));
        count.setRequireGrad(false);
        Variable avgPolicyLoss = totalPolicyLoss.div(count);
        Variable avgValueLoss = totalValueLoss.div(count);

        // 熵正则化（鼓励探索）
        Variable entropyLoss = ppoLoss.computeEntropyLoss(lastLogits);

        // 总损失 = 策略损失 + c1*价值损失 - c2*熵损失
        Variable valueLossCoef = new Variable(NdArray.of(config.getValueLossCoef()));
        valueLossCoef.setRequireGrad(false);
        Variable entropyCoef = new Variable(NdArray.of(config.getEntropyCoef()));
        entropyCoef.setRequireGrad(false);
        Variable totalLoss = avgPolicyLoss
                .add(avgValueLoss.mul(valueLossCoef))
                .sub(entropyLoss.mul(entropyCoef));

        // 反向传播: totalLoss → avgPolicyLoss/avgValueLoss → newLogProb/newValue → actor/critic 参数
        actor.clearGrads();
        critic.clearGrads();
        totalLoss.backward();

        // 梯度裁剪
        clipGradients(actor, config.getMaxGradNorm());
        clipGradients(critic, config.getMaxGradNorm());

        // 更新参数
        actorOptimizer.update();
        criticOptimizer.update();

        float lossValue = totalLoss.getValue().getNumber().floatValue();
        totalLoss.unChainBackward();

        return lossValue;
    }



    /**
     * 从 logits 提取隐藏状态供 Critic 使用
     *
     * 通过 Variable 算子（mean）操作，保持计算图连通。
     * 这样 Critic 的梯度可以通过 hidden → logits → actor 参数回传。
     *
     * @param logits actor 前向传播输出的 logits Variable（有 creator）
     * @return 隐藏状态 Variable（计算图连通）
     */
    private Variable extractHiddenState(Variable logits) {
        // 使用 mean 算子对 logits 做降维，保持计算图连通
        // logits shape 可能是 [batch_size, seq_len, vocab_size] 或 [batch_size, vocab_size]
        // mean 后得到 [batch_size, 1] 或标量，作为 Critic 的输入近似
        return logits.mean(0, true);
    }

    public List<Float> getPolicyLossHistory() {
        return new ArrayList<>(policyLossHistory);
    }

    public List<Float> getValueLossHistory() {
        return new ArrayList<>(valueLossHistory);
    }

    public List<Float> getTotalLossHistory() {
        return new ArrayList<>(totalLossHistory);
    }

    // ==================== BaseTrainer 抽象方法实现 ====================

    @Override
    protected float trainStep(Object batch) {
        RLAIFDataset.Batch rlBatch = (RLAIFDataset.Batch) batch;
        ExperienceBuffer experience = collectExperience(rlBatch);
        return ppoUpdate(experience);
    }

    @Override
    protected String getTrainerName() {
        return "PPO";
    }

    @Override
    protected void printTrainingInfo() {
        System.out.println("PPO训练器 | 配置: " + config);
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
     * 经验缓冲区
     *
     * 存储旧策略的 detach 值（float）和原始输入/标签数据（NdArray），
     * 以便 ppoUpdate 时重新前向传播构建有梯度的计算图。
     */
    private static class ExperienceBuffer {
        float[] logProbs;       // 旧策略的 logProb（detach）
        float[] values;         // 旧 Critic 的 value（detach）
        float[] rewards;
        float[] advantages;
        float[] returns;
        List<NdArray> inputDataList;   // 原始输入数据，用于重新前向传播
        List<NdArray> labelDataList;   // 原始标签数据，用于重新前向传播
        int size;
        int capacity;

        ExperienceBuffer(int capacity) {
            this.capacity = capacity;
            this.logProbs = new float[capacity];
            this.values = new float[capacity];
            this.rewards = new float[capacity];
            this.advantages = new float[capacity];
            this.returns = new float[capacity];
            this.inputDataList = new ArrayList<>(capacity);
            this.labelDataList = new ArrayList<>(capacity);
            this.size = 0;
        }

        void add(float logProb, float value, float reward, NdArray inputData, NdArray labelData) {
            if (size < capacity) {
                logProbs[size] = logProb;
                values[size] = value;
                rewards[size] = reward;
                inputDataList.add(inputData);
                labelDataList.add(labelData);
                size++;
            }
        }

        void computeAdvantages(PPOLoss ppoLoss, PPOConfig config) {
            advantages = ppoLoss.computeGAE(rewards, values, 0.0f);
            returns = ppoLoss.computeReturns(advantages, values);
        }

        int size() {
            return size;
        }
    }
}