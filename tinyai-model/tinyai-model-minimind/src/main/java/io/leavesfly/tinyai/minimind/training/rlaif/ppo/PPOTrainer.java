package io.leavesfly.tinyai.minimind.training.rlaif.ppo;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.minimind.model.MiniMindModel;

import io.leavesfly.tinyai.minimind.training.dataset.RLAIFDataset;
import io.leavesfly.tinyai.minimind.training.rlaif.BaseRLTrainer;
import io.leavesfly.tinyai.ml.model.Model;
import io.leavesfly.tinyai.ml.optimize.Adam;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

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
     * 收集经验
     */
    private ExperienceBuffer collectExperience(RLAIFDataset.Batch batch) {
        actor.setTraining(false);  // 评估模式

        int numCandidates = batch.getNumCandidates();
        int batchSize = batch.getBatchSize();
        NdArray[] candidateInputs = batch.getCandidateInputs();
        NdArray[] candidateLabels = batch.getCandidateLabels();
        float[][] rewards = batch.getRewards();

        ExperienceBuffer buffer = new ExperienceBuffer(batchSize * numCandidates);

        // 对每个候选收集经验
        for (int k = 0; k < numCandidates; k++) {
            Variable inputVar = new Variable(candidateInputs[k]);
            Variable labelVar = new Variable(candidateLabels[k]);

            // Actor前向传播
            Variable logits = actor.predict(inputVar);
            Variable logProb = computeLogProb(logits, labelVar);

            // Critic前向传播(简化:使用最后一层隐藏状态)
            Variable hidden = extractHiddenState(inputVar);
            Variable value = critic.forward(hidden);

            // 存储经验
            for (int i = 0; i < batchSize; i++) {
                buffer.add(
                        logProb.getValue().getNumber().floatValue(),
                        value.getValue().getNumber().floatValue(),
                        rewards[i][k],
                        logits.getValue(),
                        hidden.getValue()
                );
            }
        }

        // 计算GAE优势
        buffer.computeAdvantages(ppoLoss, config);

        return buffer;
    }

    /**
     * PPO更新
     */
    private float ppoUpdate(ExperienceBuffer experience) {
        actor.setTraining(true);

        // 获取经验数据
        float[] oldLogProbs = experience.logProbs;
        float[] oldValues = experience.values;
        float[] rewards = experience.rewards;
        float[] advantages = experience.advantages;
        float[] returns = experience.returns;

        // 重新计算当前策略的概率和价值
        List<Float> batchLosses = new ArrayList<>();

        // 简化实现:对整批数据更新一次
        // 实际应该mini-batch采样

        // 1. 重新前向传播
        int experienceSize = experience.size();
        float[] newLogProbsArray = new float[experienceSize];
        float[] newValuesArray = new float[experienceSize];

        for (int i = 0; i < experienceSize; i++) {
            NdArray logitsData = experience.logitsArray.get(i);
            NdArray hiddenData = experience.hiddenStates.get(i);

            // 重新计算Actor
            Variable logits = new Variable(logitsData);
            Variable labels = new Variable(NdArray.of(0.0f)); // 简化
            Variable newLogProb = computeLogProb(logits, labels);
            newLogProbsArray[i] = newLogProb.getValue().getNumber().floatValue();

            // 重新计算Critic
            Variable hidden = new Variable(hiddenData);
            Variable newValue = critic.forward(hidden);
            newValuesArray[i] = newValue.getValue().getNumber().floatValue();
        }

        // 2. 计算损失
        Variable newLogProbs = new Variable(NdArray.of(newLogProbsArray));
        Variable oldLogProbsVar = new Variable(NdArray.of(oldLogProbs));
        Variable newValues = new Variable(NdArray.of(newValuesArray));
        Variable oldValuesVar = new Variable(NdArray.of(oldValues));
        Variable dummyLogits = new Variable(NdArray.of(new float[]{0.0f}));

        Variable totalLoss = ppoLoss.computeTotalLoss(
                newLogProbs, oldLogProbsVar, advantages, returns,
                newValues, oldValuesVar, dummyLogits
        );

        // 3. 反向传播
        actor.clearGrads();
        critic.clearGrads();
        totalLoss.backward();

        // 4. 梯度裁剪
        clipGradients(actor, config.getMaxGradNorm());
        clipGradients(critic, config.getMaxGradNorm());

        // 5. 更新参数
        actorOptimizer.update();
        criticOptimizer.update();

        float lossValue = totalLoss.getValue().getNumber().floatValue();
        totalLoss.unChainBackward();

        return lossValue;
    }

    /**
     * 提取隐藏状态
     * 
     * 从模型的最后一层Transformer输出中提取隐藏状态，
     * 取序列最后一个位置的隐藏向量作为整个序列的表示。
     */
    private Variable extractHiddenState(Variable input) {
        // 通过模型前向传播获取logits，同时提取最后一层隐藏状态
        Variable logits = actor.predict(input);
        NdArray logitsData = logits.getValue();
        int[] logitsShape = logitsData.getShape().getShapeDims();

        // logits shape: [batch_size, seq_len, vocab_size]
        // 取最后一个时间步的logits作为隐藏状态的近似表示
        // 然后通过线性投影到critic的hiddenDim
        int batchSize = logitsShape.length >= 3 ? logitsShape[0] : 1;
        int vocabSize = logitsShape.length >= 3 ? logitsShape[2] : logitsShape[logitsShape.length - 1];
        int hiddenDim = critic.getHiddenDim();

        // 从logits的最后一个时间步提取特征，并截断/填充到hiddenDim
        float[] logitsBuffer = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) logitsData).buffer;
        NdArray hiddenArray = NdArray.of(Shape.of(batchSize, hiddenDim));
        float[] hiddenBuffer = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) hiddenArray).buffer;

        int seqLen = logitsShape.length >= 3 ? logitsShape[1] : 1;
        for (int b = 0; b < batchSize; b++) {
            int lastStepOffset = b * seqLen * vocabSize + (seqLen - 1) * vocabSize;
            int copyLen = Math.min(vocabSize, hiddenDim);
            System.arraycopy(logitsBuffer, lastStepOffset, hiddenBuffer, b * hiddenDim, copyLen);
        }

        return new Variable(hiddenArray);
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
     */
    private static class ExperienceBuffer {
        float[] logProbs;
        float[] values;
        float[] rewards;
        float[] advantages;
        float[] returns;
        List<NdArray> logitsArray;
        List<NdArray> hiddenStates;
        int size;
        int capacity;

        ExperienceBuffer(int capacity) {
            this.capacity = capacity;
            this.logProbs = new float[capacity];
            this.values = new float[capacity];
            this.rewards = new float[capacity];
            this.advantages = new float[capacity];
            this.returns = new float[capacity];
            this.logitsArray = new ArrayList<>(capacity);
            this.hiddenStates = new ArrayList<>(capacity);
            this.size = 0;
        }

        void add(float logProb, float value, float reward, NdArray logits, NdArray hidden) {
            if (size < capacity) {
                logProbs[size] = logProb;
                values[size] = value;
                rewards[size] = reward;
                logitsArray.add(logits);
                hiddenStates.add(hidden);
                size++;
            }
        }

        void computeAdvantages(PPOLoss ppoLoss, PPOConfig config) {
            // 使用GAE计算优势
            advantages = ppoLoss.computeGAE(rewards, values, 0.0f);
            returns = ppoLoss.computeReturns(advantages, values);
        }

        int size() {
            return size;
        }
    }
}