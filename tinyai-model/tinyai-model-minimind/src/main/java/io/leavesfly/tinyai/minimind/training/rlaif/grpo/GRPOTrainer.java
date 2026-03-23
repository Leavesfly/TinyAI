package io.leavesfly.tinyai.minimind.training.rlaif.grpo;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.minimind.model.MiniMindModel;

import io.leavesfly.tinyai.minimind.training.dataset.RLAIFDataset;
import io.leavesfly.tinyai.minimind.training.rlaif.BaseRLTrainer;
import io.leavesfly.tinyai.minimind.training.rlaif.ppo.ValueNetwork;
import io.leavesfly.tinyai.ml.optimize.Adam;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.ArrayList;
import java.util.List;

/**
 * GRPO (Group Relative Policy Optimization) 训练器
 * 
 * GRPO特点:
 * 1. 组相对优势计算
 * 2. 适合大规模候选场景(K>>2)
 * 3. 仍使用Actor-Critic架构
 * 4. 减少奖励估计方差
 * 
 * 训练流程:
 * 1. 收集K个候选回答
 * 2. 分组计算组内相对优势
 * 3. 多轮GRPO更新
 * 4. 更新Actor和Critic
 * 
 * @author leavesfly
 * @since 2024
 */
public class GRPOTrainer extends BaseRLTrainer {
    
    private final MiniMindModel actor;
    private final ValueNetwork critic;
    private final RLAIFDataset dataset;
    private final GRPOConfig config;
    private final GRPOLoss grpoLoss;
    
    private final Adam actorOptimizer;
    private final Adam criticOptimizer;
    
    private int maxEpochs;
    private int logInterval;
    private int currentEpoch;
    private int currentStep;
    
    private final List<Float> lossHistory;
    private final List<Float> rewardHistory;
    
    /**
     * 构造函数
     */
    public GRPOTrainer(MiniMindModel actor, ValueNetwork critic,
                      RLAIFDataset dataset, GRPOConfig config) {
        super(actor);
        this.actor = actor;
        this.critic = critic;
        this.dataset = dataset;
        this.config = config;
        this.grpoLoss = new GRPOLoss(config);
        
        this.actorOptimizer = new Adam(actor, config.getActorLearningRate(),
                                       0.9f, 0.999f, 1e-8f);
        // critic不是Model类型,暂不使用优化器
        this.criticOptimizer = null;
        
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
     * 训练一个epoch
     */
    protected void trainOneEpoch() {
        dataset.prepare(true);
        float epochLoss = 0.0f;
        int batchCount = 0;
        
        while (dataset.hasNext()) {
            RLAIFDataset.Batch batch = dataset.nextBatch();
            
            // 1. 收集旧策略的概率
            float[] oldLogProbs = collectOldLogProbs(batch);
            
            // 2. 多轮GRPO更新
            float avgLoss = 0.0f;
            for (int epoch = 0; epoch < config.getGrpoEpochs(); epoch++) {
                float loss = grpoUpdate(batch, oldLogProbs);
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
     * 收集旧策略的对数概率
     */
    private float[] collectOldLogProbs(RLAIFDataset.Batch batch) {
        actor.setTraining(false);
        
        int numCandidates = batch.getNumCandidates();
        int batchSize = batch.getBatchSize();
        NdArray[] candidateInputs = batch.getCandidateInputs();
        NdArray[] candidateLabels = batch.getCandidateLabels();
        
        float[] oldLogProbs = new float[batchSize * numCandidates];
        
        int idx = 0;
        for (int k = 0; k < numCandidates; k++) {
            Variable inputVar = new Variable(candidateInputs[k]);
            Variable labelVar = new Variable(candidateLabels[k]);
            
            Variable logits = actor.predict(inputVar);
            Variable logProb = computeLogProb(logits, labelVar);
            
            float logProbValue = logProb.getValue().getNumber().floatValue();
            
            for (int i = 0; i < batchSize; i++) {
                oldLogProbs[idx++] = logProbValue;
            }
        }
        
        return oldLogProbs;
    }
    
    /**
     * GRPO更新
     */
    private float grpoUpdate(RLAIFDataset.Batch batch, float[] oldLogProbs) {
        actor.setTraining(true);
        
        int numCandidates = batch.getNumCandidates();
        int batchSize = batch.getBatchSize();
        NdArray[] candidateInputs = batch.getCandidateInputs();
        NdArray[] candidateLabels = batch.getCandidateLabels();
        float[][] rewards = batch.getRewards();
        
        // 1. 计算新策略的对数概率
        float[] newLogProbs = new float[batchSize * numCandidates];
        int idx = 0;
        
        for (int k = 0; k < numCandidates; k++) {
            Variable inputVar = new Variable(candidateInputs[k]);
            Variable labelVar = new Variable(candidateLabels[k]);
            
            Variable logits = actor.predict(inputVar);
            Variable logProb = computeLogProb(logits, labelVar);
            
            float logProbValue = logProb.getValue().getNumber().floatValue();
            
            for (int i = 0; i < batchSize; i++) {
                newLogProbs[idx++] = logProbValue;
            }
        }
        
        // 2. 计算损失
        Variable newLogProbsVar = new Variable(NdArray.of(newLogProbs));
        Variable oldLogProbsVar = new Variable(NdArray.of(oldLogProbs));
        Variable dummyLogits = new Variable(NdArray.of(new float[]{0.0f}));
        
        Variable totalLoss = grpoLoss.computeTotalLoss(
            newLogProbsVar, oldLogProbsVar, rewards, dummyLogits
        );
        
        // 3. 反向传播
        actor.clearGrads();
        if (critic != null) {
            critic.clearGrads();
        }
        totalLoss.backward();
        
        // 4. 梯度裁剪
        clipGradients(actor, config.getMaxGradNorm());
        if (critic != null) {
            clipGradients(critic, config.getMaxGradNorm());
        }
        
        // 5. 更新参数
        actorOptimizer.update();
        if (critic != null && criticOptimizer != null) {
            criticOptimizer.update();
        }
        
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
        return grpoUpdate(rlBatch, oldLogProbs);
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
