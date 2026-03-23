package io.leavesfly.tinyai.minimind.training.rlaif.spo;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.minimind.model.MiniMindModel;

import io.leavesfly.tinyai.minimind.training.dataset.RLAIFDataset;
import io.leavesfly.tinyai.minimind.training.rlaif.BaseRLTrainer;
import io.leavesfly.tinyai.ml.optimize.Adam;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.ArrayList;
import java.util.List;

/**
 * SPO (Simplified Policy Optimization) 训练器
 * 
 * 简化的策略优化算法,无需Critic网络
 * 核心思想:
 * 1. 生成K个候选回答
 * 2. 计算奖励R(y)
 * 3. 计算优势A(y) = R(y) - mean(R)
 * 4. 策略梯度优化
 * 
 * @author leavesfly
 * @since 2024
 */
public class SPOTrainer extends BaseRLTrainer {
    
    private final MiniMindModel model;
    private final RLAIFDataset dataset;
    private final SPOConfig config;
    private final SPOLoss spoLoss;
    private final Adam optimizer;
    
    private int maxEpochs;
    private int logInterval;
    private int currentEpoch;
    private int currentStep;
    
    private final List<Float> lossHistory;
    private final List<Float> rewardHistory;
    
    /**
     * 构造函数
     */
    public SPOTrainer(MiniMindModel model, RLAIFDataset dataset, SPOConfig config) {
        super(model);
        this.model = model;
        this.dataset = dataset;
        this.config = config;
        this.spoLoss = new SPOLoss(config);
        this.optimizer = new Adam(model, config.getLearningRate(), 0.9f, 0.999f, 1e-8f);
        
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
    public SPOTrainer configure(int maxEpochs, int logInterval) {
        this.maxEpochs = maxEpochs;
        this.logInterval = logInterval;
        return this;
    }
    
    /**
     * 训练
     */
    public void train() {
        System.out.println("=".repeat(70));
        System.out.println("开始SPO训练");
        System.out.println("配置: " + config);
        System.out.println("样本数: " + dataset.getSampleCount());
        System.out.println("=".repeat(70));
        
        for (currentEpoch = 0; currentEpoch < maxEpochs; currentEpoch++) {
            trainOneEpoch();
        }
        
        System.out.println("\nSPO训练完成!");
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
            float loss = trainStep(batch);
            
            epochLoss += loss;
            batchCount++;
            currentStep++;
            lossHistory.add(loss);
            
            if (currentStep % logInterval == 0) {
                System.out.printf("Epoch %d | Step %d | Loss: %.4f%n",
                    currentEpoch + 1, currentStep, loss);
            }
        }
        
        System.out.printf("Epoch %d 完成 | 平均损失: %.4f%n",
            currentEpoch + 1, epochLoss / batchCount);
        
        dataset.reset();
    }
    
    /**
     * 训练一步
     */
    private float trainStep(RLAIFDataset.Batch batch) {
        model.setTraining(true);
        
        // 1. 前向传播K个候选
        int numCandidates = batch.getNumCandidates();
        NdArray[] candidateInputs = batch.getCandidateInputs();
        NdArray[] candidateLabels = batch.getCandidateLabels();
        float[][] rewards = batch.getRewards();
        
        Variable[] logits = new Variable[numCandidates];
        Variable[] labels = new Variable[numCandidates];
        
        for (int k = 0; k < numCandidates; k++) {
            Variable inputVar = new Variable(candidateInputs[k]);
            logits[k] = model.predict(inputVar);
            labels[k] = new Variable(candidateLabels[k]);
        }
        
        // 2. 计算奖励(如果没有预设奖励,使用规则奖励)
        float[][] computedRewards = computeRewards(batch, rewards);
        
        // 3. 计算SPO损失
        Variable loss = spoLoss.computeLoss(logits, labels, computedRewards);
        
        // 4. 反向传播
        model.clearGrads();
        loss.backward();
        
        // 5. 梯度裁剪
        clipGradients(model, config.getMaxGradNorm());
        
        // 6. 更新参数
        optimizer.update();
        
        float lossValue = loss.getValue().getNumber().floatValue();
        loss.unChainBackward();
        
        return lossValue;
    }
    
    /**
     * 计算奖励(简化规则奖励)
     */
    private float[][] computeRewards(RLAIFDataset.Batch batch, float[][] existingRewards) {
        int batchSize = batch.getBatchSize();
        int numCandidates = batch.getNumCandidates();
        String[][] candidateTexts = batch.getCandidateTexts();
        
        float[][] rewards = new float[batchSize][numCandidates];
        
        for (int i = 0; i < batchSize; i++) {
            for (int k = 0; k < numCandidates; k++) {
                // 如果有预设奖励就用预设的
                if (existingRewards[i][k] != 0.0f) {
                    rewards[i][k] = existingRewards[i][k];
                } else {
                    // 使用简单规则奖励
                    String text = candidateTexts[i][k];
                    float reward = 0.0f;
                    
                    // 长度奖励(适中长度)
                    int length = text.length();
                    if (length > 10 && length < 200) {
                        reward += 0.5f;
                    }
                    
                    // 重复惩罚(简单检查)
                    if (text.contains(text.substring(0, Math.min(10, length)))) {
                        reward -= 0.3f;
                    }
                    
                    rewards[i][k] = reward;
                }
            }
        }
        
        return rewards;
    }
    
    public List<Float> getLossHistory() {
        return new ArrayList<>(lossHistory);
    }
    
    public List<Float> getRewardHistory() {
        return new ArrayList<>(rewardHistory);
    }
}
