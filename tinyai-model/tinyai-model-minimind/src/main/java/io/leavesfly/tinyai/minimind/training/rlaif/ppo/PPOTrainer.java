package io.leavesfly.tinyai.minimind.training.rlaif.ppo;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.minimind.model.MiniMindModel;

import io.leavesfly.tinyai.minimind.training.dataset.RLAIFDataset;
import io.leavesfly.tinyai.ml.optimize.Adam;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.core.Parameter;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * PPO (Proximal Policy Optimization) 训练器
 * 
 * Actor-Critic架构:
 * - Actor: 策略网络(MiniMindModel)
 * - Critic: 价值网络(ValueNetwork)
 * 
 * 训练流程:
 * 1. 收集经验(生成K个候选回答)
 * 2. 计算GAE优势和回报
 * 3. 多轮PPO更新(使用经验重放)
 * 4. 更新Actor和Critic
 * 
 * @author leavesfly
 * @since 2024
 */
public class PPOTrainer {
    
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
        this.actor = actor;
        this.critic = critic;
        this.dataset = dataset;
        this.config = config;
        this.ppoLoss = new PPOLoss(config);
        
        // 创建优化器
        this.actorOptimizer = new Adam(actor, config.getActorLearningRate(), 
                                       0.9f, 0.999f, 1e-8f);
        // 修复: 将 ValueNetwork 包装为 Model 后创建优化器
        io.leavesfly.tinyai.ml.model.Model criticModel = 
            new io.leavesfly.tinyai.ml.model.Model("critic", critic);
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
    private void trainOneEpoch() {
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
        clipGradients(actor);
        clipGradients(critic);
        
        // 5. 更新参数
        actorOptimizer.update();
        criticOptimizer.update();
        
        float lossValue = totalLoss.getValue().getNumber().floatValue();
        totalLoss.unChainBackward();
        
        return lossValue;
    }
    
    /**
     * 计算对数概率(简化实现)
     */
    private Variable computeLogProb(Variable logits, Variable labels) {
        // Log softmax
        Variable logProbs = logSoftmax(logits);
        
        // 简化:返回平均对数概率
        Variable meanLogProb = logProbs.mean(0, true);
        
        return meanLogProb;
    }
    
    /**
     * 提取隐藏状态(简化实现)
     */
    private Variable extractHiddenState(Variable input) {
        // 简化:使用随机隐藏状态
        // 实际应该从模型中间层提取
        int hiddenDim = critic.getHiddenDim();
        float[] hiddenData = new float[hiddenDim];
        for (int i = 0; i < hiddenDim; i++) {
            hiddenData[i] = (float) (Math.random() - 0.5);
        }
        NdArray hiddenArray = NdArray.of(Shape.of(1, hiddenDim));
        float[] buffer = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) hiddenArray).buffer;
        System.arraycopy(hiddenData, 0, buffer, 0, hiddenDim);
        return new Variable(hiddenArray);
    }
    
    /**
     * Log Softmax
     */
    private Variable logSoftmax(Variable x) {
        Variable expX = x.exp();
        Variable sumExp = expX.sum();
        Variable logSumExp = sumExp.log();
        return x.sub(logSumExp);
    }
    
    /**
     * 梯度裁剪
     */
    private void clipGradients(Object model) {
        float maxNorm = config.getMaxGradNorm();
        if (maxNorm <= 0) return;
        
        Map<String, Parameter> params;
        if (model instanceof MiniMindModel) {
            params = ((MiniMindModel) model).getAllParams();
        } else if (model instanceof ValueNetwork) {
            // ValueNetwork返回v2.core.ParameterV1,需要跳过
            return; // 暂不支持ValueNetwork的梯度裁剪
        } else {
            return;
        }
        
        float totalNorm = 0.0f;
        for (Parameter param : params.values()) {
            if (param.getGrad() != null) {
                float[] gradData = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) param.getGrad()).buffer;
                for (float g : gradData) {
                    totalNorm += g * g;
                }
            }
        }
        
        totalNorm = (float) Math.sqrt(totalNorm);
        
        if (totalNorm > maxNorm) {
            float scale = maxNorm / (totalNorm + 1e-6f);
            for (Parameter param : params.values()) {
                if (param.getGrad() != null) {
                    float[] gradData = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) param.getGrad()).buffer;
                    for (int i = 0; i < gradData.length; i++) {
                        gradData[i] *= scale;
                    }
                }
            }
        }
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