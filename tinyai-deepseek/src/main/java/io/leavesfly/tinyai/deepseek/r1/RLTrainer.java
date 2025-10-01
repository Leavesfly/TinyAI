package io.leavesfly.tinyai.deepseek.r1;

import io.leavesfly.tinyai.func.Variable;

import io.leavesfly.tinyai.ml.Trainer;
import io.leavesfly.tinyai.ml.dataset.DataSet;
import io.leavesfly.tinyai.ml.evaluator.Evaluator;
import io.leavesfly.tinyai.ml.Monitor;
import io.leavesfly.tinyai.ml.loss.Loss;
import io.leavesfly.tinyai.ml.optimize.Optimizer;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 强化学习训练器 - 专为DeepSeek R1设计的训练组件
 * 
 * 该训练器扩展了标准的Trainer，增加了：
 * 1. 强化学习奖励计算
 * 2. 策略梯度训练
 * 3. 推理质量评估
 * 4. 自我改进机制
 * 5. 奖励塑形策略
 * 
 * 基于Python实现中的RLTrainer，使用TinyAI架构重新实现
 */
public class RLTrainer extends Trainer {
    
    private DeepSeekR1Model deepseekModel;
    
    // 强化学习特有参数
    private float baselineDecay = 0.99f;
    private float runningBaseline = 0.0f;
    private float entropyCoefficient = 0.01f;
    private float valueCoefficient = 0.1f;
    
    // 奖励计算权重
    private float accuracyRewardWeight = 1.0f;
    private float reasoningQualityWeight = 0.3f;
    private float reflectionRewardWeight = 0.2f;
    private float consistencyRewardWeight = 0.1f;
    
    // 训练统计
    private List<Float> rewardHistory;
    private List<Float> qualityHistory;
    private Map<String, Float> trainingMetrics;
    
    /**
     * 构造强化学习训练器
     * 
     * @param maxEpoch 最大训练轮次
     * @param monitor 监控器
     * @param evaluator 评估器
     */
    public RLTrainer(int maxEpoch, Monitor monitor, Evaluator evaluator) {
        super(maxEpoch, monitor, evaluator);
        
        this.rewardHistory = new ArrayList<>();
        this.qualityHistory = new ArrayList<>();
        this.trainingMetrics = new HashMap<>();
        
        initializeMetrics();
    }
    
    /**
     * 构造强化学习训练器（支持并行训练）
     * 
     * @param maxEpoch 最大训练轮次
     * @param monitor 监控器
     * @param evaluator 评估器
     * @param enableParallel 是否启用并行训练
     * @param threadCount 线程数
     */
    public RLTrainer(int maxEpoch, Monitor monitor, Evaluator evaluator,
                    boolean enableParallel, int threadCount) {
        super(maxEpoch, monitor, evaluator, enableParallel, threadCount);
        
        this.rewardHistory = new ArrayList<>();
        this.qualityHistory = new ArrayList<>();
        this.trainingMetrics = new HashMap<>();
        
        initializeMetrics();
    }
    
    /**
     * 初始化训练指标
     */
    private void initializeMetrics() {
        trainingMetrics.put("total_reward", 0.0f);
        trainingMetrics.put("average_quality", 0.0f);
        trainingMetrics.put("policy_loss", 0.0f);
        trainingMetrics.put("value_loss", 0.0f);
        trainingMetrics.put("entropy_loss", 0.0f);
        trainingMetrics.put("reflection_improvement", 0.0f);
    }
    
    /**
     * 初始化强化学习训练器
     * 
     * @param dataSet 数据集
     * @param model 模型（必须是DeepSeekR1Model）
     * @param loss 损失函数
     * @param optimizer 优化器
     */
    @Override
    public void init(DataSet dataSet, io.leavesfly.tinyai.ml.Model model, Loss loss, Optimizer optimizer) {
        // 检查模型类型
        if (!(model instanceof DeepSeekR1Model)) {
            throw new IllegalArgumentException("RLTrainer只支持DeepSeekR1Model");
        }
        
        this.deepseekModel = (DeepSeekR1Model) model;
        
        // 调用父类初始化
        super.init(dataSet, model, loss, optimizer);
        
        System.out.println("强化学习训练器初始化完成");
        System.out.println("模型类型: " + model.getClass().getSimpleName());
        System.out.println("基线衰减率: " + baselineDecay);
        System.out.println("熵系数: " + entropyCoefficient);
        System.out.println("价值系数: " + valueCoefficient);
    }
    
    /**
     * 强化学习训练步骤
     * 
     * @param inputIds 输入序列
     * @param targetIds 目标序列
     * @return 训练指标
     */
    public Map<String, Float> trainRLStep(NdArray inputIds, NdArray targetIds) {
        deepseekModel.clearGrads();
        
        // 1. 执行前向传播，获取详细结果
        DeepSeekR1Block.DeepSeekR1Result modelOutput = 
            deepseekModel.inferenceWithDetails(inputIds, null);
        
        // 2. 计算奖励信号
        RewardComponents rewards = computeRewardComponents(modelOutput, targetIds);
        float totalReward = computeTotalReward(rewards);
        
        // 3. 更新运行基线
        updateBaseline(totalReward);
        
        // 4. 计算策略梯度损失
        Variable policyLoss = computePolicyLoss(modelOutput, targetIds, totalReward - runningBaseline);
        
        // 5. 计算价值函数损失
        Variable valueLoss = computeValueLoss(totalReward);
        
        // 6. 计算熵损失（鼓励探索）
        Variable entropyLoss = computeEntropyLoss(modelOutput.getLogits());
        
        // 7. 组合总损失
        Variable totalLoss = policyLoss.add(new Variable(valueLoss.getValue().mulNum(valueCoefficient)))
                                      .add(new Variable(entropyLoss.getValue().mulNum(entropyCoefficient)));
        
        // 8. 反向传播和参数更新
        totalLoss.backward();
        
        // 梯度裁剪
        clipGradients(1.0f);
        
        // 更新参数
        // 这里需要访问optimizer，但由于父类封装，我们用一个简化方法
        performParameterUpdate();
        
        // 9. 更新统计信息
        updateTrainingMetrics(rewards, policyLoss, valueLoss, entropyLoss, totalReward);
        
        // 10. 记录历史
        rewardHistory.add(totalReward);
        qualityHistory.add(rewards.getReasoningQualityReward());
        
        return getTrainingStepMetrics(rewards, policyLoss, valueLoss, entropyLoss);
    }
    
    /**
     * 计算奖励组件
     */
    private RewardComponents computeRewardComponents(DeepSeekR1Block.DeepSeekR1Result modelOutput, 
                                                   NdArray targetIds) {
        // 1. 准确性奖励（基于交叉熵）
        float accuracyReward = computeAccuracyReward(modelOutput.getLogits(), targetIds);
        
        // 2. 推理质量奖励
        ReflectionBlock.ReflectionResult reflection = modelOutput.getReflectionResult();
        float reasoningQualityReward = reflection.getQualityScore();
        
        // 3. 反思奖励（鼓励自我改进）
        float reflectionReward = reflection.needsRefinement() ? 0.5f : 1.0f;
        
        // 4. 一致性奖励（推理步骤的一致性）
        float consistencyReward = computeConsistencyReward(modelOutput);
        
        return new RewardComponents(accuracyReward, reasoningQualityReward, 
                                  reflectionReward, consistencyReward);
    }
    
    /**
     * 计算准确性奖励
     */
    private float computeAccuracyReward(Variable logits, NdArray targetIds) {
        // 简化的交叉熵计算
        NdArray logitsData = logits.getValue();
        float totalLoss = 0.0f;
        int count = 0;
        
        int batchSize = logitsData.getShape().getDimension(0);
        int seqLen = logitsData.getShape().getDimension(1);
        int vocabSize = logitsData.getShape().getDimension(2);
        
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < Math.min(seqLen, targetIds.getShape().getDimension(1)); s++) {
                int targetToken = (int) targetIds.get(b, s);
                if (targetToken >= 0 && targetToken < vocabSize) {
                    float logit = logitsData.get(b, s, targetToken);
                    totalLoss += -Math.log(Math.max(logit, 1e-8)); // 避免log(0)
                    count++;
                }
            }
        }
        
        float avgLoss = count > 0 ? totalLoss / count : 1.0f;
        return Math.max(0.0f, 1.0f - avgLoss); // 转换为奖励
    }
    
    /**
     * 计算一致性奖励
     */
    private float computeConsistencyReward(DeepSeekR1Block.DeepSeekR1Result modelOutput) {
        // 简化实现：基于推理输出和Transformer输出的相似性
        Variable reasoningOutput = modelOutput.getReasoningOutput();
        Variable transformerOutput = modelOutput.getTransformerOutput();
        
        // 计算余弦相似度（简化版）
        float similarity = computeCosineSimilarity(reasoningOutput, transformerOutput);
        return Math.max(0.0f, similarity);
    }
    
    /**
     * 计算余弦相似度（简化版）
     */
    private float computeCosineSimilarity(Variable a, Variable b) {
        NdArray aData = a.getValue();
        NdArray bData = b.getValue();
        
        // 简化计算：只使用第一个batch的平均值
        if (aData.getShape().getDimNum() == 2 && bData.getShape().getDimNum() == 3) {
            // 如果b是3维的，取平均值
            bData = meanAlongSequence(bData);
        }
        
        float dotProduct = 0.0f;
        float normA = 0.0f;
        float normB = 0.0f;
        
        int size = Math.min(aData.getShape().size(), bData.getShape().size());
        
        for (int i = 0; i < size; i++) {
            // 使用get()方法替代getDataArrayByFlattenIndex
            float valA = aData.getNumber().floatValue(); // 简化处理，只取第一个值
            float valB = bData.getNumber().floatValue(); // 简化处理，只取第一个值
            
            dotProduct += valA * valB;
            normA += valA * valA;
            normB += valB * valB;
        }
        
        float denominator = (float) (Math.sqrt(normA) * Math.sqrt(normB));
        return denominator > 1e-8 ? dotProduct / denominator : 0.0f;
    }
    
    /**
     * 将3维数组按序列维度取平均
     */
    private NdArray meanAlongSequence(NdArray input) {
        Shape inputShape = input.getShape();
        if (inputShape.getDimNum() != 3) {
            return input;
        }
        
        int batchSize = inputShape.getDimension(0);
        int seqLen = inputShape.getDimension(1);
        int dModel = inputShape.getDimension(2);
        
        NdArray output = NdArray.zeros(Shape.of(batchSize, dModel));
        
        for (int b = 0; b < batchSize; b++) {
            for (int d = 0; d < dModel; d++) {
                float sum = 0.0f;
                for (int s = 0; s < seqLen; s++) {
                    sum += input.get(b, s, d);
                }
                output.set(sum / seqLen, b, d);
            }
        }
        
        return output;
    }
    
    /**
     * 计算总奖励
     */
    private float computeTotalReward(RewardComponents rewards) {
        return accuracyRewardWeight * rewards.getAccuracyReward() +
               reasoningQualityWeight * rewards.getReasoningQualityReward() +
               reflectionRewardWeight * rewards.getReflectionReward() +
               consistencyRewardWeight * rewards.getConsistencyReward();
    }
    
    /**
     * 更新基线
     */
    private void updateBaseline(float reward) {
        runningBaseline = baselineDecay * runningBaseline + (1 - baselineDecay) * reward;
    }
    
    /**
     * 计算策略梯度损失
     */
    private Variable computePolicyLoss(DeepSeekR1Block.DeepSeekR1Result modelOutput, 
                                     NdArray targetIds, float advantage) {
        Variable logits = modelOutput.getLogits();
        
        // 计算log概率
        Variable logProbs = logits.softMax().log();
        
        // 选择目标token的log概率
        Variable selectedLogProbs = selectTargetLogProbs(logProbs, targetIds);
        
        // REINFORCE损失：-log_prob * advantage
        NdArray scaledLogProbs = selectedLogProbs.getValue().mulNum(-advantage);
        Variable policyLoss = new Variable(scaledLogProbs.sum().divNum(scaledLogProbs.getShape().size()));
        
        return policyLoss;
    }
    
    /**
     * 选择目标token的log概率
     */
    private Variable selectTargetLogProbs(Variable logProbs, NdArray targetIds) {
        NdArray logProbsData = logProbs.getValue();
        NdArray targetData = targetIds;
        
        int batchSize = logProbsData.getShape().getDimension(0);
        int seqLen = Math.min(logProbsData.getShape().getDimension(1), 
                             targetData.getShape().getDimension(1));
        
        NdArray selectedProbs = NdArray.zeros(Shape.of(batchSize, seqLen));
        
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                int targetToken = (int) targetData.get(b, s);
                if (targetToken >= 0 && targetToken < logProbsData.getShape().getDimension(2)) {
                    selectedProbs.set(logProbsData.get(b, s, targetToken), b, s);
                }
            }
        }
        
        return new Variable(selectedProbs);
    }
    
    /**
     * 计算价值函数损失
     */
    private Variable computeValueLoss(float actualReward) {
        // 简化的价值函数损失：MSE between predicted and actual reward
        float predictedValue = runningBaseline;
        float loss = (actualReward - predictedValue) * (actualReward - predictedValue);
        
        return new Variable(NdArray.of(loss));
    }
    
    /**
     * 计算熵损失
     */
    private Variable computeEntropyLoss(Variable logits) {
        Variable probs = logits.softMax();
        Variable logProbs = probs.log();
        
        // 熵 = -sum(p * log(p))
        Variable entropy = new Variable(probs.mul(logProbs).sum().getValue().mulNum(-1.0f));
        
        return entropy;
    }
    
    /**
     * 梯度裁剪
     */
    private void clipGradients(float maxNorm) {
        // 这里需要实现梯度裁剪逻辑
        // 由于TinyAI框架的限制，这里是简化实现
        Map<String, io.leavesfly.tinyai.nnet.Parameter> params = deepseekModel.getAllParams();
        
        float totalNorm = 0.0f;
        
        // 计算梯度的总范数（简化处理）
        for (io.leavesfly.tinyai.nnet.Parameter param : params.values()) {
            if (param.getGrad() != null) {
                // 直接使用getGrad()返回的NdArray
                NdArray grad = param.getGrad();
                // 简化处理：只使用每个参数的第一个值
                float g = grad.getNumber().floatValue();
                totalNorm += g * g;
            }
        }
        
        totalNorm = (float) Math.sqrt(totalNorm);
        
        // 如果范数超过阈值，进行裁剪（简化处理）
        if (totalNorm > maxNorm) {
            float scale = maxNorm / totalNorm;
            for (io.leavesfly.tinyai.nnet.Parameter param : params.values()) {
                if (param.getGrad() != null) {
                    // 直接使用getGrad()返回的NdArray
                    NdArray grad = param.getGrad();
                    // 简化处理：创建新的梯度数组并设置
                    param.setGrad(grad.mulNum(scale));
                }
            }
        }
    }
    
    /**
     * 执行参数更新
     */
    private void performParameterUpdate() {
        // 由于无法直接访问optimizer，这里提供一个接口
        // 实际实现中需要调用父类的optimizer.update()
        // 这是一个简化的占位符实现
    }
    
    /**
     * 更新训练指标
     */
    private void updateTrainingMetrics(RewardComponents rewards, Variable policyLoss, 
                                     Variable valueLoss, Variable entropyLoss, float totalReward) {
        trainingMetrics.put("total_reward", totalReward);
        trainingMetrics.put("average_quality", rewards.getReasoningQualityReward());
        trainingMetrics.put("policy_loss", policyLoss.getValue().getNumber().floatValue());
        trainingMetrics.put("value_loss", valueLoss.getValue().getNumber().floatValue());
        trainingMetrics.put("entropy_loss", entropyLoss.getValue().getNumber().floatValue());
        trainingMetrics.put("reflection_improvement", rewards.getReflectionReward());
    }
    
    /**
     * 获取训练步骤指标
     */
    private Map<String, Float> getTrainingStepMetrics(RewardComponents rewards, Variable policyLoss, 
                                                     Variable valueLoss, Variable entropyLoss) {
        Map<String, Float> metrics = new HashMap<>();
        metrics.put("accuracy_reward", rewards.getAccuracyReward());
        metrics.put("reasoning_quality", rewards.getReasoningQualityReward());
        metrics.put("reflection_reward", rewards.getReflectionReward());
        metrics.put("consistency_reward", rewards.getConsistencyReward());
        metrics.put("policy_loss", policyLoss.getValue().getNumber().floatValue());
        metrics.put("value_loss", valueLoss.getValue().getNumber().floatValue());
        metrics.put("entropy_loss", entropyLoss.getValue().getNumber().floatValue());
        metrics.put("running_baseline", runningBaseline);
        
        return metrics;
    }
    
    /**
     * 打印训练统计
     */
    public void printTrainingStatistics() {
        System.out.println("=== 强化学习训练统计 ===");
        System.out.println("总奖励: " + trainingMetrics.get("total_reward"));
        System.out.println("平均质量: " + trainingMetrics.get("average_quality"));
        System.out.println("策略损失: " + trainingMetrics.get("policy_loss"));
        System.out.println("价值损失: " + trainingMetrics.get("value_loss"));
        System.out.println("熵损失: " + trainingMetrics.get("entropy_loss"));
        System.out.println("运行基线: " + runningBaseline);
        
        if (!rewardHistory.isEmpty()) {
            float avgReward = rewardHistory.stream().reduce(0.0f, Float::sum) / rewardHistory.size();
            System.out.println("平均历史奖励: " + avgReward);
        }
        
        if (!qualityHistory.isEmpty()) {
            float avgQuality = qualityHistory.stream().reduce(0.0f, Float::sum) / qualityHistory.size();
            System.out.println("平均历史质量: " + avgQuality);
        }
        
        System.out.println("========================");
    }
    
    /**
     * 奖励组件类
     */
    private static class RewardComponents {
        private float accuracyReward;
        private float reasoningQualityReward;
        private float reflectionReward;
        private float consistencyReward;
        
        public RewardComponents(float accuracyReward, float reasoningQualityReward, 
                              float reflectionReward, float consistencyReward) {
            this.accuracyReward = accuracyReward;
            this.reasoningQualityReward = reasoningQualityReward;
            this.reflectionReward = reflectionReward;
            this.consistencyReward = consistencyReward;
        }
        
        // Getters
        public float getAccuracyReward() { return accuracyReward; }
        public float getReasoningQualityReward() { return reasoningQualityReward; }
        public float getReflectionReward() { return reflectionReward; }
        public float getConsistencyReward() { return consistencyReward; }
    }
    
    // Getters and Setters
    public float getBaselineDecay() { return baselineDecay; }
    public void setBaselineDecay(float baselineDecay) { this.baselineDecay = baselineDecay; }
    
    public float getEntropyCoefficient() { return entropyCoefficient; }
    public void setEntropyCoefficient(float entropyCoefficient) { this.entropyCoefficient = entropyCoefficient; }
    
    public float getValueCoefficient() { return valueCoefficient; }
    public void setValueCoefficient(float valueCoefficient) { this.valueCoefficient = valueCoefficient; }
    
    public Map<String, Float> getTrainingMetrics() { return new HashMap<>(trainingMetrics); }
    public List<Float> getRewardHistory() { return new ArrayList<>(rewardHistory); }
    public List<Float> getQualityHistory() { return new ArrayList<>(qualityHistory); }
}