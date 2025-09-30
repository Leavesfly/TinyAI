package io.leavesfly.tinyai.deepseek.r1;

import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.ArrayList;
import java.util.List;

/**
 * 训练指标类
 * 
 * 记录单步训练的各种指标，用于监控训练过程。
 * 
 * @author leavesfly
 * @version 1.0
 */
public class TrainingMetrics {
    
    private int step;                    // 训练步数
    private double reward;               // 奖励值
    private double policyLoss;           // 策略损失
    private double valueLoss;            // 价值函数损失
    private double entropyLoss;          // 熵损失
    private double totalLoss;            // 总损失
    private double averageReward;        // 平均奖励
    private double averageLoss;          // 平均损失
    private int stepCount;               // 推理步骤数
    private double confidence;           // 置信度
    private boolean isError;             // 是否出错
    private String errorMessage;         // 错误信息
    
    public TrainingMetrics() {
        this.isError = false;
    }
    
    public String getSummary() {
        if (isError) {
            return String.format("Step %d: ERROR - %s", step, errorMessage);
        }
        
        return String.format(
            "Step %d: R=%.3f, L=%.3f (P=%.3f, V=%.3f, E=%.3f), Steps=%d, Conf=%.3f",
            step, reward, totalLoss, policyLoss, valueLoss, entropyLoss, stepCount, confidence
        );
    }
    
    // Getter和Setter方法
    public int getStep() { return step; }
    public void setStep(int step) { this.step = step; }
    
    public double getReward() { return reward; }
    public void setReward(double reward) { this.reward = reward; }
    
    public double getPolicyLoss() { return policyLoss; }
    public void setPolicyLoss(double policyLoss) { this.policyLoss = policyLoss; }
    
    public double getValueLoss() { return valueLoss; }
    public void setValueLoss(double valueLoss) { this.valueLoss = valueLoss; }
    
    public double getEntropyLoss() { return entropyLoss; }
    public void setEntropyLoss(double entropyLoss) { this.entropyLoss = entropyLoss; }
    
    public double getTotalLoss() { return totalLoss; }
    public void setTotalLoss(double totalLoss) { this.totalLoss = totalLoss; }
    
    public double getAverageReward() { return averageReward; }
    public void setAverageReward(double averageReward) { this.averageReward = averageReward; }
    
    public double getAverageLoss() { return averageLoss; }
    public void setAverageLoss(double averageLoss) { this.averageLoss = averageLoss; }
    
    public int getStepCount() { return stepCount; }
    public void setStepCount(int stepCount) { this.stepCount = stepCount; }
    
    public double getConfidence() { return confidence; }
    public void setConfidence(double confidence) { this.confidence = confidence; }
    
    public boolean isError() { return isError; }
    public void setError(boolean error) { isError = error; }
    
    public String getErrorMessage() { return errorMessage; }
    public void setErrorMessage(String errorMessage) { this.errorMessage = errorMessage; }
}

/**
 * 训练历史类
 * 
 * 记录整个训练过程的历史指标。
 */
class TrainingHistory {
    
    private List<TrainingMetrics> history;
    
    public TrainingHistory() {
        this.history = new ArrayList<>();
    }
    
    public void addMetrics(TrainingMetrics metrics) {
        history.add(metrics);
    }
    
    public List<TrainingMetrics> getHistory() {
        return new ArrayList<>(history);
    }
    
    public double getSuccessRate() {
        if (history.isEmpty()) {
            return 0.0;
        }
        
        long successCount = history.stream().mapToLong(m -> m.isError() ? 0 : 1).sum();
        return (double) successCount / history.size();
    }
    
    public double getAverageReward() {
        return history.stream()
                     .filter(m -> !m.isError())
                     .mapToDouble(TrainingMetrics::getReward)
                     .average()
                     .orElse(0.0);
    }
    
    public double getAverageLoss() {
        return history.stream()
                     .filter(m -> !m.isError())
                     .mapToDouble(TrainingMetrics::getTotalLoss)
                     .average()
                     .orElse(0.0);
    }
    
    public int getTotalSteps() {
        return history.size();
    }
}

/**
 * 训练数据类
 */
class TrainingData {
    
    private List<Sample> samples;
    
    public TrainingData() {
        this.samples = new ArrayList<>();
    }
    
    public void addSample(NdArray inputIds, NdArray targetIds, String question) {
        samples.add(new Sample(inputIds, targetIds, question));
    }
    
    public Sample getSample(int index) {
        return samples.get(index);
    }
    
    public int size() {
        return samples.size();
    }
    
    public static class Sample {
        private NdArray inputIds;
        private NdArray targetIds;
        private String question;
        
        public Sample(NdArray inputIds, NdArray targetIds, String question) {
            this.inputIds = inputIds;
            this.targetIds = targetIds;
            this.question = question;
        }
        
        public NdArray getInputIds() { return inputIds; }
        public NdArray getTargetIds() { return targetIds; }
        public String getQuestion() { return question; }
    }
}

/**
 * 评估结果类
 */
class EvaluationResult {
    
    private List<Double> rewards;
    private List<DeepSeekR1Result> results;
    private double averageReward;
    private double averageConfidence;
    private double successRate;
    
    public EvaluationResult() {
        this.rewards = new ArrayList<>();
        this.results = new ArrayList<>();
    }
    
    public void addSample(double reward, DeepSeekR1Result result) {
        rewards.add(reward);
        results.add(result);
    }
    
    public void computeStatistics() {
        if (rewards.isEmpty()) {
            return;
        }
        
        averageReward = rewards.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        
        averageConfidence = results.stream()
                                  .mapToDouble(DeepSeekR1Result::getFinalConfidence)
                                  .average()
                                  .orElse(0.0);
        
        long successCount = results.stream().mapToLong(r -> r.isSuccessful() ? 1 : 0).sum();
        successRate = (double) successCount / results.size();
    }
    
    public String getSummary() {
        return String.format(
            "评估结果: 样本数=%d, 平均奖励=%.3f, 平均置信度=%.3f, 成功率=%.1f%%",
            rewards.size(), averageReward, averageConfidence, successRate * 100
        );
    }
    
    public double getAverageReward() { return averageReward; }
    public double getAverageConfidence() { return averageConfidence; }
    public double getSuccessRate() { return successRate; }
    public List<Double> getRewards() { return new ArrayList<>(rewards); }
    public List<DeepSeekR1Result> getResults() { return new ArrayList<>(results); }
}