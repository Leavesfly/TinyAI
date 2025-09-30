package io.leavesfly.tinyai.deepseek.v3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.*;

/**
 * DeepSeek V3 强化学习训练器（简化版）
 * 
 * 实现基于REINFORCE算法的V3模型训练，包含：
 * 1. V3增强奖励信号计算
 * 2. 策略梯度优化（简化）
 * 3. MoE负载均衡损失
 * 4. 训练指标统计
 * 
 * @author leavesfly
 * @version 1.0
 */
public class V3RLTrainer {
    
    private DeepSeekV3Model model;
    private double learningRate;
    private double moeLocationWeight;
    private double codeQualityWeight;
    private double reasoningQualityWeight;
    
    private List<TrainingMetrics> trainingHistory;
    private int currentEpoch;
    private double bestReward;
    private long totalTrainingSteps;
    
    /**
     * 构造函数
     */
    public V3RLTrainer(DeepSeekV3Model model, double learningRate) {
        this.model = model;
        this.learningRate = learningRate;
        this.moeLocationWeight = 0.01;
        this.codeQualityWeight = 0.2;
        this.reasoningQualityWeight = 0.3;
        
        this.trainingHistory = new ArrayList<>();
        this.currentEpoch = 0;
        this.bestReward = Double.NEGATIVE_INFINITY;
        this.totalTrainingSteps = 0;
    }
    
    public V3RLTrainer(DeepSeekV3Model model) {
        this(model, 2e-5);
    }
    
    /**
     * 执行单步训练
     */
    public TrainingMetrics trainStep(Variable inputIds, Variable targetIds, TaskType taskType) {
        long startTime = System.currentTimeMillis();
        
        // 前向传播
        Variable output = model.layerForward(inputIds);
        
        // 计算V3增强奖励
        V3RewardResult rewardResult = computeV3Reward(output, targetIds, taskType);
        double totalReward = rewardResult.getTotalReward();
        
        // 计算策略损失（简化）
        double policyLoss = computePolicyLoss(output, targetIds, totalReward);
        
        // 计算MoE负载均衡损失
        double moeLoss = model.computeTotalLoadBalancingLoss() * moeLocationWeight;
        
        // 总损失
        double totalLoss = policyLoss + moeLoss;
        
        // 模拟梯度更新
        simulateGradientUpdate();
        
        totalTrainingSteps++;
        long endTime = System.currentTimeMillis();
        
        TrainingMetrics metrics = new TrainingMetrics(
            totalTrainingSteps,
            totalLoss,
            policyLoss,
            moeLoss,
            totalReward,
            rewardResult.getReasoningQuality(),
            taskType,
            rewardResult.getCodeQuality(),
            endTime - startTime
        );
        
        trainingHistory.add(metrics);
        
        if (totalReward > bestReward) {
            bestReward = totalReward;
            metrics.setNewBest(true);
        }
        
        return metrics;
    }
    
    /**
     * 计算V3增强奖励信号
     */
    public V3RewardResult computeV3Reward(Variable modelOutput, Variable targetOutput, TaskType taskType) {
        // 基础准确性奖励
        double accuracyReward = computeAccuracyReward(modelOutput, targetOutput);
        
        // 推理质量奖励
        double reasoningQuality = computeReasoningQualityReward();
        double reasoningReward = reasoningQuality * reasoningQualityWeight;
        
        // 任务特定奖励
        double taskReward = computeTaskSpecificReward(taskType);
        
        // MoE效率奖励
        double moeEfficiency = 1.0 - model.computeTotalLoadBalancingLoss();
        double moeReward = Math.max(0.0, moeEfficiency) * 0.1;
        
        // 总奖励
        double totalReward = accuracyReward + reasoningReward + taskReward + moeReward;
        
        return new V3RewardResult(
            totalReward,
            accuracyReward,
            reasoningQuality,
            getCodeQuality(taskType),
            moeEfficiency,
            0.0 // 简化验证奖励
        );
    }
    
    private double computeAccuracyReward(Variable modelOutput, Variable targetOutput) {
        // 简化实现：使用负均方误差
        try {
            NdArray outputData = modelOutput.getValue();
            NdArray targetData = targetOutput.getValue();
            
            double[] outputArray = outputData.toDoubleArray();
            double[] targetArray = targetData.toDoubleArray();
            
            double mse = 0.0;
            int minLength = Math.min(outputArray.length, targetArray.length);
            
            for (int i = 0; i < minLength; i++) {
                double diff = outputArray[i] - targetArray[i];
                mse += diff * diff;
            }
            
            if (minLength > 0) {
                mse /= minLength;
                return Math.max(-1.0, -mse);
            }
        } catch (Exception e) {
            return -0.5;
        }
        
        return -0.1;
    }
    
    private double computeReasoningQualityReward() {
        List<V3ReasoningStep> reasoningChain = model.getCurrentReasoningChain();
        if (reasoningChain.isEmpty()) {
            return 0.0;
        }
        
        double totalConfidence = 0.0;
        for (V3ReasoningStep step : reasoningChain) {
            totalConfidence += step.getConfidence();
        }
        
        return totalConfidence / reasoningChain.size();
    }
    
    private double computeTaskSpecificReward(TaskType taskType) {
        if (taskType == null) return 0.0;
        
        switch (taskType) {
            case CODING:
                return getCodeQuality(taskType) * codeQualityWeight;
            case REASONING:
            case MATH:
                return 0.1; // 复杂推理奖励
            default:
                return 0.0;
        }
    }
    
    private double getCodeQuality(TaskType taskType) {
        if (taskType != TaskType.CODING) return 0.0;
        
        CodeGenerationModule.CodeGenerationResult codeInfo = model.getCodeInfo();
        return codeInfo != null ? codeInfo.getCodeConfidence() : 0.0;
    }
    
    private double computePolicyLoss(Variable output, Variable target, double reward) {
        // 简化的REINFORCE损失
        NdArray outputData = output.getValue();
        double[] outputArray = outputData.toDoubleArray();
        
        double mean = Arrays.stream(outputArray).average().orElse(0.0);
        double variance = Arrays.stream(outputArray)
            .map(x -> Math.pow(x - mean, 2))
            .average().orElse(1.0);
        
        double logProb = -Math.log(Math.max(1e-8, variance));
        return -logProb * reward;
    }
    
    private void simulateGradientUpdate() {
        // 简化实现：重置MoE统计
        model.resetAllMoEStats();
    }
    
    /**
     * 获取训练报告
     */
    public String getTrainingReport() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== V3强化学习训练报告 ===\n");
        sb.append(String.format("总训练步数: %d\n", totalTrainingSteps));
        sb.append(String.format("当前轮次: %d\n", currentEpoch));
        sb.append(String.format("最佳奖励: %.6f\n", bestReward));
        sb.append(String.format("学习率: %.6f\n", learningRate));
        
        if (!trainingHistory.isEmpty()) {
            TrainingMetrics lastMetrics = trainingHistory.get(trainingHistory.size() - 1);
            sb.append(String.format("\n最近训练指标:\n"));
            sb.append(String.format("  - 总损失: %.6f\n", lastMetrics.getTotalLoss()));
            sb.append(String.format("  - 总奖励: %.6f\n", lastMetrics.getTotalReward()));
            sb.append(String.format("  - 推理质量: %.4f\n", lastMetrics.getReasoningQuality()));
        }
        
        sb.append("========================");
        return sb.toString();
    }
    
    // ========== Inner Classes ==========
    
    public static class V3RewardResult {
        private double totalReward;
        private double accuracyReward;
        private double reasoningQuality;
        private double codeQuality;
        private double moeEfficiency;
        private double verificationReward;
        
        public V3RewardResult(double totalReward, double accuracyReward, 
                             double reasoningQuality, double codeQuality,
                             double moeEfficiency, double verificationReward) {
            this.totalReward = totalReward;
            this.accuracyReward = accuracyReward;
            this.reasoningQuality = reasoningQuality;
            this.codeQuality = codeQuality;
            this.moeEfficiency = moeEfficiency;
            this.verificationReward = verificationReward;
        }
        
        public double getTotalReward() { return totalReward; }
        public double getAccuracyReward() { return accuracyReward; }
        public double getReasoningQuality() { return reasoningQuality; }
        public double getCodeQuality() { return codeQuality; }
        public double getMoeEfficiency() { return moeEfficiency; }
        public double getVerificationReward() { return verificationReward; }
    }
    
    public static class TrainingMetrics {
        private long step;
        private double totalLoss;
        private double policyLoss;
        private double moeLoss;
        private double totalReward;
        private double reasoningQuality;
        private TaskType taskType;
        private double codeQuality;
        private long executionTimeMs;
        private boolean newBest;
        
        public TrainingMetrics(long step, double totalLoss, double policyLoss,
                              double moeLoss, double totalReward, double reasoningQuality,
                              TaskType taskType, double codeQuality, long executionTimeMs) {
            this.step = step;
            this.totalLoss = totalLoss;
            this.policyLoss = policyLoss;
            this.moeLoss = moeLoss;
            this.totalReward = totalReward;
            this.reasoningQuality = reasoningQuality;
            this.taskType = taskType;
            this.codeQuality = codeQuality;
            this.executionTimeMs = executionTimeMs;
            this.newBest = false;
        }
        
        // Getters and Setters
        public long getStep() { return step; }
        public double getTotalLoss() { return totalLoss; }
        public double getPolicyLoss() { return policyLoss; }
        public double getMoeLoss() { return moeLoss; }
        public double getTotalReward() { return totalReward; }
        public double getReasoningQuality() { return reasoningQuality; }
        public TaskType getTaskType() { return taskType; }
        public double getCodeQuality() { return codeQuality; }
        public long getExecutionTimeMs() { return executionTimeMs; }
        public boolean isNewBest() { return newBest; }
        public void setNewBest(boolean newBest) { this.newBest = newBest; }
    }
    
    // ========== Getter Methods ==========
    
    public DeepSeekV3Model getModel() { return model; }
    public double getLearningRate() { return learningRate; }
    public List<TrainingMetrics> getTrainingHistory() { return trainingHistory; }
    public double getBestReward() { return bestReward; }
    public long getTotalTrainingSteps() { return totalTrainingSteps; }
}