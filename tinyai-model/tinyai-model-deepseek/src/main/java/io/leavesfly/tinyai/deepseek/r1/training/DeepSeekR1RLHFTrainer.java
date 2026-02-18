package io.leavesfly.tinyai.deepseek.r1.training;

import io.leavesfly.tinyai.deepseek.r1.DeepSeekR1Model;
import io.leavesfly.tinyai.deepseek.r1.training.dataset.DeepSeekR1Dataset;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.optimize.SGD;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.v2.core.Parameter;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * DeepSeek-R1强化学习训练器(RLHF - Reinforcement Learning from Human Feedback)
 * 
 * 通过人类反馈的强化学习优化推理和反思质量
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekR1RLHFTrainer {
    
    private final DeepSeekR1Model model;
    private final DeepSeekR1Dataset dataset;
    private final SGD optimizer;
    
    private int maxEpochs;
    private float learningRate;
    private float maxGradNorm;
    private float rewardWeight;  // 奖励权重
    private float qualityWeight;  // 质量分数权重
    private int logInterval;
    private String checkpointDir;
    
    private int currentEpoch;
    private int globalStep;
    private List<Float> rewardHistory;
    private List<Float> qualityHistory;
    
    public DeepSeekR1RLHFTrainer(DeepSeekR1Model model, DeepSeekR1Dataset dataset) {
        this.model = model;
        this.dataset = dataset;
        
        // RLHF学习率更小
        this.maxEpochs = 3;
        this.learningRate = 1e-5f;
        this.maxGradNorm = 0.5f;
        this.rewardWeight = 1.0f;
        this.qualityWeight = 0.5f;
        this.logInterval = 20;
        this.checkpointDir = "./checkpoints/deepseek_r1_rlhf";
        
        // 使用SGD替代Adam，减少临时NdArray对象创建，降低内存占用
        this.optimizer = new SGD(model, learningRate);
        
        this.currentEpoch = 0;
        this.globalStep = 0;
        this.rewardHistory = new ArrayList<>();
        this.qualityHistory = new ArrayList<>();
    }
    
    public DeepSeekR1RLHFTrainer configure(int maxEpochs, float learningRate,
                                           float rewardWeight, float qualityWeight) {
        this.maxEpochs = maxEpochs;
        this.learningRate = learningRate;
        this.rewardWeight = rewardWeight;
        this.qualityWeight = qualityWeight;
        // 同步学习率到优化器
        this.optimizer.setLearningRate(learningRate);
        return this;
    }
    
    public void train() {
        System.out.println("=".repeat(70));
        System.out.println("DeepSeek-R1 强化学习训练 (RLHF)");
        System.out.println("=".repeat(70));
        System.out.println("模型: " + model.getName());
        System.out.println("训练样本: " + dataset.getSampleCount());
        System.out.println("学习率: " + learningRate);
        System.out.println("奖励权重: " + rewardWeight);
        System.out.println("质量权重: " + qualityWeight);
        System.out.println("=".repeat(70));
        
        createCheckpointDir();
        
        for (currentEpoch = 0; currentEpoch < maxEpochs; currentEpoch++) {
            trainOneEpoch();
        }
        
        saveCheckpoint("final");
        System.out.println("\nRLHF训练完成!");
    }
    
    private void trainOneEpoch() {
        dataset.prepare(true);
        
        double epochReward = 0.0;
        double epochQuality = 0.0;
        int count = 0;
        
        while (dataset.hasNext()) {
            DeepSeekR1Dataset.Batch batch = dataset.nextBatch();
            
            // 前向传播获取推理结果
            Variable inputVar = new Variable(batch.getInputIds());
            DeepSeekR1Model.ReasoningResult result = model.performReasoning(inputVar);
            
            // 计算奖励信号
            float[] humanRewards = batch.getRewards();
            double moeLossReward = 1.0 - result.moeLoss;  // MoE损失越小，奖励越大
            
            // 综合奖励 = 人类反馈 + MoE质量奖励
            float avgHumanReward = calculateAverage(humanRewards);
            float qualityReward = (float) moeLossReward;
            float totalReward = rewardWeight * avgHumanReward + qualityWeight * qualityReward;
            
            // 构建损失：负奖励（最大化奖励）
            Variable rewardVar = new Variable(NdArray.of(-totalReward));
            
            // 反向传播
            model.clearGrads();
            rewardVar.backward();
            clipGradients();
            optimizer.update();
            
            rewardHistory.add(totalReward);
            qualityHistory.add(qualityReward);
            
            epochReward += totalReward;
            epochQuality += qualityReward;
            count++;
            globalStep++;
            
            if (globalStep % logInterval == 0) {
                System.out.printf("Epoch %d | Step %d | Reward: %.4f | Quality: %.4f%n",
                    currentEpoch + 1, globalStep, totalReward, qualityReward);
            }
        }
        
        System.out.printf("Epoch %d 完成 | 平均奖励: %.4f | 平均质量: %.4f%n",
            currentEpoch + 1, epochReward / count, epochQuality / count);
        
        dataset.reset();
    }
    
    private float calculateAverage(float[] values) {
        if (values == null || values.length == 0) return 0.0f;
        float sum = 0.0f;
        for (float v : values) sum += v;
        return sum / values.length;
    }
    
    private void clipGradients() {
        double totalNorm = 0.0;
        Map<String, Parameter> params = model.getModule().namedParameters("", true);
        
        for (Parameter param : params.values()) {
            if (param.grad() != null) {
                double norm = param.grad().mul(param.grad()).sum().getNumber().doubleValue();
                totalNorm += norm;
            }
        }
        
        totalNorm = Math.sqrt(totalNorm);
        
        if (totalNorm > maxGradNorm) {
            float scale = (float) (maxGradNorm / totalNorm);
            for (Parameter param : params.values()) {
                if (param.grad() != null) {
                    param.setGrad(param.grad().mulNum(scale));
                }
            }
        }
    }
    
    private void saveCheckpoint(String suffix) {
        try {
            String filepath = checkpointDir + File.separator +
                            String.format("deepseek_r1_rlhf_%s.model", suffix);
            model.saveModel(filepath);
            System.out.println("检查点已保存: " + filepath);
        } catch (Exception e) {
            System.err.println("保存失败: " + e.getMessage());
        }
    }
    
    private void createCheckpointDir() {
        try {
            Files.createDirectories(Paths.get(checkpointDir));
        } catch (Exception e) {
            System.err.println("创建目录失败: " + e.getMessage());
        }
    }
}
