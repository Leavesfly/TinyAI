package io.leavesfly.tinyai.deepseek.r1.training;

import io.leavesfly.tinyai.deepseek.r1.DeepSeekR1Model;
import io.leavesfly.tinyai.deepseek.r1.training.dataset.DeepSeekR1Dataset;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.loss.Loss;
import io.leavesfly.tinyai.ml.loss.SoftmaxCrossEntropy;
import io.leavesfly.tinyai.ml.optimize.SGD;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
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
 * 采用行业标准的 Reward-weighted Regression（奖励加权回归）算法：
 * L = -reward * sum(mask * CE_loss) / sum(mask)
 * 
 * 核心改进（对比简单奖励加权）：
 * - 使用 Chat Template 格式的结构化数据
 * - 结合 Answer-only Loss Mask，只对 assistant 回复部分计算 loss
 * - 奖励加权：高奖励样本的梯度更大，低奖励样本的梯度更小
 * - 支持 MoE 质量奖励作为辅助信号
 * 
 * 算法原理：
 * 对于每个样本 (x, reward)，计算 assistant 区域的 CE loss，
 * 然后乘以 reward 作为权重。高奖励样本被强化，低奖励样本被弱化。
 * 
 * @author leavesfly
 * @version 2.0
 */
public class DeepSeekR1RLHFTrainer {
    
    private final DeepSeekR1Model model;
    private final DeepSeekR1Dataset dataset;
    private final SoftmaxCrossEntropy elementWiseLossFunction;
    private final SGD optimizer;
    
    private int maxEpochs;
    private float learningRate;
    private float maxGradNorm;
    private float rewardWeight;
    private float qualityWeight;
    private int logInterval;
    private String checkpointDir;
    
    private int currentEpoch;
    private int globalStep;
    private List<Float> rewardHistory;
    private List<Float> lossHistory;
    
    public DeepSeekR1RLHFTrainer(DeepSeekR1Model model, DeepSeekR1Dataset dataset) {
        this.model = model;
        this.dataset = dataset;
        this.elementWiseLossFunction = new SoftmaxCrossEntropy(Loss.Reduction.NONE);
        
        this.maxEpochs = 3;
        this.learningRate = 1e-5f;
        this.maxGradNorm = 0.5f;
        this.rewardWeight = 1.0f;
        this.qualityWeight = 0.5f;
        this.logInterval = 20;
        this.checkpointDir = "./checkpoints/deepseek_r1_rlhf";
        
        this.optimizer = new SGD(model, learningRate);
        
        this.currentEpoch = 0;
        this.globalStep = 0;
        this.rewardHistory = new ArrayList<>();
        this.lossHistory = new ArrayList<>();
    }
    
    public DeepSeekR1RLHFTrainer configure(int maxEpochs, float learningRate,
                                           float rewardWeight, float qualityWeight) {
        this.maxEpochs = maxEpochs;
        this.learningRate = learningRate;
        this.rewardWeight = rewardWeight;
        this.qualityWeight = qualityWeight;
        this.optimizer.setLearningRate(learningRate);
        return this;
    }
    
    public void train() {
        System.out.println("=".repeat(70));
        System.out.println("DeepSeek-R1 强化学习训练 (RLHF - Reward-weighted Regression)");
        System.out.println("=".repeat(70));
        System.out.println("模型: " + model.getName());
        System.out.println("训练样本: " + dataset.getSampleCount());
        System.out.println("学习率: " + learningRate);
        System.out.println("奖励权重: " + rewardWeight);
        System.out.println("质量权重: " + qualityWeight);
        System.out.println("Loss Mask: " + (dataset.hasLossMasks() ? "启用（Answer-only Loss）" : "未启用"));
        System.out.println("算法: Reward-weighted Regression");
        System.out.println("=".repeat(70));
        
        createCheckpointDir();
        
        for (currentEpoch = 0; currentEpoch < maxEpochs; currentEpoch++) {
            trainOneEpoch();
        }
        
        saveCheckpoint("final");
        System.out.println("\nRLHF训练完成!");
    }
    
    /**
     * 训练一个 epoch
     * 
     * Reward-weighted Regression 核心逻辑：
     * 1. 前向传播获取 logits
     * 2. 计算逐位置 CE loss
     * 3. 应用 Answer-only Loss Mask（只对 assistant 回复部分计算）
     * 4. 乘以奖励权重：高奖励样本梯度更大
     * 5. 反向传播更新参数
     */
    private void trainOneEpoch() {
        dataset.prepare(true);
        
        double epochReward = 0.0;
        double epochLoss = 0.0;
        int count = 0;
        
        while (dataset.hasNext()) {
            DeepSeekR1Dataset.Batch batch = dataset.nextBatch();
            
            NdArray inputIds = batch.getInputIds();
            NdArray targetIds = batch.getTargetIds();
            
            // 前向传播
            Variable inputVar = new Variable(inputIds);
            DeepSeekR1Model.ReasoningResult result = model.performReasoning(inputVar);
            
            // Reshape logits 为 2D: [batchSize * seqLen, vocabSize]
            int[] logitsShape = result.logits.getValue().getShape().getShapeDims();
            int batchSize = logitsShape[0];
            int seqLen = logitsShape[1];
            int vocabSize = logitsShape[2];
            
            Variable logits2D = result.logits.reshape(Shape.of(batchSize * seqLen, vocabSize));
            Variable targetVar = new Variable(targetIds.reshape(Shape.of(batchSize * seqLen, 1)));
            
            // 计算逐元素 CE loss
            Variable elementWiseLoss = elementWiseLossFunction.loss(targetVar, logits2D);
            
            // 计算奖励信号
            float[] humanRewards = batch.getRewards();
            float avgHumanReward = calculateAverage(humanRewards);
            float moeLossReward = (float) (1.0 - result.moeLoss);
            float totalReward = rewardWeight * avgHumanReward + qualityWeight * moeLossReward;
            
            // 应用 Loss Mask 和奖励加权
            Variable loss;
            if (batch.hasLossMask()) {
                // Answer-only Loss + Reward weighting
                NdArray maskFlat = batch.getLossMask().reshape(Shape.of(batchSize * seqLen, 1));
                Variable maskVar = new Variable(maskFlat);
                Variable maskedLoss = elementWiseLoss.mul(maskVar);
                
                float maskSum = maskFlat.sum().getNumber().floatValue();
                float effectiveCount = Math.max(maskSum, 1.0f);
                
                // Reward-weighted: loss = reward * masked_CE / count
                Variable avgMaskedLoss = maskedLoss.sum().div(new Variable(NdArray.of(new float[]{effectiveCount})));
                loss = avgMaskedLoss.mul(new Variable(NdArray.of(new float[]{totalReward})));
            } else {
                // 无 mask 时直接对全序列 loss 加权
                Variable avgLoss = elementWiseLoss.mean(-1, false).mean(-1, false);
                loss = avgLoss.mul(new Variable(NdArray.of(new float[]{totalReward})));
            }
            
            float lossValue = loss.getValue().getNumber().floatValue();
            
            // 反向传播
            model.clearGrads();
            loss.backward();
            clipGradients();
            optimizer.update();
            
            // 释放计算图
            loss.unChainBackward();
            result.logits.unChainBackward();
            inputVar.unChainBackward();
            
            rewardHistory.add(totalReward);
            lossHistory.add(lossValue);
            
            epochReward += totalReward;
            epochLoss += lossValue;
            count++;
            globalStep++;
            
            if (globalStep % logInterval == 0) {
                System.out.printf("Epoch %d | Step %d | Loss: %.4f | Reward: %.4f%n",
                    currentEpoch + 1, globalStep, lossValue, totalReward);
            }
        }
        
        System.out.printf("Epoch %d 完成 | 平均Loss: %.4f | 平均奖励: %.4f%n",
            currentEpoch + 1, 
            count > 0 ? epochLoss / count : 0.0,
            count > 0 ? epochReward / count : 0.0);
        
        dataset.reset();
        System.gc();
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
