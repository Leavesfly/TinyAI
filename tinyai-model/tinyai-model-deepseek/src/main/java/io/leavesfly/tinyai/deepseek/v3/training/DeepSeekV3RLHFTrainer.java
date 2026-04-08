package io.leavesfly.tinyai.deepseek.v3.training;

import io.leavesfly.tinyai.deepseek.base.DeepSeekTrainerBase;
import io.leavesfly.tinyai.deepseek.v3.DeepSeekV3Block;
import io.leavesfly.tinyai.deepseek.v3.DeepSeekV3Config;
import io.leavesfly.tinyai.deepseek.v3.DeepSeekV3Model;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.loss.SoftmaxCrossEntropy;
import io.leavesfly.tinyai.ml.optimize.Adam;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.ArrayList;
import java.util.List;

/**
 * DeepSeek-V3 强化学习训练器 (RLHF - Reinforcement Learning from Human Feedback)
 * 
 * DeepSeek-V3 的标准训练流程为：预训练 → 监督微调(SFT) → 强化学习(RLHF)。
 * RLHF 是 V3 训练流程的最后一步，主要用于对齐人类偏好，
 * 使模型回答更有礼貌、更安全、更符合指令，提升通用任务的流畅度。
 * 
 * 采用行业标准的 Reward-weighted Regression（奖励加权回归）算法：
 * L = -reward * sum(mask * CE_loss) / sum(mask)
 * 
 * 核心特性：
 * - 使用 Chat Template 格式的结构化数据
 * - 结合 Answer-only Loss Mask，只对 assistant 回复部分计算 loss
 * - 奖励加权：高奖励样本的梯度更大，低奖励样本的梯度更小
 * - 支持 MoE 负载均衡损失作为辅助信号
 * - 使用 Adam 优化器（与 V3 Posttrain 保持一致）
 * 
 * 算法原理：
 * 对于每个样本 (x, reward)，计算 assistant 区域的 CE loss，
 * 然后乘以 reward 作为权重。高奖励样本被强化，低奖励样本被弱化。
 * 
 * 本类同时作为 R1 RLHF 训练的底层实现，因为 R1 基于 V3 底座，
 * R1 的 RLHF 训练通过委托本类完成。
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekV3RLHFTrainer extends DeepSeekTrainerBase {
    
    private final DeepSeekV3Model model;
    private final DeepSeekV3Config config;
    private final DeepSeekV3Dataset dataset;
    private final SoftmaxCrossEntropy elementWiseLossFunction;
    private final Adam optimizer;
    
    private float learningRate;
    private float rewardWeight;
    private float moeQualityWeight;
    private float moeLoadBalanceWeight;
    private List<Float> rewardHistory;
    
    /**
     * 构造函数
     * 
     * @param model DeepSeek-V3 模型
     * @param dataset 包含奖励标注的 RLHF 数据集
     */
    public DeepSeekV3RLHFTrainer(DeepSeekV3Model model, DeepSeekV3Dataset dataset) {
        super(model, 3, 0.5f, 20, "./checkpoints/deepseek_v3/rlhf");
        this.model = model;
        this.config = model.getConfig();
        this.dataset = dataset;
        this.elementWiseLossFunction = new SoftmaxCrossEntropy(SoftmaxCrossEntropy.Reduction.NONE);
        
        this.learningRate = 1e-5f;
        this.rewardWeight = 1.0f;
        this.moeQualityWeight = 0.5f;
        this.moeLoadBalanceWeight = (float) config.getLoadBalanceLossWeight();
        
        this.optimizer = new Adam(model, learningRate, 0.9f, 0.999f, 1e-8f);
        
        this.rewardHistory = new ArrayList<>();
    }
    
    /**
     * 配置训练参数
     * 
     * @param maxEpochs 最大训练轮数
     * @param learningRate 学习率
     * @param rewardWeight 人类奖励权重
     * @param moeQualityWeight MoE 质量奖励权重
     * @return 训练器自身（支持链式调用）
     */
    public DeepSeekV3RLHFTrainer configure(int maxEpochs, float learningRate,
                                           float rewardWeight, float moeQualityWeight) {
        this.maxEpochs = maxEpochs;
        this.learningRate = learningRate;
        this.rewardWeight = rewardWeight;
        this.moeQualityWeight = moeQualityWeight;
        this.optimizer.setLearningRate(learningRate);
        return this;
    }
    
    /**
     * 开始 RLHF 训练
     */
    @Override
    public void train() {
        System.out.println("=".repeat(70));
        System.out.println("DeepSeek-V3 强化学习训练 (RLHF - Reward-weighted Regression)");
        System.out.println("=".repeat(70));
        System.out.println("模型: " + model.getName());
        System.out.println("训练样本: " + dataset.getSampleCount());
        System.out.println("学习率: " + learningRate);
        System.out.println("奖励权重: " + rewardWeight);
        System.out.println("MoE质量权重: " + moeQualityWeight);
        System.out.println("MoE负载均衡权重: " + moeLoadBalanceWeight);
        System.out.println("Loss Mask: " + (dataset.hasLossMasks() ? "启用（Answer-only Loss）" : "未启用"));
        System.out.println("算法: Reward-weighted Regression");
        System.out.println("优化器: Adam");
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
     * 1. 前向传播获取 logits 和 MoE 损失
     * 2. 计算逐位置 CE loss
     * 3. 应用 Answer-only Loss Mask（只对 assistant 回复部分计算）
     * 4. 乘以奖励权重：高奖励样本梯度更大
     * 5. 融合 MoE 负载均衡损失
     * 6. 反向传播更新参数
     */
    private void trainOneEpoch() {
        dataset.prepare(true);
        
        double epochReward = 0.0;
        double epochLoss = 0.0;
        int count = 0;
        
        while (dataset.hasNext()) {
            DeepSeekV3Dataset.Batch batch = dataset.nextBatch();
            
            NdArray inputIds = batch.getInputIds();
            NdArray targetIds = batch.getTargetIds();
            
            // 前向传播（获取 logits 和 MoE 损失）
            Variable inputVar = new Variable(inputIds);
            DeepSeekV3Block.DetailedForwardResult result = model.predictWithDetails(inputVar);
            
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
            float moeLossReward = (float) (1.0 - result.avgMoELoss);
            float totalReward = rewardWeight * avgHumanReward + moeQualityWeight * moeLossReward;
            
            // 应用 Loss Mask 和奖励加权
            Variable rewardWeightedLoss;
            if (batch.hasLossMask()) {
                // Answer-only Loss + Reward weighting
                NdArray maskFlat = batch.getLossMask().reshape(Shape.of(batchSize * seqLen, 1));
                Variable maskVar = new Variable(maskFlat);
                Variable maskedLoss = elementWiseLoss.mul(maskVar);
                
                float maskSum = maskFlat.sum().getNumber().floatValue();
                float effectiveCount = Math.max(maskSum, 1.0f);
                
                // Reward-weighted: loss = reward * masked_CE / count
                Variable avgMaskedLoss = maskedLoss.sum().div(new Variable(NdArray.of(new float[]{effectiveCount})));
                rewardWeightedLoss = avgMaskedLoss.mul(new Variable(NdArray.of(new float[]{totalReward})));
            } else {
                // 无 mask 时直接对全序列 loss 加权
                Variable avgLoss = elementWiseLoss.mean(-1, false).mean(-1, false);
                rewardWeightedLoss = avgLoss.mul(new Variable(NdArray.of(new float[]{totalReward})));
            }
            
            // 融合 MoE 负载均衡损失
            Variable totalLoss = buildTotalLoss(rewardWeightedLoss, (float) result.avgMoELoss);
            float lossValue = totalLoss.getValue().getNumber().floatValue();
            
            // 反向传播
            model.clearGrads();
            totalLoss.backward();
            clipGradients();
            optimizer.update();
            
            // 释放计算图
            totalLoss.unChainBackward();
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
    
    /**
     * 融合语言模型损失与 MoE 负载均衡损失
     */
    private Variable buildTotalLoss(Variable lmLoss, float moeLoss) {
        if (moeLoadBalanceWeight <= 0) {
            return lmLoss;
        }
        Variable moeLossVar = new Variable(NdArray.of(new float[]{moeLoss * moeLoadBalanceWeight}));
        return lmLoss.add(moeLossVar);
    }
    
    @Override
    protected String getTrainerName() {
        return "DeepSeek-V3 RLHF";
    }
    
    @Override
    protected String getCheckpointPrefix() {
        return "deepseek_v3_rlhf";
    }
    
    /**
     * 判断数据集是否包含 Loss Mask
     */
    public boolean hasLossMasks() {
        return dataset.hasLossMasks();
    }
}
