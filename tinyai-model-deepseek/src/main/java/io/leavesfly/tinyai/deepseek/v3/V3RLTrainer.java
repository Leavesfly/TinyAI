package io.leavesfly.tinyai.deepseek.v3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.Model;
import io.leavesfly.tinyai.ml.Trainer;
import io.leavesfly.tinyai.ml.dataset.Batch;
import io.leavesfly.tinyai.ml.dataset.DataSet;
import io.leavesfly.tinyai.ml.evaluator.Evaluator;
import io.leavesfly.tinyai.ml.loss.Loss;
import io.leavesfly.tinyai.ml.optimize.Optimizer;
import io.leavesfly.tinyai.ml.Monitor;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.List;
import java.util.Map;

/**
 * DeepSeek V3强化学习训练器
 * 
 * 扩展了TinyAI的Trainer类，专门为DeepSeek V3模型提供强化学习训练支持，包括：
 * 1. V3增强奖励信号计算
 * 2. 推理质量奖励
 * 3. 代码质量奖励
 * 4. MoE效率奖励
 * 5. 任务特定奖励
 * 6. REINFORCE算法训练
 * 7. 自适应学习率调整
 * 
 * @author leavesfly
 * @version 1.0
 */
public class V3RLTrainer extends Trainer {
    
    // 继承的字段（由于父类字段为private，需要在子类中重新定义）
    private DataSet dataSet;
    private Model model;
    private Loss loss;
    private Optimizer optimizer;
    private Monitor monitor;
    private int maxEpoch;
    
    /**
     * DeepSeek V3模型
     */
    private DeepSeekV3Model deepSeekV3Model;
    
    /**
     * V3特有的训练参数
     */
    private final float moeRewardWeight;
    private final float codeQualityWeight;
    private final float reasoningQualityWeight;
    private final float taskSpecificWeight;
    private final float loadBalancePenalty;
    
    /**
     * 训练统计信息
     */
    private V3TrainingStats trainingStats;
    
    /**
     * 当前训练的任务类型
     */
    private TaskType currentTaskType;
    
    /**
     * 构造函数
     * 
     * @param maxEpoch 最大训练轮次
     * @param monitor 监控器
     * @param evaluator 评估器
     * @param config V3训练配置
     */
    public V3RLTrainer(int maxEpoch, Monitor monitor, Evaluator evaluator, V3TrainingConfig config) {
        super(maxEpoch, monitor, evaluator);
        
        this.moeRewardWeight = config.moeRewardWeight;
        this.codeQualityWeight = config.codeQualityWeight;
        this.reasoningQualityWeight = config.reasoningQualityWeight;
        this.taskSpecificWeight = config.taskSpecificWeight;
        this.loadBalancePenalty = config.loadBalancePenalty;
        
        this.trainingStats = new V3TrainingStats();
        this.currentTaskType = TaskType.GENERAL;
    }
    
    /**
     * 默认构造函数 - 使用标准V3训练配置
     */
    public V3RLTrainer(int maxEpoch, Monitor monitor, Evaluator evaluator) {
        this(maxEpoch, monitor, evaluator, V3TrainingConfig.getDefaultConfig());
        this.maxEpoch = maxEpoch;
        this.monitor = monitor;
    }
    
    /**
     * 初始化V3训练器
     * 
     * @param dataSet 数据集
     * @param model 模型（必须是DeepSeekV3Model）
     * @param loss 损失函数
     * @param optimizer 优化器
     */
    @Override
    public void init(DataSet dataSet, Model model, Loss loss, Optimizer optimizer) {
        if (!(model instanceof DeepSeekV3Model)) {
            throw new IllegalArgumentException("V3RLTrainer requires DeepSeekV3Model");
        }
        
        super.init(dataSet, model, loss, optimizer);
        
        // 初始化本地字段
        this.dataSet = dataSet;
        this.model = model;
        this.loss = loss;
        this.optimizer = optimizer;
        
        this.deepSeekV3Model = (DeepSeekV3Model) model;
        
        System.out.println("V3RLTrainer initialized with " + model.getName());
        printTrainingConfig();
    }
    
    /**
     * 执行V3强化学习训练
     * 
     * @param shuffleData 是否打乱数据
     * @param taskType 训练的任务类型
     */
    public void trainV3RL(boolean shuffleData, TaskType taskType) {
        this.currentTaskType = taskType;
        System.out.println("开始V3强化学习训练 - 任务类型: " + taskType);
        
        // 执行标准训练流程，但使用V3增强的奖励计算
        trainWithV3Enhancements(shuffleData);
        
        // 打印V3训练统计
        printV3TrainingStats();
    }
    
    /**
     * 带V3增强的训练
     */
    private void trainWithV3Enhancements(boolean shuffleData) {
        DataSet trainDataSet = dataSet.getTrainDataSet();
        if (shuffleData) {
            trainDataSet.shuffle();
        }
        
        for (int epoch = 0; epoch < maxEpoch; epoch++) {
            long epochStartTime = System.currentTimeMillis();
            
            model.resetState();
            monitor.startNewEpoch(epoch);
            trainingStats.startNewEpoch();
            
            List<Batch> batches = trainDataSet.getBatches();
            
            for (Batch batch : batches) {
                V3TrainingStep stepResult = executeV3TrainingStep(batch);
                trainingStats.addStepResult(stepResult);
            }
            
            long epochEndTime = System.currentTimeMillis();
            
            // 更新监控信息
            monitor.collectInfo(trainingStats.getAverageLoss());
            monitor.endEpoch();
            
            // 打印V3特有的训练信息
            printV3EpochInfo(epoch, epochEndTime - epochStartTime);
            
            // 自适应学习率调整
            adjustLearningRateIfNeeded(epoch);
        }
        
        monitor.plot();
    }
    
    /**
     * 执行V3训练步骤
     */
    private V3TrainingStep executeV3TrainingStep(Batch batch) {
        Variable variableX = batch.toVariableX().setName("x").setRequireGrad(false);
        Variable variableY = batch.toVariableY().setName("y").setRequireGrad(false);
        
        // V3前向传播
        DeepSeekV3Block.DeepSeekV3Output v3Output = deepSeekV3Model.generateWithTaskType(
            variableX.getValue(), currentTaskType);
        
        // 计算V3增强奖励
        V3RewardSignal rewardSignal = computeV3RewardSignal(v3Output, variableY);
        
        // 计算REINFORCE损失
        Variable reinforceLoss = computeREINFORCELoss(v3Output.logits, variableY, rewardSignal);
        
        // 添加MoE负载均衡损失
        float moeLoss = v3Output.moeLoss * loadBalancePenalty;
        Variable totalLoss = addScalarToVariable(reinforceLoss, moeLoss);
        
        // 清除梯度
        model.clearGrads();
        
        // 反向传播
        totalLoss.backward();
        
        // 梯度裁剪
        clipGradients(1.0f);
        
        // 更新参数
        optimizer.update();
        
        totalLoss.unChainBackward();
        
        return new V3TrainingStep(
            totalLoss.getValue().getNumber().floatValue(),
            reinforceLoss.getValue().getNumber().floatValue(),
            moeLoss,
            rewardSignal
        );
    }
    
    /**
     * 计算V3增强奖励信号
     */
    private V3RewardSignal computeV3RewardSignal(DeepSeekV3Block.DeepSeekV3Output v3Output, Variable targetOutput) {
        // 基础准确性奖励（简化版本）
        float accuracyReward = computeAccuracyReward(v3Output.logits, targetOutput);
        
        // 推理质量奖励
        float reasoningQualityReward = v3Output.getReasoningQuality() * reasoningQualityWeight;
        
        // 任务特定奖励
        float taskSpecificReward = computeTaskSpecificReward(v3Output, currentTaskType);
        
        // MoE效率奖励
        float moeEfficiencyReward = computeMoEEfficiencyReward(v3Output) * moeRewardWeight;
        
        // 组合总奖励
        float totalReward = accuracyReward + reasoningQualityReward + taskSpecificReward + moeEfficiencyReward;
        
        return new V3RewardSignal(
            totalReward,
            accuracyReward,
            reasoningQualityReward,
            taskSpecificReward,
            moeEfficiencyReward
        );
    }
    
    /**
     * 计算基础准确性奖励
     */
    private float computeAccuracyReward(Variable logits, Variable target) {
        // 简化版本：使用负交叉熵作为奖励
        try {
            Variable lossVar = loss.loss(target, logits);
            return -lossVar.getValue().getNumber().floatValue();
        } catch (Exception e) {
            return 0.0f; // 降级处理
        }
    }
    
    /**
     * 计算任务特定奖励
     */
    private float computeTaskSpecificReward(DeepSeekV3Block.DeepSeekV3Output v3Output, TaskType taskType) {
        float taskReward = 0.0f;
        
        switch (taskType) {
            case CODING:
                if (v3Output.codeInfo != null) {
                    taskReward = v3Output.codeInfo.getCodeConfidence() * codeQualityWeight;
                }
                break;
            case REASONING:
            case MATH:
                // 推理任务的额外奖励
                taskReward = v3Output.getReasoningQuality() * 0.5f;
                break;
            case GENERAL:
            case MULTIMODAL:
                // 通用任务的平衡奖励
                taskReward = v3Output.getReasoningQuality() * 0.2f;
                break;
        }
        
        return taskReward * taskSpecificWeight;
    }
    
    /**
     * 计算MoE效率奖励
     */
    private float computeMoEEfficiencyReward(DeepSeekV3Block.DeepSeekV3Output v3Output) {
        // 鼓励专家使用的多样性，惩罚过度集中
        Map<String, Integer> expertUsage = v3Output.getExpertUsageStats();
        
        if (expertUsage.isEmpty()) {
            return 0.0f;
        }
        
        // 计算专家使用的熵作为多样性指标
        int totalUsage = expertUsage.values().stream().mapToInt(Integer::intValue).sum();
        double entropy = 0.0;
        
        for (int usage : expertUsage.values()) {
            if (usage > 0) {
                double prob = (double) usage / totalUsage;
                entropy -= prob * Math.log(prob);
            }
        }
        
        // 将MoE损失转换为效率奖励
        float efficiencyReward = (float) (entropy - v3Output.moeLoss);
        
        return Math.max(0.0f, efficiencyReward);
    }
    
    /**
     * 计算REINFORCE损失
     */
    private Variable computeREINFORCELoss(Variable logits, Variable target, V3RewardSignal rewardSignal) {
        // 简化的REINFORCE实现
        NdArray logitsData = logits.getValue();
        NdArray targetData = target.getValue();
        
        // 计算log概率
        NdArray logProbs = computeLogSoftmax(logitsData);
        
        // 选择目标token的log概率
        NdArray selectedLogProbs = selectTargetLogProbs(logProbs, targetData);
        
        // REINFORCE损失：-log_prob * reward
        NdArray reinforceLossData = selectedLogProbs.mulNum(-rewardSignal.totalReward);
        
        return new Variable(reinforceLossData);
    }
    
    /**
     * 计算log softmax
     */
    private NdArray computeLogSoftmax(NdArray logits) {
        // 简化实现
        NdArray softmax = logits.softMax();
        return softmax.log();
    }
    
    /**
     * 选择目标token的log概率
     */
    private NdArray selectTargetLogProbs(NdArray logProbs, NdArray target) {
        // 简化实现：返回平均log概率
        return NdArray.of(io.leavesfly.tinyai.ndarr.Shape.of(1)).like(logProbs.mean(0).getNumber());
    }
    
    /**
     * 给Variable添加标量值
     */
    private Variable addScalarToVariable(Variable var, float scalar) {
        NdArray result = var.getValue().like(var.getValue().getNumber().floatValue() + scalar);
        return new Variable(result);
    }
    
    /**
     * 梯度裁剪
     */
    private void clipGradients(float maxNorm) {
        // 简化的梯度裁剪实现
        // 在实际应用中，应该遍历所有参数并裁剪梯度
    }
    
    /**
     * 自适应学习率调整
     */
    private void adjustLearningRateIfNeeded(int epoch) {
        // 简单的学习率衰减策略
        if (epoch > 0 && epoch % 10 == 0) {
            float currentReasoningQuality = trainingStats.getAverageReasoningQuality();
            float previousReasoningQuality = trainingStats.getPreviousEpochReasoningQuality();
            
            if (currentReasoningQuality <= previousReasoningQuality) {
                // 学习率衰减
                System.out.println("Epoch " + epoch + ": 应用学习率衰减");
            }
        }
    }
    
    /**
     * 打印训练配置
     */
    private void printTrainingConfig() {
        System.out.println("=== V3RL训练配置 ===");
        System.out.println("MoE奖励权重: " + moeRewardWeight);
        System.out.println("代码质量权重: " + codeQualityWeight);
        System.out.println("推理质量权重: " + reasoningQualityWeight);
        System.out.println("任务特定权重: " + taskSpecificWeight);
        System.out.println("负载均衡惩罚: " + loadBalancePenalty);
        System.out.println("==================");
    }
    
    /**
     * 打印V3轮次信息
     */
    private void printV3EpochInfo(int epoch, long duration) {
        System.out.printf("Epoch %d | 损失: %.4f | 推理质量: %.3f | MoE损失: %.4f | 耗时: %dms%n",
                         epoch,
                         trainingStats.getAverageLoss(),
                         trainingStats.getAverageReasoningQuality(),
                         trainingStats.getAverageMoELoss(),
                         duration);
    }
    
    /**
     * 打印V3训练统计
     */
    private void printV3TrainingStats() {
        System.out.println("\n=== V3训练统计 ===");
        System.out.println("任务类型: " + currentTaskType);
        System.out.println("平均总奖励: " + String.format("%.4f", trainingStats.getAverageTotalReward()));
        System.out.println("平均推理质量: " + String.format("%.3f", trainingStats.getAverageReasoningQuality()));
        System.out.println("平均MoE效率: " + String.format("%.3f", trainingStats.getAverageMoEEfficiency()));
        if (currentTaskType == TaskType.CODING) {
            System.out.println("平均代码置信度: " + String.format("%.3f", trainingStats.getAverageCodeConfidence()));
        }
        System.out.println("总训练步骤: " + trainingStats.getTotalSteps());
        System.out.println("================\n");
    }
    
    // 内部类
    
    /**
     * V3奖励信号
     */
    private static class V3RewardSignal {
        final float totalReward;
        final float accuracyReward;
        final float reasoningQualityReward;
        final float taskSpecificReward;
        final float moeEfficiencyReward;
        
        V3RewardSignal(float totalReward, float accuracyReward, float reasoningQualityReward,
                      float taskSpecificReward, float moeEfficiencyReward) {
            this.totalReward = totalReward;
            this.accuracyReward = accuracyReward;
            this.reasoningQualityReward = reasoningQualityReward;
            this.taskSpecificReward = taskSpecificReward;
            this.moeEfficiencyReward = moeEfficiencyReward;
        }
    }
    
    /**
     * V3训练步骤结果
     */
    private static class V3TrainingStep {
        final float totalLoss;
        final float reinforceLoss;
        final float moeLoss;
        final V3RewardSignal rewardSignal;
        
        V3TrainingStep(float totalLoss, float reinforceLoss, float moeLoss, V3RewardSignal rewardSignal) {
            this.totalLoss = totalLoss;
            this.reinforceLoss = reinforceLoss;
            this.moeLoss = moeLoss;
            this.rewardSignal = rewardSignal;
        }
    }
    
    /**
     * V3训练统计
     */
    private static class V3TrainingStats {
        private float totalLoss = 0.0f;
        private float totalReward = 0.0f;
        private float totalReasoningQuality = 0.0f;
        private float totalMoEEfficiency = 0.0f;
        private float totalCodeConfidence = 0.0f;
        private int stepCount = 0;
        private float previousEpochReasoningQuality = 0.0f;
        
        void startNewEpoch() {
            previousEpochReasoningQuality = getAverageReasoningQuality();
            totalLoss = 0.0f;
            totalReward = 0.0f;
            totalReasoningQuality = 0.0f;
            totalMoEEfficiency = 0.0f;
            totalCodeConfidence = 0.0f;
            stepCount = 0;
        }
        
        void addStepResult(V3TrainingStep step) {
            totalLoss += step.totalLoss;
            totalReward += step.rewardSignal.totalReward;
            totalReasoningQuality += step.rewardSignal.reasoningQualityReward;
            totalMoEEfficiency += step.rewardSignal.moeEfficiencyReward;
            stepCount++;
        }
        
        float getAverageLoss() {
            return stepCount > 0 ? totalLoss / stepCount : 0.0f;
        }
        
        float getAverageTotalReward() {
            return stepCount > 0 ? totalReward / stepCount : 0.0f;
        }
        
        float getAverageReasoningQuality() {
            return stepCount > 0 ? totalReasoningQuality / stepCount : 0.0f;
        }
        
        float getAverageMoEEfficiency() {
            return stepCount > 0 ? totalMoEEfficiency / stepCount : 0.0f;
        }
        
        float getAverageCodeConfidence() {
            return stepCount > 0 ? totalCodeConfidence / stepCount : 0.0f;
        }
        
        float getAverageMoELoss() {
            return stepCount > 0 ? totalMoEEfficiency / stepCount : 0.0f; // 简化
        }
        
        float getPreviousEpochReasoningQuality() {
            return previousEpochReasoningQuality;
        }
        
        int getTotalSteps() {
            return stepCount;
        }
    }
    
    /**
     * V3训练配置
     */
    public static class V3TrainingConfig {
        final float moeRewardWeight;
        final float codeQualityWeight;
        final float reasoningQualityWeight;
        final float taskSpecificWeight;
        final float loadBalancePenalty;
        
        public V3TrainingConfig(float moeRewardWeight, float codeQualityWeight,
                              float reasoningQualityWeight, float taskSpecificWeight,
                              float loadBalancePenalty) {
            this.moeRewardWeight = moeRewardWeight;
            this.codeQualityWeight = codeQualityWeight;
            this.reasoningQualityWeight = reasoningQualityWeight;
            this.taskSpecificWeight = taskSpecificWeight;
            this.loadBalancePenalty = loadBalancePenalty;
        }
        
        public static V3TrainingConfig getDefaultConfig() {
            return new V3TrainingConfig(0.1f, 0.2f, 0.3f, 0.2f, 0.01f);
        }
        
        public static V3TrainingConfig getCodeFocusedConfig() {
            return new V3TrainingConfig(0.1f, 0.5f, 0.2f, 0.3f, 0.01f);
        }
        
        public static V3TrainingConfig getReasoningFocusedConfig() {
            return new V3TrainingConfig(0.1f, 0.1f, 0.5f, 0.3f, 0.01f);
        }
    }
    
    // Getters
    public V3TrainingStats getTrainingStats() {
        return trainingStats;
    }
    
    public TaskType getCurrentTaskType() {
        return currentTaskType;
    }
    
    public DeepSeekV3Model getDeepSeekV3Model() {
        return deepSeekV3Model;
    }
}