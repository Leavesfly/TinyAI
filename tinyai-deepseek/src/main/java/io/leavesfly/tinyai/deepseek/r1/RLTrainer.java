package io.leavesfly.tinyai.deepseek.r1;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.func.loss.MeanSE;
import io.leavesfly.tinyai.ml.optimize.SGD;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * 强化学习训练器
 * 
 * 实现基于REINFORCE算法的强化学习训练，
 * 针对DeepSeek R1的推理能力进行优化。
 * 
 * 训练目标：
 * 1. 提高推理质量
 * 2. 增强自我反思能力
 * 3. 优化置信度评估
 * 4. 减少推理错误
 * 
 * 奖励信号：
 * - 准确性奖励：基于最终答案的正确性
 * - 推理质量奖励：基于反思模块的质量评估
 * - 一致性奖励：基于推理步骤的逻辑一致性
 * - 效率奖励：基于推理步骤数量的合理性
 * 
 * @author leavesfly
 * @version 1.0
 */
public class RLTrainer {
    
    private DeepSeekR1Model model;         // 要训练的模型
    private SGD optimizer;                 // 优化器
    private double learningRate;           // 学习率
    private double gamma;                  // 折扣因子
    private double entropyCoeff;           // 熵系数
    private double valueCoeff;             // 价值函数系数
    private Random random;                 // 随机数生成器
    
    // 训练统计
    private int totalSteps;                // 总训练步数
    private double averageReward;          // 平均奖励
    private double averageLoss;            // 平均损失
    
    /**
     * 构造强化学习训练器
     * 
     * @param model 要训练的模型
     * @param learningRate 学习率
     * @param gamma 折扣因子
     * @param entropyCoeff 熵系数
     * @param valueCoeff 价值函数系数
     */
    public RLTrainer(DeepSeekR1Model model, double learningRate, double gamma, 
                     double entropyCoeff, double valueCoeff) {
        this.model = model;
        this.learningRate = learningRate;
        this.gamma = gamma;
        this.entropyCoeff = entropyCoeff;
        this.valueCoeff = valueCoeff;
        this.random = new Random(42);
        
        // 初始化优化器（简化版，实际应该传入模型参数）
        this.optimizer = new SGD(learningRate);
        
        this.totalSteps = 0;
        this.averageReward = 0.0;
        this.averageLoss = 0.0;
    }
    
    /**
     * 默认构造函数
     */
    public RLTrainer(DeepSeekR1Model model, double learningRate) {
        this(model, learningRate, 0.99, 0.01, 0.1);
    }
    
    /**
     * 计算奖励信号
     * 
     * @param modelResult 模型推理结果
     * @param targetOutput 目标输出
     * @return 奖励值
     */
    public double computeReward(DeepSeekR1Result modelResult, NdArray targetOutput) {
        double totalReward = 0.0;
        
        // 1. 准确性奖励
        double accuracyReward = computeAccuracyReward(modelResult.getModelOutput(), targetOutput);
        totalReward += accuracyReward * 0.4;
        
        // 2. 推理质量奖励
        double qualityReward = computeQualityReward(modelResult);
        totalReward += qualityReward * 0.3;
        
        // 3. 一致性奖励
        double consistencyReward = computeConsistencyReward(modelResult);
        totalReward += consistencyReward * 0.2;
        
        // 4. 效率奖励
        double efficiencyReward = computeEfficiencyReward(modelResult);
        totalReward += efficiencyReward * 0.1;
        
        return totalReward;
    }
    
    /**
     * 计算准确性奖励
     */
    private double computeAccuracyReward(NdArray modelOutput, NdArray targetOutput) {
        if (modelOutput == null || targetOutput == null) {
            return 0.0;
        }
        
        // 使用负均方误差作为准确性奖励
        MeanSE mse = new MeanSE();
        Variable modelVar = new Variable(modelOutput);
        Variable targetVar = new Variable(targetOutput);
        
        NdArray loss = mse.forward(modelOutput, targetOutput);
        double mseValue = getMeanValue(loss);
        
        // 转换为奖励（损失越小，奖励越大）
        return Math.exp(-mseValue);
    }
    
    /**
     * 计算推理质量奖励
     */
    private double computeQualityReward(DeepSeekR1Result modelResult) {
        if (modelResult.getReflectionResult() == null) {
            return 0.5; // 默认中等奖励
        }
        
        ReflectionModule.ReflectionResult reflection = modelResult.getReflectionResult();
        return reflection.getQualityScore();
    }
    
    /**
     * 计算一致性奖励
     */
    private double computeConsistencyReward(DeepSeekR1Result modelResult) {
        if (modelResult.getReflectionResult() == null) {
            return 0.5; // 默认中等奖励
        }
        
        ReflectionModule.ReflectionResult reflection = modelResult.getReflectionResult();
        return reflection.getConsistencyScore();
    }
    
    /**
     * 计算效率奖励
     */
    private double computeEfficiencyReward(DeepSeekR1Result modelResult) {
        if (modelResult.getReasoningChain() == null) {
            return 0.5; // 默认中等奖励
        }
        
        ReasoningChain chain = modelResult.getReasoningChain();
        int stepCount = chain.getStepCount();
        
        // 理想步骤数为3-7步
        int idealMin = 3;
        int idealMax = 7;
        
        if (stepCount >= idealMin && stepCount <= idealMax) {
            return 1.0; // 最高效率奖励
        } else if (stepCount < idealMin) {
            return 0.5 + 0.5 * stepCount / idealMin; // 步骤太少
        } else {
            return Math.max(0.1, 1.0 - 0.1 * (stepCount - idealMax)); // 步骤太多
        }
    }
    
    /**
     * 执行一步训练
     * 
     * @param inputIds 输入序列
     * @param targetIds 目标序列
     * @param question 问题描述（可选）
     * @return 训练指标
     */
    public TrainingMetrics trainStep(NdArray inputIds, NdArray targetIds, String question) {
        try {
            // 前向传播
            DeepSeekR1Result modelResult = model.performReasoning(inputIds, question);
            
            // 计算奖励
            double reward = computeReward(modelResult, targetIds);
            
            // 计算策略损失
            double policyLoss = computePolicyLoss(modelResult, reward);
            
            // 计算价值函数损失
            double valueLoss = computeValueLoss(reward);
            
            // 计算熵损失（鼓励探索）
            double entropyLoss = computeEntropyLoss(modelResult);
            
            // 总损失
            double totalLoss = policyLoss + valueCoeff * valueLoss - entropyCoeff * entropyLoss;
            
            // 反向传播（简化实现）
            performBackwardPass(totalLoss);
            
            // 更新统计信息
            updateStatistics(reward, totalLoss);
            
            // 构建训练指标
            TrainingMetrics metrics = new TrainingMetrics();
            metrics.setStep(totalSteps);
            metrics.setReward(reward);
            metrics.setPolicyLoss(policyLoss);
            metrics.setValueLoss(valueLoss);
            metrics.setEntropyLoss(entropyLoss);
            metrics.setTotalLoss(totalLoss);
            metrics.setAverageReward(averageReward);
            metrics.setAverageLoss(averageLoss);
            
            if (modelResult.getReasoningChain() != null) {
                metrics.setStepCount(modelResult.getReasoningChain().getStepCount());
                metrics.setConfidence(modelResult.getFinalConfidence());
            }
            
            return metrics;
            
        } catch (Exception e) {
            System.err.println("训练步骤出错: " + e.getMessage());
            return createErrorMetrics(e.getMessage());
        }
    }
    
    /**
     * 计算策略损失（REINFORCE）
     */
    private double computePolicyLoss(DeepSeekR1Result modelResult, double reward) {
        // 简化的策略梯度损失
        // 实际应该基于动作概率和奖励计算
        
        if (modelResult.getReasoningChain() == null) {
            return 1.0; // 默认损失
        }
        
        ReasoningChain chain = modelResult.getReasoningChain();
        double avgConfidence = chain.getAverageConfidence();
        
        // 策略损失 = -log(prob) * reward
        // 这里用置信度近似概率
        double logProb = Math.log(Math.max(0.001, avgConfidence));
        return -logProb * reward;
    }
    
    /**
     * 计算价值函数损失
     */
    private double computeValueLoss(double reward) {
        // 简化的价值函数损失
        // 实际应该基于价值函数网络的预测和真实奖励
        
        double predictedValue = 0.5; // 简化预测值
        double target = reward;
        
        return Math.pow(predictedValue - target, 2);
    }
    
    /**
     * 计算熵损失
     */
    private double computeEntropyLoss(DeepSeekR1Result modelResult) {
        // 简化的熵计算
        // 鼓励多样性和探索
        
        if (modelResult.getReasoningChain() == null) {
            return 0.0;
        }
        
        ReasoningChain chain = modelResult.getReasoningChain();
        
        // 基于置信度方差计算熵
        double avgConfidence = chain.getAverageConfidence();
        double variance = 0.0;
        
        for (ReasoningStep step : chain.getSteps()) {
            double diff = step.getConfidence() - avgConfidence;
            variance += diff * diff;
        }
        variance /= Math.max(1, chain.getStepCount());
        
        // 方差越大，熵越大（多样性越高）
        return variance;
    }
    
    /**
     * 执行反向传播（简化实现）
     */
    private void performBackwardPass(double loss) {
        // 实际应该计算梯度并更新参数
        // 这里简化为记录损失
        
        // 模拟参数更新
        // 在实际实现中，应该调用optimizer.step()
    }
    
    /**
     * 更新训练统计信息
     */
    private void updateStatistics(double reward, double loss) {
        totalSteps++;
        
        // 指数移动平均
        double alpha = 0.1;
        averageReward = alpha * reward + (1 - alpha) * averageReward;
        averageLoss = alpha * loss + (1 - alpha) * averageLoss;
    }
    
    /**
     * 创建错误指标
     */
    private TrainingMetrics createErrorMetrics(String errorMessage) {
        TrainingMetrics metrics = new TrainingMetrics();
        metrics.setStep(totalSteps);
        metrics.setError(true);
        metrics.setErrorMessage(errorMessage);
        return metrics;
    }
    
    /**
     * 训练多个epoch
     * 
     * @param trainData 训练数据
     * @param epochs 训练轮数
     * @return 训练历史
     */
    public TrainingHistory train(TrainingData trainData, int epochs) {
        TrainingHistory history = new TrainingHistory();
        
        System.out.println("开始强化学习训练...");
        System.out.printf("训练数据: %d 样本, %d 轮次\n", trainData.size(), epochs);
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            System.out.printf("\n=== Epoch %d/%d ===\n", epoch + 1, epochs);
            
            double epochReward = 0.0;
            double epochLoss = 0.0;
            int validSamples = 0;
            
            for (int i = 0; i < trainData.size(); i++) {
                TrainingData.Sample sample = trainData.getSample(i);
                
                TrainingMetrics metrics = trainStep(
                    sample.getInputIds(),
                    sample.getTargetIds(),
                    sample.getQuestion()
                );
                
                if (!metrics.isError()) {
                    epochReward += metrics.getReward();
                    epochLoss += metrics.getTotalLoss();
                    validSamples++;
                }
                
                history.addMetrics(metrics);
                
                // 打印进度
                if ((i + 1) % 10 == 0 || i == trainData.size() - 1) {
                    System.out.printf("  样本 %d/%d - 奖励: %.3f, 损失: %.3f\n",
                                    i + 1, trainData.size(), metrics.getReward(), metrics.getTotalLoss());
                }
            }
            
            // 计算epoch统计
            if (validSamples > 0) {
                epochReward /= validSamples;
                epochLoss /= validSamples;
            }
            
            System.out.printf("Epoch %d 完成 - 平均奖励: %.3f, 平均损失: %.3f\n",
                             epoch + 1, epochReward, epochLoss);
        }
        
        System.out.println("\n训练完成!");
        printTrainingSummary(history);
        
        return history;
    }
    
    /**
     * 打印训练摘要
     */
    private void printTrainingSummary(TrainingHistory history) {
        System.out.println("\n=== 训练摘要 ===");
        System.out.printf("总训练步数: %d\n", totalSteps);
        System.out.printf("最终平均奖励: %.3f\n", averageReward);
        System.out.printf("最终平均损失: %.3f\n", averageLoss);
        System.out.printf("成功样本比例: %.1f%%\n", history.getSuccessRate() * 100);
        System.out.println("==================");
    }
    
    /**
     * 获取均值
     */
    private double getMeanValue(NdArray array) {
        double sum = 0.0;
        int count = 0;
        
        for (int i = 0; i < array.getShape().size(); i++) {
            sum += array.get(i);
            count++;
        }
        
        return count > 0 ? sum / count : 0.0;
    }
    
    /**
     * 评估模型性能
     * 
     * @param testData 测试数据
     * @return 评估结果
     */
    public EvaluationResult evaluate(TrainingData testData) {
        System.out.println("开始模型评估...");
        
        EvaluationResult result = new EvaluationResult();
        
        for (int i = 0; i < testData.size(); i++) {
            TrainingData.Sample sample = testData.getSample(i);
            
            DeepSeekR1Result modelResult = model.performReasoning(
                sample.getInputIds(),
                sample.getQuestion()
            );
            
            double reward = computeReward(modelResult, sample.getTargetIds());
            result.addSample(reward, modelResult);
        }
        
        result.computeStatistics();
        
        System.out.println("评估完成:");
        System.out.println(result.getSummary());
        
        return result;
    }
    
    // Getter方法
    public DeepSeekR1Model getModel() { return model; }
    public double getLearningRate() { return learningRate; }
    public double getGamma() { return gamma; }
    public double getEntropyCoeff() { return entropyCoeff; }
    public double getValueCoeff() { return valueCoeff; }
    public int getTotalSteps() { return totalSteps; }
    public double getAverageReward() { return averageReward; }
    public double getAverageLoss() { return averageLoss; }
    
    @Override
    public String toString() {
        return String.format("RLTrainer(lr=%.4f, gamma=%.2f, steps=%d, avgReward=%.3f)",
                           learningRate, gamma, totalSteps, averageReward);
    }
}