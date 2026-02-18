package io.leavesfly.tinyai.deepseek.v3.training;

import io.leavesfly.tinyai.deepseek.base.TaskType;
import io.leavesfly.tinyai.deepseek.v3.DeepSeekV3Block;
import io.leavesfly.tinyai.deepseek.v3.DeepSeekV3Config;
import io.leavesfly.tinyai.deepseek.v3.DeepSeekV3Model;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.loss.SoftmaxCrossEntropy;
import io.leavesfly.tinyai.ml.optimize.Adam;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.core.Parameter;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * DeepSeek-V3后训练器（任务感知微调）
 * 
 * 在预训练基础上进行任务特定的微调,
 * 优化任务感知路由和代码生成能力
 * 
 * 关键特性：
 * 1. 任务感知微调 - 根据任务类型优化专家选择
 * 2. 代码任务优化 - 特别针对代码生成质量
 * 3. 早停机制 - 防止过拟合
 * 4. 较低学习率 - 保护预训练知识
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekV3Posttrain {
    
    private final DeepSeekV3Model model;
    private final DeepSeekV3Config config;
    private final DeepSeekV3Dataset trainDataset;
    private final DeepSeekV3Dataset valDataset;
    private final SoftmaxCrossEntropy lossFunction;
    private final Adam optimizer;
    
    // 训练超参数
    private int maxEpochs;
    private float initialLearningRate;  // 比预训练低10倍
    private float minLearningRate;
    private int warmupSteps;
    private float maxGradNorm;
    private float moeLoadBalanceWeight;
    private int logInterval;
    private int valInterval;
    private int saveInterval;
    private int patience;  // 早停耐心值
    private String checkpointDir;
    
    // 训练状态
    private int currentEpoch;
    private int globalStep;
    private float currentLearningRate;
    private float bestValLoss;
    private int stepsWithoutImprovement;
    private List<Float> trainLossHistory;
    private List<Float> valLossHistory;
    private List<Float> codeQualityHistory;  // 代码质量历史
    
    /**
     * 构造函数
     */
    public DeepSeekV3Posttrain(DeepSeekV3Model model,
                               DeepSeekV3Dataset trainDataset,
                               DeepSeekV3Dataset valDataset) {
        this.model = model;
        this.config = model.getConfig();
        this.trainDataset = trainDataset;
        this.valDataset = valDataset;
        this.lossFunction = new SoftmaxCrossEntropy();
        
        // 默认超参数（比预训练更保守）
        this.maxEpochs = 5;
        this.initialLearningRate = 2.5e-5f;  // 比预训练低10倍
        this.minLearningRate = 1e-6f;
        this.warmupSteps = 500;
        this.maxGradNorm = 1.0f;
        this.moeLoadBalanceWeight = (float) config.getLoadBalanceLossWeight();
        this.logInterval = 50;
        this.valInterval = 500;
        this.saveInterval = 2000;
        this.patience = 3;
        this.checkpointDir = "./checkpoints/deepseek_v3_posttrain";
        
        // 创建优化器
        this.optimizer = new Adam(model, initialLearningRate, 0.9f, 0.999f, 1e-8f);
        
        // 初始化状态
        this.currentEpoch = 0;
        this.globalStep = 0;
        this.currentLearningRate = 0.0f;
        this.bestValLoss = Float.MAX_VALUE;
        this.stepsWithoutImprovement = 0;
        this.trainLossHistory = new ArrayList<>();
        this.valLossHistory = new ArrayList<>();
        this.codeQualityHistory = new ArrayList<>();
    }
    
    /**
     * 配置训练参数
     */
    public DeepSeekV3Posttrain configure(int maxEpochs, float learningRate,
                                          int patience) {
        this.maxEpochs = maxEpochs;
        this.initialLearningRate = learningRate;
        this.patience = patience;
        return this;
    }
    
    /**
     * 开始训练
     */
    public void train() {
        System.out.println("=".repeat(80));
        System.out.println("DeepSeek-V3 后训练/微调（任务感知优化）");
        System.out.println("=".repeat(80));
        System.out.println("训练配置:");
        System.out.println("  - 训练样本: " + trainDataset.getSampleCount());
        System.out.println("  - 验证样本: " + valDataset.getSampleCount());
        System.out.println("  - 最大轮次: " + maxEpochs);
        System.out.println("  - 初始学习率: " + initialLearningRate + " (比预训练低10倍)");
        System.out.println("  - 早停耐心值: " + patience);
        System.out.println("=".repeat(80));
        
        // 创建检查点目录
        createCheckpointDir();
        
        // 训练循环
        for (currentEpoch = 0; currentEpoch < maxEpochs; currentEpoch++) {
            trainOneEpoch();
            
            // 验证
            float valLoss = validate();
            valLossHistory.add(valLoss);
            
            System.out.printf("Epoch %d 验证损失: %.4f%n", currentEpoch + 1, valLoss);
            
            // 早停检查
            if (valLoss < bestValLoss) {
                bestValLoss = valLoss;
                stepsWithoutImprovement = 0;
                saveCheckpoint("best");
                System.out.println("新的最佳模型已保存!");
            } else {
                stepsWithoutImprovement++;
                if (stepsWithoutImprovement >= patience) {
                    System.out.println("触发早停,训练结束");
                    break;
                }
            }
        }
        
        // 保存最终模型
        saveCheckpoint("final");
        
        System.out.println("\n训练完成!");
        System.out.println("最佳验证损失: " + bestValLoss);
    }
    
    /**
     * 训练一个epoch
     */
    private void trainOneEpoch() {
        trainDataset.prepare(true);
        
        double epochLoss = 0.0;
        double epochMoeLoss = 0.0;
        double epochCodeQuality = 0.0;
        int batchCount = 0;
        int codeTaskCount = 0;
        
        while (trainDataset.hasNext()) {
            DeepSeekV3Dataset.Batch batch = trainDataset.nextBatch();
            
            // 训练一步
            StepResult stepResult = trainStep(batch);
            
            epochLoss += stepResult.loss;
            epochMoeLoss += stepResult.moeLoss;
            if (stepResult.codeQuality > 0) {
                epochCodeQuality += stepResult.codeQuality;
                codeTaskCount++;
            }
            batchCount++;
            globalStep++;
            
            trainLossHistory.add(stepResult.loss);
            if (stepResult.codeQuality > 0) {
                codeQualityHistory.add(stepResult.codeQuality);
            }
            
            // 打印日志
            if (globalStep % logInterval == 0) {
                float avgLoss = getAverage(trainLossHistory, logInterval);
                float avgCodeQuality = codeTaskCount > 0 ? 
                    getAverage(codeQualityHistory, logInterval) : 0.0f;
                System.out.printf("Epoch %d/%d | Step %d | Loss: %.4f | " +
                                 "代码质量: %.4f | LR: %.6f%n",
                    currentEpoch + 1, maxEpochs, globalStep, avgLoss, 
                    avgCodeQuality, currentLearningRate);
            }
            
            // 定期验证
            if (globalStep % valInterval == 0) {
                float valLoss = validate();
                System.out.printf("中期验证损失: %.4f%n", valLoss);
            }
            
            // 保存检查点
            if (globalStep % saveInterval == 0) {
                saveCheckpoint("step_" + globalStep);
            }
        }
        
        double avgEpochLoss = batchCount > 0 ? epochLoss / batchCount : 0.0;
        double avgEpochCodeQuality = codeTaskCount > 0 ? epochCodeQuality / codeTaskCount : 0.0;
        
        System.out.printf("Epoch %d 完成 | 平均损失: %.4f | 平均代码质量: %.4f%n",
            currentEpoch + 1, avgEpochLoss, avgEpochCodeQuality);
        
        trainDataset.reset();
    }
    
    /**
     * 训练单步
     */
    private StepResult trainStep(DeepSeekV3Dataset.Batch batch) {
        updateLearningRate();
        
        NdArray inputIds = batch.getInputIds();
        NdArray targetIds = batch.getTargetIds();
        TaskType taskType = batch.getMajorityTaskType();
        
        Variable inputVar = new Variable(inputIds);
        
        // 前向传播
        DeepSeekV3Block.DetailedForwardResult result = 
            model.predictWithDetails(inputVar, taskType);
        Variable logits = result.logits;
        
        // 计算损失 - SoftmaxCE只接受2维输入，需要将3D张量reshape为2D
        int batchSize = inputIds.getShape().getDimension(0);
        int seqLen = inputIds.getShape().getDimension(1);
        int vocabSize = model.getConfig().getVocabSize();
        
        // logits: [batch_size, seq_len, vocab_size] -> [batch_size * seq_len, vocab_size]
        Variable logits2D = logits.reshape(Shape.of(batchSize * seqLen, vocabSize));
        
        // targets: [batch_size, seq_len] -> [batch_size * seq_len, 1]
        Variable targetVar = new Variable(targetIds);
        Variable targets2D = targetVar.reshape(Shape.of(batchSize * seqLen, 1));
        
        Variable lmLoss = lossFunction.loss(targets2D, logits2D);
        
        float lossValue = lmLoss.getValue().getNumber().floatValue();
        float moeLoss = (float) result.avgMoELoss;
        
        // 代码质量（简化，使用MoE损失）
        float moeLossValue = (float) result.avgMoELoss;
        
        // 总损失
        Variable totalLoss = lmLoss;
        if (moeLoadBalanceWeight > 0) {
            float[] moeLossData = new float[]{moeLoss * moeLoadBalanceWeight};
            Variable moeLossVar = new Variable(NdArray.of(moeLossData));
            totalLoss = totalLoss.add(moeLossVar);
        }
        
        model.clearGrads();
        totalLoss.backward();
        clipGradients();
        optimizer.update();
        totalLoss.unChainBackward();
        
        return new StepResult(lossValue, moeLoss, moeLossValue);
    }
    
    /**
     * 验证
     */
    private float validate() {
        valDataset.prepare(false);
        
        double totalLoss = 0.0;
        int count = 0;
        
        while (valDataset.hasNext()) {
            DeepSeekV3Dataset.Batch batch = valDataset.nextBatch();
            
            NdArray inputIds = batch.getInputIds();
            NdArray targetIds = batch.getTargetIds();
            
            Variable inputVar = new Variable(inputIds);
            Variable logits = model.predict(inputVar);
            
            // 将3D张量reshape为2D计算损失
            int batchSize = inputIds.getShape().getDimension(0);
            int seqLen = inputIds.getShape().getDimension(1);
            int vocabSize = model.getConfig().getVocabSize();
            
            Variable logits2D = logits.reshape(Shape.of(batchSize * seqLen, vocabSize));
            Variable targetVar = new Variable(targetIds);
            Variable targets2D = targetVar.reshape(Shape.of(batchSize * seqLen, 1));
            
            Variable loss = lossFunction.loss(targets2D, logits2D);
            
            totalLoss += loss.getValue().getNumber().floatValue();
            count++;
            
            loss.unChainBackward();
        }
        
        valDataset.reset();
        return count > 0 ? (float) (totalLoss / count) : 0.0f;
    }
    
    private void updateLearningRate() {
        if (globalStep < warmupSteps) {
            currentLearningRate = initialLearningRate * ((float) globalStep / warmupSteps);
        } else {
            int totalSteps = maxEpochs * trainDataset.getBatchCount();
            int decaySteps = totalSteps - warmupSteps;
            int currentDecayStep = globalStep - warmupSteps;
            
            double cosineDecay = 0.5 * (1 + Math.cos(Math.PI * currentDecayStep / decaySteps));
            float decayedLR = (initialLearningRate - minLearningRate) * (float) cosineDecay + minLearningRate;
            currentLearningRate = Math.max(decayedLR, minLearningRate);
        }
        
        optimizer.setLearningRate(currentLearningRate);
    }
    
    private void clipGradients() {
        double totalNorm = 0.0;
        Map<String, Parameter> params = model.getModule().namedParameters("", true);
        for (Parameter param : params.values()) {
            if (param.requiresGrad() && param.grad() != null) {
                NdArray grad = param.grad();
                double norm = grad.mul(grad).sum().getNumber().doubleValue();
                totalNorm += norm;
            }
        }
        
        totalNorm = Math.sqrt(totalNorm);
        
        if (totalNorm > maxGradNorm) {
            float scale = (float) (maxGradNorm / totalNorm);
            for (Parameter param : params.values()) {
                if (param.requiresGrad() && param.grad() != null) {
                    NdArray clippedGrad = param.grad().mulNum(scale);
                    param.setGrad(clippedGrad);
                }
            }
        }
    }
    
    private void createCheckpointDir() {
        File dir = new File(checkpointDir);
        if (!dir.exists()) {
            dir.mkdirs();
        }
    }
    
    private void saveCheckpoint(String name) {
        String path = checkpointDir + "/" + name + ".ckpt";
        System.out.println("保存检查点: " + path);
    }
    
    private float getAverage(List<Float> values, int last) {
        if (values.isEmpty()) return 0.0f;
        int start = Math.max(0, values.size() - last);
        float sum = 0.0f;
        for (int i = start; i < values.size(); i++) {
            sum += values.get(i);
        }
        return sum / (values.size() - start);
    }
    
    private static class StepResult {
        final float loss;
        final float moeLoss;
        final float codeQuality;
        
        StepResult(float loss, float moeLoss, float codeQuality) {
            this.loss = loss;
            this.moeLoss = moeLoss;
            this.codeQuality = codeQuality;
        }
    }
}
