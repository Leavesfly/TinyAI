package io.leavesfly.tinyai.gpt1.training;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.gpt1.GPT1Config;
import io.leavesfly.tinyai.gpt1.GPT1Model;
import io.leavesfly.tinyai.ml.loss.Loss;
import io.leavesfly.tinyai.ml.loss.SoftmaxCrossEntropy;
import io.leavesfly.tinyai.ml.optimize.SGD;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.core.Parameter;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * GPT-1微调训练器(Posttrain/Finetune)
 * 
 * 用于在预训练模型基础上进行任务特定的微调训练
 * 支持以下功能:
 * - 较小的学习率(相比预训练)
 * - 早停机制
 * - 验证集评估
 * - 最佳模型保存
 * 
 * @author TinyAI
 * @since 2024
 */
public class GPT1Finetune {
    
    private final GPT1Model model;
    private final GPT1Config config;
    private final GPT1Dataset trainDataset;
    private final GPT1Dataset valDataset;
    private final SoftmaxCrossEntropy lossFunction;
    private final SoftmaxCrossEntropy perTokenLossFunction;  // 用于masked loss，不做归约
    private final SGD optimizer;
    
    // 微调超参数(与预训练不同)
    private int maxEpochs;
    private float learningRate;           // 微调学习率通常更小
    private float maxGradNorm;
    private int logInterval;
    private int evalInterval;             // 验证评估间隔
    private int patience;                 // 早停耐心值
    private String checkpointDir;
    
    // 训练状态
    private int currentEpoch;
    private int globalStep;
    private List<Float> trainLossHistory;
    private List<Float> valLossHistory;
    private float bestValLoss;
    private int stepsWithoutImprovement;
    
    /**
     * 构造函数
     * 
     * @param model 预训练的GPT-1模型
     * @param trainDataset 训练数据集
     * @param valDataset 验证数据集
     */
    public GPT1Finetune(GPT1Model model, GPT1Dataset trainDataset, GPT1Dataset valDataset) {
        this.model = model;
        this.config = model.getConfig();
        this.trainDataset = trainDataset;
        this.valDataset = valDataset;
        this.lossFunction = new SoftmaxCrossEntropy();
        this.perTokenLossFunction = new SoftmaxCrossEntropy(Loss.Reduction.NONE);
        
        // 微调默认超参数(学习率比预训练小10倍)
        this.maxEpochs = 5;
        this.learningRate = 2.5e-5f;  // 预训练是2.5e-4
        this.maxGradNorm = 1.0f;
        this.logInterval = 50;
        this.evalInterval = 100;
        this.patience = 3;
        this.checkpointDir = "./checkpoints/gpt1_finetune";
        
        // 创建优化器
        this.optimizer = new SGD(model, learningRate);
        
        // 初始化状态
        this.currentEpoch = 0;
        this.globalStep = 0;
        this.trainLossHistory = new ArrayList<>();
        this.valLossHistory = new ArrayList<>();
        this.bestValLoss = Float.MAX_VALUE;
        this.stepsWithoutImprovement = 0;
    }
    
    /**
     * 配置微调参数
     * 
     * @param maxEpochs 最大训练轮次
     * @param learningRate 学习率
     * @param patience 早停耐心值
     * @return this
     */
    public GPT1Finetune configure(int maxEpochs, float learningRate, int patience) {
        this.maxEpochs = maxEpochs;
        this.learningRate = learningRate;
        this.patience = patience;
        // 更新优化器学习率
        this.optimizer.setLearningRate(learningRate);
        return this;
    }
    
    /**
     * 设置检查点配置
     * 
     * @param checkpointDir 检查点目录
     * @param evalInterval 评估间隔
     * @return this
     */
    public GPT1Finetune setCheckpoint(String checkpointDir, int evalInterval) {
        this.checkpointDir = checkpointDir;
        this.evalInterval = evalInterval;
        return this;
    }
    
    /**
     * 开始微调训练
     */
    public void train() {
        System.out.println("=".repeat(60));
        System.out.println("GPT-1 微调训练 (Finetune/Posttrain)");
        System.out.println("=".repeat(60));
        System.out.println("模型配置:");
        System.out.println("  - 模型: " + model.getName());
        System.out.println("  - 参数量: " + model.getAllParams().size());
        System.out.println("微调配置:");
        System.out.println("  - 训练样本: " + trainDataset.getSampleCount());
        System.out.println("  - 验证样本: " + valDataset.getSampleCount());
        System.out.println("  - 最大轮次: " + maxEpochs);
        System.out.println("  - 学习率: " + learningRate);
        System.out.println("  - 早停耐心: " + patience);
        System.out.println("=".repeat(60));
        
        // 创建检查点目录
        createCheckpointDir();
        
        // 训练循环
        for (currentEpoch = 0; currentEpoch < maxEpochs; currentEpoch++) {
            trainOneEpoch();
            
            // 每个epoch结束后进行验证
            float valLoss = evaluate();
            valLossHistory.add(valLoss);
            
            System.out.printf("Epoch %d 验证损失: %.4f%n", currentEpoch + 1, valLoss);
            
            // 检查是否是最佳模型
            if (valLoss < bestValLoss) {
                bestValLoss = valLoss;
                stepsWithoutImprovement = 0;
                saveCheckpoint("best");
                System.out.println("✓ 保存最佳模型 (val_loss: " + String.format("%.4f", bestValLoss) + ")");
            } else {
                stepsWithoutImprovement++;
                System.out.println("连续 " + stepsWithoutImprovement + " 个epoch未改善");
                
                // 早停检查
                if (stepsWithoutImprovement >= patience) {
                    System.out.println("触发早停机制,训练结束");
                    break;
                }
            }
        }
        
        System.out.println("\n微调完成!");
        System.out.println("最佳验证损失: " + bestValLoss);
    }
    
    /**
     * 训练一个epoch
     */
    private void trainOneEpoch() {
        trainDataset.prepare(true);
        
        double epochLoss = 0.0;
        int batchCount = 0;
        
        long epochStartTime = System.currentTimeMillis();
        
        while (trainDataset.hasNext()) {
            GPT1Dataset.Batch batch = trainDataset.nextBatch();
            
            // 训练一步
            float stepLoss = trainStep(batch);
            
            epochLoss += stepLoss;
            batchCount++;
            globalStep++;
            
            trainLossHistory.add(stepLoss);
            
            // 打印日志
            if (globalStep % logInterval == 0) {
                float avgLoss = getAverageLoss(trainLossHistory, logInterval);
                System.out.printf("Epoch %d/%d | Step %d | Train Loss: %.4f%n",
                    currentEpoch + 1, maxEpochs, globalStep, avgLoss);
            }
            
            // 定期验证
            if (globalStep % evalInterval == 0) {
                float valLoss = evaluate();
                System.out.printf("  Validation Loss: %.4f%n", valLoss);
            }
        }
        
        long epochEndTime = System.currentTimeMillis();
        double avgEpochLoss = batchCount > 0 ? epochLoss / batchCount : 0.0;
        
        System.out.printf("Epoch %d 完成 | 训练损失: %.4f | 耗时: %d ms%n",
            currentEpoch + 1, avgEpochLoss, epochEndTime - epochStartTime);
        
        trainDataset.reset();
    }
    
    /**
     * 训练单步
     * 
     * 支持loss mask：当batch包含lossMask时，只在mask=1的位置计算loss（Response部分），
     * mask=0的位置（Instruction和padding部分）不参与loss计算和梯度更新。
     * 
     * @param batch 批次数据
     * @return 损失值
     */
    private float trainStep(GPT1Dataset.Batch batch) {
        NdArray inputIds = batch.getInputIds();
        NdArray targetIds = batch.getTargetIds();
        NdArray lossMask = batch.getLossMask();
        
        Variable inputVar = new Variable(inputIds);
        Variable logits = model.predict(inputVar);
        
        int[] logitsShape = logits.getValue().getShape().getShapeDims();
        int batchSize = logitsShape[0];
        int seqLen = logitsShape[1];
        int vocabSize = logitsShape[2];
        
        Variable logits2D = logits.reshape(Shape.of(batchSize * seqLen, vocabSize));
        NdArray targets2D = targetIds.reshape(Shape.of(batchSize * seqLen, 1));
        Variable targetVar = new Variable(targets2D);
        
        Variable loss;
        if (lossMask != null) {
            // 微调模式：使用masked loss，只在Response部分计算loss
            Variable perTokenLoss = perTokenLossFunction.loss(targetVar, logits2D);
            // perTokenLoss shape: [batch*seq_len, 1] 或 [batch*seq_len]
            NdArray mask1D = lossMask.reshape(Shape.of(batchSize * seqLen, 1));
            Variable maskVar = new Variable(mask1D);
            Variable maskedLoss = perTokenLoss.mul(maskVar);
            
            // 计算有效token数量（mask=1的个数），避免除以0
            float validTokenCount = mask1D.sum().getNumber().floatValue();
            if (validTokenCount < 1.0f) {
                validTokenCount = 1.0f;
            }
            loss = maskedLoss.sum().mul(new Variable(NdArray.of(1.0f / validTokenCount)));
        } else {
            // 预训练模式：所有token都参与loss计算
            loss = lossFunction.loss(targetVar, logits2D);
        }
        
        float lossValue = loss.getValue().getNumber().floatValue();
        
        model.clearGrads();
        loss.backward();
        
        // 梯度裁剪
        clipGradients();
        
        optimizer.update();
        loss.unChainBackward();
        
        return lossValue;
    }
    
    /**
     * 在验证集上评估
     * 
     * 支持loss mask：与trainStep一致，当batch包含lossMask时只在Response部分计算loss。
     * 
     * @return 验证损失
     */
    private float evaluate() {
        valDataset.prepare(false);  // 不打乱
        
        double totalLoss = 0.0;
        int batchCount = 0;
        
        while (valDataset.hasNext()) {
            GPT1Dataset.Batch batch = valDataset.nextBatch();
            
            NdArray inputIds = batch.getInputIds();
            NdArray targetIds = batch.getTargetIds();
            NdArray lossMask = batch.getLossMask();
            
            Variable inputVar = new Variable(inputIds);
            Variable logits = model.predict(inputVar);
            
            int[] logitsShape = logits.getValue().getShape().getShapeDims();
            int batchSize = logitsShape[0];
            int seqLen = logitsShape[1];
            int vocabSize = logitsShape[2];
            
            Variable logits2D = logits.reshape(Shape.of(batchSize * seqLen, vocabSize));
            NdArray targets2D = targetIds.reshape(Shape.of(batchSize * seqLen, 1));
            Variable targetVar = new Variable(targets2D);
            
            Variable loss;
            if (lossMask != null) {
                Variable perTokenLoss = perTokenLossFunction.loss(targetVar, logits2D);
                NdArray mask1D = lossMask.reshape(Shape.of(batchSize * seqLen, 1));
                Variable maskVar = new Variable(mask1D);
                Variable maskedLoss = perTokenLoss.mul(maskVar);
                
                float validTokenCount = mask1D.sum().getNumber().floatValue();
                if (validTokenCount < 1.0f) {
                    validTokenCount = 1.0f;
                }
                loss = maskedLoss.sum().mul(new Variable(NdArray.of(1.0f / validTokenCount)));
            } else {
                loss = lossFunction.loss(targetVar, logits2D);
            }
            
            totalLoss += loss.getValue().getNumber().floatValue();
            batchCount++;
        }
        
        valDataset.reset();
        
        return batchCount > 0 ? (float) (totalLoss / batchCount) : 0.0f;
    }
    
    /**
     * 梯度裁剪
     */
    private void clipGradients() {
        double totalNorm = 0.0;
        
        Map<String, Parameter> params = model.getAllParams();
        for (Parameter param : params.values()) {
            if (param.getGrad() != null) {
                NdArray grad = param.getGrad();
                double norm = grad.mul(grad).sum().getNumber().doubleValue();
                totalNorm += norm;
            }
        }
        
        totalNorm = Math.sqrt(totalNorm);
        
        if (totalNorm > maxGradNorm) {
            float scale = (float) (maxGradNorm / totalNorm);
            for (Parameter param : params.values()) {
                if (param.getGrad() != null) {
                    NdArray clippedGrad = param.getGrad().mulNum(scale);
                    param.setGrad(clippedGrad);
                }
            }
        }
    }
    
    /**
     * 保存检查点
     * 
     * @param suffix 文件名后缀
     */
    private void saveCheckpoint(String suffix) {
        try {
            String filename = String.format("gpt1_finetune_%s.model", suffix);
            String filepath = checkpointDir + File.separator + filename;
            model.saveModel(filepath);
            System.out.println("检查点已保存: " + filepath);
        } catch (Exception e) {
            System.err.println("保存检查点失败: " + e.getMessage());
        }
    }
    
    /**
     * 创建检查点目录
     */
    private void createCheckpointDir() {
        try {
            Files.createDirectories(Paths.get(checkpointDir));
        } catch (IOException e) {
            System.err.println("创建检查点目录失败: " + e.getMessage());
        }
    }
    
    /**
     * 获取平均损失
     */
    private float getAverageLoss(List<Float> history, int n) {
        if (history.isEmpty()) {
            return 0.0f;
        }
        
        int start = Math.max(0, history.size() - n);
        float sum = 0.0f;
        for (int i = start; i < history.size(); i++) {
            sum += history.get(i);
        }
        return sum / (history.size() - start);
    }
    
    /**
     * 获取训练统计信息
     */
    public FinetuneStats getStats() {
        return new FinetuneStats(
            currentEpoch,
            globalStep,
            trainLossHistory.isEmpty() ? 0.0f : trainLossHistory.get(trainLossHistory.size() - 1),
            valLossHistory.isEmpty() ? 0.0f : valLossHistory.get(valLossHistory.size() - 1),
            bestValLoss,
            stepsWithoutImprovement
        );
    }
    
    /**
     * 微调统计信息
     */
    public static class FinetuneStats {
        public final int epoch;
        public final int step;
        public final float trainLoss;
        public final float valLoss;
        public final float bestValLoss;
        public final int patienceCount;
        
        public FinetuneStats(int epoch, int step, float trainLoss, 
                           float valLoss, float bestValLoss, int patienceCount) {
            this.epoch = epoch;
            this.step = step;
            this.trainLoss = trainLoss;
            this.valLoss = valLoss;
            this.bestValLoss = bestValLoss;
            this.patienceCount = patienceCount;
        }
        
        @Override
        public String toString() {
            return String.format(
                "FinetuneStats{epoch=%d, step=%d, trainLoss=%.4f, valLoss=%.4f, bestValLoss=%.4f, patience=%d}",
                epoch, step, trainLoss, valLoss, bestValLoss, patienceCount
            );
        }
    }
}
