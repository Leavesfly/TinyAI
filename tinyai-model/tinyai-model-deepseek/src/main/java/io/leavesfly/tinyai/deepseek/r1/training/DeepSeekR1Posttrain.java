package io.leavesfly.tinyai.deepseek.r1.training;

import io.leavesfly.tinyai.deepseek.r1.DeepSeekR1Config;
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
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * DeepSeek-R1后训练器(Posttrain/Finetune)
 * 
 * 用于在预训练模型基础上进行任务特定的微调,
 * 重点优化推理质量和反思能力。
 * 
 * 支持行业标准的 Answer-only Loss Mask：
 * 当数据集包含 Loss Mask 时，只对 assistant 回复部分计算 loss，
 * user 指令部分不参与梯度更新。
 * 
 * @author leavesfly
 * @version 2.0
 */
public class DeepSeekR1Posttrain {
    
    private final DeepSeekR1Model model;
    private final DeepSeekR1Config config;
    private final DeepSeekR1Dataset trainDataset;
    private final DeepSeekR1Dataset valDataset;
    private final SoftmaxCrossEntropy lossFunction;
    private final SoftmaxCrossEntropy elementWiseLossFunction;
    private final SGD optimizer;
    
    // 后训练超参数
    private int maxEpochs;
    private float learningRate;
    private float maxGradNorm;
    private int logInterval;
    private int evalInterval;
    private int patience;
    private String checkpointDir;
    
    // 训练状态
    private int currentEpoch;
    private int globalStep;
    private List<Float> trainLossHistory;
    private List<Float> valLossHistory;
    private List<Float> qualityScoreHistory;
    private float bestValLoss;
    private int stepsWithoutImprovement;
    
    public DeepSeekR1Posttrain(DeepSeekR1Model model, 
                               DeepSeekR1Dataset trainDataset,
                               DeepSeekR1Dataset valDataset) {
        this.model = model;
        this.config = model.getConfig();
        this.trainDataset = trainDataset;
        this.valDataset = valDataset;
        this.lossFunction = new SoftmaxCrossEntropy();
        this.elementWiseLossFunction = new SoftmaxCrossEntropy(Loss.Reduction.NONE);
        
        // 后训练学习率比预训练小10倍
        this.maxEpochs = 5;
        this.learningRate = 2.5e-5f;
        this.maxGradNorm = 1.0f;
        this.logInterval = 5;
        this.evalInterval = 100;
        this.patience = 3;
        this.checkpointDir = "./checkpoints/deepseek_r1/posttrain";
        
        // 使用SGD替代Adam，减少临时NdArray对象创建，降低内存占用
        this.optimizer = new SGD(model, learningRate);
        
        this.currentEpoch = 0;
        this.globalStep = 0;
        this.trainLossHistory = new ArrayList<>();
        this.valLossHistory = new ArrayList<>();
        this.qualityScoreHistory = new ArrayList<>();
        this.bestValLoss = Float.MAX_VALUE;
        this.stepsWithoutImprovement = 0;
    }
    
    public DeepSeekR1Posttrain configure(int maxEpochs, float learningRate, int patience) {
        this.maxEpochs = maxEpochs;
        this.learningRate = learningRate;
        this.patience = patience;
        // 同步学习率到优化器
        this.optimizer.setLearningRate(learningRate);
        return this;
    }
    
    public void train() {
        System.out.println("=".repeat(70));
        System.out.println("DeepSeek-R1 后训练/微调 (Posttrain)");
        System.out.println("=".repeat(70));
        System.out.println("模型配置: " + model.getName());
        System.out.println("训练样本: " + trainDataset.getSampleCount());
        System.out.println("验证样本: " + valDataset.getSampleCount());
        System.out.println("学习率: " + learningRate);
        System.out.println("早停耐心: " + patience);
        System.out.println("Loss Mask: " + (trainDataset.hasLossMasks() ? "启用（Answer-only Loss）" : "未启用（全序列Loss）"));
        System.out.println("=".repeat(70));
        
        createCheckpointDir();
        
        for (currentEpoch = 0; currentEpoch < maxEpochs; currentEpoch++) {
            trainOneEpoch();
            
            float valLoss = evaluate();
            valLossHistory.add(valLoss);
            
            System.out.printf("Epoch %d 验证损失: %.4f%n", currentEpoch + 1, valLoss);
            
            if (valLoss < bestValLoss) {
                bestValLoss = valLoss;
                stepsWithoutImprovement = 0;
                saveCheckpoint("best");
                System.out.println("✓ 保存最佳模型 (val_loss: " + String.format("%.4f", bestValLoss) + ")");
            } else {
                stepsWithoutImprovement++;
                if (stepsWithoutImprovement >= patience) {
                    System.out.println("触发早停,训练结束");
                    break;
                }
            }
        }
        
        System.out.println("\n后训练完成! 最佳验证损失: " + bestValLoss);
    }
    
    private void trainOneEpoch() {
        trainDataset.prepare(true);
        
        while (trainDataset.hasNext()) {
            DeepSeekR1Dataset.Batch batch = trainDataset.nextBatch();
            
            NdArray inputIds = batch.getInputIds();
            NdArray targetIds = batch.getTargetIds();
            
            Variable inputVar = new Variable(inputIds);
            DeepSeekR1Model.ReasoningResult result = model.performReasoning(inputVar);
            
            // SoftmaxCE只支持2D输入，需要reshape
            int[] logitsShape = result.logits.getValue().getShape().getShapeDims();
            int batchSize = logitsShape[0];
            int seqLen = logitsShape[1];
            int vocabSize = logitsShape[2];
            
            Variable logits2D = result.logits.reshape(Shape.of(batchSize * seqLen, vocabSize));
            Variable targetVar = new Variable(targetIds.reshape(Shape.of(batchSize * seqLen, 1)));
            
            Variable loss;
            if (batch.hasLossMask()) {
                // Answer-only Loss: 使用逐元素 loss 乘以 mask，只对 assistant 回复部分计算梯度
                Variable elementWiseLoss = elementWiseLossFunction.loss(targetVar, logits2D);
                
                // 将 mask reshape 为 [batchSize * seqLen, 1] 与逐元素 loss 对齐
                NdArray maskFlat = batch.getLossMask().reshape(Shape.of(batchSize * seqLen, 1));
                Variable maskVar = new Variable(maskFlat);
                
                // masked loss = elementWiseLoss * mask
                Variable maskedLoss = elementWiseLoss.mul(maskVar);
                
                // 计算有效位置数量，避免除以零
                float maskSum = maskFlat.sum().getNumber().floatValue();
                float effectiveCount = Math.max(maskSum, 1.0f);
                
                // 对有效位置求平均: sum(maskedLoss) / count(mask==1)
                loss = maskedLoss.sum().div(new Variable(NdArray.of(new float[]{effectiveCount})));
            } else {
                // 无 mask 时使用标准 MEAN 归约
                loss = lossFunction.loss(targetVar, logits2D);
            }
            
            float lossValue = loss.getValue().getNumber().floatValue();
            float moeLoss = (float) result.moeLoss;
            
            trainLossHistory.add(lossValue);
            qualityScoreHistory.add(moeLoss);
            
            model.clearGrads();
            loss.backward();
            clipGradients();
            optimizer.update();
            
            // 彻底断开计算图，释放内存
            loss.unChainBackward();
            result.logits.unChainBackward();
            inputVar.unChainBackward();
            
            globalStep++;
            
            if (globalStep % logInterval == 0) {
                System.out.printf("Epoch %d | Step %d | Loss: %.4f | MoE Loss: %.4f%n",
                    currentEpoch + 1, globalStep, lossValue, moeLoss);
            }
        }
        
        trainDataset.reset();
        // Epoch结束后主动触发GC
        System.gc();
    }
    
    private float evaluate() {
        valDataset.prepare(false);
        
        double totalLoss = 0.0;
        int count = 0;
        
        while (valDataset.hasNext()) {
            DeepSeekR1Dataset.Batch batch = valDataset.nextBatch();
            
            Variable inputVar = new Variable(batch.getInputIds());
            DeepSeekR1Model.ReasoningResult result = model.performReasoning(inputVar);
            
            // SoftmaxCE只支持2D输入，需要reshape
            int[] logitsShape = result.logits.getValue().getShape().getShapeDims();
            int batchSize = logitsShape[0];
            int seqLen = logitsShape[1];
            int vocabSize = logitsShape[2];
            
            Variable logits2D = result.logits.reshape(Shape.of(batchSize * seqLen, vocabSize));
            Variable targetVar = new Variable(batch.getTargetIds().reshape(Shape.of(batchSize * seqLen, 1)));
            
            Variable loss;
            if (batch.hasLossMask()) {
                Variable elementWiseLoss = elementWiseLossFunction.loss(targetVar, logits2D);
                NdArray maskFlat = batch.getLossMask().reshape(Shape.of(batchSize * seqLen, 1));
                Variable maskVar = new Variable(maskFlat);
                Variable maskedLoss = elementWiseLoss.mul(maskVar);
                float maskSum = maskFlat.sum().getNumber().floatValue();
                float effectiveCount = Math.max(maskSum, 1.0f);
                loss = maskedLoss.sum().div(new Variable(NdArray.of(new float[]{effectiveCount})));
            } else {
                loss = lossFunction.loss(targetVar, logits2D);
            }
            
            totalLoss += loss.getValue().getNumber().floatValue();
            count++;
            
            // 验证时也需要释放计算图
            loss.unChainBackward();
            result.logits.unChainBackward();
            inputVar.unChainBackward();
        }
        
        valDataset.reset();
        return count > 0 ? (float) (totalLoss / count) : 0.0f;
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
                            String.format("deepseek_r1_posttrain_%s.model", suffix);
            model.saveModel(filepath);
            System.out.println("检查点已保存: " + filepath);
        } catch (Exception e) {
            System.err.println("保存失败: " + e.getMessage());
        }
    }
    
    private void createCheckpointDir() {
        try {
            Files.createDirectories(Paths.get(checkpointDir));
        } catch (IOException e) {
            System.err.println("创建目录失败: " + e.getMessage());
        }
    }
}
